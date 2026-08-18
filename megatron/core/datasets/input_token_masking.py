# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-time corruption of GPT input tokens for noisy next-token prediction."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch


class InputMaskingStrategy(ABC):
    """Select input positions to replace without modifying targets or loss masks."""

    @abstractmethod
    def select(
        self,
        eligible: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Return a boolean tensor selecting a subset of ``eligible`` positions."""


class RandomInputMaskingStrategy(InputMaskingStrategy):
    """Select an exact, uniformly random subset of eligible token positions."""

    def select(
        self,
        eligible: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Select individual positions without replacement."""
        selected = torch.zeros_like(eligible, dtype=torch.bool)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        count = int(eligible_indices.numel() * ratio)
        if count:
            order = torch.randperm(eligible_indices.numel(), generator=generator)[:count]
            selected[eligible_indices[order]] = True
        return selected


class SpanInputMaskingStrategy(InputMaskingStrategy):
    """Select randomly positioned, non-overlapping, fixed-length spans."""

    def __init__(self, span_length: int):
        if span_length <= 0:
            raise ValueError(f"input mask span length must be positive, got {span_length}")
        self.span_length = span_length

    @staticmethod
    def _eligible_runs(eligible: torch.Tensor) -> List[Tuple[int, int]]:
        """Return half-open intervals for contiguous runs of eligible positions."""
        padded = torch.nn.functional.pad(eligible.to(torch.int8), (1, 1))
        transitions = padded[1:] - padded[:-1]
        starts = torch.nonzero(transitions == 1, as_tuple=False).flatten().tolist()
        ends = torch.nonzero(transitions == -1, as_tuple=False).flatten().tolist()
        return list(zip(starts, ends))

    def select(
        self,
        eligible: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Select complete spans without crossing an ineligible position."""
        selected = torch.zeros_like(eligible, dtype=torch.bool)
        target_tokens = int(eligible.sum().item() * ratio)
        requested_spans = target_tokens // self.span_length
        if requested_spans == 0:
            return selected

        runs = self._eligible_runs(eligible)
        capacities = [(end - start) // self.span_length for start, end in runs]
        total_capacity = sum(capacities)
        num_spans = min(requested_spans, total_capacity)
        if num_spans == 0:
            return selected

        capacity_slots = torch.tensor(
            [run_idx for run_idx, capacity in enumerate(capacities) for _ in range(capacity)],
            dtype=torch.long,
        )
        chosen_slots = capacity_slots[
            torch.randperm(total_capacity, generator=generator)[:num_spans]
        ]
        spans_per_run = torch.bincount(chosen_slots, minlength=len(runs)).tolist()

        for (start, end), run_spans in zip(runs, spans_per_run):
            if run_spans == 0:
                continue

            slack = (end - start) - run_spans * self.span_length
            gaps = torch.zeros(run_spans + 1, dtype=torch.long)
            if slack:
                gap_assignments = torch.randint(run_spans + 1, (slack,), generator=generator)
                gaps = torch.bincount(gap_assignments, minlength=run_spans + 1)

            cursor = start + int(gaps[0])
            for span_idx in range(run_spans):
                selected[cursor : cursor + self.span_length] = True
                cursor += self.span_length + int(gaps[span_idx + 1])

        return selected


_STRATEGIES: Dict[str, Type[InputMaskingStrategy]] = {
    "random": RandomInputMaskingStrategy,
    "span": SpanInputMaskingStrategy,
}


def build_input_masking_strategy(name: str, span_length: int) -> InputMaskingStrategy:
    """Construct a registered masking strategy by name."""
    try:
        strategy_type = _STRATEGIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"unknown input mask strategy {name!r}; choose from: {choices}") from exc
    return strategy_type(span_length) if name == "span" else strategy_type()


def apply_input_token_masking(
    tokens: torch.Tensor,
    *,
    mask_token_id: int,
    eod_token_id: int,
    pad_token_id: Optional[int],
    ratio: float,
    strategy: InputMaskingStrategy,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace selected inputs and return the corrupted copy and position bitmap.

    EOD, padding, sequence position zero, and the first token after every EOD are
    ineligible. Consequently, a selected span cannot cross a document boundary.
    """
    if tokens.ndim != 1:
        raise ValueError(f"expected a 1D token tensor, got shape {tuple(tokens.shape)}")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"input mask ratio must be in [0, 1], got {ratio}")

    eligible = torch.ones_like(tokens, dtype=torch.bool)
    if tokens.numel():
        eligible[0] = False
    eligible &= tokens != eod_token_id
    if pad_token_id is not None:
        eligible &= tokens != pad_token_id
    if tokens.numel() > 1:
        eligible[1:] &= tokens[:-1] != eod_token_id

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    masked_positions = strategy.select(eligible, ratio, generator)
    corrupted = tokens.clone()
    corrupted[masked_positions] = mask_token_id
    return corrupted, masked_positions


def build_input_masking_loss_report(
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    input_masked_positions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build reporting-only loss aggregates split by corrupted input position."""
    losses = losses.view(-1).float()
    valid_positions = loss_mask.view(-1).bool()
    input_masked_positions = input_masked_positions.view(-1).bool()
    if losses.shape != valid_positions.shape or losses.shape != input_masked_positions.shape:
        raise ValueError(
            "losses, loss mask, and input masked positions must have the same number of elements"
        )

    masked_positions = valid_positions & input_masked_positions
    unmasked_positions = valid_positions & ~input_masked_positions
    masked_count = masked_positions.sum().detach().to(torch.int)
    unmasked_count = unmasked_positions.sum().detach().to(torch.int)
    num_tokens = valid_positions.sum().detach().to(torch.int)
    return {
        'masked-input lm loss': torch.cat(
            [torch.sum(losses * masked_positions.float()).detach().view(1), masked_count.view(1)]
        ),
        'unmasked-input lm loss': torch.cat(
            [
                torch.sum(losses * unmasked_positions.float()).detach().view(1),
                unmasked_count.view(1),
            ]
        ),
        'masked-input fraction': torch.stack([masked_count.float(), num_tokens.float()]),
    }
