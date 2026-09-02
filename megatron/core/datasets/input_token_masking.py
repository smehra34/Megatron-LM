# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-time corruption of GPT input tokens for noisy next-token prediction."""

import math
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


class VariableSpanInputMaskingStrategy(InputMaskingStrategy):
    """Select variable-length spans from a truncated geometric distribution.

    Span lengths from one through the configured maximum have probability
    proportional to ``0.5**length``. For example, a maximum of five gives
    normalized weights proportional to ``[16, 8, 4, 2, 1]``. This standard
    memoryless distribution emphasizes easier short spans while retaining a
    diminishing tail of harder spans.
    """

    def __init__(self, span_length: int):
        if span_length <= 0:
            raise ValueError(
                f"variable-span maximum length must be positive, got {span_length}"
            )
        self.max_span_length = span_length

    @staticmethod
    def _sample_span_length(max_length: int, generator: torch.Generator) -> int:
        """Sample a geometric span length, conditioned on the feasible maximum."""
        if max_length <= 0:
            raise ValueError(f"maximum span length must be positive, got {max_length}")

        # For weights 2^-length, the truncated CDF through length l is
        # (1 - 2^-l) / (1 - 2^-max_length). Inverting it avoids constructing a
        # probability vector and remains bounded for any configured maximum.
        draw = float(torch.rand((), generator=generator).item())
        normalization = 1.0 - math.ldexp(1.0, -max_length)
        tail_probability = 1.0 - draw * normalization
        length = math.floor(-math.log2(tail_probability)) + 1
        return min(length, max_length)

    def select(
        self,
        eligible: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Select complete boundary-safe spans up to the target token budget.

        At each step the length distribution is renormalized over lengths that
        fit both the remaining budget and at least one free eligible run. The
        start is then uniform over every feasible start position for that
        length. This bounded policy uses shorter feasible spans near the end,
        never rejects indefinitely, and reaches the exact floored token target
        because length one is always available while capacity remains.
        """
        selected = torch.zeros_like(eligible, dtype=torch.bool)
        remaining = int(eligible.sum().item() * ratio)
        free_runs = SpanInputMaskingStrategy._eligible_runs(eligible)

        while remaining and free_runs:
            longest_run = max(end - start for start, end in free_runs)
            feasible_max = min(self.max_span_length, remaining, longest_run)
            span_length = self._sample_span_length(feasible_max, generator)

            placements = [
                (start, end, end - start - span_length + 1)
                for start, end in free_runs
                if end - start >= span_length
            ]
            total_placements = sum(count for _, _, count in placements)
            placement = int(
                torch.randint(total_placements, (1,), generator=generator).item()
            )
            for start, end, count in placements:
                if placement < count:
                    span_start = start + placement
                    break
                placement -= count

            span_end = span_start + span_length
            selected[span_start:span_end] = True
            remaining -= span_length
            updated_runs = []
            for run_start, run_end in free_runs:
                if run_start <= span_start < run_end:
                    updated_runs.extend(
                        interval
                        for interval in ((run_start, span_start), (span_end, run_end))
                        if interval[0] < interval[1]
                    )
                else:
                    updated_runs.append((run_start, run_end))
            free_runs = updated_runs

        return selected


_STRATEGIES: Dict[str, Type[InputMaskingStrategy]] = {
    "random": RandomInputMaskingStrategy,
    "span": SpanInputMaskingStrategy,
    "variable_span": VariableSpanInputMaskingStrategy,
}


def build_input_masking_strategy(name: str, span_length: int) -> InputMaskingStrategy:
    """Construct a registered masking strategy by name."""
    try:
        strategy_type = _STRATEGIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"unknown input mask strategy {name!r}; choose from: {choices}") from exc
    return strategy_type(span_length) if name in {"span", "variable_span"} else strategy_type()


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


def build_input_mask_offsets(masked_positions: torch.Tensor) -> torch.Tensor:
    """Return the one-based position within each contiguous run of masked tokens.

    Offsets are inferred from the final corruption bitmap, so adjacent sampled
    spans form one longer effective prediction horizon. Unmasked positions are
    represented by zero.
    """
    if masked_positions.ndim != 1:
        raise ValueError(
            f"expected a 1D masked-position tensor, got shape {tuple(masked_positions.shape)}"
        )
    masked_positions = masked_positions.bool()
    indices = torch.arange(masked_positions.numel(), device=masked_positions.device)
    last_unmasked = torch.where(masked_positions, -1, indices)
    last_unmasked = torch.cummax(last_unmasked, dim=0).values
    return torch.where(masked_positions, indices - last_unmasked, 0).to(torch.int32)


def compute_token_topk_correct(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    logits_are_vocab_sharded: bool,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    max_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute global top-1 and top-k correctness without gathering full logits.

    Args:
        logits: Vocabulary logits with vocabulary in the final dimension.
        labels: Global token IDs matching the leading dimensions of ``logits``.
        logits_are_vocab_sharded: Whether the last logit dimension is a TP shard.
        tp_group: Tensor-parallel group, required for sharded logits when TP > 1.
        max_k: Largest top-k accuracy to compute.
    """
    if max_k < 1:
        raise ValueError(f"max_k must be positive, got {max_k}")
    if logits.ndim < 2 or logits.size(-1) == 0:
        raise ValueError(
            f"logits must have a non-empty vocabulary dimension, got shape {tuple(logits.shape)}"
        )
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "labels must match the leading dimensions of logits, got "
            f"{tuple(labels.shape)} and {tuple(logits.shape)}"
        )

    with torch.no_grad():
        local_k = min(max_k, logits.size(-1))
        candidate_values, candidate_ids = torch.topk(logits.detach(), local_k, dim=-1)

        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1
        if logits_are_vocab_sharded and tp_size > 1:
            vocab_shard_size = logits.size(-1)
            candidate_ids = (
                candidate_ids
                + torch.distributed.get_rank(group=tp_group) * vocab_shard_size
            )
            gathered_values = [torch.empty_like(candidate_values) for _ in range(tp_size)]
            gathered_ids = [torch.empty_like(candidate_ids) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered_values, candidate_values, group=tp_group)
            torch.distributed.all_gather(gathered_ids, candidate_ids, group=tp_group)
            candidate_values = torch.cat(gathered_values, dim=-1)
            candidate_ids = torch.cat(gathered_ids, dim=-1)

        global_k = min(max_k, candidate_values.size(-1))
        top_candidate_indices = torch.topk(candidate_values, global_k, dim=-1).indices
        top_token_ids = torch.gather(candidate_ids, dim=-1, index=top_candidate_indices)
        labels = labels.unsqueeze(-1)
        matches = top_token_ids == labels
        return matches[..., 0], matches.any(dim=-1)


class InputMaskingMetricsLoggingHelper:
    """Accumulate and log reporting-only metrics by effective mask offset."""

    tracker: Dict[str, torch.Tensor | torch.distributed.ProcessGroup | None] = {}

    @classmethod
    def save_metrics(
        cls,
        losses: torch.Tensor,
        loss_mask: torch.Tensor,
        mask_offsets: torch.Tensor,
        top1_correct: torch.Tensor,
        top5_correct: torch.Tensor,
        max_offsets: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """Accumulate detached sums and counts for one microbatch."""
        with torch.no_grad():
            losses = losses.detach().view(-1).float()
            valid = loss_mask.detach().view(-1).bool()
            offsets = mask_offsets.detach().view(-1).long()
            top1_correct = top1_correct.detach().view(-1)
            top5_correct = top5_correct.detach().view(-1)
            selected = valid & (offsets > 0)

            offsets = offsets[selected].clamp(max=max_offsets)
            size = max_offsets + 1
            device = losses.device
            updates = {
                "loss_sums": losses[selected],
                "probability_sums": torch.exp(-losses[selected]),
                "top1_sums": top1_correct[selected].float(),
                "top5_sums": top5_correct[selected].float(),
                "counts": torch.ones(offsets.numel(), device=device),
            }
            for name, values in updates.items():
                if name not in cls.tracker:
                    cls.tracker[name] = torch.zeros(size, device=device)
                cls.tracker[name].scatter_add_(0, offsets, values)
            cls.tracker["reduce_group"] = reduce_group

    @classmethod
    def log_metrics(cls, iteration: int, writer=None, wandb_writer=None) -> None:
        """Reduce, log, and clear metrics accumulated since the previous call."""
        if "counts" not in cls.tracker:
            return

        reduce_group = cls.tracker.get("reduce_group")
        metric_names = ("loss_sums", "probability_sums", "top1_sums", "top5_sums", "counts")
        metric_values = torch.stack([cls.tracker[name] for name in metric_names])
        if reduce_group is not None:
            torch.distributed.all_reduce(metric_values, group=reduce_group)

        loss_sums, probability_sums, top1_sums, top5_sums, counts = metric_values
        observed_offsets = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
        wandb_metrics = {}
        for offset in observed_offsets:
            if offset == 0:
                continue
            count = counts[offset]
            metrics = {
                f"input_masking/mask_{offset}/loss": loss_sums[offset] / count,
                f"input_masking/mask_{offset}/target_probability": probability_sums[offset]
                / count,
                f"input_masking/mask_{offset}/top1_accuracy": top1_sums[offset]
                / count
                * 100.0,
                f"input_masking/mask_{offset}/top5_accuracy": top5_sums[offset]
                / count
                * 100.0,
                f"input_masking/mask_{offset}/count": count,
            }
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(name, value, iteration)
            wandb_metrics.update(metrics)

        if wandb_writer is not None and wandb_metrics:
            wandb_writer.log(wandb_metrics, iteration)

        cls.tracker.clear()


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
