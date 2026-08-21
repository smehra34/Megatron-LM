# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.datasets.input_token_masking import (
    InputMaskingMetricsLoggingHelper,
    apply_input_token_masking,
    build_input_mask_offsets,
    build_input_masking_loss_report,
    build_input_masking_strategy,
    compute_token_topk_correct,
)


def _apply(tokens, ratio, strategy="random", span_length=1, seed=123):
    return apply_input_token_masking(
        torch.tensor(tokens),
        mask_token_id=999,
        eod_token_id=2,
        pad_token_id=3,
        ratio=ratio,
        strategy=build_input_masking_strategy(strategy, span_length),
        seed=seed,
    )


def test_random_masking_is_exact_and_deterministic():
    tokens = list(range(10, 20))
    corrupted_a, positions_a = _apply(tokens, ratio=0.4, seed=7)
    corrupted_b, positions_b = _apply(tokens, ratio=0.4, seed=7)

    assert positions_a.sum().item() == 3
    assert torch.equal(positions_a, positions_b)
    assert torch.equal(corrupted_a, corrupted_b)
    assert torch.all(corrupted_a[positions_a] == 999)
    assert torch.equal(corrupted_a[~positions_a], torch.tensor(tokens)[~positions_a])


def test_special_and_document_boundary_positions_are_ineligible():
    tokens = [10, 11, 2, 12, 13, 3, 14]
    corrupted, positions = _apply(tokens, ratio=1.0)

    assert positions.tolist() == [False, True, False, False, True, False, True]
    assert corrupted.tolist() == [10, 999, 2, 12, 999, 3, 999]


def test_span_masking_uses_complete_non_overlapping_spans_without_crossing_eod():
    tokens = [10, 11, 12, 13, 2, 14, 15, 16, 17, 18]
    _, positions = _apply(tokens, ratio=1.0, strategy="span", span_length=2, seed=9)

    assert positions.sum().item() == 6
    assert not positions[0]
    assert not positions[4]
    assert not positions[5]
    padded = torch.nn.functional.pad(positions.to(torch.int8), (1, 1))
    transitions = padded[1:] - padded[:-1]
    starts = torch.nonzero(transitions == 1).flatten().tolist()
    ends = torch.nonzero(transitions == -1).flatten().tolist()
    assert all((end - start) % 2 == 0 for start, end in zip(starts, ends))


def test_span_masking_rounds_down_to_complete_spans():
    _, positions = _apply(
        list(range(10, 20)), ratio=0.4, strategy="span", span_length=2, seed=5
    )
    assert positions.sum().item() == 2


def test_mask_offsets_use_final_contiguous_runs():
    positions = torch.tensor(
        [False, True, True, False, True, True, True, True, False, True]
    )

    offsets = build_input_mask_offsets(positions)

    assert offsets.dtype == torch.int32
    assert offsets.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]


def test_topk_correct_uses_ground_truth_membership():
    logits = torch.tensor(
        [
            [0.0, 4.0, 3.0, 2.0, 1.0, -1.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    labels = torch.tensor([1, 4, 0])

    top1, top5 = compute_token_topk_correct(
        logits, labels, logits_are_vocab_sharded=False, max_k=5
    )

    assert top1.tolist() == [True, False, False]
    assert top5.tolist() == [True, True, False]


def test_topk_correct_combines_tensor_parallel_candidates(monkeypatch):
    local_logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 3.0, 1.0]])
    labels = torch.tensor([5, 0])
    remote_values = torch.tensor([[6.0, 5.0], [7.0, 2.0]])
    remote_ids = torch.tensor([[4, 5], [3, 4]])

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    def fake_all_gather(outputs, local_candidates, group):
        outputs[0].copy_(local_candidates)
        if local_candidates.dtype.is_floating_point:
            outputs[1].copy_(remote_values)
        else:
            outputs[1].copy_(remote_ids)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    top1, top2 = compute_token_topk_correct(
        local_logits,
        labels,
        logits_are_vocab_sharded=True,
        tp_group=object(),
        max_k=2,
    )

    assert top1.tolist() == [False, False]
    assert top2.tolist() == [True, True]


def test_mask_offset_metrics_are_count_weighted_and_cleared():
    class WandbWriter:
        def __init__(self):
            self.metrics = {}

        def log(self, metrics, iteration):
            assert iteration == 17
            self.metrics.update(metrics)

    InputMaskingMetricsLoggingHelper.tracker.clear()
    losses = -torch.log(torch.tensor([[0.5, 0.25, 0.1, 0.2]]))
    offsets = torch.tensor([[1, 2, 0, 1]], dtype=torch.int32)
    loss_mask = torch.ones_like(losses)
    top1 = torch.tensor([[True, False, False, True]])
    top5 = torch.tensor([[True, True, False, True]])

    InputMaskingMetricsLoggingHelper.save_metrics(
        losses,
        loss_mask,
        offsets,
        top1,
        top5,
        max_offsets=4,
    )
    wandb_writer = WandbWriter()
    InputMaskingMetricsLoggingHelper.log_metrics(17, wandb_writer=wandb_writer)

    assert torch.isclose(wandb_writer.metrics["input_masking/mask_1/count"], torch.tensor(2.0))
    assert torch.isclose(
        wandb_writer.metrics["input_masking/mask_1/target_probability"], torch.tensor(0.35)
    )
    assert torch.isclose(
        wandb_writer.metrics["input_masking/mask_1/top1_accuracy"], torch.tensor(100.0)
    )
    assert torch.isclose(
        wandb_writer.metrics["input_masking/mask_2/top5_accuracy"], torch.tensor(100.0)
    )
    assert InputMaskingMetricsLoggingHelper.tracker == {}


def test_zero_ratio_is_noop():
    tokens = [10, 11, 12]
    corrupted, positions = _apply(tokens, ratio=0.0)
    assert corrupted.tolist() == tokens
    assert not positions.any()


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_invalid_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="must be in"):
        _apply([10, 11], ratio=ratio)


def test_invalid_span_length_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        build_input_masking_strategy("span", 0)


def test_loss_report_uses_same_position_as_masked_input():
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    masked_inputs = torch.tensor([[False, True, True, False]])

    report = build_input_masking_loss_report(losses, loss_mask, masked_inputs)

    assert report['masked-input lm loss'].tolist() == [2.0, 1.0]
    assert report['unmasked-input lm loss'].tolist() == [5.0, 2.0]
    assert report['masked-input fraction'].tolist() == [1.0, 3.0]
    assert report['masked-input lm loss'][0] + report['unmasked-input lm loss'][0] == 7.0
