# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.training.datasets.data_samplers import MegatronPretrainingSampler


def test_cyclic_validation_sampler_resumes_beyond_dataset_length():
    """Cumulative validation progress resumes modulo complete DP microbatches."""
    sampler = MegatronPretrainingSampler(
        total_samples=18,
        consumed_samples=28,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=2,
        cyclic=True,
    )

    # There are 16 usable samples. Offset 28 resumes at 12, wraps at 16, and
    # visits every complete batch exactly once. Rank zero gets two samples per batch.
    assert list(sampler) == [[12, 13], [0, 1], [4, 5], [8, 9]]


def test_cyclic_validation_sampler_uses_batch_aligned_physical_cycle():
    sampler = MegatronPretrainingSampler(
        total_samples=183952,
        consumed_samples=512000,
        micro_batch_size=4,
        data_parallel_rank=0,
        data_parallel_size=16,
        cyclic=True,
    )

    assert next(iter(sampler)) == [144128, 144129, 144130, 144131]


def test_non_cyclic_sampler_behavior_is_unchanged():
    sampler = MegatronPretrainingSampler(
        total_samples=18,
        consumed_samples=4,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    assert list(sampler) == [[6, 7], [10, 11], [14, 15]]
