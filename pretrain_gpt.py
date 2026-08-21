# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Some libraries (e.g., CUTLASS DSL) use warnings.catch_warnings() with
    # simplefilter("always"), which overrides the filters above. Override
    # showwarning as a fallback to suppress warnings that slip through.
    _original_showwarning = warnings.showwarning

    def _rank0_only_showwarning(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, (UserWarning, FutureWarning, DeprecationWarning)):
            return
        _original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _rank0_only_showwarning

from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.input_token_masking import (
    InputMaskingMetricsLoggingHelper,
    build_input_masking_loss_report,
    compute_token_topk_correct,
)
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_te_version,
    get_torch_version,
)
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_builder import ModelOptModelConfig
    from megatron.post_training.utils import maybe_enable_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

# Canonical, ordered schema of the fields ``get_batch`` returns. Kept alphabetical
# to match the historical ``sorted(batch.keys())`` order that callers unpack into.
BATCH_KEYS = [
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "hybrid_cp_group",
    "input_mask_original_tokens",
    "input_masked_positions",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "position_ids",
    "tokens",
]

_INPUT_MASK_DEBUG_PRINTED = False


def _print_input_mask_debug_example(batch: dict[str, torch.Tensor]) -> None:
    """Print and validate one actual corrupted training sequence on global rank zero."""
    global _INPUT_MASK_DEBUG_PRINTED
    args = get_args()
    original = batch.get('input_mask_original_tokens')
    # Debug reference data is never consumed by the model or CP partitioning.
    batch['input_mask_original_tokens'] = None
    if (
        _INPUT_MASK_DEBUG_PRINTED
        or not args.input_mask_debug
        or original is None
        or torch.distributed.get_rank() != 0
    ):
        return

    limit = args.input_mask_debug_tokens
    if limit <= 0:
        raise ValueError(f"--input-mask-debug-tokens must be positive, got {limit}")
    original = original[0, :limit].detach().cpu()
    corrupted = batch['tokens'][0, :limit].detach().cpu()
    masked = batch['input_masked_positions'][0, :limit].detach().cpu().bool()
    changed = original != corrupted
    if not torch.equal(changed, masked):
        raise RuntimeError(
            "input masking debug check failed: changed token positions do not match the mask"
        )
    if masked.any() and not torch.all(corrupted[masked] == args.input_mask_token_id):
        raise RuntimeError(
            "input masking debug check failed: a selected position does not contain the mask token"
        )

    print_rank_0("> input masking debug example (first training sample, shown once per job)")
    print_rank_0(
        f"> strategy={args.input_mask_strategy} ratio={args.input_mask_ratio} "
        f"mask_token={args.input_mask_token!r} mask_token_id={args.input_mask_token_id}"
    )
    print_rank_0(f"> original input ids:  {original.tolist()}")
    print_rank_0(f"> corrupted input ids: {corrupted.tolist()}")
    if batch['labels'] is not None:
        labels = batch['labels'][0, :limit].detach().cpu()
        print_rank_0(f"> causal target ids:   {labels.tolist()}")
    print_rank_0(f"> masked positions:    {masked.to(torch.int).tolist()}")
    print_rank_0(f"> displayed masked fraction: {masked.float().mean().item():.6f}")
    _INPUT_MASK_DEBUG_PRINTED = True


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_sft = args.sft
    has_cu_seqlens = is_sft or args.dataloader_inter_document_masking
    has_input_masking = args.input_mask_ratio > 0.0
    create_attention_mask_in_dataloader = args.create_attention_mask_in_dataloader
    mtp_on_this_rank = mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )
    is_hybrid_cp = args.hybrid_context_parallel

    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not mtp_on_this_rank
        and not has_cu_seqlens
    ):
        return [None for _ in BATCH_KEYS]

    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        if has_input_masking and 'input_masked_positions' not in batch:
            raise RuntimeError(
                "input masking is enabled but the dataset did not return input_masked_positions"
            )
        for key in BATCH_KEYS:
            batch[key] = (
                batch[key].cuda(non_blocking=True)
                if key in batch and batch[key] is not None
                else None
            )

    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        has_cu_seqlens=has_cu_seqlens,
        has_input_masking=has_input_masking,
        is_hybrid_cp=is_hybrid_cp,
        create_attention_mask_in_dataloader=create_attention_mask_in_dataloader,
        cp_size=cp_size,
        tp_rank=tp_rank,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=mtp_on_this_rank,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )

    _print_input_mask_debug_example(batch)

    batch = flatten_batch_for_packed_sequences(batch)

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank:
        assert has_cu_seqlens
        values = {key: None for key in BATCH_KEYS}
        values['cu_seqlens'] = batch['cu_seqlens']
        values['cu_seqlens_padded'] = batch['cu_seqlens_padded']
        values['max_seqlen'] = batch['max_seqlen']
        return [values[key] for key in BATCH_KEYS]

    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=is_hybrid_cp,
        cp_group=get_context_parallel_group(),
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_per_sequence_balancing=args.dataloader_inter_document_masking and not is_sft,
    )

    # Return values in BATCH_KEYS order so callers can unpack into the fixed
    # names regardless of any provenance fields wrappers like BlendedDataset
    # add (e.g. "dataset_id"). The for-loop above already populates every
    # BATCH_KEYS entry on tp_rank 0; other tp_ranks receive a fresh dict from
    # get_batch_on_this_tp_rank. BATCH_KEYS is already alphabetical, matching
    # the historical sorted(batch.keys()) order.
    return [batch[key] for key in BATCH_KEYS]


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


@lru_cache(maxsize=1)
def _build_cached_logits_loss_func(
    logprobs_dir, decode_threads, prefetch_factor, msc_prefetch_depth, kd_loss_alpha, ignore_errors
):
    """Build (once) the offline knowledge-distillation loss callable for cached logits.

    Memoized so the teacher log-probability reader is constructed a single time per
    process, replacing the previous module-level mutable global.
    """
    from megatron.training.distillation import LossFuncCallable

    return LossFuncCallable(
        logprobs_dir=logprobs_dir,
        decode_threads=decode_threads,
        prefetch_factor=prefetch_factor,
        msc_prefetch_depth=msc_prefetch_depth,
        kd_loss_alpha=kd_loss_alpha,
        ignore_errors=ignore_errors,
    )


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[GPTModel] = None,
    input_masked_positions: Optional[torch.Tensor] = None,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if args.logits_load_dir is not None:
        # Offline knowledge distillation loss using cached teacher log-probabilities.
        loss_func_cached_logits = _build_cached_logits_loss_func(
            logprobs_dir=args.logits_load_dir,
            decode_threads=args.logits_load_decode_threads,
            prefetch_factor=args.logits_load_prefetch_factor,
            msc_prefetch_depth=args.logits_load_msc_prefetch_depth,
            kd_loss_alpha=args.logits_load_kd_loss_alpha,
            ignore_errors=args.logits_load_ignore_errors,
        )
        loss, num_tokens, report = loss_func_cached_logits(loss_mask, output_tensor, model=model)
    elif has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
        if input_masked_positions is not None and getattr(model, 'training', True):
            report.update(
                build_input_masking_loss_report(losses, loss_mask, input_masked_positions)
            )

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def _input_masking_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    loss_mask,
    runtime_gather_output,
    compute_language_model_loss,
    scale_logits,
    context,
    **_,
):
    """Compute the normal LM loss and detached mask-offset metrics.

    Top-k candidates are selected only at masked positions, and only compact
    correctness tensors survive beyond this function. The returned loss is
    identical to the default GPT postprocess path.
    """
    logits, _ = output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )
    logits = scale_logits(logits)
    losses = compute_language_model_loss(labels, logits)

    mask_offsets = context["mask_offsets"]
    selected = (mask_offsets > 0) & loss_mask.bool()
    selected_sequence_first = selected.transpose(0, 1).contiguous()
    selected_logits = logits.detach()[selected_sequence_first]
    selected_labels = labels.transpose(0, 1).contiguous()[selected_sequence_first]

    logits_are_vocab_sharded = (
        not runtime_gather_output
        if runtime_gather_output is not None
        else not getattr(output_layer, "gather_output", False)
    )
    selected_top1, selected_top5 = compute_token_topk_correct(
        selected_logits,
        selected_labels,
        logits_are_vocab_sharded=logits_are_vocab_sharded,
        tp_group=context["tp_group"],
        max_k=5,
    )
    top1_correct = torch.zeros_like(selected_sequence_first)
    top5_correct = torch.zeros_like(selected_sequence_first)
    top1_correct[selected_sequence_first] = selected_top1
    top5_correct[selected_sequence_first] = selected_top5

    InputMaskingMetricsLoggingHelper.save_metrics(
        losses=losses,
        loss_mask=loss_mask,
        mask_offsets=mask_offsets,
        top1_correct=top1_correct.transpose(0, 1).contiguous(),
        top5_correct=top5_correct.transpose(0, 1).contiguous(),
        max_offsets=context["max_offsets"],
        reduce_group=context["reduce_group"],
    )
    return losses


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            attention_mask,
            cu_seqlens,
            cu_seqlens_padded,
            hybrid_cp_group,
            _input_mask_original_tokens,
            input_masked_positions,
            labels,
            local_cp_size,
            loss_mask,
            max_seqlen,
            position_ids,
            tokens,
        ) = get_batch(data_iterator, vp_stage)

    packed_seq_params = None
    if cu_seqlens is not None:
        # Squeeze the batch dim: the batch dict keeps cu_seqlens as (1, N)
        # for consistency, but PackedSeqParams and TE expect 1-D.
        cu_seqlens = cu_seqlens.squeeze(0)
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
        # Use real (unpadded) cu_seqlens to feed the FLOPs accounting: varlen
        # attention only computes work for real tokens within each chunk.
        update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        cu_seqlens_for_params = (
            cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
        )  # TODO(asolergi-nv): Currently there is a bug forcing cu_seqlens to be cu_seqlens_padded
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_for_params,
            cu_seqlens_kv=cu_seqlens_for_params,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=int(max_seqlen.item()),
            max_seqlen_kv=int(max_seqlen.item()),
            local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
            cp_group=hybrid_cp_group,
            tokens_per_sample=args.seq_length,
        )

    timers('batch-generator').stop()

    output_processor = None
    output_processor_context = None
    if input_masked_positions is not None and get_attr_wrapped_model(model, "training"):
        output_processor = _input_masking_output_processor
        output_processor_context = {
            "mask_offsets": input_masked_positions,
            "max_offsets": args.seq_length,
            "tp_group": mpu.get_tensor_model_parallel_group(),
            "reduce_group": mpu.get_data_parallel_group(with_context_parallel=True),
        }

    with stimer:
        if return_schedule_plan:
            assert (
                args.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                output_processor=output_processor,
                output_processor_context=output_processor_context,
            )
            return schedule_plan, partial(
                loss_func,
                loss_mask,
                model=model,
                input_masked_positions=input_masked_positions,
            )
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                output_processor=output_processor,
                output_processor_context=output_processor_context,
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(
        loss_func,
        loss_mask,
        model=model,
        input_masked_positions=input_masked_positions,
    )


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    """Whether the dataset should be built on the current rank."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    """Build the GPT (or FIM) dataset config from parsed CLI args."""
    tokenizer = build_tokenizer(args)

    input_mask_token_id = None
    if args.input_mask_ratio > 0.0:
        if args.input_mask_token is None:
            raise ValueError("--input-mask-token is required when --input-mask-ratio is positive")
        if args.sft:
            raise ValueError("input token masking is currently supported only for GPT pretraining")
        if getattr(args, 'modelopt_enabled', False):
            raise ValueError("input token masking metrics are not supported with ModelOpt")

        vocab = tokenizer.vocab
        if isinstance(vocab, dict):
            input_mask_token_id = vocab.get(args.input_mask_token)
        else:
            try:
                input_mask_token_id = vocab.index(args.input_mask_token)
            except ValueError:
                input_mask_token_id = None
        if input_mask_token_id is None:
            raise ValueError(
                f"input mask token {args.input_mask_token!r} is not in the tokenizer vocabulary; "
                "choose an existing reserved token (it will not be added automatically)"
            )
        # Masking is applied directly to already-tokenized samples using this
        # vocabulary ID. Do not validate by tokenizing the string: tokenizer
        # wrappers may legitimately add BOS/EOS IDs around otherwise atomic
        # tokens, and no text encoding occurs in the masking data path.
        if input_mask_token_id >= args.padded_vocab_size:
            raise ValueError(
                f"input mask token id {input_mask_token_id} is outside padded vocabulary size "
                f"{args.padded_vocab_size}"
            )

        protected_token_ids = set()
        for attribute in ('eod', 'bos', 'pad'):
            try:
                token_id = getattr(tokenizer, attribute)
            except (AttributeError, NotImplementedError):
                continue
            if token_id is not None:
                protected_token_ids.add(token_id)
        if input_mask_token_id in protected_token_ids:
            raise ValueError(
                f"input mask token {args.input_mask_token!r} resolves to protected token id "
                f"{input_mask_token_id}"
            )
        args.input_mask_token_id = input_mask_token_id
        print_rank_0(
            "> input token masking: "
            f"ratio={args.input_mask_ratio}, strategy={args.input_mask_strategy}, "
            f"span_length={args.input_mask_span_length}, token={args.input_mask_token!r}, "
            f"token_id={input_mask_token_id}, vocab_size={tokenizer.vocab_size}, "
            f"padded_vocab_size={args.padded_vocab_size}"
        )

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
        "inter_document_masking": args.dataloader_inter_document_masking,
        "input_mask_ratio": args.input_mask_ratio,
        "input_mask_strategy": args.input_mask_strategy,
        "input_mask_span_length": args.input_mask_span_length,
        "input_mask_token_id": input_mask_token_id,
        "input_mask_debug": args.input_mask_debug,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(
        is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence
    )
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    print_rank_0(f'> PyTorch version ................ {get_torch_version()}')
    print_rank_0(f'> Megatron-Core version .......... {mcore_version}')
    print_rank_0(f'> Transformer Engine version ... {get_te_version()}')

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    if has_nvidia_modelopt:
        maybe_enable_modelopt(args)
    if has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt_config_from_args(args, model_config_cls=ModelOptModelConfig)
    else:
        model_cfg = gpt_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
