# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by multiple model-runner families."""

import bisect
import contextlib
import math
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .interface import ScheduledInputs

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.sampler.sampler import SampleStateTensors


def get_all_rank_num_tokens(
    attn_metadata: AttentionMetadata,
    *,
    enable_attention_dp: bool,
    mapping: Mapping,
    dist: Distributed | None,
) -> list[int] | None:
    if enable_attention_dp:
        assert dist is not None, "attention DP requires a distributed communicator"
        num_tokens = attn_metadata.num_tokens
        if mapping.has_cp_helix():
            # With CP, attention uses reduce-scatter to divide tokens
            # among CP ranks. Report the post-RS token count.
            # Use tp_cp_allgather so MoE (which sees the repurposed
            # mapping where tp_size = original tp * cp) can index
            # with its tp_rank.
            num_tokens = math.ceil(num_tokens / mapping.cp_size)
            return dist.tp_cp_allgather_int64([num_tokens])[:, 0].tolist()
        return dist.tp_allgather_int64([num_tokens])[:, 0].tolist()
    return None


def get_all_rank_ctx_requests(
    num_ctx_requests: int,
    *,
    enable_attention_dp: bool,
    dist: Distributed | None,
) -> list[int] | None:
    if enable_attention_dp:
        assert dist is not None, "attention DP requires a distributed communicator"
        return dist.tp_allgather_int64([num_ctx_requests])[:, 0].tolist()
    return None


def get_padding_params(
    total_num_tokens: int,
    num_ctx_requests: int,
    attn_all_rank_num_tokens: list[int] | None,
    *,
    dist: Distributed | None,
    enable_attention_dp: bool,
    prefill_cuda_graph_backend: PrefillCudaGraphBackend,
    prefill_cuda_graph_num_tokens: list[int],
) -> tuple[int, bool, list[int] | None]:
    """
    Get the padding parameters for tensor padding.
    Return:
        padded_num_tokens: the padded number of tokens
        can_run_prefill_cuda_graph: whether a prefill CUDA graph can run
        attn_all_rank_num_tokens: the number of tokens for each rank
    """

    def get_padded_prefill_tokens(tokens: int) -> int:
        return prefill_cuda_graph_num_tokens[
            bisect.bisect_left(prefill_cuda_graph_num_tokens, tokens)
        ]

    if (
        prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
        and prefill_cuda_graph_num_tokens
    ):
        all_rank_ctx_requests = get_all_rank_ctx_requests(
            num_ctx_requests,
            enable_attention_dp=enable_attention_dp,
            dist=dist,
        )
        max_captured_num_tokens = prefill_cuda_graph_num_tokens[-1]
        if attn_all_rank_num_tokens is not None:
            has_ctx_requests = num_ctx_requests != 0 or (
                all_rank_ctx_requests is not None
                and any(ctx_requests != 0 for ctx_requests in all_rank_ctx_requests)
            )
            can_run_prefill_cuda_graph = (
                has_ctx_requests and max(attn_all_rank_num_tokens) <= max_captured_num_tokens
            )
            if can_run_prefill_cuda_graph:
                padded_num_tokens = get_padded_prefill_tokens(max(attn_all_rank_num_tokens))
                logger.debug(
                    f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens"
                )
                return padded_num_tokens, True, [padded_num_tokens] * len(attn_all_rank_num_tokens)
            else:
                logger.debug("Not all ranks can run prefill CUDA graph, disable prefill CUDA graph")
                return total_num_tokens, False, attn_all_rank_num_tokens
        elif num_ctx_requests != 0 and total_num_tokens <= max_captured_num_tokens:
            padded_num_tokens = get_padded_prefill_tokens(total_num_tokens)
            logger.debug(f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens")
            return padded_num_tokens, True, None
        else:
            logger.debug(
                f"Prefill CUDA graph cannot be used with {total_num_tokens} tokens, "
                f"{num_ctx_requests} context requests"
            )
            return total_num_tokens, False, None

    return total_num_tokens, False, attn_all_rank_num_tokens


def prepare_multimodal_indices(
    input_ids: list[int],
    *,
    model: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(input_ids, dtype=torch.int, device="cpu")
    vocab_size = model.config.vocab_size
    # `multimodal_token_ids` is the common wrapper-model contract. Keep the legacy name as a
    # fallback for models not yet migrated to `MultimodalModelMixin`.
    mm_token_ids = getattr(model, "multimodal_token_ids", None)
    if mm_token_ids is None:
        mm_token_ids = getattr(model, "mm_token_ids", None)

    text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
        input_ids, vocab_size=vocab_size, mm_token_ids=mm_token_ids
    )
    return text_token_indices, mm_token_indices


def get_top_level_model(model: Any) -> Any:
    model = getattr(model, "_orig_mod", model)
    top_level_model = getattr(model, "model", model)
    return getattr(top_level_model, "_orig_mod", top_level_model)


def get_position_id_offset(model: Any) -> int:
    offset = getattr(get_top_level_model(model), "position_id_offset", 0)
    return 0 if offset is None else int(offset)


def apply_position_id_offset(position_ids: list[int], *, model: Any) -> list[int]:
    offset = get_position_id_offset(model)
    if offset == 0:
        return position_ids
    return [position_id + offset for position_id in position_ids]


def ship_multimodal_indices(
    inputs: dict[str, Any],
    *,
    mm_token_indices_cpu: torch.Tensor,
    text_token_indices_cpu: torch.Tensor,
    num_ctx_tokens: int,
    total_num_tokens: int,
) -> None:
    """Pin and async-copy executor-precomputed MM/text token indices into
    ``inputs`` so ``fuse_input_embeds`` can skip its ``torch.where`` host
    sync. If ``total_num_tokens > num_ctx_tokens`` (KV-cache path with
    extend/draft tokens appended after the indices were computed), the
    post-context positions are appended as text. Current speculative decode
    paths do not append multimodal placeholders after the context tokens."""
    mm_token_indices_cpu = maybe_pin_memory(mm_token_indices_cpu)
    inputs["mm_token_indices"] = mm_token_indices_cpu.to("cuda", non_blocking=True)
    if total_num_tokens > num_ctx_tokens:
        extra_text = torch.arange(
            num_ctx_tokens,
            total_num_tokens,
            dtype=text_token_indices_cpu.dtype,
        )
        text_token_indices_cpu = torch.cat([text_token_indices_cpu, extra_text])
    text_token_indices_cpu = maybe_pin_memory(text_token_indices_cpu)
    inputs["text_token_indices"] = text_token_indices_cpu.to("cuda", non_blocking=True)


def make_scheduled_inputs(
    batch: ScheduledRequests,
    new_tensors_device: "SampleStateTensors | None",
    cache_indirection_buffer: torch.Tensor | None,
    *,
    enable_spec_decode: bool,
    runtime_draft_len: int,
) -> ScheduledInputs:
    """Build a scheduled record that gathers context logits when any request returns them."""
    return ScheduledInputs(
        batch=batch,
        new_tensors_device=new_tensors_device,
        cache_indirection_buffer=cache_indirection_buffer,
        gather_context_logits=any(
            request.py_return_context_logits for request in batch.context_requests
        ),
        enable_spec_decode=enable_spec_decode,
        runtime_draft_len=runtime_draft_len,
    )


def resolve_mrope_position_deltas_cache(model: torch.nn.Module | None) -> torch.Tensor | None:
    """The MRoPE delta cache held by ``model`` or by its draft model.

    ``None`` for every model that does not keep one, which is also how
    ``should_enable_overlap_headroom`` learns that the seat pool may be widened:
    the cache is sized from ``max_num_tokens`` rather than from the seat pool.
    """
    cache = getattr(model, "mrope_position_deltas_cache", None)
    if cache is None:
        cache = getattr(getattr(model, "draft_model", None), "mrope_position_deltas_cache", None)
    return cache


@contextlib.contextmanager
def moe_a2a_steady_state_budget_for_capture():
    """Force the steady-state MoE all-to-all budget across CUDA-graph capture.

    The budget is a kernel launch argument, so it is frozen into each captured
    graph. Capture happens inside the warmup window, so without this a replay
    would keep warmup's relaxed deadline for the life of the process.
    """
    set_moe_a2a_warmup(False)
    try:
        yield
    finally:
        set_moe_a2a_warmup(True)


def set_moe_a2a_warmup(in_warmup: bool) -> None:
    """Select the MoE all-to-all completion-flag budget for the current phase.

    No-op when the op is unavailable (older bindings).
    """
    try:
        torch.ops.trtllm.moe_a2a_set_warmup(in_warmup)
        logger.info(f"moe_a2a completion-flag budget: in_warmup={in_warmup}")
    except (AttributeError, RuntimeError) as e:
        logger.warning(
            f"moe_a2a_set_warmup unavailable, the all-to-all timeout "
            f"budget was not switched: {type(e).__name__}: {e}"
        )
