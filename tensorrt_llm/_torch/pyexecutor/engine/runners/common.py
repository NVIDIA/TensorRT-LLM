# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by multiple model-runner families."""

import bisect
import math
from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


def filter_cuda_graph_batch_sizes(
    cuda_graph_batch_sizes: list[int],
    max_batch_size: int,
    max_num_tokens: int,
    tokens_per_request: int,
    enable_padding: bool,
) -> list[int]:
    """Drop graph batch sizes that exceed the request or token budget."""
    max_cuda_graph_batch_size = min(
        max_batch_size,
        max_num_tokens // tokens_per_request,
    )
    if max_cuda_graph_batch_size < 1:
        return []

    result: list[int] = []
    for index, batch_size in enumerate(cuda_graph_batch_sizes):
        if batch_size <= max_cuda_graph_batch_size:
            result.append(batch_size)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_cuda_graph_batch_size):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given CUDA "
                f"graph batch sizes ({batch_size}) is larger than the "
                f"executor's max batch size ({max_cuda_graph_batch_size}). "
                f"We will pad batches to {max_cuda_graph_batch_size}."
            )
            result.append(max_cuda_graph_batch_size)
        break
    return result


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


def set_spec_metadata_all_rank_num_tokens(
    spec_metadata: SpecMetadata,
    spec_all_rank_num_tokens: list[int],
    all_rank_num_seqs: list[int],
    all_rank_num_gens: list[int] | None = None,
) -> None:
    # Eagle3 / MTP-eagle one-model use subseq_all_rank_num_tokens for
    # draft loop iterations i>0 (per-sequence counts, since each
    # sequence contributes one token per iteration).
    spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
    spec_metadata.all_rank_num_seqs = all_rank_num_seqs
    # DSpark can draft only after the target processes the current bonus token,
    # because it consumes captured target-layer hidden states for that token.
    # Prefill computes hidden states for prompt tokens; the first generated token
    # is sampled from the last prompt logits and has not itself passed through the
    # target layers. Thus context requests seed the rolling window but do not run
    # the draft. On mixed steps, num_seqs therefore over-counts the draft MoE
    # workload; gen-only per-rank counts keep the FUSED_COMM (DeepGEMM MegaMoE)
    # chunk loop identical across EP ranks.
    if all_rank_num_gens is not None:
        spec_metadata.all_rank_num_gens = all_rank_num_gens
    if (
        spec_metadata.spec_dec_mode.is_mtp_eagle_one_model()
        or spec_metadata.spec_dec_mode.is_eagle3_one_model()
    ):
        spec_metadata.subseq_all_rank_num_tokens = all_rank_num_seqs


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
