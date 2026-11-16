# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private integer metadata for stock Inferact B1/K7 preparation on SM100.

Only the nine output tensors mutate. Acceptance, embeddings, projection, MLA,
YaRN and cache/state updates retain the stock worker's order and implementation.
The caller owns the stock device-value invariants: accepted count0..8, valid
slot/token IDs and nonnegative lengths/capacities without signed-int64 overflow.
"""

from __future__ import annotations

import weakref

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM
from tensorrt_llm._torch.modules.linear import TensorParallelMode

from .inferact_boundary_kernel import Inputs, Outputs, _prepare_kernel

_OUTPUT_SPECS = (
    ((1,), torch.int64), ((2,), torch.int64), ((1, 7), torch.int64),
    ((1, 8), torch.int64), ((1,), torch.int64), ((1, 8), torch.int64),
    ((1, 8), torch.bool), ((1,), torch.int64), ((8,), torch.int64),
)


def _valid_constants(max_ctx: int, mask_token_id: int) -> bool:
    return (type(max_ctx) is int and 0 <= max_ctx <= 2**62 - 8
            and type(mask_token_id) is int and 0 <= mask_token_id < 2**31)


def _model_supported(draft_model, mapping) -> bool:
    """Admit the stock Inferact class and original target-capture convention."""
    if type(draft_model) is not MLADSparkForCausalLM:
        return False
    config = draft_model.config
    expected = {
        "model_type": "k3_dspark", "hidden_size": 7168, "intermediate_size": 14336,
        "num_hidden_layers": 5, "num_attention_heads": 64, "q_lora_rank": 1536,
        "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
        "v_head_dim": 128, "markov_rank": 256, "markov_head_type": "vanilla",
        "mla_use_nope": False, "mla_use_output_gate": False,
    }
    return (
        mapping.world_size == 16 and mapping.tp_size == 16
        and mapping.pp_size == 1 and mapping.cp_size == 1
        and not mapping.enable_attention_dp
        and draft_model.model_config.mapping == mapping
        and draft_model.model_config.torch_dtype == torch.bfloat16
        and all(getattr(config, key, None) == value for key, value in expected.items())
        and getattr(config, "target_layer_ids", None) == [2, 23, 47, 71, 89]
        and draft_model.dflash_attention_backend == "CUTEDSL"
        and draft_model._tp_split and draft_model._num_attn_layers == 5
        and draft_model._num_heads == 4 and draft_model._num_kv_heads == 1
        and draft_model._head_dim == 576 and draft_model._kv_factor == 1
        and get_sm_version() == 100
    )


def _vector(tensor, dtype, device, shape: tuple[int, ...] | None = None) -> bool:
    return (
        tensor is not None and tensor.is_cuda and tensor.device == device
        and tensor.dtype == dtype and tensor.is_contiguous()
        and (tuple(tensor.shape) == shape if shape is not None
             else tensor.ndim == 1 and tensor.numel() > 0)
    )


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    """Conservative physical span, including holes in a positive-strided view."""
    if tensor.numel() == 0 or any(stride < 0 for stride in tensor.stride()):
        raise ValueError("Expected a nonempty positive-strided protected tensor")
    elements = 1 + sum((size - 1) * stride
                       for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    begin = tensor.data_ptr()
    return begin, begin + elements * tensor.element_size()


def _disjoint_outputs(inputs, outputs) -> bool:
    protected = [_span(tensor) for tensor in inputs]
    for tensor in outputs:
        begin, end = _span(tensor)
        if any(begin < other_end and other_begin < end
               for other_begin, other_end in protected):
            return False
        protected.append((begin, end))
    return True


@torch.library.custom_op(
    "trtllm::inferact_boundary_prepare",
    mutates_args=("bonus", "lookup_ids", "qpos", "cpos", "n64", "columns",
                  "write_mask", "gen_rows", "offsets"),
)
def _prepare_into(
    accepted: torch.Tensor, count: torch.Tensor, slots: torch.Tensor,
    ctx_len: torch.Tensor, block_counts: torch.Tensor,
    bonus: torch.Tensor, lookup_ids: torch.Tensor, qpos: torch.Tensor,
    cpos: torch.Tensor, n64: torch.Tensor, columns: torch.Tensor,
    write_mask: torch.Tensor, gen_rows: torch.Tensor, offsets: torch.Tensor,
    max_ctx: int, mask_token_id: int,
) -> None:
    """Write caller-owned disjoint scratch; no input, cache or RNG mutation."""
    _prepare_kernel[(1,)](
        accepted, count, slots, ctx_len, block_counts,
        bonus, lookup_ids, qpos, cpos, n64, columns, write_mask, gen_rows, offsets,
        MAX_CTX=max_ctx, MASK_TOKEN=mask_token_id, num_warps=1,
    )


@_prepare_into.register_fake
def _prepare_into_fake(
    accepted: torch.Tensor, count: torch.Tensor, slots: torch.Tensor,
    ctx_len: torch.Tensor, block_counts: torch.Tensor,
    bonus: torch.Tensor, lookup_ids: torch.Tensor, qpos: torch.Tensor,
    cpos: torch.Tensor, n64: torch.Tensor, columns: torch.Tensor,
    write_mask: torch.Tensor, gen_rows: torch.Tensor, offsets: torch.Tensor,
    max_ctx: int, mask_token_id: int,
) -> None:
    return None


class _Scratch:
    """One stable bank for eager use or one retained graph-metadata instance."""

    def __init__(self, graph_metadata, device, max_ctx: int, mask_token_id: int) -> None:
        self.graph_metadata = graph_metadata
        self.device = device
        self.max_ctx = max_ctx
        self.mask_token_id = mask_token_id
        self.outputs = Outputs(*(torch.empty(shape, dtype=dtype, device=device)
                                 for shape, dtype in _OUTPUT_SPECS))
        self.warmed = False


class InferactBoundaryPreparation:
    """Construction-resolved admission and per-worker, per-graph scratch.

    CUDA-graph warmup uses the same retained SpecMetadata object as capture.
    A first use during capture falls back without allocation or kernel launch.
    Each graph gets its own outputs; eager use has a separate bank. Replay and
    eager forwards of a single worker must remain serialized, as required by
    the existing mutable context/cache and attention-metadata state. No global
    writable scratch or captured tensor pointer is replaced here.
    """

    def __init__(self, draft_model, mapping) -> None:
        self._model = weakref.ref(draft_model)
        self.model_supported = _model_supported(draft_model, mapping)
        self._scratch: dict[int | None, _Scratch] = {}

    def _admitted(
        self, worker, draft_model, attn_metadata, spec_metadata, accepted,
        count, slots, captured_hs, has_target_features: bool, prep,
        mask_token_id: int,
    ) -> bool:
        if not (
            self.model_supported and self._model() is draft_model and prep is None
            and has_target_features and attn_metadata.num_contexts == 0
            and attn_metadata.num_seqs == 1 and attn_metadata.num_ctx_tokens == 0
            and spec_metadata.runtime_draft_len == 7 and worker.max_draft_len == 7
            and worker._compute_block_size == 7 and worker._draft_tokens_per_req == 8
            and worker._ctx_paged and worker._ctx_page_size == 64
            and not worker._ctx_reads_rebased_view and not worker._ctx_pool_windowed
            and worker._dflash_attention_backend == "CUTEDSL"
            and worker._ctx_block_tables is not None and worker._ctx_kv_buf is not None
            and len(worker._ctx_kv_buf) == 5
            and _valid_constants(worker._max_ctx, mask_token_id)
        ):
            return False
        device = accepted.device
        embed = draft_model.draft_model_full.model.embed_tokens
        if not (
            getattr(embed, "tp_size", None) == 16
            and getattr(embed, "tp_mode", None) == TensorParallelMode.COLUMN
            and embed.weight.dtype == torch.bfloat16 and embed.weight.device == device
            and tuple(embed.weight.shape) == (10240, 7168)
            and _vector(accepted, torch.int32, device, (1, 8))
            and _vector(count, torch.int32, device, (1,))
            and _vector(slots, torch.int64, device, (1,))
            and _vector(worker._ctx_len, torch.int64, device)
            and _vector(worker._ctx_block_counts, torch.int64, device)
            and captured_hs is not None and captured_hs.is_cuda
            and captured_hs.device == device and captured_hs.dtype == torch.bfloat16
            and tuple(captured_hs.shape) == (8, 35840)
            and worker._ctx_block_tables.is_cuda
            and worker._ctx_block_tables.device == device
            and worker._ctx_block_tables.dtype == torch.int32
            and worker._ctx_block_tables.ndim == 2
            and worker._ctx_block_tables.shape[0] > 0
            and worker._ctx_block_tables.shape[1] > 0
            and worker._ctx_block_tables.stride(1) == 1
        ):
            return False
        return all(
            cache.is_cuda and cache.device == device and cache.dtype == torch.bfloat16
            and cache.ndim == 5 and cache.shape[0] > 0
            and tuple(cache.shape[1:]) == (1, 1, 64, 576)
            and all(stride >= 0 for stride in cache.stride())
            for cache in worker._ctx_kv_buf
        )

    def try_prepare(
        self, worker, draft_model, attn_metadata, spec_metadata, accepted,
        count, slots, captured_hs, has_target_features: bool, prep,
        mask_token_id: int,
    ) -> Outputs | None:
        if not self._admitted(
            worker, draft_model, attn_metadata, spec_metadata, accepted,
            count, slots, captured_hs, has_target_features, prep, mask_token_id,
        ):
            return None
        graph_metadata = spec_metadata if spec_metadata.is_cuda_graph else None
        key = None if graph_metadata is None else id(graph_metadata)
        capturing = torch.cuda.is_current_stream_capturing()
        scratch = self._scratch.get(key)
        if scratch is None:
            if capturing:
                return None
            scratch = _Scratch(graph_metadata, accepted.device, worker._max_ctx, mask_token_id)
            self._scratch[key] = scratch
        if (scratch.device != accepted.device or scratch.max_ctx != worker._max_ctx
                or scratch.mask_token_id != mask_token_id
                or scratch.graph_metadata is not graph_metadata
                or (capturing and not scratch.warmed)):
            return None
        inputs = Inputs(accepted, count, slots, worker._ctx_len, worker._ctx_block_counts)
        # Fresh output allocations cannot overlap live tensors. Recheck dynamic
        # inputs and protected views to reject accidental scratch recycling.
        protected = [*inputs, captured_hs, worker._ctx_block_tables,
                     *worker._ctx_kv_buf, draft_model.draft_model_full.model.embed_tokens.weight]
        mask_embedding = worker._trained_mask_embedding(draft_model, mask_token_id)
        if mask_embedding is not None:
            protected.append(mask_embedding)
        if not _disjoint_outputs(protected, scratch.outputs):
            return None
        _prepare_into(*inputs, *scratch.outputs, worker._max_ctx, mask_token_id)
        scratch.warmed = True
        return scratch.outputs

    @staticmethod
    def noise_embedding(embed_tokens, lookup_ids, trained_mask_embedding) -> torch.Tensor:
        """Stock B1 two-row TP lookup, trained-mask override and noise assembly."""
        combined_embed = embed_tokens(lookup_ids)
        embed_bonus = combined_embed[:1]
        embed_mask = combined_embed[1]
        if trained_mask_embedding is not None:
            embed_mask = trained_mask_embedding
        noise_embed_2d = embed_mask.expand(1, 7, -1).clone()
        noise_embed_2d[:, 0, :] = embed_bonus
        return noise_embed_2d
