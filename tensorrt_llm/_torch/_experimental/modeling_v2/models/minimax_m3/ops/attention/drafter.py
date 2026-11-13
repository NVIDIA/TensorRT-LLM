# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common Eagle3 generation preprocessing and full-context plan execution.

Targets retain their geometry, fused-RoPE eligibility and native launch policy.
The bound full-context module retains its own plan dictionary and lock.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, cast

import torch

from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
    get_multi_processor_count_for_device,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode


class DrafterFullCtxBase(TrtllmGenFmha):
    """Shared execution; target subclasses provide their policy and module bindings."""

    FULLCTX: ModuleType
    GEOMETRY_ERROR: str
    SERVES_MLA = False
    # The C++ preprocess addresses the KV pool by its block extent.
    NEEDS_BLOCK_EXTENT = True
    supports_skip_correction = False
    supports_variable_window = False
    _multi_processor_count: Optional[int] = None

    def prepare_workspace(self, q, k, v, metadata, forward_args, workspace) -> None:
        """Grow the workspace to the C++ generation preprocess's layout for this call."""
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        if self._multi_processor_count is None:
            index = q.device.index if q.device.index is not None else torch.cuda.current_device()
            self._multi_processor_count = get_multi_processor_count_for_device(index)
        attn = self.attn
        max_num_sequences = metadata.max_num_sequences or metadata.max_num_requests
        layout = thop.get_trtllm_gen_generation_workspace_layout(
            q.dtype,
            max_num_sequences,
            max(q.size(0), max_num_sequences),
            attn.num_heads,
            attn.head_dim,
            attn.rope_dim,
            attn.num_kv_heads,
        )
        required = int(layout["total_size"])
        if workspace.numel() * workspace.element_size() < required:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The drafter workspace is smaller than the generation preprocess needs."
                )
            workspace.resize_((-(-required // workspace.element_size()),))

    def _shape(self, q: torch.Tensor, metadata) -> Optional[Tuple[int, int]]:
        nreq = int(metadata.num_generations)
        if nreq <= 0 or int(q.shape[0]) % nreq:
            return None
        return int(q.shape[0]) // nreq, nreq

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        if phase == FmhaPhase.CONTEXT or metadata.num_contexts > 0:
            return False
        if phase is None and forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        attn = self.attn
        if (
            attn.num_heads != self.FULLCTX.HEADS
            or attn.num_kv_heads != self.FULLCTX.KV_HEADS
            or attn.head_dim != self.FULLCTX.HEAD_DIM
            or not QuantMode(attn.quant_mode).has_fp8_kv_cache()
        ):
            self._on_uncovered(self.GEOMETRY_ERROR)
            return False
        # The arithmetic matches the stock decode only at unit KV scales (bmm1 and bmm2 carry none here); the layer's
        # own scales are checked to be 1.0 at install, so any other per-step scale falls through to the stock path.
        if (
            forward_args.kv_scale_orig_quant is not attn.kv_scale_orig_quant
            or forward_args.kv_scale_quant_orig is not attn.kv_scale_quant_orig
        ):
            self._on_uncovered("per-step KV cache scales")
            return False
        if int(metadata.max_seq_len) > self._max_kv_len:
            self._on_uncovered("the cache length bound is beyond the plans' K/V capacity")
            return False
        shape = self._shape(q, metadata)
        if shape is None or shape not in self._recipes:
            self._on_uncovered(f"{shape} is not a covered (query rows, requests) shape")
            return False
        # Compiled before warmup (``install``); the forward never compiles one.
        if self.FULLCTX.plan_for(q.device, *shape, self._own_reducer) is None:
            self._on_missed(shape[0] * shape[1], "the PrimTS decode plan is not prebuilt")
            return False
        return super()._is_supported(q, k, v, metadata, forward_args, phase=FmhaPhase.GENERATION)

    @staticmethod
    def _multi_token(params: FmhaParams) -> bool:
        return (
            params.spec_decoding_generation_lengths is not None
            and int(params.attn.predicted_tokens_per_seq) > 1
        )

    def _run_fused_rope(self, params: FmhaParams, attention_input: torch.Tensor) -> bool:
        """The pass with the one-launch preprocess; False (nothing run) where it does not apply."""
        if not self._fused_rope_applies(params, attention_input):
            return False
        attn = params.attn
        meta = params.meta
        qlen, nreq = int(params.input_seq_length), int(params.batch_size)
        plan = self.FULLCTX.plan_for(attention_input.device, qlen, nreq, self._own_reducer)
        if plan is None:
            return False
        # The preprocess's view of the layer's cache (buildFlashinferTrtllmGenPagedKvCacheBuffers): the pool's primary
        # pointer plus the layer's offset inside its pool, sub-pages of Hkv x tokens_per_block x 128 E4M3 bytes; the
        # block table is the pool's [requests, 2, blocks] K / V sub-page indices.
        pool_index, layer_in_pool = (
            int(x) for x in meta.host_kv_cache_pool_mapping[attn.local_layer_idx, :2]
        )
        tpb = int(params.tokens_per_block)
        sub_page_bytes = attn.num_kv_heads * tpb * attn.head_dim
        base = (
            int(meta.host_kv_cache_pool_pointers[pool_index, 0])
            + layer_in_pool * int(params.kv_factor) * sub_page_bytes
        )
        pages = (int(params.total_num_blocks) - 1) * int(params.kv_factor) + 2
        kv_pages = convert_to_torch_tensor(
            TensorWrapper(base, torch.float8_e4m3fn, [pages, attn.num_kv_heads, tpb, attn.head_dim])
        )
        tables = meta.kv_cache_block_offsets[
            pool_index, params.seq_offset : params.seq_offset + nreq
        ]
        multi = self._multi_token(params)
        if not torch.cuda.is_current_stream_capturing():
            # The decode reads V page p as sub-page p + 1 of K's (kv_pages[1:] with the K table), as the preprocess
            # path checks; the op itself writes V at the table's V sub-page.
            tpb_held = (params.sequence_lengths[:nreq].to(torch.long) + tpb - 1) // tpb
            cols = torch.arange(tables.shape[-1], device=tables.device)
            in_use = cols[None, :] < tpb_held[:, None]
            offsets = (tables[:, 1] - tables[:, 0])[in_use].unique()
            if offsets.numel() != 1 or int(offsets) != 1:
                raise RuntimeError(
                    f"the drafter cache does not keep V one sub-page after K: {offsets.tolist()}"
                )
        q = torch.empty(
            int(params.num_tokens),
            attn.num_heads,
            attn.head_dim,
            dtype=torch.float8_e4m3fn,
            device=attention_input.device,
        )
        self._launch_rope(
            attention_input,
            attn.rotary_cos_sin,
            params.sequence_lengths,
            params.spec_decoding_position_offsets if multi else None,
            kv_pages,
            tables,
            q,
            qlen,
            attn.num_heads,
            attn.num_kv_heads,
            int(attn.rope_params.dim),
            tpb,
            self._fused_rope_mode,
            self._pdl,
        )
        plan.run(
            q,
            kv_pages[:-1],
            kv_pages[1:],
            tables[:, 0],
            params.sequence_lengths[:nreq],
            get_bmm1_scale(attn),
            params.output.view(params.num_tokens, attn.num_heads, attn.head_dim),
        )
        if self._on_rope_taken is not None:
            self._on_rope_taken(params.num_tokens, "spec" if multi else "plain")
        self._on_taken(params.num_tokens)
        return True

    def run_generation(self, params: FmhaParams) -> None:
        attention_input = params.qkv_input if params.qkv_input is not None else params.query_input
        if attention_input is None:
            raise RuntimeError("the drafter decode requires QKV or query input.")
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        if self._run_fused_rope(params, attention_input):
            return
        # The BF16 post-RoPE query, rounded to E4M3 below. At the unit KV scales this layer runs with, that is the FP8
        # query the stock decode gets (the preprocess's FP8 variant is e4m3(q * kv_scale_orig_quant)).
        fp8_context_fmha = False
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            _fmha_workspace,
            _cu_seqlens,
            _max_q_len,
            _max_kv_len,
            _window_left,
            is_multi_token_gen,
        ) = thop.trtllm_gen_generation_preprocess(
            attention_input,
            params.workspace,
            params.sequence_lengths,
            params.spec_decoding_generation_lengths,
            params.spec_decoding_position_offsets,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_position_deltas,
            attn.local_layer_idx,
            params.seq_offset,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            get_bmm1_scale(attn),
            1.0,
            fp8_context_fmha,
            attn.predicted_tokens_per_seq,
            get_attention_chunk_size(attn),
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            params.is_cross,
        )
        qlen, nreq = params.input_seq_length, params.batch_size
        plan = self.FULLCTX.plan_for(attention_input.device, qlen, nreq, self._own_reducer)
        # A first draft pass arrives as qlen rows per request either as a plain multi-row generation step (the
        # drafter's own, input_seq_length = qlen) or with spec-decoding lengths (multi-token generation); a 1-row
        # pass is never multi-token.
        if plan is None or (is_multi_token_gen and qlen == 1):
            raise RuntimeError(
                f"the drafter decode was selected for {qlen} query rows x {nreq} requests without a uniform plan"
            )
        rows = params.num_tokens * attn.num_heads * attn.head_dim
        q = fp8_e4m3_saturate(q_processed.flatten()[:rows])
        q = q.view(params.num_tokens, attn.num_heads, attn.head_dim)
        # K pages in row 0 of the table, V in row 1; this cache keeps V one sub-page after K, so V page p is page p + 1
        # of the pool (the two views have the same page count, as the kernel requires).
        if not torch.cuda.is_current_stream_capturing():
            # The plan runs every request at qlen rows; a multi-token pass must give each request exactly that many.
            lengths = params.spec_decoding_generation_lengths
            if is_multi_token_gen and not bool((lengths[:nreq] == qlen).all()):
                raise RuntimeError(
                    f"the drafter pass rows per request are not uniform: {lengths[:nreq].tolist()}"
                )
            # Only the blocks a request holds are laid out; the table's tail entries are unset.
            tpb = int(params.tokens_per_block)
            held = (params.sequence_lengths[:nreq].to(torch.long) + tpb - 1) // tpb
            cols = torch.arange(block_tables.shape[-1], device=block_tables.device)
            in_use = cols[None, :] < held[:, None]
            offsets = (block_tables[:nreq, 1] - block_tables[:nreq, 0])[in_use].unique()
            if offsets.numel() != 1 or int(offsets) != 1:
                raise RuntimeError(
                    f"the drafter cache does not keep V one sub-page after K: {offsets.tolist()}"
                )
        # The preprocess sizes kv_pool by blocks; its table indexes the K and V sub-pages of every block.
        # The view stops two sub-pages past the last block's K, as get_kv_subpage_pool's does.
        pages = (int(params.total_num_blocks) - 1) * int(params.kv_factor) + 2
        kv_pages = convert_to_torch_tensor(
            TensorWrapper(kv_pool.data_ptr(), kv_pool.dtype, [pages, *kv_pool.shape[1:]])
        )
        plan.run(
            q,
            kv_pages[:-1],
            kv_pages[1:],
            block_tables[:nreq, 0],
            params.sequence_lengths[:nreq],
            get_bmm1_scale(attn),
            params.output.view(params.num_tokens, attn.num_heads, attn.head_dim),
        )
        self._on_taken(params.num_tokens)


def install(
    draft_model: torch.nn.Module,
    recipes,
    max_kv_len: int,
    own_reducer: bool,
    pdl: bool,
    on_taken,
    on_uncovered,
    on_missed,
    fused_rope_mode: Optional[int] = None,
    on_rope_taken=None,
    *,
    prebuild: Callable[..., None],
    fmha_class: type[DrafterFullCtxBase],
) -> int:
    """Prebuild the plans of ``recipes`` and put ``M3DrafterFullCtxFmha`` first in the FMHA list of the TrtllmAttention
    backend of every attention module of ``draft_model``. Call before warmup. Returns the number of attention layers it
    was installed on."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    count = 0
    device = next(draft_model.parameters()).device
    for module in draft_model.modules():
        backend = getattr(module, "attn", None)
        if not isinstance(backend, TrtllmAttention) or not hasattr(backend, "_fmha_manager"):
            continue
        if not bool((backend.kv_cache_scaling_factor == 1.0).all()):
            on_uncovered("a scaled FP8 KV cache")
            continue
        prebuild(device, recipes, max_kv_len, own_reducer, pdl)
        libs = backend._fmha_manager.fmha_libs
        if not any((isinstance(lib, fmha_class) for lib in libs)):
            libs.insert(
                0,
                fmha_class(
                    backend,
                    recipes,
                    max_kv_len,
                    own_reducer,
                    on_taken,
                    on_uncovered,
                    on_missed,
                    fused_rope_mode=fused_rope_mode,
                    pdl=pdl,
                    on_rope_taken=on_rope_taken,
                ),
            )
            count += 1
    return count


class DrafterBindings(Protocol):
    PositionEmbeddingType: Any
    fwd_has_no_mrope: Callable[..., bool]


class MultiRequestDrafterBindings(DrafterBindings, Protocol):
    ROPE_ONE_REQUEST_MULTI: bool


def initialize_drafter_state(
    self: Any,
    state: DrafterBindings,
    /,
    recipes: Mapping[Tuple[int, int], Mapping],
    max_kv_len: int,
    own_reducer: bool,
    on_taken: Any,
    on_uncovered: Any,
    on_missed: Any,
    fused_rope_mode: Optional[int] = None,
    pdl: bool = False,
    on_rope_taken: Any = None,
    *,
    has_one_request_multi: bool,
) -> None:
    self._recipes = recipes
    self._max_kv_len = max_kv_len
    self._own_reducer = own_reducer
    self._fused_rope_mode = fused_rope_mode
    self._pdl = pdl
    self._on_rope_taken = on_rope_taken
    if has_one_request_multi:
        self._one_request_multi = cast(MultiRequestDrafterBindings, state).ROPE_ONE_REQUEST_MULTI
    self._on_taken = on_taken
    self._on_uncovered = on_uncovered
    self._on_missed = on_missed


def drafter_rope_applies(
    self: Any,
    state: DrafterBindings,
    /,
    params: FmhaParams,
    attention_input: torch.Tensor,
    *,
    has_one_request_multi: bool,
) -> bool:
    attn = params.attn
    width = (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim
    rotary_dim = int(attn.rope_params.dim)
    return (
        self._fused_rope_mode is not None
        and params.qkv_input is not None
        and (attention_input.dim() == 2)
        and (attention_input.dtype == torch.bfloat16)
        and attention_input.is_contiguous()
        and (int(attention_input.shape[1]) == width)
        and (not params.is_cross)
        and (int(attn.position_embedding_type) == int(state.PositionEmbeddingType.rope_gpt_neox))
        and (attn.rotary_cos_sin is not None)
        and (attn.rotary_cos_sin.dtype == torch.float32)
        and state.fwd_has_no_mrope(params.fwd)
        and (0 < rotary_dim <= attn.head_dim)
        and (rotary_dim % 16 == 0)
        and (int(params.num_tokens) == int(params.batch_size) * int(params.input_seq_length))
        and (
            has_one_request_multi
            and self._one_request_multi
            or not (self._multi_token(params) and int(params.batch_size) == 1)
        )
    )
