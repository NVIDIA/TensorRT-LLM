# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention backends using the standard TRTLLM sparse contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PredefinedAttentionMask,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .indexer import CSA2Indexer
from .metadata import CSA2TrtllmMetadata
from .params import CSA2BackendForwardArgs, CSA2Mode, CSA2Params, select_csa2_backend

if TYPE_CHECKING:
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

_HEAD_DIM = 512
_SWA_TILE = 128
# The main pool's structural ceiling is 6 * 448 = 2688, which 448 * 2**3 already
# covers, so three binades de-saturate it outright. Context spends that exponent
# out of Q's low end (see ``_prepare_native_fp8`` below), and this caps what it
# may spend when a SWA group scale asks for more than the main pool ever can.
_CONTEXT_STAGE_SCALE_MAX_EXPONENT = 3
_CSA2_KV_CACHE_DTYPES = ("auto", "bfloat16", "fp8", "fp8_ds_mla")
_CSA2_FP8_KV_CACHE_DTYPES = ("fp8", "fp8_ds_mla")


def _csa2_kv_cache_algo(quant_config) -> QuantAlgo | None:
    """Return the requested KV-cache quantization, which CSA2 honors only as FP8 staging.

    The persistent CSA2 cache formats (FP8 SWA rows, NVFP4 main and index
    records) never change with the quantization config. FP8 is accepted as a
    request for E4M3 compute staging; any other KV-cache algorithm has no
    CSA2 meaning and is rejected.
    """
    algo = None if quant_config is None else quant_config.kv_cache_quant_algo
    if algo is None:
        return None
    if algo not in (QuantAlgo.FP8, QuantAlgo.FP8.value):
        raise ValueError(
            "CSA2 persistent cache formats are fixed; only FP8 KV-cache quantization is accepted"
        )
    return QuantAlgo.FP8


def _csa2_parent_quant_config(quant_config, native_fp8: bool):
    """Give the inherited trtllm-gen forward a KV algo that matches the staging dtype."""
    algo = QuantAlgo.FP8 if native_fp8 else None
    if quant_config is None:
        return QuantConfig(kv_cache_quant_algo=algo) if native_fp8 else None
    if quant_config.kv_cache_quant_algo == algo:
        return quant_config
    # Rebuild rather than copy: the quantization modes are cached properties.
    return QuantConfig(**{**quant_config.model_dump(), "kv_cache_quant_algo": algo})


class CSA2TrtllmAttention(TrtllmAttention):
    """CSA2 sparse prediction with the inherited native trtllm-gen forward."""

    Metadata = CSA2TrtllmMetadata
    compute_backend = "trtllm"

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int = 1,
        sparse_params: CSA2Params | None = None,
        mla_params: MLAParams | None = None,
        **kwargs,
    ) -> None:
        if head_dim != _HEAD_DIM:
            raise ValueError("CSA2 TRTLLM sparse MLA requires head_dim=512")
        if num_kv_heads != 1:
            raise ValueError("CSA2 TRTLLM sparse MLA requires one KV head")
        requested_dtype = kwargs.get("kv_cache_dtype", "auto")
        if requested_dtype not in _CSA2_KV_CACHE_DTYPES:
            raise ValueError("CSA2 KV cache dtype must be auto, bfloat16, fp8 or fp8_ds_mla")
        quant_config = kwargs.get("quant_config")
        requested_algo = _csa2_kv_cache_algo(quant_config)
        sparse_params = sparse_params or CSA2Params()
        self.compute_backend = (
            select_csa2_backend(get_sm_version())
            if sparse_params.compute_backend == "auto"
            else sparse_params.compute_backend
        )
        if self.compute_backend not in ("trtllm", "flash_mla", "flashinfer"):
            raise ValueError(f"Unknown CSA2 attention backend: {self.compute_backend}")
        if self.compute_backend == "trtllm" and not is_sm_100f():
            raise ValueError("CSA2 trtllm-gen requires an SM100-family GPU")
        # ``kv_cache_config.dtype`` selects the compute staging dtype only; an
        # FP8 KV-cache algorithm in the quantization config does the same when
        # the dtype is left at ``auto``. Only native trtllm-gen consumes E4M3
        # staging, whose per-tensor KV scale is derived per forward from the
        # packed pools' group scales, so that path additionally requires
        # ``use_fp8_staging``. The library adapters and packed attention read
        # BF16 staging or the persistent rows, so they keep BF16 and drop the
        # FP8 KV algo from the inherited config. A default ``use_fp8_staging``
        # follows that same support boundary, while an explicit request for
        # staging the backend cannot serve stays an error.
        fp8_requested = requested_dtype in _CSA2_FP8_KV_CACHE_DTYPES or (
            requested_dtype == "auto" and requested_algo is QuantAlgo.FP8
        )
        staging_supported = (
            self.compute_backend == "trtllm" and not sparse_params.use_packed_sparse_attention
        )
        if sparse_params.use_fp8_staging and not staging_supported:
            raise ValueError("CSA2 FP8 staging requires trtllm-gen without packed attention")
        use_fp8_staging = (
            staging_supported
            if sparse_params.use_fp8_staging is None
            else sparse_params.use_fp8_staging
        )
        native_fp8 = fp8_requested and use_fp8_staging
        self.staging_dtype = torch.float8_e4m3fn if native_fp8 else torch.bfloat16
        kwargs["kv_cache_dtype"] = "fp8" if self.native_fp8 else "auto"
        kwargs["quant_config"] = _csa2_parent_quant_config(quant_config, self.native_fp8)
        self.indexer = None
        if mla_params is None:
            mla_params = MLAParams(
                q_lora_rank=1280,
                kv_lora_rank=448,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                v_head_dim=512,
                rope_append=False,
            )
        if (
            mla_params.kv_lora_rank,
            mla_params.qk_nope_head_dim,
            mla_params.qk_rope_head_dim,
            mla_params.v_head_dim,
            mla_params.rope_append,
        ) != (448, 448, 64, 512, False):
            raise ValueError(
                "CSA2 TRTLLM MLA requires rank/nope 448, RoPE 64, V 512 and rope_append=False"
            )
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            sparse_params=sparse_params,
            mla_params=mla_params,
            **kwargs,
        )
        self.local_layer_idx = 0  # one physical compute-pool slot in this view

    @property
    def native_fp8(self) -> bool:
        """Whether trtllm-gen consumes E4M3 staging (the only FP8 compute path)."""
        return self.staging_dtype == torch.float8_e4m3fn

    def update_quant_config(self, new_quant_config):
        """Rebuild the shared dispatcher, then apply CSA2's provider policy."""
        from ...fmha.fallback import FallbackFmha

        # A later quantization config cannot change the persistent formats or
        # the staging dtype chosen at construction; only FP8 KV is meaningful
        # and it is normalized to the staging dtype actually in use.
        _csa2_kv_cache_algo(new_quant_config)
        super().update_quant_config(_csa2_parent_quant_config(new_quant_config, self.native_fp8))
        native = (
            self.compute_backend == "trtllm" and not self.sparse_params.use_packed_sparse_attention
        )
        self._fmha_manager.fmha_libs = [
            provider
            for provider in self._fmha_manager.fmha_libs
            if native and isinstance(provider, FallbackFmha)
        ]
        if native and not self._fmha_manager.fmha_libs:
            raise ValueError(
                "CSA2 native attention requires the FallbackFmha library to be enabled"
            )

    def forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        if not self.native_fp8:
            return super().forward(q, k, v, metadata, forward_args=forward_args, **kwargs)
        args = merge_attention_forward_args(forward_args, kwargs)
        if args.attention_input_type == AttentionInputType.context_only:
            if args.quant_scale_qkv is not None or args.quant_q_buffer is not None:
                raise ValueError("CSA2 FP8 context expects unquantized BF16 Q")
            # The native context quantizer needs quant_scale_qkv without a
            # caller quant_q_buffer (which would select fused-Q and skip it).
            # Inject the scale only after parent admission, in the sparse hook.
            # A fresh carrier keeps repeated caller args safe, including errors.
            args = replace(args)
        return super().forward(q, k, v, metadata, forward_args=args)

    def create_output(self, q, *, is_quantize_output, **kwargs):
        if self.native_fp8 and (is_quantize_output or q.dtype != torch.bfloat16):
            raise ValueError("CSA2 native FP8 attention requires BF16 output")
        return super().create_output(q, is_quantize_output=is_quantize_output, **kwargs)

    def prepare_sparse_inputs(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> CSA2BackendForwardArgs:
        if self.native_fp8 and (
            forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
            or forward_args.output_sf is not None
            or forward_args.enable_dsv4_epilogue_fusion
            or (forward_args.output is not None and forward_args.output.dtype != torch.bfloat16)
        ):
            raise ValueError("CSA2 native FP8 attention requires unquantized BF16 output")
        if metadata.swa_pool.dtype != self.staging_dtype:
            raise ValueError("CSA2 tile staging dtype must match its attention backend")
        inputs = forward_args.sparse_backend_args
        if not isinstance(inputs, CSA2BackendForwardArgs):
            raise TypeError("CSA2 requires CSA2BackendForwardArgs")
        count = metadata.num_tokens
        if (
            q.shape != (count, self.num_heads * _HEAD_DIM)
            or q.dtype != torch.bfloat16
            or metadata.num_query_heads != self.num_heads
        ):
            raise ValueError("CSA2 Q geometry/dtype must match its BF16 metadata")
        if forward_args.attention_input_type not in (
            AttentionInputType.context_only,
            AttentionInputType.generation_only,
        ):
            raise ValueError("CSA2 compute tiles must contain one attention phase")
        if inputs.state is not None:
            state = inputs.state
            source = state.metadata
            manager = source.kv_cache_manager
            layout = self.sparse_params.layout
            if layout is None:
                raise ValueError("CSA2 module inputs require the model layout")
            layer = layout.layer(self.layer_idx)
            start, end = inputs.query_start, inputs.query_start + count
            if start == 0:
                source.enter_layer(layer)
                global_rows = None
                if (
                    layer.mode == CSA2Mode.FULL
                    and self.layer_idx in source.csa2_precomputed_kv_layers
                ):
                    if state.main_kv is not None or state.index_k is not None:
                        raise ValueError("CED precomputed global KV must not be projected twice")
                elif layer.mode == CSA2Mode.FULL:
                    if state.main_kv is None or state.index_k is None:
                        raise ValueError(
                            "Full mode requires main and index rows, including empty rows"
                        )
                    global_rows = (
                        layer.kv_source,
                        source.csa2_main_write_slots[layer.kv_source],
                        state.main_kv,
                        state.index_k,
                        state.main_transform,
                        state.index_transform,
                    )
                elif state.main_kv is not None or state.index_k is not None:
                    raise ValueError("Only Full mode may write shared main/index caches")
                # The SWA rows, the global rows and the index-query packing are
                # one quantize launch; the indexer then consumes packed queries.
                pack_queries = (
                    layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX)
                    and state.index_q is not None
                    and state.index_q_scale is None
                )
                packed = manager.write_layer_rows(
                    self.layer_idx,
                    source.csa2_swa_write_slots[self.layer_idx],
                    state.swa_kv,
                    state.swa_transform,
                    global_rows=global_rows,
                    index_q=state.index_q if pack_queries else None,
                )
                if packed is not None:
                    state = replace(state, index_q=packed[0], index_q_scale=packed[1])
            logical = None
            if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
                if state.index_q is None or state.index_weights is None:
                    raise ValueError("Full/Reindex mode requires index queries and weights")
                if self.indexer is None:
                    self.indexer = CSA2Indexer(
                        layout,
                        self.layer_idx,
                        state.index_q.shape[1],
                        state.index_q.shape[-1] * (2 if state.index_q_scale is not None else 1),
                        options=self.sparse_params,
                    )
                # Indexer phases and TP query splitting use the complete model
                # batch. FMHA query tiles only consume the published selections.
                if start == 0:
                    self.indexer(state, 0, state.swa_kv.shape[0])
                logical = source.csa2_indices[self.layer_idx][start:end]
            elif layer.mode == CSA2Mode.REUSE:
                if layer.index_source not in source.csa2_indices:
                    raise ValueError("CSA2 index source did not run in this forward")
                indices = source.csa2_indices[layer.index_source]
                if indices.shape[0] != state.swa_kv.shape[0]:
                    raise ValueError("CSA2 Reuse rows must match their source's packed query order")
                logical = indices[start:end]
            slots = None
            mapping = None
            if logical is not None:
                visible = source.csa2_visible_lengths[self.layer_idx][start:end]
                if self._native_slot_mapping(logical):
                    # The staging kernels resolve logical rows through the
                    # owner's live page table themselves (one launch fewer).
                    owner = source.csa2_kv_sources[self.layer_idx]
                    slots = logical
                    mapping = (
                        source.csa2_global_page_tables[owner],
                        source.csa2_global_page_sizes[owner],
                        source.csa2_global_max_positions[owner],
                        source.csa2_token_requests[start:end],
                        visible,
                    )
                else:
                    slots = source.global_slot_tile(
                        self.layer_idx, start, end, logical, visible_lengths=visible
                    )
            inputs = CSA2BackendForwardArgs(
                swa_pool=manager.get_swa_buffer(self.layer_idx),
                swa_indices=source.csa2_swa_indices[self.layer_idx][start:end],
                main_pool=None if logical is None else manager.get_main_buffer(layer.kv_source),
                topk_indices=slots,
                main_logical_indices=logical,
                main_mapping=mapping,
                stage_group=(layer.kv_source, layer.index_source, start),
                state=state,
                output_position_ids=(
                    None
                    if state.output_position_ids is None
                    else state.output_position_ids[start:end]
                ),
                output_rotary_cos_sin=state.output_rotary_cos_sin,
            )
        if (
            inputs.output_position_ids is not None or inputs.output_rotary_cos_sin is not None
        ) and not self.sparse_params.use_packed_sparse_attention:
            raise ValueError("CSA2 fused output RoPE requires packed sparse attention")
        if metadata.swa_pool.device != q.device:
            raise ValueError("CSA2 Q and metadata must be on the same CUDA device")
        return inputs

    def _native_slot_mapping(self, logical: torch.Tensor) -> bool:
        """Logical main selections may stay logical when native staging maps them."""
        from .quantization import _fused_gather_supported

        return (
            not self.sparse_params.use_packed_sparse_attention
            and logical.is_cuda
            and _fused_gather_supported(logical.device.index)
        )

    def sparse_attn_predict(self, q, k, metadata, forward_args):
        if self.compute_backend != "trtllm" or self.sparse_params.use_packed_sparse_attention:
            raise ValueError("CSA2 Flash and packed attention require their explicit module helper")
        if (
            metadata.is_cuda_graph
            and self.compute_backend == "trtllm"
            and not self.sparse_params.use_packed_sparse_attention
            and metadata.workspace.numel() == 0
        ):
            raise RuntimeError("Warm up CSA2 metadata with forward before CUDA Graph capture")
        inputs = self.prepare_sparse_inputs(q, metadata, forward_args)
        if self.native_fp8:
            # This hook runs before the parent's scale-default assignment, whose
            # scales serve Q: the staging KV scale is derived here instead, since
            # a unit one clamps every channel the packed pools decode above 448.
            context = forward_args.attention_input_type == AttentionInputType.context_only
            # Repeating a forward on one carrier is fine: it hands back the very
            # buffers derived below. Any other scale describes a KV cache this path
            # never reads, so it could only contradict the staging pool.
            owned = metadata.stage_kv_scales
            for scale in (forward_args.kv_scale_orig_quant, forward_args.kv_scale_quant_orig):
                if scale is not None and not any(scale is buffer for buffer in owned):
                    raise ValueError("CSA2 FP8 staging derives its own Q/KV scales")
            # Generation buys the staged KV range from Q, which takes the
            # reciprocal scale so BMM1's dequant product stays exactly one. That
            # product used to be a hard requirement of the split-KV reduction, which
            # rescaled its per-CtaKv maxima with a host softmax scale that never saw
            # the dequant scales; ``fmhaReduction.cu`` now reads the same device
            # ``bmm1_scale`` the main kernel does. The reciprocal stays because the
            # identical expressions also live inside the closed cubins serving the
            # other two ``MultiCtasKvMode`` values, where neither the caller nor the
            # mode selection (``FmhaAutoTuner::selectKernel()``) is auditable, so a
            # non-unit product is only safe on the mode whose source we can read.
            #
            # Context cannot pay the same way: the C++ op points dequant_scale_q and
            # dequant_scale_kv at one tensor, so the product is that scale squared and
            # Q is divided by the scale rather than multiplied by its reciprocal. That
            # direction cannot saturate Q -- it only costs resolution underneath -- so
            # context does take range, bounded by Q's own amax and by the exponent the
            # main pool can actually need. Split-KV never runs here either way: every
            # MLA multi-CTA-KV cubin is a generation kernel.
            stage_orig_quant, stage_quant_orig = metadata.derive_stage_kv_scales(
                inputs,
                q=q,
                q_shared=context,
                maximum=_CONTEXT_STAGE_SCALE_MAX_EXPONENT if context else 126,
            )
            if context:
                forward_args.kv_scale_orig_quant = stage_orig_quant
                forward_args.kv_scale_quant_orig = stage_quant_orig
            else:
                # Q's pair is the staging pair read backwards: dequant_q = 1 / dequant_kv.
                forward_args.kv_scale_orig_quant = stage_quant_orig
                forward_args.kv_scale_quant_orig = stage_orig_quant
            metadata.stage_selected(
                inputs,
                kv_scale_orig_quant=stage_orig_quant,
                kv_scale_quant_orig=stage_quant_orig,
                q_scale_quant_orig=forward_args.kv_scale_quant_orig,
                softmax_scale=1.0 / (self.q_scaling * math.sqrt(_HEAD_DIM)),
                main_logical=inputs.main_logical_indices,
            )
            if (
                forward_args.attention_input_type == AttentionInputType.generation_only
                and q.numel()
            ):
                with torch.cuda.device(q.device):
                    quant_q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                        q.contiguous(), forward_args.kv_scale_quant_orig
                    )
                forward_args.quant_q_buffer = quant_q.view(
                    metadata.num_tokens, self.num_heads, _HEAD_DIM
                )
                forward_args.mla_bmm1_scale = metadata.mla_bmm1_scale
                forward_args.mla_bmm2_scale = metadata.mla_bmm2_scale
            elif forward_args.attention_input_type == AttentionInputType.context_only:
                forward_args.quant_scale_qkv = forward_args.kv_scale_orig_quant
        else:
            metadata.stage_selected(inputs, main_logical=inputs.main_logical_indices)
        sparse = forward_args.sparse_runtime_params
        sparse.sparse_attn_kv_lens = metadata.prepared_lens
        sparse.aux_kv_cache_pool_ptr = (
            metadata.extra_pool.data_ptr()
            if inputs.topk_indices is not None and metadata.num_sparse_topk > 128
            else None
        )
        forward_args.fmha_scheduler_counter = metadata.prepared_counter
        forward_args.attention_window_size = metadata.num_sparse_topk
        # Context Q already includes RoPE and cache publication happened above.
        # A null latent cache skips the native context append/RoPE preprocessing.
        forward_args.latent_cache = (
            None
            if forward_args.attention_input_type == AttentionInputType.context_only
            else metadata.latent_placeholder
            if self.native_fp8
            else metadata.swa_pool[:, 0]
        )
        forward_args.q_pe = q.view(metadata.num_tokens, self.num_heads, _HEAD_DIM)[..., 448:]
        return metadata.prepared_indices, None

    def _supports_packed(self, q, inputs, forward_args) -> bool:
        """Validate the explicit packed attention input/output contract."""
        from .kernel import supports_packed_attention

        if (
            forward_args.attention_mask
            not in (PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL)
            or forward_args.attention_mask_data is not None
            or forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
            or forward_args.output_sf is not None
            or forward_args.enable_dsv4_epilogue_fusion
            or forward_args.attention_sinks is None
            or inputs.swa_pool is None
            or inputs.swa_indices is None
        ):
            return False
        output = forward_args.output
        if output is not None and (
            output.dtype != torch.bfloat16
            or output.device != q.device
            or not output.is_contiguous()
            or output.numel() != q.numel()
        ):
            return False
        return supports_packed_attention(
            q.view(-1, self.num_heads, _HEAD_DIM),
            inputs.swa_pool,
            inputs.main_pool,
            inputs.swa_indices,
            inputs.topk_indices,
            forward_args.attention_sinks,
            1.0 / (self.q_scaling * math.sqrt(_HEAD_DIM)),
            position_ids=inputs.output_position_ids,
            rotary_cos_sin=inputs.output_rotary_cos_sin,
        )

    @staticmethod
    def _validate_helper_output(q, forward_args) -> None:
        if (
            forward_args.attention_mask
            not in (PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL)
            or forward_args.attention_mask_data is not None
            or forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
            or forward_args.output_sf is not None
            or forward_args.enable_dsv4_epilogue_fusion
        ):
            raise ValueError("CSA2 helpers do not support the requested mask/output format")
        output = forward_args.output
        if output is not None and (
            output.shape != q.shape
            or output.dtype != q.dtype
            or output.device != q.device
            or not output.is_contiguous()
        ):
            raise ValueError("CSA2 helper output must match contiguous BF16 query geometry")
        if forward_args.attention_sinks is None:
            raise ValueError("CSA2 requires an attention sink")

    def prepare_selected(self, q, metadata, forward_args):
        self._validate_helper_output(q, forward_args)
        if self.sparse_params.use_packed_sparse_attention:
            raise ValueError("Packed CSA2 must use forward_packed without BF16 staging")
        inputs = self.prepare_sparse_inputs(q, metadata, forward_args)
        metadata.stage_selected(inputs, main_logical=inputs.main_logical_indices)
        query = q.view(metadata.num_tokens, self.num_heads, _HEAD_DIM)
        if metadata.shared_plan is not None:
            return (
                query,
                metadata.shared_pool,
                metadata.prepared_indices,
                forward_args.attention_sinks,
                1.0 / (self.q_scaling * math.sqrt(_HEAD_DIM)),
            )
        kv = metadata.swa_pool
        if metadata.num_sparse_topk > _SWA_TILE:
            kv = torch.cat((kv, metadata.extra_pool), dim=1)
        valid = (
            torch.arange(metadata.num_sparse_topk, device=q.device)[None, :]
            < metadata.prepared_lens[:, None]
        )
        return (
            query,
            kv,
            valid,
            forward_args.attention_sinks,
            1.0 / (self.q_scaling * math.sqrt(_HEAD_DIM)),
        )

    def forward_packed(self, q, metadata, forward_args):
        from .kernel import packed_sparse_attention

        if not self.sparse_params.use_packed_sparse_attention:
            raise ValueError("Packed CSA2 attention is not enabled")
        self._validate_helper_output(q, forward_args)
        inputs = self.prepare_sparse_inputs(q, metadata, forward_args)
        if not self._supports_packed(q, inputs, forward_args):
            raise ValueError(
                "Packed CSA2 does not support these attention inputs or output options"
            )
        query = q.view(metadata.num_tokens, self.num_heads, _HEAD_DIM)
        output = None if forward_args.output is None else forward_args.output.view_as(query)
        result = packed_sparse_attention(
            query,
            inputs.swa_pool,
            inputs.main_pool,
            inputs.swa_indices,
            inputs.topk_indices,
            forward_args.attention_sinks,
            1.0 / (self.q_scaling * math.sqrt(_HEAD_DIM)),
            output=output,
            position_ids=inputs.output_position_ids,
            rotary_cos_sin=inputs.output_rotary_cos_sin,
        )
        return result.flatten(1)


def get_csa2_backend(params: CSA2Params) -> type[CSA2TrtllmAttention]:
    """The single sparse backend owns metadata; the module selects compute helpers."""
    if params.compute_backend not in ("auto", "trtllm", "flash_mla", "flashinfer"):
        raise ValueError(f"Unknown CSA2 attention backend: {params.compute_backend}")
    return CSA2TrtllmAttention


class CSA2FlashMLA:
    """Module-owned FlashMLA helper, matching the DSV4 composition pattern."""

    def __init__(self, attention=None):
        self.attention = attention

    @staticmethod
    def run(q, kv, valid, sink, scale) -> torch.Tensor:
        """Use the same sparse FlashMLA entry point for selected BF16 rows."""
        indices = torch.arange(valid.numel(), device=q.device, dtype=torch.int32).reshape_as(valid)
        indices = torch.where(valid, indices, -1)
        return CSA2FlashMLA.run_shared(q, kv.reshape(-1, 512), indices, sink, scale)

    @staticmethod
    def run_shared(q, bank, indices, sink, scale) -> torch.Tensor:
        """Consume a shared BF16 row bank without expanding rows per query."""
        from tensorrt_llm.flash_mla import flash_mla_sparse_fwd

        tokens, heads, dim = q.shape
        if dim != 512 or q.dtype != torch.bfloat16 or bank.dtype != torch.bfloat16:
            raise ValueError("CSA2 FlashMLA requires BF16 Q/KV with head dimension 512")
        if heads <= 0 or heads > 128:
            raise ValueError("CSA2 FlashMLA supports up to 128 query heads")
        if tokens == 0:
            return torch.empty_like(q)
        padded_heads = 64 if heads <= 64 else 128
        q_padded = F.pad(q, (0, 0, 0, padded_heads - heads)).contiguous()
        sink_padded = F.pad(sink.float(), (0, padded_heads - heads), value=torch.inf)
        indices = F.pad(indices, (0, -indices.shape[1] % 128), value=-1).unsqueeze(1)
        output, _, _ = flash_mla_sparse_fwd(
            q_padded, bank.view(-1, 1, dim), indices, scale, d_v=dim, attn_sink=sink_padded
        )
        return output[:, :heads]

    def forward_context(self, q, metadata, forward_args):
        if forward_args.attention_input_type != AttentionInputType.context_only:
            raise ValueError("CSA2 context helper requires context_only inputs")
        return self._forward_selected(q, metadata, forward_args)

    def forward_generation(self, q, metadata, forward_args):
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            raise ValueError("CSA2 generation helper requires generation_only inputs")
        return self._forward_selected(q, metadata, forward_args)

    def _forward_selected(self, q, metadata, forward_args):
        if self.attention is None:
            raise ValueError("CSA2 helper requires its attention backend")
        inputs = self.attention.prepare_selected(q, metadata, forward_args)
        run = self.run_shared if metadata.shared_plan is not None else self.run
        result = run(*inputs).flatten(1)
        if forward_args.output is not None:
            forward_args.output.copy_(result)
            return forward_args.output
        return result


class CSA2FlashInfer:
    """Module-owned FlashInfer plans and selected-row attention execution."""

    @dataclass
    class _Plan:
        wrapper: BatchPrefillWithRaggedKVCacheWrapper | BatchPrefillWithPagedKVCacheWrapper
        packed_mask: torch.Tensor
        indices: torch.Tensor | None = None
        protected: bool = False

    @staticmethod
    def pack_query_masks(valid: torch.Tensor) -> torch.Tensor:
        """Pack each query's mask separately in FlashInfer's little-endian order."""
        padded = F.pad(valid, (0, -valid.shape[1] % 8))
        bits = padded.reshape(valid.shape[0], padded.shape[1] // 8, 8).to(torch.int32)
        shifts = torch.arange(8, device=valid.device, dtype=torch.int32)
        return (bits << shifts).sum(-1).to(torch.uint8).flatten()

    def __init__(self, attention=None) -> None:
        self.attention = attention
        self._plans: dict[tuple[int, int, int, torch.device, float], CSA2FlashInfer._Plan] = {}
        self._workspaces: dict[torch.device, torch.Tensor] = {}
        self._paged_plans: dict[tuple, CSA2FlashInfer._Plan] = {}

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        valid: torch.Tensor,
        sink: torch.Tensor,
        scale: float,
        protect: bool = False,
    ) -> torch.Tensor:
        """``protect`` retains the plan across eager prefill evictions (generation warmup)."""
        with torch.cuda.device(q.device):
            return self._forward(q, kv, valid, sink, scale, protect)

    def _forward(self, q, kv, valid, sink, scale, protect=False):
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        count, heads, dim = q.shape
        width = kv.shape[1]
        if dim != 512 or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
            raise ValueError("CSA2 FlashInfer requires BF16 Q/KV with head dimension 512")
        if count == 0 or width == 0:
            return torch.zeros_like(q)
        key = (count, heads, width, q.device, scale)
        plan = self._plans.get(key)
        if plan is None:
            with torch.cuda.device(q.device):
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                raise RuntimeError("Warm up the CSA2 FlashInfer shape before CUDA Graph capture")
            if q.device not in self._workspaces:
                # Plans use fixed query geometry and disable split-KV.
                self._workspaces[q.device] = torch.empty(
                    8 * 1024 * 1024, dtype=torch.uint8, device=q.device
                )
            qo_host = torch.arange(count + 1, dtype=torch.int32, device="cpu")
            kv_host = qo_host * width
            # Geometry is fixed per cached plan. FA2's graph-mode scheduler
            # pads CTAs but does not mask that padding with disable_split_kv
            # in FlashInfer 0.6.18. Use the fixed (non-padded) scheduler; run()
            # remains capturable because these plans and buffers never resize.
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspaces[q.device],
                kv_layout="NHD",
                backend="fa2",
                use_cuda_graph=False,
            )
            wrapper.plan(
                qo_host,
                kv_host,
                heads,
                1,
                dim,
                head_dim_vo=dim,
                custom_mask=torch.ones(count * width, dtype=torch.bool, device=q.device),
                causal=False,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                disable_split_kv=True,
            )
            # plan() packs the bool mask into its own persistent byte buffer.
            # Update that exact buffer on replay, not the original bool mask.
            plan = self._Plan(wrapper, wrapper._custom_mask_buf)
            self._plans[key] = plan
        plan.protected |= protect or torch.cuda.is_current_stream_capturing()
        # Updating the original bool mask would not update the planned wrapper.
        plan.packed_mask.copy_(self.pack_query_masks(valid))
        rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, dim).contiguous()
        output, lse = plan.wrapper.run(q.contiguous(), rows, rows, return_lse=True)
        # FA2 returns base-2 LSE. Restore the shared zero-valued sink without
        # requantizing the input cache. Empty selections have zero output.
        correction = torch.sigmoid(lse.float() * math.log(2.0) - sink.float()[None, :])
        output = output.float() * correction[..., None]
        return torch.where(valid.any(-1)[:, None, None], output, 0).to(q.dtype)

    def run_shared(self, q, bank, indices, sink, scale, protect=False) -> torch.Tensor:
        """Use one-row pages to share converted KV across sparse queries."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        count, heads, dim = q.shape
        if dim != 512 or q.dtype != torch.bfloat16 or bank.dtype != torch.bfloat16:
            raise ValueError("CSA2 FlashInfer requires BF16 Q/KV with head dimension 512")
        if count == 0:
            return torch.empty_like(q)
        width = indices.shape[1]
        if width == 0:
            return torch.zeros_like(q)
        key = (count, heads, width, q.device, scale)
        plan = self._paged_plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 FlashInfer shared rows before graph capture")
            if q.device not in self._workspaces:
                self._workspaces[q.device] = torch.empty(
                    8 * 1024 * 1024, dtype=torch.uint8, device=q.device
                )
            qo_host = torch.arange(count + 1, dtype=torch.int32, device="cpu")
            page_indices = indices.clamp_min(0).flatten().contiguous()
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspaces[q.device], kv_layout="NHD", backend="fa2", use_cuda_graph=False
            )
            wrapper.plan(
                qo_host,
                qo_host * width,
                page_indices,
                torch.ones(count, dtype=torch.int32, device="cpu"),
                heads,
                1,
                dim,
                1,
                head_dim_vo=dim,
                custom_mask=torch.ones(count * width, dtype=torch.bool, device=q.device),
                causal=False,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                disable_split_kv=True,
            )
            plan = self._Plan(wrapper, wrapper._custom_mask_buf, wrapper._paged_kv_indices_buf)
            self._paged_plans[key] = plan
        plan.protected |= protect or torch.cuda.is_current_stream_capturing()
        valid = indices >= 0
        plan.indices.copy_(indices.clamp_min(0).flatten())
        plan.packed_mask.copy_(self.pack_query_masks(valid))
        pages = bank.view(-1, 1, 1, dim)
        output, lse = plan.wrapper.run(q.contiguous(), (pages, pages), return_lse=True)
        correction = torch.sigmoid(lse.float() * math.log(2.0) - sink.float()[None, :])
        output = output.float() * correction[..., None]
        return torch.where(valid.any(-1)[:, None, None], output, 0).to(q.dtype)

    def forward_context(self, q, metadata, forward_args):
        if forward_args.attention_input_type != AttentionInputType.context_only:
            raise ValueError("CSA2 context helper requires context_only inputs")
        return self._forward_selected(q, metadata, forward_args)

    def forward_generation(self, q, metadata, forward_args):
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            raise ValueError("CSA2 generation helper requires generation_only inputs")
        return self._forward_selected(q, metadata, forward_args)

    def _forward_selected(self, q, metadata, forward_args):
        if self.attention is None:
            raise ValueError("CSA2 helper requires its attention backend")
        inputs = self.attention.prepare_selected(q, metadata, forward_args)
        shared = metadata.shared_plan is not None
        query, values, selection, _, scale = inputs
        width = selection.shape[1] if shared else values.shape[1]
        key = (query.shape[0], query.shape[1], width, q.device, scale)
        real_context = metadata.num_contexts > 0
        if real_context:
            # Whole eager phases have variable Q. Preserve generation warmup
            # and captured plans, but do not retain every historical prefill.
            for cache in (self._plans, self._paged_plans):
                for previous, plan in tuple(cache.items()):
                    if previous != key and previous[1:] == key[1:] and not plan.protected:
                        del cache[previous]
        if shared:
            with torch.cuda.device(q.device):
                result = self.run_shared(*inputs, protect=not real_context).flatten(1)
        else:
            result = self(*inputs, protect=not real_context).flatten(1)
        if forward_args.output is not None:
            forward_args.output.copy_(result)
            return forward_args.output
        return result
