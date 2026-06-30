# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CuTe DSL MLA decode FMHA library."""

import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

_LOG2_E = math.log2(math.e)

# Diagnostic for "why didn't CuteDSL engage" (e.g. seq_len_q=8 / H=16 at MTP
# draft_len=7 silently fell back). When TLLM_CUTE_DSL_GATE_LOG is set, the gate
# logs -- once per (layer_idx, supported, reason) -- the is_supported verdict
# plus the batch shape it saw (q rows, num_generations, num_contexts), so a
# single run reveals exactly which check rejects a given geometry. Host metadata
# only (no device sync) -> CUDA-graph-capture-safe.
_DEBUG_GATE = bool(os.environ.get("TLLM_CUTE_DSL_GATE_LOG"))
_GATE_LOG_SEEN = set()


class CuteDslMlaFmha(PhasedFmha):
    """Blackwell CuTe DSL FMHA library for decode-only MLA."""

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not IS_CUTLASS_DSL_AVAILABLE:
            logger.debug("CuTe DSL MLA FMHA is unavailable: nvidia-cutlass-dsl is not installed.")
            return False

        sm = get_sm_version()
        if sm not in (100, 103):
            logger.debug(f"CuTe DSL MLA FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False

        if not attn.is_mla_enable:
            logger.debug("CuTe DSL MLA FMHA is unavailable: only MLA is supported.")
            return False
        # predicted_tokens_per_seq == seq_len_q (spec_config.tokens_per_gen_step,
        # = max_draft_len + 1; 1 when no spec-decode). No hard upper bound here:
        # the decode kernel folds up to F = min(seq_len_q, M_tile // num_heads)
        # query tokens into the head dimension, and the per-request gate's
        # can_implement check is the authority on what the kernel can serve. A
        # [1, 4] cap here silently excluded CuteDSL from fmha_libs entirely for
        # MTP draft_len > 3 (e.g. seq_len_q=8), so it was never even consulted.
        if attn.predicted_tokens_per_seq is None or attn.predicted_tokens_per_seq < 1:
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: predicted_tokens_per_seq "
                f"must be >= 1, got {attn.predicted_tokens_per_seq}."
            )
            return False
        if attn.kv_lora_rank is None or attn.kv_lora_rank <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: kv_lora_rank must be positive.")
            return False
        if attn.qk_rope_head_dim is None or attn.qk_rope_head_dim <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: qk_rope_head_dim must be positive.")
            return False
        if attn.qk_nope_head_dim is None or attn.qk_nope_head_dim <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: qk_nope_head_dim must be positive.")
            return False
        if attn.kv_lora_rank != 512 or attn.qk_rope_head_dim != 64:
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: kernels require kv_lora_rank=512 and "
                f"qk_rope_head_dim=64, got kv_lora_rank={attn.kv_lora_rank}, "
                f"qk_rope_head_dim={attn.qk_rope_head_dim}."
            )
            return False
        if attn.num_heads > 128:
            logger.debug(
                f"CuTe DSL MLA FMHA is unavailable: num_heads must be <= 128, got {attn.num_heads}."
            )
            return False

        return True

    @staticmethod
    def _get_kernel_dtype(attn: "TrtllmAttention", q: torch.Tensor) -> Optional[torch.dtype]:
        if getattr(attn, "has_fp8_kv_cache", False):
            return torch.float8_e4m3fn
        if q.dtype in (torch.float16, torch.bfloat16):
            return q.dtype
        return None

    @staticmethod
    def _kernel_can_implement(
        kernel_dtype: torch.dtype,
        batch_size: int,
        seq_len_q: int,
        page_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> tuple[bool, str]:
        """Ask the CuTe DSL kernel's own ``can_implement`` whether it accepts
        this problem under the tiler the FMHA library launches with.

        The custom op only runs ``can_implement`` when the AutoTuner is engaged
        (``CuteDSLNVMlaDecodeBlackwellRunner.get_valid_tactics``); the FMHA
        library calls the op directly with ``tactic=None`` -> the default
        ``((128, 128), (128, 256))`` tiler, bypassing that check. Mirror the
        op's launch configuration here so the gate refuses any request the
        kernel cannot actually serve instead of failing at launch.
        """
        import cutlass

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import (
            BlackwellMultiHeadLatentAttentionForwardFP8,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp16 import (
            BlackwellMultiHeadLatentAttentionForwardFP16,
        )

        if kernel_dtype == torch.float8_e4m3fn:
            kernel_class = BlackwellMultiHeadLatentAttentionForwardFP8
            in_dtype, out_dtype = cutlass.Float8E4M3FN, cutlass.BFloat16
        elif kernel_dtype == torch.float16:
            kernel_class = BlackwellMultiHeadLatentAttentionForwardFP16
            in_dtype, out_dtype = cutlass.Float16, cutlass.Float16
        elif kernel_dtype == torch.bfloat16:
            kernel_class = BlackwellMultiHeadLatentAttentionForwardFP16
            in_dtype, out_dtype = cutlass.BFloat16, cutlass.BFloat16
        else:
            return False, f"Unsupported CuTe DSL kernel dtype {kernel_dtype}."

        # Default launch tiler and flags -- keep in sync with
        # ``CuteDSLNVMlaDecodeBlackwellRunner`` in cute_dsl_custom_ops.py
        # (the ``tactic=None`` path that ``_run_mla_decode`` exercises).
        mma_qk_tiler_mn, mma_pv_tiler_mn = (128, 128), (128, 256)
        if not kernel_class.can_implement(
            batch_size,
            seq_len_q,
            page_size,  # K -- mirrors the op's get_valid_tactics call
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            in_dtype,
            out_dtype,
            cutlass.Float32,  # acc_dtype
            cutlass.Float32,  # lse_dtype
            mma_qk_tiler_mn,
            mma_pv_tiler_mn,
            1,  # split_kv
            True,  # is_persistent
            True,  # is_var_seq
            False,  # is_var_split_kv
            page_size,
        ):
            return (
                False,
                "CuTe DSL MLA kernel can_implement rejected the problem "
                f"(dtype={kernel_dtype}, H={num_heads}, L={kv_lora_rank}, "
                f"R={qk_rope_head_dim}, S={seq_len_q}, B={batch_size}, "
                f"page_size={page_size}).",
            )
        return True, ""

    @staticmethod
    def _select_page_table_layer(
        block_offsets: torch.Tensor,
        layer_idx: int,
        host_kv_cache_pool_mapping: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if block_offsets.dim() == 4:
            if block_offsets.shape[2] < 1:
                return None
            if host_kv_cache_pool_mapping is not None:
                if layer_idx >= host_kv_cache_pool_mapping.shape[0]:
                    return None
                pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
            else:
                pool_idx = layer_idx if block_offsets.shape[0] > 1 else 0
            if pool_idx >= block_offsets.shape[0]:
                return None
            return block_offsets[pool_idx, :, 0, :]
        if block_offsets.dim() == 3:
            if block_offsets.shape[1] < 1:
                return None
            return block_offsets[:, 0, :]
        if block_offsets.dim() == 2:
            return block_offsets
        return None

    # ---- variable split-KV (KV-dimension parallelism) --------------------
    # The decode kernel's MMA grid is starved when batch_size is small: with
    # split_kv=1 only ~batch*heads CTAs launch, so attention-DP (batch ≈
    # concurrency/tp) leaves most SMs idle. Splitting the KV dimension lets
    # multiple CTAs cooperate on one sequence (partials reduced via an fp32
    # workspace). We mirror the kernel's own ``get_split_kv`` heuristic. The
    # kernel's SUPPORTED split mode is the VARIABLE path (is_var_split_kv=True
    # + per-sequence block_split_kvs); the fixed-split path is broken for
    # split>1. CUDA-graph-safe by construction: the scalar split_kv (which
    # bakes the launch grid) is derived from HOST-known sizes only, while the
    # per-sequence block_split_kvs is computed on-device from cache_seqs with
    # no host sync (no ``.item()``).
    _CUTE_DSL_QK_TILE_K = 128  # mma_qk_tiler_mn[1] the op launches with
    _CUTE_DSL_MAX_SPLIT_KV = 32  # kernel's get_split_kv hard cap

    def _get_max_active_blocks(self) -> int:
        """``max_active_clusters * cluster_shape[0]`` (cluster shape (2,1,1)),
        matching the op's get_split_kv input. Queried once and cached before
        CUDA-graph capture (the eager warmup populates it)."""
        cached = getattr(self, "_cute_dsl_max_active_blocks", None)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CuTe DSL MLA FMHA: max_active_blocks was not cached "
                    "before CUDA graph capture (run an eager warmup first)."
                )
            import cutlass
            hw = cutlass.utils.HardwareInfo()
            max_active_clusters = hw.get_max_active_clusters(2)  # cluster product
            cached = int(max_active_clusters) * 2  # * cluster_shape_mnk[0]
            self._cute_dsl_max_active_blocks = cached
        return cached

    @classmethod
    def _split_kv_from_max_splits(
        cls, max_splits: int, batch_size: int, seq_len_q: int, max_active_blocks: int
    ) -> int:
        """Host scalar form of the kernel's ``get_split_kv``."""
        blocks_per_batch = max(1, max_active_blocks // batch_size // (seq_len_q * 2))
        split_heur = min(max_splits, blocks_per_batch)
        k_waves = (max_splits + split_heur - 1) // split_heur
        split_wave_aware = (max_splits + k_waves - 1) // k_waves
        return min(split_wave_aware, cls._CUTE_DSL_MAX_SPLIT_KV)

    def _compute_block_split_kvs(
        self,
        cache_seqs: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        max_active_blocks: int,
        split_kv_max: int,
    ) -> torch.Tensor:
        """Per-sequence split count, vectorized over the device ``cache_seqs``
        tensor (capture-safe; no host sync). Mirrors ``get_split_kv`` with the
        per-sequence KV length ``cache_seqs[b]`` and clamps to the host grid
        max ``split_kv_max``."""
        blocks_per_batch = max(1, max_active_blocks // batch_size // (seq_len_q * 2))
        k = cache_seqs.to(torch.int64)
        tile_k = self._CUTE_DSL_QK_TILE_K
        max_splits = torch.clamp((k + tile_k - 1) // tile_k, min=1)
        split_heur = torch.clamp(max_splits, max=blocks_per_batch)
        k_waves = (max_splits + split_heur - 1) // split_heur
        split_wave_aware = (max_splits + k_waves - 1) // k_waves
        cap = min(self._CUTE_DSL_MAX_SPLIT_KV, split_kv_max)
        return torch.clamp(split_wave_aware, max=cap).to(torch.int32)

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            self.attn,
            metadata,
            forward_args,
        )
        if not supported:
            logger.debug(f"CuTe DSL MLA FMHA does not support request: {reason}")
        if _DEBUG_GATE:
            key = (self.attn.layer_idx, supported, reason)
            if key not in _GATE_LOG_SEEN:
                _GATE_LOG_SEEN.add(key)
                print(
                    "[CUTEDSL_GATE] layer=%d supported=%s q_rows=%d "
                    "num_generations=%d num_contexts=%d reason=%s"
                    % (
                        self.attn.layer_idx, supported, q.shape[0],
                        metadata.num_generations, metadata.num_contexts,
                        reason or "(ok)",
                    ),
                    flush=True,
                )
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> tuple[bool, str]:
        if fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "CuTe DSL MLA FMHA only supports generation-only attention."
        if meta.num_contexts != 0 or meta.num_generations <= 0:
            return False, "CuTe DSL MLA FMHA only supports decode-only batches."
        if meta.beam_width != 1:
            return False, f"Beam search is not supported, got beam_width={meta.beam_width}."
        # Linear-chain MTP / spec-decode (seq_len_q > 1) IS supported: the
        # kernel applies the implicit causal mask (q token t attends to KV
        # [0, K - (seq_len_q - 1) + t)). Tree / dynamic-tree spec-decode carries
        # an explicit packed mask the kernel cannot express, and an explicit
        # CUSTOM mask is likewise unsupported -> fall back to TRTLLM for those.
        if (
            fwd.attention_mask == CustomAttentionMask.CUSTOM
            or fwd.attention_mask_data is not None
            or getattr(meta, "is_spec_dec_tree", False)
            or getattr(meta, "is_spec_dec_dynamic_tree", False)
        ):
            return False, "CuTe DSL MLA FMHA does not support custom/tree speculative masks."
        if q.shape[0] % meta.num_generations != 0:
            return (
                False,
                f"num_tokens ({q.shape[0]}) must be divisible by "
                f"num_generations ({meta.num_generations}).",
            )
        seq_len_q = q.shape[0] // meta.num_generations
        # No hard upper bound on seq_len_q here: the kernel folds up to
        # F = min(seq_len_q, M_tile // num_heads) query tokens into the head
        # dimension and the can_implement check below is the authority on what
        # the kernel can actually serve for this geometry.
        if seq_len_q < 1:
            return False, f"Query length must be >= 1, got {seq_len_q}."
        if meta.kv_cache_block_offsets is None:
            return False, "Paged KV block offsets are required."
        # ``host_kv_cache_pool_mapping`` is indexed by the LOCAL (compacted)
        # layer index, not the global ``attn.layer_idx`` -- they coincide for a
        # full model but differ when the KV cache manager allocates a subset of
        # layers (e.g. PP, or the layer-wise benchmark's ``layer_mask``).
        local_layer_idx = attn.get_local_layer_idx(meta)
        page_table_layer = self._select_page_table_layer(
            meta.kv_cache_block_offsets,
            local_layer_idx,
            meta.host_kv_cache_pool_mapping,
        )
        if page_table_layer is None:
            return (
                False,
                "Unsupported KV block offsets shape "
                f"{tuple(meta.kv_cache_block_offsets.shape)} for layer_idx={attn.layer_idx}.",
            )
        if meta.kv_cache_manager is None:
            return False, "KV cache manager is required."
        if fwd.latent_cache is None:
            return False, "latent_cache is required."
        if fwd.output is None:
            return False, "output is required."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = getattr(meta.kv_cache_manager, "tokens_per_block", 0)
        if tokens_per_block <= 1 or 128 % tokens_per_block != 0:
            return (
                False,
                f"tokens_per_block must divide 128 and be greater than 1, got {tokens_per_block}.",
            )

        kernel_dtype = self._get_kernel_dtype(attn, q)
        if kernel_dtype is None:
            return (
                False,
                f"Unsupported dtype combination: q={q.dtype}, "
                f"has_fp8_kv_cache={getattr(attn, 'has_fp8_kv_cache', False)}.",
            )
        if kernel_dtype == torch.float8_e4m3fn and (
            fwd.quant_q_buffer is None or fwd.mla_bmm1_scale is None or fwd.mla_bmm2_scale is None
        ):
            return (
                False,
                "FP8 CuTe DSL MLA decode requires quant_q_buffer, "
                "mla_bmm1_scale, and mla_bmm2_scale from MLA RoPE generation.",
            )
        if kernel_dtype in (torch.float16, torch.bfloat16):
            kv_pool_dtype = meta.kv_cache_manager.get_buffers(attn.layer_idx).dtype
            if kv_pool_dtype != kernel_dtype:
                return (
                    False,
                    f"CuTe DSL MLA {kernel_dtype} fast path requires matching "
                    f"KV cache dtype, got {kv_pool_dtype}.",
                )

        # Final authority: the kernel's own can_implement under the default
        # tiler the op launches with (the FMHA library bypasses the AutoTuner's
        # can_implement filter), so a request that reaches the gate is one the
        # kernel can actually serve.
        return self._kernel_can_implement(
            kernel_dtype,
            meta.num_generations,
            seq_len_q,
            tokens_per_block,
            attn.num_heads,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
        )

    def _run_mla_decode(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        params: FmhaParams,
        kernel_dtype: torch.dtype,
    ) -> None:
        attn = params.attn
        meta = params.meta

        if kernel_dtype == torch.float8_e4m3fn:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
        elif kernel_dtype in (torch.float16, torch.bfloat16):
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
        else:
            raise ValueError(
                f"CuTe DSL MLA FMHA got unsupported kernel_dtype={kernel_dtype}; "
                "expected torch.float8_e4m3fn, torch.float16, or torch.bfloat16."
            )

        num_tokens = q.shape[0]
        batch_size = params.num_requests
        seq_len_q = num_tokens // batch_size
        if seq_len_q * batch_size != num_tokens:
            raise RuntimeError(
                f"CuTe DSL MLA decode expects num_tokens ({num_tokens}) divisible by "
                f"batch_size ({batch_size})."
            )

        d_latent = attn.kv_lora_rank
        d_rope = attn.qk_rope_head_dim
        qk_nope_head_dim = attn.qk_nope_head_dim
        if d_latent is None or d_rope is None or qk_nope_head_dim is None:
            raise RuntimeError("CuTe DSL MLA decode requires complete MLA dimensions.")

        num_heads = attn.num_heads
        page_size = params.tokens_per_block

        if kernel_dtype == torch.float8_e4m3fn and params.fwd.quant_q_buffer is not None:
            q_kernel = params.fwd.quant_q_buffer.view(torch.float8_e4m3fn).view_as(q)
        else:
            q_kernel = q if q.dtype == kernel_dtype else q.to(kernel_dtype)
        q_view = q_kernel.view(batch_size, seq_len_q, num_heads, d_latent + d_rope)

        kv_pool = meta.kv_cache_manager.get_buffers(attn.layer_idx)
        if kernel_dtype in (torch.float16, torch.bfloat16) and kv_pool.dtype != kernel_dtype:
            raise RuntimeError(
                f"CuTe DSL MLA {kernel_dtype} fast path requires matching "
                f"KV cache dtype, got {kv_pool.dtype}."
            )
        # Paged-pool layout normalization for both KV cache managers.
        # KVCacheManagerV2 exposes each layer as a densely-packed page pool.
        # KVCacheManagerV1 exposes a per-layer view over one interleaved pool,
        # where dim-0 strides by all local layers. The CuTe DSL kernel addresses
        # pages with the packed block stride, so represent v1 as a packed
        # combined-slot view and fold this layer's slot offset into the page
        # table below.
        packed_block = 1
        for size in kv_pool.shape[1:]:
            packed_block *= size
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block if packed_block else 1
        layer_in_pool = 0
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            layer_in_pool = kv_pool.storage_offset() // packed_block
            kv_pool = kv_pool.as_strided(
                (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
                (packed_block, *kv_pool.stride()[1:]),
                0,
            )
        else:
            layers_in_pool = 1

        kv_pool_typed = kv_pool.view(kernel_dtype)
        if kv_pool_typed.dim() != 5 or kv_pool_typed.shape[1] != 1 or kv_pool_typed.shape[3] != 1:
            raise RuntimeError(
                "CuTe DSL MLA decode expects KV cache layout "
                f"[num_pages, 1, page_size, 1, head_dim], got {tuple(kv_pool_typed.shape)}."
            )
        if kv_pool_typed.shape[2] != page_size or kv_pool_typed.shape[-1] < d_latent + d_rope:
            raise RuntimeError(
                "CuTe DSL MLA decode got incompatible KV cache shape "
                f"{tuple(kv_pool_typed.shape)} for page_size={page_size}, "
                f"kv_lora_rank={d_latent}, qk_rope_head_dim={d_rope}."
            )

        block_offsets = meta.kv_cache_block_offsets
        # See ``_is_supported_with_reason``: the pool mapping is local-indexed.
        local_layer_idx = attn.get_local_layer_idx(meta)
        page_table_layer = self._select_page_table_layer(
            block_offsets,
            local_layer_idx,
            meta.host_kv_cache_pool_mapping,
        )
        if page_table_layer is None:
            raise RuntimeError(
                "CuTe DSL MLA decode got unsupported KV block offsets shape "
                f"{tuple(block_offsets.shape)} for layer_idx={attn.layer_idx}."
            )
        cache_seqs_base = params.sequence_lengths.to(torch.int32)
        page_table = page_table_layer[meta.num_contexts:].transpose(0, 1).to(torch.int32)
        if layers_in_pool > 1:
            page_table = page_table + layer_in_pool

        # KVCacheManager exposes NHD pages as [num_pages, 1, page_size, 1, head_dim].
        # The CuTe DSL kernel consumes a paged [page_size, dim, num_pages] view
        # with the dim axis contiguous.
        kv_pages = kv_pool_typed[:, 0, :, 0, : d_latent + d_rope]
        c_pool_latent = kv_pages[..., :d_latent].permute(1, 2, 0)
        c_pool_rope = kv_pages[..., d_latent:].permute(1, 2, 0)

        # Variable split-KV: parallelize the KV dimension when the batch is too
        # small to fill the SMs (see the helper block above). Default ON; set
        # TLLM_CUTE_DSL_VAR_SPLIT_KV=0 to force the legacy split_kv=1 path.
        is_var_split_kv = False
        block_split_kvs = torch.empty(0, dtype=torch.int32, device=q.device)
        split_kv = 1
        workspace = torch.empty(0, dtype=torch.int8, device=q.device)
        if os.environ.get("TLLM_CUTE_DSL_VAR_SPLIT_KV", "1") != "0":
            max_active_blocks = self._get_max_active_blocks()
            # Host upper bound on KV length: per-sequence page capacity * page
            # size (page_table is [pages_per_seq, batch], a fixed shape under
            # CUDA-graph capture). Yields the grid's split_kv max on the host.
            k_max = page_table.shape[0] * page_size
            max_splits = max(1, (k_max + self._CUTE_DSL_QK_TILE_K - 1) // self._CUTE_DSL_QK_TILE_K)
            split_kv = self._split_kv_from_max_splits(
                max_splits, batch_size, seq_len_q, max_active_blocks
            )
            if split_kv > 1:
                is_var_split_kv = True
                block_split_kvs = self._compute_block_split_kvs(
                    cache_seqs_base, batch_size, seq_len_q, max_active_blocks, split_kv
                )
                # get_workspace_size = B*H*S*split_kv*(D+1)*acc_width//8; fold
                # cancels (H_eff*S_eff == H*S), acc=fp32 (width 32 -> //8 = 4).
                ws_bytes = batch_size * num_heads * seq_len_q * split_kv * (d_latent + 1) * 4
                workspace = torch.empty(ws_bytes, dtype=torch.int8, device=q.device)
            else:
                split_kv = 1

        softmax_scale = float(1.0 / (math.sqrt(qk_nope_head_dim + d_rope) * attn.q_scaling))
        output_scale = 1.0
        if kernel_dtype == torch.float8_e4m3fn:
            if params.fwd.mla_bmm1_scale is None or params.fwd.mla_bmm2_scale is None:
                raise RuntimeError("FP8 CuTe DSL MLA decode requires MLA FP8 scales.")
            cached = getattr(self, "_cute_dsl_fp8_scale", None)
            if cached is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CuTe DSL MLA FMHA: fp8 decode scale was not cached for "
                        f"layer {attn.layer_idx} before CUDA graph capture."
                    )
                softmax_scale = float(params.fwd.mla_bmm1_scale[1].item()) / _LOG2_E
                output_scale = float(params.fwd.mla_bmm2_scale[0].item())
                self._cute_dsl_fp8_scale = (softmax_scale, output_scale)
            else:
                softmax_scale, output_scale = cached
                if (
                    os.environ.get("TLLM_CUTE_DSL_SCALE_DUMP")
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    live_softmax_scale = float(params.fwd.mla_bmm1_scale[1].item()) / _LOG2_E
                    live_output_scale = float(params.fwd.mla_bmm2_scale[0].item())
                    print(
                        "[CUTEDSL_SCALE] layer=%d cached=(%.8f,%.8f) "
                        "live=(%.8f,%.8f) drift=%s"
                        % (
                            attn.layer_idx,
                            softmax_scale,
                            output_scale,
                            live_softmax_scale,
                            live_output_scale,
                            abs(live_softmax_scale - softmax_scale) > 1e-6
                            or abs(live_output_scale - output_scale) > 1e-6,
                        ),
                        flush=True,
                    )

        out_kernel_dtype = torch.bfloat16 if kernel_dtype == torch.float8_e4m3fn else kernel_dtype
        output_view = output.view(batch_size, seq_len_q, num_heads, d_latent)

        # Single fused decode over all ``seq_len_q`` query tokens. For
        # multi-query (MTP / linear spec-decode) the kernel applies the causal
        # mask internally: query token ``t`` attends to KV positions
        # ``[0, K - (seq_len_q - 1) + t)``. ``cache_seqs_base`` already counts
        # every freshly-appended token of this step (K), so token ``t``'s bound
        # equals ``cache_seqs_base - (seq_len_q - 1) + t`` -- exactly the
        # per-query trim the previous one-token-at-a-time loop applied. For
        # ``seq_len_q == 1`` this reduces to a plain decode.
        q_latent = q_view[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = q_view[..., d_latent:].permute(2, 3, 1, 0)

        # When the kernel output dtype matches the module output dtype (the
        # common fp8-KV case: both bf16), have the kernel write straight into
        # ``output`` instead of a temp buffer that is then D2D-copied back. That
        # copy was ~1.7us/call and, at small batch where the decode win is only
        # a few us, ate the win (MLA module went flat/slightly slower despite a
        # faster decode kernel). ``output_view`` is a contiguous
        # [B, S_q, H, d_latent] view, so its permute(2,3,1,0) is byte-identical
        # in layout to a fresh contiguous o_storage's -- the op's compact-shape
        # marking still holds. Only fall back to the temp+copy on a dtype
        # mismatch (the kernel can only emit out_kernel_dtype).
        write_output_direct = output.dtype == out_kernel_dtype
        if write_output_direct:
            o_storage = output_view
        else:
            o_storage = torch.empty(
                (batch_size, seq_len_q, num_heads, d_latent),
                dtype=out_kernel_dtype,
                device=q.device,
            )
        o_kernel = o_storage.permute(2, 3, 1, 0)
        lse_storage = torch.empty(
            (batch_size, seq_len_q, num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        lse = lse_storage.permute(2, 1, 0)

        # The decode path uses variable-seq mode by default (real serving has
        # unequal per-request KV lengths). Experiment toggle: set
        # TLLM_CUTE_DSL_VAR_SEQ=0 to force the fixed-length path -- only valid
        # when every sequence shares one KV length (e.g. profiling/microbench).
        is_var_seq = os.environ.get("TLLM_CUTE_DSL_VAR_SEQ", "1") != "0"

        op(
            q_latent,
            q_rope,
            c_pool_latent,
            c_pool_rope,
            page_table,
            cache_seqs_base,
            block_split_kvs,
            o_kernel,
            lse,
            workspace,
            num_heads,
            seq_len_q,
            page_size,
            True,  # is_persistent
            is_var_seq,
            is_var_split_kv,
            split_kv,
            softmax_scale,
            output_scale,
        )

        # If the kernel wrote into a temp (dtype mismatch), copy/convert back
        # into ``output``. In the common matched-dtype case the kernel already
        # wrote ``output`` directly (o_storage IS output_view), so skip the copy.
        if not write_output_direct:
            # o_kernel is [num_heads, d_latent, seq_len_q, batch_size]; restore
            # the [batch_size, seq_len_q, num_heads, d_latent] view.
            attn_out = o_kernel.permute(3, 2, 0, 1)
            output_view.copy_(attn_out.to(output.dtype))

    def run_mla_generation(
        self,
        params: FmhaParams,
    ) -> None:
        if params.qkv_input is None:
            raise RuntimeError("CuTe DSL MLA generation requires qkv_input.")
        if params.context_buf is None:
            raise RuntimeError("CuTe DSL MLA generation requires context_buf.")
        if params.sequence_lengths is None:
            raise RuntimeError("CuTe DSL MLA generation requires sequence lengths.")

        kernel_dtype = self._get_kernel_dtype(params.attn, params.qkv_input)
        if kernel_dtype is None:
            raise RuntimeError("CuTe DSL MLA generation was selected for an unsupported dtype.")

        self._run_mla_decode(
            params.qkv_input,
            params.context_buf,
            params,
            kernel_dtype,
        )
