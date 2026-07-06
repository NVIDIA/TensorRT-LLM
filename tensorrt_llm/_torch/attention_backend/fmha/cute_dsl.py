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
    def _to_cutlass_dtype(dtype: torch.dtype):
        """Map a torch dtype to the cutlass dtype the kernel expects, or None
        if the kernel has no counterpart for it."""
        import cutlass

        return {
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }.get(dtype)

    @staticmethod
    def _kernel_can_implement(
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
        batch_size: int,
        seq_len_q: int,
        page_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> tuple[bool, str]:
        """Ask the CuTe DSL kernel's own ``can_implement`` whether it accepts
        this problem under the tiler the FMHA library launches with.

        ``in_dtype`` / ``out_dtype`` are the actual kernel input/output torch
        dtypes (the input is fp8 on the fp8-KV path, the output is q's dtype);
        they are converted to cutlass dtypes here and the kernel class is
        selected from the input dtype. Passing the real dtypes lets
        can_implement reject an unsupported combination (e.g. an fp16 output on
        the fp8 input path, which the kernel does not support) instead of
        failing at launch.
        """
        import cutlass

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import (
            BlackwellMultiHeadLatentAttentionForwardFP8,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp16 import (
            BlackwellMultiHeadLatentAttentionForwardFP16,
        )

        cute_in_dtype = CuteDslMlaFmha._to_cutlass_dtype(in_dtype)
        if cute_in_dtype is None:
            return False, f"Unsupported CuTe DSL input dtype {in_dtype}."
        cute_out_dtype = CuteDslMlaFmha._to_cutlass_dtype(out_dtype)
        if cute_out_dtype is None:
            return False, f"Unsupported CuTe DSL output dtype {out_dtype}."

        kernel_class = (
            BlackwellMultiHeadLatentAttentionForwardFP8
            if cute_in_dtype == cutlass.Float8E4M3FN
            else BlackwellMultiHeadLatentAttentionForwardFP16
        )

        # Default launch tiler and flags -- keep in sync with
        # ``CuteDSLNVMlaDecodeBlackwellRunner`` in cute_dsl_custom_ops.py
        # (the ``tactic=None`` path that ``_run_mla_decode`` exercises).
        mma_qk_tiler_mn, mma_pv_tiler_mn = (128, 128), (128, 256)
        if not kernel_class.can_implement(
            batch_size,
            seq_len_q,
            2,  # A fake K to bypass can_implement check
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            cute_in_dtype,
            cute_out_dtype,
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
                f"(in_dtype={in_dtype}, out_dtype={out_dtype}, H={num_heads}, "
                f"L={kv_lora_rank}, R={qk_rope_head_dim}, S={seq_len_q}, "
                f"B={batch_size}, page_size={page_size}).",
            )
        return True, ""

    # ---- split-KV (KV-dimension parallelism) -----------------------------
    # The decode kernel's MMA grid is starved when batch_size is small: with
    # split_kv=1 only ~batch*heads CTAs launch, so attention-DP (batch ≈
    # concurrency/tp) leaves most SMs idle. Splitting the KV dimension lets
    # multiple CTAs cooperate on one sequence (partials reduced via an fp32
    # workspace). The best split is shape-dependent; choosing it is owned
    # ENTIRELY by the op's AutoTuner (profiled per shape over
    # ``CuteDSLNVMlaDecodeBlackwellRunner.get_split_kv_candidates``, cached, and
    # baked into the CUDA graph at capture). This FMHA layer only sizes the
    # workspace for the largest candidate; see ``_run_mla_decode``.

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
        return supported

    @staticmethod
    def _is_perf_favorable(num_heads: int, seq_len_q: int) -> tuple[bool, str]:
        """Perf-only allowlist, separate from the correctness checks: admit
        just the (num_heads, seq_len_q) shapes where CuteDSL decode is an
        end-to-end win over the default backend.

        Shapes where the CuTe DSL decode kernel BEATS the default TRTLLM path
        end-to-end (DeepSeek-V3 8xB200 TP=8/EP=8 A/B, ISL1024/OSL2048):
          H=16  (TP=8, attention-DP off): seq_len_q=2 +1.0%, seq_len_q=4 +2.2%
          H=128 (attention-DP on)       : seq_len_q=1 +1.4%
        Every other measured cell is at or below parity (H=128/seq_len_q=4 is
        about -14%), so the gate admits only the winning shapes and lets
        everything else fall back to the next FMHA library."""
        perf_favorable_shapes = frozenset({(16, 2), (16, 4), (128, 1)})
        if (num_heads, seq_len_q) in perf_favorable_shapes:
            return True, ""
        return False, (
            f"CuTe DSL MLA decode is not a perf win for num_heads={num_heads}, "
            f"seq_len_q={seq_len_q}; allowed (num_heads, seq_len_q): "
            f"{sorted(perf_favorable_shapes)}."
        )

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
        # Perf gate (NOT a correctness limit): only admit shapes where CuteDSL
        # beats the default path E2E; everything else falls back.
        favorable, reason = self._is_perf_favorable(attn.num_heads, seq_len_q)
        if not favorable:
            return False, reason
        if meta.kv_cache_block_offsets is None:
            return False, "Paged KV block offsets are required."
        if meta.kv_cache_manager is None:
            return False, "KV cache manager is required."
        pool_mapping = meta.host_kv_cache_pool_mapping
        if pool_mapping is None:
            return False, "KV cache pool mapping is required."
        if fwd.latent_cache is None:
            return False, "latent_cache is required."
        if fwd.output is None:
            return False, "output is required."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = getattr(meta.kv_cache_manager, "tokens_per_block", 0)
        if tokens_per_block <= 1:
            return (
                False,
                f"tokens_per_block must be greater than 1, got {tokens_per_block}.",
            )

        # The kernel type is the input dtype
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
        # Real kernel input/output torch dtypes: the input is fp8 on the fp8-KV
        # path (``kernel_dtype``), the output is written straight into
        # ``fwd.output`` (no temp buffer in ``_run_mla_decode``), so its dtype is
        # the authoritative output dtype -- can_implement rejects the request if
        # the kernel cannot emit it. ``_kernel_can_implement`` converts both to
        # cutlass dtypes internally.
        return self._kernel_can_implement(
            kernel_dtype,
            fwd.output.dtype,
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

        # dtype / batch / MLA-dim validity is already enforced before dispatch
        # (``is_available`` + ``_is_supported_with_reason`` + the kernel's
        # ``can_implement``), so they are taken as given here.
        if kernel_dtype == torch.float8_e4m3fn:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
        else:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell

        num_tokens = q.shape[0]
        batch_size = params.num_requests
        seq_len_q = num_tokens // batch_size

        d_latent = attn.kv_lora_rank
        d_rope = attn.qk_rope_head_dim
        qk_nope_head_dim = attn.qk_nope_head_dim
        num_heads = attn.num_heads
        page_size = params.tokens_per_block

        if kernel_dtype == torch.float8_e4m3fn and params.fwd.quant_q_buffer is not None:
            q_kernel = params.fwd.quant_q_buffer.view(torch.float8_e4m3fn).view_as(q)
        else:
            q_kernel = q if q.dtype == kernel_dtype else q.to(kernel_dtype)
        q_view = q_kernel.view(batch_size, seq_len_q, num_heads, d_latent + d_rope)

        kv_pool = meta.kv_cache_manager.get_buffers(attn.layer_idx)
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
        pool_mapping = meta.host_kv_cache_pool_mapping
        # Select this layer's [num_seqs, max_blocks] page table from the 4D
        # kv_cache_block_offsets via the layer -> pool mapping.
        # ``host_kv_cache_pool_mapping`` is indexed by the LOCAL (compacted)
        # layer index, not the global ``attn.layer_idx`` -- they coincide for a
        # full model but differ when the KV cache manager allocates a subset of
        # layers (e.g. PP, or the layer-wise benchmark's ``layer_mask``).
        local_layer_idx = attn.get_local_layer_idx(meta)
        pool_idx = int(pool_mapping[local_layer_idx, 0])
        page_table_layer = block_offsets[pool_idx, :, 0, :]
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

        # Split-KV parallelism is owned ENTIRELY by the op's AutoTuner: it
        # profiles the per-shape split_kv candidates
        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )
        import cutlass

        workspace = params.workspace
        # split_kv is owned by the op's AutoTuner (profiled-best per shape, baked
        # into the graph), not passed here. Variable split-KV (block_split_kvs) is
        # unused on this path (the runner's is_var_split_kv default is False).

        softmax_scale = float(1.0 / (math.sqrt(qk_nope_head_dim + d_rope) * attn.q_scaling))
        output_scale = 1.0
        if kernel_dtype == torch.float8_e4m3fn:
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

        # The kernel writes straight into ``output``. The gate's can_implement
        # (queried with the real input AND output dtype) already rejected any
        # output dtype the kernel cannot emit, so no temp buffer / dtype-convert
        # copy is needed here.
        o_kernel = output_view.permute(2, 3, 1, 0)
        lse_storage = torch.empty(
            (batch_size, seq_len_q, num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        lse = lse_storage.permute(2, 1, 0)

        op(
            q_latent,
            q_rope,
            c_pool_latent,
            c_pool_rope,
            page_table,
            cache_seqs_base,
            o_kernel,
            lse,
            workspace,
            num_heads,
            seq_len_q,
            page_size,
            softmax_scale,
            output_scale,
        )

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

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        import cutlass

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )

        required_workspace_size = CuteDSLNVMlaDecodeBlackwellRunner.get_max_workspace_size(
            self.attn.num_heads,
            q.shape[0] // metadata.num_generations,
            self.attn.kv_lora_rank,
            metadata.num_generations,
            cutlass.Float32,
        )
        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the required size for Cute DSL MLA decode."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))
