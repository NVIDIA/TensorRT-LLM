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

from .interface import FmhaPhase
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
        if attn.flashinfer_mla_backend is not None:
            logger.debug(
                "Standalone CuTe DSL MLA FMHA is unavailable: an explicit "
                "flashinfer_mla_backend delegates MLA generation to "
                "FlashInferTrtllmGenFmha."
            )
            return False

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
        # Sparse attention (DSA / RocketKV / skip-softmax): the CuTe DSL kernel
        # computes dense attention over the full paged KV and cannot honor
        # predicted sparse indices; accepting such a layer would silently drop
        # them and produce wrong results.
        if getattr(attn, "sparse_params", None) is not None:
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: sparse attention "
                f"({type(attn.sparse_params).__name__}) is not supported."
            )
            return False
        if attn.predicted_tokens_per_seq is None or attn.predicted_tokens_per_seq < 1:
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: predicted_tokens_per_seq "
                f"must be >= 1, got {attn.predicted_tokens_per_seq}."
            )
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

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase not in (None, FmhaPhase.GENERATION):
            return False
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
    def _is_perf_favorable(
        num_heads: int,
        batch_size: Optional[int],
        seq_len_q: int,
        kernel_dtype: Optional[torch.dtype],
    ) -> tuple[bool, str]:
        """Perf-only gate, separate from the correctness checks, split by the
        kernel input dtype because the measured win regions differ.

        fp8 KV: admit a (num_heads, seq_len_q) shape only above its measured
        critical batch size (``_PERF_MIN_BATCH_FP8``); everything else falls
        back to the next FMHA library.

        bf16/fp16 KV: admit the measured Kimi K3 DEP/TEP head counts and the
        existing 16-head path.

        ``batch_size=None`` evaluates only the batch-size-independent
        conditions (dtype, num_heads, seq_len_q) and skips the batch floor."""
        _BF16_FP16_PERF_FAVORABLE_HEAD_COUNTS = (6, 12, 16, 96)
        if kernel_dtype != torch.float8_e4m3fn:
            if num_heads in _BF16_FP16_PERF_FAVORABLE_HEAD_COUNTS:
                return True, ""
            return False, (
                f"CuTe DSL MLA decode on {kernel_dtype} KV is only enabled "
                "for num_heads in "
                f"{list(_BF16_FP16_PERF_FAVORABLE_HEAD_COUNTS)}, "
                f"got num_heads={num_heads}."
            )
        # Minimum per-rank decode batch size at which the CuteDSL kernel beats
        # the default TRTLLM path on the FP8-KV path, keyed by
        # (num_heads, seq_len_q).
        _PERF_MIN_BATCH_FP8 = {
            (6, 1): 1,
            (12, 1): 1,
            (16, 2): 64,
            (16, 4): 32,
            (16, 8): 16,
            # Keep H=96 off TRTLLM-Gen: its heuristic may select a 64-head
            # Q tile, which does not divide 96 and produces an invalid config.
            (96, 1): 1,
            (128, 1): 64,
            (128, 2): 32,
        }
        min_batch = _PERF_MIN_BATCH_FP8.get((num_heads, seq_len_q))
        if min_batch is None:
            return False, (
                f"CuTe DSL MLA decode is not a perf win for "
                f"num_heads={num_heads}, seq_len_q={seq_len_q}; allowed "
                f"(num_heads, seq_len_q): "
                f"{sorted(_PERF_MIN_BATCH_FP8)}."
            )
        if batch_size is not None and batch_size < min_batch:
            return False, (
                f"CuTe DSL MLA decode wins for num_heads={num_heads}, "
                f"seq_len_q={seq_len_q} only at batch_size >= {min_batch}; "
                f"got batch_size={batch_size}."
            )
        return True, ""

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> tuple[bool, str]:
        if fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "CuTe DSL MLA FMHA only supports generation-only attention."
        # Disable mixed context/generation batches until the CuTe DSL host
        # overhead is reduced. H=96 is exempt: TRTLLM-Gen may select a
        # 64-head Q tile, which does not divide 96 and produces an invalid
        # configuration after K3's 96-to-128 head padding was removed.
        if (meta.num_contexts != 0 and attn.num_heads != 96) or meta.num_generations <= 0:
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
        seq_len_q = q.shape[0] // meta.num_generations
        batch_size = meta.num_generations
        if meta.helix_position_offsets is not None:
            if seq_len_q != 1 and not meta._helix_spec_tokens_valid:
                # Multi-token decode under helix needs the per-token bound /
                # write-slot buffers of the speculative verify-group path.
                return False, "CuTe DSL MLA FMHA only supports single-token decode with Helix."
            if seq_len_q != 1 and self._get_kernel_dtype(
                    attn, q) == torch.float8_e4m3fn:
                return False, (
                    "CuTe DSL MLA FMHA helix verify groups require a bf16/fp16 "
                    "KV cache (the fp8 kernel has no per-token bounds).")
            softmax_stats = fwd.softmax_stats_tensor
            if softmax_stats is None:
                return False, "CuTe DSL MLA FMHA requires softmax_stats_tensor with Helix."
            expected_stats_shape = (q.shape[0], attn.num_heads, 2)
            if softmax_stats.shape != expected_stats_shape:
                return False, (
                    "CuTe DSL MLA FMHA requires Helix softmax stats with shape "
                    f"{expected_stats_shape}, got {tuple(softmax_stats.shape)}."
                )
            if softmax_stats.dtype != torch.float32 or not softmax_stats.is_contiguous():
                return False, (
                    "CuTe DSL MLA FMHA requires contiguous float32 Helix softmax stats, "
                    f"got dtype={softmax_stats.dtype}, contiguous={softmax_stats.is_contiguous()}."
                )

        from tensorrt_llm._torch.autotuner import AutoTuner

        # Perf gate (NOT a correctness limit).
        favorable, reason = self._is_perf_favorable(
            attn.num_heads,
            None if AutoTuner.get().is_tuning_mode else batch_size,
            seq_len_q,
            self._get_kernel_dtype(attn, q),
        )
        if not favorable:
            return False, reason
        if meta.kv_cache_manager is None:
            return False, "KV cache manager is required."
        if fwd.output is None:
            return False, "output is required."

        # The kernel type is the input dtype
        kernel_dtype = self._get_kernel_dtype(attn, q)
        if kernel_dtype is None:
            return (
                False,
                f"Unsupported dtype combination: q={q.dtype}, "
                f"has_fp8_kv_cache={getattr(attn, 'has_fp8_kv_cache', False)}.",
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
        # tiler the op launches with.
        return self._kernel_can_implement(
            kernel_dtype,
            fwd.output.dtype,
            meta.num_generations,
            seq_len_q,
            meta.tokens_per_block,
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
        # kv_cache_block_offsets.
        local_layer_idx = attn.get_local_layer_idx(meta)
        pool_idx = int(pool_mapping[local_layer_idx, 0])
        page_table_layer = block_offsets[pool_idx, :, 0, :]
        cache_seqs_base = params.sequence_lengths.to(torch.int32)
        page_table = page_table_layer[meta.num_contexts :].transpose(0, 1).to(torch.int32)
        if layers_in_pool > 1:
            page_table = page_table + layer_in_pool

        # KVCacheManager exposes NHD pages as [num_pages, 1, page_size, 1, head_dim].
        # The CuTe DSL kernel consumes a paged [page_size, dim, num_pages] view
        # with the dim axis contiguous.
        kv_pages = kv_pool_typed[:, 0, :, 0, : d_latent + d_rope]
        c_pool_latent = kv_pages[..., :d_latent].permute(1, 2, 0)
        c_pool_rope = kv_pages[..., d_latent:].permute(1, 2, 0)

        # ``params.workspace`` is the attention workspace SHARED with the C++
        # kernels, which size it from ``max_num_tokens`` and can grow it to
        # several GiB. The AutoTuner rebuilds every input carrying a
        # dynamic dim as a float32 ``torch.rand`` tensor while profiling, i.e.
        # 4x the int8 byte count, which may cause OOM.
        required_workspace_numel = math.ceil(
            self._required_workspace_size(num_heads, seq_len_q, d_latent, meta.max_num_requests)
            / params.workspace.element_size()
        )
        workspace = params.workspace[:required_workspace_numel]
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

        q_latent = q_view[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = q_view[..., d_latent:].permute(2, 3, 1, 0)

        o_kernel = output_view.permute(2, 3, 1, 0)

        op(
            q_latent,
            q_rope,
            c_pool_latent,
            c_pool_rope,
            page_table,
            cache_seqs_base,
            o_kernel,
            workspace,
            num_heads,
            seq_len_q,
            page_size,
            softmax_scale,
            output_scale,
            # Max batch size for the AutoTuner to profile.
            int(meta.max_num_requests),
            params.fwd.softmax_stats_tensor,
            # Per-token rank-local bounds, filled by
            # recompute_helix_spec_buffers. None everywhere else.
            (meta.helix_kv_bounds[:num_tokens] if
             (meta.helix_position_offsets is not None
              and meta._helix_spec_tokens_valid
              and kernel_dtype != torch.float8_e4m3fn) else None),
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

    def _required_workspace_size(
        self,
        num_heads: int,
        seq_len_q: int,
        kv_lora_rank: int,
        max_num_requests: int,
    ) -> int:
        """Bytes this kernel owns at the FRONT of the shared attention workspace."""
        import cutlass

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )

        return CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
            num_heads,
            seq_len_q,
            kv_lora_rank,
            max_num_requests,
            cutlass.Float32,
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
        required_workspace_size = self._required_workspace_size(
            self.attn.num_heads,
            q.shape[0] // metadata.num_generations,
            self.attn.kv_lora_rank,
            metadata.max_num_requests,
        )
        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the required size for Cute DSL MLA decode."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))
