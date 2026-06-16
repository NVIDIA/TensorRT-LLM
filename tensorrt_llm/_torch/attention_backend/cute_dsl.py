# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL attention backend.

Subclasses ``TrtllmAttention`` and overrides ``forward`` to dispatch the
MLA decode-only path into one of two Blackwell CuTe DSL kernels via a
single shared dispatcher (``_dispatch_cute_dsl_mla_decode``) parameterised
by the kernel's input dtype:

- ``kernel_dtype == torch.float8_e4m3fn``
    → ``torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell``
- ``kernel_dtype in (torch.float16, torch.bfloat16)``
    → ``torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell``

``forward`` picks the kernel dtype directly from runtime state
(``has_fp8_kv_cache`` → FP8; otherwise ``q.dtype`` when it is fp16/bf16 and
matches the KV cache dtype → FP16/BF16 via the FP16 op; neither match →
TRTLLM fallback). Every other code path (context /
chunked prefill / cached-KV MLA context / non-MLA / mixed batches /
unsupported SM) goes through ``super().forward`` unchanged.

Subclassing ``TrtllmAttention`` matters: ``modules/attention.py`` selects
the MLA chunked-prefill / cached-context fast paths via
``isinstance(self.mha, TrtllmAttention)``, so the CUTEDSL backend must
satisfy that check or those paths would silently fall back to the slow
default context path. This mirrors the MoE CuTeDSL integration
(``CuteDslFusedMoE(CutlassFusedMoE)``).
"""

import math
import os
from typing import Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from .interface import (AttentionForwardArgs, CustomAttentionMask,
                        PredefinedAttentionMask)
from .trtllm import TrtllmAttention, TrtllmAttentionMetadata

# log2(e): the CuTe DSL MLA kernels compute softmax via exp2 and fold this
# factor into the scale internally; the TRTLLM-produced mla_bmm1_scale already
# has it folded, so we divide it back out before handing the scale to the kernel.
_LOG2_E = math.log2(math.e)

# cutlass / kernel-class imports were used by the now-removed in-eligibility
# ``can_implement`` checks — the dispatch now goes through
# ``torch.ops.trtllm.cute_dsl_mla_decode_*_blackwell`` and the kernel-level
# Runner runs ``can_implement`` itself. ``IS_CUTLASS_DSL_AVAILABLE`` is
# still consulted in the eligibility preconditions below to short-circuit
# environments without the cutlass package.


class CuteDslAttention(TrtllmAttention):
    """CuteDSL attention backend.

    Inherits the full ``TrtllmAttention`` machinery (metadata, KV cache,
    quant flags, RoPE buffers, MLA helpers) and overrides ``forward`` to
    intercept the MLA decode-only batch. Anything that fails eligibility
    falls through to ``TrtllmAttention.forward``.
    """

    Metadata = TrtllmAttentionMetadata

    # ------------------------------------------------------------------
    # Shared preconditions (everything except dtype-specific gates).
    # ------------------------------------------------------------------
    def _cute_dsl_mla_decode_common_preconditions(
        self,
        metadata: TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs],
    ) -> bool:
        """Checks that are identical for the FP8 and FP16 paths.

        Note: phase routing (is_mla_enable / num_contexts / num_generations)
        is already handled in ``forward`` before either dtype's eligibility
        method runs, so it isn't repeated here.

        The dtype-specific eligibility methods own everything else
        (KV-cache dtype, ``q.dtype``, kernel-level ``can_implement``).
        """
        if not IS_CUTLASS_DSL_AVAILABLE:
            return False
        if get_sm_version() not in (100, 103):
            return False
        if self.predicted_tokens_per_seq is None or not (1 <= self.predicted_tokens_per_seq <= 4):
            return False
        if metadata.kv_cache_block_offsets is None:
            return False
        if forward_args is None or forward_args.latent_cache is None:
            return False
        return True

    # ==================================================================
    # MLA decode dispatch (FP8 / FP16)
    # ==================================================================

    def _dispatch_cute_dsl_mla_decode(
        self,
        q: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
        output: torch.Tensor,
        kernel_dtype: torch.dtype,
    ) -> torch.Tensor:
        """MLA decode dispatch shared by FP8 and FP16/BF16 paths.

        ``kernel_dtype`` is the in/out tensor dtype the chosen CuTe DSL
        kernel expects — ``torch.float8_e4m3fn`` for the FP8 kernel, or
        ``torch.float16`` / ``torch.bfloat16`` for the FP16 kernel class. The
        op called is selected from it.

        Assumes the MLA module has already (1) built ``q`` as the fused
        ``[num_tokens, H * (D_latent + D_rope)]`` tensor with RoPE applied
        to the rope half, and (2) appended the new token to the paged
        latent cache.
        """
        if kernel_dtype == torch.float8_e4m3fn:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
        elif kernel_dtype in (torch.float16, torch.bfloat16):
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
        else:
            raise ValueError(
                f"CuteDslAttention: unsupported kernel_dtype={kernel_dtype}; "
                "expected torch.float8_e4m3fn, torch.float16, or torch.bfloat16"
            )

        num_tokens = q.shape[0]
        num_seqs = metadata.num_generations
        seq_len_q = num_tokens // num_seqs
        assert seq_len_q * num_seqs == num_tokens, (
            f"CuteDslAttention MLA decode expects num_tokens "
            f"({num_tokens}) divisible by num_generations ({num_seqs})"
        )

        # Both kernels: L == 512, R == 64. FP16/BF16 keeps in/out dtype equal
        # to kernel_dtype; FP8 widens the attention output to bf16 below.
        d_latent = self.kv_lora_rank
        d_rope = self.qk_rope_head_dim
        h = self.num_heads
        page_size = metadata.tokens_per_block

        # q → [H, D, S_q, B].
        # FP8: use the properly-scaled fp8 query that ``mla_rope_generation``
        # already produced (``quant_q_buffer``) instead of a raw ``q.to(fp8)``.
        # The raw cast quantizes the bf16 q with scale 1, which is ~10x less
        # accurate than the scaled quantization (whose dequant is folded into
        # ``mla_bmm1_scale`` below) and blows past the fp8 test tolerance.
        # FP16/BF16: q is already the kernel dtype (or cast losslessly).
        if kernel_dtype == torch.float8_e4m3fn and forward_args.quant_q_buffer is not None:
            q_kernel = forward_args.quant_q_buffer.view(torch.float8_e4m3fn).view_as(q)
        else:
            q_kernel = q if q.dtype == kernel_dtype else q.to(kernel_dtype)
        # Kernel layout for the q tensors is logical [H, D, S_q, B] with the D
        # axis contiguous (stride 1) — see ``mark_layout_dynamic(leading_dim=1)``
        # on the op side and the kernel reference (permute_order=(2,3,1,0)).
        # ``q_view`` is [B, S_q, H, D] contiguous, so the permuted *view*
        # already has stride-1 on D; calling ``.contiguous()`` here would
        # re-lay-out to stride-1 on B and violate the kernel's contract.
        q_view = q_kernel.view(num_seqs, seq_len_q, h, d_latent + d_rope)
        q_latent = q_view[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = q_view[..., d_latent:].permute(2, 3, 1, 0)

        # Paged MLA pool view as the kernel's dtype.
        # NOTE: pool tensor handle and block-table layout for MLA depends
        # on kv-cache-manager wiring; revisit if ``get_buffers`` exposes a
        # different layout.
        kv_pool = metadata.kv_cache_manager.get_buffers(self.layer_idx)
        if kernel_dtype in (torch.float16, torch.bfloat16) and kv_pool.dtype != kernel_dtype:
            raise ValueError(
                f"CuteDslAttention MLA decode {kernel_dtype} fast path requires "
                f"matching KV cache dtype, got {kv_pool.dtype}"
            )

        # Paged-pool layout normalization (v1 vs v2 KV cache manager).
        # ``get_buffers`` returns a per-layer view of shape
        # [num_blocks, kv_factor, page_size, num_kv_heads, head_dim]. The v2
        # manager gives each layer its own densely-packed pool, so the block
        # (dim-0) stride equals the packed block size. The v1 manager stores all
        # layers interleaved in ONE pool, so the per-layer view's block stride is
        # ``layers_in_pool * packed_block`` (non-packed) with a per-layer
        # ``storage_offset``. The CuTe DSL paged TMA addresses pages with the
        # packed block stride and does not honor a non-packed one, so for v1 it
        # would read the wrong (interleaved) layer's memory. Re-expose the
        # underlying pool as a packed ``[num_blocks * layers_in_pool, ...]`` view
        # (copy-free) and fold the layer offset into the page table below:
        #   combined_block = block * layers_in_pool + layer_in_pool
        # For v2 ``layers_in_pool == 1`` and this is a no-op.
        packed_block = 1
        for s in kv_pool.shape[1:]:
            packed_block *= s
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block if packed_block else 1
        layer_in_pool = 0
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            layer_in_pool = kv_pool.storage_offset() // packed_block
            kv_pool = kv_pool.as_strided(
                (kv_pool.shape[0] * layers_in_pool, ) + tuple(kv_pool.shape[1:]),
                (packed_block, ) + tuple(kv_pool.stride()[1:]),
                0,
            )
        else:
            layers_in_pool = 1

        kv_pool_typed = kv_pool.view(kernel_dtype)
        c_pool_latent = kv_pool_typed[..., :d_latent]
        c_pool_rope = kv_pool_typed[..., d_latent : d_latent + d_rope]

        block_offsets = metadata.kv_cache_block_offsets
        if block_offsets.dim() == 4:
            # kv_cache_block_offsets is [num_pools, num_seqs, 2, max_blocks].
            # dim 0 is the *pool* index, not the layer — resolve this layer's
            # pool via the layer→pool mapping ([num_layers, 2], col 0 = pool).
            # (Indexing dim 0 by ``layer_idx`` is only valid for layer 0 / a
            # single pool and goes out of bounds for every other layer.)
            pool_mapping = metadata.host_kv_cache_pool_mapping
            pool_idx = int(pool_mapping[self.layer_idx, 0])
            page_table_layer = block_offsets[pool_idx, :, 0, :]
        elif block_offsets.dim() == 3:
            page_table_layer = block_offsets[:, 0, :]
        else:
            page_table_layer = block_offsets
        # Kernel: [max_pages, B], leading_dim=0 ⇒ pages contiguous (stride 1).
        # ``page_table_layer`` is [B, max_pages] with max_pages stride 1, so the
        # transposed *view* already has stride 1 on dim 0; ``.contiguous()``
        # would move stride 1 to B and violate the kernel's layout contract.
        page_table = page_table_layer.transpose(0, 1).to(torch.int32)

        # Fold the interleaved-pool layer offset into the page table so it
        # indexes the packed combined-slot view built above. The v1 block
        # offsets are already in combined-slot units (block * layers_in_pool),
        # so only the per-layer offset has to be added. No-op for v2
        # (layers_in_pool == 1, layer_in_pool == 0).
        if layers_in_pool > 1:
            page_table = page_table + layer_in_pool

        cache_seqs = metadata.kv_lens_cuda_runtime.to(torch.int32)

        if os.environ.get("TLLM_CUTE_DSL_DUMP"):
            print(
                "[CUTEDSL_DUMP] layer=%d kv_pool.shape=%s kv_pool.stride=%s "
                "contiguous=%s block_offsets.shape=%s page_table.shape=%s "
                "page_table=%s cache_seqs=%s" % (
                    self.layer_idx,
                    tuple(kv_pool.shape),
                    tuple(kv_pool.stride()),
                    kv_pool.is_contiguous(),
                    tuple(block_offsets.shape),
                    tuple(page_table.shape),
                    page_table.t().tolist() if page_table.numel() < 64 else "(big)",
                    cache_seqs[:8].tolist(),
                ),
                flush=True,
            )
            _pm = getattr(metadata, "host_kv_cache_pool_mapping", None)
            print(
                "[CUTEDSL_DUMP2] layer=%d pool_mapping=%s "
                "raw_block_offsets[...,0,:]=%s" % (
                    self.layer_idx,
                    _pm.tolist() if _pm is not None else None,
                    block_offsets[..., 0, :].tolist()
                    if block_offsets.dim() == 4 and block_offsets.numel() < 128
                    else "(big/other)",
                ),
                flush=True,
            )

        split_kv = 1
        workspace = torch.empty(0, dtype=torch.float32, device=q.device)
        block_split_kvs = torch.empty(0, dtype=torch.int32, device=q.device)

        # ``o`` mirrors the q layout: logical [H, D, S_q, B] with D contiguous
        # (leading_dim=1). Allocate [B, S_q, H, D] contiguous and permute so the
        # D axis keeps stride 1 (a plain contiguous [H, D, S_q, B] would put
        # stride 1 on B and fail the op's layout check).
        #
        # For the FP8 kernel we force the output dtype to bf16 rather than fp8:
        # the kernel's epilogue stores through ``o.element_type`` with a
        # width-adaptive copy (so bf16 is supported — see the kernel's
        # ``can_implement``), and writing the attention result straight to bf16
        # avoids an extra fp8 round-trip on the output. The FP16 kernel class
        # keeps its native fp16/bf16 output.
        out_kernel_dtype = torch.bfloat16 if kernel_dtype == torch.float8_e4m3fn else kernel_dtype
        o_kernel = torch.empty(
            (num_seqs, seq_len_q, h, d_latent),
            dtype=out_kernel_dtype,
            device=q.device,
        ).permute(2, 3, 1, 0)
        # ``lse`` is logical [H, S_q, B] with H contiguous (leading_dim=0); the
        # reference builds it as [B, S_q, H] then permute_order=(2,1,0).
        lse = torch.empty(
            (num_seqs, seq_len_q, h),
            dtype=torch.float32,
            device=q.device,
        ).permute(2, 1, 0)

        # MLA softmax scale follows the canonical TRT-LLM formula
        # (see modules/attention.py:1453):
        #   softmax_scale = 1 / (sqrt(qk_head_dim) * q_scaling)
        # where ``qk_head_dim`` is the *unabsorbed* Q head dim
        # ``qk_nope_head_dim + qk_rope_head_dim``. We deliberately do NOT
        # use ``(kv_lora_rank + qk_rope_head_dim)`` even though that is
        # the absorbed attention's inner dimension — the scale stays bound
        # to the original head dim so attention scores match the
        # unabsorbed reference. ``self.q_scaling`` carries any YaRN
        # ``mscale`` adjustment (set by the MLA module via
        # ``q_scaling = 1 / (mscale * mscale)`` — see attention.py:1428).
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        softmax_scale = float(1.0 / (math.sqrt(qk_head_dim) * self.q_scaling))
        output_scale = 1.0
        if (kernel_dtype == torch.float8_e4m3fn
                and forward_args.mla_bmm1_scale is not None
                and forward_args.mla_bmm2_scale is not None):
            # ``mla_rope_generation`` folds the fp8 dequant (1/(q_scale*k_scale))
            # and the softmax scale into ``mla_bmm1_scale[1]`` (the FMHA
            # scaleBmm1, see attentionOp.cpp bmm1_scale_offset=1), and the
            # output/V dequant into ``mla_bmm2_scale[0]``. The CuTe DSL kernel
            # reads q/KV as raw fp8 and applies these two scalar scales, so
            # reusing them dequantizes identically to the TRTLLM FP8 MLA path.
            # mla_bmm1_scale[1] is the TRTLLM FMHA scaleBmm1, which has log2(e)
            # PRE-folded (its softmax uses exp2). The CuTe DSL kernel folds
            # log2(e) AGAIN internally (softmax_scale_log2 = softmax_scale *
            # LOG2_E), so pass the de-folded value to avoid applying it twice.
            #
            # CUDA-graph safety: ``.item()`` is a device->host copy + sync, which
            # is illegal while a CUDA graph stream is capturing. These fp8
            # dequant scales are STATIC per layer — ``mla_rope_generation``
            # derives them from static quantization state (kv_scale_quant_orig,
            # q_scaling, quant_mode), not from per-step activations — so we read
            # them once (eagerly) and cache the host float on this per-layer
            # backend instance. The generation CUDA-graph capture runs
            # ``WARMUP_STEPS`` eager decode passes over every layer first
            # (cuda_graph_runner.capture), so the cache is always populated
            # before capture and the ``.item()`` never runs under capture.
            cached = getattr(self, "_cute_dsl_fp8_scale", None)
            if cached is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CuteDslAttention: fp8 MLA decode scale not cached for "
                        f"layer {self.layer_idx} before CUDA graph capture; "
                        "reading it via .item() during capture is illegal. The "
                        "eager warmup should have populated the cache.")
                softmax_scale = float(
                    forward_args.mla_bmm1_scale[1].item()) / _LOG2_E
                output_scale = float(forward_args.mla_bmm2_scale[0].item())
                self._cute_dsl_fp8_scale = (softmax_scale, output_scale)
            else:
                softmax_scale, output_scale = cached
                # Optional staticness check: re-read the live scale eagerly and
                # warn if it ever drifts from the cached value (would mean the
                # static-scale assumption — and thus the cache — is wrong).
                if (os.environ.get("TLLM_CUTE_DSL_SCALE_DUMP")
                        and not torch.cuda.is_current_stream_capturing()):
                    live_sm = float(
                        forward_args.mla_bmm1_scale[1].item()) / _LOG2_E
                    live_os = float(forward_args.mla_bmm2_scale[0].item())
                    print(
                        "[CUTEDSL_SCALE] layer=%d cached=(%.8f,%.8f) "
                        "live=(%.8f,%.8f) drift=%s" % (
                            self.layer_idx, softmax_scale, output_scale,
                            live_sm, live_os,
                            abs(live_sm - softmax_scale) > 1e-6
                            or abs(live_os - output_scale) > 1e-6),
                        flush=True,
                    )

        # Paged cache layout for the kernel is logical [page_size, D, num_pages]
        # with the D axis contiguous (leading_dim=1) — see the kernel reference
        # (cache shape (num_pages, page_size, D), permute_order=(1,2,0)).
        # ``get_buffers`` returns the pool as
        # [num_pages, kv_factor=1, page_size, num_kv_heads=1, head_dim]; drop the
        # two singleton axes and permute so D keeps stride 1. The latent/rope
        # slices already carry stride 1 on their (last) D axis, so no copy.
        c_latent_kernel = c_pool_latent.squeeze(3).squeeze(1).permute(1, 2, 0)
        c_rope_kernel = c_pool_rope.squeeze(3).squeeze(1).permute(1, 2, 0)

        op(
            q_latent,
            q_rope,
            c_latent_kernel,
            c_rope_kernel,
            page_table,
            cache_seqs,
            block_split_kvs,
            o_kernel,
            lse,
            workspace,
            self.num_heads,
            seq_len_q,
            page_size,
            True,  # is_persistent
            True,  # is_var_seq
            False,  # is_var_split_kv
            split_kv,
            softmax_scale,
            output_scale,
        )

        attn_out = o_kernel.permute(3, 2, 0, 1).reshape(num_tokens, h * d_latent)
        output.copy_(attn_out.to(output.dtype))
        return output

    # ==================================================================
    # forward dispatch
    # ==================================================================

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        if forward_args is None and kwargs:
            forward_args = AttentionForwardArgs(**kwargs)
            kwargs = {}

        # Phase routing — only the MLA decode-only batch is eligible for the
        # CuTe DSL fast path. Everything else falls through to TRTLLM:
        #   - non-MLA attention                       → TRTLLM
        #   - prefill / mixed batch (has context)     → TRTLLM
        #   - decode-only MLA                         → try CuteDSL FP8/FP16/BF16,
        #                                               TRTLLM on failure
        is_decode_only_mla = (
            self.is_mla_enable and metadata.num_contexts == 0 and metadata.num_generations > 0
        )

        # The CuTe DSL MLA decode kernel is non-causal: every query token
        # attends to the whole cached KV with no per-query mask. That is correct
        # for the plain decode case (the MLA module passes the default CAUSAL /
        # FULL predefined mask with no mask tensor), but it cannot honor any
        # per-query causal masking. Fall back to TRTLLM whenever masking the
        # kernel cannot express is required:
        #   - an explicit CUSTOM mask or mask tensor, or
        #   - speculative decoding (MTP / Eagle / tree) — signalled by
        #     ``metadata.use_spec_decoding``; those steps feed >1 query token per
        #     request and require a causal/tree mask the non-causal kernel lacks.
        attention_mask = (
            forward_args.attention_mask
            if forward_args is not None else PredefinedAttentionMask.CAUSAL
        )
        attention_mask_data = (
            forward_args.attention_mask_data if forward_args is not None else None
        )
        mask_supported = (
            attention_mask != CustomAttentionMask.CUSTOM
            and attention_mask_data is None
            and not getattr(metadata, "use_spec_decoding", False)
        )

        if (
            is_decode_only_mla
            and mask_supported
            and self._cute_dsl_mla_decode_common_preconditions(metadata, forward_args)
            and q.shape[0] % metadata.num_generations == 0
        ):
            # Direct dtype-based dispatch (no per-dtype eligibility helper).
            # The kernel-level Runner runs ``can_implement`` for the chosen
            # dtype; anything it rejects falls through to TRTLLM via the
            # try/except below.
            if getattr(self, "has_fp8_kv_cache", False):
                kernel_dtype = torch.float8_e4m3fn
            elif q.dtype in (torch.float16, torch.bfloat16):
                kv_pool_dtype = metadata.kv_cache_manager.get_buffers(self.layer_idx).dtype
                kernel_dtype = q.dtype if kv_pool_dtype == q.dtype else None
            else:
                kernel_dtype = None

            if kernel_dtype is not None:
                try:
                    output = q.new_empty(
                        (q.shape[0], self.num_heads * self.kv_lora_rank), dtype=q.dtype
                    )
                    result = self._dispatch_cute_dsl_mla_decode(
                        q, metadata, forward_args, output, kernel_dtype
                    )
                    # Positive confirmation (once per process) that the CuteDSL
                    # MLA decode kernel was actually engaged — symmetric to the
                    # fallback warning below so a real kernel call can never be
                    # confused with a silent TRTLLM fallback.
                    logger.warning_once(
                        "CuteDslAttention: MLA decode fast path ENGAGED "
                        "(kernel_dtype=%s); CuteDSL kernel called." % kernel_dtype,
                        key="cute_dsl_mla_decode_engaged",
                    )
                    return result
                except Exception as exc:  # noqa: BLE001
                    # Surface the fallback once per process (keyed dedup keeps
                    # it from flooding the log on every decode step) so a broken
                    # kernel can never silently masquerade as a working one.
                    logger.warning_once(
                        "CuteDslAttention: MLA decode fast path "
                        "(kernel_dtype=%s) failed (%s); falling back "
                        "to TRTLLM backend." % (kernel_dtype, exc),
                        key="cute_dsl_mla_decode_fallback",
                    )

        return super().forward(q, k, v, metadata, forward_args=forward_args, **kwargs)
