# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL attention backend.

Subclasses ``TrtllmAttention`` and overrides ``forward`` to dispatch the
MLA decode-only path into one of two Blackwell CuTe DSL kernels via a
single shared dispatcher (``_dispatch_cute_dsl_mla_decode``) parameterised
by the kernel's input dtype:

- ``kernel_dtype == torch.float8_e4m3fn``
    → ``torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell``
- ``kernel_dtype == torch.float16``
    → ``torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell``

``forward`` picks the kernel dtype directly from runtime state
(``has_fp8_kv_cache`` → FP8; otherwise ``q.dtype == torch.float16`` → FP16;
neither match → TRTLLM fallback). Every other code path (context /
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
from .interface import AttentionForwardArgs
from .trtllm import TrtllmAttention, TrtllmAttentionMetadata

_DEBUG_FALLBACK = os.environ.get("TLLM_CUTE_DSL_ATTN_DEBUG_FALLBACK", "0") == "1"

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
        """MLA decode dispatch shared by FP8 and FP16 paths.

        ``kernel_dtype`` is the in/out tensor dtype the chosen CuTe DSL
        kernel expects — ``torch.float8_e4m3fn`` for the FP8 kernel,
        ``torch.float16`` for the FP16 kernel. The op called is selected
        from it.

        Assumes the MLA module has already (1) built ``q`` as the fused
        ``[num_tokens, H * (D_latent + D_rope)]`` tensor with RoPE applied
        to the rope half, and (2) appended the new token to the paged
        latent cache.
        """
        if kernel_dtype == torch.float8_e4m3fn:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
        elif kernel_dtype == torch.float16:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
        else:
            raise ValueError(
                f"CuteDslAttention: unsupported kernel_dtype={kernel_dtype}; "
                "expected torch.float8_e4m3fn or torch.float16"
            )

        num_tokens = q.shape[0]
        num_seqs = metadata.num_generations
        seq_len_q = num_tokens // num_seqs
        assert seq_len_q * num_seqs == num_tokens, (
            f"CuteDslAttention MLA decode expects num_tokens "
            f"({num_tokens}) divisible by num_generations ({num_seqs})"
        )

        # Both kernels: L == 512, R == 64, in/out dtype == kernel_dtype.
        d_latent = self.kv_lora_rank
        d_rope = self.qk_rope_head_dim
        h = self.num_heads
        page_size = metadata.tokens_per_block

        # q → [H, D, S_q, B]. Cast to kernel dtype if upstream q is in
        # something else (e.g. bf16 model + FP8 KV — lossy but defined).
        q_kernel = q if q.dtype == kernel_dtype else q.to(kernel_dtype)
        q_view = q_kernel.view(num_seqs, seq_len_q, h, d_latent + d_rope)
        q_latent = q_view[..., :d_latent].permute(2, 3, 1, 0).contiguous()
        q_rope = q_view[..., d_latent:].permute(2, 3, 1, 0).contiguous()

        # Paged MLA pool view as the kernel's dtype.
        # NOTE: pool tensor handle and block-table layout for MLA depends
        # on kv-cache-manager wiring; revisit if ``get_buffers`` exposes a
        # different layout.
        kv_pool = metadata.kv_cache_manager.get_buffers(self.layer_idx)
        kv_pool_typed = kv_pool.view(kernel_dtype)
        c_pool_latent = kv_pool_typed[..., :d_latent]
        c_pool_rope = kv_pool_typed[..., d_latent : d_latent + d_rope]

        block_offsets = metadata.kv_cache_block_offsets
        if block_offsets.dim() == 4:
            page_table_layer = block_offsets[self.layer_idx, :, 0, :]
        elif block_offsets.dim() == 3:
            page_table_layer = block_offsets[:, 0, :]
        else:
            page_table_layer = block_offsets
        # Kernel: [max_pages, B], leading_dim=0 ⇒ pages contiguous per B.
        page_table = page_table_layer.transpose(0, 1).contiguous().to(torch.int32)

        cache_seqs = metadata.kv_lens_cuda_runtime.to(torch.int32)

        split_kv = 1
        workspace = torch.empty(0, dtype=torch.float32, device=q.device)
        block_split_kvs = torch.empty(0, dtype=torch.int32, device=q.device)

        o_kernel = torch.empty(
            (h, d_latent, seq_len_q, num_seqs),
            dtype=kernel_dtype,
            device=q.device,
        )
        lse = torch.empty(
            (h, seq_len_q, num_seqs),
            dtype=torch.float32,
            device=q.device,
        )

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

        c_latent_kernel = c_pool_latent.unsqueeze(-1).contiguous()
        c_rope_kernel = c_pool_rope.unsqueeze(-1).contiguous()

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
        #   - decode-only MLA                         → try CuteDSL FP8/FP16,
        #                                               TRTLLM on failure
        is_decode_only_mla = (
            self.is_mla_enable and metadata.num_contexts == 0 and metadata.num_generations > 0
        )

        if (
            is_decode_only_mla
            and self._cute_dsl_mla_decode_common_preconditions(metadata, forward_args)
            and q.shape[0] % metadata.num_generations == 0
        ):
            # Direct dtype-based dispatch (no per-dtype eligibility helper).
            # The kernel-level Runner runs ``can_implement`` for the chosen
            # dtype; anything it rejects falls through to TRTLLM via the
            # try/except below.
            if getattr(self, "has_fp8_kv_cache", False):
                kernel_dtype = torch.float8_e4m3fn
            elif q.dtype == torch.float16:
                kernel_dtype = torch.float16
            else:
                kernel_dtype = None

            if kernel_dtype is not None:
                try:
                    output = q.new_empty(
                        (q.shape[0], self.num_heads * self.kv_lora_rank), dtype=q.dtype
                    )
                    return self._dispatch_cute_dsl_mla_decode(
                        q, metadata, forward_args, output, kernel_dtype
                    )
                except Exception as exc:  # noqa: BLE001
                    if _DEBUG_FALLBACK:
                        logger.warning(
                            "CuteDslAttention: MLA decode fast path "
                            "(kernel_dtype=%s) failed (%s); falling back "
                            "to TRTLLM backend.",
                            kernel_dtype,
                            exc,
                        )

        return super().forward(q, k, v, metadata, forward_args=forward_args, **kwargs)
