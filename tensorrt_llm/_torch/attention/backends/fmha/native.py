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
"""Shared base of the FMHA libraries that run in the native TRT-LLM attention op."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm._utils import get_sm_version, is_sm_120f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class NativeFmha(PhasedFmha):
    """An FMHA library of the native attention op, reached through its own thop entry points.

    The op holds the library's kernel runners and the attention workspace; the library
    holds no state. Subclasses name the phases and attention variants they serve and
    implement ``native_is_supported`` and the ``native_run_*`` entry points of the phases
    they serve.
    """

    SERVES_CONTEXT: ClassVar[bool] = False
    SERVES_GENERATION: ClassVar[bool] = False
    SERVES_MHA: ClassVar[bool] = True
    SERVES_MLA: ClassVar[bool] = True

    REQUIRES_PAGED_KV = False
    # Native attention consumes pool pointers directly; sparse managers may have
    # no Role.KEY buffer from which to query a dense block extent.
    NEEDS_BLOCK_EXTENT = False
    supports_skip_correction = True
    supports_workspace_reclamation = True

    # Head sizes whose fused context FMHA kernel is proven absent for at
    # least one KV cache dtype, per SM version. This is a blocklist of
    # proven-problematic cells, not a support matrix: sourced from the
    # runtime fallback warning "Fall back to unfused MHA ... headSize = 64
    # ... in sm_103" and the silent-wrong-answer incident it caused. Inside
    # a blocklisted cell, the ``fused_context_fmha_kernel_exists`` native
    # lookup decides per KV dtype and page size which configurations the
    # running build can serve. Outside it, no probe runs: the lookup is a
    # fixed-convention diagnostic (dense causal Q_PAGED_KV, Q and output
    # precision inferred from the KV precision) that does not model every
    # configuration the op can run (MLA, cross attention, 16-bit context
    # math over an FP8 KV cache), so probing unconditionally would refuse
    # configurations the op serves correctly. The authoritative
    # exact-parameter check is the context preparation check in
    # thop/attentionOp.cpp; this gate exists to move the
    # proven-problematic cells to FMHA dispatch with a remedial error.
    # Extend as further combinations are proven.
    CONTEXT_FMHA_ABSENT_HEAD_DIMS: ClassVar[dict[int, tuple[int, ...]]] = {
        103: (64,),
    }

    @classmethod
    def _validate_paged_context_fmha(cls, metadata: "TrtllmAttentionMetadata") -> None:
        """Refuse paged-context FMHA when no kernel exists for this config.

        When ``use_paged_context_fmha`` is enabled but the fused context FMHA
        kernel is absent for this SM/head-size combination, the context phase
        falls back to unfused attention, whose context path builds K/V from the
        current chunk only: the cached prefix is DROPPED from attention and
        then OVERWRITTEN by the chunk's write-back. Any request with a
        non-zero cached length -- block reuse, partial reuse, chunked
        prefill, speculative draft tokens -- returns a plausible wrong answer
        with no error. Every native library runs this check for each batch
        with a context phase and raises instead of returning False: the
        native libraries are last in the registry, so the raise skips no
        library that could serve the batch, and it replaces the silent wrong
        answer with an error naming the cause and the remedy.

        Inside a blocklisted (SM, head_dim) cell the decision is per KV
        dtype and page size, made by asking the build what it contains via
        the ``fused_context_fmha_kernel_exists`` native lookup: a present
        kernel admits the configuration, an absent one refuses it. Nothing
        about kernel presence is hand-maintained. The check fails closed
        wherever it cannot be made: managers without a ``dtype`` or
        ``tokens_per_block`` attribute, and builds whose bindings predate
        the lookup (which also predate the op-level refusal, so passing
        unverified would reintroduce the silent corruption there).
        """
        if not metadata.use_paged_context_fmha:
            return
        manager = metadata.kv_cache_manager
        head_dim = getattr(manager, "head_dim", None) if manager else None
        if head_dim is None:
            return
        head_dims = head_dim if isinstance(head_dim, list) else [head_dim]
        sm = get_sm_version()
        absent = [dim for dim in head_dims if dim in cls.CONTEXT_FMHA_ABSENT_HEAD_DIMS.get(sm, ())]
        if not absent:
            return
        kv_dtype = getattr(manager, "dtype", None)
        tokens_per_block = getattr(manager, "tokens_per_block", None)
        if kv_dtype is not None and tokens_per_block is not None:
            kernel_exists = getattr(thop, "fused_context_fmha_kernel_exists", None)
            if kernel_exists is None:
                raise RuntimeError(
                    f"Paged-context FMHA with head_dim {absent} on SM {sm} "
                    f"requires confirming the fused context kernel against "
                    f"this build via the fused_context_fmha_kernel_exists "
                    f"lookup, but this build's bindings predate it. Bindings "
                    f"that old also predate the op-level paged-context "
                    f"refusal, so an unverified pass would risk a silent "
                    f"fall back to unfused MHA that corrupts the cached "
                    f"prefix; the configuration stays refused. Rebuild the "
                    f"bindings, or use the FlashInfer attention backend."
                )
            # Probe with every output precision the op can pair with this
            # KV precision. For an FP8 KV cache the op quantizes Q to FP8
            # but keeps dataTypeOut at the activation dtype (BF16/FP16)
            # unless FP8 attention output is enabled, and neither the
            # activation dtype nor that per-module flag is visible to this
            # metadata-only check, so one present variant admits the
            # cell; if the variant the model actually needs is the absent
            # one, the exact-parameter refusal in the AttentionOp constructor still
            # raises. An NVFP4 KV cache is read by the FP8-output kernels,
            # and a 16-bit KV cache runs matched output, mirroring the
            # binding's probe convention in
            # test_context_fmha_kernel_presence.py. Dtypes the lookup does
            # not model report absent, so they stay refused.
            if kv_dtype == DataType.FP8:
                probe_output_dtypes = (DataType.BF16, DataType.HALF, DataType.FP8)
            elif kv_dtype == DataType.NVFP4:
                probe_output_dtypes = (DataType.FP8,)
            else:
                probe_output_dtypes = (kv_dtype,)
            if all(
                any(
                    kernel_exists(
                        head_size=dim,
                        kv_cache_dtype=kv_dtype,
                        tokens_per_block=tokens_per_block,
                        output_dtype=probe_output_dtype,
                    )
                    for probe_output_dtype in probe_output_dtypes
                )
                for dim in absent
            ):
                logger.info_once(
                    f"Paged-context FMHA enabled for head_dim {absent} on SM "
                    f"{sm}: this build's fused context FMHA kernel is present "
                    f"for KV cache dtype {kv_dtype} at {tokens_per_block} "
                    f"tokens per block.",
                    key=f"paged_context_fmha_present_{sm}_{absent}_{kv_dtype}_{tokens_per_block}",
                )
                return
            cause = (
                f"this build contains no fused context FMHA kernel for that "
                f"combination with KV cache dtype {kv_dtype} at "
                f"{tokens_per_block} tokens per block"
            )
        else:
            missing = [
                name
                for name, value in (("dtype", kv_dtype), ("tokens_per_block", tokens_per_block))
                if value is None
            ]
            cause = (
                f"the KV cache manager exposes no {' or '.join(missing)}, so "
                f"the fused context FMHA kernel cannot be confirmed present "
                f"(the check fails closed)"
            )
        features = [
            name
            for name in ("chunked_prefill", "cache_reuse", "has_speculative_draft_tokens")
            if getattr(metadata.runtime_features, name, False)
        ]
        raise RuntimeError(
            f"{'/'.join(features)} requires attending to cached KV during "
            f"the context phase (use_paged_context_fmha), but for head_dim "
            f"{absent} on SM {sm} {cause}. The unfused fallback silently "
            f"drops and corrupts the cached prefix, producing plausible "
            f"wrong answers, so the configuration is refused. A build whose "
            f"--cuda_architectures does not name SM {sm} carries no kernels "
            f"for SM {sm} at all and lands here too; check the architecture "
            f"list first. Otherwise use the FlashInfer attention backend, or "
            f"disable KV block reuse, chunked prefill and speculative "
            f"decoding for this model."
        )

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not (cls.SERVES_MLA if attn.is_mla_enable else cls.SERVES_MHA):
            return False
        sparse_algorithm = getattr(attn.sparse_params, "algorithm", None)
        if sparse_algorithm in ("deepseek_v4", "dsa"):
            if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla":
                return False
            if is_sm_120f():
                return False
        return True

    def _is_call_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        del k, v
        # A context phase served by the native op with use_paged_context_fmha
        # enabled would silently corrupt the cached prefix wherever the fused
        # context kernel is absent. Raise rather than return False: the native
        # libraries come last, and the error names the cause and the remedy
        # instead of the generic no-library message. Generation-only
        # batches never run the context path, so they pass; whether a batch
        # has a context phase is part of the FMHA cache key
        # (``context_batch_size``), so the admitted result cannot be reused
        # for a context batch.
        if metadata.num_contexts > 0:
            # Extra page sizes are supported by TRTLLM-Gen, not the thop
            # paged-context kernel. Refuse before the kernel can read KV.
            extra_page_sizes = getattr(
                metadata.kv_cache_manager, "trtllm_gen_extra_tokens_per_block", ()
            )
            if (
                extra_page_sizes
                and phase != FmhaPhase.GENERATION
                and metadata.use_paged_context_fmha
                and metadata.tokens_per_block in extra_page_sizes
            ):
                raise RuntimeError(
                    f"{type(self).__name__} cannot serve paged prefill with "
                    f"{metadata.tokens_per_block}-token KV pages. "
                    "Use a supported TRTLLM-Gen configuration or disable "
                    "chunked prefill, KV reuse and speculative decoding."
                )
            self._validate_paged_context_fmha(metadata)
        # A verify group may straddle a page boundary onto two CP ranks, so
        # its KV ownership is per-token. The fused thop path cannot express
        # that: its spec-dec mask and the per-sequence helix_is_inactive_rank
        # gate both assume the new KV entries are the trailing slots of one
        # rank's kv_len. Reject rather than run it silently wrong; the native
        # libraries being last in the library list, this makes dispatch raise.
        if (
            metadata.helix_position_offsets is not None
            and metadata._helix_spec_tokens_valid
            and metadata.num_generations > 0
            # Count generation tokens only: in a mixed batch ``q`` also holds
            # the context tokens, which would otherwise trip this on a batch
            # that has exactly one query token per generation sequence.
            and q.shape[0] - metadata.num_ctx_tokens > metadata.num_generations
        ):
            return False
        if q is not None and q.dtype == torch.float8_e4m3fn:
            return False
        if forward_args.attention_mask == CustomAttentionMask.CUSTOM:
            return False
        if not (
            self.attn.is_mla_enable
            or forward_args.is_fused_qkv
            or metadata.is_cross
            or forward_args.sage_attn_num_elts_per_blk_q > 0
            or forward_args.sage_attn_num_elts_per_blk_k > 0
            or forward_args.sage_attn_num_elts_per_blk_v > 0
        ):
            return False
        if not forward_args.update_kv_cache and not metadata.is_cross:
            return False
        return True

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if not self._is_call_supported(q, k, v, metadata, forward_args, phase=phase):
            return False
        if forward_args.output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        input_type = forward_args.attention_input_type
        is_gen_only = input_type == AttentionInputType.generation_only
        has_context = metadata.num_contexts > 0 and not is_gen_only
        has_generation = (
            metadata.num_generations > 0 and input_type != AttentionInputType.context_only
        )
        check_context = has_context and phase != FmhaPhase.GENERATION
        check_generation = has_generation and phase != FmhaPhase.CONTEXT
        if not (check_context or check_generation):
            return False
        if (check_context and not self.SERVES_CONTEXT) or (
            check_generation and not self.SERVES_GENERATION
        ):
            return False
        workspace = metadata.effective_workspace
        num_tokens = q.shape[0]
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        phase_num_gen_tokens = num_gen_tokens if has_generation else 0
        if check_context:
            num_contexts = metadata.num_contexts
            params = self.context_params(
                q,
                k,
                v,
                metadata,
                forward_args,
                workspace,
                0,
                max_q_len=int(metadata.prompt_lens_cpu_runtime[:num_contexts].max()),
                max_kv_len=int(metadata.kv_lens_runtime[:num_contexts].max()),
                num_gen_tokens=phase_num_gen_tokens,
            )
            if not self.native_is_supported(params, False):
                return False
        if check_generation:
            params = self.generation_params(
                q, k, v, metadata, forward_args, workspace, 0, num_gen_tokens
            )
            if not self.native_is_supported(params, True):
                return False
        return True

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        """The native op sizes and grows the workspace itself, inside each phase call."""
        if forward_args.sage_attn_smooth_v:
            logger.warning_once(
                "The attention op quantizes V without the V smoothing this SageAttention "
                "recipe requests.",
                key="fallback_fmha_sage_attn_smooth_v",
            )

    # Keep nanobind calls eager, including when CombinedFmha delegates individual phases.
    @torch.compiler.disable
    def run_context(self, params: FmhaParams) -> None:
        self.native_run_context(params)

    @torch.compiler.disable
    def run_mla_context(self, params: FmhaParams) -> None:
        self.native_run_context(params)

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        self.native_run_generation(params)

    @torch.compiler.disable
    def run_mla_generation(self, params: FmhaParams) -> None:
        self.native_run_generation(params)

    @staticmethod
    def native_is_supported(params: FmhaParams, is_gen: bool) -> bool:
        """Whether the library serves this phase call; prepares it in the op without launching."""
        raise NotImplementedError

    @staticmethod
    def native_run_context(params: FmhaParams) -> None:
        raise NotImplementedError

    @staticmethod
    def native_run_generation(params: FmhaParams) -> None:
        raise NotImplementedError


__all__ = ["NativeFmha"]
