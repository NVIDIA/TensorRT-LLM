# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-phase FMHA for MiniMax-M3, on kernels built for a decode shape.

MsaPrefillFmha's fmha_sm100 kernel schedules a generation row like a context
row, a single query token occupying a 128-row Q tile. This library takes the
generation phase instead and dispatches by layer: a sparse layer to the Triton
block-sparse decode kernel over the blocks the indexer selected, a dense layer
to the native trtllm-gen library over the full page table.

Both need a uniform query length across the generation rows and a geometry they
support, and neither has a fallback: prepare() settles the query length per
step as metadata.msa_decode_span, and ensure_msa_available and
_validate_decode_kernel_support settle the geometry once per run.

Every import of the kernels below is function-local. This module is on the
import path of every attention.backends.trtllm import, and the minimax_m3
package init reaches msa_backend, which subclasses TrtllmAttention, so a
module-scope import here would close a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class MsaDecodeFmha(PhasedFmha):
    """MiniMax-M3 generation attention on the Triton and trtllm-gen kernels.

    Generation only: the context phase is MsaPrefillFmha's, and a mixed batch
    is served by the two together through CombinedFmha. run_context is left to
    the base class, which refuses it.
    """

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        from ..sparse.minimax_m3.kernels.msa_utils import is_msa_layer

        return is_msa_layer(attn)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        # Every generation row is this library's and every context row
        # MsaPrefillFmha's, whatever the step looks like, so each library
        # claims its one phase and their partition of the phases is total. The
        # phase-less query asks for the whole step, which neither can serve
        # alone: a mixed step belongs to the two together through CombinedFmha.
        return phase is FmhaPhase.GENERATION

    def run_generation(self, params: FmhaParams) -> None:
        from ..sparse.minimax_m3.kernels.msa_utils import write_msa_phase_kv

        metadata = params.meta
        span = metadata.msa_decode_span
        # Both kernels below map a query token to its request through the span,
        # while the phase params carry PhasedFmha's own derivation of the same
        # boundary. Checking them against each other rejects a step whose
        # generation rows were never described and one where the two disagree.
        phase = (params.seq_offset, params.input_seq_length)
        if span != phase:
            raise RuntimeError(
                "MsaDecodeFmha ran on a generation phase its decode span does "
                f"not describe: the span is {span}, while the phase starts at "
                f"row {params.seq_offset} with {params.input_seq_length} query "
                "tokens per request."
            )
        write_msa_phase_kv(
            params.attn,
            params.key_input,
            params.value_input,
            metadata,
            params.fwd.attention_input_type,
            token_offset=params.token_offset,
        )
        row_first = params.seq_offset
        row_last = row_first + metadata.num_generations
        block_table = metadata.msa_block_table[row_first:row_last]
        seq_lens = metadata.msa_seq_lens_cuda[row_first:row_last]

        kv_block_indexes = params.fwd.sparse_runtime_params.sparse_attn_indices
        if kv_block_indexes is not None:
            self._run_sparse(params, kv_block_indexes, block_table, seq_lens)
        else:
            self._run_dense(params, block_table, seq_lens)

    def _run_sparse(
        self,
        params: FmhaParams,
        kv_block_indexes: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        # This kernel also pulls in Triton, a second reason to keep it out of
        # module scope.
        from ..sparse.minimax_m3.kernels.msa_utils import msa_paged_kv
        from ..sparse.minimax_m3.kernels.triton_sparse_decode import minimax_m3_sparse_attn_decode

        attn = params.attn
        head_dim = attn.head_dim
        num_tokens = params.num_tokens
        k_paged, v_paged = msa_paged_kv(params.meta.kv_cache_manager, attn.layer_idx)
        nvfp4_args = {}
        manager = params.meta.kv_cache_manager
        if getattr(manager, "is_nvfp4_layer", lambda _: False)(attn.layer_idx):
            from .msa_prefill import _aligned_nvfp4_dequant_scales

            if params.fwd.kv_scale_quant_orig is None:
                raise RuntimeError("NVFP4 sparse decode requires dequantization scales")
            k_scale, v_scale = _aligned_nvfp4_dequant_scales(attn, params.fwd.kv_scale_quant_orig)
            scales = manager.get_block_scale_buffers(attn.layer_idx, "HND")
            nvfp4_args = dict(
                k_block_scale=scales[:, 0],
                v_block_scale=scales[:, 1],
                k_global_scale=k_scale,
                v_global_scale=v_scale,
            )
        # q may still be FP8 from a fused producer; the kernel widens it
        # in-register, so it is passed through as it arrives.
        minimax_m3_sparse_attn_decode(
            params.query_input.view(num_tokens, attn.num_heads, head_dim),
            k_paged,
            v_paged,
            # The kernel reads the top-k table head-major and the indexer
            # builds it that way for every step, so this is a view. It reads
            # every stride, so a mixed step's strided suffix works too.
            kv_block_indexes[params.token_offset : params.token_offset + num_tokens].permute(
                1, 0, 2
            ),
            block_table,
            seq_lens,
            sm_scale=(head_dim**-0.5) / float(attn.q_scaling),
            output=params.output.view(num_tokens, attn.num_heads, head_dim),
            decode_query_len=params.input_seq_length,
            **nvfp4_args,
        )

    def _run_dense(
        self,
        params: FmhaParams,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Attend the full page table on the native trtllm-gen library, for a dense M3 layer.

        A prepared-query call on the step's metadata over M3's own sub-page
        table, because the layer's pool mapping cannot address M3's pool. See
        the trtllm_gen_dense_decode module docstring for why.
        """
        from ..sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
            minimax_m3_trtllm_gen_dense_decode,
        )

        attn = params.attn
        metadata = params.meta
        head_dim = attn.head_dim
        num_tokens = params.num_tokens
        row_first = params.seq_offset
        # The sub-page block table prepare() staged, if it could; the call
        # expands its own when the factor does not match this layer's.
        staged_table, staged_factor = metadata.msa_subpage_rows(
            row_first, row_first + metadata.num_generations
        )
        minimax_m3_trtllm_gen_dense_decode(
            params.query_input.view(num_tokens, attn.num_heads, head_dim),
            metadata,
            attn.layer_idx,
            block_table,
            seq_lens,
            sm_scale=(head_dim**-0.5) / float(attn.q_scaling),
            output=params.output.view(num_tokens, attn.num_heads, head_dim),
            decode_query_len=params.input_seq_length,
            max_seq_len=int(metadata.msa_max_kv_len),
            # The attention workspace every FMHA library shares, which puts the
            # native op's scratch in a captured graph's memory pool.
            workspace=params.workspace,
            staged_subpage_table=staged_table,
            staged_subpages_per_slot=staged_factor,
        )


__all__ = ["MsaDecodeFmha"]
