# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context rows' absorbed sparse MLA on the sm103 prefill kernel, by the step's plan
(`context_plan`): `sparse_mla_prefill_index` over the indexer's selection as global pool rows,
`sparse_mla_prefill_dense_causal` over each request's causal window read through its
block-table row (a step whose context rows skipped the indexer), and
`sparse_mla_prefill_selected` (the MTP layer's selected rows, one group per query). Both
kernel modes are built at load (a first compile inside a CUDA-graph capture is a broken graph).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from . import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .cute_dsl_sparse_mla_prefill import CompiledSparseMlaPrefill, cute_dsl_sparse_mla_prefill


@dataclass(frozen=True)
class CtxSparseMlaHandle:
    """`index` / `dense_causal` the prefill kernel in its two modes, `bmm1` / `bmm2` its scale
    arrays, `stride_factor` the trunk pool's row stride per layer (set at the first forward)."""

    index: CompiledSparseMlaPrefill
    dense_causal: CompiledSparseMlaPrefill
    bmm1: torch.Tensor
    bmm2: torch.Tensor
    stride_factor: int


class SparseMlaPrefillIndex(OpWrapper):
    """The sparse MLA of the context rows' e4m3 query `quant_q` over `slots` (the selection as
    global pool rows, spanning the cached prefix: no gather) into `lat_out`, with the step's
    per-row KV windows."""

    def __call__(
        self,
        quant_q: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        lat_out: torch.Tensor,
        step: dict,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        h = handle
        cute_dsl_sparse_mla_prefill(
            quant_q, pool, slots, lat_out, step["cu_seqlen_ks"], step["cu_seqlen_ke"], h.bmm1,
            h.bmm2, compiled=h.index,
        )  # fmt: skip
        return lat_out

    def reference(
        self,
        quant_q: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        lat_out: torch.Tensor,
        step: dict,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        """`cute_dsl_sparse_mla_prefill.reference` over the same operands."""
        h = handle
        return cute_dsl_sparse_mla_prefill.reference(
            quant_q, pool, slots, lat_out, step["cu_seqlen_ks"], step["cu_seqlen_ke"], h.bmm1,
            h.bmm2, compiled=h.index,
        )  # fmt: skip


class SparseMlaPrefillDenseCausal(OpWrapper):
    """The sparse MLA of the context rows' e4m3 query over each request's causal window, read
    through its block-table row at pool layer offset `layer_offset` (the rows `to_global` of
    the engine's dense causal index would name), into `lat_out`. `plan` is the step's
    (`context_plan.dense_causal_step`): the kernel's launch is prepared at the step's first
    layer into `plan.prepared` and re-bound per layer (output and layer offset); the plan's
    fp8 query buffer is the one every layer writes."""

    def __call__(
        self,
        quant_q: torch.Tensor,
        pool: torch.Tensor,
        lat_out: torch.Tensor,
        plan,
        layer_offset: int,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        h = handle
        if plan.prepared is None:
            plan.prepared = _fmha_pre_op.prepare_dense_causal(
                h.dense_causal, quant_q, pool, lat_out, plan.ks, plan.ke, h.bmm1, h.bmm2,
                plan.block_table, plan.req_idx, h.stride_factor, layer_offset, plan.groups,
            )  # fmt: skip
        cute_dsl_sparse_mla_prefill(
            quant_q, pool, None, lat_out, plan.ks, plan.ke, h.bmm1, h.bmm2,
            compiled=plan.prepared, block_table=plan.block_table, req_idx=plan.req_idx,
            stride_factor=h.stride_factor, layer_offset=layer_offset,
            dense_causal_groups=plan.groups,
        )  # fmt: skip
        return lat_out

    def reference(
        self,
        quant_q: torch.Tensor,
        pool: torch.Tensor,
        lat_out: torch.Tensor,
        plan,
        layer_offset: int,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        """`cute_dsl_sparse_mla_prefill.reference` in its dense-causal mode."""
        h = handle
        return cute_dsl_sparse_mla_prefill.reference(
            quant_q, pool, None, lat_out, plan.ks, plan.ke, h.bmm1, h.bmm2,
            compiled=h.dense_causal, block_table=plan.block_table, req_idx=plan.req_idx,
            stride_factor=h.stride_factor, layer_offset=layer_offset,
            dense_causal_groups=plan.groups,
        )  # fmt: skip


class SparseMlaPrefillSelected(OpWrapper):
    """The MTP layer's selected context queries `q_sel` (`last.n_pad` rows of the step's e4m3
    query) on the dense-causal kernel, each its own group over its request's block-table row
    (positions 0 .. cached + len - 1: the engine's dense index, so bitwise the index path's
    rows), into `lat_out`. `last` is the step's plan (`context_plan.mtp_last_plan`), with the
    draft pool's own stride and layer offset."""

    def __call__(
        self,
        q_sel: torch.Tensor,
        pool: torch.Tensor,
        lat_out: torch.Tensor,
        last,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        h = handle
        prepared = _fmha_pre_op.prepare_dense_causal(
            h.dense_causal, q_sel, pool, lat_out, last.ks, last.ke, h.bmm1, h.bmm2,
            last.block_table, last.req, last.stride, last.layer_offset, last.groups,
        )  # fmt: skip
        cute_dsl_sparse_mla_prefill(
            q_sel, pool, None, lat_out, last.ks, last.ke, h.bmm1, h.bmm2,
            compiled=prepared, block_table=last.block_table, req_idx=last.req,
            stride_factor=last.stride, layer_offset=last.layer_offset,
            dense_causal_groups=last.groups,
        )  # fmt: skip
        return lat_out

    def reference(
        self,
        q_sel: torch.Tensor,
        pool: torch.Tensor,
        lat_out: torch.Tensor,
        last,
        handle: CtxSparseMlaHandle,
    ) -> torch.Tensor:
        """`cute_dsl_sparse_mla_prefill.reference` in its dense-causal mode."""
        h = handle
        return cute_dsl_sparse_mla_prefill.reference(
            q_sel, pool, None, lat_out, last.ks, last.ke, h.bmm1, h.bmm2,
            compiled=h.dense_causal, block_table=last.block_table, req_idx=last.req,
            stride_factor=last.stride, layer_offset=last.layer_offset,
            dense_causal_groups=last.groups,
        )  # fmt: skip


sparse_mla_prefill_index = SparseMlaPrefillIndex()
sparse_mla_prefill_dense_causal = SparseMlaPrefillDenseCausal()
sparse_mla_prefill_selected = SparseMlaPrefillSelected()
