# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context rows' MLA RoPE and paged latent append as the prefill Target calls it:
`ctx_rope_append` (the absorbed form: rope of the query's tail, the latent rows appended as
e4m3 to the pool, the e4m3 query) on the sm103 context kernel where the step and the layer's
pool take it, else the stock op; `ctx_rope_q_in_place` (the non-absorbed MHA's form: rope
back into the q_b output), on the kernel only.

The step's operands come from `context_plan.rope_step` (None: the stock op), the layer's
pool addressing from `context_plan.pool_slot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from .cute_dsl_mla_rope_append_prefill import (
    CompiledMlaRopeAppendPrefill,
    cute_dsl_mla_rope_append_prefill,
)
from .mla_rope_append_paged_kv_assign_q import mla_rope_append_paged_kv_assign_q


@dataclass(frozen=True)
class CtxRopeHandle:
    """`compiled` the context rope/append kernel (None: every step on the stock op; keyed by
    the page size, built at the first forward), `cos_sin` the rope table, the model's MLA dims
    and the KV cache's quant mode, and the row copy the stock op's operands are staged with."""

    compiled: CompiledMlaRopeAppendPrefill | None
    cos_sin: torch.Tensor | None
    heads: int
    kv_lora: int
    rope_dim: int
    quant_mode: int
    copy: RowCopyHandle
    count: Count


class CtxQuery(NamedTuple):
    """The context K-B's output as the rope reads it (`ctx_k_b`): `fused_q` `[tc, heads,
    lat_dim]` (its latent half written unless `from_kb`), `q_pe` the rope half of q, `quant_q`
    the e4m3 query (its q-base already written where `from_kb`), and whether the layer's rope
    takes the kernel (`on_kernel`)."""

    fused_q: torch.Tensor
    q_pe: torch.Tensor
    quant_q: torch.Tensor
    on_kernel: bool
    from_kb: bool


class CtxRopeAppend(OpWrapper):
    """RoPE of `query.q_pe` into the tail of `query.fused_q`, rope(k_pe) and the latent rows
    appended as e4m3 to pool layer `slot.layer_in_pool`, and the e4m3 query into
    `query.quant_q`; returns `quant_q`. On the kernel where `query.on_kernel` (counted
    `ctx_rope`; the latent read in place at the projection row's stride, else a packed copy,
    counted `ctx_rope_latent_in_place`), skipping the q-base groups where `query.from_kb`;
    else the stock op over the step's metadata (`step`, `layer` the engine's layer index),
    the query's rope tail seeded first and the e4m3 query copied after."""

    def __call__(
        self,
        query: CtxQuery,
        latent: torch.Tensor,
        step: dict,
        slot,
        layer: int,
        ctx_rope,
        handle: CtxRopeHandle,
    ) -> torch.Tensor:
        h = handle
        fused_q, q_pe, quant_q = query.fused_q, query.q_pe, query.quant_q
        h.count("ctx_rope", query.on_kernel)
        if query.on_kernel:
            in_place = latent.stride(0) in h.compiled.kernels and latent.stride(1) == 1
            h.count("ctx_rope_latent_in_place", in_place)
            # One launch: rope(q_pe) into the tail of fused_q straight from the strided view,
            # rope(k_pe) and the latent row appended as e4m3 to this layer's pool rows, and
            # e4m3(fused_q). Context requests are the first `num_contexts` of the batch. In
            # place, the latent is the projection row's tail (stride q_lora + 576), which the
            # kernel was compiled for, and nothing reads it after the append rotates its k_pe.
            cute_dsl_mla_rope_append_prefill(
                fused_q,
                q_pe,
                latent if in_place else latent.clone(memory_format=torch.contiguous_format),
                quant_q,
                step["primary_pool"],
                slot.layer_in_pool,
                slot.block_offsets,
                *ctx_rope,
                h.cos_sin,
                step["tokens_per_block"],
                h.compiled,
                skip_q_base=query.from_kb,
            )
            return quant_q
        tc, heads, lat_dim = (int(s) for s in fused_q.shape)
        # The append rotates the query's rope tail in place, reading its own pre-rotation
        # contents, and the absorbed BMM wrote only the first kv_lora columns: seed the tail.
        row_copy(fused_q[..., h.kv_lora :], q_pe, h.copy)
        mla_rope_append_paged_kv_assign_q(
            torch.reshape(fused_q, [tc, heads * lat_dim]),
            latent.contiguous(),
            step["num_contexts"],
            step["ctx_cached_token_indptr"],
            step["ctx_kv_indptr"],
            step["max_ctx_seq_len"],
            h.cos_sin,
            heads,
            # The width ahead of each head's rope tail in the query this call hands over:
            # the absorbed head is `kv_lora + rope`, so this is kv_lora, not
            # qk_nope_head_dim.
            h.kv_lora,
            h.rope_dim,
            h.kv_lora,
            step["kv_cache_block_offsets"],
            step["host_kv_cache_pool_pointers"],
            step["host_kv_cache_pool_mapping"],
            None,
            0,
            layer,
            step["tokens_per_block"],
            step["max_seq_len"],
            1,
            h.quant_mode,
        )
        row_copy(quant_q, fused_q, h.copy)
        return quant_q

    def reference(
        self,
        query: CtxQuery,
        latent: torch.Tensor,
        step: dict,
        slot,
        layer: int,
        ctx_rope,
        handle: CtxRopeHandle,
    ) -> torch.Tensor:
        """Not modelled here: the kernel's entry (`cute_dsl_mla_rope_append_prefill.reference`)
        and the stock op's carry the references; both write the same bytes."""
        raise NotImplementedError("see cute_dsl_mla_rope_append_prefill")


class CtxRopeQInPlace(OpWrapper):
    """The non-absorbed MHA's rope on the kernel's `q_in_place` mode: rope(q_pe) back into the
    query `q3` (`[tc, heads, qk_dim]`; e4m3 where it is), rope(k_pe) into the latent rows and
    the latent rows appended as e4m3 to pool layer `slot.layer_in_pool` -- the same pool
    bytes as `ctx_rope_append`. Returns the latent rows it rotated (in place at the
    projection row's stride, else a packed copy). Counted `rope_par` on the e4m3 query where
    the kernel's parallel rope variant built. The plan (`context_plan.layer_route`) sends a
    layer here only with the kernel on the step."""

    def __call__(
        self,
        q3: torch.Tensor,
        latent: torch.Tensor,
        step: dict,
        slot,
        nope: int,
        ctx_rope,
        handle: CtxRopeHandle,
    ) -> torch.Tensor:
        h = handle
        compiled = h.compiled
        fp8 = q3.dtype == torch.float8_e4m3fn
        in_place = latent.stride(0) in compiled.kernels_q_in_place and latent.stride(1) == 1
        lat = latent if in_place else latent.clone(memory_format=torch.contiguous_format)
        cute_dsl_mla_rope_append_prefill(
            None,
            q3[..., nope:],
            lat,
            None,
            step["primary_pool"],
            slot.layer_in_pool,
            slot.block_offsets,
            *ctx_rope,
            h.cos_sin,
            slot.tpb,
            compiled,
            q_in_place=True,
            q_e4m3=fp8,
        )
        if fp8 and compiled.rope_par:
            h.count("rope_par", True)
        return lat

    def reference(
        self,
        q3: torch.Tensor,
        latent: torch.Tensor,
        step: dict,
        slot,
        nope: int,
        ctx_rope,
        handle: CtxRopeHandle,
    ) -> torch.Tensor:
        """Not modelled here: the kernel's entry (`cute_dsl_mla_rope_append_prefill.reference`)
        carries the reference."""
        raise NotImplementedError("see cute_dsl_mla_rope_append_prefill")


ctx_rope_append = CtxRopeAppend()
ctx_rope_q_in_place = CtxRopeQInPlace()
