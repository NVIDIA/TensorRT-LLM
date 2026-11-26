# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context rows' attention in the non-absorbed MLA form (K / V expanded per head), taken
on a dense-causal step by the trunk layers whose weights built (`context_plan.layer_route`):

1. `ctx_rope.ctx_rope_q_in_place`: rope(q_pe) back into the query, the latent rows rotated
   and appended to the pool.
2. `mha_kv_gather`: each request's K/V source rows `[cached prefix | new]` -- on the MXF8
   expansion one launch writing them as MXF8 (cached rows as their pool bytes at 2^0, new
   rows block-quantized); else, with a cached prefix, the prefix widened from the fp8 pool.
3. `mha_kv_expand`: K / V = those latent rows x the augmented kv_b halves -- two MXF8 GEMMs
   (or one, `[K | V]` rows, where that entry built), else the prefill GEMM's e4m3 (fp8 form)
   or bf16 entries.
4. `mha_prefill`: causal varlen MHA, d 256, scale 1/16, bottom-right aligned over the
   prefix, into the layer's attention output -- or, fp8 form with the MXF8 O epilogue built,
   as o_proj's MXF8 operand (`o_operand`).

The form (fp8 or bf16, MXF8 expansion, merged launch, O epilogue) is fixed at load
(`build`); the per-step K / V scratch and source-row plan is `context_plan.mha_plan`'s.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count, try_build
from ..gemm import cute_dsl_projection as _projection_op
from ..gemm import cute_dsl_projection_mxf8 as _mxf8_op
from ..gemm.cute_dsl_projection import cute_dsl_projection
from ..gemm.cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8
from ..gemm.projection import ProjectionHandle, mxf8_takes
from . import cute_dsl_mha_kv_gather_mxf8 as _kv_gather8_op
from . import cute_dsl_mha_prefill as _mha_pre_op
from .cute_dsl_mha_kv_gather_mxf8 import cute_dsl_mha_kv_gather_mxf8
from .cute_dsl_mha_prefill import cute_dsl_mha_prefill
from .cute_dsl_mha_prefix_gather import cute_dsl_mha_prefix_gather

#: The MXF8 expansion's scale blocks are 128 rows: a row slice starts at a block.
_SF_BLOCK_ROWS = 128


@dataclass(frozen=True)
class CtxMhaHandle:
    """What the non-absorbed context attention launches. `expand` the K / V expansion runners
    (empty: the route did not build, and every context step keeps the absorbed path); `fp8`
    its form; `mha` the MHA (on strided K / V under the merged expansion); `mha_q8` / `q8_rows`
    the MHA writing O as o_proj's MXF8 operand and the generation rows' quantize into it (None:
    o_proj quantizes itself); `prefix` the cached-prefix gather (bf16 / e4m3 expansion);
    `gather8` the MXF8 expansion's gather (None: the projection GEMMs), `merged` its one-launch
    `[K | V]` runner, `idle` the gather's operands on a step without a cached prefix;
    `stride_factor` the trunk pool's row stride per layer; the model's MLA dims."""

    expand: Mapping[str, Any]
    fp8: bool
    mha: Any
    mha_q8: Any
    q8_rows: Any
    prefix: Any
    gather8: Any
    merged: Any
    idle: tuple
    stride_factor: int
    heads: int
    qk_dim: int
    v_dim: int
    kv_lora: int
    lat_dim: int
    count: Count


def build_kv_mxf8(dev, heads: int, qk: int, v: int, lat: int, lora: int):
    """The MXF8 K / V expansion's gather and its two GEMM runners (built against zero weights:
    the runners take each layer's weight per call), and the one-launch `[K | V]` runner (None
    keeps the two launches); None if the gather or a runner fails."""
    gather = try_build("MXF8 K/V gather", lambda: _kv_gather8_op.compile(dev))
    if gather is None:
        return None
    bf = dict(dtype=torch.bfloat16, device=torch.device("cuda", dev))
    runners = {
        "k": try_build(
            "MXF8 K expansion",
            lambda: _mxf8_op.build(
                "kv_b_k8", _mxf8_op.quantize_weight(torch.zeros(heads * qk, lat, **bf))
            ),
        ),
        "v": try_build(
            "MXF8 V expansion",
            lambda: _mxf8_op.build(
                "kv_b_v8", _mxf8_op.quantize_weight(torch.zeros(heads * v, lora, **bf))
            ),
        ),
    }
    if not all(runners.values()):
        return None
    merged = try_build(
        "MXF8 K | V expansion in one launch",
        lambda: _mxf8_op.build(
            "kv_b_kv8", _mxf8_op.quantize_weight(torch.zeros(heads * (qk + v), lat, **bf))
        ),
    )
    d = torch.device("cuda", dev)
    # The gather's pool / index / table operands on a step without a cached prefix: read by
    # nothing (no cached rows, new rows in order), but they must be valid tensors.
    idle = (
        torch.zeros(1, lat, dtype=torch.uint8, device=d),
        torch.zeros(1, dtype=torch.int64, device=d),
        torch.zeros(1, dtype=torch.int32, device=d),
        torch.zeros(1, 1, dtype=torch.int32, device=d),
    )
    return SimpleNamespace(gather=gather, runners=runners, merged=merged, idle=idle)


def o_operand(ph, attn_out: torch.Tensor, w8, o_proj: ProjectionHandle, handle: CtxMhaHandle):
    """o_proj's MXF8 operand `(q8, sf)` for the fp8 MHA's O epilogue to write, where that
    epilogue built and o_proj takes the MXF8 GEMM at these rows (`w8` the layer's MXF8
    o_proj weight), else None (o_proj quantizes `attn_out` itself)."""
    if handle.mha_q8 is None or not mxf8_takes(ph, attn_out, w8, None, o_proj):
        return None
    rows, dev = int(attn_out.shape[0]), attn_out.device
    return (
        torch.empty([rows, handle.heads * handle.v_dim], dtype=torch.uint8, device=dev),
        torch.empty(_mha_pre_op.sf_bytes(rows), dtype=torch.uint8, device=dev),
    )


class MhaKvGather(OpWrapper):
    """This layer's K / V source rows for `mha_kv_expand`, from the rotated latent rows `lat`
    and the step's `plan` at pool layer `slot`: on the MXF8 expansion the gather writes the
    `[cached | new]` rows as MXF8 into `plan.a8` (returns None); else with a cached prefix
    the prefix gather writes them in bf16 into `plan.cached.lat` (returns it); else `lat`."""

    def __call__(self, lat: torch.Tensor, plan, slot, handle: CtxMhaHandle):
        h = handle
        layer_offset = slot.layer_in_pool * slot.tpb
        pre = plan.cached
        if h.gather8 is not None:
            if pre is not None:
                cute_dsl_mha_kv_gather_mxf8(
                    pre.pool_rows, pre.idx, pre.rows, pre.req_idx, pre.block_table, lat,
                    plan.a8, layer_offset, h.stride_factor, slot.tpb, True, compiled=h.gather8,
                )  # fmt: skip
            else:
                pool, idx, req_idx, block_table = h.idle
                cute_dsl_mha_kv_gather_mxf8(
                    pool, idx, 0, req_idx, block_table, lat, plan.a8, 0, 1, 1, False,
                    compiled=h.gather8,
                )  # fmt: skip
            return None
        if pre is not None:
            return cute_dsl_mha_prefix_gather(
                pre.pool_rows, pre.idx, pre.rows, pre.req_idx, pre.block_table, lat, pre.lat,
                layer_offset, h.stride_factor, slot.tpb, compiled=h.prefix,
            )  # fmt: skip
        return lat

    def reference(self, lat: torch.Tensor, plan, slot, handle: CtxMhaHandle):
        """Not modelled here: the gathers' entries (`cute_dsl_mha_kv_gather_mxf8`,
        `cute_dsl_mha_prefix_gather`) carry the references."""
        raise NotImplementedError("see cute_dsl_mha_kv_gather_mxf8 / cute_dsl_mha_prefix_gather")


class MhaKvExpand(OpWrapper):
    """K / V of this layer into `plan.k` / `plan.v` from `src` (`mha_kv_gather`'s output) and
    the layer's expansion weights of the handle's form: `kvb8` (fp8: the folded K / V weights,
    MXF8 on the MXF8 expansion) or `kvb_aug` (bf16: the augmented kv_b).
    On the MXF8 expansion (counted `ctx_kvb_mxf8`) one `[K | V]` launch into `plan.kv` where
    the step's plan holds it (counted `ctx_kvb_merged`; `kvb8[0]` the stacked weight, its V tiles
    reading the first 512 columns of the rows, the bytes `a_v` holds), else two GEMMs, in row
    slices of the runners' capacity (a multiple of 128 rows, so a slice's scales start at a
    block); else the prefill GEMM per half (the V half on the latent's c_kv part in the fp8
    form) in slices of its row limit (with a cached prefix the K/V rows can exceed it, and
    the product is row-independent)."""

    def __call__(self, src, kvb8, kvb_aug, plan, handle: CtxMhaHandle) -> None:
        h = handle
        w = kvb8 if h.fp8 else kvb_aug
        if h.gather8 is not None:
            h.count("ctx_kvb_mxf8", True)
            a_k, a_v, sfa_k, sfa_v = plan.a8
            w_k, w_v = w
            blk_k = _kv_gather8_op.sf_bytes_k(_SF_BLOCK_ROWS)
            blk_v = _kv_gather8_op.sf_bytes_v(_SF_BLOCK_ROWS)
            if plan.kv is not None:
                h.count("ctx_kvb_merged", True)
                step = h.merged.launch.max_rows
                for r0 in range(0, plan.tk, step):
                    r1 = min(plan.tk, r0 + step)
                    b = r0 // _SF_BLOCK_ROWS
                    cute_dsl_projection_mxf8(
                        a_k[r0:r1], w_k, plan.kv[r0:r1], h.merged, sfa=sfa_k[b * blk_k :]
                    )
                return
            rk, rv = h.expand["k"], h.expand["v"]
            step = rk.launch.max_rows
            for r0 in range(0, plan.tk, step):
                r1 = min(plan.tk, r0 + step)
                b = r0 // _SF_BLOCK_ROWS
                cute_dsl_projection_mxf8(a_k[r0:r1], w_k, plan.k[r0:r1], rk, sfa=sfa_k[b * blk_k :])
                cute_dsl_projection_mxf8(a_v[r0:r1], w_v, plan.v[r0:r1], rv, sfa=sfa_v[b * blk_v :])
            return
        if h.fp8:
            halves = (("k", w[0], plan.k, src), ("v", w[1], plan.v, src[:, : h.kv_lora]))
        else:
            k_rows = h.heads * h.qk_dim
            halves = (("k", w[:k_rows], plan.k, src), ("v", w[k_rows:], plan.v, src))
        step = _projection_op.PREFILL_MAX_ROWS
        for r0 in range(0, plan.tk, step):
            r1 = min(plan.tk, r0 + step)
            for half, wh, dst, x in halves:
                cute_dsl_projection(x[r0:r1], wh, dst[r0:r1], h.expand[half])

    def reference(self, src, kvb8, kvb_aug, plan, handle: CtxMhaHandle) -> None:
        """Not modelled here: the GEMMs' entries (`cute_dsl_projection_mxf8`,
        `cute_dsl_projection`) carry the references; the weights are in the serving GEMM's own
        form."""
        raise NotImplementedError("see cute_dsl_projection_mxf8 / cute_dsl_projection")


class MhaPrefill(OpWrapper):
    """The causal varlen MHA of the rotated query `q3` (`[tc, heads, qk_dim]`) over the step's
    K / V (`plan`) into `out`, `cu_q` the new-row prefix sum, the cached prefix's K offsets
    from `plan.cached` (counted `ctx_mha_cached`). The fp8 form (an e4m3 `q3`, counted
    `ctx_mha_fp8`) reads the layer's per-head dequantization table `head_scales`; with `o8`
    (`o_operand`) it writes the context rows as o_proj's MXF8 operand into `o8` instead of
    `out` (counted `ctx_mha_o8`). Returns whether it wrote `o8`."""

    def __call__(
        self,
        q3: torch.Tensor,
        plan,
        out: torch.Tensor,
        cu_q: torch.Tensor,
        head_scales,
        o8,
        handle: CtxMhaHandle,
    ) -> bool:
        h = handle
        fp8 = q3.dtype == torch.float8_e4m3fn
        tc = int(q3.shape[0])
        cu_k = None if plan.cached is None else plan.cached.cu_k
        use_o8 = o8 is not None and fp8 and h.mha_q8 is not None
        cute_dsl_mha_prefill(
            q3,
            plan.k.view(plan.tk, h.heads, h.qk_dim),
            plan.v.view(plan.tk, h.heads, h.v_dim),
            torch.reshape(out, [tc, h.heads, h.v_dim]),
            cu_q,
            plan.max_len,
            compiled=h.mha_q8 if use_o8 else h.mha,
            cu_seqlens_k=cu_k,
            head_scales=head_scales if fp8 else None,
            q8_out=o8 if use_o8 else None,
        )
        assert use_o8 == (o8 is not None), "an o_proj operand the MHA did not write"
        h.count("ctx_mha", True)
        if use_o8:
            h.count("ctx_mha_o8", True)
        if fp8:
            h.count("ctx_mha_fp8", True)
        if cu_k is not None:
            h.count("ctx_mha_cached", True)
        return use_o8

    def reference(
        self,
        q3: torch.Tensor,
        plan,
        out: torch.Tensor,
        cu_q: torch.Tensor,
        head_scales,
        o8,
        handle: CtxMhaHandle,
    ) -> bool:
        """Not modelled here: the MHA's entry (`cute_dsl_mha_prefill.reference`) carries it."""
        raise NotImplementedError("see cute_dsl_mha_prefill")


mha_kv_gather = MhaKvGather()
mha_kv_expand = MhaKvExpand()
mha_prefill = MhaPrefill()
