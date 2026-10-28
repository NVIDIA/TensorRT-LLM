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
"""Decode front end of the Qwen3.5 routed experts on the TRTLLM-gen NVFP4 MoE.

For decode-sized steps, computes the router logits, top-k ids / weights, trtllm-gen's routing tables (see
:func:`q35_prerouted_sizes`) and the NVFP4 quantization of x, so ``run_moe_prerouted`` skips its routing kernel.
The routing must stay bit-identical to trtllm-gen Renormalize (``reduceTopK`` tie order, ``calcSoftmax`` sum
order); permuted-row order within an expert may differ, which the finalize does not depend on.
Prefill and steps outside the ``Q35_MOE_*`` limits take the stock path.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ...autotuner import AutoTuner
from ...q35_env import register as _q35_register

# Largest step of the router GEMM, the low-M router and the route + tables launch.
Q35_MOE_ROUTER_MAX_TOKENS = 32
# Largest step of the one-launch front (faster in-model up to MTP-6 verify at b2, TEP8/TEP16, SM 10.3).
Q35_MOE_FRONT_MAX_M = 16
# Steps of the big front: MTP-6 target verify rows 7 * b for b = 5..32. TRTLLM_Q35_MOE_FRONT_BIG_MS=<m>,... overrides
# the set (e.g. with the verify and draft steps of larger batches, together with
# TRTLLM_Q35_MOE_FRONT_BIG_C16 / _C16W above 256 tokens).
Q35_MOE_FRONT_BIG_MS = frozenset(
    int(v) for v in os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_MS", "").split(",") if v.strip()
) or frozenset(range(35, 225, 7))
# Warps per program of the per-token route + tables kernel (one warp leaves the last program's table phase
# serial).
_ROUTE_WARPS = 4

SIGN_BIT = tl.constexpr(-2147483648)
MIN_KEY = tl.constexpr(-2147483648)


@triton.jit
def _gdc_wait_with_memory_clobber():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, r, H: tl.constexpr):
    """NVFP4-quantize hidden row r (cvt_warp_fp16_to_fp4 recipe, linear SF)."""
    cols = tl.arange(0, H)
    v = tl.load(x_ptr + r * x_row_stride + cols).to(tl.float32)
    v16 = tl.reshape(v, (H // 16, 16))
    vmax = tl.max(tl.abs(v16), axis=1)
    gs = tl.load(gs_ptr)
    rcp6 = _rcp_approx(tl.full((H // 16,), 6.0, tl.float32))
    sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
    rcpgs = _rcp_approx(tl.full((H // 16,), 0.0, tl.float32) + gs)
    osc = _rcp_approx(sf8.to(tl.float32) * rcpgs)
    osc = tl.where(vmax != 0.0, osc, 0.0)
    tl.store(xsf_ptr + r * (H // 16) + tl.arange(0, H // 16), sf8.to(tl.uint8, bitcast=True))
    y = tl.reshape(v16 * osc[:, None], (H,))
    lo, hi = tl.split(tl.reshape(y, (H // 2, 2)))
    byte = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f",
        [lo, hi],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    tl.store(xq_ptr + r * (H // 2) + tl.arange(0, H // 2), byte)


@triton.jit
def _fence_acq_rel_gpu():
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; // dummy $0", "=r,~{memory}", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _q35_router_gemm_kernel(
    x_ptr,  # [M, K] bf16 activations
    x_row_stride,
    w_ptr,  # [N, K] bf16 router weight (row-major, as nn.Linear)
    out_ptr,  # [M, N] bf16 logits
    part_ptr,  # [SPLITK, BM, N] fp32 split-K partials (scratch)
    cnt_ptr,  # [N // BN] int32 arrival counters (zero between launches)
    gs_ptr,  # [1] fp32 NVFP4 global scale (QUANT only)
    xq_ptr,  # out [M, K // 2] uint8 (QUANT only)
    xsf_ptr,  # out [M, K // 16] uint8 linear scales (QUANT only)
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITK: tl.constexpr,
    QUANT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """logits = x @ w^T, one launch: (N/BN) x SPLITK CTAs, each a [BM, BN]
    partial over K/SPLITK; the last CTA of each N block sums the partials in
    split order (deterministic) and writes bf16. The weight tile does not
    depend on the producer, so it is loaded before griddepcontrol.wait.
    With QUANT, M extra CTAs NVFP4-quantize the activation rows."""
    KS: tl.constexpr = K // SPLITK
    NB: tl.constexpr = N // BN
    pid = tl.program_id(0)
    if pid >= NB * SPLITK:
        if QUANT:
            if USE_PDL:
                _gdc_wait_with_memory_clobber()
            _nvfp4_quant_row(
                x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, (pid - NB * SPLITK).to(tl.int64), K
            )
    else:
        pid_n = pid % NB
        pid_k = pid // NB
        rn = pid_n * BN + tl.arange(0, BN)
        rk = pid_k * KS + tl.arange(0, KS)
        rm = tl.arange(0, BM)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])  # [BN, KS]
        if USE_PDL:
            _gdc_wait_with_memory_clobber()
        xt = tl.load(
            x_ptr + rm[:, None] * x_row_stride + rk[None, :], mask=(rm < M)[:, None], other=0.0
        )  # [BM, KS]
        acc = tl.dot(xt, tl.trans(w), out_dtype=tl.float32)  # [BM, BN]
        tl.store(part_ptr + (pid_k * BM + rm[:, None]) * N + rn[None, :], acc)
        _fence_acq_rel_gpu()
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLITK - 1:
            _fence_acq_rel_gpu()
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLITK):
                tot += tl.load(part_ptr + (s * BM + rm[:, None]) * N + rn[None, :], volatile=True)
            tl.store(
                out_ptr + rm[:, None] * N + rn[None, :], tot.to(tl.bfloat16), mask=(rm < M)[:, None]
            )
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_quant_rows_kernel(
    x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, H: tl.constexpr, USE_PDL: tl.constexpr
):
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, tl.program_id(0).to(tl.int64), H)
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def q35_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor):
    """Standalone NVFP4 quantization of x [M, H] bf16 (linear scales), byte-
    identical to trtllm::fp4_quantize(x, gs, 16, False, False)."""
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and (h & (h - 1)) == 0
    xq = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    xsf = torch.empty((m, h // 16), dtype=torch.uint8, device=x.device)
    use_pdl = _pdl_enabled()
    _q35_quant_rows_kernel[(m,)](
        x, x.stride(0), global_scale, xq, xsf, H=h, USE_PDL=use_pdl, num_warps=8, launch_pdl=use_pdl
    )
    return xq, xsf


def q35_lowm_router(
    x: torch.Tensor, weight: torch.Tensor, global_scale: Optional[torch.Tensor] = None
):
    """Router logits on the low-M BF16 GEMM (modules/lowm_linear.py); with ``global_scale``
    the NVFP4 quantization of x first: (logits, xq, xsf). Returns None when the low-M
    kernel does not take the shape."""
    from ...modules.lowm_linear import lowm_bf16_gemm

    m = x.shape[0]
    tactic = None if m <= 16 else (16, 1)
    q = q35_nvfp4_quant(x, global_scale) if global_scale is not None else None
    logits = lowm_bf16_gemm(x, weight, out_dtype=torch.bfloat16, tactic=tactic)
    if logits is None:
        return None
    if q is not None:
        return logits, q[0], q[1]
    return logits


class Q35RouterGemm:
    """Per-module scratch + launcher for the single-launch router GEMM."""

    BN = 32
    SPLITK = 8

    def __init__(self):
        self.part = None
        self.cnt = None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ):
        """logits [M, N] bf16; with ``global_scale`` also returns the NVFP4
        quantization of x: (logits, xq [M, K/2] u8, xsf [M, K/16] u8)."""
        m, k = x.shape
        n = weight.shape[0]
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
        assert weight.is_contiguous() and x.stride(1) == 1
        assert m <= Q35_MOE_ROUTER_MAX_TOKENS
        assert n % self.BN == 0 and k % (self.SPLITK * 16) == 0
        bm = 16 if m <= 16 else 32
        dev = x.device
        if self.part is None or self.part.device != dev:
            self.part = torch.empty((self.SPLITK, 32, n), dtype=torch.float32, device=dev)
            self.cnt = torch.zeros((n // self.BN,), dtype=torch.int32, device=dev)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        quant = global_scale is not None
        if quant:
            assert (k & (k - 1)) == 0
            xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
            xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        else:
            xq = xsf = out
            global_scale = self.cnt
        use_pdl = _pdl_enabled()
        grid = (n // self.BN) * self.SPLITK + (m if quant else 0)
        _q35_router_gemm_kernel[(grid,)](
            x,
            x.stride(0),
            weight,
            out,
            self.part,
            self.cnt,
            global_scale,
            xq,
            xsf,
            m,
            N=n,
            K=k,
            BM=bm,
            BN=self.BN,
            SPLITK=self.SPLITK,
            QUANT=quant,
            USE_PDL=use_pdl,
            num_warps=4,
            launch_pdl=use_pdl,
        )
        if quant:
            return out, xq, xsf
        return out


@triton.jit
def _q35_route_tables_kernel(
    logits_ptr,  # [M, E] bf16 router logits
    ids_ptr,  # out [M, K] int32 global expert ids
    w_ptr,  # out [M, K] renormalized weights (dtype of the buffer, bf16 for trtllm-gen)
    hist_ptr,  # [NLP] int32 per-local-expert counts (zero between launches)
    rank_ptr,  # [M * K] int32 scratch: row inside the expert block
    counter_ptr,  # [1] int32 arrival ticket (zero between launches)
    e2p_ptr,  # out [M * K] int32 expanded -> permuted (-1: expert on another rank)
    p2t_ptr,  # out [>= max padded] int32 permuted -> token
    cta_expert_ptr,  # out [>= max CTAs] int32 local expert of each CTA
    cta_limit_ptr,  # out [>= max CTAs] int32 row limit of each CTA
    num_ctas_ptr,  # out [1] int32
    num_padded_ptr,  # out [1] int32
    M,
    local_start,
    x_ptr,  # [M, H] bf16 MoE input (QUANT_H > 0)
    x_row_stride,
    gs_ptr,  # [1] fp32 NVFP4 global scale (QUANT_H > 0)
    xq_ptr,  # out [M, H / 2] uint8 (QUANT_H > 0)
    xsf_ptr,  # out [M, H / 16] uint8 linear scales (QUANT_H > 0)
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TILE: tl.constexpr,
    NL: tl.constexpr,
    NLP: tl.constexpr,
    EXPANDED_PAD: tl.constexpr,
    MAX_CTAS_PER_EXPERT: tl.constexpr,
    USE_PDL: tl.constexpr,
    QUANT_H: tl.constexpr = 0,
):
    """One program per token: top-k of its router logits (reduceTopK order),
    the calcSoftmax renormalization, and the token's rows in the local experts'
    blocks (histogram atomics). The last token program to arrive lays out
    trtllm-gen's routing tables for TILE (what its routing kernel writes for
    precomputed ids), so the MoE op can skip its routing launch.

    QUANT_H > 0 (the router is a plain GEMM): M more programs NVFP4-quantize the
    H-wide rows of x for the experts, beside the routing programs (they take no
    ticket)."""
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    if QUANT_H > 0 and pid >= M:
        _nvfp4_quant_row(
            x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, (pid - M).to(tl.int64), QUANT_H
        )
    else:
        _q35_route_token(
            pid,
            logits_ptr,
            ids_ptr,
            w_ptr,
            hist_ptr,
            rank_ptr,
            counter_ptr,
            e2p_ptr,
            p2t_ptr,
            cta_expert_ptr,
            cta_limit_ptr,
            num_ctas_ptr,
            num_padded_ptr,
            M,
            local_start,
            E,
            K,
            KP,
            TILE,
            NL,
            NLP,
            EXPANDED_PAD,
            MAX_CTAS_PER_EXPERT,
        )
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_route_token(
    row,
    logits_ptr,
    ids_ptr,
    w_ptr,
    hist_ptr,
    rank_ptr,
    counter_ptr,
    e2p_ptr,
    p2t_ptr,
    cta_expert_ptr,
    cta_limit_ptr,
    num_ctas_ptr,
    num_padded_ptr,
    M,
    local_start,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TILE: tl.constexpr,
    NL: tl.constexpr,
    NLP: tl.constexpr,
    EXPANDED_PAD: tl.constexpr,
    MAX_CTAS_PER_EXPERT: tl.constexpr,
):
    """The per-token route + tables work of one program (see
    _q35_route_tables_kernel)."""
    ecols = tl.arange(0, E)
    lg = tl.load(logits_ptr + row * E + ecols)
    bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    # cub TwiddleIn for bf16 (negative -> ~bits, else bits | sign), then the
    # reduce_topk::TopKRedType packing: twiddled value in the upper 16 bits,
    # (65535 - idx) in the lower 16 -> the max picks the larger value, then the
    # smaller index. XOR with the sign bit makes the unsigned order a signed
    # int32 order.
    tw = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
    key = ((tw << 16) | (65535 - ecols)) ^ SIGN_BIT
    slots = tl.arange(0, KP)
    picked = tl.zeros([KP], dtype=tl.int32)
    vals = tl.zeros([KP], dtype=tl.float32)
    for k in tl.static_range(K):
        mx = tl.max(key, axis=0)
        ukey = mx ^ SIGN_BIT
        sel = 65535 - (ukey & 0xFFFF)
        twv = (ukey >> 16) & 0xFFFF
        vb = tl.where((twv & 0x8000) != 0, twv ^ 0x8000, twv ^ 0xFFFF)
        v = vb.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        picked = tl.where(slots == k, sel, picked)
        vals = tl.where(slots == k, v, vals)
        key = tl.where(key == mx, MIN_KEY, key)
    kmask = slots < K
    # The local picks take their row in their expert's block first: the atomics'
    # round trip overlaps the softmax.
    lid = picked - local_start
    local = kmask & (lid >= 0) & (lid < NL)
    rank = tl.atomic_add(
        hist_ptr + tl.where(local, lid, 0),
        tl.full([KP], 1, tl.int32),
        mask=local,
        sem="relaxed",
        scope="gpu",
    )
    # calcSoftmax: max over the k values, expf of the shifted values, the
    # xor-butterfly warp sum order (masks 16, 8, 4, 2, 1 over lanes 0..31 with
    # lanes >= K holding 0), IEEE division.
    vmax = tl.max(tl.where(kmask, vals, float("-inf")), axis=0)
    ex = tl.where(kmask, tl.extra.cuda.libdevice.exp(vals - vmax), 0.0)
    e0 = tl.sum(tl.where(slots == 0, ex, 0.0), axis=0)
    e1 = tl.sum(tl.where(slots == 1, ex, 0.0), axis=0)
    e2 = tl.sum(tl.where(slots == 2, ex, 0.0), axis=0)
    e3 = tl.sum(tl.where(slots == 3, ex, 0.0), axis=0)
    e4 = tl.sum(tl.where(slots == 4, ex, 0.0), axis=0)
    e5 = tl.sum(tl.where(slots == 5, ex, 0.0), axis=0)
    e6 = tl.sum(tl.where(slots == 6, ex, 0.0), axis=0)
    e7 = tl.sum(tl.where(slots == 7, ex, 0.0), axis=0)
    e8 = tl.sum(tl.where(slots == 8, ex, 0.0), axis=0)
    e9 = tl.sum(tl.where(slots == 9, ex, 0.0), axis=0)
    ssum = (((e0 + e8) + e4) + (e2 + e6)) + (((e1 + e9) + e5) + (e3 + e7))
    wts = tl.extra.cuda.libdevice.div_rn(ex, ssum)
    tl.store(ids_ptr + row * K + slots, picked, mask=kmask)
    tl.store(w_ptr + row * K + slots, wts.to(w_ptr.dtype.element_ty), mask=kmask)
    tl.store(rank_ptr + row * K + slots, rank, mask=kmask)
    _fence_acq_rel_gpu()
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == M - 1:
        _fence_acq_rel_gpu()
        i = tl.arange(0, EXPANDED_PAD)
        valid = i < M * K
        e = tl.arange(0, NLP)
        expert = e < NL
        # L2 loads (.cg) after the acquire: data other programs released before the ticket
        # (volatile loads here are serialized).
        gids = tl.load(ids_ptr + i, mask=valid, other=0, cache_modifier=".cg")
        grank = tl.load(rank_ptr + i, mask=valid, other=0, cache_modifier=".cg")
        count = tl.load(hist_ptr + e, mask=expert, other=0, cache_modifier=".cg")
        tl.store(hist_ptr + e, tl.zeros([NLP], dtype=tl.int32), mask=expert)
        num_cta = (count + TILE - 1) // TILE
        offs = tl.cumsum(num_cta, axis=0) - num_cta
        for c in tl.static_range(MAX_CTAS_PER_EXPERT):
            has = expert & (c < num_cta)
            cta = offs + c
            tl.store(cta_expert_ptr + cta, e, mask=has)
            tl.store(
                cta_limit_ptr + cta, tl.minimum((cta + 1) * TILE, offs * TILE + count), mask=has
            )
        total = tl.sum(num_cta, axis=0)
        tl.store(num_ctas_ptr, total)
        tl.store(num_padded_ptr, total * TILE)
        glid = gids - local_start
        glocal = valid & (glid >= 0) & (glid < NL)
        permuted = tl.gather(offs, tl.where(glocal, glid, 0), axis=0) * TILE + grank
        tl.store(e2p_ptr + i, tl.where(glocal, permuted, -1), mask=valid)
        tl.store(p2t_ptr + permuted, i // K, mask=glocal)
        tl.store(counter_ptr, 0)


def q35_prerouted_sizes(num_tokens: int, top_k: int, num_local: int, tile: int):
    """int32 sizes of the routing tables packed in one flat buffer, in the order
    run_moe_prerouted takes them: expanded -> permuted index, permuted index ->
    token (one tile of slack past the route map), CTA -> local expert, CTA -> row
    limit, CTA count, padded row count. Sized for an all-local problem of
    num_local experts, as the op runs a pre-routed expert-parallel layer."""
    max_ctas = max_num_tiles(num_tokens, top_k, num_local, tile)
    return [num_tokens * top_k, (max_ctas + 1) * tile, max_ctas, max_ctas, 1, 1]


def q35_prerouted_views(flat: torch.Tensor, num_tokens: int, top_k: int, num_local: int, tile: int):
    """The routing tables of ``flat`` (q35_prerouted_sizes order) as int32 views."""
    sizes = q35_prerouted_sizes(num_tokens, top_k, num_local, tile)
    return list(torch.split(flat[: sum(sizes)], sizes))


# Tile of the tactic the MoE op last ran a table buffer's shape with (by data pointer;
# table buffers live as long as the process).
_Q35_PREROUTED_TILE_CHOSEN = {}


def q35_record_prerouted_tile(flat: torch.Tensor, tile: int) -> None:
    _Q35_PREROUTED_TILE_CHOSEN[flat.data_ptr()] = tile


def q35_prerouted_tile_chosen(flat: torch.Tensor) -> Optional[int]:
    return _Q35_PREROUTED_TILE_CHOSEN.get(flat.data_ptr())


class Q35RouteTables:
    """Per-layer scratch + launcher of the per-token route + tables kernel: the
    trtllm-gen routing tables built from the router logits."""

    def __init__(self):
        self.hist = None
        self.counter = None

    def __call__(
        self,
        logits: torch.Tensor,
        top_k: int,
        local_start: int,
        num_local: int,
        tile: int,
        weight_dtype=torch.bfloat16,
        flat: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
    ):
        """(ids, wts, *tables) for the local experts [local_start, local_start +
        num_local), the tables in ``flat`` (allocated when None). x / global_scale:
        the same launch also NVFP4-quantizes x [m, H] (bf16) for the experts; the
        call then returns (ids, wts, *tables, xq [m, H/2] u8, xsf [m, H/16] u8)."""
        m, e = logits.shape
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
        assert 1 <= m <= Q35_MOE_ROUTER_MAX_TOKENS and top_k <= 10
        assert (e & (e - 1)) == 0 and (tile & (tile - 1)) == 0
        dev = logits.device
        nlp = triton.next_power_of_2(num_local)
        if self.hist is None or self.hist.numel() < nlp or self.hist.device != dev:
            self.hist = torch.zeros((nlp,), dtype=torch.int32, device=dev)
            self.counter = torch.zeros((1,), dtype=torch.int32, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=weight_dtype, device=dev)
        rank = torch.empty((m * top_k,), dtype=torch.int32, device=dev)
        if flat is None:
            flat = torch.empty(
                (sum(q35_prerouted_sizes(m, top_k, num_local, tile)),),
                dtype=torch.int32,
                device=dev,
            )
        tables = q35_prerouted_views(flat, m, top_k, num_local, tile)
        quant = x is not None
        if quant:
            assert global_scale is not None and x.dtype == torch.bfloat16 and x.shape[0] == m
            assert x.stride(1) == 1 and (x.shape[1] & (x.shape[1] - 1)) == 0
            xq = torch.empty((m, x.shape[1] // 2), dtype=torch.uint8, device=dev)
            xsf = torch.empty((m, x.shape[1] // 16), dtype=torch.uint8, device=dev)
            qargs = (x, x.stride(0), global_scale, xq, xsf)
        else:
            qargs = (logits, 0, self.counter, self.counter, self.counter)
        use_pdl = _pdl_enabled()
        _q35_route_tables_kernel[(2 * m if quant else m,)](
            logits,
            ids,
            wts,
            self.hist,
            rank,
            self.counter,
            *tables,
            m,
            local_start,
            *qargs,
            E=e,
            K=top_k,
            KP=16,
            TILE=tile,
            NL=num_local,
            NLP=nlp,
            EXPANDED_PAD=triton.next_power_of_2(Q35_MOE_ROUTER_MAX_TOKENS * top_k),
            MAX_CTAS_PER_EXPERT=triton.cdiv(Q35_MOE_ROUTER_MAX_TOKENS, tile),
            USE_PDL=use_pdl,
            QUANT_H=x.shape[1] if quant else 0,
            num_warps=_ROUTE_WARPS,
            launch_pdl=use_pdl,
        )
        if quant:
            return ids, wts, *tables, xq, xsf
        return ids, wts, *tables


class Q35PreRoute:
    """Per MoE layer state of the pre-routed front: the launchers and, per decode
    shape, the flat table buffer and its tile."""

    def __init__(self):
        self.route = Q35RouteTables()
        self.buffers = {}
        self.tiles = {}
        self.fused = None  # Q35FusedFront, made on first use
        self.route_big = None  # Q35RouteBig, made on first use

    def plan(self, m: int, top_k: int, num_local: int, device, tile_hint: Optional[int] = None):
        """(flat table buffer, tile) for an m-token step. tile_hint: the tile of
        the tactic the op will run (tables for another tile make the op fall back
        to its own routing). A buffer first made during CUDA graph capture comes
        from the graph's pool and is kept with the layer."""
        tile = tile_hint or self.tiles.get(m) or 8
        flat = self.buffers.get((m, tile))
        chosen = q35_prerouted_tile_chosen(flat) if flat is not None else None
        if chosen is not None and chosen != tile and tile_hint is None:
            self.tiles[m] = tile = chosen
            flat = self.buffers.get((m, tile))
        if flat is None:
            flat = torch.empty(
                (sum(q35_prerouted_sizes(m, top_k, num_local, tile)),),
                dtype=torch.int32,
                device=device,
            )
            self.buffers[(m, tile)] = flat
        return flat, tile

    def __call__(
        self,
        logits: torch.Tensor,
        top_k: int,
        local_start: int,
        num_local: int,
        tile_hint: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
    ):
        """(flat tables, tile, ids, bf16 weights) from one launch of the per-token
        route + tables kernel. With x / global_scale the launch also
        NVFP4-quantizes x and the result ends with (xq, xsf)."""
        m = logits.shape[0]
        flat, tile = self.plan(m, top_k, num_local, logits.device, tile_hint)
        out = self.route(
            logits,
            top_k,
            local_start,
            num_local,
            tile,
            flat=flat,
            x=x,
            global_scale=global_scale,
        )
        res = (flat, tile, out[0], out[1])
        return res + (out[8], out[9]) if x is not None else res


_Q35_FRONT_COMPILED = {}


def q35_front_fused_k_warps(m: int, n: int, k: int) -> Optional[int]:
    """Warps per router CTA of the low-M BF16 router GEMM at m rows (q35_lowm_router's tactic: the default
    tactic up to 16 rows, (16, 1) above), or None when the fused kernel cannot reproduce that GEMM (another
    tile shape or cluster split-K)."""
    from ...modules.lowm_linear import _tactic3, default_tactic

    try:
        tactic = default_tactic(m, n, k, False, False, "bf16") if m <= 16 else (16, 1)
    except ValueError:
        return None
    k_warps, n_tiles, ksplit, _, _ = _tactic3(tactic)
    if n_tiles != 1 or ksplit != 1 or k % (64 * k_warps):
        return None
    return k_warps


def q35_front_fused_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    num_local: int,
    tile: int,
    top_k: int,
) -> Optional[int]:
    """k_warps of the one-launch front for this call, or None when it does not apply."""
    from ...modules.lowm_linear import _device_supported

    if (
        not _device_supported()
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        return None
    m, k = x.shape
    n = weight.shape[0]
    if (
        not 1 <= m <= Q35_MOE_ROUTER_MAX_TOKENS
        or n != 512
        or top_k != 10
        or weight.shape[1] != k
        or not weight.is_contiguous()
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.data_ptr() % 16
        or k % 16
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
    ):
        return None
    if num_local % 32 or not 32 <= num_local <= n or tile & (tile - 1) or not 8 <= tile <= 256:
        return None
    k_warps = q35_front_fused_k_warps(m, n, k)
    if k_warps is None or num_local > 32 * k_warps or m * top_k > 32 * k_warps:
        return None
    return k_warps


def _q35_front_compile(
    m: int,
    k: int,
    n: int,
    k_warps: int,
    num_local: int,
    tile: int,
    top_k: int,
    a_stride: int,
    use_pdl: bool,
):
    key = (m, k, n, k_warps, num_local, tile, top_k, a_stride, use_pdl)
    fn = _Q35_FRONT_COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # compile outside graph capture only (eager warmup precedes it)
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute

        from ...kernels.blackwell.q35_moe_front import Q35MoeFrontKernel
        from ...modules.lowm_linear import _cute, _row_view

        kernel = Q35MoeFrontKernel(
            num_rows=m,
            k_extent=k,
            k_warps=k_warps,
            use_pdl=use_pdl,
            num_experts=n,
            top_k=top_k,
            num_local=num_local,
            tile=tile,
            a_stride=a_stride,
        )
        dev = torch.device("cuda")
        a16 = _row_view(m, k, a_stride, torch.bfloat16, dev)
        w = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
        c = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        z = cutlass.Int64(0)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = (
            cute.compile(
                kernel,
                _cute(a16),
                _cute(w),
                _cute(c),
                z,
                z,
                z,
                z,
                z,
                z,
                z,
                cutlass.Int32(0),
                stream,
            ),
            kernel.ctas,
        )
        _Q35_FRONT_COMPILED[key] = fn
    return fn


class Q35FusedFront:
    """Launcher of the one-launch MoE front with its per-layer staging buffer [32, E] int32 (the router CTA
    zeroes what it consumed, so it is zero between launches)."""

    def __init__(self):
        self.stage = None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        flat: torch.Tensor,
        tile: int,
        local_start: int,
        num_local: int,
        top_k: int,
    ):
        """(logits [m, E] bf16, xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the
        tables written into ``flat`` (q35_prerouted_views layout for (m, top_k, num_local, tile)); None when the
        fused kernel does not apply."""
        k_warps = q35_front_fused_supported(x, weight, global_scale, num_local, tile, top_k)
        if k_warps is None:
            return None
        m, k = x.shape
        n = weight.shape[0]
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(q35_prerouted_sizes(m, top_k, num_local, tile))
        compiled = _q35_front_compile(
            m, k, n, k_warps, num_local, tile, top_k, x.stride(0), _pdl_enabled()
        )
        if compiled is None:
            return None
        fn, _ = compiled
        import cuda.bindings.driver as cuda
        import cutlass

        from ...modules.lowm_linear import _cute

        dev = x.device
        if self.stage is None or self.stage.device != dev or self.stage.numel() < 32 * n:
            self.stage = torch.zeros((32 * n,), dtype=torch.int32, device=dev)
        logits = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(
            _cute(x),
            _cute(weight),
            _cute(logits),
            p64(global_scale),
            p64(xq),
            p64(xsf),
            p64(ids),
            p64(wts),
            p64(flat),
            p64(self.stage),
            cutlass.Int32(local_start),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return logits, xq, xsf, ids, wts


def _q35_front_fused(moe, weight: torch.Tensor, x: torch.Tensor, global_scale: torch.Tensor):
    """Router GEMM + routing + tables + x quant in one launch. Hands the pre-routed state to the backend (as
    _q35_preroute does) and returns (logits, xq, xsf), or None (the caller then runs the two-launch front)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    m = x.shape[0]
    if m > Q35_MOE_FRONT_MAX_M:
        return None
    top_k, local_start, num_local = _q35_route_spec(moe)
    backend = moe.backend
    tactic = q35_tuned_tactic(backend, m)
    flat, tile = state.plan(m, top_k, num_local, x.device, tile_hint=tactic[0] if tactic else None)
    if state.fused is None:
        state.fused = Q35FusedFront()
    out = state.fused(x, weight, global_scale, flat, tile, local_start, num_local, top_k)
    if out is None:
        return None
    logits, xq, xsf, ids, wts = out
    backend._q35_prerouted_pending = (flat, tile, ids, wts)
    return logits, xq, xsf


@_q35_register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global Q35_MOE_FRONT_BIG_C16, Q35_MOE_FRONT_BIG_C16W, Q35_MOE_FRONT_BIG_ROUTE_MAX
    # TRTLLM_Q35_MOE_FRONT_BIG_C16=1 (default off): token counts of Q35_MOE_FRONT_BIG_MS in 257..512 (the MTP-6
    # target verify of CUDA-graph batches 37-73, e.g. 280 / 336 / 392 / 448 at b40 / 48 / 56 / 64) take the
    # route-only launch with thread-block clusters of 16 (q35_moe_front_big.ROUTE_CLUSTER_WIDE) instead of the stock
    # routing kernels + fp4_quantize. Without it such counts keep the stock path.
    Q35_MOE_FRONT_BIG_C16 = os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_C16", "0") == "1"
    # TRTLLM_Q35_MOE_FRONT_BIG_C16W=1 (default off; with C16): also 513..1024 tokens (the MTP-6 target verify of
    # CUDA-graph batches 74-146, e.g. 560 / 672 / 784 / 896 at b80 / 96 / 112 / 128): the same clusters of 16 with up
    # to 64 tokens per routing CTA (two per warp, two-word token masks).
    Q35_MOE_FRONT_BIG_C16W = (
        Q35_MOE_FRONT_BIG_C16 and os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_C16W", "0") == "1"
    )
    Q35_MOE_FRONT_BIG_ROUTE_MAX = (
        1024 if Q35_MOE_FRONT_BIG_C16W else (512 if Q35_MOE_FRONT_BIG_C16 else 256)
    )


_Q35_ROUTE_BIG_COMPILED = {}


def _q35_route_big_compile(
    m: int,
    k: int,
    n: int,
    num_local: int,
    tile: int,
    top_k: int,
    a_stride: int,
    use_pdl: bool,
):
    key = (m, k, n, num_local, tile, top_k, a_stride, use_pdl)
    if key in _Q35_ROUTE_BIG_COMPILED:
        return _Q35_ROUTE_BIG_COMPILED[key]  # None: the kernel does not take this shape
    if torch.cuda.is_current_stream_capturing():
        return None  # compile outside graph capture only (eager warmup precedes it)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...kernels.blackwell.q35_moe_front_big import (
        ROUTE_CLUSTER,
        ROUTE_CLUSTER_WIDE,
        Q35MoeRouteBigKernel,
    )

    wide = (
        m > ROUTE_CLUSTER * 32
    )  # 257..512 (..1024 with C16W) tokens: clusters of 16 (TRTLLM_Q35_MOE_FRONT_BIG_C16)
    try:
        kernel = Q35MoeRouteBigKernel(
            num_rows=m,
            k_extent=k,
            num_experts=n,
            top_k=top_k,
            num_local=num_local,
            tile=tile,
            a_stride=a_stride,
            use_pdl=use_pdl,
            cluster=ROUTE_CLUSTER_WIDE if wide else ROUTE_CLUSTER,
            quant_clusters=3 if wide else 7,
        )
    except ValueError:
        _Q35_ROUTE_BIG_COMPILED[key] = None
        return None
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = (cute.compile(kernel, z, z, z, z, z, z, z, z, cutlass.Int32(0), stream), kernel.ctas)
    _Q35_ROUTE_BIG_COMPILED[key] = fn
    return fn


class Q35RouteBig:
    """Launcher of the big front: routing + pre-routed tables + NVFP4 x in one launch after the stock router
    GEMM (no device state carried between launches)."""

    def compiled(
        self,
        x: torch.Tensor,
        n: int,
        global_scale: torch.Tensor,
        tile: int,
        num_local: int,
        top_k: int,
    ):
        """The compiled launch for this call, or None when the kernel does not apply (checked before the router
        GEMM runs, so the caller can still take the stock path)."""
        from ...modules.lowm_linear import _device_supported

        if not _device_supported() or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        m, k = x.shape
        if (
            m > Q35_MOE_FRONT_BIG_ROUTE_MAX
            or n != 512
            or top_k != 10
            or k % 16
            or x.stride(1) != 1
            or x.stride(0) % 8
            or x.data_ptr() % 16
            or global_scale.dtype != torch.float32
            or global_scale.numel() != 1
        ):
            return None
        if num_local % 32 or not 32 <= num_local <= n or tile & (tile - 1) or not 8 <= tile <= 256:
            return None
        return _q35_route_big_compile(m, k, n, num_local, tile, top_k, x.stride(0), _pdl_enabled())

    def __call__(
        self,
        fn,
        logits: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        flat: torch.Tensor,
        tile: int,
        local_start: int,
        num_local: int,
        top_k: int,
    ):
        """(xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the tables written into
        ``flat`` (q35_prerouted_views layout for (m, top_k, num_local, tile))."""
        import cuda.bindings.driver as cuda
        import cutlass

        m, k = x.shape
        assert (
            logits.dtype == torch.bfloat16 and logits.is_contiguous() and logits.shape == (m, 512)
        )
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(q35_prerouted_sizes(m, top_k, num_local, tile))
        dev = x.device
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
        fn, _ = fn
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(
            p64(logits),
            p64(x),
            p64(global_scale),
            p64(xq),
            p64(xsf),
            p64(ids),
            p64(wts),
            p64(flat),
            cutlass.Int32(local_start),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return xq, xsf, ids, wts


def _q35_route_big(moe, gate, x: torch.Tensor, global_scale: torch.Tensor):
    """The stock router GEMM, then routing + tables + x quant in one launch above 32 tokens. Hands the
    pre-routed state to the backend and returns (logits, xq, xsf), or None (nothing ran; the caller then runs
    the stock path)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    top_k, local_start, num_local = _q35_route_spec(moe)
    backend = moe.backend
    m = x.shape[0]
    tactic = q35_tuned_tactic(backend, m)
    flat, tile = state.plan(m, top_k, num_local, x.device, tile_hint=tactic[0] if tactic else None)
    if state.route_big is None:
        state.route_big = Q35RouteBig()
    fn = state.route_big.compiled(x, gate.weight.shape[0], global_scale, tile, num_local, top_k)
    if fn is None:
        return None
    logits = gate(x)
    xq, xsf, ids, wts = state.route_big(
        fn, logits, x, global_scale, flat, tile, local_start, num_local, top_k
    )
    backend._q35_prerouted_pending = (flat, tile, ids, wts)
    return logits, xq, xsf


def _q35_stock_fp4_inputs(backend, m: int):
    """(FP4BlockScaleMoERunner, inputs) of the FP4 MoE op call the TRTLLM-gen
    NVFP4 backend makes for m tokens with router logits
    (TRTLLMGenFp4BlockScaleBase.run_moe), with meta tensors for the activations:
    enough for an autotuner cache lookup with the same key."""
    from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from ...utils import ActType_TrtllmGen
    from .trtllm_gen.kernel_inputs import (
        extract_routing_params,
        get_data_or_none,
        to_trtllm_gen_act_type,
    )

    act_type = to_trtllm_gen_act_type(backend.activation_type)
    factor = 1 if act_type in [ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu] else 2
    inter = backend.w3_w1_weight.shape[-2] // factor
    per_expert = [
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w3_w1_bias if backend.bias else None,
        backend.act_alpha,
        backend.act_beta,
        backend.act_clamp,
        backend.w2_weight,
        backend.w2_weight_scale,
        backend.w2_bias if backend.bias else None,
        get_data_or_none(backend, "fc31_scale_c"),
        get_data_or_none(backend, "fc31_alpha"),
        get_data_or_none(backend, "fc2_alpha"),
    ]
    rp = extract_routing_params(backend.routing_method)
    runner = FP4BlockScaleMoERunner(
        backend.num_slots,
        rp.top_k,
        rp.n_group,
        rp.topk_group,
        inter,
        backend.slot_start,
        backend.expert_size_per_partition,
        rp.routed_scaling_factor,
        backend.routing_method.routing_method_type,
        True,
        act_type,
        tune_max_num_tokens=backend.max_num_tokens,
        use_dp=backend.use_dp,
    )
    h2 = backend.w3_w1_weight.shape[-1]  # packed fp4: hidden / 2

    def meta(*shape):
        return torch.empty(shape, device="meta")

    inputs = [
        meta(m, backend.num_slots),
        rp.routing_bias,
        meta(m, h2),
        meta(m * h2 // 8),
        *per_expert,
        None,
        None,
    ]
    return runner, inputs


def q35_tuned_tactic(backend, m: int):
    """[tile, cfg] the FP4 MoE op of this layer will run at m tokens (the
    Qwen3.5 tactic table entry, else the autotuner's pick), or None."""
    from ...custom_ops.trtllm_gen_custom_ops import q35_pinned_moe_tactic

    runner, inputs = _q35_stock_fp4_inputs(backend, m)
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::fp4_block_scale_moe_runner", [runner], runner.tuning_config, inputs
    )
    pinned = q35_pinned_moe_tactic(runner, m, inputs[2].shape[1] * 2)
    if pinned is not None:
        tactic = pinned
    if isinstance(tactic, (list, tuple)) and len(tactic) == 2 and tactic[0] > 0:
        return [int(tactic[0]), int(tactic[1])]
    return None


def _q35_preroute_state(moe):
    """The layer's Q35PreRoute, or None when pre-routing does not apply."""
    from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from .routing import RoutingMethodType

    backend = moe.backend
    if (
        moe.routing_method.routing_method_type != RoutingMethodType.Renormalize
        or backend.num_slots != backend.num_experts
        or not FP4BlockScaleMoERunner.supports_prerouted()
    ):
        return None
    state = backend.__dict__.get("_q35_preroute")
    if state is None:
        state = backend._q35_preroute = Q35PreRoute()
    return state


def _q35_route_spec(moe):
    """(top_k, local_start, num_local) of the layer's pre-routed tables."""
    backend = moe.backend
    return (
        moe.routing_method.experts_per_token,
        backend.slot_start,
        backend.expert_size_per_partition,
    )


def _q35_preroute(
    moe,
    logits: torch.Tensor,
    x: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
):
    """Hands the TRTLLM-gen backend the routing of these logits (pre-routed
    tables) from the per-token route + tables launch. With x the launch also
    NVFP4-quantizes it: returns (xq, xsf), or None when the launch did not
    happen (the caller must then quantize x itself)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    backend = moe.backend
    tactic = q35_tuned_tactic(backend, logits.shape[0])
    top_k, local_start, num_local = _q35_route_spec(moe)
    res = state(
        logits,
        top_k,
        local_start,
        num_local,
        tile_hint=tactic[0] if tactic else None,
        x=x,
        global_scale=global_scale,
    )
    backend._q35_prerouted_pending = res[:4]
    return res[4:] if x is not None else None


def q35_prerouted_unfinalized(outputs, flat: torch.Tensor, top_k: int, wts: torch.Tensor):
    """do_finalize=False outputs of a pre-routed MoE run, shaped like the stock
    op's: [fc2 rows, [m, K] top-k weights, [m, K] int32 expanded -> permuted
    index (-1: another rank's pick)]. The weights and the index table are the
    caller's own tensors (the route kernel's output and the head of the flat
    table buffer; a custom op cannot return them, see
    fp4_block_scale_moe_runner)."""
    m = wts.shape[0]
    return [outputs[0], wts.view(m, top_k), flat[: m * top_k].view(m, top_k)]


def max_num_tiles(num_tokens: int, top_k: int, num_local_experts: int, tile: int) -> int:
    """Mirror of getMaxNumCgasInBatchDim (trtllmGen blockScaleMoe runner.h)."""
    remaining = num_tokens * top_k
    filled = min(num_local_experts, remaining)
    remaining -= filled
    tiles = filled
    if remaining > 0:
        tiles += remaining // tile
    return tiles


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _trtllm_quant_eligible(
    moe, x: torch.Tensor, max_tokens: int = Q35_MOE_ROUTER_MAX_TOKENS
) -> bool:
    """Whether ``moe`` (ConfigurableMoE over the native TRTLLM-gen NVFP4 leaf, no
    communication / load balancer) can take the front end for ``x``."""
    from .configurable_moe import ConfigurableMoE
    from .trtllm_gen.trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl

    if not isinstance(moe, ConfigurableMoE):
        return False
    backend = moe.backend
    if not isinstance(backend, TrtllmTrtllmGenNvfp4Impl) or not backend.has_nvfp4:
        return False
    if moe.comm is not None or moe.enable_dwdp or moe.use_dp:
        return False
    if moe.layer_load_balancer is not None or moe.apply_router_weight_on_input:
        return False
    if getattr(backend, "fc31_act_scale", None) is not None:
        return False
    if backend.w3_w1_weight.shape[-1] * 2 != x.shape[-1]:
        return False
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return False
    if x.dim() != 2 or x.shape[0] < 1 or x.shape[0] > max_tokens:
        return False
    if AutoTuner.get().is_tuning_mode:
        return False
    return True


def q35_routed_forward(gate, moe, x: torch.Tensor, **moe_kwargs):
    """Routed-expert output of ``moe`` for ``x`` through the decode front end (module doc), or None when it
    does not apply (the caller then runs the stock path).

    Args:
        gate: The router: ``weight`` [E, H] bf16, ``out_dtype``, ``q35_router_ok(x)``,
            ``q35_router(x, global_scale)`` (Q35RouterGemm with NVFP4 quantization of x) and ``gate(x)``
            (the stock router GEMM above Q35_MOE_ROUTER_MAX_TOKENS).
        moe: The ConfigurableMoE of the routed experts.
        x: [M, H] bf16 MoE input.
        **moe_kwargs: Forwarded to ``moe``.
    """
    from ...utils import Fp4QuantizedTensor

    # The front end serves eager and CUDA-graph calls. Under torch.compile its row-count checks (the membership
    # test in Q35_MOE_FRONT_BIG_MS) would make dynamo specialize the dynamic token count, which piecewise
    # CUDA graphs need symbolic.
    if torch.compiler.is_compiling():
        return None
    if x.dim() != 2 or gate.out_dtype != torch.bfloat16:
        return None
    if x.shape[0] in Q35_MOE_FRONT_BIG_MS and _trtllm_quant_eligible(
        moe, x, max_tokens=Q35_MOE_FRONT_BIG_ROUTE_MAX
    ):
        big = _q35_route_big(moe, gate, x, moe.backend.fc31_input_scale)
        if big is not None:
            logits, xq, xsf = big
            x_in = Fp4QuantizedTensor(xq, xsf, is_sf_swizzled=False)
            return moe(x_in, logits, output_dtype=x.dtype, **moe_kwargs)
    if not (_trtllm_quant_eligible(moe, x) and gate.q35_router_ok(x)):
        return None
    gs = moe.backend.fc31_input_scale
    state = _q35_preroute_state(moe)
    quant = None
    if state is not None:
        fused = _q35_front_fused(moe, gate.weight, x, gs)
        if fused is not None:
            logits, xq, xsf = fused
            quant = (xq, xsf)
        else:
            # The route + tables launch also quantizes x, so no launch is added.
            logits = q35_lowm_router(x, gate.weight)
            if logits is not None:
                quant = _q35_preroute(moe, logits, x=x, global_scale=gs)
    if quant is not None:
        xq, xsf = quant
    else:
        moe.backend.__dict__.pop("_q35_prerouted_pending", None)
        logits, xq, xsf = gate.q35_router(x, gs)
        if state is not None:
            _q35_preroute(moe, logits)
    x_in = Fp4QuantizedTensor(xq, xsf, is_sf_swizzled=False)
    return moe(x_in, logits, output_dtype=x.dtype, **moe_kwargs)


__all__ = [
    "Q35_MOE_FRONT_BIG_MS",
    "Q35_MOE_FRONT_MAX_M",
    "Q35_MOE_ROUTER_MAX_TOKENS",
    "Q35FusedFront",
    "Q35PreRoute",
    "Q35RouteBig",
    "Q35RouteTables",
    "Q35RouterGemm",
    "q35_lowm_router",
    "q35_nvfp4_quant",
    "q35_prerouted_sizes",
    "q35_prerouted_unfinalized",
    "q35_prerouted_views",
    "q35_record_prerouted_tile",
    "q35_routed_forward",
    "q35_tuned_tactic",
]
