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
"""Decode front end of the native TRTLLM-gen NVFP4 MoE (``supports_prerouted_routing``).

For decode-sized steps it computes the router logits, the top-k expert ids and weights, trtllm-gen's routing
tables for the tactic the MoE op runs (``prerouted_table_sizes``) and the NVFP4 quantization of x, and hands them
to the MoE as ``PreroutedRouting``, which skips its routing kernel. The routing is bit-identical to trtllm-gen
Renormalize (``reduceTopK`` tie order, ``calcSoftmax`` sum order); the row order inside an expert's block may
differ, which the finalize does not depend on. The kernels serve ``KERNEL_NUM_EXPERTS`` experts and top-k
``KERNEL_TOP_K`` (``front_eligible``); other steps take the stock path.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import is_blackwell_datacenter

from ....autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from ....cute_dsl_utils import graph_safe_cute_tensor
from ....flashinfer_utils import get_env_enable_pdl
from ....modules.fla.op import gdc_wait_with_memory_clobber
from ....modules.low_m_gemm import low_m_gemm_dispatcher
from ....modules.low_m_mma_gemm import candidate_tactics
from ....utils import Fp4QuantizedTensor
from ..impl_contract import PreroutedRouting

# Expert count and top-k the routing kernels are written for.
KERNEL_NUM_EXPERTS = 512
KERNEL_TOP_K = 10
# Largest step of the router GEMM, the low-M router and the route + tables launch.
ROUTER_MAX_TOKENS = 32
# Largest step of the one-launch front.
FUSED_FRONT_MAX_TOKENS = 16
# Largest step of the route front (route-only launch after the stock router GEMM): clusters of 8 up to 256
# tokens, clusters of 16 (moe_route_front.ROUTE_CLUSTER_WIDE) above.
ROUTE_FRONT_MAX_TOKENS = 1024
# Warps per program of the per-token route + tables kernel (one warp leaves the last program's table phase
# serial).
_ROUTE_WARPS = 4

SIGN_BIT = tl.constexpr(-2147483648)
MIN_KEY = tl.constexpr(-2147483648)


def route_front_steps(cuda_graph_batch_sizes, tokens_per_gen_step: int) -> frozenset:
    """Token counts the route front takes: the CUDA-graph decode steps (``b`` draft and ``tokens_per_gen_step * b``
    target rows per graph batch size ``b``) above the router GEMM's range. Other counts keep the stock path, so
    the front compiles no kernel for shapes an eager step happens to have."""
    return frozenset(
        m
        for b in cuda_graph_batch_sizes
        for m in (b, tokens_per_gen_step * b)
        if ROUTER_MAX_TOKENS < m <= ROUTE_FRONT_MAX_TOKENS
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
def _router_gemm_kernel(
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
                gdc_wait_with_memory_clobber()
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
            gdc_wait_with_memory_clobber()
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
def _quant_rows_kernel(
    x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, H: tl.constexpr, USE_PDL: tl.constexpr
):
    if USE_PDL:
        gdc_wait_with_memory_clobber()
    _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, tl.program_id(0).to(tl.int64), H)
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def nvfp4_quant_rows(x: torch.Tensor, global_scale: torch.Tensor):
    """Standalone NVFP4 quantization of x [M, H] bf16 (linear scales), byte-
    identical to trtllm::fp4_quantize(x, gs, 16, False, False)."""
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and (h & (h - 1)) == 0
    xq = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    xsf = torch.empty((m, h // 16), dtype=torch.uint8, device=x.device)
    use_pdl = get_env_enable_pdl()
    _quant_rows_kernel[(m,)](
        x, x.stride(0), global_scale, xq, xsf, H=h, USE_PDL=use_pdl, num_warps=8, launch_pdl=use_pdl
    )
    return xq, xsf


class RouterGemm:
    """Per-layer scratch + launcher of the single-launch Triton router GEMM (split-K over K, deterministic
    reduction), optionally with the NVFP4 quantization of x in the same launch."""

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
        assert m <= ROUTER_MAX_TOKENS
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
        use_pdl = get_env_enable_pdl()
        grid = (n // self.BN) * self.SPLITK + (m if quant else 0)
        _router_gemm_kernel[(grid,)](
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
def _route_tables_kernel(
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
        gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    if QUANT_H > 0 and pid >= M:
        _nvfp4_quant_row(
            x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, (pid - M).to(tl.int64), QUANT_H
        )
    else:
        _route_token(
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
def _route_token(
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
    _route_tables_kernel)."""
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


def prerouted_table_sizes(num_tokens: int, top_k: int, num_local: int, tile: int):
    """int32 sizes of the routing tables packed in one flat buffer, in the order
    run_moe_prerouted takes them: expanded -> permuted index, permuted index ->
    token (one tile of slack past the route map), CTA -> local expert, CTA -> row
    limit, CTA count, padded row count. Sized for an all-local problem of
    num_local experts, as the op runs a pre-routed expert-parallel layer."""
    max_ctas = max_num_tiles(num_tokens, top_k, num_local, tile)
    return [num_tokens * top_k, (max_ctas + 1) * tile, max_ctas, max_ctas, 1, 1]


def prerouted_table_views(
    flat: torch.Tensor, num_tokens: int, top_k: int, num_local: int, tile: int
):
    """The routing tables of ``flat`` (prerouted_table_sizes order) as int32 views."""
    sizes = prerouted_table_sizes(num_tokens, top_k, num_local, tile)
    return list(torch.split(flat[: sum(sizes)], sizes))


def max_num_tiles(num_tokens: int, top_k: int, num_local_experts: int, tile: int) -> int:
    """Mirror of getMaxNumCgasInBatchDim (trtllmGen blockScaleMoe runner.h)."""
    remaining = num_tokens * top_k
    filled = min(num_local_experts, remaining)
    remaining -= filled
    tiles = filled
    if remaining > 0:
        tiles += remaining // tile
    return tiles


class RouteTables:
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
        assert 1 <= m <= ROUTER_MAX_TOKENS and top_k == KERNEL_TOP_K
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
                (sum(prerouted_table_sizes(m, top_k, num_local, tile)),),
                dtype=torch.int32,
                device=dev,
            )
        tables = prerouted_table_views(flat, m, top_k, num_local, tile)
        quant = x is not None
        if quant:
            assert global_scale is not None and x.dtype == torch.bfloat16 and x.shape[0] == m
            assert x.stride(1) == 1 and (x.shape[1] & (x.shape[1] - 1)) == 0
            xq = torch.empty((m, x.shape[1] // 2), dtype=torch.uint8, device=dev)
            xsf = torch.empty((m, x.shape[1] // 16), dtype=torch.uint8, device=dev)
            qargs = (x, x.stride(0), global_scale, xq, xsf)
        else:
            qargs = (logits, 0, self.counter, self.counter, self.counter)
        use_pdl = get_env_enable_pdl()
        _route_tables_kernel[(2 * m if quant else m,)](
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
            EXPANDED_PAD=triton.next_power_of_2(ROUTER_MAX_TOKENS * top_k),
            MAX_CTAS_PER_EXPERT=triton.cdiv(ROUTER_MAX_TOKENS, tile),
            USE_PDL=use_pdl,
            QUANT_H=x.shape[1] if quant else 0,
            num_warps=_ROUTE_WARPS,
            launch_pdl=use_pdl,
        )
        if quant:
            return ids, wts, *tables, xq, xsf
        return ids, wts, *tables


_FUSED_FRONT_COMPILED = {}


def _fused_front_compile(
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
    fn = _FUSED_FRONT_COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # compile outside graph capture only (eager warmup precedes it)
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute

        from ....kernels.blackwell.moe_decode_front import MoeDecodeFrontKernel
        from ....modules.low_m_mma_gemm import _row_view

        kernel = MoeDecodeFrontKernel(
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
                graph_safe_cute_tensor(a16),
                graph_safe_cute_tensor(w),
                graph_safe_cute_tensor(c),
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
        _FUSED_FRONT_COMPILED[key] = fn
    return fn


class FusedFront:
    """Launcher of the one-launch front (router GEMM with the low-M GEMM's arithmetic + routing + tables + NVFP4
    x) with its per-layer staging buffer [32, E] int32 (the router CTA zeroes what it consumed, so it is zero
    between launches)."""

    def __init__(self):
        self.stage = None

    @staticmethod
    def supports(
        x: torch.Tensor, weight: torch.Tensor, num_local: int, tile: int, top_k: int, router_tactic
    ) -> bool:
        """Whether one launch takes this call with the router GEMM of ``router_tactic`` (a low-M warp-MMA
        (k_warps, n_tiles, ksplit, ...) for the router weight)."""
        m, k = x.shape
        k_warps, n_tiles, ksplit = router_tactic[:3]
        return (
            m <= FUSED_FRONT_MAX_TOKENS
            and n_tiles == 1
            and ksplit == 1
            and k % (64 * k_warps) == 0
            and num_local <= 32 * k_warps
            and m * top_k <= 32 * k_warps
            and weight.shape[1] == k
            and weight.is_contiguous()
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

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
        k_warps: int,
    ):
        """(logits [m, E] bf16, xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the
        tables written into ``flat`` (prerouted_table_views layout for (m, top_k, num_local, tile)); None when the
        kernel is not compiled yet inside a CUDA-graph capture."""
        m, k = x.shape
        n = weight.shape[0]
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(prerouted_table_sizes(m, top_k, num_local, tile))
        compiled = _fused_front_compile(
            m, k, n, k_warps, num_local, tile, top_k, x.stride(0), get_env_enable_pdl()
        )
        if compiled is None:
            return None
        fn, _ = compiled
        import cuda.bindings.driver as cuda
        import cutlass

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
            graph_safe_cute_tensor(x),
            graph_safe_cute_tensor(weight),
            graph_safe_cute_tensor(logits),
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


_ROUTE_FRONT_COMPILED = {}


def _route_front_compile(
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
    if key in _ROUTE_FRONT_COMPILED:
        return _ROUTE_FRONT_COMPILED[key]  # None: the kernel does not take this shape
    if torch.cuda.is_current_stream_capturing():
        return None  # compile outside graph capture only (eager warmup precedes it)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ....kernels.blackwell.moe_route_front import (
        ROUTE_CLUSTER,
        ROUTE_CLUSTER_WIDE,
        MoeRouteFrontKernel,
    )

    wide = m > ROUTE_CLUSTER * 32  # above 256 tokens: clusters of 16
    try:
        kernel = MoeRouteFrontKernel(
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
        _ROUTE_FRONT_COMPILED[key] = None
        return None
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = (cute.compile(kernel, z, z, z, z, z, z, z, z, cutlass.Int32(0), stream), kernel.ctas)
    _ROUTE_FRONT_COMPILED[key] = fn
    return fn


class RouteFront:
    """Launcher of the route front: routing + pre-routed tables + NVFP4 x in one launch after the stock router
    GEMM (no device state carried between launches)."""

    def compiled(self, x: torch.Tensor, n: int, tile: int, num_local: int, top_k: int):
        """The compiled launch for this call, or None when the kernel does not take it (checked before the router
        GEMM runs, so the caller can still take the stock path)."""
        m, k = x.shape
        if m > ROUTE_FRONT_MAX_TOKENS or x.stride(0) % 8 or x.data_ptr() % 16:
            return None
        return _route_front_compile(
            m, k, n, num_local, tile, top_k, x.stride(0), get_env_enable_pdl()
        )

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
        ``flat`` (prerouted_table_views layout for (m, top_k, num_local, tile))."""
        import cuda.bindings.driver as cuda
        import cutlass

        m, k = x.shape
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
        assert logits.shape == (m, KERNEL_NUM_EXPERTS)
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(prerouted_table_sizes(m, top_k, num_local, tile))
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


def front_eligible(moe, x: torch.Tensor, max_tokens: int) -> bool:
    """Whether ``moe`` and ``x`` take the decode front: a ConfigurableMoE over a TRTLLM-gen NVFP4 leaf that
    takes prerouted routing, without communication, load balancer or DWDP, Renormalize routing over
    ``KERNEL_NUM_EXPERTS`` experts with top-k ``KERNEL_TOP_K``, a 2-D BF16 ``x`` of 1..``max_tokens`` rows, on
    SM 10.0 / 10.3."""
    from ..configurable_moe import ConfigurableMoE
    from ..routing import RoutingMethodType
    from .fp4_block_scale import TRTLLMGenFp4BlockScaleBase

    if not isinstance(moe, ConfigurableMoE) or not is_blackwell_datacenter():
        return False
    backend = moe.backend
    if (
        not isinstance(backend, TRTLLMGenFp4BlockScaleBase)
        or not backend.supports_prerouted_routing
        or not backend.has_nvfp4
    ):
        return False
    if moe.comm is not None or moe.enable_dwdp or moe.use_dp:
        return False
    if moe.layer_load_balancer is not None or moe.apply_router_weight_on_input:
        return False
    if backend.fc31_act_scale is not None:
        return False
    routing = moe.routing_method
    if (
        routing.routing_method_type != RoutingMethodType.Renormalize
        or routing.experts_per_token != KERNEL_TOP_K
        or backend.num_slots != backend.num_experts
        or backend.num_experts != KERNEL_NUM_EXPERTS
    ):
        return False
    num_local = backend.expert_size_per_partition
    if num_local % 32 or not 32 <= num_local <= KERNEL_NUM_EXPERTS:
        return False
    if backend.w3_w1_weight.shape[-1] * 2 != x.shape[-1]:
        return False
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not 1 <= x.shape[0] <= max_tokens:
        return False
    return x.stride(1) == 1 and x.shape[1] % 16 == 0


class FrontRunner(TunableRunner):
    """The two ways a step of up to ``FUSED_FRONT_MAX_TOKENS`` tokens can produce the routed experts' inputs,
    timed whole by the AutoTuner (the experts after them are the same): tactic ``k_warps`` > 0 is the one-launch
    front with the router GEMM split over that many warps; tactic 0 (and the fallback -1) is the gate's own
    projection (the low-M race's pick) followed by the route + tables launch.

    Inputs: [x [m, H] bf16, NVFP4 global scale]. ``tile_of(m)``: the MoE tactic's tile for m tokens, or None."""

    OP = "trtllm::moe_decode_front"
    TUNING_CONFIG = TuningConfig(
        # The one-launch front is compiled per row count, so every count is its own bucket.
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=0,
                dim_idx=0,
                gen_tuning_buckets=tuple(range(1, FUSED_FRONT_MAX_TOKENS + 1)),
                map_to_tuning_buckets=lambda m: m,
            ),
        ),
        use_cold_l2_cache=True,
        use_cuda_graph=True,
        # Every rank runs rank 0's choice, as for the low-M race the router GEMM comes from.
        distributed_tuning_strategy=DistributedTuningStrategy.BROADCAST,
    )

    def __init__(
        self, front: "DecodeFront", gate, num_local: int, local_start: int, top_k: int, tile_of
    ):
        self.front = front
        self.gate = gate
        self.num_local = num_local
        self.local_start = local_start
        self.top_k = top_k
        self.tile_of = tile_of

    def unique_id(self):
        return (tuple(self.gate.weight.shape), self.num_local, self.top_k)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        x = inputs[0]
        m, k = x.shape
        tile = self.tile_of(m)
        if tile is None:
            return []
        w = self.gate.weight
        fused = sorted(
            {
                t[0]
                for t in candidate_tactics(m, w.shape[0], k, fp8=False)
                if t[1] == 1
                and t[2] == 1
                and t[3] == 0
                and FusedFront.supports(x, w, self.num_local, tile, self.top_k, t)
            }
        )
        return [0, *fused]

    def forward(self, inputs, *, tactic=-1, do_preparation=False, **kwargs):
        x, gs = inputs
        m = x.shape[0]
        tile = self.tile_of(m)
        flat = torch.empty(
            (sum(prerouted_table_sizes(m, self.top_k, self.num_local, tile)),),
            dtype=torch.int32,
            device=x.device,
        )
        args = (tile, self.local_start, self.num_local, self.top_k)
        if tactic > 0:
            out = self.front.fused(x, self.gate.weight, gs, flat, *args, tactic)
            if out is not None:
                return out
        logits = self.gate(x)
        ids, wts, *_, xq, xsf = self.front.route(
            logits,
            self.top_k,
            self.local_start,
            self.num_local,
            tile,
            flat=flat,
            x=x,
            global_scale=gs,
        )
        return logits, xq, xsf, ids, wts


class DecodeFront:
    """Per MoE layer decode front (module docstring).

    ``route_steps``: token counts the route front takes after the stock router GEMM (``route_front_steps``);
    other counts up to ``ROUTER_MAX_TOKENS`` run the router GEMM in the front, as one launch where the
    AutoTuner's race of the two (``FrontRunner``) picked it.
    """

    def __init__(self, route_steps: frozenset):
        self.route_steps = route_steps
        self.route = RouteTables()
        self.router_gemm = RouterGemm()
        self.fused = FusedFront()
        self.route_front = RouteFront()
        # Per token count: the MoE op's tactic, and per (token count, tile) the table buffer. A buffer first made
        # during CUDA-graph capture comes from the graph's pool and stays with the layer.
        self._tactics = {}
        self._tables = {}
        # Per token count: the FrontRunner tactic (k_warps of the one-launch front, 0 for the two launches).
        self._front_tactics = {}

    def _moe_tactic(self, backend, m: int) -> Optional[list]:
        tactic = self._tactics.get(m)
        if tactic is None:
            tactic = backend.decode_tactic(m)
            if tactic is not None:
                self._tactics[m] = tactic
        return tactic

    def _table_buffer(self, m: int, top_k: int, num_local: int, tile: int, device) -> torch.Tensor:
        flat = self._tables.get((m, tile))
        if flat is None:
            flat = torch.empty(
                (sum(prerouted_table_sizes(m, top_k, num_local, tile)),),
                dtype=torch.int32,
                device=device,
            )
            self._tables[(m, tile)] = flat
        return flat

    def _front_runner(self, gate, moe) -> FrontRunner:
        backend = moe.backend
        return FrontRunner(
            self,
            gate,
            backend.expert_size_per_partition,
            backend.slot_start,
            moe.routing_method.experts_per_token,
            lambda m: (t[0] if (t := self._moe_tactic(backend, m)) is not None else None),
        )

    def _front_tactic(self, gate, moe, x: torch.Tensor, gs: torch.Tensor) -> int:
        """The raced FrontRunner tactic for x's token count (0 until the race has run)."""
        m = x.shape[0]
        tactic = self._front_tactics.get(m)
        if tactic is None:
            _, tactic = AutoTuner.get().choose_one(
                FrontRunner.OP, [self._front_runner(gate, moe)], FrontRunner.TUNING_CONFIG, [x, gs]
            )
            if tactic == -1:
                return 0
            self._front_tactics[m] = tactic
        return tactic

    def _tune(self, gate, moe, x: torch.Tensor, **moe_kwargs):
        """Autotuning step: the stock router GEMM and MoE (which tune their own races), then, once the router
        race covers the front's token counts, the race of the two fronts. Returns the stock output."""
        out = moe(x, gate(x), output_dtype=x.dtype, **moe_kwargs)
        dispatcher = low_m_gemm_dispatcher(gate)
        if dispatcher is not None and x.shape[0] <= ROUTER_MAX_TOKENS:
            AutoTuner.get().choose_one(
                FrontRunner.OP,
                [self._front_runner(gate, moe)],
                FrontRunner.TUNING_CONFIG,
                [x[:FUSED_FRONT_MAX_TOKENS], moe.backend.fc31_input_scale],
            )
        return out

    def __call__(self, gate, moe, x: torch.Tensor, **moe_kwargs):
        """Routed-expert output of ``moe`` for ``x`` through the decode front, or None when it does not apply
        (the caller then runs the stock path).

        Args:
            gate: The router: ``weight`` [E, H] bf16, ``out_dtype``, and ``gate(x)`` (the stock router GEMM).
            moe: The ConfigurableMoE of the routed experts.
            x: [M, H] bf16 MoE input.
            **moe_kwargs: Forwarded to ``moe``.
        """
        # The front serves eager and CUDA-graph calls. Under torch.compile its row-count checks (the membership
        # test in route_steps) would make dynamo specialize the dynamic token count, which piecewise CUDA graphs
        # need symbolic.
        if torch.compiler.is_compiling() or gate.out_dtype != torch.bfloat16:
            return None
        m = x.shape[0] if x.dim() == 2 else 0
        route_front = m in self.route_steps
        if not front_eligible(moe, x, ROUTE_FRONT_MAX_TOKENS if route_front else ROUTER_MAX_TOKENS):
            return None
        if AutoTuner.get().is_tuning_mode:
            return self._tune(gate, moe, x, **moe_kwargs)
        backend = moe.backend
        tactic = self._moe_tactic(backend, m)
        if tactic is None:
            return None
        tile = tactic[0]
        top_k = moe.routing_method.experts_per_token
        local_start, num_local = backend.slot_start, backend.expert_size_per_partition
        flat = self._table_buffer(m, top_k, num_local, tile, x.device)
        gs = backend.fc31_input_scale
        if route_front:
            fn = self.route_front.compiled(x, gate.weight.shape[0], tile, num_local, top_k)
            if fn is None:
                return None
            logits = gate(x)
            xq, xsf, ids, wts = self.route_front(
                fn, logits, x, gs, flat, tile, local_start, num_local, top_k
            )
        else:
            dispatcher = low_m_gemm_dispatcher(gate)
            fused = None
            k_warps = (
                self._front_tactic(gate, moe, x, gs)
                if dispatcher is not None and m <= FUSED_FRONT_MAX_TOKENS
                else 0
            )
            if k_warps:
                fused = self.fused(
                    x, gate.weight, gs, flat, tile, local_start, num_local, top_k, k_warps
                )
            if fused is not None:
                logits, xq, xsf, ids, wts = fused
            elif dispatcher is not None:
                # The route + tables launch also quantizes x, so no launch is added.
                logits = gate(x)
                ids, wts, *_, xq, xsf = self.route(
                    logits, top_k, local_start, num_local, tile, flat=flat, x=x, global_scale=gs
                )
            else:
                logits, xq, xsf = self.router_gemm(x, gate.weight, gs)
                ids, wts, *_ = self.route(logits, top_k, local_start, num_local, tile, flat=flat)
        x_in = Fp4QuantizedTensor(xq, xsf, is_sf_swizzled=False)
        return moe(
            x_in,
            logits,
            output_dtype=x.dtype,
            prerouted=PreroutedRouting(flat, tactic, ids, wts),
            **moe_kwargs,
        )


__all__ = [
    "DecodeFront",
    "FrontRunner",
    "KERNEL_NUM_EXPERTS",
    "KERNEL_TOP_K",
    "ROUTE_FRONT_MAX_TOKENS",
    "ROUTER_MAX_TOKENS",
    "front_eligible",
    "nvfp4_quant_rows",
    "prerouted_table_sizes",
    "prerouted_table_views",
    "route_front_steps",
]
