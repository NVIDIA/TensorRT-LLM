# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3's MoE layer for 1..8 decode tokens in one CuTe DSL launch (``m3_moe``): the router, the routed NVFP4
experts and the shared MXFP8 expert, on caller-owned state (an ``M3MoeState`` and one ``M3MoeLayer`` per MoE layer)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.quantization.utils.fp4_utils import (
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices,
)

from ..m3_moe.op import (
    HIDDEN,
    INTER,
    SHARED_INTER,
    TOP_K,
    M3MoeLayer,
    M3MoeState,
    ar_push_desc,
    is_supported,
)

__all__ = [
    "M3MoeLayer",
    "M3MoeState",
    "ar_push_desc",
    "is_supported",
    "m3_moe_engine",
    "moe_reference",
    "moe_route",
    "mxfp8_dequant",
    "mxfp8_quant_dequant",
    "unswizzle_sf",
]

# The model's constants the kernels are built with: routed_scaling_factor, the shared expert's SwiGLU-OAI
# (swiglu_alpha, swiglu_limit, the (up + 1) offset).
ROUTED_SCALING = 2.0
SHARED_ALPHA = 1.702
SHARED_LIMIT = 7.0

_LOG2E = 1.4426950216293334961
_RCP6 = 0.16666667163372039795  # fp32(1 / 6), the kernels' reciprocal of the E2M1 maximum
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_F32_MAX = 3.40282346638528859812e38


def _f32(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def _e2m1_decode(packed: torch.Tensor) -> torch.Tensor:
    """fp32 values of packed E2M1 bytes [..., K / 2] (low nibble first) -> [..., K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """x (fp32) to the nearest E2M1 value, ties to even, saturating at 6 (cvt.rn.satfinite.e2m1x2)."""
    a = x.abs()
    q = torch.full_like(a, 6.0)
    for hi, value, inclusive in (
        (5.0, 4.0, True),
        (3.5, 3.0, False),
        (2.5, 2.0, True),
        (1.75, 1.5, False),
        (1.25, 1.0, True),
        (0.75, 0.5, False),
        (0.25, 0.0, True),
    ):
        q = torch.where(a <= hi if inclusive else a < hi, torch.full_like(a, value), q)
    return torch.where(x < 0, -q, q)


def _e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """x (fp32) to E4M3, nearest even, saturating (cvt.rn.satfinite.e4m3x2), as fp32."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def _ue8m0_up(x: torch.Tensor) -> torch.Tensor:
    """UE8M0 code of x >= 0 rounded toward +inf, saturating (cvt.rp.satfinite.ue8m0x2), int32."""
    bits = x.contiguous().view(torch.int32)
    code = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return code.clamp(0, 254)


def unswizzle_sf(sf: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    """The [rows, blocks] scale bytes of a 128x4-swizzled scale buffer (int64 indices, any byte dtype)."""
    flat = sf.reshape(-1).view(torch.uint8)
    r = torch.arange(rows, device=flat.device).view(-1, 1)
    b = torch.arange(blocks, device=flat.device).view(1, -1)
    padded_blocks = (blocks + 3) // 4 * 4
    idx = (
        (r // 128) * (128 * padded_blocks)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (b % 4)
    )
    return flat[idx]


def mxfp8_dequant(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """fp32 values of e4m3 ``q`` [R, K] with its 128x4-swizzled UE8M0 scales (one per 32 elements)."""
    rows, k = q.shape
    codes = unswizzle_sf(sf, rows, k // 32).float()
    return q.view(torch.float8_e4m3fn).float() * torch.exp2(codes - 127.0).repeat_interleave(
        32, dim=1
    )


def mxfp8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """x [R, K] through MXFP8 (one UE8M0 scale per 32, the smallest power of two >= amax / 448; e4m3 data), back to
    fp32 -- mxfp8_quantize's recipe, which every MXFP8 activation here reproduces."""
    rows, k = x.shape
    blocks = x.float().view(rows, k // 32, 32)
    code = _ue8m0_up(blocks.abs().amax(dim=-1) * _f32(1.0 / 448.0).to(x.device))
    scale = torch.exp2(code.float() - 127.0).unsqueeze(-1)
    return (_e4m3_round(blocks / scale) * scale).view(rows, k)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [M, K] @ b [N, K]^T, accumulated in fp64 and rounded once to fp32."""
    return (a.double() @ b.double().t()).float()


def _row_perm(rows: int, gated: bool) -> torch.Tensor:
    """The TRTLLM-Gen row order: stored row j holds logical row perm[j] (the gated interleave, then the 32-row shuffle
    of epilogue tile 128)."""
    probe = torch.empty(rows, 1)
    perm = get_shuffle_matrix_a_row_indices(probe, epilogue_tile_m=128)
    if gated:
        perm = get_reorder_rows_for_gated_act_gemm_row_indices(probe)[perm]
    return perm


def _nvfp4_expert(w: torch.Tensor, sf: torch.Tensor, perm_inv: torch.Tensor) -> torch.Tensor:
    """One expert's stored NVFP4 matrix (values [R, K / 2], swizzled e4m3 scales) as raw fp32 [R, K] in logical row
    order (the global scale is not applied: it is the per-expert alpha)."""
    rows, half = w.shape
    codes = unswizzle_sf(sf, rows, 2 * half // 16).view(torch.float8_e4m3fn).float()
    vals = _e2m1_decode(w) * codes.repeat_interleave(16, dim=1)
    return vals[perm_inv]


def moe_route(x: torch.Tensor, router_weight: torch.Tensor, bias: torch.Tensor):
    """MiniMax2 routing on the fp32 router GEMV: (top-4 expert ids [M, 4] int64, bf16-rounded weights [M, 4] fp32)."""
    logits = _gemm(x, router_weight)
    sig = (0.5 * torch.tanh(0.5 * logits).double() + 0.5).float()
    keys = sig + bias
    ids = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    sk = keys.gather(1, ids) - bias[ids]
    total = (sk[:, 0] + sk[:, 2]) + (sk[:, 1] + sk[:, 3])
    w = (sk * _f32(ROUTED_SCALING).to(x.device)) / (total + _f32(1e-20).to(x.device)).unsqueeze(1)
    return ids, _bf16(w)


def _nvfp4_act(x: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """The activation's NVFP4 quantization (fp4_quantize(x, sc, 16)), as raw fp32 values (q x block scale)."""
    rows, k = x.shape
    blocks = x.view(rows, k // 16, 16)
    vmax = blocks.abs().amax(dim=-1, keepdim=True)
    sff = _e4m3_round(sc * (vmax * _f32(_RCP6).to(x.device)))
    # The kernels' rcp.approx(sf x rcp.approx(sc)) lands where the correctly rounded sc / sf does on every element
    # measured (the 1 / (sf x (1 / sc)) spelling differs on ~0.06% of them).
    osc = torch.where(vmax != 0, sc / sff, torch.zeros_like(vmax))
    return (_e2m1_round(blocks * osc) * sff).view(rows, k)


def _fc1_epilogue(up: torch.Tensor, gate: torch.Tensor, esc: torch.Tensor) -> torch.Tensor:
    """TRTLLM-Gen's SwiGLU-OAI FC1 epilogue on the raw FC1 sums (fp32 [I]) and its NVFP4 requantization: the raw FC2
    input values. ``esc`` = (fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha) of the expert, with
    act_beta and act_clamp already divided by fc31_alpha (the loader's convention)."""
    a, c, alpha, beta, lim = (esc[i] for i in range(5))
    gc = torch.where(gate > lim, lim, gate)
    gc = torch.where(gate < -_F32_MAX, torch.full_like(gc, -_F32_MAX), gc)
    uc = torch.minimum(torch.maximum(up, -lim), lim)
    c1 = a * (alpha * _f32(_LOG2E).to(up.device))
    sgm = 1.0 / (torch.exp2(-(gc * c1)) + 1.0)
    h = (uc.double() * a.double() + (a * beta).double()).float() * (gc * sgm)
    blocks = h.view(-1, 16)
    amax = (blocks.abs() * c).amax(dim=-1, keepdim=True)
    sff = _e4m3_round(amax * _f32(_RCP6).to(up.device))
    osc = torch.where(sff != 0, 1.0 / sff, torch.zeros_like(sff))
    return (_e2m1_round((blocks * osc) * c) * sff).view(-1)


def _shared_expert(x: torch.Tensor, shared_mxfp8) -> torch.Tensor:
    """The shared expert as the target runs it outside the engine: MXFP8 gate_up (bf16 out), SwiGLU-OAI with the
    model's alpha / limit / (up + 1) in fp32 (bf16 out), MXFP8 down (bf16 out)."""
    gu_w, gu_sf, dn_w, dn_sf = shared_mxfp8
    gu = _bf16(_gemm(mxfp8_quant_dequant(x), mxfp8_dequant(gu_w, gu_sf)))
    gate, up = gu[:, :SHARED_INTER], gu[:, SHARED_INTER:]
    a = torch.minimum(gate, torch.full_like(gate, SHARED_LIMIT))
    ub = up.clamp(-SHARED_LIMIT, SHARED_LIMIT) + 1.0
    sg = 1.0 / (
        torch.exp2((a * _f32(-SHARED_ALPHA).to(x.device)) * _f32(_LOG2E).to(x.device)) + 1.0
    )
    h = _bf16(ub * (sg * a))
    return _bf16(_gemm(mxfp8_quant_dequant(h), mxfp8_dequant(dn_w, dn_sf)))


def moe_reference(hidden: torch.Tensor, layer: M3MoeLayer, shared: bool) -> torch.Tensor:
    """The MoE layer of the decode engines (both token ranges compute this), in plain torch: bf16 [M, 6144]."""
    x = hidden.float()
    m = x.shape[0]
    ids, weights = moe_route(x, layer.router_weight, layer.bias)
    xr = _nvfp4_act(x, layer.fc31_input_scale.reshape(()).float())
    inv31 = torch.argsort(_row_perm(2 * INTER, gated=True)).to(x.device)
    inv2 = torch.argsort(_row_perm(HIDDEN, gated=False)).to(x.device)
    y = torch.zeros(m, TOP_K, HIDDEN, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        w31 = _nvfp4_expert(layer.w3_w1_weight[e], layer.w3_w1_weight_scale[e], inv31)
        w2 = _nvfp4_expert(layer.w2_weight[e], layer.w2_weight_scale[e], inv2)
        esc = layer.escal[:, e]
        for t, k in (ids == e).nonzero().tolist():
            acc1 = _gemm(xr[t : t + 1], w31)[0]
            # The buffer holds [w3 (up); w1 (gate)].
            h = _fc1_epilogue(acc1[:INTER], acc1[INTER:], esc)
            y[t, k] = _bf16(_gemm(h.unsqueeze(0), w2)[0] * esc[5])
    acc = torch.zeros(m, HIDDEN, dtype=torch.float64, device=x.device)
    for k in range(TOP_K):
        acc = (acc + weights[:, k : k + 1].double() * y[:, k].double()).float().double()
    routed = acc.float()
    if not shared:
        return routed.to(torch.bfloat16)
    return (_shared_expert(x, layer.shared_mxfp8) + _bf16(routed)).to(torch.bfloat16)


class M3MoeEngine(OpWrapper):
    """One MiniMax-M3 MoE layer for M (1..8) decode tokens: ``layer.state.run(layer, hidden, out, arp)``.

    ``hidden`` bf16 [M, 6144] contiguous, M = the state's token count. ``layer`` (from ``M3MoeState.layer``) carries the
    TRTLLM-Gen NVFP4 expert buffers read in place, the per-expert scalars, the router weight / bias, the activation's
    NVFP4 global scale and, on a ``shared`` state, the shared expert's MXFP8 weights; its state carries the compiled
    kernel and the workspace every layer on it shares. Returns bf16 [M, 6144]: bf16(shared + routed) on a ``shared``
    state, else the routed partial. ``out``: write there instead of a new tensor. ``arp`` (``ar_push_desc``): also
    push the rows into a multicast all-reduce exchange.

    The routed part is TRTLLM-Gen's NVFP4 MoE on the fp32 router logits: sigmoid (tanh form) + correction bias top-4
    of 128 (ties to the lower id), weights sigmoid / sum * 2.0 rounded to bf16; the activation quantized to NVFP4 with
    the layer's global scale; per expert the raw FC1 sums, SwiGLU-OAI (alpha / beta / clamp), the NVFP4 requant with
    fc31_scale_c, FC2, bf16(fc2_alpha x sum); the fma chain over the top-4 in order. The shared expert is the target's
    MXFP8 SwiGLU-OAI MLP.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per token count: m_max is a trace-time constant, so each is its own build, and the target builds all
    # eight (decode batches of 1..8 requests, spec-decode verify rows up to 8). The whys name what the build differs in.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 token: pre-tiled shared expert, last phase balanced without trailing FC2",
            spec=dict(tokens=1),
        ),
        Cell(
            why="2 tokens: pre-tiled shared expert, the one build without AR_PRE",
            spec=dict(tokens=2),
        ),
        Cell(
            why="3 tokens: 4 FC2-B builder warps, the shared expert from swizzled scales",
            spec=dict(tokens=3),
        ),
        Cell(
            why="4 tokens: one token per routing warp, logits replicated 128x", spec=dict(tokens=4)
        ),
        Cell(
            why="5 tokens: 5 builder warps, a routing warp takes two tokens, logits 32x",
            spec=dict(tokens=5),
        ),
        Cell(why="6 tokens: one request verifying a 5-token draft", spec=dict(tokens=6)),
        Cell(why="7 tokens: the B2IMG global FC1 -> FC2 hand-off", spec=dict(tokens=7)),
        Cell(
            why="8 tokens: B2IMG with a 6-stage ring, 32 distinct experts at most",
            spec=dict(tokens=8),
        ),
    )

    note = """
    The kernel's elementwise math uses the PTX approximations (rcp.approx, ex2.approx, tanh) where `reference` uses
    torch's correctly rounded ones, and its GEMMs sum in tensor-core order where `reference` sums in fp64: a
    quantization boundary (NVFP4 activation / intermediate code, routing near-tie) can fall on the other side, so the
    two agree within the band the test derives, not bit for bit. The routed part is bit for bit the stock TRTLLM-Gen
    NVFP4 MoE (fp4_quantize + fp4_block_scale_moe_runner) on the same routing; the test checks that too.

    Not covered: the all-reduce push (`arp`), which needs a multicast exchange (the decode forward's boundary all-reduce
    exercises it); the non-shared state (`shared=False`), which this target never builds. Every layer on one state
    must run in one stream order: the workspace is shared and tagged per call, never reset.

    No `is_valid`: the compiled call checks every buffer's dtype, shape and alignment against its signature and
    raises on a mismatch (a misaligned `hidden`, an `out` that is not bf16 [M, 6144], a malformed `arp`).
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.state.run(layer, hidden, out, arp=arp)

    def reference(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plain torch over the layer's buffers (see the class docstring); `out` and `arp` are ignored."""
        return moe_reference(hidden, layer, shared=layer.state.shared)


m3_moe_engine = M3MoeEngine()
