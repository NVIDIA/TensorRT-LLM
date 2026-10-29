# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the CuteDSL NVFP4 FC2 entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file builds the checkpoint-shaped operands --
512 experts' down_proj in the CuteDSL weight layout, the input as the CuteDSL
FC1 returns it (permuted NVFP4 rows, swizzled block scales, zero padding rows,
unwritten bytes past the live tiles), the sort as moe_sort returns it -- and
holds the claims no cell carries: per-pair invariance across tiles and tactics,
and the contiguity guard.
"""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_grouped_gemm import (  # noqa: E501
    Fc2Expected,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_grouped_gemm import (  # noqa: E501
    cute_dsl_nvfp4_grouped_gemm as op,
)
from tensorrt_llm._torch.autotuner import AutoTuner

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the CuteDSL FC2 requires a CUDA device"

DEV = "cuda"
E, K, H, INTER, VEC = 512, 10, 2560, 640, 16
NB = INTER // VEC
#: The SwiGLU output maximum FC1's `global_sf` is calibrated to (see `_ckpt` in the entry).
ACT_AMAX = 48.0
GSF = {
    "calibrated": 448 * 6 / ACT_AMAX,
    "subnormal": 2.0**-12 * 448 * 6 / ACT_AMAX,
}

#: Tie window of `_compare`, in units of the element's accumulation magnitude
#: (derived in its docstring).
TIE = 2.0**-20
#: At most this fraction of the gated elements with a nonzero magnitude may take
#: the other side of a tie (4 of them when that is more).
FLIP_BUDGET = 1e-5


# ── the gate ──────────────────────────────────────────────────────────────


def _bf16_order(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> an integer that is monotonic in its value, consecutive between
    neighbouring bf16 values, and equal for +0 and -0."""
    bits = x.view(torch.int16).int() & 0xFFFF
    mag = bits & 0x7FFF
    return torch.where(bits >= 0x8000, -mag, mag)


def _deviation(out: torch.Tensor, ref: Fc2Expected) -> dict:
    """Every disagreement between the op and the reference on the rows it writes.

    An element is `equal` when its value matches the reference's `out` (+0 and
    -0 alike). Otherwise it is a `flip` when the two are neighbouring bf16
    values and the exact value lies within `TIE * magnitude` of the midpoint
    between them, and `outside` when it is anything else. `far` counts the
    disagreements that are not between neighbours at all.
    """
    span = ref.exact.shape[0]
    idx = torch.nonzero(ref.rows[:span]).squeeze(1)
    o = out[idx]
    r = ref.out[idx]
    v = ref.exact[idx]
    s = ref.magnitude[idx]
    of, rf = o.double(), r.double()
    finite = torch.isfinite(of)
    diff = (of != rf) | ~finite
    adj = diff & finite & ((_bf16_order(o) - _bf16_order(r)).abs() == 1)
    dist = ((v - (of + rf) / 2).abs() / s.clamp_min(1e-300)).masked_fill(~adj, 0.0)
    flip = adj & (dist <= TIE)
    outside = diff & ~flip
    return dict(
        rows=int(idx.numel()),
        elements=o.numel(),
        nonzero=int((s > 0).sum()),
        nonfinite=int((~finite).sum()),
        flips=int(flip.sum()),
        outside=int(outside.sum()),
        far=int((diff & ~adj).sum()),
    )


def _compare(out: torch.Tensor, ref: Fc2Expected) -> None:
    """Bitwise on every written row, except at bf16 rounding ties of derived width.

    The op accumulates the 640 products of a row in fp32 on the tensor cores --
    ten MMA k-steps of 64 -- multiplies by alpha in fp32 and rounds to bf16. Each
    step rounds a partial sum no larger than the element's `magnitude` (`|alpha| *
    sum |a_k w_k|`), so the op's fp32 value is within 11 roundings of `2^-24 *
    magnitude` of the exact one; only where the exact value lies that close to a
    bf16 midpoint may the op land on the other neighbour. Everywhere else the two
    agree bit for bit.

    - Flip: only to the neighbouring bf16 value, and only within `TIE` = 2^-20
      (16 fp32 roundings) of `magnitude` from the midpoint.
    - Flips: at most `FLIP_BUDGET` = 1e-5 of the gated elements whose magnitude is
      nonzero (4 when that is fewer) -- padding rows do not widen the budget.
    - No NaN or infinity.

    Every wrongly prepared operand puts most of the pair rows' elements outside the
    gate, most of them not even a bf16 neighbour away.
    """
    assert out.dtype == torch.bfloat16, out.dtype
    assert out.shape == ref.out.shape, (tuple(out.shape), tuple(ref.out.shape))
    d = _deviation(out, ref)
    assert d["rows"] > 0, d
    assert d["nonfinite"] == 0, f"non-finite output on written rows: {d}"
    assert d["outside"] == 0, f"elements off by more than a tie: {d}"
    assert d["flips"] <= max(4, FLIP_BUDGET * d["nonzero"]), f"too many ties flipped: {d}"


# ── operands ──────────────────────────────────────────────────────────────


def _swizzle(sf: torch.Tensor) -> torch.Tensor:
    """`[rows, cols]` scale bytes -> the flat 128x4-swizzled buffer; inverse of `unswizzle`."""
    rows, cols = sf.shape
    return sf.reshape(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def _quantized_experts(num_experts: int, rows: int, cols: int, seed: int):
    """Per expert, a Gaussian `[rows, cols]` matrix quantized the way the checkpoint
    stores it: NVFP4 codes, LINEAR e4m3 block scales, one `weight_scale_2`. Expert
    magnitudes spread over 2^-1..2^1 so a wrong per-expert alpha shows."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    spread = torch.exp2(torch.rand(num_experts, generator=g, device=DEV) * 2 - 1)
    w = torch.empty(num_experts, rows, cols // 2, dtype=torch.uint8, device=DEV)
    s = torch.empty(num_experts, rows, cols // VEC, dtype=torch.uint8, device=DEV)
    ws2 = torch.empty(num_experts, dtype=torch.float32, device=DEV)
    for e in range(num_experts):
        dense = torch.randn(rows, cols, generator=g, device=DEV) * (float(spread[e]) / cols**0.5)
        dense = dense.to(torch.bfloat16)
        gs = (448 * 6 / dense.abs().amax().float()).reshape(1)
        q, sf = torch.ops.trtllm.fp4_quantize(dense, gs, VEC, False, False)
        w[e] = q.view(rows, cols // 2)
        s[e] = sf.view(rows, cols // VEC)
        ws2[e] = 1.0 / gs
    return w, s, ws2


@functools.lru_cache(maxsize=None)
def _down_proj():
    """The checkpoint's `down_proj` experts: `[E, H, I/2]` codes, `[E, H, I/16]`
    linear scales, `weight_scale_2` per expert."""
    return _quantized_experts(E, H, INTER, seed=20261013)


@functools.lru_cache(maxsize=None)
def _weights(gsf: str = "calibrated"):
    """`(weight, weight_scale, alpha)` in the layout CuteDSL reads -- what the target's
    load-time transform produces. `alpha = down_proj.input_scale * weight_scale_2`,
    with the input scale the inverse of FC1's `global_sf`."""
    w, s, ws2 = _down_proj()
    s_k = torch.ops.trtllm.block_scale_interleave(s.contiguous()).view(s.shape)
    alpha = (ws2 / GSF[gsf]).contiguous()
    return w.contiguous().view(torch.float4_e2m1fn_x2), s_k, alpha


def _pair_rows(pairs: int, seed: int, zero_pairs: int = 0) -> torch.Tensor:
    """SwiGLU outputs of FC1, one bf16 `[I]` row per (token, slot) pair, in expanded
    order: `up * silu(gate)` of Gaussian halves."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    up = torch.randn(pairs, INTER, generator=g, device=DEV) * 2
    gate = torch.randn(pairs, INTER, generator=g, device=DEV) * 2
    y = (up * gate * torch.sigmoid(gate)).to(torch.bfloat16)
    y[:zero_pairs] = 0
    return y


def _fc2_input(y: torch.Tensor, sort, gsf: str, garbage_seed: int = 0):
    """This op's input as the CuteDSL FC1 returns it: each pair's row NVFP4-quantized
    with `global_sf` at its permuted row, block scales 128x4-swizzled, the padding
    rows of live tiles zero, and the rows past the live tiles holding stale bytes."""
    t2e, _t2mn, e2p, p2e, _total, n_tiles = sort
    m = p2e.numel()
    tile = m // t2e.numel()
    span = int(n_tiles) * tile
    x = torch.zeros(m, INTER, dtype=torch.bfloat16, device=DEV)
    x[e2p.reshape(-1).long()[: y.shape[0]]] = y
    gs = torch.tensor([GSF[gsf]], dtype=torch.float32, device=DEV)
    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, VEC, False, False)
    q = q.view(torch.uint8).reshape(m, INTER // 2)
    sf = sf.view(torch.uint8).reshape(m, NB)
    g = torch.Generator(device=DEV).manual_seed(garbage_seed)
    q[span:] = torch.randint(0, 256, q[span:].shape, generator=g, device=DEV, dtype=torch.uint8)
    sf[span:] = torch.randint(0, 256, sf[span:].shape, generator=g, device=DEV, dtype=torch.uint8)
    return q.view(torch.float4_e2m1fn_x2), _swizzle(sf).contiguous()


def _ids(dist: str, tokens: int, seed: int) -> torch.Tensor:
    """Expert ids `[tokens, K]` int32: each row's top-K of uniform scores."""
    assert dist == "router", dist
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.rand(tokens, E, device=DEV, generator=g)
    return scores.topk(K, dim=-1).indices.to(torch.int32)


def _sort(ids: torch.Tensor, tile: int):
    """moe_sort as the CuteDSL MoE calls it; the routing weights do not reach FC2."""
    scales = torch.full(ids.shape, 0.1, dtype=torch.float32, device=DEV)
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=tile,
    )


def _args(x, x_sf, sort, tile, gsf="calibrated"):
    """The keyword arguments the target passes, by the op's own names."""
    weight, weight_scale, alpha = _weights(gsf)
    return dict(
        input=x,
        weight=weight,
        input_scale=x_sf,
        weight_scale=weight_scale,
        alpha=alpha,
        tile_idx_to_group_idx=sort[0],
        num_non_exiting_tiles=sort[5],
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        tile_size=tile,
        output_dtype=torch.bfloat16,
        scaling_vector_size=VEC,
    )


def _problem(tokens, tile, seed, dist="router", gsf="calibrated", zero_tokens=0):
    sort = _sort(_ids(dist, tokens, seed + 1), tile)
    y = _pair_rows(tokens * K, seed, zero_pairs=zero_tokens * K)
    x, x_sf = _fc2_input(y, sort, gsf, garbage_seed=seed + 2)
    return _args(x, x_sf, sort, tile, gsf), sort


def _call(args, mma_n, cluster_n):
    """One op call with the autotuner tactic whose MMA tile is `(tile_size, mma_n)`
    and cluster `(tile_size // 128, cluster_n)`, selected through the tuner's own
    capture/replay so the call still goes through the op."""
    tuner = AutoTuner.get()
    with tuner.capture() as cap:
        op(**args)
    tile = args["tile_size"]
    want = ((tile, mma_n), (tile // 128, cluster_n))
    for config in cap:
        ((_runner, tactic),) = config
        if tactic == want:
            with tuner.replay(config):
                return op(**args)
    raise AssertionError(f"no tactic {want}")


def _pairs(out, sort, tokens):
    """Each (token, slot) pair's output row, in expanded order -- what the combine
    reads, whatever row the sort gave it."""
    return out[sort[2].reshape(-1).long()[: tokens * K]]


def _clone(v):
    """A copy that does not depend on copy kernels for the fp4 dtype."""
    if not isinstance(v, torch.Tensor):
        return v
    return v.view(torch.uint8).clone().view(v.dtype) if v.element_size() == 1 else v.clone()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_pairs_equal(a, b, what):
    assert torch.equal(_bits(a), _bits(b)), f"{what}: rows differ"


def _cell_id(cell):
    s = cell.spec
    return (
        f"t{s['tile_size']}-n{s['mma_n']}-c{s['cluster_n']}-T{s['tokens']}-{s['dist']}"
        f"-{s['gsf']}-z{s['zero_tokens']}"
    )


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[_cell_id(c) for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    assert (s["num_experts"], s["top_k"], s["hidden"], s["intermediate"]) == (E, K, H, INTER)
    seed = zlib.crc32(cell.why.encode())
    args, sort = _problem(s["tokens"], s["tile_size"], seed, s["dist"], s["gsf"], s["zero_tokens"])
    before = {k: _clone(v) for k, v in args.items() if isinstance(v, torch.Tensor)}
    out = _call(args, s["mma_n"], s["cluster_n"])
    _compare(out, op.reference(**args))
    for k, v in before.items():
        assert torch.equal(_u8(args[k]), _u8(v)), f"{k} was written"
    if s["zero_tokens"]:
        zero = sort[2][: s["zero_tokens"]].reshape(-1).long()
        assert bool((out[zero] == 0).all()), "an all-zero token's rows are not zero"


# ── invariance: a pair's row depends on the pair alone ────────────────────


def test_tiles_and_tactics_agree_per_pair(certifying):  # noqa: F811
    """Tile 128 / 256, MMA N 128 / 256, cluster N 1 / 2 and the tuner's fallback
    all produce every pair's row bit for bit the same at the 8192-row prefill, so a
    cell certifies the tactic the autotuner picks in serving as well as the one it
    ran."""
    certifying(op)
    tokens = 8192
    ids = _ids("router", tokens, seed=tokens + 1)
    y = _pair_rows(tokens * K, seed=tokens)
    seen = {}
    for tile in (128, 256):
        sort = _sort(ids, tile)
        x, x_sf = _fc2_input(y, sort, "calibrated", garbage_seed=tile)
        args = _args(x, x_sf, sort, tile)
        for mma_n in (128, 256):
            for cluster_n in (1, 2):
                seen[(tile, mma_n, cluster_n)] = _pairs(_call(args, mma_n, cluster_n), sort, tokens)
        fallback = op(**args)
        seen[(tile, "fallback")] = _pairs(fallback, sort, tokens)
        _compare(fallback, op.reference(**args))
    first = seen[(128, 128, 1)]
    for key, got in seen.items():
        _assert_pairs_equal(got, first, f"T={tokens} {key} vs (128, 128, 1)")


# ── the guard ─────────────────────────────────────────────────────────────


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _strided(t: torch.Tensor) -> torch.Tensor:
    """The same values as `t`, as a non-contiguous view into a wider buffer."""
    u = _u8(t)
    wide = torch.zeros(*u.shape[:-1], 2 * u.shape[-1], dtype=u.dtype, device=DEV)
    wide[..., : u.shape[-1]] = u
    view = wide[..., : u.shape[-1]]
    return view.view(t.dtype) if t.element_size() == 1 else view


def test_noncontiguous_operand_answers_wrongly_and_is_refused():
    """The runner hands data_ptr() to the kernel and never reads a stride: the
    same values through a strided view come back finite and wrong. The activation;
    the guard checks the other five operands the same way."""
    name = "input"
    tokens, tile = 64, 128
    args, _ = _problem(tokens, tile, seed=21)
    ref = op.reference(**args)
    bad = dict(args, **{name: _strided(args[name])})
    assert not bad[name].is_contiguous()
    assert torch.equal(_u8(bad[name]), _u8(args[name]))
    with pytest.raises(AssertionError):
        _compare(op(**bad), ref)
    with validating(op), pytest.raises(AssertionError, match=f"^{name} must be contiguous"):
        op(**bad)
