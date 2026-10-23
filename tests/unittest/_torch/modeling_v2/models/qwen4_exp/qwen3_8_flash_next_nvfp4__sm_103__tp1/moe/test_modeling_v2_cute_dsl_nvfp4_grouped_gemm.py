# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the CuteDSL NVFP4 FC2 entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file builds the checkpoint-shaped operands --
512 experts' down_proj in the CuteDSL weight layout, the input as the CuteDSL
FC1 returns it (permuted NVFP4 rows, swizzled block scales, zero padding rows,
unwritten bytes past the live tiles), the sort as moe_sort returns it -- and
holds the claims no cell carries: per-pair invariance across tiles, tactics,
row order and prefixes; CUDA-graph replay of sort, FC1 and FC2 at the captured
row counts; the regions the op leaves unwritten; that the gate rejects every
wrong layout; the guards and the loud rejections.
"""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_grouped_gemm import (  # noqa: E501
    Fc2Expected,
    unswizzle,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_grouped_gemm import (  # noqa: E501
    cute_dsl_nvfp4_grouped_gemm as op,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.utils import ActivationType

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
    "saturating": 25 * 448 * 6 / ACT_AMAX,
    "subnormal": 2.0**-12 * 448 * 6 / ACT_AMAX,
}
#: FC1's activation input scale in the chained tests: |x| = 5 maps to the e4m3 maximum.
ISC1 = 5.0 / (448 * 6)
SENTINEL = 0x5A5A

#: Tie window of `_compare`, in units of the element's accumulation magnitude
#: (derived in its docstring).
TIE = 2.0**-20
#: At most this fraction of the gated elements with a nonzero magnitude may take
#: the other side of a tie (4 of them when that is more).
FLIP_BUDGET = 1e-5

#: Row counts of a decode capture (requests x MTP verify width 1..4, up to 32
#: requests), and the eager prefix lengths: both sides of the power-of-two edges up
#: to the 8192-row ceiling.
_CAPTURED_ROWS = (1, 2, 3, 4, 5, 8, 32, 96, 128)
_PREFIXES = (1, 2, 3, 4, 5, 8, 31, 32, 33, 127, 128, 129, 512, 1023, 1024, 1025, 4095, 4096, 8191)


# ── the gate ──────────────────────────────────────────────────────────────


def _bf16_order(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> an integer that is monotonic in its value, consecutive between
    neighbouring bf16 values, and equal for +0 and -0."""
    bits = x.view(torch.int16).int() & 0xFFFF
    mag = bits & 0x7FFF
    return torch.where(bits >= 0x8000, -mag, mag)


def _deviation(out: torch.Tensor, ref: Fc2Expected, rows: torch.Tensor | None = None) -> dict:
    """Every disagreement between the op and the reference on the rows it writes
    (or on the subset `rows` of them).

    An element is `equal` when its value matches the reference's `out` (+0 and
    -0 alike). Otherwise it is a `flip` when the two are neighbouring bf16
    values and the exact value lies within `TIE * magnitude` of the midpoint
    between them, and `outside` when it is anything else. `far` counts the
    disagreements that are not between neighbours at all.
    """
    sel = ref.rows if rows is None else rows
    assert not bool((sel & ~ref.rows).any()), "deviation reads written rows only"
    span = ref.exact.shape[0]
    idx = torch.nonzero(sel[:span]).squeeze(1)
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


def _shuffle32(rows: int) -> torch.Tensor:
    """trtllm-gen's 32-row block shuffle: destination row `8v + u` <- source `4u + v`."""
    idx = torch.arange(rows, device=DEV).view(rows // 32, 8, 4)
    return idx.transpose(1, 2).reshape(-1)


@functools.lru_cache(maxsize=None)
def _weights(gsf: str = "calibrated", layout: str = "cutedsl"):
    """`(weight, weight_scale, alpha)` in the layout CuteDSL reads -- what the target's
    load-time transform produces -- or one of the wrong layouts the discrimination
    test feeds it. `alpha = down_proj.input_scale * weight_scale_2`, with the
    input scale the inverse of FC1's `global_sf`."""
    w, s, ws2 = _down_proj()
    if layout == "shuffle32":
        perm = _shuffle32(H)
        w, s = w[:, perm], s[:, perm]
    if layout == "linear_weight_scales":
        s_k = s.contiguous()
    else:
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


def _ids(dist: str, tokens: int, tile: int, seed: int) -> torch.Tensor:
    """Expert ids `[tokens, K]` int32, every row naming K distinct experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if dist == "router":
        scores = torch.rand(tokens, E, device=DEV, generator=g)
        return scores.topk(K, dim=-1).indices.to(torch.int32)
    if dist == "hot":
        hot = torch.randperm(E, device=DEV, generator=g)[:K]
        slots = torch.rand(tokens, K, device=DEV, generator=g).argsort(dim=-1)
        return hot[slots].to(torch.int32)
    if dist == "tile_edges":
        edge = (7, 300, 511)
        scores = torch.rand(tokens, E, device=DEV, generator=g)
        scores[:, list(edge)] = -1.0
        ids = scores.topk(K, dim=-1).indices.to(torch.int32)
        for slot, (expert, rows) in enumerate(zip(edge, (tile - 1, tile, tile + 1))):
            ids[:rows, slot] = expert
        return ids
    if dist == "round_robin":
        base = torch.arange(tokens, device=DEV).unsqueeze(1) * K
        return ((base + torch.arange(K, device=DEV)) % E).to(torch.int32)
    raise ValueError(dist)


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


def _args(x, x_sf, sort, tile, gsf="calibrated", layout="cutedsl"):
    """The keyword arguments the target passes, by the op's own names."""
    weight, weight_scale, alpha = _weights(gsf, layout)
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
    sort = _sort(_ids(dist, tokens, tile, seed + 1), tile)
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


def _pair_mask(sort, m, tokens):
    mask = torch.zeros(m, dtype=torch.bool, device=DEV)
    mask[sort[2].reshape(-1).long()[: tokens * K]] = True
    return mask


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


@pytest.mark.parametrize("tokens", (1, 128, 8192))
def test_tiles_and_tactics_agree_per_pair(tokens, certifying):  # noqa: F811
    """Tile 128 / 256, MMA N 128 / 256, cluster N 1 / 2 and the tuner's fallback
    all produce every pair's row bit for bit the same, so a cell certifies the
    tactic the autotuner picks in serving as well as the one it ran."""
    certifying(op)
    ids = _ids("router", tokens, 128, seed=tokens + 1)
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


@pytest.mark.parametrize("tile", (128, 256))
def test_row_order_and_repeat_do_not_change_a_pair(tile, certifying):  # noqa: F811
    """moe_sort orders an expert's rows by atomics (from T = 5). Moving every pair to
    another row of its expert's run, and calling twice, leaves each pair's row
    bitwise unchanged: the FC2 result is reproducible run to run."""
    certifying(op)
    tokens = 4096
    ids = _ids("router", tokens, tile, seed=78)
    sort = _sort(ids, tile)
    y = _pair_rows(tokens * K, seed=77)
    x, x_sf = _fc2_input(y, sort, "calibrated")
    args = _args(x, x_sf, sort, tile)
    base = _pairs(op(**args), sort, tokens)
    _assert_pairs_equal(_pairs(op(**args), sort, tokens), base, "repeat call")

    t2e, t2mn, e2p, p2e, n_tiles = sort[0], sort[1], sort[2], sort[3], int(sort[5])
    g = torch.Generator(device=DEV).manual_seed(5)
    p2e_s = p2e.clone()
    starts = torch.arange(n_tiles, device=DEV) * tile
    run_of = torch.full((p2e.numel(),), -1, dtype=torch.long, device=DEV)
    for i, (start, limit) in enumerate(zip(starts.tolist(), t2mn[:n_tiles].tolist())):
        run_of[start:limit] = int(t2e[i])
    for e in torch.unique(run_of[run_of >= 0]).tolist():
        rows = torch.nonzero(run_of == e).squeeze(1)
        p2e_s[rows] = p2e[rows[torch.randperm(rows.numel(), generator=g, device=DEV)]]
    e2p_s = e2p.clone().reshape(-1)
    real = run_of >= 0
    e2p_s[p2e_s[real].long()] = torch.nonzero(real).squeeze(1).to(torch.int32)
    shuffled = (t2e, t2mn, e2p_s.view_as(e2p), p2e_s, sort[4], sort[5])
    assert not torch.equal(e2p_s.view_as(e2p), e2p), "the shuffle moved no pair"
    xs, xs_sf = _fc2_input(y, shuffled, "calibrated")
    args_s = _args(xs, xs_sf, shuffled, tile)
    out_s = op(**args_s)
    _compare(out_s, op.reference(**args_s))
    _assert_pairs_equal(_pairs(out_s, shuffled, tokens), base, "pairs moved within their runs")


@pytest.mark.parametrize("tile", (128, 256))
def test_eager_rows_prefix_bitwise(tile, certifying):  # noqa: F811
    """The 8192-row call is gated against the reference, and every prefix call's
    pairs are bitwise the 8192-row call's."""
    certifying(op)
    tokens = 8192
    ids = _ids("router", tokens, tile, seed=9)
    y = _pair_rows(tokens * K, seed=8)
    sort = _sort(ids, tile)
    x, x_sf = _fc2_input(y, sort, "calibrated")
    args = _args(x, x_sf, sort, tile)
    full = op(**args)
    _compare(full, op.reference(**args))
    full_pairs = _pairs(full, sort, tokens)
    for t in _PREFIXES:
        sub_sort = _sort(ids[:t].contiguous(), tile)
        xs, xs_sf = _fc2_input(y[: t * K], sub_sort, "calibrated", garbage_seed=t)
        got = _pairs(op(**_args(xs, xs_sf, sub_sort, tile)), sub_sort, t)
        _assert_pairs_equal(got, full_pairs[: t * K], f"prefix T={t}")


# ── chained from the CuteDSL FC1, eager and captured ──────────────────────


@functools.lru_cache(maxsize=None)
def _fc1_weights():
    """FC1 operands for the chain: `[E, 2I, H/2]` codes, swizzled scales, per-expert
    alpha and the layer's `global_sf`. Their layout is FC1's business; what FC2
    sees is FC1's output, whatever the weights."""
    w, s, ws2 = _quantized_experts(E, 2 * INTER, H, seed=20261014)
    s_k = torch.ops.trtllm.block_scale_interleave(s.contiguous()).view(s.shape)
    alpha = (ISC1 * ws2).contiguous()
    gsf = torch.tensor([GSF["calibrated"]], dtype=torch.float32, device=DEV)
    return w.view(torch.float4_e2m1fn_x2), s_k, alpha, gsf


def _activation(tokens: int, seed: int):
    """The MoE input as the target quantizes it: `fp4_quantize(x, 1/isc1, 16, False,
    False)`, scales viewed `[T, H/16]`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, H, generator=g, device=DEV).to(torch.bfloat16)
    gs = torch.tensor([1.0 / ISC1], dtype=torch.float32, device=DEV)
    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, VEC, False, False)
    return q.view(torch.float4_e2m1fn_x2), sf.view(tokens, H // VEC)


def _fc1(xq, xsf, sort, tile):
    """The CuteDSL FC1, called as the target calls it; returns `(c, c_sf)`."""
    w1, s1, alpha1, gsf = _fc1_weights()
    return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=xq,
        weight=w1,
        input_scale=xsf,
        weight_scale=s1,
        alpha=alpha1,
        tile_idx_to_group_idx=sort[0],
        tile_idx_to_mn_limit=sort[1],
        permuted_idx_to_expanded_idx=sort[3],
        num_non_exiting_tiles=sort[5],
        global_sf=gsf,
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        tile_size=tile,
        scaling_vector_size=VEC,
        activation_type=int(ActivationType.Swiglu),
    )


@pytest.mark.parametrize("tile", (128, 256))
def test_fc1_output_feeds_it_at_prefill(tile, certifying):  # noqa: F811
    """The real FC1 output at 8192 tokens -- the target's prefill -- passes `_compare`,
    and the padding rows FC1 zeroes come out zero."""
    certifying(op)
    tokens = 8192
    xq, xsf = _activation(tokens, seed=91)
    sort = _sort(_ids("router", tokens, tile, seed=92), tile)
    h, h_sf = _fc1(xq, xsf, sort, tile)
    args = _args(h, h_sf, sort, tile)
    out = op(**args)
    ref = op.reference(**args)
    _compare(out, ref)
    pad = ref.rows & ~_pair_mask(sort, out.shape[0], tokens)
    assert int(pad.sum()) > 0
    assert bool((out[pad] == 0).all()), "a padding row of FC1's output did not come out zero"


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = fn()
    return graph, static


@pytest.mark.parametrize("tile", (128, 256))
def test_captured_rows_replay_on_new_data(tile, certifying):  # noqa: F811
    """moe_sort, FC1 and this op are captured together, as the target's graph holds
    them; each graph is replayed on two fresh activations and id sets, and its
    written rows are gated against the reference of the graph's own FC1 output and
    sort, and bitwise against an eager call on them."""
    certifying(op)
    for tokens in _CAPTURED_ROWS:
        xq, xsf = _activation(tokens, seed=tokens)
        ids = _ids("router", tokens, tile, seed=500 + tokens)
        scales = torch.full(ids.shape, 0.1, dtype=torch.float32, device=DEV)
        state = {}

        def step(xq=xq, xsf=xsf, ids=ids, scales=scales, state=state, tile=tile):
            sort = torch.ops.trtllm.moe_sort(
                token_selected_experts=ids,
                token_final_scales=scales,
                num_experts=E,
                top_k=K,
                local_expert_offset=0,
                local_num_experts=E,
                tile_tokens_dim=tile,
            )
            h, h_sf = _fc1(xq, xsf, sort, tile)
            state["sort"], state["h"], state["h_sf"] = sort, h, h_sf
            return op(**_args(h, h_sf, sort, tile))

        graph, static_out = _capture(step)
        for trial in (1, 2):
            nx, nx_sf = _activation(tokens, seed=1000 * trial + tokens)
            xq.view(torch.uint8).copy_(nx.view(torch.uint8))
            xsf.copy_(nx_sf)
            ids.copy_(_ids("router", tokens, tile, seed=7000 * trial + tokens))
            graph.replay()
            args = _args(state["h"], state["h_sf"], state["sort"], tile)
            ref = op.reference(**args)
            _compare(static_out, ref)
            eager = op(**{k: _clone(v) for k, v in args.items()})
            rows = ref.rows
            assert torch.equal(_bits(static_out[rows]), _bits(eager[rows])), (tokens, trial)
        del graph, static_out
    torch.cuda.synchronize()


# ── what the op writes ────────────────────────────────────────────────────


@pytest.mark.parametrize("tile", (128, 256))
@pytest.mark.parametrize("tokens", (3, 300))
def test_unwritten_regions(tokens, tile):
    """A sentinel planted in a captured graph's output: which rows replay writes.
    Every row of the live tiles is written -- the padding rows computed from
    whatever the input holds there, zero for FC1's zeroed padding -- and the rows
    past `num_non_exiting_tiles` are not: under a graph they keep whatever the
    buffer held."""
    args, sort = _problem(tokens, tile, seed=11)
    graph, out = _capture(lambda: op(**args))
    m = out.shape[0]
    span = int(sort[5]) * tile
    pairs = _pair_mask(sort, m, tokens)
    pad = torch.zeros(m, dtype=torch.bool, device=DEV)
    pad[:span] = ~pairs[:span]
    assert int(pad.sum()) > 0
    # With no more pairs than experts the sort sizes one tile per pair; when no
    # two pairs share an expert (T = 3 here) every tile is live.
    assert span < m or tokens * K <= E, "the sort left no tile unused"
    _bits(out).fill_(SENTINEL)
    graph.replay()
    _compare(out, op.reference(**args))
    untouched = (_bits(out) == SENTINEL).all(dim=1)
    assert bool(untouched[span:].all()), "a row past num_non_exiting_tiles was written"
    assert not bool(untouched[:span].any()), "a row of a live tile was not written"
    assert bool((out[pad] == 0).all()), "a zero padding row did not come out zero"
    pair_rows = out[pairs].clone()

    # Nonzero codes (all 0.5) under unit block scales in the padding rows.
    codes = args["input"].view(torch.uint8)
    codes[pad] = 0x11
    sf = unswizzle(args["input_scale"], m, NB).clone()
    sf[pad] = 0x38
    args["input_scale"].copy_(_swizzle(sf))
    graph.replay()
    assert bool((out[pad] != 0).any()), "nonzero padding input still gave zero rows"
    assert torch.equal(_bits(out[pairs]), _bits(pair_rows)), "padding input changed a pair row"
    _compare(out, op.reference(**args))
    del graph


# ── the gate sees a wrong layout ──────────────────────────────────────────


@pytest.mark.parametrize(
    "layout",
    (
        "shuffle32",
        "linear_weight_scales",
        "linear_input_scales",
        "alpha_rolled",
        "alpha_per_layer",
    ),
)
def test_wrong_layouts_are_rejected(layout):
    """Each operand prepared the wrong way -- trtllm-gen's 32-row shuffle, weight
    scales left linear, the input's scales linear, alpha off by one expert, one
    alpha for the whole layer -- returns finite, plausibly scaled output that
    `_compare` must reject against the reference of the correctly prepared
    operands."""
    tokens, tile = 128, 128
    good, sort = _problem(tokens, tile, seed=3)
    m = good["input"].shape[0]
    if layout in ("shuffle32", "linear_weight_scales"):
        bad = _args(good["input"], good["input_scale"], sort, tile, layout=layout)
    elif layout == "linear_input_scales":
        linear = unswizzle(good["input_scale"], m, NB).reshape(-1).contiguous()
        bad = dict(good, input_scale=linear)
    elif layout == "alpha_rolled":
        bad = dict(good, alpha=good["alpha"].roll(1).contiguous())
    else:
        bad = dict(good, alpha=torch.full_like(good["alpha"], float(good["alpha"].mean())))
    out = op(**bad)
    ref = op.reference(**good)
    assert _deviation(out, ref, rows=_pair_mask(sort, m, tokens))["nonfinite"] == 0
    with pytest.raises(AssertionError):
        _compare(out, ref)


# ── guards and loud rejections ────────────────────────────────────────────


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _strided(t: torch.Tensor) -> torch.Tensor:
    """The same values as `t`, as a non-contiguous view into a wider buffer."""
    u = _u8(t)
    if u.dim() == 1:
        wide = torch.zeros(2 * u.numel(), dtype=u.dtype, device=DEV)
        wide[::2] = u
        view = wide[::2]
    else:
        wide = torch.zeros(*u.shape[:-1], 2 * u.shape[-1], dtype=u.dtype, device=DEV)
        wide[..., : u.shape[-1]] = u
        view = wide[..., : u.shape[-1]]
    return view.view(t.dtype) if t.element_size() == 1 else view


@pytest.mark.parametrize(
    "name",
    ("input", "input_scale", "weight", "weight_scale", "alpha", "tile_idx_to_group_idx"),
)
def test_noncontiguous_operand_answers_wrongly_and_is_refused(name):
    """The runner hands data_ptr() to the kernel and never reads a stride: the
    same values through a strided view come back finite and wrong."""
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


def test_loud_rejections():
    """Raised before the launch: a tile size other than the sort's, either way
    round; 64, which the runner does not build; a non-bf16 output; and a wrong
    dtype, rank or size of every tensor operand, each tripping a runner assert."""
    tokens = 16
    args, _ = _problem(tokens, 128, seed=41)
    m = args["input"].shape[0]
    with pytest.raises(AssertionError):
        op(**dict(args, tile_size=256))
    args256, _ = _problem(tokens, 256, seed=42)
    with pytest.raises(AssertionError):
        op(**dict(args256, tile_size=128))
    with pytest.raises(ValueError, match="supports tile_size .* 128 and 256 only, but got 64"):
        op(**dict(args, tile_size=64))
    rejected = dict(
        output_dtype=torch.float16,
        alpha=args["alpha"].bfloat16(),
        input=args["input"].view(torch.uint8),
        input_scale=args["input_scale"].view(m, NB),
        weight_scale=args["weight_scale"].view(torch.float8_e4m3fn),
        scaling_vector_size=32,
        tile_idx_to_group_idx=args["tile_idx_to_group_idx"].long(),
        num_non_exiting_tiles=torch.zeros(2, dtype=torch.int32, device=DEV),
    )
    short = dict(
        alpha=args["alpha"][:1].contiguous(),
        weight_scale=args["weight_scale"][: E // 2],
        input_scale=args["input_scale"][:-4].contiguous(),
        tile_idx_to_group_idx=args["tile_idx_to_group_idx"][:-1].contiguous(),
    )
    for name, value in list(rejected.items()) + list(short.items()):
        with pytest.raises(AssertionError):
            op(**dict(args, **{name: value}))
