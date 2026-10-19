# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the CuteDSL NVFP4 FC1 entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file builds the checkpoint-shaped operands --
512 experts in the CuteDSL weight layout, the activation as the target quantizes
it, the sort as moe_sort returns it -- and holds the claims no cell carries:
per-pair invariance across tiles, tactics, row order and prefixes; CUDA-graph
replay at the captured row counts; the regions the op leaves unwritten; that
the gate rejects every wrong layout; the guards and the loud rejections.
"""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import (  # noqa: E501
    E2M1,
    E2M1_MAX,
    GATE_UP_GROUP,
    Fc1Expected,
    e2m1_codes,
    e4m3_value,
    unswizzle,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import (  # noqa: E501
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion as op,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.utils import ActivationType

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the CuteDSL FC1 requires a CUDA device"

DEV = "cuda"
E, K, H, INTER, VEC = 512, 10, 2560, 640, 16
SWIGLU = int(ActivationType.Swiglu)
#: The activation's input scale: |x| = 5 maps to the e4m3 maximum, as the
#: checkpoint's per-layer `input_scale` does for its own calibration maximum.
ISC1 = 5.0 / (448 * 6)
#: The SwiGLU output maximum `global_sf` is calibrated to (see `_ckpt` in the entry).
ACT_AMAX = 48.0
GSF = {
    "calibrated": 448 * 6 / ACT_AMAX,
    "saturating": 25 * 448 * 6 / ACT_AMAX,
    "subnormal": 2.0**-12 * 448 * 6 / ACT_AMAX,
}
SENTINEL = 0xA5

#: Tie windows of `_compare`, derived in its docstring.
SF_TIE = 2.0**-20
CODE_TIE = 2.0**-17
#: At most this fraction of block scales / codes may take the other side of a tie
#: (4 of them when that is more).
FLIP_BUDGET = 1e-5

#: Row counts of a decode capture (requests x MTP verify width 1..4, up to 32
#: requests), and the eager prefix lengths: both sides of the power-of-two edges up
#: to the 8192-row ceiling.
_CAPTURED_ROWS = (1, 2, 3, 4, 5, 8, 32, 96, 128)
_PREFIXES = (1, 2, 3, 4, 5, 8, 31, 32, 33, 127, 128, 129, 512, 1023, 1024, 1025, 4095, 4096, 8191)


# ── the gate ──────────────────────────────────────────────────────────────


def _e2m1_rank(packed: torch.Tensor) -> torch.Tensor:
    """Packed codes -> signed magnitude rank in -7..7 per element; -0 and +0 are both 0."""
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return torch.where(codes >= 8, -(codes & 7), codes & 7)


def _e2m1_rank_of(v: torch.Tensor) -> torch.Tensor:
    codes = e2m1_codes(v).long()
    return torch.where(codes >= 8, -(codes & 7), codes & 7)


def _rank_value(rank: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, dtype=torch.float64, device=rank.device)
    return lut[rank.abs()] * torch.sign(rank).double()


def _deviation(out: tuple[torch.Tensor, torch.Tensor], ref: Fc1Expected) -> dict:
    """Every disagreement between the op and the reference on the rows it writes.

    Block scales: equal, or the two neighbours of an e4m3 midpoint that the exact
    pre-rounding scale lies within `SF_TIE` (relative) of -- a `sf_flip`; anything
    else is `sf_outside`. Codes are re-derived under the op's OWN block scale (so
    a flipped scale does not cascade into its 16 codes): equal, or the two
    neighbours of an e2m1 midpoint that the exact `scaled / sf` lies within
    `CODE_TIE` of -- a `code_flip`; anything else is `code_outside`. Blocks whose
    scale is zero dequantize to zero whatever their codes hold and are not
    code-checked. `*_far` counts the disagreements that are not between
    neighbours at all.
    """
    c, c_sf = out
    m, half = c.shape
    inter = half * 2
    nb = c_sf.numel() // m
    vec = inter // nb
    rows = ref.rows
    s_op = unswizzle(c_sf.view(torch.uint8), m, nb)[rows].long()
    s_ref = unswizzle(ref.c_sf.view(torch.uint8), m, nb)[rows].long()
    pre = ref.scaled.view(-1, nb, vec).abs().amax(-1) / E2M1_MAX

    v_op, v_ref = e4m3_value(s_op), e4m3_value(s_ref)
    sf_diff = s_op != s_ref
    finite = (s_op < 0x7F) & (s_ref < 0x7F)
    sf_adj = sf_diff & finite & ((s_op - s_ref).abs() == 1)
    sf_dist = ((pre - (v_op + v_ref) / 2).abs() / pre.clamp_min(1e-300)).masked_fill(~sf_adj, 0.0)
    sf_flip = sf_adj & (sf_dist <= SF_TIE)
    sf_outside = sf_diff & ~sf_flip

    rank_op = _e2m1_rank(c.view(torch.uint8)[rows])
    scale = v_op.repeat_interleave(vec, dim=1)
    live = scale > 0
    q = torch.where(live, ref.scaled / scale, torch.zeros_like(scale))
    rank_exp = _e2m1_rank_of(q)
    code_diff = live & (rank_op != rank_exp)
    code_adj = code_diff & ((rank_op - rank_exp).abs() == 1)
    mid = (_rank_value(rank_op) + _rank_value(rank_exp)) / 2
    code_dist = (q - mid).abs().masked_fill(~code_adj, 0.0)
    code_flip = code_adj & (code_dist <= CODE_TIE)
    code_outside = code_diff & ~code_flip

    return dict(
        rows=int(rows.sum()),
        blocks=s_op.numel(),
        sf_flips=int(sf_flip.sum()),
        sf_outside=int(sf_outside.sum()),
        sf_far=int((sf_diff & ~sf_adj).sum()),
        elements=int(live.sum()),
        code_flips=int(code_flip.sum()),
        code_outside=int(code_outside.sum()),
        code_far=int((code_diff & ~code_adj).sum()),
    )


def _compare(out: tuple[torch.Tensor, torch.Tensor], ref: Fc1Expected) -> None:
    """Bitwise on every written row, except at rounding ties of bounded width.

    The op accumulates in fp32 on the tensor cores, evaluates SwiGLU with
    `ex2.approx` / `rcp.approx`, and divides by the block scale through
    `rcp.approx`; the reference is exact. The op's pre-rounding value is
    therefore off by a handful of fp32 roundings (2^-24 relative each), and
    only where the exact value lies that close to a rounding midpoint may the
    op land on its other side. Everywhere else the two agree bit for bit.

    - Block scale: a flip only within `SF_TIE` = 2^-20 (16 fp32 roundings)
      relative of an e4m3 midpoint.
    - Code: a flip only within `CODE_TIE` = 2^-17 of an e2m1 midpoint, in units
      of the block scale -- 16 fp32 roundings of a block maximum of at most 6,
      rounded up to a power of two.
    - Flips: at most `FLIP_BUDGET` = 1e-5 of the scales and of the codes (4 when
      that is fewer), so the op may not take the far side of every near-tie.

    Every wrongly prepared operand puts most of the block scales outside the gate.
    """
    c, c_sf = out
    assert c.dtype == torch.float4_e2m1fn_x2, c.dtype
    assert c_sf.dtype == torch.uint8, c_sf.dtype
    assert c.shape == ref.c.shape, (tuple(c.shape), tuple(ref.c.shape))
    assert c_sf.shape == ref.c_sf.shape, (tuple(c_sf.shape), tuple(ref.c_sf.shape))
    d = _deviation(out, ref)
    assert d["rows"] > 0, d
    assert d["sf_outside"] == 0, f"block scales off by more than a tie: {d}"
    assert d["code_outside"] == 0, f"codes off by more than a tie: {d}"
    assert d["sf_flips"] <= max(4, FLIP_BUDGET * d["blocks"]), f"too many scale ties flipped: {d}"
    assert d["code_flips"] <= max(4, FLIP_BUDGET * d["elements"]), (
        f"too many code ties flipped: {d}"
    )


# ── operands ──────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _experts(num_experts: int = E):
    """Checkpoint-format routed experts: per expert `[up; gate]` NVFP4 rows with
    LINEAR e4m3 block scales and one `weight_scale_2` shared by both halves, as the
    checkpoint stores them. Expert magnitudes spread over 2^-1..2^1 so a wrong
    per-expert alpha shows."""
    g = torch.Generator(device=DEV).manual_seed(20261013)
    spread = torch.exp2(torch.rand(num_experts, generator=g, device=DEV) * 2 - 1)
    w = torch.empty(num_experts, 2 * INTER, H // 2, dtype=torch.uint8, device=DEV)
    s = torch.empty(num_experts, 2 * INTER, H // VEC, dtype=torch.uint8, device=DEV)
    ws2 = torch.empty(num_experts, dtype=torch.float32, device=DEV)
    for e in range(num_experts):
        dense = torch.randn(2 * INTER, H, generator=g, device=DEV) * (float(spread[e]) / H**0.5)
        dense = dense.to(torch.bfloat16)
        gs = (448 * 6 / dense.abs().amax().float()).reshape(1)
        q, sf = torch.ops.trtllm.fp4_quantize(dense, gs, VEC, False, False)
        w[e] = q.view(2 * INTER, H // 2)
        s[e] = sf.view(2 * INTER, H // VEC)
        ws2[e] = 1.0 / gs
    return w, s, ws2


def _row_perm(n: int, layout: str) -> torch.Tensor:
    """Destination row -> source row of the `[up; gate]` stack."""
    idx = torch.arange(n, device=DEV)
    if layout == "group64":
        return idx.view(2, n // (2 * GATE_UP_GROUP), GATE_UP_GROUP).transpose(0, 1).reshape(-1)
    if layout == "rowwise":  # trtllm-gen's interleave, without its 32-row shuffle
        return idx.view(2, n // 2).transpose(0, 1).reshape(-1)
    if layout == "gate_first":
        half = idx.view(2, n // 2).flip(0).reshape(-1)
        return half[_row_perm(n, "group64")]
    raise ValueError(layout)


@functools.lru_cache(maxsize=None)
def _weights(layout: str = "group64", num_experts: int = E):
    """`(weight, weight_scale, alpha)` in the layout CuteDSL reads -- what the
    target's load-time transform produces -- or one of the wrong layouts the
    discrimination test feeds it."""
    w, s, ws2 = _experts(num_experts)
    perm = _row_perm(
        2 * INTER, "group64" if layout in ("linear_scales", "scales_unpermuted") else layout
    )
    w_k = w[:, perm].contiguous()
    s_rows = s if layout == "scales_unpermuted" else s[:, perm]
    if layout == "linear_scales":
        s_k = s_rows.contiguous()
    else:
        s_k = torch.ops.trtllm.block_scale_interleave(s_rows.contiguous()).view(s.shape)
    alpha = (ISC1 * ws2).contiguous()
    return w_k.view(torch.float4_e2m1fn_x2), s_k, alpha


def _activation(tokens: int, seed: int, zero_tokens: int = 0, swizzled: bool = False):
    """The target's FC1 input: bf16 hidden rows through `fp4_quantize(x, 1/isc1,
    16, False, False)`, scales viewed `[T, H/16]`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, H, generator=g, device=DEV).to(torch.bfloat16)
    x[:zero_tokens] = 0
    gs = torch.tensor([1.0 / ISC1], dtype=torch.float32, device=DEV)
    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, VEC, False, swizzled)
    return q.view(torch.float4_e2m1fn_x2), sf.view(tokens, H // VEC)


def _ids(dist: str, tokens: int, tile: int, seed: int, num_experts: int = E) -> torch.Tensor:
    """Expert ids `[tokens, K]` int32, every row naming K distinct experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if dist == "router":
        scores = torch.rand(tokens, num_experts, device=DEV, generator=g)
        return scores.topk(K, dim=-1).indices.to(torch.int32)
    if dist == "hot":
        hot = torch.randperm(num_experts, device=DEV, generator=g)[:K]
        slots = torch.rand(tokens, K, device=DEV, generator=g).argsort(dim=-1)
        return hot[slots].to(torch.int32)
    if dist == "tile_edges":
        edge = (7, 300, 511)
        scores = torch.rand(tokens, num_experts, device=DEV, generator=g)
        scores[:, list(edge)] = -1.0
        ids = scores.topk(K, dim=-1).indices.to(torch.int32)
        for slot, (expert, rows) in enumerate(zip(edge, (tile - 1, tile, tile + 1))):
            ids[:rows, slot] = expert
        return ids
    if dist == "round_robin":
        base = torch.arange(tokens, device=DEV).unsqueeze(1) * K
        return ((base + torch.arange(K, device=DEV)) % num_experts).to(torch.int32)
    raise ValueError(dist)


def _sort(ids: torch.Tensor, tile: int, num_experts: int = E):
    """moe_sort as the CuteDSL MoE calls it; the routing weights do not reach FC1."""
    scales = torch.full(ids.shape, 0.1, dtype=torch.float32, device=DEV)
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=num_experts,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=num_experts,
        tile_tokens_dim=tile,
    )


def _args(x, x_sf, sort, tile, gsf, layout="group64"):
    """The keyword arguments the target passes, by the op's own names."""
    weight, weight_scale, alpha = _weights(layout)
    t2e, t2mn, _e2p, p2e, _total, n_tiles = sort
    return dict(
        input=x,
        weight=weight,
        input_scale=x_sf,
        weight_scale=weight_scale,
        alpha=alpha,
        tile_idx_to_group_idx=t2e,
        tile_idx_to_mn_limit=t2mn,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=n_tiles,
        global_sf=torch.tensor([gsf], dtype=torch.float32, device=DEV),
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        tile_size=tile,
        scaling_vector_size=VEC,
        activation_type=SWIGLU,
    )


def _problem(tokens, tile, seed, dist="router", gsf="calibrated", zero_tokens=0):
    x, x_sf = _activation(tokens, seed, zero_tokens)
    sort = _sort(_ids(dist, tokens, tile, seed + 1), tile)
    return _args(x, x_sf, sort, tile, GSF[gsf]), sort


def _call(args, mma_n):
    """One op call with the autotuner tactic whose MMA tile is `(tile_size, mma_n)`,
    selected through the tuner's own capture/replay so the call still goes through
    the op."""
    tuner = AutoTuner.get()
    with tuner.capture() as cap:
        op(**args)
    for config in cap:
        ((_runner, tactic),) = config
        if isinstance(tactic, tuple) and tactic[0] == (args["tile_size"], mma_n):
            with tuner.replay(config):
                return op(**args)
    raise AssertionError(f"no tactic with MMA tile ({args['tile_size']}, {mma_n})")


def _pairs(out, sort, tokens):
    """Each (token, slot) pair's output row and its block-scale row, in expanded
    order -- what FC2 and the combine see of it, whatever row the sort gave it."""
    c, c_sf = out
    e2p = sort[2].reshape(-1).long()[: tokens * K]
    sf = unswizzle(c_sf, c.shape[0], c_sf.numel() // c.shape[0])
    return c.view(torch.uint8)[e2p], sf[e2p]


def _clone(v):
    """A copy that does not depend on copy kernels for the fp4 dtype."""
    if not isinstance(v, torch.Tensor):
        return v
    return v.view(torch.uint8).clone().view(v.dtype) if v.element_size() == 1 else v.clone()


def _assert_pairs_equal(a, b, what):
    assert torch.equal(a[0], b[0]), f"{what}: codes differ"
    assert torch.equal(a[1], b[1]), f"{what}: block scales differ"


def _cell_id(cell):
    s = cell.spec
    return (
        f"t{s['tile_size']}-n{s['mma_n']}-T{s['tokens']}-{s['dist']}-{s['gsf']}-z{s['zero_tokens']}"
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
    out = _call(args, s["mma_n"])
    _compare(out, op.reference(**args))
    for k, v in before.items():
        assert torch.equal(args[k].view(torch.uint8), v.view(torch.uint8)), f"{k} was written"
    if s["zero_tokens"]:
        zero = sort[2][: s["zero_tokens"]].reshape(-1).long()
        c, c_sf = out
        sf = unswizzle(c_sf, c.shape[0], INTER // VEC)
        assert bool((sf[zero] == 0).all()), "an all-zero token's block scales are not zero"
        assert bool((c.view(torch.uint8)[zero] & 0x77 == 0).all()), "a zero token's codes are not 0"


# ── invariance: a pair's output depends on the pair alone ─────────────────


@pytest.mark.parametrize("tokens", (1, 128, 8192))
def test_tiles_and_tactics_agree_per_pair(tokens, certifying):  # noqa: F811
    """Tile 128 / 256, MMA N 128 / 256 and the tuner's fallback all produce every
    pair's codes and block scales bit for bit the same, so a cell certifies the
    tactic the autotuner picks in serving as well as the one it ran."""
    certifying(op)
    x, x_sf = _activation(tokens, seed=tokens)
    ids = _ids("router", tokens, 128, seed=tokens + 1)
    seen = {}
    for tile in (128, 256):
        sort = _sort(ids, tile)
        args = _args(x, x_sf, sort, tile, GSF["calibrated"])
        for mma_n in (128, 256):
            seen[(tile, mma_n)] = _pairs(_call(args, mma_n), sort, tokens)
        seen[(tile, "fallback")] = _pairs(op(**args), sort, tokens)
        if tile == 128:
            _compare(op(**args), op.reference(**args))
    first = seen[(128, 128)]
    for key, got in seen.items():
        _assert_pairs_equal(got, first, f"T={tokens} {key} vs (128, 128)")


@pytest.mark.parametrize("tile", (128, 256))
def test_row_order_and_repeat_do_not_change_a_pair(tile, certifying):  # noqa: F811
    """moe_sort orders an expert's rows by atomics (from T = 5). Shuffling every
    expert's run of `permuted_idx_to_expanded_idx`, and calling twice, leaves each
    pair's output bitwise unchanged: the FC1 result is reproducible run to run."""
    certifying(op)
    tokens = 4096
    args, sort = _problem(tokens, tile, seed=77)
    base = _pairs(op(**args), sort, tokens)
    _assert_pairs_equal(_pairs(op(**args), sort, tokens), base, "repeat call")

    t2mn, e2p, p2e, n_tiles = sort[1], sort[2], sort[3], int(sort[5])
    g = torch.Generator(device=DEV).manual_seed(5)
    p2e_s = p2e.clone()
    starts = torch.arange(n_tiles, device=DEV) * tile
    run_of = torch.full((p2e.numel(),), -1, dtype=torch.long, device=DEV)
    for i, (start, limit) in enumerate(zip(starts.tolist(), t2mn[:n_tiles].tolist())):
        run_of[start:limit] = int(sort[0][i])
    for e in torch.unique(run_of[run_of >= 0]).tolist():
        rows = torch.nonzero(run_of == e).squeeze(1)
        p2e_s[rows] = p2e[rows[torch.randperm(rows.numel(), generator=g, device=DEV)]]
    e2p_s = e2p.clone().reshape(-1)
    real = run_of >= 0
    e2p_s[p2e_s[real].long()] = torch.nonzero(real).squeeze(1).to(torch.int32)
    shuffled = (sort[0], sort[1], e2p_s.view_as(e2p), p2e_s, sort[4], sort[5])
    args_s = dict(args, permuted_idx_to_expanded_idx=p2e_s)
    out_s = op(**args_s)
    _compare(out_s, op.reference(**args_s))
    _assert_pairs_equal(_pairs(out_s, shuffled, tokens), base, "shuffled rows")


@pytest.mark.parametrize("tile", (128, 256))
def test_eager_rows_prefix_bitwise(tile, certifying):  # noqa: F811
    """The 8192-row call is gated against the reference, and every prefix call's
    pairs are bitwise the 8192-row call's."""
    certifying(op)
    tokens = 8192
    x, x_sf = _activation(tokens, seed=8)
    ids = _ids("router", tokens, tile, seed=9)
    sort = _sort(ids, tile)
    args = _args(x, x_sf, sort, tile, GSF["calibrated"])
    full = op(**args)
    _compare(full, op.reference(**args))
    full_pairs = _pairs(full, sort, tokens)
    for t in _PREFIXES:
        sub_sort = _sort(ids[:t].contiguous(), tile)
        sub = op(**_args(x[:t], x_sf[:t], sub_sort, tile, GSF["calibrated"]))
        got = _pairs(sub, sub_sort, t)
        n = t * K
        _assert_pairs_equal(got, (full_pairs[0][:n], full_pairs[1][:n]), f"prefix T={t}")


# ── CUDA graphs: captured row counts, on new data ────────────────────────


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
    """moe_sort and this op are captured together, as the target's graph holds them;
    each graph is replayed on two fresh activations and id sets, and its written
    rows are gated against the reference and bitwise against an eager call on the
    graph's own sort output."""
    certifying(op)
    for tokens in _CAPTURED_ROWS:
        x, x_sf = _activation(tokens, seed=tokens)
        ids = _ids("router", tokens, tile, seed=500 + tokens)
        scales = torch.full(ids.shape, 0.1, dtype=torch.float32, device=DEV)
        base = _args(x, x_sf, _sort(ids, tile), tile, GSF["calibrated"])
        state = {}

        def step(ids=ids, scales=scales, base=base, state=state, tile=tile):
            sort = torch.ops.trtllm.moe_sort(
                token_selected_experts=ids,
                token_final_scales=scales,
                num_experts=E,
                top_k=K,
                local_expert_offset=0,
                local_num_experts=E,
                tile_tokens_dim=tile,
            )
            state["sort"] = sort
            return op(
                **dict(
                    base,
                    tile_idx_to_group_idx=sort[0],
                    tile_idx_to_mn_limit=sort[1],
                    permuted_idx_to_expanded_idx=sort[3],
                    num_non_exiting_tiles=sort[5],
                )
            )

        graph, static_out = _capture(step)
        static_sort = state["sort"]
        for trial in (1, 2):
            nx, nx_sf = _activation(tokens, seed=1000 * trial + tokens)
            x.view(torch.uint8).copy_(nx.view(torch.uint8))
            x_sf.copy_(nx_sf)
            ids.copy_(_ids("router", tokens, tile, seed=7000 * trial + tokens))
            graph.replay()
            args = dict(
                base,
                tile_idx_to_group_idx=static_sort[0],
                tile_idx_to_mn_limit=static_sort[1],
                permuted_idx_to_expanded_idx=static_sort[3],
                num_non_exiting_tiles=static_sort[5],
            )
            ref = op.reference(**args)
            _compare(static_out, ref)
            eager = op(**{k: _clone(v) for k, v in args.items()})
            _compare(eager, ref)
            rows = ref.rows
            assert torch.equal(
                static_out[0].view(torch.uint8)[rows], eager[0].view(torch.uint8)[rows]
            ), (tokens, trial)
            m = rows.numel()
            sf_g = unswizzle(static_out[1], m, INTER // VEC)[rows]
            sf_e = unswizzle(eager[1], m, INTER // VEC)[rows]
            assert torch.equal(sf_g, sf_e), (tokens, trial)
        del graph, static_out
    torch.cuda.synchronize()


@pytest.mark.parametrize("tile", (128, 256))
@pytest.mark.parametrize("tokens", (3, 300))
def test_unwritten_regions(tokens, tile):
    """A sentinel planted in a captured graph's outputs: which bytes replay writes.
    The rows the sort filled hold the result; the padding rows of a written tile
    are written with zero block scales and zero codes; the tiles past
    `num_non_exiting_tiles` are not written -- under a graph they keep whatever
    the buffer held."""
    args, sort = _problem(tokens, tile, seed=11)
    graph, static_out = _capture(lambda: op(**args))
    c, c_sf = static_out
    m = c.shape[0]
    for trial in (1, 2):
        nx, nx_sf = _activation(tokens, seed=60 + trial)
        args["input"].view(torch.uint8).copy_(nx.view(torch.uint8))
        args["input_scale"].copy_(nx_sf)
        c.view(torch.uint8).fill_(SENTINEL)
        c_sf.fill_(SENTINEL)
        graph.replay()
        ref = op.reference(**args)
        _compare(static_out, ref)
        n_tiles = int(sort[5])
        span = n_tiles * tile
        codes = c.view(torch.uint8)
        sf = unswizzle(c_sf, m, INTER // VEC)
        assert bool((codes[span:] == SENTINEL).all()) and bool((sf[span:] == SENTINEL).all()), (
            "a tile past num_non_exiting_tiles was written"
        )
        pad = ~ref.rows[:span]
        assert int(pad.sum()) > 0
        assert bool((sf[:span][pad] == 0).all()), "a padding row's block scales are not 0"
        assert bool((codes[:span][pad] == 0).all()), "a padding row's codes are not 0"
    del graph


# ── the gate sees a wrong layout ──────────────────────────────────────────


@pytest.mark.parametrize(
    "layout",
    (
        "rowwise",
        "gate_first",
        "linear_scales",
        "scales_unpermuted",
        "swizzled_input_scales",
        "alpha_rolled",
    ),
)
def test_wrong_layouts_are_rejected(layout):
    """Each operand prepared the wrong way -- trtllm-gen's row interleave, gate
    rows first, scales left linear, scales not given the weight's row
    permutation, the activation's scales swizzled, alpha off by one expert --
    returns finite, plausibly scaled output that `_compare` must reject against
    the reference of the correctly prepared operands."""
    tokens, tile = 128, 128
    x, x_sf = _activation(tokens, seed=3)
    sort = _sort(_ids("router", tokens, tile, seed=4), tile)
    good = _args(x, x_sf, sort, tile, GSF["calibrated"])
    if layout == "swizzled_input_scales":
        _xs, sf_swz = _activation(tokens, seed=3, swizzled=True)
        bad = dict(good, input_scale=sf_swz.reshape(tokens, H // VEC))
    elif layout == "alpha_rolled":
        bad = dict(good, alpha=good["alpha"].roll(1).contiguous())
    else:
        bad = _args(x, x_sf, sort, tile, GSF["calibrated"], layout=layout)
    out = op(**bad)
    ref = op.reference(**good)
    with pytest.raises(AssertionError):
        _compare(out, ref)


# ── guards and loud rejections ────────────────────────────────────────────


def _strided(t: torch.Tensor) -> torch.Tensor:
    """The same values as `t`, as a non-contiguous view into a wider buffer."""
    u = t.view(torch.uint8) if t.element_size() == 1 else t
    if u.dim() == 1:
        wide = torch.zeros(2 * u.numel(), dtype=u.dtype, device=DEV)
        wide[::2] = u
        view = wide[::2]
    else:
        wide = torch.zeros(*u.shape[:-1], 2 * u.shape[-1], dtype=u.dtype, device=DEV)
        wide[..., : u.shape[-1]] = u
        view = wide[..., : u.shape[-1]]
    return view.view(t.dtype) if t.element_size() == 1 else view


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


@pytest.mark.parametrize("name", ("input", "input_scale", "weight", "weight_scale", "alpha"))
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


def test_short_alpha_answers_wrongly_and_is_refused():
    """A 1-element alpha (one per layer instead of one per expert): the kernel reads
    alpha[expert] past its end without raising -- here into zeros of the same
    allocation, so every expert but the first computes zero."""
    tokens, tile = 64, 128
    args, _ = _problem(tokens, tile, seed=31)
    ref = op.reference(**args)
    buf = torch.zeros(E, dtype=torch.float32, device=DEV)
    buf[0] = args["alpha"][0]
    bad = dict(args, alpha=buf[:1])
    with pytest.raises(AssertionError):
        _compare(op(**bad), ref)
    with (
        validating(op),
        pytest.raises(AssertionError, match="alpha has 1 elements for 512 experts"),
    ):
        op(**bad)


def test_short_weight_scale_answers_wrongly_and_is_refused():
    """A weight_scale holding half the bytes the weight needs: the kernel reads past
    its end without raising -- here into zeros of the same allocation, so the upper
    half of the experts compute zero. Past the end of an allocation the same call
    faults with an illegal memory access."""
    tokens, tile = 64, 128
    args, _ = _problem(tokens, tile, seed=32)
    ref = op.reference(**args)
    ws = args["weight_scale"].reshape(-1)
    half = ws.numel() // 2
    buf = torch.zeros(ws.numel(), dtype=torch.uint8, device=DEV)
    buf[:half] = ws[:half]
    bad = dict(args, weight_scale=buf[:half].view(E, INTER, H // VEC))
    with pytest.raises(AssertionError):
        _compare(op(**bad), ref)
    with validating(op), pytest.raises(AssertionError, match="weight_scale holds"):
        op(**bad)


def test_loud_rejections():
    """Raised before the launch: a tile size other than the sort's, either way
    round; 64, which the runner does not build; a wrong scale width, a 2-element
    global_sf, a bf16 alpha, scales typed e4m3 and an activation typed uint8."""
    tokens = 16
    args, _ = _problem(tokens, 128, seed=41)
    with pytest.raises(AssertionError):
        op(**dict(args, tile_size=256))
    args256, _ = _problem(tokens, 256, seed=42)
    with pytest.raises(AssertionError):
        op(**dict(args256, tile_size=128))
    with pytest.raises(ValueError, match="Tile size 64 is not supported"):
        op(**dict(args, tile_size=64))
    with pytest.raises(AssertionError):
        op(**dict(args, input_scale=args["input_scale"][:, :-4].contiguous()))
    with pytest.raises(AssertionError):
        op(**dict(args, global_sf=torch.ones(2, dtype=torch.float32, device=DEV)))
    with pytest.raises(AssertionError):
        op(**dict(args, alpha=args["alpha"].bfloat16()))
    with pytest.raises(AssertionError):
        op(**dict(args, weight_scale=args["weight_scale"].view(torch.float8_e4m3fn)))
    with pytest.raises(AssertionError):
        op(**dict(args, input=args["input"].view(torch.uint8)))
