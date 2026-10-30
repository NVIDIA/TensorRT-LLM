# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the CuteDSL NVFP4 FC1 entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file builds the checkpoint-shaped operands --
512 experts in the CuteDSL weight layout, the activation as the target quantizes
it, the sort as moe_sort returns it -- and holds the claims no cell carries:
per-pair invariance across tiles and tactics, and the contiguity guard.
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
    "subnormal": 2.0**-12 * 448 * 6 / ACT_AMAX,
}

#: Tie windows of `_compare`, derived in its docstring.
SF_TIE = 2.0**-20
CODE_TIE = 2.0**-17
#: At most this fraction of block scales / codes may take the other side of a tie
#: (4 of them when that is more).
FLIP_BUDGET = 1e-5


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


def _row_perm(n: int) -> torch.Tensor:
    """Destination row -> source row of the `[up; gate]` stack, in `GATE_UP_GROUP`-row
    groups."""
    idx = torch.arange(n, device=DEV)
    return idx.view(2, n // (2 * GATE_UP_GROUP), GATE_UP_GROUP).transpose(0, 1).reshape(-1)


@functools.lru_cache(maxsize=None)
def _weights():
    """`(weight, weight_scale, alpha)` in the layout CuteDSL reads -- what the
    target's load-time transform produces."""
    w, s, ws2 = _experts()
    perm = _row_perm(2 * INTER)
    w_k = w[:, perm].contiguous()
    s_k = torch.ops.trtllm.block_scale_interleave(s[:, perm].contiguous()).view(s.shape)
    alpha = (ISC1 * ws2).contiguous()
    return w_k.view(torch.float4_e2m1fn_x2), s_k, alpha


def _activation(tokens: int, seed: int, zero_tokens: int = 0):
    """The target's FC1 input: bf16 hidden rows through `fp4_quantize(x, 1/isc1,
    16, False, False)`, scales viewed `[T, H/16]`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, H, generator=g, device=DEV).to(torch.bfloat16)
    x[:zero_tokens] = 0
    gs = torch.tensor([1.0 / ISC1], dtype=torch.float32, device=DEV)
    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, VEC, False, False)
    return q.view(torch.float4_e2m1fn_x2), sf.view(tokens, H // VEC)


def _ids(dist: str, tokens: int, seed: int) -> torch.Tensor:
    """Expert ids `[tokens, K]` int32: each row's top-K of uniform scores."""
    assert dist == "router", dist
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.rand(tokens, E, device=DEV, generator=g)
    return scores.topk(K, dim=-1).indices.to(torch.int32)


def _sort(ids: torch.Tensor, tile: int):
    """moe_sort as the CuteDSL MoE calls it; the routing weights do not reach FC1."""
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


def _args(x, x_sf, sort, tile, gsf):
    """The keyword arguments the target passes, by the op's own names."""
    weight, weight_scale, alpha = _weights()
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
    sort = _sort(_ids(dist, tokens, seed + 1), tile)
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


def test_tiles_and_tactics_agree_per_pair(certifying):  # noqa: F811
    """Tile 128 / 256, MMA N 128 / 256 and the tuner's fallback all produce every
    pair's codes and block scales bit for bit the same at the 8192-row prefill, so a
    cell certifies the tactic the autotuner picks in serving as well as the one it
    ran."""
    certifying(op)
    tokens = 8192
    x, x_sf = _activation(tokens, seed=tokens)
    ids = _ids("router", tokens, seed=tokens + 1)
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


# ── the guard ─────────────────────────────────────────────────────────────


def _strided(t: torch.Tensor) -> torch.Tensor:
    """The same values as `t`, as a non-contiguous view into a wider buffer."""
    u = t.view(torch.uint8) if t.element_size() == 1 else t
    wide = torch.zeros(*u.shape[:-1], 2 * u.shape[-1], dtype=u.dtype, device=DEV)
    wide[..., : u.shape[-1]] = u
    view = wide[..., : u.shape[-1]]
    return view.view(t.dtype) if t.element_size() == 1 else view


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def test_noncontiguous_operand_answers_wrongly_and_is_refused():
    """The runner hands data_ptr() to the kernel and never reads a stride: the
    same values through a strided view come back finite and wrong. The activation;
    the guard checks input_scale, weight, weight_scale and alpha the same way."""
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
