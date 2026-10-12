# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_fake_quantize entry.

The reference (`op.reference`) is the model's `fp4_act_quant(x, 16, inplace, scale_dtype=float8_e4m3fn)`
recipe written out in native torch ops: amax floor 6*2^-9, scale = e4m3(amax / 6) (IEEE division,
round-to-nearest-even, saturating at 448), y = x / scale (IEEE division), E2M1 round-to-nearest-even with
the sign of y kept, dequantized value E2M1 * scale. Every gate is bit-exact (bf16 bit patterns, packed
bytes, scale bytes).

`op.CELLS` holds the dense, layout and small-pool calls. This file keeps what a cell cannot carry: the
tie, floor and saturation properties with their deliberately-wrong controls, in place on the input,
CUDA-graph replay, the op's own rejections, the engine-size pools (far end at B_max, view-size
invariance, every row count 1..16384, every graph batch), the cache managers that size those pools,
live multi-request steps and the compiled-variant map.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.nvfp4_fake_quantize import (  # noqa: E501
    nvfp4_fake_quantize as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "nvfp4_fake_quantize requires a CUDA device"

DEV = "cuda"
W = 512  # DeepSeek-V4.1 compressed-KV latent width (head_dim)
BLOCK = 16
FLOOR = 6 * 2.0**-9  # reference amax floor for E4M3-scaled FP4 (kernel.py fp4_quant_kernel)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitude of code 0..7; bit 3 is the sign
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
NUM_SLOTS = 4096  # pool rows: plainly larger than any batch below
_IDX = {"int32": torch.int32, "int64": torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _ref_rows(x: torch.Tensor):
    """(dequantized bf16 [rows, W], packed uint8 [rows, W/2], scale bytes uint8 [rows, W/16]) of every row
    of `x`: `op.reference` of the slot-r call into fresh dense buffers."""
    n, w = x.reshape(-1, x.shape[-1]).shape
    deq = op.reference(x, torch.empty(n, w, dtype=torch.bfloat16, device=x.device))
    packed, s_bytes = op.reference(
        x,
        torch.empty(n, w // 2, dtype=torch.uint8, device=x.device),
        None,
        torch.empty(n, w // BLOCK, dtype=torch.uint8, device=x.device),
    )
    return deq, packed, s_bytes


def _control(
    x: torch.Tensor, *, floor=FLOOR, pow2_scale=False, ties_toward_zero=False, reciprocal=False
) -> torch.Tensor:
    """Dequantized bf16 values of a deliberately-wrong variant of the recipe.

    The keyword variants: a power-of-two (UE8M0-style) scale, no amax floor (a zero scale reads back as
    zero), exact midpoints rounded toward zero, and x * (1/scale) instead of x / scale.
    """
    xf = x.float().unflatten(-1, (-1, BLOCK))
    amax = xf.abs().amax(-1, keepdim=True)
    if floor is not None:
        amax = amax.clamp_min(floor)
    if pow2_scale:
        u = (amax / 6.0).contiguous().view(torch.int32)
        e = ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).to(torch.int32)
        s = (e << 23).view(torch.float32)
    else:
        s = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    y = xf * (1.0 / s) if reciprocal else xf / s
    y = torch.where(s == 0, xf * 0.0, y).clamp(-6.0, 6.0)
    a = y.abs()
    if ties_toward_zero:
        idx = (a.unsqueeze(-1) > torch.tensor(MIDPOINTS, device=x.device)).sum(-1)
        r = torch.tensor(E2M1, device=x.device)[idx]
    else:  # torch.round is round-half-to-even
        r = torch.where(
            a < 2,
            torch.round(a * 2) / 2,
            torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
        )
    return (torch.copysign(r, y) * s).flatten(-2).to(torch.bfloat16)


def _decode(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Native pair -> bf16 via an E2M1 lookup table (independent of the op)."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=packed.device)
    v = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1).flatten(-2)
    s = scale.view(torch.float8_e4m3fn).float().repeat_interleave(BLOCK, -1)
    return (v * s).to(torch.bfloat16)


def _run_core(x):
    out = torch.full(x.shape, float("nan"), dtype=torch.bfloat16, device=DEV)
    return op(x, out)


def _run_native(x):
    n = x.shape[0]
    pk = torch.full((n, x.shape[-1] // 2), 0xA5, dtype=torch.uint8, device=DEV)
    sc = torch.full((n, x.shape[-1] // BLOCK), 0xA5, dtype=torch.uint8, device=DEV)
    return op(x, pk, None, sc)


def _assert_bitexact(x):
    """Both forms of the slot-r call equal the reference bitwise; the native pair decodes to the core output."""
    deq, packed, s_bytes = _ref_rows(x)
    core = _run_core(x)
    op.compare(core, deq)
    pk, sc = _run_native(x)
    op.compare((pk, sc), (packed, s_bytes))
    assert torch.equal(_bits(_decode(pk, sc)), _bits(core)), (
        "native pair does not decode to the core output"
    )
    return core


def _random_blocks(t, width=W, lo=-16, hi=6, seed=0):
    """bf16 [t, width], each 16-block scaled by 2^k, k uniform in [lo, hi] (the low end is below the floor)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    mag = torch.randint(lo, hi + 1, (t, width // BLOCK, 1), device=DEV, generator=g).float()
    x = torch.randn(t, width // BLOCK, BLOCK, device=DEV, generator=g) * torch.exp2(mag)
    return x.view(t, width).to(torch.bfloat16)


def _tie_blocks():
    """For every positive finite E4M3 scale s: a block [6s, +-t*s for t in MIDPOINTS, 0.1s], plus the same
    blocks one bf16 ulp above and below every midpoint entry. 6s and t*s are exact in bf16, so the scale
    comes out as exactly s and y = t is an exact midpoint."""
    s = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()  # 2^-9 .. 448
    pattern = torch.tensor([6.0, *MIDPOINTS, *(-m for m in MIDPOINTS), 0.1])
    ties = (s[:, None] * pattern[None, :]).to(torch.bfloat16)  # [126, 16]
    b = ties.view(torch.int16)
    up, dn = b.clone(), b.clone()
    up[:, 1:15] += 1
    dn[:, 1:15] -= 1
    blocks = torch.cat([ties, up.view(torch.bfloat16), dn.view(torch.bfloat16)], 0)  # [378, 16]
    pad = (-blocks.shape[0]) % (W // BLOCK)
    blocks = torch.cat([blocks, torch.zeros(pad, BLOCK, dtype=torch.bfloat16)], 0)
    return blocks.view(-1, W).to(DEV)


def _floor_blocks():
    """Blocks with amax below, at and above the floor, all-zero, all -0.0, tiny negatives."""
    g = torch.Generator().manual_seed(7)
    rows = []
    for a in (
        0.0,
        2.0**-40,
        2.0**-20,
        6 * 2.0**-12,
        6 * 2.0**-11,
        6 * 2.0**-10,
        9 * 2.0**-10,
        6 * 2.0**-9 * 0.75,
        6 * 2.0**-9 * 0.99,
        6 * 2.0**-9,
        6 * 2.0**-9 * 1.01,
        6 * 2.0**-8,
    ):
        blk = (torch.rand(32, BLOCK, generator=g) * 2 - 1) * a
        blk[:, 0] = a
        rows.append(blk)
    rows.append(torch.zeros(32, BLOCK))
    rows.append(-torch.zeros(32, BLOCK))
    rows.append(-torch.rand(32, BLOCK, generator=g) * 2.0**-14)
    mixed = torch.rand(32, BLOCK, generator=g) * 0.01 - 0.005
    mixed[:, 0] = 1.0  # scale 1/6 -> 0.25*s = 0.0417: every other element rounds to +-0
    rows.append(mixed)
    return torch.cat(rows, 0).to(torch.bfloat16).view(-1, W).to(DEV)


def _every_bf16_amax():
    """One block per positive finite bf16 value a (0x0001..0x7F7F), block = a * fixed pattern (amax = a)."""
    a = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    pattern = torch.tensor(
        [
            1.0,
            -0.3,
            0.77,
            -0.51,
            0.13,
            0.9,
            -0.66,
            0.2,
            -0.99,
            0.45,
            0.06,
            -0.12,
            0.35,
            -0.8,
            0.58,
            -0.25,
        ]
    )
    blocks = (a[:, None] * pattern[None, :]).to(torch.bfloat16)
    pad = (-blocks.shape[0]) % (W // BLOCK)
    blocks = torch.cat([blocks, blocks[:pad]], 0)
    return blocks.view(-1, W).to(DEV), a.to(DEV)


def _slot_map(t, idx_dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = torch.randperm(NUM_SLOTS, device=DEV, generator=g)[:t].to(torch.int64)
    slots[3] = -1
    slots[7] = NUM_SLOTS + 5
    slots[11] = -(2**40) if idx_dtype == torch.int64 else -(2**30)
    return slots.to(idx_dtype)


def _build_x(spec):
    kind = spec["kind"]
    if kind == "random":
        width = spec["width"]
        return _random_blocks(spec["t"], width=width, seed=spec["t"] if width == W else width)
    if kind == "ties":
        return _tie_blocks()
    if kind == "floor":
        return _floor_blocks()
    if kind == "every_amax":
        return _every_bf16_amax()[0]
    if kind == "leading_dims":
        return _random_blocks(96, seed=3).view(96, 1, W)
    if kind == "row_stride":
        wide = torch.randn(96, W + 128, device=DEV).to(torch.bfloat16)
        wide[:, :W] = _random_blocks(96, seed=3)
        return wide[:, :W]
    if kind == "bf16_pool":
        return _random_blocks(37, seed=5)
    if kind == "native_pool":
        return _random_blocks(37, seed=9)
    assert kind == "zero_rows", kind
    return torch.empty(0, W, dtype=torch.bfloat16, device=DEV)


def _build(spec):
    """(x, out, slot_mapping, out_scale) for a cell: sentinel-filled buffers, so `compare` also proves
    every element the op must not write was left alone."""
    kind, layout = spec["kind"], spec.get("layout")
    x = _build_x(spec)
    rows, width = x.reshape(-1, x.shape[-1]).shape
    if kind == "bf16_pool":
        shape = {
            "flat": (NUM_SLOTS, W),
            "paged64": (NUM_SLOTS // 64, 64, W),
            "paged128": (NUM_SLOTS // 128, 128, W),
        }[layout]
        out = torch.full(shape, -7.0, dtype=torch.bfloat16, device=DEV)
        return x, out, _slot_map(rows, _IDX[spec["idx_dtype"]], seed=len(layout)), None
    if kind == "native_pool":
        slots = _slot_map(rows, _IDX[spec["idx_dtype"]], seed=13)
        if layout == "split":
            data = torch.full((NUM_SLOTS, W // 2), 0x5A, dtype=torch.uint8, device=DEV)
            scale = torch.full((NUM_SLOTS, W // BLOCK), 0x5A, dtype=torch.uint8, device=DEV)
            return x, data, slots, scale.view(torch.float8_e4m3fn)
        pages = (NUM_SLOTS,) if layout == "row288" else (NUM_SLOTS // 128, 128)
        buf = torch.full(pages + (W // 2 + W // BLOCK,), 0x5A, dtype=torch.uint8, device=DEV)
        assert buf.view(NUM_SLOTS, -1).stride(0) == 288
        return x, buf[..., : W // 2], slots, buf[..., W // 2 :]
    if kind == "zero_rows":
        out = torch.full((8, W), 3.0, dtype=torch.bfloat16, device=DEV)
        return x, out, torch.empty(0, dtype=torch.int32, device=DEV), None
    if spec["form"] == "core":
        return (
            x,
            torch.full((rows, width), float("nan"), dtype=torch.bfloat16, device=DEV),
            None,
            None,
        )
    pk = torch.full((rows, width // 2), 0xA5, dtype=torch.uint8, device=DEV)
    sc = torch.full((rows, width // BLOCK), 0xA5, dtype=torch.uint8, device=DEV)
    return x, pk, None, sc


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, out, slots, out_scale = _build(cell.spec)
    expected = op.reference(x, out, slots, out_scale)
    result = op(x, out, slots, out_scale)
    if out_scale is None:
        assert result is out, "the core form writes in place and returns out"
    else:
        assert result[0] is out and result[1] is out_scale, (
            "the native form returns (out, out_scale)"
        )
    op.compare(result, expected)
    rows = x.reshape(-1, x.shape[-1]).shape[0]
    if slots is None and out_scale is not None:
        assert torch.equal(_bits(_decode(out, out_scale)), _bits(_ref_rows(x)[0])), (
            "native pair does not decode to the core values"
        )
    if slots is not None and rows:
        s64 = slots.to(torch.int64)
        assert int(((s64 >= 0) & (s64 < NUM_SLOTS)).sum()) == rows - 3, (
            "three slots are out of range"
        )


def test_exact_ties_round_to_even():
    x = _tie_blocks()
    _, _, s_bytes = _ref_rows(x)
    # the construction really produces exact midpoints (126 scales x 14 per block)
    s = s_bytes.view(torch.float8_e4m3fn).float().repeat_interleave(BLOCK, -1)
    y = (x.float() / s).abs()
    n_ties = int((y.unsqueeze(-1) == torch.tensor(MIDPOINTS, device=DEV)).any(-1).sum())
    assert n_ties == 126 * 14, n_ties
    core = _assert_bitexact(x)
    # up-ties (0.75, 1.75, 3.5) go up, down-ties (0.25, 1.25, 2.5, 5) go down
    tie_rows = core.float().view(-1, BLOCK)[:126]
    s_rows = s.view(-1, BLOCK)[:126, :1]
    got = (tie_rows[:, 1:8] / s_rows).abs()
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], device=DEV).expand_as(got)
    assert torch.equal(got, want)
    # controls: ties toward zero, and a reciprocal multiply, both miss the reference here
    tz = _control(x, ties_toward_zero=True)
    rc = _control(x, reciprocal=True)
    n_tz = int((_bits(tz) != _bits(core)).sum())
    n_rc = int((_bits(rc) != _bits(core)).sum())
    print(
        f"ties: {n_ties} exact midpoints; control ties-toward-zero differs on {n_tz}, reciprocal on {n_rc}"
    )
    assert n_tz == 126 * 6  # +-0.75, +-1.75, +-3.5 per scale
    assert n_rc > 0


def test_floor_region():
    x = _floor_blocks()
    core = _assert_bitexact(x)
    _, sc = _run_native(x)
    amax = x.float().view(-1, BLOCK).abs().amax(-1).view(sc.shape)
    below = amax <= FLOOR
    # every block at or below the floor gets the floor scale 6*2^-9 / 6 = 2^-9 (E4M3 byte 0x01)
    assert bool((sc[below] == 0x01).all())
    # -0.0 inputs and negatives that round to zero keep the sign (code 0x8, value -0.0)
    neg_zero = _bits(core) == -32768
    assert int(neg_zero.sum()) > 0
    assert bool((torch.signbit(x) == torch.signbit(core))[core == 0].all())
    nf = _control(x, floor=None)
    n_nf = int((nf.float() != core.float()).sum())
    print(
        f"floor: {int(below.sum())} blocks at/below the floor; control no-floor differs on {n_nf} values"
    )
    assert n_nf > 0


def test_every_bf16_amax():
    x, a = _every_bf16_amax()
    core = _assert_bitexact(x)
    _, sc = _run_native(x)
    sc = sc.view(-1)[: a.numel()]
    # the scale saturates at 448 (byte 0x7E) once amax / 6 rounds past it, and never becomes NaN (0x7F)
    assert not bool((sc == 0x7F).any())
    assert bool((sc[a >= 2688.0] == 0x7E).all())
    p2 = _control(x, pow2_scale=True)
    n_p2 = int((p2.float() != core.float()).sum())
    print(
        f"every bf16 amax: {a.numel()} blocks, {int((sc == 0x7E).sum())} saturated; "
        f"control power-of-two scale differs on {n_p2} values"
    )
    assert n_p2 > 0


def test_controls_fail_on_random_data():
    x = torch.cat([_random_blocks(1024, seed=11), _tie_blocks(), _floor_blocks()], 0)
    core = _assert_bitexact(x)
    total = core.numel()
    assert torch.equal(_bits(_control(x)), _bits(core)), (
        "the control recipe without a variant is the op"
    )
    for name, kw in [
        ("power-of-two scale", dict(pow2_scale=True)),
        ("no floor", dict(floor=None)),
        ("ties toward zero", dict(ties_toward_zero=True)),
        ("reciprocal", dict(reciprocal=True)),
    ]:
        wrong = _control(x, **kw)
        n = int((_bits(wrong) != _bits(core)).sum())
        print(f"control {name}: {n} of {total} elements differ")
        assert n > 0, name


def test_in_place_on_input():
    """`out` may be `x` itself in the slot-r form: each program reads its whole row before writing it."""
    x = _random_blocks(256, seed=17)
    y = x.clone()
    expected = op.reference(y, y)
    assert op(y, y) is y
    op.compare(y, expected)


def test_cuda_graph_replay():
    t = 64
    x = torch.empty(t, W, dtype=torch.bfloat16, device=DEV)
    pool = torch.zeros(NUM_SLOTS, W, dtype=torch.bfloat16, device=DEV)
    slots = torch.empty(t, dtype=torch.int32, device=DEV)
    x.copy_(_random_blocks(t, seed=1))
    slots.copy_(_slot_map(t, torch.int32, seed=1))
    op(x, pool, slots)  # compile outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        op(x, pool, slots)
    for seed in (2, 3):
        x.copy_(_random_blocks(t, seed=seed))
        slots.copy_(_slot_map(t, torch.int32, seed=seed))
        g.replay()
        ref, _, _ = _ref_rows(x)
        s64 = slots.to(torch.int64)
        valid = (s64 >= 0) & (s64 < NUM_SLOTS)
        assert torch.equal(_bits(pool[s64[valid]]), _bits(ref[valid]))


@pytest.mark.parametrize(
    "case, match",
    [
        ("fp32_input", "x must be bfloat16"),
        ("width_not_16", "positive multiple of 16"),
        ("strided_last_dim", "unit stride along its last dim"),
        ("native_without_scale", "needs out_scale"),
        ("scale_wrong_shape", "out_scale must be"),
        ("bf16_out_with_scale", "out_scale is only written"),
        ("out_wrong_width", "out's last dim must be"),
        ("slot_wrong_length", "slot_mapping must be"),
        ("slot_float", "slot_mapping must be int32 or int64"),
        ("too_few_slots", "without slot_mapping out needs"),
        ("fp16_out", "out must be bfloat16 or uint8"),
    ],
)
def test_rejections(case, match):
    x = _random_blocks(8, seed=0)
    out = torch.empty(8, W, dtype=torch.bfloat16, device=DEV)
    pk = torch.empty(8, W // 2, dtype=torch.uint8, device=DEV)
    sc = torch.empty(8, W // BLOCK, dtype=torch.uint8, device=DEV)
    args = {
        "fp32_input": (x.float(), out),
        "width_not_16": (x[:, :24], torch.empty(8, 24, dtype=torch.bfloat16, device=DEV)),
        "strided_last_dim": (torch.empty(W, 8, dtype=torch.bfloat16, device=DEV).t(), out),
        "native_without_scale": (x, pk),
        "scale_wrong_shape": (x, pk, None, sc[:, :16]),
        "bf16_out_with_scale": (x, out, None, sc),
        "out_wrong_width": (x, pk.view(8, -1)[:, :128]),
        "slot_wrong_length": (x, out, torch.zeros(7, dtype=torch.int32, device=DEV)),
        "slot_float": (x, out, torch.zeros(8, dtype=torch.float32, device=DEV)),
        "too_few_slots": (x, out[:4]),
        "fp16_out": (x, out.half()),
    }[case]
    with pytest.raises(ValueError, match=match):
        op(*args)


# ─── the largest COMPRESS views an engine can allocate ────────────────────────

# The per-owner COMPRESS views (ratio 2 [B, 64, 512] for owners 2 / 8 / 14,
# ratio 1 [B, 128, 512] for owner 20, bf16) are sized by the V4.1
# DeepseekV4CacheManager from the per-rank KV quota, which an engine derives as
# kv_cache_fraction (0.5 at the gates) x the device's FREE memory after weights
# and activations. Free memory never exceeds total memory, so a manager built at
# 0.5 x the device's TOTAL memory (148,510,474,240 bytes on this GB300)
# allocates the largest views an engine of that configuration can have. At the
# gate settings (max_seq_len 16384, max_num_tokens 8192, max_batch_size {8, 32},
# DSpark or not; test_largest_engine_views_from_the_cache_manager) the largest
# are ratio 2 [862719, 64, 512] and ratio 1 [287573, 128, 512] (GATES_COMPRESS).
# Over the engine envelope (max_num_tokens 2048..16384, max_batch_size up
# to 256, max_seq_len up to 1,048,576, DSpark or not) the unsplit manager's
# largest are ratio 2 [1270272, 64, 512] and ratio 1 [423424, 128, 512], both at
# (max_num_tokens 2048, max_batch_size 48, no DSpark, max_seq_len 1,048,576)
# (UNSPLIT_COMPRESS, the pool scan;
# test_envelope_largest_compress_views_from_the_cache_manager re-measures them). A PP2
# stage manager holds only its 20 layers' pools, so its views are larger: stage 0
# alone (the ratio-2 sources) reaches [2118144, 64, 512] (135,561,216 slots,
# 6.9e10 elements, 138.8 GB) and stage 1 alone (the ratio-1 source) [1059584,
# 128, 512] (135,626,752 slots, 6.9e10 elements, 138.9 GB), both at (2048, 96, 0,
# 1,048,576) (the PP2 stage scan;
# test_envelope_pp2_stage_compress_views_from_the_cache_manager re-measures them).
# Those stage-alone maxima are ENGINE_COMPRESS. test_pool_size_invariance shows
# the op's result does not depend on B, so the far-end cases below run at that
# B_max. Past slot 4,194,304 every element offset
# exceeds 2^31, past 8,388,608 it exceeds 2^32 and past 67,108,864 it exceeds
# 2^35, so any 32-bit offset in the kernel's address arithmetic would land a
# row elsewhere (or fault). Each pool is allocated at the exact shape with one
# guard page past its end, every element set to a NaN bit pattern the op never
# writes for a finite row; after the call the named slots must hold the
# reference and every other element must still hold it.
GATES_COMPRESS = {"ratio2": (862719, 64), "ratio1": (287573, 128)}
UNSPLIT_COMPRESS = {"ratio2": (1270272, 64), "ratio1": (423424, 128)}
ENGINE_COMPRESS = {"ratio2": (2118144, 64), "ratio1": (1059584, 128)}
SENTINEL = 0x7FA5  # bf16 NaN bit pattern (as int16)


def _hazard_slots(num_slots, width, page_size):
    """Distinct slots on both sides of every 32-bit offset hazard of a [slots, width] bf16 pool, and both ends.

    Element offsets 2^30 (2^31 bytes), 2^31 (2^32 bytes), 2^32, 2^33, 2^34, 2^35 and 2^36 (where inside the
    pool), the first page, the last page and the last slot.
    """
    cands = {0, 1, page_size - 1, page_size}
    for limit in (2**30, 2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // width
        cands |= {s - 1, s}
    cands |= {num_slots - page_size - 1, num_slots - page_size, num_slots - 2, num_slots - 1}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _engine_pool(pages, page_size):
    """(buffer with one guard page, [pages, page_size, W] bf16 pool view), every element SENTINEL."""
    buf = torch.empty(pages + 1, page_size, W, dtype=torch.bfloat16, device=DEV)
    buf.view(torch.int16).fill_(SENTINEL)
    return buf, buf[:pages]


def _check_engine_pool(buf, pool, slots, ref):
    """Named slots hold ref bitwise; every other element of the pool and the guard page is still SENTINEL.

    Restores the named slots to SENTINEL, so the pool can be reused.
    """
    num_slots = pool.shape[0] * pool.shape[1]
    flat = pool.view(num_slots, W)
    s64 = slots.to(torch.int64)
    valid = (s64 >= 0) & (s64 < num_slots)
    idx = s64[valid]
    got = flat[idx]
    n_bad = int((_bits(got) != _bits(ref[valid])).sum())
    assert n_bad == 0, f"{n_bad} of {got.numel()} elements in the named slots differ bitwise"
    flat.view(torch.int16)[idx] = SENTINEL
    whole = buf.view(torch.int16).view(-1)
    step = 1 << 30
    n_out = sum(int((whole[i : i + step] != SENTINEL).sum()) for i in range(0, whole.numel(), step))
    assert n_out == 0, f"{n_out} elements outside the named slots were written"


def _free_engine_pool():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_engine_pool_far_end(ratio, idx_dtype):
    """Decode-like: one row per 32-bit hazard boundary and at both ends of the engine-size pool, plus skips."""
    pages, page_size = ENGINE_COMPRESS[ratio]
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    hazard = _hazard_slots(num_slots, W, page_size)
    assert hazard[-1] == num_slots - 1 and 2**34 // W in hazard
    # -1 and the first two slots past the end (which would land in the guard page) are skipped.
    slots = torch.tensor(
        hazard + [-1, num_slots, num_slots + page_size - 1], dtype=torch.int64, device=DEV
    )
    x = _random_blocks(slots.numel(), seed=31)
    op(x, pool, slots.to(idx_dtype))
    _check_engine_pool(buf, pool, slots, _ref_rows(x)[0])
    del buf, pool
    _free_engine_pool()


@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_engine_pool_prefill_spread(ratio):
    """Prefill-like, n = 16384 (the envelope's max_num_tokens): the last page written in order, every hazard
    slot, the rest spread over the pool."""
    t = _MAX_ROWS
    pages, page_size = ENGINE_COMPRESS[ratio]
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    last_page = torch.arange(num_slots - page_size, num_slots, device=DEV)
    hazard = torch.tensor(_hazard_slots(num_slots, W, page_size), device=DEV)
    fixed = torch.unique(torch.cat([last_page, hazard]))
    g = torch.Generator(device=DEV).manual_seed(8192)
    perm = torch.randperm(num_slots, device=DEV, generator=g)
    rest = perm[~torch.isin(perm, fixed)][: t - fixed.numel()]
    slots = torch.cat([last_page, fixed[~torch.isin(fixed, last_page)], rest]).to(torch.int64)
    assert slots.numel() == t and torch.unique(slots).numel() == t
    slots[torch.randperm(t, device=DEV, generator=g)[: t // 16]] = -1
    x = _sweep_rows(t, seed=8192)
    op(x, pool, slots.to(torch.int32))
    _check_engine_pool(buf, pool, slots, _ref_rows(x)[0])
    del buf, pool
    _free_engine_pool()


def test_engine_pool_cuda_graph_far_end():
    """Captured on the ratio-2 engine-size pool after a warm-up call, replayed with far-end and boundary slots."""
    t = 32
    pages, page_size = ENGINE_COMPRESS["ratio2"]
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    hazard = _hazard_slots(num_slots, W, page_size)
    x = torch.empty(t, W, dtype=torch.bfloat16, device=DEV)
    slots = torch.empty(t, dtype=torch.int32, device=DEV)
    x.copy_(_random_blocks(t, seed=40))
    slots.copy_(torch.arange(t, device=DEV))
    op(x, pool, slots)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, pool, slots)
    buf.view(torch.int16).fill_(SENTINEL)

    g = torch.Generator(device=DEV).manual_seed(41)
    hz = torch.tensor(hazard, device=DEV)
    near_end = num_slots - 1 - torch.arange(4 * page_size, device=DEV)
    near_end = near_end[~torch.isin(near_end, hz)]
    for seed in (42, 43):
        pick = torch.randperm(near_end.numel(), device=DEV, generator=g)[: t - len(hazard) - 1]
        new_slots = torch.cat([hz, torch.tensor([-1], device=DEV), near_end[pick]]).to(torch.int64)
        assert torch.unique(new_slots).numel() == t
        new_x = _random_blocks(t, seed=seed)
        x.copy_(new_x)
        slots.copy_(new_slots)
        graph.replay()
        torch.cuda.synchronize()
        _check_engine_pool(buf, pool, new_slots, _ref_rows(new_x)[0])
    del graph, buf, pool
    _free_engine_pool()


def test_engine_pool_harness_sees_a_wrapped_offset():
    """Control: the far-end check fails when rows land where a 32-bit (mod 2^32) element offset would put them.

    The op is handed the wrapped slots and the checker the true ones; both the named-slot comparison and the
    outside-the-slots sentinel scan must see it.
    """
    pages, page_size = ENGINE_COMPRESS["ratio2"]
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    true = torch.tensor(
        [s for s in _hazard_slots(num_slots, W, page_size) if s * W >= 2**32],
        dtype=torch.int64,
        device=DEV,
    )
    wrapped = (true * W % 2**32) // W
    assert true.numel() >= 5 and not torch.isin(wrapped, true).any()
    x = _random_blocks(true.numel(), seed=51)
    ref = _ref_rows(x)[0]
    op(x, pool, wrapped)
    with pytest.raises(AssertionError, match="elements in the named slots differ bitwise"):
        _check_engine_pool(buf, pool, true, ref)
    flat = pool.view(num_slots, W)
    flat.view(torch.int16)[true] = SENTINEL
    with pytest.raises(AssertionError, match="outside the named slots were written"):
        _check_engine_pool(buf, pool, torch.tensor([], dtype=torch.int64, device=DEV), ref[:0])
    del buf, pool, flat
    _free_engine_pool()


# ─── every row count 1..16384 in the core target's paged-scatter call ────────
#
# The DeepSeek-V4.1 core target writes each KV source's compressed entries with
# nvfp4_fake_quantize(latent [n, 512], compress_view, slots[:n]): ratio-2 owners
# into a [pages, 64, 512] view, the ratio-1 owner into a [pages, 128, 512] view,
# n = the step's compressed-entry count (T for ratio 1; up to T / 2 plus the
# reserved decode rows for ratio 2; at most the engine's max_num_tokens, 16384
# in the engine envelope; the KV sources are layers 2, 8, 14 and 20, so the
# decoder bounded-replay pass of layers 21-39 makes no such call),
# and slots a prefix view of the metadata's int64 [max_num_tokens]
# v41_compress_slot_mapping[r] (-1 = no slot). Rows reach the kernel only as the
# size of grid axis 0: program r loads row r and slot_mapping[r] and writes one
# slot; no kernel argument carries n.

import tensorrt_llm._torch.custom_ops.nvfp4_fake_quantize as _nv_op  # noqa: E402 (the Triton kernel object, for the variant probe)

_MAX_ROWS = 16384
_SWEEP_SLOTS = 20480  # plainly larger than the 16384 rows swept
_SWEEP_GEOMETRY = {"ratio2": 64, "ratio1": 128}  # page size of each COMPRESS view


def _jit_variants(fn) -> int:
    """Number of compiled variants Triton holds for a @triton.jit function, over all devices."""
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


def _sweep_rows(t: int, seed: int) -> torch.Tensor:
    """Random 16-blocks over 2^-16 .. 2^6, with the tie set and the floor set spliced in near the top."""
    x = _random_blocks(t, seed=seed)
    ties, floors = _tie_blocks(), _floor_blocks()
    x[2 : 2 + ties.shape[0]] = ties
    x[40 : 40 + floors.shape[0]] = floors
    return x


def _sweep_slots(t: int, num_slots: int, seed: int) -> torch.Tensor:
    """A prefix-able int64 [t] view like v41_compress_slot_mapping[r]: distinct random slots, 1/16 of them -1."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = torch.randperm(num_slots, device=DEV, generator=g)[:t].to(torch.int64)
    slots[torch.randperm(t, device=DEV, generator=g)[: t // 16]] = -1
    slots[1] = -1  # a skipped row inside the smallest prefixes too
    return slots


@pytest.mark.parametrize("ratio", list(_SWEEP_GEOMETRY))
def test_every_row_count_1_to_16384_in_the_paged_scatter_form(ratio):
    """Every prefix (x[:r], slots[:r]), r = 1..16384, writes exactly the first r rows of the 16384-row call.

    The 16384-row call is held bitwise to the native reference and to the
    sentinel scan. Then, into the same sentinel-filled pool, the op runs at
    every r in increasing order. After each call the slots named by the live
    entries of slots[:r] hold the reference rows bit for bit, and the number of
    pool elements that are not the sentinel is exactly 512 per live row: a
    skipped row, a row written to another slot, or a write past row r - 1 all
    change one of the two counts. The mismatch counts accumulate on the device;
    one sync at the end. The probe: rows are only grid axis 0, so the sweep must
    compile no Triton variant beyond the one the 16384-row call compiled.
    """
    t = _MAX_ROWS
    page_size = _SWEEP_GEOMETRY[ratio]
    buf, pool = _engine_pool(_SWEEP_SLOTS // page_size, page_size)
    flat = pool.view(_SWEEP_SLOTS, W).view(torch.int16)
    whole = buf.view(torch.int16)
    slots = _sweep_slots(t, _SWEEP_SLOTS, seed=8193 + page_size)
    x = _sweep_rows(t, seed=8193 + page_size)
    ref = _ref_rows(x)[0]

    kernel = _nv_op._nvfp4_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    op(x, pool, slots)
    n1 = _jit_variants(kernel)
    _check_engine_pool(
        buf, pool, slots, ref
    )  # the N_max call against the reference; restores the sentinel

    live = slots >= 0
    live_pos = torch.nonzero(live).squeeze(1)
    live_slots = slots[live_pos]
    ref_live = _bits(ref[live_pos])
    cnt = torch.cumsum(live.to(torch.int64), 0).tolist()
    bad_rows = torch.zeros((), dtype=torch.int64, device=DEV)
    bad_count = torch.zeros((), dtype=torch.int64, device=DEV)
    for r in range(1, t + 1):
        op(x[:r], pool, slots[:r])
        k = cnt[r - 1]
        bad_rows += (flat[live_slots[:k]] != ref_live[:k]).sum()
        bad_count += ((whole != SENTINEL).sum() - k * W).abs()
    n2 = _jit_variants(kernel)
    print(
        f"\n[{ratio} rows 1..{t}, page {page_size}] live rows {cnt[-1]}; Triton variants: {n0} before the "
        f"{t}-row call, {n1} after, {n2} after the sweep; element mismatches {int(bad_rows)}, "
        f"non-sentinel count error {int(bad_count)}"
    )
    assert int(bad_rows) == 0, (
        f"{int(bad_rows)} elements of the named slots differ from the {t}-row result"
    )
    assert int(bad_count) == 0, (
        f"non-sentinel element count off by {int(bad_count)} summed over the sweep"
    )
    assert n1 - n0 <= 1 and n2 == n1, f"row counts compiled new variants: {n0} -> {n1} -> {n2}"
    del buf, pool, flat, whole
    _free_engine_pool()


def test_row_sweep_harness_sees_a_row_or_slot_shift():
    """Control for the sweep: its checks reject a read one row off and a write one slot off, at every r tried."""
    page_size = 64
    buf, pool = _engine_pool(_SWEEP_SLOTS // page_size, page_size)
    flat = pool.view(_SWEEP_SLOTS, W).view(torch.int16)
    whole = buf.view(torch.int16)
    g = torch.Generator(device=DEV).manual_seed(66)
    slots = torch.randperm(_SWEEP_SLOTS, device=DEV, generator=g)[:66].to(
        torch.int64
    )  # no -1: every row lands
    x = _random_blocks(66, seed=66)
    ref = _bits(_ref_rows(x)[0])
    for r in (1, 2, 7, 63, 64):
        for name, xa, sa in (
            ("row +1", x[1 : r + 1], slots[:r]),
            ("slot +1", x[:r], slots[1 : r + 1]),
        ):
            whole.fill_(SENTINEL)
            op(xa, pool, sa)
            n_bad = int((flat[slots[:r]] != ref[:r]).sum())
            n_count = int((whole != SENTINEL).sum()) - r * W
            print(
                f"[control {name}] r={r}: {n_bad} / {r * W} elements differ, count error {n_count}"
            )
            assert n_bad > r * W // 2
    del buf, pool, flat, whole
    _free_engine_pool()


# ─── CUDA-graph capture at every graph batch the engines capture ─────────────
#
# The engine envelope captures every graph batch G of _ENVELOPE_GSET: every G
# 1..32 (padding off captures every batch up to max_batch_size) and TRT-LLM's
# padded list for cuda_graph_config max_batch_size 256 (1, 2, 4, 8..128 step 8,
# 192, 256). With DSpark the target-verify step is captured at 6 tokens per
# request. Rows per captured call: the ratio-1 owner writes one entry per token
# (n = G, and 6G for verify); a ratio-2 owner writes the runtime-reserved
# ceil(next_n / 2) rows per request (n = G, and 3G for verify), the reserved
# rows without a completed pair carrying slot -1.
_ENVELOPE_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
assert len(_ENVELOPE_GSET) == 46
_GRAPH_N = {
    "ratio1": tuple(
        sorted(set(_ENVELOPE_GSET) | {6 * g for g in _ENVELOPE_GSET})
    ),  # 82 row counts, 1..1536
    "ratio2": tuple(
        sorted(set(_ENVELOPE_GSET) | {3 * g for g in _ENVELOPE_GSET})
    ),  # 77 row counts, 1..768
}


def _far_end_slots(num_slots, page_size, n, g):
    """n distinct slots: the pool's last slot, then a random pick from the 32-bit hazard slots and the last
    4 pages (and, when n needs more, slots spread uniformly over the pool); for n >= 2 one of them (not the
    last slot) replaced by -1."""
    last = num_slots - 1
    hazard = torch.tensor(
        [s for s in _hazard_slots(num_slots, W, page_size) if s != last], device=DEV
    )
    near_end = last - 1 - torch.arange(4 * page_size - 1, device=DEV)
    near_end = near_end[~torch.isin(near_end, hazard)]
    rest = torch.cat([hazard, near_end])
    rest = rest[torch.randperm(rest.numel(), device=DEV, generator=g)]
    if rest.numel() < n - 1:
        spread = torch.randint(0, last, (4 * n,), device=DEV, generator=g).unique()
        spread = spread[~torch.isin(spread, rest)]
        rest = torch.cat([rest, spread[torch.randperm(spread.numel(), device=DEV, generator=g)]])
    slots = torch.cat([torch.tensor([last], device=DEV), rest[: n - 1]]).to(torch.int64)
    assert slots.numel() == n and torch.unique(slots).numel() == n
    if n >= 2:
        slots[1 + int(torch.randint(0, n - 1, (1,), device=DEV, generator=g))] = -1
    return slots


@pytest.mark.parametrize("ratio", ["ratio1", "ratio2"])
def test_cuda_graph_every_graph_batch_on_the_engine_compress_view(ratio):
    """Per n of that ratio's envelope capture set (ratio 1: G and 6G, 82 row counts to 1536; ratio 2: G and
    3G, 77 row counts to 768; G over _ENVELOPE_GSET): eager warm-up at that shape, capture, then 2 replays on new
    rows and new far-end slots.

    On the B_max bf16 COMPRESS view of that ratio. Every replay must write
    exactly the reference into the named slots and nothing else (sentinel scan
    of the whole pool plus a guard page), and equal bit for bit the same call
    run eagerly on the same inputs.
    """
    pages, page_size = ENGINE_COMPRESS[ratio]
    ns = _GRAPH_N[ratio]
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    flat = pool.view(num_slots, W)
    g = torch.Generator(device=DEV).manual_seed(700 + page_size)
    table = torch.full(
        (_MAX_ROWS,), -1, dtype=torch.int64, device=DEV
    )  # like v41_compress_slot_mapping[r]
    for n in ns:
        x = torch.empty(n, W, dtype=torch.bfloat16, device=DEV)
        slots = table[:n]
        x.copy_(_random_blocks(n, seed=700 + n))
        slots.copy_(_far_end_slots(num_slots, page_size, n, g))
        op(x, pool, slots)  # eager warm-up at this shape, as the engine's warm-up step runs it
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, pool, slots)
        buf.view(torch.int16).fill_(SENTINEL)
        for rep in (1, 2):
            new_x = _random_blocks(n, seed=700 + 1000 * rep + n)
            new_slots = _far_end_slots(num_slots, page_size, n, g)
            x.copy_(new_x)
            slots.copy_(new_slots)
            graph.replay()
            torch.cuda.synchronize()
            live = new_slots[new_slots >= 0]
            replayed = _bits(flat[live]).clone()
            ref = _ref_rows(new_x)[0]
            _check_engine_pool(buf, pool, new_slots, ref)
            op(x, pool, slots)  # the same call, eagerly, on the same inputs
            torch.cuda.synchronize()
            eager = _bits(flat[live]).clone()
            _check_engine_pool(buf, pool, new_slots, ref)
            assert torch.equal(replayed, eager), (
                f"n={n} replay {rep}: {int((replayed != eager).sum())} elements"
            )
        del graph
    print(
        f"\n[{ratio} graphs on [{pages}, {page_size}, 512]] n = {list(ns)}: every replay bitwise == eager == reference"
    )
    del buf, pool, flat
    _free_engine_pool()


# ─── any COMPRESS view B <= B_max ─────────────────────────────────────────────
#
# The engine picks the COMPRESS views' page counts from free memory, so no single
# dumped B is "the" engine's. Two tests make the coverage an upper bound:
# (1) the cache manager at the largest quota a gate-configured engine can reach
# gives B_max = ENGINE_COMPRESS (and the op is exact at the far end of the
# manager's own views), and (2) the same rows into the same slots give the same
# bits at B = the smallest view that holds them, sizes across the 2^31 / 2^32
# element-count lines, a mid size and B_max (views of one allocation), and in a
# separate exact-size allocation.

# (max_draft_len, max_batch_size) -> (ratio-2 pages of owners 2 / 8 / 14, ratio-1 pages of owner 20)
# at 0.5 x this GB300's total memory.
_BMAX_DEVICE_TOTAL = 297_020_948_480
_BMAX_COMPRESS_PAGES = {
    (5, 8): ((509184, 509183, 509182), 169728),
    (0, 8): ((527358, 527357, 527356), 175786),
    (5, 32): ((801024, 801023, 801022), 267008),
    (0, 32): ((862719, 862718, 862717), 287573),
}


def _pp_mapping(pp_stage):
    """The single-process mapping of a rank of PP2 stage `pp_stage` (tp 2, as the PP2 stage scan builds
    it; the V4.1 pools do not depend on tp), or of the unsplit manager when pp_stage is None."""
    from tensorrt_llm.mapping import Mapping

    if pp_stage is None:
        return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    return Mapping(world_size=4, rank=2 * pp_stage, tp_size=2, pp_size=2, gpus_per_node=4)


def _v41_manager(
    quota, max_draft_len, max_batch_size, max_num_tokens=8192, max_seq_len=None, pp_stage=None
):
    """A DeepSeek-V4.1 DeepseekV4CacheManager with bf16-container KV storage and the given per-rank KV quota,
    at the gate settings unless max_num_tokens / max_seq_len say otherwise (max_seq_len as the engine passes
    it: + 14 with DSpark). The storage is pinned: when TRTLLM_DSV41_KV_STORAGE is unset the manager stores the
    KV natively (the native section below builds those managers explicitly)."""
    import os
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )

    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=512,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )
    saved = os.environ.get("TRTLLM_DSV41_KV_STORAGE")
    os.environ["TRTLLM_DSV41_KV_STORAGE"] = "bf16"
    try:
        mgr = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                max_gpu_total_bytes=quota,
                host_cache_size=0,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=128,
            # DSpark (max_draft_len 5, overlap on) grows the manager's max_seq_len by 5 + 4 + 5.
            max_seq_len=max_seq_len or (16398 if max_draft_len else 16384),
            max_batch_size=max_batch_size,
            mapping=_pp_mapping(pp_stage),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1] * 43
            ),
            pretrained_config=config,
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
            max_cuda_graph_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
        )
    finally:
        if saved is None:
            os.environ.pop("TRTLLM_DSV41_KV_STORAGE", None)
        else:
            os.environ["TRTLLM_DSV41_KV_STORAGE"] = saved
    assert mgr.v41_layout.kv_storage == "bf16"
    return mgr


def test_largest_engine_views_from_the_cache_manager():
    """B_max from the manager itself, and the far-end hazard call on the manager's own B_max views.

    The quota is 0.5 x the device's total memory, the upper bound of kv_cache_fraction 0.5 x free memory.
    The largest configuration (max_batch_size 32, no DSpark) is built last and kept: on owner 2's and
    owner 14's ratio-2 views and owner 20's ratio-1 view (each filled with the sentinel first), the op
    writes rows to both sides of every 2^31..2^34 element-offset line, the first and the last page and
    the last slot; the view must hold the reference there and the sentinel everywhere else.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    quota = total // 2
    got = {}
    keys = list(_BMAX_COMPRESS_PAGES)  # the largest last
    for key in keys:
        mgr = _v41_manager(quota, *key)
        views = {o: mgr.get_buffers(o, AT.COMPRESS) for o in (2, 8, 14, 20)}
        for o, v in views.items():
            ps = 64 if o != 20 else 128
            assert tuple(v.shape[1:]) == (ps, W) and v.dtype == torch.bfloat16, (
                o,
                v.shape,
                v.dtype,
            )
            assert v.stride() == (ps * W, W, 1), (o, v.stride())
        got[key] = (tuple(views[o].shape[0] for o in (2, 8, 14)), views[20].shape[0])
        if key != keys[-1]:
            mgr.shutdown()
            del mgr, views
            _free_engine_pool()
    print(
        f"\n[quota {quota} B = 0.5 x device total] COMPRESS pages per (max_draft_len, max_batch_size): {got}"
    )
    assert got == _BMAX_COMPRESS_PAGES, got
    assert max(v[0][0] for v in got.values()) == GATES_COMPRESS["ratio2"][0]
    assert max(v[1] for v in got.values()) == GATES_COMPRESS["ratio1"][0]
    try:
        for o in (2, 14, 20):
            pool = views[o]
            page_size = pool.shape[1]
            num_slots = pool.shape[0] * page_size
            pool.view(torch.int16).fill_(SENTINEL)
            hazard = _hazard_slots(num_slots, W, page_size)
            assert hazard[-1] == num_slots - 1 and 2**34 // W in hazard
            slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device=DEV)
            x = _random_blocks(slots.numel(), seed=8620 + o)
            op(x, pool, slots)
            _check_engine_pool(pool, pool, slots, _ref_rows(x)[0])
            print(
                f"[manager's own owner-{o} view {tuple(pool.shape)}] {len(hazard)} far-end / hazard rows == "
                f"reference, rest of the view sentinel"
            )
    finally:
        mgr.shutdown()
        del mgr, views
        _free_engine_pool()


def _invariance_slots(n, page_size, seed):
    """(smallest page count holding n slots, int64 [n] prefix of a [8192] table of distinct slots in it):
    the last slot of that smallest view is named, 1/16 of the other rows -1."""
    small = -(-n // page_size)
    last = small * page_size - 1
    g = torch.Generator(device=DEV).manual_seed(seed)
    perm = torch.randperm(small * page_size, device=DEV, generator=g)
    pick = torch.cat([torch.tensor([last], device=DEV), perm[perm != last]])[:n]
    pick = pick[torch.randperm(n, device=DEV, generator=g)]
    table = torch.full(
        (_MAX_ROWS,), -1, dtype=torch.int64, device=DEV
    )  # like v41_compress_slot_mapping[r]
    slots = table[:n]
    slots.copy_(pick)
    skip = torch.randperm(n, device=DEV, generator=g)[: n // 16]
    slots[skip[slots[skip] != last]] = -1
    return small, slots


@pytest.mark.parametrize("n", [32, 8192, 16384])
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_pool_size_invariance(ratio, n):
    """The same n rows into the same slots: bitwise the same bits at every view size B, and = reference.

    B = the smallest view holding the slots, the page counts whose element count first passes 2^31, 2^32,
    2^33, 2^34, 2^35 and 2^36 (where below B_max), the gates' bound (GATES_COMPRESS), the unsplit
    manager's envelope bound (UNSPLIT_COMPRESS), a mid size B_max // 2 + 1 and the envelope B_max (a
    PP2 stage alone), each a view pool[:B] of one sentinel-filled allocation of B_max pages plus a
    guard page, then a separate allocation of exactly the smallest size. After every call the whole
    allocation (past B too) holds the sentinel outside the named slots.
    """
    pages, page_size = ENGINE_COMPRESS[ratio]
    small, slots = _invariance_slots(n, page_size, seed=2300 + n + page_size)
    x = _random_blocks(n, seed=2300 + n + page_size)
    ref = _ref_rows(x)[0]
    live = slots[slots >= 0]
    per_page = page_size * W
    sizes = sorted(
        {small, pages // 2 + 1, GATES_COMPRESS[ratio][0], UNSPLIT_COMPRESS[ratio][0], pages}
        | {
            lim // per_page + 1
            for lim in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)
            if lim // per_page + 1 < pages
        }
    )
    buf, _ = _engine_pool(pages, page_size)
    first = None
    for b in sizes:
        pool = buf[:b]
        op(x, pool, slots)
        torch.cuda.synchronize()
        got = _bits(pool.view(-1, W)[live]).clone()
        _check_engine_pool(buf, pool, slots, ref)
        first = got if first is None else first
        assert torch.equal(got, first), (
            f"B={b}: {int((got != first).sum())} elements differ from B={sizes[0]}"
        )
    del buf, pool
    _free_engine_pool()
    buf, pool = _engine_pool(small, page_size)
    op(x, pool, slots)
    torch.cuda.synchronize()
    got = _bits(pool.view(-1, W)[live]).clone()
    _check_engine_pool(buf, pool, slots, ref)
    assert torch.equal(got, first), "a separate exact-size view differs"
    print(
        f"\n[{ratio} n={n}] views B = {sizes} pages of [B, {page_size}, 512] (one allocation) and a separate "
        f"[{small}, {page_size}, 512]: {live.numel()} rows bitwise identical and == reference, nothing else written"
    )
    del buf, pool
    _free_engine_pool()


# ─── native storage on the live DeepSeek-V4.1 cache manager ──────────────────
#
# With TRTLLM_DSV41_KV_STORAGE=native the V4.1 DeepseekV4CacheManager keeps the
# compressed main KV natively: per ratio one uint8 buffer of 288-byte rows, the
# 256 packed E2M1 bytes then the 32 E4M3 bytes; the ratio-2 owners 2 / 8 / 14
# view it as [B, 64, 288] with bases one page apart (so B, B - 1, B - 2 pages),
# owner 20 (ratio 1) as [B, 128, 288]. get_v41_native_kv_views(o, COMPRESS)
# returns (codes [..., 256], scales [..., 32]), two uint8 views with the pool's
# strides. The native-storage target writes them with this op's native form:
# nvfp4_fake_quantize(latent [n, 512], codes, md.v41_compress_slot_mapping[r][:n], scales).
# The table is the manager's native COMPRESS pages at 0.5 x this GB300's total
# memory (the ceiling of kv_cache_fraction 0.5 x free memory) for every
# (max_num_tokens, max_batch_size, DSpark max_draft_len) the runs use or could
# use: the native gates run (2048, 8, 5) and (8192, 32, 5) and reach 1,954,512 /
# 651,504 pages; that cross-product's largest, (2048, 32, 0), is 2,606,019
# (ratio 2, 166.8 M rows, 48.0 GB) and 868,673 (ratio 1, 111.2 M rows, 32.0 GB)
# (GATES_NATIVE_COMPRESS). Over the engine envelope the largest native views
# are 3,886,647 / 1,295,549 pages at (2048, 128, 0, max_seq_len 1,048,576)
# (NATIVE_COMPRESS, the B_max the far-end, invariance and capture cases use).

import os  # noqa: E402
import random  # noqa: E402
from contextlib import contextmanager  # noqa: E402

_STORAGE_ENV = "TRTLLM_DSV41_KV_STORAGE"
NATIVE_ROW = 288  # 256 packed E2M1 bytes, then 32 E4M3 bytes
NATIVE_CODES = 256
# 0xFF is the E4M3 NaN: the op never writes it as a scale byte (scales are
# positive, saturating at 0x7E). A packed code byte can be 0xFF (two -6
# nibbles), so the checks compare the named rows first and then restore them.
NATIVE_SENTINEL = 0xFF
_NATIVE_COMPRESS_PAGES = {
    (2048, 8, 0): ((2_131_740, 2_131_739, 2_131_738), 710_580),
    (2048, 8, 5): ((1_954_512, 1_954_511, 1_954_510), 651_504),
    (2048, 32, 0): ((2_606_019, 2_606_018, 2_606_017), 868_673),
    (2048, 32, 5): ((2_315_604, 2_315_603, 2_315_602), 771_868),
    (8192, 8, 0): ((1_378_986, 1_378_985, 1_378_984), 459_662),
    (8192, 8, 5): ((1_302_528, 1_302_527, 1_302_526), 434_176),
    (8192, 32, 0): ((2_083_497, 2_083_496, 2_083_495), 694_499),
    (8192, 32, 5): ((1_895_082, 1_895_081, 1_895_080), 631_694),
}
_NATIVE_GATES = ((2048, 8, 5), (8192, 32, 5))
# ratio -> (B_max pages over max_batch_size 1..32 at max_num_tokens {2048, 8192}, page size, gates' largest)
GATES_NATIVE_COMPRESS = {"ratio2": (2_606_019, 64, 1_954_512), "ratio1": (868_673, 128, 651_504)}
# ratio -> (the unsplit manager's B_max pages over the engine envelope, page size, gates' largest): the
# pool scan's largest native COMPRESS views, both at (max_num_tokens 2048, max_batch_size 128, no DSpark,
# max_seq_len 1,048,576); re-measured by test_envelope_largest_compress_views_from_the_cache_manager.
UNSPLIT_NATIVE_COMPRESS = {
    "ratio2": (3_886_647, 64, 1_954_512),
    "ratio1": (1_295_549, 128, 651_504),
}
ENVELOPE_NATIVE_COMPRESS_KEY = (
    2048,
    128,
    0,
    1_048_576,
)  # (max_num_tokens, max_batch_size, max_draft_len, max_seq_len)
# PP2 (the unified target's stage split at layer 20): a stage manager holds only its 20 layers' pools. The
# PP2 stage scan's largest native COMPRESS views are stage 0 alone (ratio-2 owners 2 / 8 / 14) and stage 1
# alone (ratio-1 owner 20), both at (2048, 256, 0, 1,048,576)
# (the PP2 stage scan; the stage alone is
# the upper bound: in an engine the stages' token capacities are cut to the smaller one); re-measured by
# test_envelope_pp2_stage_compress_views_from_the_cache_manager. ratio -> (B_max pages, page size, gates' largest).
NATIVE_COMPRESS = {"ratio2": (6_480_780, 64, 1_954_512), "ratio1": (3_245_959, 128, 651_504)}
# ratio -> (the PP2 stage holding that ratio's views, the stage manager's (max_num_tokens, max_batch_size,
# max_draft_len, max_seq_len) at the native stage-alone maximum)
ENVELOPE_PP2_NATIVE_COMPRESS = {
    "ratio2": (0, (2048, 256, 0, 1_048_576)),
    "ratio1": (1, (2048, 256, 0, 1_048_576)),
}
_OWNERS = {"ratio2": (2, 8, 14), "ratio1": (20,)}


@contextmanager
def _native_storage():
    saved = os.environ.get(_STORAGE_ENV)
    os.environ[_STORAGE_ENV] = "native"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_STORAGE_ENV, None)
        else:
            os.environ[_STORAGE_ENV] = saved


def _v41_native_manager(
    max_num_tokens, max_batch_size, max_draft_len, max_seq_len=None, pp_stage=None
):
    """A V4.1 manager with native KV storage and 0.5 x the device's total memory, at the gates' max_seq_len
    unless max_seq_len says otherwise (as the engine passes it: + 14 with DSpark)."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=512,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )
    with _native_storage():
        mgr = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                max_gpu_total_bytes=total // 2,
                host_cache_size=0,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=128,
            max_seq_len=max_seq_len or (16398 if max_draft_len else 16384),
            max_batch_size=max_batch_size,
            mapping=_pp_mapping(pp_stage),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1] * 43
            ),
            pretrained_config=config,
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
            max_cuda_graph_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
        )
    assert mgr.v41_layout.kv_storage == "native"
    return mgr


def _native_compress(mgr, owner):
    """(owner's COMPRESS pool [B, page, 288] uint8, codes view, scales view), geometry asserted."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    page_size = 128 if owner == 20 else 64
    pool = mgr.get_buffers(owner, AT.COMPRESS)
    codes, scales = mgr.get_v41_native_kv_views(owner, AT.COMPRESS)
    b = pool.shape[0]
    assert pool.dtype == torch.uint8 and tuple(pool.shape) == (b, page_size, NATIVE_ROW), (
        owner,
        pool.shape,
    )
    assert pool.stride() == (page_size * NATIVE_ROW, NATIVE_ROW, 1), (owner, pool.stride())
    assert tuple(codes.shape) == (b, page_size, NATIVE_CODES) and codes.stride() == pool.stride()
    assert tuple(scales.shape) == (b, page_size, 32) and scales.stride() == pool.stride()
    assert (
        codes.data_ptr() == pool.data_ptr() and scales.data_ptr() == pool.data_ptr() + NATIVE_CODES
    )
    return pool, codes, scales


def _native_rows(x):
    """(uint8 [n, 288] native rows = packed codes then E4M3 bytes, bf16 [n, 512] core values) of the reference."""
    deq, packed, s_bytes = _ref_rows(x)
    return torch.cat([packed, s_bytes], dim=1), deq


def _native_hazard_slots(num_slots, page_size):
    """Distinct slots of a 288-byte-row pool on both sides of every byte offset 2^31..2^36 inside it (the row
    holding the offset and its neighbours), plus the first page, the last page and the last slot."""
    cands = {0, 1, page_size - 1, page_size}
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // NATIVE_ROW
        cands |= {s - 1, s, s + 1}
    cands |= {num_slots - page_size - 1, num_slots - page_size, num_slots - 2, num_slots - 1}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _count_not_sentinel(buf):
    whole = buf.view(-1)
    step = (
        1 << 26
    )  # a bool reduction materializes an int64 copy (8 bytes per element): keep chunks small
    n = torch.zeros((), dtype=torch.int64, device=DEV)
    for i in range(0, whole.numel(), step):
        n += (whole[i : i + step] != NATIVE_SENTINEL).sum()
    return n


def _check_native_pool(buf, flat, num_slots, slots, rows):
    """`flat`: the pool as [slots(+guard), 288] rows inside `buf`. The named slots hold `rows` bitwise; after
    they are restored to the sentinel, every byte of `buf` is the sentinel."""
    s64 = slots.to(torch.int64)
    live = (s64 >= 0) & (s64 < num_slots)
    idx = s64[live]
    got = flat[idx]
    n_bad = int((got != rows[live]).sum())
    assert n_bad == 0, f"{n_bad} of {got.numel()} bytes in the named slots differ bitwise"
    flat[idx] = NATIVE_SENTINEL
    n_out = int(_count_not_sentinel(buf))
    assert n_out == 0, f"{n_out} bytes outside the named slots were written"


def _native_engine_pool(pages, page_size):
    """(uint8 [pages + 1, page_size, 288] buffer with a guard page, codes view, scales view), all sentinel."""
    buf = torch.full(
        (pages + 1, page_size, NATIVE_ROW), NATIVE_SENTINEL, dtype=torch.uint8, device=DEV
    )
    return buf, buf[:pages, :, :NATIVE_CODES], buf[:pages, :, NATIVE_CODES:]


def test_native_compress_pools_from_the_cache_manager():
    """Every native configuration's COMPRESS views from the manager itself, and the hazard call on the largest.

    For each (max_num_tokens, max_batch_size, max_draft_len) at 0.5 x total
    memory: the owners' views are [B, 64, 288] (owners 2 / 8 / 14, bases one
    page apart) and [B, 128, 288] (owner 20), uint8, with the code / scale
    halves as the native views. The largest, (2048, 32, 0), is kept: on owner
    2's and owner 14's ratio-2 views and owner 20's ratio-1 view (each buffer
    filled with the sentinel first) the op's native form writes rows to both
    sides of every 2^31..2^35 byte offset, the first and the last page and the
    last slot; the buffer must hold the reference's packed codes and scale
    bytes there, decoding to the op's own core output, and the sentinel
    everywhere else.
    """
    got = {}
    keys = sorted(
        _NATIVE_COMPRESS_PAGES, key=lambda k: _NATIVE_COMPRESS_PAGES[k][0][0]
    )  # the largest last
    for key in keys:
        mgr = _v41_native_manager(*key)
        views = {o: _native_compress(mgr, o) for o in (2, 8, 14, 20)}
        base = views[2][0].data_ptr()
        for i, o in enumerate((2, 8, 14)):
            assert views[o][0].data_ptr() == base + i * 64 * NATIVE_ROW, o
        got[key] = (tuple(views[o][0].shape[0] for o in (2, 8, 14)), views[20][0].shape[0])
        if key != keys[-1]:
            mgr.shutdown()
            del mgr, views
            _free_engine_pool()
    print(
        f"\n[native, quota = 0.5 x device total] COMPRESS pages per (max_num_tokens, max_batch_size, "
        f"max_draft_len): {got}"
    )
    assert got == _NATIVE_COMPRESS_PAGES, got
    for ratio, i in (("ratio2", 0), ("ratio1", 1)):
        bmax, _, gate_max = GATES_NATIVE_COMPRESS[ratio]
        first = (lambda v: v[0][0]) if ratio == "ratio2" else (lambda v: v[1])
        assert max(first(v) for v in got.values()) == bmax
        assert max(first(got[k]) for k in _NATIVE_GATES) == gate_max
    try:
        for o in (2, 14, 20):
            pool, codes, scales = views[o]
            whole = (
                views[2][0] if o != 20 else pool
            )  # owner 2's view spans the whole ratio-2 buffer
            page_size = pool.shape[1]
            num_slots = pool.shape[0] * page_size
            whole.fill_(NATIVE_SENTINEL)
            hazard = _native_hazard_slots(num_slots, page_size)
            assert hazard[-1] == num_slots - 1
            slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device=DEV)
            x = _random_blocks(slots.numel(), seed=26060 + o)
            rows, deq = _native_rows(x)
            op(x, codes, slots, scales)
            flat = pool.view(-1, NATIVE_ROW)
            core = _run_core(x)
            assert torch.equal(_bits(core), _bits(deq))
            got_rows = flat[slots[: len(hazard)]]
            assert torch.equal(
                _bits(_decode(got_rows[:, :NATIVE_CODES], got_rows[:, NATIVE_CODES:])),
                _bits(core[: len(hazard)]),
            )
            _check_native_pool(whole, flat, num_slots, slots, rows)
            print(
                f"[manager's own native owner-{o} view {tuple(pool.shape)}] {len(hazard)} far-end / hazard rows "
                f"== reference codes + bytes, decode == core output, rest of the buffer sentinel"
            )
    finally:
        mgr.shutdown()
        del mgr, views
        _free_engine_pool()


# The max_batch_size axis: native COMPRESS pages at 0.5 x device total per (max_num_tokens,
# DSpark max_draft_len), (ratio 2 owner 2, ratio 1 owner 20) for max_batch_size 1..32 (measured on this
# GB300; owners 8 / 14 are 1 / 2 pages less than owner 2 in every configuration).
MBS_AXIS = tuple(range(1, 33))
_COMPRESS_PAGES_BY_MBS = {
    (2048, 0): (
        (
            1_954_512,
            1_954_512,
            1_954_512,
            1_954_512,
            1_954_512,
            2_171_790,
            1_954_512,
            2_131_740,
            2_131_740,
            1_954_512,
            2_104_860,
            2_104_860,
            2_104_860,
            2_104_860,
            2_104_860,
            2_345_640,
            2_345_640,
            2_345_640,
            2_345_640,
            2_443_035,
            2_528_916,
            2_528_916,
            2_468_520,
            2_468_520,
            2_541_339,
            2_541_339,
            2_541_339,
            2_419_368,
            2_419_368,
            2_550_441,
            2_606_019,
            2_606_019,
        ),
        (
            651_504,
            651_504,
            651_504,
            651_504,
            651_504,
            723_930,
            651_504,
            710_580,
            710_580,
            651_504,
            701_620,
            701_620,
            701_620,
            701_620,
            701_620,
            781_880,
            781_880,
            781_880,
            781_880,
            814_345,
            842_972,
            842_972,
            822_840,
            822_840,
            847_113,
            847_113,
            847_113,
            806_456,
            806_456,
            850_147,
            868_673,
            868_673,
        ),
    ),
    (2048, 5): (
        (
            1_954_512,
            1_954_512,
            1_954_512,
            1_954_512,
            1_736_703,
            1_776_753,
            1_776_753,
            1_954_512,
            1_954_512,
            1_954_512,
            2_104_860,
            1_954_512,
            1_954_512,
            1_954_512,
            2_198_184,
            2_198_184,
            2_198_184,
            2_069_844,
            2_069_844,
            2_171_790,
            2_262_810,
            2_262_810,
            2_345_640,
            2_233_683,
            2_310_144,
            2_310_144,
            2_209_107,
            2_209_107,
            2_209_107,
            2_255_208,
            2_315_604,
            2_315_604,
        ),
        (
            651_504,
            651_504,
            651_504,
            651_504,
            578_901,
            592_251,
            592_251,
            651_504,
            651_504,
            651_504,
            701_620,
            651_504,
            651_504,
            651_504,
            732_728,
            732_728,
            732_728,
            689_948,
            689_948,
            723_930,
            754_270,
            754_270,
            781_880,
            744_561,
            770_048,
            770_048,
            736_369,
            736_369,
            736_369,
            751_736,
            771_868,
            771_868,
        ),
    ),
    (8192, 0): (
        (
            1_116_840,
            1_040_865,
            1_040_865,
            1_040_865,
            1_040_865,
            1_220_607,
            1_220_607,
            1_378_986,
            1_378_986,
            1_518_891,
            1_518_891,
            1_439_061,
            1_439_061,
            1_439_061,
            1_674_807,
            1_674_807,
            1_674_807,
            1_674_807,
            1_674_807,
            1_776_753,
            1_869_594,
            1_954_512,
            1_875_966,
            1_804_167,
            1_881_429,
            1_881_429,
            1_881_429,
            1_881_429,
            2_020_692,
            2_020_692,
            2_083_497,
            2_083_497,
        ),
        (
            372_280,
            346_955,
            346_955,
            346_955,
            346_955,
            406_869,
            406_869,
            459_662,
            459_662,
            506_297,
            506_297,
            479_687,
            479_687,
            479_687,
            558_269,
            558_269,
            558_269,
            558_269,
            558_269,
            592_251,
            623_198,
            651_504,
            625_322,
            601_389,
            627_143,
            627_143,
            627_143,
            627_143,
            673_564,
            673_564,
            694_499,
            694_499,
        ),
    ),
    (8192, 5): (
        (
            1_040_865,
            1_040_865,
            1_040_865,
            1_040_865,
            1_040_865,
            1_148_805,
            1_148_805,
            1_302_528,
            1_302_528,
            1_439_061,
            1_439_061,
            1_367_151,
            1_367_151,
            1_367_151,
            1_598_349,
            1_598_349,
            1_598_349,
            1_598_349,
            1_529_172,
            1_627_476,
            1_720_320,
            1_736_703,
            1_736_703,
            1_736_703,
            1_751_265,
            1_751_265,
            1_751_265,
            1_751_265,
            1_889_619,
            1_889_619,
            1_895_082,
            1_895_082,
        ),
        (
            346_955,
            346_955,
            346_955,
            346_955,
            346_955,
            382_935,
            382_935,
            434_176,
            434_176,
            479_687,
            479_687,
            455_717,
            455_717,
            455_717,
            532_783,
            532_783,
            532_783,
            532_783,
            509_724,
            542_492,
            573_440,
            578_901,
            578_901,
            578_901,
            583_755,
            583_755,
            583_755,
            583_755,
            629_873,
            629_873,
            631_694,
            631_694,
        ),
    ),
}
_NATIVE_COMPRESS_PAGES_AXIS = {
    (t, s, d): ((c2[s - 1], c2[s - 1] - 1, c2[s - 1] - 2), c20[s - 1])
    for (t, d), (c2, c20) in _COMPRESS_PAGES_BY_MBS.items()
    for s in MBS_AXIS
}
assert all(_NATIVE_COMPRESS_PAGES_AXIS[k] == v for k, v in _NATIVE_COMPRESS_PAGES.items())


def test_native_compress_pools_over_every_max_batch_size():
    """B_max over the whole max_batch_size axis: the manager in native mode at 0.5 x total memory, for every
    (max_num_tokens, max_batch_size, max_draft_len) of {2048, 8192} x 1..32 x {0, 5} (128 managers;
    max_seq_len 16384 without DSpark, 16398 with it). The owners' views are [B, 64, 288] (2 / 8 / 14, bases
    one page apart, so B, B - 1, B - 2 pages) and [B, 128, 288] (20); the pages equal the table. The
    largest is B_max itself, 2,606,019 / 868,673 pages, reached at (2048, 31, 0) and (2048, 32, 0): no
    max_batch_size in 1..32 gives a larger native COMPRESS view, so the coverage below at B_max holds for
    every pool a manager at max_batch_size 1..32 allocates."""
    got = {}
    for key in _NATIVE_COMPRESS_PAGES_AXIS:
        mgr = _v41_native_manager(*key)
        try:
            assert mgr.max_batch_size == key[1] and mgr.max_seq_len == (16398 if key[2] else 16384)
            views = {o: _native_compress(mgr, o) for o in (2, 8, 14, 20)}
            base = views[2][0].data_ptr()
            for i, o in enumerate((2, 8, 14)):
                assert views[o][0].data_ptr() == base + i * 64 * NATIVE_ROW, (key, o)
            got[key] = (tuple(views[o][0].shape[0] for o in (2, 8, 14)), views[20][0].shape[0])
            del views
        finally:
            mgr.shutdown()
            del mgr
            _free_engine_pool()
    top2 = max(v[0][0] for v in got.values())
    top1 = max(v[1] for v in got.values())
    at = sorted(k for k, v in got.items() if v[0][0] == top2 or v[1] == top1)
    print(
        f"\n[native COMPRESS pages at 0.5 x device total, 128 configurations] {dict(sorted(got.items()))}\n"
        f"[axis maximum] ratio 2 {top2}, ratio 1 {top1}, at {at}"
    )
    assert got == _NATIVE_COMPRESS_PAGES_AXIS, {
        k: (v, _NATIVE_COMPRESS_PAGES_AXIS[k])
        for k, v in got.items()
        if v != _NATIVE_COMPRESS_PAGES_AXIS[k]
    }
    assert (top2, top1) == (GATES_NATIVE_COMPRESS["ratio2"][0], GATES_NATIVE_COMPRESS["ratio1"][0])
    assert at == [(2048, 31, 0), (2048, 32, 0)]


# (storage, max_num_tokens, max_batch_size, DSpark max_draft_len, max_seq_len as the engine passes it) ->
# (ratio-2 pages of owners 2 / 8 / 14, ratio-1 pages of owner 20) at 0.5 x this GB300's total memory: the
# envelope ceilings (the pool scan's maxima over 1920 managers per storage: max_num_tokens 2048..16384,
# max_batch_size 1..256 sampled, DSpark 0 / 5, max_seq_len 16384..1,048,576) and the max_batch_size-256
# configurations at max_num_tokens 16384.
_ENVELOPE_COMPRESS_PAGES = {
    ("bf16", 2048, 48, 0, 1_048_576): ((1_270_272, 1_270_271, 1_270_270), 423_424),
    ("bf16", 16384, 256, 0, 16384): ((1_035_264, 1_035_263, 1_035_262), 345_088),
    ("bf16", 16384, 256, 5, 16398): ((947_712, 947_711, 947_710), 315_904),
    ("native", 2048, 128, 0, 1_048_576): ((3_886_647, 3_886_646, 3_886_645), 1_295_549),
    ("native", 16384, 256, 0, 16384): ((2_703_360, 2_703_359, 2_703_358), 901_120),
    ("native", 16384, 256, 5, 16398): ((2_408_448, 2_408_447, 2_408_446), 802_816),
}


def test_envelope_largest_compress_views_from_the_cache_manager():
    """The envelope COMPRESS ceilings from the manager itself, and the far-end call on each ceiling
    manager's own views.

    Each configuration of _ENVELOPE_COMPRESS_PAGES is built at 0.5 x the device's total memory in its storage mode;
    the owners' views are asserted (bf16: [B, 64, 512] / [B, 128, 512] with strides (page * 512, 512, 1);
    native: [B, 64, 288] / [B, 128, 288] uint8 with code / scale halves; owners 2 / 8 / 14 one page apart)
    and their pages pinned. On the two ceiling managers (2048, 48 / 128, 0, 1,048,576), through owner 2's and
    owner 14's ratio-2 views and owner 20's ratio-1 view (each buffer filled with the sentinel first), the op
    writes rows to both sides of every 32-bit hazard (bf16 element offsets 2^30..2^35; native byte offsets
    2^31..2^36), the first and the last page and the last slot: the rows equal the reference (native: packed
    codes + scale bytes, decoding to the op's own core output) and nothing else in the buffer is written.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    got = {}
    ceilings = (
        ("bf16",) + ENVELOPE_NATIVE_COMPRESS_KEY[:1] + (48,) + ENVELOPE_NATIVE_COMPRESS_KEY[2:],
        ("native",) + ENVELOPE_NATIVE_COMPRESS_KEY,
    )
    for key in _ENVELOPE_COMPRESS_PAGES:
        storage, max_t, seqs, draft, lm = key
        if storage == "bf16":
            mgr = _v41_manager(total // 2, draft, seqs, max_num_tokens=max_t, max_seq_len=lm)
        else:
            mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm)
        try:
            assert mgr.max_seq_len == lm and mgr.max_batch_size == seqs
            if storage == "bf16":
                views = {o: mgr.get_buffers(o, AT.COMPRESS) for o in (2, 8, 14, 20)}
                for o, v in views.items():
                    ps = 64 if o != 20 else 128
                    assert tuple(v.shape[1:]) == (ps, W) and v.dtype == torch.bfloat16, (
                        o,
                        v.shape,
                        v.dtype,
                    )
                    assert v.stride() == (ps * W, W, 1), (o, v.stride())
                base = views[2].data_ptr()
                for i, o in enumerate((2, 8, 14)):
                    assert views[o].data_ptr() == base + i * 64 * W * 2, o
                got[key] = (tuple(views[o].shape[0] for o in (2, 8, 14)), views[20].shape[0])
                if key in ceilings:
                    for o in (2, 14, 20):
                        pool = views[o]
                        page_size = pool.shape[1]
                        num_slots = pool.shape[0] * page_size
                        whole = views[2] if o != 20 else pool
                        whole.view(torch.int16).fill_(SENTINEL)
                        hazard = _hazard_slots(num_slots, W, page_size)
                        assert hazard[-1] == num_slots - 1 and 2**34 // W in hazard
                        assert o == 20 or 2**35 // W in hazard
                        slots = torch.tensor(
                            hazard + [-1, num_slots], dtype=torch.int64, device=DEV
                        )
                        x = _random_blocks(slots.numel(), seed=12700 + o)
                        op(x, pool, slots)
                        _check_engine_pool(whole, pool, slots, _ref_rows(x)[0])
                        print(
                            f"\n[bf16 manager {key}, own owner-{o} view {tuple(pool.shape)}] {len(hazard)} far-end "
                            f"/ hazard rows == reference, rest of the buffer sentinel"
                        )
                    del pool, whole
                del views
            else:
                views = {o: _native_compress(mgr, o) for o in (2, 8, 14, 20)}
                base = views[2][0].data_ptr()
                for i, o in enumerate((2, 8, 14)):
                    assert views[o][0].data_ptr() == base + i * 64 * NATIVE_ROW, o
                got[key] = (tuple(views[o][0].shape[0] for o in (2, 8, 14)), views[20][0].shape[0])
                if key in ceilings:
                    for o in (2, 14, 20):
                        pool, codes, scales = views[o]
                        whole = views[2][0] if o != 20 else pool
                        page_size = pool.shape[1]
                        num_slots = pool.shape[0] * page_size
                        whole.fill_(NATIVE_SENTINEL)
                        hazard = _native_hazard_slots(num_slots, page_size)
                        assert hazard[-1] == num_slots - 1 and 2**35 // NATIVE_ROW in hazard
                        assert o == 20 or 2**36 // NATIVE_ROW in hazard
                        slots = torch.tensor(
                            hazard + [-1, num_slots], dtype=torch.int64, device=DEV
                        )
                        x = _random_blocks(slots.numel(), seed=38860 + o)
                        rows, deq = _native_rows(x)
                        op(x, codes, slots, scales)
                        flat = pool.view(-1, NATIVE_ROW)
                        core = _run_core(x)
                        assert torch.equal(_bits(core), _bits(deq))
                        got_rows = flat[slots[: len(hazard)]]
                        assert torch.equal(
                            _bits(_decode(got_rows[:, :NATIVE_CODES], got_rows[:, NATIVE_CODES:])),
                            _bits(core[: len(hazard)]),
                        )
                        _check_native_pool(whole, flat, num_slots, slots, rows)
                        print(
                            f"\n[native manager {key}, own owner-{o} view {tuple(pool.shape)}] {len(hazard)} "
                            f"far-end / hazard rows == reference codes + bytes, decode == core output, rest of "
                            f"the buffer sentinel"
                        )
                    del pool, codes, scales, whole, flat
                del views
        finally:
            mgr.shutdown()
            del mgr
            _free_engine_pool()
    print(f"\n[envelope COMPRESS pages at 0.5 x device total] {got}")
    assert got == _ENVELOPE_COMPRESS_PAGES, {
        k: (v, _ENVELOPE_COMPRESS_PAGES[k])
        for k, v in got.items()
        if v != _ENVELOPE_COMPRESS_PAGES[k]
    }
    for storage, table in (("bf16", UNSPLIT_COMPRESS), ("native", UNSPLIT_NATIVE_COMPRESS)):
        assert max(v[0][0] for k, v in got.items() if k[0] == storage) == table["ratio2"][0]
        assert max(v[1] for k, v in got.items() if k[0] == storage) == table["ratio1"][0]


# (storage, PP2 stage, max_num_tokens, max_batch_size, DSpark max_draft_len, max_seq_len) -> {owner: COMPRESS
# pages} of the stage manager built alone at 0.5 x this GB300's total memory: each storage's stage-alone maxima
# over the engine envelope. Stage 0 holds the ratio-2 owners 2 / 8 / 14 (one
# shared ratio-2 buffer, views one page apart), stage 1 the ratio-1 owner 20.
_ENVELOPE_PP2_COMPRESS_PAGES = {
    ("bf16", 0, 2048, 96, 0, 1_048_576): {2: 2_118_144, 8: 2_118_143, 14: 2_118_142},
    ("bf16", 1, 2048, 96, 0, 1_048_576): {20: 1_059_584},
    ("native", 0, 2048, 256, 0, 1_048_576): {2: 6_480_780, 8: 6_480_779, 14: 6_480_778},
    ("native", 1, 2048, 256, 0, 1_048_576): {20: 3_245_959},
}


def test_envelope_pp2_stage_compress_views_from_the_cache_manager():
    """The PP2 stage managers' COMPRESS views at their stage-alone maxima, and the far-end call on each
    manager's own views.

    Each configuration of _ENVELOPE_PP2_COMPRESS_PAGES is built alone (one process) with the mapping of a rank of
    that stage (pp_size 2, tp 2) at 0.5 x the device's total memory: pp_layers asserted, the stage's owners'
    views asserted (bf16 [B, 64 | 128, 512] with strides (page * 512, 512, 1); native [B, 64 | 128, 288]
    with code / scale halves; owners 2 / 8 / 14 one page apart) and their pages pinned. Through owner 2's
    and owner 14's views (stage 0) or owner 20's (stage 1), each buffer filled with the sentinel first, the
    op writes rows to both sides of every 32-bit hazard (bf16 element offsets 2^30..2^36; native byte offsets
    2^31..2^36), the first and the last page and the last slot: the rows equal the reference (native: packed
    codes + scale bytes, decoding to the op's own core output) and nothing else in the buffer is written.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    got = {}
    for key, want in _ENVELOPE_PP2_COMPRESS_PAGES.items():
        storage, stage, max_t, seqs, draft, lm = key
        if storage == "bf16":
            mgr = _v41_manager(
                total // 2, draft, seqs, max_num_tokens=max_t, max_seq_len=lm, pp_stage=stage
            )
        else:
            mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm, pp_stage=stage)
        try:
            assert list(mgr.pp_layers) == (list(range(20)) if stage == 0 else list(range(20, 40)))
            assert mgr.max_seq_len == lm and mgr.max_batch_size == seqs
            owners = tuple(want)
            if storage == "bf16":
                views = {o: mgr.get_buffers(o, AT.COMPRESS) for o in owners}
                for o, v in views.items():
                    ps = 64 if o != 20 else 128
                    assert tuple(v.shape[1:]) == (ps, W) and v.dtype == torch.bfloat16, (
                        o,
                        v.shape,
                        v.dtype,
                    )
                    assert v.stride() == (ps * W, W, 1), (o, v.stride())
                if stage == 0:
                    for i, o in enumerate((2, 8, 14)):
                        assert views[o].data_ptr() == views[2].data_ptr() + i * 64 * W * 2, o
                got[key] = {o: v.shape[0] for o, v in views.items()}
                for o in (2, 14) if stage == 0 else (20,):
                    pool = views[o]
                    page_size = pool.shape[1]
                    num_slots = pool.shape[0] * page_size
                    whole = views[2] if o != 20 else pool
                    whole.view(torch.int16).fill_(SENTINEL)
                    hazard = _hazard_slots(num_slots, W, page_size)
                    assert hazard[-1] == num_slots - 1 and 2**36 // W in hazard
                    slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device=DEV)
                    x = _random_blocks(slots.numel(), seed=21180 + o)
                    op(x, pool, slots)
                    _check_engine_pool(whole, pool, slots, _ref_rows(x)[0])
                    print(
                        f"\n[bf16 PP2 stage-{stage} manager {key}, own owner-{o} view {tuple(pool.shape)}] "
                        f"{len(hazard)} far-end / hazard rows == reference, rest of the buffer sentinel"
                    )
                    del pool, whole
                del views
            else:
                views = {o: _native_compress(mgr, o) for o in owners}
                if stage == 0:
                    for i, o in enumerate((2, 8, 14)):
                        assert (
                            views[o][0].data_ptr() == views[2][0].data_ptr() + i * 64 * NATIVE_ROW
                        ), o
                got[key] = {o: v[0].shape[0] for o, v in views.items()}
                for o in (2, 14) if stage == 0 else (20,):
                    pool, codes, scales = views[o]
                    whole = views[2][0] if o != 20 else pool
                    page_size = pool.shape[1]
                    num_slots = pool.shape[0] * page_size
                    whole.fill_(NATIVE_SENTINEL)
                    hazard = _native_hazard_slots(num_slots, page_size)
                    assert hazard[-1] == num_slots - 1 and 2**36 // NATIVE_ROW in hazard
                    slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device=DEV)
                    x = _random_blocks(slots.numel(), seed=64807 + o)
                    rows, deq = _native_rows(x)
                    op(x, codes, slots, scales)
                    flat = pool.view(-1, NATIVE_ROW)
                    core = _run_core(x)
                    assert torch.equal(_bits(core), _bits(deq))
                    got_rows = flat[slots[: len(hazard)]]
                    assert torch.equal(
                        _bits(_decode(got_rows[:, :NATIVE_CODES], got_rows[:, NATIVE_CODES:])),
                        _bits(core[: len(hazard)]),
                    )
                    _check_native_pool(whole, flat, num_slots, slots, rows)
                    print(
                        f"\n[native PP2 stage-{stage} manager {key}, own owner-{o} view {tuple(pool.shape)}] "
                        f"{len(hazard)} far-end / hazard rows == reference codes + bytes, decode == core output, "
                        f"rest of the buffer sentinel"
                    )
                    del pool, codes, scales, whole, flat
                del views
        finally:
            mgr.shutdown()
            del mgr
            _free_engine_pool()
    print(f"\n[PP2 stage COMPRESS pages at 0.5 x device total, stage alone] {got}")
    assert got == _ENVELOPE_PP2_COMPRESS_PAGES, {
        k: (v, _ENVELOPE_PP2_COMPRESS_PAGES[k])
        for k, v in got.items()
        if v != _ENVELOPE_PP2_COMPRESS_PAGES[k]
    }
    for storage, table in (("bf16", ENGINE_COMPRESS), ("native", NATIVE_COMPRESS)):
        assert max(v.get(2, 0) for k, v in got.items() if k[0] == storage) == table["ratio2"][0]
        assert max(v.get(20, 0) for k, v in got.items() if k[0] == storage) == table["ratio1"][0]


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_native_engine_pool_far_end(ratio, idx_dtype):
    """Decode-like on a B_max native pool with a guard page: hazard rows; -1 and past-the-end skipped."""
    pages, page_size, _ = NATIVE_COMPRESS[ratio]
    buf, codes, scales = _native_engine_pool(pages, page_size)
    num_slots = pages * page_size
    hazard = _native_hazard_slots(num_slots, page_size)
    slots = torch.tensor(
        hazard + [-1, num_slots, num_slots + page_size - 1], dtype=torch.int64, device=DEV
    )
    x = _random_blocks(slots.numel(), seed=26061 + page_size)
    op(x, codes, slots.to(idx_dtype), scales)
    _check_native_pool(buf, buf.view(-1, NATIVE_ROW), num_slots, slots, _native_rows(x)[0])
    del buf, codes, scales
    _free_engine_pool()


@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_native_engine_pool_prefill_spread(ratio):
    """Prefill-like, n = 8192 on a B_max native pool: the last page in order, every hazard slot, the rest
    spread uniformly, 1/16 of the rows -1."""
    n = _MAX_ROWS
    pages, page_size, _ = NATIVE_COMPRESS[ratio]
    buf, codes, scales = _native_engine_pool(pages, page_size)
    num_slots = pages * page_size
    last_page = torch.arange(num_slots - page_size, num_slots, device=DEV)
    hazard = torch.tensor(_native_hazard_slots(num_slots, page_size), device=DEV)
    fixed = torch.unique(torch.cat([last_page, hazard]))
    g = torch.Generator(device=DEV).manual_seed(26062 + page_size)
    rest = torch.randint(0, num_slots, (3 * n,), device=DEV, generator=g).unique()
    rest = rest[~torch.isin(rest, fixed)]
    rest = rest[torch.randperm(rest.numel(), device=DEV, generator=g)][: n - fixed.numel()]
    slots = torch.cat([last_page, fixed[~torch.isin(fixed, last_page)], rest]).to(torch.int64)
    assert slots.numel() == n and torch.unique(slots).numel() == n
    slots[torch.randperm(n, device=DEV, generator=g)[: n // 16]] = -1
    x = _sweep_rows(n, seed=26062 + page_size)
    op(x, codes, slots.to(torch.int32), scales)
    _check_native_pool(buf, buf.view(-1, NATIVE_ROW), num_slots, slots, _native_rows(x)[0])
    del buf, codes, scales
    _free_engine_pool()


def test_native_engine_pool_harness_sees_a_wrapped_offset():
    """Control: rows written where a mod-2^32 byte offset would put them fail both halves of the check."""
    pages, page_size, _ = NATIVE_COMPRESS["ratio2"]
    buf, codes, scales = _native_engine_pool(pages, page_size)
    num_slots = pages * page_size
    flat = buf.view(-1, NATIVE_ROW)
    true = torch.tensor(
        [s for s in _native_hazard_slots(num_slots, page_size) if s * NATIVE_ROW >= 2**32],
        dtype=torch.int64,
        device=DEV,
    )
    wrapped = (true * NATIVE_ROW % 2**32) // NATIVE_ROW
    assert true.numel() >= 9 and not torch.isin(wrapped, true).any()
    x = _random_blocks(true.numel(), seed=26063)
    rows = _native_rows(x)[0]
    op(x, codes, wrapped, scales)
    with pytest.raises(AssertionError, match="bytes in the named slots differ"):
        _check_native_pool(buf, flat, num_slots, true, rows)
    flat[true] = NATIVE_SENTINEL
    with pytest.raises(AssertionError, match="outside the named slots were written"):
        _check_native_pool(
            buf, flat, num_slots, torch.tensor([], dtype=torch.int64, device=DEV), rows[:0]
        )
    del buf, codes, scales, flat
    _free_engine_pool()


@pytest.mark.parametrize("n", [32, 8192, 16384])
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_native_pool_size_invariance(ratio, n):
    """The same rows into the same slots through views codes[:B] / scales[:B] of the live PP2 stage
    manager's envelope B_max view (owner 2 of stage 0, or owner 20 of stage 1, each built alone at
    (2048, 256, 0, 1,048,576)): the same bytes at every B, equal to the reference, nothing else in the buffer
    written; then a separate allocation of exactly the smallest size.

    B = the smallest view holding the slots, the page counts whose byte count first passes 2^31, 2^32,
    2^33, 2^34, 2^35 and 2^36 (where below B_max), a mid size, the gates' largest, the max_batch_size 1..32
    bound (GATES_NATIVE_COMPRESS), the unsplit manager's envelope bound (UNSPLIT_NATIVE_COMPRESS) and the
    stage's B_max.
    """
    pages, page_size, gate_max = NATIVE_COMPRESS[ratio]
    owner = 20 if ratio == "ratio1" else 2
    page_bytes = page_size * NATIVE_ROW
    stage, (max_t, seqs, draft, lm) = ENVELOPE_PP2_NATIVE_COMPRESS[ratio]
    mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm, pp_stage=stage)
    try:
        assert list(mgr.pp_layers) == (list(range(20)) if stage == 0 else list(range(20, 40)))
        pool, codes, scales = _native_compress(mgr, owner)
        assert pool.shape[0] == pages
        pool.fill_(NATIVE_SENTINEL)
        flat = pool.view(-1, NATIVE_ROW)
        small, slots = _invariance_slots(n, page_size, seed=26064 + n + page_size)
        x = (
            _sweep_rows(n, seed=26064 + n + page_size)
            if n >= 200
            else _random_blocks(n, seed=26064 + page_size)
        )
        rows = _native_rows(x)[0]
        live = slots[slots >= 0]
        sizes = sorted(
            {
                small,
                pages // 2 + 1,
                gate_max,
                GATES_NATIVE_COMPRESS[ratio][0],
                UNSPLIT_NATIVE_COMPRESS[ratio][0],
                pages,
            }
            | {
                lim // page_bytes + 1
                for lim in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)
                if lim // page_bytes + 1 < pages
            }
        )
        first = None
        for b in sizes:
            op(x, codes[:b], slots, scales[:b])
            torch.cuda.synchronize()
            got = flat[live].clone()
            _check_native_pool(pool, flat, b * page_size, slots, rows)
            first = got if first is None else first
            assert torch.equal(got, first), (
                f"B={b}: {int((got != first).sum())} bytes differ from B={sizes[0]}"
            )
    finally:
        mgr.shutdown()
        del mgr, pool, codes, scales, flat
        _free_engine_pool()
    buf, codes, scales = _native_engine_pool(small, page_size)
    op(x, codes, slots, scales)
    torch.cuda.synchronize()
    flat = buf.view(-1, NATIVE_ROW)
    got = flat[live].clone()
    _check_native_pool(buf, flat, small * page_size, slots, rows)
    assert torch.equal(got, first), "a separate exact-size native view differs"
    print(
        f"\n[native {ratio} n={n}] views B = {sizes} of the manager's owner-{owner} view and a separate "
        f"[{small}, {page_size}, 288]: {live.numel()} rows bitwise identical and == reference, nothing else written"
    )
    del buf, codes, scales, flat
    _free_engine_pool()


@pytest.mark.parametrize("ratio", list(_SWEEP_GEOMETRY))
def test_native_every_row_count_1_to_16384(ratio):
    """Every prefix (x[:r], slots[:r]), r = 1..16384, of the target's native call writes exactly the first r
    rows of the 16384-row call, and nothing else.

    A native pool of 20,480 slots in that ratio's page geometry plus a guard
    page, all bytes sentinel. The 16384-row call is held to the reference and
    the sentinel scan. Then per r: the call, the named live slots against the
    reference rows, the count of non-sentinel bytes against the reference's
    own count for the live rows, and the named slots restored to the sentinel,
    so each call's writes are checked on their own. One sync at the end. The
    probe: the sweep compiles no variant beyond the 8192-row call's.
    """
    t = _MAX_ROWS
    page_size = _SWEEP_GEOMETRY[ratio]
    buf, codes, scales = _native_engine_pool(_SWEEP_SLOTS // page_size, page_size)
    flat = buf.view(-1, NATIVE_ROW)
    slots = _sweep_slots(t, _SWEEP_SLOTS, seed=26065 + page_size)
    x = _sweep_rows(t, seed=26065 + page_size)
    rows = _native_rows(x)[0]
    kernel = _nv_op._nvfp4_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    op(x, codes, slots, scales)
    n1 = _jit_variants(kernel)
    _check_native_pool(buf, flat, _SWEEP_SLOTS, slots, rows)

    live = slots >= 0
    live_pos = torch.nonzero(live).squeeze(1)
    live_slots = slots[live_pos]
    rows_live = rows[live_pos]
    cnt = torch.cumsum(live.to(torch.int64), 0).tolist()
    # A packed code byte can equal the sentinel, so the expected count is the reference rows' own.
    expect = torch.cumsum((rows_live != NATIVE_SENTINEL).sum(1), 0)
    bad_rows = torch.zeros((), dtype=torch.int64, device=DEV)
    bad_count = torch.zeros((), dtype=torch.int64, device=DEV)
    for r in range(1, t + 1):
        op(x[:r], codes, slots[:r], scales)
        k = cnt[r - 1]
        bad_rows += (flat[live_slots[:k]] != rows_live[:k]).sum()
        bad_count += ((buf != NATIVE_SENTINEL).sum() - (expect[k - 1] if k else 0)).abs()
        flat[live_slots[:k]] = NATIVE_SENTINEL
    n2 = _jit_variants(kernel)
    print(
        f"\n[native {ratio} rows 1..{t}, page {page_size}] live rows {cnt[-1]}; Triton variants: {n0} before the "
        f"{t}-row call, {n1} after, {n2} after the sweep; byte mismatches {int(bad_rows)}, "
        f"non-sentinel count error {int(bad_count)}"
    )
    assert int(bad_rows) == 0 and int(bad_count) == 0
    assert n1 - n0 <= 1 and n2 == n1, f"row counts compiled new variants: {n0} -> {n1} -> {n2}"
    # Control: the same checks see a read one row off and a write one slot off.
    for r in (1, 2, 7, 64):
        for name, xa, sa in (
            ("row +1", x[1 : r + 1], live_slots[:r]),
            ("slot +1", x[:r], live_slots[1 : r + 1]),
        ):
            op(xa, codes, sa, scales)
            n_bad = int((flat[live_slots[:r]] != _native_rows(x[:r])[0]).sum())
            print(f"[native control {name}] r={r}: {n_bad} of {r * NATIVE_ROW} bytes differ")
            assert n_bad > r * NATIVE_ROW // 4
            buf.fill_(NATIVE_SENTINEL)
    del buf, codes, scales, flat
    _free_engine_pool()


@pytest.mark.parametrize("ratio", ["ratio1", "ratio2"])
def test_native_cuda_graph_every_graph_batch_on_the_bmax_pool(ratio):
    """Per n of that ratio's envelope capture set (ratio 1: G and 6G, 82 row counts to 1536; ratio 2: G and
    3G, 77 row counts to 768), native form on the envelope B_max native pool (3,886,647 / 1,295,549
    pages): an eager warm-up, the capture, then 2 replays on new rows and new far-end slots; every replay
    writes exactly the reference into the named slots and nothing else, and equals the same call run
    eagerly."""
    pages, page_size, _ = NATIVE_COMPRESS[ratio]
    ns = _GRAPH_N[ratio]
    buf, codes, scales = _native_engine_pool(pages, page_size)
    num_slots = pages * page_size
    flat = buf.view(-1, NATIVE_ROW)
    g = torch.Generator(device=DEV).manual_seed(26066 + page_size)
    hazard = torch.tensor(_native_hazard_slots(num_slots, page_size), device=DEV)
    last = num_slots - 1
    table = torch.full(
        (_MAX_ROWS,), -1, dtype=torch.int64, device=DEV
    )  # like v41_compress_slot_mapping[r]

    def far_slots(n):
        near = last - 1 - torch.arange(4 * page_size - 1, device=DEV)
        cand = torch.unique(torch.cat([hazard[hazard != last], near]))
        if cand.numel() < n - 1:  # large graph batches: the rest spread uniformly over the pool
            spread = torch.randint(0, last, (4 * n,), device=DEV, generator=g).unique()
            cand = torch.cat([cand, spread[~torch.isin(spread, cand)]])
        cand = cand[torch.randperm(cand.numel(), device=DEV, generator=g)]
        s = torch.cat([torch.tensor([last], device=DEV), cand[: n - 1]]).to(torch.int64)
        assert torch.unique(s).numel() == n
        if n >= 2:
            s[1 + int(torch.randint(0, n - 1, (1,), device=DEV, generator=g))] = -1
        return s

    for n in ns:
        x = torch.empty(n, W, dtype=torch.bfloat16, device=DEV)
        slots = table[:n]
        x.copy_(_random_blocks(n, seed=26066 + n))
        slots.copy_(far_slots(n))
        op(x, codes, slots, scales)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, codes, slots, scales)
        buf.fill_(NATIVE_SENTINEL)
        for rep in (1, 2):
            new_x = _random_blocks(n, seed=26066 + 1000 * rep + n)
            new_slots = far_slots(n)
            x.copy_(new_x)
            slots.copy_(new_slots)
            graph.replay()
            torch.cuda.synchronize()
            live = new_slots[new_slots >= 0]
            replayed = flat[live].clone()
            rows = _native_rows(new_x)[0]
            _check_native_pool(buf, flat, num_slots, new_slots, rows)
            op(x, codes, slots, scales)
            torch.cuda.synchronize()
            eager = flat[live].clone()
            _check_native_pool(buf, flat, num_slots, new_slots, rows)
            assert torch.equal(replayed, eager), (
                f"n={n} replay {rep}: {int((replayed != eager).sum())} bytes"
            )
        del graph
    print(
        f"\n[native {ratio} graphs on [{pages}, {page_size}, 288]] n = {list(ns)}: every replay bitwise == eager "
        f"== reference"
    )
    del buf, codes, scales, flat
    _free_engine_pool()


# ─── native form over random multi-request steps on the live manager ───


def _random_step(rng, max_tokens=8192, max_requests=32, max_len=16384):
    """([(cached, new)] contexts first (random order) then generation requests, num_contexts, gen_len)."""
    k = rng.randint(1, max_requests)
    gen_len = rng.choice([1, 6])
    g = rng.randint(0, k) if rng.random() < 0.7 else 0
    c = k - g
    budget = max_tokens - g * gen_len
    t_ctx = rng.randint(c, budget) if c else 0
    cuts = sorted(rng.sample(range(1, t_ctx), c - 1)) if c > 1 else []
    news = [b - a for a, b in zip([0] + cuts, cuts + [t_ctx])] if c else []
    reqs = [(0 if rng.random() < 0.4 else rng.randint(0, max_len - n), n) for n in news]
    rng.shuffle(reqs)
    reqs += [(rng.randint(0, max_len - gen_len), gen_len) for _ in range(g)]
    return reqs, c, gen_len


_EDGE_STEPS = [
    ([(0, 8192)], 1, 1),
    ([(0, 1)], 1, 1),
    ([(16383, 1)], 1, 1),
    ([(0, 256)] * 32, 32, 1),
    ([(16384 - 256, 256)] * 32, 32, 1),
    ([(16383, 1)] * 32, 0, 1),
    ([(16378, 6)] * 32, 0, 6),
    ([(513, 7000)] + [(16000, 6)] * 31, 1, 6),
    ([(0, 127), (0, 128), (0, 129), (127, 1), (128, 1), (8000, 192)], 6, 1),
]
_NUM_RANDOM_STEPS = 200


def _run_step(mgr, reqs, num_contexts, gen_len, uid, max_num_tokens=8192):
    """Allocate the step's requests on the manager and build its DeepSeek-V4.1 attention metadata.

    Each request's first token is unique over the test (uid + request index), so
    block reuse never shares a page between requests of different steps.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    requests = []
    for rid, (c, n) in enumerate(reqs):
        r = LlmRequest(
            request_id=rid,
            max_new_tokens=64,
            input_tokens=[uid + rid] + list(range(1, c + n)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert mgr.prepare_context(r) and mgr.resize_context(r, r.context_chunk_size), (rid, c, n)
        requests.append(r)
    md = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor([n for _, n in reqs], dtype=torch.int),
        request_ids=[r.py_request_id for r in requests],
        max_num_requests=mgr.max_batch_size,
        num_contexts=num_contexts,
        prompt_lens=[c + n for c, n in reqs],
        max_num_tokens=max_num_tokens,
        kv_cache_manager=mgr,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[c for c, _ in reqs]
        ),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=[1] * 43
        ),
    )
    if num_contexts < len(reqs) and gen_len > 1:
        md.update_spec_dec_param(
            batch_size=len(reqs),
            is_spec_decoding_enabled=False,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=gen_len - 1,
            max_total_draft_tokens=gen_len - 1,
        )
    md.prepare()
    return requests, md


def test_native_writes_over_random_multi_request_steps():
    """On the live accuracy-gate manager (8192, 32, DSpark 5; native COMPRESS 1,895,082 / 631,694).

    Per step (9 edge steps + 200 seeded random partitions of T <= 8192 into
    k <= 32 requests, contexts with random cached prefixes, decode or DSpark
    verify generation requests): the requests are allocated on the manager and
    the metadata is prepared; per ratio, every owner (2, 8, 14 at ratio 2, 20
    at ratio 1) gets random latents written by the op's native form at
    md.v41_compress_slot_mapping[r][:n] (reserved generation rows -1) through
    get_v41_native_kv_views(o, COMPRESS): the slots hold the reference's
    packed codes and scale bytes, decoding to the op's own core output of the
    same rows; after the step every other byte of both ratio buffers is still
    the sentinel. The op compiles no variant past its first native call.
    """
    mgr = _v41_native_manager(8192, 32, 5)
    rng = random.Random(26067)
    steps = list(_EDGE_STEPS) + [_random_step(rng) for _ in range(_NUM_RANDOM_STEPS)]
    kernel = _nv_op._nvfp4_fake_quantize_kernel
    try:
        views = {o: _native_compress(mgr, o) for o in (2, 8, 14, 20)}
        whole = {"ratio2": views[2][0], "ratio1": views[20][0]}
        for w in whole.values():
            w.fill_(NATIVE_SENTINEL)
        core_buf = torch.empty(_MAX_ROWS, W, dtype=torch.bfloat16, device=DEV)
        n_variants = None
        stats = dict(steps=0, rows=0, skipped=0, max_t=0, max_k=0, max_n={1: 0, 2: 0})
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(mgr, reqs, nctx, gen_len, uid=300_000 + 32 * i)
            try:
                for ratio, r in (("ratio2", 2), ("ratio1", 1)):
                    n = md.num_total_compressed_tokens[r]
                    if n == 0:
                        continue
                    slots = md.v41_compress_slot_mapping[r][:n]
                    s64 = slots.to(torch.int64)
                    live = s64 >= 0
                    for o in _OWNERS[ratio]:
                        pool, codes, scales = views[o]
                        flat = pool.view(-1, NATIVE_ROW)
                        x = _random_blocks(n, seed=400_000 + 4 * i + o)
                        op(x, codes, slots, scales)
                        rows, deq = _native_rows(x)
                        # A fixed 8192-slot out: in the identity form the slot count is an integer
                        # argument, and Triton specializes it (== 1, % 16 == 0) per call.
                        op(x, core_buf)
                        core = core_buf[:n]
                        if n_variants is None:
                            n_variants = _jit_variants(kernel)
                        assert int(s64.max()) < pool.shape[0] * pool.shape[1]
                        got = flat[s64[live]]
                        assert torch.equal(got, rows[live]), (
                            f"step {i} owner {o}: {int((got != rows[live]).sum())}"
                        )
                        assert torch.equal(
                            _bits(_decode(got[:, :NATIVE_CODES], got[:, NATIVE_CODES:])),
                            _bits(core[live]),
                        )
                        assert torch.equal(_bits(core), _bits(deq))
                        flat[s64[live]] = NATIVE_SENTINEL
                        stats["rows"] += int(live.sum())
                    stats["skipped"] += int((~live).sum())
                    stats["max_n"][r] = max(stats["max_n"][r], n)
                for ratio, w in whole.items():
                    n_out = int(_count_not_sentinel(w))
                    assert n_out == 0, (
                        f"step {i} {ratio}: {n_out} bytes outside the step's slots were written"
                    )
                stats["steps"] += 1
                stats["max_t"] = max(stats["max_t"], sum(n for _, n in reqs))
                stats["max_k"] = max(stats["max_k"], len(reqs))
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
        assert _jit_variants(kernel) == n_variants, "a step shape compiled a new variant"
        print(
            f"\n[native multi-request steps on the live manager, COMPRESS {tuple(views[2][0].shape)} / "
            f"{tuple(views[20][0].shape)}] {stats}; one compiled variant"
        )
        assert stats["steps"] == len(_EDGE_STEPS) + _NUM_RANDOM_STEPS and stats["max_t"] == 8192
        assert stats["max_k"] == 32 and stats["skipped"] > 0
    finally:
        mgr.shutdown()
        del mgr, views, whole
        _free_engine_pool()


# The envelope's step axes: T <= max_num_tokens 16384, k <= max_batch_size 256 requests per step. A step's KV
# tokens (cached + new, over its requests) are capped at _ENVELOPE_STEP_TOKENS: a native manager at (16384, 256,
# DSpark 5, max_seq_len 32,782) held 22 requests of 32,768 tokens and refused the 23rd, so 256 requests cannot
# all sit at the far end of max_seq_len at once.
_ENVELOPE_STEP_TOKENS = 560_000
_ENVELOPE_STEP_L = 32768


def _envelope_random_step(rng, max_tokens, max_requests, seq_len):
    """_random_step at the envelope's step axes with DSpark: k in 1..max_requests, contexts end at <= seq_len,
    decode requests (next_n 1) at <= seq_len, DSpark-verify requests (next_n 6) at <= seq_len + 14."""
    k = rng.randint(1, max_requests)
    gen_len = rng.choice([1, 6])
    g = rng.randint(0, k) if rng.random() < 0.7 else 0
    c = k - g
    budget = max_tokens - g * gen_len
    t_ctx = rng.randint(c, budget) if c else 0
    cuts = sorted(rng.sample(range(1, t_ctx), c - 1)) if c > 1 else []
    news = [b - a for a, b in zip([0] + cuts, cuts + [t_ctx])] if c else []
    reqs = [(0 if rng.random() < 0.4 else rng.randint(0, seq_len - n), n) for n in news]
    rng.shuffle(reqs)
    end = seq_len + 14 if gen_len == 6 else seq_len
    reqs += [(rng.randint(0, end - gen_len), gen_len) for _ in range(g)]
    return reqs, c, gen_len


def _envelope_fit(step, rng):
    """The step with its KV tokens capped at _ENVELOPE_STEP_TOKENS: in a random order, a request keeps its cached
    prefix while the rest can still fit, else gets a random prefix below 1024. Order and new tokens kept."""
    reqs, nctx, gen_len = step
    if sum(a + n for a, n in reqs) <= _ENVELOPE_STEP_TOKENS:
        return step
    out, used = list(reqs), 0
    reserve = sum(n + 1024 for _, n in reqs)
    order = list(range(len(reqs)))
    rng.shuffle(order)
    for i in order:
        a, n = reqs[i]
        reserve -= n + 1024
        if used + a + n + reserve > _ENVELOPE_STEP_TOKENS:
            a = rng.randint(0, 1023)
        out[i] = (a, n)
        used += a + n
    assert used <= _ENVELOPE_STEP_TOKENS
    return out, nctx, gen_len


def _envelope_edge_steps(max_t, seqs, L):
    """A full max_t step from position 0 and one ending at L; seqs decode requests (8 at L - 1, the rest below
    position 1152); seqs DSpark verifies (8 ending at L + 13, the rest below 1130); a context filling the step
    beside seqs - 1 verifies (7 at the far end)."""
    lm, rest = L + 14, seqs - 8
    steps = [
        ([(0, max_t)], 1, 1),
        ([(L - max_t, max_t)], 1, 1),
        ([(L - 1, 1)] * 8 + [(127 + (61 * i) % 1024, 1) for i in range(rest)], 0, 1),
        ([(lm - 6, 6)] * 8 + [(100 + (37 * i) % 1024, 6) for i in range(rest)], 0, 6),
        (
            [(0, max_t - 6 * (seqs - 1))]
            + [(lm - 6, 6)] * 7
            + [(100 + (37 * i) % 1024, 6) for i in range(rest)],
            1,
            6,
        ),
    ]
    for reqs, _, _ in steps:
        assert sum(a + b for a, b in reqs) <= _ENVELOPE_STEP_TOKENS
    return steps


def test_envelope_native_writes_over_random_steps_256_requests():
    """At the envelope's step axes, on the live native manager of max_num_tokens 16384,
    max_batch_size 256, DSpark 5, max_seq_len 32768 + 14: 5 edge steps and 200 seeded random partitions of
    T <= 16384 into k <= 256 requests (contexts first with random cached prefixes, decode at positions
    < 32,768 or DSpark verify at < 32,782), each step's KV tokens capped (_envelope_fit). Per step, exactly as
    test_native_writes_over_random_multi_request_steps: per ratio, every owner (2, 8, 14 at ratio 2, 20 at
    ratio 1) gets random latents written by the op's native form at md.v41_compress_slot_mapping[r][:n]
    (reserved generation rows -1) through get_v41_native_kv_views(o, COMPRESS); the slots hold the
    reference's packed codes and scale bytes, decoding to the op's own core output of the same rows, and
    after the step every other byte of both ratio buffers is still the sentinel; no compiled variant past the
    first native call."""
    max_t, seqs, L = 16384, 256, _ENVELOPE_STEP_L
    rng = random.Random(26_068 + seqs + L)
    edges = _envelope_edge_steps(max_t, seqs, L)
    steps = edges + [
        _envelope_fit(_envelope_random_step(rng, max_t, seqs, L), rng)
        for _ in range(_NUM_RANDOM_STEPS)
    ]
    mgr = _v41_native_manager(max_t, seqs, 5, max_seq_len=L + 14)
    kernel = _nv_op._nvfp4_fake_quantize_kernel
    try:
        assert mgr.max_seq_len == L + 14 and mgr.max_batch_size == seqs
        views = {o: _native_compress(mgr, o) for o in (2, 8, 14, 20)}
        whole = {"ratio2": views[2][0], "ratio1": views[20][0]}
        for w in whole.values():
            w.fill_(NATIVE_SENTINEL)
        core_buf = torch.empty(max_t, W, dtype=torch.bfloat16, device=DEV)
        n_variants = None
        stats = dict(steps=0, rows=0, skipped=0, max_t=0, max_k=0, max_pos=0, max_n={1: 0, 2: 0})
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(
                mgr, reqs, nctx, gen_len, uid=6_000_000 + 256 * i, max_num_tokens=max_t
            )
            try:
                for ratio, r in (("ratio2", 2), ("ratio1", 1)):
                    n = md.num_total_compressed_tokens[r]
                    if n == 0:
                        continue
                    slots = md.v41_compress_slot_mapping[r][:n]
                    s64 = slots.to(torch.int64)
                    live = s64 >= 0
                    for o in _OWNERS[ratio]:
                        pool, codes, scales = views[o]
                        flat = pool.view(-1, NATIVE_ROW)
                        x = _random_blocks(n, seed=600_000 + 4 * i + o)
                        op(x, codes, slots, scales)
                        rows, deq = _native_rows(x)
                        op(x, core_buf)  # a fixed 16384-slot out (one identity variant)
                        core = core_buf[:n]
                        if n_variants is None:
                            n_variants = _jit_variants(kernel)
                        assert int(s64.max()) < pool.shape[0] * pool.shape[1]
                        got = flat[s64[live]]
                        assert torch.equal(got, rows[live]), (
                            f"step {i} owner {o}: {int((got != rows[live]).sum())}"
                        )
                        assert torch.equal(
                            _bits(_decode(got[:, :NATIVE_CODES], got[:, NATIVE_CODES:])),
                            _bits(core[live]),
                        )
                        assert torch.equal(_bits(core), _bits(deq))
                        flat[s64[live]] = NATIVE_SENTINEL
                        stats["rows"] += int(live.sum())
                    stats["skipped"] += int((~live).sum())
                    stats["max_n"][r] = max(stats["max_n"][r], n)
                for ratio, w in whole.items():
                    n_out = int(_count_not_sentinel(w))
                    assert n_out == 0, (
                        f"step {i} {ratio}: {n_out} bytes outside the step's slots were written"
                    )
                stats["steps"] += 1
                stats["max_t"] = max(stats["max_t"], sum(n for _, n in reqs))
                stats["max_k"] = max(stats["max_k"], len(reqs))
                stats["max_pos"] = max(stats["max_pos"], max(c + n - 1 for c, n in reqs))
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
        assert _jit_variants(kernel) == n_variants, "a step shape compiled a new variant"
        print(
            f"\n[envelope native steps, ({max_t}, {seqs}, 5) at max_seq_len {L + 14}, COMPRESS "
            f"{tuple(views[2][0].shape)} / {tuple(views[20][0].shape)}] {len(edges)} edge + {_NUM_RANDOM_STEPS} "
            f"random steps: {stats}; one compiled variant"
        )
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_pos"] == L + 13 and stats["skipped"] > 0 and stats["max_n"][1] == max_t
    finally:
        mgr.shutdown()
        del mgr, views, whole
        _free_engine_pool()


def test_native_variant_map():
    """Which calls compile a new Triton variant: the native form into a manager-geometry view compiles one
    variant whatever n (rows are grid axis 0; the view's slot count, a multiple of 64, is fixed); the
    identity (core) form passes the row count as the slot count, which Triton specializes (== 1,
    % 16 == 0, other), so it compiles one variant per class and none for a second member of a class."""
    kernel = _nv_op._nvfp4_fake_quantize_kernel
    added = {}
    for ratio, page_size in (("ratio2", 64), ("ratio1", 128)):
        buf, codes, scales = _native_engine_pool(10240 // page_size, page_size)
        for n in (8192, 1, 2, 16, 17, 4096, 8191):
            slots = torch.randperm(10240, device=DEV)[:n]
            n0 = _jit_variants(kernel)
            op(_random_blocks(n, seed=n), codes, slots, scales)
            added[(f"native[{ratio}]", n)] = _jit_variants(kernel) - n0
        del buf, codes, scales
    classes = {"1": (1, 1), "16k": (16, 32, 48, 8192), "other": (2, 17, 33, 8191)}
    for name, ns in classes.items():
        for i, n in enumerate(ns):
            n0 = _jit_variants(kernel)
            op(_random_blocks(n, seed=n), torch.empty(n, W, dtype=torch.bfloat16, device=DEV))
            added[(f"identity[{name}]#{i}", n)] = _jit_variants(kernel) - n0
    print(f"\n[variant map] new variants per call: {added}")
    assert all(v == 0 for (form, n), v in added.items() if form.startswith("native") and n != 8192)
    for name, ns in classes.items():
        assert all(added[(f"identity[{name}]#{i}", n)] == 0 for i, n in enumerate(ns) if i), name
