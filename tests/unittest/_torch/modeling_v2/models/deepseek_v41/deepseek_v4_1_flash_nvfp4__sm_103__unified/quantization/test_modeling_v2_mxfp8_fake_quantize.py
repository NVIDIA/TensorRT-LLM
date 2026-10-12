# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_fake_quantize catalog entry.

The entry owns what is certified: `CELLS` is the list (bf16 container row counts and widths, in
place, strided and 3-D `x`, paged scatter, strided pools, out-of-range slots, the native 528 B
layouts and identity form, zero rows), `reference` and `compare` are the bitwise gate. This file
turns a cell's spec into tensors and keeps the checks a cell cannot carry: non-finite row
isolation, CUDA-graph replay, the deliberately-wrong recipes, the op's own rejections, and the
engine-scale cases (the largest SWA pools, every row count, CUDA-graph capture at every graph
batch, the DSpark drafter's ring and block writes, native storage on the live DeepSeek-V4.1 cache
manager).

Every comparison is bitwise (bf16 compared as int16, codes and scale bytes as uint8): the op
computes one exact value per element, so there is no tolerance to choose. The entry's reference
takes the scale exponent from torch.frexp of the float64 quotient amax / 448 rather than from the
IEEE-bit trick the kernel uses, so the two derivations are independent. (A float64 torch.log2 is
NOT usable for this: torch.ceil(torch.log2(amax.double() / 448)) gave -2 for amax = 56, i.e. for
56 / 448 = 2^-3, one too high at an exact power of two, amax = 7 * 2^k.)
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.mxfp8_fake_quantize import (  # noqa: E501
    mxfp8_fake_quantize as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mxfp8_fake_quantize requires a CUDA device"

BLOCK = 32
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4
WINDOW_KV_WIDTH = 512  # DeepSeek-V4.1 window KV head_dim (448 nope + 64 RoPE, all quantized)


def _fake_quant(x: torch.Tensor):
    """The entry's reference for the rows of a bf16 [T, W] x.

    Returns (bf16 dequantized [T, W], e4m3 codes as uint8 [T, W], UE8M0 bytes uint8 [T, W/32]).
    """
    t, w = x.reshape(-1, x.shape[-1]).shape
    y = op.reference(x, torch.empty(t, w, dtype=torch.bfloat16, device=x.device))
    codes, sbytes = op.reference(
        x,
        torch.empty(t, w, dtype=torch.uint8, device=x.device),
        None,
        torch.empty(t, w // BLOCK, dtype=torch.uint8, device=x.device),
    )
    return y, codes, sbytes


def _scale_exponent(
    amax: torch.Tensor, floor: float | None = AMAX_FLOOR, rounding=torch.ceil
) -> torch.Tensor:
    """ceil (or floor) of log2(max(amax, floor) / 448), int32, exact.

    amax is fp32 and floored in fp32 (as the reference does). The float64
    quotient of a bf16 amax by 448 is a power of two exactly when amax = 7 * 2^k
    and otherwise sits far from one, so frexp (mantissa in [0.5, 1)) classifies
    it exactly: v = m * 2^E has ceil(log2 v) = E, or E - 1 when m == 0.5, and
    floor(log2 v) = E - 1.
    """
    a = amax.float()
    if floor is not None:
        a = a.clamp_min(floor)
    m, e = torch.frexp(a.double() / E4M3_MAX)
    if rounding is torch.floor:
        return (e - 1).to(torch.int32)
    return torch.where(m == 0.5, e - 1, e).to(torch.int32)


def _control_quant(
    x: torch.Tensor,
    *,
    block: int = BLOCK,
    floor: float | None = AMAX_FLOOR,
    rounding=torch.ceil,
    scale_floor_exp: int | None = None,
) -> torch.Tensor:
    """A deliberately-wrong MXFP8 fake quantization of a bf16 [T, W] tensor -> bf16 [T, W].

    `floor`, `rounding`, `block` and `scale_floor_exp` select the control.
    """
    t, w = x.shape
    xf = x.float().view(t, w // block, block)
    e = _scale_exponent(xf.abs().amax(-1), floor, rounding)
    if scale_floor_exp is not None:
        e = e.clamp_min(scale_floor_exp)
    s = torch.exp2(e.double()).float().unsqueeze(-1)
    q = (xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (q.float() * s).view(t, w).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.uint8)


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    a, e = _bits(actual), _bits(expected)
    n_bad = (a != e).sum().item()
    assert n_bad == 0, f"{n_bad} of {a.numel()} elements differ bitwise"


def _window_kv_like(t: int, w: int = WINDOW_KV_WIDTH, seed: int = 0) -> torch.Tensor:
    """bf16 [t, w] rows whose 32-blocks cover every path of the recipe.

    Block magnitudes span 2^-30 .. 2^20 (so the 1e-4 floor, e4m3 subnormals and
    BOS-like massive rows are all present), and a quarter of the rows carry
    hand-placed blocks: all-zero, signed zeros, amax exactly 448 * 2^k (the
    ceil boundary), amax one bf16 step above it, e4m3 round-to-even ties, and
    sub-1e-4 blocks with a wide dynamic range (where the floor changes values).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = w // BLOCK
    mag = torch.randint(-30, 21, (t, nb, 1), device="cuda", generator=g).float()
    x = torch.randn(t, nb, BLOCK, device="cuda", generator=g) * torch.exp2(mag)
    for r in range(0, t, 4):
        k = r % 7 - 3
        b = (r // 4) % nb
        x[r, b] = 0.0
        x[r, (b + 1) % nb] = torch.randn(BLOCK, device="cuda", generator=g) * 2.0**k
        x[r, (b + 1) % nb, 0] = 448.0 * 2.0**k
        x[r, (b + 2) % nb] = torch.randn(BLOCK, device="cuda", generator=g) * 2.0**k
        x[r, (b + 2) % nb, 0] = 448.0 * 2.0**k * (1 + 2**-7)
        # e4m3 ties at scale 2^k: 1.0625 (-> 1.0), 1.1875 (-> 1.25), 3.25*2^-9 subnormal (-> 3*2^-9 or 4*2^-9)
        x[r, (b + 3) % nb] = 2.0**k * torch.tensor(
            [448.0, 1.0625, -1.1875, 3.25 * 2**-9, -2.5 * 2**-9, 13.5 * 2**-9, 0.5 * 2**-9, -0.0]
            * 4,
            device="cuda",
        )
        x[r, (b + 4) % nb] = torch.randn(BLOCK, device="cuda", generator=g) * 1e-6
        x[r, (b + 4) % nb, 1::2] *= 1e-3
        x[r, (b + 5) % nb] = torch.randn(BLOCK, device="cuda", generator=g) * 2.0**-10
    return x.view(t, w).to(torch.bfloat16)


# Pool widths are chosen plainly larger than every batch here: 300 pages of 128 slots (38400
# slots) and 150 pages of 256 slots, the two tokens_per_block values the DeepSeek-V4 cache
# manager accepts, against batches of 1..8192 rows. The engine's own pool sizes are certified
# further down.
POOL_PAGES = {128: 300, 256: 150}


def _pool_slots(t: int, num_slots: int, seed: int, skips: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots = torch.randperm(num_slots, device="cuda", generator=g)[:t].to(torch.int64)
    if skips:
        slots[torch.randperm(t, device="cuda", generator=g)[:skips]] = -1
    return slots


# ─── certified cells ──────────────────────────────────────────────────────────


def _snapshot(t: torch.Tensor) -> torch.Tensor:
    return t.clone() if t.dtype == torch.bfloat16 else t.view(torch.uint8).clone().view(t.dtype)


def _build(spec, seed: int):
    """A cell's spec -> (x, out, slot_mapping, out_scale, extra).

    `extra` carries what the structural checks need: `untouched` (regions of a wider parent outside
    `out` that must keep their bytes), `decodes_to_container` (decode the native slots in torch and
    hold them to the bf16 container) and `floor_byte` (the 2^-22 floor block must be present).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    kind, t = spec["kind"], spec.get("rows", 0)
    w = WINDOW_KV_WIDTH
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    slots = out_scale = None
    extra = {}
    if kind == "container":
        x = _window_kv_like(t, w=spec["width"], seed=seed)
        out = torch.empty_like(x)
    elif kind == "in_place":
        x = _window_kv_like(t, seed=seed)
        out = x
    elif kind in ("x_column_slice", "x_3d"):
        base = _window_kv_like(t, w=1024, seed=seed)
        if kind == "x_column_slice":
            x = base[:, 256:768]
            assert x.stride() == (1024, 1)
        else:
            x = base[:, None, :512]
        out = torch.empty(t, w, **bf16)
    elif kind == "scatter":
        ps = spec["page_size"]
        pages = POOL_PAGES[ps]
        x = _window_kv_like(t, seed=seed)
        slots = _pool_slots(t, pages * ps, seed=seed, skips=t // 8).to(spec["slot_dtype"])
        out = torch.randn(pages, ps, w, device="cuda", generator=g).to(torch.bfloat16)
    elif kind in ("pool_left_half", "pool_column_slice"):
        pages, page = 40, 128
        x = _window_kv_like(t, seed=seed)
        slots = _pool_slots(t, pages * page, seed=seed, skips=50)
        if kind == "pool_left_half":
            wide = torch.randn(pages, page, 2 * w, device="cuda", generator=g).to(torch.bfloat16)
            out = wide[..., :w]
            extra["untouched"] = [wide[..., w:]]
        else:
            slots = slots.to(torch.int32)
            flat_wide = torch.randn(pages * page, 640, device="cuda", generator=g).to(
                torch.bfloat16
            )
            out = flat_wide[:, 64:576]
            extra["untouched"] = [flat_wide[:, :64], flat_wide[:, 576:]]
    elif kind == "out_of_range":
        x = _window_kv_like(t, seed=seed)
        out = torch.randn(2, 128, w, device="cuda", generator=g).to(torch.bfloat16)
        slots = torch.tensor(
            [0, 256, 257, 10**9, -1, -(10**9), 255, 3], dtype=torch.int64, device="cuda"
        )
    elif kind == "native_interleaved":
        # [slots, 528] uint8 rows: 512 code bytes then 16 scale bytes.
        x = _window_kv_like(t, seed=seed)
        num_slots = 2 * t + 64
        slots = _pool_slots(t, num_slots, seed=seed, skips=t // 8)
        rows = torch.randint(
            0, 256, (num_slots, 528), dtype=torch.uint8, device="cuda", generator=g
        )
        out, out_scale = rows[:, :512], rows[:, 512:]
        extra["decodes_to_container"] = True
    elif kind == "native_footer":
        # Page = page_size code rows (512 B) then page_size scale rows (16 B), carved from one uint8 pool.
        ps, pages = spec["page_size"], 24
        x = _window_kv_like(t, seed=seed)
        slots = _pool_slots(t, pages * ps, seed=seed, skips=100).to(torch.int32)
        buf = torch.randint(
            0, 256, (pages, ps * 528), dtype=torch.uint8, device="cuda", generator=g
        )
        out = buf[:, : ps * 512].view(pages, ps, 512).view(torch.float8_e4m3fn)
        out_scale = buf[:, ps * 512 :].view(pages, ps, 16).view(torch.float8_e8m0fnu)
    elif kind == "native_identity":
        x = _window_kv_like(t, seed=seed)
        out = torch.empty(t, 512, dtype=torch.float8_e4m3fn, device="cuda")
        out_scale = torch.empty(t, 16, dtype=torch.uint8, device="cuda")
        extra["decodes_to_container"] = True
        extra["floor_byte"] = True
    else:
        assert kind == "zero_rows", kind
        x = torch.empty(0, w, **bf16)
        out = torch.randn(4, 128, w, device="cuda", generator=g).to(torch.bfloat16)
        if spec["mapped"]:
            slots = torch.empty(0, dtype=torch.int32, device="cuda")
    return x, out, slots, out_scale, extra


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, out, slots, out_scale, extra = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x_before = x.clone()
    out_before = _snapshot(out)
    scale_before = None if out_scale is None else _snapshot(out_scale)
    untouched = [(u, u.clone()) for u in extra.get("untouched", [])]
    result = op(x, out, slots, out_scale)
    if out_scale is None:
        assert result is out, "the bf16 form writes out in place and returns it"
    else:
        assert result[0] is out and result[1] is out_scale, (
            "the native form returns (out, out_scale)"
        )
    op.compare(result, op.reference(x_before, out_before, slots, scale_before))
    if out is not x:
        _assert_bitwise(x, x_before)
    for u, before in untouched:
        _assert_bitwise(u, before)
    if extra.get("decodes_to_container"):
        # native dequantization e4m3(code) * 2^(byte - 127) is bit-identical to the bf16 container
        y = _fake_quant(x_before)[0]
        s = slots.long() if slots is not None else torch.arange(x.shape[0], device="cuda")
        live = (s >= 0) & (s < out.shape[0])
        codes = out.view(torch.uint8)[s[live]].view(torch.float8_e4m3fn).float().view(-1, 16, 32)
        scale = torch.exp2(out_scale[s[live]].view(torch.uint8).float() - 127).unsqueeze(-1)
        _assert_bitwise((codes * scale).view(-1, 512).to(torch.bfloat16), y[live])
    if extra.get("floor_byte"):
        assert (out_scale == 105).any(), "the test data must contain floor-scaled (2^-22) blocks"


# ─── non-finite input, CUDA graph ─────────────────────────────────────────────


def test_non_finite_row_is_isolated() -> None:
    """A NaN/inf row (e.g. CUDA-graph padding garbage) changes no other row."""
    x = _window_kv_like(64, seed=5)
    x[7, 3] = float("nan")
    x[9, 100] = float("inf")
    x[11, :] = float("-inf")
    out = torch.empty_like(x)
    op(x, out)
    keep = torch.ones(64, dtype=torch.bool, device="cuda")
    keep[[7, 9, 11]] = False
    _assert_bitwise(out[keep], _fake_quant(x[keep].contiguous())[0])


def test_cuda_graph_capture_replay() -> None:
    """Captured after a warm-up call (Triton compiles on first use), replayed on new rows and new slots."""
    t, pages, page = 64, 16, 128
    x = torch.empty(t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
    slots = torch.empty(t, dtype=torch.int32, device="cuda")
    pool = torch.zeros(pages, page, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
    x.copy_(_window_kv_like(t, seed=1))
    slots.copy_(_pool_slots(t, pages * page, seed=1, skips=4))
    op(x, pool, slots)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, pool, slots)

    for seed in (2, 3):
        new_x = _window_kv_like(t, seed=seed)
        new_slots = _pool_slots(t, pages * page, seed=seed, skips=4)
        pool.zero_()
        x.copy_(new_x)
        slots.copy_(new_slots)
        graph.replay()
        torch.cuda.synchronize()
        op.compare(pool, op.reference(new_x, torch.zeros_like(pool), new_slots))


# ─── deliberately-wrong controls ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(dict(floor=None), id="no_1e-4_floor"),
        pytest.param(dict(rounding=torch.floor), id="floor_not_ceil_scale"),
        pytest.param(dict(block=64), id="block_64"),
        pytest.param(dict(floor=None, scale_floor_exp=-13), id="scale_floored_at_2^-13"),
    ],
)
def test_wrong_recipes_are_rejected(variant: dict) -> None:
    """Each plausible wrong recipe differs from the op on this data, so the bitwise gate can see it.

    `scale_floored_at_2^-13` floors max(amax/448, 1e-4) instead of amax: a
    different value set on every block with amax < 448 * 2^-14.
    """
    x = _window_kv_like(4096, seed=99)
    out = torch.empty_like(x)
    op(x, out)
    op.compare(out, op.reference(x, torch.empty_like(x)))
    wrong = _control_quant(x, **variant)
    n_diff = (_bits(out) != _bits(wrong)).sum().item()
    print(f"{variant}: {n_diff} of {x.numel()} elements differ from the op")
    assert n_diff > 0
    with pytest.raises(AssertionError, match="differ bitwise"):
        op.compare(out, wrong)


# ─── rejected inputs (the op's own checks) ────────────────────────────────────


def _raises(match: str, *args) -> None:
    with pytest.raises(ValueError, match=match):
        op(*args)


def test_rejected_inputs() -> None:
    x = _window_kv_like(16, seed=4)
    out = torch.empty_like(x)
    pool = torch.empty(2, 128, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
    codes = torch.empty(16, 512, dtype=torch.uint8, device="cuda")
    scales = torch.empty(16, 16, dtype=torch.uint8, device="cuda")
    _raises("x must be bfloat16", x.float(), out)
    _raises("x must be bfloat16", x.half(), out)
    _raises("x must be a CUDA tensor", x.cpu(), out)
    _raises("out must be on cuda", x, out.cpu())
    _raises("positive multiple of 32", x[:, :48], out[:, :48])
    _raises("unit stride along its last dim", x.t().contiguous().t(), out)
    _raises("rank >= 2", x[0], out)
    _raises("out's last dim must be 512", x, out[:, :256])
    _raises("out's last dim must be 512 with unit stride", x, out.t().contiguous().t())
    _raises("out must be bfloat16, float8_e4m3fn or uint8", x, out.float(), None, scales)
    _raises("needs out_scale", x, codes)
    _raises("a bfloat16 out takes None", x, out, None, scales)
    _raises("out_scale must be", x, codes, None, scales[:, :8])
    _raises("out_scale must be uint8 or float8_e8m0fnu", x, codes, None, scales.to(torch.int32))
    _raises("slot_mapping must be int32 or int64", x, pool, torch.zeros(16, device="cuda"))
    _raises(
        "slot_mapping must be \\[16\\]", x, pool, torch.zeros(15, dtype=torch.int32, device="cuda")
    )
    _raises("needs >= 16 slots", x, out[:8])
    _raises("must be \\[slots, C\\] or \\[pages, page_size, C\\]", x, pool.view(2, 1, 128, 512))


# ─── the largest SWA pool an engine can allocate ──────────────────────────────

# The SWA pool (one pool shared by all 40 layers) is sized by the V4.1
# DeepseekV4CacheManager from the per-rank KV quota, which an engine derives as
# kv_cache_fraction (0.5 at the gates) x the device's FREE memory after weights
# and activations. Free memory never exceeds total memory, so a manager built at
# 0.5 x the device's TOTAL memory (148,510,474,240 bytes on this GB300) allocates
# the largest SWA pool an engine of that configuration can have. At the gate
# settings (max_seq_len 16384, max_num_tokens 8192, max_batch_size {8, 32},
# DSpark or not; test_largest_engine_pools_from_the_cache_manager) the largest
# is [491000, 128, 512] bf16. Over the engine envelope (max_num_tokens
# 2048..16384, max_batch_size up to 256, max_seq_len up to 1,048,576, DSpark or
# not) the unsplit manager's largest is [730880, 128, 512] bf16 at
# (max_num_tokens 16384, max_batch_size 1, no DSpark), whatever max_seq_len (the
# pool scan; test_envelope_largest_swa_pools_from_the_cache_manager re-measures it).
# A PP2 stage manager holds only its 20 layers' pools, so its SWA pool is larger:
# stage 1 alone reaches [958720, 128, 512] at (16384, 2, 0, 16384) (the PP2
# stage scan; test_envelope_pp2_stage_swa_pools_from_the_cache_manager re-measures it):
# 122,716,160 slots, 6.3e10 elements, 125.7 GB. That stage-alone maximum is
# ENGINE_SWA_PAGES. test_pool_size_invariance shows the op's result does not
# depend on B, so the far-end cases below run at that B_max. Past slot 4,194,304
# every element offset exceeds 2^31, so any 32-bit
# offset in the kernel's address arithmetic would land a row somewhere else (or
# fault). The pool is allocated at that exact shape, with one guard page past
# its end, and every element is set to a NaN bit pattern the op never writes
# for a finite row: after the call, the named slots must hold the reference and
# every other element of the pool and the guard page must still hold the
# sentinel.
GATES_SWA_PAGES = 491000
UNSPLIT_SWA_PAGES = 730880
ENGINE_SWA_PAGES = 958720
ENGINE_PAGE = 128
SENTINEL = 0x7FA5  # bf16 NaN bit pattern (as int16)


def _hazard_slots(num_slots: int, width: int, page_size: int) -> list[int]:
    """Distinct slots on both sides of every 32-bit offset hazard of a [slots, width] bf16 pool, and both ends.

    Element offsets 2^30 (2^31 bytes), 2^31 (2^32 bytes), 2^32, 2^33, 2^34, 2^35
    and 2^36 (where inside the pool), the first page, the last page and the last slot.
    """
    cands = {0, 1, page_size - 1, page_size}
    for limit in (2**30, 2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // width
        cands |= {s - 1, s}
    cands |= {num_slots - page_size - 1, num_slots - page_size, num_slots - 2, num_slots - 1}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _engine_pool(pages: int, page_size: int, width: int = WINDOW_KV_WIDTH):
    """(buffer with one guard page, [pages, page_size, width] bf16 pool view), every element SENTINEL."""
    buf = torch.empty(pages + 1, page_size, width, dtype=torch.bfloat16, device="cuda")
    buf.view(torch.int16).fill_(SENTINEL)
    return buf, buf[:pages]


def _check_engine_pool(
    buf: torch.Tensor, pool: torch.Tensor, slots: torch.Tensor, y: torch.Tensor
) -> None:
    """Named slots hold y bitwise; every other element of the pool and the guard page is still SENTINEL.

    Restores the named slots to SENTINEL, so the pool can be reused.
    """
    num_slots = pool.shape[0] * pool.shape[1]
    flat = pool.view(num_slots, -1)
    s64 = slots.to(torch.int64)
    live = (s64 >= 0) & (s64 < num_slots)
    idx = s64[live]
    _assert_bitwise(flat[idx], y[live])
    flat.view(torch.int16)[idx] = SENTINEL
    whole = buf.view(torch.int16).view(-1)
    step = 1 << 30
    n_bad = sum(int((whole[i : i + step] != SENTINEL).sum()) for i in range(0, whole.numel(), step))
    assert n_bad == 0, f"{n_bad} elements outside the named slots were written"


def _free_engine_pool() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_engine_swa_pool_far_end(slot_dtype: torch.dtype) -> None:
    """Decode-like: one row per 32-bit hazard boundary and at both ends of the engine-size pool, plus skips."""
    buf, pool = _engine_pool(ENGINE_SWA_PAGES, ENGINE_PAGE)
    num_slots = ENGINE_SWA_PAGES * ENGINE_PAGE
    hazard = _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
    assert hazard[-1] == num_slots - 1 and 2**35 // WINDOW_KV_WIDTH in hazard
    # -1 and the first two slots past the end (which would land in the guard page) are skipped.
    slots = torch.tensor(
        hazard + [-1, num_slots, num_slots + ENGINE_PAGE - 1], dtype=torch.int64, device="cuda"
    )
    x = _window_kv_like(slots.numel(), seed=31)
    op(x, pool, slots.to(slot_dtype))
    _check_engine_pool(buf, pool, slots, _fake_quant(x)[0])
    del buf, pool
    _free_engine_pool()


def test_engine_swa_pool_prefill_spread() -> None:
    """Prefill-like, T = 32768 (the decoder pass's largest step): the last page written in order, every hazard
    slot, the rest spread over the pool."""
    t = _MAX_ROWS
    buf, pool = _engine_pool(ENGINE_SWA_PAGES, ENGINE_PAGE)
    num_slots = ENGINE_SWA_PAGES * ENGINE_PAGE
    last_page = torch.arange(num_slots - ENGINE_PAGE, num_slots, device="cuda")
    hazard = torch.tensor(_hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE), device="cuda")
    fixed = torch.unique(torch.cat([last_page, hazard]))
    g = torch.Generator(device="cuda").manual_seed(8192)
    perm = torch.randperm(num_slots, device="cuda", generator=g)
    rest = perm[~torch.isin(perm, fixed)][: t - fixed.numel()]
    slots = torch.cat([last_page, fixed[~torch.isin(fixed, last_page)], rest]).to(torch.int64)
    assert slots.numel() == t and torch.unique(slots).numel() == t
    slots[torch.randperm(t, device="cuda", generator=g)[: t // 16]] = -1
    x = _fast_rows(t, seed=8192)
    op(x, pool, slots.to(torch.int32))
    _check_engine_pool(buf, pool, slots, _fake_quant(x)[0])
    del buf, pool
    _free_engine_pool()


def test_engine_swa_pool_cuda_graph_far_end() -> None:
    """Captured on the engine-size pool after a warm-up call, replayed with far-end and boundary slots."""
    t = 32
    buf, pool = _engine_pool(ENGINE_SWA_PAGES, ENGINE_PAGE)
    num_slots = ENGINE_SWA_PAGES * ENGINE_PAGE
    hazard = _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
    x = torch.empty(t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
    slots = torch.empty(t, dtype=torch.int32, device="cuda")
    x.copy_(_window_kv_like(t, seed=40))
    slots.copy_(torch.arange(t, device="cuda"))
    op(x, pool, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, pool, slots)
    buf.view(torch.int16).fill_(SENTINEL)

    g = torch.Generator(device="cuda").manual_seed(41)
    hz = torch.tensor(hazard, device="cuda")
    near_end = num_slots - 1 - torch.arange(4 * ENGINE_PAGE, device="cuda")
    near_end = near_end[~torch.isin(near_end, hz)]
    for seed in (42, 43):
        pick = torch.randperm(near_end.numel(), device="cuda", generator=g)[: t - len(hazard) - 1]
        new_slots = torch.cat([hz, torch.tensor([-1], device="cuda"), near_end[pick]]).to(
            torch.int64
        )
        assert torch.unique(new_slots).numel() == t
        new_x = _window_kv_like(t, seed=seed)
        x.copy_(new_x)
        slots.copy_(new_slots)
        graph.replay()
        torch.cuda.synchronize()
        _check_engine_pool(buf, pool, new_slots, _fake_quant(new_x)[0])
    del graph, buf, pool
    _free_engine_pool()


def test_engine_swa_pool_harness_sees_a_wrapped_offset() -> None:
    """Control: the far-end check fails when rows land where a 32-bit (mod 2^32) element offset would put them.

    The op is handed the wrapped slots and the checker the true ones; both the
    named-slot comparison and the outside-the-slots sentinel scan must see it.
    """
    buf, pool = _engine_pool(ENGINE_SWA_PAGES, ENGINE_PAGE)
    num_slots = ENGINE_SWA_PAGES * ENGINE_PAGE
    true = torch.tensor(
        [
            s
            for s in _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
            if s * WINDOW_KV_WIDTH >= 2**32
        ],
        dtype=torch.int64,
        device="cuda",
    )
    wrapped = (true * WINDOW_KV_WIDTH % 2**32) // WINDOW_KV_WIDTH
    assert true.numel() >= 5 and not torch.isin(wrapped, true).any()
    x = _window_kv_like(true.numel(), seed=51)
    y = _fake_quant(x)[0]
    op(x, pool, wrapped)
    with pytest.raises(AssertionError, match="elements differ bitwise"):
        _check_engine_pool(buf, pool, true, y)
    flat = pool.view(num_slots, -1)
    flat.view(torch.int16)[true] = SENTINEL
    with pytest.raises(AssertionError, match="outside the named slots were written"):
        _check_engine_pool(buf, pool, torch.tensor([], dtype=torch.int64, device="cuda"), y[:0])
    del buf, pool, flat
    _free_engine_pool()


# ─── every row count 1..32768 in the core target's paged-scatter call ────────
#
# The DeepSeek-V4.1 core target writes each layer's window KV with
# mxfp8_fake_quantize(kv [T, 512], swa_pool [pages, 128, 512], slots[:T]),
# where T is the rank's token count this step (any integer up to the engine's
# max_num_tokens, at most 16384; in the decoder bounded-replay pass of layers
# 21-39 up to 128 x 256 = 32768 rows) and slots is a prefix view of one row of
# the metadata's int64 [num_local_layers, max_num_tokens] slot table (-1 = no
# slot). Rows reach the kernel only as the size of grid axis 0: program r loads
# row r and slot_mapping[r] and writes one slot; no kernel argument carries T.

import tensorrt_llm._torch.custom_ops.mxfp8_fake_quantize as _mx_op  # noqa: E402 (the Triton kernel object, for the variant probe)

_MAX_ROWS = 32768
_SWEEP_PAGES = 320  # 40960 slots of 128: plainly larger than the 32768 rows swept


def _jit_variants(fn) -> int:
    """Number of compiled variants Triton holds for a @triton.jit function, over all devices."""
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


def _sweep_slots(t: int, num_slots: int, seed: int) -> torch.Tensor:
    """Row 1 of an int64 [2, t] table (the metadata's layout): distinct random slots, 1/16 of them -1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.full((2, t), -1, dtype=torch.int64, device="cuda")
    slots = table[1]
    slots.copy_(torch.randperm(num_slots, device="cuda", generator=g)[:t])
    slots[torch.randperm(t, device="cuda", generator=g)[: t // 16]] = -1
    slots[1] = -1  # a skipped row inside the smallest prefixes too
    return slots


def test_every_row_count_1_to_32768_in_the_paged_scatter_form() -> None:
    """Every prefix (x[:r], slots[:r]), r = 1..32768, writes exactly the first r rows of the 32768-row call.

    The 32768-row call is held bitwise to the native reference and to the
    sentinel scan. Then, into the same sentinel-filled pool, the op runs at
    every r in increasing order. After each call the slots named by the live
    entries of slots[:r] hold the reference rows bit for bit, and the number of
    pool elements that are not the sentinel is exactly 512 per live row: a
    skipped row, a row written to another slot, or a write past row r - 1 all
    change one of the two counts. The mismatch counts accumulate on the device;
    one sync at the end. The probe: rows are only grid axis 0, so the sweep must
    compile no Triton variant beyond the one the 32768-row call compiled.
    """
    t = _MAX_ROWS
    num_slots = _SWEEP_PAGES * ENGINE_PAGE
    buf, pool = _engine_pool(_SWEEP_PAGES, ENGINE_PAGE)
    flat = pool.view(num_slots, -1).view(torch.int16)
    whole = buf.view(torch.int16)
    slots = _sweep_slots(t, num_slots, seed=8193)
    assert slots.stride(0) == 1
    x = _window_kv_like(t, seed=8193)
    y = _fake_quant(x)[0]

    kernel = _mx_op._mxfp8_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    op(x, pool, slots)
    n1 = _jit_variants(kernel)
    _check_engine_pool(
        buf, pool, slots, y
    )  # the N_max call against the reference; restores the sentinel

    live = slots >= 0
    live_pos = torch.nonzero(live).squeeze(1)
    live_slots = slots[live_pos]
    y_live = y[live_pos].view(torch.int16)
    cnt = torch.cumsum(live.to(torch.int64), 0).tolist()
    bad_rows = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_count = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(1, t + 1):
        op(x[:r], pool, slots[:r])
        k = cnt[r - 1]
        bad_rows += (flat[live_slots[:k]] != y_live[:k]).sum()
        bad_count += ((whole != SENTINEL).sum() - k * WINDOW_KV_WIDTH).abs()
    n2 = _jit_variants(kernel)
    print(
        f"\n[rows 1..{t}] live rows {cnt[-1]}; Triton variants: {n0} before the {t}-row call, {n1} after, "
        f"{n2} after the sweep; element mismatches {int(bad_rows)}, non-sentinel count error {int(bad_count)}"
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


def test_row_sweep_harness_sees_a_row_or_slot_shift() -> None:
    """Control for the sweep: its two checks reject a read one row off and a write one slot off, at every r tried."""
    num_slots = _SWEEP_PAGES * ENGINE_PAGE
    buf, pool = _engine_pool(_SWEEP_PAGES, ENGINE_PAGE)
    flat = pool.view(num_slots, -1).view(torch.int16)
    whole = buf.view(torch.int16)
    g = torch.Generator(device="cuda").manual_seed(66)
    slots = torch.randperm(num_slots, device="cuda", generator=g)[:66].to(
        torch.int64
    )  # no -1: every row lands
    x = _window_kv_like(66, seed=66)
    y = _fake_quant(x)[0].view(torch.int16)
    for r in (1, 2, 7, 63, 64):
        for name, xa, sa in (
            ("row +1", x[1 : r + 1], slots[:r]),
            ("slot +1", x[:r], slots[1 : r + 1]),
        ):
            whole.fill_(SENTINEL)
            op(xa, pool, sa)
            n_bad = int((flat[slots[:r]] != y[:r]).sum())
            n_count = int((whole != SENTINEL).sum()) - r * WINDOW_KV_WIDTH
            print(
                f"[control {name}] r={r}: {n_bad} / {r * WINDOW_KV_WIDTH} elements differ, count error {n_count}"
            )
            assert n_bad > r * WINDOW_KV_WIDTH // 2
    del buf, pool, flat, whole
    _free_engine_pool()


# ─── CUDA-graph capture at every graph batch the engines capture ─────────────
#
# The engine envelope captures every graph batch G of _ENVELOPE_GSET: every G
# 1..32 (padding off captures every batch up to max_batch_size) and TRT-LLM's
# padded list for cuda_graph_config max_batch_size 256 (1, 2, 4, 8..128 step 8,
# 192, 256). The target's SWA write runs T = G (decode) and T = 6G (DSpark
# target verify: 1 + 5 drafts per request) rows under capture; the drafter
# writes its stage rings at M = G (bonus) and 5G (back-fill) and its draft
# block at N = 5G (the ring and block tests below). _GRAPH_BS is the gates'
# earlier set {1..8, 16, 24, 32}, kept for the 99-page ring case.
_ENVELOPE_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
assert len(_ENVELOPE_GSET) == 46
_GRAPH_BS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)
_SWA_GRAPH_T = tuple(
    sorted(set(_ENVELOPE_GSET) | {6 * g for g in _ENVELOPE_GSET})
)  # 82 row counts, 1..1536
_RING_GRAPH_T = tuple(sorted({3 * b for b in _GRAPH_BS} | {5 * b for b in _GRAPH_BS}))
# The DSpark worker's ring, _kv_windows [max_bs + 1, 3, 128, 512] bf16 at max_bs 32,
# viewed as [(32 + 1) * 3, 128, 512].
_RING_PAGES = (32 + 1) * 3


def _far_end_slots(num_slots: int, page_size: int, t: int, g: torch.Generator) -> torch.Tensor:
    """t distinct slots: the pool's last slot, then a random pick from the 32-bit hazard slots and the last
    4 pages (and, when t needs more, slots spread uniformly over the pool); for t >= 2 one of them (not the
    last slot) replaced by -1."""
    last = num_slots - 1
    hazard = torch.tensor(
        [s for s in _hazard_slots(num_slots, WINDOW_KV_WIDTH, page_size) if s != last],
        device="cuda",
    )
    near_end = last - 1 - torch.arange(4 * page_size - 1, device="cuda")
    near_end = near_end[~torch.isin(near_end, hazard)]
    rest = torch.cat([hazard, near_end])
    rest = rest[torch.randperm(rest.numel(), device="cuda", generator=g)]
    if rest.numel() < t - 1:
        spread = torch.randint(0, last, (4 * t,), device="cuda", generator=g).unique()
        spread = spread[~torch.isin(spread, rest)]
        rest = torch.cat([rest, spread[torch.randperm(spread.numel(), device="cuda", generator=g)]])
    slots = torch.cat([torch.tensor([last], device="cuda"), rest[: t - 1]]).to(torch.int64)
    assert slots.numel() == t and torch.unique(slots).numel() == t
    if t >= 2:
        slots[1 + int(torch.randint(0, t - 1, (1,), device="cuda", generator=g))] = -1
    return slots


def _capture_sweep(pages: int, page_size: int, ts, seed: int) -> None:
    """Per T: eager warm-up at that shape, capture, then 2 replays on new rows and new far-end slots.

    Every replay must write exactly the reference into the named slots and
    nothing else (sentinel scan of the whole pool plus a guard page), and equal
    bit for bit the same call run eagerly on the same inputs.
    """
    buf, pool = _engine_pool(pages, page_size)
    num_slots = pages * page_size
    flat = pool.view(num_slots, -1)
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.full(
        (2, max(ts)), -1, dtype=torch.int64, device="cuda"
    )  # the metadata's int64 slot table
    for t in ts:
        x = torch.empty(t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        slots = table[1, :t]
        x.copy_(_window_kv_like(t, seed=seed + t))
        slots.copy_(_far_end_slots(num_slots, page_size, t, g))
        op(x, pool, slots)  # eager warm-up at this shape, as the engine's warm-up step runs it
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, pool, slots)
        buf.view(torch.int16).fill_(SENTINEL)
        for rep in (1, 2):
            new_x = _window_kv_like(t, seed=seed + 1000 * rep + t)
            new_slots = _far_end_slots(num_slots, page_size, t, g)
            x.copy_(new_x)
            slots.copy_(new_slots)
            graph.replay()
            torch.cuda.synchronize()
            live = new_slots[new_slots >= 0]
            replayed = flat[live].view(torch.int16).clone()
            ref = _fake_quant(new_x)[0]
            _check_engine_pool(buf, pool, new_slots, ref)
            op(x, pool, slots)  # the same call, eagerly, on the same inputs
            torch.cuda.synchronize()
            eager = flat[live].view(torch.int16).clone()
            _check_engine_pool(buf, pool, new_slots, ref)
            assert torch.equal(replayed, eager), (
                f"T={t} replay {rep}: {int((replayed != eager).sum())} elements"
            )
        del graph
    print(
        f"\n[graphs on [{pages}, {page_size}, 512]] T = {list(ts)}: every replay bitwise == eager == reference"
    )
    del buf, pool, flat
    _free_engine_pool()


def test_cuda_graph_every_graph_batch_on_the_engine_swa_pool() -> None:
    """T = G (decode) and 6G (DSpark verify) for every G of _ENVELOPE_GSET (82 row counts, 1..1536), on the B_max
    bf16 SWA pool [958720, 128, 512] (a PP2 stage alone)."""
    assert len(_SWA_GRAPH_T) == 82 and max(_SWA_GRAPH_T) == 1536
    _capture_sweep(ENGINE_SWA_PAGES, ENGINE_PAGE, _SWA_GRAPH_T, seed=700)


def test_cuda_graph_dspark_ring_and_block_rows() -> None:
    """T in 3 x and 5 x {1..8, 16, 24, 32} (stage-ring rows, draft block) on a ring-shaped [99, 128, 512] pool."""
    _capture_sweep(_RING_PAGES, ENGINE_PAGE, _RING_GRAPH_T, seed=900)


# ─── the DSpark drafter's stage-ring writes, in the drafter's own form ───────
#
# The drafter's _ring_write writes each stage s = 0, 1, 2 with
# mxfp8_fake_quantize(kv_s [M, 512], ring3, idx3[:, s]): ring3 is the worker's
# _kv_windows [max_batch_size + 1, 3, 128, 512] bf16 viewed [(max_batch_size + 1)
# * 3, 128, 512], and the slots are column s of an int64 [M, 3] table (a stride-3
# view: split(idx3, 1, 1)[s] reshaped to [M]) holding (ring_row * 3 + s) * 128 +
# (position & 127), or -1 for a masked row. M = G on every draft step's bonus
# write (forward_batched) and 5G on the multi-accept back-fill
# (write_context_windows_batched), both captured; n <= 128 on the prefill
# seeding (write_context_windows, eager), where ring3 is one ring row's
# [3, 128, 512] view. Triton specializes an integer argument equal to 1 and a
# pointer's 16-byte alignment, so the stride-3 slot map runs different compiled
# variants from the target's contiguous slot rows (stage 1's column starts 8
# bytes into the table: two variants over the three stages): these tests run them.
_ENVELOPE_RING_ROWS = 256 + 1  # the worker's ring rows at max_batch_size 256
_ENVELOPE_RING_PAGES = 3 * _ENVELOPE_RING_ROWS  # 771
_ENVELOPE_RING_M = tuple(
    sorted(set(_ENVELOPE_GSET) | {5 * g for g in _ENVELOPE_GSET})
)  # 83 row counts, 1..1280


def _ring_idx3(m: int, ring_rows: int, g: torch.Generator, masked: bool = True) -> torch.Tensor:
    """int64 [m, 3] like the drafter's idx3: G requests on distinct ring rows (the last row among them),
    k = m // G consecutive positions each (k = 1, the bonus write, while m < ring_rows; else k = 5, the
    back-fill), entry (row * 3 + s) * 128 + (position & 127); with `masked`, m // 8 whole rows -1."""
    k = 1 if m < ring_rows else 5
    assert m % k == 0 and m // k < ring_rows, (m, k)
    n_req = m // k
    rows = torch.cat(
        [
            torch.tensor([ring_rows - 1], device="cuda"),
            torch.randperm(ring_rows - 1, device="cuda", generator=g)[: n_req - 1],
        ]
    )
    p0 = torch.randint(0, 2**20, (n_req, 1), device="cuda", generator=g)
    off = (p0 + torch.arange(k, device="cuda").view(1, k)) & (ENGINE_PAGE - 1)  # [G, k]
    stage = torch.arange(3, device="cuda").view(1, 1, 3)
    idx3 = (
        ((rows.view(-1, 1, 1) * 3 + stage) * ENGINE_PAGE + off.unsqueeze(-1))
        .reshape(m, 3)
        .contiguous()
    )
    if masked and m >= 8:
        idx3[torch.randperm(m, device="cuda", generator=g)[: m // 8]] = -1
    return idx3.to(torch.int64)


def _ring_cols(idx3: torch.Tensor) -> list[torch.Tensor]:
    """The drafter's per-stage slot maps: split(idx3, 1, 1)[s] reshaped to [M], stride-3 views."""
    cols = [c.reshape(idx3.shape[0]) for c in torch.split(idx3, 1, 1)]
    assert idx3.shape[0] < 2 or all(
        c.stride(0) == 3 and c.data_ptr() == idx3.data_ptr() + 8 * s for s, c in enumerate(cols)
    )
    return cols


def test_envelope_dspark_ring_write_captured_at_every_graph_batch() -> None:
    """The drafter's ring write captured at M = G and 5G for every G of _ENVELOPE_GSET (83 row counts, 1..1280).

    As the drafter does it: three calls per graph (stage 0, 1, 2), each with its own contiguous [M, 512]
    input and a stride-3 column of one int64 [M, 3] table, into ring3 = the [257, 3, 128, 512] ring (max
    batch 256) viewed [771, 128, 512], with a guard page past it. Per M: an eager warm-up (the three calls),
    the capture, then 2 replays after new rows and a new table (new ring rows and positions, the ring's
    last row among them, 1/8 of the rows masked) are written in place. Every replay writes the reference
    into the named slots of every stage and nothing else (the whole ring and its guard page hold the
    sentinel), and equals the same three calls run eagerly.
    """
    buf, ring3 = _engine_pool(_ENVELOPE_RING_PAGES, ENGINE_PAGE)
    num_slots = _ENVELOPE_RING_PAGES * ENGINE_PAGE
    flat = ring3.view(num_slots, -1)
    g = torch.Generator(device="cuda").manual_seed(9100)
    graphs = 0
    for m in _ENVELOPE_RING_M:
        xs = [
            torch.empty(m, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda") for _ in range(3)
        ]
        idx3 = torch.empty(m, 3, dtype=torch.int64, device="cuda")
        cols = _ring_cols(idx3)
        for s in range(3):
            xs[s].copy_(_fast_rows(m, seed=9100 + 3 * m + s))
        idx3.copy_(_ring_idx3(m, _ENVELOPE_RING_ROWS, g))
        for s in range(3):  # eager warm-up at this shape
            op(xs[s], ring3, cols[s])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for s in range(3):
                op(xs[s], ring3, cols[s])
        graphs += 1
        buf.view(torch.int16).fill_(SENTINEL)
        for rep in (1, 2):
            new_x = [_fast_rows(m, seed=9100 + 100_000 * rep + 3 * m + s) for s in range(3)]
            new_idx = _ring_idx3(m, _ENVELOPE_RING_ROWS, g)
            for s in range(3):
                xs[s].copy_(new_x[s])
            idx3.copy_(new_idx)
            graph.replay()
            torch.cuda.synchronize()
            all_slots = new_idx.t().reshape(-1)
            live = all_slots[all_slots >= 0]
            assert torch.unique(live).numel() == live.numel()
            ref = torch.cat([_fake_quant(new_x[s])[0] for s in range(3)])
            replayed = flat[live].view(torch.int16).clone()
            _check_engine_pool(buf, ring3, all_slots, ref)
            for s in range(3):  # the same three calls, eagerly, on the same inputs
                op(xs[s], ring3, cols[s])
            torch.cuda.synchronize()
            eager = flat[live].view(torch.int16).clone()
            _check_engine_pool(buf, ring3, all_slots, ref)
            assert torch.equal(replayed, eager), (
                f"M={m} replay {rep}: {int((replayed != eager).sum())} elements"
            )
        del graph
    print(
        f"\n[dspark ring writes on [{_ENVELOPE_RING_PAGES}, 128, 512], stride-3 slot columns] {graphs} graphs of 3 "
        f"calls, M = {list(_ENVELOPE_RING_M)}: every replay bitwise == eager == reference, nothing else written"
    )
    assert graphs == 83 and max(_ENVELOPE_RING_M) == 1280
    del buf, ring3, flat
    _free_engine_pool()


def test_envelope_dspark_ring_write_every_row_count() -> None:
    """Every row count of the drafter's stride-3 ring write, eager: every M in 1..1280 (5 x 256, the largest the
    ring write takes), and the prefill seeding's n in 1..128 into one ring row's [3, 128, 512] view.

    The 1280-row calls (three stages, a [1280, 3] back-fill table over 256 ring rows, the last among them,
    1/8 of the rows masked) are held to the reference and the sentinel scan of the whole ring plus a guard
    page. Then per M: the three calls with (x_s[:M], cols_s[:M]) (stride-3 prefix views); the live slots of
    the first M rows hold the reference bit for bit, the count of non-sentinel elements is 512 per live
    (row, stage), and the named slots are restored, so each M is checked on its own. The probe: after the
    first stride-3 call, no M compiles a new variant. Then the seeding form: for n in 1..128, the rows at
    positions p .. p + n - 1 of ring row 256 (idx3 = s * 128 + (p & 127) into kv_windows[256], a [3, 128,
    512] view), checked the same way over the whole ring.
    """
    m_max = 5 * 256
    buf, ring3 = _engine_pool(_ENVELOPE_RING_PAGES, ENGINE_PAGE)
    num_slots = _ENVELOPE_RING_PAGES * ENGINE_PAGE
    flat = ring3.view(num_slots, -1).view(torch.int16)
    whole = buf.view(torch.int16)
    g = torch.Generator(device="cuda").manual_seed(9200)
    idx3 = _ring_idx3(m_max, _ENVELOPE_RING_ROWS, g)
    cols = _ring_cols(idx3)
    xs = [_window_kv_like(m_max, seed=9200 + s) for s in range(3)]
    ys = [_fake_quant(x)[0].view(torch.int16) for x in xs]
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    for s in range(3):
        op(xs[s], ring3, cols[s])
    n1 = _jit_variants(kernel)
    all_slots = idx3.t().reshape(-1)
    _check_engine_pool(buf, ring3, all_slots, torch.cat([_fake_quant(x)[0] for x in xs]))

    bad_rows = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_count = torch.zeros((), dtype=torch.int64, device="cuda")
    live = idx3[:, 0] >= 0
    cnt = torch.cumsum(live.to(torch.int64), 0).tolist()
    for m in range(1, m_max + 1):
        for s in range(3):
            op(xs[s][:m], ring3, cols[s][:m])
        sel = live[:m]
        for s in range(3):
            bad_rows += (flat[cols[s][:m][sel]] != ys[s][:m][sel]).sum()
        bad_count += ((whole != SENTINEL).sum() - 3 * cnt[m - 1] * WINDOW_KV_WIDTH).abs()
        flat[idx3[:m][sel].reshape(-1)] = SENTINEL
    n2 = _jit_variants(kernel)

    # The prefill seeding: n rows at positions p .. p + n - 1 into ring row 256's [3, 128, 512] view.
    window = ring3.view(_ENVELOPE_RING_ROWS, 3, ENGINE_PAGE, WINDOW_KV_WIDTH)[
        _ENVELOPE_RING_ROWS - 1
    ]
    wflat = window.reshape(3 * ENGINE_PAGE, -1).view(torch.int16)
    assert window.data_ptr() == ring3[-3:].data_ptr()
    bad_seed = torch.zeros((), dtype=torch.int64, device="cuda")
    for n in range(1, ENGINE_PAGE + 1):
        p = 1000 + 37 * n
        off = (p + torch.arange(n, device="cuda")) & (ENGINE_PAGE - 1)
        widx = (
            off.view(n, 1) + ENGINE_PAGE * torch.arange(3, device="cuda").view(1, 3)
        ).contiguous()
        wcols = _ring_cols(widx)
        for s in range(3):
            op(xs[s][:n], window, wcols[s])
        for s in range(3):
            bad_seed += (wflat[wcols[s]] != ys[s][:n]).sum()
        bad_seed += ((whole != SENTINEL).sum() - 3 * n * WINDOW_KV_WIDTH).abs()
        wflat[widx.reshape(-1)] = SENTINEL
    n3 = _jit_variants(kernel)
    print(
        f"\n[dspark ring write, stride-3 slots, M 1..{m_max}] {cnt[-1]} live rows x 3 stages; variants {n0} -> "
        f"{n1} (first stride-3 calls) -> {n2} (M sweep) -> {n3} (seeding n 1..128); element mismatches "
        f"{int(bad_rows)}, count error {int(bad_count)}; seeding mismatches + count error {int(bad_seed)}"
    )
    assert int(bad_rows) == 0 and int(bad_count) == 0 and int(bad_seed) == 0
    assert n1 - n0 <= 1 and n3 == n2 == n1, (
        f"row counts compiled new variants: {n0} -> {n1} -> {n2} -> {n3}"
    )
    del buf, ring3, flat, whole, window, wflat
    _free_engine_pool()


# ─── the pool bound: any SWA pool B <= B_max ──────────────────────────────────
#
# The engine picks the SWA pool's page count B from free memory, so no single
# dumped B is "the" engine's. Two tests make the coverage an upper bound:
# (1) the cache manager at the largest quota a gate-configured engine can reach
# gives B_max = ENGINE_SWA_PAGES (and the op is exact at the far end of the
# manager's own buffer), and (2) the same rows into the same slots give the same
# bits at B = the smallest pool that holds them, sizes across the 2^31 / 2^32
# element-count lines, a mid size and B_max (views of one allocation), and in a
# separate exact-size allocation.

# (max_draft_len, max_batch_size) -> SWA pages at 0.5 x this GB300's total memory.
_BMAX_DEVICE_TOTAL = 297_020_948_480
_BMAX_SWA_PAGES = {(0, 32): 261_600, (5, 32): 323_840, (0, 8): 468_720, (5, 8): 491_000}


def _pp_mapping(pp_stage):
    """The single-process mapping of a rank of PP2 stage `pp_stage` (tp 2, as the PP2 stage scan builds
    it; the V4.1 pools do not depend on tp), or of the unsplit manager when pp_stage is None."""
    from tensorrt_llm.mapping import Mapping

    if pp_stage is None:
        return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    return Mapping(world_size=4, rank=2 * pp_stage, tp_size=2, pp_size=2, gpus_per_node=4)


def _v41_manager(
    quota: int,
    max_draft_len: int,
    max_batch_size: int,
    max_num_tokens: int = 8192,
    max_seq_len=None,
    pp_stage=None,
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


def test_largest_engine_pools_from_the_cache_manager() -> None:
    """B_max from the manager itself, and the far-end hazard call on the manager's own B_max buffer.

    The quota is 0.5 x the device's total memory, the upper bound of
    kv_cache_fraction 0.5 x free memory. The largest configuration is built
    last and kept for the far-end call: its SWA view (layer 0; one pool for all
    40 layers) is filled with the sentinel, the op writes rows to both sides of
    every 2^31..2^34 element-offset line, the first and the last page and the
    last slot, and the view must hold the reference there and the sentinel
    everywhere else.
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
    order = sorted(_BMAX_SWA_PAGES, key=_BMAX_SWA_PAGES.get)  # the largest last
    for key in order:
        mgr = _v41_manager(quota, *key)
        pool = mgr.get_buffers(0, AT.SWA)
        got[key] = pool.shape[0]
        assert (
            tuple(pool.shape[1:]) == (ENGINE_PAGE, WINDOW_KV_WIDTH) and pool.dtype == torch.bfloat16
        )
        assert pool.stride() == (ENGINE_PAGE * WINDOW_KV_WIDTH, WINDOW_KV_WIDTH, 1), pool.stride()
        if key != order[-1]:
            mgr.shutdown()
            del mgr, pool
            _free_engine_pool()
    print(
        f"\n[quota {quota} B = 0.5 x device total] SWA pages per (max_draft_len, max_batch_size): {got}"
    )
    assert got == _BMAX_SWA_PAGES, got
    assert max(got.values()) == GATES_SWA_PAGES
    try:
        num_slots = pool.shape[0] * pool.shape[1]
        pool.view(torch.int16).fill_(SENTINEL)
        hazard = _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
        assert hazard[-1] == num_slots - 1 and 2**34 // WINDOW_KV_WIDTH in hazard
        slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device="cuda")
        x = _window_kv_like(slots.numel(), seed=4910)
        op(x, pool, slots)
        _check_engine_pool(pool, pool, slots, _fake_quant(x)[0])
        print(
            f"[manager's own SWA buffer {tuple(pool.shape)}] {len(hazard)} far-end / hazard rows == reference, "
            f"rest sentinel"
        )
    finally:
        mgr.shutdown()
        del mgr, pool
        _free_engine_pool()


def _invariance_slots(t: int, page_size: int, seed: int):
    """(smallest page count holding t slots, int64 [2, t] table row 1 of distinct slots in it): the last
    slot of that smallest pool is named, 1/16 of the other rows -1."""
    small = -(-t // page_size)
    last = small * page_size - 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(small * page_size, device="cuda", generator=g)
    pick = torch.cat([torch.tensor([last], device="cuda"), perm[perm != last]])[:t]
    pick = pick[torch.randperm(t, device="cuda", generator=g)]
    table = torch.full((2, t), -1, dtype=torch.int64, device="cuda")
    slots = table[1]
    slots.copy_(pick)
    skip = torch.randperm(t, device="cuda", generator=g)[: t // 16]
    slots[skip[slots[skip] != last]] = -1
    return small, slots


@pytest.mark.parametrize("t", [32, 8192, 32768])
def test_pool_size_invariance(t: int) -> None:
    """The same t rows into the same slots: bitwise the same bits at every pool size B, and = reference.

    B = the smallest pool holding the slots, the page counts whose element count first passes 2^31, 2^32,
    2^33, 2^34 and 2^35 (32769, 65537, 131073, 262145, 524289), the gates' bound 491000, the unsplit
    manager's envelope bound 730880, a mid size B_max // 2 + 1 and B_max (958720, a PP2 stage alone),
    each a view pool[:B] of one sentinel-filled allocation of B_max pages plus a guard page, then a separate
    allocation of exactly the smallest size. After every call the whole allocation (past B too) holds the
    sentinel outside the named slots.
    """
    small, slots = _invariance_slots(t, ENGINE_PAGE, seed=2300 + t)
    x = _window_kv_like(t, seed=2300 + t) if t <= 8192 else _fast_rows(t, seed=2300 + t)
    y = _fake_quant(x)[0]
    live = slots[slots >= 0]
    per_page = ENGINE_PAGE * WINDOW_KV_WIDTH
    sizes = sorted(
        {
            small,
            *(lim // per_page + 1 for lim in (2**31, 2**32, 2**33, 2**34, 2**35)),
            GATES_SWA_PAGES,
            UNSPLIT_SWA_PAGES,
            ENGINE_SWA_PAGES // 2 + 1,
            ENGINE_SWA_PAGES,
        }
    )
    assert {32769, 65537, 131073, 262145, 524289, 491000, 730880} < set(sizes) and sizes[
        -1
    ] == 958720
    buf, _ = _engine_pool(ENGINE_SWA_PAGES, ENGINE_PAGE)
    first = None
    for b in sizes:
        pool = buf[:b]
        op(x, pool, slots)
        torch.cuda.synchronize()
        got = pool.view(-1, WINDOW_KV_WIDTH)[live].view(torch.int16).clone()
        _check_engine_pool(buf, pool, slots, y)
        first = got if first is None else first
        assert torch.equal(got, first), (
            f"B={b}: {int((got != first).sum())} elements differ from B={sizes[0]}"
        )
    del buf, pool
    _free_engine_pool()
    buf, pool = _engine_pool(small, ENGINE_PAGE)
    op(x, pool, slots)
    torch.cuda.synchronize()
    got = pool.view(-1, WINDOW_KV_WIDTH)[live].view(torch.int16).clone()
    _check_engine_pool(buf, pool, slots, y)
    assert torch.equal(got, first), "a separate exact-size pool differs"
    print(
        f"\n[t={t}] pools B = {sizes} pages (views of one allocation) and a separate [{small}, 128, 512]: "
        f"{live.numel()} rows bitwise identical and == reference, nothing else written"
    )
    del buf, pool
    _free_engine_pool()


# ─── native storage on the live DeepSeek-V4.1 cache manager ──────────────────
#
# With TRTLLM_DSV41_KV_STORAGE=native the V4.1 DeepseekV4CacheManager keeps the
# SWA window KV natively: one uint8 pool [B, 128, 528] shared by all 40 layers,
# each 528-byte row the 512 E4M3 codes then the 16 UE8M0 bytes, and
# get_v41_native_kv_views(L, SWA) returns (codes [B, 128, 512], scales
# [B, 128, 16]), two uint8 views of that pool with strides (67584, 528, 1). The
# native-storage target writes it with this op's native form:
# mxfp8_fake_quantize(kv [T, 512], codes, md.v41_swa_slot_mapping[local(L)][:T], scales).
# B comes from free memory again. The table is the manager's native SWA pages at
# 0.5 x this GB300's total memory (the ceiling of kv_cache_fraction 0.5 x free
# memory) for every (max_num_tokens, max_batch_size, DSpark max_draft_len) the
# runs use or could use: the native gates run (2048, 8, 5) (boot) and
# (8192, 32, 5) (accuracy), 549,080 and 732,280 pages; over max_batch_size
# {8, 32} the largest is (8192, 8, 5), 854,440 pages. Over every max_batch_size
# 1..32 (the native-storage witness runs at 1) the page count is
# not monotone in max_batch_size, and B_max = 941,800 pages at (8192, 1, 0)
# (120,550,400 slots, 63.6 GB). Over the engine envelope (max_num_tokens up
# to 16384, max_batch_size up to 256, max_seq_len up to 1,048,576) the unsplit
# manager's largest is 1,098,720 pages at (16384, 1, 0) (140,636,160 slots,
# 74.3 GB), and a PP2 stage-1 manager alone reaches 1,648,320 at (16384, 1, 0)
# (210,984,960 slots, 111.4 GB; ENVELOPE_NATIVE_SWA_BMAX, the B_max the far-end,
# invariance and capture cases use). Every byte-offset hazard up to 2^35 lies
# inside all of them, and 2^36 inside the two largest.
#
# Round trips: rows the op writes natively are read back with the paired reader
# (the raw op torch.ops.trtllm.mxfp8_dequantize) and held bitwise to the op's own
# bf16-container output of the same rows, which is itself held to the recipe
# built here from native torch ops. The reader is part of what is checked, not
# the reference.

import os  # noqa: E402
import random  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import tensorrt_llm._torch.custom_ops.mxfp8_dequantize  # noqa: E402,F401 -- registers the paired reader

_STORAGE_ENV = "TRTLLM_DSV41_KV_STORAGE"
NATIVE_ROW = 528  # 512 E4M3 code bytes, then 16 UE8M0 bytes
NATIVE_CODES = 512
# 0xFF is E4M3 -NaN and the UE8M0 NaN: for a finite row the op writes neither
# (codes are clamped to +-448; the scale byte is e + 127 with e in -22..120).
NATIVE_SENTINEL = 0xFF
_NATIVE_SWA_PAGES = {
    (2048, 8, 0): 399_640,
    (2048, 8, 5): 549_080,
    (2048, 32, 0): 366_400,
    (2048, 32, 5): 569_440,
    (8192, 8, 0): 776_000,
    (8192, 8, 5): 854_440,
    (8192, 32, 0): 585_840,
    (8192, 32, 5): 732_280,
}
_NATIVE_GATES = ((2048, 8, 5), (8192, 32, 5))
MBS_AXIS = tuple(range(1, 33))  # every max_batch_size the engine configuration permits (1..32)
# Native SWA pages at 0.5 x device total per (max_num_tokens, DSpark max_draft_len), for max_batch_size 1..32
# (measured on this GB300; re-measured by test_native_swa_pools_from_the_cache_manager).
_SWA_PAGES_BY_MBS = {
    (2048, 0): (
        549_600,
        549_600,
        549_600,
        549_600,
        549_600,
        488_040,
        439_360,
        399_640,
        399_640,
        549_080,
        506_880,
        506_880,
        506_880,
        506_880,
        506_880,
        439_360,
        439_360,
        439_360,
        439_360,
        412_080,
        387_720,
        387_720,
        462_720,
        462_720,
        439_360,
        439_360,
        439_360,
        418_520,
        418_520,
        382_280,
        366_400,
        366_400,
    ),
    (2048, 5): (
        549_600,
        549_600,
        549_600,
        549_600,
        732_280,
        599_240,
        599_240,
        549_080,
        549_080,
        549_080,
        506_880,
        628_040,
        628_040,
        628_040,
        549_080,
        549_080,
        549_080,
        646_400,
        646_400,
        610_160,
        578_400,
        578_400,
        549_080,
        628_040,
        599_240,
        599_240,
        573_440,
        573_440,
        573_440,
        591_800,
        569_440,
        569_440,
    ),
    (8192, 0): (
        941_800,
        879_240,
        879_240,
        879_240,
        879_240,
        824_160,
        824_160,
        776_000,
        776_000,
        732_280,
        732_280,
        809_760,
        809_760,
        809_760,
        732_280,
        732_280,
        732_280,
        732_280,
        732_280,
        699_040,
        668_760,
        640_960,
        615_120,
        676_200,
        650_880,
        650_880,
        650_880,
        650_880,
        606_200,
        606_200,
        585_840,
        585_840,
    ),
    (8192, 5): (
        879_240,
        879_240,
        879_240,
        879_240,
        879_240,
        905_080,
        905_080,
        854_440,
        854_440,
        809_760,
        809_760,
        879_240,
        879_240,
        879_240,
        798_840,
        798_840,
        798_840,
        798_840,
        859_880,
        824_160,
        790_880,
        732_280,
        732_280,
        732_280,
        757_600,
        757_600,
        757_600,
        757_600,
        708_960,
        708_960,
        732_280,
        732_280,
    ),
}
_NATIVE_SWA_PAGES_AXIS = {
    (t, s, d): pages[s - 1] for (t, d), pages in _SWA_PAGES_BY_MBS.items() for s in MBS_AXIS
}
assert all(_NATIVE_SWA_PAGES_AXIS[k] == v for k, v in _NATIVE_SWA_PAGES.items())
NATIVE_SWA_BMAX = 941_800  # over every max_batch_size 1..32
NATIVE_SWA_BMAX_KEY = (8192, 1, 0)  # the configuration that reaches it
NATIVE_SWA_BMAX_8_32 = 854_440  # the bound over max_batch_size {8, 32} only
NATIVE_SWA_BMAX_8_32_KEY = (8192, 8, 5)
NATIVE_SWA_GATE_MAX = 732_280
NATIVE_WITNESS_KEY = (2048, 1, 5)  # the native-storage witness configuration
# The engine envelope (max_num_tokens 2048..16384, max_batch_size up to 256, max_seq_len up to 1,048,576,
# DSpark or not): the pool scan's largest native SWA pool, at (16384, 1, 0) whatever max_seq_len;
# re-measured by test_envelope_largest_swa_pools_from_the_cache_manager.
UNSPLIT_NATIVE_SWA_BMAX = 1_098_720
ENVELOPE_SWA_BMAX_KEY = (16384, 1, 0)
# PP2 (the unified target's stage split at layer 20): a stage manager holds only its 20 layers' pools. The
# PP2 stage scan's largest native SWA pool is stage 1 alone at (16384, 1, 0), whatever max_seq_len (the
# stage alone is the upper bound: in an engine the stages' token capacities are cut to the smaller one);
# re-measured by test_envelope_pp2_stage_swa_pools_from_the_cache_manager.
ENVELOPE_NATIVE_SWA_BMAX = 1_648_320
ENVELOPE_PP2_SWA_STAGE = 1
_NATIVE_FAR_PAGES = (UNSPLIT_NATIVE_SWA_BMAX, ENVELOPE_NATIVE_SWA_BMAX)
_NATIVE_PAGE_BYTES = ENGINE_PAGE * NATIVE_ROW  # 67584


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
    max_num_tokens: int, max_batch_size: int, max_draft_len: int, max_seq_len=None, pp_stage=None
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


def _native_swa(mgr, layer: int = 0):
    """(the manager's SWA pool [B, 128, 528] uint8, codes view, scales view) through `layer` (a layer of the
    manager's stage), geometry asserted."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    pool = mgr.get_buffers(layer, AT.SWA)
    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
    b = pool.shape[0]
    assert pool.dtype == torch.uint8 and tuple(pool.shape) == (b, ENGINE_PAGE, NATIVE_ROW), (
        pool.shape
    )
    assert pool.stride() == (_NATIVE_PAGE_BYTES, NATIVE_ROW, 1), pool.stride()
    assert tuple(codes.shape) == (b, ENGINE_PAGE, NATIVE_CODES) and codes.stride() == pool.stride()
    assert tuple(scales.shape) == (b, ENGINE_PAGE, 16) and scales.stride() == pool.stride()
    assert (
        codes.data_ptr() == pool.data_ptr() and scales.data_ptr() == pool.data_ptr() + NATIVE_CODES
    )
    return pool, codes, scales


def _native_rows(x: torch.Tensor):
    """(uint8 [T, 528] native rows = codes then UE8M0 bytes, bf16 [T, 512] container values) of the recipe."""
    y, codes, sbytes = _fake_quant(x)
    return torch.cat([codes, sbytes], dim=1), y


def _decode_native(rows: torch.Tensor) -> torch.Tensor:
    """bf16 [n, 512] of native rows, in native torch: e4m3(code) * 2^(byte - 127)."""
    v = rows[:, :NATIVE_CODES].contiguous().view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    s = torch.exp2(rows[:, NATIVE_CODES:].float() - 127).unsqueeze(-1)
    return (v * s).view(-1, NATIVE_CODES).to(torch.bfloat16)


def _round_trip(
    codes: torch.Tensor, scales: torch.Tensor, slots: torch.Tensor, container: torch.Tensor
) -> int:
    """Read the rows at `slots` back with the paired reader (compact: out row r = pool slot slots[r]) into a
    NaN-filled bf16 out, and hold every row whose slot is in the pool bitwise to `container` (the op's own
    bf16-container output of the same rows). Returns the number of rows checked."""
    num_slots = codes.shape[0] * codes.shape[1]
    out = torch.empty(slots.numel(), NATIVE_CODES, dtype=torch.bfloat16, device="cuda")
    out.view(torch.int16).fill_(SENTINEL)
    torch.ops.trtllm.mxfp8_dequantize(codes, scales, out, slots)
    s64 = slots.to(torch.int64)
    live = (s64 >= 0) & (s64 < num_slots)
    _assert_bitwise(out[live], container[live])
    return int(live.sum())


def _native_hazard_slots(num_slots: int, page_size: int = ENGINE_PAGE) -> list[int]:
    """Distinct slots of a 528-byte-row pool on both sides of every byte offset 2^31..2^36 inside it (the row
    holding the offset and its neighbours), plus the first page, the last page and the last slot."""
    cands = {0, 1, page_size - 1, page_size}
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // NATIVE_ROW
        cands |= {s - 1, s, s + 1}
    cands |= {num_slots - page_size - 1, num_slots - page_size, num_slots - 2, num_slots - 1}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _count_not_sentinel(buf: torch.Tensor) -> torch.Tensor:
    """Device count of bytes of `buf` (uint8, any shape, contiguous) that are not NATIVE_SENTINEL."""
    whole = buf.view(-1)
    step = (
        1 << 26
    )  # a bool reduction materializes an int64 copy (8 bytes per element): keep chunks small
    n = torch.zeros((), dtype=torch.int64, device="cuda")
    for i in range(0, whole.numel(), step):
        n += (whole[i : i + step] != NATIVE_SENTINEL).sum()
    return n


def _check_native_pool(
    buf: torch.Tensor, num_slots: int, slots: torch.Tensor, rows: torch.Tensor
) -> None:
    """buf: uint8 [pages(+guard), 128, 528] whose first num_slots slots are the pool. The named slots hold
    `rows` bitwise; every other byte of buf is the sentinel. Restores the named slots to the sentinel."""
    flat = buf.view(-1, NATIVE_ROW)
    s64 = slots.to(torch.int64)
    live = (s64 >= 0) & (s64 < num_slots)
    idx = s64[live]
    got = flat[idx]
    n_bad = int((got != rows[live]).sum())
    assert n_bad == 0, f"{n_bad} of {got.numel()} bytes in the named slots differ bitwise"
    flat[idx] = NATIVE_SENTINEL
    n_out = int(_count_not_sentinel(buf))
    assert n_out == 0, f"{n_out} bytes outside the named slots were written"


def _native_engine_pool(pages: int):
    """(uint8 [pages + 1, 128, 528] buffer with a guard page, codes view, scales view), all bytes sentinel."""
    buf = torch.full(
        (pages + 1, ENGINE_PAGE, NATIVE_ROW), NATIVE_SENTINEL, dtype=torch.uint8, device="cuda"
    )
    return buf, buf[:pages, :, :NATIVE_CODES], buf[:pages, :, NATIVE_CODES:]


def _fast_rows(t: int, seed: int) -> torch.Tensor:
    """bf16 [t, 512] window-KV-like rows, vectorized: one magnitude 2^-30..2^20 per 32-block, some blocks
    zero and some under the 1e-4 floor (scale byte 105)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mag = torch.randint(-30, 21, (t, 16, 1), device="cuda", generator=g).float()
    x = torch.randn(t, 16, BLOCK, device="cuda", generator=g) * torch.exp2(mag)
    pick = torch.randint(0, 16, (t, 1, 1), device="cuda", generator=g)
    x = torch.where(torch.arange(16, device="cuda").view(1, 16, 1) == pick, 0.0, x)
    return x.view(t, WINDOW_KV_WIDTH).to(torch.bfloat16)


def _manager_far_end_round_trip(pool, codes, scales, seed: int) -> None:
    """On a manager's own native SWA pool: sentinel-fill it, write rows through the manager's views to both
    sides of every 2^31..2^35 byte offset, the first and the last page and the last slot (plus a -1 and a
    one-past-the-end slot, skipped); the named rows hold the recipe's codes and bytes, decode in torch and
    read back through the paired reader to the op's own bf16-container output, the rest is the sentinel."""
    num_slots = pool.shape[0] * ENGINE_PAGE
    pool.fill_(NATIVE_SENTINEL)
    hazard = _native_hazard_slots(num_slots)
    assert hazard[-1] == num_slots - 1 and 2**35 // NATIVE_ROW in hazard
    slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device="cuda")
    x = _window_kv_like(slots.numel(), seed=seed)
    rows, y = _native_rows(x)
    op(x, codes, slots, scales)
    live = slots[(slots >= 0) & (slots < num_slots)]
    container = torch.empty_like(x)
    op(x, container)
    _assert_bitwise(_decode_native(pool.view(-1, NATIVE_ROW)[live]), container[: live.numel()])
    _assert_bitwise(container, y)
    assert _round_trip(codes, scales, slots, container) == len(hazard)
    _check_native_pool(pool, num_slots, slots, rows)
    print(
        f"[manager's own native SWA pool {tuple(pool.shape)}] {len(hazard)} far-end / hazard rows == "
        f"recipe codes + bytes; torch decode == mxfp8_dequantize read-back == the bf16 container; rest sentinel"
    )


def test_native_swa_pools_from_the_cache_manager() -> None:
    """Every native configuration's SWA pool from the manager itself, and the hazard call on the largest.

    For each (max_num_tokens, max_batch_size, max_draft_len) of {2048, 8192} x
    every max_batch_size 1..32 x {0, 5} (128 managers) at 0.5 x total memory:
    the pool is [B, 128, 528] uint8 for all 40 layers (one base), the views
    are its code / scale halves, and 890 bytes per token; the pages equal the
    table. B_max over the axis is 941,800 at (8192, 1, 0); over max_batch_size
    {8, 32} it is 854,440 at (8192, 8, 5); 732,280 at the gates. On both of
    those managers' own pools the far-end round trip runs: filled with the
    sentinel, the op's native form writes rows to both sides of every
    2^31..2^35 byte offset, the first and the last page and the last slot
    through the manager's views, and the pool must hold the recipe's codes and
    bytes there, decoding (in torch, and through mxfp8_dequantize) to the op's
    own bf16-container output, and the sentinel everywhere else.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    got = {}
    order = sorted(_NATIVE_SWA_PAGES_AXIS, key=_NATIVE_SWA_PAGES_AXIS.get)  # the largest last
    assert order[-1] == NATIVE_SWA_BMAX_KEY
    for key in order:
        mgr = _v41_native_manager(*key)
        pool, codes, scales = _native_swa(mgr)
        assert mgr.max_batch_size == key[1] and mgr.max_seq_len == (16398 if key[2] else 16384)
        assert mgr.get_cache_bytes_per_token() == 890, (key, mgr.get_cache_bytes_per_token())
        for layer in range(40):
            v = mgr.get_buffers(layer, AT.SWA)
            assert v.data_ptr() == pool.data_ptr() and v.shape == pool.shape, (key, layer)
        got[key] = pool.shape[0]
        if key == NATIVE_SWA_BMAX_8_32_KEY:
            assert got[key] == NATIVE_SWA_BMAX_8_32
            _manager_far_end_round_trip(pool, codes, scales, seed=8544)
        if key != order[-1]:
            mgr.shutdown()
            del mgr, pool, codes, scales, v
            _free_engine_pool()
    over = {k: v for k, v in sorted(got.items()) if v > NATIVE_SWA_BMAX_8_32}
    print(
        f"\n[native, quota = 0.5 x device total] SWA pages per (max_num_tokens, max_batch_size, "
        f"max_draft_len), 128 configurations: {dict(sorted(got.items()))}\n"
        f"[above the max_batch_size {{8, 32}} bound {NATIVE_SWA_BMAX_8_32}] {len(over)}: {over}"
    )
    assert got == _NATIVE_SWA_PAGES_AXIS, {
        k: (v, _NATIVE_SWA_PAGES_AXIS[k]) for k, v in got.items() if v != _NATIVE_SWA_PAGES_AXIS[k]
    }
    assert max(got.values()) == NATIVE_SWA_BMAX and got[NATIVE_SWA_BMAX_KEY] == NATIVE_SWA_BMAX
    assert max(got[k] for k in _NATIVE_SWA_PAGES) == NATIVE_SWA_BMAX_8_32
    assert max(got[k] for k in _NATIVE_GATES) == NATIVE_SWA_GATE_MAX
    try:
        _manager_far_end_round_trip(pool, codes, scales, seed=9418)
    finally:
        mgr.shutdown()
        del mgr, pool, codes, scales
        _free_engine_pool()


# (storage, max_num_tokens, max_batch_size, DSpark max_draft_len, max_seq_len as the engine passes it) -> SWA
# pages at 0.5 x this GB300's total memory: the envelope ceilings (the pool scan's maxima over 1920
# managers per storage: max_num_tokens 2048..16384, max_batch_size 1..256 sampled, DSpark 0 / 5, max_seq_len
# 16384..1,048,576) at two max_seq_len, and the max_batch_size-256 configurations beside them.
_ENVELOPE_SWA_PAGES = {
    ("bf16", 16384, 1, 0, 16384): 730_880,
    ("bf16", 16384, 1, 0, 1_048_576): 730_880,
    ("bf16", 16384, 256, 0, 16384): 161_520,
    ("bf16", 16384, 256, 5, 16398): 244_200,
    ("native", 16384, 1, 0, 16384): 1_098_720,
    ("native", 16384, 1, 0, 1_048_576): 1_098_720,
    ("native", 16384, 256, 0, 16384): 420_520,
    ("native", 16384, 256, 5, 16398): 617_600,
}


def test_envelope_largest_swa_pools_from_the_cache_manager() -> None:
    """The envelope SWA ceilings from the manager itself, and the far-end call on each ceiling manager's
    own pool.

    Each configuration of _ENVELOPE_SWA_PAGES is built at 0.5 x the device's total memory in its storage mode, the
    pool geometry asserted ([B, 128, 512] bf16 with strides (65536, 512, 1); [B, 128, 528] uint8 with its
    code / scale views) and B pinned: 730,880 (bf16) and 1,098,720 (native) at (16384, 1, 0) whatever
    max_seq_len. On the (16384, 1, 0, 16384) managers' own pools (filled with the sentinel first) the op
    writes rows to both sides of every 32-bit hazard, the first and the last page and the last slot:
    bf16 element offsets 2^30..2^35 (hazard rows == reference, rest sentinel); native byte offsets
    2^31..2^36 (codes + bytes == recipe, torch decode and mxfp8_dequantize read-back == the bf16 container,
    rest sentinel).
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    got = {}
    for key in _ENVELOPE_SWA_PAGES:
        storage, max_t, seqs, draft, lm = key
        if storage == "bf16":
            mgr = _v41_manager(total // 2, draft, seqs, max_num_tokens=max_t, max_seq_len=lm)
        else:
            mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm)
        try:
            assert mgr.max_seq_len == lm and mgr.max_batch_size == seqs
            far = (max_t, seqs, draft, lm) == ENVELOPE_SWA_BMAX_KEY + (16384,)
            if storage == "bf16":
                pool = mgr.get_buffers(0, AT.SWA)
                assert (
                    tuple(pool.shape[1:]) == (ENGINE_PAGE, WINDOW_KV_WIDTH)
                    and pool.dtype == torch.bfloat16
                )
                assert pool.stride() == (ENGINE_PAGE * WINDOW_KV_WIDTH, WINDOW_KV_WIDTH, 1), (
                    pool.stride()
                )
                got[key] = pool.shape[0]
                if far:
                    num_slots = pool.shape[0] * pool.shape[1]
                    pool.view(torch.int16).fill_(SENTINEL)
                    hazard = _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
                    assert hazard[-1] == num_slots - 1 and 2**35 // WINDOW_KV_WIDTH in hazard
                    slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device="cuda")
                    x = _window_kv_like(slots.numel(), seed=7308)
                    op(x, pool, slots)
                    _check_engine_pool(pool, pool, slots, _fake_quant(x)[0])
                    print(
                        f"\n[bf16 manager {key}, own SWA buffer {tuple(pool.shape)}] {len(hazard)} far-end / "
                        f"hazard rows == reference, rest sentinel"
                    )
                del pool
            else:
                pool, codes, scales = _native_swa(mgr)
                got[key] = pool.shape[0]
                if far:
                    assert 2**36 // NATIVE_ROW in _native_hazard_slots(pool.shape[0] * ENGINE_PAGE)
                    _manager_far_end_round_trip(pool, codes, scales, seed=10987)
                del pool, codes, scales
        finally:
            mgr.shutdown()
            del mgr
            _free_engine_pool()
    print(f"\n[envelope SWA pages at 0.5 x device total] {got}")
    assert got == _ENVELOPE_SWA_PAGES, {
        k: (v, _ENVELOPE_SWA_PAGES[k]) for k, v in got.items() if v != _ENVELOPE_SWA_PAGES[k]
    }
    assert max(v for k, v in got.items() if k[0] == "bf16") == UNSPLIT_SWA_PAGES
    assert max(v for k, v in got.items() if k[0] == "native") == UNSPLIT_NATIVE_SWA_BMAX


# (storage, PP2 stage, max_num_tokens, max_batch_size, DSpark max_draft_len, max_seq_len) -> SWA pages of the
# stage manager built alone at 0.5 x this GB300's total memory: each storage's stage-alone maximum over the
# engine envelope (stage 1) and the other stage at its own maximum's tuple.
_ENVELOPE_PP2_SWA_PAGES = {
    ("bf16", 0, 16384, 1, 0, 16384): 596_220,
    ("bf16", 1, 16384, 2, 0, 16384): 958_720,
    ("native", 0, 16384, 1, 0, 16384): 878_760,
    ("native", 1, 16384, 1, 0, 16384): 1_648_320,
}


def test_envelope_pp2_stage_swa_pools_from_the_cache_manager() -> None:
    """The PP2 stage managers' SWA pools, and the far-end call on each storage's largest stage pool.

    Each configuration of _ENVELOPE_PP2_SWA_PAGES is built alone (one process) with the mapping of a rank of that
    stage (pp_size 2, tp 2) at 0.5 x the device's total memory: the stage holds layers 0-19 or 20-39
    (pp_layers asserted), every layer of the stage has the same SWA view on one pool base, and B is pinned
    (bf16 [B, 128, 512] with strides (65536, 512, 1); native [B, 128, 528] with its code / scale views). On
    the two stage-1 managers' own pools (filled with the sentinel first) the op writes rows to both sides of
    every 32-bit hazard, the first and the last page and the last slot through a stage-1 layer's views:
    bf16 element offsets 2^30..2^35 (rows == reference, rest sentinel); native byte offsets 2^31..2^36 (codes
    + bytes == recipe, torch decode and mxfp8_dequantize read-back == the bf16 container, rest sentinel).
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
    got = {}
    for key in _ENVELOPE_PP2_SWA_PAGES:
        storage, stage, max_t, seqs, draft, lm = key
        if storage == "bf16":
            mgr = _v41_manager(
                total // 2, draft, seqs, max_num_tokens=max_t, max_seq_len=lm, pp_stage=stage
            )
        else:
            mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm, pp_stage=stage)
        try:
            layers = list(range(20)) if stage == 0 else list(range(20, 40))
            assert (
                list(mgr.pp_layers) == layers
                and mgr.max_seq_len == lm
                and mgr.max_batch_size == seqs
            )
            views = [mgr.get_buffers(layer, AT.SWA) for layer in layers]
            assert all(
                v.data_ptr() == views[0].data_ptr() and v.shape == views[0].shape for v in views
            ), key
            got[key] = views[0].shape[0]
            if storage == "bf16":
                pool = views[0]
                assert (
                    tuple(pool.shape[1:]) == (ENGINE_PAGE, WINDOW_KV_WIDTH)
                    and pool.dtype == torch.bfloat16
                )
                assert pool.stride() == (ENGINE_PAGE * WINDOW_KV_WIDTH, WINDOW_KV_WIDTH, 1), (
                    pool.stride()
                )
                if stage == 1:
                    num_slots = pool.shape[0] * pool.shape[1]
                    pool.view(torch.int16).fill_(SENTINEL)
                    hazard = _hazard_slots(num_slots, WINDOW_KV_WIDTH, ENGINE_PAGE)
                    assert hazard[-1] == num_slots - 1 and 2**35 // WINDOW_KV_WIDTH in hazard
                    slots = torch.tensor(hazard + [-1, num_slots], dtype=torch.int64, device="cuda")
                    x = _window_kv_like(slots.numel(), seed=9587)
                    op(x, pool, slots)
                    _check_engine_pool(pool, pool, slots, _fake_quant(x)[0])
                    print(
                        f"\n[bf16 PP2 stage-1 manager {key}, own SWA buffer {tuple(pool.shape)}] {len(hazard)} "
                        f"far-end / hazard rows == reference, rest sentinel"
                    )
                del pool
            else:
                pool, codes, scales = _native_swa(mgr, layers[0])
                if stage == 1:
                    assert 2**36 // NATIVE_ROW in _native_hazard_slots(pool.shape[0] * ENGINE_PAGE)
                    _manager_far_end_round_trip(pool, codes, scales, seed=16483)
                del pool, codes, scales
            del views
        finally:
            mgr.shutdown()
            del mgr
            _free_engine_pool()
    print(f"\n[PP2 stage SWA pages at 0.5 x device total, stage alone] {got}")
    assert got == _ENVELOPE_PP2_SWA_PAGES, {
        k: (v, _ENVELOPE_PP2_SWA_PAGES[k])
        for k, v in got.items()
        if v != _ENVELOPE_PP2_SWA_PAGES[k]
    }
    assert max(v for k, v in got.items() if k[0] == "bf16") == ENGINE_SWA_PAGES
    assert max(v for k, v in got.items() if k[0] == "native") == ENVELOPE_NATIVE_SWA_BMAX


@pytest.mark.parametrize("pages", _NATIVE_FAR_PAGES)
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_native_engine_swa_pool_far_end(slot_dtype: torch.dtype, pages: int) -> None:
    """Decode-like on a [pages, 128, 528] pool with a guard page, at the envelope B_max (1,648,320, a PP2
    stage alone) and the unsplit manager's (1,098,720): hazard rows, -1 and past-the-end skipped; the rows read
    back through mxfp8_dequantize equal the op's bf16-container output."""
    buf, codes, scales = _native_engine_pool(pages)
    num_slots = pages * ENGINE_PAGE
    hazard = _native_hazard_slots(num_slots)
    assert 2**35 // NATIVE_ROW in hazard and hazard[-1] == num_slots - 1
    slots = torch.tensor(
        hazard + [-1, num_slots, num_slots + ENGINE_PAGE - 1], dtype=torch.int64, device="cuda"
    )
    x = _window_kv_like(slots.numel(), seed=8545)
    op(x, codes, slots.to(slot_dtype), scales)
    container = torch.empty_like(x)
    op(x, container)
    _assert_bitwise(container, _native_rows(x)[1])
    assert _round_trip(codes, scales, slots.to(slot_dtype), container) == len(hazard)
    _check_native_pool(buf, num_slots, slots, _native_rows(x)[0])
    del buf, codes, scales
    _free_engine_pool()


@pytest.mark.parametrize("pages", _NATIVE_FAR_PAGES)
def test_native_engine_swa_pool_prefill_spread(pages: int) -> None:
    """Prefill-like, T = 32768 (the decoder pass's largest step) on a [pages, 128, 528] pool with a guard page
    (1,648,320 and 1,098,720): the last page in order, every hazard slot, the rest spread uniformly, 1/16 of the
    rows -1; the rows read back through mxfp8_dequantize equal the op's bf16-container output."""
    t = _MAX_ROWS
    buf, codes, scales = _native_engine_pool(pages)
    num_slots = pages * ENGINE_PAGE
    last_page = torch.arange(num_slots - ENGINE_PAGE, num_slots, device="cuda")
    hazard = torch.tensor(_native_hazard_slots(num_slots), device="cuda")
    fixed = torch.unique(torch.cat([last_page, hazard]))
    g = torch.Generator(device="cuda").manual_seed(8546)
    rest = torch.randint(0, num_slots, (3 * t,), device="cuda", generator=g).unique()
    rest = rest[~torch.isin(rest, fixed)]
    rest = rest[torch.randperm(rest.numel(), device="cuda", generator=g)][: t - fixed.numel()]
    slots = torch.cat([last_page, fixed[~torch.isin(fixed, last_page)], rest]).to(torch.int64)
    assert slots.numel() == t and torch.unique(slots).numel() == t
    slots[torch.randperm(t, device="cuda", generator=g)[: t // 16]] = -1
    x = _fast_rows(t, seed=8546)
    op(x, codes, slots.to(torch.int32), scales)
    container = torch.empty_like(x)
    op(x, container)
    _assert_bitwise(container, _native_rows(x)[1])
    assert _round_trip(codes, scales, slots.to(torch.int32), container) == t - t // 16
    _check_native_pool(buf, num_slots, slots, _native_rows(x)[0])
    del buf, codes, scales
    _free_engine_pool()


@pytest.mark.parametrize("pages", _NATIVE_FAR_PAGES)
def test_native_engine_swa_pool_harness_sees_a_wrapped_offset(pages: int) -> None:
    """Control, on [pages, 128, 528] (1,648,320 and 1,098,720): rows written where a mod-2^32 byte offset would
    put them fail both halves of the pool check (the named rows and the sentinel scan) and the round trip
    (the reader's rows at the true slots are not the container)."""
    buf, codes, scales = _native_engine_pool(pages)
    num_slots = pages * ENGINE_PAGE
    true = torch.tensor(
        [s for s in _native_hazard_slots(num_slots) if s * NATIVE_ROW >= 2**32],
        dtype=torch.int64,
        device="cuda",
    )
    wrapped = (true * NATIVE_ROW % 2**32) // NATIVE_ROW
    assert true.numel() >= 9 and not torch.isin(wrapped, true).any()
    x = _window_kv_like(true.numel(), seed=8547)
    rows = _native_rows(x)[0]
    op(x, codes, wrapped, scales)
    container = torch.empty_like(x)
    op(x, container)
    with pytest.raises(AssertionError, match="elements differ bitwise"):
        _round_trip(codes, scales, true, container)
    with pytest.raises(AssertionError, match="bytes in the named slots differ"):
        _check_native_pool(buf, num_slots, true, rows)
    buf.view(-1, NATIVE_ROW)[true] = NATIVE_SENTINEL
    with pytest.raises(AssertionError, match="outside the named slots were written"):
        _check_native_pool(
            buf, num_slots, torch.tensor([], dtype=torch.int64, device="cuda"), rows[:0]
        )
    del buf, codes, scales
    _free_engine_pool()


@pytest.mark.parametrize("t", [32, 8192, 32768])
def test_native_pool_size_invariance(t: int) -> None:
    """The same rows into the same slots through views codes[:B] / scales[:B] of the live PP2 stage-1
    manager's native SWA pool (16384, 1, 0, 16384; 1,648,320 pages, the envelope B_max): the same bytes at
    every B, equal to the recipe, read back through mxfp8_dequantize (codes[:B] / scales[:B]) bitwise as the
    op's bf16-container output, nothing else written; then a separate allocation of exactly the smallest size.

    B = the smallest pool holding the slots, the page counts whose byte count first passes 2^31, 2^32,
    2^33, 2^34, 2^35 and 2^36, a mid size, the gates' largest (732,280), the max_batch_size {8, 32} bound
    (854,440), B_max over max_batch_size 1..32 at the gates' max_num_tokens (941,800), the unsplit
    manager's envelope B_max (1,098,720) and the stage's (1,648,320).
    """
    mgr = _v41_native_manager(*ENVELOPE_SWA_BMAX_KEY, pp_stage=ENVELOPE_PP2_SWA_STAGE)
    try:
        assert list(mgr.pp_layers) == list(range(20, 40))
        pool, codes, scales = _native_swa(mgr, 20)
        assert pool.shape[0] == ENVELOPE_NATIVE_SWA_BMAX
        pool.fill_(NATIVE_SENTINEL)
        small, slots = _invariance_slots(t, ENGINE_PAGE, seed=8548 + t)
        x = _window_kv_like(t, seed=8548 + t) if t <= 8192 else _fast_rows(t, seed=8548 + t)
        rows, y = _native_rows(x)
        container = torch.empty_like(x)
        op(x, container)
        _assert_bitwise(container, y)
        live = slots[slots >= 0]
        sizes = sorted(
            {
                small,
                *(
                    lim // _NATIVE_PAGE_BYTES + 1
                    for lim in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)
                ),
                ENVELOPE_NATIVE_SWA_BMAX // 2 + 1,
                NATIVE_SWA_GATE_MAX,
                NATIVE_SWA_BMAX_8_32,
                NATIVE_SWA_BMAX,
                UNSPLIT_NATIVE_SWA_BMAX,
                ENVELOPE_NATIVE_SWA_BMAX,
            }
        )
        first = None
        for b in sizes:
            op(x, codes[:b], slots, scales[:b])
            torch.cuda.synchronize()
            got = pool.view(-1, NATIVE_ROW)[live].clone()
            assert _round_trip(codes[:b], scales[:b], slots, container) == live.numel()
            _check_native_pool(pool, b * ENGINE_PAGE, slots, rows)
            first = got if first is None else first
            assert torch.equal(got, first), (
                f"B={b}: {int((got != first).sum())} bytes differ from B={sizes[0]}"
            )
    finally:
        mgr.shutdown()
        del mgr, pool, codes, scales
        _free_engine_pool()
    buf, codes, scales = _native_engine_pool(small)
    op(x, codes, slots, scales)
    torch.cuda.synchronize()
    got = buf.view(-1, NATIVE_ROW)[live].clone()
    assert _round_trip(codes, scales, slots, container) == live.numel()
    _check_native_pool(buf, small * ENGINE_PAGE, slots, rows)
    assert torch.equal(got, first), "a separate exact-size native pool differs"
    print(
        f"\n[native t={t}] views B = {sizes} of the manager's pool and a separate [{small}, 128, 528]: "
        f"{live.numel()} rows bitwise identical and == recipe, mxfp8_dequantize read-back == bf16 container "
        f"at every B, nothing else written"
    )
    del buf, codes, scales
    _free_engine_pool()


def test_native_every_row_count_1_to_32768() -> None:
    """Every prefix (x[:r], slots[:r]), r = 1..32768, of the target's native call writes exactly the first r
    rows of the 32768-row call, and nothing else.

    A native [320, 128, 528] pool plus a guard page, all bytes sentinel. The
    32768-row call is held to the recipe and the sentinel scan. Then per r: the
    call, the named live slots against the recipe's rows, the count of
    non-sentinel bytes against 528 per live row, and the named slots restored
    to the sentinel, so each call's writes are checked on their own. One sync
    at the end. The probe: the sweep compiles no variant beyond the 32768-row
    call's.
    """
    t = _MAX_ROWS
    num_slots = _SWEEP_PAGES * ENGINE_PAGE
    buf, codes, scales = _native_engine_pool(_SWEEP_PAGES)
    flat = buf.view(-1, NATIVE_ROW)
    slots = _sweep_slots(t, num_slots, seed=8549)
    x = _window_kv_like(t, seed=8549)
    rows = _native_rows(x)[0]
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    op(x, codes, slots, scales)
    n1 = _jit_variants(kernel)
    _check_native_pool(buf, num_slots, slots, rows)

    live = slots >= 0
    live_pos = torch.nonzero(live).squeeze(1)
    live_slots = slots[live_pos]
    rows_live = rows[live_pos]
    cnt = torch.cumsum(live.to(torch.int64), 0).tolist()
    bad_rows = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_count = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(1, t + 1):
        op(x[:r], codes, slots[:r], scales)
        k = cnt[r - 1]
        bad_rows += (flat[live_slots[:k]] != rows_live[:k]).sum()
        bad_count += ((buf != NATIVE_SENTINEL).sum() - k * NATIVE_ROW).abs()
        flat[live_slots[:k]] = NATIVE_SENTINEL
    n2 = _jit_variants(kernel)
    print(
        f"\n[native rows 1..{t}] live rows {cnt[-1]}; Triton variants: {n0} before the {t}-row call, {n1} after, "
        f"{n2} after the sweep; byte mismatches {int(bad_rows)}, non-sentinel count error {int(bad_count)}"
    )
    assert int(bad_rows) == 0 and int(bad_count) == 0
    assert n1 - n0 <= 1 and n2 == n1, f"row counts compiled new variants: {n0} -> {n1} -> {n2}"
    # Control: the same two checks see a read one row off and a write one slot off.
    for r in (1, 2, 7, 64):
        for name, xa, sa in (
            ("row +1", x[1 : r + 1], live_slots[:r]),
            ("slot +1", x[:r], live_slots[1 : r + 1]),
        ):
            op(xa, codes, sa, scales)
            n_bad = int((flat[live_slots[:r]] != _native_rows(x[:r])[0]).sum())
            print(f"[native control {name}] r={r}: {n_bad} of {r * NATIVE_ROW} bytes differ")
            assert n_bad > r * NATIVE_ROW // 2
            buf.fill_(NATIVE_SENTINEL)
    del buf, codes, scales, flat
    _free_engine_pool()


# Captured native row counts: every graph batch G of the engine envelope (_ENVELOPE_GSET: every G 1..32 --
# at max_batch_size s <= 32 padding off captures every G in 1..s -- and the padded list to 256), 1 token per
# request, or 6 with DSpark verify: T = G and 6G, 82 row counts up to 1536. The gates' {1..32} and
# {6, 12, ..., 192} are part of it.
_NATIVE_GRAPH_T = _SWA_GRAPH_T
assert set(MBS_AXIS) | {6 * g for g in MBS_AXIS} <= set(_NATIVE_GRAPH_T)


def test_native_cuda_graph_every_graph_batch_on_the_bmax_pool() -> None:
    """T = G and 6G for every G of _ENVELOPE_GSET (82 row counts, 1..1536; decode and DSpark verify), native form,
    on the envelope B_max native SWA pool (1,648,320 pages, a PP2 stage alone): per T an eager warm-up,
    the capture, then 2
    replays on new rows and new far-end slots; every replay writes exactly the recipe into the named slots
    and nothing else, reads back through mxfp8_dequantize as the op's bf16 container, and equals the same
    call run eagerly."""
    buf, codes, scales = _native_engine_pool(ENVELOPE_NATIVE_SWA_BMAX)
    num_slots = ENVELOPE_NATIVE_SWA_BMAX * ENGINE_PAGE
    flat = buf.view(-1, NATIVE_ROW)
    g = torch.Generator(device="cuda").manual_seed(8550)
    hazard = torch.tensor(_native_hazard_slots(num_slots), device="cuda")
    last = num_slots - 1
    table = torch.full((2, max(_NATIVE_GRAPH_T)), -1, dtype=torch.int64, device="cuda")

    def far_slots(t):
        near = last - 1 - torch.arange(4 * ENGINE_PAGE - 1, device="cuda")
        pool_ = torch.unique(torch.cat([hazard[hazard != last], near]))
        if pool_.numel() < t - 1:  # large graph batches: the rest spread uniformly over the pool
            spread = torch.randint(0, last, (4 * t,), device="cuda", generator=g).unique()
            pool_ = torch.cat([pool_, spread[~torch.isin(spread, pool_)]])
        pool_ = pool_[torch.randperm(pool_.numel(), device="cuda", generator=g)]
        s = torch.cat([torch.tensor([last], device="cuda"), pool_[: t - 1]]).to(torch.int64)
        assert torch.unique(s).numel() == t
        if t >= 2:
            s[1 + int(torch.randint(0, t - 1, (1,), device="cuda", generator=g))] = -1
        return s

    n_round_trip = 0
    for t in _NATIVE_GRAPH_T:
        x = torch.empty(t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        slots = table[1, :t]
        x.copy_(_window_kv_like(t, seed=8550 + t))
        slots.copy_(far_slots(t))
        op(x, codes, slots, scales)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, codes, slots, scales)
        buf.fill_(NATIVE_SENTINEL)
        for rep in (1, 2):
            new_x = _window_kv_like(t, seed=8550 + 1000 * rep + t)
            new_slots = far_slots(t)
            x.copy_(new_x)
            slots.copy_(new_slots)
            graph.replay()
            torch.cuda.synchronize()
            live = new_slots[new_slots >= 0]
            replayed = flat[live].clone()
            rows, y = _native_rows(new_x)
            container = torch.empty_like(new_x)
            op(new_x, container)
            _assert_bitwise(container, y)
            n_round_trip += _round_trip(codes, scales, new_slots, container)
            _check_native_pool(buf, num_slots, new_slots, rows)
            op(x, codes, slots, scales)
            torch.cuda.synchronize()
            eager = flat[live].clone()
            _check_native_pool(buf, num_slots, new_slots, rows)
            assert torch.equal(replayed, eager), (
                f"T={t} replay {rep}: {int((replayed != eager).sum())} bytes"
            )
        del graph
    print(
        f"\n[native graphs on [{ENVELOPE_NATIVE_SWA_BMAX}, 128, 528]] {len(_NATIVE_GRAPH_T)} graphs, T = "
        f"{list(_NATIVE_GRAPH_T)}: every replay bitwise == eager == recipe; {n_round_trip} replayed rows read "
        f"back through mxfp8_dequantize == the bf16 container"
    )
    del buf, codes, scales, flat
    _free_engine_pool()


# ─── native form over random multi-request steps on the live manager ───


def _random_step(
    rng: random.Random, max_tokens: int = 8192, max_requests: int = 32, max_len: int = 16384
):
    """([(cached, new)] contexts first (random order) then generation requests, num_contexts, gen_len).

    k in 1..32 requests over T <= 8192 tokens; contexts get random lengths and a
    random cached prefix (40% none); generation requests all carry gen_len
    tokens (1 decode, 6 DSpark verify) at random positions.
    """
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


# Hand-picked steps at the envelope's edges, then the seeded random ones.
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


def _run_step(mgr, reqs, num_contexts: int, gen_len: int, uid: int, max_num_tokens: int = 8192):
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


def test_native_writes_over_random_multi_request_steps() -> None:
    """Random multi-request steps on the live accuracy-gate manager (8192, 32, DSpark 5; native SWA 732,280 pages).

    Per step (9 edge steps + 200 seeded random partitions of T <= 8192 into
    k <= 32 requests, contexts with random cached prefixes, decode or DSpark
    verify generation requests): the requests are allocated on the manager,
    the metadata is prepared, and for layers 0, 1, 20 and 39 the op's native
    form writes random rows at md.v41_swa_slot_mapping[local(L)][:T] through
    get_v41_native_kv_views(L, SWA): the slots hold the recipe's codes and
    bytes, decoding to the op's own bf16-container output of the same rows;
    after the step every other byte of the pool is still the sentinel. The
    op compiles no variant past its first native call.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    mgr = _v41_native_manager(8192, 32, 5)
    rng = random.Random(8551)
    steps = list(_EDGE_STEPS) + [_random_step(rng) for _ in range(_NUM_RANDOM_STEPS)]
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    try:
        pool, _, _ = _native_swa(mgr)
        num_slots = pool.shape[0] * ENGINE_PAGE
        assert pool.shape[0] == NATIVE_SWA_GATE_MAX
        pool.fill_(NATIVE_SENTINEL)
        flat = pool.view(-1, NATIVE_ROW)
        container_buf = torch.empty(8192, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        n_variants = None
        stats = dict(steps=0, rows=0, max_t=0, max_k=0, max_slot=0)
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(mgr, reqs, nctx, gen_len, uid=200_000 + 32 * i)
            try:
                t = sum(n for _, n in reqs)
                for layer in (0, 1, 20, 39):
                    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
                    slots = md.v41_swa_slot_mapping[mgr.layer_offsets[layer]][:t]
                    x = _fast_rows(t, seed=100_000 + 4 * i + layer)
                    op(x, codes, slots, scales)
                    rows, y = _native_rows(x)
                    # A fixed 8192-slot out: in the identity form the slot count is an integer
                    # argument, and Triton specializes it (== 1, % 16 == 0) per call.
                    op(x, container_buf)
                    container = container_buf[:t]
                    if n_variants is None:
                        n_variants = _jit_variants(kernel)
                    s64 = slots.to(torch.int64)
                    assert int(s64.min()) >= 0 and int(s64.max()) < num_slots
                    got = flat[s64]
                    assert torch.equal(got, rows), (
                        f"step {i} layer {layer}: {int((got != rows).sum())} bytes"
                    )
                    _assert_bitwise(_decode_native(got), container)
                    _assert_bitwise(container, y)
                    flat[s64] = NATIVE_SENTINEL
                    stats["max_slot"] = max(stats["max_slot"], int(s64.max()))
                n_out = int(_count_not_sentinel(pool))
                assert n_out == 0, f"step {i}: {n_out} bytes outside the step's slots were written"
                stats["steps"] += 1
                stats["rows"] += 4 * t
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_k"] = max(stats["max_k"], len(reqs))
            finally:
                for r in requests:
                    mgr.free_resources(r)
        assert _jit_variants(kernel) == n_variants, "a step shape compiled a new variant"
        print(
            f"\n[native multi-request steps on the live manager, SWA {tuple(pool.shape)}] {stats}; "
            f"one compiled variant"
        )
        assert stats["steps"] == len(_EDGE_STEPS) + _NUM_RANDOM_STEPS and stats["max_t"] == 8192
        assert stats["max_k"] == 32
    finally:
        mgr.shutdown()
        del mgr
        _free_engine_pool()


# The envelope's step axes: T <= max_num_tokens 16384, k <= max_batch_size 256 requests per step. A step's KV
# tokens (cached + new, over its requests) are capped at _ENVELOPE_STEP_TOKENS: the native manager at (16384, 256,
# DSpark 5, max_seq_len 32,782) held 22 requests of 32,768 tokens and refused the 23rd, so 256 requests cannot
# all sit at the far end of max_seq_len at once.
_ENVELOPE_STEP_TOKENS = 560_000
_ENVELOPE_STEP_L = 32768


def _envelope_random_step(rng: random.Random, max_tokens: int, max_requests: int, seq_len: int):
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


def _envelope_fit(step, rng: random.Random):
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


def _envelope_edge_steps(max_t: int, seqs: int, L: int):
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


def test_envelope_native_writes_over_random_steps_256_requests() -> None:
    """Random multi-request steps at the envelope's step axes, on the live native manager of max_num_tokens 16384,
    max_batch_size 256, DSpark 5, max_seq_len 32768 + 14: 5 edge steps and 200 seeded random partitions of
    T <= 16384 into k <= 256 requests (contexts first with random cached prefixes, decode at positions
    < 32,768 or DSpark verify at < 32,782), each step's KV tokens capped (_envelope_fit). Per step, exactly as
    test_native_writes_over_random_multi_request_steps: layers 0, 1, 20 and 39 write random rows at
    md.v41_swa_slot_mapping[local(L)][:T] through get_v41_native_kv_views(L, SWA); the slots hold the
    recipe's codes and bytes, decoding to the op's own bf16-container output, and after the step every other
    byte of the pool is still the sentinel; no compiled variant past the first native call."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    max_t, seqs, L = 16384, 256, _ENVELOPE_STEP_L
    rng = random.Random(16_384 + seqs + L)
    edges = _envelope_edge_steps(max_t, seqs, L)
    steps = edges + [
        _envelope_fit(_envelope_random_step(rng, max_t, seqs, L), rng)
        for _ in range(_NUM_RANDOM_STEPS)
    ]
    mgr = _v41_native_manager(max_t, seqs, 5, max_seq_len=L + 14)
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    try:
        assert mgr.max_seq_len == L + 14 and mgr.max_batch_size == seqs
        pool, _, _ = _native_swa(mgr)
        num_slots = pool.shape[0] * ENGINE_PAGE
        pool.fill_(NATIVE_SENTINEL)
        flat = pool.view(-1, NATIVE_ROW)
        container_buf = torch.empty(max_t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        n_variants = None
        stats = dict(steps=0, rows=0, max_t=0, max_k=0, max_pos=0, max_slot=0)
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(
                mgr, reqs, nctx, gen_len, uid=5_000_000 + 256 * i, max_num_tokens=max_t
            )
            try:
                t = sum(n for _, n in reqs)
                for layer in (0, 1, 20, 39):
                    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
                    slots = md.v41_swa_slot_mapping[mgr.layer_offsets[layer]][:t]
                    x = _fast_rows(t, seed=500_000 + 4 * i + layer)
                    op(x, codes, slots, scales)
                    rows, y = _native_rows(x)
                    op(x, container_buf)  # a fixed 16384-slot out (one identity variant)
                    container = container_buf[:t]
                    if n_variants is None:
                        n_variants = _jit_variants(kernel)
                    s64 = slots.to(torch.int64)
                    assert int(s64.min()) >= 0 and int(s64.max()) < num_slots
                    got = flat[s64]
                    assert torch.equal(got, rows), (
                        f"step {i} layer {layer}: {int((got != rows).sum())} bytes"
                    )
                    _assert_bitwise(_decode_native(got), container)
                    _assert_bitwise(container, y)
                    flat[s64] = NATIVE_SENTINEL
                    stats["max_slot"] = max(stats["max_slot"], int(s64.max()))
                n_out = int(_count_not_sentinel(pool))
                assert n_out == 0, f"step {i}: {n_out} bytes outside the step's slots were written"
                stats["steps"] += 1
                stats["rows"] += 4 * t
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_k"] = max(stats["max_k"], len(reqs))
                stats["max_pos"] = max(stats["max_pos"], max(c + n - 1 for c, n in reqs))
            finally:
                for r in requests:
                    mgr.free_resources(r)
        assert _jit_variants(kernel) == n_variants, "a step shape compiled a new variant"
        print(
            f"\n[envelope native steps, ({max_t}, {seqs}, 5) at max_seq_len {L + 14}, SWA {tuple(pool.shape)}] "
            f"{len(edges)} edge + {_NUM_RANDOM_STEPS} random steps: {stats}; one compiled variant"
        )
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_pos"] == L + 13
    finally:
        mgr.shutdown()
        del mgr
        _free_engine_pool()


# The max_batch_size axis: the target's native write on the live manager whose pool is
# B_max over every max_batch_size 1..32, (8192, 1, 0), and on the native-storage witness's (2048, 1, 5).
_MBS1_CONFIGS = (NATIVE_SWA_BMAX_KEY, NATIVE_WITNESS_KEY)
_MBS1_RANDOM_STEPS = 60


def _single_request_steps(max_t: int, draft: int, rng: random.Random, n_random: int):
    """Steps of one request (max_batch_size 1) that fit (max_num_tokens, DSpark): a full step from position 0,
    one ending at position 16383, 1 new token at 0 and at 16383 (context), a decode at 16383, and with DSpark
    a 6-token verify ending at 16397 (the manager's max_seq_len is 16398); then seeded random ones: a context
    of 1..max_t new tokens with a random cached prefix (40% none), or a decode (or, with DSpark, a verify) at
    a random position."""
    steps = [
        ([(0, max_t)], 1, 1),
        ([(16384 - max_t, max_t)], 1, 1),
        ([(0, 1)], 1, 1),
        ([(16383, 1)], 1, 1),
        ([(16383, 1)], 0, 1),
    ]
    if draft:
        steps.append(([(16392, 6)], 0, 6))
    for _ in range(n_random):
        if rng.random() < 0.5:
            n = rng.randint(1, max_t)
            steps.append(([(0 if rng.random() < 0.4 else rng.randint(0, 16384 - n), n)], 1, 1))
        else:
            gen_len = 6 if draft and rng.random() < 0.5 else 1
            steps.append(
                ([(rng.randint(0, (16398 if draft else 16384) - gen_len), gen_len)], 0, gen_len)
            )
    return steps


@pytest.mark.parametrize("cfg", _MBS1_CONFIGS, ids=[f"{t}x{s}x{d}" for t, s, d in _MBS1_CONFIGS])
def test_native_writes_at_max_batch_size_1(cfg) -> None:
    """Random steps at max_batch_size 1, on the live manager of the configuration that gives B_max over every
    max_batch_size 1..32 ((8192, 1, 0): 941,800 pages) and of the native-storage witness ((2048, 1, 5): 549,600).

    Per step (edge steps reaching T = max_num_tokens and position 16383, 16397 with DSpark, + 60 seeded
    random single-request steps): the request is allocated on the manager, the metadata prepared with the
    configuration's max_num_tokens, and for layers 0, 1, 20 and 39 the op's native form writes random rows
    at md.v41_swa_slot_mapping[local(L)][:T] through get_v41_native_kv_views(L, SWA): the slots hold the
    recipe's codes and bytes, decoding in torch and reading back through mxfp8_dequantize to the op's own
    bf16-container output; after the step every other byte of the pool is still the sentinel. The op
    compiles no variant past its first native call.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    max_t, seqs, draft = cfg
    mgr = _v41_native_manager(*cfg)
    rng = random.Random(9418 + max_t + draft)
    steps = _single_request_steps(max_t, draft, rng, _MBS1_RANDOM_STEPS)
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    try:
        pool, _, _ = _native_swa(mgr)
        num_slots = pool.shape[0] * ENGINE_PAGE
        assert pool.shape[0] == _NATIVE_SWA_PAGES_AXIS[cfg] and mgr.max_batch_size == seqs == 1
        pool.fill_(NATIVE_SENTINEL)
        flat = pool.view(-1, NATIVE_ROW)
        container_buf = torch.empty(max_t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        n_variants = None
        stats = dict(steps=0, rows=0, round_trip_rows=0, max_t=0, max_pos=0, max_slot=0)
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(
                mgr, reqs, nctx, gen_len, uid=700_000 + 8 * i, max_num_tokens=max_t
            )
            try:
                t = sum(n for _, n in reqs)
                for layer in (0, 1, 20, 39):
                    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
                    slots = md.v41_swa_slot_mapping[mgr.layer_offsets[layer]][:t]
                    x = _fast_rows(t, seed=300_000 + 4 * i + layer)
                    op(x, codes, slots, scales)
                    rows, y = _native_rows(x)
                    op(x, container_buf)  # a fixed max_t-slot out (one identity variant)
                    container = container_buf[:t]
                    if n_variants is None:
                        n_variants = _jit_variants(kernel)
                    s64 = slots.to(torch.int64)
                    assert int(s64.min()) >= 0 and int(s64.max()) < num_slots
                    got = flat[s64]
                    assert torch.equal(got, rows), (
                        f"step {i} layer {layer}: {int((got != rows).sum())} bytes"
                    )
                    _assert_bitwise(_decode_native(got), container)
                    _assert_bitwise(container, y)
                    stats["round_trip_rows"] += _round_trip(codes, scales, slots, container)
                    flat[s64] = NATIVE_SENTINEL
                    stats["max_slot"] = max(stats["max_slot"], int(s64.max()))
                n_out = int(_count_not_sentinel(pool))
                assert n_out == 0, f"step {i}: {n_out} bytes outside the step's slots were written"
                stats["steps"] += 1
                stats["rows"] += 4 * t
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_pos"] = max(stats["max_pos"], max(c + n - 1 for c, n in reqs))
            finally:
                for r in requests:
                    mgr.free_resources(r)
        assert _jit_variants(kernel) == n_variants, "a step shape compiled a new variant"
        print(
            f"\n[native steps at max_batch_size 1, {cfg}, SWA {tuple(pool.shape)}] {stats}; one compiled variant"
        )
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t
        assert stats["max_pos"] == (16397 if draft else 16383)
        assert stats["round_trip_rows"] == stats["rows"]
    finally:
        mgr.shutdown()
        del mgr
        _free_engine_pool()


def test_native_variant_map() -> None:
    """Which calls compile a new Triton variant: the native form into a manager-geometry pool compiles one
    variant whatever T (rows are grid axis 0; the pool's slot count, a multiple of 128, is fixed); the
    identity form passes the row count as the slot count, which Triton specializes (== 1, % 16 == 0,
    other), so it compiles one variant per class and none for a second member of a class."""
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    buf, codes, scales = _native_engine_pool(64)
    added = {}
    for t in (8192, 1, 2, 16, 17, 4096, 8191):
        slots = torch.randperm(64 * ENGINE_PAGE, device="cuda")[:t]
        n0 = _jit_variants(kernel)
        op(_fast_rows(t, seed=t), codes, slots, scales)
        added[("native", t)] = _jit_variants(kernel) - n0
    classes = {"1": (1, 1), "16k": (16, 32, 48, 8192), "other": (2, 17, 33, 8191)}
    for name, ts in classes.items():
        for i, t in enumerate(ts):
            n0 = _jit_variants(kernel)
            op(
                _fast_rows(t, seed=t),
                torch.empty(t, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda"),
            )
            added[(f"identity[{name}]#{i}", t)] = _jit_variants(kernel) - n0
    # The drafter's ring write: stride-3 slot columns into the [771, 128, 512] bf16 ring.
    ring = torch.zeros(
        _ENVELOPE_RING_PAGES, ENGINE_PAGE, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda"
    )
    g = torch.Generator(device="cuda").manual_seed(2201)
    for m in (1280, 1, 2, 16, 17, 256, 1275):
        cols = _ring_cols(_ring_idx3(m, _ENVELOPE_RING_ROWS, g, masked=False))
        assert all(c.stride(0) == (1 if m == 1 else 3) for c in cols), (
            m,
            [c.stride() for c in cols],
        )
        n0 = _jit_variants(kernel)
        for s in range(3):
            op(_fast_rows(m, seed=m + s), ring, cols[s])
        added[("ring stride 3", m)] = _jit_variants(kernel) - n0
    print(f"\n[variant map] new variants per call: {added}")
    assert all(v == 0 for (form, t), v in added.items() if form == "native" and t != 8192)
    # The 1280-row call compiles the stride-3 pair (stage 1's column is 8 bytes into the table, so not
    # 16-byte aligned). M = 1: the one-element column comes out of the reshape with stride 1, the target's
    # contiguous pair (already compiled in a whole-file run). So only M >= 2 is held to no new variant.
    assert all(
        v == 0 for (form, m), v in added.items() if form == "ring stride 3" and m not in (1280, 1)
    )
    for name, ts in classes.items():
        assert all(added[(f"identity[{name}]#{i}", t)] == 0 for i, t in enumerate(ts) if i), name
    del buf, codes, scales, ring


# ─── The DSpark drafter's block K/V write: the identity form ─────────────────
#
# The drafter's attention writes its draft block's K/V with
# mxfp8_fake_quantize(kv, blk_kv): `kv` a contiguous [N, 512] bf16 tensor,
# N = 5G for G <= max_batch_size (256 in the engine envelope), and `blk_kv` =
# split(buf, [N, rest], 0)[0], the first N rows of a [ring_rows * 5 + 128, 512]
# bf16 buffer built once per ring size (ring_rows = max_batch_size + 1). No
# slot_mapping: the identity form, whose slot count is the view's row count N.
# Rows reach the kernel as grid axis 0; the slot count N is a kernel argument
# that Triton specializes by class (== 1, % 16 == 0, other), so the whole row
# axis runs on three compiled variants.
_ID_N_MAX = 8192
_ID_PAD_ROWS = 128  # the drafter's buffer carries 128 rows past the largest block
_DSPARK_RING_ROWS = 256 + 1
_DSPARK_BLK_ROWS = _DSPARK_RING_ROWS * 5 + _ID_PAD_ROWS  # 1413
_DSPARK_BLOCK_T = tuple(
    5 * g for g in _ENVELOPE_GSET
)  # N = 5G for every G of _ENVELOPE_GSET: 46 blocks, 5..1280


def _sentinel_buffer(rows: int) -> torch.Tensor:
    buf = torch.empty(rows, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
    buf.view(torch.int16).fill_(SENTINEL)
    return buf


def test_identity_form_every_row_count_1_to_8192() -> None:
    """Every row count of the identity form in the drafter's call shape: out = the first T rows of a longer buffer.

    The 8192-row call is bitwise equal to the reference and leaves the 128 rows
    past it at the sentinel. Then for every T = 1..8192 the first T rows are
    reset to the sentinel and the op runs on (x[:T], buf[:T]): those rows hold
    the first T rows of the 8192-row result bit for bit, and every element past
    row T - 1 is still the sentinel. After one call per row-count class, the
    sweep compiles no Triton variant. Mismatch counts accumulate on the device;
    one sync at the end.
    """
    t = _ID_N_MAX
    x = _window_kv_like(t, seed=5160)
    y = _fake_quant(x)[0].view(torch.int16)
    buf = _sentinel_buffer(t + _ID_PAD_ROWS)
    whole = buf.view(torch.int16)
    kernel = _mx_op._mxfp8_fake_quantize_kernel
    n0 = _jit_variants(kernel)
    op(x, buf[:t])
    assert torch.equal(whole[:t], y), "the 8192-row identity call differs from the reference"
    assert bool((whole[t:] == SENTINEL).all()), "the 8192-row identity call wrote past its view"
    for r in (1, 2):  # the two other row-count classes
        op(x[:r], buf[:r])
    n1 = _jit_variants(kernel)
    whole.fill_(SENTINEL)

    bad_rows = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_tail = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(1, t + 1):
        whole[:r].fill_(SENTINEL)
        op(x[:r], buf[:r])
        bad_rows += (whole[:r] != y[:r]).sum()
        bad_tail += (whole[r:] != SENTINEL).sum()
    n2 = _jit_variants(kernel)
    print(
        f"\n[identity rows 1..{t}] Triton variants: {n0} before, {n1} after one call per class, {n2} after the "
        f"sweep; element mismatches {int(bad_rows)}, elements written past row T - 1: {int(bad_tail)}"
    )
    assert int(bad_rows) == 0, f"{int(bad_rows)} elements differ from the 8192-row result's rows"
    assert int(bad_tail) == 0, f"{int(bad_tail)} elements past the view were written"
    assert n1 - n0 <= 3 and n2 == n1, f"row counts compiled new variants: {n0} -> {n1} -> {n2}"


def test_identity_row_sweep_harness_sees_a_shift_or_an_extra_row() -> None:
    """Controls for the identity sweep: a read one row off, a write one row off and one row too many are all seen."""
    x = _window_kv_like(66, seed=5161)
    y = _fake_quant(x)[0].view(torch.int16)
    buf = _sentinel_buffer(66 + _ID_PAD_ROWS)
    whole = buf.view(torch.int16)
    for r in (1, 2, 7, 63, 64):
        cases = (
            ("row +1", x[1 : r + 1], buf[:r]),
            ("slot +1", x[:r], buf[1 : r + 1]),
            ("one row too many", x[: r + 1], buf[: r + 1]),
        )
        for name, xa, oa in cases:
            whole.fill_(SENTINEL)
            op(xa, oa)
            n_bad = int((whole[:r] != y[:r]).sum())
            n_tail = int((whole[r:] != SENTINEL).sum())
            print(
                f"[identity control {name}] r={r}: {n_bad} / {r * WINDOW_KV_WIDTH} elements differ, "
                f"{n_tail} written past row {r - 1}"
            )
            if name != "one row too many":  # the rows check sees a shifted read or write
                assert n_bad > r * WINDOW_KV_WIDTH // 2
            if name != "row +1":  # the tail check sees a write at row r
                assert n_tail == WINDOW_KV_WIDTH


def test_identity_form_dspark_block_captured_at_every_g() -> None:
    """The drafter's block K/V write captured at N = 5G for every G of _ENVELOPE_GSET (46 blocks, 5..1280),
    replayed on new rows.

    As the drafter does it at max_batch_size 256: one shared [257 * 5 + 128, 512] buffer, each graph's
    `out` the first N rows of it, an eager call at each shape before its
    capture, all graphs in one pool, largest first. Three rounds of interleaved
    replays, each on new rows written into the captured input with the whole
    buffer reset to the sentinel: the replay writes the reference into rows
    0..N-1 bit for bit, equals the same call run eagerly, differs from the
    previous round's, and leaves every row from N on at the sentinel. A control
    round writes the new rows where the graph does not read them; the replay
    then matches only the stale rows.
    """
    blk = _sentinel_buffer(_DSPARK_BLK_ROWS)
    whole = blk.view(torch.int16)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for n in sorted(_DSPARK_BLOCK_T, reverse=True):
        x = torch.empty(n, WINDOW_KV_WIDTH, dtype=torch.bfloat16, device="cuda")
        x.copy_(_window_kv_like(n, seed=n))
        out = torch.split(blk, [n, _DSPARK_BLK_ROWS - n], 0)[0]
        op(x, out)  # the engine's warm-up: one eager call at this shape
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            op(x, out)
        graphs[n] = (graph, x, out)
    torch.cuda.synchronize()

    order = _DSPARK_BLOCK_T[::2] + _DSPARK_BLOCK_T[1::2][::-1]
    previous = {}
    for rnd in range(3):
        for n in order:
            graph, x, out = graphs[n]
            new = _window_kv_like(n, seed=10_000 * (rnd + 1) + n)
            x.copy_(new)
            whole.fill_(SENTINEL)
            graph.replay()
            torch.cuda.synchronize()
            got = whole[:n].clone()
            assert torch.equal(got, _fake_quant(new)[0].view(torch.int16)), (n, rnd)
            assert bool((whole[n:] == SENTINEL).all()), (n, rnd)
            eager = torch.empty_like(x)
            op(x, eager)
            assert torch.equal(got, eager.view(torch.int16)), (n, rnd)
            if n in previous:
                assert not torch.equal(got, previous[n]), (n, rnd)
            previous[n] = got

    for n in order:
        graph, x, out = graphs[n]
        elsewhere = _window_kv_like(n, seed=99_999 + n)
        whole.fill_(SENTINEL)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(whole[:n], _fake_quant(x)[0].view(torch.int16)), n
        assert not torch.equal(whole[:n], _fake_quant(elsewhere)[0].view(torch.int16)), n
    assert len(_DSPARK_BLOCK_T) == 46 and max(_DSPARK_BLOCK_T) == 1280
    print(
        f"\n[dspark block] {len(_DSPARK_BLOCK_T)} graphs, N {_DSPARK_BLOCK_T}, 3 rounds: every replay bitwise "
        f"the reference and eager, rows past N untouched; stale-input control rejected at every N"
    )
