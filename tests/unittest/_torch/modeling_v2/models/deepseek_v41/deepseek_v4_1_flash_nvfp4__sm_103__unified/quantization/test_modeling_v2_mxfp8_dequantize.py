# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_dequantize catalog entry.

The entry owns what is certified: `CELLS` is the list (value domain, widths, the checkpoint
recipe's rows, paged gathers, the 528 B layouts, strided slots and out, the shadow pool, scatter,
zero rows), `reference` and `compare` are the bitwise gate. This file turns a cell's spec into
tensors and keeps the checks a cell cannot carry: NaN encodings, CUDA-graph replay, the op's own
rejections, the deliberately-wrong decodes, and the native pools of the live DeepSeek-V4.1 cache
manager (far-end slots, pool-size invariance, every row count, per-query window reads, the
fixed-size window buffer, CUDA-graph capture at every graph batch, renumbered reads over random
multi-request steps).

Every comparison is bitwise (bf16 compared as int16): each output element is one e4m3 value times
a power of two, so there is no tolerance to choose. The entry's reference decodes with the torch
float8_e4m3fn -> fp64 cast and torch.ldexp(code, byte - 127) in fp64, then fp64 -> fp32 (exact: at
most 4 significant bits, exponent >= -136) -> bf16 (torch's round-to-nearest-even). The kernel
instead builds the scale from the IEEE exponent bits, so the two derivations are independent.

The fake-quantized operands (codes + UE8M0 bytes of window-KV-like rows) are also produced here in
native torch, with the checkpoint reference's recipe (amax floored at 1e-4, scale
2^ceil(log2(amax / 448)), exponent from torch.frexp), so "dequantize(native) == the bf16
container's value" is checked without calling any other catalog entry.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.mxfp8_dequantize import (  # noqa: E501
    mxfp8_dequantize as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mxfp8_dequantize requires a CUDA device"

BLOCK = 32
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4
W_KV = 512  # DeepSeek-V4.1 window KV head_dim (448 nope + 64 RoPE, all quantized)
NB_KV = W_KV // BLOCK
E4M3_NAN_CODES = (0x7F, 0xFF)
DEV = "cuda"
# The synthetic pools are sized far above any batch (320 pages x 128 = 40960 slots); the cache
# manager's own pool sizes are certified on live managers further down.
POOL_PAGES = 320


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def _ref(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The entry's reference decode of every slot of a [N, W] pool, as a dense bf16 [N, W]."""
    return op.reference(codes, scales, _sentinel((codes.shape[0], codes.shape[-1])))


def _control_decode(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    bias: int = 127,
    code_dtype=torch.float8_e4m3fn,
    scale_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """A deliberately-wrong decode of uint8 codes [N, W] and UE8M0 bytes [N, W/32] -> bf16 [N, W].

    `bias`, `code_dtype` and `scale_index` (which byte serves 32-block b) select the control.
    """
    n, w = codes.shape
    v = codes.contiguous().view(code_dtype).float().double().view(n, w // BLOCK, BLOCK)
    e = (scales.long() - bias).double()
    if scale_index is not None:
        e = e[:, scale_index]
    return torch.ldexp(v, e.unsqueeze(-1)).view(n, w).float().to(torch.bfloat16)


def _random_codes(n: int, w: int, g: torch.Generator) -> torch.Tensor:
    """Uniform e4m3 code bytes, the two NaN codes replaced by +-0 (they are tested separately)."""
    c = torch.randint(0, 256, (n, w), device=DEV, dtype=torch.int32, generator=g).to(torch.uint8)
    for nan in E4M3_NAN_CODES:
        c[c == nan] = nan & 0x80
    return c


def _random_bytes(n: int, nb: int, g: torch.Generator, lo: int = 0, hi: int = 254) -> torch.Tensor:
    return torch.randint(lo, hi + 1, (n, nb), device=DEV, dtype=torch.int32, generator=g).to(
        torch.uint8
    )


def _fake_quant_native(x: torch.Tensor):
    """Checkpoint-recipe MXFP8 of bf16 [T, W]: (bf16 container values, uint8 codes, uint8 UE8M0 bytes)."""
    t, w = x.shape
    xf = x.float().view(t, w // BLOCK, BLOCK)
    a = xf.abs().amax(-1).clamp_min(AMAX_FLOOR)
    m, e = torch.frexp(
        a.double() / E4M3_MAX
    )  # exact ceil(log2(.)): mantissa 0.5 is an exact power of two
    e = torch.where(m == 0.5, e - 1, e).to(torch.int32)
    s = torch.exp2(e.double()).float().unsqueeze(-1)
    q = (xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    y = (q.float() * s).view(t, w).to(torch.bfloat16)
    return y, q.view(t, w).view(torch.uint8), (e + 127).to(torch.uint8)


def _window_kv_like(t: int, w: int = W_KV, seed: int = 0) -> torch.Tensor:
    """bf16 [t, w] rows with block magnitudes 2^-30 .. 2^20, all-zero blocks, amax = 448 * 2^k and sub-1e-4
    blocks (scale byte 105, the recipe's floor), so every scale path of the recipe is present."""
    g = _gen(seed)
    nb = w // BLOCK
    mag = torch.randint(-30, 21, (t, nb, 1), device=DEV, generator=g).float()
    x = torch.randn(t, nb, BLOCK, device=DEV, generator=g) * torch.exp2(mag)
    for r in range(0, t, 4):
        k = r % 7 - 3
        b = (r // 4) % nb
        x[r, b] = 0.0
        x[r, (b + 1) % nb, 0] = 448.0 * 2.0**k
        x[r, (b + 2) % nb] = torch.randn(BLOCK, device=DEV, generator=g) * 1e-6
    return x.view(t, w).to(torch.bfloat16)


def _n_bad(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Elements whose bf16 bits differ, counted in chunks (a bool reduction materializes an int64 copy)."""
    a = actual.reshape(-1).view(torch.int16)
    e = expected.reshape(-1).view(torch.int16)
    n = torch.zeros((), dtype=torch.int64, device=a.device)
    for i in range(0, a.numel(), 1 << 26):
        n += (a[i : i + (1 << 26)] != e[i : i + (1 << 26)]).sum()
    return int(n)


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    n_bad = _n_bad(actual, expected)
    assert n_bad == 0, f"{n_bad} of {actual.numel()} elements differ bitwise"


def _sentinel(shape, device=DEV) -> torch.Tensor:
    """bf16 filled with a NaN bit pattern the op never writes (0x7FAB)."""
    return torch.full(shape, 0x7FAB, dtype=torch.int16, device=device).view(torch.bfloat16)


def _slots(
    rows: int, num_slots: int, g: torch.Generator, dtype=torch.int32, invalid_every: int = 0
) -> torch.Tensor:
    """`rows` distinct random slots of a `num_slots` pool; every `invalid_every`-th entry is invalid
    (alternately -1 and num_slots, i.e. negative and one past the end)."""
    s = torch.randperm(num_slots, device=DEV, generator=g)[:rows].to(torch.int64)
    if invalid_every:
        bad = torch.arange(0, rows, invalid_every, device=DEV)
        s[bad] = torch.where(bad % (2 * invalid_every) == 0, -1, num_slots)
    return s.to(dtype)


def _pool(pages: int, page_size: int, w: int, g: torch.Generator, lo: int = 0, hi: int = 254):
    codes = _random_codes(pages * page_size, w, g)
    scales = _random_bytes(pages * page_size, w // BLOCK, g, lo, hi)
    if page_size == 1:
        return codes, scales
    return codes.view(pages, page_size, w), scales.view(pages, page_size, w // BLOCK)


def _gather_ref(codes, scales, src: torch.Tensor) -> torch.Tensor:
    """The entry's reference rows for `src` (invalid -> zeros) from a [slots, W] / [pages, ps, W] pool."""
    return op.reference(codes, scales, _sentinel((src.shape[0], codes.shape[-1])), src)


# ─── certified cells ──────────────────────────────────────────────────────────


def _build(spec, seed: int):
    """A cell's spec -> (codes, scales, out, src_slots, dst_slots, extra).

    `extra` carries what the structural checks need: `pool` (the tensors whose bytes must not
    change), and per kind the bf16 `container` or the `wide` parent of a strided out.
    """
    g = _gen(seed)
    kind = spec["kind"]
    src = dst = None
    extra = {}
    if kind == "identity":
        n, w = spec["rows"], spec["width"]
        codes, scales = _pool(n, 1, w, g)
        if spec["pin_extremes"]:
            # bytes 0 / 1 / 254: fp32-subnormal scale, bf16-subnormal products, overflow
            scales[:3] = torch.tensor([0, 1, 254], dtype=torch.uint8, device=DEV).view(3, 1)
            codes[:3, :256] = torch.arange(256, device=DEV, dtype=torch.int32).to(torch.uint8)
            for nan in E4M3_NAN_CODES:
                codes[:3][codes[:3] == nan] = 0
        out = _sentinel((n, w))
    elif kind == "recipe":
        t = spec["rows"]
        y, codes, scales = _fake_quant_native(_window_kv_like(t, seed=seed))
        extra["container"] = y
        out = _sentinel((t, W_KV))
    elif kind == "gather":
        ps, rows = spec["page_size"], spec["rows"]
        pages = POOL_PAGES * 128 // ps
        codes, scales = _pool(pages, ps, W_KV, g, lo=100, hi=160)
        src = _slots(rows, pages * ps, g, spec["slot_dtype"], invalid_every=16 if rows >= 16 else 0)
        out = _sentinel((rows, W_KV))
    elif kind == "interleaved":
        # The native SWA row: 512 code bytes then 16 UE8M0 bytes, 528 B per slot.
        n_slots = POOL_PAGES * 128
        buf = torch.randint(0, 256, (n_slots, 528), device=DEV, dtype=torch.int32, generator=g).to(
            torch.uint8
        )
        buf[:, :W_KV] = _random_codes(n_slots, W_KV, g)
        buf[:, W_KV:] = _random_bytes(n_slots, NB_KV, g)
        if spec["paged"]:
            buf = buf.view(POOL_PAGES, 128, 528)
        codes = buf[..., :W_KV].view(torch.float8_e4m3fn)
        scales = buf[..., W_KV:].view(spec["scale_dtype"])
        src = _slots(4096, n_slots, g, invalid_every=64)
        out = _sentinel((4096, W_KV))
        extra["pool"] = (buf,)
    elif kind == "footer":
        # One uint8 page = page_size code rows of 512 B, then page_size scale rows of 16 B.
        ps, pages = spec["page_size"], 24
        buf = torch.randint(
            0, 256, (pages, ps * 528), device=DEV, dtype=torch.int32, generator=g
        ).to(torch.uint8)
        codes = buf[:, : ps * W_KV].view(pages, ps, W_KV)
        scales = buf[:, ps * W_KV :].view(pages, ps, NB_KV)
        for nan in E4M3_NAN_CODES:
            codes[codes == nan] = 0
        scales[scales == 255] = 127
        src = _slots(2000, pages * ps, g, torch.int64, invalid_every=50)
        out = _sentinel((2000, W_KV))
        extra["pool"] = (buf,)
    elif kind == "strided":
        codes, scales = _pool(POOL_PAGES, 128, W_KV, g, lo=100, hi=160)
        src = _slots(2 * 777, POOL_PAGES * 128, g)[::2]
        wide = _sentinel((777, 2 * W_KV))
        out = wide[:, :W_KV]
        extra["wide"] = wide
    elif kind == "shadow":
        codes, scales = _pool(POOL_PAGES, 128, W_KV, g, lo=100, hi=160)
        src = dst = _slots(4 * 128, POOL_PAGES * 128, g)
        src[::37] = -1
        out = _sentinel((POOL_PAGES, 128, W_KV))
    elif kind == "scatter":
        n_src, n_dst, rows = 4096, 2048, 600
        codes, scales = _pool(n_src, 1, W_KV, g, lo=100, hi=160)
        src = _slots(rows, n_src, g, torch.int64)
        dst = _slots(rows, n_dst, g, torch.int32)
        src[0:600:5] = -1  # zero rows
        dst[1:600:7] = -1  # skipped
        dst[2:600:11] = n_dst  # skipped
        out = _sentinel((n_dst, W_KV))
    elif kind == "dst_only":
        rows = 600
        codes, scales = _pool(rows, 1, W_KV, g, lo=100, hi=160)
        dst = _slots(rows, 16 * 128, g, torch.int32)
        out = _sentinel((16, 128, W_KV))
    else:
        assert kind == "zero_rows", kind
        codes, scales = _pool(64, 1, W_KV, g)
        src = torch.empty(0, dtype=torch.int32, device=DEV)
        out = _sentinel((4, W_KV))
    extra.setdefault("pool", (codes, scales))
    return codes, scales, out, src, dst, extra


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    codes, scales, out, src, dst, extra = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    before = out.clone()
    pool_before = [t.view(torch.uint8).clone() for t in extra["pool"]]
    result = op(codes, scales, out, src, dst)
    assert result is out, "the op writes out in place and returns it"
    op.compare(out, op.reference(codes, scales, before, src, dst))
    for t, b in zip(extra["pool"], pool_before):
        assert torch.equal(t.view(torch.uint8), b), "the source pool must not be written"
    if "container" in extra:
        _assert_bitwise(out, extra["container"])
        if cell.spec["rows"] >= 32:
            assert int((scales == 105).sum()) > 0, "the 1e-4 floor block (byte 105) must be present"
    if "wide" in extra:
        _assert_bitwise(extra["wide"][:, W_KV:], _sentinel((777, W_KV)))


# ─── special encodings, graphs, rejections ────────────────────────────────────


def test_nan_encodings() -> None:
    """UE8M0 byte 255 is NaN (whole block NaN); e4m3 codes 0x7F / 0xFF are NaN (that element only)."""
    g = _gen(255)
    codes, scales = _pool(8, 1, W_KV, g, lo=110, hi=140)
    scales[1, 3] = 255
    codes[2, 7] = 0x7F
    codes[2, 100] = 0xFF
    out = _sentinel((8, W_KV))
    op(codes, scales, out)
    nan = torch.isnan(out.float())
    expect_nan = torch.zeros_like(nan)
    expect_nan[1, 96:128] = True
    expect_nan[2, 7] = True
    expect_nan[2, 100] = True
    assert torch.equal(nan, expect_nan)
    ref = _ref(codes, scales)
    assert torch.equal(torch.isnan(ref.float()), expect_nan)
    _assert_bitwise(out[~expect_nan], ref[~expect_nan])


def test_cuda_graph_replay() -> None:
    """Capture after one eager warm-up; replay after the pool and the slot list change in place."""
    g = _gen(17)
    codes, scales = _pool(POOL_PAGES, 128, W_KV, g, lo=100, hi=160)
    src = _slots(1024, POOL_PAGES * 128, g, invalid_every=32)
    out = _sentinel((1024, W_KV))
    op(codes, scales, out, src)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(codes, scales, out, src)
    for step in range(2):
        new_codes, new_scales = _pool(POOL_PAGES, 128, W_KV, g, lo=90 + step, hi=170)
        codes.copy_(new_codes)
        scales.copy_(new_scales)
        src.copy_(_slots(1024, POOL_PAGES * 128, g, invalid_every=32 + step))
        out.copy_(_sentinel((1024, W_KV)))
        graph.replay()
        torch.cuda.synchronize()
        op.compare(out, _gather_ref(codes, scales, src))


@pytest.mark.parametrize(
    "case, match",
    [
        ("codes_cpu", "codes must be a CUDA tensor"),
        ("codes_bf16", "codes must be float8_e4m3fn or uint8"),
        ("width_48", "positive multiple of 32"),
        ("codes_col_stride", "unit stride along its last dim"),
        ("codes_4d", r"codes must be \[slots, C\] or \[pages, page_size, C\]"),
        ("scales_int32", "scales must be uint8 or float8_e8m0fnu"),
        ("scales_shape", "scales must be"),
        ("scales_cpu", "scales must be on"),
        ("out_fp32", "out must be bfloat16"),
        ("out_width", "out's last dim must be 512"),
        ("out_cpu", "out must be on"),
        ("src_float", "src_slots must be int32 or int64"),
        ("src_2d", "src_slots must be 1-D"),
        ("dst_cpu", "dst_slots must be on"),
        ("len_mismatch", "same length"),
        ("out_too_small", "without dst_slots out needs >= 16 slots"),
    ],
)
def test_rejections(case: str, match: str) -> None:
    g = _gen(1)
    codes, scales = _pool(16, 1, W_KV, g)
    out = _sentinel((16, W_KV))
    src = torch.arange(16, device=DEV, dtype=torch.int32)
    args = dict(codes=codes, scales=scales, out=out, src_slots=None, dst_slots=None)
    if case == "codes_cpu":
        args["codes"] = codes.cpu()
    elif case == "codes_bf16":
        args["codes"] = codes.to(torch.bfloat16)
    elif case == "width_48":
        args["codes"], args["scales"] = codes[:, :48], scales[:, :1]
    elif case == "codes_col_stride":
        args["codes"] = torch.empty(16, 2 * W_KV, dtype=torch.uint8, device=DEV)[:, ::2]
    elif case == "codes_4d":
        args["codes"], args["scales"] = codes.view(2, 2, 4, W_KV), scales.view(2, 2, 4, NB_KV)
    elif case == "scales_int32":
        args["scales"] = scales.to(torch.int32)
    elif case == "scales_shape":
        args["scales"] = scales[:, :8]
    elif case == "scales_cpu":
        args["scales"] = scales.cpu()
    elif case == "out_fp32":
        args["out"] = out.float()
    elif case == "out_width":
        args["out"] = out[:, :256]
    elif case == "out_cpu":
        args["out"] = out.cpu()
    elif case == "src_float":
        args["src_slots"] = src.float()
    elif case == "src_2d":
        args["src_slots"] = src.view(4, 4)
    elif case == "dst_cpu":
        args["dst_slots"] = src.cpu()
    elif case == "len_mismatch":
        args["src_slots"], args["dst_slots"] = src, src[:8]
    elif case == "out_too_small":
        args["out"] = out[:8]
    before = out.clone()
    with pytest.raises(ValueError, match=match):
        op(**args)
    _assert_bitwise(out, before)


# ─── deliberately-wrong controls ──────────────────────────────────────────────


def test_wrong_recipes_are_rejected_by_the_gate() -> None:
    """The bitwise gate that passes the op fails each wrong decode of the same native rows."""
    t = 4096
    y, codes, scales = _fake_quant_native(_window_kv_like(t, seed=11))
    g = _gen(12)
    n_slots = POOL_PAGES * 128
    pool_codes = torch.zeros(n_slots, W_KV, dtype=torch.uint8, device=DEV)
    pool_scales = torch.full((n_slots, NB_KV), 127, dtype=torch.uint8, device=DEV)
    src = _slots(t, n_slots, g)
    pool_codes[src.long()] = codes
    pool_scales[src.long()] = scales
    # a distinct neighbour row in every slot next to a used one, so an off-by-one slot is visible
    nb = (src.long() + 1) % n_slots
    free = ~torch.isin(nb, src.long())
    y2, c2, s2 = _fake_quant_native(_window_kv_like(t, seed=13))
    pool_codes[nb[free]] = c2[free]
    pool_scales[nb[free]] = s2[free]

    out = _sentinel((t, W_KV))
    paged_codes = pool_codes.view(POOL_PAGES, 128, W_KV)
    paged_scales = pool_scales.view(POOL_PAGES, 128, NB_KV)
    op(paged_codes, paged_scales, out, src)
    _assert_bitwise(out, y)
    op.compare(out, _gather_ref(paged_codes, paged_scales, src))

    wrong = {
        "exponent bias 126 (scale one step high)": _control_decode(codes, scales, bias=126),
        "exponent bias 128 (scale one step low)": _control_decode(codes, scales, bias=128),
        "e5m2 decode of the code bytes": _control_decode(
            codes, scales, code_dtype=torch.float8_e5m2
        ),
        "one scale byte per 64 codes (block b reads byte b // 2)": _control_decode(
            codes, scales, scale_index=torch.arange(NB_KV, device=DEV) // 2
        ),
        "slot off by one": _ref(
            pool_codes[(src.long() + 1) % n_slots], pool_scales[(src.long() + 1) % n_slots]
        ),
    }
    report = {name: _n_bad(out, w) for name, w in wrong.items()}
    print("wrong-recipe elements differing from the op (of", t * W_KV, "):", report)
    for name, n_bad in report.items():
        assert n_bad > 0, f"control '{name}' is indistinguishable from the op's result"


# ─── native storage on the live DeepSeek-V4.1 cache manager ──────────────────
#
# With TRTLLM_DSV41_KV_STORAGE=native the V4.1 DeepseekV4CacheManager keeps the
# SWA window KV natively: one uint8 pool [B, 128, 528] for all 40 layers, 512
# E4M3 codes then 16 UE8M0 bytes per row, and get_v41_native_kv_views(L, SWA)
# returns (codes [B, 128, 512], scales [B, 128, 16]), uint8 views with strides
# (67584, 528, 1). The native-storage target writes it with mxfp8_fake_quantize's
# native form and reads the rows a step's attention references with this op,
# either compact (src_slots = the referenced pool rows, the attention's window
# indices renumbered to rows of out) or into a bf16 shadow pool of the native
# pool's geometry (src_slots = dst_slots, the indices unchanged).
# B comes from free memory. The tables are the manager's native SWA pages at
# 0.5 x this GB300's total memory for every (max_num_tokens, max_batch_size,
# DSpark max_draft_len): the native gates reach 732,280 (accuracy, (8192, 32,
# 5)); over max_batch_size {8, 32} the largest is 854,440 ((8192, 8, 5)), and
# over every max_batch_size 1..32 it is B_max = 941,800
# ((8192, 1, 0): 120,550,400 slots; the pool 63.6 GB, a shadow of its geometry
# 123.4 GB). The pool does not grow monotonically as max_batch_size falls.
#
# Round trips here write the pool with torch.ops.trtllm.mxfp8_fake_quantize
# (the writer this op pairs with) and hold the reads to that writer's own bf16
# container output of the same rows and to this file's native-torch recipe.

import os  # noqa: E402
import random  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import tensorrt_llm._torch.custom_ops.mxfp8_dequantize as _dq_op  # noqa: E402 (the Triton kernel, variant probe)
import tensorrt_llm._torch.custom_ops.mxfp8_fake_quantize  # noqa: E402,F401 -- registers the writer (round-trip inputs)

_STORAGE_ENV = "TRTLLM_DSV41_KV_STORAGE"
PAGE = 128
NATIVE_ROW = 528
NATIVE_PAGE_BYTES = PAGE * NATIVE_ROW  # 67584
BYTE_SENTINEL = 0xFF  # the writer never writes it for a finite row (E4M3 -NaN / UE8M0 NaN)
SENTINEL16 = 0x7FAB  # bf16 NaN bit pattern of _sentinel(), as int16
_BMAX_DEVICE_TOTAL = 297_020_948_480
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
MBS_AXIS = tuple(range(1, 33))  # every max_batch_size the engine configuration permits (1..32)
MBS_LISTED = (1, 2, 3, 4, 8, 16, 32)
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
NATIVE_SWA_BMAX_8_32 = 854_440  # the B_max over max_batch_size {8, 32} only
NATIVE_SWA_GATE_MAX = 732_280
# The native SWA ceiling of the engine envelope: the measured maximum over its scanned tuples at 0.5 x the
# device's memory, PP2 stage managers included (stage alone), max_num_tokens {2048, 4096, 8192, 16384} x
# max_batch_size {1..8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256} x DSpark {0, 5} x max_seq_len {16K,
# 32K, 64K, 128K, 256K, 1M}, unsplit and per PP2 stage. It is stage 1's (layers 20-39) at NATIVE_SWA_CEILING_KEY
# (the SWA pool does not follow max_seq_len). Quotas above 0.5 x and unsampled max_batch_size values are not
# covered.
NATIVE_SWA_CEILING = 1_648_320
NATIVE_SWA_CEILING_KEY = (16384, 1, 0)
NATIVE_SWA_CEILING_STAGE, NATIVE_SWA_CEILING_LAYER = 1, 20
NATIVE_SWA_UNSPLIT = 1_098_720  # the pp_size 1 ceiling, at (16384, 1, 0)
# The shadow form (a bf16 pool of the native pool's geometry) runs on a native pool of this many pages: a shadow of
# the stage ceiling is 216 GB and does not fit one GB300 beside its 111 GB source; the unsplit ceiling's does.
SWA_SHADOW_PAGES = NATIVE_SWA_UNSPLIT
# Other pool sizes the invariance test runs: an engine dump's (16384, 256, 5) pool, envelope managers' pools, the
# unsplit ceiling, stage 0's largest and stage 1's largest in a coupled PP2 engine.
_SWA_EXTRA_B = (233_840, 617_600, 1_051_040, 1_098_720, 878_760, 1_098_220)
# Distinct SWA rows one step can reference: T + 127 per request (each request's
# window reaches 127 rows before its first new token), T <= 8192, <= 32 requests.
MAX_STEP_ROWS = 8192 + 127 * 32  # 12256
SWEEP_ROWS = 16384


def _jit_variants(fn) -> int:
    """Number of compiled variants Triton holds for a @triton.jit function, over all devices."""
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


def _write(x, codes, scales, slots):
    torch.ops.trtllm.mxfp8_fake_quantize(x, codes, slots, scales)


def _container(x):
    """The writer's own bf16-container output of x (identity form, a fixed 8192-row out per call size)."""
    out = torch.empty(max(8192, x.shape[0]), W_KV, dtype=torch.bfloat16, device=DEV)
    torch.ops.trtllm.mxfp8_fake_quantize(x, out, None, None)
    return out[: x.shape[0]]


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
    """A V4.1 manager with native KV storage at the gate settings and 0.5 x the device's total memory.
    max_seq_len defaults to the gates' 16384, as the engine passes it: + 14 with DSpark."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

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
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
            if pp_stage is None
            else Mapping(world_size=2, rank=pp_stage, tp_size=1, pp_size=2, gpus_per_node=4),
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
    """(the manager's SWA pool [B, 128, 528] uint8 as `layer` sees it, codes view, scales view), geometry
    asserted."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    pool = mgr.get_buffers(layer, AT.SWA)
    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
    b = pool.shape[0]
    assert pool.dtype == torch.uint8 and tuple(pool.shape) == (b, PAGE, NATIVE_ROW), pool.shape
    assert pool.stride() == (NATIVE_PAGE_BYTES, NATIVE_ROW, 1), pool.stride()
    assert tuple(codes.shape) == (b, PAGE, W_KV) and codes.stride() == pool.stride()
    assert tuple(scales.shape) == (b, PAGE, NB_KV) and scales.stride() == pool.stride()
    assert codes.data_ptr() == pool.data_ptr() and scales.data_ptr() == pool.data_ptr() + W_KV
    return pool, codes, scales


def _ceiling_swa_manager():
    """The PP2 stage-1 manager whose native SWA pool is the envelope's ceiling (NATIVE_SWA_CEILING), built as the
    stage probe builds it (Mapping(world_size 2, rank 1, tp_size 1, pp_size 2)); (manager, pool, codes, scales),
    the page count asserted."""
    mgr = _v41_native_manager(*NATIVE_SWA_CEILING_KEY, pp_stage=NATIVE_SWA_CEILING_STAGE)
    try:
        pool, codes, scales = _native_swa(mgr, NATIVE_SWA_CEILING_LAYER)
        assert pool.shape[0] == NATIVE_SWA_CEILING, pool.shape
    except BaseException:
        mgr.shutdown()
        raise
    return mgr, pool, codes, scales


def _raw_native_pool(pages: int):
    """(uint8 [pages + 1, 128, 528] with a guard page, codes view, scales view), every byte the sentinel."""
    buf = torch.full((pages + 1, PAGE, NATIVE_ROW), BYTE_SENTINEL, dtype=torch.uint8, device=DEV)
    return buf, buf[:pages, :, :W_KV], buf[:pages, :, W_KV:]


def _native_hazard_slots(num_slots: int) -> list[int]:
    """Slots of a 528-byte-row pool on both sides of every byte offset 2^31..2^36, both ends and the last page."""
    cands = {0, 1, PAGE - 1, PAGE}
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // NATIVE_ROW
        cands |= {s - 1, s, s + 1}
    cands |= {num_slots - PAGE - 1, num_slots - PAGE, num_slots - 2, num_slots - 1}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _shadow_hazard_slots(num_slots: int) -> list[int]:
    """Slots of a 1024-byte-row bf16 shadow on both sides of every byte offset 2^31..2^37."""
    cands = set()
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36, 2**37):
        s = limit // (2 * W_KV)
        cands |= {s - 1, s}
    return sorted(c for c in cands if 0 <= c < num_slots)


def _count_not(buf: torch.Tensor, value: int) -> torch.Tensor:
    """Device count of elements of the contiguous `buf` that are not `value`, in 2^30-element chunks."""
    whole = buf.view(-1)
    step = (
        1 << 26
    )  # a bool reduction materializes an int64 copy (8 bytes per element): keep chunks small
    n = torch.zeros((), dtype=torch.int64, device=DEV)
    for i in range(0, whole.numel(), step):
        n += (whole[i : i + step] != value).sum()
    return n


def _shadow(slots: int) -> torch.Tensor:
    """A bf16 shadow [slots // 128 + 1, 128, 512] (the native pool's geometry plus a guard page), all sentinel."""
    return _sentinel((slots // PAGE + 1, PAGE, W_KV))


def _check_shadow(
    shadow: torch.Tensor, slots: torch.Tensor, expected: torch.Tensor, num_slots: int
) -> None:
    """Named (valid) slots hold `expected` bitwise; restored to the sentinel, nothing else in `shadow` changed."""
    flat = shadow.view(-1, W_KV)
    s64 = slots.to(torch.int64)
    ok = (s64 >= 0) & (s64 < num_slots)
    got = flat[s64[ok]]
    n_bad = _n_bad(got, expected[ok])
    assert n_bad == 0, f"{n_bad} of {got.numel()} shadow elements in the named slots differ bitwise"
    flat.view(torch.int16)[s64[ok]] = SENTINEL16
    n_out = int(_count_not(shadow.view(torch.int16), SENTINEL16))
    assert n_out == 0, f"{n_out} shadow elements outside the named slots were written"


def _spread_slots(num_slots: int, n: int, g: torch.Generator) -> torch.Tensor:
    """n distinct slots: the last page in order, every hazard slot, the rest spread uniformly over the pool."""
    last_page = torch.arange(num_slots - PAGE, num_slots, device=DEV)
    fixed = torch.unique(
        torch.cat([last_page, torch.tensor(_native_hazard_slots(num_slots), device=DEV)])
    )
    rest = torch.randint(0, num_slots, (3 * n,), device=DEV, generator=g).unique()
    rest = rest[~torch.isin(rest, fixed)]
    rest = rest[torch.randperm(rest.numel(), device=DEV, generator=g)][: n - fixed.numel()]
    s = torch.cat([last_page, fixed[~torch.isin(fixed, last_page)], rest]).to(torch.int64)
    assert s.numel() == n and torch.unique(s).numel() == n
    return s


def test_native_swa_pools_from_the_cache_manager() -> None:
    """Every native configuration's SWA pool from the manager, over the whole max_batch_size axis: for
    (max_num_tokens, max_batch_size, DSpark max_draft_len) in {2048, 8192} x 1..32 x {0, 5} (128 managers),
    [B, 128, 528] uint8 for all 40 layers, the views its two halves; the pages equal the table. B_max over
    the axis is 941,800 at (8192, 1, 0); over max_batch_size {8, 32} it is 854,440; 732,280 at the gates."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    got = {}
    for key in _NATIVE_SWA_PAGES_AXIS:
        mgr = _v41_native_manager(*key)
        try:
            pool, _, _ = _native_swa(mgr)
            for layer in range(40):
                assert mgr.get_buffers(layer, AT.SWA).data_ptr() == pool.data_ptr(), layer
            got[key] = pool.shape[0]
        finally:
            mgr.shutdown()
            del mgr
            torch.cuda.empty_cache()
    over = {k: v for k, v in sorted(got.items()) if v > NATIVE_SWA_BMAX_8_32}
    print(
        f"\n[native SWA pages at 0.5 x device total, 128 configurations] {got}\n"
        f"[above the max_batch_size {{8, 32}} bound {NATIVE_SWA_BMAX_8_32}] {over}"
    )
    assert got == _NATIVE_SWA_PAGES_AXIS
    assert max(got.values()) == NATIVE_SWA_BMAX and got[NATIVE_SWA_BMAX_KEY] == NATIVE_SWA_BMAX
    assert max(got[k] for k in _NATIVE_SWA_PAGES) == NATIVE_SWA_BMAX_8_32
    assert max(got[k] for k in ((2048, 8, 5), (8192, 32, 5))) == NATIVE_SWA_GATE_MAX


@pytest.mark.parametrize("rows", ["hazard_int32", "hazard_int64", "spread_12256"])
def test_native_round_trip_far_end_on_the_manager(rows: str) -> None:
    """On the live PP2 stage-1 manager's ceiling native SWA pool (NATIVE_SWA_CEILING_KEY; 1,648,320 pages,
    111.4 GB): the writer's native form puts rows at the far end (both sides of every 2^31..2^36 byte offset of
    the pool, the first / last page, the last slot, and the slots on every 2^31..2^37 byte offset of a shadow of
    its geometry) or 12,256 rows spread over the whole pool; this op reads them back compact (src_slots, -1 and
    past-the-end entries among them give zero rows) into a sentinel out, bitwise equal to the writer's own
    bf16-container output of the same rows (and to the recipe), and the native pool is left as the writer left
    it.

    The shadow form needs a bf16 shadow of the source pool's geometry, which for the stage ceiling (216 GB) does
    not fit on one GB300 beside its source. So after the manager is shut down, the shadow form runs on a
    separate native pool of SWA_SHADOW_PAGES (the unsplit ceiling, [1098720, 128, 528], the manager's strides,
    a guard page) with its own far-end rows: src_slots = dst_slots into a 144.0 GB shadow, the named slots
    equal to the container and nothing else in the shadow or its guard page written.
    """

    def far_rows(num_slots, g):
        if rows.startswith("hazard"):
            return torch.tensor(
                sorted(set(_native_hazard_slots(num_slots)) | set(_shadow_hazard_slots(num_slots))),
                dtype=torch.int64,
                device=DEV,
            )
        return _spread_slots(num_slots, MAX_STEP_ROWS, g)

    sdt = torch.int32 if rows != "hazard_int64" else torch.int64
    mgr, pool, codes, scales = _ceiling_swa_manager()
    try:
        num_slots = pool.shape[0] * PAGE
        pool.fill_(BYTE_SENTINEL)
        g = _gen(8540 + len(rows))
        valid = far_rows(num_slots, g)
        x = _window_kv_like(valid.numel(), seed=8540)
        _write(x, codes, scales, valid)
        y = _container(x)
        _assert_bitwise(y, _fake_quant_native(x)[0])
        src = torch.cat([valid, torch.tensor([-1, num_slots], device=DEV)])
        expected = torch.cat([y, torch.zeros(2, W_KV, dtype=torch.bfloat16, device=DEV)])
        out = _sentinel((src.numel(), W_KV))
        op(codes, scales, out, src.to(sdt))
        _assert_bitwise(out, expected)
        written = pool.view(-1, NATIVE_ROW)[valid]
        assert torch.equal(written[:, :W_KV], _fake_quant_native(x)[1]) and torch.equal(
            written[:, W_KV:], _fake_quant_native(x)[2]
        )
        pool.view(-1, NATIVE_ROW)[valid] = BYTE_SENTINEL
        assert int(_count_not(pool, BYTE_SENTINEL)) == 0, "the read changed the native pool"
        pool_shape, pool_stride, n_live = tuple(pool.shape), pool.stride(), valid.numel()
    finally:
        mgr.shutdown()
        del mgr, pool, codes, scales
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    buf = shadow = None
    try:
        buf, codes, scales = _raw_native_pool(SWA_SHADOW_PAGES)
        assert buf[:-1].stride() == pool_stride
        ns = SWA_SHADOW_PAGES * PAGE
        g = _gen(8547 + len(rows))
        valid = far_rows(ns, g)
        x = _window_kv_like(valid.numel(), seed=8547)
        _write(x, codes, scales, valid)
        y = _container(x)
        src = torch.cat([valid, torch.tensor([-1, ns], device=DEV)])
        expected = torch.cat([y, torch.zeros(2, W_KV, dtype=torch.bfloat16, device=DEV)])
        shadow = _shadow(ns)
        op(codes, scales, shadow[:-1], src.to(sdt), src.to(sdt))
        torch.cuda.synchronize()
        _check_shadow(shadow, src, expected, ns)
        print(
            f"\n[{rows}] {n_live} rows at the far end of the stage manager's {pool_shape} native pool: compact read "
            f"== the writer's bf16 container; {valid.numel()} far-end rows of a ({SWA_SHADOW_PAGES}, 128, 528) pool, "
            f"shadow {tuple(shadow.shape)} read == the container, nothing else written"
        )
    finally:
        buf = codes = scales = shadow = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_native_harness_sees_a_wrapped_offset() -> None:
    """Control: a read from where a mod-2^32 byte offset would put the row, and a shadow write to where a
    mod-2^32 offset would put it, both fail the checks above."""
    buf, codes, scales = _raw_native_pool(NATIVE_SWA_CEILING)
    num_slots = NATIVE_SWA_CEILING * PAGE
    true = torch.tensor(
        [s for s in _native_hazard_slots(num_slots) if s * NATIVE_ROW >= 2**32],
        dtype=torch.int64,
        device=DEV,
    )
    wrapped = (true * NATIVE_ROW % 2**32) // NATIVE_ROW
    assert true.numel() >= 9 and not torch.isin(wrapped, true).any()
    x = _window_kv_like(2 * true.numel(), seed=8541)
    _write(x, codes, scales, torch.cat([true, wrapped]))
    y = _container(x)[: true.numel()]
    out = _sentinel((true.numel(), W_KV))
    op(codes, scales, out, wrapped)
    n_bad = _n_bad(out, y)
    print(f"\n[wrapped read] {n_bad} of {out.numel()} elements differ from the true rows")
    assert n_bad > out.numel() // 2
    del buf, codes, scales
    torch.cuda.empty_cache()
    num_slots = SWA_SHADOW_PAGES * PAGE  # the shadow form's largest geometry
    shadow = _shadow(num_slots)
    src = torch.arange(true.numel(), device=DEV)
    small, small_s = _pool(1, true.numel(), W_KV, _gen(8542), lo=100, hi=160)
    ref = _ref(small.view(-1, W_KV), small_s.view(-1, NB_KV))
    s_true = torch.tensor(
        [s for s in _shadow_hazard_slots(num_slots) if s * 2 * W_KV >= 2**32],
        dtype=torch.int64,
        device=DEV,
    )
    assert s_true.numel() >= 8 and s_true.numel() <= true.numel()
    s_wrapped = (s_true * 2 * W_KV % 2**32) // (2 * W_KV)
    op(small.view(-1, W_KV), small_s.view(-1, NB_KV), shadow[:-1], src[: s_true.numel()], s_wrapped)
    with pytest.raises(AssertionError, match="shadow elements in the named slots differ"):
        _check_shadow(shadow, s_true, ref[: s_true.numel()], num_slots)
    shadow.view(-1, W_KV).view(torch.int16)[s_true] = SENTINEL16
    with pytest.raises(AssertionError, match="outside the named slots were written"):
        _check_shadow(shadow, s_true[:0], ref[:0], num_slots)
    del shadow
    torch.cuda.empty_cache()


@pytest.mark.parametrize("t", [32, MAX_STEP_ROWS])
def test_native_pool_size_invariance(t: int) -> None:
    """The same rows at the same slots read through codes[:B] / scales[:B] of the live PP2 stage-1 manager's
    ceiling pool (NATIVE_SWA_CEILING_KEY, 1,648,320 pages), compact: the same bits at every B and equal to the
    writer's container. The shadow form (into shadow[:B]) runs, after the manager is shut down, on a separate
    native pool of SWA_SHADOW_PAGES (the largest whose shadow fits beside it), with the same rows at the same
    slots, at every B up to SWA_SHADOW_PAGES: the same bits, nothing else written. Then from a separate native
    pool of exactly the smallest size.

    B = the smallest pool holding the slots, the page counts whose byte count first passes 2^31..2^36, a mid
    size, the gates' largest (732,280), the max_batch_size {8, 32} bound (854,440), the gates' axis B_max
    (941,800), _SWA_EXTRA_B and the ceiling.
    """
    mgr, pool, codes, scales = _ceiling_swa_manager()
    try:
        pool.fill_(BYTE_SENTINEL)
        small = -(-t // PAGE)
        g = _gen(8543 + t)
        last = small * PAGE - 1
        perm = torch.randperm(small * PAGE, device=DEV, generator=g)
        slots = torch.cat([torch.tensor([last], device=DEV), perm[perm != last]])[:t]
        slots = slots[torch.randperm(t, device=DEV, generator=g)].to(torch.int64)
        x = _window_kv_like(t, seed=8543 + t)
        _write(x, codes, scales, slots)
        y = _container(x)
        src = slots.clone()
        src[::16] = -1
        expected = torch.where((src >= 0)[:, None], y, torch.zeros_like(y))
        sizes = sorted(
            {
                small,
                *(
                    lim // NATIVE_PAGE_BYTES + 1
                    for lim in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)
                ),
                NATIVE_SWA_CEILING // 2 + 1,
                NATIVE_SWA_GATE_MAX,
                NATIVE_SWA_BMAX_8_32,
                NATIVE_SWA_BMAX,
                *_SWA_EXTRA_B,
                NATIVE_SWA_CEILING,
            }
        )
        first = None
        for b in sizes:
            out = _sentinel((src.numel(), W_KV))
            op(codes[:b], scales[:b], out, src)
            _assert_bitwise(out, expected)
            first = out if first is None else first
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), f"B={b} differs"
        exact = pool.view(-1, NATIVE_ROW)[: small * PAGE].clone().view(small, PAGE, NATIVE_ROW)
    finally:
        mgr.shutdown()
        del mgr, pool, codes, scales
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _buf = shadow = None
    try:
        _buf, codes, scales = _raw_native_pool(SWA_SHADOW_PAGES)
        _write(x, codes, scales, slots)
        shadow = _shadow(SWA_SHADOW_PAGES * PAGE)
        for b in (v for v in sizes if v <= SWA_SHADOW_PAGES):
            op(codes[:b], scales[:b], shadow[:b], src, src)
            torch.cuda.synchronize()
            _check_shadow(shadow, src, expected, b * PAGE)
    finally:
        _buf = codes = scales = shadow = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    out = _sentinel((src.numel(), W_KV))
    op(exact[..., :W_KV], exact[..., W_KV:], out, src)
    _assert_bitwise(out, first)
    print(
        f"\n[native t={t}] B = {sizes} (compact: views of the stage manager's ceiling pool; shadow[:B] for B <= "
        f"{SWA_SHADOW_PAGES}: a pool of that geometry) and a separate [{small}, 128, 528] pool: "
        f"{int((src >= 0).sum())} "
        f"rows bitwise identical and == the writer's container"
    )


@pytest.mark.parametrize("form", ["compact", "shadow"])
def test_native_every_row_count_1_to_16384(form: str) -> None:
    """Over the step's referenced rows, every r in 1..16384 (>= the 12,256 distinct SWA rows a
    step can reference) reads exactly the first r rows of the 16384-row call.

    The pool: a native [160, 128, 528] (20,480 slots) filled by the writer at every slot. src: 16,384
    distinct slots, 1/16 of them invalid (-1 or one past the end). compact: out is a fixed [16384, 512];
    before each call its first r rows are reset to the sentinel, after it they must equal the first r
    rows of the full call (invalid -> zero). shadow: src = dst into a [160, 128, 512] sentinel shadow;
    after each call the named valid slots hold the full call's rows, the number of non-sentinel elements
    is 512 per valid row, and the named slots are reset. One compiled variant per form over the sweep.
    """
    pages = 160
    num_slots = pages * PAGE
    g = _gen(8544)
    buf = torch.empty(pages, PAGE, NATIVE_ROW, dtype=torch.uint8, device=DEV)
    codes, scales = buf[..., :W_KV], buf[..., W_KV:]
    x = _window_kv_like(num_slots, seed=8544)
    _write(x, codes, scales, torch.arange(num_slots, device=DEV))
    y = _container(x)
    src = torch.randperm(num_slots, device=DEV, generator=g)[:SWEEP_ROWS].to(torch.int64)
    bad = torch.arange(0, SWEEP_ROWS, 16, device=DEV)
    src[bad] = torch.where(bad % 32 == 0, -1, num_slots)
    src[1] = -1
    ok = (src >= 0) & (src < num_slots)
    full = torch.where(
        ok[:, None],
        y[src.clamp(0, num_slots - 1)],
        torch.zeros(1, W_KV, dtype=torch.bfloat16, device=DEV),
    )
    kernel = _dq_op._mxfp8_dequantize_kernel
    mism = torch.zeros((), dtype=torch.int64, device=DEV)
    cnt_err = torch.zeros((), dtype=torch.int64, device=DEV)
    if form == "compact":
        out = _sentinel((SWEEP_ROWS, W_KV))
        op(codes, scales, out, src)
        _assert_bitwise(out, full)
        n1 = _jit_variants(kernel)
        o16, f16 = out.view(torch.int16), full.view(torch.int16)
        for r in range(1, SWEEP_ROWS + 1):
            o16[:r] = SENTINEL16
            op(codes, scales, out, src[:r])
            mism += (o16[:r] != f16[:r]).sum()
        n2 = _jit_variants(kernel)
    else:
        shadow = _sentinel((pages, PAGE, W_KV))
        s16 = shadow.view(-1, W_KV).view(torch.int16)
        op(codes, scales, shadow, src, src)
        _check_shadow(shadow, src, full, num_slots)
        n1 = _jit_variants(kernel)
        ok_pos = torch.nonzero(ok).squeeze(1)
        vs = src[ok_pos]
        f16 = full[ok_pos].view(torch.int16)
        cnt = torch.cumsum(ok.to(torch.int64), 0).tolist()
        for r in range(1, SWEEP_ROWS + 1):
            op(codes, scales, shadow, src[:r], src[:r])
            k = cnt[r - 1]
            mism += (s16[vs[:k]] != f16[:k]).sum()
            cnt_err += ((s16 != SENTINEL16).sum() - k * W_KV).abs()
            s16[vs[:k]] = SENTINEL16
        n2 = _jit_variants(kernel)
    print(
        f"\n[{form} rows 1..{SWEEP_ROWS}] mismatches {int(mism)}, count error {int(cnt_err)}; variants {n1} -> {n2}"
    )
    assert int(mism) == 0 and int(cnt_err) == 0
    assert n2 == n1, "a row count compiled a new variant"


def test_native_per_query_window_rows_up_to_1048576() -> None:
    """The per-query window form: src (and dst) = a step's [T, 128] window table flattened, -1 and
    duplicate entries included, up to 8192 x 128 = 1,048,576 rows.

    One 1,048,576-row call against the writer's container (compact: invalid -> zero; shadow: every
    duplicate entry names the same source row, so its writes are identical and the slot holds that row).
    Then prefixes r at 128 T - 1, 128 T, 128 T + 1 for T in {128, 256, 512, 1024, 2048, 4096, 8191, 8192}
    and 32 random r in (16384, 2^20]: each compact call equals the first r rows of the full call. One
    compiled variant.
    """
    pages = 160
    num_slots = pages * PAGE
    g = _gen(8545)
    buf = torch.empty(pages, PAGE, NATIVE_ROW, dtype=torch.uint8, device=DEV)
    codes, scales = buf[..., :W_KV], buf[..., W_KV:]
    x = _window_kv_like(num_slots, seed=8545)
    _write(x, codes, scales, torch.arange(num_slots, device=DEV))
    y = _container(x)
    t = 8192
    pos = torch.sort(torch.randint(0, num_slots - 256, (t,), device=DEV, generator=g)).values
    win = pos[:, None] - torch.arange(127, -1, -1, device=DEV)[None, :]
    win = torch.where(win >= 0, win, -1)
    win[::7, :40] = -1  # windows cut by the start of a request
    src = win.reshape(-1).to(torch.int32)
    ok = src >= 0
    full_ref = torch.where(
        ok[:, None],
        y[src.long().clamp_min(0)],
        torch.zeros(1, W_KV, dtype=torch.bfloat16, device=DEV),
    )
    out = _sentinel((src.numel(), W_KV))
    kernel = _dq_op._mxfp8_dequantize_kernel
    op(codes, scales, out, src)
    _assert_bitwise(out, full_ref)
    shadow = _sentinel((pages + 1, PAGE, W_KV))
    op(codes, scales, shadow[:-1], src, src)
    torch.cuda.synchronize()
    uniq = torch.unique(src[ok].long())
    _check_shadow(shadow, uniq, y[uniq], num_slots)
    n1 = _jit_variants(kernel)
    rs = sorted(
        {
            r
            for tt in (128, 256, 512, 1024, 2048, 4096, 8191, 8192)
            for r in (128 * tt - 1, 128 * tt, 128 * tt + 1)
            if r <= src.numel()
        }
        | {int(v) for v in torch.randint(16385, 2**20 + 1, (32,), generator=g, device=DEV)}
    )
    o16 = out.view(torch.int16)
    mism = 0
    for r in rs:
        o16[:r] = SENTINEL16
        op(codes, scales, out, src[:r])
        mism += _n_bad(out[:r], full_ref[:r])
    n2 = _jit_variants(kernel)
    print(
        f"\n[per-query window rows] {src.numel()} rows ({int(ok.sum())} valid, {uniq.numel()} distinct): compact "
        f"and shadow exact; {len(rs)} prefixes, mismatches {int(mism)}; variants {n1} -> {n2}"
    )
    assert int(mism) == 0 and n2 == n1


# Captured batch sizes: every G in 1..32. At max_batch_size s the engine captures every G in 1..s with
# padding off (padding on captures a subset of those plus s), so over the max_batch_size axis 1..32 the
# union is every G in 1..32; the gates' capture set {1..8, 16, 24, 32} is part of it.
_GRAPH_BS = MBS_AXIS
# Rows a captured decode / DSpark-verify step reads: its requests' windows as distinct rows (128 per
# request, 133 with 6-token verify), or per query (128 per query: bs and 6 x bs queries).
_GRAPH_ROWS = tuple(
    sorted(
        {128 * b for b in _GRAPH_BS} | {133 * b for b in _GRAPH_BS} | {768 * b for b in _GRAPH_BS}
    )
)


@pytest.mark.parametrize("form", ["compact", "shadow"])
def test_native_cuda_graph_every_graph_batch_on_the_bmax_pool(form: str) -> None:
    """Per captured row count: an eager warm-up, the capture, then 2 replays after the writer has put new
    rows at new far-end slots of a native pool (hazard slots, the last pages): the stage ceiling's geometry
    ([1648320, 128, 528]) compact, SWA_SHADOW_PAGES ([1098720, 128, 528]) into a shadow of that geometry. Every
    replay reads exactly the writer's container (shadow: nothing else written) and equals the same call run
    eagerly."""
    pages = NATIVE_SWA_CEILING if form == "compact" else SWA_SHADOW_PAGES
    buf, codes, scales = _raw_native_pool(pages)
    num_slots = pages * PAGE
    shadow = _shadow(num_slots) if form == "shadow" else None
    g = _gen(8546)
    hazard = torch.tensor(_native_hazard_slots(num_slots), device=DEV)
    near = num_slots - 1 - torch.arange(256 * PAGE, device=DEV)
    cand = torch.unique(torch.cat([hazard, near]))
    cand = cand[cand != num_slots - 1]
    table = torch.full((max(_GRAPH_ROWS),), -1, dtype=torch.int64, device=DEV)

    def pick(n):
        """n slots: the pool's last slot, then distinct hazard / last-256-page slots; entry 2 is -1."""
        s = torch.cat(
            [
                torch.tensor([num_slots - 1], device=DEV),
                cand[torch.randperm(cand.numel(), device=DEV, generator=g)][: n - 1],
            ]
        )
        if n > 2:
            s[2] = -1
        return s

    for n in _GRAPH_ROWS:
        src = table[:n]
        s0 = pick(n)
        src.copy_(s0)
        x = _fast_rows(n, seed=8546 + n)
        _write(x, codes, scales, s0)
        out = _sentinel((n, W_KV))
        if form == "compact":
            op(codes, scales, out, src)
        else:
            op(codes, scales, shadow[:-1], src, src)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if form == "compact":
                op(codes, scales, out, src)
            else:
                op(codes, scales, shadow[:-1], src, src)
        if shadow is not None:
            shadow.view(torch.int16).fill_(SENTINEL16)
        for rep in (1, 2):
            s = pick(n)
            x = _fast_rows(n, seed=8546 + 1000 * rep + n)
            _write(x, codes, scales, s)
            y = _container(x)
            expected = torch.where((s >= 0)[:, None], y, torch.zeros_like(y))
            src.copy_(s)
            out.view(torch.int16).fill_(SENTINEL16)
            graph.replay()
            torch.cuda.synchronize()
            if form == "compact":
                replayed = out.clone()
                _assert_bitwise(replayed, expected)
                out.view(torch.int16).fill_(SENTINEL16)
                op(codes, scales, out, src)
                _assert_bitwise(out, replayed)
            else:
                live = s[s >= 0]
                replayed = shadow.view(-1, W_KV)[live].clone()
                _check_shadow(shadow, s, expected, num_slots)
                op(codes, scales, shadow[:-1], src, src)
                torch.cuda.synchronize()
                _assert_bitwise(shadow.view(-1, W_KV)[live], replayed)
                _check_shadow(shadow, s, expected, num_slots)
        del graph
    print(
        f"\n[{form} graphs on the [{pages}, 128, 528] native pool] rows {list(_GRAPH_ROWS)}: every "
        f"replay == eager == the writer's container"
    )
    del buf, codes, scales, shadow
    torch.cuda.empty_cache()


# ─── reads over random multi-request steps on the live manager ───


def _random_step(
    rng: random.Random, max_tokens: int = 8192, max_requests: int = 32, max_len: int = 16384
):
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


def _run_step(mgr, reqs, num_contexts: int, gen_len: int, uid: int, max_num_tokens: int = 8192):
    """Allocate the step's requests on the manager and build its DeepSeek-V4.1 attention metadata as the
    engine builds it (max_num_requests = the manager's max_batch_size, the engine's max_num_tokens; each
    request's first token unique over the test, so block reuse shares no page across steps)."""
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


def test_native_reads_over_random_multi_request_steps() -> None:
    """Random multi-request steps on the live accuracy-gate manager (8192, 32, DSpark 5; native SWA 732,280 pages).

    Per step (9 edge steps + 200 seeded random partitions of T <= 8192 into k <= 32 requests, contexts
    with random cached prefixes, decode or DSpark-verify generation requests) and per layer 0, 20, 39:
    the attention's window rows are taken from the step's metadata (req_idx_per_token,
    swa_local_indices_cuda, the layer's SWA sliding block table: page * 128 + position % 128), the
    writer's native form puts the step's new rows at md.v41_swa_slot_mapping[local(L)][:T] and the
    earlier rows the windows reach at theirs, and this op reads the step's referenced rows

    - compact over the distinct referenced rows, and per query ([T * 128], -1 -> zero rows);
    - into a bf16 shadow of the pool's geometry (96 GB): src = dst = the distinct rows, and again
      src = dst = the per-query [T * 128] table (duplicates and -1 included);

    every read equal to the writer's container of the row, nothing else in the shadow written; after
    each step the native pool outside the step's rows is still the sentinel.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    mgr = _v41_native_manager(8192, 32, 5)
    rng = random.Random(8547)
    steps = list(_EDGE_STEPS) + [_random_step(rng) for _ in range(_NUM_RANDOM_STEPS)]
    kernel = _dq_op._mxfp8_dequantize_kernel
    shadow = None
    try:
        pool, _, _ = _native_swa(mgr)
        num_slots = pool.shape[0] * PAGE
        assert pool.shape[0] == NATIVE_SWA_GATE_MAX
        pool.fill_(BYTE_SENTINEL)
        flat = pool.view(-1, NATIVE_ROW)
        shadow = _shadow(num_slots)
        s16 = shadow.view(-1, W_KV).view(torch.int16)
        stats = dict(steps=0, max_t=0, max_k=0, max_distinct=0, max_per_query=0, rows=0)
        n_variants = None
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(mgr, reqs, nctx, gen_len, uid=500_000 + 32 * i)
            try:
                t = sum(n for _, n in reqs)
                req = md.req_idx_per_token[:t].long()
                loc = md.swa_local_indices_cuda[:t].long()
                for layer in (0, 20, 39):
                    local = mgr.layer_offsets[layer]
                    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
                    bt = md.sliding_block_tables[local, AT.SWA.value]
                    page = torch.where(loc >= 0, bt[req[:, None], loc.clamp_min(0) // PAGE], -1)
                    win = torch.where((loc >= 0) & (page >= 0), page * PAGE + loc % PAGE, -1)
                    assert int(((loc >= 0) & (page < 0)).sum()) == 0, (
                        "a window position without a page"
                    )
                    new = md.v41_swa_slot_mapping[local][:t]
                    uniq = torch.unique(win[win >= 0])
                    old = uniq[~torch.isin(uniq, new)]
                    assert torch.isin(new, uniq).all() and uniq.numel() <= MAX_STEP_ROWS
                    x_new = _fast_rows(t, seed=600_000 + 8 * i + layer)
                    x_old = _fast_rows(max(old.numel(), 1), seed=700_000 + 8 * i + layer)[
                        : old.numel()
                    ]
                    _write(x_old, codes, scales, old)
                    _write(x_new, codes, scales, new)  # the layer's forward write
                    # The writer's container of every referenced row, by slot.
                    slots_all = torch.cat([new.long(), old])
                    y_all = (
                        torch.cat([_container(x_new), _container(x_old)])
                        if old.numel()
                        else _container(x_new)
                    )
                    order = torch.argsort(slots_all)
                    slots_sorted, y_sorted = slots_all[order], y_all[order]

                    def lookup(s):
                        idx = torch.searchsorted(slots_sorted, s.clamp_min(0))
                        return torch.where(
                            (s >= 0)[:, None],
                            y_sorted[idx.clamp(max=slots_sorted.numel() - 1)],
                            torch.zeros(1, W_KV, dtype=torch.bfloat16, device=DEV),
                        )

                    out = _sentinel((uniq.numel(), W_KV))
                    op(codes, scales, out, uniq)
                    _assert_bitwise(out, lookup(uniq))
                    per_query = win.reshape(-1)
                    out_q = _sentinel((per_query.numel(), W_KV))
                    op(codes, scales, out_q, per_query)
                    _assert_bitwise(out_q, lookup(per_query))
                    op(codes, scales, shadow[:-1], uniq, uniq)
                    got = shadow.view(-1, W_KV)[uniq]
                    _assert_bitwise(got, lookup(uniq))
                    s16[uniq] = SENTINEL16
                    op(codes, scales, shadow[:-1], per_query, per_query)
                    _assert_bitwise(shadow.view(-1, W_KV)[uniq], lookup(uniq))
                    s16[uniq] = SENTINEL16
                    if n_variants is None:
                        n_variants = _jit_variants(kernel)
                    flat[slots_all] = BYTE_SENTINEL
                    stats["max_distinct"] = max(stats["max_distinct"], uniq.numel())
                    stats["max_per_query"] = max(stats["max_per_query"], per_query.numel())
                    stats["rows"] += uniq.numel()
                n_out = int(_count_not(shadow.view(torch.int16), SENTINEL16))
                assert n_out == 0, (
                    f"step {i}: {n_out} shadow elements outside the referenced rows were written"
                )
                n_out = int(_count_not(pool, BYTE_SENTINEL))
                assert n_out == 0, (
                    f"step {i}: {n_out} native bytes outside the step's rows were written"
                )
                stats["steps"] += 1
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_k"] = max(stats["max_k"], len(reqs))
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
        print(
            f"\n[native reads over multi-request steps, SWA {tuple(pool.shape)}, shadow {tuple(shadow.shape)}] "
            f"{stats}; variants {n_variants} -> {_jit_variants(kernel)}"
        )
        assert stats["steps"] == len(_EDGE_STEPS) + _NUM_RANDOM_STEPS and stats["max_t"] == 8192
        assert stats["max_k"] == 32
    finally:
        mgr.shutdown()
        del mgr
        shadow = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _fast_rows(t: int, seed: int) -> torch.Tensor:
    """bf16 [t, 512] window-KV-like rows, vectorized: one magnitude 2^-30..2^20 per 32-block, one zero block."""
    g = _gen(seed)
    mag = torch.randint(-30, 21, (t, NB_KV, 1), device=DEV, generator=g).float()
    x = torch.randn(t, NB_KV, BLOCK, device=DEV, generator=g) * torch.exp2(mag)
    pick = torch.randint(0, NB_KV, (t, 1, 1), device=DEV, generator=g)
    x = torch.where(torch.arange(NB_KV, device=DEV).view(1, NB_KV, 1) == pick, 0.0, x)
    return x.view(t, W_KV).to(torch.bfloat16)


def test_native_variant_map() -> None:
    """Which calls compile a new Triton variant. Rows are grid axis 0 and never a kernel argument, but the
    out's slot count is one, and Triton specializes integers (== 1, % 16 == 0, other): a compact out sized
    to the step's rows compiles one variant per class (and none for a second member of a class); a
    fixed-size compact out or a shadow of the pool's geometry compiles one variant whatever the rows."""
    kernel = _dq_op._mxfp8_dequantize_kernel
    buf = torch.empty(64, PAGE, NATIVE_ROW, dtype=torch.uint8, device=DEV)
    codes, scales = buf[..., :W_KV], buf[..., W_KV:]
    _write(_fast_rows(64 * PAGE, seed=1), codes, scales, torch.arange(64 * PAGE, device=DEV))
    fixed = _sentinel((8192, W_KV))
    shadow = _sentinel((64, PAGE, W_KV))
    added = {}
    for form in ("fixed", "shadow"):
        for n in (8192, 1, 2, 16, 17, 4096, 8191):
            src = torch.randperm(64 * PAGE, device=DEV)[:n]
            n0 = _jit_variants(kernel)
            if form == "fixed":
                op(codes, scales, fixed, src)
            else:
                op(codes, scales, shadow, src, src)
            added[(form, n)] = _jit_variants(kernel) - n0
    classes = {"1": (1, 1), "16k": (16, 32, 48, 8192), "other": (2, 17, 33, 8191)}
    for name, ns in classes.items():
        for i, n in enumerate(ns):
            src = torch.randperm(64 * PAGE, device=DEV)[:n]
            n0 = _jit_variants(kernel)
            op(codes, scales, _sentinel((n, W_KV)), src)
            added[(f"exact[{name}]#{i}", n)] = _jit_variants(kernel) - n0
    print(f"\n[variant map] new variants per call: {added}")
    assert all(
        v == 0 for (form, n), v in added.items() if form in ("fixed", "shadow") and n != 8192
    )
    for name, ns in classes.items():
        assert all(added[(f"exact[{name}]#{i}", n)] == 0 for i, n in enumerate(ns) if i), name


# ─── the native-storage target's attention read: a fixed-size out, a renumbered index table ───
#
# The native-storage target (the unified sm_103 DeepSeek-V4.1 modeling with native KV storage) reads a layer's
# window KV in one call per layer and step:
#
#     mxfp8_dequantize(codes, scales, swa_c, src_slots)
#
# - (codes, scales) = mgr.get_v41_native_kv_views(L, SWA), the live manager's native views;
# - swa_c is ONE persistent bf16 buffer [max_num_tokens * 128, 512], allocated once per manager from the
#   metadata (md.swa_local_indices_cuda is [max_num_tokens, 128]) and zero-filled, never reset;
# - src_slots is the window half of the attention's int32 index table, flattened to [T * 128]: the table is
#   [T, 128] at a ratio-0 layer (a view) and [T, 128 + 512] at a compressed layer (the window half split
#   off, so the flatten is a copy). Entries are pool slots (page * 128 + position % 128) or -1.
#
# Query t's column c lands at row t * 128 + c (a -1 column gives a zero row), and the attention is handed
# the index table with its window half replaced by that fixed numbering, arange(max_num_tokens * 128)
# .view(max_num_tokens, 128)[:T], and its compressed half unchanged. Rows >= T * 128 keep whatever an
# earlier call left and are never named. The attention on bf16 storage instead reads the bf16-container
# pool at the original index; the tests below hold the two reads bitwise equal.

_KVQ_ROWS = {
    2048: 262_144,
    8192: 1_048_576,
}  # swa_c rows at each max_num_tokens of the gates' envelope
# swa_c rows at every max_num_tokens of the engine envelope (max_num_tokens * 128, measured below).
_ENVELOPE_ROWS = {2048: 262_144, 4096: 524_288, 8192: 1_048_576, 16384: 2_097_152}
# Every graph batch of the engine envelope: every G 1..32 and TRT-LLM's padded list to max_batch_size 256.
_ENVELOPE_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
assert len(_ENVELOPE_GSET) == 46
# Row counts up to this buffer size are run one by one; the 2,097,152-row buffer's call takes over 1 ms
# (measured in the test), so its sweep is the sampled set allowed for a call of over 1 ms (_kvq_row_counts).
_ENVELOPE_EVERY_ROW_MAX = 1_048_576
_KVQ_LAYERS = {
    0: 0,
    2: 2,
    20: 1,
    39: 1,
}  # layer -> compress ratio (0: a [T, 128] table, else [T, 640])
_KVQ_LOGICAL_PAGES = (
    129  # logical pages of a request at max_seq_len 16398 (DSpark): positions <= 16397
)
S64 = 0x7FAB7FAB7FAB7FAB  # four SENTINEL16 as one int64


def _kvq_random_step(
    rng: random.Random, max_tokens: int = 8192, max_requests: int = 32, seq_len: int = 16384
):
    """_random_step over the engine's envelope with DSpark: contexts end at <= seq_len tokens, decode
    requests (next_n 1) at <= seq_len, DSpark-verify requests (next_n 6) at <= seq_len + 14 (positions to
    16397 at the gates' 16384)."""
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


_KVQ_EDGE_STEPS = list(_EDGE_STEPS) + [
    ([(16392, 6)] * 32, 0, 6),  # DSpark verify at the engine's max_seq_len 16398
    ([(8384, 8000)] + [(16392, 6)] * 31, 1, 6),
    ([(0, 8192 - 31 * 6)] + [(16392 - 97 * i, 6) for i in range(31)], 1, 6),
]


def _kvq_page_candidates(num_pages: int, n: int, g: torch.Generator) -> torch.Tensor:
    """n distinct pool pages: every page holding a hazard slot (both sides of the 2^31..2^35 byte lines) in
    random order, then the pool's last pages and random pages alternately."""
    hz = torch.tensor(
        sorted({s // PAGE for s in _native_hazard_slots(num_pages * PAGE)}), device=DEV
    )
    hz = hz[torch.randperm(hz.numel(), device=DEV, generator=g)]
    last = num_pages - 1 - torch.arange(n, device=DEV)
    last = last[~torch.isin(last, hz)]
    rnd = torch.randint(0, num_pages, (4 * n + 64,), device=DEV, generator=g).unique()
    rnd = rnd[~torch.isin(rnd, torch.cat([hz, last]))]
    rnd = rnd[torch.randperm(rnd.numel(), device=DEV, generator=g)][: last.numel()]
    tail = torch.stack([last[: rnd.numel()], rnd], 1).reshape(-1)
    cand = torch.cat([hz, tail, last[rnd.numel() :]])[:n]
    assert cand.numel() == n and torch.unique(cand).numel() == n
    return cand.to(torch.int64)


def _kvq_window(reqs, num_pages: int, g: torch.Generator) -> torch.Tensor:
    """[T, 128] int64 window table of a step whose request b holds positions c_b .. c_b + n_b - 1: the query
    at position p names positions p - 127 .. p (negative -> -1). Each (request, logical page) a window
    touches gets its own pool page from _kvq_page_candidates (so the hazard pages are always among the rows
    read); position q of request b is at slot page(b, q // 128) * 128 + q % 128."""
    req_of = torch.cat(
        [torch.full((n,), b, dtype=torch.int64) for b, (_, n) in enumerate(reqs)]
    ).to(DEV)
    pos = torch.cat([torch.arange(c, c + n) for c, n in reqs]).to(DEV)
    wpos = pos[:, None] - (PAGE - 1) + torch.arange(PAGE, device=DEV)[None]
    q = wpos.clamp_min(0)
    key = req_of[:, None] * _KVQ_LOGICAL_PAGES + q // PAGE
    touched = torch.unique(key[wpos >= 0])
    page_of = torch.full((len(reqs) * _KVQ_LOGICAL_PAGES,), -1, dtype=torch.int64, device=DEV)
    page_of[touched] = _kvq_page_candidates(num_pages, touched.numel(), g)
    return torch.where(wpos >= 0, page_of[key] * PAGE + q % PAGE, -1)


def _kvq_partition(t: int, rng: random.Random, k_max: int = 32):
    """t query tokens as contexts of up to k_max requests ending at <= 16384: request 0 from position 0
    (its first windows reach before the sequence: -1 columns), the others at random cached prefixes."""
    k = min(k_max, t, rng.randint(1, k_max))
    cuts = sorted(rng.sample(range(1, t), k - 1)) if k > 1 else []
    news = [b - a for a, b in zip([0] + cuts, cuts + [t])]
    return [(0 if b == 0 else rng.randint(0, 16384 - n), n) for b, n in enumerate(news)]


def _kvq_expected(
    win_flat: torch.Tensor, slots_sorted: torch.Tensor, y_sorted: torch.Tensor
) -> torch.Tensor:
    """The bf16-container row of each entry of win_flat (slots sorted with their container rows; -1 -> +0.0)."""
    idx = torch.searchsorted(slots_sorted, win_flat.clamp_min(0)).clamp(
        max=slots_sorted.numel() - 1
    )
    out = y_sorted[idx]
    out[win_flat < 0] = 0.0
    return out


def test_kvq_read_buffer_rows_from_the_metadata() -> None:
    """The window read buffer the target allocates, [md.swa_local_indices_cuda.shape[0] * 128, 512], over the
    whole max_batch_size axis: in every configuration (max_num_tokens, max_batch_size, DSpark max_draft_len)
    of {2048, 8192} x 1..32 x {0, 5} (128 managers; max_seq_len 16384 without DSpark, 16398 with it),
    metadata built as the engine builds it on the native manager of that configuration, and the CUDA-graph
    metadata the engine derives from it (create_cuda_graph_metadata at G = 1 and G = max_batch_size), give
    swa_local_indices_cuda [max_num_tokens, 128] int32. So the buffer is max_num_tokens * 128 rows --
    262,144 at 2048 and 1,048,576 at 8192 -- whatever the max_batch_size or DSpark setting. The engine's
    max_seq_len and the compressed tables' [max_batch_size, 132] are recorded alongside."""
    got, seen = {}, set()
    for key in _NATIVE_SWA_PAGES_AXIS:
        max_t, seqs, draft = key
        mgr = _v41_native_manager(*key)
        try:
            assert mgr.max_seq_len == (16398 if draft else 16384) and mgr.max_batch_size == seqs
            requests, md = _run_step(mgr, [(0, 100)], 1, 1, uid=950_000, max_num_tokens=max_t)
            try:
                rows = set()
                for G in sorted({1, seqs}):
                    gmd = md.create_cuda_graph_metadata(G, False, draft)
                    for m in (md, gmd):
                        sw = m.swa_local_indices_cuda
                        assert tuple(sw.shape) == (max_t, PAGE) and sw.dtype == torch.int32, (
                            key,
                            G,
                            sw.shape,
                        )
                        rows.add(sw.shape[0] * PAGE)
                        seen.add(
                            tuple(m.compress_block_tables[1].shape) + (mgr.max_blocks_per_seq,)
                        )
                        assert tuple(m.compress_block_tables[1].shape) == (
                            seqs,
                            mgr.max_blocks_per_seq,
                        )
                assert len(rows) == 1
                got[key] = rows.pop()
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
        finally:
            mgr.shutdown()
            del mgr
            torch.cuda.empty_cache()
    per_t = {t: sorted({v for k, v in got.items() if k[0] == t}) for t in _KVQ_ROWS}
    print(
        f"\n[native-storage window read buffer rows over 128 configurations] rows per max_num_tokens {per_t}; "
        f"compressed tables (max_batch_size, cols, max_blocks_per_seq) seen {sorted(seen)}"
    )
    assert got == {k: _KVQ_ROWS[k[0]] for k in _NATIVE_SWA_PAGES_AXIS}
    assert {c for _, c, _ in seen} == {132}


def _kvq_row_counts(n_rows: int) -> list[int]:
    """Every row count 1..n_rows up to _ENVELOPE_EVERY_ROW_MAX; above it (a call of over 1 ms) every
    r in 1..16384 and every r = {0, 1, -1} mod 128 (the rows of one query's window) up to n_rows, which holds
    every row count the target's call makes (128 T) and every dumped one."""
    if n_rows <= _ENVELOPE_EVERY_ROW_MAX:
        return list(range(1, n_rows + 1))
    rs = set(range(1, 16385))
    for t in range(1, n_rows // PAGE + 1):
        rs |= {t * PAGE - 1, t * PAGE, t * PAGE + 1}
    return sorted(r for r in rs if r <= n_rows)


@pytest.mark.parametrize("n_rows", sorted(_ENVELOPE_ROWS.values()))
def test_kvq_fixed_out_every_row_count(n_rows: int) -> None:
    """Every row count in the target's form, on the live accuracy-gate manager's native pool (8192, 32, 5):
    out is the fixed [n_rows, 512] read buffer (every max_num_tokens 2048, 4096, 8192, 16384 of the
    envelope), src_slots an int32 window table of n_rows / 128 queries (up to 32 requests, -1 columns, rows
    repeated by neighbouring queries, the pool's hazard and last pages) flattened. The n_rows call equals the
    writer's bf16 container; its time (median of 11 CUDA-event timings) is printed. Then every r of
    _kvq_row_counts(n_rows): out[:r] reset to the sentinel, the call with src_slots[:r], out[:r] bitwise the
    first r rows of the n_rows call and row r still the sentinel (a write past r would persist until its own
    row's check); a guard past the buffer untouched; no compiled variant added over the sweep."""
    mgr = _v41_native_manager(8192, 32, 5)
    try:
        pool, codes, scales = _native_swa(mgr)
        g = _gen(15002 + n_rows)
        rng = random.Random(15002 + n_rows)
        reqs = _kvq_partition(n_rows // PAGE, rng)
        win = _kvq_window(reqs, pool.shape[0], g)
        uniq = torch.unique(win[win >= 0])
        x = _fast_rows(uniq.numel(), seed=15003 + n_rows)
        _write(x, codes, scales, uniq)
        src = win.reshape(-1).to(torch.int32)
        big = _sentinel((n_rows + PAGE, W_KV))
        out = big[:n_rows]
        op(codes, scales, out, src)
        _assert_bitwise(out, _kvq_expected(win.reshape(-1), uniq, _container(x)))
        full = out.clone()
        times = []
        for _ in range(11):
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            op(codes, scales, out, src)
            ev1.record()
            torch.cuda.synchronize()
            times.append(ev0.elapsed_time(ev1))
        assert torch.equal(out.view(torch.int16), full.view(torch.int16))
        assert n_rows <= _ENVELOPE_EVERY_ROW_MAX or sorted(times)[5] > 1.0, (
            f"the sampled buffer's call takes {times} ms"
        )
        kernel = _dq_op._mxfp8_dequantize_kernel
        n1 = _jit_variants(kernel)
        o64, f64 = out.view(torch.int64), full.view(torch.int64)
        # One preallocated bool buffer and a logical reduction: a bool .sum() would materialize an int64
        # copy of every compare (1 GB at the largest r) and churn the allocator for a million sizes.
        neq = torch.empty((n_rows, W_KV // 4), dtype=torch.bool, device=DEV)
        mism = torch.zeros((), dtype=torch.int64, device=DEV)
        past = torch.zeros((), dtype=torch.int64, device=DEV)
        o64.fill_(S64)
        counts = _kvq_row_counts(n_rows)
        for r in counts:
            o64[:r].fill_(S64)
            op(codes, scales, out, src[:r])
            torch.ne(o64[:r], f64[:r], out=neq[:r])
            mism += neq[:r].any()
            if r < n_rows:
                past += (o64[r] != S64).any()
        n2 = _jit_variants(kernel)
        guard = int(_count_not(big[n_rows:].view(torch.int16), SENTINEL16))
        print(
            f"\n[fixed out [{n_rows}, 512], {len(reqs)} requests, {uniq.numel()} distinct rows, "
            f"{int((src < 0).sum())} -1 entries; the {n_rows}-row call {sorted(times)[5]:.3f} ms] "
            f"{len(counts)} row counts ({'every r' if n_rows <= _ENVELOPE_EVERY_ROW_MAX else 'sampled'} in "
            f"1..{n_rows}): with a prefix mismatch {int(mism)}, with a write past r {int(past)}, guard {guard}; "
            f"variants {n1} -> {n2}"
        )
        assert int(mism) == 0 and int(past) == 0 and guard == 0
        assert n2 == n1, "a row count compiled a new variant"
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _kvq_window_read_steps(mgr, steps, max_num_tokens: int, uid0: int, seed0: int):
    """Run the target's window read over `steps` on the live manager `mgr` (see
    test_kvq_window_read_renumbered_over_random_steps for what each step checks). Returns
    (pool shape, buffer shape, stats, controls)."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    container = swa_c = None
    try:
        pool, _, _ = _native_swa(mgr)
        container = torch.empty((pool.shape[0], PAGE, W_KV), dtype=torch.bfloat16, device=DEV)
        cflat = container.view(-1, W_KV)
        g = _gen(seed0)
        stats = dict(
            steps=0,
            layer_steps=0,
            max_t=0,
            max_k=0,
            max_pos=0,
            max_valid_cols=0,
            rows=0,
            zero_rows=0,
            max_distinct=0,
        )
        ctrl = {"renumbered off by one": [0, 0], "stale buffer row": [0, 0]}
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(
                mgr, reqs, nctx, gen_len, uid=uid0 + 64 * i, max_num_tokens=max_num_tokens
            )
            try:
                if swa_c is None:
                    max_t = md.swa_local_indices_cuda.shape[0]
                    assert max_t == max_num_tokens
                    swa_c = torch.zeros((max_t * PAGE, W_KV), dtype=torch.bfloat16, device=DEV)
                    win_idx = torch.arange(max_t * PAGE, dtype=torch.int32, device=DEV).view(
                        max_t, PAGE
                    )
                t = sum(n for _, n in reqs)
                assert len(reqs) <= mgr.max_batch_size and t <= max_num_tokens
                req = md.req_idx_per_token[:t].long()
                loc = md.swa_local_indices_cuda[:t].long()
                for layer, ratio in _KVQ_LAYERS.items():
                    local = mgr.layer_offsets[layer]
                    codes, scales = mgr.get_v41_native_kv_views(layer, AT.SWA)
                    bt = md.sliding_block_tables[local, AT.SWA.value]
                    page = torch.where(loc >= 0, bt[req[:, None], loc.clamp_min(0) // PAGE], -1)
                    win = torch.where((loc >= 0) & (page >= 0), page * PAGE + loc % PAGE, -1)
                    assert int(((loc >= 0) & (page < 0)).sum()) == 0, (
                        "a window position without a page"
                    )
                    new = md.v41_swa_slot_mapping[local][:t].long()
                    uniq = torch.unique(win[win >= 0])
                    old = uniq[~torch.isin(uniq, new)]
                    assert torch.isin(new, uniq).all()
                    x_new = _fast_rows(t, seed=seed0 + 2_000_000 + 8 * i + layer)
                    x_old = _fast_rows(max(old.numel(), 1), seed=seed0 + 3_000_000 + 8 * i + layer)[
                        : old.numel()
                    ]
                    for x, s in ((x_old, old), (x_new, new)):
                        _write(x, codes, scales, s)
                        torch.ops.trtllm.mxfp8_fake_quantize(x, container, s, None)
                    if ratio:
                        gidx = torch.randint(
                            -1, 1 << 20, (t, PAGE + 512), dtype=torch.int32, device=DEV, generator=g
                        )
                        gidx[:, :PAGE] = win
                        win_v, rest = gidx.split([PAGE, 512], 1)
                        src = win_v.reshape(t * PAGE)
                    else:
                        gidx, rest = win.to(torch.int32), None
                        src = gidx.reshape(t * PAGE)
                    before = swa_c.clone()
                    op(codes, scales, swa_c, src)
                    ren = win_idx[:t]
                    table = ren if rest is None else torch.cat([ren, rest], 1)
                    if rest is not None:
                        assert torch.equal(table[:, PAGE:], gidx[:, PAGE:])
                    named = table[:, :PAGE].reshape(-1).long()
                    wf = win.reshape(-1)
                    ref = cflat[wf.clamp_min(0)]
                    ref[wf < 0] = 0.0
                    n_bad = _n_bad(swa_c[named], ref)
                    assert n_bad == 0, (
                        f"step {i} layer {layer}: {n_bad} elements differ from the bf16 path"
                    )
                    n_tail = _n_bad(swa_c[t * PAGE :], before[t * PAGE :])
                    assert n_tail == 0, (
                        f"step {i} layer {layer}: {n_tail} elements past row {t * PAGE} changed"
                    )
                    valid = bool((wf >= 0).any())
                    for name, rows_of in (
                        (
                            "renumbered off by one",
                            lambda: swa_c[(named + 1).clamp(max=swa_c.shape[0] - 1)],
                        ),
                        ("stale buffer row", lambda before=before: before[named]),
                    ):
                        c = _n_bad(rows_of(), ref)
                        ctrl[name][0] += int(c > 0 or not valid)
                        ctrl[name][1] += c
                    del before, ref
                    stats["layer_steps"] += 1
                    stats["rows"] += t * PAGE
                    stats["zero_rows"] += int((wf < 0).sum())
                    stats["max_valid_cols"] = max(
                        stats["max_valid_cols"], int((win >= 0).sum(1).max())
                    )
                    stats["max_distinct"] = max(stats["max_distinct"], uniq.numel())
                stats["steps"] += 1
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_k"] = max(stats["max_k"], len(reqs))
                stats["max_pos"] = max(stats["max_pos"], max(c + n - 1 for c, n in reqs))
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
                torch.cuda.empty_cache()  # per-step gathers vary in size; keep the cache from crowding the pools
        return tuple(pool.shape), tuple(swa_c.shape), stats, ctrl
    finally:
        container = swa_c = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_kvq_window_read_renumbered_over_random_steps() -> None:
    """Random multi-request steps in the target's exact call, on the live accuracy-gate manager (8192, 32, DSpark 5).

    Per step (12 edge steps, among them DSpark verify at positions up to 16397, + 200 seeded random
    partitions of T <= 8192 into k <= 32 requests: contexts with random cached prefixes, decode (next_n 1)
    or DSpark-verify (next_n 6) requests) and per layer 0 (ratio 0), 2 (ratio 2), 20 and 39 (ratio 1):

    - the window half of the attention's index table from the step's metadata (req_idx_per_token,
      swa_local_indices_cuda, the layer's SWA sliding block table: page * 128 + position % 128, or -1);
    - the forward's window writes, twice from the same bf16 rows at the same slots: the writer's native form
      into the manager's pool, and its bf16-container form into a bf16 pool of the pool's geometry (96 GB);
    - the target's read: its [T, 128] or [T, 640] int32 table, window half flattened, into ONE persistent
      zero-filled [md.swa_local_indices_cuda.shape[0] * 128, 512] buffer shared by every layer and step;
    - the renumbered table: window half = arange(max_t * 128).view(max_t, 128)[:T], compressed half
      unchanged.

    For every (t, c): the buffer row the renumbered table names is bitwise the bf16-container pool row the
    original table names (+0.0 where it is -1), and the buffer past T * 128 is exactly as before the call.
    Controls on the same state: the renumbered table off by one row, and the buffer as it was before this
    call (a stale row), each differ on every layer-step with a valid window row.
    """
    mgr = _v41_native_manager(8192, 32, 5)
    rng = random.Random(15004)
    steps = list(_KVQ_EDGE_STEPS) + [_kvq_random_step(rng) for _ in range(_NUM_RANDOM_STEPS)]
    try:
        assert _native_swa(mgr)[0].shape[0] == NATIVE_SWA_GATE_MAX
        pool_shape, buf_shape, stats, ctrl = _kvq_window_read_steps(
            mgr, steps, 8192, uid0=1_000_000, seed0=15004
        )
        print(
            f"\n[native-storage window read over multi-request steps, SWA {pool_shape}, buffer "
            f"{buf_shape}] {stats}; controls [layer-steps detected, elements]: {ctrl}"
        )
        assert stats["steps"] == len(_KVQ_EDGE_STEPS) + _NUM_RANDOM_STEPS and stats["max_t"] == 8192
        assert (
            stats["max_k"] == 32 and stats["max_pos"] == 16397 and stats["max_valid_cols"] == PAGE
        )
        for name, (hit, _) in ctrl.items():
            assert hit == stats["layer_steps"], (
                f"control '{name}' missed {stats['layer_steps'] - hit} layer-steps"
            )
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# The max_batch_size axis: the window read on a manager built at each max_batch_size.
_KVQ_MBS_CONFIGS = [(8192, s, 5) for s in MBS_AXIS] + [
    (2048, 1, 5)
]  # (2048, 1, 5): the native-storage witness
_KVQ_MBS_RANDOM = {s: (_NUM_RANDOM_STEPS if s in MBS_LISTED else 20) for s in MBS_AXIS}


def _kvq_axis_edge_steps(max_t: int, seqs: int):
    """Edge steps that fit a (max_num_tokens, max_batch_size) configuration: the file's edge steps with at
    most `seqs` requests and `max_t` tokens, plus a full step of max_t tokens from position 0 (-1 window
    columns) and one ending at position 16383, `seqs` decode requests at 16383, `seqs` DSpark-verify
    requests at positions 16392..16397, and (seqs > 1) a context filling the step beside seqs - 1 verifies."""
    cands = [
        st for st in _KVQ_EDGE_STEPS if len(st[0]) <= seqs and sum(n for _, n in st[0]) <= max_t
    ]
    cands += [
        ([(0, max_t)], 1, 1),
        ([(16384 - max_t, max_t)], 1, 1),
        ([(16383, 1)] * seqs, 0, 1),
        ([(16392, 6)] * seqs, 0, 6),
    ]
    if seqs > 1:
        cands.append(
            ([(0, max_t - 6 * (seqs - 1))] + [(16392 - 97 * i, 6) for i in range(seqs - 1)], 1, 6)
        )
    steps = []
    for st in cands:
        if st not in steps:
            steps.append(st)
    return steps


@pytest.mark.parametrize(
    "cfg", _KVQ_MBS_CONFIGS, ids=[f"{t}x{s}x{d}" for t, s, d in _KVQ_MBS_CONFIGS]
)
def test_kvq_window_read_renumbered_at_every_max_batch_size(cfg) -> None:
    """The max_batch_size axis of the target's window read (random multi-request steps at every
    max_batch_size): the live native manager built at (max_num_tokens, max_batch_size, DSpark 5) for every
    max_batch_size 1..32 at 8192, and the native witness's (2048, 1, 5). Per configuration: the edge steps that fit it
    (_kvq_axis_edge_steps: a full max_num_tokens step, max_batch_size requests at the DSpark tail) and
    seeded random partitions of T <= max_num_tokens into k <= max_batch_size requests (200 at max_batch_size
    1, 2, 3, 4, 8, 16, 32 and at the witness, 20 at every other), each step exactly as in
    test_kvq_window_read_renumbered_over_random_steps: layers 0 / 2 / 20 / 39, the buffer
    [md.swa_local_indices_cuda.shape[0] * 128, 512] (asserted = max_num_tokens * 128), every renumbered row
    bitwise the bf16-container row of that manager's pool geometry (+0.0 for -1), rows past 128 T
    unchanged, and both controls seen on every layer-step."""
    max_t, seqs, draft = cfg
    n_random = 200 if cfg == (2048, 1, 5) else _KVQ_MBS_RANDOM[seqs]
    rng = random.Random(15_100 + 100 * seqs + max_t)
    edges = _kvq_axis_edge_steps(max_t, seqs)
    steps = edges + [
        _kvq_random_step(rng, max_tokens=max_t, max_requests=seqs) for _ in range(n_random)
    ]
    mgr = _v41_native_manager(*cfg)
    try:
        pages = _native_swa(mgr)[0].shape[0]
        assert pages == _NATIVE_SWA_PAGES_AXIS[cfg] and pages <= NATIVE_SWA_BMAX
        pool_shape, buf_shape, stats, ctrl = _kvq_window_read_steps(
            mgr, steps, max_t, uid0=2_000_000 + 100_000 * seqs, seed0=15_200 + 1_000 * seqs + max_t
        )
        print(
            f"\n[native-storage window read, max_batch_size {seqs}, max_num_tokens {max_t}, SWA {pool_shape}, buffer "
            f"{buf_shape}] {len(edges)} edge + {n_random} random steps: {stats}; controls: {ctrl}"
        )
        assert buf_shape == (max_t * PAGE, W_KV)
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_pos"] == 16397 and stats["max_valid_cols"] == PAGE
        for name, (hit, _) in ctrl.items():
            assert hit == stats["layer_steps"], (
                f"control '{name}' missed {stats['layer_steps'] - hit} layer-steps"
            )
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("n_rows", sorted(_ENVELOPE_ROWS.values()))
def test_kvq_window_read_cuda_graph_every_graph_batch_on_the_bmax_pool(n_rows: int) -> None:
    """The target's read under CUDA-graph capture, on the live PP2 stage-1 manager's native pool
    (NATIVE_SWA_CEILING_KEY; 1,648,320 pages, the envelope's measured maximum), into the fixed [n_rows, 512]
    buffer of every
    max_num_tokens 2048, 4096, 8192, 16384 (zero-filled, a sentinel guard past it).

    First the buffer's largest step, eager: n_rows / 128 queries whose windows lie on the pool's hazard
    and last pages, equal to the writer's container. That call is the only warm-up. Then per batch in every
    G of _ENVELOPE_GSET (1..32 and 40..128 step 8, 192, 256: the capture sets of every max_batch_size up to 256,
    padding on or off; the graph depends only on T) x next_n {1, 6} (decode, DSpark verify; T = G x next_n
    queries, 1..1536, at positions up to 16383 / 16397) and per table form
    ([T, 128] view; [T, 640] with the window half split and flattened inside the graph): capture, then 2
    replays after the writer puts new rows at new far-end slots and the table changes in place. Each
    replay: rows < T * 128 equal the container (+0.0 for -1), rows past them unchanged, and the same call
    run eagerly gives the same bits. No compiled variant is added after the warm-up, and the guard stays
    the sentinel.
    """
    mgr, pool, codes, scales = _ceiling_swa_manager()
    try:
        g = _gen(15005 + n_rows)
        rng = random.Random(15005 + n_rows)
        max_t = n_rows // PAGE
        big = torch.zeros((n_rows + PAGE, W_KV), dtype=torch.bfloat16, device=DEV)
        big[n_rows:].view(torch.int16).fill_(SENTINEL16)
        swa_c = big[:n_rows]
        tab640 = torch.full((max_t, PAGE + 512), -1, dtype=torch.int32, device=DEV)
        tab128 = torch.full((max_t, PAGE), -1, dtype=torch.int32, device=DEV)
        kernel = _dq_op._mxfp8_dequantize_kernel

        def fresh(reqs):
            win = _kvq_window(reqs, pool.shape[0], g)
            uniq = torch.unique(win[win >= 0])
            x = _fast_rows(uniq.numel(), seed=rng.randint(0, 2**30))
            _write(x, codes, scales, uniq)
            return win, _kvq_expected(win.reshape(-1), uniq, _container(x))

        win, expected = fresh(_kvq_partition(max_t, rng))
        tab640[:, :PAGE] = win
        op(codes, scales, swa_c, tab640.split([PAGE, 512], 1)[0].reshape(n_rows))
        _assert_bitwise(swa_c, expected)
        n1 = _jit_variants(kernel)
        graphs = 0
        for bs in _ENVELOPE_GSET:
            for next_n in (1, 6):
                t = bs * next_n
                end = 16398 if next_n == 6 else 16384
                for form in ("ratio0", "compressed"):
                    tab = tab128 if form == "ratio0" else tab640
                    win, _ = fresh([(rng.randint(0, end - next_n), next_n) for _ in range(bs)])
                    tab[:t, :PAGE] = win
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        src = (
                            tab[:t].reshape(t * PAGE)
                            if form == "ratio0"
                            else tab[:t].split([PAGE, 512], 1)[0].reshape(t * PAGE)
                        )
                        op(codes, scales, swa_c, src)
                    graphs += 1
                    for rep in (1, 2):
                        kv = [rng.randint(0, end - next_n) for _ in range(bs)]
                        if rep == 1:
                            kv[0] = end - next_n  # the last position the engine reaches
                        win, expected = fresh([(c, next_n) for c in kv])
                        tab[:t, :PAGE] = win
                        before = swa_c.clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        _assert_bitwise(swa_c[: t * PAGE], expected)
                        _assert_bitwise(swa_c[t * PAGE :], before[t * PAGE :])
                        replayed = swa_c[: t * PAGE].clone()
                        swa_c.copy_(before)
                        eager_src = (
                            tab[:t].reshape(t * PAGE)
                            if form == "ratio0"
                            else tab[:t].split([PAGE, 512], 1)[0].reshape(t * PAGE)
                        )
                        op(codes, scales, swa_c, eager_src)
                        _assert_bitwise(swa_c[: t * PAGE], replayed)
                        del before
                    del graph
        n2 = _jit_variants(kernel)
        guard = int(_count_not(big[n_rows:].view(torch.int16), SENTINEL16))
        print(
            f"\n[native-storage window read graphs, pool {tuple(pool.shape)}, buffer [{n_rows}, 512]] eager {max_t} "
            f"queries on far-end pages == the container; {graphs} graphs (bs {list(_ENVELOPE_GSET)} x next_n 1/6 x 2 "
            f"table forms): every replay == eager == the container, rows past T * 128 unchanged; variants after "
            f"the first eager call {n1} -> {n2}; guard {guard}"
        )
        assert n2 == n1 and guard == 0 and graphs == 2 * 2 * len(_ENVELOPE_GSET)
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ─── the engine envelope: max_num_tokens to 16,384, max_batch_size to 256, max_seq_len to 1,048,576 ───
#
# The target's call is the one above, unchanged: mxfp8_dequantize(codes, scales, swa_c, win.reshape(T * 128)),
# with swa_c [md.swa_local_indices_cuda.shape[0] * 128, 512]. The envelope moves T (up to max_num_tokens
# 16,384: 2,097,152 rows; under capture T = G x next_n for every G of _ENVELOPE_GSET: 128 * T up to 196,608 rows) and
# the buffer (every max_num_tokens 2048, 4096, 8192, 16384). The call has no width that follows max_seq_len.

# Native SWA pages at 0.5 x this GB300's total memory per (max_num_tokens, max_batch_size, DSpark
# max_draft_len, max_seq_len as L), measured.
_ENVELOPE_SWA_PAGES = {
    (2048, 1, 0, 16384): 549_600,
    (2048, 1, 5, 16384): 549_600,
    (2048, 32, 0, 16384): 366_400,
    (2048, 32, 5, 16384): 569_440,
    (2048, 33, 0, 16384): 366_400,
    (2048, 33, 5, 16384): 569_440,
    (2048, 64, 0, 16384): 366_400,
    (2048, 64, 5, 16384): 561_000,
    (2048, 128, 0, 16384): 342_560,
    (2048, 128, 5, 16384): 543_120,
    (2048, 256, 0, 16384): 326_680,
    (2048, 256, 5, 16384): 552_560,
    (4096, 1, 0, 16384): 659_320,
    (4096, 1, 5, 16384): 659_320,
    (4096, 32, 0, 16384): 506_880,
    (4096, 32, 5, 16384): 628_040,
    (4096, 33, 0, 16384): 506_880,
    (4096, 33, 5, 16384): 606_200,
    (4096, 64, 0, 16384): 399_640,
    (4096, 64, 5, 16384): 582_840,
    (4096, 128, 0, 16384): 361_440,
    (4096, 128, 5, 16384): 573_440,
    (4096, 256, 0, 16384): 349_520,
    (4096, 256, 5, 16384): 559_040,
    (8192, 1, 0, 16384): 941_800,
    (8192, 1, 5, 16384): 879_240,
    (8192, 32, 0, 16384): 585_840,
    (8192, 32, 5, 16384): 732_280,
    (8192, 33, 0, 16384): 585_840,
    (8192, 33, 5, 16384): 732_280,
    (8192, 64, 0, 16384): 503_400,
    (8192, 64, 5, 16384): 663_280,
    (8192, 128, 0, 16384): 423_480,
    (8192, 128, 5, 16384): 601_240,
    (8192, 256, 0, 16384): 370_840,
    (8192, 256, 5, 16384): 583_360,
    (16384, 1, 0, 16384): 1_098_720,
    (16384, 1, 5, 16384): 1_051_040,
    (16384, 32, 0, 16384): 751_640,
    (16384, 32, 5, 16384): 857_400,
    (16384, 33, 0, 16384): 751_640,
    (16384, 33, 5, 16384): 837_040,
    (16384, 64, 0, 16384): 628_040,
    (16384, 64, 5, 16384): 744_200,
    (16384, 128, 0, 16384): 506_880,
    (16384, 128, 5, 16384): 661_280,
    (16384, 256, 0, 16384): 420_520,
    (16384, 256, 5, 16384): 617_600,
    (16384, 256, 0, 20480): 358_440,
    (16384, 256, 5, 20480): 536_200,
    (16384, 256, 0, 32768): 249_720,
    (16384, 256, 5, 32768): 384_760,
    (16384, 256, 0, 65536): 137_520,
    (16384, 256, 5, 65536): 221_400,
    (16384, 256, 0, 131072): 72_480,
    (16384, 256, 5, 131072): 119_120,
    (16384, 256, 0, 262144): 37_200,
    (16384, 256, 5, 262144): 62_040,
    (16384, 256, 0, 1048576): 10_920,
    (16384, 256, 5, 1048576): 15_880,
}


def test_envelope_window_buffer_rows_from_the_metadata() -> None:
    """The window read buffer over the engine envelope: in 60 native managers -- max_num_tokens 2048,
    4096, 8192, 16384 x max_batch_size 1, 32, 33, 64, 128, 256 x DSpark 0 / 5 at max_seq_len 16384, and
    (16384, 256) x DSpark 0 / 5 at every max_seq_len 20480 .. 1,048,576 (+ 14 with DSpark, as the engine
    passes it) -- metadata built as the engine builds it and its CUDA-graph copies at G = 1 and G =
    max_batch_size have swa_local_indices_cuda [max_num_tokens, 128] int32. So swa_c is max_num_tokens * 128
    rows (262,144 / 524,288 / 1,048,576 / 2,097,152) whatever max_batch_size, DSpark or max_seq_len. The SWA
    pool pages at 0.5 x the device's memory are pinned in _ENVELOPE_SWA_PAGES."""
    got, pages = {}, {}
    for key in _ENVELOPE_SWA_PAGES:
        max_t, seqs, draft, L = key
        lm = L + 14 if draft else L
        mgr = _v41_native_manager(max_t, seqs, draft, max_seq_len=lm)
        try:
            assert mgr.max_seq_len == lm and mgr.max_batch_size == seqs
            pages[key] = _native_swa(mgr)[0].shape[0]
            requests, md = _run_step(mgr, [(0, 100)], 1, 1, uid=980_000, max_num_tokens=max_t)
            try:
                rows = set()
                for G in sorted({1, seqs}):
                    gmd = md.create_cuda_graph_metadata(G, False, draft)
                    for m in (md, gmd):
                        sw = m.swa_local_indices_cuda
                        assert tuple(sw.shape) == (max_t, PAGE) and sw.dtype == torch.int32, (
                            key,
                            G,
                            sw.shape,
                        )
                        rows.add(sw.shape[0] * PAGE)
                assert len(rows) == 1
                got[key] = rows.pop()
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
        finally:
            mgr.shutdown()
            del mgr
            torch.cuda.empty_cache()
    per_t = {t: sorted({v for k, v in got.items() if k[0] == t}) for t in _ENVELOPE_ROWS}
    print(
        f"\n[envelope window read buffer rows over {len(got)} configurations] rows per max_num_tokens {per_t}; "
        f"SWA pages {pages}; largest {max(pages.values())}"
    )
    assert got == {k: _ENVELOPE_ROWS[k[0]] for k in _ENVELOPE_SWA_PAGES}
    assert pages == _ENVELOPE_SWA_PAGES
    assert max(pages.values()) == NATIVE_SWA_UNSPLIT == pages[NATIVE_SWA_CEILING_KEY + (16384,)]


# KV tokens one step of the 256-request test may hold in total. The native manager at (16384, 256, DSpark
# 5, max_seq_len 32,782) held 22 requests of 32,768 tokens and refused the 23rd (about 720K tokens), so a
# step of 256 requests cannot put every request at the far end; steps keep their total under this bound.
_ENVELOPE_STEP_TOKENS = 560_000


def _envelope_window_edge_steps(max_t: int, seqs: int, L: int):
    """Edge steps of a (max_num_tokens, max_batch_size) configuration at max_seq_len L with DSpark, inside
    _ENVELOPE_STEP_TOKENS: a full max_t step from position 0 (-1 window columns) and one ending at L; seqs decode
    requests, 8 at L - 1 and the rest below position 1152; seqs DSpark verifies, 8 at positions up to L + 13
    and the rest below 1130; and a context filling the step beside seqs - 1 verifies (7 at the far end)."""
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
        assert sum(a + b for a, b in reqs) <= _ENVELOPE_STEP_TOKENS, reqs[:3]
    return steps


def _envelope_fit(step, rng: random.Random):
    """A random step with its total KV tokens capped at _ENVELOPE_STEP_TOKENS: its requests, in a random order,
    keep their cached prefix while the rest can still fit; the others get a random prefix below 1024. The
    request order (contexts first) and every request's new tokens are unchanged."""
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


def test_envelope_window_read_renumbered_over_random_steps_256_requests() -> None:
    """Random multi-request steps at the envelope's step axes, in the target's exact call, on the live native
    manager of the engine configuration max_num_tokens 16384, max_batch_size 256, DSpark 5, max_seq_len 32768 + 14:
    _envelope_window_edge_steps and 200 seeded random partitions of T <= 16,384 into k <= 256 requests (contexts
    first with random cached prefixes, decode next_n 1 at positions < 32,768 or DSpark verify at < 32,782),
    each step's KV tokens capped at _ENVELOPE_STEP_TOKENS (_envelope_fit). Each step exactly as
    test_kvq_window_read_renumbered_over_random_steps: layers 0 / 2 / 20 / 39, the buffer
    [md.swa_local_indices_cuda.shape[0] * 128, 512] (asserted 2,097,152 rows), every renumbered row bitwise
    the bf16-container row the original index names (+0.0 for -1), rows past 128 T unchanged, and both
    controls seen on every layer-step."""
    L, max_t, seqs = 32768, 16384, 256
    rng = random.Random(16_400 + seqs + L)
    edges = _envelope_window_edge_steps(max_t, seqs, L)
    steps = edges + [
        _envelope_fit(_kvq_random_step(rng, max_tokens=max_t, max_requests=seqs, seq_len=L), rng)
        for _ in range(_NUM_RANDOM_STEPS)
    ]
    mgr = _v41_native_manager(max_t, seqs, 5, max_seq_len=L + 14)
    try:
        pages = _native_swa(mgr)[0].shape[0]
        pool_shape, buf_shape, stats, ctrl = _kvq_window_read_steps(
            mgr, steps, max_t, uid0=4_000_000, seed0=16_500
        )
        print(
            f"\n[envelope window read, ({max_t}, {seqs}, 5) at max_seq_len {L + 14}, SWA {pool_shape}, buffer "
            f"{buf_shape}] {len(edges)} edge + {_NUM_RANDOM_STEPS} random steps: {stats}; controls: {ctrl}"
        )
        assert pages == pool_shape[0] == _ENVELOPE_SWA_PAGES[(max_t, seqs, 5, L)] and buf_shape == (
            max_t * PAGE,
            W_KV,
        )
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_pos"] == L + 13 and stats["max_valid_cols"] == PAGE
        for name, (hit, _) in ctrl.items():
            assert hit == stats["layer_steps"], (
                f"control '{name}' missed {stats['layer_steps'] - hit} layer-steps"
            )
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
