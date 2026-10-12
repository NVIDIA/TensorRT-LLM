# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_to_e4m3 catalog entry.

The entry owns what is certified: `CELLS` is the list (the compact gather at each inv_scale from a
paged and a flat pool, the engine-sized pool into the compact window buffer, the identity form,
dst_slots scatter), `reference` and `compare` are the bitwise gate. This file turns a cell's spec
into tensors and keeps the checks a cell cannot carry: normal-range exactness, saturation, signed
zeros and NaN, CUDA-graph replay, the wrong-variant controls and the op's own rejections.

Every comparison is bitwise (uint8 codes): the entry's reference decodes with the float8_e4m3fn ->
fp64 cast of the code, torch.ldexp(code, byte - 127) in fp64, the multiply by the fp32 inverse scale,
clamp to +-448, then torch's round-to-nearest-even cast to float8_e4m3fn. The kernel builds the scale
from the IEEE exponent bits and multiplies in fp32, so the two derivations are independent.

The native pool is DeepSeek-V4.1's SWA pool: 528-byte rows (512 E4M3 codes then 16 UE8M0 bytes) in
128-slot pages, passed as two strided views, with the pool sized from the engine's cache manager (an
engine runtime-value dump: [188160, 128, 528]) in the engine-pool cells and far above any batch
elsewhere.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.mxfp8_to_e4m3 import (  # noqa: E501
    mxfp8_to_e4m3 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mxfp8_to_e4m3 requires a CUDA device"

BLOCK = 32
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4
W = 512
NB = W // BLOCK
PAGE = 128
ROW_BYTES = W + NB  # 528
DEV = "cuda"
POOL_PAGES = 320  # 40960 slots, far above any batch here
E4M3_NAN_CODES = (0x7F, 0xFF)


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _fake_quant_native(x):
    """Checkpoint-recipe MXFP8 of bf16 [T, W]: (fp32 values, uint8 codes, uint8 UE8M0 bytes)."""
    t, w = x.shape
    xf = x.float().view(t, w // BLOCK, BLOCK)
    a = xf.abs().amax(-1).clamp_min(AMAX_FLOOR)
    m, e = torch.frexp(a.double() / E4M3_MAX)
    e = torch.where(m == 0.5, e - 1, e).to(torch.int32)
    s = torch.exp2(e.double()).float().unsqueeze(-1)
    q = (xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    y = (q.float() * s).view(t, w)
    return y, q.view(t, w).view(torch.uint8), (e + 127).to(torch.uint8)


def _window_kv_like(t, seed=0, lo=-12, hi=4):
    """bf16 [t, 512] rows whose 32-blocks have magnitudes 2^lo .. 2^hi (DeepSeek-V4.1's normed window KV
    lies in about 2^-10 .. 2^7), plus all-zero and sub-floor blocks."""
    g = _gen(seed)
    mag = torch.randint(lo, hi + 1, (t, NB, 1), device=DEV, generator=g).float()
    x = torch.randn(t, NB, BLOCK, device=DEV, generator=g) * torch.exp2(mag)
    for r in range(0, t, 5):
        x[r, r % NB] = 0.0
        x[r, (r + 1) % NB] = torch.randn(BLOCK, device=DEV, generator=g) * 1e-6
    return x.view(t, W).to(torch.bfloat16)


def _ref(codes, scales, inv_scale):
    """The entry's reference transcode of every slot of a [N, W] pool -> uint8 E4M3 codes [N, W]."""
    out = torch.empty(codes.shape[0], codes.shape[-1], dtype=torch.float8_e4m3fn, device=DEV)
    return op.reference(codes, scales, out, inv_scale).view(torch.uint8)


def _control(codes, scales, inv_scale, *, code_dtype=torch.float8_e4m3fn, bias=127):
    """A deliberately-wrong transcode of uint8 codes [N, W] + UE8M0 bytes [N, W/32] -> uint8 codes."""
    n, w = codes.shape
    v = codes.contiguous().view(code_dtype).float().double().view(n, w // BLOCK, BLOCK)
    e = (scales.long() - bias).double().unsqueeze(-1)
    y = torch.ldexp(v, e).float() * torch.tensor(
        inv_scale, dtype=torch.float32, device=codes.device
    )
    y = y.clamp(-E4M3_MAX, E4M3_MAX).view(n, w)
    return y.to(torch.float8_e4m3fn).view(torch.uint8)


def _native_pool(pages, seed=0):
    """[pages, 128, 528] uint8 pool + its (codes, scales) views, filled with checkpoint-recipe rows."""
    pool = torch.zeros(pages, PAGE, ROW_BYTES, dtype=torch.uint8, device=DEV)
    n = pages * PAGE
    x = _window_kv_like(n, seed)
    _, codes, sc = _fake_quant_native(x)
    flat = pool.view(n, ROW_BYTES)
    flat[:, :W] = codes
    flat[:, W:] = sc
    return pool, pool[..., :W], pool[..., W:]


def _assert_bitwise(actual, expected):
    bad = (actual.reshape(-1) != expected.reshape(-1)).sum().item()
    assert bad == 0, f"{bad} of {actual.numel()} codes differ"


# ─── certified cells ──────────────────────────────────────────────────────────


def _build(spec, seed: int):
    """A cell's spec -> (codes, scales, out, inv_scale, src_slots, dst_slots, extra).

    `extra` carries what the structural checks need: `pool` (whose bytes must not change), `tail`
    (rows past `out` in its parent buffer that must keep their fill), `last_page_start` (the
    engine pool's last page must be read) and `exactness` (the normal-range check).
    """
    g = _gen(seed)
    kind, inv_scale = spec["kind"], spec["inv_scale"]
    src = dst = None
    extra = {}
    if kind == "compact":
        # The target's call: every query's 128 window columns (with -1 tails) gathered to rows t*128+c.
        pool, codes, scales = _native_pool(POOL_PAGES, seed=seed)
        if not spec["paged"]:
            codes = codes.reshape(-1, W)
            scales = scales.reshape(-1, NB)
        T = spec["queries"]
        src = torch.randint(
            0, POOL_PAGES * PAGE, (T * 128,), device=DEV, generator=g, dtype=torch.int32
        )
        src[torch.randint(0, T * 128, (T * 20,), device=DEV, generator=g)] = -1
        buf = torch.full((T * 128 + 64, W), 0xAB, dtype=torch.uint8, device=DEV).view(
            torch.float8_e4m3fn
        )
        out = buf[: T * 128]
        extra["tail"] = buf[T * 128 :]
    elif kind == "engine_pool":
        # The engine's SWA pool and the native-storage compact window buffer [max_num_tokens * 128, 512]
        # (8192 tokens); rows named from the pool's first and last pages.
        pages, T = spec["pool_pages"], spec["queries"]
        pool = torch.zeros(pages, PAGE, ROW_BYTES, dtype=torch.uint8, device=DEV)
        codes, scales = pool[..., :W], pool[..., W:]
        n_fill = min(T * 128, 65536)
        fill_rows = torch.cat(
            [
                torch.arange(0, n_fill // 2, device=DEV),
                torch.arange(pages * PAGE - n_fill // 2, pages * PAGE, device=DEV),
            ]
        )
        x = _window_kv_like(fill_rows.numel(), seed=seed)
        _, c, s = _fake_quant_native(x)
        flat = pool.view(-1, ROW_BYTES)
        flat[fill_rows, :W] = c
        flat[fill_rows, W:] = s
        src = fill_rows[
            torch.randint(0, fill_rows.numel(), (T * 128,), device=DEV, generator=g)
        ].to(torch.int32)
        src[::13] = -1
        buf = torch.zeros(8192 * 128, W, dtype=torch.float8_e4m3fn, device=DEV)
        out = buf[: T * 128]
        extra["last_page_start"] = pages * PAGE - n_fill // 2
    elif kind == "identity":
        pool, codes, scales = _native_pool(spec["pool_pages"], seed=seed)
        out = torch.empty(spec["pool_pages"] * PAGE, W, dtype=torch.float8_e4m3fn, device=DEV)
        extra["exactness"] = True
    else:
        assert kind == "scatter", kind
        pool, codes, scales = _native_pool(8, seed=seed)
        rows = 500
        src = torch.randint(0, 8 * PAGE, (rows,), device=DEV, generator=g, dtype=torch.int64)
        dst = torch.randperm(1024, device=DEV, generator=g)[:rows]
        dst[::7] = -1
        dst[1::11] = 5000
        out = torch.full((1024, W), 0x11, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
    extra["pool"] = pool
    return codes, scales, out, inv_scale, src, dst, extra


def _check_normal_range_exact(codes, scales, out, inv_scale) -> None:
    """With a power-of-two inv_scale, every value whose scaled magnitude is in [2^-6, 448] is reproduced
    exactly: out * (1 / inv_scale) == the MXFP8 value. Only E4M3-subnormal results round."""
    n = out.shape[0]
    flat_codes, flat_scales = codes.reshape(n, W), scales.reshape(n, NB)
    # 2^(b - 127) built from the fp32 exponent bits (exact; torch.ldexp on the GPU goes through pow and
    # can be one ulp off in fp64), times the code in fp32: exact, at most 4 significant bits.
    p2 = (flat_scales.to(torch.int32) << 23).view(torch.float32)
    v = (
        flat_codes.contiguous().view(torch.float8_e4m3fn).float().view(n, NB, BLOCK)
        * p2.unsqueeze(-1)
    ).view(n, W)
    sv = (v * inv_scale).abs()
    normal = (sv >= 2.0**-6) & (sv <= E4M3_MAX)
    back = out.float() / inv_scale
    frac = normal.float().mean().item()
    n_bad = int((back[normal] != v[normal]).sum())
    print(f"inv_scale {inv_scale}: {frac:.3f} of values in the normal range, {n_bad} not exact")
    assert frac > 0.2
    assert n_bad == 0, "a normal-range value was not reproduced exactly"
    sub = (sv > 0) & (sv < 2.0**-6)
    if sub.any():
        err = (back[sub] - v[sub]).abs().max().item()
        assert err <= 2.0**-10 / inv_scale, err  # half an E4M3 subnormal step


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    codes, scales, out, inv_scale, src, dst, extra = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    before = out.view(torch.uint8).clone().view(torch.float8_e4m3fn)
    tail = None if "tail" not in extra else extra["tail"].view(torch.uint8).clone()
    pool = extra["pool"]
    pool_before = pool.clone() if pool.numel() <= 1 << 30 else None
    result = op(codes, scales, out, inv_scale, src, dst)
    assert result is out, "the op writes out in place and returns it"
    op.compare(out, op.reference(codes, scales, before, inv_scale, src, dst))
    if pool_before is not None:
        assert torch.equal(pool, pool_before), "the source pool must not be written"
    if tail is not None:
        assert torch.equal(extra["tail"].view(torch.uint8), tail), (
            "wrote past the rows it was given"
        )
    if "last_page_start" in extra:
        assert int(src.max()) >= extra["last_page_start"]  # the pool's last page is read
    if extra.get("exactness"):
        _check_normal_range_exact(codes, scales, out, inv_scale)


# ─── special encodings, graphs, controls, rejections ──────────────────────────


def test_saturates_and_keeps_signed_zero_and_nan():
    codes = torch.tensor(
        [[0x7E, 0xFE, 0x00, 0x80, 0x7F] + [0x38] * (W - 5)], dtype=torch.uint8, device=DEV
    )
    scales = torch.full((1, NB), 127 + 4, dtype=torch.uint8, device=DEV)  # x16: 448*16 saturates
    out = torch.empty(1, W, dtype=torch.float8_e4m3fn, device=DEV)
    op(codes, scales, out, 1.0)
    o = out.view(torch.uint8)[0]
    assert o[0].item() == 0x7E and o[1].item() == 0xFE, "448 * 16 must saturate to +-448"
    assert o[2].item() == 0x00 and o[3].item() == 0x80, "signed zeros"
    assert o[4].item() in E4M3_NAN_CODES, "an e4m3 NaN code stays NaN"
    scales[0, 0] = 255
    op(codes, scales, out, 1.0)
    assert all(out.view(torch.uint8)[0, i].item() in E4M3_NAN_CODES for i in range(32)), (
        "byte 255 -> NaN block"
    )


def test_cuda_graph_capture_replay():
    pool, codes, scales = _native_pool(32, seed=8)
    T = 16
    src = torch.zeros(T * 128, dtype=torch.int32, device=DEV)
    out = torch.zeros(T * 128, W, dtype=torch.float8_e4m3fn, device=DEV)
    op(codes, scales, out, 32.0, src)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(codes, scales, out, 32.0, src)
    flat = pool.view(-1, ROW_BYTES)
    for seed in range(3):
        g = _gen(100 + seed)
        src.copy_(
            torch.randint(-1, 32 * PAGE, (T * 128,), device=DEV, generator=g, dtype=torch.int32)
        )
        graph.replay()
        exp = _ref(flat[src.clamp_min(0).long(), :W], flat[src.clamp_min(0).long(), W:], 32.0)
        exp[src < 0] = 0
        _assert_bitwise(out.view(torch.uint8), exp)
        op.compare(out, op.reference(codes, scales, torch.zeros_like(out), 32.0, src))


def test_wrong_variants_are_detected():
    """Controls: the bitwise gate sees a wrong scale bias, a scale not applied, and a wrong code format."""
    pool, codes, scales = _native_pool(4, seed=9)
    out = torch.empty(4 * PAGE, W, dtype=torch.float8_e4m3fn, device=DEV)
    op(codes, scales, out, 16.0)
    op.compare(out, op.reference(codes, scales, torch.empty_like(out), 16.0))
    flat = pool.view(-1, ROW_BYTES)
    got = out.view(torch.uint8)
    for name, exp in (
        ("bias 126", _control(flat[:, :W], flat[:, W:], 16.0, bias=126)),
        ("inv_scale ignored", _control(flat[:, :W], flat[:, W:], 1.0)),
        (
            "codes read as e5m2",
            _control(flat[:, :W], flat[:, W:], 16.0, code_dtype=torch.float8_e5m2),
        ),
    ):
        frac = (got != exp).float().mean().item()
        assert frac > 0.3, f"control '{name}' differs on only {frac:.3f} of codes"


def test_rejects_bad_inputs():
    pool, codes, scales = _native_pool(1, seed=10)
    bf = torch.empty(PAGE, W, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(codes, scales, bf, 1.0)
    out = torch.empty(PAGE, W, dtype=torch.float8_e4m3fn, device=DEV)
    with pytest.raises(ValueError, match="inv_scale"):
        op(codes, scales, out, 0.0)
