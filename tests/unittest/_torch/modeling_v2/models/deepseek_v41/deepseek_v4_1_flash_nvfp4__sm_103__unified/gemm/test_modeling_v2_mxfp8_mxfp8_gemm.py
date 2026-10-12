# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_mxfp8_gemm entry (MXFP8 x MXFP8 dense GEMM, flashinfer.mm_mxfp8).

The entry owns what is certified: `CELLS` is the list (every DeepSeek-V4.1
dense FP8 linear at decode through prefill rows on each certified backend, the
8192-row prefill chunk, every kind of scale padding, fp16 output and a caller
`out`), `reference` (the exact product of the dequantized operands in true
fp32) and `compare` (the fp32-accumulation bound) are the gate. This file
builds the operands byte-first, with checkpoint-style 32x32 weight scales
expanded to the per-row form the op consumes, and keeps the checks a cell
cannot carry: the bit-exact regime, ignored padding bytes, fused projections,
the wrong-layout controls, determinism, CUDA-graph replay, the op's own
rejections and the guard, and the target's row-count sweeps, captures,
workspace stability and row-addressing controls at the core and drafter
linears.
"""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.gemm.mxfp8_mxfp8_gemm import (  # noqa: E501
    mxfp8_mxfp8_gemm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mxfp8_mxfp8_gemm requires a CUDA device"

DEV = torch.device("cuda")
BLOCK = 32  # MXFP8: one UE8M0 scale per 32 contiguous elements along K
BACKENDS = ("cudnn", "cute-dsl", "cutlass")

# The DeepSeek-V4.1-Flash dense FP8 linears, (N, K) in nn.Linear [out, in] order
# per rank: wq_a||wkv, wq_b (attention TP4 and attention DP), wo_b (attention
# TP4 and attention DP), shared w1||w3 and w2, indexer wq_b (also wq_b under
# attention TP8), engram wkv, wq_b and wo_b under attention TP2, wo_b under
# attention TP8, DSpark main_proj. wq_b is [64 / tp * 512, 1280] and wo_b
# [5120, 8 / tp * 1024] per rank (tp 1 under attention DP).
TARGET_SHAPES = (
    (1792, 5120),
    (8192, 1280),
    (32768, 1280),
    (5120, 2048),
    (5120, 8192),
    (4608, 5120),
    (5120, 2304),
    (4096, 1280),
    (25600, 6144),
    (16384, 1280),
    (5120, 4096),
    (5120, 1024),
    (5120, 15360),
)


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _swizzle(sf_2d: torch.Tensor, pad_fill: int = 0) -> torch.Tensor:
    """[rows, cols] UE8M0 bytes -> the flat 128x4-swizzled buffer, in native torch.

    Offset of the scale of (row r, block c):
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
        + (r // 128) * 128 * pad_up(cols, 4)
    Length pad_up(rows, 128) * pad_up(cols, 4); `pad_fill` goes to every offset
    no real (r, c) addresses.
    """
    rows, cols = sf_2d.shape
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device=DEV).view(-1, 1)
    c = torch.arange(cols, device=DEV).view(1, -1)
    idx = (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    out = torch.full((_pad_up(rows, 128) * padded_cols,), pad_fill, dtype=torch.uint8, device=DEV)
    out[idx.flatten()] = sf_2d.flatten()
    return out


def _e4m3(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Random e4m3 elements over the full finite code space (both NaN codes excluded)."""
    b = torch.randint(0, 256, (rows, k), generator=g, device=DEV, dtype=torch.uint8)
    b = torch.where((b & 0x7F) == 0x7F, b ^ 0x01, b)
    return b.view(torch.float8_e4m3fn)


def _activation(m: int, k: int, seed: int, sf_lo: int = 116, sf_hi: int = 124):
    """(e4m3 [M, K], linear UE8M0 [M, K/32]) built byte-first."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = _e4m3(m, k, g)
    sf = torch.randint(sf_lo, sf_hi, (m, k // BLOCK), generator=g, device=DEV, dtype=torch.uint8)
    return q, sf


def _checkpoint_weight(n: int, k: int, seed: int):
    """(e4m3 [N, K], checkpoint 32x32 block scale [ceil(N/32), K/32], row-expanded [N, K/32]).

    The checkpoint stores one UE8M0 byte per 32x32 tile. Expanding it to the 1x32
    per-row vector this op consumes is `repeat_interleave(32, dim=0)[:N]`, which
    is exact: every row of a tile carries that tile's scale.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = _e4m3(n, k, g)
    blk = torch.randint(
        118, 126, (_pad_up(n, 32) // 32, k // BLOCK), generator=g, device=DEV, dtype=torch.uint8
    )
    return q, blk, blk.repeat_interleave(BLOCK, dim=0)[:n].contiguous()


def _values(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact fp32 value of an MXFP8 operand from linear scale bytes, for the bit-exact regime."""
    return q.float() * torch.exp2(sf.float() - 127.0).repeat_interleave(BLOCK, dim=-1)


def _ref_and_bound(aq, asf, wq, wsf, out_dtype: torch.dtype = torch.bfloat16):
    """The entry's `(ref, bound)` for operands given with linear `[rows, K/32]` scale bytes."""
    return op.reference(aq, _swizzle(asf), wq, _swizzle(wsf), out_dtype)


def _ratio(out: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> float:
    """max |out - ref| / bound: the quantity the entry's gate holds at <= 1."""
    err = (out.float() - ref).abs()
    return (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()


def _check_against(out: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> float:
    """The entry's gate on a precomputed (possibly row-sliced) reference; returns the ratio."""
    op.compare(out, (ref, bound))
    return _ratio(out, ref, bound)


def _check(out: torch.Tensor, aq, asf, wq, wsf) -> float:
    """The entry's gate against its reference for these operands; returns max |err| / bound."""
    ref, bound = _ref_and_bound(aq, asf, wq, wsf, out.dtype)
    return _check_against(out, ref, bound)


def _weight_rows(wq: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """A row subset of an e4m3 weight (selected as bytes)."""
    return wq.view(torch.uint8).index_select(0, rows).view(torch.float8_e4m3fn)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    n, k, m, backend = spec["N"], spec["K"], spec["M"], spec["backend"]
    wq, _, wsf = _checkpoint_weight(n, k, seed=n + k)
    aq, asf = _activation(m, k, zlib.crc32(cell.why.encode()) % 2**31, *spec["act_sf"])
    a_sw, w_sw = _swizzle(asf), _swizzle(wsf)
    assert a_sw.numel() == _pad_up(m, 128) * _pad_up(k // BLOCK, 4)
    buf = None
    if spec["out"]:
        buf = torch.full((m, n), float("nan"), dtype=spec["out_dtype"], device=DEV)
    out = op(aq, a_sw, wq, w_sw, spec["out_dtype"], backend, buf)
    assert out.shape == (m, n) and out.dtype == spec["out_dtype"]
    if buf is not None:
        assert out.data_ptr() == buf.data_ptr(), "a caller `out` is written in place and returned"
    op.compare(out, op.reference(aq, a_sw, wq, w_sw, spec["out_dtype"], backend, buf))
    if backend == "cutlass":
        # cute-dsl and cutlass are bitwise identical to each other at every cell.
        assert torch.equal(out, op(aq, a_sw, wq, w_sw, spec["out_dtype"], "cute-dsl")), cell.why


def test_exact_regime_is_bit_exact() -> None:
    """When the fp32 sum is exact, the bf16 output is exactly one rounding of it.

    Elements are +-{0.5, 1, 2} and scales 2^{-1, 0, 1}: every product is a
    multiple of 2^-4 below 2^4, every partial sum over K <= 15360 fits in 22
    bits, so fp32 accumulation in any order is exact. Any misplaced scale, lost
    block or wrong byte would change the bits.
    """
    codes = torch.tensor(
        [0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=torch.uint8, device=DEV
    )  # +-0.5, +-1, +-2
    for n, k, m in ((1792, 5120, 8), (5120, 15360, 33), (5120, 2304, 200), (5120, 576, 5)):
        g = torch.Generator(device="cuda").manual_seed(n * 7 + k + m)
        aq = codes[torch.randint(0, 6, (m, k), generator=g, device=DEV)].view(torch.float8_e4m3fn)
        wq = codes[torch.randint(0, 6, (n, k), generator=g, device=DEV)].view(torch.float8_e4m3fn)
        asf = torch.randint(126, 129, (m, k // BLOCK), generator=g, device=DEV, dtype=torch.uint8)
        wsf = torch.randint(
            126, 129, (n // BLOCK, k // BLOCK), generator=g, device=DEV, dtype=torch.uint8
        )
        wsf = wsf.repeat_interleave(BLOCK, dim=0).contiguous()
        exact = (_values(aq, asf).double() @ _values(wq, wsf).double().T).to(torch.bfloat16)
        for backend in BACKENDS:
            out = op(aq, _swizzle(asf), wq, _swizzle(wsf), backend=backend)
            assert torch.equal(out, exact), (backend, n, k, m)


def test_scale_padding_bytes_are_ignored() -> None:
    """Whatever fills the padded rows/columns of either scale buffer never reaches the result."""
    k, n, m = 576, 160, 7  # every kind of padding at once
    wq, _, wsf = _checkpoint_weight(n, k, seed=8)
    aq, asf = _activation(m, k, seed=8)
    for backend in BACKENDS:
        zero = op(aq, _swizzle(asf, 0x00), wq, _swizzle(wsf, 0x00), backend=backend)
        # 0xFE is the largest finite E8M0 (2^127); 0xFF would be NaN.
        loud = op(aq, _swizzle(asf, 0xFE), wq, _swizzle(wsf, 0xFE), backend=backend)
        assert torch.equal(zero, loud), backend
        _check(zero, aq, asf, wq, wsf)


def test_fused_projection_equals_separate_ones() -> None:
    """Row-concatenating two weights (wq_a || wkv) and splitting the output is bitwise the two separate GEMMs."""
    k, m = 5120, 33
    wq_a, _, sf_a = _checkpoint_weight(1280, k, seed=10)
    wkv, _, sf_kv = _checkpoint_weight(512, k, seed=11)
    aq, asf = _activation(m, k, seed=12)
    a_sw = _swizzle(asf)
    for backend in BACKENDS:
        fused = op(
            aq, a_sw, torch.cat([wq_a, wkv]), _swizzle(torch.cat([sf_a, sf_kv])), backend=backend
        )
        assert torch.equal(fused[:, :1280], op(aq, a_sw, wq_a, _swizzle(sf_a), backend=backend)), (
            backend
        )
        assert torch.equal(fused[:, 1280:], op(aq, a_sw, wkv, _swizzle(sf_kv), backend=backend)), (
            backend
        )


def test_wrong_operand_layouts_fail() -> None:
    """Deliberately wrong controls: each is accepted by the op and must land far outside the gate.

    On this model family a wrong layout produces plausible numbers rather than
    an error, so the test proves the gate can see each one:
      - weight scale fed row-major (unswizzled) at the right buffer size;
      - activation scale fed row-major (unswizzled) at the right buffer size;
      - the checkpoint's 32x32 tile scale expanded by tiling instead of
        repeat_interleave (row r gets tile r mod N/32 instead of r // 32);
      - every scale byte off by one (a wrong UE8M0 bias: the result doubles).
    """
    n, k, m = 1792, 5120, 64
    wq, blk, wsf = _checkpoint_weight(n, k, seed=13)
    aq, asf = _activation(m, k, seed=13)
    ref, bound = _ref_and_bound(aq, asf, wq, wsf)
    a_sw, w_sw = _swizzle(asf), _swizzle(wsf)

    def linear_in_swizzled_size(sf: torch.Tensor, size: int) -> torch.Tensor:
        buf = torch.zeros(size, dtype=torch.uint8, device=DEV)
        buf[: sf.numel()] = sf.flatten()
        return buf

    wrong = {
        "unswizzled weight scale": (a_sw, linear_in_swizzled_size(wsf, w_sw.numel())),
        "unswizzled activation scale": (linear_in_swizzled_size(asf, a_sw.numel()), w_sw),
        "tiled 32x32 expansion": (a_sw, _swizzle(blk.repeat(BLOCK, 1)[:n].contiguous())),
        "scale bytes off by one": (a_sw, _swizzle(wsf + 1)),
    }
    for backend in BACKENDS:
        assert _check_against(op(aq, a_sw, wq, w_sw, backend=backend), ref, bound) <= 1.0
        for name, (sa, sw) in wrong.items():
            out = op(aq, sa, wq, sw, backend=backend)
            assert torch.isfinite(out).all(), (backend, name)  # plausible numbers, not an error
            with pytest.raises(AssertionError):
                _check_against(out, ref, bound)


def test_deterministic_and_inputs_untouched() -> None:
    """Repeating a call gives the same bits; no input is written."""
    n, k, m = 4608, 5120, 16
    wq, _, wsf = _checkpoint_weight(n, k, seed=14)
    aq, asf = _activation(m, k, seed=14)
    a_sw, w_sw = _swizzle(asf), _swizzle(wsf)
    snap = [t.clone() for t in (aq.view(torch.uint8), a_sw, wq.view(torch.uint8), w_sw)]
    for backend in BACKENDS:
        first = op(aq, a_sw, wq, w_sw, backend=backend)
        for _ in range(3):
            assert torch.equal(first, op(aq, a_sw, wq, w_sw, backend=backend)), backend
    for before, now in zip(snap, (aq.view(torch.uint8), a_sw, wq.view(torch.uint8), w_sw)):
        assert torch.equal(before, now)


def test_cuda_graph_replay_matches_eager() -> None:
    """After one eager call (JIT / cuDNN graph build), the call captures and replays bit-identically."""
    n, k = 8192, 1280
    wq, _, wsf = _checkpoint_weight(n, k, seed=15)
    w_sw = _swizzle(wsf)
    for m in (8, 256):
        aq, asf = _activation(m, k, seed=15 + m)
        a_sw = _swizzle(asf)
        for backend in BACKENDS:
            eager = op(aq, a_sw, wq, w_sw, backend=backend)
            out = torch.empty_like(eager)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                op(aq, a_sw, wq, w_sw, backend=backend, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op(aq, a_sw, wq, w_sw, backend=backend, out=out)
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eager), (backend, m)


def test_rejected_domains() -> None:
    """Violations the op itself rejects loudly (each observed to raise on this machine)."""
    k, n, m = 1024, 512, 8
    wq, _, wsf = _checkpoint_weight(n, k, seed=16)
    aq, asf = _activation(m, k, seed=16)
    a_sw, w_sw = _swizzle(asf), _swizzle(wsf)
    for backend in BACKENDS:
        with pytest.raises(ValueError):  # no fp32 output on any backend
            op(aq, a_sw, wq, w_sw, out_dtype=torch.float32, backend=backend)
        with pytest.raises(ValueError):  # activation scale not in the padded swizzled size
            op(aq, asf.flatten(), wq, w_sw, backend=backend)
        with pytest.raises(ValueError):  # weight scale too short
            op(aq, a_sw, wq, w_sw[:-4], backend=backend)
        w96, _, sf96 = _checkpoint_weight(96, k, seed=17)
        with pytest.raises(ValueError):  # N < 128
            op(aq, a_sw, w96, _swizzle(sf96), backend=backend)
        with pytest.raises(ValueError):  # activation not e4m3
            op(_values(aq, asf).bfloat16(), a_sw, wq, w_sw, backend=backend)


def test_guard_rejects_the_silent_domains() -> None:
    """The violations observed to be accepted and computed wrongly are rejected by `is_valid`."""
    k, n, m = 1024, 512, 8
    wq, _, wsf = _checkpoint_weight(n, k, seed=18)
    aq, asf = _activation(m, k, seed=18)
    a_sw, w_sw = _swizzle(asf), _swizzle(wsf)
    wide_a = torch.cat([aq.view(torch.uint8)] * 2, 1).view(torch.float8_e4m3fn)
    wide_w = torch.cat([wq.view(torch.uint8)] * 2, 1).view(torch.float8_e4m3fn)
    wide_out = torch.empty(m, 2 * n, dtype=torch.bfloat16, device=DEV)
    for backend in BACKENDS:
        with pytest.raises(AssertionError, match="must be contiguous"):
            with validating(op):
                op(wide_a[:, :k], a_sw, wq, w_sw, backend=backend)
        with pytest.raises(AssertionError, match="must be contiguous"):
            with validating(op):
                op(aq, a_sw, wide_w[:, :k], w_sw, backend=backend)
        with pytest.raises(AssertionError, match="out must be contiguous"):
            with validating(op):
                op(aq, a_sw, wq, w_sw, backend=backend, out=wide_out[:, :n])
    for backend in ("trtllm", "auto", "b12x"):
        with pytest.raises(AssertionError, match="backend must be one of"):
            with validating(op):
                op(aq, a_sw, wq, w_sw, backend=backend)


def test_zero_rows_raise() -> None:
    """M == 0 raises on every backend (last: cute-dsl's rejection is a failed kernel launch)."""
    k, n = 1024, 512
    wq, _, wsf = _checkpoint_weight(n, k, seed=19)
    w_sw = _swizzle(wsf)
    aq = torch.empty(0, k, dtype=torch.float8_e4m3fn, device=DEV)
    a_sw = torch.empty(0, dtype=torch.uint8, device=DEV)
    for backend in ("cutlass", "cudnn", "cute-dsl"):
        with pytest.raises(
            Exception
        ):  # RuntimeError (cutlass, cudnn); a CuTe DSL launch error (cute-dsl)
            op(aq, a_sw, wq, w_sw, backend=backend)
        torch.cuda.synchronize()


# ─── DeepSeek-V4.1 core target: every runtime row count ────────────────────
#
# The core target's twelve dense FP8 linears (TARGET_SHAPES without the DSpark
# main_proj), over its members: attention TP 2 (pp2 stages), 4 or 8, or
# attention DP. The target picks the backend by row count: cute-dsl at M <= 32,
# cudnn above. M is the rank's token count this step: any of 1..max_num_tokens,
# at most 16384 per rank. In layers 21-39 the decoder bounded replay (on every
# member, attention TP or DP) runs decoder passes over the finishing requests'
# 128-row tails plus the step's generation rows; the target cuts them into
# passes of at most the step capacity, and 32768 (128 x 256) is the uncut row
# set. wq_a||wkv, wq_b and wo_b at every member, the shared w1||w3 and w2, and
# the indexer wq_b (index layers 24, 28, 32, 36) run there, so their N_MAX is
# 32768. The Engram wkv (layers 1 and 14) stays at 16384.
CORE_SHAPES = tuple(s for s in TARGET_SHAPES if s != (5120, 15360))
N_MAX = {
    (1792, 5120): 32768,
    (8192, 1280): 32768,
    (32768, 1280): 32768,
    (5120, 2048): 32768,
    (5120, 8192): 32768,
    (4608, 5120): 32768,
    (5120, 2304): 32768,
    (4096, 1280): 32768,
    (25600, 6144): 16384,
    (16384, 1280): 32768,
    (5120, 4096): 32768,
    (5120, 1024): 32768,
}
DECODE_ROWS = tuple(range(1, 33))
# FlashInfer's cudnn backend builds one cuDNN graph per (N, K, M bucket), at the
# bucket's top, and runs it at the real M. The buckets are powers of two to
# 256, steps of 256 to 2048, steps of 512 to 4096, then powers of two. This list
# covers every bucket from 33 to 32768. For each one it has the bucket's first
# row count (farthest below the size the plan was built for), one inside, and
# the top.
BUCKET_ROWS = (
    33, 48, 61, 64, 65, 97, 128, 129, 211, 256, 257, 323, 509, 512, 513, 576, 767, 768,
    769, 862, 1000, 1024, 1025, 1153, 1280, 1281, 1423, 1536, 1537, 1709, 1792, 1793,
    1935, 2048, 2049, 2311, 2560, 2561, 2917, 3072, 3073, 3305, 3584, 3585, 3907, 4096,
    4097, 4617, 6523, 8179, 8191, 8192, 8193, 12289, 16383, 16384, 16385, 24577, 32767,
    32768,
)  # fmt: skip
# Row counts above 32 of the core target's linears in its runtime-value dumps:
# the gate runs, and the engine legs at max_num_tokens 16384 and
# max_batch_size 256 (tep4, tp4, dep4; padded graphs to 256; L to 1M).
DUMP_ROWS = (
    34, 48, 60, 96, 108, 127, 128, 129, 134, 144, 168, 192, 228, 240, 288, 330, 336, 384,
    432, 462, 480, 528, 546, 576, 597, 610, 624, 672, 720, 768, 780, 852, 1152, 1536, 1906,
    2033, 2038, 2048, 2049, 2778, 2784, 3870, 4208, 4245, 4388, 4394, 4591, 4868, 7835,
    7855, 7893, 7908, 8191, 8192, 8467, 9193, 9240, 9364, 9370, 9524, 10441, 10444, 10447,
    10451, 16298, 16308, 16316, 16320, 16322, 16327, 16328, 16329, 16333, 16340, 16378,
    16384,
)  # fmt: skip
PREFILL_ROWS = tuple(sorted(set(BUCKET_ROWS) | set(DUMP_ROWS)))
# The captured batch sizes G, per rank: every G 1..32, plus TRT-LLM's padded
# default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list for 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only; each linear sees a multiple
# of G (no MXFP8 linear runs on dep4's gathered rows).
GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
# (N, K) and the multiples of G it is captured at: G (decode) and 6G (DSpark
# target verify, 1 + 5 drafts) for the trunk linears; 5G (the drafter's 5-token
# draft block) where the drafter has a linear of that shape; G and 5G for the
# drafter-only main_proj and fused wkv x3 (its bonus rows and its ring
# back-fill). Up to 1536 rows.
CAPTURE_CELLS = (
    ((1792, 5120), (1, 5, 6)),  # wq_a || wkv: trunk, drafter
    ((8192, 1280), (1, 6)),  # wq_b, attention TP4 (16 heads per rank)
    ((32768, 1280), (1, 5, 6)),  # wq_b, attention DP (all 64 heads) and the drafter
    ((5120, 2048), (1, 6)),  # wo_b, attention TP4
    ((5120, 8192), (1, 5, 6)),  # wo_b, attention DP and the drafter
    ((4608, 5120), (1, 5, 6)),  # shared w1 || w3: trunk, drafter
    ((5120, 2304), (1, 5, 6)),  # shared w2: trunk, drafter
    ((4096, 1280), (1, 6)),  # indexer wq_b; wq_b, attention TP8 (8 heads per rank)
    ((25600, 6144), (1, 6)),  # Engram wkv
    ((16384, 1280), (1, 6)),  # wq_b, attention TP2 (32 heads per rank)
    ((5120, 4096), (1, 6)),  # wo_b, attention TP2
    ((5120, 1024), (1, 6)),  # wo_b, attention TP8
    ((5120, 15360), (1, 5)),  # drafter main_proj
    ((1536, 5120), (1, 5)),  # drafter wkv x3, fused on the row axis
)


def _capture_rows(factors) -> tuple:
    return tuple(sorted({f * g for g in GSET for f in factors}))


def _target_backend(m: int) -> str:
    """The target's policy: cute-dsl at M <= 32, cudnn above."""
    return "cute-dsl" if m <= 32 else "cudnn"


@pytest.mark.parametrize("n,k", CORE_SHAPES)
def test_core_decode_every_row_count(n: int, k: int) -> None:
    """M = 1..32 at each core linear, on cute-dsl (the target's backend there) and cudnn.

    Every M is held to the fp32-accumulation bound. cute-dsl is also held to a
    measured row invariance: at every M each row's bits equal that row's bits
    at M = 32. That holds across its three low-M kernels (N tile 8 / 16 / 32).
    """
    wq, _, wsf = _checkpoint_weight(n, k, seed=n + 3 * k)
    w_sw = _swizzle(wsf)
    aq, asf = _activation(32, k, seed=k + 32)
    ref, bound = _ref_and_bound(aq, asf, wq, wsf)
    top = op(aq, _swizzle(asf), wq, w_sw, backend="cute-dsl")
    for m in DECODE_ROWS:
        a_sw = _swizzle(asf[:m])
        for backend in ("cute-dsl", "cudnn"):
            out = op(aq[:m], a_sw, wq, w_sw, backend=backend)
            assert out.shape == (m, n) and out.dtype == torch.bfloat16
            _check_against(out, ref[:m], bound[:m])
            if backend == "cute-dsl":
                assert torch.equal(out, top[:m]), (n, k, m)


@pytest.mark.parametrize("n,k", CORE_SHAPES)
def test_core_prefill_every_cudnn_bucket(n: int, k: int) -> None:
    """Every row count in PREFILL_ROWS up to the shape's N_MAX at each core linear, on all three backends.

    cudnn is the target's backend above 32 rows. It is checked against the
    bound only, because its last bits may depend on the row count. cute-dsl
    and cutlass must match each other bitwise, and each row's bits must equal
    that row's bits at M = N_MAX (a measured row invariance).
    """
    n_max = N_MAX[(n, k)]
    wq, _, wsf = _checkpoint_weight(n, k, seed=2 * n + k)
    w_sw = _swizzle(wsf)
    aq, asf = _activation(n_max, k, seed=5 * k + 1)
    ref, bound = _ref_and_bound(aq, asf, wq, wsf)
    full = op(aq, _swizzle(asf), wq, w_sw, backend="cutlass")
    _check_against(full, ref, bound)
    worst = 0.0
    for m in (m for m in PREFILL_ROWS if m <= n_max):
        a_sw = _swizzle(asf[:m])
        outs = {}
        for backend in BACKENDS:
            out = op(aq[:m], a_sw, wq, w_sw, backend=backend)
            assert out.shape == (m, n)
            ratio = _check_against(out, ref[:m], bound[:m])
            if backend == "cudnn":
                worst = max(worst, ratio)
            outs[backend] = out
        assert torch.equal(outs["cute-dsl"], outs["cutlass"]), (n, k, m)
        assert torch.equal(outs["cutlass"], full[:m]), (n, k, m)
        del outs
    print(f"[full width M 33..{n_max}] ({n}, {k}): worst cudnn |err| / bound = {worst:.4f}")
    del ref, bound, full
    torch.cuda.empty_cache()


def _cudnn_kernel_map(aq, asf, wq, w_sw, rows) -> dict:
    """Profile one cudnn call per row count; return {M: the kernel names that call launched, in order}.

    Each call is isolated between two synchronizations inside its own
    `record_function` range, so the device kernels whose start falls in that
    range are exactly the ones the call launched (the range's own device-side
    annotation and any PyTorch kernel are excluded by name).
    """
    names = {}
    for i in range(0, len(rows), 256):
        chunk = rows[i : i + 256]
        ops = [(m, _swizzle(asf[:m])) for m in chunk]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for m, a_sw in ops:
                with record_function(f"M={m}"):
                    op(aq[:m], a_sw, wq, w_sw, backend="cudnn")
                    torch.cuda.synchronize()
        evs = prof.events()
        ranges = [
            (int(e.name[2:]), e.time_range.start, e.time_range.end)
            for e in evs
            if e.name.startswith("M=") and e.device_type == torch.autograd.DeviceType.CPU
        ]
        kernels = sorted(
            (e.time_range.start, e.name)
            for e in evs
            if e.device_type == torch.autograd.DeviceType.CUDA
            and not e.name.startswith("M=")
            and "at::native" not in e.name
        )
        for m, lo, hi in ranges:
            names[m] = tuple(nm for t, nm in kernels if lo <= t <= hi)
        del ops, prof
        torch.cuda.empty_cache()
    return names


@pytest.mark.parametrize("n,k", CORE_SHAPES)
def test_core_cudnn_every_row_count_33_to_n_max(n: int, k: int) -> None:
    """Every M from 33 to the shape's N_MAX (16384 or 32768) on cudnn, the target's backend there, at each core linear.

    Every row of every call is held to the bound on a column sample: one
    column in each 64-column block, at a position that changes from block to
    block, plus the last column. A row that is wrong, missing or misplaced
    fails on every sampled column, as the row-addressing controls show. The
    sample can miss an error confined to a few columns of one tile. Each output
    is poisoned before it is freed. The next call's output often reuses the
    block, and its rows sit at the same offsets, so a row the kernel skipped
    would otherwise keep the previous call's correct bits.

    cuDNN picks its kernel by M (nvjet tile / cluster / split-K variants that
    change inside FlashInfer's M buckets too), so the test then profiles every
    call M 33..N_MAX, derives the kernel map (maximal runs of consecutive M
    that launch the same kernel sequence), and checks the first, a middle and
    the last M of every run at FULL width against the reference (every kernel
    switch point at its edges). The full-width check at
    the bucket edges and the dumped row counts on all three backends is
    `test_core_prefill_every_cudnn_bucket`.
    """
    n_max = N_MAX[(n, k)]
    wq, _, wsf = _checkpoint_weight(n, k, seed=4 * n + k)
    w_sw = _swizzle(wsf)
    aq, asf = _activation(n_max, k, seed=7 * k + 3)
    blocks = torch.arange(n // 64, device=DEV)
    cols = torch.cat([blocks * 64 + (blocks * 37) % 64, torch.tensor([n - 1], device=DEV)])
    ref, bound = _ref_and_bound(aq, asf, _weight_rows(wq, cols), wsf.index_select(0, cols))
    bound = bound.clamp_min(torch.finfo(torch.float32).tiny)
    worst = torch.zeros((), device=DEV)
    for m in range(33, n_max + 1):
        out = op(aq[:m], _swizzle(asf[:m]), wq, w_sw, backend="cudnn")
        err = (out.index_select(1, cols).float() - ref[:m]).abs()
        worst = torch.maximum(worst, (err / bound[:m]).max())
        out.fill_(float("nan"))
        if m % 2048 == 0:
            # Each call's output is larger than any cached block, so the allocator
            # would otherwise keep a segment per size and fill the device.
            del out, err
            torch.cuda.empty_cache()
    ratio = worst.item()
    print(
        f"[cudnn M 33..{n_max}] ({n}, {k}): max |err| / bound over {cols.numel()} sampled columns = {ratio:.4f}"
    )
    assert ratio <= 1.0, ratio
    del ref, bound

    rows = list(range(33, n_max + 1))
    names = _cudnn_kernel_map(aq, asf, wq, w_sw, rows)
    runs = []
    for m in rows:
        assert names.get(m), f"no kernel attributed to the cudnn call at M={m}"
        if runs and runs[-1][2] == names[m]:
            runs[-1][1] = m
        else:
            runs.append([m, m, names[m]])
    edges = sorted({m for lo, hi, _ in runs for m in (lo, (lo + hi) // 2, hi)})
    ref, bound = _ref_and_bound(aq, asf, wq, wsf)
    edge_worst = 0.0
    for m in edges:
        out = op(aq[:m], _swizzle(asf[:m]), wq, w_sw, backend="cudnn")
        edge_worst = max(edge_worst, _check_against(out, ref[:m], bound[:m]))
        del out
    kinds = sorted({nm for _, _, nm in runs})
    print(
        f"[cudnn kernel map ({n}, {k})] {len(runs)} runs of {len(kinds)} distinct kernel sequences over M 33..{n_max}; "
        f"{len(edges)} run edges / middles at full width, worst |err| / bound {edge_worst:.4f}; "
        f"switch points (first M of each run): {[lo for lo, _, _ in runs]}"
    )
    for kind in kinds:
        print(f"    {' + '.join(nm[:120] for nm in kind)}")
    del ref, bound
    torch.cuda.empty_cache()


def _graph_sweep(n: int, k: int, rows, seed: int) -> dict:
    """Capture one call per row count on the target's backend and replay it on new activations.

    It follows the engine's order: one eager call at the same M (its warmup),
    then capture. The call is captured without `out`, as the target makes it,
    so the output lives in the graph's pool. That output is poisoned before
    every replay, then checked against the bound and bitwise against an eager
    call on the same data. A stale replay is distinguishable: the warm-up's
    output on the capture-time data (what a replay that did not rerun, or reran
    on the old inputs, would hold) must differ from each replay's output on
    most elements. Returns per-backend capture counts, the worst
    |err| / bound and the smallest differing fraction.
    """
    wq, _, wsf = _checkpoint_weight(n, k, seed=seed)
    w_sw = _swizzle(wsf)
    stats = {"cute-dsl": 0, "cudnn": 0, "worst": 0.0, "stale_min": 1.0}
    for m in rows:
        be = _target_backend(m)
        aq, asf = _activation(m, k, seed=11 * m + k)
        a_sw = _swizzle(asf)
        stale = op(aq, a_sw, wq, w_sw, torch.bfloat16, be)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(aq, a_sw, wq, w_sw, torch.bfloat16, be)
        for seed_ in (1, 2):
            nq, nsf = _activation(m, k, seed=100000 * seed_ + 11 * m + k)
            aq.view(torch.uint8).copy_(nq.view(torch.uint8))
            a_sw.copy_(_swizzle(nsf))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            stats["worst"] = max(stats["worst"], _check(out, nq, nsf, wq, wsf))
            assert torch.equal(out, op(nq, _swizzle(nsf), wq, w_sw, torch.bfloat16, be)), (
                n,
                k,
                m,
                be,
            )
            frac = (out != stale).float().mean().item()
            stats["stale_min"] = min(stats["stale_min"], frac)
            assert frac > 0.9, (n, k, m, be, frac)
        stats[be] += 1
        del graph, out, stale
    return stats


@pytest.mark.parametrize("n,k,factors", [(n, k, f) for (n, k), f in CAPTURE_CELLS])
def test_graph_capture_at_every_captured_batch(n: int, k: int, factors) -> None:
    """Each linear captured at every multiple of G it sees over GSET, on the target's backend (cute-dsl
    at M <= 32, cudnn above), and replayed twice on new activations."""
    rows = _capture_rows(factors)
    stats = _graph_sweep(n, k, rows, seed=n + k + 7)
    print(
        f"[capture ({n}, {k}) x G{factors}] {len(rows)} row counts 1..{rows[-1]}: {stats['cute-dsl']} on cute-dsl, "
        f"{stats['cudnn']} on cudnn, {2 * len(rows)} replays bitwise = eager, worst |err| / bound "
        f"{stats['worst']:.4f}, "
        f"stale output differs on >= {stats['stale_min']:.4f} of elements"
    )


def _mm_mxfp8_workspace():
    """(bytes, address) of FlashInfer's cached `mm_mxfp8_workspace` buffer on this device, or None."""
    from flashinfer import utils as fi_utils

    for (name, _), buf in fi_utils._cache_buf.items():
        if name == "mm_mxfp8_workspace":
            return buf.numel(), buf.data_ptr()
    return None


def test_cudnn_graphs_survive_eager_calls_to_n_max() -> None:
    """Graphs captured at a decode row count stay valid after eager cudnn calls at every bucket edge up to N_MAX.

    FlashInfer keeps one cached workspace buffer per device and grows it with
    `resize_` (a new allocation at a new address) whenever a cudnn call asks
    for more than it holds, so a graph captured before such a growth would
    keep the freed address. The engine captures its decode graphs after its
    warm-up and then runs eager prefill and decoder-pass steps of up to N_MAX
    rows. Here one cudnn graph per core linear is captured at M = 96 after an
    eager call; then every core linear runs eagerly at every BUCKET_ROWS value
    up to its N_MAX (the first, an interior and the last M of every FlashInfer
    bucket); the buffer's size and address must not change, and every graph
    must then replay on new data bitwise equal to eager and inside the bound.
    """
    graphs = []
    for n, k in CORE_SHAPES:
        wq, _, wsf = _checkpoint_weight(n, k, seed=41 + n + k)
        w_sw = _swizzle(wsf)
        aq, asf = _activation(96, k, seed=42 + k)
        a_sw = _swizzle(asf)
        op(aq, a_sw, wq, w_sw, torch.bfloat16, "cudnn")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(aq, a_sw, wq, w_sw, torch.bfloat16, "cudnn")
        graphs.append((n, k, wq, wsf, w_sw, aq, a_sw, graph, out))
    before = _mm_mxfp8_workspace()
    assert before is not None
    calls = 0
    for n, k, wq, _, w_sw, *_ in graphs:
        n_max = N_MAX[(n, k)]
        big_q, big_sf = _activation(n_max, k, seed=43 + k)
        for m in (m for m in BUCKET_ROWS if m <= n_max):
            op(big_q[:m], _swizzle(big_sf[:m]), wq, w_sw, torch.bfloat16, "cudnn")
            calls += 1
            assert _mm_mxfp8_workspace() == before, (n, k, m, before, _mm_mxfp8_workspace())
        del big_q, big_sf
    torch.cuda.synchronize()
    for n, k, wq, wsf, w_sw, aq, a_sw, graph, out in graphs:
        nq, nsf = _activation(96, k, seed=44 + k)
        aq.view(torch.uint8).copy_(nq.view(torch.uint8))
        a_sw.copy_(_swizzle(nsf))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(out, nq, nsf, wq, wsf)
        assert torch.equal(out, op(nq, _swizzle(nsf), wq, w_sw, torch.bfloat16, "cudnn")), (n, k)
    print(
        f"[workspace] {calls} eager cudnn calls at the bucket edges up to N_MAX after {len(graphs)} captures: "
        f"buffer stayed at {before[0]} bytes, same address; every graph replayed bitwise = eager"
    )


def test_row_addressing_controls_at_runtime_row_counts() -> None:
    """The bound catches a row-misaddressed activation at a decode and a prefill row count.

    Shared w1||w3 (4608 x 5120), at M = 7 on cute-dsl and M = 1935 on cudnn
    (the target's backends there). The op accepts each wrong operand below,
    and each result must land outside the bound:
      - activation scale rows rolled by one (row m carries row m-1's scales);
      - activation data rows rolled by one against their own scales;
      - activation scale fed row-major at the swizzled size.
    """
    n, k = 4608, 5120
    wq, _, wsf = _checkpoint_weight(n, k, seed=21)
    w_sw = _swizzle(wsf)
    for m, backend in ((7, "cute-dsl"), (1935, "cudnn")):
        aq, asf = _activation(m, k, seed=22 + m)
        a_sw = _swizzle(asf)
        assert _check(op(aq, a_sw, wq, w_sw, backend=backend), aq, asf, wq, wsf) <= 1.0
        linear = torch.zeros_like(a_sw)
        linear[: asf.numel()] = asf.flatten()
        rolled_data = torch.roll(aq.view(torch.uint8), 1, 0).view(torch.float8_e4m3fn).contiguous()
        wrong = {
            "scale rows rolled by one": (aq, _swizzle(torch.roll(asf, 1, 0).contiguous())),
            "data rows rolled by one": (rolled_data, a_sw),
            "row-major activation scale": (aq, linear),
        }
        ref, bound = _ref_and_bound(aq, asf, wq, wsf)
        for name, (qa, sa) in wrong.items():
            out = op(qa, sa, wq, w_sw, backend=backend)
            assert torch.isfinite(out).all(), (backend, name)
            ratio = (
                ((out.float() - ref).abs() / bound.clamp_min(torch.finfo(torch.float32).tiny))
                .max()
                .item()
            )
            print(f"[control M={m} {backend}] {name}: max |err| / bound = {ratio:.3g}")
            assert ratio > 1.0, (backend, name)


# ─── DeepSeek-V4.1 DSpark drafter ──────────────────────────────────────────
#
# The drafter's dense MXFP8 linears (replicated on every rank) run inside the
# target's decode CUDA graphs, and eagerly on steps that are not captured. With
# G the drafter's batch (1..256) the row count is G (forward_batched's bonus
# rows through main_proj and the fused wkv) or 5G (the draft block through
# every linear, and the batched ring back-fill through main_proj and wkv), so
# 1..1280, on the target's backend policy.
#   (5120, 15360) main_proj (stage 0), (1536, 5120) the three stages' wkv
#   fused on the row axis (stage s is output columns [512 s, 512 (s + 1)))
# run eagerly at every M 1..1280 and captured at every M 1..160 (and at G and
# 5G over GSET, CAPTURE_CELLS). The other drafter linears share a core shape
# and run at M = 5G only: wq_a || wkv (1792, 5120), wq_b (32768, 1280), wo_b
# (5120, 8192), shared w1 || w3 (4608, 5120), shared w2 (5120, 2304); their
# eager rows are inside the core shapes' 1..N_MAX and their 5G captures are in
# CAPTURE_CELLS.
DSPARK_NEW_SHAPES = ((5120, 15360), (1536, 5120))
DSPARK_ROWS = tuple(range(1, 1281))
DSPARK_GRAPH_ROWS = tuple(range(1, 161))


@pytest.mark.parametrize("n,k", DSPARK_NEW_SHAPES)
def test_dspark_every_row_count_1_to_1280(n: int, k: int) -> None:
    """main_proj and the fused wkv x3 at every M 1..1280 on all three backends, within the bound.

    cute-dsl and cutlass are bitwise equal to each other at every M (the same
    measured property as at the core shapes); cudnn is held to the bound only.
    For the fused wkv x3, at every M, each stage's 512 output columns are
    compared with that stage's own [512, 5120] GEMM on the same backend. On
    cute-dsl and cutlass they are bitwise equal at every M. On cudnn they are
    bitwise equal except at M 193..640, where the fused [1536, 5120] and the
    separate [512, 5120] GEMMs launch different kernels (the fused one leaves
    split-K at 193) and differ in the last bits of a small fraction of the
    elements; there each separate GEMM is held to the bound, and the set of M
    where they differ must be exactly 193..640 (the measured window).
    """
    wq, _, wsf = _checkpoint_weight(n, k, seed=5 * n + k)
    w_sw = _swizzle(wsf)
    aq, asf = _activation(DSPARK_ROWS[-1], k, seed=3 * k + 5)
    ref, bound = _ref_and_bound(aq, asf, wq, wsf)
    stages = []
    if n == 1536:
        stages = [
            (
                wq[512 * s : 512 * (s + 1)].contiguous(),
                _swizzle(wsf[512 * s : 512 * (s + 1)].contiguous()),
            )
            for s in range(3)
        ]
    worst = 0.0
    cudnn_differs = []
    frac_max = 0.0
    for m in DSPARK_ROWS:
        a_sw = _swizzle(asf[:m])
        outs = {}
        for backend in BACKENDS:
            out = op(aq[:m], a_sw, wq, w_sw, backend=backend)
            assert out.shape == (m, n) and out.dtype == torch.bfloat16
            worst = max(worst, _check_against(out, ref[:m], bound[:m]))
            outs[backend] = out
        assert torch.equal(outs["cute-dsl"], outs["cutlass"]), (n, k, m)
        for backend in BACKENDS if stages else ():
            differs = False
            for s, (ws_, wsf_) in enumerate(stages):
                one = op(aq[:m], a_sw, ws_, wsf_, backend=backend)
                cols = slice(512 * s, 512 * (s + 1))
                if backend != "cudnn":
                    assert torch.equal(outs[backend][:, cols], one), (backend, m, s)
                elif not torch.equal(outs[backend][:, cols], one):
                    differs = True
                    frac_max = max(frac_max, (outs[backend][:, cols] != one).float().mean().item())
                    _check_against(one, ref[:m, cols], bound[:m, cols])
            if differs:
                cudnn_differs.append(m)
    print(f"[dspark ({n}, {k}) M 1..{DSPARK_ROWS[-1]}] max |err| / bound = {worst:.4f}")
    if stages:
        print(
            f"[dspark wkv x3 stages] cute-dsl, cutlass: fused = separate at every M; cudnn: fused != separate at "
            f"{len(cudnn_differs)} M ({cudnn_differs[:1]}..{cudnn_differs[-1:]}), at most {frac_max:.4f} of a "
            "stage's elements"
        )
        assert cudnn_differs == list(range(193, 641)), cudnn_differs


@pytest.mark.parametrize("n,k", DSPARK_NEW_SHAPES)
def test_dspark_graph_capture_every_row_count(n: int, k: int) -> None:
    """main_proj and the fused wkv x3 captured at every M 1..160 on the target's backend."""
    _graph_sweep(n, k, DSPARK_GRAPH_ROWS, seed=n + k + 9)


def test_dspark_row_addressing_controls() -> None:
    """At main_proj, a row-misaddressed activation lands outside the bound on both target backends
    (M = 5 on cute-dsl, M = 160 on cudnn): scale rows rolled by one, a row-major scale buffer."""
    n, k = 5120, 15360
    wq, _, wsf = _checkpoint_weight(n, k, seed=31)
    w_sw = _swizzle(wsf)
    for m in (5, 160):
        be = _target_backend(m)
        aq, asf = _activation(m, k, seed=32 + m)
        a_sw = _swizzle(asf)
        ref, bound = _ref_and_bound(aq, asf, wq, wsf)
        assert _check_against(op(aq, a_sw, wq, w_sw, backend=be), ref, bound) <= 1.0
        linear = torch.zeros_like(a_sw)
        linear[: asf.numel()] = asf.flatten()
        wrong = {
            "scale rows rolled by one": _swizzle(torch.roll(asf, 1, 0).contiguous()),
            "row-major activation scale": linear,
        }
        for name, sa in wrong.items():
            out = op(aq, sa, wq, w_sw, backend=be)
            ratio = (
                ((out.float() - ref).abs() / bound.clamp_min(torch.finfo(torch.float32).tiny))
                .max()
                .item()
            )
            print(f"[dspark control M={m} {be}] {name}: max |err| / bound = {ratio:.3g}")
            assert ratio > 1.0, (be, name)
