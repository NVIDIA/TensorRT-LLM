# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_kv_to_e4m3 catalog entry.

The pool is DeepSeek-V4.1's native compressed-KV layout: one 288-byte row per entry, 256 packed E2M1
bytes (even element in the low nibble) then 32 E4M3 scale bytes (one per 16 elements), no global scale,
read as two strided views. Every gate is bitwise on the uint8 codes, and `out` is pre-filled with a
sentinel byte so that `compare` also proves every entry the op must skip was left alone.

The out buffer is the native-storage target's compact compressed buffer [max_batch_size * 132 * 128 / r, 512]
viewed as [B, L_r, 1, 512] with L_r = 132 * 128 / r (r = 2 and 1); the pools are the engine's page
geometries (64-row pages for ratio 2, 128-row pages for ratio 1).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.nvfp4_kv_to_e4m3 import (  # noqa: E501
    nvfp4_kv_to_e4m3 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "nvfp4_kv_to_e4m3 requires a CUDA device"

DEV = "cuda"
W = 512
BLOCK = 16
CODE_BYTES = W // 2
ROW = CODE_BYTES + W // BLOCK  # 288
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
SENT = 0x5A
COLS = 132  # compress_block_tables width at max_seq_len 16398 (native-storage runtime dump)


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _pool(pages, page_size, seed, scale_hi=0x7E):
    """[pages, page_size, 288] uint8: random code bytes, E4M3 scale bytes 0x00..scale_hi (the quantizer
    writes non-negative finite scales; its values are e4m3(amax/6) with amax >= 6*2^-9)."""
    g = _gen(seed)
    n = pages * page_size
    codes = torch.randint(0, 256, (n, CODE_BYTES), generator=g, device=DEV, dtype=torch.int32)
    scales = torch.randint(
        0, scale_hi + 1, (n, W // BLOCK), generator=g, device=DEV, dtype=torch.int32
    )
    return torch.cat([codes, scales], 1).to(torch.uint8).view(pages, page_size, ROW)


def _views(pool):
    return pool[..., :CODE_BYTES].unsqueeze(2), pool[..., CODE_BYTES:].unsqueeze(2)


def _sentinel_out(*shape):
    return torch.full(shape, SENT, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)


def _values(flat, rows):
    """fp32 [n, W] NVFP4 values of pool rows `rows` (for the error bound and the controls)."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=DEV)
    r = flat[rows]
    c = r[:, :CODE_BYTES]
    v = torch.stack([lut[(c & 0xF).long()], lut[(c >> 4).long()]], -1).flatten(-2)
    s = (
        r[:, CODE_BYTES:]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(BLOCK, -1)
    )
    return v * s


def _build_compact(spec):
    """The target's call: every compressed entry of each request of the step, into [B, L_r, 1, 512]."""
    ratio = spec["ratio"]
    page_size = 128 // ratio
    L = COLS * page_size
    pages = 2048
    pool = _pool(pages, page_size, seed=ratio)
    kv, sf = _views(pool)
    B = 8
    g = _gen(10 + ratio)
    bt = torch.stack([torch.randperm(pages, generator=g, device=DEV)[:COLS] for _ in range(B)]).to(
        torch.int32
    )
    lens = [0, 1, page_size - 1, page_size + 1, 700 // ratio, 4097 // ratio, 16398 // ratio, L]
    bt[5, 3] = -1  # a hole: its entries are skipped
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    return kv, sf, bt, sl, _sentinel_out(B, L, 1, W), spec["inv_scale"]


def _build_engine(spec):
    """Engine page counts (ratio-2 [453120, 64, .], ratio-1 [151040, 128, .]) with rows named in the
    first and last pages, into the compact buffer view of max_batch_size mbs."""
    ratio, pages, mbs = spec["ratio"], spec["pages"], spec["mbs"]
    page_size = 128 // ratio
    L = COLS * page_size
    pool = torch.zeros(pages, page_size, ROW, dtype=torch.uint8, device=DEV)
    g = _gen(20 + ratio + mbs)
    B = mbs
    bt = torch.randint(0, pages, (B, COLS), generator=g, device=DEV, dtype=torch.int32)
    bt[0, 0] = pages - 1
    bt[B - 1, 1] = 0
    used = torch.unique(bt.long())
    pool[used] = _pool(used.numel(), page_size, seed=30 + ratio)
    sl = torch.randint(1, 16398 // ratio + 1, (B,), generator=g, device=DEV, dtype=torch.int32)
    sl[0] = 16398 // ratio
    buf = torch.full((mbs * COLS * 128 // ratio, W), SENT, dtype=torch.uint8, device=DEV)
    out = buf.view(torch.float8_e4m3fn).view(mbs, L, 1, W)
    kv, sf = _views(pool)
    return kv, sf, bt, sl, out, spec["inv_scale"]


def _build_per_query(spec):
    """The target's per-query read: the pool viewed as pages of one row ([pages * bk, 1, 1, C], a slot
    number is its own page id), a [T, 512] table of the slots each query's top-k names (valid entries
    first, -1 after), seq_lens 512 for every query, into [T, 512, 1, 512]. Engine pool geometry
    (ratio-2 453120 x 64, ratio-1 151040 x 128 rows), slots named in the first and last pages."""
    ratio, T = spec["ratio"], spec["T"]
    page_size = 128 // ratio
    pages = 453120 if ratio == 2 else 151040
    pool = torch.zeros(pages, page_size, ROW, dtype=torch.uint8, device=DEV)
    g = _gen(80 + ratio + T)
    n_slots = pages * page_size
    lo = torch.randint(0, 4096, (T, 512), generator=g, device=DEV)
    hi = torch.randint(n_slots - 4096, n_slots, (T, 512), generator=g, device=DEV)
    tab = torch.where(torch.rand(T, 512, generator=g, device=DEV) < 0.5, lo, hi)
    k = torch.randint(0, 513, (T,), generator=g, device=DEV)
    k[0] = 512
    if T > 1:
        k[1] = 0
    tab = (
        torch.where(torch.arange(512, device=DEV)[None] < k[:, None], tab, -1)
        .to(torch.int32)
        .contiguous()
    )
    used = torch.unique(tab[tab >= 0].long())
    flat = pool.view(-1, ROW)
    flat[used] = _pool(1, used.numel(), seed=90 + ratio).view(-1, ROW)
    kv = pool[..., :CODE_BYTES].reshape(n_slots, 1, 1, CODE_BYTES)
    sf = pool[..., CODE_BYTES:].reshape(n_slots, 1, 1, W // BLOCK)
    assert kv.data_ptr() == pool.data_ptr(), "the per-query view must alias the pool"
    assert int(tab.max()) >= n_slots - 4096
    cols = torch.full((T,), 512, dtype=torch.int32, device=DEV)
    return kv, sf, tab, cols, _sentinel_out(T, 512, 1, W), spec["inv_scale"]


_BUILDERS = {"compact": _build_compact, "engine": _build_engine, "per_query": _build_per_query}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _BUILDERS[cell.spec["kind"]](cell.spec)
    out = args[4]
    expected = op.reference(*args)
    assert op(*args) is out, "the op writes in place and returns what it wrote"
    op.compare(out, expected)
    written = (out.view(torch.uint8) != SENT).any(-1).sum().item()
    assert written > 0, "the cell wrote nothing, so it certifies nothing"


def test_signed_zero_saturation_and_rounding():
    """-0.0 (code 0x8) stays -0; 6 * 448 saturates to 448; 6-significant-bit products round to E4M3 RNE."""
    pool = torch.zeros(1, 1, ROW, dtype=torch.uint8, device=DEV)
    pool[0, 0, 0] = 0x87  # element 0: code 7 (6.0), element 1: code 8 (-0.0)
    pool[0, 0, 1] = 0x35  # element 2: code 5 (3.0), element 3: code 3 (1.5)
    pool[0, 0, CODE_BYTES] = 0x7E  # scale 448 for block 0
    kv, sf = _views(pool)
    args = (
        kv,
        sf,
        torch.zeros(1, 1, dtype=torch.int32, device=DEV),
        torch.ones(1, dtype=torch.int32, device=DEV),
        _sentinel_out(1, 1, 1, W),
        1.0,
    )
    expected = op.reference(*args)
    out = op(*args)
    o = out.view(torch.uint8)[0, 0, 0]
    assert o[0].item() == 0x7E, "6 * 448 saturates to 448"
    assert o[1].item() == 0x80, "-0.0 keeps its sign"
    op.compare(out, expected)


def test_transcode_is_lossy_by_at_most_half_an_e4m3_ulp():
    """E2M1 x E4M3 has up to 6 significant bits: the transcode rounds, by at most 2^-4 relative in the
    normal range; measured here on random rows (the fraction that is inexact is reported)."""
    pool = _pool(16, 64, seed=40, scale_hi=0x60)
    flat = pool.view(-1, ROW)
    kv, sf = _views(pool)
    bt = torch.arange(16, dtype=torch.int32, device=DEV).view(1, 16)
    out = torch.empty(1, 1024, 1, W, dtype=torch.float8_e4m3fn, device=DEV)
    op(kv, sf, bt, torch.tensor([1024], dtype=torch.int32, device=DEV), out, 64.0)
    v = _values(flat, torch.arange(1024, device=DEV)) * 64.0
    got = out[0, :, 0].float()
    normal = (v.abs() >= 2.0**-6) & (v.abs() <= E4M3_MAX)
    rel = ((got - v).abs() / v.abs().clamp_min(1e-30))[normal]
    inexact = (got != v)[v != 0].float().mean().item()
    print(f"inexact fraction {inexact:.3f}, max rel error in normal range {rel.max().item():.4f}")
    assert rel.max().item() <= 2.0**-4 + 1e-7
    assert inexact > 0.05  # the value set is genuinely not E4M3


def test_cuda_graph_capture_replay():
    page_size = 64
    L = COLS * page_size
    pool = _pool(512, page_size, seed=50)
    kv, sf = _views(pool)
    B = 4
    bt = torch.zeros(B, COLS, dtype=torch.int32, device=DEV)
    sl = torch.ones(B, dtype=torch.int32, device=DEV)
    out = _sentinel_out(B, L, 1, W)
    op(kv, sf, bt, sl, out, 32.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(kv, sf, bt, sl, out, 32.0)
    for seed in range(3):
        g = _gen(60 + seed)
        out.view(torch.uint8).fill_(SENT)
        bt.copy_(torch.randint(0, 512, (B, COLS), generator=g, device=DEV, dtype=torch.int32))
        sl.copy_(torch.randint(0, 4000, (B,), generator=g, device=DEV, dtype=torch.int32))
        expected = op.reference(kv, sf, bt, sl, _sentinel_out(B, L, 1, W), 32.0)
        graph.replay()
        op.compare(out, expected)


def test_wrong_variants_are_detected():
    """Controls: nibble order swapped, scale per 32 instead of 16, inv_scale ignored."""
    pool = _pool(4, 64, seed=70)
    flat = pool.view(-1, ROW)
    kv, sf = _views(pool)
    out = torch.empty(1, 256, 1, W, dtype=torch.float8_e4m3fn, device=DEV)
    op(
        kv,
        sf,
        torch.arange(4, dtype=torch.int32, device=DEV).view(1, 4),
        torch.tensor([256], dtype=torch.int32, device=DEV),
        out,
        16.0,
    )
    got = out.view(torch.uint8)[0, :, 0]
    rows = torch.arange(256, device=DEV)
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=DEV)
    r = flat[rows]
    c = r[:, :CODE_BYTES]
    s16 = r[:, CODE_BYTES:].contiguous().view(torch.float8_e4m3fn).float()
    swapped = torch.stack([lut[(c >> 4).long()], lut[(c & 0xF).long()]], -1).flatten(-2)
    v = torch.stack([lut[(c & 0xF).long()], lut[(c >> 4).long()]], -1).flatten(-2)
    for name, vals in (
        ("nibbles swapped", swapped * s16.repeat_interleave(BLOCK, -1) * 16.0),
        ("scale per 32", v * s16[:, ::2].repeat_interleave(2 * BLOCK, -1) * 16.0),
        ("inv_scale ignored", v * s16.repeat_interleave(BLOCK, -1)),
    ):
        exp = vals.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        frac = (got != exp).float().mean().item()
        assert frac > 0.3, f"control '{name}' differs on only {frac:.3f}"
