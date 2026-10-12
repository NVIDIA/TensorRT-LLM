# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the indexer_k_cache_gather catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: the consumer
views and the decoded value set, the DeepGEMM consumer, skipped rows, the op's rejections and the
guard, the wrong-addressing controls, the engine-size pools, every row count up to the per-step
maximum, and pools up to the largest any DeepSeek-V4.1 cache manager can have.

The DeepSeek-V4.1 prefill index-K read path over the native MXFP4 pool: index K is quantized with
``torch.ops.trtllm.fp4_quantize(x, None, 32, True, False)`` (MXFP4, UE8M0, linear scale layout),
stored in a paged ``[pages, block_kv, 1, 68]`` uint8 pool whose page holds ``block_kv x 64`` code
bytes followed by ``block_kv x 4`` scale bytes, and gathered back to contiguous rows for
``tensorrt_llm.deep_gemm.fp8_fp4_mqa_logits``.

The pool is written here with native torch indexing (not the scatter op), the slot mappings are
computed here from a block table with native torch arithmetic, and beside the entry's reference
the gathered rows must equal the quantizer's own bytes. Deliberately wrong addressing schemes are
run as controls and must fail.
"""

import zlib

import pytest
import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.indexer_k_cache_gather import (  # noqa: E501
    indexer_k_cache_gather as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "indexer_k_cache_gather requires a CUDA device"

DEV = "cuda"
DIM = 128  # index_head_dim
CODE_BYTES = 64  # packed E2M1 bytes per key (the op's head_dim argument)
SCALE_BYTES = 4  # four UE8M0 bytes per key
KEY_BYTES = CODE_BYTES + SCALE_BYTES
# The pool's page count is the cache manager's number, never the batch's: every case allocates
# POOL_FACTOR x the pages its batch needs (+ POOL_EXTRA) and hands out the batch's pages through a
# random permutation, so "the page the block table names" and "the page this request happens to
# own" are different numbers. Chosen here, not taken from a runtime dump.
POOL_FACTOR = 3
POOL_EXTRA = 8

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
    device=DEV,
)
_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)


# ─── inputs ───────────────────────────────────────────────────────────────────────────────────


def _index_k(S: int, seed: int) -> torch.Tensor:
    """bf16 [S, 128] with per-row magnitudes spread over 2^-12..2^12 (varied UE8M0 bytes)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(S, DIM, generator=g, device=DEV)
    x = x * torch.exp2(torch.randint(-12, 13, (S, 1), generator=g, device=DEV).float())
    if S > 3:
        x[3, 32:64] = 0.0  # an all-zero 32-block
    return x.to(torch.bfloat16)


def _quantize(x: torch.Tensor):
    """The target's index-K quantizer: -> codes uint8 [S, 64], scale bytes uint8 [S, 4]."""
    codes, sf = torch.ops.trtllm.fp4_quantize(x, None, 32, True, False)
    S = x.shape[0]
    assert codes.shape == (S, CODE_BYTES) and codes.dtype == torch.uint8
    assert sf.shape == (S * SCALE_BYTES,) and sf.dtype == torch.uint8
    return codes, sf.view(S, SCALE_BYTES)


def _reference_values(x: torch.Tensor) -> torch.Tensor:
    """Checkpoint fp4_act_quant(k, 32, True) value set, native torch: fp64 [S, 128].

    Per 32-block: amax clamped to 6*2^-126, scale = 2^ceil(log2(amax * (1/6))) taken from the
    fp32 bits, y = E2M1_rne(clamp(x / scale, -6, 6)) * scale.
    """
    xf = x.float().unflatten(-1, (4, 32))
    amax = xf.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax * (1.0 / 6.0)).contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = ((e + 127) << 23).view(torch.float32)
    y = (xf / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    a = y.abs()
    code = (a.unsqueeze(-1) > _MIDPOINTS).sum(-1)
    code = code + ((a == 0.75) | (a == 1.75) | (a == 3.5)).to(code.dtype)  # ties to even code
    mag = _E2M1[code]
    return (torch.where(y < 0, -mag, mag) * scale.double().unsqueeze(-1)).flatten(-2)


def _decode(k_int8: torch.Tensor, k_sf_int32: torch.Tensor) -> torch.Tensor:
    """int8 [S, 64] + int32 [S] -> fp64 [S, 128]: element 2j in the low nibble of byte j, byte b of
    the little-endian int32 is the UE8M0 exponent of elements 32b..32b+31."""
    p = k_int8.view(torch.uint8).long()
    v = torch.stack([_E2M1[p & 0xF], _E2M1[p >> 4]], dim=-1).flatten(-2)
    w = k_sf_int32.long() & 0xFFFFFFFF
    sb = torch.stack([(w >> (8 * b)) & 0xFF for b in range(4)], dim=-1)
    scale = ((sb + (1023 - 127)) << 52).view(torch.float64)  # 2^(byte - 127), exact
    return (v.unflatten(-1, (4, 32)) * scale.unsqueeze(-1)).flatten(-2)


# ─── pool, block table, slot mappings (native torch; the caller's arithmetic) ───────────────────


def _block_tables(lens, block_kv: int, seed: int):
    """Per-request page lists from a random permutation of a pool 3x too large -> (table, pages)."""
    need = [-(-n // block_kv) for n in lens]
    num_pages = POOL_FACTOR * sum(need) + POOL_EXTRA
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_pages, generator=g)
    table = torch.full((len(lens), max(max(need), 1)), -1, dtype=torch.int32)
    off = 0
    for r, n in enumerate(need):
        table[r, :n] = perm[off : off + n].to(torch.int32)
        off += n
    return table.to(DEV), num_pages


def _positions(lens, table: torch.Tensor, block_kv: int):
    """Full slot-mapping order (request by request, position 0..len-1) -> (page, slot) int64."""
    req = torch.repeat_interleave(
        torch.arange(len(lens), device=DEV), torch.tensor(lens, device=DEV)
    )
    starts = torch.tensor([0] + list(lens[:-1]), device=DEV).cumsum(0)
    pos = torch.arange(sum(lens), device=DEV) - starts[req]
    page = table[req, pos // block_kv].long()
    return page, pos % block_kv


def _slot_mappings(page: torch.Tensor, slot: torch.Tensor, block_kv: int):
    """Flat element indices into the logical [pages, block_kv, 1, 68] view."""
    page_elems = block_kv * KEY_BYTES
    data = page * page_elems + slot * CODE_BYTES
    scale = page * page_elems + block_kv * CODE_BYTES + slot * SCALE_BYTES
    return data.contiguous(), scale.contiguous()


def _new_pool(num_pages: int, block_kv: int, seed: int, layers: int = 1, layer: int = 0):
    """Random background bytes; returns the owner layer's [pages, block_kv, 1, 68] view.

    layers > 1 builds a layer-interleaved buffer [pages, layers, block_kv, 68] whose owner view has
    page stride layers * block_kv * 68 (not contiguous)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    buf = torch.randint(0, 256, (num_pages, layers, block_kv, KEY_BYTES), generator=g, device=DEV)
    buf = buf.to(torch.uint8)
    return buf, buf[:, layer].unsqueeze(2)


def _write(pool: torch.Tensor, page, slot, codes, sf4, block_kv: int):
    """Store rows in the kernel layout: a page's block_kv x 64 code bytes, then its block_kv x 4
    scale bytes."""
    flat = pool.view(pool.shape[0], block_kv * KEY_BYTES)  # a view, also for the strided pool
    cd = slot[:, None] * CODE_BYTES + torch.arange(CODE_BYTES, device=DEV)
    cs = block_kv * CODE_BYTES + slot[:, None] * SCALE_BYTES + torch.arange(SCALE_BYTES, device=DEV)
    flat[page[:, None], cd] = codes
    flat[page[:, None], cs] = sf4


def _build(lens, block_kv: int, seed: int, layers: int = 1):
    x = _index_k(sum(lens), seed)
    codes, sf4 = _quantize(x)
    table, num_pages = _block_tables(lens, block_kv, seed)
    buf, pool = _new_pool(num_pages, block_kv, seed + 1, layers, layers // 2)
    page, slot = _positions(lens, table, block_kv)
    _write(pool, page, slot, codes, sf4, block_kv)
    d_idx, s_idx = _slot_mappings(page, slot, block_kv)
    return dict(
        x=x,
        codes=codes,
        sf4=sf4,
        table=table,
        num_pages=num_pages,
        buf=buf,
        pool=pool,
        page=page,
        slot=slot,
        d_idx=d_idx,
        s_idx=s_idx,
    )


def _row_mismatch(k, k_scale, codes, sf4) -> float:
    """Fraction of rows whose 64 code bytes or 4 scale bytes differ from the expected ones."""
    bad = (k.view(torch.uint8) != codes).any(1) | (
        k_scale.view(torch.uint8).view(-1, 4) != sf4
    ).any(1)
    return bad.float().mean().item()


def _assert_rows(k, k_scale, codes, sf4):
    n = codes.shape[0]
    assert k.dtype == torch.float8_e4m3fn and k.shape == (n, CODE_BYTES) and k.is_contiguous()
    assert k_scale.dtype == torch.float32 and k_scale.shape == (n, 1) and k_scale.is_contiguous()
    assert torch.equal(k.view(torch.uint8), codes), "code bytes differ"
    assert torch.equal(k_scale.view(torch.uint8).view(n, SCALE_BYTES), sf4), "scale bytes differ"


# ─── certified behaviour ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    block_kv, layers = spec["block_kv"], spec.get("layers", 1)
    b = _build(spec["lens"], block_kv, seed=zlib.crc32(cell.why.encode()), layers=layers)
    if layers > 1:
        # the owner layer's page view of a layer-interleaved buffer: slot mappings stay logical
        # indices into [pages, block_kv, 1, 68]
        assert not b["pool"].is_contiguous()
        assert b["pool"].stride(0) == layers * block_kv * KEY_BYTES
    before = b["buf"].clone()
    for k_start, n in spec["chunks"]:
        out = op(b["pool"], b["d_idx"], b["s_idx"], k_start, n, CODE_BYTES)
        op.compare(out, op.reference(b["pool"], b["d_idx"], b["s_idx"], k_start, n, CODE_BYTES))
        # the gathered rows are fp4_quantize's own bytes, in fresh contiguous outputs
        _assert_rows(*out, b["codes"][k_start : k_start + n], b["sf4"][k_start : k_start + n])
        # a repeat call is bitwise identical
        k2, k_scale2 = op(b["pool"], b["d_idx"], b["s_idx"], k_start, n, CODE_BYTES)
        assert torch.equal(k2.view(torch.uint8), out[0].view(torch.uint8))
        assert torch.equal(k_scale2.view(torch.uint8), out[1].view(torch.uint8))
    # the gather never writes the pool (nor, for a strided view, the other layers)
    assert torch.equal(b["buf"], before), "the gather wrote into the pool"


@pytest.mark.parametrize("block_kv", [64, 128])
def test_consumer_views_and_reference_values(block_kv):
    """The int8 / int32 views DeepGEMM takes are no-copy, contiguous, 16-byte aligned, carry the
    quantizer's bytes, and decode to the checkpoint's fp4_act_quant value set exactly."""
    lens = [700, 1, 2500, 37]
    b = _build(lens, block_kv, seed=block_kv)
    S = sum(lens)
    k, k_scale = op(b["pool"], b["d_idx"], b["s_idx"], 0, S, CODE_BYTES)
    kv = k.view(torch.int8)
    kv_sf = k_scale.view(torch.int32).reshape(-1)
    assert kv.dtype == torch.int8 and kv.shape == (S, CODE_BYTES) and kv.is_contiguous()
    assert kv_sf.dtype == torch.int32 and kv_sf.shape == (S,) and kv_sf.is_contiguous()
    assert kv.data_ptr() == k.data_ptr() and kv_sf.data_ptr() == k_scale.data_ptr()
    assert kv_sf.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0
    assert torch.equal(kv, b["codes"].view(torch.int8))
    assert torch.equal(kv_sf, b["sf4"].contiguous().view(torch.int32).view(-1))
    for byte in range(4):  # little-endian: byte b of the word is the UE8M0 of block b
        assert torch.equal((kv_sf.long() >> (8 * byte)) & 0xFF, b["sf4"][:, byte].long())
    assert torch.equal(_decode(kv, kv_sf), _reference_values(b["x"])), (
        "value set differs from fp4_act_quant"
    )


def test_negative_slot_rows_are_skipped_others_exact():
    """A row whose data or scale slot is negative is skipped; every other row is exact."""
    lens = [900]
    block_kv = 64
    b = _build(lens, block_kv, seed=5)
    d_idx, s_idx = b["d_idx"].clone(), b["s_idx"].clone()
    d_idx[::5] = -1
    s_idx[2::7] = -1
    k, k_scale = op(b["pool"], d_idx, s_idx, 0, 900, CODE_BYTES)
    keep = (d_idx >= 0) & (s_idx >= 0)
    assert k.shape == (900, CODE_BYTES) and k_scale.shape == (900, 1)
    assert torch.equal(k.view(torch.uint8)[keep], b["codes"][keep])
    assert torch.equal(k_scale.view(torch.uint8).view(900, 4)[keep], b["sf4"][keep])
    ref_k, ref_scale = op.reference(b["pool"], d_idx, s_idx, 0, 900, CODE_BYTES)
    assert torch.equal(k.view(torch.uint8)[keep], ref_k.view(torch.uint8)[keep])
    assert torch.equal(k_scale.view(torch.uint8)[keep], ref_scale.view(torch.uint8)[keep])


def test_consumer_accepts_gathered_view():
    """The gathered views drive DeepGEMM fp8_fp4_mqa_logits to the same logits, bit for bit, as
    the quantizer's own tensors; the op's raw float8 dtype is refused by the consumer."""
    lens = [1200, 800]
    b = _build(lens, 64, seed=21)
    S = sum(lens)
    k, k_scale = op(b["pool"], b["d_idx"], b["s_idx"], 0, S, CODE_BYTES)
    T, H = 64, 32
    g = torch.Generator(device=DEV).manual_seed(22)
    qx = torch.randn(T * H, DIM, generator=g, device=DEV).to(torch.bfloat16)
    q_codes, q_sf = torch.ops.trtllm.fp4_quantize(qx, None, 32, True, False)
    q = q_codes.view(torch.int8).view(T, H, CODE_BYTES)
    q_sf = q_sf.view(torch.int32).view(T, H)
    w = torch.randn(T, H, generator=g, device=DEV) * 2.0**-6
    ks = torch.zeros(T, dtype=torch.int32, device=DEV)
    ke = torch.randint(1, S + 1, (T,), generator=g, device=DEV).to(torch.int32)
    kv_src = (b["codes"].view(torch.int8), b["sf4"].contiguous().view(torch.int32).view(-1))
    ref = deep_gemm.fp8_fp4_mqa_logits((q, q_sf), kv_src, w, ks, ke, clean_logits=True)
    got = deep_gemm.fp8_fp4_mqa_logits(
        (q, q_sf),
        (k.view(torch.int8), k_scale.view(torch.int32).reshape(-1)),
        w,
        ks,
        ke,
        clean_logits=True,
    )
    assert torch.equal(got, ref)
    with pytest.raises(RuntimeError):
        deep_gemm.fp8_fp4_mqa_logits(
            (q, q_sf), (k, k_scale.view(torch.int32).reshape(-1)), w, ks, ke, clean_logits=True
        )


# ─── rejections (each probed on sm_103) ──────────────────────────────────────────────────────


def test_rejections():
    b = _build([37], 64, seed=9)
    pool, d, s = b["pool"], b["d_idx"], b["s_idx"]
    cases = [
        (
            lambda: op(pool, d, s, 0, 37, 96),
            "head_dim must be 128 \\(FP8\\) or 64",
        ),
        (
            lambda: op(pool.view(pool.shape[0], 64, 68), d, s, 0, 37, 64),
            "k_cache must be a 4D Tensor",
        ),
        (
            lambda: op(pool, d.int(), s.int(), 0, 37, 64),
            "slot_mapping_fp8 must be int64",
        ),
        (lambda: op(pool, d, s, 1, 37, 64), "slot_mapping_fp8 too short"),
        (
            lambda: op(pool.view(pool.shape[0], 32, 2, 68), d, s, 0, 37, 64),
            "k_cache dimension 2 must be 1",
        ),
        (
            lambda: op(pool, d.cpu(), s.cpu(), 0, 37, 64),
            "All tensors must be CUDA tensors",
        ),
        (
            lambda: op(pool, d.view(1, 37), s.view(1, 37), 0, 37, 64),
            "slot_mapping_fp8 must be a 1D Tensor",
        ),
        (
            lambda: op(pool, torch.stack([d, d], 1)[:, 0], s, 0, 37, 64),
            "slot_mapping_fp8 must be contiguous",
        ),
        (
            lambda: op(pool.view(torch.int8), d, s, 0, 37, 64),
            "expected scalar type Byte but found Char",
        ),
    ]
    # The op's own rejections, outside `validating` (the guard would fire first on head_dim 96).
    for fn, msg in cases:
        with pytest.raises(RuntimeError, match=msg):
            fn()
    # The entry's guard: head_dim 128 on the 68-byte pool. The op itself accepts it and returns the
    # neighbouring key's bytes (test_controls_fail shows it outside `validating`).
    with pytest.raises(AssertionError, match="must equal k_cache.size\\(3\\) - 4"):
        with validating(op):
            op(pool, d, s, 0, 37, 128)


# ─── deliberately wrong addressing schemes: every one must be visibly wrong ──────────────────


@pytest.mark.parametrize("block_kv", [64, 128])
def test_controls_fail(block_kv):
    lens = [700, 1, 2500, 37]
    b = _build(lens, block_kv, seed=31 + block_kv)
    S = sum(lens)
    codes, sf4 = b["codes"], b["sf4"]
    page, slot = b["page"], b["slot"]

    def run(d_idx, s_idx, head_dim=CODE_BYTES):
        k, k_scale = op(b["pool"], d_idx, s_idx, 0, S, head_dim)
        return k[:, :CODE_BYTES].contiguous(), k_scale

    good = _row_mismatch(*run(b["d_idx"], b["s_idx"]), codes, sf4)
    assert good == 0.0

    fractions = {}
    # 1. wrong block table: the same requests paged through another permutation of the pool (a
    #    stale table, or another owner layer's) -- every entry is a valid page, just not theirs
    wrong_table, _ = _block_tables(lens, block_kv, seed=999 + block_kv)
    wrong_page, _ = _positions(lens, wrong_table, block_kv)
    fractions["wrong block table"] = _row_mismatch(
        *run(*_slot_mappings(wrong_page, slot, block_kv)), codes, sf4
    )
    # 2. off-by-one page offset: page id + 1 (kept inside the pool)
    fractions["page id + 1"] = _row_mismatch(
        *run(*_slot_mappings((page + 1) % b["num_pages"], slot, block_kv)), codes, sf4
    )
    # 3. off-by-one slot inside the page (wraps inside the same page)
    fractions["slot + 1 within the page"] = _row_mismatch(
        *run(*_slot_mappings(page, (slot + 1) % block_kv, block_kv)), codes, sf4
    )
    # 4. key-interleaved addressing ([64 codes, 4 scales] per key) over the paged layout
    pe = block_kv * KEY_BYTES
    fractions["key-interleaved addressing"] = _row_mismatch(
        *run(page * pe + slot * KEY_BYTES, page * pe + slot * KEY_BYTES + CODE_BYTES), codes, sf4
    )
    # 5. page stride taken as block_kv x 64 (scale region forgotten)
    fractions["page stride without the scale region"] = _row_mismatch(
        *run(
            page * block_kv * CODE_BYTES + slot * CODE_BYTES,
            page * block_kv * CODE_BYTES + block_kv * CODE_BYTES + slot * SCALE_BYTES,
        ),
        codes,
        sf4,
    )

    # 6. the FP8 head_dim (128) through the unguarded op on the 68-byte pool is accepted: bytes 64..127
    #    of a row are the next slot's codes (or, for a page's last slot, its scale region)
    k128, _ = op(b["pool"], b["d_idx"], b["s_idx"], 0, S, 128)
    assert k128.shape == (S, 128)
    assert torch.equal(k128.view(torch.uint8)[:, :CODE_BYTES], codes)
    nxt = torch.cat([codes[1:], codes[:1]])
    same_page_next = torch.cat(
        [(page[1:] == page[:-1]) & (slot[1:] == slot[:-1] + 1), torch.tensor([False], device=DEV)]
    )
    upper = k128.view(torch.uint8)[:, CODE_BYTES:]
    assert torch.equal(upper[same_page_next], nxt[same_page_next]), "upper half is not the next key"
    frac_next = same_page_next.float().mean().item()

    print(f"\n[controls block_kv={block_kv} S={S}] row-mismatch fractions (correct run: {good}):")
    for name, frac in fractions.items():
        print(f"  {name}: {frac:.4f}")
    print(
        f"  head_dim 128 on the MXFP4 pool: accepted; rows whose bytes 64..127 are the next key's "
        f"codes: {frac_next:.4f}"
    )
    assert frac_next >= 0.9
    for name, frac in fractions.items():
        assert frac >= 0.9, f"control '{name}' was not caught (mismatch fraction {frac})"


# ─── the engine's pools, every row count, one kernel ─────────────────────────────────────────
#
# The core target calls this op only in a step with context tokens (`ModelingV2Core._index`,
# `if tc:` branch), with k_token_start 0 and num_tokens = v41_ctx_index_k_len[r]
# (`_prepare_v41_ctx_index_k` of the V4.1 metadata: S_r = sum over the step's context requests of
# kv_len // r, cached entries included). Context and mixed steps run eagerly, so the call is never
# captured. One request contributes at most Lm // r rows, Lm = max_seq_len (+ 14 with DSpark); the
# longest request of the envelope (max_seq_len 1,048,576, DSpark on) gives 1,048,590 at ratio 1 and
# 524,295 at ratio 2, the N_max of the exhaustive row sweep below. A step with several long context
# requests sums them (up to max_batch_size 256 of them); that axis is swept on the largest pool by
# test_rows_to_2_25_on_b_max_pool and bounded by the defect test after it. Observed in the core
# runtime dumps: up to 2938 (boot, r = 1), 24058 (the 24K passkey run, r = 1) and 12029 (r = 2).
#
# The pools are sized from the same dumps. The cache manager allocates B blocks; the ratio-2
# INDEXER_COMPRESS allocation holds 3 pages of 64 entries per block, which owners 2, 8 and 14 see
# as contiguous views starting 0, 1 and 2 pages into it ([3B - shift, 64, 1, 68]; their page ids
# are 3 x block), and the ratio-1 pool (owner 20) holds one page of 128 entries per block
# ([B, 128, 1, 68], page id = block). B = 137386 / 142336 (boot dumps) and 151040 (the largest
# pool of the runtime dumps, 453120 x 64 and 151040 x 128).
ENGINE_BLOCKS = (137386, 142336, 151040)
OWNER_SHIFT = {2: 0, 8: 1, 14: 2}
N_MAX = {1: 1_048_590, 2: 524_295}
DUMPED_ROWS = {  # s_len values the core runtime dumps hold (boot gates and an accuracy-gate run)
    1: (1, 2, 16, 48, 323, 576, 1938, 2048, 2050, 2938, 4618, 8192, 16384, 24058),
    2: (1, 8, 24, 161, 288, 969, 1024, 1025, 1469, 2309, 4096, 8192, 12029),
}
# One request's whole index K at each certified max_seq_len L: Lm // r with Lm = L + 14 (DSpark on)
# and Lm = L (DSpark off).
L_VALUES = (16384, 32768, 65536, 131072, 262144, 1048576)
ONE_REQUEST_ROWS = {
    r: tuple(sorted({(L + d) // r for L in L_VALUES for d in (0, 14)})) for r in (1, 2)
}
_ENGINE_CACHE = {}


def _engine_alloc(ratio: int, blocks: int, seed: int = 0) -> torch.Tensor:
    """The ratio's whole INDEXER_COMPRESS allocation at engine size, random background bytes.
    One per (ratio, blocks) per process; the gather never writes it (checked where used)."""
    key = (ratio, blocks)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE.clear()  # keep one engine-size allocation alive at a time
        torch.cuda.empty_cache()
        pages = 3 * blocks if ratio == 2 else blocks
        g = torch.Generator(device=DEV).manual_seed(seed + blocks + ratio)
        _ENGINE_CACHE[key] = torch.randint(
            0, 256, (pages, 128 // ratio, 1, KEY_BYTES), dtype=torch.uint8, device=DEV, generator=g
        )
    return _ENGINE_CACHE[key]


def _owner_view(alloc: torch.Tensor, ratio: int, owner: int) -> torch.Tensor:
    """What get_indexer_k_cache_buffers(owner) returns: a contiguous view `shift` pages in."""
    view = alloc[OWNER_SHIFT[owner] :] if ratio == 2 else alloc
    assert view.is_contiguous()
    return view


def _engine_table(lens, block_kv: int, blocks: int, ratio: int, seed: int):
    """Page ids for the requests, drawn from every block of the pool: the first request's first
    page is block 0 and the last request's last page is block `blocks - 1` (the far end); page id =
    3 x block at ratio 2 (the shared-block scale of the three ratio-2 owners), = block at ratio 1."""
    need = [-(-n // block_kv) for n in lens]
    total = sum(need)
    g = torch.Generator(device="cpu").manual_seed(seed)
    mid = torch.randperm(blocks - 2, generator=g)[: max(total - 2, 0)] + 1
    chosen = torch.cat([torch.tensor([0]), mid, torch.tensor([blocks - 1])])
    table = torch.full((len(lens), max(need)), -1, dtype=torch.int64)
    off = 0
    for r, n in enumerate(need):
        table[r, :n] = chosen[off : off + n]
        off += n
    return (table * (3 if ratio == 2 else 1)).to(DEV)


def _engine_rows(ratio: int, blocks: int, owner: int, lens, seed: int):
    """Write fp4_quantize bytes for `lens` requests into the owner's view of the engine-size pool
    (native torch indexing) and return the view, the slot mappings and the expected bytes."""
    block_kv = 128 // ratio
    alloc = _engine_alloc(ratio, blocks)
    view = _owner_view(alloc, ratio, owner)
    table = _engine_table(lens, block_kv, blocks, ratio, seed)
    page, slot = _positions(lens, table, block_kv)
    x = _index_k(sum(lens), seed)
    codes, sf4 = _quantize(x)
    _write(view, page, slot, codes, sf4, block_kv)
    d_idx, s_idx = _slot_mappings(page, slot, block_kv)
    return alloc, view, page, slot, d_idx, s_idx, codes, sf4


@pytest.mark.parametrize("blocks", ENGINE_BLOCKS)
@pytest.mark.parametrize("ratio,owner", [(2, 2), (2, 8), (2, 14), (1, 20)])
def test_engine_pool_far_end(ratio, owner, blocks):
    """Every owner's view of the engine-size pool: requests paged over the whole pool, the last one
    ending on the last slot of block `blocks - 1` (for owner 14 at ratio 2 and owner 20 that is the
    last byte of the allocation). Rows are the quantizer's bytes; the allocation is unchanged."""
    block_kv = 128 // ratio
    lens = [3 * block_kv, 5, 2 * block_kv + 7, 4 * block_kv]
    alloc, view, page, slot, d_idx, s_idx, codes, sf4 = _engine_rows(
        ratio, blocks, owner, lens, seed=blocks + owner
    )
    last_byte = OWNER_SHIFT.get(owner, 0) * block_kv * KEY_BYTES + int(s_idx[-1]) + SCALE_BYTES
    assert (
        int(page[-1]) == (blocks - 1) * (3 if ratio == 2 else 1) and int(slot[-1]) == block_kv - 1
    )
    before = alloc.clone()
    k, k_scale = op(view, d_idx, s_idx, 0, sum(lens), CODE_BYTES)
    _assert_rows(k, k_scale, codes, sf4)
    assert torch.equal(alloc, before), "the gather wrote into the pool"
    print(
        f"[engine pool r={ratio} owner={owner}] view {tuple(view.shape)} alloc {alloc.numel()} B; "
        f"far row's scale word ends at byte {last_byte} of the allocation; bit-exact"
    )
    del before


@pytest.mark.parametrize("ratio", [1, 2])
def test_every_row_count_prefix_bitwise(ratio):
    """Every row count on the target's call (k_token_start 0, num_tokens r), over the engine-size pool.

    (a) the N_max call returns the quantizer's bytes for all N_max rows (32 requests paged over the
    whole pool, the far end included); (b) every r in 1 .. N_max returns exactly the first r rows
    of that call, bitwise. The sweep alternates between two row orders (the second is the first
    rolled by N_max // 2 + 1), so a call that left rows unwritten in a recycled allocation would
    show the other order's bytes; (c) the dumped s_len values with a fresh packed batch each, and
    one request's whole index K at every certified max_seq_len (ONE_REQUEST_ROWS), each a fresh
    single-request batch paged over the pool."""
    n_max = N_MAX[ratio]
    owner = 14 if ratio == 2 else 20
    blocks = ENGINE_BLOCKS[-1]
    lens = [n_max // 32] * 32
    lens[-1] += n_max - sum(lens)
    alloc, view, page, slot, d_a, s_a, codes, sf4 = _engine_rows(
        ratio, blocks, owner, lens, seed=ratio
    )
    before = alloc.clone()
    k_full, ks_full = op(view, d_a, s_a, 0, n_max, CODE_BYTES)
    _assert_rows(k_full, ks_full, codes, sf4)  # (a)
    ref = [(k_full.view(torch.uint8), ks_full.view(torch.uint8).view(n_max, SCALE_BYTES))]
    shift = n_max // 2 + 1
    d_b, s_b = d_a.roll(shift).contiguous(), s_a.roll(shift).contiguous()
    ref.append((ref[0][0].roll(shift, 0).contiguous(), ref[0][1].roll(shift, 0).contiguous()))
    maps = [(d_a, s_a), (d_b, s_b)]
    assert (ref[1][0] != ref[0][0]).any(1).float().mean().item() > 0.99
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for r in range(1, n_max + 1):  # (b)
        v = r & 1
        k, ks = op(view, maps[v][0], maps[v][1], 0, r, CODE_BYTES)
        bad += (k.view(torch.uint8) != ref[v][0][:r]).any(1).sum()
        bad += (ks.view(torch.uint8).view(r, SCALE_BYTES) != ref[v][1][:r]).any(1).sum()
    bad = int(bad)
    assert torch.equal(alloc, before), "the gather wrote into the pool"
    del before
    print(
        f"[rows r={ratio}] N_max {n_max} on {tuple(view.shape)}: every r in 1..{n_max} "
        f"prefix-bitwise, mismatching rows {bad}"
    )
    assert bad == 0
    del k_full, ks_full, ref, maps, d_a, s_a, d_b, s_b, codes, sf4
    for s_len in DUMPED_ROWS[ratio]:  # (c)
        g = torch.Generator(device="cpu").manual_seed(s_len)
        cuts = sorted(torch.randperm(s_len - 1, generator=g)[: min(3, s_len - 1)].add(1).tolist())
        req_lens = [b - a for a, b in zip([0] + cuts, cuts + [s_len])]
        _, view, _, _, d, s, c, sf = _engine_rows(
            ratio, blocks, owner, req_lens, seed=s_len + ratio
        )
        k, ks = op(view, d, s, 0, s_len, CODE_BYTES)
        _assert_rows(k, ks, c, sf)
    for s_len in ONE_REQUEST_ROWS[ratio]:
        _, view, page, _, d, s, c, sf = _engine_rows(
            ratio, blocks, owner, [s_len], seed=s_len + 7 * ratio
        )
        assert int(page[-1]) == (blocks - 1) * (
            3 if ratio == 2 else 1
        )  # its last page is the far end
        k, ks = op(view, d, s, 0, s_len, CODE_BYTES)
        _assert_rows(k, ks, c, sf)
    print(
        f"[rows r={ratio}] dumped s_len {DUMPED_ROWS[ratio]} and one request's rows at every L "
        f"{ONE_REQUEST_ROWS[ratio]}: bit-exact"
    )


def test_one_kernel_at_every_row_count():
    """The op is an ahead-of-time compiled kernel whose row count is only its grid: the same single
    kernel launches at 1 row, at dumped counts and at N_max (no per-shape variant)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    n_max = N_MAX[1]
    _, view, _, _, d, s, _, _ = _engine_rows(1, ENGINE_BLOCKS[-1], 20, [n_max], seed=5)
    seen = set()
    for r in (1, 2, 1025, 2938, 24058, n_max):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(view, d, s, 0, r, CODE_BYTES)
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
        print(f"[kernels r={r}] {kernels}")
        assert len(kernels) == 1 and "indexerKCacheGatherUnifiedKernel" in kernels[0], kernels
        seen.add(kernels[0])
    assert len(seen) == 1


# Beyond the engine's pools: an allocation past 2^32 bytes, rows on the pages that straddle the
# 2^31- and 2^32-byte offsets and on the last page. The kernel unravels the int64 flat index and
# forms an int64 byte offset; a caller that computed the offsets in 32 bits would wrap them.
MARGIN_PAGES = {64: 1_000_000, 128: 500_000}  # 4.352e9 bytes either way


@pytest.mark.parametrize("block_kv", [64, 128])
def test_past_2_to_the_32_bytes_and_wrap_control(block_kv):
    page_bytes = block_kv * KEY_BYTES
    pages = MARGIN_PAGES[block_kv]
    _ENGINE_CACHE.clear()
    torch.cuda.empty_cache()
    g = torch.Generator(device=DEV).manual_seed(block_kv)
    pool = torch.randint(
        0, 256, (pages, block_kv, 1, KEY_BYTES), dtype=torch.uint8, device=DEV, generator=g
    )
    assert pool.numel() > 2**32
    far = [2**31 // page_bytes, 2**32 // page_bytes, pages - 1]
    for p, b in zip(far[:2], (2**31, 2**32)):
        assert p * page_bytes < b < (p + 1) * page_bytes  # the page straddles the boundary
    lens = [block_kv] * 3
    table = torch.tensor(far, dtype=torch.int64, device=DEV).view(3, 1)
    page, slot = _positions(lens, table, block_kv)
    x = _index_k(sum(lens), seed=block_kv)
    codes, sf4 = _quantize(x)
    _write(pool, page, slot, codes, sf4, block_kv)
    d_idx, s_idx = _slot_mappings(page, slot, block_kv)
    k, k_scale = op(pool, d_idx, s_idx, 0, sum(lens), CODE_BYTES)
    _assert_rows(k, k_scale, codes, sf4)

    # control: the same offsets wrapped to signed 32 bits (negative ones are skipped by the op,
    # the rest read the wrong bytes)
    def wrap(t):
        return ((t + 2**31) % 2**32) - 2**31

    kw, ksw = op(pool, wrap(d_idx), wrap(s_idx), 0, sum(lens), CODE_BYTES)
    frac = _row_mismatch(kw, ksw, codes, sf4)
    past = (d_idx >= 2**31).float().mean().item()
    print(
        f"[margin block_kv={block_kv}] pool {pool.numel()} B, pages {far}: bit-exact; int32-wrapped "
        f"offsets: {frac:.4f} of rows wrong ({past:.4f} of rows lie past 2^31)"
    )
    assert frac >= past > 0.5
    del pool


@pytest.mark.parametrize("ratio", [2])
def test_control_wrong_owner_view(ratio):
    """Owner 14's page ids read through owner 2's view (the three ratio-2 views overlap, shifted by
    one page each): every row is another owner's bytes."""
    blocks = ENGINE_BLOCKS[-1]
    lens = [640, 3, 200]
    alloc, view14, page, slot, d, s, codes, sf4 = _engine_rows(ratio, blocks, 14, lens, seed=77)
    view2 = _owner_view(alloc, ratio, 2)
    k, ks = op(view14, d, s, 0, sum(lens), CODE_BYTES)
    _assert_rows(k, ks, codes, sf4)
    kw, ksw = op(view2, d, s, 0, sum(lens), CODE_BYTES)
    frac = _row_mismatch(kw, ksw, codes, sf4)
    print(f"[control] owner 14's ids through owner 2's view: {frac:.4f} of rows wrong")
    assert frac >= 0.9


# Pool-size invariance. Every V4.1 persistent role of one cache manager shares one block
# count B, so B is capped by the whole GPU (284208 MiB = 298,013,687,808 B) divided by the manager's
# persistent bytes per block. The cheapest manager that holds an index-K pool is the PP2 stage-1
# manager in native storage (layers 20-39: 356 B/token, 45,568 B per block, measured); an unsplit
# native manager costs 113,920 B per block and a stage-0 one 68,352. So no V4.1 manager on a GB300
# holds more than floor(298,013,687,808 / 45,568) = 6,539,977 blocks. The largest pool measured (a
# stage-1 manager built alone at half the GPU) is 3,245,959 blocks, and the unsplit ceiling 2,615,990; both
# are run as intermediate sizes. One B_MAX serves both ratios (the ratio-2 owners live only in
# unsplit and stage-0 managers, whose own ceilings are lower), so the ratio-2 allocation at B_MAX,
# 3 x 6,539,977 pages of 64 entries (85,385,939,712 B), is a superset of any real one.
# The same rows, at the same page ids, are gathered from exact-size allocations (the pool ends
# where the allocation ends) from B_min, the smallest pool that holds their pages, up to that
# bound. Pages no row owns hold POISON, so a read outside the rows' pages would show.
B_MAX = 6_539_977
INV_BLOCKS = (481, 1493, 151040, 852832, 2_615_990, 3_245_959, B_MAX)
B_OTHER = 852832  # the control's wrong page count
POISON = 0xFF
FAR_BYTES = (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)  # those below the allocation's size


def _poisoned_alloc(ratio: int, blocks: int) -> torch.Tensor:
    """The ratio's whole INDEXER_COMPRESS allocation for `blocks` blocks, every byte POISON."""
    _ENGINE_CACHE.clear()
    torch.cuda.empty_cache()
    pages = 3 * blocks if ratio == 2 else blocks
    return torch.full((pages, 128 // ratio, 1, KEY_BYTES), POISON, dtype=torch.uint8, device=DEV)


def _wrap32(t: torch.Tensor) -> torch.Tensor:
    return (((t + 2**31) % 2**32) - 2**31).contiguous()


def _owned_pages(ratio: int, rows, seed: int):
    """The sorted allocation pages the rows live on and their content: random background bytes
    (independent of B) with every row's code bytes and scale word placed in the kernel layout.
    rows = [(owner, view page ids, slots, codes, scale bytes)]."""
    block_kv = 128 // ratio
    owned = torch.unique(torch.cat([OWNER_SHIFT.get(o, 0) + p for o, p, *_ in rows]))
    g = torch.Generator(device=DEV).manual_seed(seed)
    content = torch.randint(
        0, 256, (owned.numel(), block_kv * KEY_BYTES), generator=g, device=DEV
    ).to(torch.uint8)
    cd = torch.arange(CODE_BYTES, device=DEV)
    cs = torch.arange(SCALE_BYTES, device=DEV)
    for owner, page, slot, codes, sf4 in rows:
        i = torch.searchsorted(owned, OWNER_SHIFT.get(owner, 0) + page)
        content[i[:, None], slot[:, None] * CODE_BYTES + cd] = codes
        content[i[:, None], block_kv * CODE_BYTES + slot[:, None] * SCALE_BYTES + cs] = sf4
    return owned, content


def _native_gather(alloc: torch.Tensor, ratio: int, owner: int, d: torch.Tensor, s: torch.Tensor):
    """Native torch: the rows' bytes read from the whole allocation at the view's flat indices."""
    base = OWNER_SHIFT.get(owner, 0) * (128 // ratio) * KEY_BYTES
    flat = alloc.view(-1)
    return (
        flat[(base + d)[:, None] + torch.arange(CODE_BYTES, device=DEV)],
        flat[(base + s)[:, None] + torch.arange(SCALE_BYTES, device=DEV)],
    )


def _pool_unchanged(alloc: torch.Tensor, owned: torch.Tensor, content: torch.Tensor) -> bool:
    """The owned pages still hold `content` and every other byte of the allocation is POISON."""
    if not torch.equal(alloc[owned].view(owned.numel(), -1), content):
        return False
    alloc[owned] = POISON
    word = torch.full((8,), POISON, dtype=torch.uint8, device=DEV).view(torch.int64)
    ok = not bool((alloc.view(-1).view(torch.int64) != word).any())
    alloc[owned] = content.view(owned.numel(), -1, 1, KEY_BYTES)
    return ok


def _gather_all(alloc, ratio, rows, wrap=False, page_fn=None):
    """One call per owner through its view -> [(k bytes [n, 64], scale bytes [n, 4], d, s)]."""
    block_kv = 128 // ratio
    out = []
    for owner, page, slot, codes, _ in rows:
        d, s = _slot_mappings(page if page_fn is None else page_fn(page), slot, block_kv)
        if wrap:
            d, s = _wrap32(d), _wrap32(s)
        k, ks = op(_owner_view(alloc, ratio, owner), d, s, 0, codes.size(0), CODE_BYTES)
        out.append((k.view(torch.uint8), ks.view(torch.uint8).view(-1, SCALE_BYTES), d, s))
    torch.cuda.synchronize()
    return out


def _invariance_rows(ratio: int):
    """Five requests (3 pages + 11 entries, 5, 2 pages + 7, 5000, and 4 whole pages ending on the last
    slot of block B_min - 1), every compressed position 0 .. len-1 as the target's gather lists them,
    over B_min blocks, each used once; page ids 3 x block at ratio 2. Every owner of the ratio holds
    them with its own keys. Returns (B_min, rows)."""
    block_kv = 128 // ratio
    lens = [3 * block_kv + 11, 5, 2 * block_kv + 7, 5000, 4 * block_kv]
    b_min = sum(-(-n // block_kv) for n in lens)
    table = _engine_table(lens, block_kv, b_min, ratio, seed=60 + ratio)
    page, slot = _positions(lens, table, block_kv)
    assert int(page[-1]) == (b_min - 1) * (3 if ratio == 2 else 1) and int(slot[-1]) == block_kv - 1
    rows = []
    for owner in (2, 8, 14) if ratio == 2 else (20,):
        codes, sf4 = _quantize(_index_k(sum(lens), seed=700 + owner))
        rows.append((owner, page, slot, codes, sf4))
    return b_min, rows


@pytest.mark.parametrize("ratio", [1, 2])
def test_pool_size_invariance(ratio):
    """The same rows at the same page ids, gathered through every owner's view of the
    manager layout (ratio 2: one 3B-page allocation, owners 2 / 8 / 14 at 0 / 1 / 2 pages in;
    ratio 1: [B, 128, 1, 68]) at B = B_min, 481, 1493, 151040, 852832 and B_MAX, each an exact-size
    allocation with every unowned page POISON. At every B both outputs are bitwise identical to
    B_min's, to a native-torch gather of the allocation's bytes and to the quantizer's bytes, and
    the allocation is unchanged. Control: at B_MAX the page ids counted from the end of a
    B_OTHER-block pool (a B-dependent addressing) fail the same check."""
    b_min, rows = _invariance_rows(ratio)
    owned, content = _owned_pages(ratio, rows, seed=80 + ratio)
    sizes = (b_min,) + INV_BLOCKS
    assert list(sizes) == sorted(sizes)
    first = None
    for blocks in sizes:
        alloc = _poisoned_alloc(ratio, blocks)
        alloc[owned] = content.view(owned.numel(), -1, 1, KEY_BYTES)
        out = _gather_all(alloc, ratio, rows)
        first = out if first is None else first
        same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(out, first))
        native = True
        quant = True
        for (owner, _, _, codes, sf4), (k, ks, d, s) in zip(rows, out):
            rk, rs = _native_gather(alloc, ratio, owner, d, s)
            native &= torch.equal(k, rk) and torch.equal(ks, rs)
            quant &= torch.equal(k, codes) and torch.equal(ks, sf4)
        kept = _pool_unchanged(alloc, owned, content)
        print(
            f"[invariance r={ratio}] B={blocks}: allocation {tuple(alloc.shape)} ({alloc.numel()} B), "
            f"{sum(r[3].size(0) for r in rows)} rows on {owned.numel()} pages: equal to B_min {same}, "
            f"to the native gather {native}, to the quantizer {quant}, pool unchanged {kept}"
        )
        assert same and native and quant and kept, f"B={blocks}: result depends on the pool size"
        if blocks != sizes[-1]:
            del alloc
    k3 = 3 if ratio == 2 else 1
    wrong = _gather_all(alloc, ratio, rows, page_fn=lambda p: k3 * (B_OTHER - 1) - p)
    frac = min(_row_mismatch(k, ks, c, sf) for (_, _, _, c, sf), (k, ks, _, _) in zip(rows, wrong))
    print(
        f"[invariance control r={ratio}] page ids counted from the end of a {B_OTHER}-block pool on "
        f"B={B_MAX}: {frac:.4f} of rows wrong (lowest over owners)"
    )
    assert frac >= 0.9
    del alloc, first, out, wrong


def _far_rows(ratio: int, blocks: int, seed: int):
    """Every slot of the allocation pages that straddle each FAR_BYTES offset and of the last page
    of block blocks - 1 of every owner, read through the owner whose view holds the page
    (allocation page p belongs to owner shift p % 3 at ratio 2, view page id p - shift = 3 x block)."""
    block_kv = 128 // ratio
    page_bytes = block_kv * KEY_BYTES
    alloc_bytes = (3 * blocks if ratio == 2 else blocks) * page_bytes
    marks = [x for x in FAR_BYTES if x < alloc_bytes]
    pages = [x // page_bytes for x in marks]
    for x, p in zip(marks, pages):
        assert p * page_bytes < x < (p + 1) * page_bytes
    pages += [3 * blocks - 3, 3 * blocks - 2, 3 * blocks - 1] if ratio == 2 else [blocks - 1]
    rows = []
    for owner in (2, 8, 14) if ratio == 2 else (20,):
        shift = OWNER_SHIFT.get(owner, 0)
        mine = [p - shift for p in pages if ratio == 1 or p % 3 == shift]
        if not mine:
            continue
        assert all(m % (3 if ratio == 2 else 1) == 0 for m in mine)
        page = torch.tensor(mine, dtype=torch.int64, device=DEV).repeat_interleave(block_kv)
        slot = torch.arange(block_kv, device=DEV).repeat(len(mine))
        codes, sf4 = _quantize(_index_k(page.numel(), seed=seed + owner))
        rows.append((owner, page, slot, codes, sf4))
    return rows


@pytest.mark.parametrize("ratio", [1, 2])
def test_far_end_at_b_max(ratio):
    """Far-end addressing on the B_MAX allocation (ratio 2: 19,619,931 pages, 85,385,939,712 B;
    ratio 1: 6,539,977 pages, 56,923,959,808 B), every other page POISON: every slot of the pages
    straddling bytes 2^31 .. 2^35 (and 2^36 at ratio 2) and of the last page of block B_MAX - 1 of every owner
    (for owners 14 and 20 the last slot's scale word is the allocation's last 4 bytes). Both whole
    outputs equal a native-torch gather of the allocation and the quantizer's bytes. Control: the
    same offsets wrapped to signed 32 bits fail."""
    rows = _far_rows(ratio, B_MAX, seed=950 + ratio)
    owned, content = _owned_pages(ratio, rows, seed=990 + ratio)
    alloc = _poisoned_alloc(ratio, B_MAX)
    assert alloc.numel() > 2**35
    alloc[owned] = content.view(owned.numel(), -1, 1, KEY_BYTES)
    out = _gather_all(alloc, ratio, rows)
    ok = True
    refs = []  # kept alive through the control, so no freed block holding correct rows is reused
    for (owner, _, _, codes, sf4), (k, ks, d, s) in zip(rows, out):
        refs.append(_native_gather(alloc, ratio, owner, d, s))
        rk, rs = refs[-1]
        ok &= (
            torch.equal(k, rk)
            and torch.equal(ks, rs)
            and torch.equal(k, codes)
            and torch.equal(ks, sf4)
        )
    last = rows[-1]
    base = OWNER_SHIFT.get(last[0], 0) * (128 // ratio) * KEY_BYTES
    tail = int(base + out[-1][3][-1]) + SCALE_BYTES == alloc.numel()
    kept = _pool_unchanged(alloc, owned, content)
    print(
        f"[far end r={ratio}] B={B_MAX}, allocation {alloc.numel()} B, pages {owned.tolist()}: native "
        f"gather and quantizer bytes {ok}, owner {last[0]}'s last scale word ends the allocation {tail}, "
        f"pool unchanged {kept}"
    )
    assert ok and tail and kept
    wrong = _gather_all(alloc, ratio, rows, wrap=True)
    frac = sum(
        _row_mismatch(k, ks, c, sf) * c.size(0)
        for (_, _, _, c, sf), (k, ks, _, _) in zip(rows, wrong)
    )
    n = sum(r[3].size(0) for r in rows)
    frac /= n
    past = sum(int(((d >= 2**31) | (s >= 2**31)).sum()) for _, _, d, s in out) / n
    print(
        f"[far end control r={ratio}] offsets wrapped to signed 32 bits ({past:.4f} of rows past 2^31): "
        f"{frac:.4f} of rows wrong"
    )
    assert frac >= past > 0.5
    del alloc, out, wrong, refs


# ─── many long context requests in one step: rows past one request, up to the step maximum ───
#
# S_r sums every context request of the step, so an engine of the envelope (max_batch_size 256,
# max_seq_len 1,048,576) can pass up to 256 x 1,048,590 = 268,439,040 rows at ratio 1 and
# 134,219,520 at ratio 2; a B_MAX-block pool holds 837,117,056 entries at ratio 1, so every such
# row can address a distinct slot. Calls that large run in tens of milliseconds, so above one
# request's N_max a sample of counts replaces the exhaustive sweep: densely up to ROW_BOUND, then
# coarsely to the step maximum.
#
# ROW_BOUND is 2^25 because that is where the kernel's output offset `token_idx * head_dim` used
# to overflow 32 bits. It is only an interval boundary now: the offsets are computed in 64 bits
# since `static_cast<int64_t>(token_idx)` was added to indexerKCacheGather.cu, and the counts on
# both sides of it are certified below.
ROW_BOUND = 2**25
N_STEP_MAX = {r: 256 * N_MAX[r] for r in (1, 2)}  # max_batch_size context requests of Lm // r rows
_BIJ_A = 1_000_003  # odd prime, coprime with every B_MAX x block_kv slot count


def _bijection_rows(ratio: int, n: int, c: int, far_ends: bool = True):
    """Row i of n reads slot (A * i + c) mod P of the owner's B_MAX-block view, P = B_MAX x block_kv
    (a bijection, so rows < P read distinct slots spread over the whole pool); with far_ends, row 0
    is slot 0 of block 0 and row n - 1 the last slot of block B_MAX - 1. -> (view page ids, slots)."""
    block_kv = 128 // ratio
    k = 3 if ratio == 2 else 1
    sg = (torch.arange(n, dtype=torch.int64, device=DEV) * _BIJ_A + c) % (B_MAX * block_kv)
    page, slot = (sg // block_kv) * k, sg % block_kv
    if far_ends:
        page[0], slot[0] = 0, 0
        page[-1], slot[-1] = (B_MAX - 1) * k, block_kv - 1
    return page, slot


def _native_rows(alloc, ratio, owner, d, s, lo, hi):
    """Native torch: rows [lo, hi) read from the whole allocation at the owner view's flat indices."""
    base = OWNER_SHIFT.get(owner, 0) * (128 // ratio) * KEY_BYTES
    flat = alloc.view(-1)
    return (
        flat[(base + d[lo:hi])[:, None] + torch.arange(CODE_BYTES, device=DEV)],
        flat[(base + s[lo:hi])[:, None] + torch.arange(SCALE_BYTES, device=DEV)],
    )


def _rows_wrong(k8, ks8, alloc, ratio, owner, d, s, lo, hi, chunk=1 << 22):
    """(code rows, scale rows) of output rows [lo, hi) that differ from the native gather."""
    bc = torch.zeros((), dtype=torch.int64, device=DEV)
    bs = torch.zeros((), dtype=torch.int64, device=DEV)
    for a in range(lo, hi, chunk):
        b = min(hi, a + chunk)
        rc, rs = _native_rows(alloc, ratio, owner, d, s, a, b)
        bc += (k8[a:b] != rc).any(1).sum()
        bs += (ks8[a:b] != rs).any(1).sum()
    return int(bc), int(bs)


def _checksum(alloc: torch.Tensor) -> torch.Tensor:
    """Wrapping sum of the allocation's int64 words: a changed word changes it."""
    return alloc.view(-1).view(torch.int64).sum()


def _sampled_counts(ratio: int):
    """Row counts in (N_MAX, ROW_BOUND]: every r = 0, +1, -1 mod 2^16 (one CUDA block per row, so the
    grid has no row tile; 2^16 steps cover every power-of-two boundary from 2^16 up), every sum of
    nc whole requests at one certified max_seq_len (nc = 2..256), and ROW_BOUND - 1, ROW_BOUND."""
    lo, hi = N_MAX[ratio], ROW_BOUND
    counts = {m * 2**16 + e for m in range(lo // 2**16, hi // 2**16 + 1) for e in (-1, 0, 1)}
    counts |= {nc * x for x in ONE_REQUEST_ROWS[ratio] for nc in range(2, 257)}
    counts |= {hi - 1, hi}
    return sorted(c for c in counts if lo < c <= hi)


def _sampled_counts_high(ratio: int):
    """Row counts in (ROW_BOUND, N_STEP_MAX]: the first rows past the old 32-bit overflow point
    (ROW_BOUND + 1, +2, +3, +64, +1023), every multiple of ROW_BOUND and its neighbours (the
    overflow would have recurred at each one), the sums of nc = 32 .. 256 whole longest requests,
    and the step maximum with its predecessor."""
    lo, hi = ROW_BOUND, N_STEP_MAX[ratio]
    counts = {lo + e for e in (1, 2, 3, 64, 1023)}
    counts |= {m * lo + e for m in range(2, hi // lo + 1) for e in (-1, 0, 1)}
    counts |= {nc * N_MAX[ratio] for nc in (32, 33, 64, 65, 128, 129, 200, 255, 256)}
    counts |= {hi - 1, hi}
    return sorted(c for c in counts if lo < c <= hi)


@pytest.mark.parametrize("ratio", [2, 1])
def test_rows_to_2_25_on_b_max_pool(ratio):
    """Every row count above one request's rows, on the B_MAX allocation (owner 14's view at ratio 2,
    owner 20's at ratio 1), random background bytes. (a) The ROW_BOUND = 2^25-row call equals a
    native-torch gather of the allocation, every row; the rows read slots spread over the whole
    view by a bijection, row 0 the first slot and the last row the last slot of block B_MAX - 1,
    whose scale word is the allocation's last 4 bytes. (b) Every sampled count r in
    (N_MAX, ROW_BOUND] (_sampled_counts) returns exactly the first r rows of that call, alternating
    two row orders as in the exhaustive sweep. (c) The same single kernel launches at 2^25 rows.
    The allocation's wrapping int64 word sum is unchanged (the gather only reads)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    owner = 14 if ratio == 2 else 20
    block_kv = 128 // ratio
    alloc = _engine_alloc(ratio, B_MAX, seed=31)
    view = _owner_view(alloc, ratio, owner)
    n = ROW_BOUND
    page, slot = _bijection_rows(ratio, n, c=12_345)
    d_a, s_a = _slot_mappings(page, slot, block_kv)
    del page, slot
    base = OWNER_SHIFT.get(owner, 0) * block_kv * KEY_BYTES
    assert int(base + s_a[-1]) + SCALE_BYTES == alloc.numel()  # the far row ends the allocation
    before = _checksum(alloc)
    k_full, ks_full = op(view, d_a, s_a, 0, n, CODE_BYTES)
    k8, ks8 = k_full.view(torch.uint8), ks_full.view(torch.uint8).view(n, SCALE_BYTES)
    wrong = _rows_wrong(k8, ks8, alloc, ratio, owner, d_a, s_a, 0, n)
    print(
        f"[rows to 2^25 r={ratio}] view {tuple(view.shape)} of {alloc.numel()} B: {n}-row call vs the "
        f"native gather, code/scale rows wrong {wrong}"
    )
    assert wrong == (0, 0)  # (a)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:  # (c)
        op(view, d_a, s_a, 0, n, CODE_BYTES)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    assert len(kernels) == 1 and "indexerKCacheGatherUnifiedKernel" in kernels[0], kernels
    # (b) row words as int64 [n, 8] / int32 [n] keep the comparisons' temporaries small
    shift = n // 2 + 1
    ref = [(k8.view(torch.int64), ks8.view(torch.int32).view(-1))]
    ref.append((ref[0][0].roll(shift, 0).contiguous(), ref[0][1].roll(shift, 0).contiguous()))
    maps = [(d_a, s_a), (d_a.roll(shift).contiguous(), s_a.roll(shift).contiguous())]
    counts = _sampled_counts(ratio)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for i, r in enumerate(counts):
        v = i & 1
        k, ks = op(view, maps[v][0], maps[v][1], 0, r, CODE_BYTES)
        bad += (k.view(torch.uint8).view(torch.int64) != ref[v][0][:r]).any(1).sum()
        bad += (ks.view(torch.int32).view(-1) != ref[v][1][:r]).sum()
        del k, ks
    bad = int(bad)
    unchanged = bool(_checksum(alloc) == before)
    print(
        f"[rows to 2^25 r={ratio}] {len(counts)} sampled counts in ({N_MAX[ratio]}, {n}] "
        f"(min {counts[0]}, max {counts[-1]}): mismatching rows {bad}; one kernel at 2^25; allocation "
        f"checksum unchanged {unchanged}"
    )
    assert bad == 0 and unchanged
    del k_full, ks_full, k8, ks8, ref, maps, d_a, s_a, before


def test_rows_past_2_25_exact_with_a_skipped_band():
    """Rows at index >= 2^25 with a skipped band, the arrangement that measured the pre-fix defect:
    n = 2^26 + 2^24 rows over the B_MAX ratio-1 pool with rows [2^25, 2^26) skipped (-1). Every
    valid row must be its own; nothing writes outside the outputs. Before the kernel's offsets were
    computed in 64 bits, row 2^26 + j landed on output row j (all 2^24 of them) and output rows
    [2^26, n) were never written, while every scale word (offset token_idx * 4) was right; if those
    counts come back, the 64-bit cast is gone from indexerKCacheGather.cu."""
    ratio, owner, block_kv = 1, 20, 128
    alloc = _engine_alloc(ratio, B_MAX, seed=31)
    lo, hi = ROW_BOUND, 2 * ROW_BOUND
    n = hi + (1 << 24)
    page, slot = _bijection_rows(ratio, n, c=777, far_ends=False)
    d, s = _slot_mappings(page, slot, block_kv)
    del page, slot
    d[lo:hi] = -1
    s[lo:hi] = -1
    before = _checksum(alloc)
    k, ks = op(alloc, d, s, 0, n, CODE_BYTES)
    k8, ks8 = k.view(torch.uint8), ks.view(torch.uint8).view(n, SCALE_BYTES)
    low = _rows_wrong(k8, ks8, alloc, ratio, owner, d, s, 0, 1 << 24)
    mid = _rows_wrong(k8, ks8, alloc, ratio, owner, d, s, 1 << 24, lo)
    high = _rows_wrong(k8, ks8, alloc, ratio, owner, d, s, hi, n)
    # the pre-fix aliasing: output row j < 2^24 holding row 2^26 + j's code bytes
    alias = 0
    for a in range(0, 1 << 24, 1 << 22):
        rc, _ = _native_rows(alloc, ratio, owner, d, s, hi + a, hi + a + (1 << 22))
        alias += int((k8[a : a + (1 << 22)] == rc).all(1).sum())
    unchanged = bool(_checksum(alloc) == before)
    print(
        f"[skipped band] n={n}, rows [2^25, 2^26) skipped: code/scale rows wrong in [0, 2^24) {low}, "
        f"[2^24, 2^25) {mid}, [2^26, n) {high}; output rows j < 2^24 holding row 2^26 + j's codes "
        f"{alias} (pre-fix: {1 << 24}); pool unchanged {unchanged}"
    )
    assert low == (0, 0) and mid == (0, 0) and high == (0, 0)
    assert alias <= (1 << 24) // 1000  # the pre-fix aliasing wrote every one of them
    assert unchanged
    del k, ks, k8, ks8, d, s


@pytest.mark.parametrize("ratio", [2, 1])
def test_rows_to_the_step_maximum(ratio):
    """Row counts above ROW_BOUND, to the largest row count an engine of the envelope can pass in
    one call: N_STEP_MAX = max_batch_size 256 context requests of Lm // r rows each (268,439,040 at
    ratio 1, 134,219,520 at ratio 2), on the B_MAX allocation with random background bytes. Rows
    read slots spread over the whole view by a bijection, row 0 the first slot and the last row the
    last slot of block B_MAX - 1. (a) The N_STEP_MAX call equals a native-torch gather of the
    allocation, every row. (b) Every sampled count in (ROW_BOUND, N_STEP_MAX] (_sampled_counts_high)
    returns exactly the first rows of that call, alternating two row orders so a call that left
    rows unwritten in a recycled allocation would show the other order's bytes. (c) One kernel at
    N_STEP_MAX. The allocation's int64 word sum is unchanged."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    owner = 14 if ratio == 2 else 20
    block_kv = 128 // ratio
    alloc = _engine_alloc(ratio, B_MAX, seed=31)
    view = _owner_view(alloc, ratio, owner)
    n = N_STEP_MAX[ratio]
    page, slot = _bijection_rows(ratio, n, c=4_242)
    d_a, s_a = _slot_mappings(page, slot, block_kv)
    del page, slot
    base = OWNER_SHIFT.get(owner, 0) * block_kv * KEY_BYTES
    assert int(base + s_a[-1]) + SCALE_BYTES == alloc.numel()  # the far row ends the allocation
    before = _checksum(alloc)
    k_full, ks_full = op(view, d_a, s_a, 0, n, CODE_BYTES)
    k8, ks8 = k_full.view(torch.uint8), ks_full.view(torch.uint8).view(n, SCALE_BYTES)
    wrong = _rows_wrong(k8, ks8, alloc, ratio, owner, d_a, s_a, 0, n)
    print(
        f"[step max r={ratio}] view {tuple(view.shape)} of {alloc.numel()} B: {n}-row call vs the "
        f"native gather, code/scale rows wrong {wrong}"
    )
    assert wrong == (0, 0)  # (a)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:  # (c)
        op(view, d_a, s_a, 0, n, CODE_BYTES)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    assert len(kernels) == 1 and "indexerKCacheGatherUnifiedKernel" in kernels[0], kernels
    # (b) row words as int64 [n, 8] / int32 [n] keep the comparisons' temporaries small
    shift = n // 2 + 1
    ref = [(k8.view(torch.int64), ks8.view(torch.int32).view(-1))]
    ref.append((ref[0][0].roll(shift, 0).contiguous(), ref[0][1].roll(shift, 0).contiguous()))
    maps = [(d_a, s_a), (d_a.roll(shift).contiguous(), s_a.roll(shift).contiguous())]
    counts = _sampled_counts_high(ratio)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for i, r in enumerate(counts):
        v = i & 1
        k, ks = op(view, maps[v][0], maps[v][1], 0, r, CODE_BYTES)
        bad += (k.view(torch.uint8).view(torch.int64) != ref[v][0][:r]).any(1).sum()
        bad += (ks.view(torch.int32).view(-1) != ref[v][1][:r]).sum()
        del k, ks
    bad = int(bad)
    unchanged = bool(_checksum(alloc) == before)
    print(
        f"[step max r={ratio}] {len(counts)} sampled counts in ({ROW_BOUND}, {n}] "
        f"(min {counts[0]}, max {counts[-1]}): mismatching rows {bad}; one kernel at {n}; allocation "
        f"checksum unchanged {unchanged}"
    )
    assert bad == 0 and unchanged
    del k_full, ks_full, k8, ks8, ref, maps, d_a, s_a
