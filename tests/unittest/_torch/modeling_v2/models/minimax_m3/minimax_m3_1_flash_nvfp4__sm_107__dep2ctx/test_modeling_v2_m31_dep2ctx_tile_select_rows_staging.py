# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_TSEL_ROWS exactness by construction (CPU): an index model of selectPrefillTileKernel's two staging loops
(the family's csrc/m31/selectPrefillTileSm103.cu), thread by thread, for every launch the dispatcher can
pick (256 threads x 8 in flight with and without the stride-class minima, 1024 x 4 from 2600 blocks).

The select phase reads only the staged keys of blocks below the tile's longest row and the 32 stride-class minima per
row, so the row-major table equals the head-major one when, for the same scores, both stagings leave:
- the same tileEnd; every (row, block < tileEnd) key written exactly once, to kNegInfKey past the row's count, else the
  ordered key of the forced score; the same per-row class minima;
and the row-major staging reads no score at or past a row's count, stores four keys at once only 16-byte aligned and
inside the tile, and its class formula 4 (lane % 8) + c is the block's b % 32. The narrow kernel moves only its row
base; the op passes the view's own strides. The GPU test (test_modeling_v2_m31_dep2ctx_tile_select_rows_gpu.py)
checks the compiled kernels bit for bit.
"""

import struct

import pytest
import torch

pytestmark = pytest.mark.cpu_only

K_TILE_Q = 8
K_NEG_INF_KEY = 0xFF800000
FULL = 0xFFFFFFFF
INIT_SCORE, LOCAL_SCORE = 1.0e30, 1.0e29


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ordered_key(bits: int) -> int:
    """orderedKey on the float's bits."""
    return bits if bits & 0x80000000 else (~bits & 0x7FFFFFFF)


INIT_KEY, LOCAL_KEY = ordered_key(_f32_bits(INIT_SCORE)), ordered_key(_f32_bits(LOCAL_SCORE))


def row_pitch(num_blocks: int) -> int:
    return (num_blocks + 31) // 32 * 32 + 4


def _row_state(n_valid, q0, total_q, num_blocks, local_blocks):
    rows = []
    for c in range(K_TILE_Q):
        q = q0 + c
        if q < total_q:
            raw = n_valid[q]
            rows.append((max(0, min(raw, num_blocks)), max(raw - local_blocks, 0)))
        else:
            rows.append((0, 0))
    return rows


def _forced_key(bits_at, row, b, rows, init_blocks):
    """The key both stagings store for (row, b < tileEnd)."""
    row_end, local_start = rows[row]
    if b >= row_end:
        return K_NEG_INF_KEY, False
    bits = bits_at(row, b)
    if b < init_blocks:
        bits = _f32_bits(INIT_SCORE)
    if b >= local_start:
        bits = _f32_bits(LOCAL_SCORE)
    return ordered_key(bits), True


def stage_head_major(bits_at, rows, nt, unroll, ilp, num_blocks, init_blocks):
    """The kRowMajor == false staging: thread tid holds rows 4 jh .. 4 jh + 3 of block b = tid / 2 + (nt / 2) i."""
    pitch = row_pitch(num_blocks)
    keys, writes = {}, {}
    cls_min = [FULL] * (K_TILE_Q * 32)
    tile_end = max(end for end, _ in rows)
    per_iter = nt // 2
    for tid in range(nt):
        jh = tid % 2
        class_min = [FULL] * 4
        b0 = tid // 2
        while b0 < tile_end:
            for u in range(unroll):
                b = b0 + u * per_iter
                if b < tile_end:
                    for c in range(4):
                        r = 4 * jh + c
                        key, valid = _forced_key(bits_at, r, b, rows, init_blocks)
                        if valid and ilp:
                            class_min[c] = min(class_min[c], key)
                        keys[r * pitch + b] = key
                        writes[(r, b)] = writes.get((r, b), 0) + 1
            b0 += per_iter * unroll
        if ilp:
            cls = (tid // 2) % 32
            for c in range(4):
                if class_min[c] != FULL:
                    i = (4 * jh + c) * 32 + cls
                    cls_min[i] = min(cls_min[i], class_min[c])
    return tile_end, keys, writes, cls_min


def stage_row_major(
    bits_at, rows, nt, unroll, ilp, num_blocks, init_blocks, row_aligned, mutate=None
):
    """The kRowMajor == true staging, as written: warp w stages row w % 8, vectors v = sub * 32 + lane + 32 kWpr i."""
    pitch = row_pitch(num_blocks)
    keys, writes, loads, vec_stores = {}, {}, [], []
    cls_min = [FULL] * (K_TILE_Q * 32)
    warps_per_row = nt // 32 // K_TILE_Q
    vec_per_iter = warps_per_row * 32
    tile_end = max(end for end, _ in rows)  # every thread computes it from nValid
    n_vec = (tile_end + 3) // 4
    for tid in range(nt):
        warp, lane = tid // 32, tid % 32
        r, sub = warp % K_TILE_Q, warp // K_TILE_Q
        row_end = rows[r][0]
        vec = row_aligned[r]
        class_min = [FULL] * 4
        v0 = sub * 32 + lane
        while v0 < n_vec:
            for u in range(unroll):  # loads
                b = 4 * (v0 + u * vec_per_iter)
                if b < row_end:
                    if vec and b + 3 < row_end:
                        loads.extend((r, b + c, "vec") for c in range(4))
                    else:
                        loads.extend((r, b + c, "scalar") for c in range(4) if b + c < row_end)
            for u in range(unroll):  # keys
                b = 4 * (v0 + u * vec_per_iter)
                if b < tile_end:
                    k = []
                    for c in range(4):
                        key, valid = _forced_key(bits_at, r, b + c, rows, init_blocks)
                        if valid and ilp:
                            class_min[c] = min(class_min[c], key)
                        k.append(key)
                    if b + 3 < tile_end:
                        vec_stores.append(r * pitch + b)
                        for c in range(4):
                            keys[r * pitch + b + c] = k[c]
                            writes[(r, b + c)] = writes.get((r, b + c), 0) + 1
                    else:
                        for c in range(4):
                            if b + c < tile_end:
                                keys[r * pitch + b + c] = k[c]
                                writes[(r, b + c)] = writes.get((r, b + c), 0) + 1
            v0 += vec_per_iter * unroll
        if ilp:
            cls = 4 * (lane % 8) if mutate != "class" else 4 * (lane % 4)
            for c in range(4):
                if class_min[c] != FULL:
                    i = r * 32 + cls + c
                    cls_min[i] = min(cls_min[i], class_min[c])
    return tile_end, keys, writes, cls_min, loads, vec_stores


# The launches the dispatcher picks: v2-only (256 x 8, scalar scans), v4 (256 x 8, minima), wide (1024 x 4, minima).
LAUNCHES = [(256, 8, False), (256, 8, True), (1024, 4, True)]


def _bits_field(seed, rows, blocks):
    """Adversarial score bits: ties, +-0, both NaN signs, +-inf, finite values."""
    g = torch.Generator().manual_seed(seed)
    vals = torch.randint(-3, 4, (rows, blocks), generator=g).float().view(torch.int32)
    special = torch.tensor(
        [0x7F800000, -0x00800000, 0x7FC00000, -0x00400000, 0, -0x80000000], dtype=torch.int32
    )
    pick = torch.randint(0, 12, (rows, blocks), generator=g)
    bits = torch.where(pick < 6, special[pick.clamp(max=5)], vals)
    return (bits.to(torch.int64) & 0xFFFFFFFF).tolist()


def _check(
    n_valid, q0, total_q, num_blocks, init_blocks, local_blocks, aligned, seed, launches=LAUNCHES
):
    rows = _row_state(n_valid, q0, total_q, num_blocks, local_blocks)
    field = _bits_field(seed, K_TILE_Q, max(num_blocks, 1))
    bits_at = lambda r, b: field[r][b]  # noqa: E731
    pitch = row_pitch(num_blocks)
    for nt, unroll, ilp in launches:
        hm = stage_head_major(bits_at, rows, nt, unroll, ilp, num_blocks, init_blocks)
        rm = stage_row_major(bits_at, rows, nt, unroll, ilp, num_blocks, init_blocks, aligned)
        tile_end = hm[0]
        assert rm[0] == tile_end
        # Every (row, block < tileEnd) written exactly once by each staging, nothing else.
        want_cells = {(r, b) for r in range(K_TILE_Q) for b in range(tile_end)}
        assert set(hm[2]) == want_cells and set(rm[2]) == want_cells
        assert all(n == 1 for n in hm[2].values()) and all(n == 1 for n in rm[2].values())
        assert rm[1] == hm[1], "staged keys differ"
        if ilp:
            assert rm[3] == hm[3], "stride-class minima differ"
            direct = [FULL] * (K_TILE_Q * 32)
            for r in range(K_TILE_Q):
                for b in range(rows[r][0]):
                    key, _ = _forced_key(bits_at, r, b, rows, init_blocks)
                    direct[r * 32 + b % 32] = min(direct[r * 32 + b % 32], key)
            assert rm[3] == direct
        # Loads stay below the row's count; vector loads only on an aligned row.
        assert all(b < rows[r][0] for r, b, _ in rm[4])
        assert all(aligned[r] for r, _, kind in rm[4] if kind == "vec")
        assert all(addr % 4 == 0 and (addr % pitch) + 3 < tile_end for addr in rm[5])
    return rows


@pytest.mark.parametrize(
    "tile_end", [0, 1, 3, 4, 5, 16, 17, 31, 33, 127, 128, 129, 2599, 2600, 2601, 5856]
)
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 3)])
def test_row_major_staging_leaves_the_head_major_shared_memory(tile_end, init_blocks, local_blocks):
    num_blocks = max(tile_end, 1)
    # Rows end at the tile end, short of it, at 0, and past the table (clamped); the last two rows past totalQ.
    raw = [
        tile_end,
        max(tile_end - 1, 0),
        tile_end // 2,
        0,
        tile_end + 3,
        max(tile_end - 5, 0),
        9,
        9,
    ]
    total_q = 6
    aligned = [True, False, True, False, True, True, True, True]
    launches = LAUNCHES if tile_end < 2600 else [(1024, 4, True)]
    if tile_end > 600:
        launches = [launches[-1]]  # keep the model fast; the 256-thread loops are covered below 600
    _check(
        raw,
        0,
        total_q,
        num_blocks,
        init_blocks,
        local_blocks,
        aligned,
        seed=tile_end,
        launches=launches,
    )


def test_class_formula_is_the_block_stride_class():
    for warps_per_row in (1, 4):
        for lane in range(32):
            for sub in range(warps_per_row):
                for i in range(64):
                    v = sub * 32 + lane + 32 * warps_per_row * i
                    for c in range(4):
                        assert (4 * v + c) % 32 == 4 * (lane % 8) + c


def test_a_wrong_class_formula_is_caught():
    """Negative control: the model check fails when the row-major class index is wrong."""
    rows = _row_state([300] * 8, 0, 8, 300, 1)
    field = _bits_field(5, 8, 300)
    bits_at = lambda r, b: field[r][b]  # noqa: E731
    hm = stage_head_major(bits_at, rows, 256, 8, True, 300, 0)
    rm = stage_row_major(bits_at, rows, 256, 8, True, 300, 0, [True] * 8, mutate="class")
    assert rm[1] == hm[1] and rm[3] != hm[3]


def test_op_strides_address_the_view():
    """The rows op passes headStride = stride(0) and queryStride = stride(2) of the [H, pages, T] view of the scorer's
    [T * H, pitch] buffer (blocks contiguous); the kernels read element (h, b, q) at h * hs + b + q * qs."""
    total_q, heads, pitch, cols = 37, 4, 44, 42
    buf = torch.randn(total_q * heads, pitch)[:, :cols]
    view = buf.view(total_q, heads, -1).permute(1, 2, 0)
    assert view.stride(1) == 1
    hs, qs = view.stride(0), view.stride(2)
    flat = buf.as_strided((buf.untyped_storage().nbytes() // 4,), (1,), 0)
    h = torch.arange(heads).view(-1, 1, 1)
    b = torch.arange(cols).view(1, -1, 1)
    q = torch.arange(total_q).view(1, 1, -1)
    assert torch.equal(flat[h * hs + b + q * qs], view)
    # Row t * H + h of the buffer is (token t, head h): the scorer's row order.
    assert torch.equal(view[2, :, 5], buf[5 * heads + 2])
    # 16-byte rows when the buffer base is: pitch % 4 == 0 keeps every (h, q) row aligned.
    assert (hs % 4, qs % 4) == (0, 0)
