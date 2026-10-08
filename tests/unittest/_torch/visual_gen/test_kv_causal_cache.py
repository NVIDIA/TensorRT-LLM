# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CausalKVCacheManager: pages, the fixed region, the rolling window, and the
per-block rows attention reads."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 64
DEVICE = torch.device("cuda")
DTYPE = torch.float16  # integer stamps up to 2048 stay exact; bf16 loses them above 256


def make_cache_for(tpb, *, max_pin_tokens, window_tokens, max_staged_tokens, causal_block_sizes):
    """A cache whose pool is sized for exactly this geometry. The geometry rides on
    the object so ``open_cache`` can open it (tests only)."""
    geometry = dict(
        window_tokens=window_tokens,
        max_staged_tokens=max_staged_tokens,
        causal_block_sizes=causal_block_sizes,
    )
    cache = CausalKVCacheManager(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=tpb,
        pool_tokens=CausalKVCacheManager.pool_tokens_for(
            tokens_per_page=tpb, pin_tokens=max_pin_tokens, **geometry
        ),
    )
    cache.test_geometry = geometry
    cache.test_max_pin = max_pin_tokens
    return cache


def make_cache(tokens_per_page: int):
    """Geometry that scales with the page size so every test exercises partial
    pages, stale tokens and rotation: staged tokens are a page plus 8, window two pages."""
    tpb = tokens_per_page
    return make_cache_for(
        tpb,
        max_pin_tokens=tpb + 8,
        window_tokens=2 * tpb,
        max_staged_tokens=tpb + 8,
        causal_block_sizes=(tpb + 8, (tpb + 8) // 4),  # all staged tokens, and four blocks
    )


def open_cache(cache, pin_tokens=0):
    cache.open(pin_tokens=pin_tokens, **cache.test_geometry)


@pytest.fixture(params=[32, 128], ids=["tpb32", "tpb128"])
def cache(request):
    mgr = make_cache(request.param)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def rand_kv(n):
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return k, torch.randn_like(k)


LAYER_STAMP = 1024  # stamps of layer l are offset by l * LAYER_STAMP; exact in fp16 below 2048


def stamped_kv(positions, layer=0):
    """K/V whose every element is the token's logical position plus a per-layer offset:
    reading a slot back tells which token, and which layer's write, sits there."""
    stamp = (positions + layer * LAYER_STAMP).to(DTYPE)
    stamp = stamp[:, None, None].expand(-1, NUM_KV_HEADS, HEAD_DIM).contiguous()
    return stamp, -stamp


def layer_kv(n):
    """Different random K/V for every layer: ``[(k, v)] * NUM_LAYERS``."""
    return [rand_kv(n) for _ in range(NUM_LAYERS)]


def read_kv(cache, layer, positions):
    """Gather ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions`` straight from the pool."""
    buf = cache.kv_buffer(layer)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_page]
    slot = positions % cache.tokens_per_page
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def write_kv_reference(cache, layer, positions, k, v):
    """Scatter one token at a time; the slow, obviously-correct write ``write_range`` must match."""
    buf = cache.kv_buffer(layer)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_page]
    slot = positions % cache.tokens_per_page
    buf[page, 0, :, slot, :] = k
    buf[page, 1, :, slot, :] = v


def open_with_fixed(cache, fixed_len):
    """Open pinning ``fixed_len`` tokens, write that many random tokens at position 0,
    different in every layer, and commit them. Returns ``[(k, v)]`` per layer."""
    open_cache(cache, pin_tokens=fixed_len)
    per_layer = layer_kv(fixed_len)
    for layer, (k, v) in enumerate(per_layer):
        cache.write_range(layer, 0, k, v)
    if fixed_len:
        cache.commit(fixed_len)
    return per_layer


def row_keys(cache, layer, block_size, i):
    """The K stamps block ``i``'s row presents to the kernel, in row order."""
    buf = cache.kv_buffer(layer)
    rows = cache.page_table(block_size)
    _, kv_len = cache.causal_block_lengths(block_size)
    n = int(kv_len[i])
    pos = torch.arange(n, device=DEVICE)
    page = rows[i].long()[pos // cache.tokens_per_page]
    slot = pos % cache.tokens_per_page
    return buf[page, 0, 0, slot, 0].float()


def test_geometry_holds_fixed_window_stale_and_chunk(cache):
    tpb = cache.tokens_per_page
    assert cache.page_view_scale == NUM_LAYERS, "layers share a slot; one layer's view is strided"
    open_cache(cache, pin_tokens=cache.test_max_pin)
    tokens = cache.test_max_pin + cache.window_tokens + cache.max_staged_tokens
    assert cache.num_pages == -(-tokens // tpb) + 1
    assert cache.capacity == cache.num_pages * tpb
    # Worst case resident: full fixed region, window plus a page of stale, a chunk.
    assert cache.capacity >= tokens + tpb - 1


def test_open_backs_every_page_once_and_publishes_the_table(cache):
    open_cache(cache)
    assert cache.fixed_tokens == cache.history_tokens == cache.staging_offset == 0
    table = cache.block_table()
    assert len(table) == cache.num_pages
    assert len(set(table)) == cache.num_pages, "pages must be distinct"
    assert min(table) >= 0
    torch.testing.assert_close(
        cache.table.cpu(), torch.tensor(table, dtype=torch.int32) * cache.page_view_scale
    )
    with pytest.raises(RuntimeError):
        open_cache(cache)
    cache.close()
    cache.close()


def test_reopen_keeps_device_state_in_place(cache):
    """close() then open() refreshes the kernel-facing tensors where they are, so a
    forward captured in a CUDA graph before close() replays on live memory."""
    open_cache(cache)
    per_layer = layer_kv(cache.max_staged_tokens)
    for layer, (k, v) in enumerate(per_layer):
        cache.write_range(layer, 0, k, v)
    cache.commit(cache.max_staged_tokens)
    size = cache.causal_block_sizes[-1]
    before = [
        t.data_ptr()
        for t in (
            cache.table,
            cache.page_table(size),
            cache.causal_block_lengths(size)[1],
            cache._staged_slots,
            cache._layout(size).own_slots,
        )
    ]
    cache.close()
    with pytest.raises(RuntimeError):
        cache.table
    open_cache(cache)
    after = [
        t.data_ptr()
        for t in (
            cache.table,
            cache.page_table(size),
            cache.causal_block_lengths(size)[1],
            cache._staged_slots,
            cache._layout(size).own_slots,
        )
    ]
    assert before == after
    assert (cache.fixed_tokens, cache.history_tokens, cache.staging_offset) == (0, 0, 0)
    assert cache.table.unique().numel() == cache.num_pages, "fresh pages, no duplicates"
    for layer, (k, v) in enumerate(per_layer):
        cache.write_range(layer, 0, k, v)
    for layer, (k, v) in enumerate(per_layer):
        k_back, v_back = read_kv(cache, layer, torch.arange(cache.max_staged_tokens, device=DEVICE))
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)


def test_pinned_tokens_are_the_first_committed_and_never_evicted(cache):
    """``open(pin_tokens)`` pins the first tokens committed, whatever they are: a
    prompt in one commit, then generated tokens up to the pin size inside a later
    commit, which splits it. They survive every rotation, in every layer."""
    tpb, chunk = cache.tokens_per_page, cache.test_geometry["max_staged_tokens"]
    unopened = make_cache(tpb)
    try:
        with pytest.raises(RuntimeError):
            unopened.table
    finally:
        unopened.shutdown()
    with pytest.raises(ValueError, match="pool holds"):
        open_cache(cache, pin_tokens=cache.pool_tokens)
    prompt, sink = cache.test_max_pin - 8, 8
    open_cache(cache, pin_tokens=prompt + sink)
    prompts = layer_kv(prompt)
    for layer, (pk, pv) in enumerate(prompts):
        cache.write_range(layer, 0, pk, pv)
    cache.commit(prompt)
    assert (cache.fixed_tokens, cache.history_tokens, cache.staging_offset) == (prompt, 0, prompt)

    # The first chunk: its first `sink` tokens complete the pinned part.
    firsts = layer_kv(chunk)
    for layer, (hk, hv) in enumerate(firsts):
        cache.write_range(layer, cache.staging_offset, hk, hv)
    cache.commit(cache.max_staged_tokens)
    assert (cache.fixed_tokens, cache.history_tokens) == (prompt + sink, chunk - sink)
    for _ in range(3 * cache.capacity // chunk):  # cycle the pool several times
        for layer, (kk, vv) in enumerate(layer_kv(chunk)):
            cache.write_range(layer, cache.staging_offset, kk, vv)
        cache.commit(cache.max_staged_tokens)
        assert cache.fixed_tokens == prompt + sink
    for layer in range(NUM_LAYERS):
        (pk, pv), (hk, hv) = prompts[layer], firsts[layer]
        k_back, v_back = read_kv(cache, layer, torch.arange(prompt + sink, device=DEVICE))
        torch.testing.assert_close(k_back, torch.cat([pk, hk[:sink]]))
        torch.testing.assert_close(v_back, torch.cat([pv, hv[:sink]]))


def test_commit_beyond_a_chunk_only_into_the_pinned_part(cache):
    chunk = cache.test_geometry["max_staged_tokens"]
    open_cache(cache, pin_tokens=cache.test_max_pin)
    with pytest.raises(ValueError):
        cache.commit(cache.test_max_pin + chunk + 1)  # more than a chunk past the pin
    cache.commit(cache.test_max_pin + chunk)  # pinned part plus exactly one chunk
    assert (cache.fixed_tokens, cache.history_tokens) == (cache.test_max_pin, chunk)


def test_write_range_matches_indexed_write(cache):
    """The run-based fast write lands bytes exactly where the per-token write does."""
    open_with_fixed(cache, 20)
    tpb = cache.tokens_per_page
    for start, n in ((0, 20), (20, 40), (7, 3), (tpb - 1, 2 * tpb + 5), (5, cache.capacity - 5)):
        k, v = rand_kv(n)
        positions = torch.arange(start, start + n, device=DEVICE)
        for layer in range(NUM_LAYERS):
            write_kv_reference(cache, layer, positions, -k, -v)  # poison first
            cache.write_range(layer, start, k, v)
            k_back, v_back = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k)
            torch.testing.assert_close(v_back, v)
            write_kv_reference(cache, layer, positions, k, v)  # and the reference agrees
            k_back, v_back = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k)
            torch.testing.assert_close(v_back, v)

    # Independently: the raw pool at (table[t // tpb], t % tpb) holds token t.
    layer = NUM_LAYERS - 1
    buf = cache.kv_buffer(layer)
    assert buf.shape[1:] == (2, NUM_KV_HEADS, tpb, HEAD_DIM)
    table = cache.block_table()
    positions = torch.arange(cache.capacity, device=DEVICE)
    k_back, _ = read_kv(cache, layer, positions)
    for t in (0, 19, 20, tpb - 1, tpb, cache.capacity - 1):
        page, slot = table[t // tpb] * cache.page_view_scale, t % tpb
        torch.testing.assert_close(buf[page, 0, :, slot, :], k_back[t])

    with pytest.raises(ValueError):
        cache.write_range(0, cache.capacity - 1, *rand_kv(2))


def test_zero_values_clears_v_of_the_range_only(cache):
    """zero_values clears V of exactly the requested tokens in one layer: partial pages
    at both ends, whole pages in between, and nothing else (K, the neighbours, the
    other layers) changes."""
    open_with_fixed(cache, 20)
    tpb = cache.tokens_per_page
    positions = torch.arange(cache.capacity, device=DEVICE)
    for start, n in ((7, 3), (tpb - 1, 2 * tpb + 5), (tpb, tpb), (20, 1)):
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, 0, *stamped_kv(positions, layer))
        cache.zero_values(1, start, n)
        inside = (positions >= start) & (positions < start + n)
        for layer in range(NUM_LAYERS):
            k_want, v_want = stamped_kv(positions, layer)
            if layer == 1:
                v_want[inside] = 0
            k_back, v_back = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k_want, rtol=0, atol=0)
            torch.testing.assert_close(v_back, v_want, rtol=0, atol=0)


def row_stamps(cache, layer, size, i, plane):
    """Stamps block ``i``'s row presents to the kernel for its cached tokens, in row
    order, from the K (0) or V (1) plane."""
    buf = cache.kv_buffer(layer)
    rows = cache.page_table(size)
    n = cache.cached_tokens(size)[i]
    pos = torch.arange(n, device=DEVICE)
    page = rows[i].long()[pos // cache.tokens_per_page]
    slot = pos % cache.tokens_per_page
    return buf[page, plane, 0, slot, 0].float()


def test_edits_to_committed_tokens_reach_the_private_copies(cache):
    """write_range and zero_values on committed tokens update the private copies the
    block rows read, not only the home slots. Chunk tokens and other layers stay."""
    chunk = cache.test_geometry["max_staged_tokens"]
    sizes = (chunk, chunk // 4)
    open_with_fixed(cache, 20)
    for _ in range(3):  # past the window, so both the fixed tail and the window edge are copies
        cache.commit(cache.max_staged_tokens)
    assert cache.history_tokens > cache.window_tokens
    past = cache.staging_offset
    positions = torch.arange(past + chunk, device=DEVICE)
    for layer in range(NUM_LAYERS):
        k, v = stamped_kv(positions, layer)
        cache.write_range(layer, 0, k[:past], v[:past])
        cache.write_staged(layer, k[past:], v[past:])
    cache.commit(cache.max_staged_tokens)  # the copies are rebuilt here: the known-good path
    past = cache.staging_offset
    for layer in range(NUM_LAYERS):  # the next chunk, so the earlier-block copies hold real tokens
        k, v = stamped_kv(torch.arange(past, past + chunk, device=DEVICE), layer)
        for size in sizes:
            cache.write_staged(layer, -k, -v, size)  # negative K stamps mark staged tokens
    before = {
        (layer, size, i, plane): row_stamps(cache, layer, size, i, plane)
        for layer in range(NUM_LAYERS)
        for size in sizes
        for i in range(chunk // size)
        for plane in (0, 1)
    }

    def committed(layer, size, i):  # row entries holding committed tokens: positive K stamps
        return before[(layer, size, i, 0)] > 0

    edited = 1
    k_now, v_now = read_kv(cache, edited, torch.arange(past, device=DEVICE))
    cache.write_range(edited, 0, -k_now, -v_now)
    for (layer, size, i, plane), old in before.items():
        new = row_stamps(cache, layer, size, i, plane)
        want = old.clone()
        if layer == edited:
            want[committed(layer, size, i)] *= -1
        torch.testing.assert_close(new, want, rtol=0, atol=0)

    cache.zero_values(edited, 0, past)
    for (layer, size, i, plane), old in before.items():
        new = row_stamps(cache, layer, size, i, plane)
        want = old.clone()
        if layer == edited:
            hit = committed(layer, size, i)
            want[hit] = 0 if plane == 1 else -want[hit]
        torch.testing.assert_close(new, want, rtol=0, atol=0)


def test_eviction_keeps_the_window_and_the_fixed_region(cache):
    """Content check across many chunks with a fixed region that shares a page with the history."""
    fixed = 13
    prompts = open_with_fixed(cache, fixed)
    tpb, chunk, window = cache.tokens_per_page, cache.max_staged_tokens, cache.window_tokens
    allocated = sorted(cache.block_table())
    written = []  # one stamp per committed generator token, oldest first
    saw_stale = saw_rotation = False

    for c in range(1, 12):
        for layer in range(NUM_LAYERS):
            stamp = torch.full(
                (chunk, NUM_KV_HEADS, HEAD_DIM), float(c + 64 * layer), device=DEVICE, dtype=DTYPE
            )
            cache.write_range(layer, cache.staging_offset, stamp, -stamp)
        before = cache.block_table()
        cache.commit(cache.max_staged_tokens)
        written.extend([c] * chunk)
        saw_rotation |= cache.block_table() != before

        # Whole-page eviction leaves fewer than one page of stale tokens resident.
        assert cache.history_tokens <= window + tpb - 1
        stale = max(0, cache.history_tokens - window)
        assert stale < tpb
        saw_stale |= stale > 0
        assert cache.history_tokens - stale == min(len(written), window)
        assert cache.staging_offset == fixed + cache.history_tokens

        for layer in range(NUM_LAYERS):
            hist = torch.arange(fixed, fixed + cache.history_tokens, device=DEVICE)
            k_back, v_back = read_kv(cache, layer, hist)
            expect = torch.tensor(written[-cache.history_tokens :], device=DEVICE, dtype=DTYPE)
            expect = expect + 64 * layer
            torch.testing.assert_close(k_back[:, 0, 0], expect)
            torch.testing.assert_close(v_back[:, 0, 0], -expect)
            # The fixed region survived every rotation of the page it shares with the history.
            k_p, v_p = read_kv(cache, layer, torch.arange(fixed, device=DEVICE))
            torch.testing.assert_close(k_p, prompts[layer][0])
            torch.testing.assert_close(v_p, prompts[layer][1])

        # Eviction recycles pages; it never allocates or frees any.
        table = cache.block_table()
        # The host copy of the table rotates in lockstep with the device one.
        assert table == (cache.table // cache.page_view_scale).tolist()
        assert sorted(table) == allocated
        assert len(set(table)) == len(table)

    assert saw_stale, "test geometry should produce stale tokens"
    assert saw_rotation, "test geometry should rotate the table"


def check_rows_present_exactly_the_window(cache, fixed, size, steps=8, expect_stale=True):
    """Every block's row holds the fixed region, exactly ``window_tokens`` of history
    before the block (or fewer while the window fills), the earlier blocks it may see
    and itself; nothing stale, nothing later, nothing twice. Checked by stamping
    every token with its position and layer."""
    chunk, window = cache.test_geometry["max_staged_tokens"], cache.test_geometry["window_tokens"]
    num_blocks = chunk // size
    open_cache(cache, pin_tokens=fixed)
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, 0, *stamped_kv(torch.arange(fixed, device=DEVICE), layer))
    cache.commit(fixed)

    written: list = []  # stamp of every committed token, oldest first

    def expected_stamps(start, end, layer):
        # Fixed stamps are their positions; history stamps are the positions the
        # tokens had when written (eviction shifted them since); chunk stamps are
        # current positions. Resident history is the tail of what was committed.
        past = cache.staging_offset
        win_start = max(fixed, start - window)
        resident = written[len(written) - cache.history_tokens :]
        stamps = (
            list(range(fixed))
            + resident[win_start - fixed :]
            + list(range(max(past, win_start), end))
        )
        return sorted(t + layer * LAYER_STAMP for t in stamps)

    def check_rows(block, count, step):
        cached = cache.cached_tokens(block)
        _, kv_len = cache.causal_block_lengths(block)
        for i in range(count):
            start = cache.staging_offset + i * block
            for layer in range(NUM_LAYERS):
                expected = expected_stamps(start, start + block, layer)
                if layer == 0:
                    assert cached[i] == len(expected) - block, f"step {step} block {i}"
                    assert int(kv_len[i]) == len(expected), f"step {step} block {i}"
                got = sorted(row_keys(cache, layer, block, i).tolist())
                assert got == expected, f"step {step} block {i} layer {layer}"

    saw_stale = False
    for step in range(steps):  # from empty history through saturation and several rotations
        past = cache.staging_offset
        positions = torch.arange(past, past + chunk, device=DEVICE)
        for layer in range(NUM_LAYERS):
            cache.write_staged(layer, *stamped_kv(positions, layer), size)
        assert cache.page_table(size).shape[0] == num_blocks
        check_rows(size, num_blocks, step)
        # A one-block forward reads the whole chunk with the same window.
        for layer in range(NUM_LAYERS):
            cache.write_staged(layer, *stamped_kv(positions, layer), chunk)
        check_rows(chunk, 1, step)
        # A shorter forward of the same block size uses the leading blocks only.
        if num_blocks > 1:
            half = (num_blocks // 2) * size
            for layer in range(NUM_LAYERS):
                k, v = stamped_kv(positions[:half], layer)
                cache.write_staged(layer, k, v, size)
            check_rows(size, num_blocks // 2, step)
        # The shared pages hold the chunk too, for later blocks and chunks.
        for layer in range(NUM_LAYERS):
            k_back, _ = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, stamped_kv(positions, layer)[0])
        saw_stale |= cache.history_tokens > window
        written.extend(range(past, past + chunk))
        cache.commit(cache.max_staged_tokens)
    assert saw_stale or not expect_stale, "test geometry should hold stale tokens at some step"


def test_block_rows_present_exactly_the_window(cache):
    chunk = cache.test_geometry["max_staged_tokens"]
    check_rows_present_exactly_the_window(cache, fixed=13, size=chunk // 4)
    with pytest.raises(ValueError, match="not declared"):
        cache.causal_block_lengths(chunk // 2)
    with pytest.raises(ValueError, match="not declared"):
        cache.write_staged(0, *rand_kv(chunk // 2))


@pytest.mark.parametrize(
    "tpb, chunk, size, window, stale",
    [
        (16, 24, 4, 20, True),  # window starts on the page holding past: chunk on two pages
        (16, 32, 8, 12, True),  # window shorter than the chunk: starts inside earlier blocks
        (32, 40, 40, 8, True),  # window shorter than a page
        (32, 128, 64, 96, False),  # two blocks of two whole pages; eviction is page-aligned
    ],
)
def test_block_rows_in_odd_geometries(tpb, chunk, size, window, stale):
    cache = make_cache_for(
        tpb,
        max_pin_tokens=16,
        window_tokens=window,
        max_staged_tokens=chunk,
        causal_block_sizes=tuple(dict.fromkeys((chunk, size))),
    )
    try:
        check_rows_present_exactly_the_window(
            cache, fixed=13, size=size, steps=10, expect_stale=stale
        )
    finally:
        cache.shutdown()


def test_causal_block_lengths_follow_the_exact_window(cache):
    open_with_fixed(cache, 3)
    chunk, window = cache.max_staged_tokens, cache.window_tokens
    q, kv = cache.causal_block_lengths(chunk)
    assert q.dtype == kv.dtype == torch.int32
    assert q.tolist() == [chunk] and kv.tolist() == [3 + chunk]
    assert cache.max_causal_blocks == 4

    size = chunk // 4

    def exact(history):  # block i sees fixed + min(window, history + i*size) + itself
        return [3 + min(window, history + i * size) + size for i in range(4)]

    q4, kv4 = cache.causal_block_lengths(size)
    assert q4.tolist() == [size] * 4
    assert kv4.tolist() == exact(0)
    assert cache.causal_block_lengths(size)[1] is kv4, "one persistent pair per block size"

    cache.commit(cache.max_staged_tokens)
    # Refreshed in place by commit(), without anyone asking for them again.
    assert kv.tolist() == [3 + min(window, chunk) + chunk]
    assert kv4.tolist() == exact(chunk)

    for _ in range(4):
        cache.commit(cache.max_staged_tokens)
    assert cache.history_tokens > window
    # Saturated: exactly the window before each block, however many stale tokens are resident.
    assert kv.tolist() == [3 + window + chunk]
    assert kv4.tolist() == [3 + window + size] * 4

    with pytest.raises(ValueError, match="not declared"):
        cache.causal_block_lengths(7)


def test_copy_batch_block_offsets_encodes_the_block_rows(cache):
    open_with_fixed(cache, 3)
    for _ in range(5):
        cache.commit(cache.max_staged_tokens)
    chunk = cache.max_staged_tokens
    num_blocks, size = 3, chunk // 4  # three of the four blocks: a shorter forward
    dst = torch.full((1, 4, 2, cache.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):  # the block size must be declared first
        cache.copy_batch_block_offsets(
            dst, cache.request_ids(num_blocks), 1, num_blocks, num_blocks
        )
    cache.set_causal_block_size(size)
    cache.copy_batch_block_offsets(dst, cache.request_ids(num_blocks), 1, num_blocks, num_blocks)
    torch.cuda.synchronize()
    rows = cache.page_table(size)
    n = rows.shape[1]
    kv_offset = int(cache.kv_offset[0])
    for i in range(num_blocks):
        torch.testing.assert_close(dst[0, i, 0, :n], rows[i] * cache.kv_factor)
        torch.testing.assert_close(dst[0, i, 1, :n], rows[i] * cache.kv_factor + kv_offset)
        assert torch.count_nonzero(dst[0, i, :, n:]) == 0, "unused entries are the safe page 0"
    assert (dst[0, 3] == -7).all(), "rows beyond num_seqs are untouched"

    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [1], 1, 1, 1)
    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, cache.request_ids(1) + [1], 1, 2, 2)
    with pytest.raises(ValueError):  # more blocks than the declared size cuts the chunk into
        cache.copy_batch_block_offsets(dst, cache.request_ids(5), 1, 5, 5)


def test_write_staged_matches_write_range(cache):
    """The device-indexed chunk write lands exactly where the host-sliced write does."""
    open_with_fixed(cache, 13)
    for _ in range(3):
        cache.commit(cache.max_staged_tokens)  # move past off a page boundary and rotate once
    chunk = cache.max_staged_tokens
    positions = torch.arange(cache.staging_offset, cache.staging_offset + chunk, device=DEVICE)
    for layer in range(NUM_LAYERS):
        k, v = rand_kv(chunk)
        cache.write_range(layer, cache.staging_offset, -k, -v)  # poison first
        cache.write_staged(layer, k, v, own_tokens=False)
        k_back, v_back = read_kv(cache, layer, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        cache.write_staged(0, *rand_kv(chunk + 1))


def test_inherited_table_accessors_report_the_rotated_table(cache):
    """V2's own accessors must agree with the logical table, or raise."""
    open_with_fixed(cache, 13)
    for _ in range(6):  # rotate at least once
        cache.commit(cache.max_staged_tokens)
    expected = cache.table.tolist()
    assert cache.get_batch_cache_indices(cache.request_ids(1)) == [expected]
    two = cache.get_batch_cache_indices(cache.request_ids(2), num_blocks_per_seq=[3, 2])
    assert two == [expected[:3], expected[:2]]
    flat = cache.get_batch_cache_indices_flat(cache.request_ids(2), [3, 2])
    assert flat.dtype == torch.int32 and flat.tolist() == expected[:3] + expected[:2]
    with pytest.raises(ValueError):
        cache.get_batch_cache_indices([7])


def test_rejects_bad_geometry():
    pool = dict(num_layers=1, num_kv_heads=1, head_dim=16, dtype=torch.float16)
    for tpb in (0, 12, 24, 100):
        with pytest.raises(ValueError, match="power of two"):
            CausalKVCacheManager(tokens_per_page=tpb, pool_tokens=320, **pool)
    with pytest.raises(ValueError, match="multiple"):
        CausalKVCacheManager(tokens_per_page=32, pool_tokens=100, **pool)
    with pytest.raises(ValueError):
        CausalKVCacheManager(
            tokens_per_page=32, pool_tokens=320, **{**pool, "dtype": torch.float32}
        )
    cache = CausalKVCacheManager(tokens_per_page=32, pool_tokens=320, **pool)
    try:
        with pytest.raises(ValueError, match="at most"):
            cache.open(window_tokens=64, max_staged_tokens=40, causal_block_sizes=(40, 41))
        with pytest.raises(ValueError, match="positive"):
            cache.open(window_tokens=0, max_staged_tokens=40, causal_block_sizes=(40,))
        with pytest.raises(ValueError, match="pool holds"):
            cache.open(window_tokens=64, max_staged_tokens=320, causal_block_sizes=(320,))
        assert not cache.is_open
    finally:
        cache.shutdown()


def test_failed_open_leaves_the_cache_closed(cache, monkeypatch):
    """An open() that fails while allocating a geometry's device state releases the
    sequence and leaves nothing looking open, so the caller can open again."""

    def boom(*args, **kwargs):
        raise RuntimeError("no room for the geometry's device state")

    monkeypatch.setattr(cache, "_allocate_state", boom)
    with pytest.raises(RuntimeError, match="no room"):
        open_cache(cache, pin_tokens=9)
    assert not cache.is_open
    with pytest.raises(RuntimeError, match="not open"):
        cache.table
    monkeypatch.undo()
    open_cache(cache, pin_tokens=9)
    assert cache.is_open and len(set(cache.block_table())) == cache.num_pages


def test_each_geometry_keeps_its_device_state(cache):
    """Geometries opened on one pool keep their kernel-facing tensors at fixed
    addresses across other geometries, so a graph captured over one replays when it
    is opened again; the pool-sized table is shared and refilled."""
    tpb = cache.tokens_per_page
    narrow = dict(window_tokens=tpb, max_staged_tokens=tpb, causal_block_sizes=(tpb, tpb // 2))
    open_cache(cache, pin_tokens=9)
    wide = cache.test_geometry
    size = wide["causal_block_sizes"][-1]
    rows_a, (_, lkv_a) = cache.page_table(size), cache.causal_block_lengths(size)
    table_ptr = cache.table.data_ptr()
    cache.write_range(0, 0, *rand_kv(9))
    cache.commit(9)
    for _ in range(3):
        cache.commit(cache.max_staged_tokens)
    cached_a = cache.cached_tokens(size)
    cache.close()

    cache.open(pin_tokens=5, **narrow)
    assert cache.geometry == (tpb, tpb, (tpb, tpb // 2))
    assert cache.page_table(tpb // 2).data_ptr() != rows_a.data_ptr()
    cache.commit(5)
    cache.commit(tpb)
    assert cache.table.data_ptr() == table_ptr
    cache.close()

    open_cache(cache, pin_tokens=9)
    assert cache.page_table(size).data_ptr() == rows_a.data_ptr()
    assert cache.causal_block_lengths(size)[1].data_ptr() == lkv_a.data_ptr()
    cache.write_range(0, 0, *rand_kv(9))
    cache.commit(9)
    for _ in range(3):
        cache.commit(cache.max_staged_tokens)
    assert cache.cached_tokens(size) == cached_a
    assert len(set(cache.block_table())) == cache.num_pages


def test_commit_takes_the_tokens_actually_written(cache):
    """A rollout's first chunk is one frame: committing it must not promote the rest
    of the chunk's slots to history."""
    tpb = cache.tokens_per_page
    open_with_fixed(cache, 9)
    first = tpb + 3  # shorter than a chunk, not a page multiple
    k, v = rand_kv(first)
    cache.write_range(0, cache.staging_offset, k, v)
    cache.commit(first)
    assert cache.history_tokens == first
    assert cache.staging_offset == 9 + first
    k_back, v_back = read_kv(cache, 0, torch.arange(9, 9 + first, device=DEVICE))
    torch.testing.assert_close(k_back, k)
    torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        cache.commit(0)
    with pytest.raises(ValueError):
        cache.commit(cache.max_staged_tokens + 1)


def test_pool_bytes_for_is_what_the_gpu_loses():
    """``pool_bytes_for`` is the device memory a cache really takes, measured with the
    driver's free-memory counter, so a window sized against a budget before anything
    is built fits. The rounding beyond the raw K/V bytes is less than one allocator
    chunk, and the size grows with the window."""
    pool = dict(num_layers=3, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16)
    geometry = dict(
        tokens_per_page=32, pin_tokens=40, max_staged_tokens=120, causal_block_sizes=(120, 60)
    )
    previous = 0
    for window_tokens in (600, 6000, 60000):  # each step crosses an allocator chunk
        expected = CausalKVCacheManager.pool_bytes_for(
            window_tokens=window_tokens, **pool, **geometry
        )
        pool_tokens = CausalKVCacheManager.pool_tokens_for(window_tokens=window_tokens, **geometry)
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()
        cache = CausalKVCacheManager(tokens_per_page=32, pool_tokens=pool_tokens, **pool)
        try:
            torch.cuda.synchronize()
            free_after, _ = torch.cuda.mem_get_info()
            assert free_before - free_after == expected
            assert cache.pool_bytes == expected
            raw = cache.pool_tokens * 3 * 2 * 2 * 64 * 2
            assert 0 <= expected - raw < 32 << 20
            assert expected > previous
            previous = expected
        finally:
            cache.shutdown()
