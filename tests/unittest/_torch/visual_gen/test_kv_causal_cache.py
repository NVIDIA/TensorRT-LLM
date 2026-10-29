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
"""CausalKVCacheManager: which pages a rollout keeps, and where staged K/V lands."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.float16

# (tokens_per_page, pin_tokens, tokens_per_frame, window_frames, chunk_frames): fixed regions that
# end mid-page and on a page boundary, frames that are and are not a whole number of pages.
GEOMETRIES = [
    pytest.param(32, 48, 96, 4, 4, id="tpb32-pin48-frame96"),
    pytest.param(32, 64, 96, 3, 2, id="tpb32-pin-aligned"),
    # h - window is the last slot of a page at every chunk boundary: the window's first token
    # is alone on a page the manager must keep.
    pytest.param(32, 63, 96, 3, 2, id="tpb32-window-start-on-last-slot"),
    pytest.param(16, 17, 40, 5, 2, id="tpb16-pin17-frame40"),
    pytest.param(64, 100, 200, 2, 3, id="tpb64-pin100-frame200"),
    pytest.param(32, 0, 96, 4, 4, id="tpb32-no-pin"),
]


def make_cache(tpb, pin, frame, window_frames, chunk_frames, max_chunks=40):
    """A cache for a rollout of up to ``max_chunks`` chunks, far more than the pool holds."""
    window = (window_frames - 1) * frame
    staged = chunk_frames * frame
    return CausalKVCacheManager(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=tpb,
        pin_tokens=pin,
        window_tokens=window,
        max_staged_tokens=staged,
        max_seq_len=pin + max_chunks * staged,
    )


def stamp(layer, start, count, plane):
    """K (plane 0) or V (plane 1) of tokens ``[start, start + count)``, a function of
    layer, absolute position, head and channel alone; exact in fp16."""
    pos = torch.arange(start, start + count, device="cuda") % 512 + 1
    head = torch.arange(1, NUM_KV_HEADS + 1, device="cuda")
    chan = torch.arange(HEAD_DIM, device="cuda")
    x = pos[:, None, None] * head[None, :, None] + chan[None, None, :] + 1000 * (layer + 1)
    return (x if plane == 0 else -x).to(DTYPE)


def stage(cache, count):
    for layer in range(NUM_LAYERS):
        cache.write_staged(
            layer,
            stamp(layer, cache.staging_offset, count, 0),
            stamp(layer, cache.staging_offset, count, 1),
        )


def resident_pages(cache):
    return {p for p, page in enumerate(cache.host_page_table()) if page >= 0}


def expected_pages(cache):
    """Logical pages the manager keeps: the fixed region's pages, and every page that
    holds a token of the window before ``staging_offset`` or of the staging room."""
    tpb = cache.tokens_per_page
    history = cache.staging_offset
    end = history + cache.staging_capacity
    first_window_page = max(0, history - cache.window_tokens) // tpb
    sink_pages = -(-cache.pin_tokens // tpb)
    return {p for p in range(-(-end // tpb)) if p < sink_pages or p >= first_window_page}


def committed_chunks(cache, num_chunks):
    """Drive a rollout: the fixed region first, then ``num_chunks`` full chunks."""
    if cache.pin_tokens:
        stage(cache, cache.pin_tokens)
        cache.commit(cache.pin_tokens)
        yield
    for _ in range(num_chunks):
        stage(cache, cache.max_staged_tokens)
        cache.commit(cache.max_staged_tokens)
        yield


@pytest.mark.parametrize("tpb,pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_keeps_only_the_pages_the_window_needs(tpb, pin, frame, window_frames, chunk_frames):
    cache = make_cache(tpb, pin, frame, window_frames, chunk_frames)
    try:
        cache.open()
        assert resident_pages(cache) == expected_pages(cache)
        for _ in committed_chunks(cache, 30):
            assert resident_pages(cache) == expected_pages(cache)
        # The rollout ran far past what the pool holds, so pages were recycled.
        pool_pages = CausalKVCacheManager.pool_pages_for(
            tokens_per_page=tpb,
            pin_tokens=pin,
            window_tokens=cache.window_tokens,
            max_staged_tokens=cache.max_staged_tokens,
        )
        assert cache.staging_offset > 3 * pool_pages * tpb
    finally:
        cache.shutdown()


@pytest.mark.parametrize("tpb,pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_staged_kv_is_readable_through_the_page_table(tpb, pin, frame, window_frames, chunk_frames):
    cache = make_cache(tpb, pin, frame, window_frames, chunk_frames)
    try:
        cache.open()
        for _ in committed_chunks(cache, 12):
            history = cache.staging_offset
            # What the next forward reads of the past: the fixed region and the window.
            first = max(cache.fixed_tokens, history - cache.window_tokens)
            positions = list(range(cache.fixed_tokens)) + list(range(first, history))
            table = cache.page_table.cpu()
            for layer in range(NUM_LAYERS):
                buf = cache.kv_buffer(layer)
                for plane in (0, 1):
                    got = torch.stack([buf[table[p // tpb], plane, :, p % tpb] for p in positions])
                    want = torch.cat([stamp(layer, p, 1, plane) for p in positions])
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        cache.shutdown()


def test_restaging_overwrites_the_same_positions():
    """A denoising step stages the chunk again over the previous step's tokens."""
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        for _ in committed_chunks(cache, 3):
            pass
        start, count = cache.staging_offset, cache.max_staged_tokens
        junk = torch.full((count, NUM_KV_HEADS, HEAD_DIM), 7.0, dtype=DTYPE, device="cuda")
        for layer in range(NUM_LAYERS):
            cache.write_staged(layer, junk, -junk)
        stage(cache, count)
        table = cache.page_table.cpu()
        for layer in range(NUM_LAYERS):
            buf = cache.kv_buffer(layer)
            for p in (start, start + count // 2, start + count - 1):
                torch.testing.assert_close(
                    buf[table[p // 32], 0, :, p % 32], stamp(layer, p, 1, 0)[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    buf[table[p // 32], 1, :, p % 32], stamp(layer, p, 1, 1)[0], rtol=0, atol=0
                )
    finally:
        cache.shutdown()


def test_a_write_does_not_touch_committed_tokens():
    """Staging starts mid-page when the fixed region ends mid-page; the slots before it
    on that page hold committed tokens and must survive every staged write."""
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        stage(cache, 48)
        cache.commit(48)
        stage(cache, cache.max_staged_tokens)
        table = cache.page_table.cpu()
        buf = cache.kv_buffer(0)
        for p in range(48):
            torch.testing.assert_close(
                buf[table[p // 32], 0, :, p % 32], stamp(0, p, 1, 0)[0], rtol=0, atol=0
            )
    finally:
        cache.shutdown()


def test_fewer_tokens_than_a_full_chunk():
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        for _ in committed_chunks(cache, 2):
            pass
        stage(cache, 100)
        cache.commit(100)
        assert cache.staging_offset == 48 + 2 * cache.max_staged_tokens + 100
        assert resident_pages(cache) == expected_pages(cache)
    finally:
        cache.shutdown()


def test_table_and_ranges_follow_commits():
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        stage(cache, 48)
        cache.commit(48)
        assert cache.fixed_tokens == 48
        starts, lengths = cache.kv_ranges(4, 96)
        # Frame j of the next chunk sees the window before it and everything up to its end.
        assert starts.tolist() == [0, 0, 0, 48]  # window of 3 frames before the frame
        assert lengths.tolist() == [144, 240, 336, 384]
        table_address = cache.page_table.data_ptr()
        for _ in range(5):
            stage(cache, cache.max_staged_tokens)
            cache.commit(cache.max_staged_tokens)
        # A graph captured around a forward holds these addresses.
        assert cache.page_table.data_ptr() == table_address
        assert cache.kv_ranges(4, 96)[0].data_ptr() == starts.data_ptr()
        first = 48 + 5 * 384
        assert starts.tolist() == [first + 96 * j - 3 * 96 for j in range(4)]
        assert lengths.tolist() == [4 * 96] * 4  # the window and the frame
        host = cache.host_page_table()
        device = cache.page_table.cpu().tolist()
        assert device == [p if p >= 0 else 0 for p in host]
    finally:
        cache.shutdown()


def test_the_fixed_region_may_exceed_one_chunk():
    cache = make_cache(32, 700, 96, 3, 2)
    try:
        cache.open()
        assert cache.staging_capacity == 700 + 2 * 96
        stage(cache, 700)
        cache.commit(700)
        assert cache.fixed_tokens == 700
        assert cache.staging_capacity == 2 * 96
        assert resident_pages(cache) == expected_pages(cache)
    finally:
        cache.shutdown()


def test_reopen_starts_a_new_rollout():
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        for _ in committed_chunks(cache, 4):
            pass
        cache.close()
        assert not cache.is_open
        cache.open()
        assert cache.staging_offset == 0 and cache.fixed_tokens == 0
        assert resident_pages(cache) == expected_pages(cache)
        for _ in committed_chunks(cache, 4):
            assert resident_pages(cache) == expected_pages(cache)
    finally:
        cache.shutdown()


def test_rejects_bad_calls():
    cache = make_cache(32, 48, 96, 4, 4, max_chunks=3)
    try:
        with pytest.raises(RuntimeError, match="not open"):
            cache.commit(1)
        cache.open()
        with pytest.raises(RuntimeError, match="already open"):
            cache.open()
        with pytest.raises(ValueError, match="stages 1"):
            cache.commit(0)
        with pytest.raises(ValueError, match="stages 1"):
            cache.commit(cache.staging_capacity + 1)
        k = torch.zeros(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        with pytest.raises(TypeError, match="dtype"):
            cache.write_staged(0, k.float(), k.float())
        too_many = k.expand(cache.staging_capacity + 1, -1, -1)
        with pytest.raises(ValueError, match="does not fit"):
            cache.write_staged(0, too_many, too_many)
        with pytest.raises(ValueError, match="shape mismatch"):
            cache.write_staged(0, k, k.expand(2, -1, -1))
        for _ in committed_chunks(cache, 2):
            pass
        offset = cache.staging_offset
        with pytest.raises(ValueError, match="max_seq_len"):
            cache.commit(cache.max_staged_tokens)
        # The refused commit changed nothing.
        assert cache.staging_offset == offset
        assert resident_pages(cache) == expected_pages(cache)
    finally:
        cache.shutdown()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(tokens_per_page=24), "power of two"),
        (dict(window_tokens=-1), "non-negative"),
        (dict(max_staged_tokens=0), "positive"),
        (dict(dtype=torch.float32), "bf16/fp16"),
        (dict(max_seq_len=10), "max_seq_len"),
    ],
)
def test_rejects_bad_geometry(kwargs, match):
    args = dict(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        dtype=DTYPE,
        tokens_per_page=32,
        pin_tokens=48,
        window_tokens=96,
        max_staged_tokens=96,
        max_seq_len=1000,
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CausalKVCacheManager(**args)


def test_causal_blocks():
    assert CausalKVCacheManager.causal_blocks(384, None) == (384, 1)
    assert CausalKVCacheManager.causal_blocks(384, 96) == (96, 4)
    with pytest.raises(ValueError, match="do not split"):
        CausalKVCacheManager.causal_blocks(384, 100)


def test_staged_write_replays_in_a_cuda_graph_across_commits():
    """A graph captured around a staged write keeps writing at the current staging
    offset after commits move the window and recycle pages."""
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        for _ in committed_chunks(cache, 2):
            pass
        count = cache.max_staged_tokens
        k_in = torch.zeros(count, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        v_in = torch.zeros_like(k_in)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cache.write_staged(0, k_in, v_in)  # warm up before capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache.write_staged(0, k_in, v_in)
        for _ in range(6):  # far enough that pages are freed and reused
            k_in.copy_(stamp(0, cache.staging_offset, count, 0))
            v_in.copy_(stamp(0, cache.staging_offset, count, 1))
            graph.replay()
            table = cache.page_table.cpu()
            buf = cache.kv_buffer(0)
            for p in (cache.staging_offset, cache.staging_offset + count - 1):
                torch.testing.assert_close(
                    buf[table[p // 32], 0, :, p % 32], stamp(0, p, 1, 0)[0], rtol=0, atol=0
                )
            # Commit the chunk through the other layer's path; layer 0 holds the graph's.
            stage(cache, count)
            cache.commit(count)
    finally:
        cache.shutdown()


def test_write_staged_takes_slices_of_a_fused_projection():
    """K and V sliced from one fused QKV buffer, with token and head strides of their own, land
    at their logical positions, token for token."""
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        for _ in committed_chunks(cache, 3):
            pass
        count, hd = 200, NUM_KV_HEADS * HEAD_DIM
        k_ref = stamp(1, cache.staging_offset, count, 0)
        v_ref = stamp(1, cache.staging_offset, count, 1)
        qkv = torch.zeros(count, 3 * hd + 8, dtype=DTYPE, device="cuda")
        qkv[:, hd : 2 * hd] = k_ref.flatten(1)
        qkv[:, 2 * hd : 3 * hd] = v_ref.flatten(1)
        k = qkv[:, hd : 2 * hd].unflatten(1, (NUM_KV_HEADS, HEAD_DIM))
        v = qkv[:, 2 * hd : 3 * hd].unflatten(1, (NUM_KV_HEADS, HEAD_DIM))
        assert not k.is_contiguous()
        cache.write_staged(1, k, v)
        tpb, buf = cache.tokens_per_page, cache.kv_buffer(1)
        pos = cache.staging_offset + torch.arange(count)
        pages = cache.page_table.cpu()[pos // tpb].long().cuda()
        slots = (pos % tpb).cuda()
        torch.testing.assert_close(buf[pages, 0, :, slots], k_ref, rtol=0, atol=0)
        torch.testing.assert_close(buf[pages, 1, :, slots], v_ref, rtol=0, atol=0)
    finally:
        cache.shutdown()
