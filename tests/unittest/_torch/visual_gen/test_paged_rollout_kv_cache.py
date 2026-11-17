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
"""PagedRolloutKVCache on its own: residency, commits, and the page lists it hands a paged
attention kernel. Every token is stamped with its position and layer, so reading a page list
back tells exactly which tokens a causal block would attend to."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import (
    PagedRolloutKVCache,
    RolloutKVCacheInterface,
    RolloutSpec,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.float16  # integer stamps below 2048 are exact
DEVICE = torch.device("cuda")
LAYER_STAMP = 1024  # stamps of layer l are offset by l * LAYER_STAMP


def spec_for(*, max_pin_tokens=24, window_tokens=44, chunk=24, causal_blocks=None):
    return RolloutSpec(
        window_tokens=window_tokens,
        max_staged_tokens=chunk,
        causal_blocks=causal_blocks or {chunk: 1, chunk // 4: 4},
        max_pin_tokens=max_pin_tokens,
    )


def make_cache(tpb, spec=None, num_layers=NUM_LAYERS):
    spec = spec or spec_for()
    cache = PagedRolloutKVCache(
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=tpb,
        pool_tokens=PagedRolloutKVCache.pool_tokens_for(spec, tokens_per_page=tpb),
    )
    cache.test_spec = spec
    return cache


@pytest.fixture(params=[8, 16], ids=["tpb8", "tpb16"])
def cache(request):
    cache = make_cache(request.param)
    try:
        yield cache
    finally:
        cache.shutdown()


def stamped_kv(positions, layer=0):
    """K/V whose every element is the token's position plus a per-layer offset."""
    stamp = (torch.as_tensor(positions, device=DEVICE) + layer * LAYER_STAMP).to(DTYPE)
    stamp = stamp[:, None, None].expand(-1, NUM_KV_HEADS, HEAD_DIM).contiguous()
    return stamp, -stamp


def read_kv(cache, layer, positions):
    """K and V ``[T, heads, head_dim]`` at resident logical ``positions``, from the pool."""
    buf = cache.kv_buffer(layer)
    page, slot = (torch.from_numpy(x).to(DEVICE) for x in cache.locate(positions))
    return buf[page, 0, :, slot], buf[page, 1, :, slot]


def page_list_stamps(cache, layer, plan, i):
    """The K and V stamps causal block ``i``'s page list presents to a kernel, in order."""
    buf = cache.kv_buffer(layer)
    tables = cache.tables(plan)
    n = int(tables.seq_lens_kv[i])
    pos = torch.arange(n, device=DEVICE)
    page = tables.page_table[i].long()[pos // cache.tokens_per_page]
    slot = pos % cache.tokens_per_page
    return buf[page, 0, :, slot].float(), buf[page, 1, :, slot].float()


def forward(cache, num_tokens, size=None, stamp_offset=0):
    """Stage ``num_tokens`` stamped tokens in every layer, as a forward would."""
    plan = cache.plan(num_tokens, size)
    positions = cache.staging_offset + torch.arange(num_tokens, device=DEVICE)
    for layer in range(cache.num_layers):
        k, v = stamped_kv(positions + stamp_offset, layer)
        cache.write_staged(layer, k, v, plan)
    return plan


def expected_keys(cache, first, size):
    """Positions causal block ``[first, first + size)`` attends to: the fixed region, the
    window before it and the staged tokens up to its end."""
    fixed = cache.fixed_tokens
    start = max(first - cache.window_tokens, fixed)
    return list(range(fixed)) + list(range(start, first + size))


def check_page_lists(cache, plan, step=""):
    tables = cache.tables(plan)
    size = plan.causal_block_size
    torch.testing.assert_close(
        tables.seq_lens_q.cpu(), torch.full((plan.num_causal_blocks,), size, dtype=torch.int32)
    )
    lengths = tables.seq_lens_kv.cpu()
    torch.testing.assert_close(tables.cu_seqlens_kv.cpu()[1:], lengths.cumsum(0).int())
    n = plan.num_causal_blocks
    torch.testing.assert_close(
        tables.cu_seqlens_q.cpu(), torch.arange(n + 1, dtype=torch.int32) * size
    )
    # The TRTLLM op's format: page p's K is at 2p and its V at 2p + 1 of the pool viewed as
    # [2 * num_pages, H, page_size, D].
    assert torch.equal(tables.kv_block_offsets[:, 0], 2 * tables.page_table)
    assert torch.equal(tables.kv_block_offsets[:, 1], 2 * tables.page_table + 1)
    for i in range(plan.num_causal_blocks):
        expected = expected_keys(cache, cache.staging_offset + i * size, size)
        assert int(lengths[i]) == len(expected), f"{step} causal block {i}"
        for layer in range(cache.num_layers):
            keys, values = page_list_stamps(cache, layer, plan, i)
            want = torch.tensor(expected, dtype=torch.float32, device=DEVICE) + layer * LAYER_STAMP
            got = keys[:, 0, 0].sort().values
            assert torch.equal(got, want), f"{step} causal block {i} layer {layer}"
            assert torch.equal(keys, -values), f"{step} causal block {i} layer {layer}: K/V"
            assert torch.equal(keys, keys[:, :1, :1].expand_as(keys)), "heads or dims mixed up"


def test_implements_the_rollout_interface():
    assert issubclass(PagedRolloutKVCache, RolloutKVCacheInterface)
    assert not PagedRolloutKVCache.__abstractmethods__


def test_open_is_a_context_manager_that_closes_on_exit(cache):
    with cache.open(cache.test_spec, pin_tokens=4) as opened:
        assert opened is cache and cache.is_open and cache.spec == cache.test_spec
    assert not cache.is_open
    with pytest.raises(RuntimeError, match="not open"):
        cache.kv_buffer(0)
    with pytest.raises(RuntimeError):
        with cache.open(cache.test_spec):
            raise RuntimeError("boom")
    assert not cache.is_open
    cache.open(cache.test_spec)
    with pytest.raises(RuntimeError, match="already open"):
        cache.open(cache.test_spec)
    cache.shutdown()
    assert cache.is_shut_down and not cache.is_open
    with pytest.raises(RuntimeError, match="shut down"):
        cache.open(cache.test_spec)


@pytest.mark.parametrize("pin", [0, 5, 16, 24], ids=["no-pin", "unaligned", "page", "max"])
@pytest.mark.parametrize("size", [24, 6], ids=["one-block", "four-blocks"])
def test_page_lists_present_exactly_the_window(cache, pin, size):
    """Every causal block's page list holds the fixed region, exactly ``window_tokens`` of
    history before it (fewer while the window fills), the earlier causal blocks and itself;
    nothing stale, nothing later, nothing twice; every layer its own K/V. Through enough
    commits to recycle every resident page several times."""
    spec = cache.test_spec
    chunk = spec.max_staged_tokens
    cache.open(spec, pin_tokens=pin)
    if pin:
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, 0, *stamped_kv(torch.arange(pin), layer))
        cache.commit(pin)
    assert cache.fixed_tokens == pin
    saw_stale = False
    for step in range(18):
        plan = forward(cache, chunk, size)
        check_page_lists(cache, plan, f"step {step}")
        if plan.num_causal_blocks > 1:  # a shorter forward uses the leading causal blocks
            check_page_lists(cache, forward(cache, size * 2, size), f"step {step}, short")
            forward(cache, chunk, size)
        for layer in range(NUM_LAYERS):
            positions = cache.staging_offset + torch.arange(chunk)
            k, v = read_kv(cache, layer, positions)
            torch.testing.assert_close(k, stamped_kv(positions, layer)[0])
            torch.testing.assert_close(v, stamped_kv(positions, layer)[1])
        saw_stale |= cache.history_tokens > spec.window_tokens
        cache.commit()
    assert saw_stale, "test geometry should leave stale tokens on the window's first page"
    assert cache.staging_offset > 3 * cache.resident_tokens, "pages should be recycled"


@pytest.mark.parametrize(
    "tpb, chunk, size, window",
    [(8, 20, 5, 13), (8, 16, 8, 8), (16, 40, 10, 3), (8, 12, 1, 1), (32, 30, 3, 70)],
)
def test_page_lists_in_odd_geometries(tpb, chunk, size, window):
    """Causal blocks smaller than, equal to and larger than a page; windows shorter than a
    causal block, of one token, and longer than the pool's pages."""
    spec = spec_for(
        max_pin_tokens=7, window_tokens=window, chunk=chunk, causal_blocks={size: chunk // size}
    )
    cache = make_cache(tpb, spec)
    try:
        cache.open(spec, pin_tokens=7)
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, 0, *stamped_kv(torch.arange(7), layer))
        cache.commit(7)
        for step in range(8):
            check_page_lists(cache, forward(cache, chunk, size), f"step {step}")
            cache.commit()
    finally:
        cache.shutdown()


def test_staged_writes_overwrite_until_commit(cache):
    """Forwards between two commits rewrite the same positions; the last one is final."""
    cache.open(cache.test_spec)
    chunk = cache.max_staged_tokens
    forward(cache, chunk, stamp_offset=500)
    plan = forward(cache, chunk)
    check_page_lists(cache, plan)
    cache.commit()
    torch.testing.assert_close(
        read_kv(cache, 1, torch.arange(chunk))[0], stamped_kv(torch.arange(chunk), 1)[0]
    )


def test_write_staged_takes_strided_slices(cache):
    """K/V sliced from a fused projection (strided tokens and heads) stage without a copy
    and land like contiguous ones."""
    cache.open(cache.test_spec)
    chunk = cache.max_staged_tokens
    plan = cache.plan(chunk)
    k, v = stamped_kv(torch.arange(chunk))
    fused = torch.cat([torch.zeros_like(k), k, v], dim=1)  # [T, 3 * heads, D]
    cache.write_staged(
        0, fused[:, NUM_KV_HEADS : 2 * NUM_KV_HEADS], fused[:, 2 * NUM_KV_HEADS :], plan
    )
    back_k, back_v = read_kv(cache, 0, torch.arange(chunk))
    torch.testing.assert_close(back_k, k)
    torch.testing.assert_close(back_v, v)


def test_edits_to_committed_tokens_reach_the_private_copies(cache):
    """A committed token a causal block reads through a private copy, rewritten with
    ``write_range``, is read anew by the next forward."""
    spec = cache.test_spec
    cache.open(spec, pin_tokens=5)
    cache.write_range(0, 0, *stamped_kv(torch.arange(5)))
    cache.commit(5)
    for _ in range(3):
        forward(cache, spec.max_staged_tokens)
        cache.commit()
    edited = list(range(cache.fixed_tokens)) + list(
        range(cache.staging_offset - spec.window_tokens, cache.staging_offset)
    )
    cache.write_range(0, edited[0], *stamped_kv(torch.tensor(edited[:5]) + 300))
    cache.write_range(
        0, edited[5], *stamped_kv(torch.arange(edited[5], cache.staging_offset) + 300)
    )
    plan = forward(cache, spec.max_staged_tokens, spec.max_staged_tokens // 4)
    for i in range(plan.num_causal_blocks):
        keys = page_list_stamps(cache, 0, plan, i)[0][:, 0, 0]
        first = cache.staging_offset + i * plan.causal_block_size
        assert sorted(keys.tolist()) == sorted(
            [p + 300 for p in edited if p >= first - spec.window_tokens or p < 5]
            + list(range(cache.staging_offset, first + plan.causal_block_size))
        ), f"causal block {i}"


def test_pinned_tokens_are_the_first_committed_and_never_evicted(cache):
    spec = cache.test_spec
    cache.open(spec, pin_tokens=20)
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, 0, *stamped_kv(torch.arange(12), layer))
    cache.commit(12)  # part of the pinned prefix
    assert cache.fixed_tokens == 12
    plan = forward(cache, spec.max_staged_tokens)
    cache.commit()  # pins 8 more; the rest is history
    assert cache.fixed_tokens == 20
    for _ in range(10):
        plan = forward(cache, spec.max_staged_tokens)
        cache.commit()
    positions = torch.arange(12)
    torch.testing.assert_close(read_kv(cache, 0, positions)[0], stamped_kv(positions)[0])
    check_page_lists(cache, forward(cache, spec.max_staged_tokens))
    with pytest.raises(ValueError, match="not resident"):
        cache.locate([cache.staging_offset - spec.window_tokens - cache.tokens_per_page - 1])
    del plan


def test_commit_counts(cache):
    spec = cache.test_spec
    chunk = spec.max_staged_tokens
    cache.open(spec, pin_tokens=10)
    with pytest.raises(ValueError, match="nothing was planned"):
        cache.commit()
    with pytest.raises(ValueError, match="at most"):
        cache.commit(chunk + 11)  # only the 10 pinned tokens may exceed a chunk
    cache.commit(chunk + 10)
    assert (cache.fixed_tokens, cache.staging_offset) == (10, chunk + 10)
    forward(cache, chunk // 2, chunk // 4)
    forward(cache, chunk)
    with pytest.raises(ValueError, match="staged"):
        cache.commit(chunk // 2)
    cache.commit()  # the most a forward covered
    assert cache.staging_offset == 2 * chunk + 10
    forward(cache, chunk // 2, chunk // 4)
    cache.commit()
    assert cache.staging_offset == 2 * chunk + 10 + chunk // 2


def test_plan_follows_the_open_spec(cache):
    spec = cache.test_spec
    cache.open(spec)
    plan = cache.plan(spec.max_staged_tokens, spec.max_staged_tokens // 4)
    assert (plan.num_tokens, plan.causal_block_size, plan.num_causal_blocks) == (
        spec.max_staged_tokens,
        spec.max_staged_tokens // 4,
        4,
    )
    assert cache.plan(spec.max_staged_tokens).num_causal_blocks == 1
    assert cache.plan(12, 6).num_causal_blocks == 2  # a shorter forward, fewer causal blocks
    with pytest.raises(ValueError, match="not declared"):
        cache.plan(spec.max_staged_tokens, 4)
    with pytest.raises(ValueError, match="do not split"):
        cache.plan(spec.max_staged_tokens, 5)
    with pytest.raises(ValueError, match="exceed"):
        cache.plan(spec.max_staged_tokens + 6, 6)
    cache.close()
    with pytest.raises(RuntimeError, match="no longer open"):
        plan.validate()


def test_device_state_stays_in_place(cache):
    """Commits and reopens rewrite the device tensors a captured forward reads, never move
    them; each spec keeps its own."""
    spec = cache.test_spec
    other = spec_for(window_tokens=8)

    def addresses():
        plan = cache.plan(spec.max_staged_tokens, spec.max_staged_tokens // 4)
        return [t.data_ptr() for t in cache.tables(plan)]

    cache.open(spec, pin_tokens=3)
    before = addresses()
    for _ in range(4):
        forward(cache, spec.max_staged_tokens)
        cache.commit()
        assert addresses() == before
    cache.close()
    cache.open(other)
    assert addresses() != before
    cache.close()
    cache.open(spec, pin_tokens=20)
    assert addresses() == before


def test_rejects_bad_geometry():
    spec = spec_for()
    with pytest.raises(ValueError, match="power of two"):
        make_cache(12)
    with pytest.raises(ValueError, match="bf16/fp16"):
        PagedRolloutKVCache(
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            dtype=torch.float32,
            tokens_per_page=8,
            pool_tokens=64,
        )
    small = PagedRolloutKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        dtype=DTYPE,
        tokens_per_page=8,
        pool_tokens=PagedRolloutKVCache.pool_tokens_for(spec, tokens_per_page=8) - 8,
    )
    with pytest.raises(ValueError, match="needs"):
        small.open(spec)
    assert not small.is_open
    with pytest.raises(ValueError, match="pin_tokens"):
        make_cache(8, spec).open(spec, pin_tokens=spec.max_pin_tokens + 1)


def test_pool_bytes_for_is_what_the_pool_takes():
    spec = spec_for()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    cache = make_cache(8, spec)
    try:
        taken = torch.cuda.memory_allocated() - before
        assert taken == PagedRolloutKVCache.pool_bytes_for(
            spec,
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=DTYPE,
            tokens_per_page=8,
        )
    finally:
        cache.shutdown()
