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
"""Serving a span of cache named by its content: name it, take it, address it.

The loaded backend is checked directly; the two are compared through a
subprocess, since only one can be live per process.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    CacheTier,
    GpuCacheTierConfig,
    KVCacheDesc,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    TokenId,
    _introspection,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
    TemporaryCudaStream,
    init_cuda_once,
    temporary_sys_path,
)

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    import content_probe
    from test_kv_cache_manager_v2 import create_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS_PER_BLOCK = content_probe.TOKENS_PER_BLOCK
NUM_BLOCKS = content_probe.NUM_BLOCKS
CHAIN_TOKENS = NUM_BLOCKS * TOKENS_PER_BLOCK
# Long enough that the host reaches its assertions while the stall is still
# running, and generous so it stays that way on a fast GPU. The same lever the
# block-offset overlap race regression uses to hold a copy open.
_STALL_CYCLES = 2_000_000_000
GPU_QUOTA = 16 << 20
HOST_QUOTA = 16 << 20
DISK_QUOTA = 16 << 20

_BACKEND = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower()
_OTHER_BACKEND = "python" if _BACKEND == "cpp" else "cpp"

requires_cpp_backend = pytest.mark.skipif(
    _BACKEND != "cpp", reason="the padding cold-page codec is native only"
)


@pytest.fixture
def manager():
    """Device memory only: enough to commit a chain and then name it back."""
    init_cuda_once()
    mgr = KVCacheManager(create_config(TOKENS_PER_BLOCK, GPU_QUOTA, 0, 0, 2, None, 0))
    try:
        yield mgr
    finally:
        mgr.clear_reusable_blocks()
        mgr.shutdown()


@pytest.fixture
def tiered_manager():
    """Device, host and disk, so every answer a level can give is reachable."""
    init_cuda_once()
    mgr = KVCacheManager(
        create_config(TOKENS_PER_BLOCK, GPU_QUOTA, HOST_QUOTA, DISK_QUOTA, 2, None, 0)
    )
    try:
        yield mgr
    finally:
        mgr.clear_reusable_blocks()
        mgr.shutdown()


@pytest.fixture
def swa_manager():
    """Two life cycles -- a sliding window and full attention -- over one tree.

    Blocks are made by hand under this one, so the collector stays off across the
    whole fixture and is put back as found even if the manager fails to build.
    """
    init_cuda_once()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        mgr = KVCacheManager(
            create_config(TOKENS_PER_BLOCK, GPU_QUOTA, 0, 0, 2, TOKENS_PER_BLOCK, 0)
        )
        try:
            yield mgr
        finally:
            mgr.clear_reusable_blocks()
            mgr.shutdown()
    finally:
        if was_enabled:
            gc.enable()


def _commit_and_name(manager, first_token: int = 0):
    """Commit whole blocks, then derive the name a holder would be handed."""
    tokens = [TokenId(first_token + i) for i in range(CHAIN_TOKENS)]
    content_probe.commit_chain(manager, tokens)
    return tokens, content_probe.chain_keys(tokens)


def _uncomputed_name():
    """A well-formed chain in this namespace that nobody here ever computed."""
    return content_probe.chain_keys([TokenId(9000 + i) for i in range(CHAIN_TOKENS)])


def _commit_and_name_a_partial_block(manager, first_token: int = 5000):
    """Commit a chain ending part-way into a block, then name every link.

    ``stop_committing`` commits that last block at its real coverage, so the key
    naming it resolves to a block holding fewer tokens than a whole one.
    """
    tokens = [TokenId(first_token + i) for i in range(CHAIN_TOKENS + 2)]
    content_probe.commit_chain(manager, tokens)
    return tokens, content_probe.chain_keys(tokens)


def _layout(descs):
    """Everything a reader needs in order to address a slot at a level."""
    return [
        (
            int(desc.pool_group_index),
            int(desc.num_slots),
            [(int(p.pool_index), int(p.base_address), int(p.slot_bytes)) for p in desc.pools],
        )
        for desc in descs
    ]


# ---- servable_chain --------------------------------------------------------


def test_an_empty_chain_is_answered_rather_than_rejected(manager):
    # The capability probe run at assembly calls exactly this. Raising here
    # would make a build that works look like it cannot serve content at all.
    assert manager.servable_chain([], []) is None


def test_a_committed_chain_is_servable_at_the_level_holding_it(manager):
    _, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    assert manager.servable_chain(keys, life_cycles) == (GPU_LEVEL, CHAIN_TOKENS)


def test_a_chain_nobody_here_computed_is_not_servable(manager):
    _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    assert manager.servable_chain(_uncomputed_name(), life_cycles) is None


def test_a_chain_is_servable_for_exactly_the_blocks_it_names(manager):
    _, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    shorter = (NUM_BLOCKS - 1) * TOKENS_PER_BLOCK
    assert manager.servable_chain(keys[:-1], life_cycles) == (GPU_LEVEL, shorter)
    # The root is a namespace, not a block, so it names nothing to serve.
    assert manager.servable_chain(keys[:1], life_cycles) is None


def test_a_chain_that_forks_from_the_tree_is_servable_only_up_to_the_fork(manager):
    _, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    forked = keys[:-1] + [content_probe.ABSENT_KEY]
    shorter = (NUM_BLOCKS - 1) * TOKENS_PER_BLOCK
    # Answered short rather than refused: whether short is good enough is the
    # caller's test, and it is all-or-none there.
    assert manager.servable_chain(forked, life_cycles) == (GPU_LEVEL, shorter)
    # A bytes subclass is that same name: the binding asks PyBytes_Check, so an
    # exact type check on the other side would refuse what this one answers.
    subclassed = keys[:-1] + [content_probe.KeyLike(content_probe.ABSENT_KEY)]
    assert manager.servable_chain(subclassed, life_cycles) == (GPU_LEVEL, shorter)


def test_naming_no_life_cycle_leaves_nothing_known_to_be_covered(manager):
    _, keys = _commit_and_name(manager)
    assert manager.servable_chain(keys, []) is None


def test_a_life_cycle_this_manager_does_not_have_is_refused(manager):
    """An id this manager does not have is refused, not answered.

    Unchecked, ``-1`` names the last life cycle in one backend and reads past
    the front of a TypedVec in the other. Neither of those is a refusal.
    """
    _, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    num_life_cycles = len(_introspection.life_cycle_pool_group_indices(manager))
    # The premise: an id this manager does have is answered, not refused.
    assert manager.servable_chain(keys, life_cycles) is not None
    with pytest.raises(ValueError):
        manager.servable_chain(keys, [num_life_cycles])
    with pytest.raises(ValueError):
        manager.servable_chain(keys, [-1])


def test_a_name_that_is_not_a_digest_is_refused_rather_than_missed(manager):
    """A corrupt name is refused, not reported as a cache miss.

    Both backends refuse it at the same point: the binding as it casts the
    sequence, the pure-Python tree as it walks it.
    """
    _, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    for bad in (
        content_probe.ABSENT_KEY[:31],
        content_probe.ABSENT_KEY + b"\x00",
        bytearray(content_probe.ABSENT_KEY),
        content_probe.ABSENT_KEY.hex(),
    ):
        with pytest.raises(ValueError):
            manager.servable_chain(keys[:-1] + [bad], life_cycles)


def test_a_chain_naming_a_block_that_is_not_whole_is_not_servable(manager):
    """A chain whose last block is short is not servable at all.

    All-or-nothing cannot serve a fraction of a block, and the peer would read
    the untouched tail as if it were content.
    """
    _, keys = _commit_and_name_a_partial_block(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    # The premise: the same name, up to the last whole block, is servable.
    assert manager.servable_chain(keys[:-1], life_cycles) == (GPU_LEVEL, CHAIN_TOKENS)
    assert manager.servable_chain(keys, life_cycles) is None


def _commit_behind_a_stall(manager, tokens):
    """Commit a chain whose pages cannot be read until a GPU stall clears.

    Closing does not synchronize, so a stall ahead of the commit leaves what a
    migration in flight leaves. Returns the releasing and the pending event.
    """
    with TemporaryCudaStream([]) as stream_ctx:
        stalled = torch.cuda.ExternalStream(int(stream_ctx.handle))
        with torch.cuda.stream(stalled):
            torch.cuda._sleep(_STALL_CYCLES)
        still_stalled = torch.cuda.Event()
        still_stalled.record(stalled)

        kv = manager.create_kv_cache(None, tokens)
        assert kv.resume(stream_ctx.handle)
        assert kv.resize(len(tokens))
        already = kv.num_committed_tokens
        if already < len(tokens):
            kv.commit(tokens[already:])
        kv.stop_committing()
        kv.close()
    return stream_ctx.take_finish_event(), still_stalled


def test_a_page_still_being_filled_answers_the_same_as_a_landed_one(manager):
    """The answer is topology, and a copy in flight does not change topology.

    A host gate here was tried and removed, because the query holds nothing.
    This pins that it stays gone. Held open on purpose, not raced.
    """
    tokens = [TokenId(i) for i in range(CHAIN_TOKENS)]
    finish, still_stalled = _commit_behind_a_stall(manager, tokens)
    keys = content_probe.chain_keys(tokens)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    try:
        # The premise, stated rather than assumed. If a faster GPU ever drains
        # the stall before the host arrives, raise _STALL_CYCLES -- the
        # assertion below is the point and must not be weakened to suit it.
        assert not still_stalled.query(), "the stall drained before the chain was asked about"
        stalled_answer = manager.servable_chain(keys, life_cycles)
    finally:
        finish.synchronize()
    assert stalled_answer == (GPU_LEVEL, CHAIN_TOKENS)
    assert manager.servable_chain(keys, life_cycles) == stalled_answer


def test_a_chain_uncovered_for_a_life_cycle_this_worker_reads_is_refused(swa_manager):
    """Every matched block needs a page for every life cycle asked about.

    A block out of the sliding window keeps its match, but a peer reading that
    life cycle would be sent a page that is not there.
    """
    swa_lc = _introspection.swa_life_cycle_ids(swa_manager)[0]
    life_cycles = _introspection.attention_life_cycle_ids(swa_manager)
    assert len(life_cycles) == 2
    full_lc = next(lc for lc in life_cycles if lc != swa_lc)

    tokens = [TokenId(i) for i in range(2 * TOKENS_PER_BLOCK)]
    out_of_window = [0, 0]
    out_of_window[full_lc] = TOKENS_PER_BLOCK
    covered = [TOKENS_PER_BLOCK, TOKENS_PER_BLOCK]

    first = _introspection.make_test_block(swa_manager, tokens[:TOKENS_PER_BLOCK], out_of_window)
    try:
        second = _introspection.make_test_block(
            swa_manager, tokens[TOKENS_PER_BLOCK:], covered, first
        )
        try:
            keys = content_probe.chain_keys(tokens)
            assert keys[1:] == [
                _introspection.test_block_key(first),
                _introspection.test_block_key(second),
            ]
            assert swa_manager.servable_chain(keys, [full_lc]) == (
                GPU_LEVEL,
                2 * TOKENS_PER_BLOCK,
            )
            assert swa_manager.servable_chain(keys, life_cycles) is None
        finally:
            _introspection.close_test_block(second)
    finally:
        _introspection.close_test_block(first)


# ---- create_kv_cache_from_keys ---------------------------------------------


def test_an_empty_chain_cannot_be_held(manager):
    # An empty chain names nothing, so there is no honest prompt length to
    # state. A bad argument and not a fault, so ValueError -- and the same one
    # from either backend, since refusing differently is a divergence too.
    with pytest.raises(ValueError):
        manager.create_kv_cache_from_keys([])


def test_holding_a_chain_by_name_lands_on_the_pages_matching_it_by_tokens(manager):
    tokens, keys = _commit_and_name(manager)
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    by_tokens = manager.create_kv_cache(None, tokens)
    by_keys = manager.create_kv_cache_from_keys(keys)
    try:
        assert by_tokens.num_committed_tokens == CHAIN_TOKENS
        for lc_id in life_cycles:
            assert list(by_keys.get_aggregated_page_indices(lc_id, valid_only=True)) == list(
                by_tokens.get_aggregated_page_indices(lc_id, valid_only=True)
            )
    finally:
        by_keys.close()
        by_tokens.close()


def test_holding_a_chain_nobody_here_computed_holds_nothing(manager):
    life_cycles = _introspection.attention_life_cycle_ids(manager)
    held = manager.create_kv_cache_from_keys(_uncomputed_name())
    try:
        for lc_id in life_cycles:
            assert list(held.get_aggregated_page_indices(lc_id, valid_only=True)) == []
    finally:
        held.close()


def test_a_chain_naming_a_block_that_is_not_whole_cannot_be_held(manager):
    """A chain whose last block is short cannot be held either.

    The holder commits under the default scope, not the chain's -- a key is a
    digest and does not invert -- so the remainder lands under the wrong root.
    """
    _, keys = _commit_and_name_a_partial_block(manager)
    whole = manager.create_kv_cache_from_keys(keys[:-1])
    try:
        with pytest.raises(ValueError):
            manager.create_kv_cache_from_keys(keys)
    finally:
        whole.close()


def test_a_held_chain_still_refuses_to_plan_a_drop(manager):
    """A holder is left committing, so planning a drop raises.

    The plan would match against the default root rather than the one the chain
    was named in, and that gate is the only thing standing between the two.
    """
    _, keys = _commit_and_name(manager)
    held = manager.create_kv_cache_from_keys(keys)
    try:
        with pytest.raises(Exception) as caught:
            held.plan_committed_block_drop()
        # By name and message, not by class: each backend raises its own
        # LogicError, and a bare Exception would take the AttributeError a
        # rename produces -- which is the regression this is here to catch.
        assert type(caught.value).__name__ == "LogicError"
        assert "stop_committing" in str(caught.value)
    finally:
        held.close()


# ---- pool_group_descs_at ---------------------------------------------------


def test_the_hot_level_layout_is_the_layout_already_published(tiered_manager):
    # The published layout is this call at the hot level. Two copies of the
    # same walk drift, and the one that drifts is the one read less often.
    hot = _introspection.pool_group_descs_at(tiered_manager, GPU_LEVEL)
    assert hot is not None
    assert _layout(hot) == _layout(list(tiered_manager.pool_group_descs))


def test_a_disk_level_has_no_base_address_to_publish(tiered_manager):
    tiers = list(tiered_manager.cache_tier_list)
    # Disk addresses by (fd, offset), which is not a region a transfer agent
    # can be handed.
    assert _introspection.pool_group_descs_at(tiered_manager, tiers.index(CacheTier.DISK)) is None


def test_a_host_level_publishes_its_own_slots_at_the_hot_slot_size(tiered_manager):
    tiers = list(tiered_manager.cache_tier_list)
    hot = _introspection.pool_group_descs_at(tiered_manager, GPU_LEVEL)
    host = _introspection.pool_group_descs_at(tiered_manager, tiers.index(CacheTier.HOST_MEM))
    assert hot is not None and host is not None
    assert len(host) == len(hot)
    for host_desc, hot_desc in zip(host, hot):
        host_pools = list(host_desc.pools)
        hot_pools = list(hot_desc.pools)
        # Same payload, other memory: a reader addressing a host slot as the hot
        # layout has to land on the slot it asked for and not past it.
        assert [int(p.slot_bytes) for p in host_pools] == [int(p.slot_bytes) for p in hot_pools]
        assert all(int(p.base_address) != 0 for p in host_pools)
        assert [int(p.base_address) for p in host_pools] != [int(p.base_address) for p in hot_pools]


def test_a_level_this_build_does_not_have_is_not_memory_either(tiered_manager):
    # Declared as `list[PoolGroupDesc] | None`, and the page-table path catches
    # only TypeError, so anything other than an answer here kills the worker.
    beyond_last = len(list(tiered_manager.cache_tier_list))
    assert _introspection.pool_group_descs_at(tiered_manager, beyond_last) is None
    # Below the first as well as past the last: the level arrives as a plain
    # int, and a negative one clears an upper-bound test on either backend.
    assert _introspection.pool_group_descs_at(tiered_manager, -1) is None


@requires_cpp_backend
def test_a_level_grouped_unlike_the_hot_one_cannot_be_addressed_as_one():
    """A cold-page codec regroups life cycles, and the arithmetic stops holding.

    A caller reaches this tier as "the same groups, one level down". A level
    shaped otherwise does not fail on mismatch -- it lands on an unrelated pool.
    """
    init_cuda_once()
    unit = 1 << 20
    config = KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[GpuCacheTierConfig(quota=24 * unit), GpuCacheTierConfig(quota=32 * unit)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0), buffers=[BufferConfig(role="key", size=2 * unit)]
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role="key", size=2 * unit)],
                sliding_window_size=8,
                num_sink_tokens=0,
            ),
        ],
        initial_pool_ratio=[0.5, 0.5],
        constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=12, history_length=0)])],
        max_util_for_resume=1.0,
    )
    codec = _introspection.create_test_padding_cold_page_codec({0: 2 * unit, 1: 4 * unit})
    regrouped = KVCacheManager(config, cold_page_codec=codec)
    try:
        hot_groups = set(_introspection.life_cycle_pool_group_indices(regrouped, 0))
        cold_groups = set(_introspection.life_cycle_pool_group_indices(regrouped, 1))
        # Precondition: the padding is what splits one hot group into two.
        assert len(hot_groups) != len(cold_groups)
        assert _introspection.pool_group_descs_at(regrouped, 1) is None
    finally:
        regrouped.shutdown()


# ---- the two implementations against each other ----------------------------


def _run_probe(backend: str) -> dict:
    # The answers come back in a file of our own: the child's stdout carries the
    # version banner tensorrt_llm prints on import, so the probe does not own it.
    # Captured all the same -- a probe that fails silently is worse than a noisy one.
    with tempfile.TemporaryDirectory() as tmp_dir:
        answers_path = os.path.join(tmp_dir, f"{backend}.json")
        completed = subprocess.run(
            [sys.executable, content_probe.__file__, answers_path],
            env=dict(os.environ, TLLM_KV_CACHE_MANAGER_V2_BACKEND=backend),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert completed.returncode == 0, (
            f"the {backend} backend could not answer the probe:\n{completed.stderr}"
        )
        # Exited cleanly and wrote nothing: said in as many words, because the
        # alternative is a decode error naming the wrong problem.
        assert os.path.exists(answers_path), (
            f"the {backend} probe exited cleanly without writing its answers\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        with open(answers_path, encoding="utf-8") as fh:
            answers = json.load(fh)
    if "unavailable" in answers:
        pytest.skip(f"the {backend} backend is not built here: {answers['unavailable']}")
    return answers


@pytest.fixture(scope="module")
def backend_answers():
    """Both implementations' answers to one set of questions, from two processes."""
    return content_probe.collect(), _run_probe(_OTHER_BACKEND)


def test_both_backends_name_the_same_chain(backend_answers):
    here, there = backend_answers
    # A name derived two ways that disagree anywhere misses every time, and
    # says nothing when it does.
    assert there["tokens_per_block"] == here["tokens_per_block"]
    assert there["life_cycles"] == here["life_cycles"]
    assert there["keys"] == here["keys"]


def test_both_backends_locate_the_same_chain(backend_answers):
    here, there = backend_answers
    assert there["servable"] == here["servable"]
    assert there["held_page_counts"] == here["held_page_counts"]
    # Answered, not refused, and answered as the fork: neither backend read the
    # subclass as a different name.
    assert here["servable"]["forked_subclass"] == here["servable"]["forked"]


def test_both_backends_refuse_an_empty_chain_the_same_way(backend_answers):
    here, there = backend_answers
    # Refused rather than reconciled, in the one entry point whose whole
    # purpose is that both give the same answer -- so refusing differently is
    # the same defect one level down.
    assert there["empty_chain_refusal"] is not None
    assert here["empty_chain_refusal"] is not None
    assert there["empty_chain_refusal"] == here["empty_chain_refusal"]


def test_both_backends_refuse_the_same_names(backend_answers):
    here, there = backend_answers
    # A guard only one backend has is a peer that reads a fraction of a block,
    # or a page nobody vouched for, as if it were content.
    assert all(answer.startswith("raised:") for answer in here["refusals"].values())
    assert there["refusals"] == here["refusals"]


def test_both_backends_leave_a_holder_unable_to_plan_a_drop(backend_answers):
    here, there = backend_answers
    # A drop plan matches on the holder's scope, which is not the chain's.
    assert here["held_drop_plan"].startswith("raised:")
    assert there["held_drop_plan"] == here["held_drop_plan"]


def test_both_backends_address_the_same_levels(backend_answers):
    here, there = backend_answers
    assert there["cache_tiers"] == here["cache_tiers"]
    assert here["hot_matches_published"] and there["hot_matches_published"]
    # Which levels are memory, and how many pools each publishes. A level one
    # backend calls addressable and the other does not is a page table that is
    # right on one worker and wrong on its peer.
    assert there["pool_groups"] == here["pool_groups"]
