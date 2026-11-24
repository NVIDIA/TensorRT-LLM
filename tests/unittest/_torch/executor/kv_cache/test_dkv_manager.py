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
"""KVCacheManagerV2(dkv=True) on the CPU: the group, the reuse depth every rank can serve,
the per-step page-table exchange, the snapshot unit and the token <-> byte conversions."""

import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

TPB = 64


def _bare(**attributes) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    for name, value in attributes.items():
        setattr(manager, name, value)
    return manager


class _Recorder:
    """A one-rank stand-in for the group's all-gather: records what the rank sends."""

    def __init__(self, size: int, others=None) -> None:
        self.size, self.sent, self._others = size, [], others or []

    def __call__(self, obj):
        self.sent.append(obj)
        return [obj, *self._others] if self._others else [obj] * self.size


def _mapping(**overrides):
    fields = dict(tp_size=1, cp_size=8, tp_rank=0, cp_rank=3, pp_size=1, enable_attention_dp=False)
    fields.update(overrides)
    helix = fields.pop("helix", False)
    return SimpleNamespace(**fields, has_cp_helix=lambda: helix)


def test_the_group_is_the_attention_tp_group(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm

    monkeypatch.setattr(kvm.Distributed, "get", staticmethod(lambda mapping: SimpleNamespace()))
    dkv = KVCacheManagerV2._make_dkv(_mapping(), TPB)
    assert (dkv.size, dkv.rank, dkv.dkv_tokens_per_block) == (8, 3, 512)
    dkv = KVCacheManagerV2._make_dkv(_mapping(tp_size=2, cp_size=4, tp_rank=1), TPB)
    assert (dkv.size, dkv.rank) == (8, 7)
    for bad in (dict(helix=True), dict(enable_attention_dp=True), dict(pp_size=2)):
        with pytest.raises(ValueError, match="dkv"):
            KVCacheManagerV2._make_dkv(_mapping(**bad), TPB)


@pytest.mark.parametrize("partial", [False, True], ids=["whole_blocks", "partial"])
def test_the_reuse_depth_is_the_group_minimum(partial):
    """Each rank probes its own tree; the group reuses the smallest match: in tokens with partial
    reuse, rounded down to whole DKV blocks without."""
    gather = _Recorder(4, others=[1280, 9984, 600 if partial else 512])
    dkv = Dkv(4, 0, TPB, gather)
    probed = []

    def probe(tokens, lora_task_id=None, cache_salt=None):
        probed.append(len(tokens))
        return 1000

    manager = _bare(
        _dkv=dkv,
        enable_partial_reuse=partial,
        _context_reuse_tokens=lambda req, limit: list(range(2000 if limit is None else limit)),
        probe_prefix_match_length=probe,
    )
    req = SimpleNamespace(py_request_id=7, lora_task_id=None, cache_salt=None)
    # This rank holds 1000 tokens (768 in whole 256-token DKV blocks); another holds less.
    if partial:
        assert manager._dkv_agreed_reuse_limit(req, None) == 600
        assert gather.sent == [1000] and probed == [2000]
    else:
        assert manager._dkv_agreed_reuse_limit(req, None) == 512
        assert gather.sent == [768] and probed == [2000]


class _Group:
    """One rank's side of the group's all-gathers, in call order: records what this rank sends;
    ``answers[i]`` are the other ranks' objects of call i (None: the same as this rank's)."""

    def __init__(self, size, answers=()):
        self.size, self.sent, self._answers = size, [], list(answers)

    def __call__(self, obj):
        self.sent.append(obj)
        others = self._answers.pop(0) if self._answers else None
        return [obj, *(others if others is not None else [obj] * (self.size - 1))]


class _Pins:
    """The native manager's side of the reuse pins: suspended caches created and closed.

    ``claims`` caps what a pin claims: a number, or a function of its lookup length.
    """

    def __init__(self, claims, forks=None):
        self.claims, self.live, self.excluded = claims, {}, set()
        # (where a lookup of n tokens leaves the tree, the depth attention pages support there)
        self.forks = forks or (lambda n: (n, n))

    def claim(self, num_tokens):
        if callable(self.claims):
            return self.claims(num_tokens)
        return min(num_tokens, self.claims)

    def mark_stats_excluded(self, pin_id):
        self.excluded.add(pin_id)

    def clear_stats_excluded(self, pin_id):
        self.excluded.discard(pin_id)

    def create_kv_cache(self, scope, tokens, id=None):
        pins = self

        class _Pin:
            num_committed_tokens = pins.claim(len(tokens))
            cuda_stream = None

            def _get_num_reusable_tokens_before_pruning(self):
                return pins.forks(len(tokens))[0]

            def _get_num_reusable_tokens_before_hybrid_pruning(self):
                return pins.forks(len(tokens))[1]

            def discard_pending_stats(self):
                pass

            def close(self):
                del pins.live[id]

        pins.live[id] = (len(tokens), _Pin())
        return pins.live[id][1]


def _dkv_manager(gather, depths, partial=True, claims=10**9, **extra):
    """A one-rank DKV manager whose radix tree holds ``depths[request id]`` tokens of each prompt."""
    probed = []

    def probe(req, limit):
        probed.append((req.py_request_id, limit))
        depth = depths[req.py_request_id]
        return depth if limit is None else min(depth, limit)

    manager = _bare(
        _dkv=Dkv(4, 0, TPB, gather),
        enable_block_reuse=True,
        enable_partial_reuse=partial,
        kv_cache_map={},
        impl=_Pins(claims),
        _stream=SimpleNamespace(cuda_stream=0),
        _spec_recompute_tail=0,
        _context_reuse_tokens=lambda req, limit=None: list(range(5000 if limit is None else limit)),
        _derive_reuse_salt=lambda salt: salt,
        _probe_reuse_depth=probe,
        **extra,
    )
    return manager, probed


def _ctx(rid, first=True):
    return SimpleNamespace(
        py_request_id=rid, is_first_context_chunk=first, lora_task_id=None, cache_salt=None
    )


@pytest.mark.parametrize("partial", [False, True], ids=["whole_blocks", "partial"])
def test_a_step_agrees_on_every_new_context_with_one_collective(partial):
    """Every first chunk without a cache is probed first, then the group takes each request's
    minimum in one all-gather; the step's claims then need no collective, whatever their limits."""
    # The other ranks (their depths rounded as this rank's): request 1 agrees at another rank's
    # depth, request 2 at this rank's own, request 3 has no reuse anywhere; nobody probes request 5
    # (a later chunk) or 6 (cache held).
    low = 900 if partial else 512
    others = [[low, 4000, 0], [1000, 3072, 0], [3072, 3072, 0]]
    gather = _Group(4, answers=[others])
    manager, probed = _dkv_manager(gather, {1: 2000, 2: 1500, 3: 0, 6: 700}, partial)
    manager.kv_cache_map[6] = object()
    requests = [_ctx(1), _ctx(2), _ctx(3), _ctx(5, first=False), _ctx(6)]
    manager.agree_dkv_context_reuse(requests)
    one = 1500 if partial else 1280  # 1500 in whole 256-token DKV blocks
    assert gather.sent == [[2000 if partial else 1792, one, 0]]
    assert manager._dkv_agreed == {1: low, 2: one, 3: 0}
    assert sorted(rid for rid, _ in probed) == [1, 2, 3]
    # The claims: capped by their own limits, no further collective.
    assert manager._dkv_agreed_reuse_limit(requests[0], None) == low
    assert manager._dkv_agreed_reuse_limit(requests[1], 1200) == (1200 if partial else 1024)
    assert manager._dkv_agreed_reuse_limit(requests[1], 0) == 0
    assert manager.probe_context_reuse(requests[0]) == low
    assert len(gather.sent) == 1
    # Nothing is held until the scheduler pins what the step may claim; one collective checks
    # that every rank's pin claims its agreed depth.
    assert manager.impl.live == {}
    manager.pin_dkv_context_reuse(requests)
    assert gather.sent[1:] == [[low, one]]
    assert manager._dkv_reuse_stats == (2, low + one, 0, 0)
    assert sorted(n for n, _ in manager.impl.live.values()) == sorted([low, one])
    assert manager.impl.excluded == set(manager.impl.live)
    assert [manager.dkv_claim_ready(r) for r in requests] == [True] * 5
    manager.release_dkv_reuse_pins()
    assert manager.impl.live == {} and manager.impl.excluded == set()
    assert manager._dkv_agreed == {}


def test_a_step_without_new_contexts_runs_no_collective():
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {})
    manager.agree_dkv_context_reuse([_ctx(5, first=False)])
    manager.pin_dkv_context_reuse([_ctx(5, first=False)])
    assert gather.sent == []


def test_the_agreed_depth_is_one_both_paired_pools_can_claim():
    """With a paired draft pool, each rank steps down to a depth both its pools can claim (each
    probe returns a claimable depth), and the draft prefix is pinned as well."""
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 2000})
    # The draft pool claims whole 256-token blocks of what it holds (1900 tokens).
    draft = _bare(
        kv_cache_map={},
        impl=_Pins(10**9),
        _stream=SimpleNamespace(cuda_stream=0),
        _context_reuse_tokens=lambda req, limit=None: list(range(5000 if limit is None else limit)),
        _derive_reuse_salt=lambda salt: salt,
        _probe_reuse_depth=lambda req, limit: min(1900, 10**9 if limit is None else limit)
        // 256
        * 256,
    )
    manager.agree_dkv_context_reuse([_ctx(1)], draft)
    manager.pin_dkv_context_reuse([_ctx(1)], draft)
    assert gather.sent == [[1792], [1792]]
    assert manager._dkv_agreed == {1: 1792}
    assert [n for n, _ in draft.impl.live.values()] == [1792]
    manager.release_dkv_reuse_pins()
    assert draft.impl.live == {}


def _short_inside_blocks(num_tokens):
    """A lookup cut inside a 1024-token block falls back to the block before it (the radix
    tree's partial-match pass is skipped at a node with 32 or more children)."""
    return num_tokens if num_tokens % 1024 == 0 else num_tokens // 1024 * 1024


def test_a_pin_that_claims_less_moves_the_group_to_its_claim():
    """This rank probed 2500 tokens, but its pin, looked up only to the agreed depth, claims the
    1024 before the block that depth ends inside: the group moves to 1024 and pins it again."""
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 2500}, claims=_short_inside_blocks)
    req = _ctx(1)
    manager.agree_dkv_context_reuse([req])
    manager.pin_dkv_context_reuse([req])
    assert gather.sent == [[2500], [2048], [2048]]
    assert manager._dkv_agreed == {1: 2048}
    assert [n for n, _ in manager.impl.live.values()] == [2048]
    assert manager.dkv_claim_ready(req)
    assert manager._dkv_agreed_reuse_limit(req, None) == 2048
    # One request checked at 2500 agreed tokens, lowered by 452; a later check of the same
    # request (it was not admitted) is not counted again.
    assert manager._dkv_reuse_stats == (1, 2500, 1, 452)
    manager.release_dkv_reuse_pins()
    manager.agree_dkv_context_reuse([req])
    manager.pin_dkv_context_reuse([req])
    assert manager._dkv_reuse_stats == (1, 2500, 1, 452)


def test_another_rank_s_short_claim_moves_this_rank_too():
    gather = _Group(4, answers=[None, [[2000], [512], [2000]]])
    manager, _ = _dkv_manager(gather, {1: 2000})
    req = _ctx(1)
    manager.agree_dkv_context_reuse([req])
    manager.pin_dkv_context_reuse([req])
    assert gather.sent == [[2000], [2000], [512]]
    assert manager._dkv_agreed == {1: 512}
    assert [n for n, _ in manager.impl.live.values()] == [512]


def test_a_depth_the_ranks_never_agree_on_reuses_nothing():
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 2000}, claims=lambda n: max(0, n - 64))
    req = _ctx(1)
    manager.agree_dkv_context_reuse([req])
    manager.pin_dkv_context_reuse([req], max_rounds=3)
    assert len(gather.sent) == 4  # the agreement and three checks
    assert manager._dkv_agreed == {1: 0} and manager.impl.live == {}
    assert manager.dkv_claim_ready(req)
    assert manager._dkv_agreed_reuse_limit(req, None) == 0


def test_only_a_verified_or_empty_claim_is_ready():
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 0, 2: 500, 3: 600})
    manager.kv_cache_map[4] = object()
    requests = [_ctx(1), _ctx(2), _ctx(3)]
    manager.agree_dkv_context_reuse(requests)
    manager.pin_dkv_context_reuse(requests[2:])
    assert [manager.dkv_claim_ready(r) for r in requests] == [True, False, True]
    # Held cache, later chunk: ready; a first chunk the agreement did not cover: not ready.
    assert manager.dkv_claim_ready(_ctx(4))
    assert manager.dkv_claim_ready(_ctx(6, first=False))
    assert not manager.dkv_claim_ready(_ctx(5))


def test_unpinning_frees_every_other_prefix_and_defers_the_unclaimed():
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 500, 2: 600, 3: 700})
    requests = [_ctx(1), _ctx(2), _ctx(3)]
    manager.agree_dkv_context_reuse(requests)
    manager.pin_dkv_context_reuse(requests)
    manager.kv_cache_map[1] = object()  # request 1 claimed already
    assert manager.unpin_dkv_context_reuse(keep=2)
    assert [n for n, _ in manager.impl.live.values()] == [600]
    assert manager._dkv_agreed == {1: 500, 2: 600}
    assert manager.dkv_claim_ready(requests[1]) and not manager.dkv_claim_ready(requests[2])
    assert not manager.unpin_dkv_context_reuse(keep=2)
    manager.release_dkv_reuse_pins()
    assert manager.impl.live == {} and manager.impl.excluded == set()


class _ThreadGroup:
    """The all-gather of ranks that run as threads of this process."""

    def __init__(self, size):
        self._barrier = threading.Barrier(size)
        self._objects = [None] * size

    def rank(self, rank):
        def all_gather(obj):
            self._objects[rank] = obj
            self._barrier.wait()
            gathered = list(self._objects)
            self._barrier.wait()
            return gathered

        return all_gather


def test_ranks_with_different_trees_claim_the_same_depth():
    """Regression (cep16 "this rank's pool claims 0"): rank 1's tree holds the prompt deeper, but
    the request's prefix node there has many children, so a lookup cut at the agreed 2500 tokens
    claims only 2048 on it while rank 0 claims 2500. The pins' check makes both claim 2048."""
    group = _ThreadGroup(2)
    results = {}

    def run(rank, depth, claims):
        manager, _ = _dkv_manager(None, {1: depth}, claims=claims)
        manager._dkv = Dkv(2, rank, TPB, group.rank(rank))
        req = _ctx(1)
        manager.agree_dkv_context_reuse([req])
        first = {rid: manager.impl.claim(n) for rid, n in manager._dkv_agreed.items()}
        manager.pin_dkv_context_reuse([req])
        claimed = [pin.num_committed_tokens for _, pin in manager.impl.live.values()]
        results[rank] = (first, dict(manager._dkv_agreed), claimed)

    threads = [
        threading.Thread(target=run, args=(0, 2500, 10**9)),
        threading.Thread(target=run, args=(1, 4000, _short_inside_blocks)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    # Without the check the ranks would claim different depths for the same request.
    assert results[0][0] == {1: 2500} and results[1][0] == {1: 2048}
    assert results[0][1:] == results[1][1:] == ({1: 2048}, [2048])


def test_the_exchange_sends_only_what_changed():
    gather = _Recorder(2)
    dkv = Dkv(2, 0, TPB, gather)
    pages = {7: [4, 5, 6]}
    caches = {
        7: SimpleNamespace(num_blocks=3, get_base_page_indices=lambda lc: pages[7] + [-1]),
    }
    manager = _bare(_dkv=dkv, _dkv_life_cycles=(1,), _dkv_sent={}, kv_cache_map=caches)
    batch = SimpleNamespace(
        context_requests=[SimpleNamespace(py_request_id=7)], generation_requests=[]
    )
    manager._exchange_dkv_pages(batch)
    caches[7].num_blocks = 4
    pages[7] = [4, 9, 6, 8]  # block 1 moved, block 3 is new
    manager._exchange_dkv_pages(batch)
    manager._exchange_dkv_pages(batch)  # nothing changed: an empty exchange, still collective
    assert gather.sent == [
        ([((7, 1), 0, [4, 5, 6])], False),
        ([((7, 1), 1, [9, 6, 8])], False),
        ([], False),
    ]
    assert dkv.peer_pages((7, 1)) == [[4, 9, 6, 8], [4, 9, 6, 8]]


def _native(levels=(0, 0), committed=0, history=0, tpb=512):
    return SimpleNamespace(
        cached_tokens_by_level=list(levels),
        num_committed_tokens=committed,
        history_length=history,
        manager=SimpleNamespace(tokens_per_block=tpb),
    )


def test_resume_copies_only_restores_and_partial_blocks():
    from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import resume_copies

    # Fresh: a reuse whose pages all sat on the GPU, in whole blocks, copies nothing.
    assert not resume_copies(_native((0, 0), 0), fresh=True)
    assert not resume_copies(_native((1024, 0), 1024), fresh=True)
    # Pages back from the host tier, or the private copy of the block a match ends inside.
    assert resume_copies(_native((512, 512), 1024), fresh=True)
    assert resume_copies(_native((1000, 0), 1000), fresh=True)
    # A cache resumed again may get its pages back from the host tier once it holds history.
    assert not resume_copies(_native(history=0), fresh=False)
    assert resume_copies(_native(history=64), fresh=False)


def test_a_resume_that_copies_fences_the_next_exchange():
    """The manager notes the copies of a resume (fresh, then again after a suspension) and sends
    the flag with its next page exchange; the flag does not outlive that exchange."""
    gather = _Recorder(2)
    dkv = Dkv(2, 0, TPB, gather)
    native = _native((128, 128), 256, tpb=128)
    native.is_active = False

    def resume(stream):
        native.is_active = True
        return True

    native.resume = resume
    manager = _bare(
        _dkv=dkv,
        _dkv_resumed=set(),
        _dkv_life_cycles=(),
        _dkv_sent={},
        kv_cache_map={},
        _cache_stream=lambda rid: 0,
        _restore_page_index_bufs=lambda rid, kv: None,
    )
    batch = SimpleNamespace(context_requests=[], generation_requests=[])
    assert manager._resume_and_restore(7, native)
    manager._exchange_dkv_pages(batch)
    manager._exchange_dkv_pages(batch)
    assert [copied for _, copied in gather.sent] == [True, False]
    assert dkv.take_fence() and not dkv.take_fence()
    # Fresh, the reuse came whole from the GPU: no copy. Resumed again with history: a copy.
    native.cached_tokens_by_level, native.is_active = [256, 0], False
    native.history_length = 0
    assert manager._resume_and_restore(8, native)
    manager._exchange_dkv_pages(batch)
    native.is_active, native.history_length = False, 300
    assert manager._resume_and_restore(8, native)
    manager._exchange_dkv_pages(batch)
    assert [copied for _, copied in gather.sent[2:]] == [False, True]


def test_a_sender_reads_its_own_slots_as_they_are_now_not_as_last_exchanged():
    """A finished context leaves the batch, so no exchange follows.

    When the manager then moves a block (seen at send time on cep8 -> helix8), the exchanged own
    table names a slot the request no longer holds; the sender reads the manager's current slots.
    """
    dkv = Dkv(2, 0, TPB, _Recorder(2))
    pages = {7: [166, 165, 9]}
    caches = {7: SimpleNamespace(num_blocks=3, get_base_page_indices=lambda lc: pages[7] + [-1])}
    manager = _bare(_dkv=dkv, _dkv_life_cycles=(1,), _dkv_sent={}, kv_cache_map=caches)
    batch = SimpleNamespace(
        context_requests=[SimpleNamespace(py_request_id=7)], generation_requests=[]
    )
    manager._exchange_dkv_pages(batch)
    pages[7] = [38, 37, 9]  # moved after the request's last step
    assert dkv.peer_pages((7, 1))[0] == [166, 165, 9]
    assert manager.dkv_own_slots(7, 1) == [38, 37, 9]


def test_snapshot_points_round_to_the_dkv_block_only_without_partial_reuse():
    plain = _bare(_dkv=None, tokens_per_block=TPB, _ledger_tokens_per_block=TPB)

    def dkv(partial):
        return _bare(
            _dkv=Dkv(8, 0, TPB, _Recorder(8)),
            enable_partial_reuse=partial,
            tokens_per_block=TPB,
            _ledger_tokens_per_block=8 * TPB,
        )

    assert plain._reuse_tokens_per_block == TPB
    assert dkv(False)._reuse_tokens_per_block == 512
    assert dkv(True)._reuse_tokens_per_block == TPB


def test_token_capacity_counts_global_tokens(monkeypatch):
    """A rank's budget holds its own page of each DKV block: global tokens = local x size."""
    monkeypatch.setattr(
        KVCacheManagerV2, "_get_max_tokens_from_quota_impl", lambda self, quota: quota / 10
    )
    monkeypatch.setattr(
        KVCacheManagerV2, "_get_quota_from_max_tokens_impl", lambda self, tokens: tokens * 10
    )
    # The manager reads the shard size off its DKV group (_kv_shard_size_of).
    manager = _bare(
        _dkv=Dkv(8, 0, TPB, _Recorder(8)), tokens_per_block=TPB, _ledger_tokens_per_block=512
    )
    # 100000 bytes -> 10000 local tokens -> 156 whole pages -> 156 * 64 * 8 global tokens.
    assert manager._get_max_tokens_from_quota(100_000) == 156 * 64 * 8
    # 1000 global tokens -> 2 DKV blocks -> 2 local pages of 64 tokens -> 1280 bytes.
    assert manager._get_quota_from_max_tokens(1000) == 2 * 64 * 10
    one = _bare(_dkv=None, tokens_per_block=TPB, _ledger_tokens_per_block=TPB)
    assert one._get_max_tokens_from_quota(100_000) == 10_000


def test_a_snapshot_interval_off_the_dkv_block_is_refused():
    """Without partial reuse a DKV resumes on whole DKV blocks, so periodic snapshots must sit
    on them (the manager checks only then)."""

    def config(interval, reuse=True):
        return SimpleNamespace(
            enable_block_reuse=reuse,
            mamba_state_config=SimpleNamespace(periodic_snapshot_interval=interval),
        )

    check = KVCacheManagerV2._check_dkv_snapshot_interval
    for ok in (config(0), config(1024), config(256, reuse=False)):
        check(ok, 512)
    with pytest.raises(ValueError, match="multiple of DKV block"):
        check(config(256), 512)


def test_the_group_agrees_on_each_candidate_s_fork_in_the_first_check():
    """With branch snapshots, every request the step may claim (agreed at 0 too) has its fork
    measured with the uncut lookup; the first check's collective carries the forks and the group
    keeps each one's minimum."""
    gather = _Group(4, answers=[None, [[2000, 1000, 2500, 1000, 2300]] * 3])
    manager, _ = _dkv_manager(gather, {1: 0, 2: 2000}, _commit_branch_snapshots=True)
    own = {1: (900, 900, 4999), 2: (2600, 2400, 4999)}
    manager._probe_dkv_fork = lambda req: own[req.py_request_id]
    requests = [_ctx(1), _ctx(2)]
    manager.agree_dkv_context_reuse(requests)
    manager.pin_dkv_context_reuse(requests)
    assert gather.sent == [[0, 2000], [2000, 900, 2600, 900, 2400]]
    assert manager._dkv_forks == {1: (900, 900, 4999), 2: (2500, 2300, 4999)}
    assert manager._dkv_agreed == {1: 0, 2: 2000}
    req = requests[1]
    assert manager._branch_fork(req, None, 0) == (2500, 2300, 4999)
    manager.release_dkv_reuse_pins()
    assert manager._dkv_forks == {}
    # A request that waits keeps its first measurement: the next step does not probe it again.
    manager._probe_dkv_fork = None
    manager.agree_dkv_context_reuse(requests)
    manager.pin_dkv_context_reuse(requests)
    assert gather.sent[-1] == [2000, 900, 2600, 900, 2400]


def test_without_branch_snapshots_no_fork_is_measured():
    gather = _Group(4)
    manager, _ = _dkv_manager(gather, {1: 0})
    manager.agree_dkv_context_reuse([_ctx(1)])
    manager.pin_dkv_context_reuse([_ctx(1)])
    assert gather.sent == [[0]] and manager._dkv_forks == {}


def test_a_fork_probe_reads_the_uncut_lookup_and_closes():
    manager, _ = _dkv_manager(
        _Group(4), {}, _commit_branch_snapshots=True, _context_claim_limit=lambda req: None
    )
    manager.impl.forks = lambda n: (n - 1700, n - 1900)
    assert manager._probe_dkv_fork(_ctx(1)) == (3300, 3100, 5000)
    assert manager.impl.live == {} and manager.impl.excluded == set()


def test_a_claim_without_an_agreed_fork_keeps_its_own_match():
    manager, _ = _dkv_manager(_Group(4), {})
    claim = SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: 700,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: 640,
    )
    assert manager._branch_fork(_ctx(1), claim, 700) == (700, 640, 700)
