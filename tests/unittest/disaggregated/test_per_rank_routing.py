# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for per-rank hierarchical routing in the centralized KV-cache router."""

import time

import pytest

from tensorrt_llm.serve.kv_cache_router import (CentralizedKVCacheRouter,
                                                KvCacheEventReport,
                                                KVCacheRouterServer,
                                                WorkerLoadReport,
                                                WorkerReporter)

TPB = 32


def stored_event(parent_hash, block_hashes, layer_group_id=None):
    ev = {
        "event_id": 0,
        "data": {
            "type": "stored",
            "parent_hash": parent_hash,
            "blocks": [{"block_hash": h} for h in block_hashes],
        },
        "window_size": 0,
    }
    if layer_group_id is not None:
        ev["layer_group_id"] = layer_group_id
    return ev


def removed_event(block_hashes, layer_group_id=None):
    ev = {
        "event_id": 0,
        "data": {
            "type": "removed",
            "block_hashes": list(block_hashes),
        },
        "window_size": 0,
    }
    if layer_group_id is not None:
        ev["layer_group_id"] = layer_group_id
    return ev


def load_report(worker_id, seq, active=0, queued=0, max_bs=64, ns="ctx"):
    return WorkerLoadReport(worker_id=worker_id,
                            namespace=ns,
                            seq=seq,
                            num_active_requests=active,
                            num_queued_requests=queued,
                            max_batch_size=max_bs)


# ------------------------------------------------- parse_worker_id


def test_parse_worker_id_composite():
    from tensorrt_llm.serve.kv_cache_router.router_core import _parse_worker_id
    assert _parse_worker_id("inst_abc:rank0") == ("inst_abc", 0)
    assert _parse_worker_id("inst_abc:rank7") == ("inst_abc", 7)
    assert _parse_worker_id("my-inst-id:rank12") == ("my-inst-id", 12)


def test_parse_worker_id_legacy():
    from tensorrt_llm.serve.kv_cache_router.router_core import _parse_worker_id
    assert _parse_worker_id("legacy_worker") == ("legacy_worker", None)
    assert _parse_worker_id("no_rank_suffix") == ("no_rank_suffix", None)


def test_parse_worker_id_colon_in_instance():
    from tensorrt_llm.serve.kv_cache_router.router_core import _parse_worker_id
    # Instance ID itself contains colons (e.g. UUID-like)
    assert _parse_worker_id("a:b:c:rank3") == ("a:b:c", 3)


# ------------------------------------------------- per-rank event ingest


def test_per_rank_events_build_combined_trie():
    """Per-rank reports with :rankN suffix build both rank trie and combined trie."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # Rank 0 holds blocks [1, 2, 3]
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    # Rank 1 holds blocks [1, 2, 4]
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 4])]))

    # Load reports for both ranks
    router.apply_load_report(load_report("inst0:rank0", 0))
    router.apply_load_report(load_report("inst0:rank1", 0))

    # Register instance address
    router.register_worker_address("inst0", "http://host0:8000")

    # Query [1, 2, 3] — rank0 is a better match
    sel = router.select_worker("ctx", [1, 2, 3])
    assert sel is not None
    assert sel.worker_id == "inst0"
    assert sel.address == "http://host0:8000"
    assert sel.matched_blocks == 3  # combined trie sees [1,2,3]
    assert sel.dp_rank == 0  # rank0 has better per-rank match


def test_per_rank_selects_best_rank_by_match():
    """Phase 2 picks the rank with the longest prefix match."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # Rank 0: blocks [1, 2]
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    # Rank 1: blocks [1, 2, 3, 4]
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3, 4])]))

    router.apply_load_report(load_report("inst0:rank0", 0))
    router.apply_load_report(load_report("inst0:rank1", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    sel = router.select_worker("ctx", [1, 2, 3, 4])
    assert sel.dp_rank == 1  # rank1 has full match


def test_per_rank_selects_by_load_when_match_equal():
    """When cache match is equal, phase 2 prefers lower load."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB, load_weight=1.0)

    # Both ranks have same blocks
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))

    # rank0 is busy, rank1 is idle
    router.apply_load_report(load_report("inst0:rank0", 0, active=10))
    router.apply_load_report(load_report("inst0:rank1", 0, active=0))
    router.register_worker_address("inst0", "http://host0:8000")

    sel = router.select_worker("ctx", [1, 2, 3])
    assert sel.dp_rank == 1  # rank1 is idle


# ------------------------------------------------- instance-level routing


def test_two_instances_selects_better_match():
    """Phase 1 picks the instance with the better combined trie match."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # inst0 has [1, 2, 3, 4] across ranks
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3, 4])]))
    router.apply_load_report(load_report("inst0:rank0", 0))

    # inst1 has [1, 2] only
    router.apply_event_report(
        KvCacheEventReport("inst1:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_load_report(load_report("inst1:rank0", 0))

    router.register_worker_address("inst0", "http://host0:8000")
    router.register_worker_address("inst1", "http://host1:8000")

    sel = router.select_worker("ctx", [1, 2, 3, 4])
    assert sel.worker_id == "inst0"
    assert sel.matched_blocks == 4


def test_two_instances_load_breaks_tie():
    """Phase 1 prefers lower instance load when cache match is equal."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB, load_weight=1.0)

    # Both instances have same blocks
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_event_report(
        KvCacheEventReport("inst1:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))

    # inst0 is loaded, inst1 is idle
    router.apply_load_report(load_report("inst0:rank0", 0, active=20))
    router.apply_load_report(load_report("inst1:rank0", 0, active=0))

    router.register_worker_address("inst0", "http://host0:8000")
    router.register_worker_address("inst1", "http://host1:8000")

    sel = router.select_worker("ctx", [1, 2])
    assert sel.worker_id == "inst1"


# ------------------------------------------------- combined trie refcount


def test_combined_trie_refcount_across_ranks():
    """Hash is only removed from combined trie when NO rank holds it."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # Both ranks store block hash 1
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 3])]))

    # Remove hash 1 from rank0 only
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=1,
                           events=[removed_event([1, 2])]))

    router.apply_load_report(load_report("inst0:rank0", 0))
    router.apply_load_report(load_report("inst0:rank1", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    # Combined trie still has hash 1 (from rank1)
    sel = router.select_worker("ctx", [1, 3])
    assert sel.matched_blocks == 2  # rank1 still has both


def test_combined_trie_removes_when_all_ranks_evict():
    """Hash removed from combined trie once all ranks remove it."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))

    # Both ranks remove hash 1
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=1,
                           events=[removed_event([1])]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=1,
                           events=[removed_event([1])]))

    router.apply_load_report(load_report("inst0:rank0", 0))
    router.apply_load_report(load_report("inst0:rank1", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    # Hash 1 is gone from combined trie — match starts at hash 2
    sel = router.select_worker("ctx", [1, 2])
    assert sel.matched_blocks == 0  # prefix match breaks at hash 1


# ------------------------------------------------- mixed legacy + per-rank


def test_legacy_and_per_rank_coexist():
    """Legacy workers (no :rank suffix) compete alongside per-rank instances."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # Legacy worker with good match
    router.apply_event_report(
        KvCacheEventReport("legacy_w", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3, 4, 5])]))
    router.apply_load_report(load_report("legacy_w", 0))
    router.register_worker_address("legacy_w", "http://legacy:8000")

    # Per-rank instance with partial match
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_load_report(load_report("inst0:rank0", 0))
    router.register_worker_address("inst0", "http://inst0:8000")

    sel = router.select_worker("ctx", [1, 2, 3, 4, 5])
    # Legacy worker has better match (5 blocks vs 2)
    assert sel.worker_id == "legacy_w"
    assert sel.dp_rank is None  # legacy has no dp_rank


# ------------------------------------------------- suspend/stale per-rank


def test_per_rank_suspend_on_stale_load():
    """Instance suspended when all ranks' load reports are stale."""
    clk = {"t": 1000.0}
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      load_suspend_s=3.0,
                                      clock=lambda: clk["t"])

    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("inst0:rank0", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    assert router.select_worker("ctx", [1, 2, 3]) is not None

    # Advance past load_suspend_s
    clk["t"] += 4.0
    assert router.select_worker("ctx", [1, 2, 3]) is None

    # Fresh load revives it
    router.apply_load_report(load_report("inst0:rank0", 1))
    assert router.select_worker("ctx", [1, 2, 3]) is not None


def test_per_rank_stale_eviction():
    """Instance evicted after stale_timeout_s with no reports."""
    clk = {"t": 1000.0}
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      stale_timeout_s=10.0,
                                      clock=lambda: clk["t"])

    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))
    router.apply_load_report(load_report("inst0:rank0", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    clk["t"] += 11.0
    router.evict_stale_workers()
    # Instance fully removed
    assert router.select_worker("ctx", [1, 2]) is None


# ------------------------------------------------- layer group refcount per-rank


def test_per_rank_layer_group_refcount():
    """Multi-layer-group stored events tracked per rank; removal only on last."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    # Two layer groups store the same hash on rank0
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2], layer_group_id=0)]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=1,
                           events=[stored_event(None, [1, 2], layer_group_id=1)]))

    # Remove from layer_group 0 only
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=2,
                           events=[removed_event([1, 2], layer_group_id=0)]))

    router.apply_load_report(load_report("inst0:rank0", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    # Still matched — layer_group 1 retains
    sel = router.select_worker("ctx", [1, 2])
    assert sel.matched_blocks == 2


def test_combined_trie_drops_block_after_all_layer_groups_evict():
    """Regression: a block stored under multiple layer groups on one rank must
    disappear from the instance combined trie once ALL its layer groups evict.

    The combined refcount counts distinct rank-holders, so it must be bumped
    once per rank acquisition -- NOT once per stored event. A block is re-emitted
    once per layer group, so counting events over-counts by the layer-group
    factor while the removed path decrements once per rank, leaving the count
    stuck >0 and the evicted block stale in the combined trie. Stale entries
    mis-route phase-1 to an instance that no longer holds the prefix, which
    silently tanks the cache-hit rate. This guards 'none' mode specifically,
    where ONLY the combined trie is consulted."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      rank_routing_algo="none")
    router.register_worker_address("inst0", "http://host0:8000")

    # Same block stored under two layer groups on the same rank.
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2], layer_group_id=0)]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=1,
                           events=[stored_event(None, [1, 2], layer_group_id=1)]))
    router.apply_load_report(load_report("inst0:rank0", 0))

    sel = router.select_worker("ctx", [1, 2])
    assert sel is not None and sel.matched_blocks == 2

    # Evict BOTH layer groups -> the rank no longer holds the block at all.
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=2,
                           events=[removed_event([1, 2], layer_group_id=0)]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=3,
                           events=[removed_event([1, 2], layer_group_id=1)]))

    # Combined trie must now match nothing -- the block is fully evicted.
    sel = router.select_worker("ctx", [1, 2])
    assert sel is None or sel.matched_blocks == 0, (
        f"evicted block still routable (matched_blocks="
        f"{getattr(sel, 'matched_blocks', None)}) -- stale combined-trie entry")


# ------------------------------------------------- full snapshot per-rank


def test_per_rank_full_snapshot_replaces():
    """Full snapshot replaces a rank's trie and adjusts combined trie."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)

    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_event_report(
        KvCacheEventReport("inst0:rank1", "ctx", seq=0,
                           events=[stored_event(None, [1, 2])]))

    # rank0 sends full snapshot with different blocks
    router.apply_event_report(
        KvCacheEventReport("inst0:rank0", "ctx", seq=1,
                           events=[stored_event(None, [9, 10])],
                           is_full_snapshot=True))

    router.apply_load_report(load_report("inst0:rank0", 0))
    router.apply_load_report(load_report("inst0:rank1", 0))
    router.register_worker_address("inst0", "http://host0:8000")

    # Old rank0 blocks [1,2,3] gone; rank1 still has [1,2]
    sel = router.select_worker("ctx", [1, 2, 3])
    # Combined trie: hash 1 from rank1, hash 2 from rank1. Prefix = 2 blocks.
    assert sel.matched_blocks == 2

    # New blocks [9, 10] accessible
    sel = router.select_worker("ctx", [9, 10])
    assert sel.matched_blocks == 2
    assert sel.dp_rank == 0  # only rank0 has [9,10]


# ------------------------------------------------- ZMQ per-rank loopback


def test_zmq_per_rank_reporter_to_router():
    """Per-rank reporters push events that build hierarchical state."""
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    server = KVCacheRouterServer(router, address="tcp://127.0.0.1:*")
    endpoint, hmac_key = server.address
    server.start()
    try:
        router.register_worker_address("inst0", "http://host0:8000")

        reporters = []
        for rank in range(2):
            events = [stored_event(None, [1, 2, 3] if rank == 0 else [1, 2])]
            r = WorkerReporter(
                worker_id=f"inst0:rank{rank}",
                namespace="ctx",
                router_address=endpoint,
                hmac_key=hmac_key,
                get_events=lambda timeout_ms, e=events: e,
                get_load=lambda: (0, 0),
                max_batch_size=64,
                event_interval_s=0.01,
                load_interval_s=0.01,
            )
            r.start()
            reporters.append(r)

        try:
            deadline = time.time() + 5.0
            sel = None
            while time.time() < deadline:
                sel = router.select_worker("ctx", [1, 2, 3])
                if sel is not None and sel.matched_blocks == 3:
                    break
                time.sleep(0.02)
            assert sel is not None, "router never received per-rank reports"
            assert sel.worker_id == "inst0"
            assert sel.matched_blocks == 3
            assert sel.dp_rank == 0  # rank0 has full [1,2,3]
        finally:
            for r in reporters:
                r.stop()
    finally:
        server.stop()


# ------------------------------------------------- RouteHint + ADP router


def test_route_hint_dataclass():
    """RouteHint round-trips through DisaggregatedParams."""
    from tensorrt_llm.disaggregated_params import DisaggregatedParams, RouteHint
    hint = RouteHint(dp_rank=3)
    params = DisaggregatedParams(route_hint=hint)
    assert params.route_hint.dp_rank == 3


def test_route_hint_protocol_conversion():
    """RouteHint survives protocol -> internal conversion and back."""
    from tensorrt_llm.serve.openai_protocol import (
        DisaggregatedParams as ProtoDisagg,
        RouteHint as ProtoRouteHint,
        to_llm_disaggregated_params,
        to_disaggregated_params,
    )

    proto = ProtoDisagg(request_type="context_only",
                        route_hint=ProtoRouteHint(dp_rank=5))
    internal = to_llm_disaggregated_params(proto)
    assert internal.route_hint is not None
    assert internal.route_hint.dp_rank == 5

    # Round-trip back
    proto_back = to_disaggregated_params(internal)
    assert proto_back.route_hint is not None
    assert proto_back.route_hint.dp_rank == 5


def test_route_hint_none_passes_through():
    """No route_hint -> None in conversion."""
    from tensorrt_llm.serve.openai_protocol import (
        DisaggregatedParams as ProtoDisagg,
        to_llm_disaggregated_params,
    )
    proto = ProtoDisagg(request_type="context_only")
    internal = to_llm_disaggregated_params(proto)
    assert internal.route_hint is None


# ------------------------------------------------- Selection dp_rank field


def test_selection_dp_rank_field():
    """Selection dataclass carries dp_rank."""
    from tensorrt_llm.serve.kv_cache_router.messages import Selection
    sel = Selection(worker_id="inst0", address="http://host0:8000",
                    matched_blocks=5, dp_rank=3)
    assert sel.dp_rank == 3

    # Default is None
    sel2 = Selection(worker_id="inst0")
    assert sel2.dp_rank is None


# ------------------------------------------------- KvCacheConfig new fields


def test_kv_cache_config_per_rank_fields():
    """KvCacheConfig has the new per-rank routing fields with defaults."""
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    cfg = KvCacheConfig()
    assert cfg.per_rank_routing is False
    assert cfg.centralized_router_report_address is None

    cfg2 = KvCacheConfig(per_rank_routing=True,
                         centralized_router_report_address="tcp://router:5557")
    assert cfg2.per_rank_routing is True
    assert cfg2.centralized_router_report_address == "tcp://router:5557"


# ------------------------------------------------- per-instance lock concurrency


@pytest.mark.parametrize("algo", ["none", "adp"])
def test_concurrent_ingest_and_select_no_deadlock_or_corruption(algo):
    """Hammer event ingest on multiple instances while selecting concurrently.

    Exercises the per-instance ('per tree') locking: ingest to one instance must
    run without deadlocking against selects or ingest to others, and the routing
    state must stay self-consistent (selects always return a registered address
    and a valid dp_rank). Guards the struct->instance lock ordering against
    regressions (a reversed acquisition would deadlock this test)."""
    import threading

    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      rank_routing_algo=algo)
    n_inst, n_rank = 3, 4
    addrs = {}
    for i in range(n_inst):
        iid = f"inst{i}"
        addrs[iid] = f"http://host{i}:8000"
        router.register_worker_address(iid, addrs[iid])
        for rk in range(n_rank):
            wid = f"{iid}:rank{rk}"
            router.apply_event_report(KvCacheEventReport(
                wid, "ctx", seq=0,
                events=[stored_event(None, [1, 2, 3], layer_group_id=0)]))
            router.apply_load_report(load_report(wid, 0, active=5))

    stop = threading.Event()
    errors = []

    def ingest_worker(inst_idx):
        seq = 1
        try:
            while not stop.is_set():
                iid = f"inst{inst_idx}"
                for rk in range(n_rank):
                    wid = f"{iid}:rank{rk}"
                    h = [100 + seq * 10 + j for j in range(6)]
                    router.apply_event_report(KvCacheEventReport(
                        wid, "ctx", seq=seq,
                        events=[stored_event(None, h, layer_group_id=0)]))
                    router.apply_event_report(KvCacheEventReport(
                        wid, "ctx", seq=seq + 1,
                        events=[removed_event(h, layer_group_id=0)]))
                    router.apply_load_report(load_report(wid, seq, active=rk))
                seq += 2
        except Exception as e:  # noqa: BLE001
            errors.append(("ingest", e))

    def select_worker_loop():
        try:
            for _ in range(4000):
                sel = router.select_worker("ctx", [1, 2, 3])
                if sel is not None:
                    assert sel.address in addrs.values()
                    if sel.dp_rank is not None:
                        assert 0 <= sel.dp_rank < n_rank
        except Exception as e:  # noqa: BLE001
            errors.append(("select", e))

    threads = [threading.Thread(target=ingest_worker, args=(i,))
               for i in range(n_inst)]
    threads += [threading.Thread(target=select_worker_loop) for _ in range(3)]
    for t in threads:
        t.start()
    # Let the select loops finish; then stop ingest.
    for t in threads[n_inst:]:
        t.join(timeout=30)
        assert not t.is_alive(), "select loop hung -- possible deadlock"
    stop.set()
    for t in threads[:n_inst]:
        t.join(timeout=5)
        assert not t.is_alive(), "ingest loop hung -- possible deadlock"

    assert not errors, f"concurrent access errors: {errors[:3]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
