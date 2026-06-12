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
"""Unit tests for the centralized KV-cache router (reporter + router core)."""

import time

import pytest

from tensorrt_llm.serve.kv_cache_router import (CentralizedKVCacheRouter,
                                                KvCacheEventReport,
                                                KVCacheRouterServer,
                                                WorkerLoadReport,
                                                WorkerPrefixTrie,
                                                WorkerReporter)

TPB = 32  # tokens per block


def stored_event(parent_hash, block_hashes):
    return {
        "event_id": 0,
        "data": {
            "type": "stored",
            "parent_hash": parent_hash,
            "blocks": [{
                "block_hash": h
            } for h in block_hashes],
        },
        "window_size": 0,
    }


def removed_event(block_hashes):
    return {
        "event_id": 0,
        "data": {
            "type": "removed",
            "block_hashes": list(block_hashes)
        },
        "window_size": 0,
    }


def load_report(worker_id, seq, active=0, queued=0, max_bs=64, ns="ctx"):
    return WorkerLoadReport(worker_id=worker_id,
                            namespace=ns,
                            seq=seq,
                            num_active_requests=active,
                            num_queued_requests=queued,
                            max_batch_size=max_bs)


# --------------------------------------------------------------------- trie


def test_trie_longest_prefix_match():
    trie = WorkerPrefixTrie()
    # w0 holds [1,2,3]; w1 holds [1,2]; w2 holds [9]
    trie.add("w0", [1, 2, 3])
    trie.add("w1", [1, 2])
    trie.add("w2", [9])

    m = trie.match([1, 2, 3, 4])
    assert m == {"w0": 3, "w1": 2}  # w2 absent (no match at depth 1)

    # Query that only shares the first block.
    assert trie.match([1, 5]) == {"w0": 1, "w1": 1}
    # No shared prefix at all.
    assert trie.match([7, 8]) == {}


def test_trie_remove_and_remove_worker():
    trie = WorkerPrefixTrie()
    trie.add("w0", [1, 2, 3])
    trie.add("w1", [1, 2, 3])

    trie.remove("w0", [3])
    assert trie.match([1, 2, 3]) == {"w0": 2, "w1": 3}

    trie.remove_worker("w1")
    assert trie.match([1, 2, 3]) == {"w0": 2}
    assert not trie.has_worker("w1")


# ----------------------------------------------------------------- ingest


def test_apply_event_report_updates_trie():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("w0", 0))

    sel = router.select_worker("ctx", [1, 2, 3])
    assert sel is not None and sel.worker_id == "w0"
    assert sel.matched_blocks == 3

    # A removed event shortens the match.
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=1, events=[removed_event([3])]))
    sel = router.select_worker("ctx", [1, 2, 3])
    assert sel.matched_blocks == 2


def test_stale_seq_dropped():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=5,
                           events=[stored_event(None, [1, 2])]))
    # Stale (seq <= last) event must be ignored.
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=3, events=[removed_event([1, 2])]))
    router.apply_load_report(load_report("w0", 0))
    assert router.select_worker("ctx", [1, 2]).matched_blocks == 2


def test_full_snapshot_replaces_table():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    # Snapshot with a different block set replaces everything.
    router.apply_event_report(
        KvCacheEventReport("w0",
                           "ctx",
                           seq=1,
                           events=[stored_event(None, [9])],
                           is_full_snapshot=True))
    router.apply_load_report(load_report("w0", 0))
    assert router.select_worker("ctx", [1, 2, 3]) .matched_blocks == 0
    assert router.select_worker("ctx", [9]).matched_blocks == 1


# ----------------------------------------------------------------- scoring


def test_scoring_prefers_cache_match():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    # w0 has the prefix cached, w1 does not; equal (zero) load.
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("w0", 0))
    router.apply_load_report(load_report("w1", 0))
    assert router.select_worker("ctx", [1, 2, 3]).worker_id == "w0"


def test_scoring_prefers_lower_load_without_match():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_load_report(load_report("busy", 0, active=50))
    router.apply_load_report(load_report("idle", 0, active=0))
    # No cache match anywhere -> lower load wins.
    assert router.select_worker("ctx", [1, 2, 3]).worker_id == "idle"


def test_namespace_isolation():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("ctx0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("ctx0", 0, ns="ctx"))
    router.apply_load_report(load_report("gen0", 0, ns="gen"))

    # Query in "gen" never sees the ctx worker.
    sel = router.select_worker("gen", [1, 2, 3])
    assert sel.worker_id == "gen0"
    assert sel.matched_blocks == 0
    # Unknown namespace -> None.
    assert router.select_worker("nope", [1, 2, 3]) is None


def test_suspend_routing_on_stale_load():
    clk = {"t": 1000.0}
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      load_suspend_s=3.0,
                                      stale_timeout_s=60.0,
                                      clock=lambda: clk["t"])
    # w_fresh keeps reporting load; w_quiet stops after t=1000 but its cache is
    # the better prefix match.
    router.apply_event_report(
        KvCacheEventReport("w_quiet", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("w_quiet", 0))
    router.apply_load_report(load_report("w_fresh", 0))

    # Within the window, the cache-rich w_quiet wins.
    assert router.select_worker("ctx", [1, 2, 3]).worker_id == "w_quiet"

    # Advance past load_suspend_s without a new load report from w_quiet; it is
    # suspended even though its cache still matches.
    clk["t"] += 4.0
    router.apply_load_report(load_report("w_fresh", 1))  # only w_fresh refreshes
    assert router.select_worker("ctx", [1, 2, 3]).worker_id == "w_fresh"

    # w_quiet resumes reporting load -> eligible again, cache match wins.
    router.apply_load_report(load_report("w_quiet", 1))
    assert router.select_worker("ctx", [1, 2, 3]).worker_id == "w_quiet"


def test_suspend_then_all_suspended_returns_none():
    clk = {"t": 0.0}
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      load_suspend_s=3.0,
                                      stale_timeout_s=60.0,
                                      clock=lambda: clk["t"])
    router.apply_load_report(load_report("w0", 0))
    assert router.select_worker("ctx", []) is not None
    clk["t"] += 5.0  # no fresh load anywhere
    assert router.select_worker("ctx", []) is None


def test_stale_worker_eviction():
    clk = {"t": 1000.0}
    router = CentralizedKVCacheRouter(tokens_per_block=TPB,
                                      stale_timeout_s=10.0,
                                      clock=lambda: clk["t"])
    router.apply_load_report(load_report("w0", 0))
    assert router.select_worker("ctx", []) is not None
    clk["t"] += 11.0
    router.evict_stale_workers()
    assert router.select_worker("ctx", []) is None


# ------------------------------------------------ worker_id -> address map


def test_address_resolution_in_selection():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.register_worker_address("w0", "http://host0:8000")
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("w0", 0))

    sel = router.select_worker("ctx", [1, 2, 3])
    assert sel.worker_id == "w0"
    assert sel.address == "http://host0:8000"


def test_address_unknown_until_registered():
    # Events can arrive before /server_info is fetched; address is None then.
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1])]))
    router.apply_load_report(load_report("w0", 0))
    sel = router.select_worker("ctx", [1])
    assert sel.worker_id == "w0" and sel.address is None

    router.register_worker_address("w0", "http://host0:8000")
    assert router.select_worker("ctx", [1]).address == "http://host0:8000"


def test_address_rebind_supersedes_stale_id():
    # The instance at a URL restarts under a new worker_id; the URL must map to
    # the new id only, and the old id must be forgotten.
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.register_worker_address("old", "http://host0:8000")
    router.register_worker_address("new", "http://host0:8000")
    assert router.address_of("old") is None
    assert router.address_of("new") == "http://host0:8000"


def test_unregister_removes_address_and_routing_state():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.register_worker_address("w0", "http://host0:8000")
    router.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3])]))
    router.apply_load_report(load_report("w0", 0))
    assert router.select_worker("ctx", [1, 2, 3]) is not None

    # Adapter only knows the address when a server leaves the pool.
    router.unregister_worker_address(address="http://host0:8000")
    assert router.address_of("w0") is None
    # Routing state is gone immediately, not left for stale-eviction.
    assert router.select_worker("ctx", [1, 2, 3]) is None


# ------------------------------------------------------ ADP: one instance


def test_adp_instance_routed_as_single_target():
    # A gathered ADP report carries the union of all ranks' blocks under one
    # worker_id; the router treats it as a single target.
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    router.apply_event_report(
        KvCacheEventReport("adp_inst", "ctx", seq=0,
                           events=[stored_event(None, [1, 2, 3, 4])]))
    router.apply_load_report(load_report("adp_inst", 0))
    sel = router.select_worker("ctx", [1, 2, 3, 4])
    assert sel.worker_id == "adp_inst"
    assert sel.matched_blocks == 4


# --------------------------------------------------- ZMQ loopback (e2e)


def test_zmq_loopback_reporter_to_router():
    router = CentralizedKVCacheRouter(tokens_per_block=TPB)
    server = KVCacheRouterServer(router, address="tcp://127.0.0.1:*")
    endpoint, hmac_key = server.address
    server.start()
    try:
        events = [stored_event(None, [1, 2, 3])]
        reporter = WorkerReporter(
            worker_id="w0",
            namespace="ctx",
            router_address=endpoint,
            hmac_key=hmac_key,
            get_events=lambda timeout_ms: events,
            get_load=lambda: (2, 1),
            max_batch_size=64,
            event_interval_s=0.01,
            load_interval_s=0.01,
        )
        reporter.start()
        try:
            # Wait for the pushed reports to land.
            deadline = time.time() + 5.0
            sel = None
            while time.time() < deadline:
                sel = router.select_worker("ctx", [1, 2, 3])
                if sel is not None and sel.matched_blocks == 3:
                    break
                time.sleep(0.02)
            assert sel is not None, "router never received reports"
            assert sel.worker_id == "w0"
            assert sel.matched_blocks == 3
        finally:
            reporter.stop()
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
