# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""End-to-end tests for the centralized KV-cache router (KV-cache manager V2).

Wires the *real* components together in one process, using the V2 KV-cache
manager only (the C++ manager is left untouched):

    in-process LLM (use_kv_cache_manager_v2=True)
        -> KVCacheManagerV2.event_manager   (radix tree enqueues stored/removed)
        -> WorkerReporter  (drains kv_cache_manager.get_latest_events)
        -> real ZMQ PUSH  -> KVCacheRouterServer (PULL, HMAC)
        -> CentralizedKVCacheRouter
        -> select_worker(namespace, block_hashes)

Real generations populate the V2 radix tree, so the manager emits genuine
``stored`` events. The query hashes are computed with ``block_key_hasher`` over
the *same* prompt tokens, proving cross-side block-hash compatibility: the
hashes a router computes for a request must equal the hashes the V2 manager
emits for the blocks it stored.

The multi-worker test (2 ctx + 2 gen) captures real V2 events once from a single
LLM (4 live engines would not fit one GPU), then replays those real event
payloads through four real reporters over real ZMQ. This keeps full hash
fidelity while exercising multi-worker selection and namespace isolation.

Single GPU. Marked threadleak-exempt because the engine, KV-cache manager and
the reporter/router all spawn background threads.
"""

import os
import time

import pytest

# The reporter must reach the KV-cache manager in-process, so the executor has
# to run single-process (otherwise LLM uses GenerationExecutorProxy and the
# engine lives in another process). Must be set before importing tensorrt_llm.
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.kv_cache_router import (CentralizedKVCacheRouter,
                                                KVCacheRouterServer,
                                                WorkerReporter, block_key_hasher)

pytestmark = pytest.mark.threadleak(enabled=False)

TOKENS_PER_BLOCK = 32
NAMESPACE = "ctx"


def _v2_llm():
    model_path = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    # Leave kv_cache_event_hash_algo at its default ("auto" -> v1_block_key,
    # even for V2), so the V2 manager emits V1-style hashes that the router's
    # block_key_hasher reproduces. This is what makes the cross-side hash
    # assertions meaningful.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                    event_buffer_max_size=1024,
                                    enable_block_reuse=True,
                                    tokens_per_block=TOKENS_PER_BLOCK,
                                    use_kv_cache_manager_v2=True)
    return LLM(model=model_path,
               kv_cache_config=kv_cache_config,
               enable_autotuner=False)


def _v2_kv_cache_manager(llm):
    """Reach the in-process V2 KV-cache manager from the LLM executor."""
    engine = llm._executor.engine
    return engine.resource_manager.resource_managers[
        ResourceManagerType.KV_CACHE_MANAGER]


def _expected_block_hashes(tokens):
    """Hash *tokens* the way a request is hashed at query time.

    Mirrors BlockHashMixin._compute_block_hashes: chained per-block, excluding
    the final token (which is not part of a block key).
    """
    hashes = []
    end = len(tokens) - 1
    t = 0
    while t < end:
        t_end = min(t + TOKENS_PER_BLOCK, end)
        parent = hashes[-1] if hashes else None
        hashes.append(block_key_hasher(tokens[t:t_end], parent))
        t += TOKENS_PER_BLOCK
    return hashes


def _generate_and_capture_events(llm, kv_cache_manager, prompt_ids):
    """Run a real generation and return the serialized ``stored`` events.

    Drains the V2 event manager (the same queue the reporter consumes), so the
    returned dicts are exactly what a worker would push for *prompt_ids*.
    """
    kv_cache_manager.get_latest_events(0)  # clear created/startup events
    sampling = SamplingParams(max_tokens=8, temperature=0.0)
    llm.generate([prompt_ids], sampling_params=sampling)

    events = []
    deadline = time.time() + 30.0
    while time.time() < deadline:
        kv_cache_manager.flush_iteration_events()
        batch = kv_cache_manager.get_latest_events(0)
        if batch:
            events.extend(KVCacheEventSerializer.serialize(batch))
        if any(e.get("data", {}).get("type") == "stored" for e in events):
            break
        time.sleep(0.05)
    assert any(e.get("data", {}).get("type") == "stored" for e in events), (
        "generation did not emit a 'stored' event")
    return events


def _replay_get_events(events):
    """One-shot ``get_events`` for a reporter: deliver *events* once as the
    initial snapshot, then nothing (the worker holds a static block table)."""
    state = {"sent": False}

    def get_events(timeout_ms):  # noqa: ARG001 - signature match
        if state["sent"]:
            return []
        state["sent"] = True
        return events

    return get_events


def _wait_until(predicate, timeout_s=30.0, interval_s=0.1):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def test_centralized_router_e2e_cache_locality():
    """A request whose prefix the V2 manager cached is routed back to that worker."""
    llm = _v2_llm()
    kv_cache_manager = _v2_kv_cache_manager(llm)

    router = CentralizedKVCacheRouter(tokens_per_block=TOKENS_PER_BLOCK,
                                      load_suspend_s=60.0,
                                      stale_timeout_s=600.0)
    router.register_worker_address("ctx-0", "http://localhost:8000")

    server = KVCacheRouterServer(router, address="tcp://127.0.0.1:*")
    endpoint, hmac_key = server.address
    server.start()

    reporter = WorkerReporter(
        worker_id="ctx-0",
        namespace=NAMESPACE,
        router_address=endpoint,
        hmac_key=hmac_key,
        get_events=kv_cache_manager.get_latest_events,
        get_load=lambda: (0, 0),
        max_batch_size=1,
        event_interval_s=0.02,
        load_interval_s=0.05,
    )
    reporter.start()

    try:
        prompt_ids = list(range(10, 10 + 2 * TOKENS_PER_BLOCK + 1))
        llm.generate([prompt_ids],
                     sampling_params=SamplingParams(max_tokens=8,
                                                    temperature=0.0))

        query_hashes = _expected_block_hashes(prompt_ids)
        assert query_hashes, "test setup should produce at least one block"

        def matched():
            sel = router.select_worker(NAMESPACE, query_hashes)
            return sel is not None and sel.matched_blocks > 0

        assert _wait_until(matched, timeout_s=60.0), (
            "router never received a matching report from the worker")

        sel = router.select_worker(NAMESPACE, query_hashes)
        assert sel.worker_id == "ctx-0"
        assert sel.address == "http://localhost:8000"
        assert sel.matched_blocks >= 1, (
            "router-side block hashes did not match V2 manager-emitted hashes; "
            "check block_key_hasher vs the V2 manager's hash algorithm")
    finally:
        reporter.stop()
        server.stop()
        llm.shutdown()


def test_centralized_router_e2e_two_ctx_two_gen():
    """2 ctx + 2 gen workers: cache locality, namespace isolation, load tiebreak.

    Captures real V2 events for two distinct prompts from one LLM, then replays
    them through four real reporters (ctx-0/ctx-1, gen-0/gen-1) so each worker
    holds a real, distinct block table. Asserts:
      * a prompt is routed to the worker that cached it (within its namespace);
      * a ctx query never returns a gen worker and vice versa;
      * with no cache match, the least-loaded worker in the namespace wins.
    """
    llm = _v2_llm()
    kv_cache_manager = _v2_kv_cache_manager(llm)

    # Two distinct prompts -> distinct block hashes (different token ids).
    prompt_a = list(range(1000, 1000 + 2 * TOKENS_PER_BLOCK + 1))
    prompt_b = list(range(5000, 5000 + 2 * TOKENS_PER_BLOCK + 1))
    prompt_c = list(range(9000, 9000 + 2 * TOKENS_PER_BLOCK + 1))  # uncached

    events_a = _generate_and_capture_events(llm, kv_cache_manager, prompt_a)
    events_b = _generate_and_capture_events(llm, kv_cache_manager, prompt_b)

    hashes_a = _expected_block_hashes(prompt_a)
    hashes_b = _expected_block_hashes(prompt_b)
    hashes_c = _expected_block_hashes(prompt_c)

    router = CentralizedKVCacheRouter(tokens_per_block=TOKENS_PER_BLOCK,
                                      load_suspend_s=60.0,
                                      stale_timeout_s=600.0)

    server = KVCacheRouterServer(router, address="tcp://127.0.0.1:*")
    endpoint, hmac_key = server.address
    server.start()

    # worker_id -> (namespace, replayed events, (active, queued) load)
    # ctx-1 is the least-loaded ctx worker; gen-0 the least-loaded gen worker.
    workers = {
        "ctx-0": (NAMESPACE, events_a, (8, 4)),
        "ctx-1": ("ctx", events_b, (1, 0)),
        "gen-0": ("gen", events_a, (0, 0)),
        "gen-1": ("gen", events_b, (5, 2)),
    }
    reporters = []
    try:
        for wid, (ns, events, load) in workers.items():
            router.register_worker_address(wid, f"http://localhost/{wid}")
            rep = WorkerReporter(
                worker_id=wid,
                namespace=ns,
                router_address=endpoint,
                hmac_key=hmac_key,
                get_events=_replay_get_events(events),
                get_load=(lambda l=load: l),
                max_batch_size=16,
                event_interval_s=0.05,
                load_interval_s=0.05,
            )
            rep.start()
            reporters.append(rep)

        # Wait until all four workers are known (load reports landed) and the
        # cache tables are populated.
        def ready():
            a = router.select_worker("ctx", hashes_a)
            b = router.select_worker("ctx", hashes_b)
            g = router.select_worker("gen", hashes_a)
            return (a is not None and a.matched_blocks > 0 and b is not None
                    and b.matched_blocks > 0 and g is not None)

        assert _wait_until(ready, timeout_s=60.0), (
            "router did not receive all worker reports")

        # Cache locality within ctx: A -> ctx-0, B -> ctx-1.
        assert router.select_worker("ctx", hashes_a).worker_id == "ctx-0"
        assert router.select_worker("ctx", hashes_b).worker_id == "ctx-1"

        # Namespace isolation: prompt A is cached on ctx-0 and gen-0, but a gen
        # query must stay within the gen pool.
        gen_sel = router.select_worker("gen", hashes_a)
        assert gen_sel.worker_id == "gen-0"
        assert gen_sel.worker_id not in ("ctx-0", "ctx-1")
        # And a ctx query for B's prefix never returns the gen worker holding B.
        assert router.select_worker("ctx", hashes_b).worker_id != "gen-1"

        # No cache match in ctx -> least-loaded ctx worker (ctx-1) wins.
        c_sel = router.select_worker("ctx", hashes_c)
        assert c_sel.matched_blocks == 0
        assert c_sel.worker_id == "ctx-1"
    finally:
        for rep in reporters:
            rep.stop()
        server.stop()
        llm.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
