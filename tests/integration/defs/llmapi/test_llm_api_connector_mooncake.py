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
"""End-to-end tests for the mooncake-store KV connector against a live pool.

The unit tests drive the connector's own APIs. These drive a model: a real
engine publishes its KV cache into a real Mooncake pool, a second engine
replays it, and the generated tokens have to agree. That covers the seams no
narrower test reaches, namely the KV cache layout a `KVCacheManagerV2` actually
reports, per-rank namespacing under tensor parallelism, and the scheduler
interaction when a request is preempted while the connector is still reading
its pages.

One master and pool serve the whole module and nothing else on the machine.
Each test gets a key namespace of its own; see `mooncake_pool`.
"""

import contextlib
import json
import math
import os
import socket
from types import SimpleNamespace

import pytest
from test_common.mooncake_utils import running_master_on_free_ports

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    parse_size,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ledger import read_segments
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    POOL_MANIFEST_NAME,
    provision_pool,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig, MooncakeStoreConfig

from ..conftest import llm_models_root
from .test_llm_api_connector import E2E_MIN_TOKEN_AGREEMENT

MODEL_PATH = "Qwen3/Qwen3-0.6B"
#: What the fixture lends the pool. Small enough to pass the node budget check
#: wherever this runs.
SEGMENT_SIZE = "2GiB"

# Long enough to cover several KV blocks. A prefix shorter than one block is
# never published, which would leave the reuse assertions vacuous.
PROMPT = (
    "Nvidia Corporation is an American technology company headquartered in Santa Clara, "
    "California. Founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, it "
    "develops graphics processing units (GPUs), system on a chips (SoCs), and application "
    "programming interfaces (APIs) for data science, high-performance computing, and mobile "
    "and automotive applications. It is also a dominant supplier of artificial intelligence "
    "hardware and software. Tell me about the company."
)


def _common_prefix(left, right) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def assert_tokens_agree(reference, replayed, *, context: str) -> None:
    """Require a long common prefix rather than exact equality.

    Reusing cached KV skips prefill for the matched blocks, which changes the
    attention reduction order, so greedy decoding can split on a near-tie in
    the last tokens even when the restored K/V are bit-identical. Wrong or
    misaddressed KV instead diverges early and degenerates, so a prefix floor
    stays meaningful without making the test a coin flip. Same convention as
    `test_llm_api_connector.test_connector_e2e_persistent_cache`.
    """
    assert len(replayed) == len(reference), (
        f"{context}: generation length changed, {len(reference)} tokens against {len(replayed)}."
    )

    common = _common_prefix(reference, replayed)
    floor = math.floor(len(reference) * E2E_MIN_TOKEN_AGREEMENT)
    assert common >= floor, (
        f"{context}: diverged at token {common} of {len(reference)}, below the "
        f"{floor}-token floor, so the KV the engine read was wrong rather than "
        "merely numerically different.\n"
        f"  reference ids: {list(reference)}\n"
        f"  replayed ids:  {list(replayed)}"
    )


@contextlib.contextmanager
def pool_capacity(config_path: str, segment_size):
    """Lend the pool named by `config_path` a segment for the body's duration.

    The engines lend nothing, so this is the pool's whole capacity and every
    object lands here. An engine's own segment would go when the engine does,
    making cross-engine reuse depend on where the pool placed its objects.

    Closed before its master goes down, so the segment is unmounted rather
    than abandoned.
    """
    from mooncake.store import MooncakeDistributedStore

    config = MooncakeStoreConnectorConfig.from_file(config_path)
    store = MooncakeDistributedStore()
    status = store.setup(
        socket.gethostbyname(socket.gethostname()),
        config.metadata_server,
        parse_size(segment_size, strict_units=True),
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if status != 0:
        raise RuntimeError(
            f"the test fixture could not lend the pool a segment: "
            f"MooncakeDistributedStore.setup failed with status {status} "
            f"(master={config.master_server_address!r})"
        )
    try:
        yield
    finally:
        store.close()


def _set_namespace(config_path: str, namespace: str) -> None:
    """Point the rendered client config at `namespace`, renaming into place."""
    with open(config_path) as handle:
        config = json.load(handle)
    config["namespace"] = namespace
    staging = f"{config_path}.partial"
    with open(staging, "w") as handle:
        json.dump(config, handle, indent=2)
    os.replace(staging, config_path)


@pytest.fixture(scope="module")
def _mooncake_pool_for_module(tmp_path_factory):
    """One master, one rendered client config and one lent segment.

    The `LLM` API does not provision a pool itself, only `trtllm-serve` calls
    `maybe_provision_pool`, so the test does it. That also exports
    `MOONCAKE_CONFIG_PATH`, which is how a worker process finds the pool.

    The engines lend nothing; this fixture holds the pool's only segment. See
    `pool_capacity`.

    Pages pass through pinned host memory, which needs nothing of this node's
    hardware beyond what the fixture already assumes.
    """
    root = tmp_path_factory.mktemp("mooncake")
    master_dir = root / "master"
    client_dir = root / "client"
    master_dir.mkdir()
    client_dir.mkdir()

    with running_master_on_free_ports(str(master_dir), protocol="tcp"):
        store_config = MooncakeStoreConfig(
            pool=f"file://{master_dir / POOL_MANIFEST_NAME}",
            model_key=MODEL_PATH,
            segment_size=0,
            stage_through_host=True,
            run_dir=str(client_dir),
        )
        with provision_pool(store_config, run_dir=str(client_dir)) as config_path:
            with pool_capacity(config_path, SEGMENT_SIZE):
                yield SimpleNamespace(
                    config=store_config,
                    run_dir=str(client_dir),
                    config_path=config_path,
                )


@pytest.fixture
def mooncake_pool(_mooncake_pool_for_module, request):
    """The module's pool, under a key namespace of this test's own.

    Not a pool per test: a worker process can outlive the test it was spawned
    for and still resolve the `MOONCAKE_CONFIG_PATH` it was born with, by
    which time a per-test master has stopped accepting connections. One master
    per module keeps every path a worker might hold pointing at a live pool.

    Keys begin with `<namespace>/<model key>`, so the namespace is what keeps
    the tests apart. Rewritten in the shared config rather than rendered per
    test, so a worker reading the path late still gets the current test's.
    """
    _set_namespace(_mooncake_pool_for_module.config_path, f"trtllm-{request.node.name}")
    yield _mooncake_pool_for_module


def llm_kwargs(pool, **overrides) -> dict:
    """Engine settings shared by the tests here.

    `use_kv_cache_manager_v2` is not optional: the connector implements the V2
    worker contract (`register_kv_cache_layout`) and none of the V1 pool
    accessors, so a run that silently fell back to V1 would not reach it.
    """
    kwargs = dict(
        model=f"{llm_models_root()}/{MODEL_PATH}",
        backend="pytorch",
        kv_connector_config=KvCacheConnectorConfig(
            connector="mooncake-store", mooncake_store=pool.config
        ),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.2, use_kv_cache_manager_v2=True),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def matched_token_counts(monkeypatch):
    """Every answer the leader gave about how much of a prompt the pool held.

    Only usable when the executor runs in this process, since a spawned
    worker's leader would not see the patch.
    """
    counts = []
    original = MooncakeStoreConnectorScheduler.get_num_new_matched_tokens

    def recording(self, request, num_computed_tokens):
        result = original(self, request, num_computed_tokens)
        counts.append(result[0])
        return result

    monkeypatch.setattr(MooncakeStoreConnectorScheduler, "get_num_new_matched_tokens", recording)
    return counts


@pytest.fixture
def single_process_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


@pytest.mark.threadleak(enabled=False)
def test_mooncake_e2e_cross_engine_prefix_reuse(
    single_process_worker, mooncake_pool, matched_token_counts
):
    """A second engine replays a prefix the first one published.

    The first run must match nothing, or the pool was dirty and the comparison
    proves nothing. The second must match something, or the token agreement
    would hold just as well with the connector switched off.
    """
    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

    cold = LLM(**llm_kwargs(mooncake_pool))
    try:
        cold_output = cold.generate([PROMPT], sampling_params)
    finally:
        cold.shutdown()
    cold_token_ids = list(cold_output[0].outputs[0].token_ids)

    assert matched_token_counts and not any(matched_token_counts), (
        f"The first run already found tokens in the pool (matched counts "
        f"{matched_token_counts}), so it was not a cold start and the reuse "
        "comparison below is meaningless."
    )
    matched_token_counts.clear()

    warm = LLM(**llm_kwargs(mooncake_pool))
    try:
        warm_output = warm.generate([PROMPT], sampling_params)
    finally:
        warm.shutdown()
    warm_token_ids = list(warm_output[0].outputs[0].token_ids)

    assert matched_token_counts and max(matched_token_counts) > 0, (
        f"The second engine read nothing back from the pool (matched counts "
        f"{matched_token_counts}), so it recomputed the prefix and this test "
        "would pass with the connector disabled."
    )

    assert_tokens_agree(cold_token_ids, warm_token_ids, context="Cross-engine prefix reuse")


@pytest.mark.skip_less_device(2)
@pytest.mark.threadleak(enabled=False)
def test_mooncake_e2e_prefix_reuse_across_tensor_parallel_ranks(mooncake_pool):
    """Save and reuse at TP 2, where each rank owns a shard of every page.

    Three things only break above one rank. Keys are namespaced per rank, so a
    shard written under the wrong name is never found again. The save thread
    starts on device 0 and has to adopt the rank's device, or every rank but
    zero fails its copies. And a prefix only counts when every rank's shard is
    present, so a lookup that asked about one rank alone would hand out pages
    the others never saved.

    The leader runs in a spawned process here, so its matched-token answers are
    not observable. What is observable is that both ranks joined the pool and
    that the replayed tokens agree, and a rank whose saves failed reports the
    error on the next connector call rather than silently writing nothing.
    """
    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)
    kwargs = llm_kwargs(mooncake_pool, tensor_parallel_size=2)

    cold = LLM(**kwargs)
    try:
        cold_output = cold.generate([PROMPT], sampling_params)
    finally:
        cold.shutdown()

    segments = read_segments(mooncake_pool.run_dir)
    assert {record.rank for record in segments} == {0, 1}, (
        "Both ranks should have joined the pool, but the ledger records ranks "
        f"{sorted(record.rank for record in segments)}. A rank that never opened a "
        "store handle stores no shard, and every prefix it is asked about reads as "
        "a miss."
    )

    warm = LLM(**kwargs)
    try:
        warm_output = warm.generate([PROMPT], sampling_params)
    finally:
        warm.shutdown()

    assert_tokens_agree(
        list(cold_output[0].outputs[0].token_ids),
        list(warm_output[0].outputs[0].token_ids),
        context="Prefix reuse at TP 2",
    )


@pytest.mark.threadleak(enabled=False)
def test_mooncake_e2e_chunked_prefill_survives_a_small_cache(mooncake_pool):
    """Chunked prefill against a cache too small to hold every request.

    This is the preemption path the connector complicates: a victim's pages may
    still be under an outstanding save, so `preempt_request` declines and parks
    the request until the save drains. Get that wrong and the run either stalls,
    or frees pages the connector is still reading and stores KV that belongs to
    a different request.

    The reference run differs only in cache size and the connector, so the
    chunking is the same on both sides.

    Unlike its neighbours this one leaves the worker in its own process. The
    iteration statistics it asserts on reach the caller over RPC, and an
    in-process worker has no RPC server to answer, so they come back empty.

    The bound on context tokens is what makes this more than a liveness check.
    Chunks partition a prompt, so each prompt token is prefilled once when
    nothing is preempted, and a victim that recomputes adds another pass over
    the same tokens. Saves are write-through, so a resumed victim should find
    its prefix in the pool and load it instead, which is what keeps the count
    near one pass and is the behaviour this bound confirms.
    """
    # Distinct prefixes, so the requests compete for cache rather than sharing
    # pages. A page is only reclaimed when a generating request cannot get one,
    # so the pressure has to come from decode growth rather than prompt length:
    # these six fit at admission and outgrow the cache while generating.
    prompts = [f"{index}. {PROMPT} " * 2 for index in range(6)]
    sampling_params = SamplingParams(max_tokens=256, ignore_eos=True)
    # `max_seq_len` bounds the warmup allocation, which would otherwise size
    # itself against this model's 40k position limit and dwarf the cache below.
    # `max_batch_size` has to exceed the number of prompts, or the scheduler
    # limits concurrency before the cache does and nothing is ever preempted.
    engine = dict(
        enable_chunked_prefill=True,
        max_num_tokens=256,
        max_batch_size=8,
        max_seq_len=1024,
    )

    reference = LLM(
        **llm_kwargs(
            mooncake_pool,
            kv_connector_config=None,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.3, use_kv_cache_manager_v2=True
            ),
            **engine,
        )
    )
    try:
        reference_outputs = reference.generate(prompts, sampling_params)
    finally:
        reference.shutdown()

    segments_before = len(read_segments(mooncake_pool.run_dir))
    llm = LLM(
        **llm_kwargs(
            mooncake_pool,
            enable_iter_perf_stats=True,
            # Keep every iteration rather than the most recent 1000, so the
            # sums below cover the whole run.
            iter_stats_max_iterations=-1,
            # Bounds a V2 cache, which max_tokens does not; 256MiB is roughly 2300 tokens.
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=256 << 20, use_kv_cache_manager_v2=True
            ),
            **engine,
        )
    )
    try:
        outputs = llm.generate(prompts, sampling_params)
        stats = llm.get_stats(timeout=30)
    finally:
        llm.shutdown()

    # Every sum below is over this list, so an empty one would make them all
    # read zero and report that as a scheduler result.
    assert stats, "The engine returned no iteration statistics to assert on."

    # The assertions below are about the scheduler, so a connector that did
    # nothing would satisfy them. This run's own rank reaching the ledger is
    # the evidence that the pool was in the loop, counted against a reading
    # from before the engine started because the ledger outlives one test.
    assert len(read_segments(mooncake_pool.run_dir)) > segments_before, (
        "This run's rank recorded nothing, so the connector never opened the store "
        "and no save was ever in flight for a preemption to wait on."
    )

    for index, (reference_output, output) in enumerate(zip(reference_outputs, outputs)):
        assert_tokens_agree(
            list(reference_output.outputs[0].token_ids),
            list(output.outputs[0].token_ids),
            context=f"Request {index} under preemption",
        )

    inflight = [entry.get("inflightBatchingStats", {}) for entry in stats]
    paused = sum(entry.get("numPausedRequests", 0) for entry in inflight)
    assert paused > 0, (
        f"No request was ever paused across {len(stats)} iterations, so the "
        "cache was not small enough to reach the preemption path and this test "
        "exercised nothing."
    )

    prompt_tokens = sum(len(output.prompt_token_ids) for output in reference_outputs)
    context_tokens = sum(entry.get("numCtxTokens", 0) for entry in inflight)
    # One recompute per request is expected; a loop of them is not.
    budget = 3 * prompt_tokens
    assert context_tokens <= budget, (
        f"The engine prefilled {context_tokens} context tokens for {prompt_tokens} "
        f"prompt tokens, past the {budget}-token budget. Requests are being "
        "preempted and recomputed repeatedly rather than making progress."
    )
