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

"""Threaded single-process NIXL harness for V2 disaggregated KV transfer tests.

Creates one ``KvCacheTransceiverV2`` per rank inside a single process using
threads plus a Barrier-based ``Distributed`` mock, then drives a full
ctx-send / gen-receive transfer and hands verification back to the caller.
Model specifics (cache-manager construction, pool initialization, post-transfer
verification) are injected as hooks; see ``test_deepseek_v4_kv_transfer.py``
and ``test_minimax_m3_kv_transfer.py`` for the two current users.
"""

import os
import threading
import uuid
from typing import Dict, List, Optional, Protocol, Sequence, TypeVar

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

# Reduce NIXL threads for unit test: default 8 threads per agent causes heavy
# contention when creating multiple agents on a single GPU in the same process.
os.environ.setdefault("TRTLLM_NIXL_NUM_THREADS", "0")


# ---------------------------------------------------------------------------
# Harness-scale constants shared by all model-specific manager factories
# ---------------------------------------------------------------------------
TOKENS_PER_BLOCK = 128
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 16
VOCAB_SIZE = 129280

_CacheManagerT = TypeVar("_CacheManagerT", bound=KVCacheManagerV2)
_CacheManagerT_co = TypeVar("_CacheManagerT_co", bound=KVCacheManagerV2, covariant=True)
_CacheManagerT_contra = TypeVar("_CacheManagerT_contra", bound=KVCacheManagerV2, contravariant=True)


class ManagerFactory(Protocol[_CacheManagerT_co]):
    """Build one cache manager per rank of a (tp x pp) instance.

    Model-specific knobs (dtype, compress ratios, sparse layers, ...) are
    closed over by the caller rather than threaded through the harness.
    """

    def __call__(
        self,
        tp: int,
        pp: int,
        enable_dp: bool,
        /,
    ) -> Sequence[_CacheManagerT_co]: ...


class CacheInitializer(Protocol[_CacheManagerT_contra]):
    def __call__(
        self,
        managers: Sequence[_CacheManagerT_contra],
        tp: int,
        /,
        *,
        seed_base: int = 0,
        fill_random: bool = True,
    ) -> None: ...


class CacheVerifier(Protocol[_CacheManagerT_contra]):
    def __call__(
        self,
        *,
        request_lengths: List[int],
        ctx_managers: Sequence[_CacheManagerT_contra],
        gen_managers: Sequence[_CacheManagerT_contra],
        ctx_tp: int,
        ctx_pp: int,
        gen_tp: int,
        gen_pp: int,
        ctx_enable_dp: bool,
        gen_enable_dp: bool,
        ctx_request_ids: List[int],
        gen_request_ids: List[int],
    ) -> None: ...


# ---------------------------------------------------------------------------
# ThreadSafeDistributed: threading.Barrier-based Distributed mock
# ---------------------------------------------------------------------------
class ThreadSafeDistributed:
    """Distributed mock using threading.Barrier for single-process multi-rank testing.

    Provides the same interface as TorchDistributedWrapper from test_py_cache_transceiver_mp.py
    but uses Barrier + Lock + shared dict instead of torch.distributed.
    """

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        tp_size: int,
        pp_size: int,
        tp_rank: int,
        pp_rank: int,
        shared: dict,
    ):
        self.rank = local_rank
        self._world_size = world_size
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._tp_rank = tp_rank
        self._pp_rank = pp_rank
        self._s = shared
        self._bcast_idx = 0
        self._ag_idx = 0
        self._pp_ag_idx = 0
        self._tp_ag_idx = 0

    @property
    def tp_size(self):
        return self._tp_size

    @property
    def pp_size(self):
        return self._pp_size

    @property
    def world_size(self):
        return self._world_size

    def broadcast(self, obj, root=0):
        idx = self._bcast_idx
        self._bcast_idx += 1
        key = f"bcast_{idx}"
        if self.rank == root:
            self._s[key] = obj
        self._s["barrier"].wait()
        result = self._s[key]
        self._s["barrier"].wait()
        return result

    def allgather(self, obj):
        idx = self._ag_idx
        self._ag_idx += 1
        key = f"ag_{idx}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._world_size
            self._s[key][self.rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def pp_allgather(self, obj):
        idx = self._pp_ag_idx
        self._pp_ag_idx += 1
        key = f"pp_ag_{idx}_tp{self._tp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._pp_size
            self._s[key][self._pp_rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def tp_allgather(self, obj):
        idx = self._tp_ag_idx
        self._tp_ag_idx += 1
        key = f"tp_ag_{idx}_pp{self._pp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._tp_size
            self._s[key][self._tp_rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------
def run_concurrent(items, fn):
    """Run fn(item) for each item concurrently in threads and propagate errors."""
    errors = [None] * len(items)
    results = [None] * len(items)

    def _worker(idx, item):
        try:
            results[idx] = fn(item)
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=_worker, args=(i, item)) for i, item in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, err in enumerate(errors):
        if err is not None:
            raise err
    return results


def _create_transceiver_in_thread(rank, mapping, cache_manager, dist_mock, config, results, errors):
    """Thread target: create one KvCacheTransceiverV2."""
    try:
        tc = KvCacheTransceiverV2(
            mapping=mapping,
            dist=dist_mock,
            kv_cache_manager=cache_manager,
            cache_transceiver_config=config,
        )
        results[rank] = tc
    except Exception as e:
        errors[rank] = e


def create_instance_transceivers(
    tp: int,
    pp: int,
    enable_dp: bool,
    cache_managers: Sequence[KVCacheManagerV2],
    config: CacheTransceiverConfig,
) -> List[KvCacheTransceiverV2]:
    """Create KvCacheTransceiverV2 for all ranks via threaded init."""
    world_size = tp * pp
    shared = {"barrier": threading.Barrier(world_size), "lock": threading.Lock()}
    results = [None] * world_size
    errors = [None] * world_size
    threads = []

    for rank in range(world_size):
        pp_rank = rank // tp
        tp_rank = rank % tp
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=tp,
            pp_size=pp,
            enable_attention_dp=enable_dp,
        )
        dist_mock = ThreadSafeDistributed(rank, world_size, tp, pp, tp_rank, pp_rank, shared)
        t = threading.Thread(
            target=_create_transceiver_in_thread,
            args=(
                rank,
                mapping,
                cache_managers[rank],
                dist_mock,
                config,
                results,
                errors,
            ),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for rank, err in enumerate(errors):
        if err is not None:
            raise err

    return results


def get_ctx_info_endpoint(tc: KvCacheTransceiverV2) -> Optional[str]:
    """Extract the context_info_endpoint from a transceiver's disaggregated params."""
    endpoints = tc.get_disaggregated_params().get("ctx_info_endpoint") or []
    return endpoints[0] if endpoints else None


def get_layers_per_pp(num_layers: int, pp_size: int) -> List[int]:
    """Return a list of layer counts per PP rank (mirrors C++ getLayerNumPPRank).

    When num_layers is not evenly divisible by pp_size, the first
    (num_layers % pp_size) ranks get one extra layer.
    Matches Mapping.pp_layers / torch.tensor_split behaviour.
    """
    base = num_layers // pp_size
    extra = num_layers % pp_size
    return [base + (1 if r < extra else 0) for r in range(pp_size)]


def run_kv_transfer_test(
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    update_before_transfer: bool = True,
    *,
    manager_factory: ManagerFactory[_CacheManagerT],
    init_fn: CacheInitializer[_CacheManagerT],
    verify_fn: CacheVerifier[_CacheManagerT],
) -> None:
    """Run one ctx->gen KV transfer with injectable model-specific cache hooks."""
    ctx_world = ctx_tp * ctx_pp
    gen_world = gen_tp * gen_pp

    # Mix of block-aligned and non-aligned lengths for boundary testing.
    # TOKENS_PER_BLOCK=128: 65=half+1, 256=2x exact, 129=1x+1, 383=3x-1
    request_lengths = [65, 256, 129, 383]

    # ===== 1. Create cache managers =====
    ctx_managers = list(manager_factory(ctx_tp, ctx_pp, ctx_enable_dp))
    gen_managers = list(manager_factory(gen_tp, gen_pp, gen_enable_dp))

    # ===== 2. Initialize data =====
    # ctx: random data, seed=pp_rank (same across TP, different across PP)
    init_fn(ctx_managers, ctx_tp, seed_base=1000, fill_random=True)
    # gen: zeros
    init_fn(gen_managers, gen_tp, fill_random=False)

    # ===== 3. Create KvCacheTransceiverV2 instances (threaded init) =====
    config = CacheTransceiverConfig(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        max_tokens_in_buffer=512,
    )
    ctx_tcs = create_instance_transceivers(ctx_tp, ctx_pp, ctx_enable_dp, ctx_managers, config)
    gen_tcs = create_instance_transceivers(gen_tp, gen_pp, gen_enable_dp, gen_managers, config)

    try:
        ctx_info_endpoint = get_ctx_info_endpoint(ctx_tcs[0])

        # ===== 4. Create requests and determine handle map =====
        # handle_map: rank -> [(req_idx, ctx_request, gen_request)]
        ctx_handle_map: Dict[int, List] = {r: [] for r in range(ctx_world)}
        gen_handle_map: Dict[int, List] = {r: [] for r in range(gen_world)}
        ctx_request_ids: List[int] = []
        gen_request_ids: List[int] = []

        sampling_params = SamplingParams()

        for req_idx, req_len in enumerate(request_lengths):
            unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
            ctx_rid = req_idx * 2
            gen_rid = req_idx * 2 + 1
            ctx_request_ids.append(ctx_rid)
            gen_request_ids.append(gen_rid)

            ctx_dp_rank = req_idx % ctx_tp if ctx_enable_dp else 0

            ctx_request = LlmRequest(
                request_id=ctx_rid,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
            )
            ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)

            gen_request = LlmRequest(
                request_id=gen_rid,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            )
            gen_request.py_disaggregated_params = DisaggregatedParams(
                ctx_request_id=ctx_rid,
                ctx_dp_rank=ctx_dp_rank,
                ctx_info_endpoint=ctx_info_endpoint,
                disagg_request_id=unique_rid,
            )

            for rank in range(ctx_world):
                tp_rank = rank % ctx_tp
                should_handle = (not ctx_enable_dp) or (req_idx % ctx_tp == tp_rank)
                if should_handle:
                    ctx_handle_map[rank].append((req_idx, ctx_request))

            for rank in range(gen_world):
                tp_rank = rank % gen_tp
                should_handle = (not gen_enable_dp) or (req_idx % gen_tp == tp_rank)
                if should_handle:
                    gen_handle_map[rank].append((req_idx, gen_request))

        # ===== 5. Allocate KV cache for all ranks =====
        # prepare_resources is a no-op for non-draft KVCacheManagerV2.
        # All ranks must allocate BEFORE mutating shared request objects
        # (add_new_token changes is_first_context_chunk).
        #
        # Gen ranks take the disagg-gen-init path: prepare_disagg_gen_init
        # sizes the cache for the full prompt and pre-declares
        # history_length=prompt_len, matching what the V2 scheduler's
        # _try_schedule_disagg_gen_init does in production so the
        # transceiver's TRANS_COMPLETE contract check is satisfied.
        # Ctx ranks take the regular prefill path (prepare_context +
        # resize_context).
        gen_batches: Dict[int, ScheduledRequests] = {}
        for rank in range(gen_world):
            reqs = [req for _, req in gen_handle_map[rank]]
            if reqs:
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = reqs
                for req in reqs:
                    gen_managers[rank].prepare_disagg_gen_init(req)
                gen_batches[rank] = batch

        ctx_batches: Dict[int, ScheduledRequests] = {}
        for rank in range(ctx_world):
            reqs = [req for _, req in ctx_handle_map[rank]]
            if reqs:
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = reqs
                for req in reqs:
                    ctx_managers[rank].prepare_context(req)
                    ctx_managers[rank].resize_context(req, req.context_chunk_size)
                ctx_batches[rank] = batch

        # ===== 5.5. context_current_position + add_new_token =====
        # Set position on each unique request once (needed for transfer metadata).
        seen: set = set()
        for rank in range(ctx_world):
            for _, req in ctx_handle_map[rank]:
                if req.py_request_id not in seen:
                    req.context_current_position = req.prompt_len
                    req.add_new_token(req.prompt_len, 0)
                    seen.add(req.py_request_id)

        seen = set()
        for rank in range(gen_world):
            for _, req in gen_handle_map[rank]:
                if req.py_request_id not in seen:
                    req.context_current_position = req.prompt_len
                    req.add_new_token(req.prompt_len, 0)
                    seen.add(req.py_request_id)

        # ===== 5.6. update_resources BEFORE transfer (mode: update_before) =====
        if update_before_transfer:
            for rank, batch in ctx_batches.items():
                ctx_managers[rank].update_resources(batch)
            for rank, batch in gen_batches.items():
                gen_managers[rank].update_resources(batch)

        # ===== 6. gen receive + ctx send =====
        for rank in range(gen_world):
            for _, req in gen_handle_map[rank]:
                gen_tcs[rank].request_and_receive_async(req)
        for rank in range(ctx_world):
            for _, req in ctx_handle_map[rank]:
                ctx_tcs[rank].respond_and_send_async(req)

        # ===== 7. Wait for completion (threaded, dist calls inside) =====
        run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))

        # ===== 7.5. update_resources AFTER transfer (mode: update_after) =====
        if not update_before_transfer:
            for rank, batch in ctx_batches.items():
                ctx_managers[rank].update_resources(batch)
            for rank, batch in gen_batches.items():
                gen_managers[rank].update_resources(batch)

        # ===== 8. Verify =====
        verify_fn(
            request_lengths=request_lengths,
            ctx_managers=ctx_managers,
            gen_managers=gen_managers,
            ctx_tp=ctx_tp,
            ctx_pp=ctx_pp,
            gen_tp=gen_tp,
            gen_pp=gen_pp,
            ctx_enable_dp=ctx_enable_dp,
            gen_enable_dp=gen_enable_dp,
            ctx_request_ids=ctx_request_ids,
            gen_request_ids=gen_request_ids,
        )

    finally:
        for tc in ctx_tcs + gen_tcs:
            try:
                tc.shutdown()
            except Exception:
                pass
        for mgr in ctx_managers + gen_managers:
            try:
                mgr.shutdown()
            except Exception:
                pass
