# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch

import tensorrt_llm.bindings
from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base import CacheExtent, CacheKind, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.agent import use_pure_python_transfer_agent
from tensorrt_llm._torch.disaggregation.base.transfer import (
    RxSessionBase,
    SessionStatus,
    TxSessionBase,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    CtxTransferStatus,
    GenTransferStatus,
    KvCacheTransceiver,
)
from tensorrt_llm._torch.disaggregation.native.bounce import (
    config_from_size as bounce_config_from_size,
)
from tensorrt_llm._torch.disaggregation.native.fetch import PeerFetch
from tensorrt_llm._torch.disaggregation.native.perf_logger import perf_log_manager
from tensorrt_llm._torch.disaggregation.native.publish import PeerPublish
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    create_cache_reuse_adapter,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_physical_pool,
    get_pool_view_num_layers,
)
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings import DataType, LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

_FP4_MLA_OWNERSHIP_BRIDGE_ENV = "TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE"


def _find_consensus_request_ids(request_ids_all_ranks, sync_size):
    frequency_map = defaultdict(int)
    consensus = []
    for rid in chain.from_iterable(request_ids_all_ranks):
        frequency_map[rid] += 1
    for rid, freq in sorted(frequency_map.items(), key=lambda x: x[1], reverse=True):
        if freq == sync_size:
            consensus.append(rid)
        else:
            break
    return consensus


def _validate_fp4_mla_bridge_profile(
    mapping: Mapping,
    kv_cache_manager: KVCacheManager,
    cache_transceiver_config: CacheTransceiverConfig,
) -> bool:
    """Select the bridge only for its explicit no-retry FP4-MLA deployment cell."""
    if os.getenv(_FP4_MLA_OWNERSHIP_BRIDGE_ENV, "0") != "1":
        return False
    fp4_mla_layout = (
        isinstance(kv_cache_manager, KVCacheManagerV2)
        and kv_cache_manager.is_disagg
        and kv_cache_manager.dtype == DataType.NVFP4
        and kv_cache_manager.kv_cache_type == CacheTypeCpp.SELFKONLY
    )
    supported = (
        fp4_mla_layout
        and os.getenv("TRTLLM_DISAGG_NO_RETRY", "0") == "1"
        and not use_pure_python_transfer_agent()
        and cache_transceiver_config.kv_transfer_timeout_ms is not None
        and cache_transceiver_config.kv_transfer_timeout_ms > 0
        and mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and cache_transceiver_config.kv_cache_bounce_size_mb == 0
        and not cache_transceiver_config.enable_pipelined_transfer
        and os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") != "1"
        and os.getenv("TRTLLM_DISAGG_LAYERWISE") != "1"
    )
    if not supported:
        raise ValueError(
            "FP4 MLA lifecycle bridge requires a disaggregated NVFP4 SELFKONLY "
            "KVCacheManagerV2, the C++ NIXL agent binding, a finite timeout, "
            "no-retry ADP, PP1/CP1, "
            "async monolithic non-layerwise transfer, and bounce disabled"
        )
    return True


class KvCacheTransceiverV2(KvCacheTransceiver):
    @property
    def consumes_transfer_buffer(self) -> bool:
        return False

    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._dist: Distributed = dist
        self._kv_cache_manager = kv_cache_manager
        self._mapping = mapping
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        if self.kv_transfer_timeout_ms is None:
            raise ValueError("KvCacheTransceiverV2 requires a finite kv_transfer_timeout_ms")
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms
        self._sender_future_timeout_ms = (
            cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        )
        transfer_timeout_s = self.kv_transfer_timeout_ms / 1000.0
        sender_wait_slice_s = (
            self._sender_future_timeout_ms / 1000.0
            if self._sender_future_timeout_ms is not None
            else None
        )
        self._check_compatible()
        enforce_physical_ownership = _validate_fp4_mla_bridge_profile(
            mapping, kv_cache_manager, cache_transceiver_config
        )
        self._fp4_mla_bridge_enabled = enforce_physical_ownership
        self._reuse_adapter: CacheReuseAdapter = create_cache_reuse_adapter(kv_cache_manager)

        self._device_id = torch.cuda.current_device()
        logger.info(f"device_id: {self._device_id} in KvCacheTransceiverV2")
        # Setup interleaves MPI collectives (broadcast/allgather below) with
        # per-rank native NIXL/UCX initialization inside TransferWorker. A rank
        # that blocks or dies in the native phase leaves its peers stuck in the
        # next collective, so log each phase per rank to make the blocking
        # rank/phase identifiable from the logs (see the 'setup:' lines).
        rank = self._dist.rank
        logger.info(f"KvCacheTransceiverV2 setup: rank={rank} broadcast instance name (collective)")
        self._instance_name = self._broadcast_instance_name()
        logger.info(
            f"KvCacheTransceiverV2 setup: rank={rank} creating TransferWorker "
            "(native NIXL agent init + KV memory registration)"
        )
        self._transfer_worker = TransferWorker(
            TransferWorkerConfig(
                kv_cache_manager=kv_cache_manager,
                device_id=self._device_id,
                instance_name=self._instance_name,
                # Context-only requests are released after KV transfer completes, so many batches
                # can be in-flight simultaneously. AuxBuffer holds only small CPU metadata, so a
                # large multiplier is cheap.
                max_concurrent_sessions=max(1, int(kv_cache_manager.max_batch_size)) * 20000,
                tx_timeout_s=sender_wait_slice_s,
                tx_overall_timeout_s=transfer_timeout_s,
                rx_timeout_s=transfer_timeout_s,
                enforce_physical_ownership=enforce_physical_ownership,
                # kv_cache_bounce_size_mb is the shared bounce capacity; agent_bounce_buffer_enable
                # routes it to exactly one implementation: the Python bounce below (per-region,
                # size 0 = off; the per-transfer size gates are internal, tuned via env:
                # TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS for plain-KV payloads,
                # TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES for recurrent-state payloads) or the C++
                # transfer-agent staging buffer (bounce v2, one buffer shared by send and recv).
                bounce=bounce_config_from_size(
                    0
                    if cache_transceiver_config.agent_bounce_buffer_enable
                    else cache_transceiver_config.kv_cache_bounce_size_mb
                ),
                agent_buffer_size_mb=(
                    cache_transceiver_config.kv_cache_bounce_size_mb
                    if cache_transceiver_config.agent_bounce_buffer_enable
                    else 0
                ),
                agent_bounce_params=cache_transceiver_config.agent_bounce_params,
            )
        )
        if enforce_physical_ownership:
            logger.info(
                "FP4 MLA KV ownership bridge ENABLED: "
                f"rank={rank}/{mapping.world_size}, backend=NIXL, runtime=PYTHON, "
                "request_schedule_required=GENERATION_FIRST, attention_dp=True, pp=1, cp=1, "
                "retry=False, async=True, layerwise=False, bounce_mb=0, "
                f"kv_transfer_timeout_ms={self.kv_transfer_timeout_ms}"
            )
        logger.info(
            f"KvCacheTransceiverV2 setup: rank={rank} TransferWorker ready; "
            "broadcast context endpoint (collective)"
        )
        self._dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
        self._context_info_endpoint = self._broadcast_context_endpoint()
        self._init_sync_policy()
        logger.info(f"KvCacheTransceiverV2 setup: rank={rank} exchange rank info (collective)")
        self._exchange_rank_info()
        logger.info(f"KvCacheTransceiverV2 setup: rank={rank} complete")

        self._send_sessions: Dict[int, TxSessionBase] = {}
        self._recv_sessions: Dict[int, RxSessionBase] = {}
        self._send_reqs = {}
        self._recv_reqs = {}
        self._wait_reqs = {}
        self._page_table = self._transfer_worker.page_table
        try:
            self._enable_pipelined_transfer = self._resolve_pipelined_transfer(
                cache_transceiver_config
            )
        except ValueError:
            # The transfer worker is already live by this point.
            self._transfer_worker.shutdown()
            raise
        # _chunk_num_bytes() is this rank's KV shard, so scale by tp_size to get the request total (kv_cache_size),
        # except under attention DP where the local count already is the total.
        # Helix CP ranks hold disjoint block sets, so they scale the request
        # total the same way TP shards do (metric only).
        self._kv_size_rank_factor = (
            1 if mapping.enable_attention_dp else max(1, mapping.tp_size * mapping.cp_size)
        )

        # Sticky role markers; flip True once any session opens, used to short-circuit
        # per-iter tp_allgather when this transceiver never sends/receives.
        self._ever_had_send_session: bool = False
        self._ever_had_recv_session: bool = False

    def _broadcast_instance_name(self) -> str:
        if self._dist.rank == 0:
            name = str(uuid.uuid4())
            self._dist.broadcast(name, 0)
            return name
        return cast(str, self._dist.broadcast(None, 0))

    def _broadcast_context_endpoint(self) -> str:
        if self._dist.rank == 0:
            endpoint = self._transfer_worker.rank_info_server_endpoint or ""
            self._dist.broadcast(endpoint, 0)
            return endpoint
        return cast(str, self._dist.broadcast(None, 0))

    def _init_sync_policy(self):
        m = self._mapping
        self._ctx_need_tp_sync = m.tp_size > 1 and not m.enable_attention_dp
        self._ctx_need_pp_sync = m.pp_size > 1
        if m.enable_attention_dp:
            # DP groups schedule independently, but the PP ranks and the helix CP
            # ranks of one DP group share the same requests, so gen-side consensus
            # spans PP x CP (like the C++ transceiver's mGroupDataComm). Without it
            # a CP rank can start decoding before its partner received the KV and
            # the helix all-to-all deadlocks.
            self._gen_need_sync = m.pp_size * m.cp_size > 1
            self._gen_allgather: Callable = self._dp_group_allgather
        else:
            self._gen_need_sync = m.world_size > 1
            self._gen_allgather = self._dist.allgather

    def _dp_group_allgather(self, obj):
        """Allgather over the PP x CP ranks of this DP group (CP first, then PP)."""
        m = self._mapping
        gathered = list(self._dist.cp_allgather(obj)) if m.cp_size > 1 else [obj]
        if m.pp_size > 1:
            gathered = [x for part in self._dist.pp_allgather(gathered) for x in part]
        return gathered

    def _exchange_rank_info(self):
        endpoints = cast(list, self._dist.allgather(self._transfer_worker.sender_endpoint))
        layer_num = len(self._kv_cache_manager.pp_layers)
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager) and not isinstance(
            self._kv_cache_manager, MambaHybridCacheManagerV2
        ):
            layer_num += len(self._kv_cache_manager._impl.mamba_layer_offsets)
        layer_num_per_pp = cast(list, getattr(self._dist, "pp_allgather")(layer_num))
        self._transfer_worker.populate_instance_and_rank_info(
            endpoints=endpoints, layer_num_per_pp=layer_num_per_pp
        )
        logger.info(f"transfer worker ctx_server_endpoints: {endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self._context_info_endpoint: {self._context_info_endpoint}")

    def get_status_dump(self) -> str:
        """Return a one-line summary of transceiver state for debugging hangs."""

        def summarize(
            sessions: Dict[int, Any],
            include_receiver_ready: bool,
        ) -> str:
            sessions_snapshot = list(sessions.values())
            status_counts = Counter()
            receiver_ready = 0
            for session in sessions_snapshot:
                status = session.status
                if isinstance(status, SessionStatus):
                    status_counts[status] += 1
                else:
                    status_counts["unknown"] += 1

                if include_receiver_ready:
                    receiver_ready += int(bool(session.receiver_ready))

            fields = [
                f"sessions={len(sessions_snapshot)}",
                f"init={status_counts[SessionStatus.INIT]}",
                f"ready_to_transfer={status_counts[SessionStatus.READY]}",
                f"transferring={status_counts[SessionStatus.TRANSFERRING]}",
                f"transferred={status_counts[SessionStatus.TRANSFERRED]}",
                f"error={status_counts[SessionStatus.ERROR]}",
                f"cancelled={status_counts[SessionStatus.CANCELLED]}",
                f"unknown={status_counts['unknown']}",
            ]
            if include_receiver_ready:
                fields.append(f"peer_ready={receiver_ready}/{len(sessions_snapshot)}")
            return ", ".join(fields)

        tx_status = summarize(self._send_sessions, include_receiver_ready=True)
        rx_status = summarize(self._recv_sessions, include_receiver_ready=False)
        return (
            f"KV cache transceiver | backend=NIXL | TX({tx_status}) | RX({rx_status}) | "
            f"waiting_for_peer_info={len(self._wait_reqs)}"
        )

    def shutdown(self) -> None:
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            shutdown_lock = threading.Lock()
            self._shutdown_lock = shutdown_lock
        with shutdown_lock:
            self._shutdown_once()

    def _shutdown_once(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        # This flag records completed teardown, not an attempted shutdown. If
        # an active owner refuses closure, leave it false so shutdown can be
        # retried after the physical operation drains.
        # Close receive owners before touching send-side or worker state. The
        # separate drain snapshot would not be sufficient: late evidence can make an
        # owner unretirable before close() acquires its lock.
        for rid, session in list(self._recv_sessions.items()):
            self._close_session_or_raise(session, rid, "shutdown")
        for rid, session in list(self._send_sessions.items()):
            self._close_session_or_raise(session, rid, "shutdown")
        self._send_sessions.clear()
        self._send_reqs.clear()
        self._recv_sessions.clear()
        self._recv_reqs.clear()
        self._transfer_worker.shutdown()
        self._shutdown_complete = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

    def _get_mamba_slot_for_request(self, req: LlmRequest) -> Optional[int]:
        """Get the mamba state slot index for a request, or None."""
        if isinstance(self._kv_cache_manager, MambaHybridCacheManagerV2):
            if self._kv_cache_manager.local_num_mamba_layers > 0:
                return self._kv_cache_manager._request_id_to_state_index[req.py_request_id]
        elif isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            return self._kv_cache_manager.mamba_cache_index[req.py_request_id]
        return None

    def _create_chunk(self, req: LlmRequest) -> Chunk:
        """Just the blocks, for the callers that have no use for the rest of the extent."""
        return self._describe_local(req)

    def _create_cache_extent(self, req: LlmRequest) -> CacheExtent:
        """This request's whole prompt as the sole piece of a transfer, under the name it is
        registered by."""
        rid = get_unique_rid(req)
        assert rid is not None
        return CacheExtent(name=rid, local=self._describe_local(req))

    def _describe_local(self, req: LlmRequest) -> Chunk:
        """The blocks this rank holds, one list per layer group.

        Every paged group gets ``ceil(prompt_len / tpb)`` entries indexed by block ordinal: the
        local pool slot, or -1 where this side has nothing there. Eviction and allocation state
        come straight from the cache manager (``get_block_ordinals``), so ctx and gen never have
        to agree on *when* a block left the window -- the sender simply pairs the ordinals both
        sides still hold. The only request-derived bound is prompt_len, which drops the
        speculative tail and the ctx first-token block (num_extra_kv_tokens slots are not
        transferred; both sides use prompt_len, so the ranges stay consistent). On the receiver
        the already-cached prefix is masked so the sender skips it. A STATE group carries one
        slot, or nothing when this rank has none.
        """
        adapter = self._reuse_adapter
        tpb = adapter.tokens_per_block
        assert self._page_table is not None
        layer_groups = self._page_table.layer_groups
        prompt_blocks = (req.prompt_len + tpb - 1) // tpb

        is_gen_only = req.is_generation_only_request
        cached_per_lg = (
            adapter.get_cached_token_count_per_layer_group(req, layer_groups)
            if is_gen_only
            else [0] * len(layer_groups)
        )

        empty = np.array([], dtype=np.int64)
        groups = []
        kinds = [lg.kind for lg in layer_groups]
        for idx, lg in enumerate(layer_groups):
            if lg.kind == CacheKind.STATE:
                slot = self._get_mamba_slot_for_request(req)
                group = np.array([slot], dtype=np.int64) if slot is not None else empty
            else:
                # Block lists carry beam 0 only (beam-search attention reads
                # prompt positions through beam 0's block table), so the
                # positional path applies to every beam width.
                ordinals = adapter.get_block_ordinals(req, idx, lg)
                group = self._positional_window(ordinals, prompt_blocks, cached_per_lg[idx] // tpb)
            groups.append(group)

        return Chunk(
            block_ids_per_layer_groups=groups,
            kind_per_layer_group=kinds,
            token_range=TokenRange(start=0, end=req.prompt_len),
            is_last=True,
        )

    @staticmethod
    def _positional_window(
        ordinals: np.ndarray, prompt_blocks: int, cached_blocks: int
    ) -> np.ndarray:
        """Fit a manager block table into the prompt's ordinal range.

        Pads with -1 when fewer blocks are allocated than the prompt spans
        (incremental ctx allocation), truncates the speculative / first-token
        tail, and masks the receiver's cached prefix.
        """
        window = np.full(prompt_blocks, -1, dtype=np.int64)
        n = min(int(ordinals.size), prompt_blocks)
        window[:n] = ordinals[:n]
        window[: min(cached_blocks, prompt_blocks)] = -1
        return window

    def _chunk_num_bytes(self, chunk: Chunk) -> int:
        """Local-rank KV bytes covered by a chunk (sum of num_valid_blocks * pool.slot_bytes), enough to populate
        kv_cache_size and unblock the perf-metric timestamps that gate on it.

        Counterpart accounting: the bounce reserve sizing (bounce/impl.py block_bytes_per_group)
        computes per-block bytes for the same layer groups but reads pool 0 only, while this sums
        every pool view of a group. The pool-0-only sizing gap for multi-pool attention groups is
        tracked under TRTLLM-15194; keep the two accountings in mind together when changing either.
        """
        pt = self._page_table
        if pt is None:
            return 0
        total = 0
        for lg_id, block_ids in enumerate(chunk.block_ids_per_layer_groups):
            if block_ids is None or block_ids.size == 0:
                continue
            n = int((block_ids >= 0).sum())
            if n == 0:
                continue
            lg = pt.layer_groups[lg_id]
            for pv in lg.pool_views:
                pool = get_physical_pool(pt, lg_id, pv.pool_idx)
                if lg.kind == CacheKind.STATE:
                    # STATE: n=1 (one slot), but transfer covers all layers of
                    # the view. The physical slot may hold several roles, so
                    # size by the view's per-layer bytes, not the pool's slot.
                    num_layers = get_pool_view_num_layers(pv)
                    total += num_layers * pv.bytes_per_layer
                else:
                    # Attention: n blocks, each slot covers all layers.
                    total += n * pool.slot_bytes
        return total

    @staticmethod
    def _need_aux_transfer(req: LlmRequest) -> bool:
        params = req.py_disaggregated_params
        return params is not None and params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST

    def _validate_bridge_req(self, req: LlmRequest, synchronous: bool = False) -> bool:
        if not getattr(self, "_fp4_mla_bridge_enabled", False):
            return True
        params = req.py_disaggregated_params
        rid = None if params is None else params.disagg_request_id
        if (
            synchronous
            or params is None
            or params.schedule_style != DisaggScheduleStyle.GENERATION_FIRST
            or type(rid) is not int
            or rid < 0
        ):
            logger.error(
                "FP4 MLA lifecycle bridge requires async generation-first requests "
                "with a non-negative integer disagg_request_id"
            )
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            return False
        return True

    def _ctx_consensus(self, local_ids: list) -> list:
        # TP consensus: ensure all TP ranks have peer info
        sync_size = self._dist.tp_size if self._ctx_need_tp_sync else 1
        all_ranks = self._dist.tp_allgather(local_ids) if self._ctx_need_tp_sync else [local_ids]
        ready_ids = _find_consensus_request_ids(all_ranks, sync_size)

        # PP consensus: ensure all PP ranks have peer info before promoting.
        # In PP, the first PP rank schedules and propagates to others. If a
        # request is promoted on the first rank but peer info hasn't arrived
        # on other ranks, respond_and_send_async on those ranks would fail
        # to dispatch the KV transfer (gen-first skips listener dispatch).
        # TODO: This is a workaround for functionality: pp_allgather impacts
        # the pp loop performance. One possible solution is to let pp rank0
        # decide the ready request ids, the other pp ranks treat the unready
        # request as ctx-first requests.
        if self._ctx_need_pp_sync:
            pp_all_ranks = getattr(self._dist, "pp_allgather")(ready_ids)
            ready_ids = _find_consensus_request_ids(pp_all_ranks, self._mapping.pp_size)

        return ready_ids

    def _gen_consensus(self, local_ids: list) -> list:
        if not self._gen_need_sync:
            return list(local_ids)
        all_ranks = self._gen_allgather(local_ids)
        return _find_consensus_request_ids(all_ranks, len(all_ranks))

    @staticmethod
    def _union(all_lists: List[List[int]]) -> set:
        merged: set = set()
        for ids in all_lists:
            merged.update(ids)
        return merged

    @staticmethod
    def _intersection(all_lists: List[List[int]], n_ranks: int) -> set:
        if n_ranks == 0:
            return set()
        cnt: Dict[int, int] = defaultdict(int)
        for ids in all_lists:
            for rid in set(ids):
                cnt[rid] += 1
        return {rid for rid, c in cnt.items() if c == n_ranks}

    def _consensus_outcome(
        self,
        to_process,
        cancelled,
        failed,
        completed,
        allgather: Callable,
        need_sync: bool,
        locally_quiesced=None,
    ):
        # CANCELLED/FAILED on any rank → global; COMPLETED only when ALL ranks agree.
        # Quiescence, when requested, also requires agreement from every rank.
        # Batch the id lists into one allgather to cut the per-step collective count.
        local_outcome = [list(cancelled), list(failed), list(completed)]
        if locally_quiesced is not None:
            local_outcome.append(list(locally_quiesced))
        if not need_sync:
            packed = [local_outcome]
        else:
            packed = list(allgather(local_outcome))
        all_c = [p[0] for p in packed]
        all_f = [p[1] for p in packed]
        all_done = [p[2] for p in packed]
        n = len(all_c)
        global_cancelled = self._union(all_c)
        global_failed = self._union(all_f)
        global_completed = self._intersection(all_done, n)
        new_cancelled = [rid for rid in to_process if rid in global_cancelled]
        cancel_set = set(new_cancelled)
        new_failed = [rid for rid in to_process if rid in global_failed and rid not in cancel_set]
        terminal = cancel_set | set(new_failed)
        new_completed = [
            rid for rid in to_process if rid in global_completed and rid not in terminal
        ]
        if locally_quiesced is not None:
            all_quiesced = [p[3] for p in packed]
            global_quiesced = self._intersection(all_quiesced, n)
            new_quiesced = [rid for rid in to_process if rid in global_quiesced]
            return new_cancelled, new_failed, new_completed, new_quiesced
        return new_cancelled, new_failed, new_completed

    def _gen_consensus_outcome(self, to_process, cancelled, failed, completed):
        # A failure/cancellation may be global, but reuse is safe only after
        # every participating rank has drained its local physical accessor.
        locally_retirable = []
        for rid in to_process:
            session = self._recv_sessions[rid]
            if not self._ownership_blocks_retirement(session):
                locally_retirable.append(rid)
        new_cancelled, new_failed, new_completed, globally_retirable = self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            self._gen_allgather,
            self._gen_need_sync,
            locally_retirable,
        )
        retirable = set(globally_retirable)
        return (
            [rid for rid in new_cancelled if rid in retirable],
            [rid for rid in new_failed if rid in retirable],
            new_completed,
        )

    def _ctx_consensus_outcome(self, to_process, cancelled, failed, completed, locally_quiesced):
        # TP first, then PP. A local timeout remains nonterminal, so it is
        # represented by the absence of that request from completed.
        c, f, d, q = self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            self._dist.tp_allgather,
            self._ctx_need_tp_sync,
            locally_quiesced,
        )
        if self._ctx_need_pp_sync:
            pp_allgather: Callable = getattr(self._dist, "pp_allgather")
            c, f, d, q = self._consensus_outcome(to_process, c, f, d, pp_allgather, True, q)
        return c, f, d, q

    def _sync_transfer_timing(self, reqs: list):
        """Allgather timing for a batch of completed requests in one collective.

        Matches C++ ``batchUpdateKVCacheTransferBW()`` in ``cacheTransceiver.cpp``.
        Only runs when ``TRTLLM_KVCACHE_TIME_OUTPUT_PATH`` is set (same gate
        as C++) and multi-rank sync is needed.  All ranks that participate in
        the allgather update their local request objects.
        """
        if not reqs:
            return
        if not os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH"):
            return
        if not self._gen_need_sync:
            return

        # Pack local timing for all completed requests into one dict.
        local_data = {
            get_unique_rid(req): (
                req.get_kv_cache_transfer_start(),
                req.get_kv_cache_transfer_end(),
                req.kv_cache_size,
            )
            for req in reqs
        }

        # Single allgather for the whole batch.
        all_data = self._gen_allgather(local_data)

        # Merge: per-rid min(start), max(end), sum(size) across ranks.
        merged: dict = {}
        for rank_data in all_data:
            for rid, (start, end, size) in rank_data.items():
                if rid in merged:
                    prev = merged[rid]
                    merged[rid] = (
                        min(prev[0], start),
                        max(prev[1], end),
                        prev[2] + size,
                    )
                else:
                    merged[rid] = (start, end, size)

        # Every rank updates its own local requests.
        rid_to_req = {get_unique_rid(r): r for r in reqs}
        for rid, (min_start, max_end, total_size) in merged.items():
            req = rid_to_req.get(rid)
            if req is not None:
                req.set_kv_cache_transfer_start(min_start)
                req.set_kv_cache_transfer_end(max_end)
                req.set_kv_cache_size(total_size)

    def _collect_done(self, sessions: dict, reqs: dict):
        """Scan sessions and return (completed_rids, failed_rids)."""
        completed, failed = [], []
        for rid, session in sessions.items():
            if session.is_completed():
                completed.append(rid)
            elif session.has_failed():
                if self._ownership_blocks_retirement(session):
                    continue
                failed.append(rid)
        return completed, failed

    def _build_to_process(
        self, sessions: dict, consensus: list, wait_num: int, block_all: bool
    ) -> list:
        if block_all:
            return list(sessions.keys())
        to_process = list(consensus)
        for rid in sessions:
            if len(to_process) >= wait_num:
                break
            if rid not in to_process:
                to_process.append(rid)
        return to_process

    def _close_failed_sessions(
        self, sessions: dict, reqs: dict, failed: list, mark_retired: bool = False
    ):
        for rid in failed:
            session = sessions.get(rid)
            if session is None:
                continue
            self._close_session_or_raise(session, rid, "failed")
            req = reqs.pop(rid, None)
            if req is not None:
                if not mark_retired:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                else:
                    req.py_kv_send_session_retired = True
            sessions.pop(rid, None)

    def _retire_send_session(
        self,
        rid: int,
        req: Optional[LlmRequest] = None,
        *,
        outcome: str = "retired",
        session_already_closed: bool = False,
    ) -> None:
        """Close a send session and prevent later chunks from recreating it."""
        session = self._send_sessions.get(rid)
        if session is not None and not session_already_closed:
            if getattr(session, "_enforce_physical_ownership", False):
                self._close_session_or_raise(session, rid, outcome)
            else:
                session.close()
        self._send_sessions.pop(rid, None)
        # Early teardown may occur before _send_reqs is populated.
        req = req if req is not None else self._send_reqs.get(rid)
        self._send_reqs.pop(rid, None)
        if req is not None:
            req.py_kv_send_session_retired = True

    @staticmethod
    def _ownership_blocks_retirement(session: object) -> bool:
        """Whether an ownership-enabled session still holds physical resources."""
        if not getattr(session, "_enforce_physical_ownership", False):
            return False
        resources_drained = getattr(session, "resources_drained", None)
        return resources_drained is None or not resources_drained()

    def _close_session_or_raise(self, session: object, rid: int, outcome: str) -> None:
        """Close one terminal session or fail-stop instead of diverging locally."""
        if self._ownership_blocks_retirement(session):
            raise RuntimeError(
                f"refusing to retire {outcome} KV transfer rid={rid}: "
                "physical resources remain active"
            )
        if session.close() is False:
            raise RuntimeError(
                f"refusing to retire {outcome} KV transfer rid={rid}: session close refused"
            )

    def _apply_aux(self, session, req: LlmRequest):
        """Unpack aux tokens from session into request's context_phase_params."""
        session.unpack_aux(req)
        first_gen_tokens = req.py_first_gen_tokens  # type: ignore[attr-defined]
        draft_tokens = req.py_draft_tokens
        if req.context_phase_params is None:
            assert req.py_request_id is not None
            req.context_phase_params = ContextPhaseParams(
                first_gen_tokens=first_gen_tokens,
                req_id=req.py_request_id,
                opaque_state=b"",
                draft_tokens=draft_tokens,
                ctx_dp_rank=0,
                disagg_info_endpoint="",
            )
        else:
            req.context_phase_params.first_gen_tokens = first_gen_tokens
            req.context_phase_params.draft_tokens = draft_tokens

    def _get_or_create_send_session(self, req: LlmRequest) -> Optional[TxSessionBase]:
        self._ever_had_send_session = True
        rid = get_unique_rid(req)
        assert rid is not None
        if rid not in self._send_sessions:
            if req.py_kv_send_session_retired:
                logger.warning(
                    f"rid={rid}: send session already retired; failing the request "
                    "rather than re-creating one with no peer registration"
                )
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                return None
            self._send_sessions[rid] = self._transfer_worker.create_tx_session(req)
        return self._send_sessions[rid]

    def _finalize_send(self, req: LlmRequest, session: TxSessionBase):
        """Pack aux and set context phase params. Call after the last slice."""
        rid = get_unique_rid(req)
        assert rid is not None
        if self._need_aux_transfer(req):
            session.pack_aux(req)
            session.send_aux()
        req.context_phase_params = ContextPhaseParams(
            first_gen_tokens=[],
            req_id=rid,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=self._dp_rank,
            disagg_info_endpoint=self._context_info_endpoint,
        )
        self._send_reqs[rid] = req

    @property
    def pipeline_transfer_enabled(self) -> bool:
        """Whether pipelined prefill-transfer is enabled."""
        return self._enable_pipelined_transfer

    def has_inflight_transfer(self, req: LlmRequest) -> bool:
        """Whether a transfer session still owns this request's resources."""
        rid = get_unique_rid(req)
        return rid in self._send_sessions or rid in self._recv_sessions

    def has_retired_send_session(self, req: LlmRequest) -> bool:
        """Whether req's send session was torn down before its last slice."""
        return req.py_kv_send_session_retired and get_unique_rid(req) not in self._send_sessions

    def _build_prefill_extent(
        self,
        req: LlmRequest,
    ) -> Optional[CacheExtent]:
        """Build an extent for completed prefill blocks, or None."""
        assert req.py_beam_width == 1, "beam_width > 1 is not supported for chunked KV transfer"
        rid = get_unique_rid(req)
        assert rid is not None
        self._send_reqs[rid] = req

        chunk_start_pos, chunk_end_pos = req.py_last_context_chunk
        tpb = self._kv_cache_manager.tokens_per_block

        # Include any reused prefix in the first transferred chunk.
        is_first_chunk = chunk_start_pos == req.prepopulated_prompt_len
        is_last_chunk = req.context_remaining_length == 0
        # Defer partial blocks except at the prompt's final chunk.
        chunk_start = 0 if is_first_chunk else chunk_start_pos // tpb
        chunk_end = (chunk_end_pos + tpb - 1) // tpb if is_last_chunk else chunk_end_pos // tpb

        # The final chunk is sent even when it contains no complete block.
        if chunk_end <= chunk_start and not is_last_chunk:
            return None

        whole = self._describe_local(req)
        all_block_ids = whole.block_ids_per_layer_groups
        chunk_block_ids = []
        assert self._page_table is not None
        for lg, block_ids in zip(self._page_table.layer_groups, all_block_ids):
            window_size = getattr(lg, "sliding_window_size", None)
            if window_size is not None and window_size < req.prompt_len:
                # SWA pages can leave the active window between chunks. Defer
                # the group and send its complete final active window at once.
                chunk_block_ids.append(block_ids if is_last_chunk else np.full_like(block_ids, -1))
            else:
                # Positional: keep the chunk's ordinals, blank everything else.
                chunk = np.full_like(block_ids, -1)
                chunk[chunk_start:chunk_end] = block_ids[chunk_start:chunk_end]
                chunk_block_ids.append(chunk)
        if not is_last_chunk and not any((ids >= 0).any() for ids in chunk_block_ids):
            return None
        # The block window is rounded up; the span this piece delivers is not. Every reader below
        # rounds back to blocks, and a recurrent-state group reads the end as an exact checkpoint.
        chunk_end_token = min(chunk_end * tpb, req.prompt_len) if is_last_chunk else chunk_end * tpb
        return CacheExtent(
            name=rid,
            local=Chunk(
                block_ids_per_layer_groups=chunk_block_ids,
                kind_per_layer_group=whole.kind_per_layer_group,
                token_range=TokenRange(start=chunk_start * tpb, end=chunk_end_token),
                is_last=is_last_chunk,
            ),
        )

    @nvtx_range("KvCacheTransceiverV2.respond_and_send_async")
    def respond_and_send_async(self, req: LlmRequest) -> None:
        """Send the request's next KV slice to the generation server."""

        if not self._validate_bridge_req(req):
            return
        self._ever_had_send_session = True
        # Keep the latest slice's transfer-start timestamp.
        req.set_kv_cache_transfer_start(tensorrt_llm.bindings.global_steady_clock_now())
        rid = get_unique_rid(req)
        assert rid is not None
        session = self._get_or_create_send_session(req)
        if session is None:
            return
        bridge_enabled = getattr(self, "_fp4_mla_bridge_enabled", False)
        if bridge_enabled:
            # Root the request before backend admission so its source pages
            # outlive every ownership-enabled NIXL operation.
            self._send_reqs[rid] = req
            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
            # A remote cancellation can win before the local TxSession is
            # created. Sender.setup_session() records that terminal state and
            # reports safe pre-submission failures to every known receiver;
            # leave retirement to the normal status path without attempting
            # to publish KV or auxiliary memory afterward.
            if session.has_failed():
                return
        try:
            if self.pipeline_transfer_enabled:
                extent = self._build_prefill_extent(req)
                if extent is None:
                    return
            else:
                extent = self._create_cache_extent(req)
            chunk = extent.local
            # The handle that comes back is the contract's answer about this piece. What retires
            # the request is the sweep over the session tables, as it was before.
            PeerPublish(session, req).publish(extent)

            if chunk.is_last:
                self._finalize_send(req, session)
                if not bridge_enabled:
                    # Preserve the legacy pipelined-transfer state boundary:
                    # intermediate chunks must remain schedulable for prefill.
                    req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        except Exception as error:
            if bridge_enabled:
                cast(Any, session).set_exception(f"transfer admission failed: {error}")
            raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_sync")
    def request_and_receive_sync(self, req: LlmRequest) -> None:
        if not self._validate_bridge_req(req, synchronous=True):
            return
        rid = get_unique_rid(req)
        self._ever_had_recv_session = True
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_sync: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        session = None
        fetches = None
        try:
            extent = self._create_cache_extent(req)
            fetches = self._open_peer_source(req)
            # Same submission the asynchronous entry makes; what differs is who waits. The session
            # underneath is read back for the blocking wait, the auxiliary buffer and the close.
            fetches.fetch(extent)
            session = self._legacy_session(fetches)
            self._recv_sessions[rid] = session
            self._recv_reqs[rid] = req
            result = session.wait_complete(blocking=True)

            if result == WaitResult.COMPLETED:
                # KV-transfer timing setters deferred to #15871 (clock-source consistency); size only.
                req.set_kv_cache_size(
                    self._chunk_num_bytes(extent.local) * self._kv_size_rank_factor
                )
                if self._need_aux_transfer(req):
                    self._apply_aux(session, req)
                self._assert_disagg_history_declared(req)
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            else:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
        except Exception:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            raise
        finally:
            # The adapter opens the session before it submits, so a submission that raises leaves
            # one to close while the local above is still unassigned; read it back off the adapter.
            if session is None and fetches is not None:
                session = self._legacy_session(fetches)
            close_succeeded = session is None or session.close() is not False
            if close_succeeded:
                self._recv_sessions.pop(rid, None)
                self._recv_reqs.pop(rid, None)
            else:
                logger.error(
                    f"request_and_receive_sync: retaining rid={rid} because receive "
                    "resources remain active"
                )

    def _open_peer_source(self, req: LlmRequest) -> PeerFetch:
        """The paired backend: pull this request's cache from the worker that ran its context."""
        return PeerFetch(self._transfer_worker, req)

    @staticmethod
    def _legacy_session(fetches: PeerFetch):
        """The native session under the adapter, for the sweep that retires it and for the
        auxiliary buffer.

        Named as the escape hatch it is: both jobs are the reason the contract does not cover
        everything here yet.
        """
        return fetches.session

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest) -> None:
        """Start background KV cache receive from the context server.

        The receiver always uses a single monolithic slice.  Chunking is
        sender-only: the sender splits its source blocks into chunks and
        slices the receiver's destination blocks to match each chunk.

        Args:
            req: The generation request whose KV cache blocks to receive
                into.
        """
        if not self._validate_bridge_req(req):
            return
        self._ever_had_recv_session = True
        req.set_kv_cache_transfer_start(tensorrt_llm.bindings.global_steady_clock_now())
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_async: rid={rid} already has a recv session, skipping"
            )
            return
        extent = self._create_cache_extent(req)
        req.py_kv_cache_xfer_bytes = self._chunk_num_bytes(extent.local) * self._kv_size_rank_factor
        fetches = self._open_peer_source(req)
        # Claimed to be transferring only once there is something to transfer: a builder that
        # raises above leaves the request where it was, not in a state nothing advances.
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        # Root the destination before publication can escape. Registering only on success leaves a
        # failed publication as a session the sweep can see but cannot pair with a request.
        self._recv_reqs[rid] = req
        try:
            # The handle that comes back is the contract's answer about this piece. What retires
            # the request is the sweep over the session tables, as it was before.
            fetches.fetch(extent)
        except Exception:
            # No session means no publication and nothing the sweep could ever pair the request
            # with, so the registration made here is undone here and the request goes terminal.
            if self._legacy_session(fetches) is None:
                del self._recv_reqs[rid]
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
            raise
        finally:
            # The session exists even when publication failed, and the legacy sweep owns it.
            # TODO: An idle bounce reservation is not handed back on the failure path.
            session = self._legacy_session(fetches)
            if session is not None:
                self._recv_sessions[rid] = session

    def check_context_transfer_status(
        self, at_least_request_num: Optional[int], mark_complete: bool = False
    ) -> CtxTransferStatus:
        # A worker that never sends KV has nothing to reconcile here, so skip the consensus. Safe
        # because the flag flips together on every rank and never resets, so they all skip in step;
        # gating on the live session dict instead would not be, since a cancel clears it per-rank.
        # Keep the original sweep (only when tp/pp sync is on) so nothing is leaked.
        if not self._ever_had_send_session:
            if self._ctx_need_tp_sync or self._ctx_need_pp_sync:
                self._transfer_worker.sweep_stale_req_infos()
            return CtxTransferStatus([], [])
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_sessions_for_interval(
                self._send_sessions,
                self._send_reqs,
                wait_num,
                self._sender_future_timeout_ms,
            )

        local_completed, local_failed = self._collect_done(self._send_sessions, self._send_reqs)
        to_process = self._build_to_process(
            self._send_sessions,
            self._ctx_consensus(local_completed + local_failed),
            0 if need_progress else wait_num,
            block_all,
        )

        completed, failed, cancelled = [], [], []
        for rid in to_process:
            session = self._send_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if session.status == SessionStatus.CANCELLED:
                if getattr(session, "_enforce_physical_ownership", False) and (
                    session.has_transferring_tasks()
                ):
                    continue
                cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result is None:
                continue
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} exceeded "
                    f"kv_transfer_timeout_ms={self.kv_transfer_timeout_ms}ms; "
                    "keeping it in progress"
                )
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                failed.append(rid)

        # CANCELLED/ERROR are logical terminal states; a fabric write may still
        # be reading the request's pages. Include each rank's physical-writer
        # state in the outcome exchange so retirement also requires consensus.
        locally_quiesced = [
            rid for rid in to_process if not self._send_sessions[rid].has_transferring_tasks()
        ]
        cancelled, failed, completed, quiesced_ids = self._ctx_consensus_outcome(
            to_process, cancelled, failed, completed, locally_quiesced
        )
        quiesced = set(quiesced_ids)
        cancelled = [rid for rid in cancelled if rid in quiesced]
        failed = [rid for rid in failed if rid in quiesced]

        for rid in cancelled:
            self._retire_send_session(rid, outcome="cancelled")

        for rid in completed:
            req = self._send_reqs[rid]
            self._retire_send_session(rid, outcome="completed")
            if mark_complete:
                req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
        self._close_failed_sessions(self._send_sessions, self._send_reqs, failed, mark_retired=True)

        # Sweep orphaned RecvReqInfo entries from ADP broadcast on non-assigned
        # DP ranks (entries that will never have a TxSession created for them).
        self._transfer_worker.sweep_stale_req_infos()

        if getattr(self, "_fp4_mla_bridge_enabled", False):
            # CtxTransferStatus has no cancellation channel. In the qualified
            # no-retry bridge, a globally quiesced cancellation is a terminal
            # send failure that the executor must consume to release its
            # matching AsyncTransferManager claim.
            failed.extend(rid for rid in cancelled if rid not in failed)
        return CtxTransferStatus(completed, failed)

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]) -> GenTransferStatus:
        if not self._ever_had_recv_session and not self._gen_need_sync:
            return GenTransferStatus([], [], [])
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_gen_sessions_for_poll_interval(wait_num)

        local_completed, local_failed = self._collect_done(self._recv_sessions, self._recv_reqs)
        to_process = self._build_to_process(
            self._recv_sessions,
            self._gen_consensus(local_completed + local_failed),
            0 if need_progress else wait_num,
            block_all,
        )

        completed, failed, cancelled = [], [], []
        for rid in to_process:
            session = self._recv_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if session.status == SessionStatus.CANCELLED:
                if self._ownership_blocks_retirement(session):
                    continue
                # Session cancelled — either by local cancel_request() (user
                # cancel) or by a remote CANCEL_SESSION message (e.g. CTX
                # server timeout).  Return the req objects so the caller can
                # distinguish the two cases and set the appropriate state.
                cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                req = self._recv_reqs[rid]
                if session.transfer_end_time is not None:
                    req.set_kv_cache_transfer_end(session.transfer_end_time)
                if session.kv_cache_size_bytes > 0:
                    req.set_kv_cache_size(session.kv_cache_size_bytes)
                completed.append(rid)
            elif result == WaitResult.FAILED:
                # RxSession.wait_complete() already withholds FAILED while an
                # ownership-enabled accessor remains active. Keep the caller
                # boundary fail-closed as well so alternate/test session
                # implementations cannot authorize request retirement early.
                if self._ownership_blocks_retirement(session):
                    continue
                failed.append(rid)
            # else: None — KV done but aux still in flight; re-poll next cycle

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed = self._gen_consensus_outcome(
            to_process, cancelled, failed, completed
        )

        cancelled_reqs = []
        for rid in cancelled:
            session = self._recv_sessions[rid]
            self._close_session_or_raise(session, rid, "cancelled")
            cancelled_reqs.append(self._recv_reqs[rid])
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]

        # Log gen-side transfer summary after consensus.
        if completed and os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH"):
            # Batch-sync timing for all completed requests in one allgather.
            self._sync_transfer_timing([self._recv_reqs[rid] for rid in completed])
            for rid in completed:
                req = self._recv_reqs[rid]
                perf_log_manager.log_gen_transfer_summary(
                    unique_rid=rid,
                    instance_name=self._instance_name,
                    instance_rank=self._mapping.rank,
                    gen_side_transfer_time_ms=req.kv_cache_transfer_time_ms,
                    kv_cache_size=req.kv_cache_size,
                )

        for rid in completed:
            session = self._recv_sessions[rid]
            req = self._recv_reqs[rid]
            # transfer_end already stamped at completion detection above.
            req.set_kv_cache_size(getattr(req, "py_kv_cache_xfer_bytes", 0))
            if self._need_aux_transfer(req):
                self._apply_aux(session, req)
            self._assert_disagg_history_declared(req)
            self._close_session_or_raise(session, rid, "completed")
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]
        if failed:
            logger.warning(
                f"Disagg gen transfer FAILED rank={self._dist.rank} "
                f"rids={failed} gen_need_sync={self._gen_need_sync}"
            )
        self._close_failed_sessions(self._recv_sessions, self._recv_reqs, failed)

        return GenTransferStatus(completed, failed, cancelled_reqs)

    def _poll_gen_sessions_for_poll_interval(self, wait_num: int) -> None:
        self._poll_sessions_for_interval(
            self._recv_sessions,
            self._recv_reqs,
            wait_num,
            self.kv_transfer_poll_interval_ms,
        )

    def _poll_sessions_for_interval(
        self,
        sessions: dict,
        reqs: dict,
        wait_num: int,
        poll_interval_ms: Optional[int],
    ) -> None:
        # The exit condition can only ever count in-flight sessions, so a
        # target above len(sessions) is unsatisfiable and the loop would sleep
        # out the whole interval for nothing. The idle executor loop hits
        # exactly that: check_context_transfer_status(1) with no in-flight
        # sends burns a full kv_transfer_sender_future_timeout_ms per
        # iteration and delays scheduling of newly arrived requests
        # (nvbugs 6647405). Clamping is safe under rank-divergent session
        # counts because this helper is purely local (no collectives).
        wait_num = min(wait_num, len(sessions))
        if wait_num <= 0:
            return
        poll_interval_s = (poll_interval_ms or 0) / 1000.0
        deadline = time.monotonic() + poll_interval_s
        while True:
            completed, failed = self._collect_done(sessions, reqs)
            if len(completed) + len(failed) >= wait_num:
                return
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            for session in sessions.values():
                session.wait_complete(blocking=False)
            time.sleep(min(0.001, remaining_s))

    def check_gen_transfer_complete(self):
        return len(self._recv_sessions) == 0

    def _assert_disagg_history_declared(self, req: LlmRequest) -> None:
        """Verify the V2 scheduler pre-declared prompt_len as history.

        Call right before the TRANS_COMPLETE state transition.  The V2
        scheduler's ``_try_schedule_disagg_gen_init`` calls
        ``prepare_disagg_gen_init``, which sets ``kv_cache.history_length``
        to ``prompt_len`` at allocation time so SWA stale computation
        skips pre-window blocks. If that contract is violated, SWA /
        sparse-attn pools may fill with pre-window prompt blocks and the
        V2 scheduler can deadlock under high concurrency (e.g., benchmark
        fill-phase).

        No-op for V1 managers (which lack ``get_history_length``) and for
        V2 caches with only full-context life cycles (where the watermark
        has no allocation effect).
        """
        get_history = getattr(self._kv_cache_manager, "get_history_length", None)
        if get_history is None:
            return
        prompt_len = getattr(req, "prompt_len", None)
        if not prompt_len or prompt_len <= 0:
            return
        history = get_history(req)
        if history is None:
            # Cache was already released (e.g., cancelled mid-transfer); nothing to verify.
            return
        if history < prompt_len:
            raise RuntimeError(
                f"req {req.py_request_id}: kv_cache.history_length={history} "
                f"< prompt_len={prompt_len} at TRANS_COMPLETE boundary. "
                f"V2 scheduler must call prepare_disagg_gen_init() in "
                f"_try_schedule_disagg_gen_init."
            )

    def cancel_request(self, req: LlmRequest) -> bool:
        """Cancel the transfer for the given request.

        Returns False if any task is mid-write (TRANSFERRING); caller must
        retry next iteration. Returns True when safe to free KV memory.
        """
        rid = get_unique_rid(req)

        # Not yet started (generation-first wait queue).
        self._wait_reqs.pop(rid, None)

        has_transferring = False

        if rid in self._send_sessions:
            self._send_sessions[rid].cancel()
            if self._send_sessions[rid].has_transferring_tasks():
                has_transferring = True
            elif self._send_sessions[rid].close() is False:
                has_transferring = True
            else:
                self._retire_send_session(
                    rid,
                    req,
                    outcome="cancelled",
                    session_already_closed=True,
                )

        if rid in self._recv_sessions:
            self._recv_sessions[rid].cancel()
            if self._recv_sessions[rid].has_transferring_tasks():
                has_transferring = True
            elif self._recv_sessions[rid].close() is False:
                has_transferring = True
            else:
                del self._recv_reqs[rid]
                del self._recv_sessions[rid]

        if has_transferring:
            return False
        return True

    def get_disaggregated_params(self) -> Dict[str, Any]:
        # Keep this aligned with fields populated in respond_and_send_async().
        # These values are server-level metadata used to seed generation-first
        # requests before context-phase response data arrives.
        #
        # With ADP (enable_attention_dp), ctx_dp_rank is not known at
        # registration time because the context scheduler has not yet assigned
        # the request to a DP rank.  Return None so that the gen-side Receiver
        # broadcasts REQUEST_DATA to all ctx DP ranks.  The actual ctx_dp_rank
        # is stamped into ContextPhaseParams by respond_and_send_async() after
        # the prefill is scheduled.
        ctx_dp_rank = None if self._mapping.enable_attention_dp else self._dp_rank
        return {
            "ctx_dp_rank": ctx_dp_rank,
            "ctx_info_endpoint": [self._context_info_endpoint]
            if self._context_info_endpoint
            else None,
        }

    def prepare_context_requests(self, requests: List[LlmRequest]) -> None:
        # Place new generation-first context requests into wait state, then
        # use allgather consensus to promote ready requests to CONTEXT_INIT.
        for req in requests:
            if not self._validate_bridge_req(req):
                continue
            rid = get_unique_rid(req)
            if rid not in self._send_sessions:
                self._wait_reqs[rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

        # Nothing waiting on any rank, so skip the consensus. The waiting set is the same on every
        # rank, so they all skip together.
        if not self._wait_reqs:
            return

        # Check which waiting requests have peer info locally, then allgather
        # consensus so all TP/PP ranks agree before promoting.
        # Without consensus, background peer info arriving at different times on
        # different ranks causes scheduling mismatches → hang.
        local_ready = [
            rid
            for rid in self._wait_reqs
            if self._transfer_worker.has_all_peer_req_infos_for_send(rid)
        ]
        for rid in self._ctx_consensus(local_ready):
            self._wait_reqs[rid].state = LlmRequestState.CONTEXT_INIT
            del self._wait_reqs[rid]

    def _check_compatible(self):
        if self._mapping.cp_size != 1 and not self._mapping.has_cp_helix():
            raise ValueError(
                f"KvCacheTransceiverV2: _check_compatible: unsupported context parallelism "
                f"(cp_size={self._mapping.cp_size}, cp_type={self._mapping.cp_config.get('cp_type')}); "
                f"only cp_size == 1 or helix CP is supported"
            )

    def _resolve_pipelined_transfer(self, cfg: CacheTransceiverConfig) -> bool:
        """Return whether this rank supports pipelined KV transfer."""
        if not cfg.enable_pipelined_transfer:
            return False
        blockers = []
        # The Python bounce reserves a receiver region for the whole request, not per chunk.
        # The C++ transfer-agent bounce (agent_bounce_buffer_enable) stages each transfer request
        # independently below the Python layer, so a pipelined chunk is just another request.
        # TODO: This only sees the local config, so a chunking sender and a bouncing receiver pass.
        if cfg.kv_cache_bounce_size_mb != 0 and not cfg.agent_bounce_buffer_enable:
            blockers.append(
                f"the Python bounce buffer (kv_cache_bounce_size_mb="
                f"{cfg.kv_cache_bounce_size_mb} without agent_bounce_buffer_enable)"
            )
        if isinstance(self._kv_cache_manager, (MambaHybridCacheManager, MambaHybridCacheManagerV2)):
            blockers.append("a Mamba/hybrid cache manager")
        # Refused for the configuration, not for the piece: the per-chunk guard downstream only sees
        # a piece short of the whole prompt, so a prompt that fits one chunk would slip past it.
        if self._mapping.cp_size > 1:
            blockers.append(f"context parallelism (cp_size={self._mapping.cp_size})")
        # Policies other than all_reusable defer the whole prompt's commit to the final
        # chunk. Committing a prefix block that a concurrent request already committed
        # rebases it onto that request's page and frees ours, which an earlier chunk's
        # in-flight read may still be sourcing from.
        if (
            isinstance(self._kv_cache_manager, KVCacheManagerV2)
            and self._kv_cache_manager.enable_block_reuse
            and self._kv_cache_manager.block_reuse_policy != BlockReusePolicy.ALL_REUSABLE
        ):
            blockers.append(
                f"block reuse policy '{self._kv_cache_manager.block_reuse_policy}' "
                f"(only '{BlockReusePolicy.ALL_REUSABLE}' is supported)"
            )
        if blockers:
            raise ValueError(
                "enable_pipelined_transfer is not supported with " + "; ".join(blockers)
            )
        return True

    def commit_blocks_for_reuse(self, req) -> None:
        self._reuse_adapter.commit_blocks_for_reuse(req)

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
