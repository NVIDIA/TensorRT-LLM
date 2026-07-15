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

import threading
import time
import uuid
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    RxSessionBase,
    SessionStatus,
    TokenRange,
    TxSessionBase,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.native.bounce import (
    config_from_size as bounce_config_from_size,
)
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    create_cache_reuse_adapter,
)
from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

# A non-drained transceiver is deliberately retained even if its caller drops
# the last reference. It owns the LlmRequest/session objects that keep receive
# targets from being reclaimed before terminal transport evidence arrives.
_NON_DRAINED_TRANSCEIVERS: set["KvCacheTransceiverV2"] = set()


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


class KvCacheTransceiverV2(KvCacheTransceiver):
    @property
    def requires_physical_drain_before_request_release(self) -> bool:
        return True

    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._session_admission_lock = threading.RLock()
        self._shutdown_started = False
        self._shutdown = False
        self._dist: Distributed = dist
        self._kv_cache_manager = kv_cache_manager
        self._mapping = mapping
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms
        self._sender_future_timeout_ms = (
            cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        )
        self._check_compatible()
        self._reuse_adapter: CacheReuseAdapter = create_cache_reuse_adapter(kv_cache_manager)

        self._device_id = torch.cuda.current_device()
        logger.info(f"device_id: {self._device_id} in KvCacheTransceiverV2")
        self._instance_name = self._broadcast_instance_name()
        self._transfer_worker = TransferWorker(
            TransferWorkerConfig(
                kv_cache_manager=kv_cache_manager,
                device_id=self._device_id,
                instance_name=self._instance_name,
                # Context-only requests are released after KV transfer completes, so many batches
                # can be in-flight simultaneously. AuxBuffer holds only small CPU metadata, so a
                # large multiplier is cheap.
                max_concurrent_sessions=max(1, int(kv_cache_manager.max_batch_size)) * 20000,
                tx_timeout_s=self._sender_future_timeout_ms / 1000.0,
                rx_timeout_s=self.kv_transfer_timeout_ms / 1000.0,
                # Size 0 turns bounce off; the block-count gate is internal (tuned via env).
                bounce=bounce_config_from_size(cache_transceiver_config.kv_cache_bounce_size_mb),
            )
        )
        self._dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
        self._context_info_endpoint = self._broadcast_context_endpoint()
        self._init_sync_policy()
        self._exchange_rank_info()

        self._send_sessions: Dict[int, TxSessionBase] = {}
        self._recv_sessions: Dict[int, RxSessionBase] = {}
        self._send_reqs = {}
        self._recv_reqs = {}
        self._wait_reqs = {}
        self._page_table = self._transfer_worker.page_table
        # _slice_num_bytes() is this rank's KV shard, so scale by tp_size to get the request total (kv_cache_size),
        # except under attention DP where the local count already is the total.
        self._kv_size_rank_factor = 1 if mapping.enable_attention_dp else max(1, mapping.tp_size)

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
        self._gen_need_sync = not (m.world_size == 1 or (m.enable_attention_dp and m.pp_size == 1))
        pp_allgather: Callable = getattr(self._dist, "pp_allgather")
        self._gen_allgather: Callable = (
            pp_allgather if m.enable_attention_dp else self._dist.allgather
        )

    def _exchange_rank_info(self):
        endpoints = cast(list, self._dist.allgather(self._transfer_worker.sender_endpoint))
        layer_num = len(self._kv_cache_manager.pp_layers)
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            layer_num += len(self._kv_cache_manager._impl.mamba_layer_offsets)
        layer_num_per_pp = cast(list, getattr(self._dist, "pp_allgather")(layer_num))
        self._transfer_worker.populate_instance_and_rank_info(
            endpoints=endpoints, layer_num_per_pp=layer_num_per_pp
        )
        logger.info(f"transfer worker ctx_server_endpoints: {endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self._context_info_endpoint: {self._context_info_endpoint}")

    def _get_session_admission_lock(self):
        """Return the launch/shutdown gate, including partial-init cleanup."""
        admission_lock = getattr(self, "_session_admission_lock", None)
        if admission_lock is None:
            admission_lock = threading.RLock()
            self._session_admission_lock = admission_lock
        return admission_lock

    def shutdown(self) -> bool:
        admission_lock = self._get_session_admission_lock()
        with admission_lock:
            if getattr(self, "_shutdown", False):
                return True
            # Retain the ownership root before any fallible cancellation or
            # cleanup. A failed shutdown must remain retryable even when the
            # caller releases its last reference after this attempt.
            _NON_DRAINED_TRANSCEIVERS.add(self)
            # Close admission before taking any session snapshot. The same lock
            # covers owner enrollment through publication/worker enqueue, so a
            # launch cannot resume against a worker that shutdown just destroyed.
            self._shutdown_started = True
            cancel_failed_session_ids: set[int] = set()
            for direction, sessions in (
                ("send", self._send_sessions),
                ("receive", self._recv_sessions),
            ):
                for session in list(sessions.values()):
                    try:
                        # Cancellation closes future publication but leaves
                        # already-exposed resources owned until terminal
                        # results arrive.
                        session.cancel()
                    except Exception as e:
                        cancel_failed_session_ids.add(id(session))
                        logger.error(
                            f"KvCacheTransceiverV2 shutdown failed to cancel {direction} "
                            f"session {session.disagg_request_id}: {e}"
                        )
            try:
                worker_drained = self._transfer_worker.shutdown()
            except Exception as e:
                logger.error(
                    "KvCacheTransceiverV2 worker shutdown did not drain; retaining "
                    f"sessions and request resources for retry: {e}"
                )
                return False
            if not worker_drained:
                return False

            close_failed = False
            for direction, sessions, requests in (
                ("send", self._send_sessions, self._send_reqs),
                ("receive", self._recv_sessions, self._recv_reqs),
            ):
                for rid, session in list(sessions.items()):
                    if id(session) in cancel_failed_session_ids:
                        continue
                    try:
                        session.close()
                    except Exception as e:
                        close_failed = True
                        logger.error(
                            f"KvCacheTransceiverV2 shutdown failed to close {direction} "
                            f"session {session.disagg_request_id}: {e}"
                        )
                        continue
                    if sessions.get(rid) is session:
                        sessions.pop(rid, None)
                    requests.pop(rid, None)

            if cancel_failed_session_ids or close_failed:
                return False
            getattr(self, "_wait_reqs", {}).clear()
            self._shutdown = True
            _NON_DRAINED_TRANSCEIVERS.discard(self)
            return True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        try:
            drained = self.shutdown()
        except Exception as shutdown_error:
            if _exc_type is None:
                raise
            logger.error(
                "KvCacheTransceiverV2 shutdown also failed while propagating "
                f"{_exc_type.__name__}: {shutdown_error}"
            )
            return False
        if not drained:
            if _exc_type is None:
                raise RuntimeError("KV cache transceiver shutdown did not drain")
            logger.error(
                "KV cache transceiver shutdown did not drain while propagating "
                f"{_exc_type.__name__}"
            )
        return False

    def _create_kv_slice(self, req: LlmRequest) -> KVSlice:
        adapter = self._reuse_adapter
        tpb = adapter.tokens_per_block
        assert self._page_table is not None
        layer_groups = self._page_table.layer_groups

        is_gen_only = req.is_generation_only_request()
        cached_per_lg = (
            adapter.get_cached_token_count_per_layer_group(req, layer_groups)
            if is_gen_only
            else [0] * len(layer_groups)
        )

        token_range = None
        if req.prompt_len > 0:
            # end must match the trimmed block list below (ceil(prompt_len / tpb)
            # blocks). num_extra_kv_tokens slots (speculative decoding) are not
            # transferred. In the previously added support for ctx disabling
            # speculative decoding while gen enables it, both sides currently
            # use prompt_len as the transfer range, so the ranges stay
            # consistent.
            # TODO: the accuracy impact of not transferring num_extra_kv_tokens
            # on MTP and other speculative decoding paths is currently unclear;
            # revisit whether these extra KV slots need to be transferred.
            token_range = TokenRange(start=0, end=req.prompt_len)

        groups = []
        for idx, lg in enumerate(layer_groups):
            if isinstance(lg, MambaLayerGroup):
                groups.append(np.array([], dtype=np.int64))
                continue
            block_ids = adapter.get_block_ids(req, idx, lg)
            # Limit to prompt_len blocks, matching C++ cacheFormatter behavior.
            total_blocks = (req.prompt_len + tpb - 1) // tpb
            if block_ids.size > total_blocks:
                block_ids = block_ids[:total_blocks]
            window_size = lg.sliding_window_size

            if window_size is not None:
                # Drop stale blocks the manager may still expose (V1 pre-eviction).
                stale_end = max(0, (req.prompt_len + 1 - window_size) // tpb)
                expected_valid = max(0, total_blocks - stale_end)
                # Stale prefix already pruned above; skip reuse-hit blocks that
                # land inside the window. Clamp to 0: ctx side has cached_per_lg
                # synthetically 0, and a reuse hit may fall entirely inside the
                # stale region (those blocks were already pruned, no extra skip).
                cache_skip = max(0, cached_per_lg[idx] // tpb - stale_end)
            else:
                total_blocks = (req.prompt_len + tpb - 1) // tpb
                expected_valid = total_blocks
                cache_skip = cached_per_lg[idx] // tpb

            block_ids = self._trim_packed_beam_block_ids(
                block_ids,
                beam_width=req.py_beam_width,
                total_blocks=total_blocks,
                expected_valid=expected_valid,
                cache_skip=cache_skip,
            )

            groups.append(block_ids)

        mamba_state_index = None
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            mamba_state_index = self._kv_cache_manager.mamba_cache_index[req.py_request_id]

        return KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=groups,
            mamba_state_index=mamba_state_index,
            token_range=token_range,
        )

    def _slice_num_bytes(self, slice: KVSlice) -> int:
        """Local-rank KV bytes covered by a slice (sum of num_valid_blocks * pool.slot_bytes), enough to populate
        kv_cache_size and unblock the perf-metric timestamps that gate on it."""
        pt = self._page_table
        if pt is None:
            return 0
        total = 0
        for lg_id, block_ids in enumerate(slice.block_ids_per_layer_groups):
            if block_ids is None or block_ids.size == 0:
                continue
            lg = pt.layer_groups[lg_id]
            if isinstance(lg, MambaLayerGroup):
                continue
            n = int((block_ids >= 0).sum())
            if n == 0:
                continue
            for pv in lg.pool_views:
                pool = get_physical_pool(pt, lg_id, pv.pool_idx)
                total += n * pool.slot_bytes
        return total

    @staticmethod
    def _split_packed_beam_block_ids(
        block_ids: np.ndarray,
        beam_width: int,
        total_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split 1-D block IDs into beam-0 prefix and appended beam-tail blocks."""
        if beam_width <= 1 or block_ids.size <= total_blocks:
            return block_ids, np.array([], dtype=np.int64)
        tail_count = min(beam_width - 1, block_ids.size - total_blocks)
        if tail_count <= 0:
            return block_ids, np.array([], dtype=np.int64)
        return block_ids[:-tail_count], block_ids[-tail_count:]

    @staticmethod
    def _trim_packed_beam_block_ids(
        block_ids: np.ndarray,
        beam_width: int,
        total_blocks: int,
        expected_valid: int,
        cache_skip: int,
    ) -> np.ndarray:
        """Trim/skip beam-0 blocks while preserving packed beam-tail blocks."""
        if expected_valid <= 0:
            return np.array([], dtype=np.int64)

        beam0_block_ids, tail_block_ids = KvCacheTransceiverV2._split_packed_beam_block_ids(
            block_ids, beam_width, total_blocks
        )

        if beam0_block_ids.size > expected_valid:
            beam0_block_ids = (
                beam0_block_ids[-expected_valid:]
                if expected_valid > 0
                else np.array([], dtype=np.int64)
            )
            if beam0_block_ids.size == 0:
                tail_block_ids = np.array([], dtype=np.int64)

        if cache_skip > 0:
            if cache_skip < beam0_block_ids.size:
                beam0_block_ids = beam0_block_ids[cache_skip:]
            else:
                beam0_block_ids = np.array([], dtype=np.int64)
                tail_block_ids = np.array([], dtype=np.int64)

        if tail_block_ids.size == 0:
            return beam0_block_ids
        return np.concatenate([beam0_block_ids, tail_block_ids])

    @staticmethod
    def _need_aux_transfer(req: LlmRequest) -> bool:
        params = req.py_disaggregated_params
        return params is not None and params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST

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
        sync_size = (
            self._mapping.pp_size if self._mapping.enable_attention_dp else self._mapping.world_size
        )
        all_ranks = self._gen_allgather(local_ids) if self._gen_need_sync else [local_ids]
        return _find_consensus_request_ids(all_ranks, sync_size)

    @staticmethod
    def _allgather_or_passthrough(local_value: list, allgather: Callable, need_sync: bool) -> list:
        if not need_sync:
            return [list(local_value)]
        return list(allgather(list(local_value)))

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
        cleanup_ready,
        allgather: Callable,
        need_sync: bool,
    ):
        """Agree on logical outcome without outrunning local physical drain.

        Cancellation/failure is a union because one rank's logical failure must
        fail the distributed request. Cleanup readiness is an intersection:
        every rank must independently prove its local source/target accessors
        terminal before any rank removes its request owner.
        """
        payload = [
            list(cancelled),
            list(failed),
            list(completed),
            list(cleanup_ready),
        ]
        gathered = self._allgather_or_passthrough(payload, allgather, need_sync)
        all_c = [rank_payload[0] for rank_payload in gathered]
        all_f = [rank_payload[1] for rank_payload in gathered]
        all_done = [rank_payload[2] for rank_payload in gathered]
        all_ready = [rank_payload[3] for rank_payload in gathered]
        n = len(gathered)
        global_cancelled = self._union(all_c)
        global_failed = self._union(all_f)
        global_completed = self._intersection(all_done, n)
        globally_ready = self._intersection(all_ready, n)
        new_cancelled = [
            rid for rid in to_process if rid in global_cancelled and rid in globally_ready
        ]
        cancel_set = set(new_cancelled)
        new_failed = [
            rid
            for rid in to_process
            if rid in global_failed and rid in globally_ready and rid not in cancel_set
        ]
        terminal = cancel_set | set(new_failed)
        new_completed = [
            rid
            for rid in to_process
            if rid in global_completed and rid in globally_ready and rid not in terminal
        ]
        return new_cancelled, new_failed, new_completed

    def _gen_consensus_outcome(self, to_process, cancelled, failed, completed, cleanup_ready):
        return self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            cleanup_ready,
            self._gen_allgather,
            self._gen_need_sync,
        )

    def _ctx_consensus_outcome(
        self, to_process, cancelled, failed, completed, timed_out, cleanup_ready
    ):
        # TP first, then PP.  timed_out is local-only (back-off signal).
        c, f, d = self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            cleanup_ready,
            self._dist.tp_allgather,
            self._ctx_need_tp_sync,
        )
        if self._ctx_need_pp_sync:
            pp_allgather: Callable = getattr(self._dist, "pp_allgather")
            # A TP group appears ready to the PP stage only after the first
            # consensus admitted one terminal outcome for that request.
            c, f, d = self._consensus_outcome(
                to_process,
                c,
                f,
                d,
                c + f + d,
                pp_allgather,
                True,
            )
        return c, f, d, timed_out

    def _collect_done(self, sessions: dict, reqs: dict):
        """Scan sessions and return (completed_rids, failed_rids)."""
        completed, failed = [], []
        for rid, session in sessions.items():
            if session.is_completed():
                completed.append(rid)
            elif session.has_failed():
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
        self,
        sessions: dict,
        reqs: dict,
        failed: list,
        expected_sessions: dict,
        expected_reqs: dict,
    ) -> list:
        """Retire only the exact failed owners observed by one status poll."""
        retired = []
        with self._get_session_admission_lock():
            for rid in failed:
                session = expected_sessions.get(rid)
                req = expected_reqs.get(rid)
                if sessions.get(rid) is not session or reqs.get(rid) is not req:
                    continue
                session.close()
                if sessions.get(rid) is session and reqs.get(rid) is req:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                    reqs.pop(rid, None)
                    sessions.pop(rid, None)
                    retired.append(rid)
        return retired

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

    def _get_or_create_send_session(self, req: LlmRequest) -> TxSessionBase:
        rid = get_unique_rid(req)
        assert rid is not None
        if rid not in self._send_sessions:
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

    @nvtx_range("KvCacheTransceiverV2.respond_and_send_async")
    def respond_and_send_async(self, req: LlmRequest):
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            rid = get_unique_rid(req)
            assert rid is not None
            existing_owner = self._send_reqs.get(rid)
            if existing_owner is not None and existing_owner is not req:
                raise RuntimeError(f"send request {rid} already has a different live source owner")
            owner_newly_enrolled = existing_owner is None
            self._ever_had_send_session = True
            # Enroll the source owner before a worker can launch gather/NIXL.
            self._send_reqs[rid] = req
            session = None
            try:
                session = self._get_or_create_send_session(req)
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                session.send(self._create_kv_slice(req))
                self._finalize_send(req, session)
            except Exception:
                if session is None:
                    if owner_newly_enrolled:
                        self._send_reqs.pop(rid, None)
                else:
                    try:
                        session.cancel()
                        if not session.has_transferring_tasks():
                            session.close()
                            self._send_sessions.pop(rid, None)
                            self._send_reqs.pop(rid, None)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"respond_and_send_async: retaining rid={rid} after "
                            f"exception-path cleanup could not prove drain: {cleanup_error}"
                        )
                raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_sync")
    def request_and_receive_sync(self, req: LlmRequest):
        admission_lock = self._get_session_admission_lock()
        with admission_lock:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            admitted = self._request_and_receive_sync_admitted(req)

        if admitted is None:
            return
        session, kv_slice = admitted
        rid = get_unique_rid(req)
        assert rid is not None

        try:
            result = session.wait_complete(blocking=True)
        except Exception:
            with admission_lock:
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                    session.cancel()
                    if not session.has_transferring_tasks():
                        session.close()
                        self._recv_sessions.pop(rid, None)
                        self._recv_reqs.pop(rid, None)
            raise

        # Cancellation and shutdown use the same gate. They may have retired
        # this session while wait_complete() was blocked, so only the exact
        # enrolled owner may consume AUX data or remove the map entries.
        with admission_lock:
            owns_session = (
                self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req
            )
            if not owns_session:
                if result != WaitResult.COMPLETED or session.status == SessionStatus.CANCELLED:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                return
            try:
                if result == WaitResult.COMPLETED:
                    # KV-transfer timing setters deferred to #15871 (clock-source consistency);
                    # size only.
                    req.set_kv_cache_size(
                        self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
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
                session.close()
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    self._recv_sessions.pop(rid, None)
                    self._recv_reqs.pop(rid, None)

    def _request_and_receive_sync_admitted(self, req: LlmRequest):
        rid = get_unique_rid(req)
        assert rid is not None
        if rid in self._recv_sessions:
            if self._recv_reqs.get(rid) is not req:
                raise RuntimeError(
                    f"receive request {rid} already has a different live destination owner"
                )
            logger.warning(
                f"request_and_receive_sync: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        self._recv_reqs[rid] = req
        session = None
        try:
            session = self._transfer_worker.create_rx_session(req)
            self._recv_sessions[rid] = session
            kv_slice = self._create_kv_slice(req)
            session.receive(kv_slice)
            return session, kv_slice
        except Exception:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            if session is None:
                if self._recv_reqs.get(rid) is req:
                    self._recv_reqs.pop(rid, None)
            else:
                session.cancel()
                if not session.has_transferring_tasks():
                    session.close()
                    if self._recv_sessions.get(rid) is session:
                        self._recv_sessions.pop(rid, None)
                    if self._recv_reqs.get(rid) is req:
                        self._recv_reqs.pop(rid, None)
            raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest):
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            self._ever_had_recv_session = True
            rid = get_unique_rid(req)
            assert rid is not None
            if rid in self._recv_sessions:
                if self._recv_reqs.get(rid) is not req:
                    raise RuntimeError(
                        f"receive request {rid} already has a different live destination owner"
                    )
                logger.warning(
                    f"request_and_receive_async: rid={rid} already has a recv session, skipping"
                )
                return
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
            # Enroll the request owner before session construction can allocate an
            # AUX slot and before receive() can publish KV/AUX target addresses.
            # If either step raises, keep every owner enrolled so cleanup continues
            # through the normal physical-drain gate rather than dropping the only
            # Python-level owner during exception unwinding.
            self._recv_reqs[rid] = req
            session = None
            try:
                session = self._transfer_worker.create_rx_session(req)
                self._recv_sessions[rid] = session
                kv_slice = self._create_kv_slice(req)
                req.py_kv_cache_xfer_bytes = (
                    self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
                )
                session.receive(kv_slice)
            except Exception:
                if session is None:
                    # Constructor rollback owns any pre-publication allocation.
                    self._recv_reqs.pop(rid, None)
                else:
                    try:
                        session.cancel()
                        drained = not session.has_transferring_tasks()
                        if drained:
                            session.close()
                            self._recv_sessions.pop(rid, None)
                            self._recv_reqs.pop(rid, None)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"request_and_receive_async: retaining rid={rid} after "
                            f"exception-path cleanup could not prove drain: {cleanup_error}"
                        )
                raise

    def check_context_transfer_status(
        self,
        at_least_request_num: Optional[int],
        mark_complete: bool = False,
    ):
        # This is also the native sender's periodic control-result progress
        # hook. Run it before the idle fast path so cancel-before-session
        # failures are retried even when no TxSession is ever created.
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                return [], []
            self._transfer_worker.sweep_stale_req_infos()
            if not self._ever_had_send_session:
                return [], []
            send_sessions = dict(self._send_sessions)
            send_reqs = {rid: self._send_reqs[rid] for rid in send_sessions}
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0

        local_completed, local_failed = self._collect_done(send_sessions, send_reqs)
        to_process = self._build_to_process(
            send_sessions,
            self._ctx_consensus(local_completed + local_failed),
            wait_num,
            block_all,
        )

        completed, timed_out, failed, cancelled, cleanup_ready = [], [], [], [], []
        for rid in to_process:
            session = send_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if result is None:
                # Logical cancellation can precede source gather/NIXL drain.
                # Keep the source owner until wait_complete reports a terminal
                # physical state.
                continue
            if session.status == SessionStatus.CANCELLED:
                cancelled.append(rid)
                cleanup_ready.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
                cleanup_ready.append(rid)
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} timed out after {self._sender_future_timeout_ms}ms"
                )
                timed_out.append(rid)
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                failed.append(rid)
                cleanup_ready.append(rid)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed, timed_out = self._ctx_consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            timed_out,
            cleanup_ready,
        )

        retired_completed = []
        with self._get_session_admission_lock():
            for rid in completed:
                session = send_sessions.get(rid)
                req = send_reqs.get(rid)
                if (
                    self._send_sessions.get(rid) is not session
                    or self._send_reqs.get(rid) is not req
                ):
                    continue
                session.close()
                if mark_complete:
                    req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                if self._send_sessions.get(rid) is session and self._send_reqs.get(rid) is req:
                    self._send_reqs.pop(rid, None)
                    self._send_sessions.pop(rid, None)
                    retired_completed.append(rid)
        # A remotely cancelled context transfer is terminal for its local
        # AsyncTransferManager entry. Report it as an error so PyExecutor can
        # unpin the request instead of silently losing the transceiver session.
        terminal_failed = failed + cancelled
        retired_failed = self._close_failed_sessions(
            self._send_sessions,
            self._send_reqs,
            terminal_failed,
            send_sessions,
            send_reqs,
        )

        return retired_completed, retired_failed

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]):
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                return [], [], []
            if not self._ever_had_recv_session and not self._gen_need_sync:
                return [], [], []
            recv_sessions = dict(self._recv_sessions)
            recv_reqs = {rid: self._recv_reqs[rid] for rid in recv_sessions}
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_gen_sessions_for_poll_interval(wait_num, recv_sessions, recv_reqs)

        local_completed, local_failed = self._collect_done(recv_sessions, recv_reqs)
        to_process = self._build_to_process(
            recv_sessions,
            self._gen_consensus(local_completed + local_failed),
            0 if need_progress else wait_num,
            block_all,
        )

        completed, failed, cancelled, cleanup_ready = [], [], [], []
        for rid in to_process:
            session = recv_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if result not in (WaitResult.COMPLETED, WaitResult.FAILED):
                # Logical cancellation/failure may precede physical drain.
                # Neither a non-blocking miss nor a timeout proves that request
                # cleanup can no longer race a transfer accessor.
                continue
            cleanup_ready.append(rid)
            if session.status == SessionStatus.CANCELLED:
                # Session cancelled — either by local cancel_request() (user
                # cancel) or by a remote CANCEL_SESSION message (e.g. CTX
                # server timeout).  Return the req objects so the caller can
                # distinguish the two cases and set the appropriate state.
                cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result == WaitResult.FAILED:
                failed.append(rid)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed = self._gen_consensus_outcome(
            to_process, cancelled, failed, completed, cleanup_ready
        )

        retired_completed = []
        cancelled_reqs = []
        with self._get_session_admission_lock():
            for rid in cancelled:
                session = recv_sessions.get(rid)
                req = recv_reqs.get(rid)
                if (
                    self._recv_sessions.get(rid) is not session
                    or self._recv_reqs.get(rid) is not req
                ):
                    continue
                session.close()
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    self._recv_reqs.pop(rid, None)
                    self._recv_sessions.pop(rid, None)
                    cancelled_reqs.append(req)

            for rid in completed:
                session = recv_sessions.get(rid)
                req = recv_reqs.get(rid)
                if (
                    self._recv_sessions.get(rid) is not session
                    or self._recv_reqs.get(rid) is not req
                ):
                    continue
                # transfer_end already stamped at completion detection above.
                req.set_kv_cache_size(getattr(req, "py_kv_cache_xfer_bytes", 0))
                if self._need_aux_transfer(req):
                    self._apply_aux(session, req)
                self._assert_disagg_history_declared(req)
                session.close()
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    self._recv_reqs.pop(rid, None)
                    self._recv_sessions.pop(rid, None)
                    retired_completed.append(rid)
        retired_failed = self._close_failed_sessions(
            self._recv_sessions,
            self._recv_reqs,
            failed,
            recv_sessions,
            recv_reqs,
        )
        if retired_failed:
            logger.warning(
                f"Disagg gen transfer FAILED rank={self._dist.rank} "
                f"rids={retired_failed} gen_need_sync={self._gen_need_sync}"
            )

        return retired_completed, retired_failed, cancelled_reqs

    def _poll_gen_sessions_for_poll_interval(
        self, wait_num: int, sessions: dict, reqs: dict
    ) -> None:
        poll_interval_s = (self.kv_transfer_poll_interval_ms or 0) / 1000.0
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
        with self._get_session_admission_lock():
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
        assert rid is not None
        with self._get_session_admission_lock():
            # A stale cancellation must not remove a replacement request that
            # happens to reuse the same integer request ID.
            if self._wait_reqs.get(rid) is req:
                self._wait_reqs.pop(rid, None)

            cleanup_pending = False
            for direction, sessions, reqs in (
                ("send", self._send_sessions, self._send_reqs),
                ("receive", self._recv_sessions, self._recv_reqs),
            ):
                session = sessions.get(rid)
                if session is None or reqs.get(rid) is not req:
                    continue
                try:
                    session.cancel()
                    if session.has_transferring_tasks():
                        cleanup_pending = True
                        continue
                    session.close()
                except Exception as error:
                    cleanup_pending = True
                    logger.error(
                        f"Failed to cancel {direction} KV transfer for request {rid}; "
                        f"retaining its owner for retry: {error}"
                    )
                    continue
                if sessions.get(rid) is session and reqs.get(rid) is req:
                    reqs.pop(rid, None)
                    sessions.pop(rid, None)

            return not cleanup_pending

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

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # Place new generation-first context requests into wait state, then
        # use allgather consensus to promote ready requests to CONTEXT_INIT.
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            pending_admissions = {}
            planned_wait_owners = dict(self._wait_reqs)
            for req in requests:
                rid = get_unique_rid(req)
                assert rid is not None
                send_owner = self._send_reqs.get(rid)
                if rid in self._send_sessions:
                    if send_owner is not req:
                        raise RuntimeError(
                            f"send request {rid} already has a different live source owner"
                        )
                    continue
                if send_owner is not None and send_owner is not req:
                    raise RuntimeError(
                        f"send request {rid} already has a different live source owner"
                    )
                wait_owner = planned_wait_owners.get(rid)
                if wait_owner is not None and wait_owner is not req:
                    raise RuntimeError(
                        f"waiting request {rid} already has a different live source owner"
                    )
                planned_wait_owners[rid] = req
                pending_admissions[rid] = req

            # Validate the full batch before publishing any owner. A collision
            # must not leave an earlier request stranded in the wait ledger.
            for rid, req in pending_admissions.items():
                self._wait_reqs[rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

            if not self._wait_reqs:
                return

            # Check which waiting requests have peer info while the worker is
            # protected from concurrent shutdown. Promotion happens after the
            # collective and therefore revalidates each exact owner below.
            ready_owners = {
                rid: owner
                for rid, owner in self._wait_reqs.items()
                if self._transfer_worker.has_all_peer_req_infos_for_send(rid)
            }

        # Without consensus, background peer info arriving at different times on
        # different ranks causes scheduling mismatches → hang. Do not hold the
        # local admission lock across the distributed collective.
        ready_ids = self._ctx_consensus(list(ready_owners))
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                return
            for rid in ready_ids:
                owner = ready_owners.get(rid)
                if owner is not None and self._wait_reqs.get(rid) is owner:
                    owner.state = LlmRequestState.CONTEXT_INIT
                    self._wait_reqs.pop(rid, None)

    def _check_compatible(self):
        if self._mapping.cp_size != 1:
            raise ValueError(
                f"KvCacheTransceiverV2: _check_compatible: only support context parallelism is 1: "
                f"cp_size: {self._mapping.cp_size}"
            )

    def commit_blocks_for_reuse(self, req) -> None:
        self._reuse_adapter.commit_blocks_for_reuse(req)

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
