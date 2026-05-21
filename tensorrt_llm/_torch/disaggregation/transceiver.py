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
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    create_cache_reuse_adapter,
)
from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MambaHybridCacheManager,
    PythonMambaCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping


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
    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._dist: Distributed = dist
        self._kv_cache_manager = kv_cache_manager
        self._mapping = mapping
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
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
            assert isinstance(self._kv_cache_manager._impl, PythonMambaCacheManager), (
                "CppMambaCacheManager is not supported with Python transceiver, please set TRTLLM_USE_CPP_MAMBA=0"
            )
            layer_num += len(self._kv_cache_manager._impl.mamba_layer_offsets)
        layer_num_per_pp = cast(list, getattr(self._dist, "pp_allgather")(layer_num))
        self._transfer_worker.populate_instance_and_rank_info(
            endpoints=endpoints, layer_num_per_pp=layer_num_per_pp
        )
        logger.info(f"transfer worker ctx_server_endpoints: {endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self._context_info_endpoint: {self._context_info_endpoint}")

    def shutdown(self):
        if getattr(self, "_shutdown", False):
            return
        self._shutdown = True
        for session in list(self._send_sessions.values()):
            session.close()
        for session in list(self._recv_sessions.values()):
            session.close()
        self._send_sessions.clear()
        self._send_reqs.clear()
        self._recv_sessions.clear()
        self._recv_reqs.clear()
        self._transfer_worker.shutdown()

    def _create_kv_slice(
        self,
        req: LlmRequest,
        token_range: Optional[TokenRange] = None,
        is_last_slice: bool = True,
    ) -> KVSlice:
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

        if token_range is None and req.prompt_len > 0:
            token_range = TokenRange(start=0, end=req.prompt_len)

        groups = []
        for idx, lg in enumerate(layer_groups):
            if isinstance(lg, MambaLayerGroup):
                groups.append(np.array([], dtype=np.int64))
                continue
            block_ids = adapter.get_block_ids(req, idx, lg)
            window_size = lg.sliding_window_size

            if window_size is not None:
                # Drop stale blocks the manager may still expose (V1 pre-eviction).
                total_blocks = (req.prompt_len + tpb - 1) // tpb
                stale_end = max(0, (req.prompt_len + 1 - window_size) // tpb)
                expected_valid = max(0, total_blocks - stale_end)
                if block_ids.size > expected_valid:
                    block_ids = (
                        block_ids[-expected_valid:]
                        if expected_valid > 0
                        else np.array([], dtype=np.int64)
                    )
                # Adapter contract: SWA cached_tokens >= stale_end*tpb (clamped).
                cache_skip = cached_per_lg[idx] // tpb - stale_end
                assert cache_skip >= 0, (
                    f"SWA adapter must clamp cached_tokens to >= stale_end*tpb "
                    f"(cached={cached_per_lg[idx]}, stale_end*tpb={stale_end * tpb})"
                )
            else:
                cache_skip = cached_per_lg[idx] // tpb

            if cache_skip > 0:
                block_ids = (
                    block_ids[cache_skip:]
                    if cache_skip < block_ids.size
                    else np.array([], dtype=np.int64)
                )

            groups.append(block_ids)

        mamba_state_index = None
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            mamba_state_index = self._kv_cache_manager.mamba_cache_index[req.py_request_id]

        return KVSlice(
            is_last_slice=is_last_slice,
            block_ids_per_layer_groups=groups,
            mamba_state_index=mamba_state_index,
            token_range=token_range,
        )

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
    def _allgather_or_passthrough(
        local_ids, allgather: Callable, need_sync: bool
    ) -> List[List[int]]:
        if not need_sync:
            return [list(local_ids)]
        return list(allgather(list(local_ids)))

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
        self, to_process, cancelled, failed, completed, allgather: Callable, need_sync: bool
    ):
        # CANCELLED/FAILED on any rank → global; COMPLETED only when ALL ranks agree.
        all_c = self._allgather_or_passthrough(cancelled, allgather, need_sync)
        all_f = self._allgather_or_passthrough(failed, allgather, need_sync)
        all_done = self._allgather_or_passthrough(completed, allgather, need_sync)
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
        return new_cancelled, new_failed, new_completed

    def _gen_consensus_outcome(self, to_process, cancelled, failed, completed):
        return self._consensus_outcome(
            to_process, cancelled, failed, completed, self._gen_allgather, self._gen_need_sync
        )

    def _ctx_consensus_outcome(self, to_process, cancelled, failed, completed, timed_out):
        # TP first, then PP.  timed_out is local-only (back-off signal).
        c, f, d = self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            self._dist.tp_allgather,
            self._ctx_need_tp_sync,
        )
        if self._ctx_need_pp_sync:
            pp_allgather: Callable = getattr(self._dist, "pp_allgather")
            c, f, d = self._consensus_outcome(to_process, c, f, d, pp_allgather, True)
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

    def _close_failed_sessions(self, sessions: dict, reqs: dict, failed: list):
        for rid in failed:
            reqs[rid].state = LlmRequestState.DISAGG_TRANS_ERROR
            sessions[rid].close()
            del reqs[rid]
            del sessions[rid]

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
        session = self._get_or_create_send_session(req)
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        session.send(self._create_kv_slice(req))
        self._finalize_send(req, session)

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_sync")
    def request_and_receive_sync(self, req: LlmRequest):
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_sync: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        session = None
        try:
            session = self._transfer_worker.create_rx_session(req)
            self._recv_sessions[rid] = session
            self._recv_reqs[rid] = req
            session.receive(self._create_kv_slice(req))
            result = session.wait_complete(blocking=True)

            if result == WaitResult.COMPLETED:
                if self._need_aux_transfer(req):
                    self._apply_aux(session, req)
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            else:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
        except Exception:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            raise
        finally:
            if session is not None:
                session.close()
            self._recv_sessions.pop(rid, None)
            self._recv_reqs.pop(rid, None)

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest):
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_async: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        session = self._transfer_worker.create_rx_session(req)
        self._recv_sessions[rid] = session
        session.receive(self._create_kv_slice(req))
        self._recv_reqs[rid] = req

    def check_context_transfer_status(
        self, at_least_request_num: Optional[int], mark_complete: bool = False
    ):
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0

        local_completed, local_failed = self._collect_done(self._send_sessions, self._send_reqs)
        to_process = self._build_to_process(
            self._send_sessions,
            self._ctx_consensus(local_completed + local_failed),
            wait_num,
            block_all,
        )

        completed, timed_out, failed, cancelled = [], [], [], []
        for rid in to_process:
            session = self._send_sessions[rid]
            result = session.wait_complete()
            if session.status == SessionStatus.CANCELLED:
                cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} timed out after {self._sender_future_timeout_ms}ms"
                )
                timed_out.append(rid)
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                failed.append(rid)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed, timed_out = self._ctx_consensus_outcome(
            to_process, cancelled, failed, completed, timed_out
        )

        for rid in cancelled:
            self._send_sessions[rid].close()
            del self._send_reqs[rid]
            del self._send_sessions[rid]

        for rid in completed:
            if mark_complete:
                self._send_reqs[rid].state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
            self._send_sessions[rid].close()
            del self._send_reqs[rid]
            del self._send_sessions[rid]
        self._close_failed_sessions(self._send_sessions, self._send_reqs, failed)

        # Sweep orphaned RecvReqInfo entries from ADP broadcast on non-assigned
        # DP ranks (entries that will never have a TxSession created for them).
        self._transfer_worker.sweep_stale_req_infos()

        return completed, failed

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]):
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0

        local_completed, local_failed = self._collect_done(self._recv_sessions, self._recv_reqs)
        to_process = self._build_to_process(
            self._recv_sessions,
            self._gen_consensus(local_completed + local_failed),
            wait_num,
            block_all,
        )

        completed, failed, cancelled = [], [], []
        for rid in to_process:
            session = self._recv_sessions[rid]
            result = session.wait_complete(blocking=block_all)
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
            # else: None — KV done but aux still in flight; re-poll next cycle

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed = self._gen_consensus_outcome(
            to_process, cancelled, failed, completed
        )

        cancelled_reqs = []
        for rid in cancelled:
            cancelled_reqs.append(self._recv_reqs[rid])
            self._recv_sessions[rid].close()
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]

        for rid in completed:
            session = self._recv_sessions[rid]
            req = self._recv_reqs[rid]
            if self._need_aux_transfer(req):
                self._apply_aux(session, req)
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            session.close()
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]
        self._close_failed_sessions(self._recv_sessions, self._recv_reqs, failed)

        return completed, failed, cancelled_reqs

    def check_gen_transfer_complete(self):
        return len(self._recv_sessions) == 0

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
            else:
                self._send_sessions[rid].close()
                del self._send_reqs[rid]
                del self._send_sessions[rid]

        if rid in self._recv_sessions:
            self._recv_sessions[rid].cancel()
            if self._recv_sessions[rid].has_transferring_tasks():
                has_transferring = True
            else:
                self._recv_sessions[rid].close()
                del self._recv_reqs[rid]
                del self._recv_sessions[rid]

        if has_transferring:
            return False  # mid-write; caller must retry
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

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # Place new generation-first context requests into wait state, then
        # use allgather consensus to promote ready requests to CONTEXT_INIT.
        for req in requests:
            rid = get_unique_rid(req)
            if rid not in self._send_sessions:
                self._wait_reqs[rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

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
        if self._mapping.cp_size != 1:
            raise ValueError(
                f"KvCacheTransceiverV2: _check_compatible: only support context parallelism is 1: "
                f"cp_size: {self._mapping.cp_size}"
            )

    def commit_blocks_for_reuse(self, req) -> None:
        self._reuse_adapter.commit_blocks_for_reuse(req)

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
