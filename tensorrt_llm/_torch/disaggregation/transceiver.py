import uuid
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, cast

import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    RxSessionBase,
    TxSessionBase,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.utils import get_global_layer_ids
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2
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
        self._is_v2_manager = isinstance(kv_cache_manager, KVCacheManagerV2)

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
        layer_num_per_pp = cast(
            list, getattr(self._dist, "pp_allgather")(len(self._kv_cache_manager.pp_layers))
        )
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

    def _get_block_ids(self, req: LlmRequest, group_idx: int, lg) -> list:
        if self._is_v2_manager:
            kv_cache_map = getattr(self._kv_cache_manager, "kv_cache_map")
            return list(
                kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                    group_idx, valid_only=True
                )
            )
        else:
            first_layer = get_global_layer_ids(lg)[0]
            return self._kv_cache_manager.get_batch_cache_indices(
                [req.py_request_id], layer_idx=first_layer
            )[0]

    def _create_kv_slice(self, req: LlmRequest) -> KVSlice:
        tpb = self._kv_cache_manager.tokens_per_block
        groups = []
        assert self._page_table is not None
        for idx, lg in enumerate(self._page_table.layer_groups):
            block_ids = self._get_block_ids(req, idx, lg)

            # Filter to only window-relevant blocks for sliding window layer groups.
            # Computes the expected number of non-stale blocks (using the same
            # eviction formula as update_resources) and keeps only the tail.
            # This works correctly regardless of whether update_resources has
            # been called:
            #   - Pre-eviction: all blocks present → trim to last N.
            #   - Post-eviction (V2 valid_only=True): stale blocks already
            #     removed → len == expected_valid, so the condition is false.
            window_size = lg.sliding_window_size
            if window_size is not None:
                total_blocks = (req.prompt_len + tpb - 1) // tpb
                stale_end = max(0, (req.prompt_len + 1 - window_size) // tpb)
                expected_valid = total_blocks - stale_end
                if expected_valid <= 0:
                    block_ids = []
                elif len(block_ids) > expected_valid:
                    block_ids = block_ids[-expected_valid:]

            groups.append(list(block_ids))

        return KVSlice(is_last_slice=True, block_ids_per_layer_groups=groups)

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

    def respond_and_send_async(self, req: LlmRequest):
        rid = get_unique_rid(req)
        assert rid is not None
        if rid not in self._send_sessions:
            self._send_sessions[rid] = self._transfer_worker.create_tx_session(req)
        session = self._send_sessions[rid]
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        session.send(self._create_kv_slice(req))
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

    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError("request_and_receive_sync is not implemented")

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

        completed, timed_out, failed = [], [], []
        for rid in to_process:
            session = self._send_sessions[rid]
            result = session.wait_complete()
            if result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} timed out after {self._sender_future_timeout_ms}ms"
                )
                timed_out.append(rid)
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                failed.append(rid)

        for rid in completed:
            if mark_complete:
                self._send_reqs[rid].state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
            self._send_sessions[rid].close()
            del self._send_reqs[rid]
            del self._send_sessions[rid]
        self._close_failed_sessions(self._send_sessions, self._send_reqs, failed)

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

        completed, failed = [], []
        for rid in to_process:
            result = self._recv_sessions[rid].wait_complete(blocking=block_all)
            if result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result == WaitResult.FAILED:
                failed.append(rid)
            # else: None — KV done but aux still in flight; re-poll next cycle

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

        return completed, failed

    def check_gen_transfer_complete(self):
        return len(self._recv_sessions) == 0

    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError("cancel_request is not implemented")

    def get_disaggregated_params(self) -> Dict[str, Any]:
        # Keep this aligned with fields populated in respond_and_send_async().
        # These values are server-level metadata used to seed generation-first
        # requests before context-phase response data arrives.
        return {
            "ctx_dp_rank": self._dp_rank,
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

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
