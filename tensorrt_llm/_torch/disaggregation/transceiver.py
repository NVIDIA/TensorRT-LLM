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
                # On/off switch via config (size 0 => None => per-block path).
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

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

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
            # Align with KV cache allocation (prepare_disagg_gen_init /
            # _get_context_bytes), which reserves prompt_len +
            # num_extra_kv_tokens slots for speculative decoding methods
            # (e.g. EAGLE3) that consume extra KV positions per request.
            num_extra_kv_tokens = getattr(self._kv_cache_manager, "num_extra_kv_tokens", 0) or 0
            token_range = TokenRange(start=0, end=req.prompt_len + num_extra_kv_tokens)

        groups = []
        for idx, lg in enumerate(layer_groups):
            if isinstance(lg, MambaLayerGroup):
                groups.append(np.array([], dtype=np.int64))
                continue
            block_ids = adapter.get_block_ids(req, idx, lg)
            # Limit to prompt_len blocks, matching C++ cacheFormatter behavior.
            # Extra blocks from num_extra_kv_tokens (speculative decoding) have
            # uninitialized KV data and must not be transferred.
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
            is_last_slice=is_last_slice,
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
        self._ever_had_send_session = True
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
            kv_slice = self._create_kv_slice(req)
            session.receive(kv_slice)
            result = session.wait_complete(blocking=True)

            if result == WaitResult.COMPLETED:
                # KV-transfer timing setters deferred to #15871 (clock-source consistency); size only.
                req.set_kv_cache_size(self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor)
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
            if session is not None:
                session.close()
            self._recv_sessions.pop(rid, None)
            self._recv_reqs.pop(rid, None)

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest):
        self._ever_had_recv_session = True
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_async: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        session = self._transfer_worker.create_rx_session(req)
        self._recv_sessions[rid] = session
        kv_slice = self._create_kv_slice(req)
        req.py_kv_cache_xfer_bytes = self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
        session.receive(kv_slice)
        self._recv_reqs[rid] = req

    def check_context_transfer_status(
        self, at_least_request_num: Optional[int], mark_complete: bool = False
    ):
        if not self._ever_had_send_session and not (
            self._ctx_need_tp_sync or self._ctx_need_pp_sync
        ):
            return [], []
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
            result = session.wait_complete(blocking=block_all)
            if session.status == SessionStatus.CANCELLED:
                cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result is None:
                continue
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
        if not self._ever_had_recv_session and not self._gen_need_sync:
            return [], [], []
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
            # transfer_end already stamped at completion detection above.
            req.set_kv_cache_size(getattr(req, "py_kv_cache_xfer_bytes", 0))
            if self._need_aux_transfer(req):
                self._apply_aux(session, req)
            self._assert_disagg_history_declared(req)
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            session.close()
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]
        if failed:
            logger.warning(
                f"Disagg gen transfer FAILED rank={self._dist.rank} "
                f"rids={failed} gen_need_sync={self._gen_need_sync}"
            )
        self._close_failed_sessions(self._recv_sessions, self._recv_reqs, failed)

        return completed, failed, cancelled_reqs

    def _poll_gen_sessions_for_poll_interval(self, wait_num: int) -> None:
        poll_interval_s = (self.kv_transfer_poll_interval_ms or 0) / 1000.0
        deadline = time.monotonic() + poll_interval_s
        while True:
            completed, failed = self._collect_done(self._recv_sessions, self._recv_reqs)
            if len(completed) + len(failed) >= wait_num:
                return
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            for session in self._recv_sessions.values():
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
