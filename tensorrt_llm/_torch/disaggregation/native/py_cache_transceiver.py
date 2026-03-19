import concurrent
import uuid
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List

import torch

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, get_unique_rid
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
from tensorrt_llm._torch.disaggregation.resource.utils import get_global_layer_ids
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


def _find_consensus_request_ids(request_ids_all_ranks, sync_size):
    frequency_map = defaultdict(int)
    consensus_request_ids = []
    for request_id in list(chain.from_iterable(request_ids_all_ranks)):
        frequency_map[request_id] += 1
    sorted_frequency_map = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
    for request_id, frequency in sorted_frequency_map:
        if frequency == sync_size:
            consensus_request_ids.append(request_id)
        else:
            break
    return consensus_request_ids


class PyNativeCacheTransceiver(KvCacheTransceiver):
    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self.dist: Distributed = dist
        self.kv_cache_manager = kv_cache_manager
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.mapping = mapping
        self._check_compatible()
        self.sender_future_timeout_ms = (
            cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        )
        instance_name = None
        if dist.rank == 0:
            instance_name = str(uuid.uuid4())
            dist.broadcast(instance_name, 0)
        else:
            instance_name = dist.broadcast(instance_name, 0)

        self.instance_name = instance_name

        # device_id = mapping.local_rank
        self.device_id = torch.cuda.current_device()
        logger.info(f"device_id: {self.device_id} in PyNativeCacheTransceiver")

        # Aux payload carries first-gen and draft tokens in generation-first flow.
        self.aux_buffer = AuxBuffer(
            # * 2 to allow back-to-back batches, one in transferring, one in preparing next batch
            max_slot_num=max(1, int(self.kv_cache_manager.max_batch_size)) * 2,
            beam_width=max(1, int(getattr(self.kv_cache_manager, "max_beam_width", 1))),
            max_draft_len=max(0, int(getattr(self.kv_cache_manager, "max_draft_len", 0))),
            device="cpu",
        )

        self.transfer_worker = TransferWorker(
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            device_id=self.device_id,
            instance_name=instance_name,
            aux_buffer=self.aux_buffer,
        )

        self.context_info_endpoint = None
        self.dp_rank = self.mapping.tp_rank if self.mapping.enable_attention_dp else 0
        if self.dist.rank == 0:
            self.context_info_endpoint = self.transfer_worker._rank_info_server.endpoint
            self.dist.broadcast(self.context_info_endpoint, 0)
        else:
            self.context_info_endpoint = self.dist.broadcast(self.context_info_endpoint, 0)

        self.mapping = mapping

        self.ctx_need_tp_sync = mapping.tp_size > 1 and (not mapping.enable_attention_dp)

        self.gen_need_sync = not (
            mapping.world_size == 1 or (mapping.enable_attention_dp and mapping.pp_size == 1)
        )
        self.gen_sync_allgather_fun = (
            self.dist.pp_allgather if mapping.enable_attention_dp else self.dist.allgather
        )
        ctx_server_endpoint = self.transfer_worker._sender.endpoint
        layer_num = len(self.kv_cache_manager.pp_layers)

        ctx_server_endpoints = self.dist.allgather(ctx_server_endpoint)
        layer_num_per_pp = self.dist.pp_allgather(layer_num)
        self.transfer_worker.populate_instance_and_rank_info(
            endpoints=ctx_server_endpoints, layer_num_per_pp=layer_num_per_pp
        )

        logger.info(f"transfer worker  ctx_server_endpoints: {ctx_server_endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self.context_info_endpoint: {self.context_info_endpoint}")
        self.send_sessions = {}  # request_id to send_session
        self.send_task_ids = {}  # request_id to send_task_id
        self.recv_sessions = {}  # request_id to recv_session
        self.recv_task_ids = {}  # request_id to recv_task_id
        self.send_req_id_to_request = {}  # request_id to request (for send)
        self.recv_req_id_to_request = {}  # request_id to request (for recv)
        self.wait_req_id_to_request = {}  # request_id to request (for gen-first waiting-scheduler)
        self.page_table = self.transfer_worker._rank_info.page_table
        # Check if using V2 manager (has kv_cache_map attribute)
        self.is_v2_manager = hasattr(self.kv_cache_manager, "kv_cache_map")

    def shutdown(self):
        if self.transfer_worker is not None:
            self.transfer_worker.shutdown()

    def _create_kv_slice(self, req: LlmRequest):
        # Get block_ids for each layer group
        block_ids_per_layer_groups: List[List[int]] = []
        tokens_per_block = self.kv_cache_manager.tokens_per_block

        for group_idx, lg in enumerate(self.page_table.layer_groups):
            if self.is_v2_manager:
                # V2: Use get_aggregated_page_indices for efficient slot indices
                group_id = group_idx
                block_ids = list(
                    self.kv_cache_manager.kv_cache_map[
                        req.py_request_id
                    ].get_aggregated_page_indices(group_id, valid_only=True)
                )
            else:
                # V1: Use get_batch_cache_indices
                first_global_layer_id = get_global_layer_ids(lg)[0]
                block_ids = self.kv_cache_manager.get_batch_cache_indices(
                    [req.py_request_id], layer_idx=first_global_layer_id
                )[0]

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
                total_blocks = (req.prompt_len + tokens_per_block - 1) // tokens_per_block
                stale_end = max(0, (req.prompt_len + 1 - window_size) // tokens_per_block)
                expected_valid = total_blocks - stale_end
                if expected_valid <= 0:
                    block_ids = []
                elif len(block_ids) > expected_valid:
                    block_ids = block_ids[-expected_valid:]

            block_ids_per_layer_groups.append(list(block_ids))

        return KVSlice(is_last_slice=True, block_ids_per_layer_groups=block_ids_per_layer_groups)

    @staticmethod
    def _need_aux_transfer(req: LlmRequest) -> bool:
        params = req.py_disaggregated_params
        return params is not None and params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST

    # starts background transfer to send this request's KV cache to the gen server, attaches ContextPhaseParams metadata
    def respond_and_send_async(self, req: LlmRequest):
        unique_rid = get_unique_rid(req)
        if unique_rid not in self.send_sessions:
            send_session = self.transfer_worker.create_tx_session(req)
            self.send_sessions[unique_rid] = send_session
        else:
            send_session = self.send_sessions[unique_rid]
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        # stores block ids, not raw data
        kv_slice = self._create_kv_slice(req)
        # sending actual kv data
        send_task_id = send_session.send(kv_slice)
        if self._need_aux_transfer(req):
            send_session.send_aux()
        self.send_task_ids[unique_rid] = send_task_id

        # contains metadata about itself so the gen server can see
        req.context_phase_params = ContextPhaseParams(
            first_gen_tokens=[],
            req_id=unique_rid,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=self.dp_rank,
            disagg_info_endpoint=self.context_info_endpoint,
        )
        self.send_req_id_to_request[unique_rid] = req

        return

    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError("request_and_receive_sync is not implemented")

    # starts background listener to receive KV cache from the ctx server into this request's pre-allocated blocks.
    def request_and_receive_async(self, req: LlmRequest):
        unique_rid = get_unique_rid(req)
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

        # create rx session for receiving blocks
        recv_session = self.transfer_worker.create_rx_session(req)
        self.recv_sessions[unique_rid] = recv_session
        kv_slice = self._create_kv_slice(req)
        recv_task_id = recv_session.receive(kv_slice)
        self.recv_task_ids[unique_rid] = recv_task_id
        self.recv_req_id_to_request[unique_rid] = req

    def check_context_transfer_status(self, at_least_request_num: int, mark_complete: bool = False):
        block_all = at_least_request_num is None

        wait_num = at_least_request_num if not block_all else 0

        local_completed_request_ids = []
        local_failed_request_ids = []
        for request_id, session in self.send_sessions.items():
            req = self.send_req_id_to_request[request_id]
            need_aux = self._need_aux_transfer(req)
            session_status = session.state.status
            if need_aux:
                if session_status == SessionStatus.AUX_TRANSFERRED:
                    local_completed_request_ids.append(request_id)
                elif session_status == SessionStatus.ERROR:
                    local_failed_request_ids.append(request_id)
            elif session_status == SessionStatus.TRANSFERRED:
                local_completed_request_ids.append(request_id)
            elif session_status == SessionStatus.ERROR:
                local_failed_request_ids.append(request_id)
        local_sync_request_ids = local_completed_request_ids + local_failed_request_ids

        if self.ctx_need_tp_sync:
            sync_request_ids_all_ranks = self.dist.tp_allgather(local_sync_request_ids)
        else:
            sync_request_ids_all_ranks = [local_sync_request_ids]

        sync_size = self.dist.tp_size if self.ctx_need_tp_sync else 1

        to_complete_request_ids = _find_consensus_request_ids(sync_request_ids_all_ranks, sync_size)
        for request_id in self.send_req_id_to_request.keys():
            if len(to_complete_request_ids) >= wait_num:
                break
            if request_id not in to_complete_request_ids:
                to_complete_request_ids.append(request_id)
        if block_all:
            to_complete_request_ids = self.send_req_id_to_request.keys()
        completed_request_ids = []
        timeout_request_ids = []
        failed_request_ids = []
        for request_id in to_complete_request_ids:
            session = self.send_sessions[request_id]
            try:
                if session.wait_complete(
                    self.send_task_ids[request_id],
                    wait_aux=True,
                    timeout_ms=self.sender_future_timeout_ms,
                ):
                    completed_request_ids.append(request_id)
            except concurrent.futures.TimeoutError:
                logger.warning(f"TxSession {session.unique_rid} timed out waiting for completion")
                timeout_request_ids.append(request_id)
            except Exception:
                logger.warning(f"TxSession {session.unique_rid} failed to complete")
                failed_request_ids.append(request_id)

        for request_id in completed_request_ids + failed_request_ids:
            if request_id in completed_request_ids:
                if mark_complete:
                    self.send_req_id_to_request[
                        request_id
                    ].state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
            elif request_id in failed_request_ids:
                self.send_req_id_to_request[request_id].state = LlmRequestState.DISAGG_TRANS_ERROR
            del self.send_req_id_to_request[request_id]
            self.transfer_worker.clear_session(self.send_sessions[request_id])
            del self.send_sessions[request_id]
            del self.send_task_ids[request_id]

        return completed_request_ids, failed_request_ids

    def check_gen_transfer_status(self, at_least_request_num: int):
        block_all = at_least_request_num is None

        wait_num = at_least_request_num if not block_all else 0

        local_completed_request_ids = []
        local_failed_request_ids = []
        for request_id, session in self.recv_sessions.items():
            req = self.recv_req_id_to_request[request_id]
            need_aux = self._need_aux_transfer(req)
            session_status = session.state.status
            if need_aux:
                if session_status == SessionStatus.AUX_TRANSFERRED:
                    local_completed_request_ids.append(request_id)
                elif session_status == SessionStatus.ERROR:
                    local_failed_request_ids.append(request_id)
            elif session_status == SessionStatus.TRANSFERRED:
                local_completed_request_ids.append(request_id)
            elif session_status == SessionStatus.ERROR:
                local_failed_request_ids.append(request_id)
        local_sync_request_ids = local_completed_request_ids + local_failed_request_ids

        if self.gen_need_sync:
            sync_request_ids = self.gen_sync_allgather_fun(local_sync_request_ids)
        else:
            sync_request_ids = [local_sync_request_ids]

        frequency_map = {}
        for request_id in list(chain.from_iterable(sync_request_ids)):
            frequency_map[request_id] = frequency_map.get(request_id, 0) + 1
        sorted_frequency_map = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        sync_size = (
            self.mapping.pp_size if self.mapping.enable_attention_dp else self.mapping.world_size
        )
        to_complete_request_ids = []
        for request_id, frequency in sorted_frequency_map:
            if frequency == sync_size:
                to_complete_request_ids.append(request_id)
            else:
                break
        for request_id in self.recv_sessions.keys():
            if len(to_complete_request_ids) >= wait_num:
                break
            if request_id not in to_complete_request_ids:
                to_complete_request_ids.append(request_id)
        if block_all:
            to_complete_request_ids = list(self.recv_sessions.keys())
        completed_request_ids = []
        failed_request_ids = []
        for request_id in to_complete_request_ids:
            recv_task_id = self.recv_task_ids[request_id]
            recv_session = self.recv_sessions[request_id]
            req = self.recv_req_id_to_request[request_id]
            if recv_session.wait_complete(recv_task_id, wait_aux=self._need_aux_transfer(req)):
                completed_request_ids.append(request_id)
            else:
                failed_request_ids.append(request_id)

        for request_id in completed_request_ids + failed_request_ids:
            if request_id in completed_request_ids:
                self.recv_req_id_to_request[
                    request_id
                ].state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            elif request_id in failed_request_ids:
                self.recv_req_id_to_request[request_id].state = LlmRequestState.DISAGG_TRANS_ERROR
            del self.recv_req_id_to_request[request_id]
            self.transfer_worker.clear_session(self.recv_sessions[request_id])
            del self.recv_sessions[request_id]
            del self.recv_task_ids[request_id]

        return

    def check_gen_transfer_complete(self):
        return len(self.recv_sessions) == 0

    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError("cancel_request is not implemented")

    def get_disaggregated_params(self) -> Dict[str, Any]:
        # Keep this aligned with fields populated in respond_and_send_async().
        # These values are server-level metadata used to seed generation-first
        # requests before context-phase response data arrives.
        return {
            "ctx_dp_rank": self.dp_rank,
            "ctx_info_endpoint": self.context_info_endpoint,
        }

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # Place new generation-first context requests into wait state, then
        # use tp_allgather consensus to promote ready requests to CONTEXT_INIT.
        for req in requests:
            unique_rid = get_unique_rid(req)
            if unique_rid not in self.send_sessions:
                self.wait_req_id_to_request[unique_rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

        # Check which waiting requests have peer info locally, then use
        # tp_allgather consensus so all TP ranks agree before promoting.
        # Without consensus, background peer info arriving at different
        # times on different ranks causes scheduling mismatches → hang.
        # Place tp sync here because this function runs in every iteration
        # but check_context_transfer_status runs when can_queue is True
        local_ready_request_ids = []
        for request_id in self.wait_req_id_to_request.keys():
            if self.transfer_worker.has_all_peer_req_infos_for_send(request_id):
                local_ready_request_ids.append(request_id)

        if self.ctx_need_tp_sync:
            ready_request_ids_all_ranks = self.dist.tp_allgather(local_ready_request_ids)
        else:
            ready_request_ids_all_ranks = [local_ready_request_ids]

        sync_size = self.dist.tp_size if self.ctx_need_tp_sync else 1
        ready_request_ids = _find_consensus_request_ids(ready_request_ids_all_ranks, sync_size)

        for request_id in ready_request_ids:
            self.wait_req_id_to_request[request_id].state = LlmRequestState.CONTEXT_INIT
            del self.wait_req_id_to_request[request_id]

    def _check_compatible(self):
        if self.mapping.cp_size != 1:
            raise ValueError(
                f"PyNativeCacheTransceiver: _check_compatible: only support context parallelism is 1: "
                f"cp_size: {self.mapping.cp_size}"
            )
        return

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
