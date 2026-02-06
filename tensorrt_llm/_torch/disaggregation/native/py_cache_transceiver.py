import concurrent
import uuid
from itertools import chain
from typing import Any, Dict, List

import torch

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


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

        self.transfer_worker = TransferWorker(
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            device_id=self.device_id,
            instance_name=instance_name,
        )

        self.context_info_endpoint = None
        self.dp_rank = self.mapping.tp_rank if self.mapping.enable_attention_dp else 0
        if self.dist.rank == 0:
            self.context_info_endpoint = self.transfer_worker._instance_info_server.endpoint
            self.dist.broadcast(self.context_info_endpoint, 0)
        else:
            self.context_info_endpoint = self.dist.broadcast(self.context_info_endpoint, 0)

        self.mapping = mapping

        self.ctx_need_tp_sync = mapping.tp_size > 1 and (not mapping.enable_attention_dp)

        self.gen_need_sync = not (
            mapping.world_size == 1 or (mapping.enable_attention_dp and mapping.pp_size > 1)
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

        logger.info(f" transfer worker  ctx_server_endpoints: {ctx_server_endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self.context_info_endpoint: {self.context_info_endpoint}")
        self.send_sessions = {}  # request_id to send_session
        self.send_task_ids = {}  # request_id to send_task_id
        self.recv_sessions = {}  # request_id to recv_session
        self.recv_task_ids = {}  # request_id to recv_task_id
        self.send_req_id_to_request = {}  # request_id to request (for send)
        self.recv_req_id_to_request = {}  # request_id to request (for recv)
        self.kv_pool_attrs = self.transfer_worker._rank_info.kv_pool_attrs
        # Check if using V2 manager (has kv_cache_map attribute)
        self.is_v2_manager = hasattr(self.kv_cache_manager, "kv_cache_map")

    def _create_kv_slice(self, req: LlmRequest):
        # Get block_ids for each layer group
        block_ids_per_layer_groups: List[List[int]] = []
        tokens_per_block = self.kv_cache_manager.tokens_per_block

        for group_attrs in self.kv_pool_attrs.layer_group_attrs_list:
            if self.is_v2_manager:
                # V2: Use get_aggregated_page_indices for efficient slot indices
                block_ids = list(
                    self.kv_cache_manager.kv_cache_map[
                        req.py_request_id
                    ].get_aggregated_page_indices(group_attrs.group_id, valid_only=True)
                )
            else:
                # V1: Use get_batch_cache_indices
                first_global_layer_id = group_attrs.global_layer_ids[0]
                block_ids = self.kv_cache_manager.get_batch_cache_indices(
                    [req.py_request_id], layer_id=first_global_layer_id
                )[0]
                # Filter by window_size if request_len > window_size
                window_size = group_attrs.sliding_window_size
                if window_size is not None:
                    max_blocks_in_window = window_size // tokens_per_block + 1
                    if len(block_ids) > max_blocks_in_window:
                        block_ids = block_ids[-max_blocks_in_window:]

            block_ids_per_layer_groups.append(list(block_ids))

        return KVSlice(is_last_slice=True, block_ids_per_layer_groups=block_ids_per_layer_groups)

    def respond_and_send_async(self, req: LlmRequest):
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        send_session = self.transfer_worker.create_tx_session(req)
        self.send_sessions[req.request_id] = send_session
        kv_slice = self._create_kv_slice(req)
        send_task_id = send_session.send(kv_slice)
        self.send_task_ids[req.request_id] = send_task_id

        req.context_phase_params = ContextPhaseParams(
            first_gen_tokens=[],
            req_id=req.request_id,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=self.dp_rank,
            disagg_info_endpoint=self.context_info_endpoint,
        )
        self.send_req_id_to_request[req.request_id] = req

        return

    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError("request_and_receive_sync is not implemented")

    def request_and_receive_async(self, req: LlmRequest):
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        recv_session = self.transfer_worker.create_rx_session(req)
        self.recv_sessions[req.request_id] = recv_session
        kv_slice = self._create_kv_slice(req)
        recv_task_id = recv_session.receive(kv_slice)
        self.recv_task_ids[req.request_id] = recv_task_id
        self.recv_req_id_to_request[req.request_id] = req
        return

    def check_context_transfer_status(self, at_least_request_num: int, mark_complete: bool = False):
        block_all = at_least_request_num is None

        wait_num = at_least_request_num if not block_all else 0

        local_completed_request_ids = []
        local_failed_request_ids = []
        for request_id, session in self.send_sessions.items():
            if session.state.status == SessionStatus.TRANSFERRED:
                local_completed_request_ids.append(request_id)
            elif session.state.status == SessionStatus.ERROR:
                local_failed_request_ids.append(request_id)
        local_sync_request_ids = local_completed_request_ids + local_failed_request_ids
        if self.ctx_need_tp_sync:
            sync_request_ids = self.dist.tp_allgather(local_sync_request_ids)
        else:
            sync_request_ids = [local_sync_request_ids]

        frequency_map = {}
        for request_id in list(chain.from_iterable(sync_request_ids)):
            frequency_map[request_id] = frequency_map.get(request_id, 0) + 1
        sorted_frequency_map = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        sync_size = self.dist.tp_size if self.ctx_need_tp_sync else 1
        to_complete_request_ids = []
        for request_id, frequency in sorted_frequency_map:
            if frequency == sync_size:
                to_complete_request_ids.append(request_id)
            else:
                break
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
            future = self.send_sessions[request_id]._kv_tasks[self.send_task_ids[request_id]].future
            try:
                sync_status = future.result(timeout=self.sender_future_timeout_ms / 1000.0)
                if sync_status == "SUCCESS":
                    completed_request_ids.append(request_id)
                else:
                    failed_request_ids.append(request_id)
            except concurrent.futures.TimeoutError:
                timeout_request_ids.append(request_id)
                logger.warning(
                    f"Request {request_id} timed out waiting for context KV cache transfer after",
                    f"{self.sender_future_timeout_ms} milliseconds.",
                )
            except Exception:
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
            if session.state.status == SessionStatus.TRANSFERRED:
                local_completed_request_ids.append(request_id)
            elif session.state.status == SessionStatus.ERROR:
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
            future = self.recv_sessions[request_id]._kv_tasks[self.recv_task_ids[request_id]].future
            try:
                sync_status = future.result()
                if sync_status == "SUCCESS":
                    completed_request_ids.append(request_id)
                else:
                    failed_request_ids.append(request_id)
            except Exception:
                failed_request_ids.append(request_id)
        for request_id in completed_request_ids + failed_request_ids:
            future = self.recv_sessions[request_id]._kv_tasks[self.recv_task_ids[request_id]].future
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
        # self.transfer_worker.cancel_request(req)

    def get_disaggregated_params(self) -> Dict[str, Any]:
        raise NotImplementedError("get_disaggregated_params is not implemented")

    def prepare_context_requests(self, requests: List[LlmRequest]):
        raise NotImplementedError("prepare_context_requests is not implemented")

    def _check_compatible(self):
        if self.mapping.cp_size != 1:
            raise ValueError(
                f"PyNativeCacheTransceiver: _check_compatible: only support context parallelism is 1: "
                f"cp_size: {self.mapping.cp_size}"
            )

        # if self.kv_cache_manager.is_vswa:
        #     raise ValueError("PyNativeCacheTransceiver: _check_compatible: VSWA is not supported")
        return

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
