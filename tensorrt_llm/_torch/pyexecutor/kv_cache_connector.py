# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
This file contains the primary interface for the KV Cache Connector.

The KV Cache Connector is a component that allows for remote KV cache access.
It is responsible for:
- Orchestrating the loading and saving of KV cache blocks.
- Managing asynchronous block tx/rx.

It can be used to provide functionalities such as:
1. Disagg
2. KV offload/onboard
3. KV cache sharing
4. P2P KV cache transfer
etc.

The Connector API is split into two parts:
1. The scheduler, which is responsible for orchestration, and building metadata for the workers.
2. The worker, which performs and monitors transfers indicated by the scheduler's metadata.

To implement a custom KV connector, you need to implement both the scheduler and worker-side interfaces.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple)

import torch

from tensorrt_llm._utils import mpi_allgather, mpi_broadcast, mpi_rank
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnectorManager as KvCacheConnectorManagerCpp
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from .resource_manager import KVCacheManager


# Used to store data for a single inflight request.
@dataclass
class RequestData:
    # The request ID.
    request_id: int
    # The new tokens that were generated in the prior forward pass.
    new_tokens: List[int]
    # The new block IDs allocated in the prior forward pass.
    new_block_ids: List[int]
    # The position of the latest token with computed (valid) kv cache values.
    computed_position: int
    # The number of scheduled tokens for the upcoming forward pass.
    num_scheduled_tokens: int


# A class to store some basic data regarding all inflight requests.
# This is used when calling `build_connector_meta` on the scheduler.
@dataclass
class SchedulerOutput:
    # Requests being scheduled for the first time. Requests will show up in `new_request` exactly once.
    new_requests: List[RequestData] = field(default_factory=list)

    # Requests being scheduled, that have already shown up in `new_requests`.
    cached_requests: List[RequestData] = field(default_factory=list)


class KvCacheConnectorWorker(ABC):

    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        self._metadata = None
        super().__init__()

    def bind_connector_meta(self, metadata: object):
        self._metadata = metadata

    def get_connector_meta(self) -> object:
        return self._metadata

    def _clear_connector_meta(self):
        self._metadata = None

    def register_forward_pass_callable(self) -> Callable:
        """
        This callable will be called at the end of the forward pass.

        Any CUDA calls which happen in the callable will execute on the
        same stream as the forward pass.

        This method is typically used by the connector to insert a
        cuda event into the forward pass cuda stream to obtain a
        signal of when it's appropriate to start offloading cache blocks.
        """

    @abstractmethod
    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """
        Register the KV cache tensors to the worker.
        This can be used for something like NIXL registration.

        Args:
            kv_cache_tensor: The contiguous KV cache tensor.
        """

    @abstractmethod
    def start_load_kv(self, stream: torch.cuda.Stream):
        """
        Begin loading the KV cache in preparation for the next forward pass.
        Specific blocks to transfer are indicated by the scheduler's metadata.
        """

    @abstractmethod
    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Wait for a layer to finish being loaded before proceeding with the forward pass on the layer.
        Note: This function is called immediately before the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Begin saving the KV cache for a layer.
        Note: This function is called immediately after the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to save.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def wait_for_save(self, stream: torch.cuda.Stream):
        """
        Block until all synchronous saving operations are complete. Called at the end of the forward pass.
        """

    @abstractmethod
    def get_finished(
            self, finished_gen_req_ids: List[int],
            started_loading_req_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Get the requests that have finished loading and saving.

        Args:
            finished_gen_req_ids: The IDs of the requests that have finished generating tokens, and are now asynchronously saving.
            started_loading_req_ids: The IDs of the requests that have started asynchronously loading.

        Returns:
            The IDs of the requests that have finished saving.
            The IDs of the requests that have finished loading.

        Note: IDs may only be returned from this call after they've been provided in the `finished_gen_req_ids` and `started_loading_req_ids` arguments.
        Additionally, the runtime will only take action based on these returned IDs once they've been returned by ALL workers. This allows some workers to take longer than others to complete the operations.
        """


class KvCacheConnectorScheduler(ABC):

    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        super().__init__()

    @abstractmethod
    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """
        Build the metadata for the worker.
        This is called by the KV Cache Manager when adding a sequence.
        Args:
            scheduler_output: The data for all inflight requests.

        Returns:
            The metadata for the workers.
        """

    @abstractmethod
    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> Tuple[int, bool]:
        """
        Get the number of tokens that can be loaded from remote KV cache.
        This does not include the tokens already matched on device (indicated by `num_computed_tokens`).

        Args:
            request: The request to get the number of tokens for.
            num_computed_tokens: The number of tokens already matched on device.

        Returns:
            The number of tokens that can be loaded from remote KV cache.
            Whether the tokens will be loaded asynchronously.
        """

    @abstractmethod
    def request_finished(self, request: LlmRequest,
                         cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            request: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving operations.
            If true, this indicates that the kv cache manager should wait to deallocate the blocks until the saving has completed (determined by `get_finished` on the workers).
        """

    @abstractmethod
    def update_state_after_alloc(self, request: LlmRequest,
                                 block_ids: List[int]):
        """
        Called after get_num_new_matched_tokens is called to provide the block ids to the scheduler.

        Args:
            request: The request that was allocated resources.
            block_ids: The KV cacheblock IDs that were allocated.
        """


# An internal dataclass to handle async saving/loading requests.
@dataclass
class AsyncRequests:
    saving: Dict[int, LlmRequest]
    loading: Dict[int, LlmRequest]

    def add_from(self, other: 'AsyncRequests'):
        """
        Remove requests from the other `AsyncRequests` object, and add them to this one.
        """
        self.saving.update(other.saving)
        self.loading.update(other.loading)

        other.saving = dict()
        other.loading = dict()

    def extract_by_id(self, saving_ids: List[int],
                      loading_ids: List[int]) -> 'AsyncRequests':
        """
        Extract the requests with the given IDs from this `AsyncRequests` object.

        Args:
            saving_ids: The IDs of the requests to extract.
            loading_ids: The IDs of the requests to extract.
        """
        new_async_requests = AsyncRequests(dict(), dict())

        for req_id in saving_ids:
            new_async_requests.saving[req_id] = self.saving[req_id]
            del self.saving[req_id]
        for req_id in loading_ids:
            new_async_requests.loading[req_id] = self.loading[req_id]
            del self.loading[req_id]

        return new_async_requests

    @property
    def saving_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being saved asynchronously.
        """
        return set(self.saving.keys())

    @property
    def loading_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being loaded asynchronously.
        """
        return set(self.loading.keys())


class KvCacheConnectorSchedulerOutputRequest:

    def __init__(self):
        self.block_ids = []
        self.tokens = []

    def update_and_build_data(self, req: LlmRequest,
                              kv_cache_manager: "KVCacheManager"):
        block_ids = kv_cache_manager.get_cache_indices(req)
        tokens = req.get_tokens(0)

        new_block_ids = block_ids[len(self.block_ids):]
        new_tokens = tokens[len(self.tokens):]

        self.block_ids.extend(new_block_ids)
        self.tokens.extend(new_tokens)

        if req.state in (LlmRequestState.CONTEXT_INIT,
                         LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS):
            computed_position = req.context_current_position
            num_scheduled_tokens = min(req.context_remaining_length,
                                       req.context_chunk_size)
        else:
            computed_position = len(tokens) - 1
            num_scheduled_tokens = 1  # Specdec with draft tokens is not supported yet.

        return RequestData(req.request_id, new_tokens, new_block_ids,
                           computed_position, num_scheduled_tokens)


class KvCacheConnectorSchedulerOutputManager:

    def __init__(self):
        self.requests = defaultdict(KvCacheConnectorSchedulerOutputRequest)
        self.external_loads = dict()

    def build_scheduler_output(self, scheduled_batch: ScheduledRequests,
                               new_async_requests: AsyncRequests,
                               kv_cache_manager: "KVCacheManager"):
        scheduler_output = SchedulerOutput()

        for req in scheduled_batch.context_requests:
            if req.request_id in new_async_requests.loading_ids:
                continue

            is_new = req.request_id not in self.requests

            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager)

            # Don't include the connector matched tokens in the initial scheduler output.
            if req.request_id in self.external_loads:
                request_data.computed_position -= self.external_loads[
                    req.request_id]

            if is_new:
                scheduler_output.new_requests.append(request_data)
            else:
                scheduler_output.cached_requests.append(request_data)

        for req in scheduled_batch.generation_requests:
            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager)

            scheduler_output.cached_requests.append(request_data)

        self.external_loads = dict()

        return scheduler_output

    def record_new_matched_tokens(self, request: LlmRequest,
                                  num_new_matched_tokens: int):
        self.external_loads[request.request_id] = num_new_matched_tokens


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):
    """
    The KvCacheConnectorManager is used to manager connector-related state.

    It has the following responsibilities:
    1. Managing the state of async requests (both offload and onboard)
    2. Handling MPI communication. We only run the leader on one rank, but need the results of the leader API on all ranks.

    Note: This class is solely an implementation detail, and is not part of the connector interface itself.
    When implementing a connector API, you do not need to implement this class.
    """

    def __init__(self, worker: KvCacheConnectorWorker,
                 scheduler: Optional[KvCacheConnectorScheduler]):
        assert (scheduler is not None) == (
            mpi_rank() == 0), "The scheduler may only exist on rank 0!"

        super().__init__()

        self.worker = worker
        self.scheduler = scheduler

        # Requests that haven't yet been passed into get_finished.
        self.new_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been passed into get_finished, but haven't yet been returned.
        self.pending_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been returned from get_finished locally, but haven't yet been returned by all workers.
        self.local_finished_async_requests = AsyncRequests(dict(), dict())

        # Requests that have finished loading asynchronously.
        self.finished_async_loading_requests = dict()

        self._scheduler_output = None
        self.scheduler_output_manager = KvCacheConnectorSchedulerOutputManager()

    def _run_on_leader(self, f: Callable[[], Any]) -> Any:
        """
        Run a function on the leader rank, and broadcast the result to all other ranks.
        """
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            res = f()
        else:
            res = None
        return mpi_broadcast(res, root=0)

    def get_num_new_matched_tokens(self, request: LlmRequest,
                                   num_computed_tokens: int) -> int:
        if request.is_generation_only_request:
            raise RuntimeError(
                "Connector API is not supported for generation-only requests!")

        num_tokens, load_kv_async = self._run_on_leader(
            lambda: self.scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens))

        if num_tokens == 0 and load_kv_async:
            raise RuntimeError(
                "load_kv_async must be False when num_tokens is 0!")

        # TODO(jthomson04): This part is a bit ugly.
        # When the connector indicates that a request will be loaded asynchronously, we need to suspend it's execution.
        # This is problematic, since at the point when this function is called, the request has already been scheduled!
        # Because of this, we need to remove it from our list of scheduled requests (see `take_scheduled_requests_pending_load`).
        if load_kv_async:
            self.new_async_requests.loading[request.request_id] = request

        self.scheduler_output_manager.record_new_matched_tokens(
            request, num_tokens)

        return num_tokens

    def should_add_sequence(self, request: LlmRequest) -> bool:
        req_id = request.request_id
        return req_id not in self.finished_async_loading_requests

    def build_scheduler_output(self, scheduled_batch: ScheduledRequests,
                               kv_cache_manager: "KVCacheManager"):
        self._scheduler_output = self.scheduler_output_manager.build_scheduler_output(
            scheduled_batch, self.new_async_requests, kv_cache_manager)

    def take_scheduled_requests_pending_load(
            self, scheduled_requests: ScheduledRequests):
        """
        Remove context requests from our list of scheduled requests that are being loaded asynchronously.
        This is done to prevent the runtime from attempting to load the KV cache for these requests.

        Args:
            scheduled_requests: The scheduled requests.

        Returns:
            The scheduled requests with the context requests that are being loaded asynchronously removed.
        """
        allowed_context_requests = []

        for req in scheduled_requests.context_requests:
            # If this request is being loaded asynchronously, in addition to removing it from the list of scheduled requests,
            # we also need to update it's state.
            if req.request_id in self.new_async_requests.loading.keys():
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

                # Replace the request with the canonical request.
                self.new_async_requests.loading[req.request_id] = req
            else:
                allowed_context_requests.append(req)

        # Update the list of scheduled requests.
        scheduled_requests.context_requests = allowed_context_requests

    def handle_metadata(self) -> object:
        metadata = self._run_on_leader(
            lambda: self.scheduler.build_connector_meta(self._scheduler_output))

        self._scheduler_output = None

        self.worker.bind_connector_meta(metadata)

    def request_finished(self, req: LlmRequest,
                         cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            req: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving operations. If true, we do not immediately call free_resources on the request.
        """

        if req.request_id in self.finished_async_loading_requests:
            del self.finished_async_loading_requests[req.request_id]

        saving_async = self._run_on_leader(
            lambda: self.scheduler.request_finished(req, cache_block_ids))

        # This is similar to take_scheduled_requests_pending_load.
        # We need to update the request's state to indicate that it's still being used, but isn't schedulable.
        if saving_async:
            self.new_async_requests.saving[req.request_id] = req
            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        return saving_async

    def get_finished(self) -> List[LlmRequest]:
        """
        Process requests that have finished loading and saving.

        Returns:
            The requests that have newly finished saving.
        """
        started_loading_req_ids = list(self.new_async_requests.loading_ids)
        finished_gen_req_ids = list(self.new_async_requests.saving_ids)

        # Add the requests to our list of outstanding (still in progress) requests.
        self.pending_async_requests.add_from(self.new_async_requests)

        # Pass these newly finished requests into get_finished, and get the list of requests that have finished saving and loading.
        (finished_saving,
         finished_loading) = self.worker.get_finished(finished_gen_req_ids,
                                                      started_loading_req_ids)

        # Remove the requests from our pending list that have finished locally.
        new_local_finished_async_requests = self.pending_async_requests.extract_by_id(
            finished_saving, finished_loading)

        # Add these requests to our list of locally finished requests.
        self.local_finished_async_requests.add_from(
            new_local_finished_async_requests)

        # Broadcast this whole list to all other workers.
        finished_saving = list(self.local_finished_async_requests.saving_ids)
        finished_loading = list(self.local_finished_async_requests.loading_ids)

        all_results = mpi_allgather((finished_saving, finished_loading))

        # Find only the requests that have been reported complete by all workers.
        intersect_finished_saving = set.intersection(
            *[set(res[0]) for res in all_results])
        intersect_finished_loading = set.intersection(
            *[set(res[1]) for res in all_results])

        # Remove these requests from our list of locally finished requests.
        all_finished = self.local_finished_async_requests.extract_by_id(
            intersect_finished_saving, intersect_finished_loading)

        # For requests that have finished loading, move them back to the context state.
        for id, req in all_finished.loading.items():
            req.state = LlmRequestState.CONTEXT_INIT
            self.finished_async_loading_requests[id] = req

        # Return the requests that have finished saving.
        # The execution loop will call _terminate_request on these requests.
        return list(all_finished.saving.values())

    def update_state_after_alloc(self, req: LlmRequest, block_ids: List[int]):
        if self.scheduler is not None:
            self.scheduler.update_state_after_alloc(req, block_ids)

    def set_scheduler_output(self, scheduler_output: SchedulerOutput):
        self._scheduler_output = scheduler_output

    def layer_pre_hook(self, module, *args):
        self.worker.wait_for_layer_load(module.layer_idx,
                                        torch.cuda.current_stream())

    def layer_post_hook(self, module, *args):
        self.worker.save_kv_layer(module.layer_idx, torch.cuda.current_stream())
