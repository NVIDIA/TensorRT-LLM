import datetime
import logging
from typing import Dict, List, Optional
import traceback
import torch
import asyncio
from tensorrt_llm._utils import nvtx_range
from ...distributed import Distributed, MPIDist

from ..py_executor import PyExecutor, _get_from_request_queue
from ..model_engine import ModelEngine, PyTorchModelEngine

from tensorrt_llm.inputs import create_input_processor
import queue
import threading
from tensorrt_llm.executor.multimodal import MultimodalRequest, MultimodalResponse
from ..llm_request import ExecutorResponse
from tensorrt_llm._torch.multimodal import SharedTensorContainer
from torchvision.transforms import ToTensor
from ..py_executor import RequestQueueItem
logger = logging.getLogger(__name__)

class MMExecutor(PyExecutor):

    def __init__(self,
                 resource_manager,
                 scheduler,
                 model_engine: ModelEngine,
                 dist: Distributed,
                 enable_overlap_scheduler: bool = False,
                 max_num_active_requests: int = 10,
                 max_batch_size: int = 8,
                 start_worker: bool = True):

        self.device_id = torch.cuda.current_device()
        self.global_rank = dist.rank
        self.request_queue = queue.Queue()
        self.next_req_id = 0

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.dist = dist
        self.enable_overlap_scheduler = enable_overlap_scheduler

        # enqueue and _fetch_new_requests used data
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.shutdown_event = threading.Event()

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.canceled_req_ids = set()
        # _executor_loop private data
        self.max_num_active_requests = max_num_active_requests # TODO: remove this should be the same as max_batch_size
        self.active_requests = []
        self.is_shutdown = False
        self.event_loop = self._executor_loop
        self.max_batch_size = max_batch_size
        print(f"max_batch_size: {self.max_batch_size}")
        print(f"max_num_active_requests: {self.max_num_active_requests}")

        # Start worker if needed
        self.worker_started = False
        self.worker_lock = threading.Lock()
        if start_worker:
            self.start_worker()

    def start_worker(self):
        self.worker_lock.acquire()
        try:
            if self.worker_started == False:
                self.worker_thread = threading.Thread(target=self.event_loop,
                                                      daemon=True)
                self.worker_thread.start()
                self.worker_started = True
        finally:
            self.worker_lock.release()

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(self):
        total_num_active_requests = len(self.active_requests)
        total_max_num_active_requests = self.max_num_active_requests

        timeout = None if total_num_active_requests == 0 else datetime.timedelta(
            0)
        new_requests = []
        if self.dist.rank == 0:
            new_requests = _get_from_request_queue(
                self.request_queue, timeout,
                total_max_num_active_requests - total_num_active_requests)

        if self.dist.world_size > 1:
            new_requests = self.dist.broadcast(new_requests, root=0)
        return new_requests

    def _merge_tp_requests(self, new_requests: List[RequestQueueItem]):
        for request in new_requests:
            if request is None:
                return True
        for req_item in new_requests:
            self.active_requests.append(req_item.request)  # type: ignore
        return False

    def _executor_loop(self):
        """
        Simplified version of the executor loop that handles multimodal requests.
        Focuses only on basic functionality without complex features.
        """
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        iter_count = 0
        while not got_finish_signal or len(self.active_requests) > 0:
            # Get new requests
            new_requests = self._fetch_new_requests()
            # TODO: support DP across all requests in the batch
            got_finish_signal = (new_requests is not None and self._merge_tp_requests(new_requests)) or got_finish_signal

            # Exit if no more work to do
            if got_finish_signal and len(self.active_requests) == 0:
                break

            # Schedule requests
            scheduled_batch, batch_size = self._schedule()

            assert batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource.")
            print(f"in executor loop, iter_count: {iter_count}, batch_size: {batch_size}, active_requests: {len(self.active_requests)}")

            self.num_scheduled_requests = batch_size
            logger.debug(
                f'has {len(self.active_requests)} active_request, '
                f'scheduled {len(scheduled_batch)} requests'
            )

            finished_requests = []
            # Process batch
            if batch_size > 0:
                # TODO: add resource manager for multimodal executor
                # self.resource_manager.prepare_resources(scheduled_batch) # only sequency manager?

                batch_outputs = self._forward(scheduled_batch)

                # TODO: Handle canceled requests for multimodal executor
                self._handle_cancelled_requests()
                finished_requests = self._handle_responses(scheduled_batch, batch_outputs)

                # Free resources
                #self.resource_manager.update_resources(scheduled_batch)

            # TODO: add iter perf stats for multimodal executor
            iter_count += 1
            # if self.enable_iter_perf_stats:

        # Cleanup when loop is done
        self._executor_loop_cleanup()

    def _schedule(self):
        # This is a simple static scheduler that only considers active requests, and max batch size
        num_to_schedule = min(len(self.active_requests), self.max_batch_size)
        if num_to_schedule == 0:
            return []

        scheduled_requests = self.active_requests[:num_to_schedule]
        batch_size = len(scheduled_requests)
        return scheduled_requests, batch_size

    @nvtx_range("_forward")
    def _forward(self,
                      scheduled_requests):
        @nvtx_range(
            f"[Executor] _forward_step: {len(scheduled_requests)} mm reqs"
        )
        def forward(scheduled_requests, resource_manager):
            return self.model_engine.forward(scheduled_requests,
                                             resource_manager)

        try:
            outputs = forward(scheduled_requests, self.resource_manager)
            return outputs
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in forward function: {error_msg}")
            self._handle_errors(error_msg)
            return None

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Dict[int, ExecutorResponse]):
        logger.debug(
            f'before enqueue, rank = {self.dist.rank}, responses = {responses}')
        if self.dist.rank == 0:
            with self.response_cv:
                for req_id, resp in responses.items():
                    if isinstance(resp, MultimodalResponse):
                        if resp.cp_event is not None:
                            # if we need to cpy embedding to host, we can sync here
                            resp.cp_event.synchronize()
                        # We only store/enqueue the handle here
                        resp.embedding_handle = [SharedTensorContainer.from_tensor(resp.embeddings)]
                        resp.embeddings = None
                        resp.cp_event = None

                    if req_id in self.responses.keys():
                        self.responses[req_id].append(resp)
                    else:
                        self.responses.update({req_id: [resp]})
                self.response_cv.notify_all()

    @nvtx_range("_handle_responses")
    def _handle_responses(self, scheduled_requests, batch_outputs):
        """Handle responses from postprocess_batch_outputs using existing infrastructure."""
        new_responses = {}
        new_active_requests = []
        if batch_outputs is None:
            for request in scheduled_requests:
                if request.has_error() or request not in self.active_requests:
                    continue
                response = request.create_response()
                if response:
                    response.set_final()
                    new_responses[request.id] = response

            self._enqueue_responses(new_responses)
            self.active_requests = [req for req in self.active_requests if req.id not in new_responses]
            return scheduled_requests

        mm_embeddings = batch_outputs['mm_embeddings']
        mrope_config = batch_outputs['mrope_config']
        batch_request_offsets = batch_outputs['batch_request_offsets']

        # Process each request's portion of the fused embeddings
        for i, request in enumerate(scheduled_requests):
            assert isinstance(request, MultimodalRequest), "request should be a MultimodalRequest"
            if request.has_error() or request not in self.active_requests:
                continue
            start_idx = batch_request_offsets[i]
            end_idx = batch_request_offsets[i + 1]

            # Create response for this request
            response = request.create_response()
            if response:
                # Extract this request's portion of embeddings
                request_embedding = mm_embeddings[start_idx:end_idx]

                # Attach the fused embedding directly to the response
                response.set_embeddings(request_embedding, cp_event=None)

                # Attach mrope config if available
                if mrope_config is not None:
                    response.set_mrope_config(mrope_config)

                response.set_final()
                new_responses.update({request.id: response})
                # self._terminate_request(request) # TODO: add resource manager for multimodal executor

        self._enqueue_responses(new_responses)
        self.active_requests = [req for req in self.active_requests if req.id not in new_responses]
        return scheduled_requests # finished requests

    def _handle_cancelled_requests(self):
        #TODO: properly handle canceled ids in pp case
        if self.dist.has_tp:
            self.canceled_req_ids = self.dist.broadcast(self.canceled_req_ids,
                                                        root=0)

        if len(self.canceled_req_ids) == 0:
            return

        cancelled_responses = {}
        left_requests = []
        for request in self.active_requests:
            req_id = request.id
            if req_id in self.canceled_req_ids:
                # TODO: As for now, all resources are on-the-fly, so we don't need to free resources here
                # but in future, when we add embedding tensor pool, we need to evict and free resources here
                # self._terminate_request(request)
                cancelled_responses[req_id] = request.create_response()
                self.canceled_req_ids.remove(req_id)
            else:
                left_requests.append(request)
        self.active_requests = left_requests

        # When enable attention dp, each rank does not have full copy of requests
        # so we need to remove the cancel requests not in the local rank
        self.canceled_req_ids.clear()

        # enqueue the cancelled requests' responses as they are not
        # active_requests and be discarded in the sampler loop.
        self._enqueue_responses(cancelled_responses)

    def shutdown(self):
        """
        Signals the server to shutdown.
        """
        try:
            self.enqueue_lock.acquire()
            self.request_queue.put(None)
            self.active = False
        finally:
            self.enqueue_lock.release()
        self.shutdown_event.wait()
        self.worker_thread.join()
        self.worker_started = False
        del self.model_engine

    def enqueue_request(self,
                        request: MultimodalRequest,
                        query: Optional[List] = None):
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            req_id = self.next_req_id
            self.request_queue.put(RequestQueueItem(req_id, request))
            self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
        return req_id

    def enqueue_requests(self, requests: List[MultimodalRequest]):
        """
        Enqueue new requests
        """
        req_ids = []
        try:
            self.enqueue_lock.acquire()
            assert self.active, "MMPyExecutor has already been shutdown."
            for request in requests:
                self.request_queue.put(
                    RequestQueueItem(self.next_req_id, request))
                req_ids.append(self.next_req_id)
                self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
        return req_ids


    def _handle_errors(self, error_msg: Optional[str] = None):
        error_responses = {}
        error_msg = error_msg or "error"
        for request in self.active_requests:
            req_id = request.id
            # use the same error response as the llm executor
            error_responses[req_id] = ExecutorResponse(
                req_id, error_msg, client_id=request.id)
        self.active_requests.clear()
        self._enqueue_responses(error_responses)

    def cancel_request(self, id: int):
        """
        Cancel the request with provided request id
        Args:
            id (int): The request id for which to cancel the response
        """
        self.canceled_req_ids.add(id)

    def get_latest_kv_cache_events(self):
        return []

    def get_latest_iteration_stats(self):
        return []


class MultimodalModelEngine(PyTorchModelEngine):
    def __init__(
        self,
        model_path: str,
        pytorch_backend_config = None,
        max_batch_size: Optional[int] = 8,
        dist: Optional[MPIDist] = None,
    ):
        self.pytorch_backend_config = pytorch_backend_config
        self.dist = dist
        self.max_batch_size = max_batch_size
        self.model = create_input_processor(model_path, None)

    async def _prepare_inputs(self, scheduled_requests):
        """Prepare inputs for batch processing.

        Args:
            scheduled_requests: List[MultimodalRequest]

        Returns:
            Tuple[Dict[str, List[MultimodalItem]], List[int]]:
                - Dict mapping modality to ordered list of items
                - List of offsets for each request (based on token lengths)
        """
        # 1. prefetch all the contents
        all_mm_items = []
        for request in scheduled_requests:
            all_mm_items.extend(request.items)
        # 2. Process items and track their completion
        processed_items = {}  # Dict to track processed items by (req_id, item_id)

        # Process items asynchronously
        for ready_item in all_mm_items:
            # Calculate token length and preprocess
            # TODO: Need to converge for all models on this, currently we don't have an uniform way to get the token length
            # https://github.com/vllm-project/vllm/blob/54631f826233dbd1c046f9a70e98bc2e25edff1a/vllm/model_executor/models/llava.py#L151
            #ready_item.length = self.model.get_num_image_tokens(image_width=ready_item.data.width, image_height=ready_item.data.height)
            # TODO: VLLM output lenght is not correct, need to fix it
            image_size = (ready_item.data.height, ready_item.data.width)
            ready_item.length = self.model.image_size_to_num_tokens(image_size)
            # If not converted to tensor, the results will be off
            ready_item.data =  ToTensor()(ready_item.data)
            ready_item.data = self.model._preprocess([ready_item.data])[0] # _preprocess involves H2D transfer
            processed_items[(ready_item.req_id, ready_item.id)] = ready_item

        # 3. Reconstruct batch_mm_items in correct order and calculate offsets
        batch_mm_items = {}
        batch_request_offsets = []
        current_offset = 0

        for request in scheduled_requests:
            batch_request_offsets.append(current_offset)
            request_offset = 0
            for item in request.items:
                # Get the processed item in original order
                processed_item = processed_items[(item.req_id, item.id)]

                # Set the start position within this request
                processed_item.offset = request_offset
                # Update request_offset for next item
                request_offset += processed_item.length

                # Add to batch_mm_items maintaining request/item order
                if processed_item.modality_type not in batch_mm_items:
                    batch_mm_items[processed_item.modality_type] = []
                batch_mm_items[processed_item.modality_type].append(processed_item)

            current_offset += request_offset
        batch_request_offsets.append(current_offset)
        return batch_mm_items, batch_request_offsets

    @torch.inference_mode()
    def _model_forward(self, batch_mm_items, batch_request_offsets):
        batch_mm_data = {
            modality: [item.data for item in items]
            for modality, items in batch_mm_items.items()
        }
        # Stack all mm tensors and issue one encoder forward pass
        # Shape after torch.cat (total_patches, 3, pix_height, pix_width) - image only
        for item in batch_mm_items['image']:
            batch_mm_input = torch.cat(batch_mm_data['image'])
        batch_mm_features = self.model._process(batch_mm_input) if len(batch_mm_data['image']) > 0 else None
        assert batch_mm_features.shape[0] == sum([item.length for item in batch_mm_items['image']]), "batch_mm_features should have the same length as sum of item.length"
        assert batch_mm_features.dim() == 2, "batch_mm_features should be a 2D tensor"

        # TODO: add mrope config which seems need input ids of llm request, deferring for now
        mrope_config = None

        if batch_mm_features is None:
            return None

        return {
            "mm_embeddings": batch_mm_features,
            "mrope_config": mrope_config,
            "batch_request_offsets": batch_request_offsets
        }

    def forward(self, scheduled_requests, resource_manager = None):
        batch_mm_items, batch_request_offsets = asyncio.run(self._prepare_inputs(scheduled_requests))
        return self._model_forward(batch_mm_items, batch_request_offsets)