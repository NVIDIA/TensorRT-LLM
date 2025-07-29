from typing import Optional

from tensorrt_llm._utils import mpi_broadcast, mpi_rank
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnectorManager as KvCacheConnectorManagerCpp
from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnectorScheduler as KvCacheConnectorSchedulerCpp
from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnectorWorker as KvCacheConnectorWorkerCpp
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest

from .scheduler import ScheduledRequests


class KvCacheConnectorWorker(KvCacheConnectorWorkerCpp):

    def __init__(self):
        super().__init__()

    def bind_connector_meta(self, metadata: object):
        self._metadata = metadata

    def get_connector_meta(self) -> object:
        return self._metadata

    def _clear_connector_meta(self):
        self._metadata = None


class KvCacheConnectorScheduler(KvCacheConnectorSchedulerCpp):

    def __init__(self):
        super().__init__()

    def build_connector_metadata(self, metadata: object):
        return None


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):

    def __init__(self, worker: KvCacheConnectorWorker,
                 scheduler: Optional[KvCacheConnectorScheduler]):
        assert (scheduler is not None) == (
            mpi_rank() == 0), "The scheduler may only exist on rank 0!"

        super().__init__()

        self.worker = worker
        self.scheduler = scheduler

        self.requests_awaiting_async_load_init = set()

        self.requests_awaiting_async_load_complete = []

    def get_num_new_matched_tokens(self, request: LlmRequest,
                                   num_computed_tokens: int) -> int:
        if self.scheduler is not None:
            num_tokens, load_kv_async = self.scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens)

            mpi_broadcast((num_tokens, load_kv_async), root=0)
        else:
            num_tokens, load_kv_async = mpi_broadcast(None, root=0)

        if num_tokens == 0 and load_kv_async:
            raise RuntimeError(
                "load_kv_async must be False when num_tokens is 0!")

        if load_kv_async:
            self.requests_awaiting_async_load_init.add(request.request_id)

        return num_tokens

    def build_connector_metadata(self) -> object:
        if self.scheduler is not None:
            metadata = self.scheduler.build_connector_metadata()
        else:
            metadata = None

        metadata = mpi_broadcast(metadata, root=0)

        self.worker.bind_connector_meta(metadata)

    def take_scheduled_requests_pending_transfer(
            self, scheduled_requests: ScheduledRequests) -> ScheduledRequests:
        allowed_context_requests = []
        async_load_requests = []

        for req in scheduled_requests.context_requests:
            if req.request_id in self.requests_awaiting_async_load_init:
                async_load_requests.append(req)
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
                self.requests_awaiting_async_load_init.remove(req.request_id)
                self.requests_awaiting_async_load_complete.append(req)
            else:
                allowed_context_requests.append(req)

        scheduled_requests.context_requests = allowed_context_requests

        return scheduled_requests
