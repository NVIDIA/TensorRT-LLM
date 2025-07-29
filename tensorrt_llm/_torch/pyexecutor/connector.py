from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._utils import mpi_allgather, mpi_broadcast, mpi_rank
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


@dataclass
class Finished:
    saving: dict[int, LlmRequest]
    loading: dict[int, LlmRequest]

    def add_from(self, other: 'Finished'):
        self.saving.update(other.saving)
        self.loading.update(other.loading)

        other.saving = dict()
        other.loading = dict()

    def extract_by_id(self, saving_ids: list[int], loading_ids: list[int]):

        new_finished = Finished(dict(), dict())

        for req_id in saving_ids:
            new_finished.saving[req_id] = self.saving[req_id]
            del self.saving[req_id]
        for req_id in loading_ids:
            new_finished.loading[req_id] = self.loading[req_id]
            del self.loading[req_id]

        return new_finished

    def saving_ids(self) -> set[int]:
        return set(self.saving.keys())

    def loading_ids(self) -> set[int]:
        return set(self.loading.keys())

    @staticmethod
    def intersection(*all_finished: 'Finished') -> 'Finished':
        if len(all_finished) == 0:
            return Finished(dict(), dict())

        saving_ids = set.intersection(
            *[finished.saving_ids() for finished in all_finished])
        loading_ids = set.intersection(
            *[finished.loading_ids() for finished in all_finished])
        return Finished(
            dict([(req_id, all_finished[0].saving[req_id])
                  for req_id in saving_ids]),
            dict([(req_id, all_finished[0].loading[req_id])
                  for req_id in loading_ids]))

    def __sub__(self, other: 'Finished') -> 'Finished':
        return Finished(self.saving - other.saving,
                        self.loading - other.loading)


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):

    def __init__(self, worker: KvCacheConnectorWorker,
                 scheduler: Optional[KvCacheConnectorScheduler]):
        assert (scheduler is not None) == (
            mpi_rank() == 0), "The scheduler may only exist on rank 0!"

        super().__init__()

        self.worker = worker
        self.scheduler = scheduler

        # Requests that haven't yet been passed into get_finished.
        self.new_finished = Finished(dict(), dict())

        # Requests that have been passed into get_finished, but haven't yet been returned.
        self.pending_finished = Finished(dict(), dict())

        # Requests that have been returned from get_finished locally, but haven't yet been returned by all workers.
        self.local_finished = Finished(dict(), dict())

    def get_num_new_matched_tokens(self, request: LlmRequest,
                                   num_computed_tokens: int) -> int:
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            res = self.scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens)
        else:
            res = None

        (num_tokens, load_kv_async) = mpi_broadcast(res, root=0)

        if num_tokens == 0 and load_kv_async:
            raise RuntimeError(
                "load_kv_async must be False when num_tokens is 0!")

        if load_kv_async:
            self.new_finished.loading[request.request_id] = request
            request.is_kv_cache_connector_async_onboard = True

        return num_tokens

    def build_connector_metadata(self) -> object:
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            metadata = self.scheduler.build_connector_metadata()
        else:
            metadata = None

        metadata = mpi_broadcast(metadata, root=0)

        self.worker.bind_connector_meta(metadata)

    def request_finished(self, req: LlmRequest) -> bool:
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            saving_async = self.scheduler.request_finished(req)
        else:
            saving_async = None

        saving_async = mpi_broadcast(saving_async, root=0)

        if saving_async:
            self.new_finished.saving[req.request_id] = req
            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        return saving_async

    def take_scheduled_requests_pending_load(
            self, scheduled_requests: ScheduledRequests) -> ScheduledRequests:
        allowed_context_requests = []

        for req in scheduled_requests.context_requests:
            if req.request_id in self.new_finished.loading.keys():
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
            else:
                allowed_context_requests.append(req)

        scheduled_requests.context_requests = allowed_context_requests

        return scheduled_requests

    def get_finished(self) -> list[LlmRequest]:
        started_loading_req_ids = list(self.new_finished.loading_ids())
        finished_gen_req_ids = list(self.new_finished.saving_ids())

        self.pending_finished.add_from(self.new_finished)
        (finished_saving,
         finished_loading) = self.worker.get_finished(finished_gen_req_ids,
                                                      started_loading_req_ids)

        new_local_finished = self.pending_finished.extract_by_id(
            finished_saving, finished_loading)

        # Get all pending finished requests for this worker.
        self.local_finished.add_from(new_local_finished)

        # Broadcast this to all other workers.
        finished_saving = list(self.local_finished.saving_ids())
        finished_loading = list(self.local_finished.loading_ids())

        all_results = mpi_allgather((finished_saving, finished_loading))

        # Find only the requests that have been reported complete by all workers.
        intersect_finished_saving = set.intersection(
            *[set(res[0]) for res in all_results])
        intersect_finished_loading = set.intersection(
            *[set(res[1]) for res in all_results])

        all_finished = self.local_finished.extract_by_id(
            intersect_finished_saving, intersect_finished_loading)

        # For requests that have finished loading, move them back to the context state.
        for req in all_finished.loading.values():
            req.state = LlmRequestState.CONTEXT_INIT

        return list(all_finished.saving.values())
