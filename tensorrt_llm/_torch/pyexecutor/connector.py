from typing import Optional

from tensorrt_llm._utils import mpi_broadcast, mpi_rank
from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnectorManager as KvCacheConnectorManagerCpp
from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker, LlmRequest)


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):

    def __init__(self, worker: KvCacheConnectorWorker,
                 scheduler: Optional[KvCacheConnectorScheduler]):
        assert (scheduler is not None) == (
            mpi_rank() == 0), "The scheduler may only exist on rank 0!"
        super().__init__(worker, scheduler)

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.scheduler is not None:
            result = self.scheduler.getNumNewMatchedTokens(
                request, num_computed_tokens)
        else:
            result = None

        return mpi_broadcast(result, root=0)
