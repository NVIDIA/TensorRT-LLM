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

    def get_num_new_matched_tokens(self,
                                   request: LlmRequest = None,
                                   num_computed_tokens: int = None) -> int:
        if self.scheduler is not None:
            num_tokens, load_kv_async = self.scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens)

            mpi_broadcast((num_tokens, load_kv_async), root=0)
        else:
            num_tokens, load_kv_async = mpi_broadcast(None, root=0)

        # TODO: Do some stuff in the future to handle load_kv_async.

        return num_tokens
