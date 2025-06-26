import itertools

from .llm_request import LlmRequest
from .resource_manager import BaseResourceManager, SlotManager
from .scheduler import ScheduledRequests


class SeqSlotManager(BaseResourceManager):

    def __init__(self, max_num_sequences: int):
        self.slot_manager = SlotManager(max_num_sequences)

    def get_max_resource_count(self) -> int:
        return self.slot_manager.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 1

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for llm_req in itertools.chain(scheduled_batch.context_requests,
                                       scheduled_batch.generation_requests):
            if (llm_req.is_context_init_state and llm_req.seq_slot is None) or \
                llm_req.is_disagg_generation_transmission_complete:
                llm_req.seq_slot = self.slot_manager.add_slot(
                    llm_req.request_id)
                if llm_req.return_perf_metrics:
                    llm_req.create_first_scheduled_time()

    def free_resources(self, request: LlmRequest) -> None:
        self.slot_manager.remove_slot(request.request_id)
