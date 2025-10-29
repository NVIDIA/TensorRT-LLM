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
        for llm_req in scheduled_batch.all_requests():
            if llm_req.is_disagg_generation_init_state:
                logger.info(
                    f"Skip assigning sequence slot for DISAGG_GENERATION_INIT request."
                )
                continue
            if llm_req.seq_slot is None or llm_req.is_disagg_generation_transmission_complete:
                llm_req.seq_slot = self.slot_manager.add_slot(
                    llm_req.request_id)
                llm_req.py_seq_slot = llm_req.seq_slot
                if llm_req.return_perf_metrics:
                    llm_req.set_first_scheduled_time()

    def free_resources(self, request: LlmRequest) -> None:
        self.slot_manager.remove_slot(request.request_id)
