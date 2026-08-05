from tensorrt_llm.logger import logger
from tensorrt_llm.serve.perf_time_events_writer import emit_event

from .llm_request import LlmRequest
from .resource_manager import BaseResourceManager, SlotManager
from .scheduler import ScheduledRequests


def _ctx_request_id(llm_req: LlmRequest):
    """Cross-worker time-event join key (ctx <-> gen <-> router).

    Reads ``py_disaggregated_params.ctx_request_id`` the same way
    ``py_executor._disagg_ctx_request_id`` does; ``None`` on non-disagg runs.
    """
    params = getattr(llm_req, "py_disaggregated_params", None)
    return getattr(params, "ctx_request_id", None) if params is not None else None


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
                # Time-event #10a: gen-init request admitted -- it is allowed to
                # pull KV cache but deliberately not given a seq-slot yet. This
                # branch re-fires on every scheduler pass while the request waits
                # in INIT state, so guard to the first fire. emit_event is inert
                # unless TRTLLM_PERF_TIME_EVENTS_PATH is set.
                if (llm_req.return_perf_metrics and
                        not getattr(llm_req, "py_te_gen_init_scheduled", False)):
                    llm_req.py_te_gen_init_scheduled = True
                    emit_event("gen", "gen_init_scheduled",
                               request_id=llm_req.request_id,
                               ctx_request_id=_ctx_request_id(llm_req))
                continue
            if llm_req.seq_slot is None or llm_req.is_disagg_generation_transmission_complete:
                llm_req.seq_slot = self.slot_manager.add_slot(
                    llm_req.request_id)
                llm_req.py_seq_slot = llm_req.seq_slot
                if llm_req.return_perf_metrics:
                    llm_req.set_first_scheduled_time()
                    # Time-event #10b: real scheduler admission (ctx prefill or
                    # gen decode). Distinct from #10a -- this is the seq-slot
                    # grant, gen-side only after KV transfer completes. Guard to
                    # first fire (the block re-enters when a gen request flips
                    # to TRANS_COMPLETE).
                    if not getattr(llm_req, "py_te_first_scheduled", False):
                        llm_req.py_te_first_scheduled = True
                        role = ("gen" if llm_req.is_generation_only_request()
                                else "ctx")
                        emit_event(role, f"{role}_first_scheduled",
                                   request_id=llm_req.request_id,
                                   ctx_request_id=_ctx_request_id(llm_req))

    def free_resources(self, request: LlmRequest) -> None:
        self.slot_manager.remove_slot(request.request_id)
