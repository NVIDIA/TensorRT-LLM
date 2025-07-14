from itertools import chain

from ordered_set import OrderedSet

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import *
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter

class APIDrafter(Drafter):

    def __init__(
        self,
        spec_config: "ExternalAPIConfig",
        endpoint: str,
    ):
        super().__init__(spec_resource_manager=None)
        self.max_draft_len = spec_config.max_draft_len
        assert endpoint is not None, "API endpoint is required for external API speculative decoding."
        self.endpoint = endpoint
    
    def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ):
        # TODO: Implement API endpoint call here
        return []

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
    ) -> None:
        # Sort by request_id when py_batch_idx is None as a fallback.
        # This happens in the disagg case: for a set of new requests, we draft
        # before forward_step, so py_batch_idx is not assigned.
        for request in sorted(
                scheduled_requests.generation_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            # Add new token to a copy of the generated tokens to find new draft tokens
            prefix = list(request.get_tokens()[0])  # Get a copy

            # Generate draft tokens
            draft_tokens = self.get_draft_tokens(
                prefix,
                request.request_id,
                request.py_end_id,
                request.py_orig_prompt_len + request.py_max_new_tokens,
            )
            # Pad length to `self.max_draft_len`
            if len(draft_tokens) > 0:
                pad_length = self.max_draft_len - len(draft_tokens)
                draft_tokens.extend([request.py_end_id] * pad_length)
            request.py_draft_tokens = draft_tokens
