from dataclasses import dataclass
from typing import List, Optional

# isort: off
# needed before trying to import bindings to load tensorrt_libs
import tensorrt as trt  # noqa
# isort: on

from tensorrt_llm.bindings import executor as tllme


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
    """

    request_type: Optional[str] = None
    first_gen_tokens: Optional[List[int]] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        return tllme.ContextPhaseParams(
            self.first_gen_tokens, self.ctx_request_id, self.opaque_state, self.draft_tokens
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )
