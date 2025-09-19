from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

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
        draft_tokens (List[int]): The draft tokens of the generation request

        multimodal_embedding_handles (List[Dict[str, Any]]): The resulting multimodal embedding handles from ViT.
        multimodal_hashes (List[List[int]]): The multimodal hashes of each multimodal item in the request.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None

    # E-P Disaggregated Params
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = (
        None  # multimodal embedding handles should be a list of cudaIPC handles for each mm_embedding
    )
    multimodal_hashes: Optional[List[List[int]]] = (
        None  # user provided mm hashes should be a list of 8 integers
    )

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

    def __post_init__(self):
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
        if self.multimodal_embedding_handles is not None:
            if self.multimodal_hashes is not None:
                # if mm hashes are provided, kvcache reuse can be enabled
                assert len(self.multimodal_embedding_handles) == len(self.multimodal_hashes), (
                    "multimodal_embedding_handles and multimodal_hashes must have the same length"
                )
                for mm_hash in self.multimodal_hashes:
                    assert isinstance(mm_hash, list), "mm_hash must be a list"
                    assert len(mm_hash) == 8, "mm_hash must be a list of 8 integers"
                    assert all(isinstance(x, int) for x in mm_hash), "mm_hash must contain integers"
            else:
                # if user did not provide mm embedding handles, kvcache reuse will be disabled
                assert len(self.multimodal_embedding_handles) > 0, (
                    "multimodal_embedding_handles must be provided"
                )
                vals = np.random.randint(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=8, dtype=np.int32
                ).tolist()
                self.multimodal_hashes = [vals] * len(self.multimodal_embedding_handles)
