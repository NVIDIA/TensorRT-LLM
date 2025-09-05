from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from tensorrt_llm.inputs.multimodal import MultimodalInput

# isort: off
# needed before trying to import bindings to load tensorrt_libs
import tensorrt as trt  # noqa
# isort: on

from tensorrt_llm.bindings import executor as tllme


@dataclass(slots=True, kw_only=True)
class MultimodalDisaggParams:
    """Multimodal disaggregated serving parameters.

    Args:
        ctx_request_id (int): The context request id
        prompt_token_ids (List[int]): The prompt token ids after tokenization and multimodal token expansion
        multimodal_input (MultimodalInput): MultimodalInput object containing multimodal hashes, positions, and lengths.
        mm_embedding_handles (List[Dict[str, Any]]): Multimodal embedding handles for each multimodal item.
        opaque_state(bytes): Any additional state needing to be exchanged between mm encoder and ctx instance (reserved for future use)
    """
    ctx_request_id: Optional[int] = None
    prompt_token_ids: Optional[List[int]] = None
    multimodal_input: Optional[MultimodalInput] = None
    mm_embedding_handles: Optional[List[Dict[str, Any]]] = None
    opaque_state: Optional[bytes] = None
