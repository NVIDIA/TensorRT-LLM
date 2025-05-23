from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class MultimodalParams:
    """
    Parameters for multimodal parameters in disaggregated serving as the interface between mm_encoder and llm servers.

    This class holds information needed to reconstruct prompt_token_ids and mm_embedding in LLM servers.

    Args:
        embeddings (Optional[Dict[str, Any]]):
            Metadata for reconstructing embedding tensors via CUDA IPC.
            The tensor data is stored in shared memory or device memory (shared by cudaIPC) for LLM server reconstruction.

        mrope_config (Optional[dict]):
            Configuration for multimodal Rotary Position Embedding parameters.

        num_items (Optional[int]):
            Number of multimodal items in the batch. Used to reconstruct input_ids.

        item_offsets (Optional[List[int]]):
            Offsets for positioning each multimodal item in the sequence.

        item_token_length (Optional[List[int]]):
            Token lengths for each multimodal item.

    Note:
        As an experimental feature, all fields are currently optional to allow flexibility during development.
        In future, we should stabilize the interface by defining a fixed set of required fields.
    """
    #embeddings: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[Dict[str, Any]]] = None # TODO: change to Dict[str, Any]
    mrope_config: Optional[dict] = None
    num_items: Optional[int] = 0
    item_offsets: Optional[List[int]] = None
    item_token_length: Optional[List[int]] = None