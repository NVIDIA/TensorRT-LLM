"""Multimodal utilities for handling images and other media types in TensorRT-LLM."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from blake3 import blake3
from torchvision.transforms import ToPILImage

import tensorrt_llm
from tensorrt_llm.logger import logger

# Default hasher
default_hasher = blake3


@dataclass
class MultimodalInput:
    multimodal_hashes: List[List[int]]
    """Hash values for multimodal data items (e.g., images).

    Each element is a list of 8 integers representing the hash digest of a multimodal item.
    """

    multimodal_positions: List[int]
    """Starting positions of each contiguous multimodal token chunk in the token sequence.

    Contains only the start position of each chunk, not all positions of multimodal tokens.
    This is different from mm_positions elsewhere which contains all positions.
    """

    multimodal_lengths: List[int]
    """Length of each contiguous multimodal token chunk, including any special tokens.

    Each span is unique to its multimodal item and may include special tokens for some models,
    (e.g., image_end_token, image_break_token for mistral3) mixed with the actual multimodal tokens.
    """

    def __post_init__(self):
        """Validate input data structure and consistency."""
        # Validate multimodal_hashes
        if not isinstance(self.multimodal_hashes, list):
            raise TypeError("multimodal_hashes must be a list")

        # Check that hashes are lists of consistent length containing integers
        if not all(isinstance(h, list) for h in self.multimodal_hashes):
            raise TypeError("Each element in multimodal_hashes must be a list")

        # Check consistent length of hash arrays
        hash_lengths = [len(h) for h in self.multimodal_hashes]
        if min(hash_lengths) != max(hash_lengths):
            raise ValueError(
                f"All hash arrays must have the same length, got lengths: {hash_lengths}"
            )

        # Check that positions and lengths are valid
        if not all(isinstance(x, int) for x in self.multimodal_positions):
            raise TypeError("multimodal_positions must contain only integers")

        if not all(isinstance(x, int) for x in self.multimodal_lengths):
            raise TypeError("multimodal_lengths must contain only integers")

        # Check position and length arrays match in size
        if len(self.multimodal_positions) != len(self.multimodal_lengths):
            raise ValueError(
                f"Position and length arrays must match in size: "
                f"positions={len(self.multimodal_positions)}, lengths={len(self.multimodal_lengths)}"
            )

    @classmethod
    def from_components(cls, mm_hashes: List[List[int]],
                        mm_positions: List[int],
                        mm_lengths: List[int]) -> 'MultimodalInput':
        return cls(multimodal_hashes=mm_hashes,
                   multimodal_positions=mm_positions,
                   multimodal_lengths=mm_lengths)

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert data to tensors"""
        return (
            # int32 to match the type in TRTLLM SizeType32
            torch.tensor(self.multimodal_hashes, dtype=torch.int32),
            torch.tensor(self.multimodal_positions, dtype=torch.int32),
            torch.tensor(self.multimodal_lengths, dtype=torch.int32))


@dataclass
class MultimodalRuntimeData:
    """Runtime data for tracking multimodal token caching and reuse per request sequence.

    This class tracks which multimodal tokens are cached vs. need to be processed
    for each request sequence during both KV cache reuse and chunked prefill scenarios.

    Attributes:
        past_seen_token_num: Total number of tokens seen in previous iterations (cached)
        mm_token_lengths: Length of each multimodal token chunk
        mm_token_positions: Starting positions of each multimodal token chunk
        chunk_end_pos: End position of the current chunk for chunked prefill
        special_token_offsets: Starting positions of special tokens in the union of all multimodal token chunks (optional, depending on the model)

        num_unseen_mm_tokens: Number of multimodal tokens that are cached (computed)
        num_mm_tokens_in_chunk: Number of multimodal tokens in the current chunk (computed)
        total_mm_tokens_in_request: Total number of multimodal tokens in the request sequence (computed)

        num_unseen_special_tokens: Number of special tokens that are cached (computed)
        num_special_tokens_in_chunk: Number of special tokens in the current chunk (computed)
        total_special_tokens_in_request: Total number of special tokens in the request sequence (computed)
    """
    past_seen_token_num: int
    mm_token_lengths: List[int]
    mm_token_positions: List[int]
    chunk_end_pos: int
    special_token_offsets: List[int]

    num_unseen_mm_tokens: Optional[int] = None
    num_mm_tokens_in_chunk: Optional[int] = None
    total_mm_tokens_in_request: Optional[int] = None

    num_unseen_special_tokens: Optional[int] = 0
    num_special_tokens_in_chunk: Optional[int] = 0
    total_special_tokens_in_request: Optional[int] = 0

    # TODO: fine-grained control of encoder runner/cache to each mm_item

    def __post_init__(self):
        # Validate input data
        if self.total_mm_tokens_in_request is None:
            self.total_mm_tokens_in_request = sum(self.mm_token_lengths)
        if len(self.mm_token_positions) != len(self.mm_token_lengths):
            raise ValueError(
                f"mm_token_positions ({len(self.mm_token_positions)}) and mm_token_lengths ({len(self.mm_token_lengths)}) must have the same length"
            )

        if self.past_seen_token_num < 0:
            raise ValueError(
                f"past_seen_token_num must be non-negative, got {self.past_seen_token_num}"
            )

        if any(length <= 0 for length in self.mm_token_lengths):
            raise ValueError(
                f"All mm_token_lengths must be positive, got {self.mm_token_lengths}"
            )

        if any(pos < 0 for pos in self.mm_token_positions):
            raise ValueError(
                f"All mm_token_positions must be non-negative, got {self.mm_token_positions}"
            )

        if self.num_unseen_mm_tokens is None or self.num_mm_tokens_in_chunk is None:
            # Compute cached multimodal tokens based on positions and cached tokens
            self.num_unseen_mm_tokens = 0
            self.num_mm_tokens_in_chunk = 0
            remainder = 0
            for pos, length in zip(self.mm_token_positions,
                                   self.mm_token_lengths):
                if pos + length <= self.past_seen_token_num:
                    self.num_unseen_mm_tokens += length
                elif pos < self.past_seen_token_num:
                    # Partial overlap - only count the cached portion
                    self.num_unseen_mm_tokens += self.past_seen_token_num - pos
                    self.num_mm_tokens_in_chunk += min(
                        self.chunk_end_pos,
                        pos + length) - self.past_seen_token_num
                else:
                    if pos + length > self.chunk_end_pos:
                        # Partial overlap - only count the cached portion
                        if pos < self.chunk_end_pos:
                            self.num_mm_tokens_in_chunk += self.chunk_end_pos - pos
                        else:
                            remainder += length
                    else:
                        # Full overlap - count the entire mm item chunk
                        self.num_mm_tokens_in_chunk += length

        if len(self.special_token_offsets) > 0:
            self.num_unseen_special_tokens = sum(
                1 for offset in self.special_token_offsets
                if offset < self.num_unseen_mm_tokens)
            mm_tokens_end_pos = self.num_unseen_mm_tokens + self.num_mm_tokens_in_chunk
            self.num_special_tokens_in_chunk = sum(
                1 for offset in self.special_token_offsets
                if self.num_unseen_mm_tokens <= offset < mm_tokens_end_pos)

            self.total_special_tokens_in_request = len(
                self.special_token_offsets)

        if self.num_unseen_mm_tokens + self.num_mm_tokens_in_chunk + remainder > sum(
                self.mm_token_lengths):
            raise ValueError(
                f"num_unseen_mm_tokens ({self.num_unseen_mm_tokens}) + num_mm_tokens_in_chunk ({self.num_mm_tokens_in_chunk}) + remainder ({remainder}) must be less than or equal to sum of mm_token_lengths ({sum(self.mm_token_lengths)})"
            )


@dataclass
class MultimodalParams:
    """Unified container for multimodal parameters.

    This class encapsulates all multimodal-related data that flows through the system,
    providing a clean interface for handling multimodal inputs across different models.

    Attributes:
        multimodal_input: Multimodal input data with hashing information.
        multimodal_data: Processed multimodal data containing embeddings, configurations,
                        and modality-specific data organized by type.
        multimodal_runtime: Runtime data for tracking multimodal token caching and reuse
                           during KV cache scenarios. Contains information about cached
                           tokens, multimodal token positions, and lengths for efficient
                           processing during inference.

    Structure of multimodal_data:
        {
            "mrope_config": {
                "mrope_rotary_cos_sin": torch.Tensor,    # Rotary embeddings (Qwen2/2.5-VL)
                "mrope_position_deltas": torch.Tensor,   # Position deltas (Qwen2/2.5-VL)
            },
            "multimodal_embedding": torch.Tensor,        # Pre-computed vision embeddings
            "image": {
                "pixel_values": torch.Tensor,
                "image_height": torch.Tensor | List[int],
                "image_width": torch.Tensor | List[int],
            },
            "video": {
                "pixel_values": torch.Tensor,
                "video_height": torch.Tensor | List[int],
                "video_width": torch.Tensor | List[int],
            },
            "special_token_offsets": List[int],          # List of starting positions of special tokens in the union of all multimodal token chunks, if available
            # ... other modalities
        }
    """

    multimodal_input: Optional[MultimodalInput] = None
    multimodal_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    multimodal_runtime: Optional[MultimodalRuntimeData] = None

    def __post_init__(self):
        """Ensure default values are properly set."""
        if self.multimodal_data is None:
            self.multimodal_data = {}

    def _is_shared_tensor_dict(self, obj: Any) -> bool:
        """Check if an object is a shared tensor dictionary.

        Args:
            obj: Object to check

        Returns:
            True if the object is a shared tensor dictionary, False otherwise
        """
        if not isinstance(obj, dict):
            return False

        # Check for required keys that uniquely identify a shared tensor dict
        required_keys = {'method_key'}
        if not required_keys.issubset(obj.keys()):
            return False

        # Additional validation based on method_key
        method_key = obj.get('method_key')

        # Import here to avoid circular imports
        from tensorrt_llm._torch.shared_tensor import \
            _SharedTensorRebuildMethodRegistry

        if method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CUDA:
            cuda_keys = {'tensor_size', 'storage_handle', 'storage_device'}
            return cuda_keys.issubset(obj.keys())
        elif method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CPU:
            cpu_keys = {'tensor_size', 'storage_handle', 'manager_handle'}
            return cpu_keys.issubset(obj.keys())

        return False

    def _apply_tensor_operation(
            self, input_data: Union[torch.Tensor, List, dict, None],
            operation: str, **kwargs) -> Union[torch.Tensor, List, dict, None]:
        """Apply tensor operations recursively to nested data structures.

        This method handles three types of operations:
        - "to_handle": Convert tensors to shared tensor dictionaries
        - "to_tensor": Convert shared tensor dictionaries back to tensors
        - "to_device": Move tensors to specified device

        Args:
            input_data: Input data structure (tensor, list, dict, or None)
            operation: Operation to apply
            **kwargs: Additional arguments for the operation

        Returns:
            Transformed data structure
        """
        # Handle None case
        if input_data is None:
            return None

        # Handle list case - recursively process each element
        if isinstance(input_data, list):
            return [
                self._apply_tensor_operation(item, operation, **kwargs)
                for item in input_data
            ]

        # Handle dictionary case
        if isinstance(input_data, dict):
            if operation == "to_tensor" and self._is_shared_tensor_dict(
                    input_data):
                # Convert shared tensor dict back to tensor
                try:
                    # Import here to avoid circular imports
                    from tensorrt_llm._torch.shared_tensor import \
                        SharedTensorContainer

                    return SharedTensorContainer.from_dict(
                        input_data).get_local_view()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore tensor from shared tensor dict: {e}"
                    )
            else:
                # Regular dictionary - recursively process values
                return {
                    key: self._apply_tensor_operation(value, operation,
                                                      **kwargs)
                    for key, value in input_data.items()
                }

        # Handle tensor case
        if isinstance(input_data, torch.Tensor):
            if operation == "to_handle":
                try:
                    # Import here to avoid circular imports
                    from tensorrt_llm._torch.shared_tensor import \
                        SharedTensorContainer
                    return SharedTensorContainer.from_tensor(
                        input_data).dump_to_dict()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to convert tensor to shared tensor: {e}")
            elif operation == "to_device":
                device = kwargs.get('device')
                if device is None:
                    raise ValueError(
                        "Device must be specified for 'to_device' operation")

                pin_memory = kwargs.get('pin_memory', False)
                try:
                    if pin_memory and input_data.device.type == 'cpu':
                        return input_data.pin_memory().to(device,
                                                          non_blocking=True)
                    else:
                        return input_data.to(device, non_blocking=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to move tensor to device {device}: {e}")

        # For any other type, return as-is
        return input_data

    def to_handle(self, element: str) -> None:
        """Move specified multimodal data element to shared tensor.

        Args:
            element: Element to move (only "multimodal_data" is supported)

        Raises:
            ValueError: If element is not "multimodal_data"
            RuntimeError: If tensor conversion fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to convert

        transformed_data = self._apply_tensor_operation(data, "to_handle")
        setattr(self, element, transformed_data)

    def to_tensor(self, element: str) -> None:
        """Move specified multimodal data element from shared tensor.

        Args:
            element: Element to restore (only "multimodal_data" is supported)

        Raises:
            ValueError: If element is not "multimodal_data"
            RuntimeError: If tensor restoration fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to restore

        restored_data = self._apply_tensor_operation(data, "to_tensor")
        setattr(self, element, restored_data)

    def to_device(self,
                  element: str,
                  device: str,
                  pin_memory: bool = False,
                  target_keywords: Optional[List[str]] = None) -> None:
        """Move specified multimodal data element to target device.

        Args:
            element: Element to move (only "multimodal_data" is supported)
            device: Target device (e.g., "cuda", "cpu")
            pin_memory: Whether to pin memory for asynchronous transfers
            target_keywords: Optional list of keyword paths to filter which data to move.
                    Each string can be a simple key or dot-separated path
                    (e.g., ["image.pixel_values", "mrope_config"])
                    If provided, only data matching these paths will be moved to device.

        Raises:
            ValueError: If element is not "multimodal_data" or device is invalid
            RuntimeError: If device transfer fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to move

        # If keyword is specified, only move data for those keyword paths
        if target_keywords is not None:
            if not isinstance(data, dict):
                raise ValueError(
                    f"multimodal_data must be a dictionary when keyword is specified, "
                    f"got {type(data)}")

            # Process multiple keyword paths
            transformed_data = self._move_multiple_paths_to_device(
                data, target_keywords, device, pin_memory)
        else:
            # Move all data as before
            transformed_data = self._apply_tensor_operation(
                data, "to_device", device=device, pin_memory=pin_memory)

        setattr(self, element, transformed_data)

    def _move_multiple_paths_to_device(self, data: Dict[str, Any],
                                       target_keywords: List[str], device: str,
                                       pin_memory: bool) -> Dict[str, Any]:
        """Move multiple nested data paths to device.

        Args:
            data: The multimodal data dictionary
            target_keywords: List of keyword paths (can be dot-separated)
            device: Target device
            pin_memory: Whether to pin memory

        Returns:
            Updated data dictionary with specified paths moved to device
        """
        result = data
        for keyword_path in target_keywords:
            # Parse each keyword path
            if '.' in keyword_path:
                key_path = keyword_path.split('.')
            else:
                key_path = [keyword_path]

            # Navigate to the target location and move data
            current = result
            parent_path = key_path[:-1]
            target_key = key_path[-1]

            # Navigate to the parent dictionary
            for key in parent_path:
                if not isinstance(current, dict) or key not in current:
                    # Path doesn't exist, skip this keyword path
                    break
                current = current[key]
            else:
                # Check if the target key exists and move it to device
                if isinstance(current, dict) and target_key in current:
                    current[target_key] = self._apply_tensor_operation(
                        current[target_key],
                        "to_device",
                        device=device,
                        pin_memory=pin_memory)

        return result

    def strip_for_generation(self) -> None:
        """Strip multimodal data for generation processing.

        Keeps only mrope_position_deltas and removes all other multimodal data
        (embeddings, images, etc.) as they're not needed during generation.
        """
        if not self.multimodal_data:
            return

        # Extract mrope_position_deltas before clearing
        mrope_position_deltas = None
        if 'mrope_config' in self.multimodal_data:
            mrope_config = self.multimodal_data['mrope_config']
            if isinstance(mrope_config,
                          dict) and 'mrope_position_deltas' in mrope_config:
                mrope_position_deltas = mrope_config['mrope_position_deltas']

        # Clear all data and restore only position deltas if they exist
        self.multimodal_data = {}
        if mrope_position_deltas is not None:
            self.multimodal_data['mrope_config'] = {
                'mrope_position_deltas': mrope_position_deltas
            }

    def has_content(self) -> bool:
        """Check if this object contains any multimodal data."""
        return bool(self.multimodal_input or self.multimodal_data)


@dataclass
class MultimodalServerConfig():
    media_io_kwargs: Optional[dict] = None


# adopt from vllm : https://github.com/vllm-project/vllm/blob/main/vllm/vllm/multimodal/hash.py
def serialize_item(obj: object) -> bytes:
    # Simple cases
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (int, float)):
        return np.array(obj).tobytes()

    if isinstance(obj, PIL.Image.Image):
        return np.array(obj.convert("RGBA")).tobytes()
    if isinstance(obj, torch.Tensor):
        return obj.numpy().tobytes()
    if isinstance(obj, np.ndarray):
        return obj.tobytes()

    raise ValueError(f"Unsupported object type: {type(obj)}")


def apply_mm_hashes(mm_data: Dict[str, Any],
                    hash_lib=default_hasher) -> Dict[str, List[str]]:
    """Apply hashing to multimodal data items."""

    def _hash_image(image):
        # TODO: possible hash collision w/ this simplified version (vllm/PR/17378)
        hasher = hash_lib()
        if isinstance(image, torch.Tensor):
            # Ensure tensor is on CPU and contiguous for consistent hashing
            image = image.detach().cpu().contiguous()
            hasher.update(serialize_item(image))
        elif isinstance(image, list):
            # Hash each frame with a separator to avoid collisions between [A,B] and [AB]
            for frame in image:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        elif isinstance(image, tensorrt_llm.inputs.utils.VideoData):
            frames = image.frames
            for frame in frames:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        else:
            hasher.update(serialize_item(image))

        return hasher.hexdigest()

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    # TODO: need to hash both modality and item to distinguish modality (vllm/PR)
    mm_hashes = {
        modality: [_hash_image(item) for item in items]
        for modality, items in mm_items.items()
    }
    return mm_hashes


def hexdigest_to_int32(hex_digest: str) -> List[int]:
    """Convert a 256-bit hexadecimal digest to 8 int32 values."""
    if len(hex_digest) != 64:
        raise ValueError(
            f"Expected 64 character hexadecimal string, got {len(hex_digest)}")

    result = []
    for i in range(0, 64, 8):
        hex_chunk = hex_digest[i:i + 8]
        value = int(hex_chunk, 16)
        if value > 0x7FFFFFFF:  # Check if the highest bit is set (value > 2^31-1)
            value = value - 0x100000000  # Convert to signed by subtracting 2^32
        result.append(value)
    return result


def find_mm_token_lengths(mm_data: Dict[str, Any],
                          input_processor: Any) -> List[int]:
    """Get the maximum contiguous multimodal token lengths from multimodal data items.

    Returns the total token count for each multimodal item, including any special tokens
    (e.g., image_begin, image_end, image_break) that may be mixed with the actual
    multimodal content tokens. This mm_token_lengths represents the full contiguous chunk from beginning
    to end, not just pure image/video/audio tokens.
    """

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    num_mm_tokens = {}

    for modality, items in mm_items.items():
        if not hasattr(input_processor, f"get_num_tokens_per_{modality}"):
            raise AttributeError(
                f"Input processor {type(input_processor).__name__} does not have 'get_num_tokens_per_{modality}' method required for multimodal hashing."
            )

        modality_token_lengths = []
        for item in items:
            if modality == "image":
                if isinstance(item, torch.Tensor):
                    item = ToPILImage()(item)
                num_tokens = input_processor.get_num_tokens_per_image(
                    image=item, )
                modality_token_lengths.append(num_tokens)
            elif modality == "video":
                if isinstance(item, tensorrt_llm.inputs.utils.VideoData):
                    item = item.frames
                assert isinstance(item, list), "Video must be a list of frames"
                if isinstance(item[0], torch.Tensor):
                    item = [ToPILImage()(frame) for frame in item]
                num_tokens = input_processor.get_num_tokens_per_video(
                    video=item, )
                modality_token_lengths.append(num_tokens)
            else:
                # TODO: add audio support if needed
                raise ValueError(f"Unsupported modality: {modality}")

        num_mm_tokens[modality] = modality_token_lengths

    return num_mm_tokens  # flatten all mm instances to a single list


def find_mm_token_positions(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    num_mm_tokens: List[int],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None
) -> Tuple[List[int], List[int]]:
    """Get starting positions of contiguous multimodal token chunks using known lengths.

    This function finds multimodal tokens (with IDs > vocab_size or matching mm_token_ids)
    and uses the provided lengths in num_mm_tokens to identify where each contiguous chunk starts.
    Each chunk in num_mm_tokens is assumed to be a contiguous block of multimodal tokens for each multimodal item, and may include special tokens (e.g., image_begin, image_end, image_break) within the chunk.

    Note: at least one of vocab_size or mm_token_ids must be provided. If mm_token_ids
    is provided, vocab_size is ignored.

    Args:
        input_ids: Token sequence (tensor, list, or numpy array)
        num_mm_tokens: List of contiguous chunk lengths for each multimodal item
        vocab_size: Size of the model's vocabulary (used to identify tokens > vocab_size)
        mm_token_ids: Specific token IDs that represent multimodal tokens
        mm_special_token_ids: Specific token IDs that represent special multimodal tokens

    Returns:
        List of starting positions for each contiguous multimodal token
        (Optional) List of starting positions of special tokens in the union of all multimodal token chunks
    """
    if mm_token_ids is None and vocab_size is None:
        raise ValueError(
            "Provide either mm_token_ids or vocab_size to find multimodal token positions"
        )
    if mm_token_ids is not None and vocab_size is not None:
        logger.debug(
            "Both mm_token_ids and vocab_size are provided, using mm_token_ids and ignoring vocab_size"
        )

    # Convert input_ids to tensor if needed
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)

    # Create mask for multimodal tokens including special tokens if provided
    if mm_token_ids is None:
        mm_mask = input_ids >= vocab_size
        if mm_special_token_ids is not None:
            mm_special_token_ids = mm_special_token_ids.to(
                device=input_ids.device, dtype=input_ids.dtype)
            mm_mask = mm_mask | torch.isin(input_ids, mm_special_token_ids)
    else:
        mm_token_ids = mm_token_ids.to(device=input_ids.device,
                                       dtype=input_ids.dtype)
        if mm_token_ids.ndim != 1:
            raise ValueError("mm_token_ids must be a 1D tensor")
        if mm_special_token_ids is not None:
            mm_special_token_ids = mm_special_token_ids.to(
                device=input_ids.device, dtype=input_ids.dtype)
            mm_token_ids = torch.unique(
                torch.cat([mm_token_ids, mm_special_token_ids]))
        else:
            mm_token_ids = torch.unique(mm_token_ids)
        mm_mask = torch.isin(input_ids, mm_token_ids)

    # If no multimodal tokens found, return empty list
    if not torch.any(mm_mask):
        return []

    # Get positions of all multimodal tokens
    mm_positions = torch.where(mm_mask)[0].tolist()
    assert len(mm_positions) == sum(
        num_mm_tokens
    ), f"Number of multimodal tokens does not match sum of all lengths"

    # Use num_mm_tokens to find the starting position of each chunk
    start_positions = []
    current_position = 0

    # Process each expected length
    for length in num_mm_tokens:
        if current_position < len(mm_positions):
            # Add the starting position of this chunk
            start_positions.append(mm_positions[current_position])
            # Move to the next chunk
            current_position += length

    start_special_token_positions = []
    if mm_special_token_ids is not None:
        mm_token_ids = input_ids[mm_positions]
        special_token_mask_in_mm = torch.isin(mm_token_ids,
                                              mm_special_token_ids)
        start_special_token_positions = torch.where(
            special_token_mask_in_mm)[0].tolist()

    return start_positions, start_special_token_positions


def validate_mm_inputs(prompt_token_ids: Union[torch.Tensor, List[int],
                                               np.ndarray],
                       mm_hashes: List[List[int]], start_positions: List[int],
                       num_mm_tokens: List[int]) -> None:
    """Validates multimodal inputs for consistency and correctness."""
    # Validate number of hashes matches number of chunks
    if len(mm_hashes) != len(num_mm_tokens):
        raise AssertionError(
            f"Number of hashes ({len(mm_hashes)}) does not match "
            f"number of multimodal chunks ({len(num_mm_tokens)})")

    # Validate number of start positions matches number of chunks
    if len(start_positions) != len(num_mm_tokens):
        raise AssertionError(
            f"Number of start positions ({len(start_positions)}) does not match "
            f"number of multimodal chunks ({len(num_mm_tokens)})")
    # Validate each chunk's position and length
    prompt_len = len(prompt_token_ids)
    # Verify start_positions are sorted
    if not all(start_positions[i] < start_positions[i + 1]
               for i in range(len(start_positions) - 1)):
        raise AssertionError(
            "start_positions must be sorted in ascending order")
    for chunk_idx, (start_pos,
                    chunk_len) in enumerate(zip(start_positions,
                                                num_mm_tokens)):
        if start_pos < 0:
            raise AssertionError(
                f"Invalid negative start position {start_pos} for chunk {chunk_idx}"
            )

        if start_pos + chunk_len > prompt_len:
            raise AssertionError(
                f"Multimodal chunk {chunk_idx} at position {start_pos} with length {chunk_len} "
                f"exceeds input sequence length {prompt_len}")

        # Check for overlap with next chunk
        if chunk_idx < len(start_positions) - 1:
            next_start = start_positions[chunk_idx + 1]
            if start_pos + chunk_len > next_start:
                raise AssertionError(
                    f"Multimodal chunk {chunk_idx} at position {start_pos} with length {chunk_len} "
                    f"overlaps with chunk {chunk_idx + 1} at position {next_start}"
                )
