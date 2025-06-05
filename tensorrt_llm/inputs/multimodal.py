"""Multimodal utilities for handling images and other media types in TensorRT-LLM."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL
import torch
from blake3 import blake3
from torchvision.transforms import ToPILImage

# Default hasher
default_hasher = blake3


@dataclass
class MultimodalInput:
    multimodal_hashes: List[List[int]]
    """Hash values for multimodal data items (e.g., images).

    Each element is a list of 8 integers representing the hash digest of a multimodal item.
    """

    multimodal_positions: List[int]
    """Starting positions of each multimodal chunk in the token sequence.

    Contains only the start position of each chunk, not all positions of multimodal tokens.
    This is different from mm_positions elsewhere which contains all positions.
    """

    multimodal_lengths: List[int]
    """Length (number of tokens) of each multimodal item.

    Combined with multimodal_positions, this defines the token spans for each multimodal item.
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
        # only support single modality w/ PIL.Image.Image for now
        # TODO: possible hash collision w/ this simplified version (vllm/PR/17378)
        hasher = hash_lib()
        if isinstance(image, torch.Tensor):
            # TODO: Device tensor hashing is an open issue. Limited hashing to CPU for now.
            image = image.cpu()
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
    """Get multimodal token lengths from multimodal data items. """

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    num_mm_tokens = {}

    for modality, items in mm_items.items():
        if modality != "image":
            #TODO: support other modalities
            raise ValueError(
                f"Unsupported modality: {modality}. Only 'image' modality is currently supported for hashing."
            )
        if not hasattr(input_processor, "get_num_tokens_per_image"):
            #TODO: backward compatibility for models that don't yet have get_num_tokens_per_image implemented
            #TODO: only support qwen2_vl for now
            raise AttributeError(
                f"Input processor {type(input_processor).__name__} does not have 'get_num_tokens_per_image' method required for multimodal hashing."
            )

        modality_token_lengths = []
        for item in items:
            if isinstance(item, torch.Tensor):
                item = ToPILImage()(item)
            num_tokens = input_processor.get_num_tokens_per_image(
                image_width=item.width,
                image_height=item.height,
            )
            modality_token_lengths.append(num_tokens)

        num_mm_tokens[modality] = modality_token_lengths

    return num_mm_tokens['image']  # flatten all mm instances to a single list


def find_mm_token_positions(input_ids: Union[torch.Tensor, List[int],
                                             np.ndarray],
                            num_mm_tokens: List[int],
                            vocab_size: int,
                            mm_token_ids: torch.Tensor = None) -> List[int]:
    """Get multimodal token positions using IDs > vocab_size and known lengths.

    This function finds multimodal tokens (with IDs > vocab_size) and uses the
    provided lengths in num_mm_tokens to identify where each chunk starts.
    This works even when there are no gaps between different image sequences
    (e.g., when all images use the same token IDs).

    Args:
        input_ids: Token sequence (tensor, list, or numpy array)
        num_mm_tokens: List of lengths for each multimodal token chunk
        vocab_size: Size of the model's vocabulary
        mm_token_ids (optional): possible token ids for multimodal tokens

    Returns:
        List of starting positions for each multimodal token chunk
    """
    # Convert input_ids to tensor if needed
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)

    # Create mask for multimodal tokens
    if mm_token_ids is None:
        mm_mask = input_ids >= vocab_size
    else:
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

    return start_positions


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
