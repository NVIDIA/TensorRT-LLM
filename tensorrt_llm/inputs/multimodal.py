# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multimodal utilities for handling images and other media types in TensorRT-LLM."""

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from blake3 import blake3

import tensorrt_llm
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

# Default hasher
default_hasher = blake3


@dataclass(frozen=True)
class MultimodalItemRun:
    prompt_start: int
    run_length: int
    non_embed_offsets: Tuple[int, ...] = ()


MultimodalItemRuns = List[List[MultimodalItemRun]]


@dataclass
class PreparedMultimodalItemRuns:
    multimodal_item_runs: MultimodalItemRuns
    multimodal_embedding_lengths: List[int]
    multimodal_prompt_lengths: List[int]
    multimodal_item_run_spans: List[List[Tuple[int, int]]]


def unpack_multimodal_item_run(run: Any, field_name: str, item_idx: int,
                               run_idx: int) -> Tuple[Any, Any, Any]:
    """Return a multimodal item-run tuple from dataclass or tuple-shaped input."""
    if all(
            hasattr(run, attr)
            for attr in ("prompt_start", "run_length", "non_embed_offsets")):
        return run.prompt_start, run.run_length, run.non_embed_offsets
    if isinstance(run, (list, tuple)) and len(run) == 3:
        return run
    raise TypeError(f"{field_name}[{item_idx}][{run_idx}] must be a "
                    "MultimodalItemRun or "
                    "(prompt_start, run_length, non_embed_offsets) tuple")


def _validate_multimodal_hashes(mm_hashes: List[List[int]]) -> None:
    if not isinstance(mm_hashes, list):
        raise TypeError("multimodal_hashes must be a list")
    for item_idx, mm_hash in enumerate(mm_hashes):
        if not isinstance(mm_hash, list):
            raise TypeError(f"multimodal_hashes[{item_idx}] must be a list")
        if len(mm_hash) != 8:
            raise ValueError(
                f"multimodal_hashes[{item_idx}] must be a list of 8 integers")
        if not all(isinstance(hash_part, int) for hash_part in mm_hash):
            raise TypeError(
                f"multimodal_hashes[{item_idx}] must contain only integers")


def prepare_multimodal_item_runs(
        mm_hashes: Optional[List[List[int]]],
        mm_item_runs: MultimodalItemRuns,
        prompt_len: Optional[int] = None) -> PreparedMultimodalItemRuns:
    """Validate and derive multimodal item-run metadata in one pass."""
    if mm_hashes is not None:
        _validate_multimodal_hashes(mm_hashes)
    field_name = "multimodal_item_runs"
    if not isinstance(mm_item_runs, list):
        raise TypeError(f"{field_name} must be a list")
    if mm_hashes is not None and len(mm_item_runs) != len(mm_hashes):
        raise ValueError(
            f"{field_name} length ({len(mm_item_runs)}) must match "
            f"multimodal_hashes length ({len(mm_hashes)})")

    normalized_runs: MultimodalItemRuns = []
    embedding_lengths: List[int] = []
    prompt_lengths: List[int] = []
    item_run_spans: List[List[Tuple[int, int]]] = []
    occupied_runs: List[Tuple[int, int, int, int]] = []

    for item_idx, item_runs in enumerate(mm_item_runs):
        if not isinstance(item_runs, list):
            raise TypeError(f"{field_name}[{item_idx}] must be a list")
        if len(item_runs) == 0:
            raise ValueError(f"{field_name}[{item_idx}] must not be empty")

        normalized_item_runs: List[MultimodalItemRun] = []
        item_embedding_length = 0
        item_prompt_length = 0
        item_spans: List[Tuple[int, int]] = []
        previous_end: Optional[int] = None
        for run_idx, run in enumerate(item_runs):
            prompt_start, run_length, non_embed_offsets = (
                unpack_multimodal_item_run(run, field_name, item_idx, run_idx))
            if not isinstance(prompt_start, int) or not isinstance(
                    run_length, int):
                raise TypeError(
                    f"{field_name}[{item_idx}][{run_idx}] start/length must be integers"
                )
            if prompt_start < 0:
                raise ValueError(
                    f"{field_name}[{item_idx}][{run_idx}] must not contain negative positions"
                )
            if run_length <= 0:
                raise ValueError(
                    f"{field_name}[{item_idx}][{run_idx}] must have positive length"
                )
            run_end = prompt_start + run_length
            if prompt_len is not None and run_end > prompt_len:
                raise ValueError(
                    f"{field_name}[{item_idx}][{run_idx}] ends at "
                    f"{run_end}, exceeding input sequence length {prompt_len}")
            if non_embed_offsets is None:
                non_embed_offsets = ()
            if not isinstance(non_embed_offsets, (list, tuple)):
                raise TypeError(
                    f"{field_name}[{item_idx}][{run_idx}] non-embed offsets must be a list"
                )
            normalized_offsets = tuple(non_embed_offsets)
            if not all(
                    isinstance(offset, int) for offset in normalized_offsets):
                raise TypeError(
                    f"{field_name}[{item_idx}][{run_idx}] non-embed offsets must contain only integers"
                )
            if any(offset < 0 or offset >= run_length
                   for offset in normalized_offsets):
                raise ValueError(
                    f"{field_name}[{item_idx}][{run_idx}] non-embed offsets must be within the run"
                )
            if normalized_offsets != tuple(sorted(set(normalized_offsets))):
                raise ValueError(
                    f"{field_name}[{item_idx}][{run_idx}] non-embed offsets must be ordered and unique"
                )
            if previous_end is not None and prompt_start < previous_end:
                raise ValueError(
                    f"{field_name}[{item_idx}] must be ordered and non-overlapping"
                )

            normalized_run = MultimodalItemRun(
                prompt_start=prompt_start,
                run_length=run_length,
                non_embed_offsets=normalized_offsets)
            normalized_item_runs.append(normalized_run)
            item_embedding_length += run_length - len(
                normalized_run.non_embed_offsets)
            item_prompt_length += run_length
            item_spans.append((prompt_start, run_length))
            occupied_runs.append((prompt_start, run_end, item_idx, run_idx))
            previous_end = run_end

        normalized_runs.append(normalized_item_runs)
        embedding_lengths.append(item_embedding_length)
        prompt_lengths.append(item_prompt_length)
        item_run_spans.append(item_spans)

    occupied_runs.sort()
    for (_prev_start, prev_end, prev_item_idx,
         prev_run_idx), (run_start, _, item_idx,
                         run_idx) in zip(occupied_runs,
                                         occupied_runs[1:],
                                         strict=False):
        if run_start < prev_end:
            raise ValueError(f"{field_name} must be globally non-overlapping "
                             f"but [{item_idx}][{run_idx}] overlaps "
                             f"[{prev_item_idx}][{prev_run_idx}]")

    return PreparedMultimodalItemRuns(
        multimodal_item_runs=normalized_runs,
        multimodal_embedding_lengths=embedding_lengths,
        multimodal_prompt_lengths=prompt_lengths,
        multimodal_item_run_spans=item_run_spans,
    )


def _positions_to_item_runs(
        item_positions: List[List[int]],
        item_embed_flags: Optional[List[List[bool]]] = None
) -> MultimodalItemRuns:
    runs_by_item: MultimodalItemRuns = []
    if item_embed_flags is None:
        item_embed_flags = [[True] * len(positions)
                            for positions in item_positions]
    for item_idx, (positions, embed_flags) in enumerate(
            zip(item_positions, item_embed_flags, strict=True)):
        if not positions:
            runs_by_item.append([])
            continue
        if len(positions) != len(embed_flags):
            raise ValueError(
                f"item_positions[{item_idx}] and item_embed_flags[{item_idx}] "
                "must have the same length")
        item_runs: List[MultimodalItemRun] = []
        run_start = positions[0]
        previous_position = positions[0]
        run_length = 1
        non_embed_offsets: List[int] = [] if embed_flags[0] else [0]
        for position, is_embed in zip(positions[1:],
                                      embed_flags[1:],
                                      strict=True):
            if position == previous_position + 1:
                if not is_embed:
                    non_embed_offsets.append(run_length)
                run_length += 1
            else:
                item_runs.append(
                    MultimodalItemRun(
                        prompt_start=run_start,
                        run_length=run_length,
                        non_embed_offsets=tuple(non_embed_offsets)))
                run_start = position
                run_length = 1
                non_embed_offsets = [] if is_embed else [0]
            previous_position = position
        item_runs.append(
            MultimodalItemRun(prompt_start=run_start,
                              run_length=run_length,
                              non_embed_offsets=tuple(non_embed_offsets)))
        runs_by_item.append(item_runs)
    return runs_by_item


def strip_mm_data_for_generation(mm_data: Dict[str, Any]) -> None:
    """Clear `mm_data` in place, retaining only `mrope_config.mrope_position_deltas`.

    Shared primitive behind `MultimodalParams.strip_for_generation` (which applies
    it to a detached copy, preserving its caller's dict) and the post-prefill
    release path in `py_executor` (which applies it directly to the shared
    `request.py_multimodal_data` to actually free pinned encoder outputs).
    """
    if not mm_data:
        return
    mrope_config = mm_data.get('mrope_config')
    mrope_deltas = None
    if isinstance(mrope_config, dict):
        mrope_deltas = mrope_config.get('mrope_position_deltas')
    mm_data.clear()
    if mrope_deltas is not None:
        mm_data['mrope_config'] = {'mrope_position_deltas': mrope_deltas}


@dataclass
class MultimodalInput:
    """Per-item multimodal metadata for request layout and KV-cache hashing.

    Indexed per logical multimodal item (one image, video, or audio clip).
    `multimodal_item_runs` is the sole prompt-layout source of truth.
    """

    multimodal_hashes: List[List[int]]
    """Hash digest per logical unit (list of 8 int32 each)."""

    multimodal_item_runs: MultimodalItemRuns
    """Exact prompt token runs covered by each multimodal item.

    The outer list must match ``multimodal_hashes``. Each inner list contains
    compact ``MultimodalItemRun`` entries for that multimodal item.
    ``non_embed_offsets`` are local offsets inside the run that belong to the
    multimodal prompt layout but do not consume encoder-output embedding rows.
    Empty offsets mean the full run maps to embeddings.
    """

    multimodal_uuids: Optional[List[Optional[str]]] = None
    """Optional user-provided UUIDs for multimodal data items.

    When provided, these UUIDs will be returned in KV cache events instead of the
    computed hash hex string. This enables deterministic cache identification across
    sessions using user-defined stable identifiers.

    Each element can be:
    - A string UUID: Used as the cache identifier (returned in events)
    - None: Falls back to content-based hashing for that item

    If the UUID string is longer than 32 bytes, it will be hashed internally
    for cache key computation, but the original UUID string is preserved and
    returned in KV cache events.
    """

    prompt_len: InitVar[Optional[int]] = None
    """Optional prompt length used to validate item-run prompt bounds."""

    multimodal_embedding_lengths: List[int] = field(init=False)
    """Per-item encoder-output row counts, precomputed from item runs."""

    multimodal_prompt_lengths: List[int] = field(init=False)
    """Per-item prompt-owned token counts, precomputed from item runs."""

    multimodal_item_run_spans: List[List[Tuple[int, int]]] = field(init=False)
    """Per-item prompt spans, precomputed from item runs."""

    def __post_init__(self, prompt_len: Optional[int]):
        """Validate input data structure and consistency."""
        prepared_item_runs = prepare_multimodal_item_runs(
            self.multimodal_hashes,
            self.multimodal_item_runs,
            prompt_len=prompt_len)
        self.multimodal_item_runs = prepared_item_runs.multimodal_item_runs
        self.multimodal_embedding_lengths = (
            prepared_item_runs.multimodal_embedding_lengths)
        self.multimodal_prompt_lengths = (
            prepared_item_runs.multimodal_prompt_lengths)
        self.multimodal_item_run_spans = (
            prepared_item_runs.multimodal_item_run_spans)

        # Validate multimodal_uuids if provided
        if self.multimodal_uuids is not None:
            if not isinstance(self.multimodal_uuids, list):
                raise TypeError("multimodal_uuids must be a list")
            if len(self.multimodal_uuids) != len(self.multimodal_hashes):
                raise ValueError(
                    f"multimodal_uuids length ({len(self.multimodal_uuids)}) must match "
                    f"multimodal_hashes length ({len(self.multimodal_hashes)})")
            for i, uuid in enumerate(self.multimodal_uuids):
                if uuid is not None and not isinstance(uuid, str):
                    raise TypeError(
                        f"multimodal_uuids[{i}] must be a string or None, got {type(uuid)}"
                    )

    @classmethod
    def from_components(
        cls,
        mm_hashes: List[List[int]],
        mm_item_runs: MultimodalItemRuns,
        mm_uuids: Optional[List[Optional[str]]] = None,
        prompt_len: Optional[int] = None,
    ) -> 'MultimodalInput':
        return cls(
            multimodal_hashes=mm_hashes,
            multimodal_item_runs=mm_item_runs,
            multimodal_uuids=mm_uuids,
            prompt_len=prompt_len,
        )


@dataclass
class MultimodalRuntimeData:
    """Runtime data for tracking multimodal embedding caching and reuse per request sequence.

    Constructed from `py_multimodal_data["multimodal_embed_mask_cumsum"]`
    (int64 CPU cumsum populated by the producer); counts are derived via
    three O(1) cumsum lookups. Handles non-contiguous embedding positions
    (inline specials, interleaved text) natively.

    Attributes:
        past_seen_token_num: Total tokens already processed in previous iterations (cached)
        chunk_end_pos: End position of the current chunk for chunked prefill
        embed_mask_cumsum: int64 prefix sum of the flat embedding-slot mask

        num_cached_mm_tokens: Number of embeddings already cached (computed)
        num_mm_tokens_in_chunk: Number of embeddings in the current chunk (computed)
        total_embeds_in_request: Total embeddings in the request (computed)
    """
    past_seen_token_num: int
    chunk_end_pos: int
    embed_mask_cumsum: torch.Tensor

    num_cached_mm_tokens: Optional[int] = None
    num_mm_tokens_in_chunk: Optional[int] = None
    total_embeds_in_request: Optional[int] = None

    def __post_init__(self):
        if self.past_seen_token_num < 0:
            raise ValueError(
                f"past_seen_token_num must be non-negative, got {self.past_seen_token_num}"
            )
        if self.embed_mask_cumsum is None:
            raise ValueError(
                "MultimodalRuntimeData requires embed_mask_cumsum.")

        cs = self.embed_mask_cumsum
        # int(cs[idx]) below would D2H-sync if cs lived on CUDA.
        assert cs.device.type == "cpu", f"embed_mask_cumsum must be CPU, got {cs.device}"
        assert cs.numel() > 0, "embed_mask_cumsum must be non-empty"
        assert self.chunk_end_pos >= self.past_seen_token_num, (
            f"chunk_end_pos ({self.chunk_end_pos}) < past_seen_token_num "
            f"({self.past_seen_token_num})")
        assert 0 <= self.past_seen_token_num <= cs.numel(), (
            f"past_seen_token_num {self.past_seen_token_num} out of range "
            f"[0, {cs.numel()}]")
        assert self.chunk_end_pos <= cs.numel(), (
            f"chunk_end_pos {self.chunk_end_pos} > cumsum length {cs.numel()}")

        self.num_cached_mm_tokens = (int(cs[self.past_seen_token_num - 1])
                                     if self.past_seen_token_num > 0 else 0)
        self.num_mm_tokens_in_chunk = (int(cs[self.chunk_end_pos - 1]) -
                                       self.num_cached_mm_tokens
                                       if self.chunk_end_pos > 0 else 0)
        self.total_embeds_in_request = int(cs[-1])


# Keys under `MultimodalParams.multimodal_data` that are CPU-resident metadata
# and must never be moved to GPU by `MultimodalParams.to_device`.
# Extend only after auditing each key's consumers.
_CPU_ONLY_MULTIMODAL_DATA_KEYS = frozenset({
    "multimodal_embed_mask_cumsum",
})


@dataclass
class MultimodalParams:
    """Unified container for multimodal parameters.

    This class encapsulates all multimodal-related data that flows through the system,
    providing a clean interface for handling multimodal inputs across different models.

    Attributes:
        multimodal_input: Multimodal input data with hashing information.
        multimodal_data: Processed multimodal data containing embeddings, configurations,
                        and modality-specific data organized by type.
        multimodal_runtime: Runtime data for chunked multimodal embedding reuse.
                           Counts are derived from the embedding-mask cumsum;
                           prompt layout lives in `multimodal_input.multimodal_item_runs`.

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
                # Regular dictionary - recursively process values.
                if operation == "to_device":
                    return {
                        key: (value if key in _CPU_ONLY_MULTIMODAL_DATA_KEYS
                              else self._apply_tensor_operation(
                                  value, operation, **kwargs))
                        for key, value in input_data.items()
                    }
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
                        return maybe_pin_memory(input_data).to(
                            device, non_blocking=True)
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
                    if target_key in _CPU_ONLY_MULTIMODAL_DATA_KEYS:
                        logger.warning_once(
                            "to_device('%s') skipped: key is CPU-only "
                            "multimodal metadata.",
                            keyword_path,
                            key="mm_cpu_only_skip",
                        )
                        continue
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
        # Detach from the caller's dict, then apply the in-place primitive.
        self.multimodal_data = dict(self.multimodal_data)
        strip_mm_data_for_generation(self.multimodal_data)

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
    if isinstance(obj, (tuple, list)):
        # Support compound types like audio (np.ndarray, sample_rate).
        # Use length-delimited framing so sequences with different element
        # boundaries (e.g. ["ab", "c"] vs ["a", "bc"]) cannot collide.
        container_tag = b"T" if isinstance(obj, tuple) else b"L"
        parts = [container_tag, len(obj).to_bytes(8, "big", signed=False)]
        for x in obj:
            payload = serialize_item(x)
            parts.append(len(payload).to_bytes(8, "big", signed=False))
            parts.append(payload)
        return b"".join(parts)

    raise ValueError(f"Unsupported object type: {type(obj)}")


def apply_mm_hashes(
    mm_data: Dict[str, Any],
    mm_uuids: Optional[Dict[str, List[Optional[str]]]] = None,
    hash_lib=default_hasher
) -> Tuple[Dict[str, List[str]], Optional[List[Optional[str]]]]:
    """Apply hashing to multimodal data, one hash per multimodal item.

    When a UUID is provided for an item, the hash is computed from both the UUID
    and the content together: BLAKE3(UUID || Content). This ensures:
    - Cache correctness: Different content always produces different hashes
    - User isolation: Same content with different UUIDs produces different hashes
    - The original UUID string is preserved and returned in KV cache events

    Args:
        mm_data: Dictionary of modality -> data items
        mm_uuids: Optional dictionary of modality -> list of UUID strings.
                  Use None for items that should use content-based hashing only.
        hash_lib: Hash function to use (default: blake3)

    Returns:
        Tuple of:
        - Dictionary of modality -> list of hash hex strings (64 chars each)
        - Flattened list of original UUID strings (or None for content-hashed items)
    """

    def _hash_content(hasher, item):
        """Hash the content of a multimodal item into the provided hasher."""
        if isinstance(item, torch.Tensor):
            # Ensure tensor is on CPU and contiguous for consistent hashing
            item = item.detach().cpu().contiguous()
            hasher.update(serialize_item(item))
        elif isinstance(item, list):
            # Hash each frame with a separator to avoid collisions between [A,B] and [AB]
            for frame in item:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        elif isinstance(item, tensorrt_llm.inputs.utils.VideoData):
            frames = item.frames
            for frame in frames:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        else:
            hasher.update(serialize_item(item))

    def _hash_item(item):
        """Hash only the content of a multimodal item (no UUID)."""
        # TODO: possible hash collision w/ this simplified version (vllm/PR/17378)
        hasher = hash_lib()
        _hash_content(hasher, item)
        return hasher.hexdigest()

    def _hash_item_with_uuid(item, uuid: str):
        """Hash UUID and content together: BLAKE3(UUID || Content).

        This creates a unique hash that incorporates both the user-provided
        identifier and the actual content, ensuring cache correctness while
        supporting user-defined cache isolation.
        """
        hasher = hash_lib()
        # Hash UUID first with delimiters to prevent length-extension ambiguity
        hasher.update(b"<uuid>")
        hasher.update(uuid.encode('utf-8'))
        hasher.update(b"</uuid>")
        # Then hash the content
        hasher.update(b"<content>")
        _hash_content(hasher, item)
        hasher.update(b"</content>")
        return hasher.hexdigest()

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }

    # Collect UUIDs in the same order as items
    all_uuids: List[Optional[str]] = []
    mm_hashes: Dict[str, List[str]] = {}

    for modality, items in mm_items.items():
        modality_uuids = None
        if mm_uuids is not None and modality in mm_uuids:
            modality_uuids = mm_uuids[modality]
            if not isinstance(modality_uuids, list):
                modality_uuids = [modality_uuids]
            if len(modality_uuids) != len(items):
                raise ValueError(
                    f"UUID list length ({len(modality_uuids)}) doesn't match "
                    f"data items length ({len(items)}) for modality '{modality}'"
                )

        hashes = []
        for i, item in enumerate(items):
            uuid = modality_uuids[i] if modality_uuids else None
            if uuid is not None:
                # Hash UUID + content together for cache correctness
                hashes.append(_hash_item_with_uuid(item, uuid))
                all_uuids.append(uuid)  # Store original UUID
            else:
                # Fall back to content-only hashing
                hashes.append(_hash_item(item))
                all_uuids.append(None)

        mm_hashes[modality] = hashes

    # Return None for uuids if no UUIDs were provided at all
    return mm_hashes, all_uuids if mm_uuids is not None else None


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


def int32_to_hexdigest(int32_values: List[int]) -> str:
    """Convert 8 int32 values back to a 64-character hexadecimal digest.

    This is the inverse of hexdigest_to_int32.

    Args:
        int32_values: List of 8 signed int32 values

    Returns:
        64-character hexadecimal string representing the 32-byte hash
    """
    if len(int32_values) != 8:
        raise ValueError(f"Expected 8 int32 values, got {len(int32_values)}")

    result = []
    for value in int32_values:
        # Convert signed int32 back to unsigned
        if value < 0:
            value = value + 0x100000000
        # Format as 8 hex characters (zero-padded)
        result.append(f'{value:08x}')
    return ''.join(result)


def find_mm_token_lengths(
    mm_data: Dict[str, Any],
    input_processor: Any,
    *,
    multimodal_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[int]]:
    """Get the token lengths of each multimodal item.

    Returns the total token count for each multimodal item, including any special tokens
    (e.g., image_begin, image_end, image_break) that may be mixed with the actual
    multimodal content tokens. This mm_token_lengths represents the full chunk from beginning
    to end, not just pure image/video/audio tokens.

    """
    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    num_mm_tokens = {}

    mm_video_dict = (multimodal_data or {}).get("video") or {}
    video_grid_thw = mm_video_dict.get("video_grid_thw")

    for modality, items in mm_items.items():
        if not hasattr(input_processor, f"get_num_tokens_per_{modality}"):
            raise AttributeError(
                f"Input processor {type(input_processor).__name__} does not have 'get_num_tokens_per_{modality}' method required for multimodal hashing."
            )

        video_grid_thw_for_items = None
        if modality == "video" and video_grid_thw is not None:
            if len(video_grid_thw) == len(items):
                video_grid_thw_for_items = video_grid_thw
            else:
                logger.warning(
                    "find_mm_token_lengths: video_grid_thw row count "
                    f"({len(video_grid_thw)}) does not match number of "
                    f"videos in mm_data ({len(items)}); falling back to "
                    "per-item recompute without video_grid_thw.")

        modality_token_lengths = []
        for idx, item in enumerate(items):
            if modality == "image":
                num_tokens = input_processor.get_num_tokens_per_image(
                    image=item, )
                modality_token_lengths.append(num_tokens)
            elif modality == "video":
                if isinstance(item, tensorrt_llm.inputs.utils.VideoData):
                    item = item.frames
                assert isinstance(item, list), "Video must be a list of frames"
                call_kwargs = {"video": item}
                if video_grid_thw_for_items is not None:
                    # TODO(TRTLLM-11951): Replace this Qwen-VL-specific
                    # metadata wiring with a generic per-item processed
                    # metadata route. Keep this for now: Qwen3-VL needs the
                    # processor-produced video_grid_thw for correct video token
                    # counts.
                    call_kwargs["video_grid_thw"] = video_grid_thw_for_items[
                        idx]
                num_tokens = input_processor.get_num_tokens_per_video(
                    **call_kwargs)
                modality_token_lengths.append(num_tokens)
            elif modality == "audio":
                num_tokens = input_processor.get_num_tokens_per_audio(
                    audio=item)
                modality_token_lengths.append(num_tokens)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        num_mm_tokens[modality] = modality_token_lengths

    return num_mm_tokens  # flatten all mm instances to a single list


# Keys in py_multimodal_data that carry metadata (not vision/audio content).
# If py_multimodal_data has ONLY these keys, the request has no real MM
# payload (e.g. mrope-only warmup on an mrope-enabled model) and the
# check_mm_embed_cumsum_if_needed gate short-circuits.
_MM_METADATA_ONLY_KEYS = frozenset({
    "mrope_config",
    "multimodal_embed_mask_cumsum",
    "special_token_offsets",
    "layout_metadata",
})


def _has_mm_payload_keys(py_multimodal_data: Optional[dict]) -> bool:
    """True iff py_multimodal_data contains vision/video/audio content keys.

    Metadata-only payloads (`mrope_config` on mrope warmup,
    `multimodal_embed_mask_cumsum` alone, `special_token_offsets` alone,
    `layout_metadata`) return False — those don't carry real MM content
    that the model needs to fuse embeddings for.
    """
    if not py_multimodal_data:
        return False
    return bool(set(py_multimodal_data.keys()) - _MM_METADATA_ONLY_KEYS)


# TODO(TRTLLM-11951): fold this gate into MultimodalRuntimeData.__post_init__
# so new call sites cannot forget to invoke it.
def check_mm_embed_cumsum_if_needed(
    py_multimodal_data: Optional[dict],
    *,
    begin_compute: int,
    end_compute: int,
    prompt_len: int,
) -> None:
    """Raise iff chunked prefill or KV-cache reuse is in effect AND MM data is present without embed cumsum.

    Triggers on the two cases where the scheduler advances less than the
    whole prompt in one step:
      * `begin_compute > 0` — KV-cache reuse: a prefix was served from cache, OR
      * `end_compute < prompt_len` — chunked prefill: the scheduler split the
        remaining tokens across iterations.

    Both cases require `multimodal_embed_mask_cumsum` to derive per-chunk
    embedding counts in `MultimodalRuntimeData`. Full-prefill, no-reuse
    iterations don't: `MultimodalRuntimeData` stays `None` and
    `find_input_mm_embeds` handles the full payload.

    When the cumsum is missing outside the chunked-prefill / KV-reuse cases,
    log a one-shot warning via `logger.warning_once` and proceed.
    """
    assert 0 <= begin_compute <= end_compute <= prompt_len, (
        f"invalid window: {begin_compute}..{end_compute}/{prompt_len}")
    if not _has_mm_payload_keys(py_multimodal_data):
        return
    if py_multimodal_data.get("multimodal_embed_mask_cumsum") is not None:
        return

    is_chunked_or_reused = (begin_compute > 0) or (end_compute < prompt_len)
    mm_keys = set(py_multimodal_data.keys()) - _MM_METADATA_ONLY_KEYS

    if is_chunked_or_reused:
        raise ValueError(
            f"Request requires multimodal_embed_mask_cumsum for chunked prefill "
            f"or KV-cache reuse (begin_compute={begin_compute}, "
            f"end_compute={end_compute}, prompt_len={prompt_len}) but "
            f"py_multimodal_data has keys {mm_keys} with no cumsum. The input "
            f"processor may be missing a discriminator (override "
            f"get_mm_token_ids or ensure get_vocab_size "
            f"resolves).")

    logger.warning_once(
        "multimodal_embed_mask_cumsum missing on multimodal request (keys=%s); "
        "running without mask-aware accounting. This is fine for full-prefill "
        "iterations but will fail if this request is later chunked or reuses "
        "KV cache.",
        mm_keys,
        key="mm_embed_cumsum_missing_non_partial",
    )


def _as_cpu_tensor(
        input_ids: Union[torch.Tensor, List[int], np.ndarray]) -> torch.Tensor:
    """Coerce input_ids to a CPU torch.Tensor without copying when possible."""
    input_ids_tensor = torch.as_tensor(input_ids)
    if input_ids_tensor.device.type != "cpu":
        raise ValueError("prompt_token_ids must be CPU-resident when computing "
                         f"multimodal metadata, got {input_ids_tensor.device}.")
    return input_ids_tensor


def _compute_mm_masks(
    input_ids: torch.Tensor,
    vocab_size: Optional[int],
    mm_token_ids: Optional[torch.Tensor],
    mm_special_token_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute MM masks in a single pass.

    Returns `(mm_mask, embed_mask, special_mask)` where:
      - `mm_mask[i]` is True iff position i is any MM token (embed slot OR special)
      - `embed_mask[i]` is True iff position i is an embed slot (specials excluded)
      - `special_mask[i]` is True iff position i is a special (None when
        `mm_special_token_ids` is not provided)

    At least one of `vocab_size` or `mm_token_ids` must be provided; if
    both, `mm_token_ids` takes precedence (matching legacy behavior).

    Example: `input_ids=[1, 5, 5, 6, 5, 7, 2]`,
    `mm_token_ids=[5]`, `mm_special_token_ids=[6, 7]` gives
    `embed_mask=[F, T, T, F, T, F, F]` and
    `mm_mask=[F, T, T, T, T, T, F]`.

    Each call performs at most two full-sequence `isin` scans — one for
    regular MM tokens and one for specials — and reuses them to derive all
    three masks. Callers that need multiple masks should share one invocation
    rather than recomputing predicates.
    """
    if mm_token_ids is None and vocab_size is None:
        raise ValueError(
            "Provide either mm_token_ids or vocab_size to compute multimodal masks"
        )
    if mm_token_ids is not None and vocab_size is not None:
        logger.debug(
            "Both mm_token_ids and vocab_size are provided, using mm_token_ids and ignoring vocab_size"
        )

    if mm_special_token_ids is not None:
        mm_special_token_ids = mm_special_token_ids.to(device=input_ids.device,
                                                       dtype=input_ids.dtype)
        special_mask = torch.isin(input_ids, mm_special_token_ids)
    else:
        special_mask = None

    if mm_token_ids is not None:
        mm_token_ids = mm_token_ids.to(device=input_ids.device,
                                       dtype=input_ids.dtype)
        if mm_token_ids.ndim != 1:
            raise ValueError("mm_token_ids must be a 1D tensor")
        embed_mask = torch.isin(input_ids, mm_token_ids)
    else:
        embed_mask = input_ids >= vocab_size

    if special_mask is not None:
        embed_mask &= ~special_mask

    mm_mask = embed_mask if special_mask is None else embed_mask | special_mask
    return mm_mask, embed_mask, special_mask


def find_mm_token_item_runs(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    num_mm_tokens: List[int],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None,
    *,
    precomputed_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> MultimodalItemRuns:
    """Get exact prompt token runs for each multimodal item."""
    if precomputed_masks is None:
        input_ids_tensor = _as_cpu_tensor(input_ids)
        mm_mask, embed_mask, _ = _compute_mm_masks(input_ids_tensor, vocab_size,
                                                   mm_token_ids,
                                                   mm_special_token_ids)
    else:
        mm_mask, embed_mask = precomputed_masks
        if mm_mask.dtype != torch.bool or embed_mask.dtype != torch.bool:
            raise ValueError(
                "precomputed multimodal masks must be bool tensors")
        if mm_mask.device.type != "cpu" or embed_mask.device.type != "cpu":
            raise ValueError(
                "precomputed multimodal masks must be CPU-resident")
        if mm_mask.shape != embed_mask.shape:
            raise ValueError(
                "precomputed multimodal masks must have matching shapes")
    if not torch.any(mm_mask):
        return []

    mm_positions = torch.where(mm_mask)[0].tolist()
    expected_mm_tokens = sum(num_mm_tokens)
    if len(mm_positions) != expected_mm_tokens:
        raise ValueError(
            "Number of multimodal tokens does not match sum of all lengths: "
            f"found {len(mm_positions)}, expected {expected_mm_tokens}")

    mm_embed_flags = embed_mask[mm_mask].tolist()
    item_positions = []
    item_embed_flags = []
    current_position = 0
    for length in num_mm_tokens:
        item_positions.append(mm_positions[current_position:current_position +
                                           length])
        item_embed_flags.append(
            mm_embed_flags[current_position:current_position + length])
        current_position += length

    return _positions_to_item_runs(item_positions, item_embed_flags)


def _find_mm_special_token_offsets_from_masks(
    mm_mask: torch.Tensor,
    special_mask: Optional[torch.Tensor],
) -> List[int]:
    """Compute special-token offsets within the flattened multimodal token stream.

    Args:
        mm_mask: Boolean tensor; True for multimodal tokens.
        special_mask: Optional boolean tensor for special MM tokens.

    Returns:
        Indices within the flattened MM-token stream where special MM tokens appear.

    Example: if MM prompt positions are `[1, 2, 3, 4, 5]` and prompt
    positions `[3, 5]` are special, returns `[2, 4]`.
    """
    if not torch.any(mm_mask) or special_mask is None:
        return []

    mm_positions = torch.where(mm_mask)[0]
    return torch.where(special_mask[mm_positions])[0].tolist()
