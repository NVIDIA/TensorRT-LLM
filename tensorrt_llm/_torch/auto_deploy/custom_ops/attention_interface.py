"""Attention Interface to handle various attention operators and cache operations.

This module provides an interface between the high-level runtime and cache management system and
the low-level functional attention operators. The interface is designed to provide a homogeneous
object-oriented interface to the high-level runtime via the SequenceInfo dataclass. The SequenceInfo
is also responsible for functionalizing information about the sequence and pass it on the the
various attention interface. The AttentionDescriptor is the main interface to the attention operator
and operates on a purely functional paradigm that is compatible with the torch custom op system.

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple, Type, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch._ops import OpOverloadPacket
from torch.fx import Node
from torch.types import Number

from ...._utils import nvtx_range
from ..utils.logger import ad_logger

Constant = Union[int, float, str, None]


class PrepareMetadataHostCallable(Protocol):
    def __call__(self, **sequence_info_args: torch.Tensor) -> None: ...


class InputBuffer:
    """Manages contiguous memory buffers for efficient host-to-device transfers.

    This class consolidates multiple tensors into a single contiguous buffer on both
    host (pinned memory) and device. This enables efficient bulk transfers with a
    single async H2D copy instead of multiple small copies.

    The buffer layout places the truncatable tensor (typically cache_loc) last,
    allowing partial copies when the full buffer isn't needed.

    Usage:
        1. Create InputBuffer with tensor specifications (name, max_numel, dtype)
        2. Use store() to write data to the pinned host buffer
        3. Call copy_to_device() to perform a single async H2D transfer
        4. Access device tensors via get_view()
    """

    def __init__(self, tensor_specs: List[Tuple[str, int, torch.dtype]]):
        """Initialize the InputBuffer.

        Args:
            tensor_specs: Ordered list of (name, max_numel, dtype) tuples.
                         The last tensor is treated as truncatable during copy.
        """
        self._tensor_specs = {name: (numel, dtype) for name, numel, dtype in tensor_specs}
        self._tensor_order = [name for name, _, _ in tensor_specs]

        # Calculate offsets for each tensor (aligned to dtype's element size)
        self._offsets: Dict[str, int] = {}
        self._byte_sizes: Dict[str, int] = {}

        current_offset = 0
        for name, numel, dtype in tensor_specs:
            # Align to the tensor's element size for proper memory access
            alignment = dtype.itemsize
            aligned_offset = (current_offset + alignment - 1) // alignment * alignment
            byte_size = numel * dtype.itemsize
            self._offsets[name] = aligned_offset
            self._byte_sizes[name] = byte_size
            current_offset = aligned_offset + byte_size

        # Total buffer size
        self._total_bytes = current_offset

        # Allocate contiguous buffers (device buffer starts on default device, use to() to move)
        self._device_buffer = torch.empty(self._total_bytes, dtype=torch.uint8)
        self._host_buffer = torch.empty(
            self._total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )

        # Create persistent views into device and host buffers
        # Persistent views help us identify the arguments as static during graph capture.
        self._device_views = self._create_views(self._device_buffer)
        self._host_views = self._create_views(self._host_buffer)

        # Track current lengths for each tensor (for truncation optimization)
        self._current_lengths: Dict[str, int] = {name: 0 for name in self._tensor_order}

    def _create_views(self, buffer: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create views into the given buffer for each tensor."""
        views = {}
        for name in self._tensor_order:
            offset = self._offsets[name]
            byte_size = self._byte_sizes[name]
            _, dtype = self._tensor_specs[name]
            views[name] = buffer[offset : offset + byte_size].view(dtype)
        return views

    @property
    def tensor_names(self) -> List[str]:
        """Return the list of tensor names in buffer order."""
        return self._tensor_order.copy()

    @property
    def _truncatable_name(self) -> str:
        """Return the name of the truncatable tensor."""
        return self._tensor_order[-1]

    @property
    def total_bytes(self) -> int:
        """Total size of the buffer in bytes."""
        return self._total_bytes

    @property
    def device(self) -> torch.device:
        """Return the device of the device buffer."""
        return self._device_buffer.device

    def get_view(self, name: str) -> torch.Tensor:
        """Get the device tensor view for the specified name.

        Args:
            name: Name of the tensor.

        Returns:
            A view into the device buffer for the specified tensor.
        """
        return self._device_views[name]

    def get_view_at_current_length(self, name: str) -> torch.Tensor:
        """Get the device tensor view for the specified name at the current length.

        Args:
            name: Name of the tensor.

        Returns:
            A view into the device buffer for the specified tensor at the current length.
        """
        return self._device_views[name][: self._current_lengths[name]]

    def get_host_view(self, name: str) -> torch.Tensor:
        """Get the host tensor view for the specified name.

        Args:
            name: Name of the tensor.

        Returns:
            A view into the pinned host buffer for the specified tensor.
        """
        return self._host_views[name]

    def get_capacity(self, name: str) -> int:
        """Get the maximum number of elements for the specified tensor.

        Args:
            name: Name of the tensor.

        Returns:
            Maximum number of elements that can be stored.
        """
        numel, _ = self._tensor_specs[name]
        return numel

    def get_current_length(self, name: str) -> int:
        """Get the current stored length for the specified tensor.

        Args:
            name: Name of the tensor.

        Returns:
            Number of elements currently stored in the tensor.
        """
        return self._current_lengths[name]

    def store(
        self,
        name: str,
        data: List[Number],
        fill_value: Optional[Number] = None,
    ) -> int:
        """Store data into the host buffer.

        Args:
            name: Name of the tensor to store to.
            data: List of values to store.
            fill_value: Optional value to fill the entire tensor with before storing.
                       If None, only the provided data is written.

        Returns:
            Number of elements stored.
        """
        numel, dtype = self._tensor_specs[name]
        host_view = self.get_host_view(name)

        # Fill with default value if specified
        if fill_value is not None:
            host_view.fill_(fill_value)

        # Convert list to tensor and copy to host buffer
        length = len(data)
        assert length <= numel, f"Data too large for buffer '{name}': {length} > {numel}"

        temp_tensor = torch.tensor(data, dtype=dtype)
        host_view[:length].copy_(temp_tensor)

        self._current_lengths[name] = length
        return length

    def copy_to_device(self) -> None:
        """Copy from host buffer to device buffer.

        Uses the current length of the truncatable tensor (last in spec) to minimize
        transfer size. All tensors before the truncatable one are fully copied.
        """
        # Calculate bytes to copy based on truncatable tensor's current length
        truncatable_len = self._current_lengths[self._truncatable_name]
        truncatable_offset = self._offsets[self._truncatable_name]
        truncatable_dtype = self._tensor_specs[self._truncatable_name][1]
        copy_bytes = truncatable_offset + truncatable_len * truncatable_dtype.itemsize

        # Single async copy
        with nvtx_range("ad_input_buffer_h2d_copy"):
            self._device_buffer[:copy_bytes].copy_(
                self._host_buffer[:copy_bytes], non_blocking=True
            )

    def resize(self, name: str, new_capacity: int) -> None:
        """Resize a tensor's capacity.

        This operation is only supported for the last tensor in the buffer to avoid
        complex offset recalculations.

        Args:
            name: Name of the tensor to resize.
            new_capacity: New maximum number of elements for the tensor.
        """
        assert name == self._truncatable_name, (
            f"Can only resize the last tensor in the buffer ('{self._truncatable_name}'). "
            f"Attempted to resize '{name}'."
        )

        old_numel, dtype = self._tensor_specs[name]
        if new_capacity <= old_numel:
            return  # No need to resize if new capacity is smaller or equal

        # Update tensor specs
        self._tensor_specs[name] = (new_capacity, dtype)

        # Calculate new byte size for this tensor
        new_byte_size = new_capacity * dtype.itemsize
        self._byte_sizes[name] = new_byte_size

        # Update total bytes (offset stays the same since it's the last tensor)
        self._total_bytes = self._offsets[name] + new_byte_size

        # Resize device buffer in-place
        self._device_buffer.resize_(self._total_bytes)

        # Host buffer must be re-allocated to ensure we have pinned memory
        old_host_buffer = self._host_buffer
        self._host_buffer = torch.empty(
            self._total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._host_buffer[: old_host_buffer.numel()].copy_(old_host_buffer)
        del old_host_buffer

        # Recreate views after the update
        self._device_views = self._create_views(self._device_buffer)
        self._host_views = self._create_views(self._host_buffer)

    def to(self, *args, **kwargs) -> None:
        """Move the device buffer to a new device/dtype.

        Note: This recreates the device views after moving.
        """
        old_device = self._device_buffer.device
        self._device_buffer = self._device_buffer.to(*args, **kwargs)

        # Recreate views if device changed
        if old_device != self._device_buffer.device:
            self._device_views = self._create_views(self._device_buffer)


class CacheConfig(BaseModel):
    """Cache configuration for attention-related dtypes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    dtype: Optional[torch.dtype] = Field(default=None, description="KV cache dtype.")
    mamba_dtype: Optional[torch.dtype] = Field(default=None, description="Mamba cache dtype.")
    delta_dtype: Optional[torch.dtype] = Field(
        default=torch.float32, description="Delta cache dtype. Defaults to float32."
    )

    @field_validator("dtype", "mamba_dtype", "delta_dtype", mode="before")
    @classmethod
    def _coerce_dtype(cls, value):
        if value is None or isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            dtype = getattr(torch, value, None)
            assert isinstance(dtype, torch.dtype), f"Invalid {dtype=}"
            return dtype
        return value

    def __or__(self, other: "CacheConfig") -> "CacheConfig":
        """Combine two CacheConfig objects field-wise using Python's `or` semantics.

        For each field, selects the first non-None value between `self` and `other`.
        """
        if not isinstance(other, CacheConfig):
            raise NotImplementedError(f"Cannot combine CacheConfig with {type(other)}")
        merged_kwargs = {}
        for field_name in type(self).model_fields.keys():
            merged_kwargs[field_name] = getattr(self, field_name) or getattr(other, field_name)
        return CacheConfig(**merged_kwargs)


class SequenceInfo:
    """An interface to hold information about how the sequence is laid out and stored in cache.

    We assume the sequence + cache is laid out in the following way. Also note that we differentiate
    between arguments that are originally part of the model/graph and arguments that are needed for
    the attention operator when we switch to cached+flattened attention.

    ### ORIGINAL MODEL ARGUMENTS ###################################################################
    - input_ids: [id_0, ..., id_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total]. We use [b, 1] to denote generate-only batches.
    - position_ids: [pos_0, ..., pos_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total] indicating absolute position ids for every token
      in the input_ids sequence. We use [b, 1] to denote generate-only batches.

    NOTE: ``input_ids`` and ``position_ids`` are initially expected to be of shape [b, seq_len]
    before we switch to cached+flattened attention.

    ### EXTRA ARGUMENTS PROVIDED TO THE INTERFACE ##################################################
    Those are extra arguments that can be provided to the interface and they are stored as follows:
    - _extra_args: dictionary of extra arguments with currently active values.

    ### AVAILABLE ARGUMENTS TO BE ADDED BY THE TRANSFORMS IF NEEDED ################################
    - seq_len: [s_0, s_1, ..., s_{b-1}] such that s_total = sum(s_i)
      Describes how long each sequence is. For example,
      input_ids[:s_0] will correspond to sequence 0 in the batch and input_ids[s_0:s_1] will
      correspond to sequence 1 in the batch.
    - cu_seqlen: [0, s_0, s_0+s_1, ..., s_total]
      Cumulative sequence lengths of shape [b+1]. cu_seqlen[i+1] - cu_seqlen[i] gives the length
      of sequence i.
    - input_pos: [pos_0, ..., pos_{b-1}]
      Corresponds to the total number of tokens that have already been cached for each sequence
      in the batch (i.e., the starting position in the cache for new tokens).
    - pages_per_seq: [ps_0, ps_1, ..., ps_{b-1}] where ps_i is the number of pages allocated for
      sequence i. Note that, for example, cache_loc[sum(ps_0:ps_{i-1}):sum(ps_0:ps_i)] will
      correspond to the pages associated with sequence i in the batch.
    - cu_num_pages: [0, ps_0, ps_0+ps_1, ..., sum(ps_i)]
      Cumulative number of pages of shape [b+1]. cu_num_pages[i+1] - cu_num_pages[i] gives the
      number of pages for sequence i.
    - seq_len_with_cache: [pos_0+s_0, pos_1+s_1, ..., pos_{b-1}+s_{b-1}]
      Total sequence length including cached tokens for each sequence (input_pos + seq_len).
    - last_page_len: [lpl_0, lpl_1, ..., lpl_{b-1}]
      Number of valid tokens in the last page for each sequence. Computed as
      (seq_len_with_cache - 1) % page_size + 1.
    - slot_idx: [slot_0, slot_1, ..., slot_{b-1}]
      Corresponds to the slot index of each sequence in the batch.
    - use_initial_states: [bool_0, bool_1, ..., bool_{b-1}]
      Per-sequence boolean indicating whether initial states should be used (True if input_pos > 0).
    - batch_info: [num_prefill, num_prefill_tokens, num_decode]
      Batch metadata containing the number of prefill sequences, total prefill tokens, and number
      of decode sequences.
    - cache_loc: [c_0, c_1, ..., c_{np-1}] where np is total number of pages allocated to describe
      all sequences in the batch. Each value is a page index in the cache.
    - logits_gather_indices: [g_0, g_1, ..., g_{s_total-1}]
      Gather indices used by the gather_logits_before_lm_head custom op to gather logits before the LM head.
    - logits_gather_info: [num_tokens_to_gather, gather_required]. Info for the
      gather_logits_before_lm_head custom op to gather logits before the LM head.
    - _gather_idx: [g_0, g_1, ..., g_{s_total-1}]
      Gather indices used by the overlap scheduler to reorder input tokens.
    - _mask_scatter_indices: [m_0, m_1, ..., m_{s_total-1}]
      Mask scatter indices used by the overlap scheduler to scatter results back.

    NOTE: all tensors are also accessible as host tensors with the suffix "_host". For example,
    the tensor "batch_info" is accessible as "batch_info_host" on the host.

    ################################################################################################

    Here are a couple of notes to emphasize this notation:

    - The total number of allocated token space for sequence i is given by ps_i * page_size. This is
      the total number of tokens that can be cached for each sequence.

    - NOTE: It must hold that pos_i + s_i <= ps_i * page_size for all i in [0, b-1]. Moreover, it is
      the responsibility of the cache manager and/or runtime to ensure sufficient page allocation
      for each sequence.

    """

    def __init__(
        self,
        max_seq_len: int = 1,
        max_batch_size: int = 1,
        page_size: int = 0,
        max_num_tokens: Optional[int] = None,
        vocab_size_padded: Optional[int] = None,
    ):
        """Initialize the SequenceInfo object.

        Args:
            max_seq_len: corresponds to the maximum sequence length of the input sequence. It
                includes the tokens in the input sequence and the tokens generated by the model.
            max_batch_size: corresponds to the maximum number of sequences (or requests) that the
                model can process.
            page_size: corresponds to the page size of the cache. For an unpaged cache, the page
                size should be set to max_seq_len. Also note that two sequences in a batch can not
                share a page.
            max_num_tokens: corresponds to the maximum number of tokens that the model can process
                across all sequences in the batch. If a batch is composed of context-only requests
                of input sequence length ISL, then the maximum number of sequences possible in the
                batch is min (max_batch_size, max_num_tokens // ISL). Similarly, if a batch is
                composed of generate-only requests, then the maximum number of sequences possible in
                the batch is min (max_batch_size, max_num_tokens).
            vocab_size_padded: corresponds to the padded vocabulary size of the model.
        Returns:
            None
        """
        # set up basic attributes
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.page_size = page_size if page_size > 0 else max_seq_len
        self.vocab_size_padded = vocab_size_padded
        # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
        # (max_batch_size, max_seq_len) input in trtllm runtime.
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        max_seq_len_adjusted = self.max_seq_len + 1

        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/9883 clean up this hack
        self.max_state_slots = max_batch_size + 1

        # if the provided max_num_tokens is less than the max_batch_size * max_seq_len_adjusted,
        # we use the provided max_num_tokens. If max_num_tokens provided is more, we still use
        # max_batch_size * max_seq_len_adjusted since the extra tokens cannot be used.
        self.max_num_tokens = self.max_batch_size * max_seq_len_adjusted
        if max_num_tokens is not None and max_num_tokens > 0:
            self.max_num_tokens = min(self.max_num_tokens, max_num_tokens)

        # Num pages can not be less than max_batch_size.
        self._num_pages = max(
            self.max_batch_size,
            (self.max_num_tokens) // self.page_size  # floored number of pages
            + (self.max_num_tokens / self.max_batch_size % self.page_size > 0)  # check for overflow
            * self.max_batch_size,  # +1 page per sequence if overflow is required
        )
        # sanity check
        assert self.num_pages >= self.max_batch_size, "num_pages can't be less than max_batch_size"

        # cache_loc requires some special treatment due to block reuse. Note that the constraint for
        # cache_loc with block_reuse is as follows:
        # 0 <= cache_loc < num_pages
        # len(cache_loc) <= max_num_cache_loc_assignments
        max_num_cache_loc_assignments = (
            max_seq_len_adjusted // self.page_size + 1
        ) * self.max_batch_size

        # log parameters
        ad_logger.info(
            f"[SequenceInfo:] {self.max_seq_len=}, {self.max_batch_size=}, {self.page_size=}, "
            f"{self.max_num_tokens=} (inferred), {max_num_tokens=} (provided), {self.num_pages=}, "
            f"{max_num_cache_loc_assignments=}"
        )

        # indicator if extra args are activated that are needed for cached attention backends
        self._use_flattened_layout = False

        # TENSOR FIELDS ############################################################################
        # Define tensor specifications for the InputBuffer
        # Order matters: cache_loc is placed LAST for truncation optimization during H2D copy
        # Format: (name, max_numel, dtype)
        tensor_specs: List[Tuple[str, int, torch.dtype]] = [
            # TENSOR FIELDS FOR UNCACHED ATTENTION
            ("input_ids", self.max_num_tokens, torch.int),
            ("position_ids", self.max_num_tokens, torch.long),
            # TENSOR FIELDS FOR CACHED ATTENTION
            ("seq_len", self.max_batch_size, torch.int),
            ("cu_seqlen", self.max_batch_size + 1, torch.int),
            ("input_pos", self.max_batch_size, torch.int),
            ("pages_per_seq", self.max_batch_size, torch.int),
            ("cu_num_pages", self.max_batch_size + 1, torch.int),
            ("seq_len_with_cache", self.max_batch_size, torch.int),
            ("last_page_len", self.max_batch_size, torch.int),
            ("slot_idx", self.max_batch_size, torch.long),
            ("use_initial_states", self.max_batch_size, torch.bool),
            ("batch_info", 3, torch.int),
            ("logits_gather_indices", self.max_num_tokens, torch.long),
            ("logits_gather_info", 2, torch.int),
            # OTHER FIELDS WHERE WE NEED EFFICIENT HOST<>DEVICE TRANSFER
            ("_gather_idx", self.max_num_tokens, torch.int),
            ("_mask_scatter_indices", self.max_num_tokens, torch.int),
            # cache_loc is LAST for truncation optimization (it's the largest tensor)
            ("cache_loc", max_num_cache_loc_assignments, torch.int),
        ]

        # Create the InputBuffer that manages contiguous host and device memory
        # Starts on default device; use to() to move to target device
        self._input_buffer = InputBuffer(tensor_specs)
        self._available_args = set(self._input_buffer.tensor_names) | {
            f"{name}_host" for name in self._input_buffer.tensor_names
        }

        # Initialize args_list from tensor specs
        self._args_list: Dict[str, List[int]] = {
            name: [0] * numel for name, numel, _ in tensor_specs
        }

        self._active_args = ("input_ids", "position_ids")
        self._shapeable_args = ("input_ids", "position_ids", "input_ids_host", "position_ids_host")
        ############################################################################################

        # EXTRA TENSOR FIELDS ######################################################################
        self._extra_args: Dict[str, Optional[torch.Tensor]] = {}
        ############################################################################################

        # HOST PREPARE FOR ATTENTION FORWARD #######################################################
        self._host_prepare_functions: List[Tuple[PrepareMetadataHostCallable, List[str]]] = []

        # call reset once to set a consistent initial state
        self.reset()

    @property
    def device(self) -> torch.device:
        return self._input_buffer.device

    def _shape_for_forward(self, tnsr: torch.Tensor) -> torch.Tensor:
        """Shape the tensor for the forward pass based on the current attention mode.

        Args:
            tnsr: The tensor to shape assumed to be in shape [batch_size*seq_len, ...]

        Returns:
            The shaped tensor flattened or unflattened based on the current attention mode.
        """
        # check if we are still running uncached attention in which case we are also still
        # operate on unflattened tensors with explicit [batch_size, seq_len, ...] shape
        # generate-only batches are also formatted like this (i.e. [b, 1])
        if not self._use_flattened_layout or self.is_generate:
            bs = len(self.seq_len)
            sl = self.seq_len[0]
        # use [1,total_len] shape to indicate non-generate-only batch for cached attention
        else:
            bs, sl = 1, self.total_num_tokens

        # truncate to total tokens now, reshape, and return
        return tnsr[: self.total_num_tokens].view(bs, sl, *tnsr.shape[1:])

    def _get_arg(self, name: str) -> torch.Tensor:
        """Get the argument from the input buffer either on device or host."""
        if name.endswith("_host"):
            arg = self._input_buffer.get_host_view(name.replace("_host", ""))
        else:
            arg = self._input_buffer.get_view(name)
        return self._shape_for_forward(arg) if name in self._shapeable_args else arg

    def _named_args(self, include_extra_args: bool = True) -> Dict[str, torch.Tensor]:
        args = {k: self._get_arg(k) for k in self._active_args}

        # check other args to include
        if include_extra_args:
            args.update(self._extra_args)

        return args

    @property
    def available_args(self) -> Set[str]:
        """Return a list of available arguments."""
        return self._available_args

    @property
    def named_args(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of named arguments.

        These arguments contain all arguments that are managed by this interface and are required
        to run a model's forward pass including all extra arguments.

        Cached arguments are only included if the attention mode is cached to reflect that after
        switching to cached attention, the cached arguments are required for a forward pass.
        """
        return self._named_args(include_extra_args=True)

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return a tuple of arguments."""
        return tuple(self.named_args.values())

    @property
    def seq_len(self) -> List[int]:
        return self._args_list["seq_len"].copy()

    @property
    def input_pos(self) -> List[int]:
        return self._args_list["input_pos"].copy()

    @property
    def cache_loc(self) -> List[int]:
        return self._args_list["cache_loc"].copy()

    @property
    def pages_per_seq(self) -> List[int]:
        return self._args_list["pages_per_seq"].copy()

    @property
    def num_sequences(self) -> int:
        return len(self.seq_len)

    @property
    def total_num_tokens(self) -> int:
        return sum(self.seq_len)

    @property
    def is_generate(self) -> bool:
        return all(sl == 1 for sl in self.seq_len)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @num_pages.setter
    def num_pages(self, value):
        self._num_pages = value
        # Check if we need to resize cache_loc (it's the last tensor in the buffer)
        cache_loc_capacity = self._input_buffer.get_capacity("cache_loc")
        if value > cache_loc_capacity:
            ad_logger.info(
                f"Resizing cache_loc capacity from {cache_loc_capacity} to {value} "
                f"to accommodate num_pages={value}"
            )
            # Resize the input buffer (cache_loc is the last tensor, so this is supported)
            self._input_buffer.resize("cache_loc", value)
            # Also resize the args_list to match
            old_size = len(self._args_list["cache_loc"])
            self._args_list["cache_loc"].extend([0] * (value - old_size))

    @property
    def is_paged(self) -> bool:
        return self.page_size < self.max_seq_len

    @property
    def page_assignments(self) -> List[List[int]]:
        """Return the page assignments for each sequence."""
        return self._get_page_assignments(self.cache_loc, self.pages_per_seq)

    @staticmethod
    def _get_page_assignments(
        cache_locations: List[int], pages_per_sequence: List[int]
    ) -> List[List[int]]:
        """Get nested page assignments from cache locations and pages per sequence as list of lists.

        Args:
            cache_locations: A flat list of cache locations for each sequence ordered by sequence.
            pages_per_sequence: A list of number of pages per sequence.

        Returns:
            A list of page assignments for each sequence ordered by sequence.
            For example:
                cache_locations: [0, 4, 2]
                pages_per_sequence: [2, 1]
                --> returns [[0, 4], [2]]
        """
        return [
            c_loc_one_seq.tolist()
            for c_loc_one_seq in torch.split(torch.tensor(cache_locations), pages_per_sequence)
        ]

    @staticmethod
    def _get_cache_locations_and_pages_per_sequence(
        page_assignments: List[List[int]],
    ) -> Tuple[List[int], List[int]]:
        """Get cache locations and pages per sequence from nested page assignments (lists of lists).

        Args:
            page_assignments: A list of page assignments for each sequence ordered by sequence.
        Returns:
            A tuple of:
                cache_locations: A flat list of cache locations for each sequence ordered by sequence.
                pages_per_sequence: A list of number of pages per sequence.

        Example:
            page_assignments: [[0, 4], [2]]
            --> returns ([0, 4, 2], [2, 1])

        """
        cache_loc_flat = [p_idx for pages in page_assignments for p_idx in pages]
        pages_per_seq = [len(p) for p in page_assignments]
        return cache_loc_flat, pages_per_seq

    def activate_arg(self, arg_name: str) -> bool:
        """Activate a desired argument.

        The first time this function is called we will also switch to the flattened input layout.

        Returns:
            True if the argument was activated, False if already activated.
        """
        assert arg_name in self.available_args, f"{arg_name=} not found in {self.available_args}"
        self._use_flattened_layout = True
        if arg_name not in self._active_args:
            self._active_args += (arg_name,)
            return True
        return False

    def to(self, *args, **kwargs) -> None:
        # Move the InputBuffer (which recreates views automatically)
        self._input_buffer.to(*args, **kwargs)

        # Move extra args
        for k, v in self._extra_args.items():
            if v is not None:
                self._extra_args[k] = v.to(*args, **kwargs)

    def set_example_sequence(
        self,
        input_ids: Optional[Sequence[Sequence[int]]] = None,
        position_ids: Optional[Sequence[Sequence[int]]] = None,
        **extra_args,
    ) -> None:
        """Set an example sequence useful for testing and export purposes without cache history."""
        # use a best guess default for input_ids if not provided
        if input_ids is None:
            bs, seq_len = min(2, self.max_batch_size), min(4, self.max_seq_len)
            input_ids = torch.ones(bs, seq_len, dtype=torch.int).tolist()

        # figure out page assignments
        pages_per_seq = [
            len(ids_one_seq) // self.page_size + (len(ids_one_seq) % self.page_size > 0)
            for ids_one_seq in input_ids
        ]
        cache_loc = list(range(sum(pages_per_seq)))

        # vanilla slot indices
        slot_idx = list(range(len(input_ids)))

        self.nest_sequences(
            input_ids,
            position_ids,  # will be auto-inferred if None
            input_pos=0,  # no cache history
            cache_loc=cache_loc,  # vanilla page assignments
            pages_per_seq=pages_per_seq,  # vanilla page assignments
            slot_idx=slot_idx,  # vanilla slot indices
            **extra_args,
        )

    def set_max_num_tokens_sample(self) -> None:
        """Set an example sequence with max_num_tokens."""
        # TODO (lucaslie): understand what this implies for extra arguments
        seq_len = self.max_num_tokens // self.max_batch_size
        input_ids = torch.ones(self.max_batch_size, seq_len, dtype=torch.int).tolist()
        self.set_example_sequence(input_ids)

    def set_generate_only_batch(self, batch_size: Optional[int] = None) -> None:
        """Set an example sequence for generate-only batch."""
        batch_size = batch_size or self.max_batch_size
        self.set_example_sequence(
            [[1]] * batch_size,
            logits_gather_info=[batch_size, 0],
        )

    def reset(self) -> None:
        """Reset the sequence information.

        After reset the sequence information should correspond to a "generate-only" batch of
        sequences (b, s==1) without cache history.
        """
        self.set_generate_only_batch()

    @staticmethod
    def _flatten(nested_seqs: Sequence[Sequence[int]]) -> List[int]:
        return [
            val
            for lst in nested_seqs
            for val in (lst.detach().tolist() if isinstance(lst, torch.Tensor) else lst)
        ]

    def _store_arg(
        self,
        name: str,
        tnsr_like: List[Number],
        reset_val: Optional[Number] = None,
        force_copy: bool = False,
    ) -> None:
        """Store the argument into the pinned host buffer for later batch transfer to device.

        The data is stored in the host-side pinned memory buffer managed by InputBuffer.
        The actual H2D transfer happens in a single batch at the end of nest_sequences().

        Args:
            name: Name of the argument to store.
            tnsr_like: List of values to store.
            reset_val: Value to reset/fill the tensor with before writing data.
            force_copy: Whether to force immediate copy to device (for use outside nest_sequences).
        """
        with nvtx_range(f"ad_store_on_host_seq_info_arg_{name}"):
            # Always store list object for Python access
            self._args_list[name] = tnsr_like.copy()

            # Only store to buffer when the argument is active or force_copy is True
            if not (name in self._active_args or f"{name}_host" in self._active_args or force_copy):
                return

            # Store to the InputBuffer's pinned host memory
            self._input_buffer.store(name, tnsr_like, fill_value=reset_val)

    def _store_extra_arg(
        self, name: str, tnsr_like: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]
    ) -> None:
        with nvtx_range(f"ad_store_extra_arg_{name}"):
            if tnsr_like is not None:
                if not isinstance(tnsr_like, torch.Tensor):
                    if len(tnsr_like) > 1:
                        tnsr_like = torch.cat(tnsr_like)
                    else:
                        tnsr_like = tnsr_like[0]
                self._extra_args[name] = tnsr_like.to(self.device, non_blocking=True)
            else:
                self._extra_args[name] = None

    @nvtx_range("ad_get_unique_value")
    def _get_unique_value(self, occupied: Set[int], max_val: int) -> int:
        """Get un unoccupied value from the range indicated by max_val."""
        # Return the smallest free value; fall back to 0 if none
        for candidate in range(max_val):
            if candidate not in occupied:
                return candidate
        return 0

    @nvtx_range("ad_nest_sequences")
    def nest_sequences(
        self,
        input_ids: Sequence[Sequence[int]],
        position_ids: Optional[Sequence[Sequence[int]]] = None,
        seq_len: Optional[Sequence[int]] = None,
        input_pos: Optional[Union[Sequence[int], int]] = None,
        batch_info: Optional[Sequence[int]] = None,
        cu_seqlen: Optional[Sequence[int]] = None,
        cache_loc: Optional[Sequence[int]] = None,
        pages_per_seq: Optional[Sequence[int]] = None,
        cu_num_pages: Optional[Sequence[int]] = None,
        seq_len_with_cache: Optional[Sequence[int]] = None,
        last_page_len: Optional[Sequence[int]] = None,
        slot_idx: Optional[Sequence[int]] = None,
        use_initial_states: Optional[Sequence[bool]] = None,
        logits_gather_indices: Optional[Sequence[int]] = None,
        logits_gather_info: Optional[Sequence[int]] = None,
        _gather_idx: Optional[Sequence[int]] = None,
        _mask_scatter_indices: Optional[Sequence[int]] = None,
        **extra_args: Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> None:
        """Create and store sequence information for the next forward pass.

        Args:
            input_ids: List of sequences of input_ids.
            position_ids: List of sequences of position_ids for each token. If None, auto-inferred
                from input_pos and seq_len.
            seq_len: List of sequence lengths for each sequence. If None, inferred from input_ids.
            input_pos: Absolute starting position in the cache for each sequence. Can be a single
                int (applied to all sequences) or a list of ints.
            batch_info: Batch metadata as [num_prefill, num_prefill_tokens, num_decode]. If None,
                auto-computed from seq_len.
            cu_seqlen: Cumulative sequence lengths of shape [b+1]. If None, auto-computed from
                seq_len.
            cache_loc: Flat list of page indices for all sequences. Must be provided together with
                pages_per_seq.
            pages_per_seq: Number of pages allocated per sequence. Must be provided together with
                cache_loc.
            cu_num_pages: Cumulative number of pages of shape [b+1]. If None, auto-computed from
                pages_per_seq.
            seq_len_with_cache: Total sequence length including cached tokens (input_pos + seq_len)
                for each sequence. If None, auto-computed.
            last_page_len: Number of valid tokens in the last page for each sequence. If None,
                auto-computed from seq_len_with_cache.
            slot_idx: Slot index for each sequence in the batch.
            use_initial_states: Per-sequence boolean indicating if the initial states should be
                used. If None, auto-computed as (input_pos > 0).
            logits_gather_indices: Gather indices for the logits before/after the LM head.
            logits_gather_info: Info list containing [num_tokens_to_gather, gather_required].
            _gather_idx: Gather indices for the overlap scheduler to reorder input tokens.
            _mask_scatter_indices: Mask scatter indices for the overlap scheduler.
            extra_args: Extra arguments to be stored in the interface.

        This i/f will ensure that all sequence info args are updated accordingly. Reset values are
        chosen as "neutral" values so that for cases like rounding up batch sizes for cudagraph we
        only write to unused buffers/caches.
        """
        ### UPDATE SEQUENCE LENGTH AND INPUT POSITION FIRST SINCE IT'S USED FOR OTHER UPDATES ######
        if seq_len is None:
            seq_len = [len(ids) for ids in input_ids]
        self._store_arg("seq_len", seq_len)

        # check for updated input_pos (i.e. cache start position)
        if input_pos is not None:
            self._store_arg(
                "input_pos",
                [input_pos] * self.num_sequences if isinstance(input_pos, int) else input_pos,
            )

        ### UPDATE MAIN INPUTS #####################################################################
        # set new input_ids and make sure to flatten it
        self._store_arg("input_ids", self._flatten(input_ids))

        # set new position_ids and make sure to flatten it
        if position_ids is None:
            position_ids = [
                [num for num in range(in_pos, in_pos + seq_len)]
                for in_pos, seq_len in zip(self.input_pos, self.seq_len)
            ]
        self._store_arg("position_ids", self._flatten(position_ids))

        ### UPDATE OTHER (DERIVATIVE) METADATA #####################################################
        # check for updated batch_info_tensor
        if batch_info is None:
            num_prefill = sum(s_l > 1 for s_l in seq_len)
            num_prefill_tokens = sum(s_l for s_l in seq_len if s_l > 1)
            num_decode = len(seq_len) - num_prefill
            batch_info = [num_prefill, num_prefill_tokens, num_decode]
        self._store_arg("batch_info", batch_info)

        if cu_seqlen is None:
            cu_seqlen = torch.zeros(len(seq_len) + 1, dtype=torch.int)
            cu_seqlen[1:] = torch.cumsum(torch.tensor(seq_len), dim=0)
            cu_seqlen = cu_seqlen.tolist()
        self._store_arg("cu_seqlen", cu_seqlen)

        # check for updated page_assignments
        assert (cache_loc is None) == (pages_per_seq is None), (
            "cache_loc and pages_per_seq must beeither both None or both set"
        )
        if cache_loc is not None and pages_per_seq is not None:
            self._store_arg("cache_loc", cache_loc)
            self._store_arg("pages_per_seq", pages_per_seq)

        # update cumulative number of pages
        if cu_num_pages is None:
            pages_per_seq = self.pages_per_seq
            cu_num_pages = torch.zeros(len(pages_per_seq) + 1, dtype=torch.int)
            cu_num_pages[1:] = torch.cumsum(torch.tensor(pages_per_seq), dim=0)
            cu_num_pages = cu_num_pages.tolist()
        self._store_arg("cu_num_pages", cu_num_pages)

        # update sequence length with cache
        if seq_len_with_cache is None:
            seq_len_with_cache = [i_p + s_l for i_p, s_l in zip(self.input_pos, self.seq_len)]
        self._store_arg("seq_len_with_cache", seq_len_with_cache)

        # update last page length
        if last_page_len is None:
            last_page_len = [(slwc - 1) % self.page_size + 1 for slwc in seq_len_with_cache]
        self._store_arg("last_page_len", last_page_len)

        # check for updated slot_idx
        if slot_idx is not None:
            self._store_arg("slot_idx", slot_idx)

        # check for updated use_initial_states
        if use_initial_states is None:
            use_initial_states = [i_p > 0 for i_p in self.input_pos]
        self._store_arg("use_initial_states", use_initial_states)

        # check for updated logits_gather_indices
        if logits_gather_indices is None:
            # default is to gather all logits
            logits_gather_indices = list(range(self.total_num_tokens))
        self._store_arg("logits_gather_indices", logits_gather_indices, force_copy=True)

        # check for updated logits_gather_info
        if logits_gather_info is None:
            logits_gather_info = [len(logits_gather_indices), 1]
        self._store_arg("logits_gather_info", logits_gather_info, force_copy=True)

        ### UPDATE OVERLAP SCHEDULER METADATA ######################################################
        # check for updated _gather_idx
        if _gather_idx is not None:
            self._store_arg("_gather_idx", _gather_idx, force_copy=True)

        # check for updated _mask_scatter_indices
        if _mask_scatter_indices is not None:
            self._store_arg("_mask_scatter_indices", _mask_scatter_indices, force_copy=True)

        ### UPDATE EXTRA INPUTS ####################################################################
        self._extra_args = {}
        for key, value in extra_args.items():
            self._store_extra_arg(key, value)

        ### BATCH COPY TO DEVICE ###################################################################
        # Perform a single async H2D copy for all device tensors
        # The copy is truncated at the end of cache_loc to minimize transfer size
        self._input_buffer.copy_to_device()

    @nvtx_range("ad_rescatter_input_ids")
    def rescatter_input_ids(self, ungathered_input_ids: torch.Tensor):
        """Re-scatter the provided ungathered input ids into the input_ids tensor.

        Args:
            ungathered_input_ids: The input ids on the device from which to gather using the stored
                gather and mask scatter indices.

        Returns:
            None

        This function will assume that we are in a generate-only batch.
        """
        # retrieve input_ids and gather_ids on device
        input_ids_device = self._input_buffer.get_view_at_current_length("input_ids")
        gather_ids_device = self._input_buffer.get_view_at_current_length("_gather_idx")
        mask_scatter_indices_device = self._input_buffer.get_view_at_current_length(
            "_mask_scatter_indices"
        )

        torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
            ungathered_input_ids, gather_ids_device, mask_scatter_indices_device, input_ids_device
        )

    # TODO: remove once https://github.com/NVIDIA/TensorRT-LLM/issues/9878 is fixed and
    # logits gather is enabled by default (only keep squeeze_logits)
    @nvtx_range("ad_maybe_gather_logits")
    def maybe_gather_and_squeeze_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Maybe gather the logits if logits have not been gathered yet."""
        num_tokens = logits.shape[0] * logits.shape[1]
        num_tokens_to_gather, gather_required = self._get_arg("logits_gather_info_host").tolist()
        if gather_required and num_tokens_to_gather < num_tokens:
            logits = torch.ops.auto_deploy.gather_logits_before_lm_head(
                logits,
                self._get_arg("logits_gather_indices"),
                self._get_arg("logits_gather_info_host"),
            )
        return logits.squeeze(int(self.is_generate))

    @nvtx_range("ad_unnest_sequences")
    def unnest_sequences(self, t_nested: torch.Tensor) -> List[torch.Tensor]:
        t_squeezed = t_nested.squeeze(int(self.is_generate))
        return list(torch.split(t_squeezed, self.seq_len))

    def register_host_prepare_for_attention_forward(
        self, host_function: PrepareMetadataHostCallable, args: List[str]
    ):
        self._host_prepare_functions.append((host_function, args))

    def run_host_prepare_for_attention_forward(self) -> None:
        for host_function, args in self._host_prepare_functions:
            host_function(**{arg: self._get_arg(arg) for arg in args})


class MHACallable(Protocol):
    def __call__(
        self,
        *qkv_metadata_and_caches: Union[torch.Tensor, Constant],
    ) -> torch.Tensor: ...


class PrepareMetadataCallable(Protocol):
    def __call__(
        self, *sequence_info_args_and_constants: Union[torch.Tensor, Constant]
    ) -> List[torch.Tensor]: ...


class GetCacheCallable(Protocol):
    def __call__(self, sequence_info: SequenceInfo) -> torch.Tensor: ...


class GetBufferCallable(GetCacheCallable):
    pass


CacheInitializerDict = Dict[str, GetCacheCallable]
BufferInitializerDict = Dict[str, GetBufferCallable]
AttentionLayout = Literal["bsnd", "bnsd"]


class AttentionDescriptor(ABC):
    """An interface to define a functional attention operator.

    The main logic is contained with the actual attention op as well as the prepare_metadata op. The
    prepare_metadata op is responsible for converting the standardized sequence info into metadata
    specific to the attention op.
    """

    @classmethod
    @abstractmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""

    @classmethod
    @abstractmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the source op and the cached attention op."""

    @classmethod
    @abstractmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""

    @classmethod
    @abstractmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""

    @classmethod
    @abstractmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op .

        The attention_op should follow the below signature:

        ```
        def attention_op(
            *qkv,       # list of tensors corresponding to Q, K, V as in source attention op
            *meta_std,  # standard metadata fields identified by matching arg names!
            *meta_extra,# metadata about the sequences as returned by the prepare_metadata op
            *caches,    # contains layer-specific caches per provided cache initializers
            *buffers,   # global buffers used by the attention op as provided by buffer initializers
            *constants, # basic arguments (int, float, str, None) added as CONSTANTS in the graph
        ) -> torch.Tensor: ...
        ```

        **Note that the attention op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**

        **Note that the `qkv` tuple should be consistent across both the cached attention
        op and the source attention op that it is replacing.**

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments that are expected by the attention op."""
        raise NotImplementedError

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Get the prepare_metadata op.

        The prepare_metadata op should follow the below signature:

        ```
        def prepare_extra_metadata(
            *desired_graph_inputs,  # matched by arg names in the signature of the prepare_metadata op
            *constant_inputs, # as returned by this function
        ) -> List[torch.Tensor]: ...
        ```
        The metadata should contain all necessary extra global information required for the
        underlying attention op to process the input sequence and the returned list of tensors will
        be passed as additional arguments to each invocation of the attention op in the graph.

        This may not be needed for all attention ops if the standard metadata is sufficient.

        prepare_metadata is called once at the beginning of the forward pass for each attention op
        detected in the graph.

        **Note that the prepare_metadata op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**

        Returns:
            - prepare_metadata_op: The prepare_metadata op callable.
            - num_meta_out: The number of extra metadata tensors to return.
            - const_args: A list of constant arguments to pass to the prepare_metadata op.
        """
        return None, 0, []

    @classmethod
    @abstractmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize the caches.

        The key corresponds to the argument name used in the attention op signature. The function
        key doesn't need to be unique across multiple attention nodes in the graph. The key used to
        describe the cache in the graph will be patched with the attention node index to ensure
        uniqueness.

        ``get_cache_initializers`` will be called *once* during cache initialization and before
        the initial forward pass for each attention op detected in the graph. The caches will be
        managed by the global CacheManager and passed back to the attention op during the forward
        pass.

        If the cache initializer requires information about the attention op, it can retrieve
        the necessary information from the source attention node and cache config.
        """

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize buffers.

        The key corresponds to the buffer name used in the graph module and will **not**
        be patched unlike a cache key. Hence, it is a **global** key that is shared across all
        attention ops in the model much like a regular buffer in an nn.Module. That means if this
        i/f is called for multiple attention ops, the same buffer will be shared across all of them
        if this function provides the same key multiple times.

        Buffers are initialize *once* after the model initialization and before the initial forward
        pass for each attention op detected in the graph. The buffer will be managed by the global
        CacheManager and passed back to the attention op during the forward pass.

        If the buffer initializer requires information about the attention op, it can retrieve
        the necessary information from the source attention node.
        """
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Provide a list of constant arguments to be passed to the attention op.

        The constant arguments are passed to the attention op as additional arguments after the
        caches and buffers. The constants are expected to be of type int, float, str, or None.
        """
        return []

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Get function that performs host-side prep for the forward pass for the attention op.

        This method is responsible for preparing the attention op for the forward pass.
        This function is not expected to be graph capturable or compatible with cuda graphs. It can
        use any argument from the SequenceInfo interface as input argument to its function.
        """
        return None


class AttentionRegistry:
    """A simple registry to look up different attention implementations."""

    _attention_registry: Dict[str, Type["AttentionDescriptor"]] = {}

    @classmethod
    def register(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        def decorator(attention_cls: Type["AttentionDescriptor"]):
            assert kernel_source not in cls._attention_registry, (
                f"Attention source {kernel_source} already registered."
            )
            cls._attention_registry[kernel_source] = attention_cls
            return attention_cls

        return decorator

    @classmethod
    def get(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        assert cls.has(kernel_source), f"Attention source {kernel_source} not registered."
        return cls._attention_registry[kernel_source]

    @classmethod
    def has(cls, kernel_source: str) -> bool:
        return kernel_source in cls._attention_registry
