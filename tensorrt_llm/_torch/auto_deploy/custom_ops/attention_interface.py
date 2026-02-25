# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention Interface to handle various attention operators and cache operations.

This module provides an interface between the high-level runtime and cache management system and
the low-level functional attention operators. The interface is designed to provide a homogeneous
object-oriented interface to the high-level runtime via the SequenceInfo dataclass. The SequenceInfo
is also responsible for functionalizing information about the sequence and pass it on the the
various attention interface. The AttentionDescriptor is the main interface to the attention operator
and operates on a purely functional paradigm that is compatible with the torch custom op system.

"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple, Type, Union

import numpy as np
import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node
from torch.types import Number

from tensorrt_llm.llmapi.llm_args import KvCacheConfig

from ...._utils import nvtx_range, str_dtype_to_torch
from ..utils.logger import ad_logger

Constant = Union[int, float, str, None]

# Torch dtype → numpy dtype for fast list-to-tensor conversion.
# numpy's list→array conversion is ~2-3x faster than torch.tensor(list) for large lists.
_TORCH_TO_NUMPY_DTYPE: Dict[torch.dtype, np.dtype] = {
    torch.int: np.int32,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.double: np.float64,
    torch.float16: np.float16,
    torch.bool: np.bool_,
}


def _list_to_tensor(data: list, dtype: torch.dtype) -> torch.Tensor:
    """Convert a Python list to a tensor, using numpy for speed."""
    np_dtype = _TORCH_TO_NUMPY_DTYPE.get(dtype)
    if np_dtype is not None:
        return torch.from_numpy(np.array(data, dtype=np_dtype))
    return torch.tensor(data, dtype=dtype)


class PrepareMetadataHostCallable(Protocol):
    def __call__(self, **sequence_info_args: torch.Tensor) -> None: ...


class InputBuffer:
    """Manages memory buffers for efficient host-to-device transfers.

    Supports two categories of tensors:

    - **Contiguous tensors** (default): packed into a single contiguous buffer on both
      host (pinned) and device. Copied in one bulk async H2D transfer.
    - **Truncatable tensors** (``truncatable=True`` in spec): each gets its own separate
      host+device buffer pair, copied independently with truncation to actual length.
      Use this for large, variable-length tensors (e.g., ``cache_loc``) to avoid
      copying unused capacity.

    Usage:
        1. Create InputBuffer with tensor specifications
        2. Use store() to write data to the pinned host buffer
        3. Call copy_to_device() to perform async H2D transfers
        4. Access device tensors via get_view()
    """

    def __init__(self, tensor_specs: List[Tuple]):
        """Initialize the InputBuffer.

        Args:
            tensor_specs: Ordered list of tensor specs. Each element is either:
                - ``(name, max_numel, dtype)`` for contiguous tensors (default)
                - ``(name, max_numel, dtype, True)`` for truncatable tensors
        """
        # Parse specs into canonical form: (name, numel, dtype, truncatable)
        parsed = []
        for spec in tensor_specs:
            if len(spec) == 4:
                name, numel, dtype, truncatable = spec
            else:
                name, numel, dtype = spec
                truncatable = False
            parsed.append((name, numel, dtype, truncatable))

        self._tensor_specs: Dict[str, Tuple[int, torch.dtype]] = {
            name: (numel, dtype) for name, numel, dtype, _ in parsed
        }
        self._tensor_order = [name for name, _, _, _ in parsed]
        self._truncatable_names: Set[str] = {name for name, _, _, t in parsed if t}
        self._contiguous_names = [name for name, _, _, t in parsed if not t]

        # Track current lengths for each tensor (for truncation optimization)
        self._current_lengths: Dict[str, int] = {name: 0 for name in self._tensor_order}

        # === CONTIGUOUS BUFFER (small, fixed-size tensors) ===
        self._offsets: Dict[str, int] = {}
        self._byte_sizes: Dict[str, int] = {}

        current_offset = 0
        for name in self._contiguous_names:
            numel, dtype = self._tensor_specs[name]
            alignment = dtype.itemsize
            aligned_offset = (current_offset + alignment - 1) // alignment * alignment
            byte_size = numel * dtype.itemsize
            self._offsets[name] = aligned_offset
            self._byte_sizes[name] = byte_size
            current_offset = aligned_offset + byte_size

        self._total_bytes = current_offset

        if self._total_bytes > 0:
            self._device_buffer = torch.empty(self._total_bytes, dtype=torch.uint8)
            self._host_buffer = torch.empty(
                self._total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        else:
            self._device_buffer = torch.empty(0, dtype=torch.uint8)
            self._host_buffer = torch.empty(0, dtype=torch.uint8, device="cpu")

        # Create persistent views into contiguous buffers
        self._device_views: Dict[str, torch.Tensor] = {}
        self._host_views: Dict[str, torch.Tensor] = {}
        self._create_contiguous_views()

        # === TRUNCATABLE BUFFERS (large, variable-length tensors) ===
        self._trunc_device_bufs: Dict[str, torch.Tensor] = {}
        self._trunc_host_bufs: Dict[str, torch.Tensor] = {}
        for name in self._truncatable_names:
            numel, dtype = self._tensor_specs[name]
            byte_size = numel * dtype.itemsize
            self._trunc_device_bufs[name] = torch.empty(byte_size, dtype=torch.uint8)
            self._trunc_host_bufs[name] = torch.empty(
                byte_size, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            # Create typed views
            self._device_views[name] = self._trunc_device_bufs[name].view(dtype)
            self._host_views[name] = self._trunc_host_bufs[name].view(dtype)

    def _create_contiguous_views(self) -> None:
        """Create typed views into the contiguous host and device buffers."""
        for name in self._contiguous_names:
            offset = self._offsets[name]
            byte_size = self._byte_sizes[name]
            _, dtype = self._tensor_specs[name]
            self._device_views[name] = self._device_buffer[offset : offset + byte_size].view(dtype)
            self._host_views[name] = self._host_buffer[offset : offset + byte_size].view(dtype)

    @property
    def tensor_names(self) -> List[str]:
        """Return the list of tensor names in spec order."""
        return self._tensor_order.copy()

    @property
    def total_bytes(self) -> int:
        """Total size of the contiguous buffer in bytes."""
        return self._total_bytes

    @property
    def device(self) -> torch.device:
        """Return the device of the device buffer."""
        return self._device_buffer.device

    def get_view(self, name: str) -> torch.Tensor:
        """Get the device tensor view for the specified name."""
        return self._device_views[name]

    def get_view_at_current_length(self, name: str) -> torch.Tensor:
        """Get the device tensor view truncated to the current stored length."""
        return self._device_views[name][: self._current_lengths[name]]

    def get_host_view(self, name: str) -> torch.Tensor:
        """Get the host tensor view for the specified name."""
        return self._host_views[name]

    def get_capacity(self, name: str) -> int:
        """Get the maximum number of elements for the specified tensor."""
        numel, _ = self._tensor_specs[name]
        return numel

    def get_current_length(self, name: str) -> int:
        """Get the current stored length for the specified tensor."""
        return self._current_lengths[name]

    def store(
        self,
        name: str,
        data: torch.Tensor,
        fill_value: Optional[Number] = None,
    ) -> int:
        """Store a tensor into the pinned host buffer.

        Args:
            name: Name of the tensor to store to.
            data: 1-D torch.Tensor to store.
            fill_value: Optional value to fill the entire buffer with before storing.

        Returns:
            Number of elements stored.
        """
        numel, dtype = self._tensor_specs[name]
        host_view = self.get_host_view(name)

        if fill_value is not None:
            host_view.fill_(fill_value)

        length = data.numel()
        assert length <= numel, f"Data too large for buffer '{name}': {length} > {numel}"
        # Use numpy for the memcpy into pinned memory — avoids torch dispatcher overhead
        dst = host_view[:length].numpy()
        src = (data if data.dtype == dtype else data.to(dtype)).numpy()
        np.copyto(dst, src)

        self._current_lengths[name] = length
        return length

    def copy_to_device(self) -> None:
        """Copy from host buffers to device buffers.

        Contiguous tensors are copied in a single bulk transfer.
        Truncatable tensors are each copied independently, truncated to actual length.
        """
        with nvtx_range("ad_input_buffer_h2d_copy"):
            # Copy contiguous buffer in one shot
            if self._total_bytes > 0:
                self._device_buffer[: self._total_bytes].copy_(
                    self._host_buffer[: self._total_bytes], non_blocking=True
                )

            # Copy each truncatable tensor independently, truncated to current length
            for name in self._truncatable_names:
                length = self._current_lengths[name]
                if length > 0:
                    _, dtype = self._tensor_specs[name]
                    copy_bytes = length * dtype.itemsize
                    self._trunc_device_bufs[name][:copy_bytes].copy_(
                        self._trunc_host_bufs[name][:copy_bytes], non_blocking=True
                    )

    def resize(self, name: str, new_capacity: int) -> None:
        """Resize a truncatable tensor's capacity.

        Only truncatable tensors can be resized (they have independent buffers).

        Args:
            name: Name of the tensor to resize.
            new_capacity: New maximum number of elements for the tensor.
        """
        assert name in self._truncatable_names, (
            f"Can only resize truncatable tensors. '{name}' is not truncatable. "
            f"Truncatable tensors: {self._truncatable_names}"
        )

        old_numel, dtype = self._tensor_specs[name]
        if new_capacity <= old_numel:
            return

        self._tensor_specs[name] = (new_capacity, dtype)
        new_byte_size = new_capacity * dtype.itemsize

        # Resize device buffer in-place
        self._trunc_device_bufs[name].resize_(new_byte_size)

        # Host buffer must be re-allocated for pinned memory
        old_host = self._trunc_host_bufs[name]
        self._trunc_host_bufs[name] = torch.empty(
            new_byte_size, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._trunc_host_bufs[name][: old_host.numel()].copy_(old_host)
        del old_host

        # Recreate typed views
        self._device_views[name] = self._trunc_device_bufs[name].view(dtype)
        self._host_views[name] = self._trunc_host_bufs[name].view(dtype)

    def to(self, *args, **kwargs) -> None:
        """Move all device buffers to a new device/dtype."""
        old_device = self._device_buffer.device

        # Move contiguous buffer
        self._device_buffer = self._device_buffer.to(*args, **kwargs)

        # Move truncatable buffers
        for name in self._truncatable_names:
            self._trunc_device_bufs[name] = self._trunc_device_bufs[name].to(*args, **kwargs)

        # Recreate views if device changed
        if old_device != self._device_buffer.device:
            self._create_contiguous_views()
            for name in self._truncatable_names:
                _, dtype = self._tensor_specs[name]
                self._device_views[name] = self._trunc_device_bufs[name].view(dtype)


# TODO (lucaslie): as this list is growing we may want to "upstream" the active arguments to
# nest_sequences and _prepare_inputs to skip on unnecessary computations.
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
    - max_seq_info: [max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size]
      Model-level constants for the attention kernel: maximum context length (equal to max_seq_len),
      maximum number of KV cache blocks per sequence (ceil(max_seq_len / tokens_per_block)),
      block offset multiplier derived from kv_cache strides, and maximum batch size. These are
      set once via update_cache_information() after cache initialization and remain constant.
    - page_seq_indices: [si_0, si_1, ..., si_{np-1}] where si_j is the sequence index that page j
      belongs to. For example, if seq 0 has 2 pages and seq 1 has 3, then
      page_seq_indices = [0, 0, 1, 1, 1].
    - page_in_seq: [pi_0, pi_1, ..., pi_{np-1}] where pi_j is the page index within its sequence.
      For example, if seq 0 has 2 pages and seq 1 has 3, then page_in_seq = [0, 1, 0, 1, 2].
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
        max_seq_len: int,
        max_batch_size: int,
        tokens_per_block: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        vocab_size_padded: Optional[int] = None,
    ):
        """Initialize the SequenceInfo object.

        Args:
            max_seq_len: corresponds to the maximum sequence length of the input sequence. It
                includes the tokens in the input sequence and the tokens generated by the model.
            max_batch_size: corresponds to the maximum number of sequences (or requests) that the
                model can process.
            tokens_per_block: corresponds to the tokens per block of the cache.
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
        self.tokens_per_block = tokens_per_block or max_seq_len
        self.max_blocks_per_seq = math.ceil(max_seq_len / self.tokens_per_block)
        # NOTE (lucaslie): +1 is a WAR to address issue when using flashinfer attention with
        # (max_batch_size, max_seq_len) input in trtllm runtime.
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        self.max_num_tokens = max_num_tokens or (max_seq_len + 1) * max_batch_size

        # will store num_blocks later...
        self._num_blocks = None

        # TODO (lucaslie): can we remove this eventually from this i/f?
        self.vocab_size_padded = vocab_size_padded

        # NOTE: we keep an extra state slot around to simplify cuda graph padding
        # WHY?
        # Requests that just finished won't free their used resources immediately. Specifically, the
        # running order is self.scheduler.schedule_request, self._forward_step() and
        # self._process_previous_batch() in the PyExecutor. Hence, the current forward step will
        # remove finished requests but will not remove mamba_cache immediately and therefore it
        # won't be available in time for padding in the next forward step.
        self.max_num_state_slots = max_batch_size + 1

        # log parameters
        ad_logger.info(
            f"[SequenceInfo:] {self.max_seq_len=}, {self.max_batch_size=}, {self.max_num_tokens=}, "
            f"{self.max_blocks_per_seq=}, {self.tokens_per_block=}"
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
            # [max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size]
            ("max_seq_info", 4, torch.int),
            ("logits_gather_indices", self.max_num_tokens, torch.long),
            ("logits_gather_info", 2, torch.int),
            # OTHER FIELDS WHERE WE NEED EFFICIENT HOST<>DEVICE TRANSFER
            ("_gather_idx", self.max_num_tokens, torch.int),
            ("_mask_scatter_indices", self.max_num_tokens, torch.int),
            # TRUNCATABLE TENSORS: large, variable-length, each independently truncated during
            # H2D copy to avoid copying unused capacity.
            # NOTE: sufficient for max_num_tokens forward pass. will be resized when KVCacheManager
            # is created.
            ("cache_loc", self.max_num_tokens, torch.int, True),
            ("page_seq_indices", self.max_num_tokens, torch.int, True),
            ("page_in_seq", self.max_num_tokens, torch.int, True),
        ]

        # Create the InputBuffer that manages contiguous host and device memory
        # Starts on default device; use to() to move to target device
        self._input_buffer = InputBuffer(tensor_specs)
        self._available_args = set(self._input_buffer.tensor_names) | {
            f"{name}_host" for name in self._input_buffer.tensor_names
        }

        # Initialize args_list from tensor specs (all entries are tensors)
        self._args_list: Dict[str, torch.Tensor] = {
            spec[0]: torch.zeros(spec[1], dtype=spec[2]) for spec in tensor_specs
        }

        self._active_args = ("input_ids", "position_ids")
        self._shapeable_args = ("input_ids", "position_ids", "input_ids_host", "position_ids_host")

        # Args that require copy to InputBuffer but are NOT included in named_args / graph inputs.
        # Pre-populated with args that are always needed regardless of graph inclusion.
        self._requires_copy: Set[str] = {
            "max_seq_info",  # written once at cache init (update_cache_information)
            "page_seq_indices",  # used by host-prepare (not a graph input)
            "page_in_seq",  # used by host-prepare (not a graph input)
            "logits_gather_indices",  # always needed for logits gathering
            "logits_gather_info",  # always needed for logits gathering
            "_gather_idx",  # overlap scheduler metadata
            "_mask_scatter_indices",  # overlap scheduler metadata
        }
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
        return self._args_list["seq_len"].tolist()

    @property
    def input_pos(self) -> List[int]:
        return self._args_list["input_pos"].tolist()

    @property
    def cache_loc(self) -> List[int]:
        return self._args_list["cache_loc"].tolist()

    @property
    def pages_per_seq(self) -> List[int]:
        return self._args_list["pages_per_seq"].tolist()

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
    def page_assignments(self) -> List[List[int]]:
        """Return the page assignments for each sequence."""
        return self._get_page_assignments(self.cache_loc, self.pages_per_seq)

    @property
    def num_blocks(self) -> int:
        assert self._num_blocks is not None, "num_blocks not set yet"
        return self._num_blocks

    def estimate_cache_tokens_per_forward(self) -> int:
        """Estimate the max number of tokens that will be cached for a forward pass.

        It is estimated assuming a worst-case allocation of tokens across sequences in a batch.
        """
        seq_len = math.ceil(self.max_num_tokens / self.max_batch_size)
        num_blocks_estimate_per_seq = math.ceil(seq_len / self.tokens_per_block)
        num_blocks_estimate = num_blocks_estimate_per_seq * self.max_batch_size
        return num_blocks_estimate * self.tokens_per_block

    def update_cache_information(self, num_blocks: int, block_offset_multiplier: int = 0) -> None:
        """Update cache information after cache manager creation.

        Sets num_blocks and block_offset_multiplier, writes max_seq_info to the host buffer
        (constant after this call), and resizes cache_loc if needed.
        """
        # set num_blocks and block_offset_multiplier
        self._num_blocks = num_blocks

        # write max_seq_info once (constant after this call)
        max_seq_info = [
            self.max_seq_len,
            self.max_blocks_per_seq,
            block_offset_multiplier,
            self.max_batch_size,
        ]
        self._store_arg("max_seq_info", max_seq_info)

        # get current capacity
        cache_loc_capacity = self._input_buffer.get_capacity("cache_loc")

        # cache_loc requires some special treatment due to block reuse. Note that the constraint for
        # cache_loc with block_reuse is as follows:
        # 0 <= cache_loc < num_blocks
        # len(cache_loc) <= num_blocks * max_batch_size
        # NOTE: # However, num_blocks * max_batch_size may be a potentially very large number and an
        # overestimation, see we assume at most 10x reuse of blocks.
        estimated_capacity = num_blocks * min(10, self.max_batch_size)

        # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
        # (max_batch_size, max_seq_len) input in trtllm runtime.
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        estimated_capacity = estimated_capacity + 1

        if estimated_capacity > cache_loc_capacity:
            # Resize all truncatable page tensors together (they share the same max size)
            for tensor_name in ("cache_loc", "page_seq_indices", "page_in_seq"):
                self._input_buffer.resize(tensor_name, estimated_capacity)
                self._args_list[tensor_name] = torch.zeros(estimated_capacity, dtype=torch.int)

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

    def require_copy(self, arg_name: str) -> bool:
        """Mark an argument as requiring copy to InputBuffer.

        Unlike activate_arg, this does NOT add the arg to _named_args / graph inputs.
        It only ensures _store_arg writes to InputBuffer so _get_arg returns updated values.

        Use cases:
        - Host-prepare functions that need args outside the CUDA graph
        - Args that are always needed in InputBuffer regardless of graph inclusion

        Returns:
            True if the argument was newly marked, False if already marked.
        """
        assert arg_name in self.available_args, f"{arg_name=} not found in {self.available_args}"
        if arg_name not in self._requires_copy:
            self._requires_copy.add(arg_name)
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
            # Use batch_size >= 2 for export to prevent torch.export from specializing
            # the batch dimension when max_batch_size=1 (dimension value 1 triggers static optimization)
            bs, seq_len = max(2, min(2, self.max_batch_size)), min(4, self.max_seq_len)
            input_ids = torch.ones(bs, seq_len, dtype=torch.int).tolist()

        # figure out page assignments
        pages_per_seq = [
            len(ids_one_seq) // self.tokens_per_block
            + (len(ids_one_seq) % self.tokens_per_block > 0)
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
        """Set an example sequence with max_num_tokens.

        The per-sequence length is capped to the maximum that fits in the paged KV cache
        (max_blocks_per_seq * tokens_per_block) to avoid exceeding block_offsets capacity.
        """
        max_cache_tokens_per_seq = self.max_blocks_per_seq * self.tokens_per_block
        seq_len = min(self.max_num_tokens // self.max_batch_size, max_cache_tokens_per_seq)
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
        data: "Union[List[Number], torch.Tensor]",
        reset_val: Optional[Number] = None,
    ) -> None:
        """Store the argument into the pinned host buffer for later batch transfer to device.

        The data is stored in the host-side pinned memory buffer managed by InputBuffer.
        The actual H2D transfer happens in a single batch at the end of nest_sequences().

        Lists are converted to tensors at the boundary so the rest of the pipeline is
        tensor-only.

        Args:
            name: Name of the argument to store.
            data: List of values or a 1-D torch.Tensor to store.
            reset_val: Value to reset/fill the tensor with before writing data.
        """
        with nvtx_range(f"ad_store_on_host_seq_info_arg_{name}"):
            # Convert to tensor at the boundary (numpy is ~2-3x faster than torch.tensor for large lists)
            # TODO: move this to self._input_buffer.store() when _args_list get deprecated
            if not isinstance(data, torch.Tensor):
                _, dtype = self._input_buffer._tensor_specs[name]
                data = _list_to_tensor(data, dtype)

            self._args_list[name] = data

            # Only store to buffer when the argument is active or requires copy
            is_active = name in self._active_args or f"{name}_host" in self._active_args
            is_required = name in self._requires_copy or f"{name}_host" in self._requires_copy
            if not (is_active or is_required):
                return

            # Store to the InputBuffer's pinned host memory
            self._input_buffer.store(name, data, fill_value=reset_val)

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
        _ungathered_input_ids: Optional[torch.Tensor] = None,
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
            _ungathered_input_ids: Optional tensor of ungathered input ids from the overlap
                scheduler. If provided, triggers rescatter_input_ids after H2D copy.
            extra_args: Extra arguments to be stored in the interface.

        This i/f will ensure that all sequence info args are updated accordingly. Reset values are
        chosen as "neutral" values so that for cases like rounding up batch sizes for cudagraph we
        only write to unused caches.
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

            # Resolve cu_num_pages: use caller-provided value or derive from pages_per_seq
            if cu_num_pages is None:
                pps_t = self._args_list["pages_per_seq"]
                cu_num_pages = torch.zeros(len(pps_t) + 1, dtype=torch.int)
                cu_num_pages[1:] = pps_t.cumsum(0)
            self._store_arg("cu_num_pages", cu_num_pages)

            # Compute page_seq_indices and page_in_seq using vectorized torch ops,
            # reusing the stored cu_num_pages tensor instead of recomputing the cumsum.
            # page_seq_indices[j] = which sequence page j belongs to
            # page_in_seq[j] = which page within that sequence (0-indexed)
            pages_per_seq_t = self._args_list["pages_per_seq"]
            cu_pages_t = self._args_list["cu_num_pages"]
            seq_indices = torch.arange(len(pages_per_seq), dtype=torch.int)
            page_seq_indices_t = torch.repeat_interleave(seq_indices, pages_per_seq_t)
            total_pages = cu_pages_t[-1].item()
            page_in_seq_t = torch.arange(total_pages, dtype=torch.int) - torch.repeat_interleave(
                cu_pages_t[:-1], pages_per_seq_t
            )
            self._store_arg("page_seq_indices", page_seq_indices_t)
            self._store_arg("page_in_seq", page_in_seq_t)

        # update sequence length with cache
        if seq_len_with_cache is None:
            seq_len_with_cache = [i_p + s_l for i_p, s_l in zip(self.input_pos, self.seq_len)]
        self._store_arg("seq_len_with_cache", seq_len_with_cache)

        # update last page length
        if last_page_len is None:
            last_page_len = [(slwc - 1) % self.tokens_per_block + 1 for slwc in seq_len_with_cache]
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
        self._store_arg("logits_gather_indices", logits_gather_indices)

        # check for updated logits_gather_info
        if logits_gather_info is None:
            logits_gather_info = [len(logits_gather_indices), 1]
        self._store_arg("logits_gather_info", logits_gather_info)

        ### UPDATE OVERLAP SCHEDULER METADATA ######################################################
        # check for updated _gather_idx
        if _gather_idx is not None:
            self._store_arg("_gather_idx", _gather_idx)

        # check for updated _mask_scatter_indices
        if _mask_scatter_indices is not None:
            self._store_arg("_mask_scatter_indices", _mask_scatter_indices)

        ### UPDATE EXTRA INPUTS ####################################################################
        self._extra_args = {}
        for key, value in extra_args.items():
            self._store_extra_arg(key, value)

        ### BATCH COPY TO DEVICE ###################################################################
        # Perform a single async H2D copy for all device tensors
        # The copy is truncated at the end of cache_loc to minimize transfer size
        self._input_buffer.copy_to_device()

        ### RESCATTER + HOST PREPARE ###############################################################
        # Rescatter input_ids if ungathered tokens are provided (overlap scheduler)
        if _ungathered_input_ids is not None:
            self.rescatter_input_ids(_ungathered_input_ids)

        # Run host-prepare functions for attention forward (e.g. trtllm block_offsets computation)
        self.run_host_prepare_for_attention_forward()

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
        # Ensure all host-prepare args are stored to InputBuffer via _requires_copy
        for arg in args:
            self.require_copy(arg)

    def run_host_prepare_for_attention_forward(self) -> None:
        for host_function, args in self._host_prepare_functions:
            host_function(**{arg: self._get_arg(arg) for arg in args})


class ResourceHandler(ABC):
    """An abstract interface to handle a generic resource needed by attention operators.

    The ResourceHandler interface standardizes operations that the cached sequence interface
    performs on the resources providing an abstract handle.
    """

    @property
    def is_paged(self) -> bool:
        """Whether the resource is paged.

        If the resource is paged, it will participate in the resize computation of the caches and
        needs to implement the _get_bytes_per_token method.
        """
        return False

    @property
    def bytes_per_token(self) -> int:
        """The size of the resource per token."""
        if self.is_paged:
            return self._get_bytes_per_token()
        else:
            raise NotImplementedError(f"Resource {self.__class__.__name__} is not paged.")

    def _get_bytes_per_token(self) -> int:
        """The size of the resource per token."""
        raise NotImplementedError(
            f"Resource {self.__class__.__name__} needs to implement _get_bytes_per_token."
        )

    @abstractmethod
    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Initialize the resource for the given sequence info."""


class KVPagedResourceHandler(ResourceHandler):
    """Handler for paged KV cache resources.

    This handler indicates the resource should be managed by the standard KVCacheManager.

    Args:
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension of each head.
        dtype: The dtype of the KV cache.
        kv_factor: The factor of the KV cache. Default is 2 for combined k/v cache.
        kv_layout: Memory layout for the KV cache. Either "HND" (head-num-dim) or "NHD" (num-head-dim).
            Default is "HND" which is the standard layout for flashinfer.
    """

    @property
    def is_paged(self) -> bool:
        """Whether the resource is paged."""
        return True

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        kv_factor: int = 2,
        kv_layout: Literal["HND", "NHD"] = "HND",
    ) -> None:
        """Initialize the KVPagedResourceHandler.

        Args:
            num_kv_heads: Number of key-value heads.
            head_dim: Dimension of each head.
            dtype: The dtype of the KV cache.
            kv_factor: The factor of the KV cache. Default is 2.
            kv_layout: Memory layout - "HND" or "NHD". Default is "HND".
        """
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.kv_factor = kv_factor
        assert kv_factor in [1, 2], f"Invalid kv_factor: {kv_factor}"
        self.kv_layout = kv_layout

    def __eq__(self, other: Optional[ResourceHandler]) -> bool:
        """Check compatibility for KVCacheManager (head_dim and dtype must match)."""
        if type(other) is not type(self):
            return False
        return (
            self.head_dim == other.head_dim
            and self.dtype == other.dtype
            and self.kv_factor == other.kv_factor
            and self.kv_layout == other.kv_layout
        )

    def _get_bytes_per_token(self) -> int:
        """The size of the resource per token in bytes."""
        return self.num_kv_heads * self.kv_factor * self.head_dim * self.dtype.itemsize

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate paged resource locally when not managed by KVCacheManager.

        Args:
            sequence_info: Sequence information containing device info.

        Returns:
            Allocated tensor with shape depending on kv_layout:
            - NHD: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
            - HND: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
        """
        if self.kv_layout == "HND":
            return torch.empty(
                sequence_info.num_blocks,
                self.kv_factor,
                self.num_kv_heads,
                sequence_info.tokens_per_block,
                self.head_dim,
                device=sequence_info.device,
                dtype=self.dtype,
            )
        elif self.kv_layout == "NHD":
            return torch.empty(
                sequence_info.num_blocks,
                sequence_info.tokens_per_block,
                self.kv_factor,
                self.num_kv_heads,
                self.head_dim,
                device=sequence_info.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Invalid kv_layout: {self.kv_layout}")


class StateResourceHandler(ResourceHandler):
    """Handler for per-sequence state resources (e.g., Mamba SSM/conv states).

    These resources have shape [max_batch_size, *state_shape] and can be either:
    - Managed by MambaHybridCacheManager (for typed subclasses SSMResourceHandler, CausalConvResourceHandler)
    - Allocated locally via allocate() (for generic StateResourceHandler or when constraints don't hold)

    Subclasses should define state_shape as a property that returns the appropriate shape.
    """

    def __init__(self, *state_shape: int, dtype: torch.dtype) -> None:
        """Initialize the StateResourceHandler.

        Args:
            state_shape: The shape of a single state resource.
            dtype: The dtype of the state resource.
        """
        self._state_shape = state_shape
        self.dtype = dtype

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Return the state shape. Subclasses may override this as a property."""
        return self._state_shape

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate state resource locally (fallback when not managed by cache manager)."""
        return torch.empty(
            sequence_info.max_num_state_slots,
            *self.state_shape,
            device=sequence_info.device,
            dtype=self.dtype,
        )

    def __eq__(self, other: Optional[ResourceHandler]) -> bool:
        """Check compatibility for MambaHybridCacheManager state resources."""
        if type(other) is not type(self):
            return False
        return self.state_shape == other.state_shape and self.dtype == other.dtype


class SSMResourceHandler(StateResourceHandler):
    """Handler for SSM state resources that maps directly to MambaCacheManager's ssm_states buffer.

    These resources have shape [max_batch_size, num_heads, head_dim, d_state] and are
    managed by MambaHybridCacheManager via the ssm_states buffer when compatible.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        d_state: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the SSMResourceHandler.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            d_state: SSM state size.
            dtype: Data type for the state.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_state = d_state
        # Call parent with dtype only (state_shape comes from property)
        super().__init__(dtype=dtype)

    @property
    def state_shape(self) -> Tuple[int, int, int]:
        """Return the SSM state shape: (num_heads, head_dim, d_state)."""
        return (self.num_heads, self.head_dim, self.d_state)


class CausalConvResourceHandler(StateResourceHandler):
    """Handler for causal conv state resources that maps to MambaCacheManager's conv_states buffer.

    These resources have shape [max_batch_size, conv_dim, d_conv - 1] and are
    managed by MambaHybridCacheManager via the conv_states buffer when compatible.

    Note: d_conv is the kernel size, and (d_conv - 1) is the state size stored in the cache.
    """

    def __init__(
        self,
        conv_dim: int,
        d_conv: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the CausalConvResourceHandler.

        Args:
            conv_dim: Convolution dimension (typically in_channels).
            d_conv: Kernel size. The cache stores d_conv - 1 elements.
            dtype: Data type for the state.
        """
        self.conv_dim = conv_dim
        self.d_conv = d_conv  # kernel_size
        # Call parent with dtype only (state_shape comes from property)
        super().__init__(dtype=dtype)

    @property
    def state_shape(self) -> Tuple[int, int]:
        """Return the conv state shape: (conv_dim, d_conv - 1)."""
        return (self.conv_dim, self.d_conv - 1)


class UnpagedResourceHandler(ResourceHandler):
    """Handler for per-token unpaged resources (e.g., unpaged KV caches).

    These resources have shape [max_batch_size, max_seq_len, *token_shape].
    They are allocated locally and not managed by MambaHybridCacheManager.
    """

    def __init__(self, *token_shape: int, dtype: torch.dtype) -> None:
        """Initialize the UnpagedResourceHandler.

        Args:
            token_shape: The shape of the resource per token.
            dtype: The dtype of the resource.
        """
        self.token_shape = token_shape
        self.dtype = dtype

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Initialize the unpaged resource for the given sequence info."""
        return torch.empty(
            sequence_info.max_num_state_slots,
            sequence_info.max_seq_len,
            *self.token_shape,
            device=sequence_info.device,
            dtype=self.dtype,
        )


class MHACallable(Protocol):
    def __call__(
        self,
        *qkv_metadata_and_caches: Union[torch.Tensor, Constant],
    ) -> torch.Tensor: ...


class PrepareMetadataCallable(Protocol):
    def __call__(
        self, *sequence_info_args_and_constants: Union[torch.Tensor, Constant]
    ) -> List[torch.Tensor]: ...


AttentionLayout = Literal["bsnd", "bnsd"]

ResourceHandlerDict = Dict[str, ResourceHandler]


class AttentionDescriptor(ABC):
    """An interface to define a functional attention operator.

    The main logic is contained with the actual attention op as well as the prepare_metadata op. The
    prepare_metadata op is responsible for converting the standardized sequence info into metadata
    specific to the attention op.
    """

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
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Provide a dictionary of resource handlers that can be used to initialize the resources.

        The key corresponds to the argument name used in the attention op signature. The function
        key doesn't need to be unique across multiple attention nodes in the graph. The key used to
        describe the cache in the graph will be patched with the attention node index to ensure
        uniqueness.

        The resource will be initialized before the initial forward pass and will be managed by the
        global CacheManager and passed back to the model during the forward pass.

        If the cache initializer requires information about the attention op, it can retrieve
        the necessary information from the source attention node and cache config.
        """

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Provide a list of constant arguments to be passed to the attention op.

        The constant arguments are passed to the attention op as additional arguments after the
        caches. The constants are expected to be of type int, float, str, or None.
        """
        return []

    @staticmethod
    def resolve_cache_dtype(dtype_config: str, fallback_dtype: torch.dtype) -> torch.dtype:
        """Resolve cache dtype from KvCacheConfig dtype string to torch.dtype.

        Args:
            dtype_config: The dtype string from KvCacheConfig (e.g., "auto", "float16", "bfloat16").
            fallback_dtype: The fallback dtype to use when dtype_config is "auto".

        Returns:
            The resolved torch.dtype.
        """
        if dtype_config == "auto":
            return fallback_dtype
        return str_dtype_to_torch(dtype_config)

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
