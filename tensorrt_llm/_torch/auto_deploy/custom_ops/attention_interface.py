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

from ...._utils import nvtx_range, prefer_pinned, str_dtype_to_torch
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
                self._total_bytes, dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned()
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
                byte_size, dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned()
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

    @property
    def host_device(self) -> torch.device:
        """Return the device of the host buffer."""
        return self._host_buffer.device

    def get_view(self, name: str, truncate: bool = False) -> torch.Tensor:
        """Get the device tensor view for the specified name."""
        if truncate:
            return self._device_views[name][: self.get_current_length(name)]
        return self._device_views[name]

    def get_host_view(self, name: str, truncate: bool = False) -> torch.Tensor:
        """Get the host tensor view for the specified name."""
        if truncate:
            return self._host_views[name][: self.get_current_length(name)]
        return self._host_views[name]

    def get_capacity(self, name: str) -> int:
        """Get the maximum number of elements for the specified tensor."""
        numel, _ = self._tensor_specs[name]
        return numel

    def get_current_length(self, name: str) -> int:
        """Get the current stored length for the specified tensor."""
        return self._current_lengths[name]

    def stage(
        self,
        name: str,
        data: Union[torch.Tensor, List[Number]],
        fill_value: Optional[Number] = None,
    ) -> int:
        """Stage data into the pinned host buffer.

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

        # convert to tensor via numpy (numpy is ~2-3x faster than torch.tensor for large lists)
        if not isinstance(data, torch.Tensor):
            data = _list_to_tensor(data, dtype)

        length = data.numel()
        assert length <= numel, f"Data too large for buffer '{name}': {length} > {numel}"
        # Use numpy for the memcpy into pinned memory — avoids torch dispatcher overhead
        dst = host_view[:length].numpy()
        src = (data if data.dtype == dtype else data.to(dtype)).numpy()
        np.copyto(dst, src)

        self._current_lengths[name] = length
        return length

    def copy_(self, name: str, src: torch.Tensor) -> None:
        """Copy a tensor into the buffer.

        NOTE: only the buffer that matches the source device is updated!

        Args:
            name: Name of the tensor in the buffer.
            src: host-side or device-side source tensor whose elements are copied into the first
                ``src.numel()`` positions of the buffer views.
        """
        n = src.numel()
        if src.device == self._host_buffer.device:
            self._host_views[name][:n].copy_(src)
        elif src.device == self._device_buffer.device:
            self._device_views[name][:n].copy_(src, non_blocking=True)
        else:
            raise RuntimeError(f"Unexpected device: {src.device=}")
        self._current_lengths[name] = n

    def copy_to_device(self) -> None:
        """Copy from host buffers to device buffers.

        Contiguous tensors are copied in a single bulk transfer.
        Truncatable tensors are each copied independently, truncated to actual length.
        """
        with nvtx_range("ad_input_buffer_h2d_copy"):
            # Copy contiguous buffer in one shot
            if self._total_bytes > 0:
                h_buffer = self._host_buffer[: self._total_bytes]
                d_buffer = self._device_buffer[: self._total_bytes]
                d_buffer.copy_(h_buffer, non_blocking=True)

            # Copy each truncatable tensor independently, truncated to current length
            for name in self._truncatable_names:
                length = self.get_current_length(name)
                if length > 0:
                    _, dtype = self._tensor_specs[name]
                    copy_bytes = length * dtype.itemsize
                    trunc_d_buf = self._trunc_device_bufs[name][:copy_bytes]
                    trunc_h_buf = self._trunc_host_bufs[name][:copy_bytes]
                    trunc_d_buf.copy_(trunc_h_buf, non_blocking=True)

    def copy_to_host(self) -> None:
        """Copy from device buffer to host buffer.

        Mirrors ``copy_to_device``: uses the current length of the truncatable tensor
        (last in spec) to minimize transfer size.
        """
        with nvtx_range("ad_input_buffer_d2h_copy"):
            # Copy contiguous buffer in one shot
            if self._total_bytes > 0:
                h_buffer = self._host_buffer[: self._total_bytes]
                d_buffer = self._device_buffer[: self._total_bytes]
                h_buffer.copy_(d_buffer, non_blocking=True)

            # Copy each truncatable tensor independently, truncated to current length
            for name in self._truncatable_names:
                length = self.get_current_length(name)
                if length > 0:
                    _, dtype = self._tensor_specs[name]
                    copy_bytes = length * dtype.itemsize
                    trunc_d_buf = self._trunc_device_bufs[name][:copy_bytes]
                    trunc_h_buf = self._trunc_host_bufs[name][:copy_bytes]
                    trunc_h_buf.copy_(trunc_d_buf, non_blocking=True)

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
            new_byte_size, dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned()
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
      in the input_ids sequence. We use [b, 1] to denote generate-only batches. Usually derived from
      input_pos.

    NOTE: ``input_ids`` and ``position_ids`` are initially expected to be of shape [b, seq_len]
    before we switch to cached+flattened attention.

    ### EXTRA ARGUMENTS PROVIDED TO THE INTERFACE ##################################################
    Those are extra arguments that can be provided to the interface and they are stored as follows:
    - _extra_args: dictionary of extra arguments with currently active values.

    ### BASIC ARGUMENTS TO DESCRIBE RAGGED ATTENTION+CACHE INPUTS ##################################
    - input_pos: [pos_0, ..., pos_{b-1}]
      Corresponds to the total number of tokens that have already been cached for each sequence
      in the batch (i.e., the starting position in the cache for new tokens).
    - cu_seqlen: [0, s_0, s_0+s_1, ..., s_total]
      Cumulative sequence lengths of shape [b+1]. cu_seqlen[i+1] - cu_seqlen[i] gives the length
      of sequence i.
    - cache_loc: [c_0, c_1, ..., c_{np-1}] where np is total number of pages allocated to describe
      all sequences in the batch. Each value is a page index in the cache.
    - cu_num_pages: [0, ps_0, ps_0+ps_1, ..., sum(ps_i)]
      Cumulative number of pages of shape [b+1]. cu_num_pages[i+1] - cu_num_pages[i] gives the
      number of pages for sequence i.
    - last_page_len: [lpl_0, lpl_1, ..., lpl_{b-1}]
      Number of valid tokens in the last page for each sequence. Computed as
      (seq_len_with_cache - 1) % page_size + 1.
    - slot_idx: [slot_0, slot_1, ..., slot_{b-1}]
      Corresponds to the slot index of each sequence in the batch.

    ### INFO OBJECTS THAT ARE AVAILABLE TO DESCRIBE THE INPUTS IN A MORE COMPACT WAY ###############
    - batch_info: [num_prefill, num_prefill_tokens, num_extend, num_extend_tokens, num_decode, num_decode_tokens]
      Batch metadata containing the number of requests and tokens for each of the three request
      types: prefill, extend (spec dec), and decode.
    - max_seq_info: [max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size]
      Model-level constants for the attention kernel: maximum context length (equal to max_seq_len),
      maximum number of KV cache blocks per sequence (ceil(max_seq_len / tokens_per_block)),
      block offset multiplier derived from kv_cache strides, and maximum batch size. These are
      set once via update_cache_information() after cache initialization and remain constant.

    ### ADDITIONAL ARGUMENTS AVAILABLE THAT ARE DERIVED FROM THE BASIC ARGUMENTS ###################
    - seq_len: [s_0, s_1, ..., s_{b-1}] such that s_total = sum(s_i)
      Describes how long each sequence is. For example,
      input_ids[:s_0] will correspond to sequence 0 in the batch and input_ids[s_0:s_1] will
      correspond to sequence 1 in the batch.
    - pages_per_seq: [ps_0, ps_1, ..., ps_{b-1}] where ps_i is the number of pages allocated for
      sequence i. Note that, for example, cache_loc[sum(ps_0:ps_{i-1}):sum(ps_0:ps_i)] will
      correspond to the pages associated with sequence i in the batch.
    - seq_len_with_cache: [pos_0+s_0, pos_1+s_1, ..., pos_{b-1}+s_{b-1}]
      Total sequence length including cached tokens for each sequence (input_pos + seq_len).
    - use_initial_states: [bool_0, bool_1, ..., bool_{b-1}]
      Per-sequence boolean indicating whether initial states should be used (True if input_pos > 0).

    ### OTHER ARGUMENTS USED BY THE RUNTIME ########################################################
    - extra_page_per_seq: [ep_0, ep_1, ..., ep_{b-1}]
      Extra page per sequence for deferred page insertion. If no extra pages is needed, this is -1.
    - token_gather_indices: [g_0, g_1, ..., g_{s_total-1}]
      Gather indices used by the gather_tokens custom op to gather logits before the LM head.
    - tokens_gather_info: [num_tokens_to_gather, gather_required]. Info for the
      gather_tokens custom op to gather tokens for which we don't need the output logits.
    - _gather_idx: [g_0, g_1, ..., g_{s_total-1}]
      Gather indices used by the overlap scheduler to reorder input tokens.
    - _mask_scatter_indices: [m_0, m_1, ..., m_{s_total-1}]
      Mask scatter indices used by the overlap scheduler to scatter results back.

    NOTE: all tensors are also accessible as host tensors with the host suffix as host version. For
    example, the tensor "batch_info" is accessible as "batch_info_host" on the host.


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
            ### ORIGINAL MODEL ARGUMENTS ###########################################################
            ("input_ids", self.max_num_tokens, torch.int),
            ("position_ids", self.max_num_tokens, torch.long),
            ### BASIC ARGUMENTS TO DESCRIBE RAGGED ATTENTION+CACHE INPUTS ##########################
            ("input_pos", self.max_batch_size, torch.int),
            ("cu_seqlen", self.max_batch_size + 1, torch.int),
            ("cache_loc", self.max_num_tokens, torch.int, True),
            ("cu_num_pages", self.max_batch_size + 1, torch.int),
            ("last_page_len", self.max_batch_size, torch.int),
            ("slot_idx", self.max_batch_size, torch.long),
            ### INFO OBJECTS THAT ARE AVAILABLE TO DESCRIBE THE INPUTS IN A MORE COMPACT WAY #######
            ("batch_info", 3, torch.int),
            ("max_seq_info", 4, torch.int),
            ### ADDITIONAL ARGUMENTS AVAILABLE THAT ARE DERIVED FROM THE BASIC ARGUMENTS ###########
            ("seq_len", self.max_batch_size, torch.int),
            ("pages_per_seq", self.max_batch_size, torch.int),
            ("seq_len_with_cache", self.max_batch_size, torch.int),
            ("use_initial_states", self.max_batch_size, torch.bool),
            ### OTHER ARGUMENTS USED BY THE RUNTIME ################################################
            ("extra_page_per_seq", self.max_batch_size, torch.int),
            ("token_gather_indices", self.max_num_tokens, torch.long),
            ("tokens_gather_info", 2, torch.int),
            ("_gather_idx", self.max_num_tokens, torch.int),
            ("_mask_scatter_indices", self.max_num_tokens, torch.int),
        ]

        # Create the InputBuffer that manages contiguous host and device memory
        # Starts on default device; use to() to move to target device
        self._input_buffer = InputBuffer(tensor_specs)
        self._available_args = set(self._input_buffer.tensor_names) | {
            name + self._host_suffix for name in self._input_buffer.tensor_names
        }

        # active args that are included in the graph inputs
        self._active_args = ("input_ids", "position_ids")
        # args out of the active args that are shapeable
        self._shapeable_args = {"input_ids", "position_ids"}

        # args that are required by the host-side prepare function
        self._active_host_prep_args: Set[str] = set()
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

    @property
    def host_device(self) -> torch.device:
        return self._input_buffer.host_device

    @property
    def _host_suffix(self) -> str:
        return "_host"

    def unflatten(self, tnsr: torch.Tensor) -> torch.Tensor:
        """Shape the tensor for the forward pass based on the current attention mode.

        Args:
            tnsr: The tensor to shape assumed to be in shape [batch_size*seq_len, ...]

        Returns:
            The shaped tensor flattened or unflattened based on the current attention mode.
        """
        # check if we are still running uncached attention in which case we are also still
        # operate on unflattened tensors with explicit [batch_size, seq_len, ...] shape
        # generate-only batches are also formatted like this (i.e. [b, 1])
        total_num_tokens = self.total_num_tokens
        if not self._use_flattened_layout or self.is_generate_only:
            bs = self.num_sequences
            sl = total_num_tokens // bs
        # use [1,total_len] shape to indicate non-generate-only batch for cached attention
        else:
            bs, sl = 1, total_num_tokens

        # truncate to total tokens now, reshape, and return
        return tnsr[:total_num_tokens].view(bs, sl, *tnsr.shape[1:])

    def flatten(self, tnsr: torch.Tensor) -> torch.Tensor:
        """Flatten the tensor after the forward pass based on the current attention mode.

        Args:
            tnsr: The tensor to flatten.

        Returns:
            The flattened tensor.
        """
        return tnsr.squeeze(int(self.is_generate_only))

    def get_arg(
        self, name: str, truncate: Optional[bool] = None, unflatten: Optional[bool] = None
    ) -> torch.Tensor:
        """Get the argument from the input buffer either on device or host.

        Args:
            name: The name of the argument.
            truncate: Whether to truncate the tensor to the current length. Default is None, which
                means we will truncate when unflattening and no truncation otherwise.
            unflatten: Whether to unflatten the tensor. Default is None, which means we will use
                _shapeable_args to determine if we should unflatten.

        Returns:
            The argument tensor.
        """

        is_host = name.endswith(self._host_suffix)
        name = name.removesuffix(self._host_suffix)
        if is_host:
            arg = self._input_buffer.get_host_view(name, truncate=truncate)
        else:
            arg = self._input_buffer.get_view(name, truncate=truncate)
        if unflatten is True or (unflatten is None and name in self._shapeable_args):
            assert truncate is not False, "unflattening requires truncation"
            arg = self.unflatten(arg)
        return arg

    def _named_args(self, include_extra_args: bool = True) -> Dict[str, torch.Tensor]:
        args = {k: self.get_arg(k) for k in self._active_args}

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
    def num_sequences(self) -> int:
        return self.get_arg("batch_info_host")[::2].sum().item()

    @property
    def total_num_tokens(self) -> int:
        return self.get_arg("batch_info_host")[1:].sum().item()

    @property
    def is_generate_only(self) -> bool:
        return self.get_arg("batch_info_host")[0].item() == 0

    def get_nested_page_assignments(self) -> List[List[int]]:
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
        cache_loc_host = self.get_arg("cache_loc_host", truncate=True)
        pages_per_seq_host = self.get_arg("pages_per_seq_host", truncate=True)
        return [
            c_loc_one_seq.tolist()
            for c_loc_one_seq in torch.split(cache_loc_host, pages_per_seq_host.tolist())
        ]

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
        self._stage_arg("max_seq_info", max_seq_info)

        # get current capacity
        cache_loc_capacity = self._input_buffer.get_capacity("cache_loc")

        # take estimated capacity from provided information
        estimated_capacity = self.max_batch_size * self.max_blocks_per_seq

        # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
        # (max_batch_size, max_seq_len) input in trtllm runtime.
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        estimated_capacity = estimated_capacity + 1

        if estimated_capacity > cache_loc_capacity:
            self._input_buffer.resize("cache_loc", estimated_capacity)

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
        input_ids: Optional[torch.Tensor] = None,
        cache_loc: Optional[Sequence[int]] = None,
        cu_num_pages: Optional[Sequence[int]] = None,
        slot_idx: Optional[Sequence[int]] = None,
        **extra_args,
    ) -> None:
        """Set an example sequence useful for testing and export purposes without cache history."""
        # use a best guess default for input_ids if not provided
        if input_ids is None:
            # Use batch_size >= 2 for export to prevent torch.export from specializing
            # the batch dimension when max_batch_size=1 (dimension value 1 triggers static optimization)
            bs, seq_len = max(2, min(2, self.max_batch_size)), min(4, self.max_seq_len)
            input_ids = torch.ones(bs, seq_len, dtype=torch.int)
        assert input_ids.ndim == 2, f"input_ids must be 2D, got {input_ids.ndim}"

        bs, seq_len = input_ids.shape

        # heuristic for cache assignment
        if cu_num_pages is None:
            num_pages = seq_len // self.tokens_per_block + (seq_len % self.tokens_per_block > 0)
            cu_num_pages = torch.arange(bs + 1, dtype=torch.int) * num_pages
        if cache_loc is None:
            cache_loc = torch.arange(cu_num_pages[-1])
        assert len(cache_loc) >= cu_num_pages[-1]
        if slot_idx is None:
            slot_idx = torch.arange(bs)
        assert len(slot_idx) >= bs

        self.nest_sequences(
            input_ids.flatten(),
            cu_seqlen=torch.arange(bs + 1, dtype=torch.int) * seq_len,
            input_pos=0,  # no cache history
            cache_loc=cache_loc,  # vanilla page assignments
            cu_num_pages=cu_num_pages,  # vanilla page assignments
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
        input_ids = torch.ones(self.max_batch_size, seq_len, dtype=torch.int)
        self.set_example_sequence(input_ids)

    def set_generate_only_batch(self, batch_size: Optional[int] = None) -> None:
        """Set an example sequence for generate-only batch."""
        batch_size = batch_size or self.max_batch_size
        self.set_example_sequence(
            torch.ones(batch_size, 1, dtype=torch.int), tokens_gather_info=[batch_size, 0]
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

    def _is_active(self, name: str, check_both: bool = True) -> bool:
        """Check if the host or device-side argument is an active model argument."""
        if check_both:
            name = name.removesuffix(self._host_suffix)
            name_host = name + self._host_suffix
        else:
            name_host = None
        return name in self._active_args or name_host in self._active_args

    def _is_active_host_prep(self, name: str, check_both: bool = True) -> bool:
        """Check if the host-side argument is an active host-side prepare argument."""
        if check_both:
            name = name.removesuffix(self._host_suffix)
            name_host = name + self._host_suffix
        else:
            name_host = None
        return name in self._active_host_prep_args or name_host in self._active_host_prep_args

    def _is_required(self, name: str, check_both: bool = True) -> bool:
        """Check if the argument is required anywhere.

        Returns:
            True if the device or host argument is required for either the graph inputs or the
            host-side prepare function.
        """
        return self._is_active(name, check_both) or self._is_active_host_prep(name, check_both)

    def _stage_arg(
        self,
        name: str,
        data: Union[List[Number], torch.Tensor, None],
        reset_val: Optional[Number] = None,
    ):
        """Stage the argument into the pinned host buffer for later batch transfer to device.

        The data is stored in the host-side pinned memory buffer managed by InputBuffer.
        The actual H2D transfer happens in a single batch at the end of nest_sequences().

        Lists are converted to tensors at the boundary so the rest of the pipeline is
        tensor-only.

        Args:
            name: Name of the argument to store.
            data: List of values or a 1-D torch.Tensor to store.
            reset_val: Value to reset/fill the tensor with before writing data.
        """
        # make sure it is provided if required
        if self._is_required(name):
            assert data is not None, f"data is required for {name}"

        if data is None:
            return None

        with nvtx_range(f"ad_stage_{name}_on_host"):
            # Store to the InputBuffer's pinned host memory
            name = name.removesuffix(self._host_suffix)
            self._input_buffer.stage(name, data, fill_value=reset_val)

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
        ### BASIC INPUTS FOR RAGGED ATTENTION + CACHES #############################################
        input_ids: Union[Sequence[int], torch.Tensor],
        cu_seqlen: Union[Sequence[int], torch.Tensor],
        input_pos: Union[Sequence[int], int, torch.Tensor],
        batch_info: Union[Sequence[int], torch.Tensor, None] = None,
        cache_loc: Union[Sequence[int], torch.Tensor, None] = None,
        cu_num_pages: Union[Sequence[int], torch.Tensor, None] = None,
        extra_page_per_seq: Optional[Sequence[int]] = None,
        slot_idx: Union[Sequence[int], torch.Tensor, None] = None,
        ### RUNTIME ARGUMENTS ######################################################################
        token_gather_indices: Union[Sequence[int], torch.Tensor, None] = None,
        tokens_gather_info: Union[Sequence[int], torch.Tensor, None] = None,
        _gather_idx: Union[Sequence[int], torch.Tensor, None] = None,
        _mask_scatter_indices: Union[Sequence[int], torch.Tensor, None] = None,
        _ungathered_input_ids: Optional[torch.Tensor] = None,
        ### EXTRA INPUTS FOR MULTIMODAL MODELS #####################################################
        **extra_args: Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> None:
        """Create and store sequence information for the next forward pass.

        Args:
            input_ids: Flattened list of input_ids.
            cu_seqlen: Cumulative sequence lengths for all sequences.
            input_pos: Absolute starting position in the cache for each sequence. Can be a single
                int (applied to all sequences) or a list of ints.
            batch_info: Batch metadata as [num_prefill, num_prefill_tokens, num_extend,
                num_extend_tokens, num_decode, num_decode_tokens]. If None, heuristic is used to
                compute it from seq_len. NOTE: the heuristic makes potentially incorrect assumptions
                about the batch composition. batch_info should be provided explicitly to ensure
                correctness.
            cache_loc: Flat list of page indices for all sequences. Must be provided together with
                cu_num_pages.
            cu_num_pages: Cumulative number of pages for all sequences. Must be provided together with
                cache_loc.
            extra_page_per_seq: Extra page per sequence for deferred page insertion.
            slot_idx: Slot index for each sequence in the batch.
            token_gather_indices: Gather indices for the logits before/after the LM head.
            tokens_gather_info: Info list containing [num_tokens_to_gather, gather_required].
            _gather_idx: Gather indices for the overlap scheduler to reorder input tokens.
            _mask_scatter_indices: Mask scatter indices for the overlap scheduler.
            _ungathered_input_ids: Optional tensor of ungathered input ids from the overlap
                scheduler. If provided, triggers rescatter_input_ids after H2D copy.
            extra_args: Extra arguments to be stored in the interface.

        This i/f will ensure that all sequence info args are updated accordingly. Reset values are
        chosen as "neutral" values so that for cases like rounding up batch sizes for cudagraph we
        only write to unused caches.
        """
        ### UPDATE CU_SEQLEN, BATCH INFO, AND INPUT_POS FIRST SINCE IT'S USED FOR OTHER UPDATES ####
        self._stage_arg("cu_seqlen", cu_seqlen)
        csl_host = self.get_arg("cu_seqlen_host", truncate=True)

        sl_host = csl_host[1:] - csl_host[:-1]
        self._stage_arg("seq_len", sl_host)

        # check for updated batch_info_tensor
        if batch_info is None:
            # NOTE: assumes no extend requests, decode requests are all of length 1 and come after
            # prefill requests.
            num_decode = int((sl_host.flip(0) == 1).cumprod(0).sum())
            num_prefill = len(sl_host) - num_decode
            num_prefill_tokens = int(sl_host.sum()) - num_decode
            batch_info = [num_prefill, num_prefill_tokens, num_decode]
        self._stage_arg("batch_info", batch_info)

        # check for updated input_pos (i.e. cache start position)
        if isinstance(input_pos, int):
            input_pos = torch.full((self.num_sequences,), input_pos, dtype=torch.int)
        self._stage_arg("input_pos", input_pos)
        ip_host = self.get_arg("input_pos_host", truncate=True)

        ### UPDATE REQUIRED INPUTS #################################################################
        # set new input_ids and make sure to flatten it
        self._stage_arg("input_ids", input_ids)

        ### UPDATE EXTRA INPUTS ####################################################################
        self._extra_args = {}
        for key, value in extra_args.items():
            self._store_extra_arg(key, value)

        ### UPDATE CACHE ASSIGNMENTS (needs to be provided if required!) ###########################
        # check for updated page assignments
        assert (cache_loc is None) == (cu_num_pages is None), "Both must be provided together!"
        self._stage_arg("cache_loc", cache_loc)
        self._stage_arg("cu_num_pages", cu_num_pages)
        lpl_host = (ip_host + sl_host - 1) % self.tokens_per_block + 1
        self._stage_arg("last_page_len", lpl_host)

        # check for updated slot_idx
        self._stage_arg("slot_idx", slot_idx)

        # check for updated extra_page_per_seq
        self._stage_arg("extra_page_per_seq", extra_page_per_seq)

        ### UPDATE OPTIONAL DERIVATIVE METADATA ####################################################
        if self._is_required("position_ids"):
            # set new position_ids and make sure to flatten it
            # position_ids for each sequence is in the range [input_pos, input_pos + seq_len - 1]
            ip_np = ip_host.numpy()
            sl_np = sl_host.numpy()
            base = np.repeat(ip_np, sl_np)
            group_starts = np.repeat(np.cumsum(sl_np) - sl_np, sl_np)
            offsets = np.arange(sl_np.sum()) - group_starts
            position_ids = torch.from_numpy(base + offsets)  # zero-copy back
            self._stage_arg("position_ids", position_ids)

        # update cumulative number of pages
        if self._is_required("pages_per_seq"):
            assert cu_num_pages is not None, "cu_num_pages is required for pages_per_seq"
            cu_num_pages_host = self.get_arg("cu_num_pages_host", truncate=True)
            pages_per_seq = cu_num_pages_host[1:] - cu_num_pages_host[:-1]
            self._stage_arg("pages_per_seq", pages_per_seq)

        # update sequence length with cache
        if self._is_required("seq_len_with_cache"):
            seq_len_with_cache = ip_host + sl_host
            self._stage_arg("seq_len_with_cache", seq_len_with_cache)

        # check for updated use_initial_states
        if self._is_required("use_initial_states"):
            use_initial_states = ip_host > 0
            self._stage_arg("use_initial_states", use_initial_states)

        ### UPDATE LOGITS GATHERING METADATA using heuristic if not provided #######################
        # default is to gather all logits
        if token_gather_indices is None:
            token_gather_indices = torch.arange(self.total_num_tokens, dtype=torch.long)
        self._stage_arg("token_gather_indices", token_gather_indices)

        # check for updated tokens_gather_info
        if tokens_gather_info is None:
            tokens_gather_info = [len(token_gather_indices), 1]
        self._stage_arg("tokens_gather_info", tokens_gather_info)

        ### UPDATE OVERLAP SCHEDULER METADATA ######################################################
        # check for updated _gather_idx
        if _gather_idx is not None:
            self._stage_arg("_gather_idx", _gather_idx)

        # check for updated _mask_scatter_indices
        if _mask_scatter_indices is not None:
            self._stage_arg("_mask_scatter_indices", _mask_scatter_indices)

        ### BATCH COPY TO DEVICE ###################################################################
        # Perform a single async H2D copy for all device tensors
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
        torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
            ungathered_input=ungathered_input_ids,
            gather_ids=self.get_arg("_gather_idx", truncate=True),
            mask_indices=self.get_arg("_mask_scatter_indices", truncate=True),
            out=self.get_arg("input_ids", truncate=True, unflatten=False),
        )

    # TODO: remove once https://github.com/NVIDIA/TensorRT-LLM/issues/9878 is fixed and
    # logits gather is enabled by default (only keep squeeze_logits)
    @nvtx_range("ad_maybe_gather_and_squeeze")
    def maybe_gather_and_squeeze(self, logits: torch.Tensor) -> torch.Tensor:
        """Maybe gather the logits if logits have not been gathered yet."""
        num_tokens = logits.shape[0] * logits.shape[1]
        num_tokens_to_gather, gather_required = self.get_arg("tokens_gather_info_host").tolist()
        if gather_required and num_tokens_to_gather < num_tokens:
            logits = torch.ops.auto_deploy.gather_tokens(
                logits,
                self.get_arg("token_gather_indices"),
                self.get_arg("tokens_gather_info_host"),
            )
        return self.flatten(logits)

    @nvtx_range("ad_unnest_sequences")
    def unnest_sequences(self, t_nested: torch.Tensor) -> List[torch.Tensor]:
        t_squeezed = self.flatten(t_nested)
        seq_len_list = self.get_arg("seq_len_host", truncate=True).tolist()
        return list(torch.split(t_squeezed, seq_len_list))

    def register_host_prepare_for_attention_forward(
        self, host_function: PrepareMetadataHostCallable, args: List[str]
    ):
        self._host_prepare_functions.append((host_function, args))
        # Ensure all host-prepare args are stored to InputBuffer via _requires_copy
        for arg in args:
            self._active_host_prep_args.add(arg)

    def run_host_prepare_for_attention_forward(self) -> None:
        for host_function, args in self._host_prepare_functions:
            host_function(**{arg: self.get_arg(arg) for arg in args})


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
