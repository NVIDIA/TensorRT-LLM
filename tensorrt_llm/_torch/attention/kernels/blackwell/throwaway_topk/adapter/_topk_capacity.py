# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Address-stable step inputs and pool ownership for MQA TopK."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from threading import RLock
from typing import Any
import weakref
import torch


_deferred_metadata_modules: list[Any] = []
_deferred_metadata_lock = RLock()
_logger = logging.getLogger(__name__)


def _retire_metadata_module(module):
    """Queue retirement without calling CUDA from a finalizer thread."""
    with _deferred_metadata_lock:
        _deferred_metadata_modules.append(module)


def _drain_deferred_metadata_modules():
    """Release queued libraries at an eager API boundary.

    Capture status is stream-local; callers must not invoke this boundary while
    another stream is undergoing global capture.
    """
    global _deferred_metadata_modules
    with _deferred_metadata_lock:
        if not _deferred_metadata_modules:
            return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except RuntimeError:
        return
    with _deferred_metadata_lock:
        pending, _deferred_metadata_modules = _deferred_metadata_modules, []
    for module in pending:
        try:
            module.unload()
        except Exception:
            _logger.exception("Failed to unload a retired CUDA metadata module")


def _bind_metadata_module_lifetime(compiled):
    """Defer metadata module unloading to an eager planning boundary.

    The finalizer retains only the module, avoiding a callable reference cycle
    and preventing cyclic GC from unloading a library during graph capture.
    """
    module = getattr(compiled, "jit_module", None)
    if module is not None:
        weakref.finalize(compiled, _retire_metadata_module, module)
    return compiled


def _validate_step_shape(capacity, num_query_rows, next_n):
    if (
        type(num_query_rows) is not int
        or not 0 <= num_query_rows <= capacity.max_chunk_size
    ):
        raise ValueError("num_query_rows must fit max_chunk_size")
    if next_n is not None and (type(next_n) is not int or next_n not in (1, 2, 4)):
        raise ValueError("next_n must be None, 1, 2 or 4")


@dataclass(frozen=True)
class _TopKCapacity:
    """Immutable host-side storage bounds for one TopK planning bucket.

    Context lengths count original tokens; physical lengths are computed by
    flooring each request separately at the selected compression ratio.
    ``max_chunk_size`` bounds supplied Q storage across the entire step, while
    ``max_batch_size`` bounds the number of requests. Fields are positive int32
    values, with compression ratio 1/4; bools are rejected.
    Scorer specialization and device resource checks occur when building the chain.
    """

    num_heads: int
    head_dim: int
    compression_ratio: int
    max_context_length: int
    max_chunk_size: int
    max_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "num_heads",
            "head_dim",
            "max_context_length",
            "max_chunk_size",
            "max_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive int32 value")
        if self.head_dim % 32:
            raise ValueError("head_dim must be divisible by 32")
        if type(self.compression_ratio) is not int or self.compression_ratio not in (
            1,
            4,
        ):
            raise ValueError("compression_ratio must be 1 or 4")


class _CapacityStepInputs:
    """Own stable payload/prefix buffers and sanitized per-step row metadata."""

    def __init__(
        self,
        capacity: _TopKCapacity,
        *,
        device,
        num_keys,
        qkv_dtype,
        weights_dtype,
        refresh_type,
    ):
        from ._topk_runtime import _require_cute

        _require_cute()
        from cutlass.cute.runtime import from_dlpack

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("capacity device must be CUDA")
        if type(num_keys) is not int or not 0 <= num_keys < 2**31:
            raise ValueError("num_keys must be a nonnegative int32 value")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("capacity buffers must be prepared outside capture")
            device = torch.device("cuda", torch.cuda.current_device())
        self.capacity = capacity
        self.num_keys = num_keys
        self.device = device
        self.q = torch.empty(
            (
                capacity.max_chunk_size,
                capacity.num_heads,
                capacity.head_dim // (2 if qkv_dtype == "mxfp4" else 1),
            ),
            dtype=torch.uint8 if qkv_dtype == "mxfp4" else torch.float8_e4m3fn,
            device=self.device,
        )
        self.weights = torch.empty(
            (capacity.max_chunk_size, capacity.num_heads),
            dtype=weights_dtype,
            device=self.device,
        )
        prefix_shape = (capacity.max_batch_size + 1,)
        self.qo_indptr = torch.zeros(
            prefix_shape, dtype=torch.int32, device=self.device
        )
        self.kv_indptr = torch.zeros_like(self.qo_indptr)
        self.step_shape = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._written_step_shape: tuple[int, int] | None = None
        self._skip_repeated_step_shape = True
        self.safe_qo_indptr = torch.zeros_like(self.qo_indptr)
        self.physical_kv_indptr = torch.zeros_like(self.qo_indptr)
        self.active_shape = torch.zeros_like(self.step_shape)
        self.row_ends = torch.zeros(
            capacity.max_chunk_size, dtype=torch.int32, device=self.device
        )
        self.row_requests = torch.full_like(self.row_ends, -1)
        self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
        tensors = (
            self.qo_indptr,
            self.kv_indptr,
            self.step_shape,
            self.safe_qo_indptr,
            self.physical_kv_indptr,
            self.active_shape,
            self.row_ends,
            self.row_requests,
            self.error,
        )
        self._args = tuple(from_dlpack(tensor, assumed_align=4) for tensor in tensors)
        self._refresh_kernel = refresh_type(
            capacity.max_context_length,
            capacity.max_chunk_size,
            capacity.max_batch_size,
            capacity.compression_ratio,
            num_keys,
        )
        self._compiled = None
        self._next_n = 0
        from ._topk_host_hints import _HostHintValidation

        self._host_hint_validation = _HostHintValidation(capacity, self.device)

    def input_buffers(self) -> dict[str, torch.Tensor]:
        return dict(
            q=self.q,
            weights=self.weights,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.kv_indptr,
        )

    def _payload_copies(self, q, weights):
        """Validate layer payload layouts without enqueuing any copies."""
        c = self.capacity
        if not all(isinstance(t, torch.Tensor) for t in (q, weights)):
            raise ValueError("step inputs must be tensors")
        if (
            q.device != self.device
            or q.dtype != self.q.dtype
            or q.ndim != 3
            or q.shape[1:] != self.q.shape[1:]
            or q.shape[0] > c.max_chunk_size
            or not q.is_contiguous()
            or weights.device != self.device
            or weights.dtype != self.weights.dtype
            or weights.shape != (q.shape[0], c.num_heads)
            or weights.stride(1) != 1
            or weights.stride(0) < c.num_heads
            or (weights.dtype == torch.float32 and not weights.is_contiguous())
        ):
            raise ValueError("Q and weights must match the planned shapes and dtypes")
        return (
            (self.q[: q.shape[0]], q),
            (self.weights[: q.shape[0]], weights),
        )

    def _prefix_copies(self, qo_indptr, kv_indptr):
        """Validate step prefix layouts without reading their device values."""
        if not all(isinstance(t, torch.Tensor) for t in (qo_indptr, kv_indptr)):
            raise ValueError("step inputs must be tensors")
        if (
            qo_indptr.device != self.device
            or kv_indptr.device != self.device
            or qo_indptr.dtype != torch.int32
            or kv_indptr.dtype != torch.int32
            or qo_indptr.ndim != 1
            or kv_indptr.shape != qo_indptr.shape
            or not 1 <= qo_indptr.numel() <= self.capacity.max_batch_size + 1
            or not qo_indptr.is_contiguous()
            or not kv_indptr.is_contiguous()
        ):
            raise ValueError("indptrs must be matching contiguous CUDA int32 [B+1]")
        return (
            (self.qo_indptr[: qo_indptr.numel()], qo_indptr),
            (self.kv_indptr[: kv_indptr.numel()], kv_indptr),
        )

    def _owner_ranges(self):
        """(byte range, data_ptr) of every staging buffer.

        The staging buffers are allocated once, so their ranges are computed once and reused
        until the set of buffers (identity or storage) changes.
        """
        from ._topk_runtime import _byte_range

        owners = tuple(self.input_buffers().values())
        key = tuple((id(owner), owner.data_ptr()) for owner in owners)
        cached = getattr(self, "_owner_range_cache", None)
        if cached is None or cached[0] != key:
            ranges = tuple((_byte_range(owner), owner.data_ptr()) for owner in owners)
            cached = (key, ranges)
            self._owner_range_cache = cached
        return cached[1]

    def _copy_inputs(self, copies) -> None:
        """Reject all cross-aliases before enqueuing the first staging copy."""
        from ._topk_runtime import _byte_range, _overlap

        owners = self._owner_ranges()
        # A staging copy cannot overwrite another source before it is consumed.
        for destination, source in copies:
            source_range = _byte_range(source)
            source_ptr = source.data_ptr()
            destination_ptr = destination.data_ptr()
            for owner_range, owner_ptr in owners:
                if _overlap(source_range, owner_range) and not (
                    owner_ptr == destination_ptr == source_ptr
                ):
                    raise ValueError("step inputs must not cross-alias staging buffers")
        # All host metadata/alias checks precede every copy.
        for destination, source in copies:
            if destination.data_ptr() != source.data_ptr():
                destination.copy_(source)

    def stage_payload(self, q, weights) -> None:
        """Stage one layer without changing step metadata."""
        self._copy_inputs(self._payload_copies(q, weights))

    def stage_step(
        self,
        qo_indptr,
        kv_indptr,
        *,
        num_query_rows,
        next_n=None,
        _shape_validated=False,
    ):
        """Stage shared prefixes/map and publish the supplied Q extent."""
        if not _shape_validated:
            _validate_step_shape(self.capacity, num_query_rows, next_n)
        copies = self._prefix_copies(qo_indptr, kv_indptr)
        self._copy_inputs(copies)
        step_shape = (num_query_rows, qo_indptr.numel() - 1)
        self._next_n = 0 if next_n is None else next_n
        # Replayed captures can write a shape without updating the host cache.
        # Once captured, always publish it, including on repeated eager steps.
        if torch.cuda.is_current_stream_capturing():
            self._skip_repeated_step_shape = False
        if not self._skip_repeated_step_shape or step_shape != self._written_step_shape:
            # CUDA is little-endian: one int64 fill publishes both int32 fields.
            self.step_shape.view(torch.int64).fill_(step_shape[0] | step_shape[1] << 32)
            self._written_step_shape = step_shape

    def refresh(self) -> None:
        from ._topk_runtime import _current_stream

        self.prepare()
        assert self._compiled is not None
        with torch.cuda.device(self.device):
            self._compiled(*self._args, self._next_n, _current_stream(self.device))
            self._host_hint_validation.check(self)

    @property
    def is_prepared(self):
        return self._compiled is not None

    def prepare(self) -> None:
        """Compile metadata consumers outside capture without moving their buffers."""
        from ._topk_runtime import _compile_for_device, _current_stream

        if self.is_prepared:
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("metadata must be prepared outside capture")
            compiled = _bind_metadata_module_lifetime(
                _compile_for_device(
                    self.device,
                    self._refresh_kernel,
                    *self._args,
                    0,
                    _current_stream(self.device),
                )
            )
        # Publish only after compilation succeeds, so a failed first preparation
        # can be retried with the same staging addresses.
        self._compiled = compiled
