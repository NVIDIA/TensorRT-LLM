"""Exact host byte buffers, aligned unit layouts, and capacity checks for streamed tensor units."""

import mmap
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import UNIT_ALIGN, StateDictMetadata, TensorMetadata, Unit, active_dtype

_EXACT_PIN_BYTES = 2 << 20


def host_bytes(nbytes: int, *, pin: bool = False) -> torch.Tensor:
    """Return an uninitialized host byte buffer of exactly ``nbytes``, page-locked when ``pin``.
    torch's pinned allocator rounds every request up to a power of two, so a 4.4 GiB unit would hold 8 GiB.
    Pinned buffers from ``_EXACT_PIN_BYTES`` up are anonymous mappings registered with CUDA instead; the
    registration ends when the last view of the buffer is released.
    """
    if not pin or nbytes < _EXACT_PIN_BYTES:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    buffer = torch.frombuffer(region, dtype=torch.uint8)
    cudart = torch.cuda.cudart()
    if int(cudart.cudaHostRegister(buffer.data_ptr(), nbytes, 0)) != 0:
        raise RuntimeError(f"could not page-lock {nbytes} host bytes")
    # The mapping outlives every tensor view of it, so it is unregistered just before it is unmapped.
    weakref.finalize(region, cudart.cudaHostUnregister, buffer.data_ptr()).atexit = False
    return buffer


def _storage_bytes(
    name: str, clean: TensorMetadata, strides: Mapping[str, tuple[int, ...]] | None, requested_dtype: torch.dtype | None
) -> int:
    """Bytes one tensor's storage covers at its active dtype, stride gaps included."""
    shape = clean[0]
    stride = None if strides is None else strides.get(name)
    elements = (
        shape.numel()
        if stride is None or not shape.numel()
        else 1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
    )
    return elements * active_dtype(clean, requested_dtype).itemsize


def unit_offsets(
    metadata: StateDictMetadata,
    strides: Mapping[str, tuple[int, ...]] | None = None,
    requested_dtype: torch.dtype | None = None,
) -> dict[str, int]:
    """Return each tensor's byte offset in one contiguous unit buffer, aligned to ``UNIT_ALIGN``.
    Tensors are sized at their active dtypes and projected strides. Keys follow the unit's metadata order, which
    is also ascending offset order: the last key's tensor ends the buffer.
    """
    offsets: dict[str, int] = {}
    end = 0
    for name, clean in metadata.items():
        offsets[name] = end + (-end) % UNIT_ALIGN
        end = offsets[name] + _storage_bytes(name, clean, strides, requested_dtype)
    return offsets


def _buffer_end(
    offsets: Mapping[str, int],
    metadata: StateDictMetadata,
    strides: Mapping[str, tuple[int, ...]] | None,
    requested_dtype: torch.dtype | None,
) -> int:
    """End of the last tensor in ``unit_offsets`` order: the bytes the unit buffer needs."""
    if not offsets:
        return 0
    last = next(reversed(offsets))
    return offsets[last] + _storage_bytes(last, metadata[last], strides, requested_dtype)


def layout_bytes(
    metadata: StateDictMetadata,
    strides: Mapping[str, tuple[int, ...]] | None = None,
    requested_dtype: torch.dtype | None = None,
) -> int:
    """Return the byte capacity for one aligned unit and its projected tensor strides.
    ``metadata`` describes clean tensors; ``requested_dtype`` sizes them at their ``active_dtype``.
    """
    return _buffer_end(unit_offsets(metadata, strides, requested_dtype), metadata, strides, requested_dtype)


def aligned_region_bytes(size: int) -> int:
    """Round a preceding buffer region up so the next tensor starts at ``UNIT_ALIGN``."""
    return size + (-size) % UNIT_ALIGN


def unit_views(
    raw: torch.Tensor,
    metadata: StateDictMetadata,
    strides: Mapping[str, tuple[int, ...]] | None = None,
    requested_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Make typed views of a unit's tensors in a caller-owned byte buffer without copying.
    ``metadata`` describes clean tensors; with ``requested_dtype`` the views have their ``active_dtype``.
    """
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("unit views need a contiguous one-dimensional uint8 buffer")
    offsets = unit_offsets(metadata, strides, requested_dtype)
    needed = _buffer_end(offsets, metadata, strides, requested_dtype)
    if raw.numel() < needed:
        raise ValueError(f"buffer has {raw.numel()} bytes; unit needs {needed}")
    views: dict[str, torch.Tensor] = {}
    for name, clean in metadata.items():
        shape, dtype = clean[0], active_dtype(clean, requested_dtype)
        stride = None if strides is None else strides.get(name)
        values = raw.narrow(0, offsets[name], _storage_bytes(name, clean, strides, requested_dtype)).view(dtype)
        views[name] = values.reshape(shape) if stride is None else torch.as_strided(values, shape, stride)
    return views


@dataclass(frozen=True, slots=True)
class Capacity:
    """Buffer sizes and counts, allowing host and device storage to differ."""

    host_buffer_bytes: int
    device_buffer_bytes: int
    host_buffers: int
    device_buffers: int

    @property
    def buffer_bytes(self) -> int:
        """Largest buffer size, for callers that use equal host and device buffers."""
        return max(self.host_buffer_bytes, self.device_buffer_bytes)

    @property
    def host_bytes(self) -> int:
        """Total bytes reserved across host buffers."""
        return self.host_buffer_bytes * self.host_buffers

    @property
    def device_bytes(self) -> int:
        """Total bytes reserved across device buffers."""
        return self.device_buffer_bytes * self.device_buffers


def assess_capacity(
    units: Sequence[Unit],
    prefetch_depth: int = 1,
    *,
    host_budget: int | None = None,
    device_budget: int | None = None,
    strides: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
    host_buffers: int | None = None,
    device_buffers: int | None = None,
    host_buffer_bytes: int | None = None,
    device_buffer_bytes: int | None = None,
    requested_dtype: torch.dtype | None = None,
) -> Capacity:
    """Size the largest selected unit and the requested host and device buffers.
    Without count overrides, use equal host and device counts of ``prefetch_depth + 1`` buffers.
    Builders supply their actual counts and any different buffer sizes.
    """
    if not units or prefetch_depth < 0:
        raise ValueError("need units and a nonnegative prefetch depth")
    count = min(len(units), prefetch_depth + 1)
    if any(
        value is not None and value < 0
        for value in (host_buffers, device_buffers, host_buffer_bytes, device_buffer_bytes)
    ):
        raise ValueError("buffer counts and sizes must be nonnegative")
    largest = (
        max(
            layout_bytes(unit.metadata, None if strides is None else strides.get(unit.unit_id), requested_dtype)
            for unit in units
        )
        if host_buffer_bytes is None or device_buffer_bytes is None
        else 0
    )
    result = Capacity(
        largest if host_buffer_bytes is None else host_buffer_bytes,
        largest if device_buffer_bytes is None else device_buffer_bytes,
        count if host_buffers is None else host_buffers,
        count if device_buffers is None else device_buffers,
    )
    if (host_budget is not None and result.host_bytes > host_budget) or (
        device_budget is not None and result.device_bytes > device_budget
    ):
        raise ValueError(
            f"requested prefetch depth needs {result.host_bytes} host bytes and {result.device_bytes} device bytes"
        )
    return result
