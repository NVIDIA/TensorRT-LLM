"""Device abstraction for CUDA, Apple Silicon (MPS), and CPU backends.
Centralizes backend detection and the handful of APIs that genuinely differ
across accelerators (synchronization, allocator cache, memory queries, RNG
state). Selection order is CUDA -> MPS -> CPU.
CUDA-only optimizations (FlashAttention, Triton blockwise FP8/FP6,
bitsandbytes, NCCL) are gated at their call sites, not here. MPS in particular
has no ``float64`` support and no fp8 dtype support; use
:func:`highest_precision_float` to stay within what the backend can represent.
"""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)

DeviceSpec = torch.device | None


def is_mps_available() -> bool:
    """Return whether PyTorch can use the Apple Metal/MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def get_preferred_device(local_rank: int | None = None) -> torch.device:
    """Prefer CUDA, then MPS, then CPU.
    ``local_rank`` is only meaningful for CUDA multi-process launches. MPS exposes
    a single logical device in PyTorch, so rank-based indexing is not used there.
    """
    if torch.cuda.is_available():
        index = torch.cuda.current_device() if local_rank is None else local_rank
        return torch.device("cuda", index)
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceSpec = None, *, local_rank: int | None = None) -> torch.device:
    """Return *device*, or the best available accelerator when it is ``None``."""
    if device is None:
        return get_preferred_device(local_rank=local_rank)
    return device


def supports_float64(device: DeviceSpec) -> bool:
    """Return whether *device* can represent ``torch.float64``.
    MPS has no double-precision support; CUDA and CPU do.
    """
    return resolve_device(device).type != "mps"


def highest_precision_float(device: DeviceSpec) -> torch.dtype:
    """Return the widest float the backend supports: ``float64`` on CUDA/CPU,
    ``float32`` on MPS.
    Use for numerically sensitive accumulators (e.g. sampler ODE math) that
    request double precision but must degrade gracefully on MPS.
    """
    return torch.float64 if supports_float64(device) else torch.float32


def allow_async_transfer(device: torch.device) -> bool:
    """Whether a fused host-to-device + dtype-convert transfer to *device* may be non-blocking.
    Only CUDA orders a non-blocking transfer against the consuming kernel
    (same-stream ordering). On MPS the consumer is not fenced against it and
    may read the destination before the transfer lands, silently corrupting the
    consuming computation, so transfers block everywhere but CUDA.
    """
    return device.type == "cuda"


def upload_from_host(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a host tensor to *device*, without blocking the host where the backend allows it.
    The transfer reads a pinned copy, so ``host`` may be modified as soon as this returns. A tensor
    that is not on the host is moved with ``.to``.
    """
    if not allow_async_transfer(device) or host.device.type != "cpu":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)


def synchronize_device(device: DeviceSpec = None) -> None:
    """Synchronize CUDA or MPS work if the selected backend supports it."""
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and is_mps_available():
        torch.mps.synchronize()


def empty_device_cache(device: DeviceSpec = None) -> None:
    """Release cached allocator memory for CUDA or MPS."""
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and is_mps_available():
        torch.mps.empty_cache()


def cuda_activation_budget_bytes(device: DeviceSpec = None) -> int:
    """Bytes still available for new CUDA allocations in this process.
    Honors ``torch.cuda.set_per_process_memory_fraction``: ``mem_get_info`` reports
    raw device free memory and does *not* shrink under a fraction cap, so we use
    ``fraction * device_total - memory_reserved`` (also capped by raw free).
    """
    if not torch.cuda.is_available():
        return 0
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return 0
    idx = resolved.index if resolved.index is not None else torch.cuda.current_device()
    free_raw, _total_raw = torch.cuda.mem_get_info(idx)
    props_total = torch.cuda.get_device_properties(idx).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction(idx)
    reserved = torch.cuda.memory_reserved(idx)
    under_fraction = max(0, int(fraction * props_total) - int(reserved))
    return min(int(free_raw), under_fraction)


def cuda_allocatable_bytes(device: DeviceSpec = None) -> int:
    """Bytes a new CUDA allocation in this process can still get, counting the allocator's cache.
    Like :func:`cuda_activation_budget_bytes`, but reserved blocks that hold no tensor count as
    available: the caching allocator serves new allocations from them and releases them to the
    driver before an allocation fails. ``fraction * device_total - memory_allocated``, capped by
    raw free memory plus those cached blocks.
    """
    if not torch.cuda.is_available():
        return 0
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return 0
    idx = resolved.index if resolved.index is not None else torch.cuda.current_device()
    free_raw, _total_raw = torch.cuda.mem_get_info(idx)
    props_total = torch.cuda.get_device_properties(idx).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction(idx)
    allocated = int(torch.cuda.memory_allocated(idx))
    cached = max(0, int(torch.cuda.memory_reserved(idx)) - allocated)
    return max(0, min(int(free_raw) + cached, int(fraction * props_total) - allocated))


def mps_activation_budget_bytes(device: DeviceSpec = None) -> int:
    """Bytes still available for new MPS allocations in this process.
    ``recommended_max_memory`` is Metal's recommended working-set size, which is
    below physical RAM (unified memory is shared with the rest of the system), so it
    is the conservative ceiling to plan against. ``driver_allocated_memory`` -- not
    ``current_allocated_memory`` -- is subtracted because it counts blocks the
    allocator is holding in its cache, mirroring the CUDA path's use of
    ``memory_reserved``.
    Unlike the CUDA ceiling this one is advisory: exceeding Metal's recommendation
    degrades to swapping rather than raising, so callers get a budget that errs
    small instead of a hard limit.
    """
    if not is_mps_available():
        return 0
    if resolve_device(device).type != "mps":
        return 0
    recommended = int(torch.mps.recommended_max_memory())
    allocated = int(torch.mps.driver_allocated_memory())
    return max(0, recommended - allocated)


def activation_budget_bytes(device: DeviceSpec = None) -> int:
    """Bytes available for new activations on *device*, across backends.
    Use this rather than the per-backend helpers when sizing work to fit in memory
    (e.g. DiffVAE decode tiling): a CUDA-only probe silently yields a zero budget
    elsewhere, which reads as "nothing fits" instead of "unknown".
    Returns 0 on CPU. Host RAM is not probed, so callers that support CPU execution
    must treat 0 as "no budget information" and pick their own fallback rather than
    concluding that no work fits.
    """
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        return cuda_activation_budget_bytes(resolved)
    if resolved.type == "mps":
        return mps_activation_budget_bytes(resolved)
    return 0


def cleanup_accelerator_memory(device: DeviceSpec = None) -> None:
    """Run Python GC and release CUDA/MPS allocator caches."""
    gc.collect()
    empty_device_cache(device)
    synchronize_device(device)
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logger.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)
