"""Scoped pinned staging for weight-load host-to-device copies.

On some platform/driver configurations (observed on GB300 NVL72 with driver
590.48.01 under specific per-node profiling configurations), pageable
host-to-device copies issued during checkpoint materialization can degrade
into a pathological per-work-item polling crawl inside the CUDA driver,
stretching sub-second copies to tens of minutes. The visible symptom is a
weight-load "hang": typically one rank per node stops making progress in the
``Loading weights`` phase with zero errors, and eventually completes after
36 min to 4+ hours. Staging the copies through a pinned buffer routes them
onto the pinned-DMA path, which is unaffected.

Usage (gated by ``TRTLLM_PINNED_WEIGHT_STAGING=1``, default off):

    from tensorrt_llm._torch import pinned_weight_staging
    with pinned_weight_staging.staging_scope():
        ...load weights...

Inside the scope, every pageable CPU->CUDA ``Tensor.to`` / ``Tensor.copy_``
is rerouted through a per-dtype pinned staging buffer. The staged copy is
forced synchronous and lock-protected because the buffer is reused. On scope
exit the original tensor methods are restored and all staging buffers are
freed (best-effort ``host_empty_cache`` so the pinned pages actually return
to the OS), so pinned-host consumers that come up later -- e.g. the KV-cache
host offload pool sized by ``host_cache_size`` -- see zero residual
footprint, and steady-state serving copies are untouched.
"""
import os
import threading

import torch

from tensorrt_llm.logger import logger

_lock = threading.RLock()
_depth = 0
_bufs = {}
_orig_to = None
_orig_copy = None


def _enabled() -> bool:
    return os.environ.get("TRTLLM_PINNED_WEIGHT_STAGING", "0") == "1"


def _pinned_clone(src: torch.Tensor) -> torch.Tensor:
    s = src.contiguous()
    n = s.numel()
    b = _bufs.get(s.dtype)
    if b is None or b.numel() < n:
        _bufs[s.dtype] = b = torch.empty(n,
                                         dtype=s.dtype,
                                         pin_memory=True)
    v = b[:n].view(s.shape)
    v.copy_(s)
    return v


def _target_is_cuda(args, kwargs) -> bool:
    d = kwargs.get("device", None)
    if d is None:
        for a in args:
            if isinstance(a, torch.device):
                d = a
                break
            if isinstance(a, str):
                try:
                    d = torch.device(a)
                except (RuntimeError, ValueError):
                    d = None
                break
            if isinstance(a, torch.Tensor):
                d = a.device
                break
    elif not isinstance(d, torch.device):
        try:
            d = torch.device(d)
        except (RuntimeError, ValueError, TypeError):
            d = None
    return d is not None and getattr(d, "type", None) == "cuda"


def _staged_to(self, *args, **kwargs):
    try:
        if (self.device.type == "cpu" and self.layout == torch.strided
                and self.numel() > 0 and not self.is_pinned()
                and _target_is_cuda(args, kwargs)):
            # The staging buffer is reused; the copy must stay synchronous.
            kwargs.pop("non_blocking", None)
            with _lock:
                return _orig_to(_pinned_clone(self), *args, **kwargs)
    except Exception:
        pass
    return _orig_to(self, *args, **kwargs)


def _staged_copy(self, src, non_blocking=False):
    try:
        if (self.device.type == "cuda" and isinstance(src, torch.Tensor)
                and src.device.type == "cpu" and src.layout == torch.strided
                and src.numel() > 0 and not src.is_pinned()):
            with _lock:
                return _orig_copy(self, _pinned_clone(src), False)
    except Exception:
        pass
    return _orig_copy(self, src, non_blocking)


class staging_scope:
    """Reroute pageable CPU->CUDA copies through pinned staging for the
    dynamic extent of a weight load; restore and free everything on exit.

    Re-entrant: nested scopes share one installation, freed at depth 0.
    A no-op unless ``TRTLLM_PINNED_WEIGHT_STAGING=1``.
    """

    def __enter__(self):
        global _depth, _orig_to, _orig_copy
        if not _enabled():
            return self
        with _lock:
            _depth += 1
            if _depth == 1:
                _orig_to = torch.Tensor.to
                _orig_copy = torch.Tensor.copy_
                torch.Tensor.to = _staged_to
                torch.Tensor.copy_ = _staged_copy
                logger.info(
                    "[pinned_weight_staging] staging scope enter: pageable "
                    "CPU->CUDA weight copies go through pinned buffers")
        return self

    def __exit__(self, exc_type, exc, tb):
        global _depth, _orig_to, _orig_copy
        if not _enabled():
            return False
        with _lock:
            _depth -= 1
            if _depth <= 0:
                _depth = 0
                if _orig_to is not None:
                    torch.Tensor.to = _orig_to
                if _orig_copy is not None:
                    torch.Tensor.copy_ = _orig_copy
                freed = sum(b.numel() * b.element_size()
                            for b in _bufs.values())
                _bufs.clear()
                # Best-effort: push the freed pinned blocks out of torch's
                # caching host allocator so the pages return to the OS.
                hec = (getattr(torch.cuda, "host_empty_cache", None)
                       or getattr(torch._C, "_host_emptyCache", None))
                if hec is not None:
                    try:
                        hec()
                    except Exception:
                        pass
                logger.info(
                    "[pinned_weight_staging] staging scope exit: released "
                    f"{freed / (1 << 30):.2f} GiB of pinned staging buffers")
        return False
