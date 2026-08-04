"""Opt-in pinned staging for weight-load host-to-device copies.

Some driver/platform combinations degrade pageable H2D copies during
checkpoint loading into extreme slowdowns (observed on GB300 with driver
590.48.01: one rank per node stalls in ``Loading weights`` for 36 min to
4+ hours, no errors). Staging through pinned memory avoids the affected
driver path.

With ``TRTLLM_PINNED_WEIGHT_STAGING=1``, ``staging_scope()`` reroutes
pageable CPU->CUDA ``Tensor.to``/``Tensor.copy_`` through a reusable
per-dtype pinned buffer for the duration of the scope. On exit the original
methods are restored and the buffers are freed, leaving no pinned-memory
footprint behind. Default off: without the env var the scope is a no-op.
"""

import os
import threading

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

_lock = threading.RLock()
_depth = 0
_bufs = {}
_orig_to = None
_orig_copy = None


def _enabled() -> bool:
    return os.environ.get("TRTLLM_PINNED_WEIGHT_STAGING", "0") == "1" and prefer_pinned()


def _pinned_clone(src: torch.Tensor) -> torch.Tensor:
    s = src.contiguous()
    n = s.numel()
    b = _bufs.get(s.dtype)
    if b is None or b.numel() < n:
        _bufs[s.dtype] = b = torch.empty(n, dtype=s.dtype, pin_memory=prefer_pinned())
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
        if (
            self.device.type == "cpu"
            and self.layout == torch.strided
            and self.numel() > 0
            and not self.is_pinned()
            and _target_is_cuda(args, kwargs)
        ):
            # The staging buffer is reused, so the copy must stay
            # synchronous. non_blocking/copy may be passed positionally as
            # bools; force them to False.
            args = tuple(False if isinstance(a, bool) else a for a in args)
            kwargs.pop("non_blocking", None)
            with _lock:
                return _orig_to(_pinned_clone(self), *args, **kwargs)
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        logger.debug(f"[pinned_weight_staging] to() fallback: {e}")
    return _orig_to(self, *args, **kwargs)


def _staged_copy(self, src, non_blocking=False):
    try:
        if (
            self.device.type == "cuda"
            and isinstance(src, torch.Tensor)
            and src.device.type == "cpu"
            and src.layout == torch.strided
            and src.numel() > 0
            and not src.is_pinned()
        ):
            with _lock:
                return _orig_copy(self, _pinned_clone(src), False)
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        logger.debug(f"[pinned_weight_staging] copy_() fallback: {e}")
    return _orig_copy(self, src, non_blocking)


class _StagingScope:
    """Re-entrant: nested scopes share one installation, freed at depth 0."""

    def __enter__(self):
        global _depth, _orig_to, _orig_copy
        # Record activation so a mid-scope env change cannot skip cleanup.
        self._active = _enabled()
        if not self._active:
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
                    "CPU->CUDA weight copies go through pinned buffers"
                )
        return self

    def __exit__(self, exc_type, exc, tb):
        global _depth, _orig_to, _orig_copy
        if not getattr(self, "_active", False):
            return False
        with _lock:
            _depth -= 1
            if _depth <= 0:
                _depth = 0
                if _orig_to is not None:
                    torch.Tensor.to = _orig_to
                if _orig_copy is not None:
                    torch.Tensor.copy_ = _orig_copy
                freed = sum(b.numel() * b.element_size() for b in _bufs.values())
                _bufs.clear()
                # Return freed pinned pages to the OS, not just to torch's
                # caching host allocator.
                hec = getattr(torch.cuda, "host_empty_cache", None) or getattr(
                    torch._C, "_host_emptyCache", None
                )
                if hec is not None:
                    try:
                        hec()
                    except (RuntimeError, AttributeError) as e:
                        logger.debug(f"[pinned_weight_staging] host_empty_cache: {e}")
                logger.info(
                    "[pinned_weight_staging] staging scope exit: released "
                    f"{freed / (1 << 30):.2f} GiB of pinned staging buffers"
                )
        return False


def staging_scope() -> _StagingScope:
    return _StagingScope()
