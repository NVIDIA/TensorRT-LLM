"""V23 VA-probe diagnostic helper.

One-shot ``cudaPointerGetAttributes`` + VA printer for tracking which
device / host-pinned tensor lands in a given VA range. Used to identify
the source of UVM page-fault ping-pong on B200 with HMM coherence.

Enabled by ``AD_VA_PROBE=1``. Default off; when off, every entry-point
function is a single env-var lookup + return.

Output goes to stderr (so it lands in server.log) with prefix
``[AD_VA_PROBE]``. Each label is printed at most once per process to
avoid log spam during steady-state decode.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any, Optional, Tuple

import torch

_ENABLED: Optional[bool] = None
_LABEL_COUNTS: dict[str, int] = {}
_CUDART: Optional[ctypes.CDLL] = None


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("AD_VA_PROBE", "0") == "1"
    return _ENABLED


def _cudart() -> Optional[ctypes.CDLL]:
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    try:
        lib = ctypes.CDLL("libcudart.so")
        lib.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaPointerGetAttributes.restype = ctypes.c_int
        lib.cudaGetLastError.restype = ctypes.c_int
        _CUDART = lib
        return _CUDART
    except Exception:
        return None


_MEM_TYPE_STR = {
    -2: "SkippedDuringCapture",
    0: "Unregistered",
    1: "Host",
    2: "Device",
    3: "Managed",
}


def _ptr_attrs(ptr: int) -> Tuple[int, int]:
    """Return (mem_type, device) via cudaPointerGetAttributes; (-1, -1) on fail.

    Skipped during CUDA stream capture (CUDA API restriction; calling it
    during capture corrupts the captured stream and segfaults later).
    """
    lib = _cudart()
    if lib is None or ptr == 0:
        return (-1, -1)
    try:
        if torch.cuda.is_current_stream_capturing():
            return (-2, -1)  # sentinel for "capture-in-progress, skipped"
    except Exception:
        pass
    buf = ctypes.create_string_buffer(64)
    rc = lib.cudaPointerGetAttributes(buf, ctypes.c_void_p(ptr))
    try:
        lib.cudaGetLastError()
    except Exception:
        pass
    if rc != 0:
        return (-1, -1)
    mem_type = int.from_bytes(buf.raw[0:4], "little")
    device = int.from_bytes(buf.raw[4:8], "little", signed=True)
    return (mem_type, device)


def _format_tensor(name: str, t: Any) -> str:
    if t is None:
        return f"  {name}: None"
    if not isinstance(t, torch.Tensor):
        return f"  {name}: non-tensor type={type(t).__name__} value={t!r}"
    ptr = t.data_ptr()
    size = t.element_size() * t.numel()
    try:
        is_pinned = bool(t.is_pinned()) if not t.is_cuda else False
    except Exception:
        is_pinned = False
    mem_type, device = _ptr_attrs(ptr)
    mem_str = _MEM_TYPE_STR.get(mem_type, f"?({mem_type})")
    return (
        f"  {name}: ptr=0x{ptr:x} end=0x{ptr + size:x} size={size}B "
        f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} pinned={is_pinned} "
        f"mem_type={mem_str}"
    )


def va_probe(label: str, *args, _times: int = 1, **kwargs) -> None:
    """Print VA + cudaPointerGetAttributes for tensors at this label up to ``_times`` times.

    Accepts positional args (will be labeled arg0/arg1/...) and named kwargs.
    Use kwargs for readability. Set ``_times`` > 1 to sample a per-iter callsite
    over multiple iters and confirm VA stability across the PyTorch caching
    allocator.
    """
    if not _enabled():
        return
    n = _LABEL_COUNTS.get(label, 0)
    if n >= _times:
        return
    _LABEL_COUNTS[label] = n + 1

    lines = [f"[AD_VA_PROBE] {label} (call #{n + 1}/{_times})"]
    for i, a in enumerate(args):
        lines.append(f"[AD_VA_PROBE] {_format_tensor(f'arg{i}', a)}")
    for k, v in kwargs.items():
        lines.append(f"[AD_VA_PROBE] {_format_tensor(k, v)}")
    print("\n".join(lines), file=sys.stderr, flush=True)


def reset_seen() -> None:
    _LABEL_COUNTS.clear()
