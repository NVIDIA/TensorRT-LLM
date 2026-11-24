# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver function attributes of compiled CuTe DSL kernels (SM100/SM103/SM107).

An SM splits its unified L1 / shared memory when a CTA lands on it while it is idle, and keeps
that split while any CTA runs on it. The split follows the preferred shared-memory carveout of the
kernel (``CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT``, percent of the maximum shared
memory; ``-1``, no preference, gives a kernel with little shared memory a small split). A CTA of
another kernel is placed on a busy SM if its kernel has no preference or prefers a split at least
as large as the SM's current one (measured on SM107). So a small kernel that runs first and should
leave room for a large-shared-memory kernel launched beside it (a programmatic dependent, or a
kernel on another stream) needs carveout 100; with ``-1`` the large kernel's CTAs wait for it to
exit.

The CuTe DSL passes a launch's ``preferred_smem_carveout`` to the driver. A launch with
``min_blocks_per_mp > 1`` and no carveout gets one derived by the DSL from the kernel's own shared
memory, just enough for that many of its CTAs.

The helpers here read and write these attributes on the kernels of a compiled CuTe DSL function
(``cute.compile`` result) through the driver API, without recompiling:

* :func:`set_shared_carveout` writes the preferred carveout (and optionally the dynamic
  shared-memory limit) on every kernel of a compiled function;
* :func:`apply_shared_carveout` does it only under ``TRTLLM_K3_CUTE_CARVEOUT_FIX=1`` and returns
  the compiled function, so a compile site can wrap its ``cute.compile`` call;
* :func:`kernel_attributes` reports the attributes of every kernel (tests and probes).

The attributes stick until the function's CUDA library is unloaded; set them before graph capture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cuda.bindings.driver as cuda
import torch

from ..features import active as _features

#: Preferred carveout of the maximum shared-memory split (percent).
MAX_SHARED_CARVEOUT = 100

_ATTR = cuda.CUfunction_attribute
_CARVEOUT_ATTR = _ATTR.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
_MAX_DYNAMIC_SMEM_ATTR = _ATTR.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
_REPORTED_ATTRS = {
    "carveout": _CARVEOUT_ATTR,
    "max_dynamic_smem": _MAX_DYNAMIC_SMEM_ATTR,
    "static_smem": _ATTR.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    "registers": _ATTR.CU_FUNC_ATTRIBUTE_NUM_REGS,
    "max_threads": _ATTR.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
}


def carveout_fix_enabled() -> bool:
    """``features.cute_carveout_fix``: K3 kernels that run ahead of large-shared-memory kernels (the
    attn_res kernels, L2 prefetch tables built without a carveout) get carveout 100 through the
    driver API (:func:`apply_shared_carveout`)."""
    return _features().cute_carveout_fix


def _checked(result: tuple, what: str) -> Any:
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


def _device_index(device: Optional[int | torch.device]) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    return int(device)


def _kernels(compiled: Any, device_index: int) -> List[Tuple[str, Any]]:
    """``(name, CUkernel)`` of every kernel of ``compiled`` (loaded on ``device_index`` first)."""
    with torch.cuda.device(device_index):
        compiled.to(device_index)
    jit_module = getattr(compiled, "jit_module", None)
    libraries = list(getattr(jit_module, "cuda_library", None) or [])
    if not libraries:
        raise RuntimeError(f"{type(compiled).__name__} has no loaded CUDA library")
    kernels = []
    for library in libraries:
        handle = cuda.CUlibrary(int(library))
        count = _checked(cuda.cuLibraryGetKernelCount(handle), "cuLibraryGetKernelCount")
        for kernel in _checked(
            cuda.cuLibraryEnumerateKernels(count, handle), "cuLibraryEnumerateKernels"
        ):
            name = _checked(cuda.cuKernelGetName(kernel), "cuKernelGetName")
            kernels.append((name.decode() if isinstance(name, bytes) else str(name), kernel))
    return kernels


def set_shared_carveout(
    compiled: Any,
    carveout: int,
    device: Optional[int | torch.device] = None,
    max_dynamic_smem: Optional[int] = None,
) -> None:
    """Write the preferred shared-memory carveout on every kernel of ``compiled``.

    Args:
        compiled: a ``cute.compile`` result (CUDA dialect executor).
        carveout: preferred carveout in percent of the maximum shared memory (0-100), or -1 for
            the driver's choice.
        device: device whose kernels are updated (default: the current device).
        max_dynamic_smem: if given, also ``CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES``.
    """
    if not -1 <= carveout <= 100:
        raise ValueError(f"carveout={carveout} must be a percentage or -1")
    index = _device_index(device)
    cu_device = _checked(cuda.cuDeviceGet(index), "cuDeviceGet")
    for _, kernel in _kernels(compiled, index):
        if max_dynamic_smem is not None:
            _checked(
                cuda.cuKernelSetAttribute(
                    _MAX_DYNAMIC_SMEM_ATTR, max_dynamic_smem, kernel, cu_device
                ),
                "cuKernelSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
            )
        _checked(
            cuda.cuKernelSetAttribute(_CARVEOUT_ATTR, carveout, kernel, cu_device),
            "cuKernelSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT)",
        )


def apply_shared_carveout(
    compiled: Any, carveout: Optional[int], device: Optional[int | torch.device] = None
) -> Any:
    """``compiled``, with ``carveout`` written by :func:`set_shared_carveout` when
    ``TRTLLM_K3_CUTE_CARVEOUT_FIX=1`` and ``carveout`` is not ``None``."""
    if carveout is not None and carveout_fix_enabled():
        set_shared_carveout(compiled, carveout, device)
    return compiled


def kernel_attributes(
    compiled: Any, device: Optional[int | torch.device] = None
) -> List[Dict[str, Any]]:
    """Name and driver attributes (:data:`_REPORTED_ATTRS`) of every kernel of ``compiled`` as the
    current context sees them."""
    index = _device_index(device)
    out = []
    with torch.cuda.device(index):
        for name, kernel in _kernels(compiled, index):
            function = _checked(cuda.cuKernelGetFunction(kernel), "cuKernelGetFunction")
            entry: Dict[str, Any] = {"name": name}
            for label, attribute in _REPORTED_ATTRS.items():
                entry[label] = _checked(
                    cuda.cuFuncGetAttribute(attribute, function), f"cuFuncGetAttribute({label})"
                )
            out.append(entry)
    return out


__all__ = [
    "MAX_SHARED_CARVEOUT",
    "apply_shared_carveout",
    "carveout_fix_enabled",
    "kernel_attributes",
    "set_shared_carveout",
]
