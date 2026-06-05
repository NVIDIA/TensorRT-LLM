# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL user-buffer registration for VisualGen collectives.

Two complementary mechanisms are provided:

1. **CUMEM mode** (``enable_nccl_cumem()``): sets ``NCCL_CUMEM_ENABLE=1``
   before ``init_process_group``.  NCCL then allocates its internal scratch
   via cuMemCreate (VMM), enabling zero-copy for all collectives on VMM-backed
   tensors.  No per-tensor API call required.  Requires NCCL ≥ 2.21.

2. **Explicit registration** (``NCCLBufferRegistrar``): calls
   ``ncclCommRegister`` / ``ncclCommDeregister`` on a specific tensor via
   ctypes.  Useful for pinning a persistent latent buffer that is used in the
   same collective call across all denoising steps (same pointer, same
   communicator, same offset alignment).  Requires NCCL ≥ 2.19.

   **Constraint (from NCCL docs)**: if *any* rank passes a registered buffer
   to a collective, *all* other ranks in the same communicator must also pass
   their registered counterpart.  The registrar is therefore all-or-nothing
   per communicator.

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
"""

import ctypes
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Minimum NCCL version codes for each feature.
_NCCL_MIN_REG = (2, 19, 0)    # ncclCommRegister / ncclCommDeregister
_NCCL_MIN_CUMEM = (2, 21, 0)  # NCCL_CUMEM_ENABLE / VMM allocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nccl() -> Optional[ctypes.CDLL]:
    """Try to load libnccl.so from standard locations."""
    for lib_name in ("libnccl.so.2", "libnccl.so"):
        try:
            return ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    return None


def _nccl_version(lib: ctypes.CDLL) -> tuple:
    """Return (major, minor, patch) from ncclGetVersion."""
    ver = ctypes.c_int(0)
    lib.ncclGetVersion(ctypes.byref(ver))
    v = ver.value
    return (v // 10000, (v // 100) % 100, v % 100)


def _check_nccl_version(lib: ctypes.CDLL, required: tuple, feature: str) -> bool:
    ver = _nccl_version(lib)
    if ver < required:
        logger.warning(
            f"NCCL {ver[0]}.{ver[1]}.{ver[2]} < "
            f"{required[0]}.{required[1]}.{required[2]}: "
            f"{feature} not available"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Mechanism 1: CUMEM env-var (set before init_process_group)
# ---------------------------------------------------------------------------

def enable_nccl_cumem() -> bool:
    """Enable NCCL VMM-based buffer registration via ``NCCL_CUMEM_ENABLE=1``.

    Must be called **before** ``torch.distributed.init_process_group()``.
    Returns True if the env var was set (or was already set), False if NCCL
    is too old to support it.

    When enabled, NCCL allocates its scratch buffers with ``cuMemCreate``
    (VMM).  Any CUDA tensor allocated by PyTorch's caching allocator that
    happens to be VMM-backed (``expandable_segments:True`` on CUDA 11.8+)
    will also be eligible for zero-copy collectives automatically.
    """
    lib = _load_nccl()
    if lib is None:
        logger.warning("libnccl.so not found; cannot enable CUMEM mode")
        return False
    if not _check_nccl_version(lib, _NCCL_MIN_CUMEM, "NCCL_CUMEM_ENABLE"):
        return False

    if os.environ.get("NCCL_CUMEM_ENABLE") == "1":
        logger.debug("NCCL_CUMEM_ENABLE already set")
    else:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        logger.info("Set NCCL_CUMEM_ENABLE=1 (VMM-based buffer registration)")
    return True


# ---------------------------------------------------------------------------
# Mechanism 2: Explicit per-tensor ncclCommRegister
# ---------------------------------------------------------------------------

def _extract_comm_ptr(group: dist.ProcessGroup) -> Optional[int]:
    """Best-effort: extract the raw ``ncclComm_t`` pointer from a PyTorch process group.

    PyTorch does not expose this pointer in a stable public API.  We try a
    sequence of progressively more fragile introspection methods and return
    ``None`` if none succeed.
    """
    try:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        backend = group._get_backend(device)

        # PyTorch ≥ 2.6 ProcessGroupNCCL exposes _get_comm() returning the
        # ncclComm_t as an integer.
        if hasattr(backend, "_get_comm"):
            return int(backend._get_comm(device))

        # Older releases: the comm is stored in backend._comm (a capsule).
        # We can cast the capsule's void* with ctypes.
        if hasattr(backend, "_comm"):
            import ctypes
            cap = backend._comm
            # PyCapsule_GetPointer is exposed through ctypes.pythonapi
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            ptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, None)
            return ptr

    except Exception as exc:
        logger.debug(f"Could not extract NCCL comm ptr: {exc}")
    return None


class NCCLBufferRegistrar:
    """Registers specific tensors with ``ncclCommRegister`` for zero-copy collectives.

    All ranks in the communicator **must** register the corresponding tensor
    before using it in a collective; partial registration is undefined behaviour.

    Typical usage::

        registrar = NCCLBufferRegistrar(ulysses_group)
        if registrar.available:
            registrar.register_all([latent, scratch_q, scratch_kv])
        # ... run denoising loop (same tensor pointers) ...
        registrar.deregister_all()

    Registration is a no-op (and ``available`` is False) when:
    - libnccl.so cannot be loaded
    - NCCL version < 2.19
    - the raw ncclComm_t pointer cannot be extracted from the process group
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._group = process_group
        self._lib: Optional[ctypes.CDLL] = None
        self._comm_ptr: Optional[int] = None
        self._handles: dict[int, ctypes.c_void_p] = {}  # data_ptr → opaque handle
        self.available = False

        lib = _load_nccl()
        if lib is None:
            logger.warning("NCCLBufferRegistrar: libnccl.so not found")
            return
        if not _check_nccl_version(lib, _NCCL_MIN_REG, "ncclCommRegister"):
            return
        comm_ptr = _extract_comm_ptr(process_group)
        if comm_ptr is None:
            logger.info(
                "NCCLBufferRegistrar: could not extract ncclComm_t; "
                "CUMEM env-var mode is still active if enabled"
            )
            return

        self._lib = lib
        self._comm_ptr = comm_ptr
        self.available = True
        logger.info("NCCLBufferRegistrar ready (ncclCommRegister available)")

    def register(self, tensor: torch.Tensor) -> bool:
        """Register a single tensor.  Returns True on success."""
        if not self.available:
            return False
        key = tensor.data_ptr()
        if key in self._handles:
            return True  # already registered
        handle = ctypes.c_void_p(None)
        ret = self._lib.ncclCommRegister(
            ctypes.c_void_p(self._comm_ptr),
            ctypes.c_void_p(key),
            ctypes.c_size_t(tensor.nbytes),
            ctypes.byref(handle),
        )
        if ret == 0:  # ncclSuccess
            self._handles[key] = handle
            logger.debug(f"Registered tensor @{key:#x} ({tensor.nbytes} B)")
            return True
        logger.warning(f"ncclCommRegister returned error {ret} for tensor @{key:#x}")
        return False

    def register_all(self, tensors) -> int:
        """Register a list of tensors.  Returns the number successfully registered."""
        return sum(self.register(t) for t in tensors)

    def deregister(self, tensor: torch.Tensor):
        key = tensor.data_ptr()
        if key in self._handles:
            self._lib.ncclCommDeregister(
                ctypes.c_void_p(self._comm_ptr),
                self._handles.pop(key),
            )
            logger.debug(f"Deregistered tensor @{key:#x}")

    def deregister_all(self):
        if not self.available or self._lib is None:
            return
        for key, handle in list(self._handles.items()):
            self._lib.ncclCommDeregister(
                ctypes.c_void_p(self._comm_ptr),
                handle,
            )
        self._handles.clear()

    def __del__(self):
        try:
            self.deregister_all()
        except Exception:
            pass
