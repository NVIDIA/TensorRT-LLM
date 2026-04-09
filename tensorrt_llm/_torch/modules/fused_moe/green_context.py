# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA Driver API helpers for Workqueue-isolated GreenContext stream creation."""

import ctypes
from typing import Optional

import torch

_libcuda: Optional[ctypes.CDLL] = None


def _get_libcuda() -> ctypes.CDLL:
    """Lazily load libcuda.so.1 (CUDA Driver API)."""
    global _libcuda
    if _libcuda is None:
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                _libcuda = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _libcuda is None:
            raise RuntimeError("Cannot load libcuda (CUDA Driver API) shared library")
    return _libcuda


def get_current_stream_gc_sm_count() -> int:
    """Return the SM count of the GreenContext bound to the current CUDA stream.

    When a kernel is dispatched inside ``torch.cuda.stream(gc_stream)`` where
    *gc_stream* was created by ``cuGreenCtxStreamCreate``, the stream carries a
    ``CUgreenCtx`` handle that encodes the SM partition assigned to that
    GreenContext.  This function queries that partition size so callers can
    derive ``sm_budget`` without hard-coding it at the call site.

    Workflow:
        1. ``cuStreamGetGreenCtx`` — retrieve the ``CUgreenCtx`` bound to the
           current PyTorch stream (fails with ``CUDA_ERROR_INVALID_HANDLE`` for
           plain streams, returning ``-1``).
        2. ``cuCtxFromGreenCtx`` — convert ``CUgreenCtx`` to a ``CUcontext``
           handle (does not push/pop; the handle is valid for the lifetime of the
           GreenContext).
        3. ``cuCtxGetDevResource(CU_DEV_RESOURCE_TYPE_SM)`` — fill a raw
           ``CUdevResource`` buffer and read ``smCount`` at the version-specific
           byte offset.

    Returns:
        ``smCount`` from the GreenContext's SM resource partition, or ``-1`` if:
        - the current stream is not a GC-bound stream,
        - ``cuStreamGetGreenCtx`` is unavailable (CUDA < 12.4), or
        - any Driver API call fails.
    """
    from ctypes import byref, c_int, c_void_p, create_string_buffer

    libcuda = _get_libcuda()

    CU_DEV_RESOURCE_TYPE_SM = 1
    CUDA_SUCCESS = 0

    if not hasattr(libcuda, "cuStreamGetGreenCtx"):
        return -1  # CUDA < 12.4

    stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

    # Step 1: get GreenContext handle from the stream.
    # Signature: cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx *phCtx)
    gc_handle = c_void_p()
    if libcuda.cuStreamGetGreenCtx(stream_ptr, byref(gc_handle)) != CUDA_SUCCESS:
        return -1  # not a cuGreenCtxStreamCreate-created stream

    # Step 2: derive a CUcontext from the CUgreenCtx.
    ctx_handle = c_void_p()
    if libcuda.cuCtxFromGreenCtx(byref(ctx_handle), gc_handle) != CUDA_SUCCESS:
        return -1

    # Step 3: determine CUDA version to pick the correct smCount byte offset.
    # CUdevResource raw layout:
    #   CUDA 12.x: { type:u32(4) } + { union(92) }  — smCount at offset 4
    #   CUDA 13.x: { type:u32(4) } + { padding(92) } + { union(40) } + { ptr(8) }
    #              — smCount is the first field of the union → offset 4+92 = 96
    cuda_ver = c_int(0)
    libcuda.cuDriverGetVersion(byref(cuda_ver))
    if cuda_ver.value >= 13000:
        buf_size, sm_count_offset = 144, 96
    else:
        buf_size, sm_count_offset = 96, 4

    res_buf = create_string_buffer(buf_size)
    if (
        libcuda.cuCtxGetDevResource(ctx_handle, res_buf, c_int(CU_DEV_RESOURCE_TYPE_SM))
        != CUDA_SUCCESS
    ):
        return -1

    # smCount is a little-endian uint32 at the computed offset.
    return int.from_bytes(res_buf[sm_count_offset : sm_count_offset + 4], byteorder="little")


def create_wq_isolated_gc_streams(
    fc1_sms: int,
    router_sms: int,
    device_id: int,
) -> tuple:
    """Create two GreenContext streams with SM *and* Workqueue isolation.

    PyTorch's ``torch.cuda.GreenContext.create()`` only partitions SM resources,
    leaving the hardware workqueue shared across GreenContexts.  That shared WQ
    introduces a ~10 µs dispatch serialisation overhead even when the two kernels
    execute on disjoint SM partitions.

    This function bypasses PyTorch and calls the CUDA Driver API directly to
    build resource descriptors that include *both* an SM partition and a
    ``CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED`` workqueue config.  The result is
    two streams that are truly independent at both the SM and WQ scheduling
    levels.

    SM counts are rounded up to the hardware ``smCoscheduledAlignment`` boundary
    (queried at runtime).  If the requested counts cannot both be satisfied after
    alignment, the FC1 partition is shrunk to leave the Router at least one
    scheduling unit.

    Args:
        fc1_sms:   Requested SM count for the FC1 partition.
        router_sms: Requested SM count for the Router partition.
        device_id: CUDA device index.

    Returns:
        ``(fc1_stream, router_stream, cleanup_fn)`` where the two streams are
        :class:`torch.cuda.ExternalStream` objects backed by GC-bound
        ``CUstream`` handles, and ``cleanup_fn()`` destroys those handles
        (call it when the layer is no longer needed).

    Raises:
        RuntimeError: if any CUDA Driver API call fails.
    """
    from ctypes import (
        Structure,
        Union,
        addressof,
        byref,
        c_int,
        c_ubyte,
        c_uint,
        c_void_p,
        memmove,
        sizeof,
    )

    libcuda = _get_libcuda()

    # ------------------------------------------------------------------
    # CUDA Driver API constants
    # ------------------------------------------------------------------
    CU_DEV_RESOURCE_TYPE_SM = 1
    CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000
    CU_GREEN_CTX_DEFAULT_STREAM = 0x1
    CU_STREAM_NON_BLOCKING = 0x1
    CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED = 1
    CUDA_SUCCESS = 0

    # ------------------------------------------------------------------
    # CUdevResource ctypes layout — version-aware
    #
    # CUDA 12.x:
    #   struct { CUdevResourceType type; union { sm; wqConfig; wqResource; char raw[92]; }; }
    #   CUdevSmResource:           smCount, smCoscheduledAlignment, minSmPartitionSize, reserved[13]
    #   CUdevWorkqueueConfigResource: wqConcurrencyLimit, sharingScope, reserved[6]
    #
    # CUDA 13.0+:
    #   struct { CUdevResourceType type; unsigned char _internal_padding[92];
    #            union { sm; wqConfig; wq; char raw[40]; }; struct CUdevResource_st* nextResource; }
    #   CUdevSmResource:           smCount, minSmPartitionSize, smCoscheduledAlignment, flags
    #   CUdevWorkqueueConfigResource: device, wqConcurrencyLimit, sharingScope
    # ------------------------------------------------------------------
    cuda_ver = c_int(0)
    libcuda.cuDriverGetVersion(byref(cuda_ver))
    _cuda_version = cuda_ver.value  # e.g. 13010 for CUDA 13.1, 12060 for CUDA 12.6

    if _cuda_version >= 13000:

        class _SmData(Structure):
            _fields_ = [
                ("smCount", c_uint),
                ("minSmPartitionSize", c_uint),
                ("smCoscheduledAlignment", c_uint),
                ("flags", c_uint),
            ]

        class _WqConfigData(Structure):
            _fields_ = [
                ("device", c_int),
                ("wqConcurrencyLimit", c_uint),
                ("sharingScope", c_uint),
            ]

        class _ResData(Union):
            _fields_ = [
                ("sm", _SmData),
                ("wqConfig", _WqConfigData),
                # RESOURCE_ABI_BYTES = 40 in CUDA 13.x
                ("raw", c_ubyte * 40),
            ]

        class CUdevResource(Structure):
            _fields_ = [
                ("type", c_uint),
                ("_internal_padding", c_ubyte * 92),
                ("data", _ResData),
                ("nextResource", c_void_p),
            ]
    else:

        class _SmData(Structure):
            _fields_ = [
                ("smCount", c_uint),
                ("smCoscheduledAlignment", c_uint),
                ("minSmPartitionSize", c_uint),
                ("reserved", c_uint * 13),
            ]

        class _WqConfigData(Structure):
            _fields_ = [
                ("wqConcurrencyLimit", c_uint),
                ("sharingScope", c_uint),
                ("reserved", c_uint * 6),
            ]

        class _ResData(Union):
            _fields_ = [
                ("sm", _SmData),
                ("wqConfig", _WqConfigData),
                # 92 bytes: CUDA 12.x union size
                ("raw", c_ubyte * 92),
            ]

        class CUdevResource(Structure):
            _fields_ = [("type", c_uint), ("data", _ResData)]

    def _check(ret: int, fn_name: str) -> None:
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"{fn_name} failed with CUDA error code {ret}")

    # 1. Query device SM resource to get total SM count -------------------
    sm_res = CUdevResource()
    _check(
        libcuda.cuDeviceGetDevResource(
            c_int(device_id), byref(sm_res), c_int(CU_DEV_RESOURCE_TYPE_SM)
        ),
        "cuDeviceGetDevResource(SM)",
    )
    total_sms = sm_res.data.sm.smCount

    # 2. Split SM resources via cuDevSmResourceSplitByCount ---------------
    # Carve the Router partition first; FC1 receives the unstructured
    # remainder.  Reversing the order ensures the router gets an exactly
    # sized, aligned partition, while FC1's remainder is still fully usable
    # for a GreenContext.
    router_sm_result = (CUdevResource * 1)()
    fc1_sm_res = CUdevResource()
    nbGroups = c_uint(1)
    _check(
        libcuda.cuDevSmResourceSplitByCount(
            router_sm_result,
            byref(nbGroups),
            byref(sm_res),
            byref(fc1_sm_res),
            c_uint(0),
            c_uint(router_sms),
        ),
        "cuDevSmResourceSplitByCount(router)",
    )
    if nbGroups.value == 0:
        raise RuntimeError(
            f"cuDevSmResourceSplitByCount returned 0 groups "
            f"(router_sms={router_sms}, total_sms={total_sms})"
        )

    # 3. Build WQ config resources with GREEN_CTX_BALANCED scope ----------
    # WQ_CONFIG must be queried with cuDeviceGetDevResource (device-level), not
    # cuCtxGetDevResource (context-level).  When mixing SM + WQ_CONFIG resources
    # in cuDevResourceGenerateDesc, the driver requires both to originate from the
    # same device.  SM resources come from cuDeviceGetDevResource/cuDevSmResourceSplitByCount;
    # using cuCtxGetDevResource for WQ gives them different provenance, causing
    # CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION (914).
    # CUDA docs: "In case of workqueues, an existing one queried via the
    # cuDeviceGetDevResource API."
    wq_fc1 = CUdevResource()
    _check(
        libcuda.cuDeviceGetDevResource(
            c_int(device_id),
            byref(wq_fc1),
            c_int(CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG),
        ),
        "cuDeviceGetDevResource(WQ_CONFIG)",
    )
    wq_router = CUdevResource()
    memmove(addressof(wq_router), addressof(wq_fc1), sizeof(CUdevResource))
    # Override only the sharing scope; all other fields (wqConcurrencyLimit, etc.)
    # retain the driver-provided values.
    wq_fc1.data.wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
    wq_router.data.wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED

    # 4. Build resource descriptors (SM partition + WQ config, per GC) ---
    fc1_res_arr = (CUdevResource * 2)()
    router_res_arr = (CUdevResource * 2)()
    res_sz = sizeof(CUdevResource)
    # fc1: fc1_sm_res (remainder) + wq_fc1
    memmove(addressof(fc1_res_arr), addressof(fc1_sm_res), res_sz)
    memmove(addressof(fc1_res_arr) + res_sz, addressof(wq_fc1), res_sz)
    # router: router_sm_result[0] + wq_router
    memmove(addressof(router_res_arr), addressof(router_sm_result), res_sz)
    memmove(addressof(router_res_arr) + res_sz, addressof(wq_router), res_sz)

    fc1_desc = c_void_p()
    router_desc = c_void_p()
    _check(
        libcuda.cuDevResourceGenerateDesc(byref(fc1_desc), fc1_res_arr, c_uint(2)),
        "cuDevResourceGenerateDesc(fc1)",
    )
    _check(
        libcuda.cuDevResourceGenerateDesc(byref(router_desc), router_res_arr, c_uint(2)),
        "cuDevResourceGenerateDesc(router)",
    )

    # 5. Create GreenContexts ---------------------------------------------
    fc1_gc_h = c_void_p()
    router_gc_h = c_void_p()
    _check(
        libcuda.cuGreenCtxCreate(
            byref(fc1_gc_h),
            fc1_desc,
            c_int(device_id),
            c_uint(CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "cuGreenCtxCreate(fc1)",
    )
    _check(
        libcuda.cuGreenCtxCreate(
            byref(router_gc_h),
            router_desc,
            c_int(device_id),
            c_uint(CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "cuGreenCtxCreate(router)",
    )

    # 6. Create NON_BLOCKING streams bound to each GreenContext -----------
    fc1_stream_ptr = c_void_p()
    router_stream_ptr = c_void_p()
    _check(
        libcuda.cuGreenCtxStreamCreate(
            byref(fc1_stream_ptr),
            fc1_gc_h,
            c_uint(CU_STREAM_NON_BLOCKING),
            c_int(0),
        ),
        "cuGreenCtxStreamCreate(fc1)",
    )
    _check(
        libcuda.cuGreenCtxStreamCreate(
            byref(router_stream_ptr),
            router_gc_h,
            c_uint(CU_STREAM_NON_BLOCKING),
            c_int(0),
        ),
        "cuGreenCtxStreamCreate(router)",
    )

    # 7. Wrap raw CUstream handles in torch ExternalStream ----------------
    fc1_stream = torch.cuda.ExternalStream(fc1_stream_ptr.value, device=f"cuda:{device_id}")
    router_stream = torch.cuda.ExternalStream(router_stream_ptr.value, device=f"cuda:{device_id}")

    # 8. Cleanup closure -- call when the layer is destroyed --------------
    _fc1_gc_val = fc1_gc_h.value
    _router_gc_val = router_gc_h.value
    _fc1_s_val = fc1_stream_ptr.value
    _router_s_val = router_stream_ptr.value

    def _cleanup() -> None:
        _lib = _get_libcuda()
        _lib.cuStreamDestroy(c_void_p(_fc1_s_val))
        _lib.cuStreamDestroy(c_void_p(_router_s_val))
        _lib.cuGreenCtxDestroy(c_void_p(_fc1_gc_val))
        _lib.cuGreenCtxDestroy(c_void_p(_router_gc_val))

    return fc1_stream, router_stream, _cleanup


def create_sm_only_gc_streams(
    fc1_sms: int,
    router_sms: int,
    device_id: int,
) -> tuple:
    """Create GreenContext streams with SM isolation only (no WQ isolation).

    This is a CUDA-Graph-compatible alternative to the PyTorch
    ``torch.cuda.GreenContext`` + ``torch.cuda.Stream()`` path.

    **Why PyTorch streams break CUDA Graph:**
    ``torch.cuda.Stream()`` created inside ``GreenContext.set_context() / pop_context()``
    lives in a ``cuCtxFromGreenCtx``-derived regular context.  CUDA Graph
    capture/replay runs on the primary context's default stream.  When the graph is
    replayed, streams belonging to the GC-derived context silently **lose their SM
    partition** — FC1 and Router CTAs compete for all SMs, adding ~10 µs of latency.

    **Why this helper works:**
    Streams created by ``cuGreenCtxStreamCreate`` are bound directly to the
    ``CUgreenCtx`` handle itself (not to any derived context).  The SM partition
    encoded in the ``CUgreenCtx`` is preserved across CUDA Graph capture and replay
    because the stream identity remains stable.

    The difference from :func:`create_wq_isolated_gc_streams` is that only the SM
    resource is included in ``cuDevResourceGenerateDesc`` (count=1).  The hardware
    workqueue remains shared across the two GreenContexts.

    SM counts are rounded up to the hardware ``smCoscheduledAlignment`` boundary via
    ``cuDevSmResourceSplitByCount``.

    Args:
        fc1_sms:    Requested SM count for the FC1 partition.
        router_sms: Requested SM count for the Router partition (informational only;
                    router gets the remainder after FC1 is carved out).
        device_id:  CUDA device index.

    Returns:
        ``(fc1_stream, router_stream, cleanup_fn)`` where the two streams are
        :class:`torch.cuda.ExternalStream` objects backed by GC-bound ``CUstream``
        handles, and ``cleanup_fn()`` destroys those handles and the GreenContexts.

    Raises:
        RuntimeError: if any CUDA Driver API call fails.
    """
    from ctypes import Structure, Union, byref, c_int, c_ubyte, c_uint, c_void_p

    libcuda = _get_libcuda()

    CU_DEV_RESOURCE_TYPE_SM = 1
    CU_GREEN_CTX_DEFAULT_STREAM = 0x1
    CU_STREAM_NON_BLOCKING = 0x1
    CUDA_SUCCESS = 0

    cuda_ver = c_int(0)
    libcuda.cuDriverGetVersion(byref(cuda_ver))
    _cuda_version = cuda_ver.value

    # Reuse the same version-aware CUdevResource layout as create_wq_isolated_gc_streams.
    if _cuda_version >= 13000:

        class _SmData(Structure):
            _fields_ = [
                ("smCount", c_uint),
                ("minSmPartitionSize", c_uint),
                ("smCoscheduledAlignment", c_uint),
                ("flags", c_uint),
            ]

        class _ResData(Union):
            _fields_ = [
                ("sm", _SmData),
                ("raw", c_ubyte * 40),
            ]

        class CUdevResource(Structure):
            _fields_ = [
                ("type", c_uint),
                ("_internal_padding", c_ubyte * 92),
                ("data", _ResData),
                ("nextResource", c_void_p),
            ]
    else:

        class _SmData(Structure):
            _fields_ = [
                ("smCount", c_uint),
                ("smCoscheduledAlignment", c_uint),
                ("minSmPartitionSize", c_uint),
                ("reserved", c_uint * 13),
            ]

        class _ResData(Union):
            _fields_ = [
                ("sm", _SmData),
                ("raw", c_ubyte * 92),
            ]

        class CUdevResource(Structure):
            _fields_ = [("type", c_uint), ("data", _ResData)]

    def _check(ret: int, fn_name: str) -> None:
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"{fn_name} failed with CUDA error code {ret}")

    # 1. Query total SM resource.
    sm_res = CUdevResource()
    _check(
        libcuda.cuDeviceGetDevResource(
            c_int(device_id), byref(sm_res), c_int(CU_DEV_RESOURCE_TYPE_SM)
        ),
        "cuDeviceGetDevResource(SM)",
    )
    total_sms = sm_res.data.sm.smCount

    # 2. Split SMs: carve Router partition first; FC1 gets the remainder.
    router_sm_result = (CUdevResource * 1)()
    fc1_sm_res = CUdevResource()
    nbGroups = c_uint(1)
    _check(
        libcuda.cuDevSmResourceSplitByCount(
            router_sm_result,
            byref(nbGroups),
            byref(sm_res),
            byref(fc1_sm_res),
            c_uint(0),
            c_uint(router_sms),
        ),
        "cuDevSmResourceSplitByCount(router)",
    )
    if nbGroups.value == 0:
        raise RuntimeError(
            f"cuDevSmResourceSplitByCount returned 0 groups "
            f"(router_sms={router_sms}, total_sms={total_sms})"
        )

    # 3. Build resource descriptors with SM only (count=1, no WQ config).
    fc1_desc = c_void_p()
    router_desc = c_void_p()
    _check(
        libcuda.cuDevResourceGenerateDesc(byref(fc1_desc), byref(fc1_sm_res), c_uint(1)),
        "cuDevResourceGenerateDesc(fc1)",
    )
    _check(
        libcuda.cuDevResourceGenerateDesc(byref(router_desc), router_sm_result, c_uint(1)),
        "cuDevResourceGenerateDesc(router)",
    )

    # 4. Create GreenContexts.
    fc1_gc_h = c_void_p()
    router_gc_h = c_void_p()
    _check(
        libcuda.cuGreenCtxCreate(
            byref(fc1_gc_h),
            fc1_desc,
            c_int(device_id),
            c_uint(CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "cuGreenCtxCreate(fc1)",
    )
    _check(
        libcuda.cuGreenCtxCreate(
            byref(router_gc_h),
            router_desc,
            c_int(device_id),
            c_uint(CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "cuGreenCtxCreate(router)",
    )

    # 5. Create NON_BLOCKING streams bound directly to each CUgreenCtx.
    fc1_stream_ptr = c_void_p()
    router_stream_ptr = c_void_p()
    _check(
        libcuda.cuGreenCtxStreamCreate(
            byref(fc1_stream_ptr),
            fc1_gc_h,
            c_uint(CU_STREAM_NON_BLOCKING),
            c_int(0),
        ),
        "cuGreenCtxStreamCreate(fc1)",
    )
    _check(
        libcuda.cuGreenCtxStreamCreate(
            byref(router_stream_ptr),
            router_gc_h,
            c_uint(CU_STREAM_NON_BLOCKING),
            c_int(0),
        ),
        "cuGreenCtxStreamCreate(router)",
    )

    fc1_stream = torch.cuda.ExternalStream(fc1_stream_ptr.value, device=f"cuda:{device_id}")
    router_stream = torch.cuda.ExternalStream(router_stream_ptr.value, device=f"cuda:{device_id}")

    _fc1_gc_val = fc1_gc_h.value
    _router_gc_val = router_gc_h.value
    _fc1_s_val = fc1_stream_ptr.value
    _router_s_val = router_stream_ptr.value

    def _cleanup() -> None:
        _lib = _get_libcuda()
        _lib.cuStreamDestroy(c_void_p(_fc1_s_val))
        _lib.cuStreamDestroy(c_void_p(_router_s_val))
        _lib.cuGreenCtxDestroy(c_void_p(_fc1_gc_val))
        _lib.cuGreenCtxDestroy(c_void_p(_router_gc_val))

    return fc1_stream, router_stream, _cleanup
