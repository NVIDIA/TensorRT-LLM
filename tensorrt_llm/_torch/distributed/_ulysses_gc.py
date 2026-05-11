# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Green Context partition manager for Ulysses split-QKV pipeline.

Used by UlyssesAttention to create non-overlapping SM partitions for the
bench v11_split_hybrid_batched timeline:

    gc_comp_stream      : large partition (~136 SMs) — Q/K GEMM + qk_norm + rope
    gc_selfcopy_stream  : small partition (~12 SMs)  — V/Q alltoall self-copy
    pri_comm_stream     : primary ctx, ordinary aux stream — peer copies + barrier signal
    default stream      : primary ctx, all 148 SMs       — V GEMM, output GEMM, SDPA

Torch's ``torch.cuda.GreenContext.create()`` API has an aliasing bug when 2 GCs
are live simultaneously, so we use raw CUDA driver API
(``cuDevSmResourceSplitByCount`` + ``cuGreenCtxCreate``) via ``ctypes``.

NO fallback: if GC partition creation fails, we raise. A non-GC fallback would
silently regress to the serial pV→pri_comm timeline (no GEMM ∥ self-copy
overlap), losing most of the optimization.
"""
from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional, Tuple

import torch


# -----------------------------------------------------------------------------
# Raw CUDA driver bindings
# -----------------------------------------------------------------------------
_libcuda = ctypes.CDLL("libcuda.so.1")

_CUresult = ctypes.c_int
_CUdevice = ctypes.c_int
_CUgreenCtx = ctypes.c_void_p
_CUstream = ctypes.c_void_p
_CUdevResourceDesc = ctypes.c_void_p

# CUdevResource = 144 B in CUDA 13 (verified). offset 0 = type (4B), then 140 B
# of internal driver state.
class CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("_internal", ctypes.c_ubyte * 140),
    ]


CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 0x1
CU_STREAM_NON_BLOCKING = 0x1


_libcuda.cuInit.argtypes = [ctypes.c_uint]
_libcuda.cuInit.restype = _CUresult

_libcuda.cuDeviceGetDevResource.argtypes = [
    _CUdevice,
    ctypes.POINTER(CUdevResource),
    ctypes.c_int,
]
_libcuda.cuDeviceGetDevResource.restype = _CUresult

_libcuda.cuDevSmResourceSplitByCount.argtypes = [
    ctypes.POINTER(CUdevResource),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(CUdevResource),
    ctypes.POINTER(CUdevResource),
    ctypes.c_uint,
    ctypes.c_uint,
]
_libcuda.cuDevSmResourceSplitByCount.restype = _CUresult

_libcuda.cuDevResourceGenerateDesc.argtypes = [
    ctypes.POINTER(_CUdevResourceDesc),
    ctypes.POINTER(CUdevResource),
    ctypes.c_uint,
]
_libcuda.cuDevResourceGenerateDesc.restype = _CUresult

_libcuda.cuGreenCtxCreate.argtypes = [
    ctypes.POINTER(_CUgreenCtx),
    _CUdevResourceDesc,
    _CUdevice,
    ctypes.c_uint,
]
_libcuda.cuGreenCtxCreate.restype = _CUresult

_libcuda.cuGreenCtxStreamCreate.argtypes = [
    ctypes.POINTER(_CUstream),
    _CUgreenCtx,
    ctypes.c_uint,
    ctypes.c_int,
]
_libcuda.cuGreenCtxStreamCreate.restype = _CUresult

_libcuda.cuGetErrorString.argtypes = [_CUresult, ctypes.POINTER(ctypes.c_char_p)]
_libcuda.cuGetErrorString.restype = _CUresult


def _check(res: int, op: str = "") -> None:
    if res != 0:
        err = ctypes.c_char_p()
        _libcuda.cuGetErrorString(res, ctypes.byref(err))
        msg = err.value.decode() if err.value else "<no msg>"
        raise RuntimeError(f"CUDA driver {op!r} failed (err={res}): {msg}")


def _create_two_partitions(
    device_id: int,
    large_sms: int = 136,
) -> Tuple[ctypes.c_void_p, ctypes.c_void_p, "torch.cuda.ExternalStream", "torch.cuda.ExternalStream"]:
    """Carve the 148-SM pool into a large + remaining-small partition.

    Strategy: split LARGE first (request ``large_sms``), then use the
    ``remaining`` resource directly as the small partition (B200 GPC granularity
    means asking for an explicit small_sms count is not honored — the leftover
    is whatever is left). Retries large with decreasing min if the first
    attempt fails (driver enforces GPC-aligned counts).
    """
    _libcuda.cuInit(0)
    dev = _CUdevice(device_id)

    full = CUdevResource()
    _check(
        _libcuda.cuDeviceGetDevResource(
            dev, ctypes.byref(full), CU_DEV_RESOURCE_TYPE_SM
        ),
        "cuDeviceGetDevResource",
    )

    split_large = CUdevResource()
    r1 = CUdevResource()
    full_bk = CUdevResource()
    ctypes.memmove(
        ctypes.byref(full_bk), ctypes.byref(full), ctypes.sizeof(CUdevResource)
    )

    nb = ctypes.c_uint(1)
    success = False
    actual_large = None
    last_err = None
    for attempt in range(large_sms, 4, -4):
        ctypes.memmove(
            ctypes.byref(full), ctypes.byref(full_bk), ctypes.sizeof(CUdevResource)
        )
        nb.value = 1
        res = _libcuda.cuDevSmResourceSplitByCount(
            ctypes.byref(split_large),
            ctypes.byref(nb),
            ctypes.byref(full),
            ctypes.byref(r1),
            0,
            attempt,
        )
        if res == 0:
            actual_large = attempt
            success = True
            break
        last_err = res
    if not success:
        _check(last_err, "split large")

    split_small = r1

    desc_small = _CUdevResourceDesc()
    _check(
        _libcuda.cuDevResourceGenerateDesc(
            ctypes.byref(desc_small), ctypes.byref(split_small), 1
        ),
        "desc small",
    )
    desc_large = _CUdevResourceDesc()
    _check(
        _libcuda.cuDevResourceGenerateDesc(
            ctypes.byref(desc_large), ctypes.byref(split_large), 1
        ),
        "desc large",
    )

    gc_small = _CUgreenCtx()
    _check(
        _libcuda.cuGreenCtxCreate(
            ctypes.byref(gc_small), desc_small, dev, CU_GREEN_CTX_DEFAULT_STREAM
        ),
        "gc small",
    )
    gc_large = _CUgreenCtx()
    _check(
        _libcuda.cuGreenCtxCreate(
            ctypes.byref(gc_large), desc_large, dev, CU_GREEN_CTX_DEFAULT_STREAM
        ),
        "gc large",
    )

    s_small = _CUstream()
    _check(
        _libcuda.cuGreenCtxStreamCreate(
            ctypes.byref(s_small), gc_small, CU_STREAM_NON_BLOCKING, 0
        ),
        "stream small",
    )
    s_large = _CUstream()
    _check(
        _libcuda.cuGreenCtxStreamCreate(
            ctypes.byref(s_large), gc_large, CU_STREAM_NON_BLOCKING, 0
        ),
        "stream large",
    )

    torch_small = torch.cuda.ExternalStream(s_small.value, device=device_id)
    torch_large = torch.cuda.ExternalStream(s_large.value, device=device_id)
    return gc_small, gc_large, torch_small, torch_large


# -----------------------------------------------------------------------------
# Singleton manager
# -----------------------------------------------------------------------------
class UlyssesPipelineStreams:
    """Per-device singleton holding the 3 auxiliary streams used by the Ulysses
    split-QKV pipeline:

    - ``pri_comm_stream``    : ordinary cuStream on primary ctx
    - ``gc_comp_stream``     : on GC-large partition (~136 SMs)
    - ``gc_selfcopy_stream`` : on GC-small partition (remainder)

    Plus persistent ``torch.cuda.Event``s for cross-stream sync.
    """

    __slots__ = (
        "device_id",
        "pri_comm_stream",
        "gc_comp_stream",
        "gc_selfcopy_stream",
        "gc_selfcopy_handle",
        "_gc_small",
        "_gc_large",
        "ev_v",
        "ev_q",
        "ev_k",
        "ev_done",
    )

    _instances: Dict[int, "UlyssesPipelineStreams"] = {}

    def __init__(self, device_id: int, large_sms: int = 136) -> None:
        self.device_id = device_id
        # pri_comm: ordinary stream on primary ctx (NCCL alltoall ordering).
        self.pri_comm_stream = torch.cuda.Stream(device=device_id)

        # GC partitions: large (~136 SM) for Q/K compute; small (remainder)
        # for self-copy memcpy. Stream switches happen INSIDE @disable region
        # (`_pre_attn_alltoall_pipeline`) so dynamo does not trace through them.
        # Set ULYSSES_DISABLE_GC=1 to fall back to plain streams (no SM
        # partition) for debugging.
        if os.environ.get("ULYSSES_DISABLE_GC", "0") == "1":
            self._gc_small = None
            self._gc_large = None
            self.gc_comp_stream = torch.cuda.Stream(device=device_id)
            self.gc_selfcopy_stream = torch.cuda.Stream(device=device_id)
            self.gc_selfcopy_handle = int(self.gc_selfcopy_stream.cuda_stream)
        else:
            try:
                gc_small, gc_large, s_small, s_large = _create_two_partitions(
                    device_id, large_sms=large_sms)
                self._gc_small = gc_small
                self._gc_large = gc_large
                # gc_comp on large partition (compute), selfcopy on small.
                self.gc_comp_stream = s_large
                self.gc_selfcopy_stream = s_small
                self.gc_selfcopy_handle = int(s_small.cuda_stream)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"GC partition creation failed ({e}); falling back to "
                    "plain torch.cuda.Stream (no SM partition).")
                self._gc_small = None
                self._gc_large = None
                self.gc_comp_stream = torch.cuda.Stream(device=device_id)
                self.gc_selfcopy_stream = torch.cuda.Stream(device=device_id)
                self.gc_selfcopy_handle = int(self.gc_selfcopy_stream.cuda_stream)

        # Events are created per-call inside `_pre_attn_alltoall_pipeline`
        # (the @disable region) — kept here only for back-compat eager use.
        self.ev_v = torch.cuda.Event()
        self.ev_q = torch.cuda.Event()
        self.ev_k = torch.cuda.Event()
        self.ev_done = torch.cuda.Event()

    @classmethod
    def get(cls, device_id: int, large_sms: int = 136) -> "UlyssesPipelineStreams":
        inst = cls._instances.get(device_id)
        if inst is None:
            inst = cls(device_id=device_id, large_sms=large_sms)
            cls._instances[device_id] = inst
        return inst
