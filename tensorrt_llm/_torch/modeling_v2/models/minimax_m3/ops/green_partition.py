# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A device's SMs split into a main green context of ``main_sms`` SMs (coscheduled in 4s) and the remainder, each with a
non-blocking stream: work launched on a part's stream runs only on that part's SMs.

The driver path of the family's ``kernels/m3ll/locality`` (``cuDevSmResourceSplit`` + ``cuGreenCtxCreate`` +
``cuGreenCtxStreamCreate``, the streams wrapped as ``torch.cuda.ExternalStream``), with each part's SM set read back by
a ``%smid`` probe launched on that part. Built once per (device, main_sms) and kept for the process, before any CUDA
graph capture; a graph that forks onto a part's stream and joins back captures like any other fork / join.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

import torch

#: The SMs one coscheduled group spans in the main part (the 4-CTA clusters its kernels may launch).
COSCHEDULED_SMS = 4

_PROBE_SRC = r"""
extern "C" __global__ void smid_probe(int* out) {
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        out[blockIdx.x] = (int)s;
    }
}
"""


@dataclass(frozen=True)
class GreenPartition:
    """One device's main / side split: the probed SM sets and a stream on each part."""

    device: int
    num_sms: int
    main_sms: FrozenSet[int]
    side_sms: FrozenSet[int]
    main_stream: torch.cuda.ExternalStream
    side_stream: torch.cuda.ExternalStream

    @property
    def disjoint(self) -> bool:
        return not (self.main_sms & self.side_sms)

    def describe(self) -> str:
        return (
            f"main {len(self.main_sms)} / side {len(self.side_sms)} of {self.num_sms} SMs"
            f"{'' if self.disjoint else ' (OVERLAPPING)'}"
        )


_LOCK = threading.Lock()
_PARTS: Dict[Tuple[int, int], GreenPartition] = {}
# The driver objects behind each partition, kept alive with it (the streams die with their green contexts).
_KEEP: Dict[Tuple[int, int], list] = {}


def _ck(res, what: str):
    """cuda-python returns (CUresult, *values): raise on an error, return the values."""
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"green partition: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _probe_cubin(nvrtc, major: int, minor: int) -> bytes:
    prog = _ck(
        nvrtc.nvrtcCreateProgram(_PROBE_SRC.encode(), b"smid_probe.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
        raise RuntimeError("green partition: the %smid probe did not compile")
    n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * n
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    return cubin


def get(device: int, main_sms: int) -> GreenPartition:
    """The (cached) split of ``device`` into ``main_sms`` SMs and the rest; raises where the driver cannot split it or
    when called inside a CUDA-graph capture before it exists."""
    key = (int(device), int(main_sms))
    with _LOCK:
        part = _PARTS.get(key)
        if part is not None:
            return part
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("green partition: build it before CUDA-graph capture")
        part, keep = _build(*key)
        _PARTS[key] = part
        _KEEP[key] = keep
        return part


def _build(device: int, main_sms: int):
    from cuda.bindings import driver as d
    from cuda.bindings import nvrtc

    _ck(d.cuInit(0), "cuInit")
    dev = _ck(d.cuDeviceGet(device), "cuDeviceGet")
    num_sms = int(
        _ck(
            d.cuDeviceGetAttribute(
                d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev
            ),
            "SM count",
        )
    )
    if not 0 < main_sms < num_sms:
        raise ValueError(f"green partition: {main_sms} main SMs on a {num_sms}-SM device")
    full = _ck(
        d.cuDeviceGetDevResource(dev, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM), "dev resource"
    )
    params = d.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
    params.smCount = main_sms
    for name in ("coscheduledSmCount", "coscheduledSMCount"):
        if hasattr(params, name):
            setattr(params, name, COSCHEDULED_SMS)
    res, rem = _ck(d.cuDevSmResourceSplit(1, full, 0, [params]), "cuDevSmResourceSplit")
    keep, streams = [], []
    for part in (res[0], rem):
        desc = _ck(d.cuDevResourceGenerateDesc([part], 1), "resource desc")
        green = _ck(
            d.cuGreenCtxCreate(desc, dev, d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM),
            "cuGreenCtxCreate",
        )
        stream = _ck(
            d.cuGreenCtxStreamCreate(green, d.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
            "cuGreenCtxStreamCreate",
        )
        keep.append((green, stream))
        streams.append(torch.cuda.ExternalStream(int(stream), device=f"cuda:{device}"))
    major, minor = torch.cuda.get_device_capability(device)
    cubin = _probe_cubin(nvrtc, major, minor)
    blocks = 4 * num_sms  # more blocks than SMs: every SM of the part shows up
    buf = torch.full((blocks,), -1, dtype=torch.int32, device=f"cuda:{device}")
    sets = []
    for green, stream in keep:
        ctx = _ck(d.cuCtxFromGreenCtx(green), "ctx from green ctx")
        _ck(d.cuCtxPushCurrent(ctx), "push ctx")
        try:
            mod = _ck(d.cuModuleLoadData(cubin), "probe module")
            fn = _ck(d.cuModuleGetFunction(mod, b"smid_probe"), "probe fn")
            buf.fill_(-1)
            torch.cuda.synchronize(device)
            arg = ctypes.c_void_p(buf.data_ptr())
            _ck(
                d.cuLaunchKernel(
                    fn, blocks, 1, 1, 32, 1, 1, 0, stream, ((arg,), (ctypes.c_void_p,)), 0
                ),
                "probe launch",
            )
            _ck(d.cuStreamSynchronize(stream), "probe sync")
            sets.append(frozenset(int(x) for x in buf.tolist() if x >= 0))
            d.cuModuleUnload(mod)
        finally:
            d.cuCtxPopCurrent()
    part = GreenPartition(device, num_sms, sets[0], sets[1], streams[0], streams[1])
    return part, keep
