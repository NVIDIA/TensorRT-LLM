# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' exact finalize (+ the shared expert's add) written into a caller-owned buffer.

``mv2_minimax_m3::minimax_m31_moe_finalize`` allocates its output. A caller whose consumer needs the result in a fixed
buffer (the TP2 CTX sequence-parallel boundary's IPC input slot) would then pay a full T x hidden copy.
:func:`finalize_into` runs the same kernel with ``out`` pointing at that buffer:

    out[t, h] = bf16(sum_k fma(w[t, k], fc2[idx[t, k], h], acc)) over k in order (idx -1 skipped), fp32 accumulate;
    with shared: bf16(float(shared[t, h]) + float(that)).

The source below is ``cpp/tensorrt_llm/kernels/minimaxM31/moeFinalizeSm107.cu`` (same per-element fp32 fma chain, the
same two round-to-nearest-even conversions, the same (threads, vectors per thread) table), so the result is bit for
bit the native op's. It is CUDA C++ compiled at runtime with NVRTC for this device (as ``m3_ctxlen_resume``) without
headers: bf16 is carried as its 16-bit pattern, widened by a shift and narrowed by ``cvt.rn.bf16.f32`` (what
``__float2bfloat16_rn`` emits on SM80+). It is launched with the driver API on the current torch stream.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Dict, Optional, Tuple

import torch

_LOCK = threading.Lock()
_KERNEL_NAME = "m31_moe_finalize_out"
_VEC = 8
# cfg -> (threads, 16-byte vectors per thread): the native op's table (cfg -1 = entry 0).
CFG_TABLE = {
    0: (256, 1),
    1: (128, 2),
    2: (64, 4),
    3: (128, 1),
    4: (64, 2),
    5: (192, 2),
    6: (96, 2),
    7: (32, 4),
}
_W_TYPES = {torch.float32: "float", torch.bfloat16: "unsigned short"}

_SRC = r"""
typedef W_T w_t;
__device__ __forceinline__ float w2f(float v) { return v; }
__device__ __forceinline__ float w2f(unsigned short v) { return __uint_as_float(((unsigned int) v) << 16); }
__device__ __forceinline__ float b2f(unsigned short v) { return __uint_as_float(((unsigned int) v) << 16); }
__device__ __forceinline__ unsigned short f2b(float f)
{
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}
extern "C" __global__ void __launch_bounds__(THREADS) m31_moe_finalize_out(const unsigned short* __restrict__ fc2,
    const int* __restrict__ idx, const w_t* __restrict__ weights, const unsigned short* __restrict__ shared,
    unsigned short* __restrict__ out, long long ld_fc2, int hidden)
{
    long long const t = blockIdx.x;
    int const col0 = blockIdx.y * (THREADS * NV * 8) + threadIdx.x * 8;
    int p[TOPK];
    float w[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        p[k] = idx[t * TOPK + k];
        w[k] = w2f(weights[t * TOPK + k]);
    }
    uint4 x[NV][TOPK];
    uint4 shv[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        int const col = col0 + v * THREADS * 8;
        if (SHARED)
            shv[v] = col < hidden ? __ldg(reinterpret_cast<const uint4*>(shared + t * hidden + col))
                                  : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
            x[v][k] = (p[k] >= 0 && col < hidden)
                ? __ldg(reinterpret_cast<const uint4*>(fc2 + static_cast<long long>(p[k]) * ld_fc2 + col))
                : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        int const col = col0 + v * THREADS * 8;
        if (col >= hidden)
            continue;
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
        {
            if (p[k] < 0)
                continue;
            const unsigned short* xv = reinterpret_cast<const unsigned short*>(&x[v][k]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] = __fmaf_rn(w[k], b2f(xv[j]), acc[j]);
        }
        uint4 r;
        unsigned short* rv = reinterpret_cast<unsigned short*>(&r);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            rv[j] = f2b(acc[j]);
        if (SHARED)
        {
            const unsigned short* sv = reinterpret_cast<const unsigned short*>(&shv[v]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                rv[j] = f2b(b2f(sv[j]) + b2f(rv[j]));
        }
        *reinterpret_cast<uint4*>(out + t * hidden + col) = r;
    }
}
"""

# (device, weights dtype, top_k, shared, cfg) -> (module, function)
_Key = Tuple[int, torch.dtype, int, bool, int]
_FUNCS: Dict[_Key, object] = {}
_FAILED: Dict[_Key, str] = {}


def _ck(res, what: str):
    """cuda-python returns (CUresult, *values): raise on an error, return the values."""
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"m31_moe_finalize_out: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _compile(key: _Key):
    """NVRTC-compile one instance for ``key``'s device and load it into torch's (primary) context of that device."""
    from cuda.bindings import driver as d
    from cuda.bindings import nvrtc

    device, w_dtype, topk, shared, cfg = key
    threads, nv = CFG_TABLE[cfg]
    torch.cuda.set_device(device)
    torch.zeros(1, device=f"cuda:{device}")  # make torch's primary context current on this thread
    dev = _ck(d.cuDeviceGet(device), "cuDeviceGet")
    major = _ck(
        d.cuDeviceGetAttribute(
            d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
        ),
        "cc major",
    )
    minor = _ck(
        d.cuDeviceGetAttribute(
            d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
        ),
        "cc minor",
    )
    src = (
        f"#define THREADS {threads}\n#define NV {nv}\n#define TOPK {topk}\n#define SHARED {int(shared)}\n"
        + _SRC.replace("W_T", _W_TYPES[w_dtype])
    )
    prog = _ck(
        nvrtc.nvrtcCreateProgram(src.encode(), b"m31_moe_finalize_out.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17"]
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
    if int(err) != 0:
        n = nvrtc.nvrtcGetProgramLogSize(prog)[1]
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(
            f"m31_moe_finalize_out: compile failed: {err}: {log.decode(errors='replace')}"
        )
    n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * n
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    nvrtc.nvrtcDestroyProgram(prog)
    ctx = _ck(d.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if ctx is None or int(ctx) == 0:
        raise RuntimeError("m31_moe_finalize_out: no current CUDA context after torch init")
    mod = _ck(d.cuModuleLoadData(cubin), "cuModuleLoadData")
    fn = _ck(d.cuModuleGetFunction(mod, _KERNEL_NAME.encode()), "cuModuleGetFunction")
    return mod, fn


def _function(key: _Key):
    """The loaded kernel for ``key``, compiling it on first use; None (with the reason kept) if that failed."""
    fn = _FUNCS.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        if key in _FUNCS:
            return _FUNCS[key]
        if key in _FAILED:
            return None
        try:
            _FUNCS[key] = _compile(key)
        except Exception as e:  # noqa: BLE001 - any failure leaves the caller on the native op
            _FAILED[key] = f"{type(e).__name__}: {e}"
            return None
    return _FUNCS[key]


def failure(device: int, w_dtype: torch.dtype, topk: int, shared: bool, cfg: int) -> Optional[str]:
    """Why the instance for these parameters could not be built, or None."""
    return _FAILED.get((device, w_dtype, topk, shared, cfg if cfg >= 0 else 0))


def outside_contract(
    fc2: torch.Tensor,
    weights: torch.Tensor,
    idx: torch.Tensor,
    shared: Optional[torch.Tensor],
    out: torch.Tensor,
    cfg: int,
) -> Optional[str]:
    """Why :func:`finalize_into` does not take these operands, or None (the native op's checks, plus ``out``)."""
    if not (fc2.is_cuda and weights.is_cuda and idx.is_cuda and out.is_cuda):
        return "an operand is not on CUDA"
    if fc2.dim() != 2 or fc2.dtype != torch.bfloat16 or fc2.stride(1) != 1:
        return "fc2 is not bf16 [P, H] rows"
    hidden = fc2.shape[1]
    if hidden % _VEC or fc2.stride(0) % _VEC or fc2.data_ptr() % 16:
        return "fc2 rows are not 16-byte aligned"
    if (
        weights.dim() != 2
        or not weights.is_contiguous()
        or weights.dtype not in _W_TYPES
        or idx.shape != weights.shape
        or idx.dtype != torch.int32
        or not idx.is_contiguous()
    ):
        return "weights / idx are not contiguous [T, K] (fp32 / bf16, int32)"
    if weights.device != fc2.device or idx.device != fc2.device:
        return "weights / idx are not on fc2's device"
    tokens, topk = weights.shape
    if not 1 <= topk <= 10:
        return f"top_k {topk} outside 1..10"
    if cfg not in CFG_TABLE and cfg != -1:
        return f"cfg {cfg}"
    for name, t in (("shared", shared), ("out", out)):
        if t is None:
            continue
        if (
            t.dtype != torch.bfloat16
            or t.dim() != 2
            or tuple(t.shape) != (tokens, hidden)
            or not t.is_contiguous()
            or t.data_ptr() % 16
            or t.device != fc2.device
        ):
            return f"{name} is not a contiguous, 16-byte aligned bf16 [T, H] on fc2's device"
    if shared is not None and shared.data_ptr() == out.data_ptr():
        return "out aliases shared"
    # A CTA may write a row of ``out`` before another CTA reads the same bytes as a row of ``fc2``.
    if tokens and fc2.shape[0]:
        fc2_elems = (fc2.shape[0] - 1) * fc2.stride(0) + hidden
        fc2_end = fc2.data_ptr() + fc2_elems * fc2.element_size()
        out_end = out.data_ptr() + out.numel() * out.element_size()
        if out.data_ptr() < fc2_end and fc2.data_ptr() < out_end:
            return "out overlaps fc2"
    return None


def finalize_into(
    fc2: torch.Tensor,
    weights: torch.Tensor,
    idx: torch.Tensor,
    shared: Optional[torch.Tensor],
    out: torch.Tensor,
    cfg: int = -1,
) -> bool:
    """Write the finalize (+ ``shared``) into ``out``. False (nothing launched) when the kernel cannot be built here;
    the operands must pass :func:`outside_contract`."""
    tokens, topk = weights.shape
    hidden = fc2.shape[1]
    cfg = cfg if cfg >= 0 else 0
    key = (fc2.device.index, weights.dtype, int(topk), shared is not None, cfg)
    loaded = _function(key)
    if loaded is None:
        return False
    if tokens == 0:
        return True
    from cuda.bindings import driver as d

    _, fn = loaded
    threads, nv = CFG_TABLE[cfg]
    stream = d.CUstream(torch.cuda.current_stream(fc2.device).cuda_stream)
    args = (
        (
            ctypes.c_void_p(fc2.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(weights.data_ptr()),
            # Without shared the kernel never reads this pointer; cuda-python rejects a null c_void_p (value None).
            ctypes.c_void_p(shared.data_ptr() if shared is not None else fc2.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_longlong(fc2.stride(0)),
            ctypes.c_int(hidden),
        ),
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_int,
        ),
    )
    grid_y = (hidden + threads * nv * _VEC - 1) // (threads * nv * _VEC)
    _ck(
        d.cuLaunchKernel(fn, int(tokens), grid_y, 1, threads, 1, 1, 0, stream, args, 0),
        "cuLaunchKernel",
    )
    return True
