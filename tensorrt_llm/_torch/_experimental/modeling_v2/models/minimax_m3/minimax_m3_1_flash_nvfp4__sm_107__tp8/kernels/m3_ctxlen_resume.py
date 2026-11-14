# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8 CTXLEN_FUSE: the DFlash drafter's stored-context resume update in one CUDA C++ launch.

The in-tree first drafter inputs (``DFlashWorker.prepare_1st_drafter_inputs``, manager-bound pool) re-derive each
generation request's drafter context length from the target position of its first input token, for slots whose
context is stored:

    first_input = num_ctx_tokens + tokens_per_req * arange(num_gens)
    ctx_len[slots] = where(ctx_stored[slots], position_ids[first_input].long().clamp(max=max_ctx), ctx_len[slots])

That is nine eager kernels per step in the captured decode graph (multiply, add, two gathers, cast, clamp, gather,
select, index_put). :func:`ctxlen_resume_` does the same in one launch: thread ``i`` reads ``slots[i]``, and when
that slot's context is stored writes ``min(int64(position_ids[num_ctx_tokens + tokens_per_req * i]), max_ctx)`` to
``ctx_len[slot]``, cast to ``ctx_len``'s dtype; a slot whose context is not stored keeps its length. Integer only,
so the result is bitwise the eager form's (slots are distinct per step, as the eager ``index_put`` also assumes).

The kernel is CUDA C++ compiled at runtime with NVRTC for this device (as the family ``kernels/m3ll/locality`` probe)
and launched with the driver API on the current torch stream, so CUDA-graph capture records it like any other launch.
Compilation runs outside capture (:func:`prepare`); a failure there leaves the item off (the eager form runs) and says
why.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Dict, Optional, Tuple

import torch

_LOCK = threading.Lock()
_KERNEL_NAME = "tp8_ctxlen_resume"
_THREADS = 128  # capacity: fixed: one thread per generation request; a CTA covers 128 requests

# position_ids dtype -> C type; ctx_len dtype -> C type
_POS_TYPES = {torch.int32: "int", torch.int64: "long long"}
_LEN_TYPES = {torch.int32: "int", torch.int64: "long long"}
_SLOT_TYPES = {torch.int32: "int", torch.int64: "long long"}

_SRC = r"""
typedef POS_T pos_t;
typedef LEN_T len_t;
typedef SLOT_T slot_t;
extern "C" __global__ void __launch_bounds__(128) tp8_ctxlen_resume(
    const pos_t* __restrict__ pos, long long pos_stride, const slot_t* __restrict__ slots,
    const unsigned char* __restrict__ stored, len_t* ctx_len, long long first, long long tokens_per_req,
    int num_gens, long long max_ctx)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_gens) {
        long long const slot = (long long) slots[i];
        if (stored[slot]) {
            long long const v = (long long) pos[(first + tokens_per_req * (long long) i) * pos_stride];
            ctx_len[slot] = (len_t) (v < max_ctx ? v : max_ctx);
        }
    }
}
"""

# (device index, pos dtype, len dtype, slot dtype) -> (module, function)
_FUNCS: Dict[Tuple[int, torch.dtype, torch.dtype, torch.dtype], object] = {}
_FAILED: Dict[Tuple[int, torch.dtype, torch.dtype, torch.dtype], str] = {}


def _ck(res, what: str):
    """cuda-python returns (CUresult, *values): raise on an error, return the values."""
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"CTXLEN_FUSE: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _compile(device: int, pos_dtype, len_dtype, slot_dtype):
    """NVRTC-compile the kernel for ``device`` and load it into torch's (primary) context of that device."""
    from cuda.bindings import driver as d
    from cuda.bindings import nvrtc

    torch.cuda.set_device(device)
    torch.cuda.init()
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
        _SRC.replace("POS_T", _POS_TYPES[pos_dtype])
        .replace("LEN_T", _LEN_TYPES[len_dtype])
        .replace("SLOT_T", _SLOT_TYPES[slot_dtype])
    )
    prog = _ck(
        nvrtc.nvrtcCreateProgram(src.encode(), b"tp8_ctxlen_resume.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17"]
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
    if int(err) != 0:
        n = nvrtc.nvrtcGetProgramLogSize(prog)[1]
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"CTXLEN_FUSE: compile failed: {err}: {log.decode(errors='replace')}")
    n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * n
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    nvrtc.nvrtcDestroyProgram(prog)
    ctx = _ck(d.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if ctx is None or int(ctx) == 0:
        raise RuntimeError("CTXLEN_FUSE: no current CUDA context after torch init")
    mod = _ck(d.cuModuleLoadData(cubin), "cuModuleLoadData")
    fn = _ck(d.cuModuleGetFunction(mod, _KERNEL_NAME.encode()), "cuModuleGetFunction")
    return mod, fn


def _key(position_ids: torch.Tensor, ctx_len: torch.Tensor, slots: torch.Tensor):
    return (ctx_len.device.index, position_ids.dtype, ctx_len.dtype, slots.dtype)


def prepare(
    device: int,
    pos_dtypes=(torch.int32, torch.int64),
    len_dtype=torch.int64,
    slot_dtype=torch.int64,
) -> Optional[str]:
    """Compile the kernels for ``device`` outside capture. Returns None when they are ready, else the reason the item
    stays off."""
    reason = None
    for pos_dtype in pos_dtypes:
        key = (device, pos_dtype, len_dtype, slot_dtype)
        with _LOCK:
            if key in _FUNCS:
                continue
            try:
                _FUNCS[key] = _compile(device, pos_dtype, len_dtype, slot_dtype)
            except Exception as e:  # noqa: BLE001 - any failure leaves the eager form on
                _FAILED[key] = f"{type(e).__name__}: {e}"
                reason = _FAILED[key]
    return reason


def outside_contract(
    position_ids: torch.Tensor, slots: torch.Tensor, ctx_stored: torch.Tensor, ctx_len: torch.Tensor
) -> Optional[str]:
    """Why the launch does not take these operands (the eager form then runs), or None."""
    if not (position_ids.is_cuda and slots.is_cuda and ctx_stored.is_cuda and ctx_len.is_cuda):
        return "an operand is not on CUDA"
    if len({t.device for t in (position_ids, slots, ctx_stored, ctx_len)}) != 1:
        return "operands on different devices"
    if position_ids.dim() != 1 or slots.dim() != 1 or ctx_stored.dim() != 1 or ctx_len.dim() != 1:
        return (
            f"ranks pos {position_ids.dim()} slots {slots.dim()} stored {ctx_stored.dim()} "
            f"len {ctx_len.dim()}"
        )
    if (
        position_ids.dtype not in _POS_TYPES
        or ctx_len.dtype not in _LEN_TYPES
        or slots.dtype not in _SLOT_TYPES
    ):
        return f"dtypes pos {position_ids.dtype} len {ctx_len.dtype} slots {slots.dtype}"
    if ctx_stored.dtype != torch.bool:
        return f"ctx_stored dtype {ctx_stored.dtype}"
    if slots.stride(0) != 1 or ctx_stored.stride(0) != 1 or ctx_len.stride(0) != 1:
        return "non-unit stride on slots / ctx_stored / ctx_len"
    if _FUNCS.get(_key(position_ids, ctx_len, slots)) is None:
        return "kernel not compiled for these dtypes on this device"
    return None


def ctxlen_resume_(
    position_ids: torch.Tensor,
    slots: torch.Tensor,
    ctx_stored: torch.Tensor,
    ctx_len: torch.Tensor,
    num_ctx_tokens: int,
    tokens_per_req: int,
    num_gens: int,
    max_ctx: int,
) -> None:
    """The one-launch resume update, in place on ``ctx_len`` (operands checked by :func:`outside_contract`)."""
    if num_gens <= 0:
        return
    from cuda.bindings import driver as d

    _, fn = _FUNCS[_key(position_ids, ctx_len, slots)]
    stream = d.CUstream(torch.cuda.current_stream(ctx_len.device).cuda_stream)
    args = (
        (
            ctypes.c_void_p(position_ids.data_ptr()),
            ctypes.c_longlong(position_ids.stride(0)),
            ctypes.c_void_p(slots.data_ptr()),
            ctypes.c_void_p(ctx_stored.data_ptr()),
            ctypes.c_void_p(ctx_len.data_ptr()),
            ctypes.c_longlong(int(num_ctx_tokens)),
            ctypes.c_longlong(int(tokens_per_req)),
            ctypes.c_int(int(num_gens)),
            ctypes.c_longlong(int(max_ctx)),
        ),
        (
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
            ctypes.c_longlong,
        ),
    )
    grid = (int(num_gens) + _THREADS - 1) // _THREADS
    _ck(d.cuLaunchKernel(fn, grid, 1, 1, _THREADS, 1, 1, 0, stream, args, 0), "cuLaunchKernel")


def ctxlen_resume_eager_(
    position_ids: torch.Tensor,
    slots: torch.Tensor,
    ctx_stored: torch.Tensor,
    ctx_len: torch.Tensor,
    num_ctx_tokens: int,
    tokens_per_req: int,
    num_gens: int,
    max_ctx: int,
) -> None:
    """The in-tree eager form (the reference of the unit tests), in place on ``ctx_len``."""
    first_input = num_ctx_tokens + tokens_per_req * torch.arange(
        0, num_gens, dtype=torch.long, device=ctx_len.device
    )
    ctx_len[slots] = torch.where(
        ctx_stored[slots],
        position_ids[first_input].long().clamp(max=max_ctx),
        ctx_len[slots],
    )
