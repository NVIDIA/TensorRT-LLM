# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step-commit launch whose descriptor table travels in the kernel's parameter block.

``step_commit._step_commit_kernel`` reads its descriptor table, the COPY payloads
and the BLOCKS page rows from pinned host memory, so each program waits for two
or three dependent round trips over C2C (``count``, then the other fields, then
the payload or page row) before its store. ``InlineCommit`` packs the same
descriptors into one ``__grid_constant__`` struct of at most ``PARAM_BYTES``,
with the COPY payloads and the BLOCKS page indices copied into it on the host,
and launches ``step_commit_inline`` with it: a program reads its row from the
constant bank and touches only the memory its descriptor writes, plus the
device source of a GATHER.

Semantics (per descriptor, as ``step_commit``'s module docstring states them,
in the same int32 arithmetic): ``COPY`` ``dst[i] = staged[i]``; ``GATHER``
``dst[i * a6] = src[i * a4] + a5``; ``FILL`` ``dst[i * a6] = a5``; ``BLOCKS``
``K[i] = 0`` if the page index is -1 else ``a4 * page``, ``V[i]`` the same plus
``a5`` (``V`` at ``a6``). The host reads the COPY payload from the staging block
and the BLOCKS page indices from the pinned page row at ``pack`` time; the
stock kernel reads the same bytes at its execution time, and both are written
by the host before the launch and not changed until the launch has executed
(``step_commit.StepStaging``'s lifetime rule, and the executor's wait for the
step's prepare event before it changes the host page table), so the words are
the same.

The launch does not read the staging block after ``launch`` returns: the
driver copies the parameter block when the launch is issued.

The kernel is compiled with NVRTC for the current device on first use and
launched through the driver API on the current stream. When the toolchain or
the launch is unavailable, ``available()`` is False and callers keep the stock
kernel.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Optional

import numpy as np
import torch

#: Descriptor rows and 32-bit payload words one parameter block holds; a launch with
#: more falls back to the stock kernel.
MAX_ROWS = 40
MAX_WORDS = 520
#: Threads per program, and elements one program covers (as the stock kernel's ``_BLOCK``).
_THREADS = 256
_BLOCK = 1024

_SOURCE = r"""
#define MAX_ROWS %(rows)d
#define MAX_WORDS %(words)d
struct Row {
  long long dst;
  long long src;   // GATHER: device address; COPY / BLOCKS: first word in words[]
  long long a4;    // GATHER: source stride; BLOCKS: scale
  long long a6;    // GATHER / FILL: destination stride; BLOCKS: V address
  int kind;
  int count;
  int a5;          // GATHER: add; FILL: value; BLOCKS: kv offset
  int pad;
};
struct Table {
  int n;
  int words_used;
  Row rows[MAX_ROWS];
  int words[MAX_WORDS];
};
extern "C" __global__ void step_commit_inline(const __grid_constant__ Table t) {
  if ((int)blockIdx.x >= t.n) return;
  const Row r = t.rows[blockIdx.x];
  const int stride = (int)(blockDim.x * gridDim.y);
  int* dst = reinterpret_cast<int*>(r.dst);
  for (int i = (int)(blockIdx.y * blockDim.x + threadIdx.x); i < r.count; i += stride) {
    if (r.kind == 0) {
      dst[i] = t.words[r.src + i];
    } else if (r.kind == 1) {
      const int* src = reinterpret_cast<const int*>(r.src);
      unsigned v = (unsigned)src[(long long)i * r.a4] + (unsigned)r.a5;
      dst[(long long)i * r.a6] = (int)v;
    } else if (r.kind == 2) {
      dst[(long long)i * r.a6] = r.a5;
    } else {
      int page = t.words[r.src + i];
      unsigned scaled = (unsigned)page * (unsigned)(int)r.a4;
      int* dst_v = reinterpret_cast<int*>(r.a6);
      dst[i] = page == -1 ? 0 : (int)scaled;
      dst_v[i] = page == -1 ? 0 : (int)(scaled + (unsigned)r.a5);
    }
  }
}
"""


class _Row(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_int64),
        ("src", ctypes.c_int64),
        ("a4", ctypes.c_int64),
        ("a6", ctypes.c_int64),
        ("kind", ctypes.c_int32),
        ("count", ctypes.c_int32),
        ("a5", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _Table(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int32),
        ("words_used", ctypes.c_int32),
        ("rows", _Row * MAX_ROWS),
        ("words", ctypes.c_int32 * MAX_WORDS),
    ]


#: Bytes of the parameter block (must stay within the classic 4 KB kernel-parameter limit).
PARAM_BYTES = ctypes.sizeof(_Table)
assert PARAM_BYTES <= 4096, PARAM_BYTES

_ROW_DTYPE = np.dtype(
    [
        ("dst", "<i8"),
        ("src", "<i8"),
        ("a4", "<i8"),
        ("a6", "<i8"),
        ("kind", "<i4"),
        ("count", "<i4"),
        ("a5", "<i4"),
        ("pad", "<i4"),
    ]
)
assert _ROW_DTYPE.itemsize == ctypes.sizeof(_Row)


def _to_i32(v: np.ndarray) -> np.ndarray:
    """The low 32 bits of int64 values as int32 (Triton's ``.to(tl.int32)`` of an int64)."""
    return (v & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


class _Driver:
    """The NVRTC-built module of one device and the driver entry points (loaded once per device)."""

    def __init__(self, device: int):
        self.device = device
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.cuda.cuLaunchKernel.restype = ctypes.c_int
        self.cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.cuda.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.cuda.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        with torch.cuda.device(device):
            torch.cuda.current_stream()  # the primary context exists and is current
            self.ctx = self._context()
            image = self._compile(torch.cuda.get_device_capability(device))
            module = ctypes.c_void_p()
            self._check(self.cuda.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
            func = ctypes.c_void_p()
            self._check(
                self.cuda.cuModuleGetFunction(ctypes.byref(func), module, b"step_commit_inline"),
                "cuModuleGetFunction",
            )
        self.module, self.func = module, func

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed with CUresult {rc}")

    def _context(self) -> ctypes.c_void_p:
        ctx = ctypes.c_void_p()
        self._check(self.cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            self._check(
                self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.device),
                "cuDevicePrimaryCtxRetain",
            )
            self._check(self.cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        return ctx

    @staticmethod
    def _compile(capability: tuple[int, int]) -> bytes:
        """The ``step_commit_inline`` image for sm_107; RuntimeError for any other ``capability``."""
        if tuple(capability) != (10, 7):
            raise RuntimeError(
                f"step_commit_inline is built for sm_107 only, not compute capability {capability}"
            )
        from cuda.bindings import nvrtc

        def ok(result):
            err, *rest = result
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"NVRTC: {nvrtc.nvrtcGetErrorString(err)[1]}")
            return rest[0] if len(rest) == 1 else rest

        source = (_SOURCE % {"rows": MAX_ROWS, "words": MAX_WORDS}).encode()
        failures = []
        # SASS for the device; else PTX for the driver to finish (an NVRTC without this target).
        for arch, binary in (("sm_107", True), ("compute_107", False)):
            prog = ok(nvrtc.nvrtcCreateProgram(source, b"step_commit_inline.cu", 0, [], []))
            try:
                opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
                err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
                if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    size = ok(nvrtc.nvrtcGetProgramLogSize(prog))
                    log = b" " * size
                    nvrtc.nvrtcGetProgramLog(prog, log)
                    failures.append(f"{arch}: {log.decode(errors='replace').strip()}")
                    continue
                if binary:
                    size = ok(nvrtc.nvrtcGetCUBINSize(prog))
                    image = b" " * size
                    err = nvrtc.nvrtcGetCUBIN(prog, image)[0]
                else:
                    size = ok(nvrtc.nvrtcGetPTXSize(prog))
                    image = b" " * size
                    err = nvrtc.nvrtcGetPTX(prog, image)[0]
                if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    failures.append(f"{arch}: {nvrtc.nvrtcGetErrorString(err)[1]}")
                    continue
                return image
            finally:
                nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(f"NVRTC compile failed: {failures}")

    def launch(self, table: _Table, rows: int, chunks: int, stream: int) -> None:
        ctx = ctypes.c_void_p()
        self.cuda.cuCtxGetCurrent(ctypes.byref(ctx))
        if not ctx.value:
            self._check(self.cuda.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        params = (ctypes.c_void_p * 1)(ctypes.addressof(table))
        self._check(
            self.cuda.cuLaunchKernel(
                self.func,
                rows,
                chunks,
                1,
                _THREADS,
                1,
                1,
                0,
                stream,
                ctypes.cast(params, ctypes.c_void_p),
                None,
            ),
            "cuLaunchKernel",
        )


_drivers: dict[int, Optional[_Driver]] = {}
_errors: dict[int, str] = {}
_lock = threading.Lock()


def _driver(device: int) -> Optional[_Driver]:
    with _lock:
        if device not in _drivers:
            try:
                _drivers[device] = _Driver(device)
            except (OSError, RuntimeError, ImportError, AttributeError) as exc:
                _drivers[device] = None
                _errors[device] = repr(exc)
        return _drivers[device]


def available(device: Optional[int] = None) -> bool:
    """Whether the inline kernel is built and launchable on ``device`` (the current one by default)."""
    return _driver(torch.cuda.current_device() if device is None else device) is not None


def unavailable_reason(device: Optional[int] = None) -> Optional[str]:
    """Why ``available`` is False on ``device``, or None."""
    device = torch.cuda.current_device() if device is None else device
    _driver(device)
    return _errors.get(device)


class InlineCommit:
    """Packs descriptor rows of a ``StepStaging`` into a parameter block and launches them."""

    def __init__(self):
        self._table = _Table()
        raw = np.frombuffer((ctypes.c_uint8 * PARAM_BYTES).from_buffer(self._table), dtype=np.uint8)
        rows_at, words_at = _Table.rows.offset, _Table.words.offset
        self._rows = raw[rows_at : rows_at + MAX_ROWS * _ROW_DTYPE.itemsize].view(_ROW_DTYPE)
        self._words = raw[words_at : words_at + 4 * MAX_WORDS].view(np.int32)
        self._max_count = 0

    def pack(self, desc: np.ndarray, data: np.ndarray, data_base: int) -> bool:
        """Pack ``desc`` (int64 ``[k, DESC_FIELDS]``, every count positive) into the parameter block.

        Args:
            desc: the descriptor rows to launch, in ``StepStaging``'s layout.
            data: the staging block's data area (uint8), holding every COPY payload.
            data_base: the data area's address (COPY sources are addresses inside it).

        Returns:
            False, with nothing packed, when the rows or their payload words do not fit, or a
            COPY source is not inside ``data``.
        """
        k = desc.shape[0]
        if k == 0 or k > MAX_ROWS:
            return False
        kinds = desc[:, 0]
        counts = desc[:, 3]
        inline = (kinds == 0) | (kinds == 3)
        if int(counts[inline].sum()) > MAX_WORDS or int(counts.max()) >= 1 << 31:
            return False
        copies = kinds == 0
        offs = desc[copies, 2] - data_base
        if offs.size and (
            int(offs.min()) < 0 or int((offs + 4 * counts[copies]).max()) > data.size
        ):
            return False
        rows = self._rows[:k]
        rows["dst"] = desc[:, 1]
        rows["a4"] = desc[:, 4]
        rows["a6"] = desc[:, 6]
        rows["kind"] = kinds
        rows["count"] = counts
        rows["a5"] = _to_i32(desc[:, 5])
        rows["pad"] = 0
        src = desc[:, 2].copy()
        used = 0
        for i in np.flatnonzero(inline):
            n = int(counts[i])
            if kinds[i] == 0:
                off = int(desc[i, 2]) - data_base
                self._words[used : used + n] = data[off : off + 4 * n].view(np.int32)
            else:
                page_row = (ctypes.c_int32 * n).from_address(int(desc[i, 2]))
                self._words[used : used + n] = np.frombuffer(page_row, dtype=np.int32)
            src[i] = used
            used += n
        rows["src"] = src
        self._table.n = k
        self._table.words_used = used
        self._max_count = int(counts.max())
        return True

    def launch(self, stream: Optional[int] = None) -> None:
        """Launch the packed rows on ``stream`` (the current stream's handle by default)."""
        driver = _driver(torch.cuda.current_device())
        if driver is None:
            raise RuntimeError(f"step_commit_inline unavailable: {unavailable_reason()}")
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        chunks = max(1, -(-self._max_count // _BLOCK))
        driver.launch(self._table, self._table.n, chunks, stream)
