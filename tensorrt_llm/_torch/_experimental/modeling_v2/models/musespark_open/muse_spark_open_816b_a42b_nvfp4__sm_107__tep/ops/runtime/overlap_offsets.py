# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overlap scheduler's two input corrections of a captured decode step as one launch.

With the overlap scheduler and speculative decoding, ``_preprocess_inputs`` adds the
previous step's accepted-token offsets to the generation rows' position ids and KV
lengths (``position_ids += previous_pos_id_offsets_cuda`` and ``kv_lens_cuda +=
previous_kv_lens_offsets_cuda``). In a decode CUDA graph these are its first two
nodes, two 1-CTA ``CUDAFunctor_add<int>`` kernels back to back, and the target's first
kernel waits for the second. ``OverlapOffsetsFold`` is the model's
``RuntimeHooks.overlap_offsets_applier``: inside a graph capture it issues both adds as
one 1-CTA kernel (``ia_fold``), so the graph starts with one launch fewer.

Forms (``OverlapOffsetsFold``'s ``fold``): ``1`` the fold; ``2`` the fold, which also
lets its programmatic dependent (the target's first kernel when that is launched with
PDL) launch at the fold's entry (``griddepcontrol.launch_dependents``). The
``step_loop_inline_commits`` fast path declares the applier with ``2`` (``build_applier``);
without it the model declares no applier and the two adds run as built.

Values: each element is the int32 sum of the two built-in adds, computed with the
same wrap-around (unsigned 32-bit addition), so every word equals the built-in's.
The applier takes only a step being captured into a CUDA graph whose four views are
int32, contiguous, on the current device and of equal pairwise length; eager steps,
graph warm-up runs and every other shape keep the two built-in adds (the call site
runs them when ``apply`` returns False). Capture-time symmetry: the capture's
post-processing subtracts the offsets eagerly, as it does after the built-in adds.

Ownership and order: the inputs (the four buffers) are written before the graph
launch by the step's input preparation and not inside the graph before this node;
the fold is their only writer inside the graph and runs where the first built-in add
ran (the graph's first node on the main stream). Its readers are the target's
kernels after it: the first of them is a programmatic dependent (layer 0's QKV table
gather, ``kernels/tt_table/target_l0.py``, which waits for the fold's completion
before its first read and releases its own dependents only after that wait), and
every side branch forked after it has a full edge to it. So every reader that read
the built-in adds' result reads the fold's, including the sliding-window split's
pre-wait loads of the positions and KV lengths (``kernels/swa_block/attention.py``:
the writer reaches layer 0's first kernel through a full dependency or a waiting
programmatic one). With ``2`` the fold triggers at
entry, which releases only the gather's launch; the gather still waits for the
fold's completion before it reads anything or triggers.

Attribution (``OverlapOffsetsFold``'s ``stamps=<K>``; ``0``, off, in the applier the
model declares; for attribution only, not for timed runs): each captured correction
records ``%globaltimer`` at its kernel's entry and exit into a device ring
(``RING_RECORDS`` records of ``RECORD_WORDS`` int64: ``n``, first launch entry, exit,
second launch entry, exit, ``kv_lens[0]`` and ``position_ids[0]`` after the add,
form), and after every K-th graph replay the ring is copied to pinned host memory on
the replay's stream and written by a background thread to
``DEFAULT_STAMPS_DIR/ia-rank<r>-<nnnn>.{bin,json}``. With ``fold`` ``0`` the stamps
keep the built-in shape: two 1-CTA launches, one per add (form 1); with the fold,
one launch (form 0).
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import threading
import time
from typing import List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

#: Directory the attribution stamps are written to.
DEFAULT_STAMPS_DIR = "/logs/ia_stamps"

#: Threads of the 1-CTA kernel; it loops over longer views.
_THREADS = 128
#: Longest view the applier takes (a decode step's generation tokens / requests).
MAX_ELEMENTS = 1 << 16
#: Ring layout (int64 words): word 0 counts the recorded steps, then the records.
RING_HEADER = 8
RING_RECORDS = 1024
RECORD_WORDS = 8
#: Snapshots per written file.
SNAPSHOT_BATCH = 16

_SOURCE = r"""
#define RING_HEADER %(header)d
#define RING_RECORDS %(records)d
#define RECORD_WORDS %(record)d
struct Args {
  long long pos;      // int32 [n_pos], updated in place
  long long pos_off;  // int32 [n_pos]
  long long kv;       // int32 [n_kv], updated in place
  long long kv_off;   // int32 [n_kv]
  long long ring;     // 0, or the attribution ring
  int n_pos;
  int n_kv;
  int trigger;        // 1: release the programmatic dependent at entry
  int role;           // stamps: 0 both adds, 1 the positions add, 2 the KV-lengths add
};
__device__ __forceinline__ long long ia_now() {
  long long t;
  asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t));
  return t;
}
extern "C" __global__ void __launch_bounds__(%(threads)d) ia_fold(const __grid_constant__ Args a) {
  long long t0 = 0;
  if (a.ring != 0 && threadIdx.x == 0) t0 = ia_now();
  if (a.trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int* pos = reinterpret_cast<int*>(a.pos);
  const int* pos_off = reinterpret_cast<const int*>(a.pos_off);
  int* kv = reinterpret_cast<int*>(a.kv);
  const int* kv_off = reinterpret_cast<const int*>(a.kv_off);
  for (int i = (int)threadIdx.x; i < a.n_pos + a.n_kv; i += (int)blockDim.x) {
    if (i < a.n_pos) {
      pos[i] = (int)((unsigned)pos[i] + (unsigned)pos_off[i]);
    } else {
      const int j = i - a.n_pos;
      kv[j] = (int)((unsigned)kv[j] + (unsigned)kv_off[j]);
    }
  }
  if (a.ring == 0) return;
  __syncthreads();
  if (threadIdx.x != 0) return;
  long long* ring = reinterpret_cast<long long*>(a.ring);
  const long long t1 = ia_now();
  const long long n = ring[0];
  long long* r = ring + RING_HEADER + RECORD_WORDS * (n & (RING_RECORDS - 1));
  if (a.role != 2) {
    r[0] = n;
    r[1] = t0;
    r[2] = t1;
    r[6] = a.n_pos > 0 ? pos[0] : -1;
    r[7] = a.role == 0 ? 0 : 1;
  }
  if (a.role == 0) {
    r[3] = 0;
    r[4] = 0;
  }
  if (a.role != 1) {
    if (a.role == 2) {
      r[3] = t0;
      r[4] = t1;
    }
    r[5] = a.n_kv > 0 ? kv[0] : -1;
    ring[0] = n + 1;
  }
}
"""


class _Args(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_int64),
        ("pos_off", ctypes.c_int64),
        ("kv", ctypes.c_int64),
        ("kv_off", ctypes.c_int64),
        ("ring", ctypes.c_int64),
        ("n_pos", ctypes.c_int32),
        ("n_kv", ctypes.c_int32),
        ("trigger", ctypes.c_int32),
        ("role", ctypes.c_int32),
    ]


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with CUresult {rc}")


def _compile(capability: tuple) -> bytes:
    """The ``ia_fold`` image for sm_107: SASS, else PTX for the driver to finish.

    Raises:
        RuntimeError: for a ``capability`` other than (10, 7).
    """
    if tuple(capability) != (10, 7):
        raise RuntimeError(f"ia_fold is built for sm_107 only, not compute capability {capability}")
    from cuda.bindings import nvrtc

    def ok(result):
        err, *rest = result
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC: {nvrtc.nvrtcGetErrorString(err)[1]}")
        return rest[0] if len(rest) == 1 else rest

    source = (
        _SOURCE
        % {
            "header": RING_HEADER,
            "records": RING_RECORDS,
            "record": RECORD_WORDS,
            "threads": _THREADS,
        }
    ).encode()
    failures = []
    for arch, binary in (("sm_107", True), ("compute_107", False)):
        prog = ok(nvrtc.nvrtcCreateProgram(source, b"ia_fold.cu", 0, [], []))
        try:
            opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
            err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                size = ok(nvrtc.nvrtcGetProgramLogSize(prog))
                log = b" " * size
                nvrtc.nvrtcGetProgramLog(prog, log)
                failures.append(f"{arch}: {log.decode(errors='replace').strip()}")
                continue
            get_size, get = (
                (nvrtc.nvrtcGetCUBINSize, nvrtc.nvrtcGetCUBIN)
                if binary
                else (nvrtc.nvrtcGetPTXSize, nvrtc.nvrtcGetPTX)
            )
            image = b" " * ok(get_size(prog))
            err = get(prog, image)[0]
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                failures.append(f"{arch}: {nvrtc.nvrtcGetErrorString(err)[1]}")
                continue
            return image
        finally:
            nvrtc.nvrtcDestroyProgram(prog)
    raise RuntimeError(f"NVRTC compile of ia_fold failed: {failures}")


class _Kernel:
    """``ia_fold`` built for one device and launched through the driver API."""

    def __init__(self, device: int):
        self.device = device
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        cuda.cuLaunchKernel.restype = ctypes.c_int
        cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cuda.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.cuda = cuda
        with torch.cuda.device(device):
            torch.cuda.current_stream()  # the primary context exists and is current
            self.ctx = ctypes.c_void_p()
            _check(cuda.cuCtxGetCurrent(ctypes.byref(self.ctx)), "cuCtxGetCurrent")
            if not self.ctx.value:
                raise RuntimeError("no current CUDA context after torch initialised the device")
            image = _compile(torch.cuda.get_device_capability(device))
            module = ctypes.c_void_p()
            _check(cuda.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
            func = ctypes.c_void_p()
            _check(
                cuda.cuModuleGetFunction(ctypes.byref(func), module, b"ia_fold"),
                "cuModuleGetFunction",
            )
        self.module, self.func = module, func

    def launch(self, args: _Args, stream: int) -> None:
        ctx = ctypes.c_void_p()
        self.cuda.cuCtxGetCurrent(ctypes.byref(ctx))
        if not ctx.value:
            _check(self.cuda.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        params = (ctypes.c_void_p * 1)(ctypes.addressof(args))
        _check(
            self.cuda.cuLaunchKernel(
                self.func,
                1,
                1,
                1,
                _THREADS,
                1,
                1,
                0,
                stream,
                ctypes.cast(params, ctypes.c_void_p),
                None,
            ),
            "cuLaunchKernel ia_fold",
        )


def _rank() -> int:
    """The process's CUDA device index, which names its stamp files."""
    return int(torch.cuda.current_device())


def ring_words() -> int:
    return RING_HEADER + RECORD_WORDS * RING_RECORDS


class _Snapshots:
    """Copies the ring into pinned host slots after every K-th graph replay (stream-ordered
    after that replay) and writes full batches of slots from a background thread."""

    def __init__(self, ring: torch.Tensor, k: int, directory: str):
        self.ring = ring
        self.k = int(k)
        self.dir = directory
        self.rank = _rank()
        self.calls = 0
        self.files = 0
        words = ring.numel()
        self.sets = [
            [
                torch.empty(words, dtype=torch.int64, pin_memory=prefer_pinned())
                for _ in range(SNAPSHOT_BATCH)
            ]
            for _ in range(2)
        ]
        self.events = [[torch.cuda.Event() for _ in range(SNAPSHOT_BATCH)] for _ in range(2)]
        self.cur = 0
        self.index: List[dict] = []
        self.writer: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def after_replay(self) -> None:
        self.calls += 1
        if self.calls % self.k or torch.cuda.is_current_stream_capturing():
            return
        with self.lock:
            slot = len(self.index)
            host = self.sets[self.cur][slot]
            host.copy_(self.ring, non_blocking=True)
            self.events[self.cur][slot].record()
            self.index.append(dict(wall=time.time(), replay=self.calls, slot=slot))
            if len(self.index) >= SNAPSHOT_BATCH:
                self._flush()

    def _flush(self) -> None:
        if not self.index:
            return
        if self.writer is not None:
            self.writer.join()
        os.makedirs(self.dir, exist_ok=True)
        meta = dict(
            rank=self.rank,
            header=RING_HEADER,
            records=RING_RECORDS,
            record=RECORD_WORDS,
            fields=["n", "entry0", "exit0", "entry1", "exit1", "kv0", "pos0", "form"],
            snapshots=self.index,
        )
        base = os.path.join(self.dir, f"ia-rank{self.rank}-{self.files:04d}")
        n = len(self.index)
        self.writer = threading.Thread(
            target=_write,
            args=(base, self.sets[self.cur][:n], self.events[self.cur][:n], meta),
            daemon=True,
        )
        self.writer.start()
        self.files += 1
        self.index = []
        self.cur ^= 1

    def close(self) -> None:
        try:
            with self.lock:
                self._flush()
            if self.writer is not None:
                self.writer.join(timeout=30)
        except (RuntimeError, OSError) as exc:  # at exit: report, never raise
            print(
                f"[musespark_open] overlap offsets fold: stamp flush at exit failed: {exc!r}",
                flush=True,
            )


def _write(
    base: str, slots: List[torch.Tensor], events: List[torch.cuda.Event], meta: dict
) -> None:
    events[-1].synchronize()
    with open(base + ".bin", "wb") as f:
        for host in slots:
            f.write(host.numpy().tobytes())
    with open(base + ".json", "w") as f:
        json.dump(meta, f)


_REPLAY_HOOK = {}


def _install_replay_hook(snaps: _Snapshots) -> None:
    """Wrap ``CUDAGraphRunner.replay`` once so every replay counts toward the next snapshot."""
    if _REPLAY_HOOK:
        _REPLAY_HOOK["snaps"] = snaps
        return
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as _cgr

    original = _cgr.CUDAGraphRunner.replay

    def replay(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        _REPLAY_HOOK["snaps"].after_replay()
        return out

    _cgr.CUDAGraphRunner.replay = replay
    _REPLAY_HOOK.update(snaps=snaps, original=original)
    atexit.register(lambda: _REPLAY_HOOK["snaps"].close())


class OverlapOffsetsFold:
    """The model's ``OverlapOffsetsApplier`` (``tensorrt_llm/_torch/pyexecutor/runtime_hooks.py``).

    Args:
        fold: 0 to keep the built-in shape (two launches; only with ``stamps``), 1 the fold,
            2 the fold releasing its programmatic dependent at entry.
        stamps: snapshot period K of the attribution stamps (0: no stamps).
        device: the CUDA device of the views it takes (the current one by default).
    """

    def __init__(self, fold: int, stamps: int = 0, device: Optional[int] = None):
        if fold not in (0, 1, 2) or stamps < 0 or (fold == 0 and stamps == 0):
            raise ValueError(
                f"[musespark_open] overlap offsets fold: bad configuration fold={fold} stamps={stamps}"
            )
        self.fold = fold
        self.device = torch.cuda.current_device() if device is None else int(device)
        self.kernel = _Kernel(self.device)
        self.ring: Optional[torch.Tensor] = None
        self.snapshots: Optional[_Snapshots] = None
        if stamps:
            self.ring = torch.zeros(ring_words(), dtype=torch.int64, device=f"cuda:{self.device}")
            self.snapshots = _Snapshots(self.ring, stamps, DEFAULT_STAMPS_DIR)
        self.captured = 0
        self.declined = 0

    def _takes(self, *views: torch.Tensor) -> bool:
        pos, pos_off, kv, kv_off = views
        if pos.numel() != pos_off.numel() or kv.numel() != kv_off.numel():
            return False
        if pos.numel() + kv.numel() == 0 or pos.numel() + kv.numel() > MAX_ELEMENTS:
            return False
        return all(
            t.is_cuda
            and t.dtype == torch.int32
            and t.dim() == 1
            and t.is_contiguous()
            and t.device.index == self.device
            for t in views
        )

    def _args(self, pos, pos_off, kv, kv_off, n_pos: int, n_kv: int, role: int) -> _Args:
        ring = 0 if self.ring is None else int(self.ring.data_ptr())
        return _Args(
            int(pos.data_ptr()),
            int(pos_off.data_ptr()),
            int(kv.data_ptr()),
            int(kv_off.data_ptr()),
            ring,
            int(n_pos),
            int(n_kv),
            1 if self.fold == 2 else 0,
            int(role),
        )

    def apply(
        self,
        position_ids: torch.Tensor,
        position_offsets: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
    ) -> bool:
        views = (position_ids, position_offsets, kv_lens, kv_lens_offsets)
        if not torch.cuda.is_current_stream_capturing():
            return False
        if not self._takes(*views):
            self.declined += 1
            return False
        stream = torch.cuda.current_stream().cuda_stream
        n_pos, n_kv = position_ids.numel(), kv_lens.numel()
        if self.fold:
            self.kernel.launch(self._args(*views, n_pos, n_kv, 0), stream)
        else:
            # Attribution only: the built-in shape, one launch per add, each stamped.
            self.kernel.launch(self._args(*views, n_pos, 0, 1), stream)
            self.kernel.launch(self._args(*views, 0, n_kv, 2), stream)
        self.captured += 1
        return True

    def run_eager(self, *views: torch.Tensor) -> None:
        """Run the fold on the current stream outside a capture (tests and the install's check)."""
        if not self._takes(*views):
            raise ValueError(
                "[musespark_open] overlap offsets fold: views the applier does not take"
            )
        stream = torch.cuda.current_stream().cuda_stream
        n_pos, n_kv = views[0].numel(), views[2].numel()
        if self.fold:
            self.kernel.launch(self._args(*views, n_pos, n_kv, 0), stream)
        else:
            self.kernel.launch(self._args(*views, n_pos, 0, 1), stream)
            self.kernel.launch(self._args(*views, 0, n_kv, 2), stream)


def _self_check(applier: OverlapOffsetsFold) -> None:
    """One eager run against the two built-in adds on edge values (int32 wrap included)."""
    device = f"cuda:{applier.device}"
    imax, imin = 2**31 - 1, -(2**31)
    pos = torch.tensor([0, 5, imax, imin, -7, 123456, 9, 1], dtype=torch.int32, device=device)
    pos_off = torch.tensor([3, -5, 1, -1, 7, 0, 4, imax], dtype=torch.int32, device=device)
    kv = torch.tensor([450000, imax], dtype=torch.int32, device=device)
    kv_off = torch.tensor([-4, 2], dtype=torch.int32, device=device)
    want_pos, want_kv = pos + pos_off, kv + kv_off
    ring_before = None if applier.ring is None else applier.ring.clone()
    applier.run_eager(pos, pos_off, kv, kv_off)
    torch.cuda.current_stream().synchronize()
    if applier.ring is not None:
        applier.ring.copy_(ring_before)
    if not (torch.equal(pos, want_pos) and torch.equal(kv, want_kv)):
        raise RuntimeError(
            "[musespark_open] overlap offsets fold: install check differs from the built-in adds"
        )


def build_applier(fold: int, stamps: int = 0) -> OverlapOffsetsFold:
    """An ``OverlapOffsetsFold(fold, stamps)`` built and checked on the current device.

    The ``step_loop_inline_commits`` fast path declares ``build_applier(2)``.

    Raises:
        ValueError: on a configuration ``OverlapOffsetsFold`` does not take.
        RuntimeError: if the kernel cannot be built or differs from the built-in adds.
    """
    try:
        applier = OverlapOffsetsFold(fold, stamps)
    except (OSError, ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"[musespark_open] overlap offsets fold: kernel unavailable: {exc!r}"
        ) from exc
    _self_check(applier)
    if applier.snapshots is not None:
        _install_replay_hook(applier.snapshots)
    form = {
        0: "two stamped launches (the built-in shape)",
        1: "one launch",
        2: "one launch, dependent released at entry",
    }[fold]
    logger.info(
        f"[musespark_open] overlap offsets fold: the adds as {form} in captured decode graphs "
        f"(fold={fold}, stamps={stamps}"
        + (f", stamps into {applier.snapshots.dir}, ATTRIBUTION ONLY" if applier.snapshots else "")
        + "); install check vs the built-in adds passed"
    )
    return applier
