# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in GPU keepalive for the benchmark fill gate.

With ``TRTLLM_GPU_KEEPALIVE=1`` a generation worker that is blocked on the
benchmark fill gate (``TLLM_BENCHMARK_REQ_QUEUES_SIZE``) keeps its GPU visibly
busy instead of idling through the wait, so GPU-activity metrics do not read
0 while the context tier fills it. Off by default; does no useful work.

The primitive is a Triton kernel with one warp per CTA and two CTAs per SM
spinning on an FMA chain in ~100 ms chunks. SM-activity metrics count SMs
with at least one resident warp, so this reads like a loaded GPU at ~3% warp
occupancy and slows co-running kernels by only a few percent. Chunk length is
calibrated at runtime in ns per iteration, so nothing is GPU-specific.

Design points:

* The kernel is compiled in a subprocess first: a Triton compile failure can
  abort the process (SIGABRT), which no in-process ``try`` can catch. The
  child is polled from :meth:`tick` and never blocks the executor thread; if
  it does not pass, a low-duty ``torch.mm`` fallback is used.
* Everything device-side is allocated at the first tick, on the executor
  thread and outside the creator's ``executor_extra`` memory scope (which an
  engine sleep releases), and freed again when the gate opens.
* Work is launched only while the gate is closed (never during warmup) and
  drained when the gate opens, so it never overlaps CUDA-graph capture or a
  forward.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess  # nosec B404 - runs this very file with sys.executable to isolate a Triton compiler abort
import sys
import tempfile
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from logging import Logger

KEEPALIVE_ENV_VAR_NAME = "TRTLLM_GPU_KEEPALIVE"

_PERIOD_SEC = 0.1  # chunk length; keep under NVML's ~1/6 s utilization sample
_GRID_MULT = 2  # CTAs per SM, so every SM gets a warp even if another kernel holds some
_QUEUE_DEPTH = 2  # chunks queued ahead of the GPU; bounds the drain when the gate opens
_MM_DUTY = 0.05  # duty of the torch.mm fallback; a saturating GEMM slows other kernels ~8x
_MM_DIM = 4096
_SPIN_CALIBRATION_ITERS = 2_000_000
_SPIN_MIN_ITERS = 1_000
_SPIN_MAX_ITERS = (1 << 31) - 1  # n_iters is an unspecialized i32 kernel argument
_SELFTEST_TIMEOUT_SEC = 120.0
_SELFTEST_OK_MARKER = "KEEPALIVE_SELFTEST_OK"
_SELFTEST_RC_NO_CONTEXT = 3  # child could not create a CUDA context (e.g. EXCLUSIVE_PROCESS)
# Verdict per device for this process: PyExecutor may be built twice per rank
# (KV-cache estimation) and the ~15 s compile check cannot change in between.
_SELFTEST_VERDICT: dict[int, bool] = {}

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit(do_not_specialize=["n_iters"])
    def _spin_kernel(n_iters, sink_ptr):
        # The store keeps the chain alive so the loop is not optimised away.
        # do_not_specialize: one compiled variant for every iteration count,
        # so no JIT compile ever happens on the executor thread.
        x = tl.zeros([32], dtype=tl.float32) + tl.program_id(0)
        for _ in range(n_iters):
            x = x * 0.999 + 1.0
        tl.store(sink_ptr + tl.arange(0, 32), x)


def _log() -> "Logger":
    # Lazy: the subprocess self-test runs this file by path and must not
    # import the tensorrt_llm package.
    from tensorrt_llm.logger import logger

    return logger


class _SelftestChild:
    """The subprocess that compiles and launches the spin kernel once.

    Output goes to temporary files, not pipes, so the child can never block
    on a full pipe while nobody reads it.
    """

    def __init__(self, device_index: int):
        self._out = tempfile.TemporaryFile()
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell, interpreter is sys.executable
            [sys.executable, os.path.abspath(__file__), "--selftest", str(device_index)],
            stdout=self._out,
            stderr=self._err,
            env={**os.environ, "PYTHONSAFEPATH": "1"},  # keep this directory off sys.path
            start_new_session=True,  # own process group: kill() also reaches ptxas et al.
        )
        self.started = time.monotonic()

    def poll(self) -> int | None:
        return self._proc.poll()

    def kill(self) -> None:
        """Stop the whole compiler process tree, not just the Python wrapper."""
        try:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._proc.wait()
        finally:
            self._close()

    def output(self) -> tuple[bytes, str]:
        """(stdout, last stderr lines); call once, after the child exited."""
        self._out.seek(0)
        self._err.seek(0)
        out = self._out.read()
        tail = " | ".join(self._err.read().decode(errors="replace").strip().splitlines()[-3:])
        self._close()
        return out, tail

    def _close(self) -> None:
        self._out.close()
        self._err.close()


def _run_selftest(device_index: int) -> int:
    """Subprocess entry: compile, launch once, print the marker. Exit code is the verdict."""
    if not _HAVE_TRITON:
        return 2
    try:
        torch.cuda.set_device(device_index)
        sink = torch.zeros(32, dtype=torch.float32, device=f"cuda:{device_index}")
    except RuntimeError:
        return _SELFTEST_RC_NO_CONTEXT
    _spin_kernel[(4,)](1000, sink, num_warps=1)
    torch.cuda.synchronize()
    print(_SELFTEST_OK_MARKER, flush=True)
    return 0


class GpuKeepalive:
    """Keeps the GPU visibly non-idle while the executor waits at the fill gate.

    Call :meth:`tick` on every iteration that finds the gate closed,
    :meth:`drain` when it opens and :meth:`close` when the executor loop
    exits. All are rank-local, issue no collectives and never raise: any
    error disables the keepalive.
    """

    def __init__(self, device):
        # No CUDA work here: PyExecutor is constructed inside a memory scope
        # that sleep() releases. Everything device-side happens in _initialize.
        self.device = torch.device(device if not isinstance(device, int) else f"cuda:{device}")
        self.period_s = _PERIOD_SEC
        self.queue_depth = _QUEUE_DEPTH
        self.mode: str | None = None
        self._initialized = False
        self._disabled = False
        self._stream: torch.cuda.Stream | None = None
        self._selftest: _SelftestChild | None = None
        self._inflight: collections.deque[torch.cuda.Event] = collections.deque()
        self._launches = 0
        self._last_launch = 0.0
        self._active = False

    # ------------------------------------------------------------------ setup
    def _initialize(self) -> bool:
        """Device-side setup, driven from tick(); False while the self-test child runs."""
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
            num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            self._grid = _GRID_MULT * num_sms
        device_index = self.device.index if self.device.index is not None else 0
        if device_index not in _SELFTEST_VERDICT:
            verdict = self._poll_selftest(device_index)
            if verdict is None:
                return False
            _SELFTEST_VERDICT[device_index] = verdict
        if _SELFTEST_VERDICT[device_index]:
            self._init_spin()
        else:
            self._init_mm()
        self._initialized = True
        _log().info(
            f"[gpu-keepalive] enabled mode={self.mode} grid={self._grid} "
            f"period={self.period_s}s depth={self.queue_depth}"
        )
        return True

    def _poll_selftest(self, device_index: int) -> bool | None:
        """Drive the subprocess compile check without blocking; None while pending."""
        if not _HAVE_TRITON:
            return False
        if self._selftest is None:
            self._selftest = _SelftestChild(device_index)
            return None
        rc = self._selftest.poll()
        if rc is None:
            if time.monotonic() - self._selftest.started < _SELFTEST_TIMEOUT_SEC:
                return None
            self._selftest.kill()
            _log().warning(
                f"[gpu-keepalive] spin self-test did not finish within {_SELFTEST_TIMEOUT_SEC:.0f}s; "
                "using torch.mm"
            )
            verdict = False
        else:
            out, tail = self._selftest.output()
            verdict = rc == 0 and _SELFTEST_OK_MARKER.encode() in out
            if not verdict and rc == _SELFTEST_RC_NO_CONTEXT:
                _log().warning(
                    "[gpu-keepalive] spin self-test could not create a CUDA context in a child "
                    f"process (compute mode EXCLUSIVE_PROCESS?): {tail}; using torch.mm"
                )
            elif not verdict:
                _log().warning(
                    f"[gpu-keepalive] spin self-test failed (rc={rc}): {tail}; using torch.mm"
                )
        self._selftest = None
        return verdict

    def _use_on_private_stream(self, *tensors: torch.Tensor) -> None:
        """Hand buffers created on the current stream to the private stream.

        The private stream first waits for their initialisation, and the
        allocator is told they are in use there, so their blocks are not
        recycled under in-flight chunks. They stay in the current stream's
        pool, so releasing them gives the memory back to the model's pool.
        """
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        for t in tensors:
            t.record_stream(self._stream)

    def _init_spin(self) -> None:
        self._sink = torch.zeros(32, dtype=torch.float32, device=self.device)
        self._use_on_private_stream(self._sink)
        self._ns_per_iter = self._calibrate_spin()
        if not (self._ns_per_iter > 0.0):
            raise RuntimeError(f"spin calibration returned {self._ns_per_iter} ns/iter")
        self._recalibrate_when_hot = True
        self.mode = "spin"

    def _calibrate_spin(self) -> float:
        """Measure ns per spin iteration on the private stream."""
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._stream):
            _spin_kernel[(self._grid,)](10_000, self._sink, num_warps=1)
            beg.record(self._stream)
            _spin_kernel[(self._grid,)](_SPIN_CALIBRATION_ITERS, self._sink, num_warps=1)
            end.record(self._stream)
        end.synchronize()
        return beg.elapsed_time(end) * 1e6 / _SPIN_CALIBRATION_ITERS

    def _init_mm(self) -> None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw = dict(dtype=dtype, device=self.device)
        # torch.ones, not randn: do not consume the default CUDA generator.
        self._a = torch.ones(_MM_DIM, _MM_DIM, **kw)
        self._b = torch.ones(_MM_DIM, _MM_DIM, **kw)
        self._c = torch.empty(_MM_DIM, _MM_DIM, **kw)
        self._use_on_private_stream(self._a, self._b, self._c)
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._stream):
            for _ in range(5):
                torch.mm(self._a, self._b, out=self._c)
            beg.record(self._stream)
            for _ in range(20):
                torch.mm(self._a, self._b, out=self._c)
            end.record(self._stream)
        end.synchronize()
        self._mm_ms = beg.elapsed_time(end) / 20.0
        self.mode = "mm"

    def _release(self) -> None:
        """Free everything device-side; a later closed gate re-initialises lazily."""
        if self._selftest is not None:
            try:
                self._selftest.kill()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            self._selftest = None
        self._inflight.clear()
        for name in ("_sink", "_a", "_b", "_c"):
            if hasattr(self, name):
                setattr(self, name, None)
        self._stream = None
        self._initialized = False
        self.mode = None

    # ---------------------------------------------------------------- launch
    def _launch_chunk(self) -> None:
        """Queue one chunk of ~period_s GPU work on the private stream."""
        done = torch.cuda.Event()  # created first, so a failure here submits nothing
        with torch.cuda.stream(self._stream):
            if self.mode == "spin":
                iters = int(self.period_s * 1e9 / self._ns_per_iter)
                iters = max(_SPIN_MIN_ITERS, min(_SPIN_MAX_ITERS, iters))
                _spin_kernel[(self._grid,)](iters, self._sink, num_warps=1)
            else:
                n = max(1, int(self.period_s * _MM_DUTY * 1e3 / self._mm_ms))
                for _ in range(n):
                    torch.mm(self._a, self._b, out=self._c)
            done.record(self._stream)
        self._inflight.append(done)
        self._launches += 1

    def _reap_inflight(self) -> int:
        """Drop completed chunks from the front; return how many remain."""
        while self._inflight and self._inflight[0].query():
            self._inflight.popleft()
        return len(self._inflight)

    def _disable(self, reason: str) -> None:
        if self._disabled:
            return
        self._disabled = True
        self._active = False
        _log().warning(f"[gpu-keepalive] disabled: {reason}")
        try:
            if self._stream is not None:
                self._stream.synchronize()  # anything already submitted must finish
        except Exception:  # noqa: BLE001 - best-effort
            pass
        self._release()

    # ---------------------------------------------------------------- public
    @property
    def launches(self) -> int:
        return self._launches

    def tick(self) -> bool:
        """Queue a chunk while the gate is closed; True if one was launched."""
        if self._disabled:
            return False
        try:
            if not self._initialized and not self._initialize():
                return False
            if self._reap_inflight() >= self.queue_depth:
                return False
            now = time.monotonic()
            if self.mode == "mm" and now - self._last_launch < self.period_s:
                return False  # one ~5 ms GEMM burst per period keeps the fallback at its duty
            if not self._active:
                self._active = True
                _log().info("[gpu-keepalive] fill gate closed: keeping the GPU busy")
            self._launch_chunk()
            self._last_launch = now
            if self.mode == "spin" and self._recalibrate_when_hot and self._launches >= 2:
                # Clocks have boosted by now; re-measure once so chunks match period_s.
                self._recalibrate_when_hot = False
                self._ns_per_iter = self._calibrate_spin()
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort: never fail the executor loop
            self._disable(f"error: {exc!r}")
            return False

    def drain(self) -> None:
        """Wait for queued chunks (at most queue_depth * period_s) and free the device side."""
        self._finish(f"fill gate opened after {self._launches} chunks")

    def close(self) -> None:
        """Executor shutdown: like :meth:`drain`, also while the gate is still closed."""
        self._finish(f"closed with the fill gate still closed after {self._launches} chunks")

    def _finish(self, message: str) -> None:
        if self._inflight:
            try:
                self._inflight[-1].synchronize()
            except Exception as exc:  # noqa: BLE001 - best-effort: never fail the executor loop
                self._disable(f"error: {exc!r}")
        if self._active:
            self._active = False
            _log().info(f"[gpu-keepalive] {message}")
        self._release()  # also stops a still-running self-test child

    @classmethod
    def create_from_env(cls, device) -> GpuKeepalive | None:
        """Build a keepalive if ``TRTLLM_GPU_KEEPALIVE=1``, else None. Never raises.

        Construction touches no CUDA state; the device side is set up at the
        first :meth:`tick`, where any failure disables the keepalive.
        """
        if os.environ.get(KEEPALIVE_ENV_VAR_NAME, "0") != "1":
            return None
        try:
            return cls(device)
        except Exception as exc:  # noqa: BLE001 - opt-in best-effort feature must not fail executor init
            _log().warning(f"[gpu-keepalive] not active: {exc!r}")
            return None


if __name__ == "__main__":  # pragma: no cover - subprocess self-test entry
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(_run_selftest(int(sys.argv[2])))
    sys.exit(2)
