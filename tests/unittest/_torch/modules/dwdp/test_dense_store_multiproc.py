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
"""Multi-process GPU tests of the dense DWDP FABRIC row store.

Every DWDP rank is its own process with one GPU visible, like an independently launched
``trtllm-serve`` context worker.  The ranks meet on a ``torch.distributed.TCPStore`` that rank 0
hosts on 127.0.0.1, through ``StoreDwdpGroup``; each rank sets a ``test_done/<rank>`` key when it
is finished with the store and rank 0 exits only after it has seen every rank's key (last man
out).  The test re-runs this file as the worker (``--rank``, ``--size``, ``--port``, ``--out``);
each worker writes a JSON report that the parent turns into a pass, a failure listing every
finding, or a skip.

Every rank builds the same seeded toy decoder layers on its GPU and collects them through
``layout.collect_dense_layout``: blobs of up to about 6 MB, a replicated layer, a gap in the layer
indices, segments with nothing to move (``B = 0``), a blob of 777 bytes, blob sizes that are not
multiples of 16, and a trailing layer with nothing to move at all, which the chunked layout puts
in a chunk without rows (its handle is one granule).  Each rank then runs, in the same collective
order:

1. a FABRIC probe (create and export one handle), gathered so that all ranks skip together when
   a rank cannot use FABRIC handles, and the layout digests, which must agree;
2. a cycle over the single-chunk layout: ``FabricShardStore`` allocate, ``write_local_rows``,
   synchronize, export, allgather, ``import_peers``; local chunks must be mapped read-write and
   every peer chunk read-only; the plans run through ``DenseCopyLauncher`` on a copy stream in
   each transfer mode (gather, batched, per-slice) and every slot must reassemble every blob
   byte-exact against ``write_blob``; then a barrier, ``release`` twice (the second is a no-op),
   and a barrier;
3. the same cycle over the layout with ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB`` set (several handles
   per rank, the last one without rows), after a release of everything: allocating takes at
   least the store's bytes of device memory, and after every rank released, free device memory
   is back at its pre-allocation level within a small tolerance (rank ``p``'s chunks are freed
   only once every peer released its imports of them, so a leaked import shows on its exporter);
4. a cycle whose ``import_peers`` fails at its second read-only mapping, and one whose
   ``allocate`` fails at its second chunk: ``release`` frees what the failed step created, and
   the memory comes back as in 3.

The test runs once per gather kernel: the register kernel (``sm``) and the bulk-copy kernel that
``TRTLLM_DWDP_GATHER_KERNEL=tma`` selects, which production uses and which loads the peers' rows
with ``cp.async.bulk`` through their read-only mappings.  The kernel is chosen once per process,
so each worker gets the variable from the parent; it runs its gather passes under
``torch.profiler`` and reports how many launches of each kernel reached the GPU, and every launch
must be of the selected kernel.  Without ``trtllm::dwdp_fabric_gather`` the workers run the other
two modes and the test skips after they passed.  Where the profiler records no device activity
(an image whose CUPTI cannot profile the GPU), the kernel is unverified, and the test skips after
every other check passed.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import functools
import json
import math
import os
import random
import signal
import site
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.dwdp import fabric_gather, transport, vmm
from tensorrt_llm._torch.modules.dwdp.dense import store as dense_store
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher, build_copy_plans
from tensorrt_llm._torch.modules.dwdp.dense.store import FabricShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    CopyMode,
    DenseLayout,
    SlotBuffer,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.rendezvous import StoreDwdpGroup

_MASTER_ADDR = "127.0.0.1"
_GROUP_TIMEOUT_S = 120  # every collective, and the store connect
_GROUP_DEADLINE_S = 420.0  # importing TensorRT-LLM plus every cycle, per scenario
_MEMORY_TOLERANCE = 1 << 20  # less than one 2 MiB granule, the smallest handle
_MEMORY_POLLS = 20
_MEMORY_POLL_S = 0.25

FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

BF16 = torch.bfloat16
F16 = torch.float16
F32 = torch.float32
FP8 = torch.float8_e4m3fn
I32 = torch.int32
U8 = torch.uint8

#: Storages below this stay replicated in the toy layout.
TOY_MIN_BYTES = 64
#: ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB`` of the chunked layout: one or two layers per handle.
CHUNK_MB = 2

#: Decoder layer -> (attention tensors, MoE-dense tensors) as ``(dtype, shape)``; None: the layer
#: stays replicated.  Blob sizes: layer 1 A 6303801 (two anchors), D 2048000; layer 2 A 777, D 0
#: (a 5-byte tensor stays); layer 3 A 6205438, D 1332000 (a 21-byte tensor stays); layer 4 A
#: 6144000, D 0 (no tensor); layer 6 A 4004000, D 5734400 (a 3-byte tensor stays); layer 7 A 0,
#: D 0 (a 5-byte and a 3-byte tensor stay).  Layer 6 alone exceeds ``CHUNK_MB``, so layer 7 gets
#: a chunk of its own, without rows.
LAYERS = {
    0: None,
    1: ([(BF16, (1536, 2048)), (U8, (12345,))], [(F32, (512, 1000))]),
    2: ([(U8, (777,))], [(U8, (5,))]),
    3: ([(FP8, (2048, 3000)), (F16, (30719,))], [(I32, (1000, 333)), (FP8, (3, 7))]),
    4: ([(BF16, (2048, 1500))], []),
    6: ([(F32, (1000, 1001))], [(BF16, (4096, 700)), (U8, (3,))]),
    7: ([(U8, (5,))], [(U8, (3,))]),
}

POISON = 0x5A  # Slot bytes before a copy.

#: Selects the gather kernel; read once per process by the op.
_GATHER_KERNEL_ENV = "TRTLLM_DWDP_GATHER_KERNEL"
#: A library defining ``trtllm::dwdp_fabric_gather``; the workers get this process's value, so
#: they resolve the op as this process does.
_GATHER_LIB_ENV = "TRTLLM_DWDP_GATHER_LIB"
#: The gather kernels, ``sm`` (the variable unset) and ``tma`` (production), and the name of each
#: one's device function in a profile.
GATHER_KERNELS = {"sm": "dwdpFabricGatherKernel", "tma": "dwdpFabricGatherTmaKernel"}

# What a worker must not inherit: launcher identities (importing TensorRT-LLM initializes MPI, and
# each worker is a standalone MPI singleton, like a plain trtllm-serve process) and the DWDP
# knobs, which the test sets for the workers itself.
_DROPPED_ENV_PREFIXES = (
    "SLURM_",
    "PMIX_",
    "PMI_",
    "OMPI_",
    "PRTE_",
    "I_MPI_",
    "HYDRA_",
    "MPI_",
    "TRTLLM_DWDP_",
)
_DROPPED_ENV_VARS = ("TLLM_DISABLE_MPI", "TLLM_LOG_LEVEL_BY_MODULE")


# ---------------------------------------------------------------------------------------------
# Toy layers (identical on every rank)
# ---------------------------------------------------------------------------------------------


class ToyRoot(nn.Module):
    """A dense root holding one Parameter per ``(dtype, shape)``, filled with random bytes."""

    def __init__(self, tensors, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        for index, (dtype, shape) in enumerate(tensors):
            nbytes = math.prod(shape) * dtype.itemsize
            data = torch.randint(0, 256, (nbytes,), generator=generator, dtype=torch.uint8)
            weight = data.view(dtype).reshape(shape).to(device)
            self.register_parameter(f"w{index}", nn.Parameter(weight, requires_grad=False))


class ToyLayer(nn.Module):
    """A decoder layer with the ``dense_dwdp_roots`` seam; ``spec`` None keeps it replicated."""

    def __init__(self, spec, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        self.covered = spec is not None
        attn, moe = spec if spec is not None else ([], [])
        self.attn = ToyRoot(attn, generator, device)
        self.moe = ToyRoot(moe, generator, device)

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        return [(segment, getattr(self, segment))] if self.covered else []


def toy_layers(device: torch.device) -> list[tuple[int, ToyLayer]]:
    generator = torch.Generator().manual_seed(4321)
    return [(index, ToyLayer(LAYERS[index], generator, device)) for index in sorted(LAYERS)]


def collect(layers, num_ranks: int, device: torch.device, *, chunk_mb: int) -> DenseLayout:
    config = DenseConfig(enabled=True, min_bytes=TOY_MIN_BYTES, store_chunk_mb=chunk_mb)
    return collect_dense_layout(
        layers,
        config,
        num_ranks=num_ranks,
        device=device,
        attr_allow=frozenset(),
        exclude=frozenset(),
    )


# ---------------------------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------------------------


def _visible_gpus() -> list[str]:
    """The ``CUDA_VISIBLE_DEVICES`` entries of this process's GPUs, in device order."""
    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in range(count)]
    return [entry.strip() for entry in visible.split(",") if entry.strip()][:count]


def _require_gpus(num_ranks: int) -> list[str]:
    """One GPU per rank and FABRIC peer handles, else skip; returns the ranks' GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    gpus = _visible_gpus()
    if len(gpus) < num_ranks:
        pytest.skip(f"needs {num_ranks} GPUs, found {len(gpus)}")
    if vmm.peer_handle_type() != FABRIC:
        pytest.skip("the dense DWDP store needs FABRIC peer handles (aarch64 / NVL72)")
    return gpus[:num_ranks]


def _package_root() -> str | None:
    """The PYTHONPATH entry that makes workers import this process's ``tensorrt_llm``."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(tensorrt_llm.__file__)))
    site_dirs = {os.path.realpath(p) for p in [*site.getsitepackages(), site.getusersitepackages()]}
    return None if root in site_dirs else root


def _worker_env(gpu: str, gather_kernel: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_DROPPED_ENV_PREFIXES) and key not in _DROPPED_ENV_VARS
    }
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"})
    if gather_kernel != "sm":
        env[_GATHER_KERNEL_ENV] = gather_kernel
    if os.environ.get(_GATHER_LIB_ENV):
        env[_GATHER_LIB_ENV] = os.environ[_GATHER_LIB_ENV]
    root = _package_root()
    if root:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    return env


def _port_is_free(port: int) -> bool:
    """Whether a listener could bind ``port`` on every IPv4 and IPv6 address."""
    for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        with sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind((host, port))
            except OSError:
                return False
    return True


def _pick_port() -> int:
    """A free store port below the ephemeral range, away from the torch and DWDP defaults."""
    low = 32768
    with contextlib.suppress(OSError, ValueError):
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low = int(f.read().split()[0])
    candidates = [p for p in range(20000, min(low, 32768)) if not 29400 <= p < 29700]
    random.SystemRandom().shuffle(candidates)
    for port in candidates[:256]:
        if _port_is_free(port):
            return port
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@dataclass
class _Outcome:
    """What the workers of one group did, as the parent saw it and as they reported."""

    workdir: Path
    reports: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    returncodes: dict[int, int | None] = field(default_factory=dict)
    hung: list[int] = field(default_factory=list)

    def log(self, rank: int) -> str:
        path = self.workdir / f"rank{rank}.log"
        return path.read_text(errors="replace") if path.exists() else ""

    def describe(self, ranks: list[int] | None = None) -> str:
        lines = [f"workdir {self.workdir}"]
        for rank, report in self.reports.items():
            lines.append(
                f"  rank {rank}: rc={self.returncodes.get(rank)} "
                f"report={'missing' if report is None else 'written'}"
            )
        for rank in (ranks if ranks is not None else list(self.reports))[:4]:
            lines.append(f"--- rank {rank} log tail ---")
            lines.append(self.log(rank)[-3000:])
        return "\n".join(lines)


def _run_group(workdir: Path, gpus: list[str], gather_kernel: str) -> _Outcome:
    """Run one worker per GPU and wait until all exit or the deadline passes."""
    num_ranks = len(gpus)
    port = _pick_port()
    outcome = _Outcome(workdir)
    procs: dict[int, subprocess.Popen] = {}
    try:
        for rank, gpu in enumerate(gpus):
            with open(workdir / f"rank{rank}.log", "wb") as log:
                procs[rank] = subprocess.Popen(
                    [sys.executable, "-u", "-X", "faulthandler", os.path.abspath(__file__)]
                    + ["--rank", str(rank), "--size", str(num_ranks), "--port", str(port)]
                    + ["--parent-pid", str(os.getpid())]
                    + ["--out", str(workdir / f"rank{rank}.json")],
                    cwd=workdir,
                    env=_worker_env(gpu, gather_kernel),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        deadline = time.monotonic() + _GROUP_DEADLINE_S
        while any(proc.poll() is None for proc in procs.values()):
            if time.monotonic() > deadline:
                outcome.hung = [rank for rank, proc in procs.items() if proc.poll() is None]
                for rank in outcome.hung:  # dump the Python stacks into the log, then kill
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(procs[rank].pid, signal.SIGUSR1)
                time.sleep(1.0)
                break
            time.sleep(0.2)
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
        for proc in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
    for rank in range(num_ranks):
        outcome.returncodes[rank] = procs[rank].returncode if rank in procs else None
        path = workdir / f"rank{rank}.json"
        outcome.reports[rank] = json.loads(path.read_text()) if path.exists() else None
    return outcome


@pytest.mark.parametrize("gather_kernel", list(GATHER_KERNELS))
@pytest.mark.parametrize("num_ranks", [2, 4], ids=["2ranks", "4ranks"])
def test_fabric_store_across_processes(num_ranks, gather_kernel, tmp_path, capfd):
    """Real FABRIC stores on one GPU per process: rows reassemble byte-exact, teardown leaks nothing.

    See the module docstring for the cycles each rank runs; ``gather_kernel`` is the kernel of
    the gather mode.
    """
    gpus = _require_gpus(num_ranks)
    out = _run_group(tmp_path, gpus, gather_kernel)
    assert not out.hung, f"ranks {out.hung} hung and were killed\n{out.describe(out.hung)}"
    missing = [rank for rank, report in out.reports.items() if report is None]
    assert not missing, f"ranks {missing} wrote no report\n{out.describe(missing)}"

    failures = [
        f"rank {rank}: {failure}"
        for rank, report in out.reports.items()
        for failure in report["failures"]
    ]
    if failures:
        shown = "\n".join(f"- {failure}" for failure in failures[:40])
        pytest.fail(f"{len(failures)} failure(s)\n{shown}\n{out.describe()}")
    skips = {rank: report["skip"] for rank, report in out.reports.items() if report["skip"]}
    if skips:
        assert len(skips) == num_ranks, f"only ranks {sorted(skips)} skipped: {skips}"
        pytest.skip(f"rank 0: {skips[0]}")
    bad_exits = {rank: code for rank, code in out.returncodes.items() if code != 0}
    assert not bad_exits, f"workers exited with {bad_exits}\n{out.describe(list(bad_exits))}"

    lines = []
    for rank, report in out.reports.items():
        info = report["info"]
        assert info["compared"] == info["expected_compared"] > 0, info
        # One access check per rank and chunk, in the single-chunk and the chunked cycle.
        assert info["access_checked"] == num_ranks * (1 + info["chunks"]), info
        assert set(info["leaked_bytes"]) == {"chunked", "partial import", "partial allocate"}
        notes = f", notes {report['notes']}" if report["notes"] else ""
        lines.append(
            f"rank {rank}: store {info['store_bytes']} B in 1 chunk and "
            f"{info['chunked_store_bytes']} B in {info['chunks']}, allocating the latter took "
            f"{info['allocated_bytes']} B, leaked {info['leaked_bytes']}, "
            f"{info['compared']} slot comparisons, {info['gather_launches']} gather launches "
            f"ran as {info['gather_kernels']}{notes}"
        )
    with capfd.disabled():
        print(f"\n[dense DWDP FABRIC store] {num_ranks} ranks: " + "; ".join(lines))

    ran = [rank for rank, report in out.reports.items() if report["info"]["gather_kernels"]]
    if not ran:
        pytest.skip(
            "trtllm::dwdp_fabric_gather is not registered by this build: the batched and "
            "per-slice modes passed, the gather mode did not run"
        )
    assert len(ran) == num_ranks, f"only ranks {ran} ran the gather mode"
    unprofiled = {
        rank: report["info"]["unprofiled_passes"]
        for rank, report in out.reports.items()
        if report["info"]["unprofiled_passes"]
    }
    if unprofiled:
        pytest.skip(
            f"torch.profiler recorded no device activity in the gather passes {unprofiled} "
            f"(CUPTI cannot profile this GPU), so the {gather_kernel} kernel is unverified; the "
            "stores, the reassembly in every transfer mode and the teardown passed"
        )
    for rank, report in out.reports.items():
        info = report["info"]
        launches = info["gather_launches"]
        want = {kind: launches if kind == gather_kernel else 0 for kind in GATHER_KERNELS}
        assert launches > 0 and info["gather_kernels"] == want, (
            f"rank {rank}: of {launches} gather launches, the profile shows "
            f"{info['gather_kernels']}; all of them must be {gather_kernel}"
        )


# ---------------------------------------------------------------------------------------------
# Worker side (one DWDP rank)
# ---------------------------------------------------------------------------------------------


class _Skip(Exception):
    """Raised in a worker when this machine cannot run the test."""


class _Report:
    """Failures, notes and facts a worker hands back to the parent."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.info: dict[str, Any] = {
            "compared": 0,
            "expected_compared": 0,
            "access_checked": 0,
            "leaked_bytes": {},
            "gather_launches": 0,  # Layers with something to copy, over every gather pass.
            "gather_kernels": None,  # Once the gather mode ran: profiled launches per kernel.
            "unprofiled_passes": [],  # Gather passes the profiler recorded no device activity of.
        }
        self.skip: str | None = None

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return ok

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Record an exception of the step as a failure, so the collectives stay in step."""
        try:
            yield
        except Exception:  # every rank must reach the next collective; the error is reported
            self.failures.append(f"{name} raised:\n{traceback.format_exc()}")

    def to_json(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "notes": self.notes,
            "info": self.info,
            "skip": self.skip,
        }


def _die_with_parent(parent_pid: int) -> None:
    """SIGKILL this worker when the test process dies (PR_SET_PDEATHSIG)."""
    with contextlib.suppress(AttributeError, OSError):
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() != parent_pid:
        os._exit(3)


def _fabric_problem(device_id: int) -> str | None:
    """None if this process can create and export a FABRIC handle, else what failed."""
    if vmm.peer_handle_type() != FABRIC:
        return f"peer handles are {vmm.peer_handle_type().name}, not FABRIC"
    granularity = vmm.get_allocation_granularity(device_id)
    try:
        handle = vmm.create_fabric_handle(granularity, device_id)
    except RuntimeError as exc:
        return f"cannot create a FABRIC handle: {exc}"
    try:
        transport._export_handle(handle, FABRIC)
    except RuntimeError as exc:
        return f"cannot export a FABRIC handle: {exc}"
    finally:
        vmm.release_handle(handle)
    return None


def _free_memory(device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)[0]


def _check_memory(report: _Report, device: torch.device, baseline: int, label: str) -> None:
    """Free device memory is back at ``baseline`` (polled: peers may release a moment later)."""
    leaked = baseline - _free_memory(device)
    for _ in range(_MEMORY_POLLS):
        if leaked <= _MEMORY_TOLERANCE:
            break
        time.sleep(_MEMORY_POLL_S)
        leaked = baseline - _free_memory(device)
    report.info["leaked_bytes"][label] = leaked
    report.check(
        leaked <= _MEMORY_TOLERANCE,
        f"{label}: {leaked} bytes of device memory did not come back after every rank released "
        f"(tolerance {_MEMORY_TOLERANCE})",
    )


def _check_access(
    report: _Report, store: FabricShardStore, rank: int, device_id: int, label: str
) -> None:
    """Local chunks are mapped read-write, every peer chunk read-only."""
    cuda = vmm.cuda
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id
    read = int(cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ)
    read_write = int(cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    for peer, bases in enumerate(store.chunk_bases()):
        for chunk, base in enumerate(bases):
            flags = int(vmm.check_cu_result(cuda.cuMemGetAccess(location, base)))
            want = read_write if peer == rank else read
            report.check(
                flags == want,
                f"{label}: rank {peer}'s chunk {chunk} is mapped with access flags {flags}, "
                f"expected {want}",
            )
            report.info["access_checked"] += 1


def _blob(segment, num_ranks: int, device: torch.device) -> torch.Tensor:
    out = torch.empty(num_ranks * segment.row_bytes, dtype=torch.uint8, device=device)
    write_blob(segment, out, num_ranks=num_ranks)
    return out


def _holding_slot(layout: DenseLayout, rings, segment) -> SlotBuffer:
    return rings[segment.segment][layout.slot_index(segment.segment, segment.layer)]


def _launch_and_compare(
    report: _Report,
    layout: DenseLayout,
    rings,
    expected,
    launcher: DenseCopyLauncher,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    label: str,
) -> None:
    """Copy every layer on the copy stream; each slot must hold its blob and poison elsewhere."""
    compute_stream, copy_stream = streams
    for layer in layout.layers:
        segments = layout.layer_segments(layer)
        for segment in segments:
            _holding_slot(layout, rings, segment).storage.fill_(POISON)
        copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(copy_stream):
            launcher(layer)
        compute_stream.wait_stream(copy_stream)
        torch.cuda.synchronize(copy_stream.device)
        for segment in segments:
            slot = _holding_slot(layout, rings, segment)
            nbytes = layout.num_ranks * segment.row_bytes
            name = f"{label}/layer {layer}/{segment.segment.value}"
            got = slot.blob_view(nbytes)
            if not torch.equal(got, expected[segment.key]):
                wrong = torch.nonzero(got != expected[segment.key]).flatten()
                row, offset = divmod(int(wrong[0]), segment.row_bytes)
                report.failures.append(
                    f"{name}: {wrong.numel()} wrong bytes, the first in row {row} at byte {offset}"
                )
            outside = (slot.storage[: slot.pad], slot.storage[slot.pad + nbytes :])
            report.check(
                all(bool((part == POISON).all()) for part in outside),
                f"{name}: slot bytes outside the blob were written",
            )
            report.info["compared"] += 1


def _profile_gather(report: _Report, run: Callable[[], None], launches: int, label: str) -> None:
    """Run a gather pass under ``torch.profiler``; count each gather kernel's launches.

    Args:
        report: Receives the counts (``gather_kernels``) and the launches the pass makes, or the
            pass in ``unprofiled_passes`` when the profiler recorded no device activity.
        run: The pass.
        launches: Gather launches the pass makes: one per layer with something to copy.
        label: Names the pass.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    names = [evt.name for evt in prof.events() if evt.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        # No device event at all, not even the comparisons' kernels: CUPTI cannot profile this
        # GPU. The pass ran and was compared; the parent skips only the kernel check.
        report.info["unprofiled_passes"].append(label)
    counts = report.info["gather_kernels"] or dict.fromkeys(GATHER_KERNELS, 0)
    for kind, kernel in GATHER_KERNELS.items():
        counts[kind] += sum(kernel in name for name in names)
    report.info["gather_kernels"] = counts
    report.info["gather_launches"] += launches


def _check_reassembly(
    report: _Report, layout: DenseLayout, store: FabricShardStore, rank: int, device, label: str
) -> None:
    """Every transfer mode rebuilds every blob in its slot from the peers' and the local rows."""
    rings = {kind: allocate_slot_ring(spec, device) for kind, spec in layout.slot_specs.items()}
    plans = build_copy_plans(
        layout,
        rank=rank,
        chunk_bases=store.chunk_bases(),
        slot_bases={kind: [slot.base for slot in ring] for kind, ring in rings.items()},
    )
    expected = {
        key: _blob(segment, layout.num_ranks, device) for key, segment in layout.segments.items()
    }
    streams = (torch.cuda.current_stream(device), torch.cuda.Stream(device))
    modes = [mode for mode in CopyMode if mode is not CopyMode.GATHER or fabric_gather.gather_op()]
    if len(modes) < len(CopyMode):
        report.notes.append("trtllm::dwdp_fabric_gather is not registered; gather mode not run")
    for mode in modes:
        # Launch arguments are built on the stream that launches them, as in setup P3.
        with torch.cuda.stream(streams[1]):
            launcher = DenseCopyLauncher(plans, mode, device=device)
        name = f"{label}/{mode.value}"
        run = functools.partial(
            _launch_and_compare, report, layout, rings, expected, launcher, streams, name
        )
        try:
            if mode is CopyMode.GATHER:
                launches = sum(not plans[layer].is_empty for layer in layout.layers)
                _profile_gather(report, run, launches, name)
            else:
                run()
        finally:
            launcher.release()
    report.info["expected_compared"] += len(modes) * len(layout.segments)


@contextlib.contextmanager
def _failing(owner: object, name: str, call_number: int) -> Iterator[None]:
    """Make the ``call_number``-th call of ``owner.name`` raise instead of running."""
    original = getattr(owner, name)
    calls = 0

    def wrapper(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == call_number:
            raise RuntimeError(f"injected failure of {name} call {call_number}")
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _check_released(report: _Report, store: FabricShardStore, label: str) -> None:
    """``release`` twice; afterwards the store refuses to hand out addresses."""
    store.release()
    store.release()
    try:
        store.chunk_bases()
    except RuntimeError:
        pass
    else:
        report.failures.append(f"{label}: chunk_bases still answered after release")


def _cycle(
    report: _Report,
    group: StoreDwdpGroup,
    layout: DenseLayout,
    rank: int,
    device: torch.device,
    *,
    label: str,
    measure: bool,
    fail_import: bool = False,
) -> None:
    """One store lifetime; three collectives: the payload allgather and two barriers."""
    baseline = _free_memory(device) if measure else None
    store = FabricShardStore(layout, rank, device)
    payload = None
    with report.step(f"{label}: allocate, write the rows and export"):
        store.allocate()
        if baseline is not None:
            taken = baseline - _free_memory(device)
            report.info["allocated_bytes"] = taken
            report.check(
                taken >= layout.store_bytes - _MEMORY_TOLERANCE,
                f"{label}: allocate took {taken} bytes of device memory; the store has "
                f"{layout.store_bytes}",
            )
        store.write_local_rows()
        torch.cuda.synchronize(device)
        payload = store.export()
    payloads = group.allgather(payload)
    if fail_import:
        with report.step(f"{label}: import_peers with its second read-only mapping failing"):
            try:
                with _failing(dense_store, "_set_read_only_access", 2):
                    store.import_peers(payloads)
            except RuntimeError as exc:
                report.check("injected failure" in str(exc), f"{label}: import raised {exc!r}")
            else:
                report.failures.append(f"{label}: import_peers did not fail")
    else:
        with report.step(f"{label}: import the peers and gather"):
            store.import_peers(payloads)
            _check_access(report, store, rank, device.index, label)
            _check_reassembly(report, layout, store, rank, device, label)
    group.Barrier()  # No rank reads a peer's rows any more.
    with report.step(f"{label}: release"):
        _check_released(report, store, label)
    group.Barrier()  # Every rank released its imports of this rank's chunks.
    if baseline is not None:
        _check_memory(report, device, baseline, label)


def _partial_allocate(report: _Report, group, layout: DenseLayout, rank: int, device) -> None:
    """``allocate`` fails at its second chunk; ``release`` frees the chunks it made."""
    label = "partial allocate"
    baseline = _free_memory(device)
    store = FabricShardStore(layout, rank, device)
    with report.step(f"{label}: allocate with the second chunk's access failing"):
        try:
            with _failing(vmm, "set_access", 2):
                store.allocate()
        except RuntimeError as exc:
            report.check("injected failure" in str(exc), f"{label}: allocate raised {exc!r}")
        else:
            report.failures.append(f"{label}: allocate did not fail")
    with report.step(f"{label}: release"):
        _check_released(report, store, label)
    group.Barrier()
    _check_memory(report, device, baseline, label)


def _run_rank(report: _Report, group: StoreDwdpGroup, rank: int, size: int, device) -> None:
    problems = group.allgather(_fabric_problem(device.index))
    problems = [f"rank {peer}: {problem}" for peer, problem in enumerate(problems) if problem]
    if problems:
        raise _Skip("; ".join(problems))

    layers = toy_layers(device)
    single = collect(layers, size, device, chunk_mb=0)
    chunked = collect(layers, size, device, chunk_mb=CHUNK_MB)
    digests = group.allgather((single.digest, chunked.digest))
    report.check(len(set(digests)) == 1, f"the ranks' layouts differ: {digests}")
    report.check(
        len(single.chunks) == 1 and len(chunked.chunks) >= 2,
        f"expected 1 and at least 2 store chunks, got {len(single.chunks)} and "
        f"{len(chunked.chunks)}",
    )
    last = chunked.chunks[-1]
    report.check(
        last.layers == (max(LAYERS),) and last.nbytes == 0,
        f"expected the chunked layout to end with a chunk of layer {max(LAYERS)} without rows, "
        f"got layers {last.layers} with {last.nbytes} bytes of rows",
    )
    report.info.update(
        store_bytes=single.store_bytes,
        chunked_store_bytes=chunked.store_bytes,
        chunks=len(chunked.chunks),
    )

    _cycle(report, group, single, rank, device, label="single chunk", measure=False)
    _cycle(report, group, chunked, rank, device, label="chunked", measure=True)
    _cycle(
        report, group, chunked, rank, device, label="partial import", measure=True, fail_import=True
    )
    _partial_allocate(report, group, chunked, rank, device)


def _worker(report: _Report, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise _Skip("CUDA is not available in the worker")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    from torch.distributed import TCPStore

    tcp = TCPStore(
        host_name=_MASTER_ADDR,
        port=args.port,
        is_master=args.rank == 0,
        timeout=timedelta(seconds=_GROUP_TIMEOUT_S),
        wait_for_workers=False,
    )
    group = StoreDwdpGroup(
        global_rank=args.rank,
        dwdp_size=args.size,
        master_addr=_MASTER_ADDR,
        master_port=args.port,
        timeout_s=_GROUP_TIMEOUT_S,
        store=tcp,
    )
    try:
        _run_rank(report, group, args.rank, args.size, device)
    finally:
        # Last man out: rank 0 hosts the store, so it leaves only after every rank is done.
        try:
            tcp.set(f"test_done/{args.rank}", b"1")
            if args.rank == 0:
                keys = [f"test_done/{rank}" for rank in range(args.size)]
                tcp.wait(keys, timedelta(seconds=_GROUP_TIMEOUT_S))
        except RuntimeError as exc:  # the store's own errors; the test reports the rest
            report.notes.append(f"last man out failed: {exc!r}")


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="dense DWDP FABRIC store worker (one rank)")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _die_with_parent(args.parent_pid)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    report = _Report()
    try:
        _worker(report, args)
    except _Skip as exc:
        report.skip = str(exc)
    except Exception:  # the worker's top level: any error becomes a reported failure
        report.failures.append(f"the worker raised:\n{traceback.format_exc()}")
    finally:
        out = Path(args.out)
        partial = out.with_suffix(".partial")
        partial.write_text(json.dumps(report.to_json(), indent=2, default=str))
        os.replace(partial, out)
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
