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

"""Small, model-free infrastructure benchmark used only by CI dry runs.

The filename intentionally does not match pytest's normal ``test_*.py``
pattern.  Dry-run jobs pass this module explicitly after selecting its node ID
through the dedicated ``infra_dry_run`` test-db context.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import sys
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Optional

import torch

_DISTRIBUTED_TIMEOUT_SECONDS = 900
_MATRIX_SIZE = 32


@contextmanager
def _bounded_wait(seconds: int):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("distributed dry-run timeout requires the main thread")

    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("distributed dry-run timeout cannot replace an active ITIMER_REAL")

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"distributed dry-run operation exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


def _required_int(environ: Mapping[str, str], name: str) -> int:
    value = environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for an externally launched rank")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error


def _external_rank_context(
    environ: Mapping[str, str],
) -> Optional[tuple[int, int, int]]:
    rank_names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    present = [name in environ for name in rank_names]
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, is_present in zip(rank_names, present) if not is_present]
        raise RuntimeError(f"incomplete distributed rank environment; missing {missing}")

    rank = _required_int(environ, "RANK")
    local_rank = _required_int(environ, "LOCAL_RANK")
    world_size = _required_int(environ, "WORLD_SIZE")
    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise RuntimeError(f"RANK {rank} is outside WORLD_SIZE {world_size}")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    return rank, local_rank, world_size


def _run_cpu(torch_module=torch) -> None:
    torch_module.manual_seed(0)
    left = torch_module.full(
        (_MATRIX_SIZE, _MATRIX_SIZE), 0.25, dtype=torch_module.float32, device="cpu"
    )
    right = torch_module.full(
        (_MATRIX_SIZE, _MATRIX_SIZE), 0.5, dtype=torch_module.float32, device="cpu"
    )
    output = torch_module.matmul(left, right)
    expected = torch_module.full_like(output, _MATRIX_SIZE * 0.25 * 0.5)
    if output.device.type != "cpu" or output.dtype != torch_module.float32:
        raise RuntimeError("CPU benchmark did not produce a CPU FP32 tensor")
    if not torch_module.isfinite(output).all().item():
        raise RuntimeError("CPU benchmark produced non-finite values")
    if not torch_module.equal(output, expected):
        raise RuntimeError("CPU benchmark produced an unexpected deterministic result")


def _run_cuda_matmul(local_rank: int, torch_module=torch) -> float:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for this infrastructure dry-run stage")
    device_count = torch_module.cuda.device_count()
    if not 0 <= local_rank < device_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} is outside the {device_count} visible CUDA devices"
        )

    torch_module.cuda.set_device(local_rank)
    device = torch_module.device("cuda", local_rank)
    torch_module.manual_seed(1000 + local_rank)
    torch_module.cuda.manual_seed_all(1000 + local_rank)
    left = torch_module.full(
        (_MATRIX_SIZE, _MATRIX_SIZE), 0.25, dtype=torch_module.float16, device=device
    )
    right = torch_module.full(
        (_MATRIX_SIZE, _MATRIX_SIZE), 0.5, dtype=torch_module.float16, device=device
    )
    output = torch_module.matmul(left, right)
    expected = torch_module.full_like(output, _MATRIX_SIZE * 0.25 * 0.5)
    if output.device.type != "cuda" or output.dtype != torch_module.float16:
        raise RuntimeError("GPU benchmark did not produce a CUDA FP16 tensor")
    if not torch_module.isfinite(output).all().item():
        raise RuntimeError("GPU benchmark produced non-finite values")
    if not torch_module.equal(output, expected):
        raise RuntimeError("GPU benchmark produced an unexpected deterministic result")
    torch_module.cuda.synchronize(device)
    return float(output.float().sum().item())


def _validate_rank_summaries(summaries: list[list[float]], world_size: int) -> None:
    expected_ranks = list(range(world_size))
    observed_ranks = sorted(int(summary[0]) for summary in summaries)
    if observed_ranks != expected_ranks:
        raise RuntimeError(
            f"observed ranks {observed_ranks} do not match expected {expected_ranks}"
        )
    if any(int(summary[1]) != world_size for summary in summaries):
        raise RuntimeError("rank summaries contain inconsistent world sizes")
    checksums = [summary[2] for summary in summaries]
    if any(abs(checksum - checksums[0]) > 1e-3 for checksum in checksums[1:]):
        raise RuntimeError("rank summaries contain inconsistent CUDA checksums")


def _run_distributed_rank(
    rank: int,
    local_rank: int,
    world_size: int,
    timeout_seconds: int = _DISTRIBUTED_TIMEOUT_SECONDS,
    torch_module=torch,
) -> list[float]:
    distributed = torch_module.distributed
    if not distributed.is_available() or not distributed.is_nccl_available():
        raise RuntimeError("NCCL distributed support is required for multi-GPU dry runs")

    try:
        if not distributed.is_initialized():
            distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )

        checksum = _run_cuda_matmul(local_rank, torch_module)
        device = torch_module.device("cuda", local_rank)
        local_summary = torch_module.tensor(
            [float(rank), float(world_size), checksum],
            dtype=torch_module.float64,
            device=device,
        )
        reduced_checksum = torch_module.tensor(checksum, dtype=torch_module.float64, device=device)
        distributed.all_reduce(reduced_checksum)
        expected_total = checksum * world_size
        if abs(float(reduced_checksum.item()) - expected_total) > 1e-3:
            raise RuntimeError("NCCL all-reduce produced an unexpected checksum")

        gathered = [torch_module.empty_like(local_summary) for _ in range(world_size)]
        distributed.all_gather(gathered, local_summary)
        summaries = [summary.cpu().tolist() for summary in gathered]
        _validate_rank_summaries(summaries, world_size)
        return [float(rank), float(world_size), checksum]
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _llmapi_rank_task(timeout_seconds: int) -> list[float]:
    """Run on every rank already owned by ``trtllm-llmapi-launch``."""
    from mpi4py import MPI

    rank_context = _external_rank_context(os.environ)
    if rank_context is None:
        raise RuntimeError("LLMAPI worker is missing its distributed rank environment")
    env_rank, local_rank, env_world_size = rank_context
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if (env_rank, env_world_size) != (rank, world_size):
        raise RuntimeError(
            "LLMAPI worker rank environment does not match its MPI communicator: "
            f"env=({env_rank}, {env_world_size}), mpi=({rank}, {world_size})"
        )
    return _run_distributed_rank(rank, local_rank, world_size, timeout_seconds)


def _worker_import_module():
    """Load this file under the top-level name visible from the worker cwd.

    Pytest may collect this file as ``defs.infra_dry_run_benchmark``, while the
    MGMN workers run from this file's directory and can import it only as
    ``infra_dry_run_benchmark``.  Both RemoteMpiCommSession and multiprocessing
    serialize callables, so worker functions must come from that deterministic
    top-level module and its directory must be inherited by spawn children.
    """
    module_path = Path(__file__).resolve()
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = Path(__file__).stem
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, __file__)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load worker task module from {__file__}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != module_path:
            raise RuntimeError(f"{module_name} resolves to {module_file}, expected {__file__}")
    return module


def _pickleable_llmapi_rank_task():
    return _worker_import_module()._llmapi_rank_task


def _run_with_existing_llmapi_launcher(
    world_size: int,
    timeout_seconds: int = _DISTRIBUTED_TIMEOUT_SECONDS,
    session_factory=None,
) -> None:
    if session_factory is None:
        from tensorrt_llm.executor.utils import create_mpi_comm_session

        session_factory = create_mpi_comm_session

    session = session_factory(world_size)
    try:
        with _bounded_wait(timeout_seconds + 60):
            summaries = session.submit_sync(_pickleable_llmapi_rank_task(), timeout_seconds)
        if isinstance(summaries, BaseException):
            raise RuntimeError("LLMAPI rank task failed") from summaries
        if not isinstance(summaries, list) or len(summaries) != world_size:
            raise RuntimeError(
                "LLMAPI launcher returned an incomplete rank result set: "
                f"expected {world_size}, got {summaries!r}"
            )
        _validate_rank_summaries(summaries, world_size)
    finally:
        # RemoteMpiCommSessionClient.shutdown() is intentionally a no-op.  The
        # outer launcher owns and stops the MGMN server after pytest exits.
        session.shutdown()


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_rank_worker(
    local_rank: int, world_size: int, master_port: int, timeout_seconds: int
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    _run_distributed_rank(local_rank, local_rank, world_size, timeout_seconds)


def test_infra_dry_run_benchmark() -> None:
    """Exercise CPU or every assigned GPU without downloading external data."""
    stage_name = os.environ.get("stageName", "")
    if stage_name.startswith("CPU-"):
        _run_cpu()
        return

    rank_context = _external_rank_context(os.environ)
    if rank_context is not None:
        rank, local_rank, world_size = rank_context
        if world_size == 1:
            _run_cuda_matmul(local_rank)
        elif os.environ.get("TLLM_SPAWN_PROXY_PROCESS") == "1":
            if rank != 0:
                raise RuntimeError(
                    "only LLMAPI rank 0 may run the infrastructure pytest controller"
                )
            _run_with_existing_llmapi_launcher(world_size)
        else:
            _run_distributed_rank(rank, local_rank, world_size)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this infrastructure dry-run stage")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("no CUDA devices are visible to the infrastructure dry run")
    if device_count == 1:
        _run_cuda_matmul(0)
        return

    torch.multiprocessing.spawn(
        _worker_import_module()._local_rank_worker,
        args=(device_count, _reserve_local_port(), _DISTRIBUTED_TIMEOUT_SECONDS),
        nprocs=device_count,
        join=True,
    )
