# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce divergent GEN cancellation and collective order with two MPI ranks.

Run with a matching TRT-LLM build from the repository root::

    timeout 45s mpirun -np 2 python tests/unittest/_torch/disaggregation/repro_tp_gen_collective.py --mode baseline
    timeout 45s mpirun -np 2 python tests/unittest/_torch/disaggregation/repro_tp_gen_collective.py --mode fixed
    timeout 45s mpirun -np 2 python tests/unittest/_torch/disaggregation/repro_tp_gen_collective.py --mode forced-skew

The baseline reads the pre-fix executor and coordinator methods from --base-ref,
then demonstrates an allgather/broadcast deadlock. The fixed modes must exit
normally. The baseline prints both stacks after seven seconds and is stopped by
the outer timeout (exit 124). The forced-skew mode checks admission when a
previous divergence has already occurred. --baseline-dir can supply baseline_executor.py and
baseline_coordinator.py when the Git history is unavailable inside a container.

Request/transfer progress and KV resource effects are controlled fakes. MPI,
executor cancellation, coordinator admission/receive, and the Python
transceiver's status-consensus call are real. No model or NIXL transfer runs.
"""

import argparse
import ast
import faulthandler
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import MethodType, SimpleNamespace

from coordinator_harness import CoordinatorHarness, TransferRequest
from mpi4py import MPI

import tensorrt_llm._torch.disaggregation.orchestration.coordinator as coordinator_module
import tensorrt_llm._torch.pyexecutor.py_executor as executor_module
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.mapping import Mapping

_BASE = "3cee20415044a0d5d128fc2dbddb0b9daaf46512"
_ROOT = Path(__file__).resolve().parents[4]


def _baseline_method(source: str, filename: str, name: str, context: dict) -> Callable:
    """Load the specified original method without importing its entire module."""
    tree = ast.parse(source, filename=filename)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    scope = dict(context)
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), scope)
    return scope[name]


def _read_baseline(args: argparse.Namespace, kind: str, source_path: str) -> str:
    """Read pre-fix source from Git or an explicitly supplied directory."""
    if args.baseline_dir:
        return (args.baseline_dir / f"baseline_{kind}.py").read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), "show", f"{args.base_ref}:{source_path}"], text=True
    )


def main() -> None:
    """Run one rank of the selected bounded reproduction scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "fixed", "forced-skew"), required=True)
    parser.add_argument("--base-ref", default=_BASE)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    if size != 2:
        raise ValueError("Run this reproduction with exactly two MPI ranks")

    dist = MPIDist(Mapping(world_size=size, rank=rank, tp_size=size))
    harness = CoordinatorHarness(dist=dist, is_kv_manager_v2=True, consumes_transfer_buffer=False)
    engine = object.__new__(executor_module.PyExecutor)
    engine.dist = dist
    engine.enable_attention_dp = False
    engine.kv_connector_manager = None
    engine.kv_cache_transceiver = harness.transceiver
    engine._disagg_coordinator = harness.coordinator
    engine.canceled_req_ids = [42]
    engine.waiting_queue = SimpleNamespace(remove_by_ids=lambda ids: None)
    request = TransferRequest(
        42, is_context_only_request=False, state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    )
    request.finish_by_reason = lambda reason: setattr(request, "is_finished", True)
    engine.active_requests = [request]
    # Rank 0's receive has drained; rank 1 still owns transfer resources.
    engine._try_cancel_request = lambda request: rank == 0

    if args.mode == "baseline":
        old_executor = _read_baseline(
            args, "executor", "tensorrt_llm/_torch/pyexecutor/py_executor.py"
        )
        old_coordinator = _read_baseline(
            args, "coordinator", "tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py"
        )
        engine._handle_canceled_requests = MethodType(
            _baseline_method(
                old_executor,
                "baseline_executor.py",
                "_handle_canceled_requests",
                vars(executor_module),
            ),
            engine,
        )
        harness.coordinator.align_gen_admission_across_cp = MethodType(
            _baseline_method(
                old_coordinator,
                "baseline_coordinator.py",
                "align_gen_admission_across_cp",
                vars(coordinator_module),
            ),
            harness.coordinator,
        )

    engine._handle_canceled_requests()
    expected_finished = args.mode == "baseline" and rank == 0
    assert request.is_finished is expected_finished
    print(f"{args.mode} rank={rank}: finished_cancel={request.is_finished}", flush=True)
    MPI.COMM_WORLD.Barrier()

    candidate = TransferRequest(
        43, is_context_only_request=False, state=LlmRequestState.DISAGG_GENERATION_INIT
    )
    local = [candidate] if rank == 0 and (request.is_finished or args.mode == "forced-skew") else []
    admitted = harness.coordinator.align_gen_admission_across_cp(local)
    print(f"{args.mode} rank={rank}: admitted={[r.py_request_id for r in admitted]}", flush=True)

    # Execute the real consensus path with no physical receive sessions.
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver._gen_idle_agreed = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_allgather = dist.allgather
    harness.transceiver.check_gen_transfer_status = lambda at_least: (
        transceiver.check_gen_transfer_status(at_least)
    )
    faulthandler.dump_traceback_later(7, repeat=False)
    print(f"{args.mode} rank={rank}: entering receive_gen_init", flush=True)
    harness.coordinator.receive_gen_init(admitted)
    print(f"{args.mode} rank={rank}: entering next request broadcast", flush=True)
    dist.broadcast("next iteration" if rank == 0 else None, root=0)
    faulthandler.cancel_dump_traceback_later()
    print(f"{args.mode} rank={rank}: completed", flush=True)


if __name__ == "__main__":
    main()
