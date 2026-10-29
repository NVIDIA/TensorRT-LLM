# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank IPC arithmetic, shared-workspace interleave and CUDA graph matrix.

Run with CUDA_VISIBLE_DEVICES=0,1,2,3. The launcher kills all ranks on any failure or
at its deadline. The communication is real IPC; gloo only exchanges allocation handles.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

DEADLINE_S = 900
_WORKER_FLAG = "--rank-worker"
HIDDEN = 6144
EPS = 1e-5


def _run_one_rank():
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 4
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (10, 7)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", rank)
    )
    host_group = dist.new_group(backend="gloo")
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as shared_ar
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.comm.intra_node_moe_finalize import (  # noqa: E501
        intra_node_moe_finalize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.comm.intra_node_reduce import (  # noqa: E501
        IpcCollectives,
        intra_node_reduce,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.comm.pre_moe_fp4_linear import (  # noqa: E501
        pre_moe_fp4_linear,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating

    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    class HostExchange:
        def tp_allgather(self, obj, *, small_payload=False):
            result = [None] * world
            dist.all_gather_object(result, obj, group=host_group)
            return result

    local_group = dist.new_group(ranks=list(range(world)), backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())
    communicator.Distributed.get = staticmethod(lambda mapping: HostExchange())
    mapping = Mapping(
        world_size=4, rank=rank, gpus_per_node=4, tp_size=4, moe_tp_size=4, moe_ep_size=1
    )
    table = get_allreduce_workspace(mapping)
    ipc, reason = IpcCollectives.bind(mapping, [table, table])
    assert ipc is not None, reason
    state = ipc.workspace
    assert (
        state.table is table and state.flags is state.owners[3] and state.layout is state.owners[4]
    )
    # One rank cannot bind a lookalike table: all ranks must decline before a launch.
    rejected, reason = IpcCollectives.bind(mapping, [table.clone() if rank == 0 else table])
    assert rejected is None and "rank 0" in reason
    assert not ipc.supported(25) and not ipc.supported(0)
    args = shared_ar.allreduce_group_args(mapping, int(AllReduceStrategy.ONESHOT))
    assert args["workspace"] is table

    def tensor(rows, seed, r=rank):
        g = torch.Generator(device="cuda").manual_seed(seed * 1009 + r)
        return torch.randn((rows, HIDDEN), device="cuda", generator=g).to(torch.bfloat16)

    def norm_for(seed):
        return SimpleNamespace(weight=(1 + tensor(1, seed, 0)[0] * 0.1), variance_epsilon=EPS)

    def shared_plain(x, residual, norm):
        return shared_ar.allreduce(
            x,
            residual=residual,
            norm_weight=norm.weight,
            scale=None,
            bias=None,
            op=shared_ar.RESIDUAL_RMS_NORM,
            eps=EPS,
            trigger_completion_at_end=True,
            **args,
        )

    def close_pair(actual, expected, exact_residual=True):
        torch.testing.assert_close(
            actual[1], expected[1], rtol=0, atol=0 if exact_residual else 0.03125
        )
        # Both normalize BF16 residuals. A different CTA reduction tree may move the norm by one BF16 step.
        torch.testing.assert_close(actual[0], expected[0], rtol=0.008, atol=0.0002)

    def check_state(rows, previous):
        torch.cuda.synchronize()
        assert state.flags.cpu().tolist() == [0, 0, (previous + 1) % 3]
        assert state.layout.cpu().tolist() == [rows * HIDDEN * world, state.buffer_bytes]

    # Interleave a large shared oneshot dirty range with small owned kernels, without resets.
    rows_to_test = [cell.spec["rows"] for cell in intra_node_reduce.CELLS] + [1, 18, 3]
    for iteration, rows in enumerate(rows_to_test):
        norm = norm_for(101 + iteration)
        shared_plain(tensor(256, 11 + iteration), tensor(256, 29, 0), norm)
        assert state.layout[0].item() == 256 * HIDDEN * world
        x, residual = tensor(rows, 31 + iteration), tensor(rows, 73, 0)
        saved_x, saved_residual = x.clone(), residual.clone()
        old_flag = state.flags[2].item()
        if rank == iteration % world:
            time.sleep(0.02)
        with validating(intra_node_reduce):
            actual = ipc.plain_reduce(
                norm, x, residual, trigger_completion_at_end=iteration % 2 == 0
            )
        check_state(rows, old_flag)
        expected = shared_plain(x, residual, norm)
        close_pair(actual, expected)
        torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
        torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
        # Independent rank-distinct arithmetic catches a dropped or repeated rank.
        reference = intra_node_reduce.reference(
            [tensor(rows, 31 + iteration, r) for r in range(world)],
            residual,
            norm.weight,
            EPS,
            fp32_acc=os.getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32", "1") != "0",
        )
        close_pair(actual, reference)
        total = tensor(rows, 31 + iteration, 0).float()
        for other in range(1, world):
            if os.getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32", "1") != "0":
                total = total + tensor(rows, 31 + iteration, other).float()
            else:
                total = (total.to(torch.bfloat16) + tensor(rows, 31 + iteration, other)).float()
        reduced = total.to(torch.bfloat16) + residual
        torch.testing.assert_close(actual[1], reduced, rtol=0, atol=0)

    def moe_payload(rows, seed):
        fc2 = tensor(rows * 8, seed)
        weights = torch.full((rows, 8), 0.125, dtype=torch.bfloat16, device="cuda")
        indices = torch.arange(rows * 8, dtype=torch.int32, device="cuda").reshape(rows, 8)
        indices[:, 3] = -1
        return fc2, weights, indices, tensor(rows, seed + 1), tensor(rows, seed + 2, 0)

    for cell in intra_node_moe_finalize.CELLS:
        rows = cell.spec["rows"]
        norm = norm_for(211)
        fc2, weights, indices, shared, residual = moe_payload(rows, 181)
        with validating(intra_node_moe_finalize):
            actual = ipc.finalize(fc2, weights, indices, shared, residual, norm)
        expected = torch.ops.trtllm.moe_finalize_allreduce(
            input=fc2,
            residual=residual,
            norm_weight=norm.weight,
            expanded_idx_to_permuted_idx=indices,
            shared_expert_output=shared,
            expert_scale_factor=weights,
            workspace=table,
            rank=rank,
            nranks=world,
            eps=EPS,
        )
        close_pair(actual, expected)
        reference = intra_node_moe_finalize.reference(
            [tensor(rows * 8, 181, r) for r in range(world)],
            [weights] * world,
            [indices] * world,
            [tensor(rows, 182, r) for r in range(world)],
            residual,
            norm.weight,
            EPS,
        )
        close_pair(actual, reference)
        local = []
        for other in range(world):
            fc = tensor(rows * 8, 181, other).reshape(rows, 8, HIDDEN)
            acc = torch.zeros((rows, HIDDEN), device="cuda")
            for slot in range(8):
                if slot != 3:
                    acc = acc + fc[:, slot].float() * 0.125
            local.append((acc + tensor(rows, 182, other).float()).to(torch.bfloat16))
        reduced = local[0]
        for other in range(1, world):
            reduced = reduced + local[other]
        torch.testing.assert_close(actual[1], reduced + residual, rtol=0, atol=0)

    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    for cell in pre_moe_fp4_linear.CELLS:
        rows = cell.spec["rows"]
        norm = norm_for(303)
        x, residual = tensor(rows, 281), tensor(rows, 282, 0)
        with validating(pre_moe_fp4_linear):
            actual = ipc.premoe(x, residual, norm, scale)
        reference = pre_moe_fp4_linear.reference(
            [tensor(rows, 281, r) for r in range(world)],
            residual,
            norm.weight,
            EPS,
            scale,
            fp32_acc=os.getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32", "1") != "0",
        )
        close_pair(actual[:2], reference[:2])
        close_pair(actual[:2], shared_plain(x, residual, norm))
        packed, sf = torch.ops.trtllm.fp4_quantize(actual[0], scale, 16, False, False)
        torch.testing.assert_close(actual[2], packed.reshape_as(actual[2]), rtol=0, atol=0)
        torch.testing.assert_close(
            actual[3], sf.view(torch.uint8).reshape_as(actual[3]), rtol=0, atol=0
        )

    # Replay the same addresses through all three collective shapes, allowing flags to rotate.
    for rows in (1, 6, 18):
        norm = norm_for(411)
        x, residual = tensor(rows, 401), tensor(rows, 402, 0)
        fc2, weights, indices, shared, _ = moe_payload(rows, 403)

        def sequence():
            a = ipc.plain_reduce(norm, x, residual)
            b = ipc.finalize(fc2, weights, indices, shared, a[1], norm)
            # The shared FP4 kernel preloads residual before its PDL dependency wait.
            # Complete this synthetic direct dependency before allowing that preload.
            return ipc.premoe(b[0], b[1].clone(), norm, scale)

        sequence()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sequence()
        for step in range(8):
            x.copy_(tensor(rows, 501 + step))
            if rank == step % world:
                time.sleep(0.01)
            graph.replay()
            got = tuple(t.clone() for t in captured)
            eager = sequence()
            for a, b in zip(got, eager):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert state.flags[0].item() == 0
        del graph

    # Functionalization must fail before a peer-buffer write, not clone only the pointer table.
    try:
        torch.func.functionalize(
            lambda a: torch.ops.trtllm.glm_intra_node_reduce(
                a, residual, norm.weight, EPS, *state.args(), True
            )
        )(x)
    except RuntimeError as exc:
        assert "functionalization" in str(exc).lower()
    else:
        raise AssertionError("opaque IPC functionalization unexpectedly succeeded")
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {rank}] IPC arithmetic/interleave/FP4/graph checks passed", flush=True)
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per visible device, rendezvous over env://, all killed on
    the first nonzero exit or at the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"a collective test needs at least 2 devices, got {world_size}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # See the allgather matrix: inheriting the launching srun step's PMI
    # namespace lets one failing rank take the whole step down unreported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        try:
            _run_one_rank()
        except BaseException:
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
    _spawn_ranks()
    print("OK")
