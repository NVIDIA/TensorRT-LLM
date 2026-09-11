"""Minimal Rubin PCG regression test for named ProcessGroup collectives.

Run it with any TP size; use two four-GPU Rubin nodes to reproduce TP8:

    TLLM_DISABLE_MPI=1 python -m torch.distributed.run --nnodes=2 \
        --nproc_per_node=4 --node_rank=<0|1> --master_addr=<head> \
        --master_port=29500 tests/torch/distributed/rubin_pcg_named_collectives.py

It deliberately does not load a model.  It executes the same named C++
ProcessGroup collectives selected by ``ops.py`` under CUDA graph capture and
replay, then validates their outputs against the deterministic expected values.
The functional-collective path is included as the Python fallback control.
"""

import argparse
import os
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

import tensorrt_llm  # Registers TensorRT-LLM custom operators.
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy


def _custom_workload(x: torch.Tensor, ranks: list[int], rank: int,
                     group_name: str) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
    """Exercise the exact named C++ collective operators used under PCG."""
    reduced = torch.ops.trtllm.allreduce_pg_by_name(
        input=x,
        residual=None,
        norm_weight=None,
        scale=None,
        bias=None,
        workspace=None,
        group=ranks,
        rank=rank,
        group_name=group_name,
        strategy=int(AllReduceStrategy.NCCL),
        op=int(AllReduceFusionOp.NONE),
        eps=1e-6,
        trigger_completion_at_end=True,
    )[0]
    gathered = torch.ops.trtllm.allgather_pg_by_name(x, None, ranks,
                                                      group_name)
    # Every rank provides world-size equal chunks.  Reduce-scatter should
    # therefore produce the same sum on each rank.
    scattered = torch.ops.trtllm.reducescatter_pg_by_name(
        x.repeat((len(ranks), 1)), None, ranks, rank, group_name)
    return reduced, gathered, scattered


def _functional_workload(x: torch.Tensor, group: dist.ProcessGroup,
                         world_size: int) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
    """Graph-native control path used by the Rubin Python workaround."""
    reduced = funcol.all_reduce(x, "sum", group)
    gathered = funcol.all_gather_tensor(x, 0, group)
    scattered = funcol.reduce_scatter_tensor(x.repeat((world_size, 1)), "sum",
                                               0, group)
    return reduced, gathered, scattered


def _expected(x: torch.Tensor, world_size: int) -> Tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:
    # Inputs on rank r contain r + 1.  The sum across TP is triangular(world).
    summed = torch.full_like(x, world_size * (world_size + 1) / 2)
    gathered = torch.cat(
        [torch.full_like(x, rank + 1) for rank in range(world_size)], dim=0)
    return summed, gathered, summed


def _assert_outputs(label: str, outputs: Tuple[torch.Tensor, ...],
                    expected: Tuple[torch.Tensor, ...]) -> None:
    for name, actual, want in zip(("allreduce", "allgather", "reducescatter"),
                                  outputs, expected):
        torch.testing.assert_close(
            actual,
            want,
            rtol=0,
            atol=0,
            msg=lambda message: f"{label} {name}: {message}",
        )


def _capture_and_replay(workload: Callable[[torch.Tensor], Tuple[torch.Tensor,
                                                                  ...]],
                        x: torch.Tensor, replays: int,
                        expected: Tuple[torch.Tensor, ...], label: str, warmup_iterations: int = 3) -> None:
    # Warmup is outside capture, matching the engine's PCG capture lifecycle.
    for _ in range(warmup_iterations):
        _assert_outputs(label + " warmup", workload(x), expected)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = workload(x)
    torch.cuda.synchronize()

    for replay in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        _assert_outputs(f"{label} replay={replay}", captured, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument(
        "--cold-capture",
        action="store_true",
        help="Capture the named custom operators without ordinary eager calls.",
    )
    parser.add_argument(
        "--skip-native-warmup",
        action="store_true",
        help="Do not invoke the fixed wheel's pre-capture topology warmup.",
    )
    parser.add_argument("--skip-functional-control", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This regression test requires a Rubin CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("Run with torchrun and at least two GPUs")

    # Match Ray/TorchDist: CUDA tensors use NCCL and topology metadata uses Gloo.
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.new_group(list(range(world_size)), backend="cuda:nccl,cpu:gloo")
    group_name = str(group.group_name)

    # AllreduceOp's topology probe uses TensorRT-LLM's native world/local
    # ProcessGroup registry. The engine initializes it through TorchDist;
    # reproduce that setup here so this test follows the production path.
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_group = None
    for first_rank in range(0, world_size, local_world_size):
        node_ranks = list(range(first_rank, first_rank + local_world_size))
        candidate = dist.new_group(node_ranks, backend="cuda:nccl,cpu:gloo")
        if rank in node_ranks:
            local_group = candidate
    assert local_group is not None
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())

    # Must be eager: this is the production pre-capture native warmup.
    ranks = list(range(world_size))
    if not args.skip_native_warmup:
        torch.ops.trtllm.allreduce_pg_warmup_by_name(
            torch.ones(1, device=device, dtype=torch.float32), ranks, rank,
            group_name)

    try:
        x = torch.full((args.tokens, args.hidden_size), rank + 1,
                       device=device, dtype=torch.bfloat16)
        expected = _expected(x, world_size)

        _capture_and_replay(
            lambda value: _custom_workload(value, ranks, rank, group_name),
            x,
            args.replays,
            expected,
            "named-custom",
            warmup_iterations=0 if args.cold_capture else 3,
        )
        if not args.skip_functional_control:
            _capture_and_replay(
                lambda value: _functional_workload(value, group, world_size),
                x,
                args.replays,
                expected,
                "functional-control",
                warmup_iterations=3,
            )
        dist.barrier()
        if rank == 0:
            print(
                f"PASS: {world_size}-GPU named collectives survived CUDA graph "
                f"capture and {args.replays} replays (tokens={args.tokens}, "
                f"hidden_size={args.hidden_size})",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
