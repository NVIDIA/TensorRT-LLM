"""NCCL_SYMMETRIC must not write into an NCCL window buffer while capturing.

Replaying a graph that captured such an all-reduce yields wrong results (on
TP=4 GB200: a non-finite residual at the first global-attention layer and a
decode collapse). A 2x2 ablation showed the output buffer is solely
responsible; a window input is harmless.

The test asserts that invariant instead of trying to reproduce the corruption:
outside the full model the numerical failure does not reproduce, and a
pointer-stability assertion gives false negatives -- an allocator fix that
removed all cross-graph aliasing passed it while the model still collapsed.
"""

import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceParams,
                                             AllReduceStrategy)
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

HIDDEN = 1024
TOKENS = 8


def _in_window(tensor, group):
    """True if ``tensor`` is backed by a registered NCCL window buffer."""
    if not hasattr(torch.ops.trtllm, "is_nccl_window_buffer"):
        return False
    return bool(torch.ops.trtllm.is_nccl_window_buffer(tensor, list(group)))


def _run(rank, world):
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    ar = AllReduce(mapping=mapping,
                   strategy=AllReduceStrategy.NCCL_SYMMETRIC,
                   dtype=torch.bfloat16)
    group = mapping.tp_group
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16,
                    device=torch.device("cuda", rank))

    def call(t):
        return ar(t,
                  all_reduce_params=AllReduceParams(
                      fusion_op=AllReduceFusionOp.NONE))

    # Warm up on a side stream: required before an NCCL-involving capture, and
    # it is what populates the window pool.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            eager_out = call(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Reported, not asserted: a pool miss may legitimately fall back, but the
    # fix must not disable the window path outside capture.
    eager_in_window = _in_window(eager_out, group)

    static_in = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = call(static_in)
    torch.cuda.synchronize()

    captured_in_window = _in_window(captured_out, group)
    print(
        f"[rank {rank}] eager_output_in_window={eager_in_window} "
        f"captured_output_in_window={captured_in_window}",
        flush=True)

    assert not captured_in_window, (
        f"[rank {rank}] the NCCL_SYMMETRIC all-reduce output captured into a "
        f"CUDA graph is backed by an NCCL window buffer; replaying such a "
        f"graph yields wrong results")

    static_in.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured_out).all(), (
        f"[rank {rank}] replayed all-reduce produced non-finite values")


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs for TP=4")
def test_nccl_symmetric_output_not_in_window_under_capture():
    world = tensorrt_llm.mpi_world_size()
    if world < 2:
        pytest.skip("needs a multi-rank launch")
    _run(tensorrt_llm.mpi_rank(), world)
    MPI.COMM_WORLD.barrier()


if __name__ == "__main__":
    _run(tensorrt_llm.mpi_rank(), tensorrt_llm.mpi_world_size())
    MPI.COMM_WORLD.barrier()
    if tensorrt_llm.mpi_rank() == 0:
        print("NCCL_SYMMETRIC_CUDAGRAPH_WINDOW_TEST_OK", flush=True)
