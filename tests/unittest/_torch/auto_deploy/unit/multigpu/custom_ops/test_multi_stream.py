"""Unit tests for multi-stream linear op."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.auto_deploy.distributed.common import initialize_or_skip

# Register this module for cloudpickle serialization for MPI workers
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_multi_stream_linear_single_rank(tensor_parallel_size: int):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    initialize_or_skip(rank=rank, world_size=tensor_parallel_size, port=29500)

    try:
        # required for the _multi_stream_test_utils module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../_utils_test"))
        from _multi_stream_test_utils import (
            ParallelTwoLinear,
            replace_multi_stream_linear_with_aux_stream_wrapper,
        )

        from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
            cuda_stream_manager,
        )

        in_dim, out_dim = 128, 256
        cuda_stream_manager.add_device(torch.cuda.current_device())
        model = (
            nn.Sequential(
                ParallelTwoLinear(in_dim, out_dim, tensor_parallel_size),
                ParallelTwoLinear(out_dim, out_dim, tensor_parallel_size),
            )
            .eval()
            .to("cuda")
        )

        # Example input used for export
        example_input = torch.randn(4, in_dim).to("cuda")

        # Export the graph
        egm = torch.export.export(model, (example_input,))
        gm = egm.module()

        test_x = torch.randn(4, in_dim).to("cuda")
        ref_output = model(test_x)

        # pattern matching and replace
        gm, num_replaced = replace_multi_stream_linear_with_aux_stream_wrapper(gm)
        print(gm.graph)
        assert num_replaced == 2
        y = gm(test_x)
        assert torch.allclose(y, ref_output)

        static_x = torch.randn(4, in_dim).to("cuda")
        static_output = torch.randn(4, out_dim).to("cuda")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(gm(static_x))

        static_x.copy_(test_x)
        graph.replay()

        assert torch.allclose(static_output, ref_output)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [1, 2], indirect=True)
def test_multi_stream_linear(mpi_pool_executor):
    """Test all_reduce operation across multiple GPUs."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        run_multi_stream_linear_single_rank,
        *zip(*[(tensor_parallel_size,)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True
