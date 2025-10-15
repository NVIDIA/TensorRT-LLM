import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    weights, hidden_size, dtype):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def mlp_forward(x, hidden_size, dtype, tensor_parallel_size,
                tensor_parallel_rank, weights):
    x = x.cuda()

    l0 = Linear(
        in_features=hidden_size,
        out_features=4 * hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
    )
    l0.load_weights([dict(weight=weights[0])])
    l0.cuda()
    l1 = Linear(
        in_features=4 * hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    l1.load_weights([dict(weight=weights[1])])
    l1.cuda()

    mlp = torch.compile(lambda x: l1.forward(l0.forward(x)), fullgraph=True)
    output = mlp(x)

    # torch run
    l0 = nn.Linear(in_features=hidden_size,
                   out_features=4 * hidden_size,
                   bias=False,
                   dtype=dtype)
    l0.weight.data.copy_(weights[0])
    l0.cuda()

    l1 = nn.Linear(in_features=4 * hidden_size,
                   out_features=hidden_size,
                   bias=False,
                   dtype=dtype)
    l1.weight.data.copy_(weights[1])
    l1.cuda()

    torch_output = l1.forward(l0.forward(x))

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output, rtol=0.05, atol=0.05)


@torch.inference_mode
def column_linear_forward(x, hidden_size, dtype, tensor_parallel_size,
                          tensor_parallel_rank, weights):

    x = x.cuda()
    l0 = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    )
    l0.load_weights([dict(weight=weights[0])])
    l0.cuda()

    l0 = torch.compile(l0, fullgraph=True)
    output = l0.forward(x)

    # torch run
    l0 = nn.Linear(in_features=hidden_size,
                   out_features=hidden_size,
                   bias=False,
                   dtype=dtype)
    l0.weight.data.copy_(weights[0])
    l0.cuda()

    torch_output = l0.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def row_linear_forward(x, hidden_size, dtype, tensor_parallel_size,
                       tensor_parallel_rank, weights):

    x = x.cuda()
    l0 = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    l0.load_weights([dict(weight=weights[0])])
    l0.cuda()

    xs = torch.chunk(x, 2, dim=-1)
    l0 = torch.compile(l0, fullgraph=True)
    output = l0.forward(xs[tensor_parallel_rank])

    # torch run
    l0 = nn.Linear(in_features=hidden_size,
                   out_features=hidden_size,
                   bias=False,
                   dtype=dtype)
    l0.weight.data.copy_(weights[0])
    l0.cuda()

    torch_output = l0.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output, rtol=0.05, atol=0.05)


@torch.inference_mode
def row_linear_norm_fusion_forward(x, hidden_size, dtype, tensor_parallel_size,
                                   tensor_parallel_rank, weights):

    x = x.cuda()
    residual = torch.randn_like(x)
    norm_weight = torch.randn((1, hidden_size), dtype=dtype, device="cuda")
    eps = 1e-6
    fusion_params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=norm_weight,
        eps=eps,
    )

    l0 = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    l0.load_weights([dict(weight=weights[0])])
    l0.cuda()

    xs = torch.chunk(x, 2, dim=-1)
    l0 = torch.compile(l0, fullgraph=True)
    final_output, inter_output = l0.forward(
        xs[tensor_parallel_rank],
        all_reduce_params=fusion_params,
    )

    # torch run
    l0 = nn.Linear(in_features=hidden_size,
                   out_features=hidden_size,
                   bias=False,
                   dtype=dtype)
    l0.weight.data.copy_(weights[0])
    l0.cuda()

    torch_output = l0.forward(x)
    torch_inter_output = torch_output + residual
    torch_final_output = rms_norm(torch_inter_output, norm_weight, eps)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(
        inter_output,
        torch_inter_output,
        rtol=0.05,
        atol=0.15,
    )
    torch.testing.assert_close(
        final_output,
        torch_final_output,
        rtol=0.05,
        atol=0.15,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mlp(mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    l0_weight = torch.randn((4 * hidden_size, hidden_size), dtype=dtype)
    l1_weight = torch.randn((hidden_size, 4 * hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, mlp_forward, x, [l0_weight, l1_weight],
                hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_column_linear(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 10
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    l0_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, column_linear_forward, x, [l0_weight],
                hidden_size, dtype)] * 2))
    if hidden_size % 2 != 0:
        with pytest.raises(AssertionError):
            for r in results:
                assert r is True
    else:
        for r in results:
            assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [16, 15],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_linear(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    l0_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_linear_forward, x, [l0_weight],
                hidden_size, dtype)] * 2))
    if hidden_size % 2 != 0:
        with pytest.raises(AssertionError):
            for r in results:
                assert r is True
    else:
        for r in results:
            assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("seq_len", [2, 32], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [16, 256], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_linear_norm_fusion(seq_len, hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    l0_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_linear_norm_fusion_forward, x,
                [l0_weight], hidden_size, dtype)] * 2))
    for r in results:
        assert r is True
