import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from torch import nn
from utils.util import skip_pre_blackwell

import tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

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


def check_accuracy(a, b, atol, rtol, percent):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if not (mismatch_percent < 1 - percent):
        raise Exception("Mismatch percentage is %f for rtol %f" %
                        (mismatch_percent, rtol))


@torch.inference_mode
def fp4_row_linear_allreduce(tp_size, local_rank, seq_len, output_size,
                             hidden_size, dtype, output_ref, x_sf_global,
                             w_sf_global, x_fp4s, w_fp4, x_sf_blocks,
                             w_sf_block_unswizzled):
    output_ref = output_ref.cuda()
    x_sf_global = x_sf_global.cuda()
    w_sf_global = w_sf_global.cuda()
    x_fp4 = x_fp4s[local_rank].cuda()
    w_fp4 = w_fp4.cuda()
    x_sf_block = x_sf_blocks[local_rank].cuda()
    w_sf_block_unswizzled = w_sf_block_unswizzled.cuda()

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l0 = Linear(
        in_features=hidden_size,
        out_features=output_size,
        bias=False,
        dtype=dtype,
        quant_config=qc,
        mapping=Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=local_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )

    l0.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(torch.float8_e4m3fn),
        'weight_scale_2':
        1.0 / w_sf_global.cpu()
    }])

    l0.cuda()
    # TODO: parameters['weight']' size mismatch at index 0
    # l0 = torch.compile(l0)
    with torch.inference_mode(), autotune():
        output = l0.forward((x_fp4, x_sf_block))

    torch.cuda.synchronize()
    check_accuracy(output, output_ref, atol=0.05, rtol=0.05, percent=0.99)


def fp4_row_linear_allreduce_run_single_rank(func, tp_size, seq_len,
                                             output_size, hidden_size, dtype,
                                             output_ref, x_sf_global,
                                             w_sf_global, x_fp4s, w_fp4,
                                             x_sf_blocks,
                                             w_sf_block_unswizzled):
    local_rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(local_rank)

    try:
        func(tp_size, local_rank, seq_len, output_size, hidden_size, dtype,
             output_ref, x_sf_global, w_sf_global, x_fp4s, w_fp4, x_sf_blocks,
             w_sf_block_unswizzled)
    except Exception as e:
        print(f"Error: {e}")
        raise
    return True


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("seq_len", [256, 400], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("output_size", [32, 64], ids=lambda x: f"output:{x}")
@pytest.mark.parametrize("hidden_size", [128, 256], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=lambda x: f"dtype:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2],
                         indirect=True,
                         ids=lambda x: f"tp_size:{x}")
def test_fp4_row_linear_allreduce(seq_len, output_size, hidden_size, dtype,
                                  mpi_pool_executor, monkeypatch):
    monkeypatch.setenv("TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", "1")
    torch.manual_seed(42)
    tp_size = mpi_pool_executor.num_workers

    x = torch.randn((seq_len, hidden_size), dtype=dtype).cuda()
    w = torch.randn((output_size, hidden_size), dtype=dtype).cuda()

    scaling_vector_size = 16
    x_sf_global = (448 * 6) / x.abs().max().float()
    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(x, x_sf_global,
                                                      scaling_vector_size,
                                                      False)
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)
    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(output_size, 128), -1)))

    with torch.inference_mode():
        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        output_ref = torch.ops.trtllm.fp4_gemm(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, dtype)

    torch.cuda.synchronize()

    xs = [x.contiguous().cuda() for x in torch.chunk(x, tp_size, dim=-1)]
    x_fp4s = []
    x_sf_blocks = []
    for i in range(tp_size):
        _fp4, _sf_block = torch.ops.trtllm.fp4_quantize(xs[i], x_sf_global,
                                                        scaling_vector_size,
                                                        False)
        x_fp4s.append(_fp4.cpu())
        x_sf_blocks.append(_sf_block.cpu())

    output_ref = output_ref.cpu()
    x_sf_global = x_sf_global.cpu()
    w_sf_global = w_sf_global.cpu()
    w_fp4 = w_fp4.cpu()
    w_sf_block_unswizzled = w_sf_block_unswizzled.cpu()

    results = mpi_pool_executor.map(
        fp4_row_linear_allreduce_run_single_rank,
        *zip(*[(fp4_row_linear_allreduce, tp_size, seq_len, output_size,
                hidden_size, dtype, output_ref, x_sf_global, w_sf_global,
                x_fp4s, w_fp4, x_sf_blocks, w_sf_block_unswizzled)] * tp_size))

    for r in results:
        assert r is True
