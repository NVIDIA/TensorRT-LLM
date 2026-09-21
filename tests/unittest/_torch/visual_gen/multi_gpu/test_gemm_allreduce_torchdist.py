# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchDist rendezvous tests for fused GEMM + all-reduce.

These tests intentionally use the same multi-backend ProcessGroup configuration
as VisualGen: CPU collectives route through Gloo and CUDA collectives through
NCCL.
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm.bindings import ipc_nvls_supported
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.fixture(autouse=True, scope="module")
def _cleanup_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)
    os.environ.pop("TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", None)


def _init_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=6.25e-2)


def _run_raw_runner(rank: int, world_size: int) -> None:
    process_group = dist.group.WORLD
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(1234 + rank)
        input = torch.randn(16, 128, device="cuda", dtype=dtype)
        weight = torch.randn(64, 128, device="cuda", dtype=dtype)

        expected = input @ weight.t()
        dist.all_reduce(expected, group=process_group)

        runner = torch.classes.trtllm.GemmAllreduceRunner(dtype, dtype, process_group.boxed())
        assert runner.get_num_configs() > 0

        # The second call exercises the global allocation/workspace cache.
        for _ in range(2):
            output = runner.run_gemm(input, weight, -1)
            _assert_close(output, expected)

    torch.manual_seed(2468 + rank)
    input_fp8 = torch.randn(16, 128, device="cuda").to(torch.float8_e4m3fn)
    weight_fp8 = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
    for output_dtype in (torch.float16, torch.bfloat16):
        expected = (input_fp8.float() @ weight_fp8.float().t()).to(output_dtype)
        dist.all_reduce(expected, group=process_group)
        runner = torch.classes.trtllm.GemmAllreduceRunner(
            torch.float8_e4m3fn, output_dtype, process_group.boxed()
        )
        output = runner.run_gemm(input_fp8, weight_fp8, -1)
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=1.25e-1)


def _quantize_weight_128x128(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference FP8 quantization matching FP8_BLOCK_SCALES checkpoints."""
    n, k = weight.shape
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        ((n + 127) // 128, (k + 127) // 128), dtype=torch.float32, device=weight.device
    )
    for n_block in range(scales.shape[0]):
        for k_block in range(scales.shape[1]):
            block = weight[n_block * 128 : (n_block + 1) * 128, k_block * 128 : (k_block + 1) * 128]
            scale = block.abs().max().float().clamp_min(torch.finfo(torch.float32).tiny) / 448.0
            scales[n_block, k_block] = scale
            quantized[n_block * 128 : (n_block + 1) * 128, k_block * 128 : (k_block + 1) * 128] = (
                block.float() / scale
            ).to(torch.float8_e4m3fn)
    return quantized, scales


def _run_fp8_blockscale_runner(rank: int) -> None:
    process_group = dist.group.WORLD
    torch.manual_seed(1357 + rank)
    m, n, k = 65, 256, 256
    input = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
    weight_fp8, weight_scale = _quantize_weight_128x128(weight)

    # The Hopper kernel scales every K=128 partial before adding it to the
    # FP32 accumulator. Reproduce that order rather than dequantizing the
    # complete operands and issuing one matmul.
    expected = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    padded_m = (m + 3) // 4 * 4
    input_scale = input_scale[: padded_m * (k // 128)].view(k // 128, padded_m)
    for k_block in range(k // 128):
        partial = (
            input_fp8[:, k_block * 128 : (k_block + 1) * 128].float()
            @ weight_fp8[:, k_block * 128 : (k_block + 1) * 128].float().t()
        )
        row_scale = input_scale[k_block, :m, None]
        column_scale = weight_scale[:, k_block].repeat_interleave(128)[:n]
        expected += partial * row_scale * column_scale[None, :]
    expected = expected.to(torch.bfloat16)
    dist.all_reduce(expected, group=process_group)

    runner = torch.classes.trtllm.Fp8BlockScaleGemmAllreduceRunner(
        torch.bfloat16, process_group.boxed()
    )
    assert runner.get_num_configs() > 0
    for _ in range(2):
        output = runner.run_gemm(input_fp8, weight_fp8, input_scale, weight_scale, -1)
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=1.0)


def _run_distinct_process_groups(rank: int) -> None:
    groups = [dist.new_group([0, 1], backend="cuda:nccl,cpu:gloo") for _ in range(2)]
    torch.manual_seed(4321 + rank)
    input = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)

    for process_group in groups:
        expected = input @ weight.t()
        dist.all_reduce(expected, group=process_group)
        runner = torch.classes.trtllm.GemmAllreduceRunner(
            torch.float16, torch.float16, process_group.boxed()
        )
        output = runner.run_gemm(input, weight, -1)
        _assert_close(output, expected)


def _run_visual_gen_linear(rank: int, world_size: int) -> None:
    mapping = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)

    # VisualGen's CUDA DeviceMesh inherits both backends from the world group.
    process_group = mapping.tp_group_pg
    cpu_token = torch.tensor([rank + 1], dtype=torch.int32)
    dist.all_reduce(cpu_token, group=process_group)
    assert cpu_token.item() == world_size * (world_size + 1) // 2

    linear = Linear(
        in_features=128 * world_size,
        out_features=64,
        bias=False,
        dtype=torch.float16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=AllReduceStrategy.NCCL,
    ).cuda()
    assert linear.use_fused_gemm_allreduce
    assert linear._gemm_allreduce_process_group is process_group

    torch.manual_seed(9876 + rank)
    input = torch.randn(2, 4, 128, device="cuda", dtype=torch.float16)
    linear.weight.data.normal_()

    expected = torch.nn.functional.linear(input, linear.weight)
    dist.all_reduce(expected, group=process_group)

    # This attribute is created only by
    # UnquantizedLinearMethod.apply_linear_allreduce. Its transition proves the
    # forward did not take the ordinary GEMM + AllReduce fallback.
    assert not hasattr(linear, "_torch_dist_gemm_allreduce_runner")
    output = linear(input)

    assert output.shape == (2, 4, 64)
    assert linear._torch_dist_gemm_allreduce_runner is not None
    _assert_close(output, expected)


def _run_visual_gen_fp8_blockscale_linear(rank: int, world_size: int) -> None:
    mapping = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
    process_group = mapping.tp_group_pg
    linear = Linear(
        in_features=256 * world_size,
        out_features=256,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        allreduce_strategy=AllReduceStrategy.NCCL,
    ).cuda()
    assert linear.use_fused_gemm_allreduce

    torch.manual_seed(8642 + rank)
    input = torch.randn(2, 5, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    weight_fp8, weight_scale = _quantize_weight_128x128(weight)
    linear.weight.data.copy_(weight_fp8)
    linear.weight_scale.data.copy_(weight_scale)

    input_2d = input.reshape(-1, input.shape[-1])
    input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input_2d)
    expected = torch.zeros(input_2d.shape[0], weight.shape[0], device="cuda", dtype=torch.float32)
    padded_m = (input_2d.shape[0] + 3) // 4 * 4
    input_scale = input_scale[: padded_m * (input_2d.shape[1] // 128)].view(
        input_2d.shape[1] // 128, padded_m
    )
    for k_block in range(input_2d.shape[1] // 128):
        partial = (
            input_fp8[:, k_block * 128 : (k_block + 1) * 128].float()
            @ weight_fp8[:, k_block * 128 : (k_block + 1) * 128].float().t()
        )
        row_scale = input_scale[k_block, : input_2d.shape[0], None]
        column_scale = weight_scale[:, k_block].repeat_interleave(128)
        expected += partial * row_scale * column_scale[None, :]
    expected = expected.to(torch.bfloat16)
    dist.all_reduce(expected, group=process_group)
    expected = expected.reshape(*input.shape[:-1], weight.shape[0])

    assert not hasattr(linear, "_torch_dist_fp8_blockscale_gemm_allreduce_runner")
    output = linear(input)
    assert linear._torch_dist_fp8_blockscale_gemm_allreduce_runner is not None
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=1.25e-1)


def _worker(rank: int, world_size: int, port: int, include_visual_gen: bool) -> None:
    try:
        _init_dist(rank, world_size, port)
        _run_raw_runner(rank, world_size)
        _run_fp8_blockscale_runner(rank)
        if world_size == 2:
            _run_distinct_process_groups(rank)
        if include_visual_gen:
            _run_visual_gen_linear(rank, world_size)
            _run_visual_gen_fp8_blockscale_linear(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_multi_gpu(world_size: int, include_visual_gen: bool = False) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs, have {torch.cuda.device_count()}")
    if torch.cuda.get_device_capability(0) != (9, 0):
        pytest.skip("TorchDist fused GEMM+allreduce currently supports SM90 only")
    if not ipc_nvls_supported():
        pytest.skip("NVLS multicast is not supported")

    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _worker,
            args=(world_size, port, include_visual_gen),
            nprocs=world_size,
            join=True,
        )
    )


@pytest.mark.gpu2
def test_torchdist_gemm_allreduce_and_visual_gen_linear() -> None:
    _run_multi_gpu(world_size=2, include_visual_gen=True)


@pytest.mark.gpu4
def test_torchdist_gemm_allreduce_four_ranks() -> None:
    _run_multi_gpu(world_size=4)
