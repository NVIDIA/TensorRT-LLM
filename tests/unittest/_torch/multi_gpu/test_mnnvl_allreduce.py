# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

SEQ_LEN_CASES = (
    (1, ),
    (4, ),
    (15, ),
    (32, ),
    (128, ),
    (31, 11, 27, 4),  # Switching one/two shot in MNNVL
    (12, 2048),  # reallocate workspace in MNNVL
)
NCCL_SYMMETRIC_SEQ_LEN_CASES = tuple(seq_len for seq_len in SEQ_LEN_CASES
                                     if 2048 not in seq_len)
HIDDEN_SIZES = (8, 2880, 7168, 7176, 8192, 16384)
DTYPES = (torch.bfloat16, )
FUSION_CASES = (True, False)
QUANT_DTYPES = (torch.float16, torch.bfloat16)
QUANT_SEQ_LEN_CASES = (16, 2048)
QUANT_HIDDEN_SIZE = 128
QUANT_FUSION_OPS = (
    pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                 id="residual_rms_norm_quant_fp8"),
    pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
                 id="residual_rms_norm_out_quant_fp8"),
    pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                 id="residual_rms_norm_quant_nvfp4",
                 marks=skip_pre_blackwell),
    pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
                 id="residual_rms_norm_out_quant_nvfp4",
                 marks=skip_pre_blackwell),
)


def _seq_len_id(seq_len: tuple[int, ...]):
    return f"seqlen:{list(seq_len)}"


def _dtype_id(dtype: torch.dtype):
    return f"dtype:{torch.finfo(dtype).dtype}"


def _max_nccl_symmetric_buffer_size(dtype: torch.dtype):
    max_seq_len = max(max(seq_len) for seq_len in NCCL_SYMMETRIC_SEQ_LEN_CASES)
    return max_seq_len * max(HIDDEN_SIZES) * torch.empty(
        (), dtype=dtype).element_size()


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def fp8_quant(input: torch.Tensor, scale: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    inv_scale = scale.reciprocal()
    qinput = (input.float() * inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qinput.to(torch.float8_e4m3fn)


def dequant(input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return (input.to(torch.float32) * scale).to(dtype)


def dequant_nvfp4(quant: torch.Tensor, scale_out: torch.Tensor,
                  global_scale: torch.Tensor):
    import torch
    return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        quant.cpu(), scale_out.cpu(), 1 / global_scale.cpu(), 16, 1, True)


def check_quant_accuracy(a: torch.Tensor, b: torch.Tensor, atol: float,
                         rtol: float, percent: float):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = float(count / a.numel())
    if not mismatch_percent < 1 - percent:
        raise AssertionError(
            f"Mismatch percentage is {mismatch_percent:f} for rtol {rtol:f}")


def fp4_quantize(input: torch.Tensor, scale: torch.Tensor):
    import torch
    return torch.ops.trtllm.fp4_quantize(input, scale, 16, False)


def run_single_rank(
    tensor_parallel_size,
    single_rank_forward_func,
    input_list,
    residual_list,
    norm_weight,
    eps,
    hidden_size,
    dtype,
    fused_add_norm,
    reference_output_list,
    strategy,
    max_userbuffers_size,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(
            input_list,
            residual_list,
            norm_weight,
            eps,
            hidden_size,
            dtype,
            tensor_parallel_size,
            rank,
            fused_add_norm,
            reference_output_list,
            strategy,
            max_userbuffers_size,
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


def run_quant_single_rank(
    tensor_parallel_size,
    single_rank_forward_func,
    input,
    residual,
    norm_weight,
    scale,
    eps,
    dtype,
    fusion_op,
    reference_norm,
    reference_residual,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, norm_weight, scale, eps,
                                 dtype, tensor_parallel_size, rank, fusion_op,
                                 reference_norm, reference_residual)
    except Exception:
        traceback.print_exc()
        raise
    return True


def run_reject_single_rank(tensor_parallel_size, single_rank_forward_func,
                           input, residual, norm_weight, scale):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, norm_weight, scale,
                                 tensor_parallel_size, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def mnnvl_quant_fusion_forward(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    fusion_op: AllReduceFusionOp,
    reference_norm: torch.Tensor,
    reference_residual: torch.Tensor,
):
    input = input.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    scale = scale.cuda()
    reference_norm = reference_norm.cuda()
    reference_residual = reference_residual.cuda()

    os.environ["TLLM_TEST_MNNVL"] = "1"
    MPI.COMM_WORLD.barrier()

    allreduce = AllReduce(
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        strategy=AllReduceStrategy.MNNVL,
        dtype=dtype,
    )

    output = allreduce(
        input,
        all_reduce_params=AllReduceParams(
            fusion_op=fusion_op,
            residual=residual,
            norm_weight=norm_weight,
            scale=scale,
            eps=eps,
        ),
    )

    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        quant_out, residual_out = output
        calc_output = dequant(quant_out, scale, dtype)
        ref_output = dequant(fp8_quant(reference_norm, scale), scale, dtype)
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        norm_out, quant_out, residual_out = output
        torch.testing.assert_close(norm_out,
                                   reference_norm,
                                   rtol=0.05,
                                   atol=0.15)
        calc_output = dequant(quant_out, scale, dtype)
        ref_output = dequant(fp8_quant(reference_norm, scale), scale, dtype)
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        quant_out, scale_out, residual_out = output
        ref_quant, ref_scale_out = fp4_quantize(reference_norm.clone(), scale)
        calc_output = dequant_nvfp4(quant_out, scale_out, scale).cuda()
        ref_output = dequant_nvfp4(ref_quant, ref_scale_out, scale).cuda()
    else:
        norm_out, quant_out, scale_out, residual_out = output
        torch.testing.assert_close(norm_out,
                                   reference_norm,
                                   rtol=0.05,
                                   atol=0.15)
        ref_quant, ref_scale_out = fp4_quantize(reference_norm.clone(), scale)
        calc_output = dequant_nvfp4(quant_out, scale_out, scale).cuda()
        ref_output = dequant_nvfp4(ref_quant, ref_scale_out, scale).cuda()

    check_quant_accuracy(calc_output,
                         ref_output,
                         atol=0.05,
                         rtol=0.15,
                         percent=0.99)
    torch.testing.assert_close(residual_out,
                               reference_residual,
                               rtol=0.05,
                               atol=0.15)


@torch.inference_mode()
def mnnvl_nvfp4_reject_fp32_forward(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
):
    input = input.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    scale = scale.cuda()

    os.environ["TLLM_TEST_MNNVL"] = "1"
    MPI.COMM_WORLD.barrier()

    allreduce = AllReduce(
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        strategy=AllReduceStrategy.MNNVL,
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError,
                       match="NVFP4 quantization requires FP16 or BF16"):
        allreduce(
            input,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                residual=residual,
                norm_weight=norm_weight,
                scale=scale,
                eps=1e-5,
            ),
        )


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
    norm_weight: torch.Tensor,
    eps: float,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    fusion: bool,
    reference_output_list: list[tuple[torch.Tensor, ...]],
    strategy: AllReduceStrategy,
    max_userbuffers_size: int | None,
):

    # Move all tensors to GPU
    x_list = [x.cuda() for x in x_list]
    residual_list = [residual.cuda() for residual in residual_list]
    norm_weight = norm_weight.cuda()
    reference_output_list = [
        tuple(t.cuda() for t in ref_output)
        for ref_output in reference_output_list
    ]

    if strategy == AllReduceStrategy.NCCL_SYMMETRIC:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        assert max_userbuffers_size is not None
        ub.initialize_userbuffers_manager(tensor_parallel_size, 1, 1,
                                          tensor_parallel_rank,
                                          torch.cuda.device_count(),
                                          max_userbuffers_size)
    elif strategy == AllReduceStrategy.MNNVL:
        os.environ["TLLM_TEST_MNNVL"] = "1"

    MPI.COMM_WORLD.barrier()

    # Create a single AllReduce instance to be reused for all sequence lengths
    allreduce = AllReduce(
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        strategy=strategy,
        dtype=dtype,
    )

    # Since all the modules here are provided by TRT-LLM,
    # so it has to be fullgraph compatible
    def func(input, residual, norm_weight, eps, enable_fusion):
        if enable_fusion:
            output, residual = allreduce(
                input,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=norm_weight,
                    eps=eps,
                ),
            )
            return (output, residual)
        else:
            output = allreduce(input)
            return (output, )

    try:
        # Process each sequence length using the same AllReduce instance
        for i, (x, residual, reference_output) in enumerate(
                zip(x_list, residual_list, reference_output_list)):
            output = func(x.clone(), residual.clone(), norm_weight, eps, fusion)

            torch.testing.assert_close(
                output[0],
                reference_output[0],
                rtol=0.05,
                atol=0.15,
            )

            if fusion:
                torch.testing.assert_close(
                    output[1],
                    reference_output[1],
                    rtol=0.05,
                    atol=0.15,
                )
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _run_row_linear_residual_norm_fusion(seq_len,
                                         hidden_size,
                                         dtype,
                                         strategy,
                                         fusion,
                                         mpi_pool_executor,
                                         max_userbuffers_size=None):
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers

    # Create norm_weight once (same for all sequence lengths)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    eps = 1e-5

    # Create lists of tensors for each sequence length
    x_list = []
    residual_list = []
    reference_output_list = []

    for seq_len_val in seq_len:
        x = torch.randn((tensor_parallel_size, seq_len_val, hidden_size),
                        dtype=dtype)
        residual = torch.randn((seq_len_val, hidden_size), dtype=dtype)
        reference_output = (torch.sum(x, dim=0), )
        if fusion:
            residual_out = reference_output[0] + residual
            reference_output = (rms_norm(residual_out.to(torch.float32),
                                         norm_weight,
                                         eps).to(dtype), residual_out)

        x_list.append(x)
        residual_list.append(residual)
        reference_output_list.append(reference_output)

    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[
            (
                tensor_parallel_size,
                row_linear_residual_norm_fusion_forward,
                [x[i, :, :] for x in x_list
                 ],  # Extract the i-th rank's data from each sequence length
                residual_list,
                norm_weight,
                eps,
                hidden_size,
                dtype,
                fusion,
                reference_output_list,
                strategy,
                max_userbuffers_size,
            ) for i in range(tensor_parallel_size)
        ]),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len", SEQ_LEN_CASES, ids=_seq_len_id)
@pytest.mark.parametrize("hidden_size",
                         HIDDEN_SIZES,
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion", FUSION_CASES, ids=["fusion", "no_fusion"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_row_linear_residual_norm_fusion(seq_len, hidden_size, dtype,
                                               fusion, mpi_pool_executor):
    _run_row_linear_residual_norm_fusion(seq_len, hidden_size, dtype,
                                         AllReduceStrategy.MNNVL, fusion,
                                         mpi_pool_executor)


def _make_quant_scale(reference_norm: torch.Tensor,
                      fusion_op: AllReduceFusionOp) -> torch.Tensor:
    amax = reference_norm.abs().max().float().clamp_min(1e-6)
    if fusion_op in (AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                     AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4):
        return ((448 * 6) / amax).to(torch.float32)
    return (amax / 448).to(torch.float32)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len",
                         QUANT_SEQ_LEN_CASES,
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("dtype", QUANT_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion_op", QUANT_FUSION_OPS)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_quant_fusion(seq_len, dtype, fusion_op, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers
    hidden_size = QUANT_HIDDEN_SIZE
    eps = 1e-5

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    reference_residual = torch.sum(x, dim=0) + residual
    reference_norm = rms_norm(reference_residual.to(torch.float32), norm_weight,
                              eps).to(dtype)
    scale = _make_quant_scale(reference_norm, fusion_op)

    results = mpi_pool_executor.map(
        run_quant_single_rank,
        *zip(*[(
            tensor_parallel_size,
            mnnvl_quant_fusion_forward,
            x[i, :, :],
            residual,
            norm_weight,
            scale,
            eps,
            dtype,
            fusion_op,
            reference_norm,
            reference_residual,
        ) for i in range(tensor_parallel_size)]),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_nvfp4_rejects_fp32_before_launch(mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers
    seq_len = 16
    hidden_size = QUANT_HIDDEN_SIZE
    dtype = torch.float32

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    scale = torch.tensor(1.0, dtype=torch.float32)

    results = mpi_pool_executor.map(
        run_reject_single_rank,
        *zip(*[(
            tensor_parallel_size,
            mnnvl_nvfp4_reject_fp32_forward,
            x[i, :, :],
            residual,
            norm_weight,
            scale,
        ) for i in range(tensor_parallel_size)]),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len",
                         NCCL_SYMMETRIC_SEQ_LEN_CASES,
                         ids=_seq_len_id)
@pytest.mark.parametrize("hidden_size",
                         HIDDEN_SIZES,
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion", FUSION_CASES, ids=["fusion", "no_fusion"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_nccl_symmetric_row_linear_residual_norm_fusion(seq_len, hidden_size,
                                                        dtype, fusion,
                                                        mpi_pool_executor):
    _run_row_linear_residual_norm_fusion(
        seq_len,
        hidden_size,
        dtype,
        AllReduceStrategy.NCCL_SYMMETRIC,
        fusion,
        mpi_pool_executor,
        max_userbuffers_size=_max_nccl_symmetric_buffer_size(dtype),
    )
