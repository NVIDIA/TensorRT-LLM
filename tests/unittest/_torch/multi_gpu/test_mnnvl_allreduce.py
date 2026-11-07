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
import gc
import os
import pickle
import platform
import sys
import traceback

import cloudpickle
import pytest
import torch
from _torch.helpers import mpi_worker_env, r128c4_scale_offsets
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce, get_or_scale_allreduce_mnnvl_workspace)
from tensorrt_llm._torch.nccl_window_tensor_scope import \
    discard_nccl_window_tensor_outputs
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
HIDDEN_SIZES = (8, 2880, 7168, 7176, 8192, 16384)
DTYPES = (torch.bfloat16, )
FUSION_CASES = (True, False)
QUANT_DTYPES = (torch.float16, torch.bfloat16)
QUANT_HIDDEN_SIZE = 2880
# Quant fusion uses tp=2 and 2-byte inputs in this test. The first two shapes
# cover one-shot and two-shot at the same real hidden size; the wide hidden-size
# case also crosses the two-shot threshold while keeping the fused RMSNorm stage
# on the CGA grid-policy path.
QUANT_SHAPE_CASES = (
    pytest.param(16, 2880, id="oneshot_m16_n2880"),
    pytest.param(2048, 2880, id="twoshot_m2048_n2880"),
    pytest.param(32, 16384, id="twoshot_cga_m32_n16384"),
)
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


def mnnvl_checkpoint_graph_forward(tensor_parallel_size: int,
                                   tensor_parallel_rank: int, num_tokens: int):
    env_names = ("TLLM_TEST_MNNVL", "TRTLLM_FORCE_MNNVL_AR")
    previous_env = {
        name: (name in os.environ, os.environ.get(name))
        for name in env_names
    }
    tensor_parallel_rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(tensor_parallel_rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    mapping = None
    allreduce = None
    graph = None
    input_ = None
    output = None
    workspace = None
    try:
        MPI.COMM_WORLD.barrier()

        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        )
        stale_workspace = MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(
            mapping, None)
        del stale_workspace
        gc.collect()
        MPI.COMM_WORLD.barrier()
        with torch.inference_mode():
            allreduce = AllReduce(
                mapping=mapping,
                strategy=AllReduceStrategy.MNNVL,
                dtype=torch.bfloat16,
            )
            input_ = torch.full((num_tokens, 128),
                                tensor_parallel_rank + 1,
                                dtype=torch.bfloat16,
                                device="cuda")
            output = allreduce(input_)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = allreduce(input_)

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 3))

        workspace = allreduce.mnnvl_allreduce.allreduce_mnnvl_workspaces[
            mapping]
        uc_address = workspace["uc_buffer"].data_ptr()
        mc_address = workspace["handle"].get_mc_buffer((1, ), torch.float32,
                                                       0).data_ptr()

        for cycle in range(3):
            allreduce.mnnvl_allreduce.checkpoint_prepare()
            allreduce.mnnvl_allreduce.checkpoint_prepare()
            assert not workspace["handle"].is_mapped()
            with pytest.raises(
                    RuntimeError,
                    match="does not have an active MPI communicator"):
                workspace["handle"].checkpoint_restore_complete(True)
            with pytest.raises(RuntimeError, match="handles are not attached"):
                allreduce(input_)

            if cycle == 0:
                reordered_comm = MPI.COMM_WORLD.Split(
                    0, tensor_parallel_size - tensor_parallel_rank - 1)
                with pytest.raises(
                        RuntimeError,
                        match="restore communicator validation failed"):
                    workspace["handle"].checkpoint_restore(
                        reordered_comm.py2f())
                reordered_comm.Free()

            fresh_comm = MPI.COMM_WORLD.Split(0, tensor_parallel_rank)
            allreduce.mnnvl_allreduce.checkpoint_restore(fresh_comm)
            fresh_comm.Free()
            allreduce.mnnvl_allreduce.checkpoint_restore(workspace["comm"])
            assert workspace["handle"].is_mapped()
            assert workspace["uc_buffer"].data_ptr() == uc_address
            assert workspace["handle"].get_mc_buffer((1, ), torch.float32,
                                                     0).data_ptr() == mc_address

            with torch.inference_mode():
                input_.fill_(tensor_parallel_rank + 2 + cycle)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,
                                       torch.full_like(output, 5 + 2 * cycle))

        allreduce.mnnvl_allreduce.checkpoint_prepare()
        abort_comm = MPI.COMM_WORLD.Split(0, tensor_parallel_rank)
        assert workspace["handle"].checkpoint_restore(abort_comm.py2f())
        with pytest.raises(RuntimeError,
                           match="all-reduce protocol reset failed"):
            workspace["handle"].checkpoint_restore_complete(
                tensor_parallel_rank != 0)
        assert not workspace["handle"].is_mapped()
        abort_comm.Free()
        return tensor_parallel_rank, previous_env
    finally:
        workspace = None
        output = None
        input_ = None
        graph = None
        allreduce = None
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        gc.collect()
        for name, (was_present, value) in previous_env.items():
            if was_present:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)


def mnnvl_checkpoint_worker_env(_: int):
    MPI.COMM_WORLD.barrier()
    env_names = ("TLLM_TEST_MNNVL", "TRTLLM_FORCE_MNNVL_AR")
    return tensorrt_llm.mpi_rank(), {
        name: (name in os.environ, os.environ.get(name))
        for name in env_names
    }


def mnnvl_growth_graph_forward(tensor_parallel_size: int,
                               tensor_parallel_rank: int) -> bool:
    """A graph captured on the MNNVL workspace replays correctly after an eager call grew the workspace, a replaced
    workspace that no capture used is not kept, and growth during capture is refused."""
    env_names = ("TLLM_TEST_MNNVL", "TRTLLM_FORCE_MNNVL_AR")
    previous_env = {
        name: (name in os.environ, os.environ.get(name))
        for name in env_names
    }
    tensor_parallel_rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(tensor_parallel_rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    mapping = None
    try:
        MPI.COMM_WORLD.barrier()
        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        )
        MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        MNNVLAllReduce.allreduce_mnnvl_retired_workspaces.pop(mapping, None)
        gc.collect()
        MPI.COMM_WORLD.barrier()
        expected = tensor_parallel_size * (tensor_parallel_size + 1) // 2
        with torch.inference_mode():
            allreduce = AllReduce(
                mapping=mapping,
                strategy=AllReduceStrategy.MNNVL,
                dtype=torch.bfloat16,
            )
            input_ = torch.full((1, 7168),
                                tensor_parallel_rank + 1,
                                dtype=torch.bfloat16,
                                device="cuda")
            allreduce(input_)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = allreduce(input_)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,
                                       torch.full_like(output, expected))

            before = MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping]
            # Two-shot footprint 2 * 4096 * 7168 * 2 B: well past the initial 16 MiB per Lamport buffer.
            big = torch.ones((4096, 7168), dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(
                allreduce(big), torch.full_like(big, tensor_parallel_size))
            after = MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping]
            assert after["buffer_size_bytes"] > before["buffer_size_bytes"]
            assert MNNVLAllReduce.allreduce_mnnvl_retired_workspaces[
                mapping] == [before]
            assert before["handle"].is_mapped()
            del big

            for step in range(3):
                input_.fill_(tensor_parallel_rank + 2 + step)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output,
                    torch.full_like(
                        output, expected + tensor_parallel_size * (1 + step)))

            # No capture used `after`: growing past it keeps only `before`, which the graph still launches on.
            wider = torch.ones((6144, 7168),
                               dtype=torch.bfloat16,
                               device="cuda")
            torch.testing.assert_close(
                allreduce(wider), torch.full_like(wider, tensor_parallel_size))
            assert MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping][
                "buffer_size_bytes"] > after["buffer_size_bytes"]
            assert MNNVLAllReduce.allreduce_mnnvl_retired_workspaces[
                mapping] == [before]
            del wider, after
            input_.fill_(tensor_parallel_rank + 1)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,
                                       torch.full_like(output, expected))

            huge = torch.ones((8192, 7168), dtype=torch.bfloat16, device="cuda")
            with pytest.raises(RuntimeError, match="during CUDA graph capture"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    allreduce(huge)
        return True
    finally:
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
            MNNVLAllReduce.allreduce_mnnvl_retired_workspaces.pop(mapping, None)
        gc.collect()
        for name, (was_present, value) in previous_env.items():
            if was_present:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)


def mnnvl_checkpoint_rejects_wrong_membership(world_size: int,
                                              world_rank: int) -> bool:
    env_names = ("TLLM_TEST_MNNVL", "TRTLLM_FORCE_MNNVL_AR")
    previous_env = {
        name: (name in os.environ, os.environ.get(name))
        for name in env_names
    }
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    mapping = None
    allreduce = None
    try:
        world_rank = tensorrt_llm.mpi_rank()
        torch.cuda.set_device(world_rank)
        MPI.COMM_WORLD.barrier()
        mapping = Mapping(world_size=world_size,
                          tp_size=2,
                          pp_size=2,
                          rank=world_rank)
        MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        allreduce = AllReduce(mapping=mapping,
                              strategy=AllReduceStrategy.MNNVL,
                              dtype=torch.bfloat16)
        allreduce(torch.ones((1, 128), dtype=torch.bfloat16, device="cuda"))
        workspace = MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping]
        allreduce.mnnvl_allreduce.checkpoint_prepare()

        wrong_group = 0 if world_rank in (0, 3) else 1
        wrong_comm = MPI.COMM_WORLD.Split(wrong_group, mapping.tp_rank)
        with pytest.raises(RuntimeError,
                           match="restore communicator validation failed"):
            workspace["handle"].checkpoint_restore(wrong_comm.py2f())
        wrong_comm.Free()

        correct_comm = MPI.COMM_WORLD.Split(mapping.pp_rank, mapping.tp_rank)
        allreduce.mnnvl_allreduce.checkpoint_restore(correct_comm)
        correct_comm.Free()
        assert workspace["handle"].is_mapped()
        return True
    finally:
        allreduce = None
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        gc.collect()
        for name, (was_present, value) in previous_env.items():
            if was_present:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)


@torch.inference_mode()
def mnnvl_rms_norm_fusion_forward(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    reference_norm: torch.Tensor,
):
    input = input.cuda()
    norm_weight = norm_weight.cuda()
    reference_norm = reference_norm.cuda()

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
    assert allreduce.mnnvl_allreduce is not None
    params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RMS_NORM,
        norm_weight=norm_weight,
        eps=eps,
    )
    # Call the MNNVL module directly: AllReduce.forward would silently fall
    # back to another strategy if MNNVL declined the fusion.
    output = allreduce.mnnvl_allreduce(input, all_reduce_params=params)
    assert isinstance(output, torch.Tensor), type(output)
    torch.testing.assert_close(output, reference_norm, rtol=0.05, atol=0.15)


def run_rms_norm_single_rank(tensor_parallel_size, input, norm_weight, eps,
                             dtype, reference_norm):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mnnvl_rms_norm_fusion_forward(input, norm_weight, eps, dtype,
                                      tensor_parallel_size, rank,
                                      reference_norm)
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
            with discard_nccl_window_tensor_outputs((x, residual)):
                output = func(x.clone(), residual.clone(), norm_weight, eps,
                              fusion)

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


def _run_row_linear_residual_norm_fusion(seq_len, hidden_size, dtype, strategy,
                                         fusion, mpi_pool_executor):
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


@pytest.mark.skipif(
    platform.machine().lower() != "aarch64" or torch.cuda.device_count() < 2
    or not MnnvlMemory.supports_mnnvl(),
    reason="requires at least two GB200 GPUs with fabric-backed MNNVL",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("num_tokens", [1, 4096], ids=["oneshot", "twoshot"])
def test_mnnvl_checkpoint_preserves_cuda_graph_addresses(
        mpi_pool_executor, num_tokens) -> None:
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        mnnvl_checkpoint_graph_forward,
        [tensor_parallel_size] * tensor_parallel_size,
        range(tensor_parallel_size),
        [num_tokens] * tensor_parallel_size,
    )
    previous_env_by_rank = dict(results)
    current_env_by_rank = dict(
        mpi_pool_executor.map(
            mnnvl_checkpoint_worker_env,
            range(tensor_parallel_size),
        ))
    assert set(current_env_by_rank) == set(range(tensor_parallel_size))
    assert current_env_by_rank == previous_env_by_rank


@pytest.mark.skipif(
    platform.machine().lower() != "aarch64" or torch.cuda.device_count() < 4
    or not MnnvlMemory.supports_mnnvl(),
    reason="requires four GB200 GPUs with fabric-backed MNNVL",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_mnnvl_checkpoint_rejects_wrong_group_membership(
        mpi_pool_executor) -> None:
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        mnnvl_checkpoint_rejects_wrong_membership,
        [world_size] * world_size,
        range(world_size),
    )
    assert all(results)


@pytest.mark.skipif(
    platform.machine().lower() != "aarch64" or torch.cuda.device_count() < 2
    or not MnnvlMemory.supports_mnnvl(),
    reason="requires at least two GB200 GPUs with fabric-backed MNNVL",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_workspace_growth_keeps_captured_graphs(
        mpi_pool_executor) -> None:
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        mnnvl_growth_graph_forward,
        [tensor_parallel_size] * tensor_parallel_size,
        range(tensor_parallel_size),
    )
    assert all(results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mnnvl_workspace_creation_refuses_graph_capture() -> None:
    """Creating or growing the MNNVL workspace allocates collectively, so under
    CUDA-graph capture it raises before entering the allocation. One process,
    a group of one: the guard runs before any MNNVL call."""
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="during CUDA graph capture"):
            get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
    assert mapping not in MNNVLAllReduce.allreduce_mnnvl_workspaces


def _make_quant_scale(reference_norm: torch.Tensor,
                      fusion_op: AllReduceFusionOp) -> torch.Tensor:
    amax = reference_norm.abs().max().float().clamp_min(1e-6)
    if fusion_op in (AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                     AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4):
        return ((448 * 6) / amax).to(torch.float32)
    return (amax / 448).to(torch.float32)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len, hidden_size", QUANT_SHAPE_CASES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion_op", QUANT_FUSION_OPS)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_quant_fusion(seq_len, hidden_size, dtype, fusion_op,
                            mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers
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
@pytest.mark.parametrize("seq_len, hidden_size", (
    pytest.param(16, 3584, id="oneshot_m16_n3584"),
    pytest.param(2048, 3584, id="twoshot_m2048_n3584"),
))
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_rms_norm_fusion_without_residual(seq_len, hidden_size,
                                                mpi_pool_executor):
    """AllReduce + RMSNorm with no residual operand (Kimi K3 routed latent)."""
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    reference_norm = rms_norm(
        torch.sum(x, dim=0).to(torch.float32), norm_weight, eps).to(dtype)

    results = mpi_pool_executor.map(
        run_rms_norm_single_rank,
        *zip(*[(
            tensor_parallel_size,
            x[i, :, :],
            norm_weight,
            eps,
            dtype,
            reference_norm,
        ) for i in range(tensor_parallel_size)]),
    )
    for r in results:
        assert r is True


@mpi_worker_env(TLLM_TEST_MNNVL="1")
def run_rms_norm_mxfp8_single_rank(tensor_parallel_size, seq_len, hidden_size):
    """RMS_NORM_QUANT_MXFP8 against the low-M GEMM's quantization of the RMS_NORM output of the
    same call, and Kimi K3's latent_up on that operand against latent_up on the BF16 rows."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        from tensorrt_llm._torch.kernels.blackwell import \
            fp8_block_scale_low_m_gemm as low_m
        from tensorrt_llm._torch.models.modeling_kimi_linear import \
            _Fp8BlockScaleWeightReadLinear

        allreduce = AllReduce(mapping=Mapping(world_size=tensor_parallel_size,
                                              tp_size=tensor_parallel_size,
                                              rank=rank),
                              strategy=AllReduceStrategy.MNNVL,
                              dtype=torch.bfloat16)
        assert allreduce.mnnvl_runs_fusion(
            AllReduceFusionOp.RMS_NORM_QUANT_MXFP8, hidden_size)
        assert not allreduce.mnnvl_runs_fusion(
            AllReduceFusionOp.RMS_NORM_QUANT_MXFP8, hidden_size + 128)
        gen = torch.Generator(device="cuda").manual_seed(rank + 3)
        x = (torch.randn(seq_len, hidden_size, device="cuda", generator=gen) *
             0.5).to(torch.bfloat16)
        x[0, :32] = 0  # an all-zero group takes the amax floor
        norm_weight = (1 + 0.1 * torch.randn(
            hidden_size,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(0))).to(
                torch.bfloat16)
        eps = 1e-6
        norm = allreduce(x,
                         all_reduce_params=AllReduceParams(
                             fusion_op=AllReduceFusionOp.RMS_NORM,
                             norm_weight=norm_weight,
                             eps=eps))
        xq, xs = allreduce(x,
                           all_reduce_params=AllReduceParams(
                               fusion_op=AllReduceFusionOp.RMS_NORM_QUANT_MXFP8,
                               norm_weight=norm_weight,
                               eps=eps))
        ref_q, ref_s = low_m.mxfp8_quantize_reference(norm)
        assert xq.dtype == torch.float8_e4m3fn and xs.dtype == torch.uint8
        assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
        assert torch.equal(
            xs[r128c4_scale_offsets(seq_len, hidden_size, "cuda")], ref_s)

        weight = (torch.randn(7168, hidden_size, device="cuda", generator=gen) *
                  0.02).to(torch.bfloat16)
        up = _Fp8BlockScaleWeightReadLinear(
            *_Fp8BlockScaleWeightReadLinear.quantize_weight(weight), 7168)
        up.post_load_weights()
        if up.takes_prequantized_input(seq_len):
            assert torch.equal(up.forward_prequantized(xq, xs), up(norm))
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("seq_len", [1, 16, 96, 300], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("hidden_size", [3072, 7168], ids=lambda h: f"n{h}")
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_rms_norm_mxfp8_fusion(seq_len, hidden_size, mpi_pool_executor):
    """AllReduce + RMSNorm + MXFP8 (Kimi K3 routed latent for latent_up), one-shot and
    two-shot."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = mpi_pool_executor.map(
        run_rms_norm_mxfp8_single_rank,
        *zip(*[(world, seq_len, hidden_size)] * world))
    for r in results:
        assert r is True


@torch.inference_mode()
@mpi_worker_env(TLLM_TEST_MNNVL="1")
def run_own_workspace_single_rank(tensor_parallel_size, hidden_size,
                                  early_state):
    """An all-reduce on a workspace of its own (MnnvlOwnWorkspace): its own buffers, one-shot up
    to its own limit, interleaved with calls on the group's shared workspace; with
    ``early_state`` bitwise the same calls on a workspace without it."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        from tensorrt_llm._torch.distributed.ops import (
            MnnvlOwnWorkspace, mnnvl_allreduce_transport)
        from tensorrt_llm._torch.kernels.blackwell import \
            fp8_block_scale_low_m_gemm as low_m

        MPI.COMM_WORLD.barrier()
        mapping = Mapping(world_size=tensor_parallel_size,
                          tp_size=tensor_parallel_size,
                          rank=rank)
        big = 96
        limit = big * hidden_size * tensor_parallel_size * 2
        shared = AllReduce(mapping=mapping,
                           strategy=AllReduceStrategy.MNNVL,
                           dtype=torch.bfloat16)
        own = AllReduce(mapping=mapping,
                        strategy=AllReduceStrategy.MNNVL,
                        dtype=torch.bfloat16,
                        mnnvl_own_workspace=MnnvlOwnWorkspace(
                            f"test_own_{early_state}", limit, early_state))
        own_ws = own.mnnvl_allreduce._workspace()
        shared_ws = shared.mnnvl_allreduce._workspace()
        assert own_ws["buffer_size_bytes"] == limit
        assert own_ws["uc_buffer"].data_ptr(
        ) != shared_ws["uc_buffer"].data_ptr()
        assert own.mnnvl_allreduce._transport(big, hidden_size)[0]
        assert not own.mnnvl_allreduce._transport(big + 1, hidden_size)[0]
        assert mnnvl_allreduce_transport(big, hidden_size, tensor_parallel_size,
                                         torch.bfloat16, limit)[0]

        norm_weight = (1 + 0.1 * torch.randn(
            hidden_size,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(0))).to(
                torch.bfloat16)
        eps = 1e-6
        params = AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM,
                                 norm_weight=norm_weight,
                                 eps=eps)
        for step, m in enumerate((1, big, 16, big, 1)):

            def inputs(r):
                gen = torch.Generator(device="cuda").manual_seed(100 * step + r)
                return (
                    torch.randn(m, hidden_size, device="cuda", generator=gen) *
                    0.5).to(torch.bfloat16)

            x = inputs(rank)
            total = sum(inputs(r).float() for r in range(tensor_parallel_size))
            ref = rms_norm(
                total.to(torch.bfloat16).float(), norm_weight.float(),
                eps).to(torch.bfloat16)
            out_own = own(x, all_reduce_params=params)
            out_shared = shared(x, all_reduce_params=params)
            if early_state:
                # Early state only moves the state read and the re-arm: the same bits.
                plain = AllReduce(mapping=mapping,
                                  strategy=AllReduceStrategy.MNNVL,
                                  dtype=torch.bfloat16,
                                  mnnvl_own_workspace=MnnvlOwnWorkspace(
                                      "test_own_False", limit))
                assert torch.equal(out_own, plain(x, all_reduce_params=params))
            torch.testing.assert_close(out_own, ref, rtol=0.05, atol=0.15)
            if shared.mnnvl_allreduce._transport(m, hidden_size)[0]:
                # Both one-shot: the same kernel on other buffers.
                assert torch.equal(out_own, out_shared)
            else:
                torch.testing.assert_close(out_own,
                                           out_shared,
                                           rtol=0.02,
                                           atol=0.05)
            xq, xs = own(x,
                         all_reduce_params=AllReduceParams(
                             fusion_op=AllReduceFusionOp.RMS_NORM_QUANT_MXFP8,
                             norm_weight=norm_weight,
                             eps=eps))
            ref_q, _ = low_m.mxfp8_quantize_reference(out_own)
            assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
        # A two-shot size falls back to the strategies after MNNVL; the own workspace keeps its
        # size. The fallback may return an NCCL window tensor, which needs a lease scope (the
        # decoder layer's in a model).
        x = (torch.randn(big + 1, hidden_size, device="cuda") * 0.5).to(
            torch.bfloat16)
        with discard_nccl_window_tensor_outputs(x):
            torch.testing.assert_close(own(x), shared(x), rtol=0.02, atol=0.05)
        assert own.mnnvl_allreduce._workspace()["buffer_size_bytes"] == limit
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("early_state", [False, True],
                         ids=["late_state", "early_state"])
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_own_workspace(mpi_pool_executor, early_state):
    """The Kimi K3 routed latent all-reduce's own workspace, one-shot limit and early state."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = mpi_pool_executor.map(run_own_workspace_single_rank,
                                    *zip(*[(world, 3584, early_state)] * world))
    for r in results:
        assert r is True


@torch.inference_mode()
def run_fp8_payload_single_rank(tensor_parallel_size, input, residual,
                                norm_weight, eps, reference_residual):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        input, residual = input.cuda(), residual.cuda()
        norm_weight = norm_weight.cuda()
        reference_residual = reference_residual.cuda()
        os.environ["TLLM_TEST_MNNVL"] = "1"
        MPI.COMM_WORLD.barrier()
        mapping = Mapping(world_size=tensor_parallel_size,
                          tp_size=tensor_parallel_size,
                          rank=rank)

        def params(fp8_payload):
            return AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=eps,
                fp8_payload=fp8_payload,
            )

        allreduce = MNNVLAllReduce(mapping, torch.bfloat16)
        _, res_exact = allreduce(input, all_reduce_params=params(False))
        norm_fp8, res_fp8 = allreduce(input, all_reduce_params=params(True))
        torch.cuda.synchronize()
        # Two E4M3 roundings (3 mantissa bits) on a per-256-element scale.
        bound = 0.15 * reference_residual.float().abs().max().item()
        err = (res_fp8.float() - reference_residual.float()).abs().max().item()
        assert err <= bound, (err, bound)
        assert not torch.equal(res_fp8,
                               res_exact), "the E4M3 payload was not used"
        assert torch.isfinite(norm_fp8.float()).all()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # The pool worker is reused: later tests must not see the MNNVL override.
        os.environ.pop("TLLM_TEST_MNNVL", None)
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_fp8_payload_residual_rms_norm(mpi_pool_executor):
    """fp8_payload: the two-shot residual RMSNorm call exchanges E4M3, close to the exact sum."""
    torch.manual_seed(7)
    tensor_parallel_size = mpi_pool_executor.num_workers
    seq_len, hidden_size, eps = 256, 4096, 1e-6  # two-shot (> 1 MiB message)
    x = torch.randn((tensor_parallel_size, seq_len, hidden_size),
                    dtype=torch.bfloat16)
    residual = torch.randn((seq_len, hidden_size), dtype=torch.bfloat16)
    norm_weight = torch.randn((hidden_size, ), dtype=torch.bfloat16)
    reference_residual = (torch.sum(x.float(), dim=0) + residual.float()).to(
        torch.bfloat16)
    results = mpi_pool_executor.map(
        run_fp8_payload_single_rank,
        *zip(*[(tensor_parallel_size, x[i], residual, norm_weight, eps,
                reference_residual) for i in range(tensor_parallel_size)]),
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
@pytest.mark.parametrize("seq_len", SEQ_LEN_CASES, ids=_seq_len_id)
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
    )
