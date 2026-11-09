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
import pickle
import platform
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce, _MnnvlDispatchPlan, get_or_scale_allreduce_mnnvl_workspace)
from tensorrt_llm._torch.nccl_window_tensor_scope import \
    discard_nccl_window_tensor_outputs
from tensorrt_llm.bindings.internal.runtime import (McastGPUBuffer,
                                                    MnnvlTransport,
                                                    MnnvlWorkspace)
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
    tensor_parallel_rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(tensor_parallel_rank)
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
            allreduce = MNNVLAllReduce(
                mapping=mapping,
                dispatch_plan=_MnnvlDispatchPlan(
                    transport=MnnvlTransport.FABRIC),
                dtype=torch.bfloat16,
            )
            input_ = torch.full((num_tokens, 128),
                                tensor_parallel_rank + 1,
                                dtype=torch.bfloat16,
                                device="cuda")
            output = allreduce(input_, AllReduceParams())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = allreduce(input_, AllReduceParams())

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 3))

        workspace = allreduce.allreduce_mnnvl_workspaces[mapping]
        uc_address = workspace["uc_buffer"].data_ptr()
        mc_address = workspace["handle"].get_mc_buffer((1, ), torch.float32,
                                                       0).data_ptr()

        for cycle in range(3):
            allreduce.checkpoint_prepare()
            allreduce.checkpoint_prepare()
            assert not workspace["handle"].is_mapped()
            with pytest.raises(
                    RuntimeError,
                    match="does not have an active MPI communicator"):
                workspace["handle"].checkpoint_restore_complete(True)
            with pytest.raises(RuntimeError, match="handles are not attached"):
                allreduce(input_, AllReduceParams())

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
            allreduce.checkpoint_restore(fresh_comm)
            fresh_comm.Free()
            allreduce.checkpoint_restore(workspace["comm"])
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

        allreduce.checkpoint_prepare()
        abort_comm = MPI.COMM_WORLD.Split(0, tensor_parallel_rank)
        assert workspace["handle"].checkpoint_restore(abort_comm.py2f())
        with pytest.raises(RuntimeError,
                           match="all-reduce protocol reset failed"):
            workspace["handle"].checkpoint_restore_complete(
                tensor_parallel_rank != 0)
        assert not workspace["handle"].is_mapped()
        abort_comm.Free()
        return True
    finally:
        workspace = None
        output = None
        input_ = None
        graph = None
        allreduce = None
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        gc.collect()


def mnnvl_growth_graph_forward(tensor_parallel_size: int,
                               tensor_parallel_rank: int) -> bool:
    """A graph captured on the MNNVL workspace replays correctly after an eager call grew the workspace, a replaced
    workspace that no capture used is not kept, and growth during capture is refused."""
    tensor_parallel_rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(tensor_parallel_rank)
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

            def oversized_input(workspace):
                # Exceed the usable capacity, including any allocation-granularity slack.
                bytes_per_token = 2 * input_.shape[-1] * input_.element_size()
                num_tokens = (
                    workspace["buffer_size_bytes"] // bytes_per_token +
                    tensor_parallel_size)
                return torch.ones((num_tokens, input_.shape[-1]),
                                  dtype=input_.dtype,
                                  device=input_.device)

            big = oversized_input(before)
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
            wider = oversized_input(after)
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

            huge = oversized_input(
                MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping])
            with pytest.raises(RuntimeError, match="during CUDA graph capture"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    allreduce(huge)
        return True
    finally:
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
            MNNVLAllReduce.allreduce_mnnvl_retired_workspaces.pop(mapping, None)
        gc.collect()


def mnnvl_checkpoint_rejects_wrong_membership(world_size: int,
                                              world_rank: int) -> bool:
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
        allreduce = MNNVLAllReduce(
            mapping=mapping,
            dispatch_plan=_MnnvlDispatchPlan(transport=MnnvlTransport.FABRIC),
            dtype=torch.bfloat16)
        allreduce(torch.ones((1, 128), dtype=torch.bfloat16, device="cuda"),
                  AllReduceParams())
        workspace = MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping]
        allreduce.checkpoint_prepare()

        wrong_group = 0 if world_rank in (0, 3) else 1
        wrong_comm = MPI.COMM_WORLD.Split(wrong_group, mapping.tp_rank)
        with pytest.raises(RuntimeError,
                           match="restore communicator validation failed"):
            workspace["handle"].checkpoint_restore(wrong_comm.py2f())
        wrong_comm.Free()

        correct_comm = MPI.COMM_WORLD.Split(mapping.pp_rank, mapping.tp_rank)
        allreduce.checkpoint_restore(correct_comm)
        correct_comm.Free()
        assert workspace["handle"].is_mapped()
        return True
    finally:
        allreduce = None
        if mapping is not None:
            MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        gc.collect()


def mnnvl_rejects_shifted_workspace(
        tensor_parallel_size: int,
        tensor_parallel_rank: int,
        use_qualified_workspace: bool = False) -> bool:
    from tensorrt_llm._torch._experimental.modeling_v2._native import \
        load_family_ops

    kimi_ops = load_family_ops(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl")
    rank = tensorrt_llm.mpi_rank()
    assert rank == tensor_parallel_rank
    torch.cuda.set_device(rank)
    if use_qualified_workspace:
        buffer = MnnvlWorkspace(256, MnnvlTransport.FABRIC, rank,
                                MPI.COMM_WORLD.py2f())
        assert buffer.is_available, buffer.reason
    else:
        buffer = McastGPUBuffer(256, tensor_parallel_size, rank, rank, True,
                                MPI.COMM_WORLD.py2f())
    base = buffer.get_uc_buffer(rank, (64, ), torch.float32, 0)
    multicast = buffer.get_mc_buffer((64, ), torch.float32, 0)
    assert multicast.shape == base.shape
    assert multicast.dtype == base.dtype
    if use_qualified_workspace:
        assert buffer.get_local_buffer().data_ptr() == base.data_ptr()
    for peer in range(tensor_parallel_size):
        assert buffer.get_uc_buffer(peer, (1, ), torch.float32, 0).numel() == 1
    sliced = base[4:]
    shifted_storage = buffer.get_uc_buffer(rank, (60, ), torch.float32, 4)
    assert sliced.storage_offset() == 4
    assert shifted_storage.storage_offset() == 0
    assert sliced.data_ptr() == base.data_ptr() + 16
    assert shifted_storage.data_ptr() == sliced.data_ptr()

    # Invalid input dimensions prevent a kernel launch even if workspace validation is missing.
    input_tensor = torch.zeros((8, ), dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.ones_like(input_tensor)
    flags = torch.zeros((9, ), dtype=torch.uint32, device="cuda")

    def invoke(workspace: torch.Tensor) -> None:
        torch.ops.trtllm.mnnvl_fusion_allreduce(input_tensor, norm_weight, None,
                                                1e-5, workspace, flags, True,
                                                None,
                                                int(AllReduceFusionOp.RMS_NORM))

    def invoke_allgather(workspace: torch.Tensor) -> None:
        torch.ops.trtllm.glm_mnnvl_allgather(input_tensor.float(), workspace,
                                             flags, tensor_parallel_size)

    def invoke_hist_topk(workspace: torch.Tensor) -> None:
        torch.ops.trtllm.glm_dsa_hist_topk(input_tensor, flags, flags,
                                           workspace, flags, 1,
                                           tensor_parallel_size)

    def invoke_hop(workspace: torch.Tensor) -> None:
        rows = input_tensor.view(1, 8)
        torch.ops.trtllm.glm_mnnvl_hop(rows, rows, norm_weight, 1e-5, workspace,
                                       flags, False)

    def invoke_draft(workspace: torch.Tensor) -> None:
        torch.ops.trtllm.glm_draft_sample_glue(logits=input_tensor,
                                               exchange_buffer=workspace,
                                               exchange_flags=flags,
                                               group_size=tensor_parallel_size,
                                               draft_tokens=flags,
                                               draft_step=0,
                                               embed_table=None,
                                               hidden=None,
                                               enorm_weight=None,
                                               hnorm_weight=None,
                                               eps=1e-5,
                                               col_lo=0,
                                               col_hi=0,
                                               glue_out=None,
                                               positions_in=None,
                                               positions_out=None,
                                               gather_ids=None,
                                               seq_lens=None,
                                               fill_seq_lens=False,
                                               kv_lens=None,
                                               kv_update=0,
                                               rewind_accepted=None,
                                               draft_len=0,
                                               block_offsets=None,
                                               dsa_layout=[],
                                               req_idx_per_token=None,
                                               slot_mapping_fp8=None,
                                               slot_mapping_scale=None,
                                               gen_kv_indptr=None,
                                               gen_cached_token_indptr=None,
                                               accepted_tokens=None,
                                               batch_indices=None,
                                               num_accepted=None,
                                               next_new_tokens=None)

    def invoke_kimi_fusion(workspace: torch.Tensor) -> None:
        kimi_ops.k3_mnnvl_fusion_allreduce(input_tensor, norm_weight, None,
                                           1e-5, workspace, flags, True, None,
                                           int(AllReduceFusionOp.RMS_NORM))

    def invoke_kimi_attn_res(workspace: torch.Tensor) -> None:
        kimi_ops.mnnvl_allreduce_attn_res(input_tensor, None, input_tensor,
                                          norm_weight, norm_weight, norm_weight,
                                          1e-5, 1e-5, workspace, flags)

    def invoke_kimi_allgather_split(workspace: torch.Tensor) -> None:
        kimi_ops.mnnvl_allgather_split(input_tensor, 0, tensor_parallel_size,
                                       workspace, flags)

    # Each valid base reaches an independent argument check before any kernel launch.
    for call, argument_error in (
        (invoke, "input must have at least 2 dimensions"),
        (invoke_allgather, "input must be BF16"),
        (invoke_hist_topk, "logits must be a row-major FP32 matrix"),
        (invoke_hop, "comm_buffer must be the"),
        (invoke_draft, r"logits must be \[rows, vocab_shard\]"),
        (invoke_kimi_fusion, "input must have at least 2 dimensions"),
        (invoke_kimi_attn_res, r"input must be \[num_tokens, hidden\]"),
        (invoke_kimi_allgather_split, "input must be a contiguous"),
    ):
        for workspace in (base, base.view(8, 8)):
            with pytest.raises(RuntimeError, match=argument_error):
                call(workspace)
        for workspace in (sliced, shifted_storage):
            with pytest.raises(RuntimeError,
                               match="local unicast allocation base"):
                call(workspace)
    return True


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
    assert all(results)


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


@pytest.mark.skipif(
    platform.machine().lower() != "aarch64" or torch.cuda.device_count() < 2
    or not MnnvlMemory.supports_mnnvl(),
    reason="requires at least two GB200 GPUs with fabric-backed MNNVL",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("use_qualified_workspace", [False, True],
                         ids=["raw-buffer", "qualified-workspace"])
def test_mnnvl_rejects_shifted_workspace(mpi_pool_executor,
                                         use_qualified_workspace) -> None:
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        mnnvl_rejects_shifted_workspace,
        [tensor_parallel_size] * tensor_parallel_size,
        range(tensor_parallel_size),
        [use_qualified_workspace] * tensor_parallel_size,
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


@torch.inference_mode()
def run_fp8_payload_single_rank(tensor_parallel_size, input, residual,
                                norm_weight, eps, reference_residual):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        input, residual = input.cuda(), residual.cuda()
        norm_weight = norm_weight.cuda()
        reference_residual = reference_residual.cuda()

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


@torch.inference_mode()
def run_early_pdl_release_single_rank(tensor_parallel_size, input, residual,
                                      norm_weight):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    previous_mnnvl = os.environ.get("TLLM_TEST_MNNVL")
    try:
        input, residual = input.cuda(), residual.cuda()
        norm_weight = norm_weight.cuda()
        os.environ["TLLM_TEST_MNNVL"] = "1"
        MPI.COMM_WORLD.barrier()
        mapping = Mapping(world_size=tensor_parallel_size,
                          tp_size=tensor_parallel_size,
                          rank=rank)
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            mapping,
            input.dtype,
            buffer_size_bytes=MNNVLAllReduce.get_required_workspace_size(
                *input.shape, tensor_parallel_size, input.dtype))
        comm = (workspace["uc_buffer"].view(input.dtype).view(3, -1),
                workspace["buffer_flags"])

        def chain(trigger):
            # The norm call is a PDL dependent of the sum call and reads its output.
            total, = torch.ops.trtllm.mnnvl_fusion_allreduce(
                input,
                None,
                None,
                None,
                *comm,
                False,
                None,
                int(AllReduceFusionOp.NONE),
                trigger_completion_at_end=trigger)
            return (total, *torch.ops.trtllm.mnnvl_fusion_allreduce(
                total,
                norm_weight,
                residual,
                1e-6,
                *comm,
                True,
                None,
                int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
                trigger_completion_at_end=trigger))

        for early, default in zip(chain(False), chain(True)):
            assert torch.equal(early, default)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # The pool worker is reused: later tests see the setting they had before.
        if previous_mnnvl is None:
            os.environ.pop("TLLM_TEST_MNNVL", None)
        else:
            os.environ["TLLM_TEST_MNNVL"] = previous_mnnvl
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_oneshot_early_pdl_release(mpi_pool_executor):
    """trigger_completion_at_end=False: one-shot and PDL dependent outputs match the default bit for bit."""
    torch.manual_seed(11)
    tensor_parallel_size = mpi_pool_executor.num_workers
    seq_len, hidden_size = 32, 2560  # one-shot (<= 1 MiB message)
    x = torch.randn((tensor_parallel_size, seq_len, hidden_size),
                    dtype=torch.bfloat16)
    residual = torch.randn((seq_len, hidden_size), dtype=torch.bfloat16)
    norm_weight = torch.randn((hidden_size, ), dtype=torch.bfloat16)
    results = mpi_pool_executor.map(
        run_early_pdl_release_single_rank,
        *zip(*[(tensor_parallel_size, x[i], residual, norm_weight)
               for i in range(tensor_parallel_size)]),
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
