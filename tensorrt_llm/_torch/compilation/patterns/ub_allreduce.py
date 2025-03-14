from operator import getitem
from typing import Optional

import torch
from torch._inductor.pattern_matcher import (CallFunction, Ignored, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceStrategy

aten = torch.ops.aten
from tensorrt_llm.mapping import Mapping


def register_ub_allreduce(custom_pass: PatternMatcherPass):
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    # Only match AUTO strategy all-reduce
    strategy = int(AllReduceStrategy.AUTO)
    fusion = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

    mm_dtype = KeywordArg('mm_dtype')
    trtllm_cublas_scaled_mm_default = CallFunction(
        torch.ops.trtllm.cublas_scaled_mm.default, KeywordArg('mm0_a'),
        KeywordArg('mm0_b'), KeywordArg('mm0_a_scale'),
        KeywordArg('mm0_b_scale'), KeywordArg('mm0_bias'), mm_dtype, -1)
    trtllm_allreduce_default = CallFunction(
        torch.ops.trtllm.allreduce.default,
        trtllm_cublas_scaled_mm_default,
        Ignored(), [KeywordArg('residual_in'),
                    KeywordArg('gamma')],
        mapping.tp_group,
        strategy,
        Ignored(),
        fusion,
        KeywordArg('eps'),
        True,
        False,
        False,
        _users=2)
    getitem_0 = CallFunction(getitem, trtllm_allreduce_default, 0)
    tensorrt_llm_static_quantize_e4m3_per_tensor_default = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        getitem_0,
        KeywordArg('scale'),
        _users=2)
    getitem_1 = CallFunction(
        getitem, tensorrt_llm_static_quantize_e4m3_per_tensor_default, 0)
    getitem_2 = CallFunction(
        getitem, tensorrt_llm_static_quantize_e4m3_per_tensor_default, 1)
    aten_t_default = CallFunction(torch.ops.aten.t.default, KeywordArg('mm1_b'))
    trtllm_cublas_scaled_mm_default_1 = CallFunction(
        torch.ops.trtllm.cublas_scaled_mm.default, getitem_1,
        aten_t_default, getitem_2, KeywordArg('mm1_b_scale'),
        KeywordArg('mm1_bias'), mm_dtype, -1)
    getitem_3 = CallFunction(getitem, trtllm_allreduce_default, 1)
    ub_ar_pattern = MultiOutputPattern(
        [trtllm_cublas_scaled_mm_default_1, getitem_3])

    def empty_pattern(
        mm0_a: torch.Tensor,
        mm0_b: torch.Tensor,
        mm0_a_scale: torch.Tensor,
        mm0_b_scale: torch.Tensor,
        mm0_bias: Optional[torch.Tensor],
        mm_dtype: torch.dtype,
        residual_in: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        mm1_b: torch.Tensor,
        mm1_b_scale: torch.Tensor,
        mm1_bias: Optional[torch.Tensor],
    ):
        return

    def target_pattern(
        mm0_a: torch.Tensor,
        mm0_b: torch.Tensor,
        mm0_a_scale: torch.Tensor,
        mm0_b_scale: torch.Tensor,
        mm0_bias: Optional[torch.Tensor],
        mm_dtype: torch.dtype,
        residual_in: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        mm1_b: torch.Tensor,
        mm1_b_scale: torch.Tensor,
        mm1_bias: Optional[torch.Tensor],
    ):
        all_reduce_output = torch.ops.trtllm.ub_scaled_mm_allreduce_quant_scaled_mm_op(
            mm0_a, mm0_b, mm0_a_scale, mm0_b_scale, mm0_bias, mm_dtype,
            residual_in, gamma, mapping.tp_group, eps, scale, mm1_b,
            mm1_b_scale, mm1_bias)
        full_residual = torch.ops.trtllm.userbuffers_allreduce_finalize(
            all_reduce_output[1])
        return all_reduce_output[0], full_residual

    def extra_check(match: Match) -> bool:
        # Userbuffers allreduce only supports BF16/FP16
        if not isinstance(match.ctx.pattern_to_node[mm_dtype], torch.dtype):
            return False
        dt = match.ctx.pattern_to_node[mm_dtype]
        if dt != torch.float16 and dt != torch.bfloat16:
            return False
        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=ub_ar_pattern,
        extra_check=extra_check,
    )


def register_ub_allreduce_finalize(custom_pass: PatternMatcherPass):
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )
    trtllm_userbuffers_allreduce_finalize_default = CallFunction(
        torch.ops.trtllm.userbuffers_allreduce_finalize.default,
        KeywordArg("sharded_residual"))
    trtllm_ub_scaled_mm_allreduce_quant_scaled_mm_op_default = CallFunction(
        torch.ops.trtllm.ub_scaled_mm_allreduce_quant_scaled_mm_op.default,
        KeywordArg("mm0_a"),
        KeywordArg("mm0_b"),
        KeywordArg("mm0_a_scale"),
        KeywordArg("mm0_b_scale"),
        KeywordArg("mm0_bias"),
        KeywordArg("mm_dtype"),
        trtllm_userbuffers_allreduce_finalize_default,
        KeywordArg("gamma"),
        mapping.tp_group,
        KeywordArg("eps"),
        KeywordArg("scale"),
        KeywordArg("mm1_b"),
        KeywordArg("mm1_b_scale"),
        KeywordArg("mm1_bias"),
        _users=2,
    )
    getitem_0 = CallFunction(
        getitem, trtllm_ub_scaled_mm_allreduce_quant_scaled_mm_op_default, 0)
    getitem_1 = CallFunction(
        getitem, trtllm_ub_scaled_mm_allreduce_quant_scaled_mm_op_default, 1)
    ub_ar_finalize_pattern = MultiOutputPattern([getitem_0, getitem_1])

    def empty_pattern(
        mm0_a: torch.Tensor,
        mm0_b: torch.Tensor,
        mm0_a_scale: torch.Tensor,
        mm0_b_scale: torch.Tensor,
        mm0_bias: Optional[torch.Tensor],
        mm_dtype: torch.dtype,
        sharded_residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        mm1_b: torch.Tensor,
        mm1_b_scale: torch.Tensor,
        mm1_bias: Optional[torch.Tensor],
    ):
        return

    def target_pattern(
        mm0_a: torch.Tensor,
        mm0_b: torch.Tensor,
        mm0_a_scale: torch.Tensor,
        mm0_b_scale: torch.Tensor,
        mm0_bias: Optional[torch.Tensor],
        mm_dtype: torch.dtype,
        sharded_residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        mm1_b: torch.Tensor,
        mm1_b_scale: torch.Tensor,
        mm1_bias: Optional[torch.Tensor],
    ):
        all_reduce_output = torch.ops.trtllm.ub_scaled_mm_allreduce_quant_scaled_mm_op(
            mm0_a, mm0_b, mm0_a_scale, mm0_b_scale, mm0_bias, mm_dtype,
            sharded_residual, gamma, mapping.tp_group, eps, scale, mm1_b,
            mm1_b_scale, mm1_bias)
        return all_reduce_output[0], all_reduce_output[1]

    def extra_check(match: Match) -> bool:
        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=ub_ar_finalize_pattern,
        extra_check=extra_check,
    )
