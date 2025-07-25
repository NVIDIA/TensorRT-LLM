from operator import getitem
from typing import List, Optional

import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, Ignored,
                                             KeywordArg, Match,
                                             MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

import tensorrt_llm

from ...distributed import AllReduceFusionOp, AllReduceStrategy

aten = torch.ops.aten
from tensorrt_llm.mapping import Mapping


def register_ar_residual_norm(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )
    residual_key = KeywordArg("residual")
    trtllm_allreduce_default = CallFunction(
        torch.ops.trtllm.allreduce.default, KeywordArg("input"), None, None,
        None, None, KeywordArg("workspace"), mapping.tp_group,
        KeywordArg("strategy"), int(AllReduceFusionOp.NONE), Ignored(),
        KeywordArg("trigger_completion_at_end"))
    getitem_x = CallFunction(getitem, trtllm_allreduce_default, 0)
    add_Tensor = CallFunction(aten.add.Tensor,
                              getitem_x,
                              residual_key,
                              _users=MULTIPLE)
    _torch_rms_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    ar_residual_norm_pattern = MultiOutputPattern(
        [_torch_rms_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        workspace: torch.LongTensor,
        residual: torch.Tensor,
        strategy: int,
        norm_weight: torch.nn.Parameter,
        eps: float,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        workspace: torch.LongTensor,
        residual: torch.Tensor,
        strategy: int,
        norm_weight: torch.nn.Parameter,
        eps: float,
        trigger_completion_at_end: bool,
    ):
        all_reduce_output = torch.ops.trtllm.allreduce(
            input, residual, norm_weight, None, None, workspace,
            mapping.tp_group, int(strategy),
            int(AllReduceFusionOp.RESIDUAL_RMS_NORM), float(eps),
            trigger_completion_at_end)
        return all_reduce_output[0], all_reduce_output[1]

    def extra_check(match: Match) -> bool:
        # Residual should be a tensor
        residual_node = match.ctx.pattern_to_node[residual_key]
        if not isinstance(residual_node, torch.fx.graph.Node):
            return False
        getitem_node = match.ctx.pattern_to_node[getitem_x]
        if not isinstance(getitem_node, torch.fx.graph.Node):
            return False

        getitem_node_shape = getitem_node.meta["tensor_meta"].shape
        residual_node_shape = residual_node.meta["tensor_meta"].shape

        if getitem_node_shape != residual_node_shape:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=ar_residual_norm_pattern,
        extra_check=extra_check,
    )


def check_f16_bf16_input(match, input_node) -> bool:
    input = match.ctx.pattern_to_node[input_node]
    if not isinstance(input, torch.fx.graph.Node):
        return False
    dtype = input.meta["tensor_meta"].dtype
    if dtype != torch.float16 and dtype != torch.bfloat16:
        return False
    return True


def check_non_ub_strategy(match, strategy_node) -> bool:
    strategy = match.ctx.pattern_to_node[strategy_node]
    if not isinstance(strategy, int):
        return False
    if strategy == int(AllReduceStrategy.UB):
        return False
    return True


def register_ar_residual_norm_out_fp8_quant(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    input_node = KeywordArg("input")
    strategy_node = KeywordArg("strategy")
    allreduce_default = CallFunction(torch.ops.trtllm.allreduce.default,
                                     input_node,
                                     KeywordArg("residual"),
                                     KeywordArg("gamma"),
                                     None,
                                     None,
                                     KeywordArg("workspace"),
                                     mapping.tp_group,
                                     strategy_node,
                                     int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
                                     KeywordArg("eps"),
                                     KeywordArg("trigger_completion_at_end"),
                                     _users=2)
    getitem_0 = CallFunction(getitem, allreduce_default, 0, _users=2)
    getitem_1 = CallFunction(getitem, allreduce_default, 1)
    static_quantize_e4m3_per_tensor_default = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        getitem_0,
        KeywordArg("scale"),
        _users=2)
    getitem_2 = CallFunction(getitem,
                             static_quantize_e4m3_per_tensor_default,
                             0,
                             _users=2)
    getitem_3 = CallFunction(getitem, static_quantize_e4m3_per_tensor_default,
                             1)
    pattern = MultiOutputPattern([getitem_0, getitem_1, getitem_2, getitem_3
                                  ])  # norm_out, residual_out, quant_out, scale

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        allreduce = torch.ops.trtllm.allreduce(
            input, residual, gamma, scale, None, workspace, mapping.tp_group,
            int(strategy),
            int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8), float(eps),
            trigger_completion_at_end)
        return allreduce[0], allreduce[2], allreduce[1], scale

    def extra_check(match: Match) -> bool:
        return check_f16_bf16_input(
            match, input_node) and check_non_ub_strategy(match, strategy_node)

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=pattern,
        extra_check=extra_check,
    )


def register_ar_residual_norm_fp8_quant(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    input_node = KeywordArg("input")
    strategy_node = KeywordArg("strategy")
    allreduce_default = CallFunction(torch.ops.trtllm.allreduce.default,
                                     input_node,
                                     KeywordArg("residual"),
                                     KeywordArg("gamma"),
                                     None,
                                     None,
                                     KeywordArg("workspace"),
                                     mapping.tp_group,
                                     strategy_node,
                                     int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
                                     KeywordArg("eps"),
                                     KeywordArg("trigger_completion_at_end"),
                                     _users=2)
    getitem_0 = CallFunction(getitem, allreduce_default, 0)
    getitem_1 = CallFunction(getitem, allreduce_default, 1)
    static_quantize_e4m3_per_tensor_default = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        getitem_0,
        KeywordArg("scale"),
        _users=2)
    getitem_2 = CallFunction(getitem,
                             static_quantize_e4m3_per_tensor_default,
                             0,
                             _users=2)
    getitem_3 = CallFunction(getitem, static_quantize_e4m3_per_tensor_default,
                             1)
    pattern = MultiOutputPattern([getitem_1, getitem_2,
                                  getitem_3])  # residual_out, quant_out, scale

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        allreduce = torch.ops.trtllm.allreduce(
            input, residual, gamma, scale, None, workspace, mapping.tp_group,
            int(strategy), int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8),
            float(eps), trigger_completion_at_end)
        return allreduce[1], allreduce[0], scale

    def extra_check(match: Match) -> bool:
        return check_f16_bf16_input(
            match, input_node) and check_non_ub_strategy(match, strategy_node)

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=pattern,
        extra_check=extra_check,
    )


def register_ar_residual_norm_out_fp4_quant(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    input_node = KeywordArg("input")
    strategy_node = KeywordArg("strategy")
    allreduce_default = CallFunction(torch.ops.trtllm.allreduce.default,
                                     input_node,
                                     KeywordArg("residual"),
                                     KeywordArg("gamma"),
                                     None,
                                     None,
                                     KeywordArg("workspace"),
                                     mapping.tp_group,
                                     strategy_node,
                                     int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
                                     KeywordArg("eps"),
                                     KeywordArg("trigger_completion_at_end"),
                                     _users=2)
    getitem_0 = CallFunction(getitem, allreduce_default, 0, _users=2)
    getitem_1 = CallFunction(getitem, allreduce_default, 1)
    fp4_quant_default = CallFunction(torch.ops.trtllm.fp4_quantize.default,
                                     getitem_0,
                                     KeywordArg("scale"),
                                     16,
                                     _users=2)
    getitem_2 = CallFunction(getitem, fp4_quant_default, 0, _users=2)
    getitem_3 = CallFunction(getitem, fp4_quant_default, 1)
    pattern = MultiOutputPattern([getitem_0, getitem_1, getitem_2, getitem_3])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        allreduce = torch.ops.trtllm.allreduce(
            input, residual, gamma, scale, None, workspace, mapping.tp_group,
            int(strategy),
            int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4),
            float(eps), trigger_completion_at_end)
        return allreduce[0], allreduce[3], allreduce[1], allreduce[2]

    def extra_check(match: Match) -> bool:
        return check_f16_bf16_input(
            match, input_node) and check_non_ub_strategy(match, strategy_node)

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=pattern,
        extra_check=extra_check,
    )


def register_ar_residual_norm_fp4_quant(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    input_node = KeywordArg("input")
    strategy_node = KeywordArg("strategy")
    allreduce_default = CallFunction(torch.ops.trtllm.allreduce.default,
                                     input_node,
                                     KeywordArg("residual"),
                                     KeywordArg("gamma"),
                                     None,
                                     None,
                                     KeywordArg("workspace"),
                                     mapping.tp_group,
                                     strategy_node,
                                     int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
                                     KeywordArg("eps"),
                                     KeywordArg("trigger_completion_at_end"),
                                     _users=2)
    getitem_0 = CallFunction(getitem, allreduce_default, 0)
    getitem_1 = CallFunction(getitem, allreduce_default, 1)
    fp4_quant_default = CallFunction(torch.ops.trtllm.fp4_quantize.default,
                                     getitem_0,
                                     KeywordArg("scale"),
                                     16,
                                     _users=2)
    getitem_2 = CallFunction(getitem, fp4_quant_default, 0, _users=2)
    getitem_3 = CallFunction(getitem, fp4_quant_default, 1)
    pattern = MultiOutputPattern([getitem_1, getitem_2, getitem_3])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        workspace: torch.LongTensor,
        strategy: int,
        eps: float,
        scale: torch.Tensor,
        trigger_completion_at_end: bool,
    ):
        allreduce = torch.ops.trtllm.allreduce(
            input, residual, gamma, scale, None, workspace, mapping.tp_group,
            int(strategy), int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4),
            float(eps), trigger_completion_at_end)
        return allreduce[2], allreduce[0], allreduce[1]

    def extra_check(match: Match) -> bool:
        return check_f16_bf16_input(
            match, input_node) and check_non_ub_strategy(match, strategy_node)

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=pattern,
        extra_check=extra_check,
    )


def register_ub_patterns(custom_passes: List[PatternMatcherPass]):
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    def register_convert_supported_ar_to_ub(custom_pass: PatternMatcherPass):
        strategy = int(AllReduceStrategy.AUTO)
        input_node = KeywordArg('input')
        fusion = KeywordArg('fusion_op')
        trtllm_allreduce_default = CallFunction(
            torch.ops.trtllm.allreduce.default, input_node,
            KeywordArg('residual_in'), KeywordArg('gamma'), KeywordArg('scale'),
            None, Ignored(), mapping.tp_group, strategy, fusion,
            KeywordArg('eps'), Ignored())

        def empty_convert_supported_ar_to_ub(
            input: torch.Tensor,
            residual_in: torch.Tensor,
            gamma: torch.Tensor,
            scale: Optional[torch.Tensor],
            fusion_op: int,
            eps: float,
        ):
            return

        def target_convert_supported_ar_to_ub(
            input: torch.Tensor,
            residual_in: torch.Tensor,
            gamma: torch.Tensor,
            scale: Optional[torch.Tensor],
            fusion_op: int,
            eps: float,
        ):
            input = torch.ops.trtllm.copy_to_userbuffers(input)
            all_reduce_output = torch.ops.trtllm.allreduce(
                input, residual_in, gamma, scale, None, None, mapping.tp_group,
                int(AllReduceStrategy.UB), fusion_op, eps, False)
            finalize_output = torch.ops.trtllm.userbuffers_allreduce_finalize(
                all_reduce_output[-1], False)
            all_reduce_output[-1] = finalize_output
            return all_reduce_output

        def extra_check_convert_supported_ar_to_ub(match: Match) -> bool:
            if not check_f16_bf16_input(match, input_node):
                return False

            fusion_value = match.ctx.pattern_to_node[fusion]
            if not isinstance(fusion_value, int):
                return False
            if fusion_value != int(
                    AllReduceFusionOp.RESIDUAL_RMS_NORM
            ) and fusion_value != int(
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
            ) and fusion_value != int(
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4):
                return False

            return True

        register_replacement(
            empty_convert_supported_ar_to_ub,
            target_convert_supported_ar_to_ub,
            [],
            fwd_only,
            custom_pass,
            search_fn_pattern=trtllm_allreduce_default,
            extra_check=extra_check_convert_supported_ar_to_ub,
        )

    def register_ub_prologue_patterns(custom_pass: PatternMatcherPass):

        def register_scaled_mm_prologue(custom_pass: PatternMatcherPass):
            trtllm_cublas_scaled_mm_default = CallFunction(
                torch.ops.trtllm.cublas_scaled_mm.default, KeywordArg('mm0_a'),
                KeywordArg('mm0_b'), KeywordArg('mm0_a_scale'),
                KeywordArg('mm0_b_scale'), KeywordArg('mm0_bias'),
                KeywordArg('mm_dtype'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   trtllm_cublas_scaled_mm_default)

            def empty_scaled_mm_prologue_pattern(
                mm0_a: torch.Tensor,
                mm0_b: torch.Tensor,
                mm0_a_scale: torch.Tensor,
                mm0_b_scale: torch.Tensor,
                mm0_bias: Optional[torch.Tensor],
                mm_dtype: torch.dtype,
            ):
                return

            def target_scaled_mm_prologue_pattern(
                mm0_a: torch.Tensor,
                mm0_b: torch.Tensor,
                mm0_a_scale: torch.Tensor,
                mm0_b_scale: torch.Tensor,
                mm0_bias: Optional[torch.Tensor],
                mm_dtype: torch.dtype,
            ):
                scaled_mm_output = torch.ops.trtllm.cublas_scaled_mm(
                    mm0_a, mm0_b, mm0_a_scale, mm0_b_scale, mm0_bias, mm_dtype,
                    True)
                return scaled_mm_output

            # No extra check needed as the output dtype of scaled_mm has been verified when
            # ub_copy is inserted.
            register_replacement(
                empty_scaled_mm_prologue_pattern,
                target_scaled_mm_prologue_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_copy,
            )

        def register_nvfp4_gemm_prologue(custom_pass: PatternMatcherPass):
            trtllm_nvfp4_gemm_default = CallFunction(
                torch.ops.trtllm.nvfp4_gemm.default, KeywordArg('act_fp4'),
                KeywordArg('weight'), KeywordArg('act_sf'),
                KeywordArg('weight_scale'), KeywordArg('alpha'),
                KeywordArg('output_dtype'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   trtllm_nvfp4_gemm_default)

            def empty_nvfp4_gemm_prologue_pattern(
                act_fp4: torch.Tensor,
                weight: torch.Tensor,
                act_sf: torch.Tensor,
                weight_scale: torch.Tensor,
                alpha: torch.Tensor,
                output_dtype: torch.dtype,
            ):
                return

            def target_nvfp4_gemm_prologue_pattern(
                act_fp4: torch.Tensor,
                weight: torch.Tensor,
                act_sf: torch.Tensor,
                weight_scale: torch.Tensor,
                alpha: torch.Tensor,
                output_dtype: torch.dtype,
            ):
                nvfp4_gemm_output = torch.ops.trtllm.nvfp4_gemm(
                    act_fp4, weight, act_sf, weight_scale, alpha, output_dtype,
                    True)
                return nvfp4_gemm_output

            # No extra check needed as the output dtype of nvfp4_gemm has been verified when
            # ub_copy is inserted.
            register_replacement(
                empty_nvfp4_gemm_prologue_pattern,
                target_nvfp4_gemm_prologue_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_copy,
            )

        def register_mm_prologue(custom_pass: PatternMatcherPass):
            aten_mm_default = CallFunction(aten.mm.default, KeywordArg('mm0_a'),
                                           KeywordArg('mm0_b'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   aten_mm_default)

            def empty_mm_prologue_pattern(
                mm0_a: torch.Tensor,
                mm0_b: torch.Tensor,
            ):
                return

            def target_mm_prologue_pattern(
                mm0_a: torch.Tensor,
                mm0_b: torch.Tensor,
            ):
                mm_output = torch.ops.trtllm.matmul_to_ub(mm0_a, mm0_b)
                return mm_output

            # No extra check needed as the output dtype of mm has been verified when
            # ub_copy is inserted.
            register_replacement(
                empty_mm_prologue_pattern,
                target_mm_prologue_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_copy,
            )

        def register_add_prologue(custom_pass: PatternMatcherPass):
            aten_add_default = CallFunction(aten.add.Tensor,
                                            KeywordArg('add_a'),
                                            KeywordArg('add_b'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   aten_add_default)

            def empty_add_prologue_pattern(
                add_a: torch.Tensor,
                add_b: torch.Tensor,
            ):
                return

            def target_add_prologue_pattern(
                add_a: torch.Tensor,
                add_b: torch.Tensor,
            ):
                add_output = torch.ops.trtllm.add_to_ub(add_a, add_b)
                return add_output

            # No extra check needed as the output dtype of add has been verified when
            # ub_copy is inserted.
            register_replacement(
                empty_add_prologue_pattern,
                target_add_prologue_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_copy,
            )

        register_scaled_mm_prologue(custom_pass)
        register_nvfp4_gemm_prologue(custom_pass)
        register_mm_prologue(custom_pass)
        register_add_prologue(custom_pass)

    def register_ub_finalize_patterns(custom_pass: PatternMatcherPass):
        trtllm_userbuffers_allreduce_finalize_default = CallFunction(
            torch.ops.trtllm.userbuffers_allreduce_finalize.default,
            KeywordArg("sharded_residual"), False)
        trtllm_allreduce_default = CallFunction(
            torch.ops.trtllm.allreduce.default, KeywordArg("input"),
            trtllm_userbuffers_allreduce_finalize_default, KeywordArg("gamma"),
            KeywordArg("scale"), Ignored(), Ignored(), mapping.tp_group,
            int(AllReduceStrategy.UB), KeywordArg("fusion_op"),
            KeywordArg("eps"), Ignored())

        def empty_finalize_pattern(
            input: torch.Tensor,
            sharded_residual: torch.Tensor,
            gamma: torch.Tensor,
            scale: Optional[torch.Tensor],
            fusion_op: int,
            eps: float,
        ):
            return

        def target_finalize_pattern(
            input: torch.Tensor,
            sharded_residual: torch.Tensor,
            gamma: torch.Tensor,
            scale: Optional[torch.Tensor],
            fusion_op: int,
            eps: float,
        ):
            all_reduce_output = torch.ops.trtllm.allreduce(
                input, sharded_residual,
                gamma, scale, None, None, mapping.tp_group,
                int(AllReduceStrategy.UB), fusion_op, eps, False)
            return all_reduce_output

        register_replacement(
            empty_finalize_pattern,
            target_finalize_pattern,
            [],
            fwd_only,
            custom_pass,
            search_fn_pattern=trtllm_allreduce_default,
        )

    custom_passes.append(PatternMatcherPass())
    register_convert_supported_ar_to_ub(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_ub_prologue_patterns(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_ub_finalize_patterns(custom_passes[-1])


def register_ar_fusions(custom_passes: List[PatternMatcherPass],
                        enable_ub: bool):
    register_ar_residual_norm(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_ar_residual_norm_fp8_quant(custom_passes[-1])
    register_ar_residual_norm_fp4_quant(custom_passes[-1])
    # AR-Residual-Norm-Out-Quant-X is not supported by Userbuffers kernel.
    if not enable_ub:
        register_ar_residual_norm_out_fp8_quant(custom_passes[-1])
        register_ar_residual_norm_out_fp4_quant(custom_passes[-1])

    if enable_ub:
        register_ub_patterns(custom_passes)
