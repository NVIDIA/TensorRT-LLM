from operator import getitem
from typing import List, Optional

import torch
from torch._inductor.pattern_matcher import (CallFunction, Ignored, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

import tensorrt_llm

from ...distributed import AllReduceFusionOp, AllReduceStrategy

aten = torch.ops.aten
from tensorrt_llm.mapping import Mapping


def register_ub_patterns(custom_passes: List[PatternMatcherPass]):
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )

    def register_ub_allreduce_quantize_fusion(custom_pass: PatternMatcherPass):
        strategy = int(AllReduceStrategy.AUTO)
        fusion = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

        def register_fp8_quant_pattern(custom_pass: PatternMatcherPass):
            input_node = KeywordArg('input')
            trtllm_allreduce_default = CallFunction(
                torch.ops.trtllm.allreduce.default,
                input_node,
                KeywordArg('residual_in'),
                KeywordArg('gamma'),
                None,
                None,
                Ignored(),
                mapping.tp_group,
                strategy,
                fusion,
                KeywordArg('eps'),
                Ignored(),
                _users=2)
            allreduce_output = CallFunction(getitem, trtllm_allreduce_default,
                                            0)
            residual_out = CallFunction(getitem, trtllm_allreduce_default, 1)
            tensorrt_llm_static_quantize_e4m3_per_tensor_default = CallFunction(
                torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
                allreduce_output,
                KeywordArg('scale'),
                _users=2)
            quant_output = CallFunction(
                getitem, tensorrt_llm_static_quantize_e4m3_per_tensor_default,
                0)
            scale_out = CallFunction(
                getitem, tensorrt_llm_static_quantize_e4m3_per_tensor_default,
                1)
            fp8_quant_pattern = MultiOutputPattern(
                [quant_output, scale_out, residual_out])

            def empty_fp8_quant_pattern(
                input: torch.Tensor,
                residual_in: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
                scale: torch.Tensor,
            ):
                return

            def target_fp8_quant_pattern(
                input: torch.Tensor,
                residual_in: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
                scale: torch.Tensor,
            ):
                input = torch.ops.trtllm.copy_to_userbuffers(input)
                all_reduce_output = torch.ops.trtllm.allreduce(
                    input, residual_in, gamma, scale, None, None,
                    mapping.tp_group, int(AllReduceStrategy.UB),
                    int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8), eps,
                    True)
                finalize_output = torch.ops.trtllm.userbuffers_allreduce_finalize(
                    all_reduce_output[1], False)
                return all_reduce_output[0], scale, finalize_output

            def extra_check_fp8_quant_pattern(match: Match) -> bool:
                input = match.ctx.pattern_to_node[input_node]
                if not isinstance(input, torch.fx.graph.Node):
                    return False
                dtype = input.meta["tensor_meta"].dtype
                # UB only supports FP16/BF16 input
                if dtype != torch.float16 and dtype != torch.bfloat16:
                    return False
                return True

            register_replacement(
                empty_fp8_quant_pattern,
                target_fp8_quant_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=fp8_quant_pattern,
                extra_check=extra_check_fp8_quant_pattern,
            )

        def register_fp4_quant_pattern(custom_pass: PatternMatcherPass):
            input_node = KeywordArg('input')
            trtllm_allreduce_default = CallFunction(
                torch.ops.trtllm.allreduce.default,
                input_node,
                KeywordArg('residual_in'),
                KeywordArg('gamma'),
                None,
                Ignored(),
                Ignored(),
                mapping.tp_group,
                strategy,
                fusion,
                KeywordArg('eps'),
                Ignored(),
                _users=2)
            allreduce_output = CallFunction(getitem, trtllm_allreduce_default,
                                            0)
            residual_out = CallFunction(getitem, trtllm_allreduce_default, 1)
            tensorrt_llm_fp4_quantize_default = CallFunction(
                torch.ops.trtllm.fp4_quantize.default,
                allreduce_output,
                KeywordArg('scale'),
                16,
                _users=2)
            quant_output = CallFunction(getitem,
                                        tensorrt_llm_fp4_quantize_default, 0)
            scale_out = CallFunction(getitem, tensorrt_llm_fp4_quantize_default,
                                     1)
            fp4_quant_pattern = MultiOutputPattern(
                [quant_output, scale_out, residual_out])

            def empty_fp4_quant_pattern(
                input: torch.Tensor,
                residual_in: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
                scale: torch.Tensor,
            ):
                return

            def target_fp4_quant_pattern(
                input: torch.Tensor,
                residual_in: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
                scale: torch.Tensor,
            ):
                input = torch.ops.trtllm.copy_to_userbuffers(input)
                all_reduce_output = torch.ops.trtllm.allreduce(
                    input, residual_in, gamma, scale, None, None,
                    mapping.tp_group, int(AllReduceStrategy.UB),
                    int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4), eps,
                    True)
                finalize_output = torch.ops.trtllm.userbuffers_allreduce_finalize(
                    all_reduce_output[-1], False)
                return all_reduce_output[0], all_reduce_output[
                    1], finalize_output

            def extra_check_fp4_quant_pattern(match: Match) -> bool:
                input = match.ctx.pattern_to_node[input_node]
                if not isinstance(input, torch.fx.graph.Node):
                    return False
                dtype = input.meta["tensor_meta"].dtype
                # UB only supports FP16/BF16 input
                if dtype != torch.float16 and dtype != torch.bfloat16:
                    return False
                return True

            register_replacement(
                empty_fp4_quant_pattern,
                target_fp4_quant_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=fp4_quant_pattern,
                extra_check=extra_check_fp4_quant_pattern,
            )

        register_fp8_quant_pattern(custom_pass)
        register_fp4_quant_pattern(custom_pass)

    def register_convert_supported_ar_to_ub(custom_pass: PatternMatcherPass):
        strategy = int(AllReduceStrategy.AUTO)
        # TODO: Also handle scale once the allreduce interface does not contain
        # dynamic number of tensors.
        input_node = KeywordArg('input')
        fusion = KeywordArg('fusion_op')
        trtllm_allreduce_default = CallFunction(
            torch.ops.trtllm.allreduce.default, input_node,
            KeywordArg('residual_in'), KeywordArg('gamma'), KeywordArg('scale'),
            None, Ignored(), mapping.tp_group, strategy, fusion,
            KeywordArg('eps'), Ignored())
        convert_pattern = MultiOutputPattern([trtllm_allreduce_default])

        def empty_convert_supported_ar_to_ub(
            input: torch.Tensor,
            residual_in: torch.Tensor,
            gamma: torch.Tensor,
            scale: torch.Tensor,
            fusion_op: int,
            eps: float,
        ):
            return

        def target_convert_supported_ar_to_ub(
            input: torch.Tensor,
            residual_in: torch.Tensor,
            gamma: torch.Tensor,
            scale: torch.Tensor,
            fusion_op: int,
            eps: float,
        ):
            input = torch.ops.trtllm.copy_to_userbuffers(input)
            all_reduce_output = torch.ops.trtllm.allreduce(
                input, residual_in, gamma, scale, None, None, mapping.tp_group,
                int(AllReduceStrategy.UB), fusion_op, eps, True)
            finalize_output = torch.ops.trtllm.userbuffers_allreduce_finalize(
                all_reduce_output[-1], False)
            all_reduce_output[-1] = finalize_output
            return all_reduce_output

        def extra_check_convert_supported_ar_to_ub(match: Match) -> bool:
            input = match.ctx.pattern_to_node[input_node]
            if not isinstance(input, torch.fx.graph.Node):
                return False
            dtype = input.meta["tensor_meta"].dtype
            # UB only supports FP16/BF16 input
            if dtype != torch.float16 and dtype != torch.bfloat16:
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
            search_fn_pattern=convert_pattern,
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
            scaled_mm_prologue_pattern = MultiOutputPattern([ub_copy])

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
                search_fn_pattern=scaled_mm_prologue_pattern,
            )

        def register_nvfp4_prologue(custom_pass: PatternMatcherPass):
            trtllm_nvfp4_gemm_default = CallFunction(
                torch.ops.trtllm.nvfp4_gemm.default, KeywordArg('act_fp4'),
                KeywordArg('weight'), KeywordArg('act_sf'),
                KeywordArg('weight_scale'), KeywordArg('alpha'),
                KeywordArg('output_dtype'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   trtllm_nvfp4_gemm_default)
            nvfp4_gemm_prologue_pattern = MultiOutputPattern([ub_copy])

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
                search_fn_pattern=nvfp4_gemm_prologue_pattern,
            )

        def register_mm_prologue(custom_pass: PatternMatcherPass):
            aten_mm_default = CallFunction(torch.ops.aten.mm.default,
                                           KeywordArg('mm0_a'),
                                           KeywordArg('mm0_b'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   aten_mm_default)
            mm_prologue_pattern = MultiOutputPattern([ub_copy])

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
                search_fn_pattern=mm_prologue_pattern,
            )

        def register_add_prologue(custom_pass: PatternMatcherPass):
            aten_add_default = CallFunction(torch.ops.aten.add.Tensor,
                                            KeywordArg('add_a'),
                                            KeywordArg('add_b'))
            ub_copy = CallFunction(torch.ops.trtllm.copy_to_userbuffers,
                                   aten_add_default)
            add_prologue_pattern = MultiOutputPattern([ub_copy])

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
                search_fn_pattern=add_prologue_pattern,
            )

        register_scaled_mm_prologue(custom_pass)
        register_nvfp4_prologue(custom_pass)
        register_mm_prologue(custom_pass)
        register_add_prologue(custom_pass)

    def register_ub_finalize_patterns(custom_pass: PatternMatcherPass):
        # TODO: Unify the finalize patterns once the allreduce interface does not contain
        # dynamic number of tensors.
        def allreduce_quant_finalize_pattern(custom_pass: PatternMatcherPass):
            trtllm_userbuffers_allreduce_finalize_default = CallFunction(
                torch.ops.trtllm.userbuffers_allreduce_finalize.default,
                KeywordArg("sharded_residual"), False)
            trtllm_allreduce_default = CallFunction(
                torch.ops.trtllm.allreduce.default, KeywordArg("input"),
                trtllm_userbuffers_allreduce_finalize_default,
                KeywordArg("gamma"), KeywordArg("scale"), Ignored(), Ignored(),
                mapping.tp_group, int(AllReduceStrategy.UB),
                KeywordArg("fusion_op"), KeywordArg("eps"), Ignored())
            ub_ar_finalize_pattern = MultiOutputPattern(
                [trtllm_allreduce_default])

            def empty_quant_finalize_pattern(
                input: torch.Tensor,
                sharded_residual: torch.Tensor,
                gamma: torch.Tensor,
                scale: torch.Tensor,
                fusion_op: int,
                eps: float,
            ):
                return

            def target_quant_finalize_pattern(
                input: torch.Tensor,
                sharded_residual: torch.Tensor,
                gamma: torch.Tensor,
                scale: torch.Tensor,
                fusion_op: int,
                eps: float,
            ):
                all_reduce_output = torch.ops.trtllm.allreduce(
                    input, sharded_residual, gamma,
                    scale, None, None, mapping.tp_group,
                    int(AllReduceStrategy.UB), fusion_op, eps, True)
                return all_reduce_output

            register_replacement(
                empty_quant_finalize_pattern,
                target_quant_finalize_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_ar_finalize_pattern,
            )

        def allreduce_half_finalize_pattern(custom_pass: PatternMatcherPass):
            trtllm_userbuffers_allreduce_finalize_default = CallFunction(
                torch.ops.trtllm.userbuffers_allreduce_finalize.default,
                KeywordArg("sharded_residual"), False)
            trtllm_allreduce_default = CallFunction(
                torch.ops.trtllm.allreduce.default, KeywordArg("input"),
                trtllm_userbuffers_allreduce_finalize_default,
                KeywordArg("gamma"), Ignored(), Ignored(), Ignored(),
                mapping.tp_group, int(AllReduceStrategy.UB),
                int(AllReduceFusionOp.RESIDUAL_RMS_NORM), KeywordArg("eps"),
                Ignored())
            ub_ar_finalize_pattern = MultiOutputPattern(
                [trtllm_allreduce_default])

            def empty_half_finalize_pattern(
                input: torch.Tensor,
                sharded_residual: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
            ):
                return

            def target_half_finalize_pattern(
                input: torch.Tensor,
                sharded_residual: torch.Tensor,
                gamma: torch.Tensor,
                eps: float,
            ):
                all_reduce_output = torch.ops.trtllm.allreduce(
                    input, sharded_residual, gamma, None, None, None,
                    mapping.tp_group, int(AllReduceStrategy.UB),
                    int(AllReduceFusionOp.RESIDUAL_RMS_NORM), eps, True)
                return all_reduce_output

            register_replacement(
                empty_half_finalize_pattern,
                target_half_finalize_pattern,
                [],
                fwd_only,
                custom_pass,
                search_fn_pattern=ub_ar_finalize_pattern,
            )

        allreduce_quant_finalize_pattern(custom_pass)
        allreduce_half_finalize_pattern(custom_pass)

    custom_passes.append(PatternMatcherPass())
    register_ub_allreduce_quantize_fusion(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_convert_supported_ar_to_ub(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_ub_prologue_patterns(custom_passes[-1])

    custom_passes.append(PatternMatcherPass())
    register_ub_finalize_patterns(custom_passes[-1])
