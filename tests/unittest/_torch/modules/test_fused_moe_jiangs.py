import os
import pickle
import sys
from contextlib import contextmanager
from itertools import product
from typing import Dict, List, Optional
from unittest import mock

import _torch.helpers
import cloudpickle
import pytest
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
from _torch.helpers import (calc_woq_tolerence, per_block_cast_to_fp8,
                            per_block_cast_to_fp8_e8m0,
                            per_token_cast_to_fp8_e8m0)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig
from utils.util import (check_accuracy, skip_blackwell, skip_blackwell_geforce,
                        skip_neither_ada_nor_hopper_unittest, skip_no_hopper,
                        skip_pre_blackwell, skip_pre_hopper)

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import \
    CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import \
    DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.interface import (
    AlltoallMethodType, MoEWeightLoadingMode)

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from tensorrt_llm._torch.modules.fused_moe import (
    BaseMoeRoutingMethod, CutlassFusedMoE, TRTLLMGenFusedMoE,
    DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod, TritonFusedMoE,
    create_moe, WideEPMoE)
from tensorrt_llm._torch.modules.fused_moe.quantization import \
    NVFP4CutlassFusedMoEMethod
# isort: on
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._utils import get_sm_version, mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(_torch.helpers)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@skip_neither_ada_nor_hopper_unittest
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "weight_loading_mode",
    [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.W4A8_CUSTOM])
def test_fused_moe_w4afp8(dtype, weight_loading_mode):
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        SEQ_LEN = 4
        HIDDEN_SIZE = 768
        INTERMEDIATE_SIZE = 640
        SCALING_GROUP_SIZE = 128
        NUM_EXPERTS = 3
        TOP_K = 2
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        affine_coeff = 0.005

        lut = {
            "weight":
            "weight",
            "weight_scale":
            ("weight_scale_inv" if weight_loading_mode
             == MoEWeightLoadingMode.W4A8_CUSTOM else "weight_scale"),
            "weight_scale_2":
            "weight_scale_2",
            "pre_quant_scale":
            "pre_quant_scale",
            "input_scale":
            "input_scale",
        }

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            # ModelOpt W4A8 packs pairs of 4b weights in the output dimension into one 8b element.
            if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_shape = (INTERMEDIATE_SIZE // 2, HIDDEN_SIZE)
                w2_shape = (HIDDEN_SIZE // 2, INTERMEDIATE_SIZE)
                w3_shape = (INTERMEDIATE_SIZE // 2, HIDDEN_SIZE)
            # The custom W4A8 quantization script examples/quantization/quantize_mixed_precision_moe.py
            # packs pairs of 4b weight in the input dimension into one 8b element.
            if weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:
                w1_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
                w2_shape = (HIDDEN_SIZE, INTERMEDIATE_SIZE // 2)
                w3_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)

            # The weights in int4 precision.
            w1_weight = torch.randint(-128, 127, w1_shape,
                                      dtype=torch.int8).cuda()
            w2_weight = torch.randint(-128, 127, w2_shape,
                                      dtype=torch.int8).cuda()
            w3_weight = torch.randint(-128, 127, w3_shape,
                                      dtype=torch.int8).cuda()

            # The pre-quant scale to be multiplied with the input activation.
            # Use random pre-quant scales [0.95, 1.05] instead of fixed 1.0 to ensure the kernel handles
            # non-uniform pre-quant scaling factors correctly
            w1_pre_quant_scale = torch.rand(
                HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1 + 0.95
            w2_pre_quant_scale = torch.rand(
                INTERMEDIATE_SIZE, dtype=dtype, device="cuda") * 0.1 + 0.95
            w3_pre_quant_scale = torch.rand(
                HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1 + 0.95

            # The weight scale to dequantize int4 weights (by multiplication).
            w1_scale = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
                dtype=dtype,
                device="cuda") * affine_coeff
            w2_scale = torch.randn(
                (HIDDEN_SIZE, INTERMEDIATE_SIZE // SCALING_GROUP_SIZE),
                dtype=dtype,
                device="cuda") * affine_coeff
            w3_scale = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
                dtype=dtype,
                device="cuda") * affine_coeff

            # The input scale to quantize the input activation (by division).
            w1_input_scale = torch.randn(1, dtype=torch.float32,
                                         device="cuda") * 0.2
            w2_input_scale = w1_input_scale
            w3_input_scale = w1_input_scale

            # The weight scale 2 to quantize the dequantized weights (by division).
            w1_weight_scale_2 = torch.ones([1],
                                           dtype=torch.float32,
                                           device="cuda")
            w2_weight_scale_2 = w1_weight_scale_2
            w3_weight_scale_2 = w1_weight_scale_2

            # Prepare weights.
            weights[f"{expert_id}.w1.{lut['weight']}"] = w1_weight
            weights[f"{expert_id}.w2.{lut['weight']}"] = w2_weight
            weights[f"{expert_id}.w3.{lut['weight']}"] = w3_weight
            weights[f"{expert_id}.w1.{lut['input_scale']}"] = w1_input_scale
            weights[f"{expert_id}.w2.{lut['input_scale']}"] = w2_input_scale
            weights[f"{expert_id}.w3.{lut['input_scale']}"] = w3_input_scale
            weights[f"{expert_id}.w1.{lut['weight_scale']}"] = w1_scale
            weights[f"{expert_id}.w2.{lut['weight_scale']}"] = w2_scale
            weights[f"{expert_id}.w3.{lut['weight_scale']}"] = w3_scale
            weights[
                f"{expert_id}.w1.{lut['pre_quant_scale']}"] = w1_pre_quant_scale
            weights[
                f"{expert_id}.w2.{lut['pre_quant_scale']}"] = w2_pre_quant_scale
            weights[
                f"{expert_id}.w3.{lut['pre_quant_scale']}"] = w3_pre_quant_scale
            weights[
                f"{expert_id}.w1.{lut['weight_scale_2']}"] = w1_weight_scale_2
            weights[
                f"{expert_id}.w2.{lut['weight_scale_2']}"] = w2_weight_scale_2
            weights[
                f"{expert_id}.w3.{lut['weight_scale_2']}"] = w3_weight_scale_2

        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
        fused_moe = CutlassFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=False,
            model_config=ModelConfig(quant_config=quant_config),
            weight_loading_mode=weight_loading_mode)
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        def ref():
            results = torch.zeros_like(x)
            selected_experts, final_scales = routing_method.apply(router_logits)
            for e_idx in range(NUM_EXPERTS):
                mask = selected_experts == e_idx
                activated_tokens = mask.sum(1).bool()
                act = x[activated_tokens, :]
                if act.shape[0] == 0:
                    continue
                final_scale = (final_scales *
                               mask).sum(1)[activated_tokens].unsqueeze(1)

                # weights
                def unpack_weights(weight: torch.Tensor) -> torch.Tensor:
                    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
                    if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                        return unpacker(weight.cpu().T.contiguous()).cuda()
                    else:
                        return unpacker(weight.cpu()).T.contiguous().cuda()

                w1 = unpack_weights(weights[f"{e_idx}.w1.{lut['weight']}"])
                w2 = unpack_weights(weights[f"{e_idx}.w2.{lut['weight']}"])
                w3 = unpack_weights(weights[f"{e_idx}.w3.{lut['weight']}"])
                w3_w1 = torch.cat([w3, w1], dim=-1)

                # weight_scale
                s1 = weights[f"{e_idx}.w1.{lut['weight_scale']}"].T.contiguous(
                ).cuda()
                s2 = weights[f"{e_idx}.w2.{lut['weight_scale']}"].T.contiguous(
                ).cuda()
                s3 = weights[f"{e_idx}.w3.{lut['weight_scale']}"].T.contiguous(
                ).cuda()
                s3_s1 = torch.cat([s3, s1], dim=-1)

                # input_scale
                p1 = weights[f"{e_idx}.w1.{lut['input_scale']}"].cuda()
                p2 = weights[f"{e_idx}.w2.{lut['input_scale']}"].cuda()
                p3 = weights[f"{e_idx}.w3.{lut['input_scale']}"].cuda()
                p3_p1 = torch.max(p1, p3)

                # pre_quant_scale
                a1 = a2 = a3 = a1_a3 = None
                if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                    a1 = weights[
                        f"{e_idx}.w1.{lut['pre_quant_scale']}"].T.contiguous(
                        ).cuda()
                    a2 = weights[
                        f"{e_idx}.w2.{lut['pre_quant_scale']}"].T.contiguous(
                        ).cuda()
                    a3 = weights[
                        f"{e_idx}.w3.{lut['pre_quant_scale']}"].T.contiguous(
                        ).cuda()
                    a1_a3 = torch.max(a1, a3)

                # weight_scale_2
                q1 = q2 = q3 = q3_q1 = None
                if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                    q1 = weights[f"{e_idx}.w1.{lut['weight_scale_2']}"].cuda()
                    q2 = weights[f"{e_idx}.w3.{lut['weight_scale_2']}"].cuda()
                    q3 = weights[f"{e_idx}.w2.{lut['weight_scale_2']}"].cuda()
                    q3_q1 = torch.max(q3, q1)

                # forward pass
                def process_layer(
                    act,
                    weight,
                    weight_scale,
                    input_scale,
                    pre_quant_scale=None,
                    weight_scale_2=None,
                ):
                    if pre_quant_scale is not None:
                        act = act * pre_quant_scale
                    act = (torch.clamp((act / input_scale), -448.0,
                                       448.0).to(torch.float8_e4m3fn).to(dtype))
                    weight = (weight.float() * weight_scale.repeat_interleave(
                        128, dim=0).float()).to(dtype)
                    if weight_scale_2 is not None:
                        weight /= weight_scale_2
                    output = torch.matmul(act, weight) * input_scale
                    if weight_scale_2 is not None:
                        output *= weight_scale_2
                    return output

                # fc13
                fc1 = process_layer(
                    act,
                    w3_w1,
                    s3_s1,
                    p3_p1,
                    pre_quant_scale=a1_a3,
                    weight_scale_2=q3_q1,
                )
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)

                # fc2
                fc2 = process_layer(fc1,
                                    w2,
                                    s2,
                                    p2,
                                    pre_quant_scale=a2,
                                    weight_scale_2=q2)

                results[activated_tokens, :] += (fc2 * final_scale).to(
                    results.dtype)
            return results

        AutoTuner.get().clear_cache()
        with torch.inference_mode():
            ref_output = ref()

        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        # Explicitly capture context for kernel testing
        with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
            output = fused_moe.forward(x, router_logits)

        # Test all kernel tactics
        for tactic in all_tactics:
            with AutoTuner.get().replay(tactic), torch.inference_mode():
                output = fused_moe.forward(x, router_logits)
                # assert that result does not contain NaN or is all 0s
                assert not torch.isnan(output).any(), "output contains NaN"
                assert torch.nonzero(output).numel() > 0, "output is empty"
                torch.testing.assert_close(output,
                                           ref_output,
                                           rtol=1e-2,
                                           atol=0.1)

        torch.cuda.synchronize()
        assert not torch.isnan(ref_output).any(), "ref_output contains NaN"
        assert not torch.isnan(output).any(), "output contains NaN"
        assert torch.nonzero(output).numel() > 0, "output is empty"
        assert torch.nonzero(ref_output).numel() > 0, "ref_output is empty"
        # Final comparison
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [768, 2880])
@pytest.mark.parametrize(
    "moe_backend",
    [
        # smVersion
        pytest.param("TRTLLM",
                     marks=[skip_blackwell_geforce, skip_pre_blackwell]),
        pytest.param(
            "CUTLASS",
            marks=[skip_pre_hopper, skip_blackwell, skip_blackwell_geforce]),
    ],
)
def test_fused_moe_wfp4a16(dtype, hidden_size, moe_backend):
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        ######################################################################
        # SEQ_LEN = 4
        # HIDDEN_SIZE = hidden_size
        # INTERMEDIATE_SIZE = 640
        # SCALING_GROUP_SIZE = 32
        # NUM_EXPERTS = 4
        # TOP_K = 2
        ######################################################################
        SEQ_LEN = 16
        HIDDEN_SIZE = hidden_size
        INTERMEDIATE_SIZE = 640
        SCALING_GROUP_SIZE = 32
        NUM_EXPERTS = 8
        TOP_K = 8
        ######################################################################
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.randint(0,
                                      256,
                                      (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                      dtype=torch.uint8,
                                      device='cuda')
            w2_weight = torch.randint(0,
                                      256,
                                      (HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
                                      dtype=torch.uint8,
                                      device='cuda')
            w3_weight = torch.randint(0,
                                      256,
                                      (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                      dtype=torch.uint8,
                                      device='cuda')

            w1_scale = torch.randint(
                118,
                123, (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
                dtype=torch.uint8,
                device='cuda')
            w2_scale = torch.randint(
                118,
                123, (HIDDEN_SIZE, INTERMEDIATE_SIZE // SCALING_GROUP_SIZE),
                dtype=torch.uint8,
                device='cuda')
            w3_scale = torch.randint(
                118,
                123, (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
                dtype=torch.uint8,
                device='cuda')

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
            # WFP4A16FusedMoEMethod
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_scale
            # MXFP4WeightFusedMoEMethod
            weights[f"{expert_id}.w1.weight_scale"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_scale

        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4)

        # Create pretrained_config with necessary parameters
        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = NUM_EXPERTS
        pretrained_config.hidden_size = HIDDEN_SIZE
        pretrained_config.intermediate_size = INTERMEDIATE_SIZE
        pretrained_config.torch_dtype = dtype

        fused_moe = create_moe(routing_method=routing_method,
                               reduce_results=False,
                               model_config=ModelConfig(
                                   pretrained_config=pretrained_config,
                                   quant_config=quant_config,
                                   moe_backend=moe_backend))
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        def ref():
            results = torch.zeros_like(x)
            selected_experts, final_scales = routing_method.apply(router_logits)
            unpacker = torch.ops.trtllm.mxfp4_dequantize_unswizzled
            for e_idx in range(NUM_EXPERTS):
                mask = selected_experts == e_idx
                activated_tokens = mask.sum(1).bool()
                act = x[activated_tokens, :]
                if act.shape[0] == 0:
                    continue
                final_scale = (final_scales *
                               mask).sum(1)[activated_tokens].unsqueeze(1)

                # weights and scales
                w1 = weights[f"{e_idx}.w1.weight"]
                s1 = weights[f"{e_idx}.w1.weight_scale_inv"]
                w2 = weights[f"{e_idx}.w2.weight"]
                s2 = weights[f"{e_idx}.w2.weight_scale_inv"]
                w3 = weights[f"{e_idx}.w3.weight"]
                s3 = weights[f"{e_idx}.w3.weight_scale_inv"]

                # converted weights
                w1 = unpacker(w1.cpu(), s1.cpu(), SCALING_GROUP_SIZE).to(
                    dtype=x.dtype, device=x.device).T.contiguous()
                w2 = unpacker(w2.cpu(), s2.cpu(), SCALING_GROUP_SIZE).to(
                    dtype=x.dtype, device=x.device).T.contiguous()
                w3 = unpacker(w3.cpu(), s3.cpu(), SCALING_GROUP_SIZE).to(
                    dtype=x.dtype, device=x.device).T.contiguous()
                w3_w1 = torch.cat([w3, w1], dim=-1)

                fc1 = torch.matmul(act, w3_w1)
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)
                fc2 = torch.matmul(fc1, w2)
                results[activated_tokens, :] += (fc2 * final_scale).to(
                    results.dtype)
            return results

        AutoTuner.get().clear_cache()
        with torch.inference_mode():
            ref_output = ref()

        nvtx.range_push("autotune")
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)
        nvtx.range_pop()

        from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
        # Get the C++ FusedMoeRunner to query tactic descriptions
        cpp_runner = next(iter(MoERunner.runner_dict.values()))

        cache = AutoTuner.get().profiling_cache.cache
        for key, value in cache.items():
            custom_op, runner_cls, runner_id, shape_profile = key
            runner_id_val, tactic, min_time = value
            gemm_idx = 1 if "gemm1" in custom_op else 2
            desc = cpp_runner.get_tactic_desc(gemm_idx, tactic)
            print(f"Op: {custom_op}, Runner: {runner_cls}, Shape: {shape_profile}")
            print(f"  -> Best tactic: {tactic}, Time: {min_time:.6f}ms")
            print(f"  -> {desc}")

        # Explicitly capture context for kernel testing
        with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
            output = fused_moe.forward(x, router_logits)

        # Test all kernel tactics
        for tactic in all_tactics:
            with AutoTuner.get().replay(tactic), torch.inference_mode():
                output = fused_moe.forward(x, router_logits)
                check_accuracy(output,
                               ref_output,
                               rtol=1e-2,
                               atol=0.1,
                               percent=0.99)

        # compare
        torch.cuda.synchronize()
        check_accuracy(output, ref_output, rtol=1e-2, atol=0.1, percent=0.99)


@skip_no_hopper
@pytest.mark.parametrize("experts", [8, 128])
@pytest.mark.parametrize(
    "hidden_size, intermediate_size",
    [
        (2880, 2880),
        (2880, 1440),
        (2880, 720),
        (2880, 360),
    ],
)
@pytest.mark.parametrize("fp8_activation", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dynamic_quant", [True, False])
def test_fused_moe_triton_mxfp4(experts, hidden_size, intermediate_size,
                                fp8_activation, bias, dynamic_quant):
    if fp8_activation:
        pytest.skip("Latest Triton requires BF16 activation on Hopper")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        dtype = torch.bfloat16
        SEQ_LEN = 8
        HIDDEN_SIZE = hidden_size
        INTERMEDIATE_SIZE = intermediate_size
        NUM_EXPERTS = experts
        TOP_K = 4
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

        w1_weight = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w2_weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w1_bias = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE),
                              dtype=dtype).cuda()
        w2_bias = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=dtype).cuda()
        w3_bias = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE),
                              dtype=dtype).cuda()

        from triton_kernels.numerics_details.mxfp import (
            downcast_to_mxfp_torch, upcast_from_mxfp_torch)

        def fp32_to_mxfp4(tensor):
            tensor = tensor.transpose(1, 2).contiguous()
            tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor,
                                                               torch.uint8,
                                                               axis=1)
            tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
            tensor_scales = tensor_scales.transpose(1, 2).contiguous()
            return tensor_fp4, tensor_scales

        def mxfp4_to_fp32(tensor, scales):
            tensor = tensor.transpose(1, 2).contiguous()
            scales = scales.transpose(1, 2).contiguous()
            tensor = upcast_from_mxfp_torch(tensor,
                                            scales,
                                            torch.float32,
                                            axis=1)
            return tensor.transpose(1, 2).contiguous()

        w1_weight_fp4, w1_weight_scale = fp32_to_mxfp4(w1_weight)
        w2_weight_fp4, w2_weight_scale = fp32_to_mxfp4(w2_weight)
        w3_weight_fp4, w3_weight_scale = fp32_to_mxfp4(w3_weight)
        w1_weight_qdq = mxfp4_to_fp32(w1_weight_fp4, w1_weight_scale)
        w2_weight_qdq = mxfp4_to_fp32(w2_weight_fp4, w2_weight_scale)
        w3_weight_qdq = mxfp4_to_fp32(w3_weight_fp4, w3_weight_scale)

        # Since we don't have mxfp4 reference, we run the ref in bf16 after q-dq
        weights = {}
        for expert_id in range(NUM_EXPERTS):
            weights[f"{expert_id}.w1.weight"] = w1_weight_qdq[expert_id]
            weights[f"{expert_id}.w2.weight"] = w2_weight_qdq[expert_id]
            weights[f"{expert_id}.w3.weight"] = w3_weight_qdq[expert_id]
            if bias:
                weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
                weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
                weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]

        ref_fused_moe = RefGatedMLPFusedMoE(num_experts=NUM_EXPERTS,
                                            routing_method=routing_method,
                                            hidden_size=HIDDEN_SIZE,
                                            intermediate_size=INTERMEDIATE_SIZE,
                                            dtype=dtype,
                                            model_config=ModelConfig(),
                                            bias=bias)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)
        torch.cuda.synchronize()

        # Now we run the TritonFusedMoE with MXFP4 weights
        weights = {}

        for expert_id in range(NUM_EXPERTS):
            if dynamic_quant:
                weights[f"{expert_id}.w1.weight"] = w1_weight_qdq[expert_id]
                weights[f"{expert_id}.w2.weight"] = w2_weight_qdq[expert_id]
                weights[f"{expert_id}.w3.weight"] = w3_weight_qdq[expert_id]
            else:
                weights[f"{expert_id}.w1.weight"] = w1_weight_fp4[expert_id]
                weights[f"{expert_id}.w2.weight"] = w2_weight_fp4[expert_id]
                weights[f"{expert_id}.w3.weight"] = w3_weight_fp4[expert_id]
                weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale[
                    expert_id]
                weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale[
                    expert_id]
                weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale[
                    expert_id]
            if bias:
                weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
                weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
                weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]

        quant_algo = QuantAlgo.W4A8_MXFP4_FP8 if fp8_activation else QuantAlgo.W4A16_MXFP4
        quant_config = QuantConfig(quant_algo=quant_algo)
        fused_moe = TritonFusedMoE(num_experts=NUM_EXPERTS,
                                   routing_method=routing_method,
                                   hidden_size=HIDDEN_SIZE,
                                   intermediate_size=INTERMEDIATE_SIZE,
                                   dtype=dtype,
                                   reduce_results=True,
                                   bias=bias,
                                   model_config=ModelConfig(
                                       quant_config=quant_config,
                                       mapping=mapping))
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
        torch.cuda.synchronize()

        # Evaluate outputs

        # There can be one off mismatch in the outputs due to different kernel implementations
        # Here we check certain percent of the outputs are within the tolerance
        check_accuracy(output, ref_output, rtol=0.6, atol=0.6, percent=0.945)


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig(),
                 use_cute_dsl_blockscaling_mm: bool = False,
                 bias=False,
                 swiglu_alpha: Optional[float] = None,
                 swiglu_beta: Optional[float] = None,
                 swiglu_limit: Optional[float] = None):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        def custom_swiglu(x):
            gate, value = x.chunk(2, dim=-1)
            if swiglu_limit is not None and swiglu_limit != float("inf"):
                gate = gate.clamp(max=swiglu_limit)
                value = value.clamp(min=-swiglu_limit, max=swiglu_limit)

            alpha = swiglu_alpha if swiglu_alpha is not None else 1.0
            gate_act = gate * torch.sigmoid(gate * alpha)

            beta = swiglu_beta if swiglu_beta is not None else 0.0

            return gate_act * (value + beta)

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=bias,
                dtype=self.dtype,
                config=model_config,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
                activation=custom_swiglu
                if swiglu_alpha is not None else F.silu,
            ) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        selected_experts, routing_weights = self.routing_method.apply(
            router_logits)

        final_hidden_states = torch.zeros(hidden_states.shape,
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]

            output = self.experts[expert_id](expert_inputs)
            final_hidden_states[batch_idx] += routing_weights[
                batch_idx, nth_expert, None] * output.float()

        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]

            gate_up_proj_weights[0]['weight'] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]['weight'] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]['weight'] = weights[f"{expert}.w2.weight"]
            if self.bias:
                gate_up_proj_weights[0]['bias'] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]['bias'] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]['bias'] = weights[f"{expert}.w2.bias"]

            if self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
            elif self.quant_config and self.quant_config.quant_algo in (
                    QuantAlgo.NVFP4, QuantAlgo.W4A8_NVFP4_FP8):
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
                gate_up_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w1.weight_scale_2"]
                gate_up_proj_weights[1]['weight_scale_2'] = weights[
                    f"{expert}.w3.weight_scale_2"]
                down_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w2.weight_scale_2"]
            elif (self.quant_config and self.quant_config.quant_algo
                  == QuantAlgo.FP8_BLOCK_SCALES):
                gate_up_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]["weight_scale"] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w2.weight_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

