import pickle
import sys
from itertools import product
from typing import Dict, List, Optional
from unittest import mock

import _torch.helpers
import cloudpickle
import pytest
import torch
import torch.nn as nn
from _torch.helpers import (calc_woq_tolerence, per_block_cast_to_fp8,
                            per_block_cast_to_fp8_e8m0,
                            per_token_cast_to_fp8_e8m0)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import (check_accuracy, skip_blackwell, skip_blackwell_geforce,
                        skip_neither_ada_nor_hopper_unittest,
                        skip_non_hopper_unittest, skip_pre_blackwell,
                        skip_pre_hopper)

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import \
    CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import \
    DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import \
    AlltoallMethodType
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from tensorrt_llm._torch.modules.fused_moe import (
    BaseMoeRoutingMethod, CutlassFusedMoE, TRTLLMGenFusedMoE,
    DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod, TritonFusedMoE,
    create_moe, WideEPMoE)
# isort: on
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import \
    IS_TRITON_KERNELS_AVAILABLE
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(_torch.helpers)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.mark.parametrize(
    "moe_backend, dtype, experts, routing_cls, bias",
    product(["CUTLASS", "VANILLA", "TRITON"], [torch.float16, torch.bfloat16],
            [3, 8, 512], [DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod],
            [True, False]))
def test_fused_moe(moe_backend,
                   dtype,
                   experts,
                   routing_cls,
                   bias,
                   mapping=None):

    if moe_backend == "TRITON":
        if not IS_TRITON_KERNELS_AVAILABLE:
            pytest.skip("Triton kernels are not available")
        if dtype != torch.bfloat16:
            pytest.skip("Unsupported for TritonFusedMoE")
        if routing_cls != RenormalizeMoeRoutingMethod:
            pytest.skip("Unsupported for TritonFusedMoE")

    if bias and moe_backend not in ["TRITON"]:
        pytest.skip("Bias not supported.")

    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()

    torch.cuda.set_device(mapping.rank)

    with torch.device(f'cuda:{mapping.rank}'):
        SEQ_LEN = 8
        HIDDEN_SIZE = 64
        INTERMEDIATE_SIZE = 32
        NUM_EXPERTS = experts
        TOP_K = 2
        routing_method = routing_cls(top_k=TOP_K)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            if bias:
                w1_bias = torch.randn((INTERMEDIATE_SIZE, ), dtype=dtype).cuda()
                w2_bias = torch.randn((HIDDEN_SIZE, ), dtype=dtype).cuda()
                w3_bias = torch.randn((INTERMEDIATE_SIZE, ), dtype=dtype).cuda()
                weights[f"{expert_id}.w1.bias"] = w1_bias
                weights[f"{expert_id}.w2.bias"] = w2_bias
                weights[f"{expert_id}.w3.bias"] = w3_bias
            w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
        fused_moe = create_moe(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            model_config=ModelConfig(mapping=mapping, moe_backend=moe_backend),
            bias=bias,
        )
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        ref_fused_moe = RefGatedMLPFusedMoE(num_experts=NUM_EXPERTS,
                                            routing_method=routing_method,
                                            hidden_size=HIDDEN_SIZE,
                                            intermediate_size=INTERMEDIATE_SIZE,
                                            dtype=dtype,
                                            model_config=ModelConfig(),
                                            bias=bias)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

    # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
    m = SEQ_LEN
    while m >= 2:
        x = torch.randn((m, HIDDEN_SIZE), dtype=dtype, device="cuda")
        router_logits = torch.randn((m, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref_fused_moe.forward(x, router_logits)

        # Evaluate outputs
        torch.cuda.synchronize()
        # There can be one off mismatch in the outputs due to different kernel implementations
        # Here we check 99% of the outputs are within the tolerance
        # The CutlassFusedMoE case fails as well without this change on H100 for bf16
        check_accuracy(output, ref_output, rtol=0.2, atol=0.2, percent=0.984)
        m //= 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("moe_cls", ["CUTLASS", "VANILLA"])
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_fused_moe_multi_gpu(moe_cls, ep_size):
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            test_fused_moe,
            *zip(
                *[(moe_cls, torch.bfloat16, 512, DefaultMoeRoutingMethod, False,
                   Mapping(world_size=world_size,
                           tp_size=world_size,
                           moe_ep_size=ep_size,
                           moe_tp_size=world_size // ep_size))] * world_size),
        )
        for r in results:
            assert r is None


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("alltoall_method_type", [
    AlltoallMethodType.MNNVL, AlltoallMethodType.DeepEP,
    AlltoallMethodType.DeepEPLowLatency
],
                         ids=lambda s: s.name)
def test_fused_moe_alltoall(alltoall_method_type):
    world_size = 4
    dtype = torch.bfloat16
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = 72
    TOP_K = 6
    MAX_NUM_TOKENS = 2048

    def per_rank_test_fused_moe_alltoall(job_id):
        routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
        mapping = Mapping(world_size=world_size,
                          rank=mpi_rank(),
                          tp_size=world_size,
                          moe_ep_size=world_size,
                          moe_tp_size=1,
                          enable_attention_dp=True)
        torch.cuda.set_device(mapping.rank)
        torch.manual_seed(mapping.rank)

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.empty((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype)
            w2_weight = torch.empty((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=dtype)
            w3_weight = torch.empty((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype)
            torch.nn.init.xavier_uniform_(w1_weight)
            torch.nn.init.xavier_uniform_(w2_weight)
            torch.nn.init.xavier_uniform_(w3_weight)
            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
        with mock.patch.object(WideEPMoE,
                               "select_alltoall_method_type",
                               return_value=alltoall_method_type):
            alltoall_model = WideEPMoE(
                num_experts=NUM_EXPERTS,
                routing_method=routing_method,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                dtype=dtype,
                reduce_results=True,
                model_config=ModelConfig(mapping=mapping,
                                         max_num_tokens=MAX_NUM_TOKENS,
                                         moe_max_num_tokens=MAX_NUM_TOKENS),
            )
        alltoall_model.to("cuda")
        alltoall_model.load_weights([weights])

        ref_model = CutlassFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            model_config=ModelConfig(mapping=mapping,
                                     max_num_tokens=MAX_NUM_TOKENS),
        )
        ref_model.to("cuda")
        ref_model.load_weights([weights])

        # Evaluate the outputs on a variant sequence length to verify the robustness of alltoall methods
        m = MAX_NUM_TOKENS
        while m >= 1:
            x = torch.randn((m, HIDDEN_SIZE), dtype=dtype, device="cuda")
            router_logits = torch.randn((m, NUM_EXPERTS),
                                        dtype=dtype,
                                        device="cuda")
            all_rank_num_tokens = [m] * mapping.world_size

            with torch.inference_mode():
                output = alltoall_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)
                ref_output = ref_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)

            if alltoall_method_type == AlltoallMethodType.MNNVL and output.ndim == 3:
                output = output.sum(dim=1)
            print(f"output: {output.shape}")
            print(f"ref_output: {ref_output.shape}")
            # Evaluate outputs
            torch.testing.assert_close(output,
                                       ref_output,
                                       rtol=0.05,
                                       atol=0.003)
            m //= 2

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(per_rank_test_fused_moe_alltoall,
                               range(world_size))
        for r in results:
            assert r is None


@pytest.mark.skip(reason="https://nvbugs/5467531")
@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("alltoall_method_type", [
    AlltoallMethodType.MNNVL, AlltoallMethodType.DeepEP,
    AlltoallMethodType.DeepEPLowLatency
],
                         ids=lambda s: s.name)
def test_fused_moe_alltoall_fp4(alltoall_method_type):

    world_size = 4
    dtype = torch.bfloat16
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = 72
    TOP_K = 6
    MAX_NUM_TOKENS = 2048

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x_list_world = []
    weights_world = []

    for i in range(world_size):
        x_list = []
        m = MAX_NUM_TOKENS
        while m >= 1:
            x = torch.randn((m, HIDDEN_SIZE), dtype=dtype, device="cuda")
            x_list.append(x.cuda(i))
            m //= 2

        x_abs_max = torch.cat([x.flatten() for x in x_list]).abs().max().float()
        x_sf_global = (448 * 6) / x_abs_max

        weights = {}
        for expert_id in range(NUM_EXPERTS):

            w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w1_sf_global = (448 * 6) / w1_weight.abs().max().float()

            w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w2_sf_global = (448 * 6) / w2_weight.abs().max().float()

            w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w3_sf_global = (448 * 6) / w3_weight.abs().max().float()

            w3_w1_global = min(
                w1_sf_global,
                w3_sf_global)  # w3 global and w1 global must be the same

            SCALING_VECTOR_SIZE = 16

            w1_weight_nvfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
                w1_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
            w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w1_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

            w2_weight_nvfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
                w2_weight, w2_sf_global, SCALING_VECTOR_SIZE, False)
            w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w2_sf_block.cpu().view(HIDDEN_SIZE, -1))

            w3_weight_nvfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
                w3_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
            w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w3_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

            w1_input_scale = x_sf_global.cuda(i)
            w2_input_scale = x_sf_global.cuda(i)
            w3_input_scale = x_sf_global.cuda(i)

            weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4.cuda(i)
            weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4.cuda(i)
            weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4.cuda(i)
            weights[
                f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.cuda(i)
            weights[
                f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.cuda(i)
            weights[
                f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.cuda(i)

            weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale.cuda(
                i)
            weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale.cuda(
                i)
            weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale.cuda(
                i)
            weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global.cuda(
                i)
            weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global.cuda(
                i)
            weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global.cuda(
                i)

        x_list_world.append(x_list)
        weights_world.append(weights)

    def per_rank_test_fused_moe_alltoall(job_id):
        routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
        mapping = Mapping(world_size=world_size,
                          rank=mpi_rank(),
                          tp_size=world_size,
                          moe_ep_size=world_size,
                          moe_tp_size=1,
                          enable_attention_dp=True)
        torch.cuda.set_device(mapping.rank)
        torch.manual_seed(mapping.rank)

        x_list = x_list_world[mapping.rank]
        weights = weights_world[mapping.rank]

        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
        with mock.patch.object(WideEPMoE,
                               "select_alltoall_method_type",
                               return_value=alltoall_method_type):
            alltoall_model = WideEPMoE(
                num_experts=NUM_EXPERTS,
                routing_method=routing_method,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                dtype=dtype,
                reduce_results=True,
                model_config=ModelConfig(mapping=mapping,
                                         max_num_tokens=MAX_NUM_TOKENS,
                                         quant_config=quant_config),
            )
        alltoall_model.to("cuda")
        alltoall_model.load_weights([weights])

        ref_model = CutlassFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            model_config=ModelConfig(mapping=mapping,
                                     max_num_tokens=MAX_NUM_TOKENS,
                                     quant_config=quant_config),
        )
        ref_model.to("cuda")
        ref_model.load_weights([weights])

        # Evaluate the outputs on a variant sequence length to verify the robustness of alltoall methods
        m = MAX_NUM_TOKENS
        i = 0
        while m >= 1:
            x = x_list[i]
            i += 1
            router_logits = torch.randn((m, NUM_EXPERTS),
                                        dtype=dtype,
                                        device="cuda")
            all_rank_num_tokens = [m] * mapping.world_size

            with torch.inference_mode():
                output = alltoall_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)
                ref_output = ref_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)

            # Evaluate outputs
            torch.testing.assert_close(output, ref_output, rtol=0.05, atol=0.5)
            m //= 2

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(per_rank_test_fused_moe_alltoall,
                               range(world_size))
        for r in results:
            assert r is None


@skip_pre_hopper
@pytest.mark.parametrize("moe_backend", ["CUTLASS", "TRITON"])
@pytest.mark.parametrize("routing_cls",
                         [DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8(moe_backend, dtype, routing_cls, bias):

    if moe_backend == "TRITON":
        if not IS_TRITON_KERNELS_AVAILABLE:
            pytest.skip("Triton kernels are not available")
        if dtype != torch.bfloat16:
            pytest.skip("Unsupported for TritonFusedMoE")
        if routing_cls != RenormalizeMoeRoutingMethod:
            pytest.skip("Unsupported for TritonFusedMoE")

    if bias and moe_backend not in ["TRITON"]:
        pytest.skip("Bias not supported.")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        SEQ_LEN = 4
        HIDDEN_SIZE = 64
        INTERMEDIATE_SIZE = 32
        NUM_EXPERTS = 3
        TOP_K = 2
        routing_method = routing_cls(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
        _, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
        x_scale = x_scale.float().squeeze()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            if bias:
                w1_bias = torch.randn((INTERMEDIATE_SIZE, ), dtype=dtype).cuda()
                w2_bias = torch.randn((HIDDEN_SIZE, ), dtype=dtype).cuda()
                w3_bias = torch.randn((INTERMEDIATE_SIZE, ), dtype=dtype).cuda()
                weights[f"{expert_id}.w1.bias"] = w1_bias
                weights[f"{expert_id}.w2.bias"] = w2_bias
                weights[f"{expert_id}.w3.bias"] = w3_bias
            w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=dtype,
                                    device="cuda")

            w1_weight_fp8, w1_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w1_weight)
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w2_weight_fp8, w2_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w2_weight)
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w3_weight_fp8, w3_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w3_weight)
            w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w1_input_scale = x_scale.cuda()
            w2_input_scale = x_scale.cuda()
            w3_input_scale = x_scale.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale.float()
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale.float()
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale.float()
            weights[f"{expert_id}.w1.input_scale"] = w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = w3_input_scale

        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
        fused_moe = create_moe(num_experts=NUM_EXPERTS,
                               routing_method=routing_method,
                               hidden_size=HIDDEN_SIZE,
                               intermediate_size=INTERMEDIATE_SIZE,
                               dtype=dtype,
                               reduce_results=False,
                               model_config=ModelConfig(
                                   quant_config=quant_config,
                                   moe_backend=moe_backend),
                               bias=bias)
        fused_moe.cuda()
        fused_moe.load_weights([weights])

        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        ref_fused_moe = RefGatedMLPFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            model_config=ModelConfig(quant_config=quant_config),
            bias=bias)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()
        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref_fused_moe.forward(x, router_logits)

        # compare
        torch.cuda.synchronize()
        check_accuracy(output, ref_output, rtol=0.04, atol=0.1, percent=0.99)


def set_tensor_value_2(x, num_row, num_cols):
    # Create 2x2 base pattern matrix
    pattern = torch.tensor([[0.2, -0.5], [-0.3, 0.1]], device=x.device)

    # Repeat pattern to cover entire matrix
    repeated = pattern.repeat((num_row + 1) // 2,
                              (num_cols + 1) // 2)[:num_row, :num_cols]

    x.copy_(repeated)


def set_tensor_value_3(x, num_row, num_cols):
    # Create 3x3 base pattern matrix
    pattern = torch.tensor(
        [[0.1, 0.21, 0.31], [0.3, 0.6, 0.1], [0.11, 0.51, 0.62]],
        device=x.device)

    # Repeat pattern to cover entire matrix
    repeated = pattern.repeat((num_row + 2) // 3,
                              (num_cols + 2) // 3)[:num_row, :num_cols]

    x.copy_(repeated)


def set_tensor_value_4(x, num_row, num_cols):
    # Create 4x4 base pattern matrix
    pattern = torch.tensor(
        [
            [0.1, 0.21, 0.31, 0.41],
            [0.3, 0.6, 0.1, 0.2],
            [0.11, 0.51, 0.61, 0.71],
            [0.11, 0.52, 0.62, 0.72],
        ],
        device=x.device,
    )

    # Repeat pattern to cover entire matrix
    repeated = pattern.repeat((num_row + 3) // 4,
                              (num_cols + 3) // 4)[:num_row, :num_cols]

    x.copy_(repeated)


@pytest.mark.skip(reason="https://nvbugs/5565565")
@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize(
    "alltoall_method_type",
    [AlltoallMethodType.MNNVL, AlltoallMethodType.NotEnabled],
    ids=lambda s: s.name)
def test_fused_moe_fp8_blockwise_wide_ep(alltoall_method_type):
    """Test WideEPMoE with FP8 block-wise quantization using DeepGemmFusedMoE as reference."""

    world_size = 4
    dtype = torch.bfloat16
    # Reduce model size to avoid MPI int32 overflow
    HIDDEN_SIZE = 768
    INTERMEDIATE_SIZE = 512
    NUM_EXPERTS = 16
    TOP_K = 2
    MAX_NUM_TOKENS = 256

    # The MPI can not support FP8, so create weights on each rank
    def per_rank_test_fused_moe_alltoall_fp8_blockwise(job_id):
        routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
        mapping = Mapping(world_size=world_size,
                          rank=mpi_rank(),
                          tp_size=world_size,
                          moe_ep_size=world_size,
                          moe_tp_size=1,
                          enable_attention_dp=True)
        torch.cuda.set_device(mapping.rank)
        # Use same seed for all ranks to ensure consistency
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Generate test data locally on each rank
        x_list = []
        m = MAX_NUM_TOKENS
        while m >= 1:
            x = torch.randn((m, HIDDEN_SIZE), dtype=dtype, device="cuda")
            set_tensor_value_2(x, m, HIDDEN_SIZE)
            x_list.append(x)
            m //= 2

        # Generate weights locally on each rank (same weights due to same seed)
        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype,
                device="cuda") / HIDDEN_SIZE
            w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=dtype,
                                    device="cuda")
            w3_weight = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype,
                device="cuda") / HIDDEN_SIZE

            set_tensor_value_3(w1_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)
            set_tensor_value_4(w2_weight, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            set_tensor_value_3(w3_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)

            # FP8 block-wise quantization
            w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8_e8m0(
                w1_weight)
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8_e8m0(
                w2_weight)
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8_e8m0(
                w3_weight)
            w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale

        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

        # Test WideEPMoE with alltoall method
        with mock.patch.object(WideEPMoE,
                               "select_alltoall_method_type",
                               return_value=alltoall_method_type):
            alltoall_model = WideEPMoE(
                num_experts=NUM_EXPERTS,
                routing_method=routing_method,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                dtype=dtype,
                reduce_results=True,
                model_config=ModelConfig(mapping=mapping,
                                         max_num_tokens=MAX_NUM_TOKENS,
                                         quant_config=quant_config),
            )
        alltoall_model.to("cuda")
        alltoall_model.load_weights([weights])
        alltoall_model.post_load_weights()

        # Use DeepGemmFusedMoE as reference
        ref_model = DeepGemmFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            model_config=ModelConfig(mapping=mapping,
                                     max_num_tokens=MAX_NUM_TOKENS,
                                     quant_config=quant_config),
        )
        ref_model.to("cuda")
        ref_model.load_weights([weights])
        ref_model.post_load_weights()

        # Evaluate the outputs on variant sequence lengths
        m = MAX_NUM_TOKENS
        i = 0
        while m >= 1:
            x = x_list[i]
            i += 1
            router_logits = torch.randn((m, NUM_EXPERTS),
                                        dtype=dtype,
                                        device="cuda")
            all_rank_num_tokens = [m] * mapping.world_size
            with torch.inference_mode():
                output = alltoall_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)
                ref_output = ref_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False)

            # Evaluate outputs with relaxed tolerance for FP8
            # If WideEPMoE output has TOP_K dimension, reduce it to match DeepGemmFusedMoE
            if output.dim() == 3 and output.shape[1] == TOP_K:
                output = output.sum(dim=1)
            torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
            m //= 2

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(per_rank_test_fused_moe_alltoall_fp8_blockwise,
                               range(world_size))
        for r in results:
            assert r is None


@skip_pre_blackwell
@pytest.mark.parametrize(
    "dtype, num_experts, seq_len, hidden_size, RoutingMethodCls",
    product(
        [torch.bfloat16],
        [72],
        [128, 256, 384, 512, 1024, 2048, 4096, 8192],
        [2560],
        [DefaultMoeRoutingMethod],
    ),
)
def test_fused_moe_fp8_blockwise_deepgemm(dtype,
                                          num_experts,
                                          seq_len,
                                          hidden_size,
                                          RoutingMethodCls,
                                          mapping=None):
    SEQ_LEN = seq_len
    HIDDEN_SIZE = hidden_size
    INTERMEDIATE_SIZE = 256
    NUM_EXPERTS = num_experts
    TOP_K = 2

    routing_method = RoutingMethodCls(top_k=TOP_K)

    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    # Note: we use some special values init x and weight, otherwise the test will false positive failed.
    set_tensor_value_2(x, SEQ_LEN, HIDDEN_SIZE)

    x = x.cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    w3_w1_weight_scales = []
    w2_weight_scales = []
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() / HIDDEN_SIZE
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() / HIDDEN_SIZE
        set_tensor_value_3(w1_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        set_tensor_value_4(w2_weight, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        set_tensor_value_3(w3_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)

        w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8_e8m0(w1_weight)
        w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8_e8m0(w2_weight)
        w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8_e8m0(w3_weight)
        w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
        weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
        weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale

        w3_w1_weight_scales.append(
            torch.cat([w3_weight_scale, w1_weight_scale], dim=0))
        w2_weight_scales.append(w2_weight_scale)

    w3_w1_weight_scales = torch.stack(w3_w1_weight_scales, dim=0).cuda()
    w2_weight_scales = torch.stack(w2_weight_scales, dim=0).cuda()

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    fused_moe = DeepGemmFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=quant_config, mapping=mapping),
    )
    fused_moe.cuda()
    fused_moe.load_weights([weights])
    fused_moe.post_load_weights()

    def swiglu_fused_moe(x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x

    def grouped_gemm(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor,
                     b_sf: torch.Tensor,
                     offset_array: torch.Tensor) -> torch.Tensor:
        d = torch.empty((a.shape[0], b.shape[1]),
                        device=b.device,
                        dtype=torch.bfloat16)
        m_indices = torch.empty(a.shape[0], device=b.device, dtype=torch.int32)
        for idx in range(offset_array.numel() - 1):
            m_indices[offset_array[idx]:offset_array[idx + 1]] = idx

        num_groups, n, k_ = b.shape
        d = torch.empty((a.shape[0], b.shape[1]),
                        device=b.device,
                        dtype=torch.bfloat16)
        m_indices = torch.empty(a.shape[0], device=b.device, dtype=torch.int32)
        for idx in range(offset_array.numel() - 1):
            m_indices[offset_array[idx]:offset_array[idx + 1]] = idx

        for g in range(num_groups):
            aa = a[offset_array[g]:offset_array[g + 1], :].to(torch.bfloat16)
            aa_sf = a_sf[offset_array[g]:offset_array[g + 1], :]
            aa_dq = aa * aa_sf.repeat_interleave(
                128, dim=1)[:aa.shape[0], :aa.shape[1]]
            bb = b[g, :, :].to(torch.bfloat16)
            bb_sf = b_sf[g, :, :]
            bb_dq = bb * bb_sf.repeat_interleave(128, dim=0).repeat_interleave(
                128, dim=1)[:bb.shape[0], :bb.shape[1]]
            d[offset_array[g]:offset_array[g + 1], :] = (aa_dq @ bb_dq.t())
        return d

    token_selected_experts, token_final_scales = routing_method.apply(
        router_logits)
    t_idx = 0
    permuted_data_tensor = torch.empty((x.shape[0] * TOP_K, x.shape[1]),
                                       device=x.device,
                                       dtype=torch.bfloat16)
    expert_first_token_offset_tensor = torch.zeros(NUM_EXPERTS + 1,
                                                   dtype=torch.int32)
    unpermute_map = []
    scales = []
    for e_idx in range(NUM_EXPERTS):
        for idx, token in enumerate(x):
            for i, selected_expert in enumerate(token_selected_experts[idx]):
                if e_idx == selected_expert:
                    permuted_data_tensor[t_idx, :] = token
                    unpermute_map.append(idx)
                    scales.append(token_final_scales[idx, i])
                    t_idx += 1
        expert_first_token_offset_tensor[e_idx + 1] = t_idx

    act_input_fp8, act_input_sf = per_token_cast_to_fp8_e8m0(
        permuted_data_tensor)
    h1 = grouped_gemm(
        a=act_input_fp8,
        b=fused_moe.w3_w1_weight,
        a_sf=act_input_sf,
        b_sf=w3_w1_weight_scales,
        offset_array=expert_first_token_offset_tensor,
    )
    h2 = swiglu_fused_moe(h1)
    act_input_fp8, act_input_sf = per_token_cast_to_fp8_e8m0(h2)
    h3 = grouped_gemm(
        a=act_input_fp8,
        b=fused_moe.w2_weight,
        a_sf=act_input_sf,
        b_sf=w2_weight_scales,
        offset_array=expert_first_token_offset_tensor,
    )
    ref_output = torch.zeros_like(x)
    for token_idx, h3_token in enumerate(h3):
        original_idx = unpermute_map[token_idx]
        ref_output[original_idx, :] += h3_token * scales[token_idx]

    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "dtype, num_experts, seq_len, hidden_size, RoutingMethodCls, WeightLoadingMode",
    product(
        [torch.bfloat16],
        [72],
        [128, 256, 384, 512, 1024, 2048, 4096, 8192],
        [2560],
        [DefaultMoeRoutingMethod],
        [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ],
    ),
)
def test_fused_moe_fp8_blockwise_cute_dsl(dtype,
                                          num_experts,
                                          seq_len,
                                          hidden_size,
                                          RoutingMethodCls,
                                          WeightLoadingMode,
                                          mapping=None):
    SEQ_LEN = seq_len
    HIDDEN_SIZE = hidden_size
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = num_experts
    TOP_K = 6

    routing_method = RoutingMethodCls(top_k=TOP_K)

    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    # Note: we use some special values init x and weight, otherwise the test will false positive failed.
    set_tensor_value_2(x, SEQ_LEN, HIDDEN_SIZE)

    x = x.cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                dtype=dtype,
                                device="cuda")

    weights = {}

    if WeightLoadingMode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
        weights['gate_up_proj'] = {}
        weights['down_proj'] = {}
        weights['gate_up_proj_weight_scale'] = {}
        weights['down_proj_weight_scale'] = {}

    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype,
                                device="cuda")
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype,
                                device="cuda")
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype,
                                device="cuda")
        set_tensor_value_3(w1_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        set_tensor_value_4(w2_weight, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        set_tensor_value_3(w3_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)

        w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8(w1_weight)
        w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8(w2_weight)
        w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8(w3_weight)
        w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
        weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
        weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale

        if WeightLoadingMode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            weights['gate_up_proj'][expert_id] = torch.cat(
                [w3_weight_fp8, w1_weight_fp8],
                dim=-2).transpose(0, 1).contiguous()
            weights['down_proj'][expert_id] = w2_weight_fp8.transpose(
                0, 1).contiguous()
            weights['gate_up_proj_weight_scale'][expert_id] = torch.cat(
                [w3_weight_scale, w1_weight_scale],
                dim=-2).transpose(0, 1).contiguous()
            weights['down_proj_weight_scale'][
                expert_id] = w2_weight_scale.transpose(0, 1).contiguous()
        elif WeightLoadingMode == MoEWeightLoadingMode.VANILLA:
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    fused_moe = CuteDslFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=quant_config, mapping=mapping),
        weight_loading_mode=WeightLoadingMode,
    )
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        model_config=ModelConfig(quant_config=quant_config),
        # Note: use deepgemm mm will cause accuracy error, so we use trtllmgen mm here
        use_cute_dsl_blockscaling_mm=True,
    )
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()

    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    return True


@skip_non_hopper_unittest
@pytest.mark.parametrize(
    "dtype, num_experts, seq_len, hidden_size, RoutingMethodCls, WeightLoadingMode",
    product(
        [torch.bfloat16],
        [72],
        [128, 256, 384, 512, 1024, 2048, 4096, 8192],
        [2560],
        [DefaultMoeRoutingMethod],
        [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ],
    ),
)
def test_fused_moe_fp8_blockwise_cutlass(dtype,
                                         num_experts,
                                         seq_len,
                                         hidden_size,
                                         RoutingMethodCls,
                                         WeightLoadingMode,
                                         mapping=None):
    SEQ_LEN = seq_len
    HIDDEN_SIZE = hidden_size
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = num_experts
    TOP_K = 6

    routing_method = RoutingMethodCls(top_k=TOP_K)

    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    # Note: we use some special values init x and weight, otherwise the test will false positive failed.
    set_tensor_value_2(x, SEQ_LEN, HIDDEN_SIZE)

    x = x.cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                dtype=dtype,
                                device="cuda")

    weights = {}

    if WeightLoadingMode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
        weights['gate_up_proj'] = {}
        weights['down_proj'] = {}
        weights['gate_up_proj_weight_scale'] = {}
        weights['down_proj_weight_scale'] = {}

    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype,
                                device="cuda")
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype,
                                device="cuda")
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype,
                                device="cuda")
        set_tensor_value_3(w1_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        set_tensor_value_4(w2_weight, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        set_tensor_value_3(w3_weight, INTERMEDIATE_SIZE, HIDDEN_SIZE)

        w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8(w1_weight)
        w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8(w2_weight)
        w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8(w3_weight)
        w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
        weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
        weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale

        if WeightLoadingMode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            weights['gate_up_proj'][expert_id] = torch.cat(
                [w3_weight_fp8, w1_weight_fp8],
                dim=-2).transpose(0, 1).contiguous()
            weights['down_proj'][expert_id] = w2_weight_fp8.transpose(
                0, 1).contiguous()
            weights['gate_up_proj_weight_scale'][expert_id] = torch.cat(
                [w3_weight_scale, w1_weight_scale],
                dim=-2).transpose(0, 1).contiguous()
            weights['down_proj_weight_scale'][
                expert_id] = w2_weight_scale.transpose(0, 1).contiguous()
        elif WeightLoadingMode == MoEWeightLoadingMode.VANILLA:
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    fused_moe = CutlassFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=quant_config, mapping=mapping),
        weight_loading_mode=WeightLoadingMode,
    )
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        model_config=ModelConfig(quant_config=quant_config),
    )
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()

    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    return True


@skip_non_hopper_unittest
@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("routing_method", [DefaultMoeRoutingMethod])
@pytest.mark.parametrize(
    "weight_loading_mode",
    [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ])
def test_fused_moe_fp8_blockwise_cutlass_multi_gpu(ep_size, routing_method,
                                                   weight_loading_mode):
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            test_fused_moe_fp8_blockwise_cutlass,
            *zip(*[(
                torch.bfloat16,
                72,
                384,
                384,
                routing_method,
                weight_loading_mode,
                Mapping(
                    world_size=world_size,
                    tp_size=world_size,
                    moe_ep_size=ep_size,
                    moe_tp_size=world_size // ep_size,
                ),
            )] * world_size),
        )
        for r in results:
            assert r is True


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("routing_method", [DefaultMoeRoutingMethod])
@pytest.mark.parametrize(
    "weight_loading_mode",
    [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ])
def test_fused_moe_fp8_blockwise_cute_dsl_multi_gpu(ep_size, routing_method,
                                                    weight_loading_mode):
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            test_fused_moe_fp8_blockwise_cute_dsl,
            *zip(*[(
                torch.bfloat16,
                72,
                384,
                384,
                routing_method,
                weight_loading_mode,
                Mapping(
                    world_size=world_size,
                    tp_size=world_size,
                    moe_ep_size=ep_size,
                    moe_tp_size=world_size // ep_size,
                ),
            )] * world_size),
        )
        for r in results:
            assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "moe_backend",
    [pytest.param("TRTLLM", marks=skip_blackwell_geforce), "CUTLASS"])
def test_fused_moe_nvfp4(dtype, moe_backend):

    if moe_backend == "TRTLLM" and dtype == torch.float16:
        pytest.skip("TRTLLM NVFP4 MoE backend does not support float16 yet")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        SCALING_VECTOR_SIZE = 16

        SEQ_LEN = 4
        HIDDEN_SIZE = 512
        INTERMEDIATE_SIZE = 512
        NUM_EXPERTS = 4
        TOP_K = 2
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
        x_sf_global = (448 * 6) / x.abs().max().float()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype,
                device="cuda") * 0.05
            w1_sf_global = (448 * 6) / w1_weight.abs().max().float()

            w2_weight = torch.randn(
                (HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype,
                device="cuda") * 0.05
            w2_sf_global = (448 * 6) / w2_weight.abs().max().float()

            w3_weight = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype,
                device="cuda") * 0.05
            w3_sf_global = (448 * 6) / w3_weight.abs().max().float()

            w3_w1_global = min(
                w1_sf_global,
                w3_sf_global)  # w3 global and w1 global must be the same

            w1_weight_nvfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
                w1_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
            w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w1_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

            w2_weight_nvfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
                w2_weight, w2_sf_global, SCALING_VECTOR_SIZE, False)
            w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w2_sf_block.cpu().view(HIDDEN_SIZE, -1))

            w3_weight_nvfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
                w3_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
            w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w3_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

            w1_input_scale = x_sf_global.cuda()
            w2_input_scale = x_sf_global.cuda()
            w3_input_scale = x_sf_global.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
            weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
            weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
            weights[
                f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[
                f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[
                f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
            weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
            weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
            weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global

        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
        fused_moe = create_moe(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            model_config=ModelConfig(quant_config=quant_config,
                                     moe_backend=moe_backend),
        )
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
        ref_fused_moe = RefGatedMLPFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            model_config=ModelConfig(quant_config=quant_config))
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref_fused_moe.forward(x, router_logits)

        # compare
        torch.cuda.synchronize()
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "moe_backend",
    [pytest.param("TRTLLM", marks=skip_blackwell_geforce), "CUTLASS"])
def test_fused_moe_w4a8_nvfp4_fp8(moe_backend):
    dtype = torch.bfloat16
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        SCALING_VECTOR_SIZE = 32

        SEQ_LEN = 4
        HIDDEN_SIZE = 512
        INTERMEDIATE_SIZE = 512
        NUM_EXPERTS = 4
        TOP_K = 2
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
        x_sf_global = 448 / x.abs().max().float()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=torch.float32,
                                    device="cpu")
            w1_sf_global = (448) / w1_weight.abs().max().float()

            w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                    dtype=torch.float32,
                                    device="cpu")
            w2_sf_global = (448) / w2_weight.abs().max().float()

            w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                    dtype=torch.float32,
                                    device="cpu")
            w3_sf_global = (448) / w3_weight.abs().max().float()

            w3_w1_global = min(
                w1_sf_global,
                w3_sf_global)  # w3 global and w1 global must be the same

            w1_weight_nvfp4, w1_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w1_weight * w3_w1_global, SCALING_VECTOR_SIZE, 1, False)
            w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w1_sf_block.view(INTERMEDIATE_SIZE, -1))

            w2_weight_nvfp4, w2_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w2_weight * w2_sf_global, SCALING_VECTOR_SIZE, 1, False)
            w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w2_sf_block.view(HIDDEN_SIZE, -1))

            w3_weight_nvfp4, w3_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w3_weight * w3_w1_global, SCALING_VECTOR_SIZE, 1, False)
            w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w3_sf_block.view(INTERMEDIATE_SIZE, -1))

            w1_weight_nvfp4 = w1_weight_nvfp4.cuda()
            w1_sf_block_unswizzled = w1_sf_block_unswizzled.cuda()
            w2_weight_nvfp4 = w2_weight_nvfp4.cuda()
            w2_sf_block_unswizzled = w2_sf_block_unswizzled.cuda()
            w3_weight_nvfp4 = w3_weight_nvfp4.cuda()
            w3_sf_block_unswizzled = w3_sf_block_unswizzled.cuda()

            w1_input_scale = x_sf_global.cuda()
            w2_input_scale = x_sf_global.cuda()
            w3_input_scale = x_sf_global.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
            weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
            weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
            weights[
                f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[
                f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[
                f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
                    torch.float8_e4m3fn).cuda()
            weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
            weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
            weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
            weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global

        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_NVFP4_FP8)
        fused_moe = TRTLLMGenFusedMoE(num_experts=NUM_EXPERTS,
                                      routing_method=routing_method,
                                      hidden_size=HIDDEN_SIZE,
                                      intermediate_size=INTERMEDIATE_SIZE,
                                      dtype=dtype,
                                      reduce_results=False,
                                      model_config=ModelConfig(
                                          quant_config=quant_config,
                                          moe_backend=moe_backend))
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
        ref_fused_moe = RefGatedMLPFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            model_config=ModelConfig(quant_config=quant_config))
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref_fused_moe.forward(x, router_logits)

        # compare
        torch.cuda.synchronize()
        torch.testing.assert_close(output, ref_output, rtol=1e-1, atol=0.5)


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
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        torch.cuda.synchronize()
        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref()

        torch.cuda.synchronize()
        # assert that result does not contain NaN or is all 0s
        assert not torch.isnan(ref_output).any(), "ref_output contains NaN"
        assert not torch.isnan(output).any(), "output contains NaN"
        assert torch.nonzero(output).numel() > 0, "output is empty"
        assert torch.nonzero(ref_output).numel() > 0, "ref_output is empty"
        # compare
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "moe_backend",
    [pytest.param("TRTLLM", marks=skip_blackwell_geforce), "CUTLASS"])
@pytest.mark.parametrize("bias", [True, False])
def test_fused_moe_mxfp4_mxfp8(moe_backend, bias):

    SCALING_VECTOR_SIZE = 32
    dtype = torch.bfloat16
    SEQ_LEN = 128
    HIDDEN_SIZE = 256
    INTERMEDIATE_SIZE = 256
    NUM_EXPERTS = 8
    TOP_K = 4
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda() * 0.1
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        if bias:
            w1_bias = torch.randn(
                (INTERMEDIATE_SIZE, ), dtype=dtype).cuda() * 0.1
            w2_bias = torch.randn((HIDDEN_SIZE, ), dtype=dtype).cuda() * 0.1
            w3_bias = torch.randn(
                (INTERMEDIATE_SIZE, ), dtype=dtype).cuda() * 0.1
            weights[f"{expert_id}.w1.bias"] = w1_bias
            weights[f"{expert_id}.w2.bias"] = w2_bias
            weights[f"{expert_id}.w3.bias"] = w3_bias
        w1_weight = torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.1
        w2_weight = torch.randn(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda() * 0.1
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()

        w1_weight_mxfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
            w1_weight, None, SCALING_VECTOR_SIZE, True)
        w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
            w1_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        w2_weight_mxfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
            w2_weight, None, SCALING_VECTOR_SIZE, True)
        w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
            w2_sf_block.cpu().view(HIDDEN_SIZE, -1))

        w3_weight_mxfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
            w3_weight, None, SCALING_VECTOR_SIZE, True)
        w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
            w3_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        weights[f"{expert_id}.w1.weight"] = w1_weight_mxfp4
        weights[f"{expert_id}.w2.weight"] = w2_weight_mxfp4
        weights[f"{expert_id}.w3.weight"] = w3_weight_mxfp4
        weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
            torch.uint8).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
            torch.uint8).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
            torch.uint8).cuda()

    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    fused_moe = create_moe(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=quant_config,
                                 moe_backend=moe_backend),
        bias=bias,
    )
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        bias=bias,
        model_config=ModelConfig(quant_config=quant_config))
    ref_fused_moe.cuda()
    ref_fused_moe.load_weights([weights])

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)


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
        SEQ_LEN = 4
        HIDDEN_SIZE = hidden_size
        INTERMEDIATE_SIZE = 640
        SCALING_GROUP_SIZE = 32
        NUM_EXPERTS = 4
        TOP_K = 2
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
        fused_moe = create_moe(num_experts=NUM_EXPERTS,
                               routing_method=routing_method,
                               hidden_size=HIDDEN_SIZE,
                               intermediate_size=INTERMEDIATE_SIZE,
                               dtype=dtype,
                               reduce_results=False,
                               model_config=ModelConfig(
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
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        torch.cuda.synchronize()
        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref()

        # compare
        torch.cuda.synchronize()
        check_accuracy(output, ref_output, rtol=1e-2, atol=0.1, percent=0.99)


@skip_pre_hopper
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
    if not IS_TRITON_KERNELS_AVAILABLE:
        pytest.skip("Triton kernels are not available")
    if torch.cuda.get_device_capability()[0] < 10 and fp8_activation:
        pytest.skip("Latest Triton requires BF16 activation on Hopper")
    if torch.cuda.get_device_capability()[0] >= 10 and not fp8_activation:
        pytest.skip("Latest Triton requires FP8 activation on Blackwell")

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.int8])
def test_fused_moe_int8_woq_per_channel(dtype, weight_dtype):

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        SEQ_LEN = 4
        HIDDEN_SIZE = 768
        INTERMEDIATE_SIZE = 640
        NUM_EXPERTS = 3
        TOP_K = 2
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")

        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                    dtype=dtype,
                                    device="cuda")

        weight_id = 1  # 1 for w8a16, 2 for w4a16
        quant_config = QuantConfig(quant_algo=QuantAlgo.W8A16)
        weights = {}
        for expert_id in range(NUM_EXPERTS):
            w1_weight = torch.randint(
                -128,
                127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // weight_id),
                dtype=torch.int8).cuda()
            w2_weight = torch.randint(
                -128,
                127, (HIDDEN_SIZE, INTERMEDIATE_SIZE // weight_id),
                dtype=torch.int8).cuda()
            w3_weight = torch.randint(
                -128,
                127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // weight_id),
                dtype=torch.int8).cuda()

            w1_scale = torch.randn(
                (INTERMEDIATE_SIZE), dtype=dtype, device="cuda") / HIDDEN_SIZE
            w2_scale = torch.randn(
                (HIDDEN_SIZE), dtype=dtype, device="cuda") / INTERMEDIATE_SIZE
            w3_scale = torch.randn(
                (INTERMEDIATE_SIZE), dtype=dtype, device="cuda") / HIDDEN_SIZE

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
            weights[f"{expert_id}.w1.weight_scale"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_scale

        fused_moe = CutlassFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=False,
            model_config=ModelConfig(quant_config=quant_config))
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
                w1 = weights[f"{e_idx}.w1.weight"].T.contiguous().cuda()
                w2 = weights[f"{e_idx}.w2.weight"].T.contiguous().cuda()
                w3 = weights[f"{e_idx}.w3.weight"].T.contiguous().cuda()
                w3_w1 = torch.cat([w3, w1], dim=-1)
                # scales
                s1 = weights[f"{e_idx}.w1.weight_scale"].cuda()
                s2 = weights[f"{e_idx}.w2.weight_scale"].cuda()
                s3 = weights[f"{e_idx}.w3.weight_scale"].cuda()
                s3_s1 = torch.cat([s3, s1], dim=-1)
                # calculation
                w3_w1 = (w3_w1.float() * s3_s1).to(dtype)
                fc1 = torch.matmul(act, w3_w1)
                fc1, gate = fc1.chunk(2, dim=-1)
                act = fc1 * torch.nn.functional.silu(gate)
                w2 = (w2.float() * s2).to(dtype)
                fc2 = torch.matmul(act, w2)
                results[activated_tokens, :] += (fc2 * final_scale).to(
                    results.dtype)
            return results

        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)

        torch.cuda.synchronize()
        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref()

        # compare
        torch.cuda.synchronize()
        atol = calc_woq_tolerence(ref_output, weight_dtype)
        torch.testing.assert_close(output, ref_output, rtol=1e-7, atol=atol)


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig(),
                 use_cute_dsl_blockscaling_mm: bool = False,
                 bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=bias,
                dtype=self.dtype,
                config=model_config,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
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
