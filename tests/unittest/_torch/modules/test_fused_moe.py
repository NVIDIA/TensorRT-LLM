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
from _torch.helpers import (per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0,
                            per_token_cast_to_fp8_e8m0)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import (skip_neither_ada_nor_hopper_unittest,
                        skip_non_hopper_unittest, skip_pre_blackwell,
                        skip_pre_hopper)

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   CutlassFusedMoE,
                                                   DefaultMoeRoutingMethod,
                                                   RenormalizeMoeRoutingMethod,
                                                   VanillaMoE, WideEPMoE)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import \
    CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import \
    DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import \
    AlltoallMethodType
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
    "moe_cls, dtype, experts, RoutingMethodCls",
    product([CutlassFusedMoE, VanillaMoE], [torch.float16, torch.bfloat16],
            [3, 8, 512],
            [DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod]))
def test_fused_moe(moe_cls, dtype, experts, RoutingMethodCls, mapping=None):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = experts
    TOP_K = 2
    routing_method = RoutingMethodCls(top_k=TOP_K)
    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                dtype=dtype,
                                device="cuda")

    weights = {}
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
        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
    fused_moe = moe_cls(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(mapping=mapping),
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
                                        model_config=ModelConfig())
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
        torch.testing.assert_close(output, ref_output, rtol=0.5, atol=0.5)
        m //= 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("moe_cls", [CutlassFusedMoE, VanillaMoE])
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_fused_moe_multi_gpu(moe_cls, ep_size):
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            test_fused_moe,
            *zip(*[(moe_cls, torch.bfloat16, 512, DefaultMoeRoutingMethod,
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
                                         max_num_tokens=MAX_NUM_TOKENS),
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
                    all_rank_max_num_tokens=m,
                    use_dp_padding=False)
                ref_output = ref_model.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    all_rank_max_num_tokens=m,
                    use_dp_padding=False)

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


@skip_pre_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8(dtype):
    SEQ_LEN = 4
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
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
    fused_moe = CutlassFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=False,
        model_config=ModelConfig(quant_config=quant_config))
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
        model_config=ModelConfig(quant_config=quant_config))
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()
    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.2)


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
        b_sf=fused_moe.quant_scales[0],
        offset_array=expert_first_token_offset_tensor,
    )
    h2 = swiglu_fused_moe(h1)
    act_input_fp8, act_input_sf = per_token_cast_to_fp8_e8m0(h2)
    h3 = grouped_gemm(
        a=act_input_fp8,
        b=fused_moe.w2_weight,
        a_sf=act_input_sf,
        b_sf=fused_moe.quant_scales[1],
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


@skip_non_hopper_unittest
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
def test_fused_moe_fp8_blockwise(dtype,
                                 num_experts,
                                 seq_len,
                                 hidden_size,
                                 RoutingMethodCls,
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
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    fused_moe = CuteDslFusedMoE(
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

    fused_moe_origin = CutlassFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=quant_config, mapping=mapping),
    )
    fused_moe_origin.cuda()
    fused_moe_origin.load_weights([weights])

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
        output_origin = fused_moe_origin.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output_origin, output, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(output_origin, ref_output, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    return True


@skip_non_hopper_unittest
@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("routing_method", [DefaultMoeRoutingMethod])
def test_fused_moe_fp8_blockwise_multi_gpu(ep_size, routing_method):
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            test_fused_moe_fp8_blockwise,
            *zip(*[(
                torch.bfloat16,
                72,
                384,
                384,
                routing_method,
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
def test_fused_moe_nvfp4(dtype):
    SCALING_VECTOR_SIZE = 16

    SEQ_LEN = 4
    HIDDEN_SIZE = 128
    INTERMEDIATE_SIZE = 128
    NUM_EXPERTS = 3
    TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    x_sf_global = (448 * 6) / x.abs().max().float()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS),
                                dtype=dtype,
                                device="cuda")

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

        w1_weight_nvfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
            w1_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
        w1_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
            w1_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        w2_weight_nvfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
            w2_weight, w2_sf_global, SCALING_VECTOR_SIZE, False)
        w2_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
            w2_sf_block.cpu().view(HIDDEN_SIZE, -1))

        w3_weight_nvfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
            w3_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
        w3_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
            w3_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        w1_input_scale = x_sf_global.cuda()
        w2_input_scale = x_sf_global.cuda()
        w3_input_scale = x_sf_global.cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
        weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
        weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
        weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
        weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
        weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
        weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
        weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
        weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global

    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
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
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


@skip_neither_ada_nor_hopper_unittest
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_w4afp8(dtype):

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

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randint(-128,
                                  127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                  dtype=torch.int8).cuda()
        w2_weight = torch.randint(-128,
                                  127, (HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
                                  dtype=torch.int8).cuda()
        w3_weight = torch.randint(-128,
                                  127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                  dtype=torch.int8).cuda()

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

        w1_input = torch.randn(1, dtype=torch.float32, device="cuda") * 0.02
        w2_input = w1_input
        w3_input = w1_input

        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_scale
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_scale
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_scale
        weights[f"{expert_id}.w1.input_scale"] = w1_input
        weights[f"{expert_id}.w2.input_scale"] = w2_input
        weights[f"{expert_id}.w3.input_scale"] = w3_input

    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
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
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        for e_idx in range(NUM_EXPERTS):
            mask = selected_experts == e_idx
            activated_tokens = mask.sum(1).bool()
            act = x[activated_tokens, :]
            if act.shape[0] == 0:
                continue
            final_scale = (final_scales *
                           mask).sum(1)[activated_tokens].unsqueeze(1)

            # weights
            w1 = weights[f"{e_idx}.w1.weight"]
            w1 = unpacker(w1.cpu()).T.contiguous().cuda()
            w2 = weights[f"{e_idx}.w2.weight"]
            w2 = unpacker(w2.cpu()).T.contiguous().cuda()
            w3 = weights[f"{e_idx}.w3.weight"]
            w3 = unpacker(w3.cpu()).T.contiguous().cuda()
            w3_w1 = torch.cat([w3, w1], dim=-1)

            # scales
            s1 = weights[f"{e_idx}.w1.weight_scale_inv"].T.contiguous().cuda()
            s2 = weights[f"{e_idx}.w2.weight_scale_inv"].T.contiguous().cuda()
            s3 = weights[f"{e_idx}.w3.weight_scale_inv"].T.contiguous().cuda()
            s3_s1 = torch.cat([s3, s1], dim=-1)

            # prequant / alpha
            p1 = weights[f"{e_idx}.w1.input_scale"].cuda()
            p2 = weights[f"{e_idx}.w2.input_scale"].cuda()
            p3 = weights[f"{e_idx}.w3.input_scale"].cuda()
            p3_p1 = max(p1, p3)

            act = torch.clamp((act / p3_p1), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            w3_w1 = (w3_w1.float() *
                     s3_s1.repeat_interleave(128, dim=0).float()).to(dtype)
            fc1 = torch.matmul(act, w3_w1) * p3_p1
            fc1, gate = fc1.chunk(2, dim=-1)
            fc1 = fc1 * torch.nn.functional.silu(gate)

            act = torch.clamp((fc1 / p2), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            w2 = (w2.float() *
                  s2.repeat_interleave(128, dim=0).float()).to(dtype)
            fc2 = torch.matmul(act, w2) * p2
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
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=False,
                dtype=self.dtype,
                config=model_config,
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
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.NVFP4:
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

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)
