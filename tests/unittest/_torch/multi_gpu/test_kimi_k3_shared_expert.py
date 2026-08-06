# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 tensor-parallel shared-expert tests."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.kimi_k3_moe._mlp import SituAndMul
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _run_shared_expert_rank(
    tensor_parallel_size,
    hidden_states,
    gate_weight,
    up_weight,
    down_weight,
    routed_latent_partials,
    routed_norm_weight,
    routed_up_weight,
    situ_beta,
    situ_linear_beta,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=rank,
        )
        model_config = ModelConfig(mapping=mapping, quant_config=QuantConfig())
        shared_expert = GatedMLP(
            hidden_size=hidden_states.shape[-1],
            intermediate_size=gate_weight.shape[0],
            bias=False,
            activation=SituAndMul(
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                use_fused_activation=True,
            ),
            dtype=hidden_states.dtype,
            config=model_config,
            reduce_output=True,
            is_shared_expert=True,
        )
        routed_all_reduce = AllReduce(mapping=mapping, dtype=hidden_states.dtype)
        shared_expert.gate_up_proj.load_weights([{"weight": gate_weight}, {"weight": up_weight}])
        shared_expert.down_proj.load_weights([{"weight": down_weight}])
        shared_expert.cuda()

        local_intermediate = gate_weight.shape[0] // tensor_parallel_size
        assert shared_expert.gate_up_proj.weight.shape == (
            2 * local_intermediate,
            hidden_states.shape[-1],
        )
        assert shared_expert.down_proj.weight.shape == (
            hidden_states.shape[-1],
            local_intermediate,
        )
        assert shared_expert.down_proj.reduce_output

        hidden_states = hidden_states.cuda()
        routed_latent_partial = routed_latent_partials[rank].cuda()
        main_event = torch.cuda.Event()
        shared_event = torch.cuda.Event()
        aux_stream = torch.cuda.Stream()
        main_event.record()
        with torch.cuda.stream(aux_stream):
            main_event.wait()
            shared_output = shared_expert(hidden_states)
            shared_event.record()
        shared_event.wait()
        routed_latent = routed_all_reduce(routed_latent_partial)
        routed_latent_float = routed_latent.float()
        routed_normalized = routed_latent_float * torch.rsqrt(
            routed_latent_float.square().mean(dim=-1, keepdim=True) + 1e-5
        )
        routed_normalized = (routed_normalized * routed_norm_weight.cuda()).to(hidden_states.dtype)
        output = shared_output + F.linear(routed_normalized, routed_up_weight.cuda())
        gate_up = torch.cat(
            [
                F.linear(hidden_states, gate_weight.cuda()),
                F.linear(hidden_states, up_weight.cuda()),
            ],
            dim=-1,
        )
        shared_reference = F.linear(
            SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)(gate_up),
            down_weight.cuda(),
        )
        routed_latent_reference = routed_latent_partials.sum(dim=0).cuda().float()
        routed_normalized_reference = routed_latent_reference * torch.rsqrt(
            routed_latent_reference.square().mean(dim=-1, keepdim=True) + 1e-5
        )
        routed_normalized_reference = (routed_normalized_reference * routed_norm_weight.cuda()).to(
            hidden_states.dtype
        )
        reference = shared_reference + F.linear(
            routed_normalized_reference, routed_up_weight.cuda()
        )
        torch.cuda.synchronize()
        # NCCL's bf16 reduction can differ from the host reference sum by one
        # bf16 ULP before the nonlinear routed RMSNorm.
        torch.testing.assert_close(output, reference, rtol=1.6e-2, atol=2e-3)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_kimi_k3_shared_allreduce_precedes_routed_allreduce(mpi_pool_executor):
    """Shared GatedMLP AR on the aux stream completes before routed AR."""
    torch.manual_seed(29)
    tensor_parallel_size = mpi_pool_executor.num_workers
    num_tokens, hidden_size, intermediate_size, latent_size = 8, 256, 384, 128
    dtype = torch.bfloat16
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype) * 0.5
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype) * 0.05
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype) * 0.05
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.05
    routed_latent_partials = (
        torch.randn(tensor_parallel_size, num_tokens, latent_size, dtype=dtype) * 0.05
    )
    routed_norm_weight = torch.randn(latent_size, dtype=torch.float32) * 0.05 + 1
    routed_up_weight = torch.randn(hidden_size, latent_size, dtype=dtype) * 0.05
    args = (
        tensor_parallel_size,
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
        routed_latent_partials,
        routed_norm_weight,
        routed_up_weight,
        4.0,
        25.0,
    )
    results = mpi_pool_executor.map(
        _run_shared_expert_rank,
        *zip(*[args] * tensor_parallel_size),
    )
    assert all(results)
