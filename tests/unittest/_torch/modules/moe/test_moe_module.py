# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import os
import pickle
import sys
from contextlib import nullcontext

import cloudpickle
import pytest
import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig
from utils.util import getSMVersion

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def _skip_helper(quant_algo):
    if quant_algo == QuantAlgo.NVFP4 and getSMVersion() < 100:
        pytest.skip("This test is not supported in pre-Blackwell architecture")


def _test_moe_worker(
    moe_backend,
    dtype,
    quant_algo,
    mapping=None,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
):
    # Hardcode some parameters for testing
    # activation and weight related
    seq_len = 4
    top_k = 2
    num_experts = 8
    hidden_size = 512
    intermediate_size = 512

    # Other parameters
    finalize_fusion = True

    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()

    all_rank_num_tokens = [seq_len] * mapping.world_size

    torch.cuda.set_device(mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Create route method
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)

        # Create activation and weight
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        if enable_eplb:
            # Here we create same router_logits for all tokens to force the eplb update weights
            router_logits = torch.randn((1, num_experts), dtype=dtype, device="cuda").repeat(
                seq_len, 1
            )
        else:
            router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(quant_algo, x)
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        if enable_eplb:
            # Keep the tensor on CPU for eplb
            for key in weights:
                if isinstance(weights[key], torch.Tensor):
                    weights[key] = weights[key].to("cpu")

        # Deepcopy the CPU weight since when eplb turns on, fused moe may advise_tensor_pageout in post load weight.
        ref_weights = copy.deepcopy(weights) if enable_eplb else weights

        # Create pretrained config
        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        if enable_eplb:
            moe_load_balancer_config = MoeLoadBalancerConfig(
                num_slots=num_slots,
                layer_updates_per_iter=layer_updates_per_iter,
            )
        else:
            moe_load_balancer_config = None

        model_config = ModelConfig(
            pretrained_config=pretrained_config,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=moe_backend,
            moe_disable_finalize_fusion=not finalize_fusion,
            moe_load_balancer=moe_load_balancer_config,
        )

        moe_load_balancer = nullcontext()
        if enable_eplb:
            # A simple implementation of maybe_create_moe_load_balancer for unit test.
            ep_rank = model_config.mapping.moe_ep_rank
            ep_size = model_config.mapping.moe_ep_size
            model_config.moe_load_balancer.setup(ep_rank=ep_rank, ep_size=ep_size)
            moe_load_balancer = MoeLoadBalancer(
                ep_rank=ep_rank,
                ep_size=ep_size,
                layer_updates_per_iter=model_config.moe_load_balancer.layer_updates_per_iter,
            )

        with moe_load_balancer:
            # Create fused MoE module
            fused_moe = create_moe(
                routing_method=routing_method, reduce_results=True, model_config=model_config
            )

            fused_moe.load_weights([weights])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{mapping.rank}")

            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                moe_load_balancer.finalize_model()

            ref_fused_moe = quantize_util.create_ref_module(routing_method)
            ref_fused_moe.load_weights([ref_weights])
            ref_fused_moe.cuda(f"cuda:{mapping.rank}")

            # Evaluate the outputs
            def _run_forward(x, router_logits, skip_ref=False):
                with torch.inference_mode():
                    ref_output = None if skip_ref else ref_fused_moe.forward(x, router_logits)
                    if isinstance(moe_load_balancer, MoeLoadBalancer):
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            output = fused_moe.forward(
                                x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                            )
                    else:
                        output = fused_moe.forward(
                            x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                        )
                torch.cuda.synchronize()
                return ref_output, output

            load_expert_ids = None
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)
                load_expert_ids = moe_load_balancer.single_layer_load_balancers[
                    0
                ].get_old_rank_expert_ids()

            ref_output, output = _run_forward(x, router_logits)
            ref_fused_moe.check_accuracy(output, ref_output)

            if enable_eplb:
                # Multi iter run for eplb
                assert isinstance(moe_load_balancer, MoeLoadBalancer), (
                    "Moe load balancer should be created when eplb is enabled"
                )
                extra_steps = 3
                for _ in range(extra_steps):
                    _, output = _run_forward(x, router_logits, skip_ref=True)
                    ref_fused_moe.check_accuracy(output, ref_output)
                assert moe_load_balancer.iter_id == extra_steps + 1, (
                    "Iter id should be equal to extra steps + 1 after multiple iterations"
                )

                current_expert_ids = moe_load_balancer.single_layer_load_balancers[
                    0
                ].get_old_rank_expert_ids()
                assert load_expert_ids != current_expert_ids, (
                    "Expert ids after eplb update should be different from the initial loaded ones"
                )


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.NVFP4,
    ],
    ids=lambda val: f"quant_algo={val}",
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        "CUTLASS",
        "TRTLLM",
    ],
    ids=lambda val: f"moe_backend={val}",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
    ids=lambda val: f"dtype={val}",
)
def test_moe(dtype, moe_backend, quant_algo):
    # Enable configurable moe by default
    if moe_backend == "TRTLLM":
        if dtype == torch.float16 and quant_algo == QuantAlgo.NVFP4:
            pytest.skip("TRTLLM NVFP4 MoE backend does not support float16 yet")
    _skip_helper(quant_algo)

    _test_moe_worker(moe_backend=moe_backend, dtype=dtype, quant_algo=quant_algo)


def _test_moe_multi_gpu(
    comm_method_type,
    moe_backend,
    quant_algo,
    dtype,
    ep_size,
    world_size,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
):
    def init_worker(custom_paths, comm_method_type):
        # Update the sys.path to align with main process for submodule import
        for custom_path in custom_paths:
            if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
                sys.path.append(custom_path)

        # Set comm method
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = comm_method_type

    with MPIPoolExecutor(
        initializer=init_worker, initargs=(sys.path, comm_method_type), max_workers=world_size
    ) as executor:
        results = executor.map(
            _test_moe_worker,
            *zip(
                *[
                    (
                        moe_backend,
                        dtype,
                        quant_algo,
                        Mapping(
                            world_size=world_size,
                            tp_size=world_size,
                            moe_ep_size=ep_size,
                            moe_tp_size=world_size // ep_size,
                            enable_attention_dp=True,
                        ),
                        enable_eplb,
                        layer_updates_per_iter,
                        num_slots,
                    )
                ]
                * world_size
            ),
        )
        for r in results:
            assert r is None


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.NVFP4,
    ],
    ids=lambda val: f"quant_algo={val}",
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        "CUTLASS",
        "TRTLLM",
    ],
    ids=lambda val: f"moe_backend={val}",
)
@pytest.mark.parametrize(
    "comm_method_type",
    [
        "NVLINK_ONE_SIDED",
        "NVLINK_TWO_SIDED",
    ],
    ids=lambda val: f"comm_method_type={val}",
)
def test_moe_multi_gpu(comm_method_type, moe_backend, quant_algo):
    _skip_helper(quant_algo)

    dtype = torch.bfloat16
    ep_size = 4
    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        ep_size=ep_size,
        world_size=world_size,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.skipif(
    not _tbr.is_host_accessible_device_memory_supported(),
    reason="needs support of host accessible device memory",
)
@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.NVFP4,
    ],
    ids=lambda val: f"quant_algo={val}",
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        "CUTLASS",
    ],
    ids=lambda val: f"moe_backend={val}",
)
@pytest.mark.parametrize(
    "comm_method_type",
    [
        "NVLINK_ONE_SIDED",
    ],
    ids=lambda val: f"comm_method_type={val}",
)
@pytest.mark.parametrize(
    "num_slots",
    [
        16,
    ],
    ids=lambda val: f"num_slots={val}",
)
@pytest.mark.parametrize(
    "layer_updates_per_iter",
    [
        1,
    ],
    ids=lambda val: f"layer_updates_per_iter={val}",
)
def test_moe_multi_gpu_eplb(
    layer_updates_per_iter, num_slots, comm_method_type, moe_backend, quant_algo
):
    _skip_helper(quant_algo)

    dtype = torch.bfloat16
    ep_size = 4
    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        ep_size=ep_size,
        world_size=world_size,
        enable_eplb=True,
        layer_updates_per_iter=layer_updates_per_iter,
        num_slots=num_slots,
    )
