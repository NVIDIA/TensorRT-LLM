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
import os

import pytest
import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig
from utils.util import getSMVersion

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo


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
def test_moe(dtype, moe_backend, quant_algo, mocker):
    # Enable configurable moe by default
    mocker.patch.dict(os.environ, {"ENABLE_CONFIGURABLE_MOE": "1"})
    if moe_backend == "TRTLLM":
        if dtype == torch.float16 and quant_algo == QuantAlgo.NVFP4:
            pytest.skip("TRTLLM NVFP4 MoE backend does not support float16 yet")
    if quant_algo == QuantAlgo.NVFP4 and getSMVersion() < 100:
        pytest.skip("This test is not supported in pre-Blackwell architecture")

    # Hardcode some parameters for testing
    # activation and weight related
    seq_len = 4
    top_k = 2
    num_experts = 8
    hidden_size = 512
    intermediate_size = 512
    # Other parameters
    finalize_fusion = True

    # Create mapping for current rank
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # Create route method
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

        # Create activation and weight
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
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

        # Create pretrained config
        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        # Create fused MoE module
        fused_moe = create_moe(
            routing_method=routing_method,
            reduce_results=True,
            model_config=ModelConfig(
                pretrained_config=pretrained_config,
                quant_config=quant_config,
                moe_backend=moe_backend,
                moe_disable_finalize_fusion=not finalize_fusion,
            ),
        )
        fused_moe.load_weights([weights])
        fused_moe.post_load_weights()
        fused_moe.cuda()

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        # Evaluate the outputs
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)
            output = fused_moe.forward(x, router_logits)

        ref_fused_moe.check_accuracy(output, ref_output)
