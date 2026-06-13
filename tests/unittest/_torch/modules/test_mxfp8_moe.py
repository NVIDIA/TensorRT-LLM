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

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    FusedMoEQuantScalesMXFP8,
    MXFP8CutlassFusedMoEMethod,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def test_mxfp8_moe_quant_scales_tuple_shape():
    """FusedMoEQuantScalesMXFP8 must expose fc31_weight_block_scale and
    fc2_weight_block_scale as its two fields (order matters: the eventual
    fused_moe op call constructs `list(self.quant_scales)`).
    """
    t = FusedMoEQuantScalesMXFP8(
        fc31_weight_block_scale=torch.empty(0, dtype=torch.uint8),
        fc2_weight_block_scale=torch.empty(0, dtype=torch.uint8),
    )
    assert t._fields == ("fc31_weight_block_scale", "fc2_weight_block_scale")


def test_mxfp8_moe_supported_on_sm_in_table():
    """can_implement(QuantAlgo.MXFP8) must return True on Blackwell sm100/103."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    sm_major = torch.cuda.get_device_capability()[0]
    if sm_major < 10:
        pytest.skip("MXFP8 MoE requires sm100+")
    can, reason = CutlassFusedMoE.can_implement(QuantAlgo.MXFP8)
    assert can, f"MXFP8 MoE rejected by can_implement: {reason}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 MoE construct/load needs sm100+",
)
def test_mxfp8_moe_method_create_weights():
    """MXFP8CutlassFusedMoEMethod.create_weights must allocate e4m3 weights
    plus uint8 UE8M0 scale tensors with the expected per-expert shapes.
    """

    class _Stub(torch.nn.Module):
        # Minimal stand-in for a CutlassFusedMoE module: only the attributes
        # the method reads during create_weights need to exist.
        def __init__(self):
            super().__init__()
            self.expert_size_per_partition = 4
            self.hidden_size = 256
            self.intermediate_size_per_partition = 128
            self.expand_intermediate_size_per_partition = 2 * 128
            self.bias = False
            self.dtype = torch.bfloat16
            self.tp_size = 1
            self.tp_rank = 0
            self.weight_loading_mode = None
            self.initial_local_expert_ids = list(range(4))

    module = _Stub().cuda()
    method = MXFP8CutlassFusedMoEMethod()
    method.create_weights(module)

    assert module.w3_w1_weight.dtype == torch.float8_e4m3fn
    assert module.w2_weight.dtype == torch.float8_e4m3fn
    assert module.w3_w1_weight.shape == (4, 256, 256)
    assert module.w2_weight.shape == (4, 256, 128)

    # SF storage: int32-packed (4 UE8M0 per int32) matching MXFP4 MoE
    # convention. K chunked by BLOCK_SIZE*BLOCK_SCALES_VEC_SIZE = 32*4 = 128.
    assert module.w3_w1_weight_scale.dtype == torch.int32
    assert module.w2_weight_scale.dtype == torch.int32
    assert module.w3_w1_weight_scale.shape == (4, 256, 256 // 128)
    assert module.w2_weight_scale.shape == (4, 256, 128 // 128)

    # quant_scales should be a FusedMoEQuantScalesMXFP8 NamedTuple.
    assert isinstance(module.quant_scales, FusedMoEQuantScalesMXFP8)
    assert module.quant_scales.fc31_weight_block_scale is module.w3_w1_weight_scale
    assert module.quant_scales.fc2_weight_block_scale is module.w2_weight_scale


def test_mxfp8_moe_quant_config_dispatch_table_entry():
    """QuantConfig(QuantAlgo.MXFP8) must be recognized by the support table
    (separate from can_implement which also checks SM)."""
    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    assert qc.quant_algo == QuantAlgo.MXFP8
    assert QuantAlgo.MXFP8 in CutlassFusedMoE._QUANT_SUPPORT_TABLE


def _mxfp8_moe_kernel_available():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    return hasattr(torch.ops.trtllm, "fused_moe")


@pytest.mark.skipif(
    not _mxfp8_moe_kernel_available(), reason="MXFP8 MoE kernel needs sm100+ and built op"
)
def test_mxfp8_moe_forward_smoke():
    """End-to-end: build a small MXFP8 CutlassFusedMoE, load synthetic e4m3
    weights + UE8M0 scales, and run a forward pass. Verifies the new
    use_mxfp8_weight_scaling flag plumbs all the way down to the
    block-scaled MoE kernel and that the kernel executes without crash.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod, MoEWeightLoadingMode
    from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
    from tensorrt_llm.mapping import Mapping

    NUM_EXPERTS = 4
    HIDDEN_SIZE = 256
    INTERMEDIATE_SIZE = 128
    SEQ_LEN = 8
    TOP_K = 2
    dtype = torch.bfloat16

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    mapping = Mapping()
    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype)
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype)
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype)
        w1_e4m3, w1_sf = quant_bf16_to_mxfp8(w1, 32)
        w2_e4m3, w2_sf = quant_bf16_to_mxfp8(w2, 32)
        w3_e4m3, w3_sf = quant_bf16_to_mxfp8(w3, 32)
        weights[f"{expert_id}.w1.weight"] = w1_e4m3
        weights[f"{expert_id}.w2.weight"] = w2_e4m3
        weights[f"{expert_id}.w3.weight"] = w3_e4m3
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_sf
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_sf
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_sf

    fused_moe = CutlassFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(quant_config=qc, mapping=mapping),
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    )
    fused_moe.cuda()
    fused_moe.load_weights([weights])
    fused_moe.post_load_weights()

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype, device="cuda")

    with torch.inference_mode():
        out = fused_moe.forward(x, router_logits)

    torch.cuda.synchronize()
    assert out.shape == (SEQ_LEN, HIDDEN_SIZE), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "MXFP8 MoE output has non-finite values"
