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
"""Which activations ``TritonFusedMoE`` claims during MoE resolution.

Triton gates the activation *family*, not ``swiglu_gptoss_style`` — see
``TritonFusedMoE.can_implement`` for why (nvbugs/6660905).

Selection-time only: these run without a cubin and need no GPU. Kernel-level
MXFP4 numerics live in ``_torch/modules/test_fused_moe.py``
(``test_fused_moe_triton_mxfp4``, which covers bias on and off).
"""

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonFusedMoE
from tensorrt_llm._torch.modules.fused_moe.impl_contract import MoEEnvironment, MoERejectReason
from tensorrt_llm._torch.modules.fused_moe.impl_environment import override_moe_environment
from tensorrt_llm._torch.modules.fused_moe.moe_resolution import (
    impl_class_for,
    infer_swiglu_gptoss_style,
    resolve_moe_impl,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _resolve_triton(*, bias=False, swiglu_alpha=None, swiglu_beta=None, activation_type=None):
    """Resolve a W4A16_MXFP4 TRITON request the way ``create_moe`` does.

    ``swiglu_gptoss_style`` goes through ``infer_swiglu_gptoss_style`` rather
    than being hard-coded, because ``resolve_moe_impl`` takes it as a parameter:
    passing a literal would leave the activation gate unreached and every
    assertion below would hold vacuously.
    """
    cfg = ModelConfig()
    cfg.moe_backend = "TRITON"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4)
    # SM90 is Triton's only window, so activation stays the lone variable.
    with override_moe_environment(MoEEnvironment(sm=90)):
        return resolve_moe_impl(
            cfg,
            dtype=torch.bfloat16,
            routing=RenormalizeMoeRoutingMethod(top_k=4),
            swiglu_gptoss_style=infer_swiglu_gptoss_style(
                bias=bias,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                activation_type=activation_type,
            ),
            bias=bias,
            activation_type=activation_type,
        )


def test_triton_serves_plain_swiglu_mxfp4():
    """Qwen3-30B-A3B W4A16_MXFP4: no bias, no alpha/beta -> Triton, not Cutlass."""
    report = _resolve_triton()
    # The regression degraded to Cutlass here and only failed later, at load.
    assert impl_class_for(report) is TritonFusedMoE
    assert report.selected_by == "pinned"
    assert not report.degraded


def test_triton_still_serves_gptoss_swiglu():
    """The gpt-oss package keeps resolving to Triton (the pre-existing case)."""
    report = _resolve_triton(
        bias=True,
        swiglu_alpha=torch.tensor([1.702]),
        swiglu_beta=torch.tensor([1.0]),
    )
    assert impl_class_for(report) is TritonFusedMoE
    assert not report.degraded


@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Geglu, id="geglu"),
        pytest.param(ActivationType.Relu2, id="relu2"),
    ],
)
def test_triton_degrades_on_non_swiglu_activation(activation_type):
    """A non-SwiGLU activation has no Triton path and must still degrade."""
    report = _resolve_triton(activation_type=activation_type)
    assert impl_class_for(report) is CutlassFusedMoE
    assert report.degraded
    assert report.degraded_from.reason is MoERejectReason.ACTIVATION_UNSUPPORTED
