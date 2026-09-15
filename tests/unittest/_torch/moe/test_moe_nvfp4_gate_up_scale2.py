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
"""CPU tests for NVFP4 MoE gate/up global weight-scale handling.

An NVFP4 checkpoint may store a different ``weight_scale_2`` for the gate (w1)
and up (w3) projections of the same expert. Kernels that take a single FC1
alpha have to reconcile the two into one value; the trtllm-gen kernels take
both, so they can reproduce the checkpoint exactly.

Everything here exercises the pure-Python weight-loading logic on small
synthetic tensors, so it runs on CPU.
"""

import torch

from tensorrt_llm._torch.moe.fused_moe import quantization
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4FusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod,
)


class _StubNVFP4Method(NVFP4FusedMoEMethod):
    """Concrete NVFP4 method with the block-scale loaders stubbed out.

    Only the alpha bookkeeping is under test, so these never run.
    """

    def load_expert_w3_w1_weight_scale_nvfp4(self, *args, **kwargs):
        raise NotImplementedError

    def load_expert_w2_weight_scale_nvfp4(self, *args, **kwargs):
        raise NotImplementedError


class _SingleAlphaMethod(_StubNVFP4Method):
    """Stand-in for a backend whose FC1 kernel takes one alpha (e.g. Cutlass)."""

    supports_split_gate_up_weight_scale_2 = False


class _SplitAlphaMethod(_StubNVFP4Method):
    """Stand-in for a backend whose FC1 kernel takes gate and up alphas."""

    supports_split_gate_up_weight_scale_2 = True


def _make_method(cls):
    # The methods under test only use `self` for class attributes and helper
    # methods, so skip __init__ (it wants a full MoE module).
    return cls.__new__(cls)


def _make_module(num_experts: int) -> torch.nn.Module:
    module = torch.nn.Module()
    module.fc31_input_scale = torch.nn.Parameter(torch.tensor(1.0 / 6.0), requires_grad=False)
    module.fc2_input_scale = torch.nn.Parameter(torch.tensor(1.0 / 3.0), requires_grad=False)
    module.fc31_alpha = torch.nn.Parameter(torch.zeros(num_experts), requires_grad=False)
    module.fc2_alpha = torch.nn.Parameter(torch.zeros(num_experts), requires_grad=False)
    return module


def _scales(w1: float, w3: float, w2: float) -> dict:
    return {
        "w1": torch.tensor(w1),
        "w3": torch.tensor(w3),
        "w2": torch.tensor(w2),
    }


def _expected_alpha(weight_scale_2: float, input_scale: torch.Tensor):
    # Same formula as load_expert_fc31_alpha_nvfp4 / load_expert_fc2_alpha_nvfp4.
    return 1.0 / (input_scale * (1.0 / torch.tensor(weight_scale_2)))


def test_equal_gate_up_scales_unchanged():
    """Equal gate/up scales: both backends produce the same single alpha."""
    tmp = {0: _scales(0.25, 0.25, 0.5), 1: _scales(0.125, 0.125, 0.5)}

    alphas = []
    for cls in (_SingleAlphaMethod, _SplitAlphaMethod):
        module = _make_module(2)
        up_alpha = torch.zeros(2)
        _make_method(cls)._reconcile_and_compute_alphas(
            module, tmp, module.fc31_alpha.data, module.fc2_alpha.data, dst_fc31_up_alpha=up_alpha
        )
        # The up half agrees with the gate half, so fc31_scale_c is unaffected.
        torch.testing.assert_close(up_alpha, module.fc31_alpha.data)
        alphas.append(module.fc31_alpha.data.clone())

    torch.testing.assert_close(alphas[0], alphas[1])
    torch.testing.assert_close(
        alphas[0],
        torch.stack(
            [
                _expected_alpha(0.25, module.fc31_input_scale.data),
                _expected_alpha(0.125, module.fc31_input_scale.data),
            ]
        ),
    )


def test_split_backend_keeps_both_scales():
    """A split-scale backend keeps the gate scale in fc31_alpha and the up
    scale in the separate up alpha, with no reconciliation."""
    module = _make_module(1)
    up_alpha = torch.zeros(1)
    tmp = {0: _scales(0.25, 0.5, 0.5)}

    _make_method(_SplitAlphaMethod)._reconcile_and_compute_alphas(
        module, tmp, module.fc31_alpha.data, module.fc2_alpha.data, dst_fc31_up_alpha=up_alpha
    )

    torch.testing.assert_close(
        module.fc31_alpha.data[0], _expected_alpha(0.25, module.fc31_input_scale.data)
    )
    torch.testing.assert_close(up_alpha[0], _expected_alpha(0.5, module.fc31_input_scale.data))
    assert not torch.allclose(module.fc31_alpha.data[0], up_alpha[0])


def test_split_backend_fc31_scale_c_uses_up_column():
    """fc31_scale_c rescales the gated intermediate, so it must come from the
    up half's alpha rather than the gate half's."""
    module = _make_module(1)
    module.expert_size_per_partition = 1
    module.fc31_scale_c = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    tmp = {0: _scales(0.25, 0.5, 0.5)}

    method = _make_method(NVFP4TRTLLMGenFusedMoEBaseMethod)
    assert method.supports_split_gate_up_weight_scale_2
    module.fc31_up_alpha = torch.zeros(1)
    method._reconcile_and_compute_alphas(
        module,
        tmp,
        module.fc31_alpha.data,
        module.fc2_alpha.data,
        dst_fc31_up_alpha=module.fc31_up_alpha,
    )
    up_alpha = module.fc31_up_alpha.clone()

    method._compute_fc31_scale_c(module)

    torch.testing.assert_close(module.fc31_scale_c.data, module.fc2_input_scale.data * up_alpha)
    assert not torch.allclose(
        module.fc31_scale_c.data, module.fc2_input_scale.data * module.fc31_alpha.data
    )
    # The staging tensor is dropped once consumed.
    assert not hasattr(module, "fc31_up_alpha")


def test_single_alpha_backend_reconciles_and_warns(monkeypatch):
    """A backend without split support falls back to the max() reconcile and
    says so."""
    module = _make_module(1)
    tmp = {0: _scales(0.25, 0.5, 0.5)}

    warnings = []
    monkeypatch.setattr(quantization.logger, "warning", warnings.append)
    _make_method(_SingleAlphaMethod)._reconcile_and_compute_alphas(
        module, tmp, module.fc31_alpha.data, module.fc2_alpha.data
    )

    torch.testing.assert_close(
        module.fc31_alpha.data[0], _expected_alpha(0.5, module.fc31_input_scale.data)
    )
    assert len(warnings) == 1
    assert "w1_weight_scale_2 != w3_weight_scale_2" in warnings[0]


def test_resolve_gate_up_weight_scale_2():
    """The resolve helper reconciles only for single-alpha backends."""
    gate = torch.tensor(0.25)
    up = torch.tensor(0.5)

    split = _make_method(_SplitAlphaMethod)._resolve_gate_up_weight_scale_2(gate, up)
    assert split[0] is gate and split[1] is up

    single = _make_method(_SingleAlphaMethod)._resolve_gate_up_weight_scale_2(gate, up)
    torch.testing.assert_close(single[0], up)
    torch.testing.assert_close(single[1], up)

    # Equal scales are returned untouched by both.
    for cls in (_SplitAlphaMethod, _SingleAlphaMethod):
        resolved = _make_method(cls)._resolve_gate_up_weight_scale_2(gate, gate)
        torch.testing.assert_close(resolved[0], gate)
        torch.testing.assert_close(resolved[1], gate)
