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

import torch
from torch import nn

from tensorrt_llm._torch.modules.rms_norm import RMSNorm


def _make_eagle3_for_causal_lm(hidden_size, num_capture_layers, dtype):
    """Build a minimal mock of Eagle3ForCausalLM with fc_norm enabled."""
    from types import SimpleNamespace

    model = SimpleNamespace()
    model.dtype = dtype
    model.hidden_size = hidden_size

    fc_norm_layers = nn.ModuleList(
        [RMSNorm(hidden_size=hidden_size, eps=1e-5, dtype=dtype) for _ in range(num_capture_layers)]
    ).cuda()
    model.fc_norm = fc_norm_layers

    model._norm_before_fc = True
    model.input_norm = RMSNorm(
        hidden_size=hidden_size * num_capture_layers, eps=1e-5, dtype=dtype
    ).cuda()

    fc_linear = nn.Linear(
        hidden_size * num_capture_layers, hidden_size, bias=False, dtype=dtype
    ).cuda()
    model.fc = fc_linear

    obj = SimpleNamespace()
    obj.model = model

    from tensorrt_llm._torch.models.modeling_speculative import Eagle3ForCausalLM

    obj.apply_eagle3_fc = Eagle3ForCausalLM.apply_eagle3_fc.__get__(obj)

    return obj


def test_apply_eagle3_fc_with_fc_norm():
    hidden_size = 16
    num_capture_layers = 3
    dtype = torch.bfloat16
    batch_size = 2

    obj = _make_eagle3_for_causal_lm(hidden_size, num_capture_layers, dtype)

    hidden_states = torch.randn(
        batch_size, hidden_size * num_capture_layers, device="cuda", dtype=dtype
    )

    # Track calls to each fc_norm layer
    norm_call_inputs = []
    original_norms = [n for n in obj.model.fc_norm]
    for i, norm in enumerate(original_norms):
        orig_forward = norm.forward

        def make_hook(idx, orig_fn):
            def hooked(x):
                norm_call_inputs.append((idx, x.clone()))
                return orig_fn(x)

            return hooked

        norm.forward = make_hook(i, orig_forward)

    # Patch input_norm to detect if it gets called (it should NOT)
    input_norm_called = []
    orig_input_norm_forward = obj.model.input_norm.forward

    def input_norm_spy(x):
        input_norm_called.append(True)
        return orig_input_norm_forward(x)

    obj.model.input_norm.forward = input_norm_spy

    result = obj.apply_eagle3_fc(hidden_states)

    # fc_norm chunks and normalizes each chunk
    assert len(norm_call_inputs) == num_capture_layers
    chunks = hidden_states.chunk(num_capture_layers, dim=-1)
    for idx, (call_idx, call_input) in enumerate(norm_call_inputs):
        assert call_idx == idx
        assert call_input.shape == (batch_size, hidden_size)
        assert torch.allclose(call_input, chunks[idx], atol=1e-5)

    # input_norm must NOT be called (fc_norm takes precedence over _norm_before_fc)
    assert len(input_norm_called) == 0

    # Result has correct shape from fc projection
    assert result.shape == (batch_size, hidden_size)
