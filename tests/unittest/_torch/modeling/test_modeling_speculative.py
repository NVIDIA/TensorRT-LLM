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
"""Unit tests for Eagle3ForCausalLM.apply_eagle3_fc fc_norm branch.

Tests call the real production method (Eagle3ForCausalLM.apply_eagle3_fc) on a
minimal mock object to verify the per-chunk RMSNorm logic introduced for
Eagle 3.1 models with config.fc_norm=True.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_speculative import Eagle3ForCausalLM
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


class _FakeDraftModel(nn.Module):
    """Minimal stand-in for Eagle3DraftModel with attributes used by apply_eagle3_fc.

    Note: nn.Linear is used as a stand-in for the TRT-LLM Linear module since
    only the normalization path is being tested, not the linear implementation.
    """

    def __init__(
        self, hidden_size: int, num_capture_layers: int, use_fc_norm: bool, norm_before_fc: bool
    ):
        super().__init__()
        self.dtype = torch.float32
        self.hidden_size = hidden_size
        self.hidden_size_in = hidden_size
        self._norm_before_fc = norm_before_fc

        total_in = hidden_size * num_capture_layers
        # nn.Linear as stand-in for tensorrt_llm Linear; only normalization is tested.
        self.fc = nn.Linear(total_in, hidden_size, bias=False)
        # Initialize fc weights to identity-like for predictability
        nn.init.eye_(self.fc.weight[:hidden_size, :hidden_size])

        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size=hidden_size,
                        eps=1e-5,
                        dtype=torch.float32,
                    )
                    for _ in range(num_capture_layers)
                ]
            )
        else:
            self.fc_norm = None

        if norm_before_fc and not use_fc_norm:
            self.input_norm = RMSNorm(
                hidden_size=total_in,
                eps=1e-5,
                dtype=torch.float32,
            )
        else:
            self.input_norm = None


class _FakeEagle3ForCausalLM(nn.Module):
    """Minimal stand-in that holds a model attribute for apply_eagle3_fc.

    The actual apply_eagle3_fc method from Eagle3ForCausalLM is called as an
    unbound method on instances of this class, ensuring the production code
    path is exercised.
    """

    def __init__(self, model: _FakeDraftModel):
        super().__init__()
        self.model = model


@pytest.mark.parametrize("num_capture_layers", [2, 3])
def test_apply_eagle3_fc_with_fc_norm(num_capture_layers: int):
    """Verify apply_eagle3_fc applies per-chunk RMSNorm when fc_norm is enabled."""
    hidden_size = 64
    num_tokens = 4
    device = "cuda"

    # --- Build model with fc_norm enabled ---
    draft_model = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        use_fc_norm=True,
        norm_before_fc=False,
    ).to(device)
    eagle3 = _FakeEagle3ForCausalLM(draft_model).to(device)

    # Create input with shape (num_tokens, hidden_size * num_capture_layers)
    # Use non-unit values so normalization has a visible effect
    torch.manual_seed(42)
    input_hidden = torch.randn(num_tokens, hidden_size * num_capture_layers, device=device)

    # Call the REAL production method as an unbound method on our fake object
    output = Eagle3ForCausalLM.apply_eagle3_fc(eagle3, input_hidden.clone())

    # Assertion 1: output shape is (num_tokens, hidden_size)
    assert output.shape == (num_tokens, hidden_size), (
        f"Expected shape ({num_tokens}, {hidden_size}), got {output.shape}"
    )

    # Assertion 2: output matches manual per-chunk RMSNorm + fc computation
    chunks = input_hidden.chunk(num_capture_layers, dim=-1)
    normed_chunks = []
    for i, chunk in enumerate(chunks):
        normed_chunks.append(draft_model.fc_norm[i](chunk))
    normed_concat = torch.cat(normed_chunks, dim=-1)
    expected_output = draft_model.fc(normed_concat)

    assert torch.allclose(output, expected_output, atol=1e-5), (
        "apply_eagle3_fc output does not match manual per-chunk RMSNorm + fc"
    )

    # Assertion 3: normalization actually modifies the tensor -- the fc_norm
    # path produces different output than passing unnormalized input through fc
    raw_fc_output = draft_model.fc(input_hidden)
    differs = not torch.allclose(output, raw_fc_output, atol=1e-5)
    assert differs, (
        "fc_norm should produce different output than passing raw input through fc, "
        "i.e. normalization must actually modify the hidden states before projection"
    )

    # --- Assertion 4: fc_norm disabled path ---
    # Build model with fc_norm disabled, norm_before_fc also disabled
    draft_model_no_norm = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        use_fc_norm=False,
        norm_before_fc=False,
    ).to(device)
    # Copy fc weights so we can compare
    draft_model_no_norm.fc.weight.data.copy_(draft_model.fc.weight.data)
    eagle3_no_norm = _FakeEagle3ForCausalLM(draft_model_no_norm).to(device)

    output_no_norm = Eagle3ForCausalLM.apply_eagle3_fc(eagle3_no_norm, input_hidden.clone())

    # Without fc_norm, the output should equal raw fc(input) without normalization
    expected_no_norm = draft_model_no_norm.fc(input_hidden)
    assert torch.allclose(output_no_norm, expected_no_norm, atol=1e-5), (
        "Without fc_norm, apply_eagle3_fc should pass input directly to fc"
    )


if __name__ == "__main__":
    pytest.main([__file__])
