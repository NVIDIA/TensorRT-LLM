# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Eagle3ForCausalLM.apply_eagle3_fc fc_norm branch."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_speculative import Eagle3ForCausalLM
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


class _FakeDraftModel(nn.Module):
    """Minimal stand-in for Eagle3DraftModel with attributes used by apply_eagle3_fc."""

    def __init__(self, hidden_size, num_capture_layers, dtype, use_fc_norm, use_norm_before_fc):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self._norm_before_fc = use_norm_before_fc

        in_features = hidden_size * num_capture_layers

        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size=hidden_size,
                        eps=1e-5,
                        dtype=dtype,
                    )
                    for _ in range(num_capture_layers)
                ]
            ).cuda()
        else:
            self.fc_norm = None

        if use_norm_before_fc:
            self.input_norm = RMSNorm(
                hidden_size=in_features,
                eps=1e-5,
                dtype=dtype,
            ).cuda()
        else:
            self.input_norm = None

        self.fc = nn.Linear(in_features, hidden_size, bias=False, dtype=dtype).cuda()


class _FakeEagle3Wrapper:
    """Minimal wrapper that has a .model attribute for calling the real apply_eagle3_fc."""

    def __init__(self, model):
        self.model = model


@pytest.mark.parametrize("num_capture_layers", [2, 3], ids=["2_layers", "3_layers"])
def test_apply_eagle3_fc_with_fc_norm(num_capture_layers):
    """Test that apply_eagle3_fc correctly chunks, normalizes per-chunk, and projects."""
    torch.manual_seed(42)

    hidden_size = 64
    dtype = torch.float32
    batch_size = 4
    device = "cuda"

    model = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        dtype=dtype,
        use_fc_norm=True,
        use_norm_before_fc=True,  # Should be ignored when fc_norm is set
    )

    wrapper = _FakeEagle3Wrapper(model)

    # Input: concatenated hidden states from multiple capture layers
    in_features = hidden_size * num_capture_layers
    hidden_states = torch.randn(batch_size, in_features, dtype=dtype, device=device)

    # Call the REAL production method as an unbound method
    result = Eagle3ForCausalLM.apply_eagle3_fc(wrapper, hidden_states)

    # Assert output shape is (batch_size, hidden_size)
    assert result.shape == (batch_size, hidden_size), (
        f"Expected shape ({batch_size}, {hidden_size}), got {result.shape}"
    )

    # Verify the fc_norm path was taken by manually computing expected result
    chunks = hidden_states.chunk(num_capture_layers, dim=-1)
    assert len(chunks) == num_capture_layers, (
        f"Expected {num_capture_layers} chunks, got {len(chunks)}"
    )

    # Each chunk should have size hidden_size
    for i, chunk in enumerate(chunks):
        assert chunk.shape == (batch_size, hidden_size), f"Chunk {i} shape mismatch: {chunk.shape}"

    # Manually apply per-chunk norm and concat
    normed_chunks = []
    for norm, chunk in zip(model.fc_norm, chunks):
        normed_chunks.append(norm(chunk))
    normed_concat = torch.cat(normed_chunks, dim=-1)

    # Apply fc
    expected = model.fc(normed_concat)

    # Results should match exactly (same computation path)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # Verify fc_norm takes priority over _norm_before_fc:
    # If we disable fc_norm and enable _norm_before_fc, result should differ
    torch.manual_seed(42)
    model_no_fc_norm = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        dtype=dtype,
        use_fc_norm=False,
        use_norm_before_fc=True,
    )
    # Share same fc weights so only the normalization differs
    model_no_fc_norm.fc.weight.data.copy_(model.fc.weight.data)

    wrapper_no_fc_norm = _FakeEagle3Wrapper(model_no_fc_norm)
    result_norm_before_fc = Eagle3ForCausalLM.apply_eagle3_fc(wrapper_no_fc_norm, hidden_states)

    # The two paths should produce different results (different normalization)
    # Per-chunk norm != whole-tensor norm for non-trivial inputs
    assert not torch.allclose(result, result_norm_before_fc, atol=1e-5), (
        "fc_norm path and _norm_before_fc path produced identical results; "
        "fc_norm should apply per-chunk normalization which differs from "
        "whole-tensor normalization"
    )
