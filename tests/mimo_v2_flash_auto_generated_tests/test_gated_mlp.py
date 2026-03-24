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
"""
Module-level test: GatedMLP for MiMo-V2-Flash layer 0 (dense MLP).

Compares TRT-LLM GatedMLP output against a reference MiMoV2MLP implementation
using real checkpoint weights from layer 0.

The HF MiMoV2MLP is reimplemented inline to avoid importing the full
HuggingFace modeling code (which has transformers version constraints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from safetensors import safe_open

CHECKPOINT_DIR = "/workspace/MiMo-V2-Flash/MiMo-V2-Flash"
SHARD_FILE = f"{CHECKPOINT_DIR}/model_0.safetensors"

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 16384


# ---- Reference HF implementation (standalone, no transformers dependency) ----

class RefMiMoV2MLP(nn.Module):
    """Reference implementation of MiMoV2MLP (gate-up-down with SiLU)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


# ---- Utilities ----

def load_layer0_mlp_weights():
    """Load layer 0 MLP weights from the checkpoint."""
    f = safe_open(SHARD_FILE, framework="pt", device="cpu")
    weights = {}
    for name in [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale_inv",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale_inv",
    ]:
        weights[name] = f.get_tensor(name)
    return weights


def dequantize_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize FP8 block-scaled weight to BF16."""
    out_features, in_features = weight_fp8.shape
    weight_f32 = weight_fp8.to(torch.float32)
    n_blocks_out = scale_inv.shape[0]
    n_blocks_in = scale_inv.shape[1]

    result = torch.zeros(out_features, in_features, dtype=torch.float32)
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            r_start = i * block_size
            r_end = min((i + 1) * block_size, out_features)
            c_start = j * block_size
            c_end = min((j + 1) * block_size, in_features)
            result[r_start:r_end, c_start:c_end] = (
                weight_f32[r_start:r_end, c_start:c_end] * scale_inv[i, j]
            )
    return result.to(torch.bfloat16)


def create_ref_mlp(weights):
    """Create a reference MLP with dequantized weights."""
    ref_mlp = RefMiMoV2MLP(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    gate_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.gate_proj.weight"],
        weights["model.layers.0.mlp.gate_proj.weight_scale_inv"],
    )
    up_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.up_proj.weight"],
        weights["model.layers.0.mlp.up_proj.weight_scale_inv"],
    )
    down_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.down_proj.weight"],
        weights["model.layers.0.mlp.down_proj.weight_scale_inv"],
    )

    ref_mlp.gate_proj.weight = nn.Parameter(gate_w)
    ref_mlp.up_proj.weight = nn.Parameter(up_w)
    ref_mlp.down_proj.weight = nn.Parameter(down_w)
    return ref_mlp.to(torch.bfloat16)


def create_trtllm_mlp(weights):
    """Create a TRT-LLM GatedMLP with dequantized weights (no quantization)."""
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    trt_mlp = GatedMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        bias=False,
        dtype=torch.bfloat16,
    )

    gate_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.gate_proj.weight"],
        weights["model.layers.0.mlp.gate_proj.weight_scale_inv"],
    )
    up_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.up_proj.weight"],
        weights["model.layers.0.mlp.up_proj.weight_scale_inv"],
    )
    down_w = dequantize_fp8_block(
        weights["model.layers.0.mlp.down_proj.weight"],
        weights["model.layers.0.mlp.down_proj.weight_scale_inv"],
    )

    # TRT-LLM GatedMLP fuses gate and up into gate_up_proj
    gate_up_w = torch.cat([gate_w, up_w], dim=0)
    trt_mlp.gate_up_proj.weight = nn.Parameter(gate_up_w)
    trt_mlp.down_proj.weight = nn.Parameter(down_w)
    return trt_mlp.to(torch.bfloat16)


class TestGatedMLP:
    """Test GatedMLP consistency between reference HF impl and TRT-LLM."""

    @pytest.fixture(scope="class")
    def weights(self):
        return load_layer0_mlp_weights()

    @pytest.fixture(scope="class")
    def ref_mlp(self, weights):
        return create_ref_mlp(weights).eval()

    @pytest.fixture(scope="class")
    def trt_mlp(self, weights):
        return create_trtllm_mlp(weights).eval()

    def test_weight_shapes(self, ref_mlp, trt_mlp):
        """Verify weight shapes match expectations."""
        assert ref_mlp.gate_proj.weight.shape == (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        assert ref_mlp.up_proj.weight.shape == (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        assert ref_mlp.down_proj.weight.shape == (HIDDEN_SIZE, INTERMEDIATE_SIZE)

        assert trt_mlp.gate_up_proj.weight.shape == (
            2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
        ), f"TRT gate_up_proj shape mismatch: {trt_mlp.gate_up_proj.weight.shape}"
        assert trt_mlp.down_proj.weight.shape == (HIDDEN_SIZE, INTERMEDIATE_SIZE)

    def test_gate_up_fusion(self, weights, trt_mlp):
        """Verify gate+up fusion: top half = gate, bottom half = up."""
        gate_w = dequantize_fp8_block(
            weights["model.layers.0.mlp.gate_proj.weight"],
            weights["model.layers.0.mlp.gate_proj.weight_scale_inv"],
        )
        up_w = dequantize_fp8_block(
            weights["model.layers.0.mlp.up_proj.weight"],
            weights["model.layers.0.mlp.up_proj.weight_scale_inv"],
        )

        fused = trt_mlp.gate_up_proj.weight.data.cpu()
        assert torch.equal(fused[:INTERMEDIATE_SIZE], gate_w), (
            "Gate portion of fused weight does not match gate_proj"
        )
        assert torch.equal(fused[INTERMEDIATE_SIZE:], up_w), (
            "Up portion of fused weight does not match up_proj"
        )

    def test_output_match(self, ref_mlp, trt_mlp):
        """Compare reference and TRT-LLM MLP outputs on the same input."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_mlp(x)
            x_2d = x.view(-1, HIDDEN_SIZE)
            trt_out = trt_mlp(x_2d).view(1, 4, HIDDEN_SIZE)

        if not torch.allclose(ref_out, trt_out, atol=1e-2, rtol=1e-2):
            max_diff = (ref_out - trt_out).abs().max().item()
            mean_diff = (ref_out - trt_out).abs().mean().item()
            raise AssertionError(
                f"GatedMLP output mismatch!\n"
                f"  Ref output shape: {ref_out.shape}, TRT output shape: {trt_out.shape}\n"
                f"  Max absolute diff: {max_diff}\n"
                f"  Mean absolute diff: {mean_diff}\n"
                f"  Ref output sample: {ref_out[0, 0, :8]}\n"
                f"  TRT output sample: {trt_out[0, 0, :8]}"
            )

    def test_output_match_larger_input(self, ref_mlp, trt_mlp):
        """Compare outputs on a larger input to stress-test."""
        torch.manual_seed(123)
        x = torch.randn(2, 8, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_mlp(x)
            x_2d = x.view(-1, HIDDEN_SIZE)
            trt_out = trt_mlp(x_2d).view(2, 8, HIDDEN_SIZE)

        if not torch.allclose(ref_out, trt_out, atol=1e-2, rtol=1e-2):
            max_diff = (ref_out - trt_out).abs().max().item()
            mean_diff = (ref_out - trt_out).abs().mean().item()
            raise AssertionError(
                f"GatedMLP output mismatch on larger input!\n"
                f"  Max absolute diff: {max_diff}\n"
                f"  Mean absolute diff: {mean_diff}"
            )
