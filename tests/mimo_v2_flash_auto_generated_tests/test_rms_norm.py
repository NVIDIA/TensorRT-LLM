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
Module-level test: RMSNorm for MiMo-V2-Flash.

Compares TRT-LLM RMSNorm output against a reference MiMoV2RMSNorm
implementation using real checkpoint weights.

The HF MiMoV2RMSNorm is reimplemented inline to avoid importing the full
HuggingFace modeling code (which has transformers version constraints).

Tests:
1. input_layernorm (layer 0) -- standalone forward (no residual)
2. post_attention_layernorm (layer 0) -- standalone forward
3. Fused residual+norm pattern
4. Final model.norm
"""

import torch
import torch.nn as nn
import pytest

from safetensors import safe_open

CHECKPOINT_DIR = "/workspace/MiMo-V2-Flash/MiMo-V2-Flash"
SHARD_LAYER0 = f"{CHECKPOINT_DIR}/model_0.safetensors"
SHARD_FINAL = f"{CHECKPOINT_DIR}/model_final.safetensors"

HIDDEN_SIZE = 4096
EPS = 1e-5


# ---- Reference HF implementation (standalone) ----

class RefMiMoV2RMSNorm(nn.Module):
    """Reference implementation of MiMoV2RMSNorm (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


# ---- Utilities ----

def load_norm_weight(shard_file, weight_key):
    """Load a single norm weight from a safetensors shard."""
    f = safe_open(shard_file, framework="pt", device="cpu")
    return f.get_tensor(weight_key)


def create_ref_rmsnorm(weight):
    """Create reference RMSNorm with given weight."""
    ref_norm = RefMiMoV2RMSNorm(HIDDEN_SIZE, eps=EPS)
    ref_norm.weight = nn.Parameter(weight.clone().to(torch.bfloat16))
    return ref_norm.to(torch.bfloat16)


def create_trtllm_rmsnorm(weight):
    """Create TRT-LLM RMSNorm with given weight."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    trt_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=EPS,
        dtype=torch.bfloat16,
    )
    trt_norm.weight = nn.Parameter(weight.clone().to(torch.bfloat16))
    return trt_norm


class TestRMSNormInputLayernorm:
    """Test input_layernorm from layer 0."""

    @pytest.fixture(scope="class")
    def weight(self):
        return load_norm_weight(
            SHARD_LAYER0, "model.layers.0.input_layernorm.weight"
        )

    @pytest.fixture(scope="class")
    def ref_norm(self, weight):
        return create_ref_rmsnorm(weight).eval()

    @pytest.fixture(scope="class")
    def trt_norm(self, weight):
        return create_trtllm_rmsnorm(weight).eval()

    def test_weight_shapes(self, ref_norm, trt_norm):
        """Verify weight shapes."""
        assert ref_norm.weight.shape == (HIDDEN_SIZE,)
        assert trt_norm.weight.shape == (HIDDEN_SIZE,)

    def test_standalone_forward(self, ref_norm, trt_norm):
        """Compare standalone forward (no residual)."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_norm(x)
            trt_out = trt_norm(x)

        if not torch.allclose(ref_out, trt_out, atol=1e-3, rtol=1e-3):
            max_diff = (ref_out - trt_out).abs().max().item()
            mean_diff = (ref_out - trt_out).abs().mean().item()
            raise AssertionError(
                f"RMSNorm (input_layernorm) standalone output mismatch!\n"
                f"  Max absolute diff: {max_diff}\n"
                f"  Mean absolute diff: {mean_diff}\n"
                f"  Ref output sample: {ref_out[0, 0, :8]}\n"
                f"  TRT output sample: {trt_out[0, 0, :8]}"
            )


class TestRMSNormPostAttentionLayernorm:
    """Test post_attention_layernorm from layer 0."""

    @pytest.fixture(scope="class")
    def weight(self):
        return load_norm_weight(
            SHARD_LAYER0, "model.layers.0.post_attention_layernorm.weight"
        )

    @pytest.fixture(scope="class")
    def ref_norm(self, weight):
        return create_ref_rmsnorm(weight).eval()

    @pytest.fixture(scope="class")
    def trt_norm(self, weight):
        return create_trtllm_rmsnorm(weight).eval()

    def test_standalone_forward(self, ref_norm, trt_norm):
        """Compare standalone forward."""
        torch.manual_seed(99)
        x = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_norm(x)
            trt_out = trt_norm(x)

        if not torch.allclose(ref_out, trt_out, atol=1e-3, rtol=1e-3):
            max_diff = (ref_out - trt_out).abs().max().item()
            raise AssertionError(
                f"RMSNorm (post_attention_layernorm) output mismatch!\n"
                f"  Max absolute diff: {max_diff}"
            )


class TestRMSNormFusedResidual:
    """Test TRT-LLM fused residual+norm pattern used in decoder layer forward.

    In the TRT-LLM decoder layer:
      hidden_states, residual = self.input_layernorm(hidden_states, residual)

    This is equivalent to the HF pattern:
      new_residual = hidden_states + residual
      normed = rmsnorm(new_residual)
    """

    @pytest.fixture(scope="class")
    def weight(self):
        return load_norm_weight(
            SHARD_LAYER0, "model.layers.0.input_layernorm.weight"
        )

    @pytest.fixture(scope="class")
    def trt_norm(self, weight):
        return create_trtllm_rmsnorm(weight).eval()

    @pytest.fixture(scope="class")
    def ref_norm(self, weight):
        return create_ref_rmsnorm(weight).eval()

    def test_fused_residual_norm(self, ref_norm, trt_norm):
        """Compare fused residual+norm against manual reference equivalent."""
        torch.manual_seed(77)
        hidden_states = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)
        residual = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            # TRT-LLM fused path
            trt_normed, trt_residual = trt_norm(hidden_states, residual)

            # Reference equivalent (manual)
            ref_residual = hidden_states.float() + residual.float()
            ref_residual_bf16 = ref_residual.to(torch.bfloat16)
            ref_normed = ref_norm(ref_residual_bf16)

        if not torch.allclose(
            trt_residual, ref_residual_bf16, atol=1e-3, rtol=1e-3
        ):
            max_diff = (trt_residual - ref_residual_bf16).abs().max().item()
            raise AssertionError(
                f"Fused residual mismatch! Max diff: {max_diff}"
            )

        if not torch.allclose(trt_normed, ref_normed, atol=1e-3, rtol=1e-3):
            max_diff = (trt_normed - ref_normed).abs().max().item()
            raise AssertionError(
                f"Fused norm output mismatch! Max diff: {max_diff}"
            )


class TestRMSNormFinalNorm:
    """Test model.norm (final normalization layer)."""

    @pytest.fixture(scope="class")
    def weight(self):
        return load_norm_weight(SHARD_FINAL, "model.norm.weight")

    @pytest.fixture(scope="class")
    def ref_norm(self, weight):
        return create_ref_rmsnorm(weight).eval()

    @pytest.fixture(scope="class")
    def trt_norm(self, weight):
        return create_trtllm_rmsnorm(weight).eval()

    def test_final_norm_output(self, ref_norm, trt_norm):
        """Compare final norm outputs."""
        torch.manual_seed(55)
        x = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_norm(x)
            trt_out = trt_norm(x)

        if not torch.allclose(ref_out, trt_out, atol=1e-3, rtol=1e-3):
            max_diff = (ref_out - trt_out).abs().max().item()
            raise AssertionError(
                f"Final norm output mismatch! Max diff: {max_diff}"
            )

    def test_final_norm_fused_residual(self, ref_norm, trt_norm):
        """Test final norm with fused residual (as used in model forward:
        hidden_states, _ = self.norm(hidden_states, residual))."""
        torch.manual_seed(33)
        hidden_states = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)
        residual = torch.randn(1, 4, HIDDEN_SIZE, dtype=torch.bfloat16)

        with torch.no_grad():
            trt_normed, trt_residual = trt_norm(hidden_states, residual)

            ref_residual = hidden_states.float() + residual.float()
            ref_residual_bf16 = ref_residual.to(torch.bfloat16)
            ref_normed = ref_norm(ref_residual_bf16)

        if not torch.allclose(trt_normed, ref_normed, atol=1e-3, rtol=1e-3):
            max_diff = (trt_normed - ref_normed).abs().max().item()
            raise AssertionError(
                f"Final norm fused residual output mismatch! Max diff: {max_diff}"
            )
