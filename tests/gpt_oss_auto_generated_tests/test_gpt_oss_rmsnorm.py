# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Module-level test for RMSNorm: compares TRT-LLM RMSNorm against HF GptOssRMSNorm.

This test verifies numerical equivalence between the two implementations,
particularly around the dtype casting order difference:
  - HF GptOssRMSNorm: (weight * hidden_states_fp32).to(input_dtype)
  - TRT-LLM RMSNorm:  weight * hidden_states.to(input_dtype)
"""

import sys
import os
import pytest
import torch

# ---------------------------------------------------------------------------
# HF GptOssRMSNorm reference (standalone, no HF dependency needed)
# ---------------------------------------------------------------------------

class GptOssRMSNorm(torch.nn.Module):
    """Exact replica of HF GptOssRMSNorm from modular_gpt_oss.py."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        # GptOss: multiply in FP32, THEN cast
        return (self.weight * hidden_states).to(input_dtype)


# ---------------------------------------------------------------------------
# TRT-LLM RMSNorm reference (standalone fallback path, no flashinfer)
# ---------------------------------------------------------------------------

class TrtllmRMSNormReference(torch.nn.Module):
    """Mimics the generic (non-flashinfer, non-cuda_tile) path of TRT-LLM RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=dtype)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        # TRT-LLM: cast hidden_states FIRST, then multiply with weight
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return hidden_states


# ---------------------------------------------------------------------------
# Config constants matching GPT-OSS-20B
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2880
EPS = 1e-5
CHECKPOINT_PATH = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/"


def _load_hf_norm_weight(layer_idx: int = 0, norm_name: str = "input_layernorm"):
    """Load a norm weight from the HF checkpoint if available."""
    import json

    index_path = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None

    with open(index_path) as f:
        index = json.load(f)

    weight_name = f"model.layers.{layer_idx}.{norm_name}.weight"
    weight_map = index.get("weight_map", {})
    shard_file = weight_map.get(weight_name)
    if shard_file is None:
        return None

    from safetensors.torch import load_file

    shard_path = os.path.join(CHECKPOINT_PATH, shard_file)
    tensors = load_file(shard_path)
    return tensors.get(weight_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRMSNormCastingOrder:
    """Test the dtype casting order difference between HF and TRT-LLM."""

    @pytest.fixture
    def random_input_bf16(self):
        """Random BF16 input tensor (batch=2, seq=8, hidden=2880)."""
        torch.manual_seed(42)
        return torch.randn(2, 8, HIDDEN_SIZE, dtype=torch.bfloat16)

    @pytest.fixture
    def random_weight_bf16(self):
        """Random BF16 weight -- not all-ones, to expose casting differences."""
        torch.manual_seed(123)
        return torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)

    def test_fp32_exact_match(self, random_input_bf16):
        """In FP32 both formulations are identical."""
        x = random_input_bf16.float()

        hf_norm = GptOssRMSNorm(HIDDEN_SIZE, eps=EPS).float()
        trt_norm = TrtllmRMSNormReference(HIDDEN_SIZE, eps=EPS, dtype=torch.float32)

        # Use the same weight
        with torch.no_grad():
            w = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
            hf_norm.weight.copy_(w)
            trt_norm.weight.copy_(w)

        hf_out = hf_norm(x)
        trt_out = trt_norm(x)

        assert torch.allclose(hf_out, trt_out, atol=1e-6), (
            f"FP32 outputs differ! max diff = {(hf_out - trt_out).abs().max().item()}"
        )

    def test_bf16_casting_difference(self, random_input_bf16, random_weight_bf16):
        """
        In BF16, the casting order matters.
        HF: (weight_bf16 * hidden_fp32).to(bf16)  -- multiply in FP32
        TRT-LLM: weight_bf16 * hidden.to(bf16)    -- multiply in BF16

        This test documents the expected numerical difference.
        """
        x = random_input_bf16  # BF16

        hf_norm = GptOssRMSNorm(HIDDEN_SIZE, eps=EPS)
        trt_norm = TrtllmRMSNormReference(HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16)

        with torch.no_grad():
            hf_norm.weight.copy_(random_weight_bf16)
            trt_norm.weight.copy_(random_weight_bf16)

        hf_out = hf_norm(x)
        trt_out = trt_norm(x)

        max_diff = (hf_out.float() - trt_out.float()).abs().max().item()
        mean_diff = (hf_out.float() - trt_out.float()).abs().mean().item()

        # Report the difference magnitude
        print(f"\n[BF16 casting order test]")
        print(f"  max  abs diff: {max_diff:.6e}")
        print(f"  mean abs diff: {mean_diff:.6e}")
        print(f"  hf_out  sample: {hf_out[0, 0, :5]}")
        print(f"  trt_out sample: {trt_out[0, 0, :5]}")

        # The difference should be small (within BF16 precision) but non-zero
        # for non-trivial weights. atol=1e-2 is generous for BF16.
        close_enough = torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2)
        if not close_enough:
            raise AssertionError(
                f"BF16 outputs differ beyond BF16 tolerance! "
                f"max diff = {max_diff:.6e}, mean diff = {mean_diff:.6e}. "
                f"hf_out sample: {hf_out[0, 0, :5]}, "
                f"trt_out sample: {trt_out[0, 0, :5]}"
            )

        # Now test strict match -- this is expected to FAIL for non-trivial weights
        strict_match = torch.allclose(hf_out, trt_out, atol=0, rtol=0)
        if strict_match:
            print("  NOTE: Outputs are bitwise identical (unexpected for non-trivial BF16 weights)")
        else:
            print(f"  CONFIRMED: Casting order causes numerical difference (max={max_diff:.6e})")

    def test_bf16_with_checkpoint_weights(self, random_input_bf16):
        """
        Load actual checkpoint weights and compare outputs.
        Skipped if checkpoint is not available.
        """
        weight = _load_hf_norm_weight(layer_idx=0, norm_name="input_layernorm")
        if weight is None:
            pytest.skip("Checkpoint not available at " + CHECKPOINT_PATH)

        x = random_input_bf16
        weight_bf16 = weight.to(torch.bfloat16)

        hf_norm = GptOssRMSNorm(HIDDEN_SIZE, eps=EPS)
        trt_norm = TrtllmRMSNormReference(HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16)

        with torch.no_grad():
            hf_norm.weight.copy_(weight_bf16)
            trt_norm.weight.copy_(weight_bf16)

        hf_out = hf_norm(x)
        trt_out = trt_norm(x)

        max_diff = (hf_out.float() - trt_out.float()).abs().max().item()
        mean_diff = (hf_out.float() - trt_out.float()).abs().mean().item()

        print(f"\n[Checkpoint weight test]")
        print(f"  weight dtype: {weight.dtype}, shape: {weight.shape}")
        print(f"  max  abs diff: {max_diff:.6e}")
        print(f"  mean abs diff: {mean_diff:.6e}")

        # With real weights the difference should still be small
        assert torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2), (
            f"Outputs differ with real checkpoint weights! "
            f"max diff = {max_diff:.6e}, mean diff = {mean_diff:.6e}"
        )

        if max_diff > 0:
            print(f"  CONFIRMED: Casting order causes numerical difference with real weights")


class TestRMSNormWithTrtllmModule:
    """Test using the actual TRT-LLM RMSNorm module (requires tensorrt_llm)."""

    @pytest.fixture
    def random_input_bf16(self):
        torch.manual_seed(42)
        return torch.randn(2, 8, HIDDEN_SIZE, dtype=torch.bfloat16)

    def test_trtllm_rmsnorm_generic_path(self, random_input_bf16):
        """
        Compare HF GptOssRMSNorm vs actual TRT-LLM RMSNorm on CUDA.
        """
        try:
            from tensorrt_llm._torch.modules.rms_norm import RMSNorm as TrtRMSNorm
        except ImportError:
            pytest.skip("tensorrt_llm not importable")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = random_input_bf16.cuda()

        # Create both modules
        hf_norm = GptOssRMSNorm(HIDDEN_SIZE, eps=EPS).cuda()
        trt_norm = TrtRMSNorm(
            hidden_size=HIDDEN_SIZE,
            eps=EPS,
            dtype=torch.bfloat16,
            use_gemma=False,
        ).cuda()

        # Use identical non-trivial weights
        torch.manual_seed(999)
        w = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
        with torch.no_grad():
            hf_norm.weight.copy_(w)
            trt_norm.weight.copy_(w)

        hf_out = hf_norm(x)
        trt_out = trt_norm(x)

        max_diff = (hf_out.float() - trt_out.float()).abs().max().item()
        mean_diff = (hf_out.float() - trt_out.float()).abs().mean().item()

        print(f"\n[TRT-LLM RMSNorm module test]")
        print(f"  max  abs diff: {max_diff:.6e}")
        print(f"  mean abs diff: {mean_diff:.6e}")

        # This test documents the mismatch caused by casting order
        if max_diff > 0:
            print(f"  MISMATCH DETECTED: TRT-LLM RMSNorm does not exactly match "
                  f"GptOssRMSNorm due to dtype casting order.")
            print(f"  HF:      (weight * hidden_states_fp32).to(bf16)")
            print(f"  TRT-LLM: weight * hidden_states.to(bf16)")

        # We still expect them to be close within BF16 precision
        assert torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2), (
            f"TRT-LLM RMSNorm output differs too much from HF GptOssRMSNorm! "
            f"max diff = {max_diff:.6e}, mean diff = {mean_diff:.6e}. "
            f"This is due to casting order: HF multiplies weight*hidden in FP32, "
            f"TRT-LLM casts hidden to BF16 first."
        )

    def test_trtllm_rmsnorm_with_checkpoint_weights(self, random_input_bf16):
        """
        Compare with real checkpoint weights using actual TRT-LLM module.
        """
        try:
            from tensorrt_llm._torch.modules.rms_norm import RMSNorm as TrtRMSNorm
        except ImportError:
            pytest.skip("tensorrt_llm not importable")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        weight = _load_hf_norm_weight(layer_idx=0, norm_name="input_layernorm")
        if weight is None:
            pytest.skip("Checkpoint not available at " + CHECKPOINT_PATH)

        x = random_input_bf16.cuda()
        weight_bf16 = weight.to(torch.bfloat16)

        hf_norm = GptOssRMSNorm(HIDDEN_SIZE, eps=EPS).cuda()
        trt_norm = TrtRMSNorm(
            hidden_size=HIDDEN_SIZE,
            eps=EPS,
            dtype=torch.bfloat16,
            use_gemma=False,
        ).cuda()

        with torch.no_grad():
            hf_norm.weight.copy_(weight_bf16)
            trt_norm.weight.copy_(weight_bf16)

        hf_out = hf_norm(x)
        trt_out = trt_norm(x)

        max_diff = (hf_out.float() - trt_out.float()).abs().max().item()
        mean_diff = (hf_out.float() - trt_out.float()).abs().mean().item()

        print(f"\n[TRT-LLM RMSNorm with checkpoint weights]")
        print(f"  max  abs diff: {max_diff:.6e}")
        print(f"  mean abs diff: {mean_diff:.6e}")

        # Strict check at BF16 precision
        if not torch.allclose(hf_out, trt_out, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                f"TRT-LLM RMSNorm output differs from HF GptOssRMSNorm with real weights! "
                f"max diff = {max_diff:.6e}, mean diff = {mean_diff:.6e}. "
                f"Root cause: dtype casting order mismatch."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
