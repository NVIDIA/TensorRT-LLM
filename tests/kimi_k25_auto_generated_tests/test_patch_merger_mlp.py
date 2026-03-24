# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
"""Module-level test for PatchMergerMLP: TRT-LLM vs HuggingFace."""

import sys
import types

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Import TRT-LLM PatchMergerMLP
# ---------------------------------------------------------------------------
sys.path.insert(
    0, "/home/scratch.huig_gpu/TensorRT-LLM_LQ")

from tensorrt_llm._torch.models.modeling_kimi_k25 import (  # noqa: E402
    PatchMergerMLP as TRTLLMPatchMergerMLP,
)

# ---------------------------------------------------------------------------
# 2. Import HuggingFace PatchMergerMLP from the checkpoint directory
# ---------------------------------------------------------------------------
# We load it dynamically because it is not an installed package.
HF_MODELING_PATH = (
    "/workspace/MiMo-V2-Flash/Kimi-K2.5-NVFP4/modeling_kimi_k25.py"
)


def _load_hf_module():
    """Dynamically import the HF modeling file and return PatchMergerMLP."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hf_modeling_kimi_k25", HF_MODELING_PATH)
    mod = importlib.util.module_from_spec(spec)
    # The HF file may import from transformers; ensure it is available.
    spec.loader.exec_module(mod)
    return mod.PatchMergerMLP


HFPatchMergerMLP = _load_hf_module()

# ---------------------------------------------------------------------------
# Constants matching Kimi-K2.5 config
# ---------------------------------------------------------------------------
MM_HIDDEN_SIZE = 1152
TEXT_HIDDEN_SIZE = 7168
MERGE_KERNEL_SIZE = (2, 2)
PROJECTOR_LN_EPS = 1e-5
HIDDEN_SIZE = MM_HIDDEN_SIZE * MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]
# = 4608


# ---------------------------------------------------------------------------
# Helper: create a fake HF vision config so HF PatchMergerMLP can init
# ---------------------------------------------------------------------------
def _make_hf_config():
    cfg = types.SimpleNamespace()
    cfg.mm_hidden_size = MM_HIDDEN_SIZE
    cfg.hidden_size = TEXT_HIDDEN_SIZE
    cfg.merge_kernel_size = MERGE_KERNEL_SIZE
    cfg.projector_ln_eps = PROJECTOR_LN_EPS
    return cfg


# ---------------------------------------------------------------------------
# Helper: copy weights from HF module to TRT-LLM module
# ---------------------------------------------------------------------------
def _copy_weights(src: nn.Module, dst: nn.Module):
    """Copy state_dict from *src* (HF) to *dst* (TRT-LLM).

    Both modules share the same parameter names because the TRT-LLM code
    mirrors the HF structure exactly (pre_norm, proj.0, proj.1, proj.2).
    """
    sd = src.state_dict()
    missing, unexpected = dst.load_state_dict(sd, strict=True)
    assert not missing, f"Missing keys when copying weights: {missing}"
    assert not unexpected, f"Unexpected keys when copying weights: {unexpected}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # Use fp32 for exact comparison


class TestPatchMergerMLPConsistency:
    """Compare TRT-LLM PatchMergerMLP against HuggingFace PatchMergerMLP."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)

        # -- HF module --
        hf_cfg = _make_hf_config()
        self.hf_module = HFPatchMergerMLP(hf_cfg).to(DEVICE, DTYPE).eval()

        # -- TRT-LLM module --
        self.trt_module = TRTLLMPatchMergerMLP(
            mm_hidden_size=MM_HIDDEN_SIZE,
            text_hidden_size=TEXT_HIDDEN_SIZE,
            merge_kernel_size=MERGE_KERNEL_SIZE,
            projector_ln_eps=PROJECTOR_LN_EPS,
        ).to(DEVICE, DTYPE).eval()

        # Copy weights so both modules have identical parameters.
        _copy_weights(self.hf_module, self.trt_module)

    # ------------------------------------------------------------------
    # Test 1: list-of-tensors input (normal inference path)
    # ------------------------------------------------------------------
    def test_list_input_single_image(self):
        """Single image: list with one tensor of shape (N, 4, 1152)."""
        N = 64  # number of merged patches
        K = MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]  # 4
        x = [torch.randn(N, K, MM_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)]

        with torch.no_grad():
            out_hf = self.hf_module(x)
            out_trt = self.trt_module(x)

        assert len(out_hf) == len(out_trt) == 1
        hf_tensor = out_hf[0]
        trt_tensor = out_trt[0]
        assert hf_tensor.shape == trt_tensor.shape, (
            f"Shape mismatch: HF={hf_tensor.shape}, TRT={trt_tensor.shape}")
        assert hf_tensor.shape == (N, TEXT_HIDDEN_SIZE), (
            f"Expected shape ({N}, {TEXT_HIDDEN_SIZE}), got {hf_tensor.shape}")
        close = torch.allclose(hf_tensor, trt_tensor, rtol=1e-3, atol=1e-3)
        if not close:
            diff = (hf_tensor - trt_tensor).abs()
            raise AssertionError(
                f"Output mismatch in test_list_input_single_image.\n"
                f"  max_diff={diff.max().item():.6e}, "
                f"mean_diff={diff.mean().item():.6e}\n"
                f"  HF output sample: {hf_tensor[0, :5]}\n"
                f"  TRT output sample: {trt_tensor[0, :5]}\n"
                f"  shapes: HF={hf_tensor.shape}, TRT={trt_tensor.shape}")

    # ------------------------------------------------------------------
    # Test 2: list-of-tensors input with multiple images (batch)
    # ------------------------------------------------------------------
    def test_list_input_multiple_images(self):
        """Multiple images: list with several tensors of varying N."""
        K = MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]
        sizes = [32, 64, 128]
        x = [
            torch.randn(n, K, MM_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
            for n in sizes
        ]

        with torch.no_grad():
            out_hf = self.hf_module(x)
            out_trt = self.trt_module(x)

        assert len(out_hf) == len(out_trt) == len(sizes)
        for i, n in enumerate(sizes):
            hf_t = out_hf[i]
            trt_t = out_trt[i]
            assert hf_t.shape == trt_t.shape == (n, TEXT_HIDDEN_SIZE), (
                f"Image {i}: shape mismatch HF={hf_t.shape}, TRT={trt_t.shape}")
            close = torch.allclose(hf_t, trt_t, rtol=1e-3, atol=1e-3)
            if not close:
                diff = (hf_t - trt_t).abs()
                raise AssertionError(
                    f"Output mismatch for image {i} (N={n}).\n"
                    f"  max_diff={diff.max().item():.6e}, "
                    f"mean_diff={diff.mean().item():.6e}")

    # ------------------------------------------------------------------
    # Test 3: batched tensor input (non-list path)
    # ------------------------------------------------------------------
    def test_tensor_input(self):
        """Batched tensor input of shape (B, N, K, C)."""
        B, N = 2, 48
        K = MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]
        x = torch.randn(B, N, K, MM_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            out_hf = self.hf_module(x)
            out_trt = self.trt_module(x)

        assert out_hf.shape == out_trt.shape, (
            f"Shape mismatch: HF={out_hf.shape}, TRT={out_trt.shape}")
        expected_shape = (B, N, TEXT_HIDDEN_SIZE)
        assert out_hf.shape == expected_shape, (
            f"Expected {expected_shape}, got {out_hf.shape}")
        close = torch.allclose(out_hf, out_trt, rtol=1e-3, atol=1e-3)
        if not close:
            diff = (out_hf - out_trt).abs()
            raise AssertionError(
                f"Output mismatch in test_tensor_input.\n"
                f"  max_diff={diff.max().item():.6e}, "
                f"mean_diff={diff.mean().item():.6e}\n"
                f"  HF shape={out_hf.shape}, TRT shape={out_trt.shape}")

    # ------------------------------------------------------------------
    # Test 4: verify state_dict key names match exactly
    # ------------------------------------------------------------------
    def test_state_dict_keys_match(self):
        """Both modules must produce identical state_dict key sets."""
        hf_keys = set(self.hf_module.state_dict().keys())
        trt_keys = set(self.trt_module.state_dict().keys())
        assert hf_keys == trt_keys, (
            f"state_dict key mismatch.\n"
            f"  HF only: {hf_keys - trt_keys}\n"
            f"  TRT only: {trt_keys - hf_keys}")

    # ------------------------------------------------------------------
    # Test 5: verify parameter shapes match exactly
    # ------------------------------------------------------------------
    def test_parameter_shapes_match(self):
        """All parameters must have identical shapes."""
        hf_sd = self.hf_module.state_dict()
        trt_sd = self.trt_module.state_dict()
        for key in hf_sd:
            assert key in trt_sd, f"Key {key} missing from TRT-LLM module"
            assert hf_sd[key].shape == trt_sd[key].shape, (
                f"Shape mismatch for {key}: "
                f"HF={hf_sd[key].shape}, TRT={trt_sd[key].shape}")

    # ------------------------------------------------------------------
    # Test 6: verify expected weight names from checkpoint
    # ------------------------------------------------------------------
    def test_expected_weight_names(self):
        """Checkpoint weight names (after prefix stripping) must be present."""
        expected_names = {
            "pre_norm.weight",
            "pre_norm.bias",
            "proj.0.weight",
            "proj.0.bias",
            "proj.2.weight",
            "proj.2.bias",
        }
        trt_keys = set(self.trt_module.state_dict().keys())
        assert expected_names == trt_keys, (
            f"Weight name mismatch.\n"
            f"  Expected: {sorted(expected_names)}\n"
            f"  Got: {sorted(trt_keys)}")
