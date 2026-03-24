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
"""Module-level tests comparing TRT-LLM vision tower modules against HuggingFace.

Tests: Learnable2DInterpPosEmbDivided_fixed, MoonVision3dPatchEmbed,
       Rope2DPosEmbRepeated, MLP2, MoonViTEncoderLayer, MoonViT3dEncoder,
       tpool_patch_merger, MoonViT3dPretrainedModel, PatchMergerMLP.
"""

import sys
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Import TRT-LLM modules (the code under test)
# --------------------------------------------------------------------------- #
sys.path.insert(0, "/home/scratch.huig_gpu/TensorRT-LLM_LQ")
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    Learnable2DInterpPosEmbDivided_fixed as TRT_Learnable2DInterpPosEmbDivided_fixed,
    MoonVision3dPatchEmbed as TRT_MoonVision3dPatchEmbed,
    Rope2DPosEmbRepeated as TRT_Rope2DPosEmbRepeated,
    MLP2 as TRT_MLP2,
    MoonViTEncoderLayer as TRT_MoonViTEncoderLayer,
    MoonViT3dEncoder as TRT_MoonViT3dEncoder,
    tpool_patch_merger as trt_tpool_patch_merger,
    MoonViT3dPretrainedModel as TRT_MoonViT3dPretrainedModel,
    PatchMergerMLP as TRT_PatchMergerMLP,
    get_1d_sincos_pos_embed as trt_get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid as trt_get_1d_sincos_pos_embed_from_grid,
    _get_rope_shape as trt_get_rope_shape,
    _apply_rope_vision as trt_apply_rope_vision,
)

# --------------------------------------------------------------------------- #
# Import HuggingFace modules (the reference)
# --------------------------------------------------------------------------- #
sys.path.insert(0, "/workspace/MiMo-V2-Flash/Kimi-K2.5-NVFP4")
from modeling_kimi_k25 import (
    Learnable2DInterpPosEmbDivided_fixed as HF_Learnable2DInterpPosEmbDivided_fixed,
    MoonVision3dPatchEmbed as HF_MoonVision3dPatchEmbed,
    Rope2DPosEmbRepeated as HF_Rope2DPosEmbRepeated,
    MLP2 as HF_MLP2,
    MoonViTEncoderLayer as HF_MoonViTEncoderLayer,
    MoonViT3dEncoder as HF_MoonViT3dEncoder,
    tpool_patch_merger as hf_tpool_patch_merger,
    MoonViT3dPretrainedModel as HF_MoonViT3dPretrainedModel,
    PatchMergerMLP as HF_PatchMergerMLP,
    get_1d_sincos_pos_embed as hf_get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid as hf_get_1d_sincos_pos_embed_from_grid,
    apply_rope as hf_apply_rope,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # Use float32 for comparison precision


# =========================================================================== #
# Helper: copy state_dict between modules
# =========================================================================== #
def copy_weights(src: nn.Module, dst: nn.Module):
    """Copy all matching parameters and buffers from src to dst."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    matched = {}
    for k in dst_sd:
        if k in src_sd:
            assert src_sd[k].shape == dst_sd[k].shape, (
                f"Shape mismatch for {k}: src={src_sd[k].shape}, dst={dst_sd[k].shape}")
            matched[k] = src_sd[k]
    missing = set(dst_sd.keys()) - set(matched.keys())
    if missing:
        # Non-persistent buffers won't appear in state_dict, that is expected
        print(f"[copy_weights] keys in dst but not src (may be non-persistent buffers): {missing}")
    dst.load_state_dict(matched, strict=False)


# =========================================================================== #
# Test 1: get_1d_sincos_pos_embed_from_grid
# =========================================================================== #
class TestSincosHelper:
    def test_sincos_pos_embed_from_grid(self):
        embed_dim = 64
        pos = np.arange(10, dtype=np.float32)
        trt_out = trt_get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
        hf_out = hf_get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
        assert np.allclose(trt_out, hf_out, atol=1e-6), (
            f"get_1d_sincos_pos_embed_from_grid mismatch: max_diff={np.max(np.abs(trt_out - hf_out))}")

    def test_sincos_pos_embed(self):
        embed_dim = 1152
        t_size = 4
        trt_out = trt_get_1d_sincos_pos_embed(embed_dim, t_size)
        hf_out = hf_get_1d_sincos_pos_embed(embed_dim, t_size)
        assert np.allclose(trt_out, hf_out, atol=1e-6), (
            f"get_1d_sincos_pos_embed mismatch: max_diff={np.max(np.abs(trt_out - hf_out))}")


# =========================================================================== #
# Test 2: Learnable2DInterpPosEmbDivided_fixed
# =========================================================================== #
class TestLearnable2DInterpPosEmbDividedFixed:
    @pytest.fixture
    def modules(self):
        h, w, nf, dim = 64, 64, 4, 1152
        hf_mod = HF_Learnable2DInterpPosEmbDivided_fixed(h, w, nf, dim).to(DEVICE, DTYPE)
        trt_mod = TRT_Learnable2DInterpPosEmbDivided_fixed(h, w, nf, dim).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_same_resolution(self, modules):
        hf_mod, trt_mod = modules
        seq_len = 64 * 64  # h * w
        x = torch.randn(seq_len, 1152, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[1, 64, 64]], device=DEVICE)
        hf_out = hf_mod(x, grid_thws)
        trt_out = trt_mod(x, grid_thws)
        assert torch.allclose(hf_out, trt_out, rtol=1e-4, atol=1e-4), (
            f"Learnable2DInterpPosEmbDivided_fixed (same res) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}, "
            f"hf_shape={hf_out.shape}, trt_shape={trt_out.shape}")

    def test_different_resolution(self, modules):
        hf_mod, trt_mod = modules
        h, w = 32, 32
        seq_len = h * w
        x = torch.randn(seq_len, 1152, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[1, h, w]], device=DEVICE)
        hf_out = hf_mod(x, grid_thws)
        trt_out = trt_mod(x, grid_thws)
        assert torch.allclose(hf_out, trt_out, rtol=1e-3, atol=1e-3), (
            f"Learnable2DInterpPosEmbDivided_fixed (diff res) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}")

    def test_multi_frame(self, modules):
        hf_mod, trt_mod = modules
        t, h, w = 2, 64, 64
        seq_len = t * h * w
        x = torch.randn(seq_len, 1152, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[t, h, w]], device=DEVICE)
        hf_out = hf_mod(x, grid_thws)
        trt_out = trt_mod(x, grid_thws)
        assert torch.allclose(hf_out, trt_out, rtol=1e-4, atol=1e-4), (
            f"Learnable2DInterpPosEmbDivided_fixed (multi-frame) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}")


# =========================================================================== #
# Test 3: MoonVision3dPatchEmbed
# =========================================================================== #
class TestMoonVision3dPatchEmbed:
    @pytest.fixture
    def modules(self):
        hf_mod = HF_MoonVision3dPatchEmbed(
            out_dim=1152, patch_size=14,
            pos_emb_height=64, pos_emb_width=64, pos_emb_time=4
        ).to(DEVICE, DTYPE)
        trt_mod = TRT_MoonVision3dPatchEmbed(
            out_dim=1152, patch_size=14,
            pos_emb_height=64, pos_emb_width=64, pos_emb_time=4
        ).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward(self, modules):
        hf_mod, trt_mod = modules
        # 4 patches: 2x2 grid, each patch is 14x14 pixels
        num_patches = 4
        x = torch.randn(num_patches, 3, 14, 14, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[1, 2, 2]], device=DEVICE)
        hf_out = hf_mod(x, grid_thws)
        trt_out = trt_mod(x, grid_thws)
        assert torch.allclose(hf_out, trt_out, rtol=1e-4, atol=1e-4), (
            f"MoonVision3dPatchEmbed mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}, "
            f"hf_shape={hf_out.shape}, trt_shape={trt_out.shape}")


# =========================================================================== #
# Test 4: Rope2DPosEmbRepeated
# =========================================================================== #
class TestRope2DPosEmbRepeated:
    @pytest.fixture
    def modules(self):
        dim = 72  # hidden_dim / num_heads = 1152 / 16 = 72
        hf_mod = HF_Rope2DPosEmbRepeated(dim, 512, 512)
        trt_mod = TRT_Rope2DPosEmbRepeated(dim, 512, 512)
        return hf_mod, trt_mod

    def test_precompute_freqs_cis(self, modules):
        hf_mod, trt_mod = modules
        hf_cis = hf_mod._precompute_freqs_cis(torch.device(DEVICE))
        trt_cis = trt_mod._precompute_freqs_cis(torch.device(DEVICE))
        assert torch.allclose(hf_cis, trt_cis, rtol=1e-5, atol=1e-5), (
            f"Rope2DPosEmbRepeated._precompute_freqs_cis mismatch: "
            f"max_diff={torch.max(torch.abs(hf_cis - trt_cis)).item()}")

    def test_get_freqs_cis(self, modules):
        hf_mod, trt_mod = modules
        grid_thws = torch.tensor([[1, 4, 4]], device=DEVICE)
        hf_cis = hf_mod.get_freqs_cis(grid_thws, torch.device(DEVICE))
        trt_cis = trt_mod.get_freqs_cis(grid_thws, torch.device(DEVICE))
        assert torch.allclose(hf_cis, trt_cis, rtol=1e-5, atol=1e-5), (
            f"Rope2DPosEmbRepeated.get_freqs_cis mismatch: "
            f"max_diff={torch.max(torch.abs(hf_cis - trt_cis)).item()}")


# =========================================================================== #
# Test 5: apply_rope / _apply_rope_vision
# =========================================================================== #
class TestApplyRope:
    def test_apply_rope(self):
        num_heads = 16
        head_dim = 72
        seq_len = 16
        # Create query and key
        xq = torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        xk = torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        # Create freqs_cis (complex) of shape (seq_len, head_dim/2)
        dim_range = torch.arange(0, head_dim, 4)[:(head_dim // 4)].float().to(DEVICE)
        freqs = 1.0 / (10000 ** (dim_range / head_dim))
        pos = torch.arange(seq_len, dtype=torch.float32, device=DEVICE)
        x_freqs = torch.outer(pos, freqs)
        y_freqs = torch.outer(pos, freqs)
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        freqs_cis = freqs_cis.reshape(seq_len, -1)

        hf_q, hf_k = hf_apply_rope(xq, xk, freqs_cis)
        trt_q, trt_k = trt_apply_rope_vision(xq, xk, freqs_cis)
        assert torch.allclose(hf_q, trt_q, rtol=1e-4, atol=1e-4), (
            f"apply_rope Q mismatch: max_diff={torch.max(torch.abs(hf_q - trt_q)).item()}")
        assert torch.allclose(hf_k, trt_k, rtol=1e-4, atol=1e-4), (
            f"apply_rope K mismatch: max_diff={torch.max(torch.abs(hf_k - trt_k)).item()}")


# =========================================================================== #
# Test 6: MLP2
# =========================================================================== #
class TestMLP2:
    @pytest.fixture
    def modules(self):
        activation = nn.GELU(approximate='tanh')
        hf_mod = HF_MLP2([1152, 4304, 1152], activation).to(DEVICE, DTYPE)
        trt_mod = TRT_MLP2([1152, 4304, 1152], activation).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward(self, modules):
        hf_mod, trt_mod = modules
        x = torch.randn(16, 1152, device=DEVICE, dtype=DTYPE)
        hf_out = hf_mod(x)
        trt_out = trt_mod(x)
        assert torch.allclose(hf_out, trt_out, rtol=1e-4, atol=1e-4), (
            f"MLP2 mismatch: max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}")


# =========================================================================== #
# Test 7: MoonViTEncoderLayer
# =========================================================================== #
class TestMoonViTEncoderLayer:
    @pytest.fixture
    def modules(self):
        try:
            from transformers.activations import PytorchGELUTanh
        except ImportError:
            PytorchGELUTanh = lambda: nn.GELU(approximate='tanh')

        hf_mod = HF_MoonViTEncoderLayer(
            num_heads=16, hidden_dim=1152, mlp_dim=4304,
            activation=PytorchGELUTanh(), attn_bias=True,
            attn_implementation='eager',  # Use eager to avoid flash_attn dependency issues
        ).to(DEVICE, DTYPE)
        trt_mod = TRT_MoonViTEncoderLayer(
            num_heads=16, hidden_dim=1152, mlp_dim=4304,
            activation=PytorchGELUTanh(), attn_bias=True,
        ).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward(self, modules):
        hf_mod, trt_mod = modules
        seq_len = 16  # 4x4 grid
        x = torch.randn(seq_len, 1152, device=DEVICE, dtype=DTYPE)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
        max_seqlen = seq_len

        # Build freqs_cis for 4x4 grid
        dim = 72
        rope_mod = TRT_Rope2DPosEmbRepeated(dim, 512, 512)
        grid_thws = torch.tensor([[1, 4, 4]], device=DEVICE)
        freqs_cis = rope_mod.get_freqs_cis(grid_thws, torch.device(DEVICE))

        hf_out = hf_mod(x, cu_seqlens, max_seqlen, rope_freqs_cis=freqs_cis)
        trt_out = trt_mod(x, cu_seqlens, max_seqlen, rope_freqs_cis=freqs_cis)
        assert torch.allclose(hf_out, trt_out, rtol=1e-3, atol=1e-3), (
            f"MoonViTEncoderLayer mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}")


# =========================================================================== #
# Test 8: MoonViT3dEncoder
# =========================================================================== #
class TestMoonViT3dEncoder:
    @pytest.fixture
    def modules(self):
        try:
            from transformers.activations import PytorchGELUTanh
        except ImportError:
            PytorchGELUTanh = lambda: nn.GELU(approximate='tanh')

        block_cfg = {
            'num_heads': 16,
            'hidden_dim': 1152,
            'mlp_dim': 4304,
            'activation': PytorchGELUTanh(),
            'attn_bias': True,
        }
        # Use only 2 layers for speed
        hf_block_cfg = dict(block_cfg, attn_implementation='eager')
        hf_mod = HF_MoonViT3dEncoder(
            hidden_dim=1152, num_layers=2, block_cfg=hf_block_cfg
        ).to(DEVICE, DTYPE)
        trt_mod = TRT_MoonViT3dEncoder(
            hidden_dim=1152, num_layers=2, block_cfg=block_cfg
        ).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward(self, modules):
        hf_mod, trt_mod = modules
        seq_len = 16  # 4x4 grid
        x = torch.randn(seq_len, 1152, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[1, 4, 4]], device=DEVICE)
        hf_out = hf_mod(x, grid_thws)
        trt_out = trt_mod(x, grid_thws)
        assert torch.allclose(hf_out, trt_out, rtol=1e-3, atol=1e-3), (
            f"MoonViT3dEncoder mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out - trt_out)).item()}")


# =========================================================================== #
# Test 9: tpool_patch_merger
# =========================================================================== #
class TestTpoolPatchMerger:
    def test_single_frame(self):
        d = 1152
        t, h, w = 1, 4, 4
        x = torch.randn(t * h * w, d, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[t, h, w]], device=DEVICE)
        hf_out = hf_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        trt_out = trt_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        assert len(hf_out) == len(trt_out) == 1
        assert torch.allclose(hf_out[0], trt_out[0], rtol=1e-5, atol=1e-5), (
            f"tpool_patch_merger (single frame) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out[0] - trt_out[0])).item()}")

    def test_multi_frame(self):
        d = 1152
        t, h, w = 2, 4, 4
        x = torch.randn(t * h * w, d, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[t, h, w]], device=DEVICE)
        hf_out = hf_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        trt_out = trt_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        assert len(hf_out) == len(trt_out) == 1
        assert torch.allclose(hf_out[0], trt_out[0], rtol=1e-5, atol=1e-5), (
            f"tpool_patch_merger (multi frame) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out[0] - trt_out[0])).item()}")

    def test_multi_batch(self):
        d = 1152
        t1, h1, w1 = 1, 4, 4
        t2, h2, w2 = 2, 4, 4
        total = t1 * h1 * w1 + t2 * h2 * w2
        x = torch.randn(total, d, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[t1, h1, w1], [t2, h2, w2]], device=DEVICE)
        hf_out = hf_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        trt_out = trt_tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2))
        assert len(hf_out) == len(trt_out) == 2
        for i in range(2):
            assert torch.allclose(hf_out[i], trt_out[i], rtol=1e-5, atol=1e-5), (
                f"tpool_patch_merger (multi batch, item {i}) mismatch: "
                f"max_diff={torch.max(torch.abs(hf_out[i] - trt_out[i])).item()}")


# =========================================================================== #
# Test 10: PatchMergerMLP
# =========================================================================== #
class TestPatchMergerMLP:
    @pytest.fixture
    def modules(self):
        # HF uses a config object
        class HFProjectorConfig:
            mm_hidden_size = 1152
            hidden_size = 7168
            merge_kernel_size = (2, 2)
            projector_ln_eps = 1e-5
        hf_mod = HF_PatchMergerMLP(HFProjectorConfig()).to(DEVICE, DTYPE)
        trt_mod = TRT_PatchMergerMLP(
            mm_hidden_size=1152,
            text_hidden_size=7168,
            merge_kernel_size=(2, 2),
            projector_ln_eps=1e-5,
        ).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward_list_input(self, modules):
        hf_mod, trt_mod = modules
        # Input from tpool_patch_merger: list of (N, K, C) tensors
        item = torch.randn(4, 4, 1152, device=DEVICE, dtype=DTYPE)
        hf_out = hf_mod([item])
        trt_out = trt_mod([item])
        assert isinstance(hf_out, list) and isinstance(trt_out, list)
        assert torch.allclose(hf_out[0], trt_out[0], rtol=1e-4, atol=1e-4), (
            f"PatchMergerMLP (list input) mismatch: "
            f"max_diff={torch.max(torch.abs(hf_out[0] - trt_out[0])).item()}")


# =========================================================================== #
# Test 11: MoonViT3dPretrainedModel (full vision tower)
# =========================================================================== #
class TestMoonViT3dPretrainedModel:
    @pytest.fixture
    def modules(self):
        """Create small-config vision tower for testing."""
        try:
            from transformers.activations import PytorchGELUTanh
        except ImportError:
            PytorchGELUTanh = lambda: nn.GELU(approximate='tanh')

        # Create a minimal config for TRT-LLM
        class TRTConfig:
            hidden_size = 1152
            patch_size = 14
            init_pos_emb_height = 64
            init_pos_emb_width = 64
            init_pos_emb_time = 4
            pos_emb_type = 'divided_fixed'
            num_attention_heads = 16
            num_hidden_layers = 2  # Use only 2 layers for speed
            intermediate_size = 4304
            merge_kernel_size = (2, 2)
            merge_type = 'sd2_tpool'

        trt_mod = TRT_MoonViT3dPretrainedModel(TRTConfig()).to(DEVICE, DTYPE)

        # Create HF version with matching config
        from transformers.configuration_utils import PretrainedConfig as HFPretrainedConfig
        hf_config = HFPretrainedConfig()
        hf_config.hidden_size = 1152
        hf_config.patch_size = 14
        hf_config.init_pos_emb_height = 64
        hf_config.init_pos_emb_width = 64
        hf_config.init_pos_emb_time = 4
        hf_config.pos_emb_type = 'divided_fixed'
        hf_config.num_attention_heads = 16
        hf_config.num_hidden_layers = 2  # Match TRT
        hf_config.intermediate_size = 4304
        hf_config.merge_kernel_size = (2, 2)
        hf_config.video_attn_type = 'spatial_temporal'
        hf_config.merge_type = 'sd2_tpool'
        hf_config._attn_implementation = 'eager'

        hf_mod = HF_MoonViT3dPretrainedModel(hf_config).to(DEVICE, DTYPE)
        copy_weights(hf_mod, trt_mod)
        return hf_mod, trt_mod

    def test_forward(self, modules):
        hf_mod, trt_mod = modules
        # 4 patches in a 2x2 grid, single frame
        num_patches = 4
        pixel_values = torch.randn(num_patches, 3, 14, 14, device=DEVICE, dtype=DTYPE)
        grid_thws = torch.tensor([[1, 2, 2]], device=DEVICE)

        hf_out = hf_mod(pixel_values, grid_thws)
        trt_out = trt_mod(pixel_values, grid_thws)

        assert isinstance(hf_out, list) and isinstance(trt_out, list)
        assert len(hf_out) == len(trt_out)
        for i in range(len(hf_out)):
            assert torch.allclose(hf_out[i], trt_out[i], rtol=1e-3, atol=1e-3), (
                f"MoonViT3dPretrainedModel output[{i}] mismatch: "
                f"max_diff={torch.max(torch.abs(hf_out[i] - trt_out[i])).item()}, "
                f"hf_shape={hf_out[i].shape}, trt_shape={trt_out[i].shape}")


# =========================================================================== #
# Test 12: Weight name compatibility with checkpoint
# =========================================================================== #
class TestWeightNameCompatibility:
    def test_vision_tower_weight_names(self):
        """Verify TRT-LLM module's state_dict keys match HF checkpoint weight names."""
        class TRTConfig:
            hidden_size = 1152
            patch_size = 14
            init_pos_emb_height = 64
            init_pos_emb_width = 64
            init_pos_emb_time = 4
            pos_emb_type = 'divided_fixed'
            num_attention_heads = 16
            num_hidden_layers = 2
            intermediate_size = 4304
            merge_kernel_size = (2, 2)
            merge_type = 'sd2_tpool'

        trt_mod = TRT_MoonViT3dPretrainedModel(TRTConfig())
        trt_keys = set(trt_mod.state_dict().keys())

        # Expected keys (based on checkpoint analysis)
        expected_keys = {
            'patch_embed.proj.weight',
            'patch_embed.proj.bias',
            'patch_embed.pos_emb.weight',
            'encoder.final_layernorm.weight',
            'encoder.final_layernorm.bias',
        }
        for b in range(2):
            for suffix in [
                'wqkv.weight', 'wqkv.bias',
                'wo.weight', 'wo.bias',
                'norm0.weight', 'norm0.bias',
                'norm1.weight', 'norm1.bias',
                'mlp.fc0.weight', 'mlp.fc0.bias',
                'mlp.fc1.weight', 'mlp.fc1.bias',
            ]:
                expected_keys.add(f'encoder.blocks.{b}.{suffix}')

        assert expected_keys == trt_keys, (
            f"Weight name mismatch!\n"
            f"  Missing from TRT-LLM: {expected_keys - trt_keys}\n"
            f"  Extra in TRT-LLM: {trt_keys - expected_keys}")

    def test_projector_weight_names(self):
        """Verify TRT-LLM PatchMergerMLP state_dict keys match HF checkpoint."""
        trt_mod = TRT_PatchMergerMLP(
            mm_hidden_size=1152, text_hidden_size=7168,
            merge_kernel_size=(2, 2), projector_ln_eps=1e-5,
        )
        trt_keys = set(trt_mod.state_dict().keys())
        expected_keys = {
            'pre_norm.weight', 'pre_norm.bias',
            'proj.0.weight', 'proj.0.bias',
            'proj.2.weight', 'proj.2.bias',
        }
        assert expected_keys == trt_keys, (
            f"Projector weight name mismatch!\n"
            f"  Missing: {expected_keys - trt_keys}\n"
            f"  Extra: {trt_keys - expected_keys}")
