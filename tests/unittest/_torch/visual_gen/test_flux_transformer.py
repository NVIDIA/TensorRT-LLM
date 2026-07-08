# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLUX transformer models.

Tests cover:
- Model structure and instantiation
- Forward pass sanity checks
- Numerical correctness vs HuggingFace
"""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig

# FLUX.1 dev config (12B params)
FLUX1_CONFIG = {
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
    "torch_dtype": "bfloat16",
    "axes_dim": [16, 56, 56],
}

# FLUX.2 dev config (35B params)
FLUX2_CONFIG = {
    "attention_head_dim": 128,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 4096,
    "num_attention_heads": 48,
    "num_layers": 8,
    "num_single_layers": 48,
    "patch_size": 1,
    "pooled_projection_dim": 768,
    "torch_dtype": "bfloat16",
    "axes_dim": [32, 32, 32, 32],
}


def reduce_flux_config(mem_for_full_model: int, config_dict: dict):
    """Reduce model size if insufficient GPU memory."""
    _, total_mem = torch.cuda.mem_get_info()
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = max(1, int(config_dict["num_layers"] * model_fraction))
        num_single_layers = max(1, int(config_dict["num_single_layers"] * model_fraction))
        config_dict["num_layers"] = min(num_layers, 2)
        config_dict["num_single_layers"] = min(num_single_layers, 4)


def _make_fake_flux2_parallel_attn():
    from tensorrt_llm._torch.visual_gen.models.flux.attention import Flux2ParallelSelfAttention

    attn = Flux2ParallelSelfAttention.__new__(Flux2ParallelSelfAttention)
    gate_up_proj = SimpleNamespace(
        _weights_created=True,
        has_nvfp4=True,
        has_bias=False,
        out_features=256,
        use_cute_dsl_blockscaling_mm=True,
        input_scale=torch.ones(1),
        pre_quant_scale=None,
        force_dynamic_quantization=False,
    )
    down_proj = SimpleNamespace(
        _weights_created=True,
        has_nvfp4=True,
        in_features=128,
        input_scale=torch.ones(1),
        pre_quant_scale=None,
        force_dynamic_quantization=False,
    )
    attn.to_qkv_mlp_proj = SimpleNamespace(
        tp_size=2,
        qkv_proj=object(),
        mlp_proj=gate_up_proj,
    )
    attn.to_out = SimpleNamespace(mlp_proj=down_proj)
    return attn, gate_up_proj, down_proj


def test_flux2_single_stream_fp4out_guard_requires_compatible_packed_width(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux import attention as flux_attention

    monkeypatch.setattr(flux_attention.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: True)
    attn, gate_up_proj, down_proj = _make_fake_flux2_parallel_attn()

    hidden_states = torch.empty(1, 128, 16)
    assert attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    down_proj.in_features = 64
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    gate_up_proj.out_features = 258
    down_proj.in_features = 129
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)


def test_flux2_single_stream_fp4out_guard_requires_static_nvfp4(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux import attention as flux_attention

    monkeypatch.setattr(flux_attention.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: True)
    attn, gate_up_proj, down_proj = _make_fake_flux2_parallel_attn()
    hidden_states = torch.empty(1, 128, 16)

    gate_up_proj.has_nvfp4 = False
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    gate_up_proj.has_nvfp4 = True
    gate_up_proj.force_dynamic_quantization = True
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    gate_up_proj.force_dynamic_quantization = False
    gate_up_proj.pre_quant_scale = torch.ones(16)
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    gate_up_proj.pre_quant_scale = None
    gate_up_proj.input_scale = None
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    gate_up_proj.input_scale = torch.ones(1)
    down_proj.force_dynamic_quantization = True
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)

    down_proj.force_dynamic_quantization = False
    down_proj.input_scale = None
    assert not attn._can_project_hidden_mlp_with_fp4out(hidden_states)


def test_flux2_single_stream_cute_dsl_guard_requires_interleaved_weights(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux import attention as flux_attention

    monkeypatch.setattr(flux_attention.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: True)
    attn, gate_up_proj, _ = _make_fake_flux2_parallel_attn()

    assert attn._can_project_hidden_mlp_with_cute_dsl()

    gate_up_proj.use_cute_dsl_blockscaling_mm = False
    assert not attn._can_project_hidden_mlp_with_cute_dsl()


def test_flux2_joint_qkv_mlp_enables_cutedsl_for_mlp_only():
    from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import FluxJointQKVMLPProj

    proj = FluxJointQKVMLPProj(
        in_dim=256,
        q_dim=256,
        kv_dim=256,
        mlp_dim=512,
        quant_config=QuantConfig(),
        skip_create_weights_in_init=True,
        use_cute_dsl_blockscaling_mm=True,
        mapping=Mapping(world_size=2, rank=0, tp_size=2),
    )

    assert not proj.qkv_proj.use_cute_dsl_blockscaling_mm
    assert proj.mlp_proj.use_cute_dsl_blockscaling_mm


def test_flux2_moe_swiglu_reorders_gate_up(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux.attention import Flux2ParallelSelfAttention

    attn = Flux2ParallelSelfAttention.__new__(Flux2ParallelSelfAttention)
    captured = {}

    def fake_quantize(x, *args):
        captured["input"] = x
        return torch.empty(x.shape[0], x.shape[1] // 4), torch.empty(1)

    class FakeProj:
        input_scale = torch.ones(1)

        def __call__(self, x):
            return torch.empty(1, 2, 4)

    attn.to_out = SimpleNamespace(mlp_proj=FakeProj())
    attn._combine_split_projection = lambda attn_out, mlp_out: mlp_out
    monkeypatch.setattr(torch.ops.trtllm, "moe_swiglu_nvfp4_quantize", fake_quantize)

    gate = torch.full((1, 2, 4), 1.0)
    up = torch.full((1, 2, 4), 2.0)
    attn._project_split_output_with_fp4_mlp(torch.empty(1), torch.cat((gate, up), dim=-1))

    expected = torch.cat((up, gate), dim=-1).reshape(2, 8)
    torch.testing.assert_close(captured["input"][:2], expected)


def test_flux2_mlp_fp4_guard_requires_blackwell(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux import attention as flux_attention

    monkeypatch.setattr(flux_attention.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: True)
    attn, _, _ = _make_fake_flux2_parallel_attn()
    assert attn._can_project_mlp_out_from_fp4()

    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: False)
    assert not attn._can_project_mlp_out_from_fp4()


def test_flux2_mlp_fp4_guard_requires_static_output_quant(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux import attention as flux_attention

    monkeypatch.setattr(flux_attention.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_attention, "is_sm_100f", lambda: True)
    attn, _, down_proj = _make_fake_flux2_parallel_attn()

    assert attn._can_project_mlp_out_from_fp4()

    down_proj.input_scale = None
    assert not attn._can_project_mlp_out_from_fp4()

    down_proj.input_scale = torch.ones(1)
    down_proj.pre_quant_scale = torch.ones(128)
    assert not attn._can_project_mlp_out_from_fp4()

    down_proj.pre_quant_scale = None
    down_proj.force_dynamic_quantization = True
    assert not attn._can_project_mlp_out_from_fp4()


class TestFluxTransformer(unittest.TestCase):
    """Unit tests for FLUX transformer models."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_model_config(
        self, config_dict: dict, backend: str = "VANILLA"
    ) -> DiffusionModelConfig:
        """Create DiffusionModelConfig from config dict."""
        pretrained_config = SimpleNamespace(**config_dict)
        return DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
            skip_create_weights_in_init=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux1_model_structure(self):
        """Test FLUX.1 model can be instantiated with correct structure."""
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import (
            FluxTransformer2DModel,
        )

        config = deepcopy(FLUX1_CONFIG)
        config["num_layers"] = 1
        config["num_single_layers"] = 1

        model_config = self._create_model_config(config)
        model = FluxTransformer2DModel(model_config).to(self.DEVICE)

        # Check model structure
        self.assertTrue(hasattr(model, "transformer_blocks"))
        self.assertTrue(hasattr(model, "single_transformer_blocks"))
        self.assertEqual(len(model.transformer_blocks), 1)
        self.assertEqual(len(model.single_transformer_blocks), 1)

        # Check key components
        self.assertTrue(hasattr(model, "x_embedder"))
        self.assertTrue(hasattr(model, "context_embedder"))
        self.assertTrue(hasattr(model, "time_text_embed"))
        self.assertTrue(hasattr(model, "pos_embed"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux2_model_structure(self):
        """Test FLUX.2 model can be instantiated with correct structure."""
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import (
            Flux2Transformer2DModel,
        )

        config = deepcopy(FLUX2_CONFIG)
        config["num_layers"] = 1
        config["num_single_layers"] = 1

        model_config = self._create_model_config(config)
        model = Flux2Transformer2DModel(model_config).to(self.DEVICE)

        self.assertTrue(hasattr(model, "transformer_blocks"))
        self.assertTrue(hasattr(model, "single_transformer_blocks"))
        self.assertEqual(len(model.transformer_blocks), 1)
        self.assertEqual(len(model.single_transformer_blocks), 1)

    def test_flux2_dual_stream_ffn_enables_cutedsl_blockscaling(self):
        """FLUX.2 dual-stream FFNs should reach the fused NVFP4 SwiGLU path."""
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import (
            Flux2TransformerBlock,
        )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch(
                "tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2.is_sm_100f",
                return_value=True,
            ),
        ):
            model_config = self._create_model_config(FLUX2_CONFIG)
            model_config.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
            model_config.skip_create_weights_in_init = True
            block = Flux2TransformerBlock(
                dim=16,
                num_attention_heads=2,
                attention_head_dim=8,
                mlp_ratio=2.0,
                config=model_config,
            )

        self.assertTrue(block.ff.use_cute_dsl_blockscaling_mm)
        self.assertTrue(block.ff_context.use_cute_dsl_blockscaling_mm)

    def test_flux2_dual_stream_ffn_disables_cutedsl_blockscaling_off_sm100f(self):
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import (
            Flux2TransformerBlock,
        )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch(
                "tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2.is_sm_100f",
                return_value=False,
            ),
        ):
            model_config = self._create_model_config(FLUX2_CONFIG)
            model_config.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
            model_config.skip_create_weights_in_init = True
            block = Flux2TransformerBlock(
                dim=16,
                num_attention_heads=2,
                attention_head_dim=8,
                mlp_ratio=2.0,
                config=model_config,
            )

        self.assertFalse(block.ff.use_cute_dsl_blockscaling_mm)
        self.assertFalse(block.ff_context.use_cute_dsl_blockscaling_mm)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux1_forward_sanity(self):
        """Test FLUX.1 forward pass produces valid output."""
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import (
            FluxTransformer2DModel,
        )

        config = deepcopy(FLUX1_CONFIG)
        config["num_layers"] = 1
        config["num_single_layers"] = 1

        model_config = self._create_model_config(config)
        model = FluxTransformer2DModel(model_config).to(self.DEVICE, dtype=torch.bfloat16).eval()

        batch_size = 1
        height, width = 64, 64
        seq_len = (height // 2) * (width // 2)
        text_seq_len = 128

        hidden_states = torch.randn(
            batch_size, seq_len, config["in_channels"], device=self.DEVICE, dtype=torch.bfloat16
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            text_seq_len,
            config["joint_attention_dim"],
            device=self.DEVICE,
            dtype=torch.bfloat16,
        )
        pooled_projections = torch.randn(
            batch_size, config["pooled_projection_dim"], device=self.DEVICE, dtype=torch.bfloat16
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.bfloat16)
        guidance = torch.tensor([3.5], device=self.DEVICE, dtype=torch.bfloat16)
        img_ids = torch.zeros(batch_size, seq_len, 3, device=self.DEVICE)
        txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=self.DEVICE)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )

        # Model returns {"sample": tensor}
        if isinstance(output, dict):
            output = output["sample"]

        # Note: With random weights, NaN can occur. For unit tests, we only check shape.
        # Full numerical correctness is tested in TestFluxHuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux2_forward_sanity(self):
        """Test FLUX.2 forward pass produces valid output."""
        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import (
            Flux2Transformer2DModel,
        )

        config = deepcopy(FLUX2_CONFIG)
        config["num_layers"] = 1
        config["num_single_layers"] = 1

        model_config = self._create_model_config(config)
        model = Flux2Transformer2DModel(model_config).to(self.DEVICE, dtype=torch.bfloat16).eval()

        batch_size = 1
        height, width = 32, 32
        seq_len = (height // 2) * (width // 2)
        text_seq_len = 128

        hidden_states = torch.randn(
            batch_size, seq_len, config["in_channels"], device=self.DEVICE, dtype=torch.bfloat16
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            text_seq_len,
            config["joint_attention_dim"],
            device=self.DEVICE,
            dtype=torch.bfloat16,
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.bfloat16)
        guidance = torch.tensor([3.5], device=self.DEVICE, dtype=torch.bfloat16)
        # FLUX.2 uses 4-axis RoPE
        img_ids = torch.zeros(seq_len, 4, device=self.DEVICE)
        txt_ids = torch.zeros(text_seq_len, 4, device=self.DEVICE)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )

        # Model returns {"sample": tensor}
        if isinstance(output, dict):
            output = output["sample"]

        # Note: With random weights, NaN can occur. For unit tests, we only check shape.
        # Full numerical correctness is tested in TestFluxHuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)


class TestFluxHuggingFaceComparison(unittest.TestCase):
    """Test FLUX models match HuggingFace reference implementation."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_model_config(
        self, config_dict: dict, backend: str = "VANILLA"
    ) -> DiffusionModelConfig:
        """Create DiffusionModelConfig from config dict."""
        pretrained_config = SimpleNamespace(**config_dict)
        return DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
            skip_create_weights_in_init=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux1_allclose_to_hf(self):
        """Test TRT-LLM FLUX.1 transformer matches HuggingFace output."""
        try:
            from diffusers import FluxTransformer2DModel as HFFluxTransformer2DModel
        except ImportError:
            self.skipTest("diffusers not installed")

        from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import (
            FluxTransformer2DModel,
        )

        torch.manual_seed(42)

        config = deepcopy(FLUX1_CONFIG)
        config["num_layers"] = 1
        config["num_single_layers"] = 2

        dtype = torch.bfloat16

        # Create HuggingFace model with random weights
        hf_model = (
            HFFluxTransformer2DModel(
                patch_size=config["patch_size"],
                in_channels=config["in_channels"],
                num_layers=config["num_layers"],
                num_single_layers=config["num_single_layers"],
                attention_head_dim=config["attention_head_dim"],
                num_attention_heads=config["num_attention_heads"],
                joint_attention_dim=config["joint_attention_dim"],
                pooled_projection_dim=config["pooled_projection_dim"],
                guidance_embeds=config["guidance_embeds"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Create TRT-LLM model
        model_config = self._create_model_config(config)
        trtllm_model = FluxTransformer2DModel(model_config).to(self.DEVICE, dtype=dtype).eval()

        # Copy weights from HF to TRT-LLM
        hf_state_dict = hf_model.state_dict()
        trtllm_model.load_weights(hf_state_dict)

        # Create test inputs
        batch_size = 1
        height, width = 32, 32
        seq_len = (height // 2) * (width // 2)
        text_seq_len = 64

        generator = torch.Generator(device=self.DEVICE).manual_seed(42)
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            config["in_channels"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            text_seq_len,
            config["joint_attention_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        pooled_projections = torch.randn(
            batch_size,
            config["pooled_projection_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        timestep = torch.tensor([500.0], device=self.DEVICE, dtype=dtype)
        guidance = torch.tensor([3.5], device=self.DEVICE, dtype=dtype)
        img_ids = torch.zeros(batch_size, seq_len, 3, device=self.DEVICE)
        txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=self.DEVICE)

        # Run both models
        with (
            torch.no_grad(),
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ),
        ):
            hf_output = hf_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep / 1000,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )

        # Model returns {"sample": tensor}
        if isinstance(trtllm_output, dict):
            trtllm_output = trtllm_output["sample"]

        # Compare outputs
        hf_output = hf_output.float()
        trtllm_output = trtllm_output.float()

        cos_sim = F.cosine_similarity(
            hf_output.flatten().unsqueeze(0), trtllm_output.flatten().unsqueeze(0)
        ).item()

        max_diff = (hf_output - trtllm_output).abs().max().item()

        print("\n[FLUX.1 HF Comparison]")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max diff: {max_diff:.6f}")

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
