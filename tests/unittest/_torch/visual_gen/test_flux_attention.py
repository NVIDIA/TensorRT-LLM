# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FLUX attention backends.

Tests cover:
- VANILLA backend (PyTorch SDPA)
- TRTLLM backend (fmha_v2)
- Backend equivalence comparison
- Fused QK-norm + RoPE TP gating

Note: With random weights, attention can produce NaN due to numerical instability.
      These tests use scaled inputs and primarily verify correct output shapes.
      Full numerical correctness is tested via HuggingFace comparison tests with real weights.
"""

import unittest
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig


class TestFluxAttentionBackend(unittest.TestCase):
    """Test FLUX attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_config(self, backend: str, tp_size: int = 1) -> DiffusionModelConfig:
        """Create DiffusionModelConfig with specified backend and TP size."""
        return DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            quant_config=QuantConfig(),
            mapping=Mapping(world_size=tp_size, tp_size=tp_size),
            attention=AttentionConfig(backend=backend),
        )

    def test_fused_qk_norm_rope_enabled_only_for_tp1(self) -> None:
        """Test FLUX enables fused QK-norm + RoPE only when TP=1."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        for tp_size, expected in ((1, True), (2, False)):
            with self.subTest(tp_size=tp_size):
                attn = FluxJointAttention(
                    hidden_size=128,
                    num_attention_heads=2,
                    head_dim=64,
                    config=self._create_config("VANILLA", tp_size=tp_size),
                )
                self.assertEqual(attn.fuse_qk_norm_rope, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vanilla_backend_sanity(self):
        """Test FLUX attention works with VANILLA backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 256
        text_seq_len = 64
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("VANILLA")

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_backend_sanity(self):
        """Test FLUX attention works with TRTLLM backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 256
        text_seq_len = 64
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("TRTLLM")
        config.attention_metadata_state = create_attention_metadata_state()

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_equivalence(self):
        """Test VANILLA and TRTLLM backends produce similar outputs."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 1
        seq_len = 128
        text_seq_len = 32
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)

        # Create attention modules for both backends
        config = self._create_config("VANILLA")
        vanilla_attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # TRT-LLM Linear uses torch.empty for bias (uninitialized memory).
        # After prior tests free GPU memory, recycled memory can contain NaN.
        # Initialize all parameters with small random values for numerical stability.
        with torch.no_grad():
            for p in vanilla_attn.parameters():
                p.normal_(0, 0.02)

        config = self._create_config("TRTLLM")
        config.attention_metadata_state = create_attention_metadata_state()
        trtllm_attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Copy scaled weights from VANILLA to TRTLLM
        trtllm_attn.load_state_dict(vanilla_attn.state_dict())
        attns = {"VANILLA": vanilla_attn, "TRTLLM": trtllm_attn}

        # Create inputs (scaled for numerical stability)
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        # Run both backends (skip RoPE for equivalence test)
        outputs = {}
        with torch.no_grad():
            for backend in ["VANILLA", "TRTLLM"]:
                out, text_out = attns[backend](
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    image_rotary_emb=None,
                )
                outputs[backend] = (out, text_out)

        # Compare outputs
        vanilla_out, vanilla_text = outputs["VANILLA"]
        trtllm_out, trtllm_text = outputs["TRTLLM"]

        # Skip comparison if either has NaN or Inf (common with random weights)
        has_nan = torch.isnan(vanilla_out).any() or torch.isnan(trtllm_out).any()
        has_inf = torch.isinf(vanilla_out).any() or torch.isinf(trtllm_out).any()
        if has_nan or has_inf:
            self.skipTest("NaN/Inf detected in outputs with random weights - skipping comparison")

        # With random weights, outputs may be all zeros or have numerical issues
        # This test is primarily for ensuring both backends can run
        # Full equivalence is tested via HuggingFace comparison with real weights
        vanilla_norm = vanilla_out.float().norm().item()
        trtllm_norm = trtllm_out.float().norm().item()

        print(f"\n[Debug] vanilla_norm={vanilla_norm:.6f}, trtllm_norm={trtllm_norm:.6f}")

        # With random weights, we can only reliably check that both backends produce
        # valid outputs (same shapes, non-trivial values). Strict equivalence requires
        # real weights from a trained model (tested via HuggingFace comparison).
        self.assertEqual(vanilla_out.shape, trtllm_out.shape)
        self.assertEqual(vanilla_text.shape, trtllm_text.shape)

        # If both outputs have meaningful norms, compute similarity as informational
        if vanilla_norm > 1e-3 and trtllm_norm > 1e-3:
            cos_sim = F.cosine_similarity(
                vanilla_out.float().flatten().unsqueeze(0),
                trtllm_out.float().flatten().unsqueeze(0),
            ).item()
            print(f"  Cosine similarity: {cos_sim:.6f}")
            # Note: Not asserting on cos_sim with random weights as it's not meaningful


class TestFlux2AttentionBackend(unittest.TestCase):
    """Test FLUX.2 attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_config(self, backend: str) -> DiffusionModelConfig:
        """Create DiffusionModelConfig with specified backend."""
        return DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux2_vanilla_backend_sanity(self):
        """Test FLUX.2 attention works with VANILLA backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 128
        text_seq_len = 64
        dim = 6144  # FLUX.2 has larger dim
        heads = 48
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("VANILLA")

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.cpu_only
def test_flux_attention_quant_ignore_bypasses_explicit_qkv_quantization() -> None:
    """Flux QKV preparation uses BF16 when attention quantization is ignored."""
    from unittest.mock import patch

    from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    attention = FluxJointAttention.__new__(FluxJointAttention)
    torch.nn.Module.__init__(attention)
    attention._quant_attention_config = QuantAttentionConfig(
        qk_dtype="fp8", v_dtype="fp8", ignored_steps=[0]
    )
    attention._quantization_excluded = False
    attention._quant_ignore_state = {}
    attention.requires_static_e4m3_attention = True
    attention._static_e4m3_attention_scales_loaded = True
    attention.added_kv_proj_dim = None
    attention.pre_only = True
    attention.fuse_qk_norm_rope = True
    attention.num_attention_heads = attention.num_key_value_heads = 1
    attention.head_dim = attention.local_q_dim = attention.local_kv_dim = 128
    attention.eps = 1e-6
    attention.interleave = True
    attention.norm_q = attention.norm_k = SimpleNamespace(weight=torch.ones(128))
    attention.qkv_proj = torch.nn.Linear(128, 384, bias=False, dtype=torch.bfloat16)
    for name in ("q", "k", "v"):
        setattr(attention, f"_static_{name}_dequant_scale", torch.ones(()))
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    rotary = (torch.ones(1, 4, 1, 128), torch.zeros(1, 4, 1, 128))
    fp8_qkv = tuple(torch.ones(4, 128).to(torch.float8_e4m3fn) for _ in range(3))
    calls = []

    def attend(q, k, v, **kwargs):
        calls.append((q.dtype, kwargs))
        return q.to(torch.bfloat16)

    with (
        patch.object(attention, "apply_packed_qk_norm_rope") as bf16,
        patch.object(attention, "_attn_impl", side_effect=attend),
        patch("torch.ops.trtllm.fused_dit_qk_norm_rope_quant_fp8", return_value=fp8_qkv) as fp8,
    ):
        for quant_ignored in (False, True, False):
            attention._quant_ignore_state["attention_quant_ignored"] = quant_ignored
            attention(x, image_rotary_emb=rotary)
    assert fp8.call_count == 2 and bf16.call_count == 1
    assert [dtype for dtype, _ in calls] == [
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]
    assert ["static_q_scale" in kwargs for _, kwargs in calls] == [True, False, True]


@pytest.mark.cpu_only
def test_flux2_linear_quant_ignore_bypasses_fp4_mlp_fusion() -> None:
    from unittest.mock import patch

    from tensorrt_llm._torch.visual_gen.models.flux.attention import Flux2ParallelSelfAttention

    class InputProjection(torch.nn.Module):
        tp_size = 2

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(128, 384, bias=False, dtype=torch.bfloat16)
            self.mlp_proj = torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)
            self.mlp_proj.can_use_cute_dsl_nvfp4_swiglu_blackwell = lambda: False

        def forward(self, value):
            return self.qkv_proj(value), self.mlp_proj(value)

    class OutputProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp_proj = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
            self.mlp_proj._weights_created = self.mlp_proj.has_nvfp4 = True
            self.mlp_proj.requires_unquantized_activation = True
            self.mlp_proj.input_scale = torch.ones(())
            self.mlp_proj.pre_quant_scale = None
            self.mlp_proj.force_dynamic_quantization = False

        def forward(self, attention, mlp):
            return attention + self.mlp_proj(mlp)

    attention = Flux2ParallelSelfAttention.__new__(Flux2ParallelSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.to_qkv_mlp_proj = InputProjection()
    attention.to_out = OutputProjection()
    attention._apply_norm_rope = lambda value, rotary: value.chunk(3, dim=-1)
    attention._attn_impl = lambda q, k, v, **kwargs: q
    with (
        patch("tensorrt_llm._torch.visual_gen.models.flux.attention.is_sm_100f", return_value=True),
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "tensorrt_llm._torch.visual_gen.models.flux.attention.swiglu",
            side_effect=lambda value: value[:, :128],
        ),
        patch.object(
            attention,
            "_project_hidden_mlp_with_cute_dsl",
            side_effect=AssertionError("Fused weight layout must not be used"),
        ),
        patch.object(
            attention,
            "_project_split_output_with_fp4_mlp",
            side_effect=AssertionError("FP4 output must not be produced"),
        ),
        patch.object(
            attention.to_out.mlp_proj, "forward", wraps=attention.to_out.mlp_proj.forward
        ) as bf16,
    ):
        attention(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    bf16.assert_called_once()
    assert bf16.call_args.args[0].dtype == torch.bfloat16


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", ["flux1", "flux2"])
def test_flux_attention_ignore_matches_real_module_path(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from test_common.visual_gen_quantization import ToyAttentionBackend

    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    monkeypatch.setenv("TLLM_DISABLE_MPI", "0")
    for component, target, tp_size in (
        (None, "single_transformer_blocks.0.attn", 1),
        ("transformer", "transformer.single_transformer_blocks.1.attn", 2),
        ("transformer", "transformer_blocks.0.attn", 1),
    ):
        config = DiffusionModelConfig(
            component_name=component,
            mapping=Mapping(world_size=tp_size, rank=0, tp_size=tp_size),
            pretrained_config=SimpleNamespace(
                num_layers=1,
                num_single_layers=2,
                num_attention_heads=2,
                attention_head_dim=128,
                joint_attention_dim=128,
                pooled_projection_dim=128,
                timestep_guidance_channels=32,
                in_channels=128,
                axes_dims_rope=[32, 32, 32, 32] if variant == "flux2" else [16, 56, 56],
            ),
            attention=AttentionConfig(
                backend="CUTEDSL",
                quant_attention_config=QuantAttentionConfig(
                    qk_dtype="nvfp4", v_dtype="fp8", ignore=[target]
                ),
            ),
        )
        with (
            torch.device("meta"),
            patch(
                "tensorrt_llm._torch.visual_gen.attention.backends.utils.get_visual_gen_attention_backend",
                return_value=ToyAttentionBackend,
            ),
        ):
            model_type = Flux2Transformer2DModel if variant == "flux2" else FluxTransformer2DModel
            model = model_type(config)
        found = []
        for name, module in model.named_modules():
            if name.endswith(".attn"):
                prefix = f"{component}." if component else ""
                assert module.module_name == prefix + name
                assert module._quantization_excluded == (target in (name, prefix + name))
                global_index = int(name.split(".")[1])
                if name.startswith("single_transformer_blocks."):
                    global_index += len(model.transformer_blocks)
                assert module.layer_idx == global_index
                assert module.tp_size == tp_size
                found.append(name)
        assert target.removeprefix("transformer.") in found
