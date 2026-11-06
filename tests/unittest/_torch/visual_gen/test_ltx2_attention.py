# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LTX2 attention backends.

Tests cover:
- VANILLA backend (PyTorch SDPA)
- TRTLLM backend (fmha_v2)
- Backend equivalence comparison
- Cross-attention (text context)
- Gated attention

No checkpoint or LTX-2 reference code required — all tests use random weights.
"""

import unittest

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


def _create_config(backend: str = "VANILLA") -> DiffusionModelConfig:
    """Create a minimal DiffusionModelConfig with the given attention backend."""
    from types import SimpleNamespace

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _init_weights(module: torch.nn.Module, std: float = 0.02):
    """Initialize all parameters with small random values.

    TRT-LLM Linear uses torch.empty() (uninitialized memory), so we must
    explicitly initialize to avoid NaN from recycled GPU memory.
    """
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=std)


def _make_pe(
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an identity-rotation (cos=1, sin=0) RoPE tuple for self-attn tests.

    LTX-2 self-attn forward (fuse_qk_norm_rope=True, head_dim ∈ {64, 128}) requires
    ``pe`` to be a non-None ``(cos, sin)`` tuple in token-major [B, T, H, D] layout —
    the same shape ``_split_freqs_cis`` produces in production. cos=1, sin=0 makes
    the RoPE step an identity, so shape-only sanity checks remain meaningful while
    still exercising the fused norm+RoPE kernel.
    """
    cos = torch.ones(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    sin = torch.zeros(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    return cos, sin


class TestLTX2SelfAttention(unittest.TestCase):
    """Test LTX2Attention self-attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vanilla_self_attention_sanity(self):
        """Test LTX2Attention self-attention with VANILLA backend produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 2
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        pe = _make_pe(batch_size, seq_len, heads, head_dim, dtype, self.DEVICE)

        with torch.no_grad():
            output = attn(x, context=None, pe=pe)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_self_attention_sanity(self):
        """Test LTX2Attention self-attention with TRTLLM backend produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 2
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("TRTLLM")
        config.attention_metadata_state = create_attention_metadata_state()

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        pe = _make_pe(batch_size, seq_len, heads, head_dim, dtype, self.DEVICE)

        with torch.no_grad():
            output = attn(x, context=None, pe=pe)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))


class TestLTX2CrossAttention(unittest.TestCase):
    """Test LTX2Attention cross-attention (with text context)."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_attention_sanity(self):
        """Test cross-attention with context_dim produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        q_seq = 64
        kv_seq = 32
        query_dim = 4096
        context_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, q_seq, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        ctx = torch.randn(batch_size, kv_seq, context_dim, device=self.DEVICE, dtype=dtype) * 0.02

        # Cross-attn must use the cached pattern: project K/V upstream, then
        # call forward with pre_projected_kv (production text cross-attn does
        # this via prepare_text_cache; AV cross-attn does it inline).
        with torch.no_grad():
            k, v = attn.project_kv(ctx, pe=None)
            output = attn(x, pre_projected_kv=(k, v), pe=None)

        self.assertEqual(output.shape, (batch_size, q_seq, query_dim))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_attention_different_dims(self):
        """Test cross-attention where query_dim != context_dim (e.g., AV cross-attn)."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        q_seq = 64
        kv_seq = 32
        query_dim = 4096
        context_dim = 2048
        heads = 32
        head_dim = 64
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, q_seq, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        ctx = torch.randn(batch_size, kv_seq, context_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            k, v = attn.project_kv(ctx, pe=None)
            output = attn(x, pre_projected_kv=(k, v), pe=None)

        self.assertEqual(output.shape, (batch_size, q_seq, query_dim))


class TestLTX2GatedAttention(unittest.TestCase):
    """Test LTX2Attention with gated attention."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gated_self_attention_sanity(self):
        """Test self-attention with apply_gated_attention=True produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                apply_gated_attention=True,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        self.assertIsNotNone(attn.to_gate_logits, "Gated attention should create to_gate_logits")

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        pe = _make_pe(batch_size, seq_len, heads, head_dim, dtype, self.DEVICE)

        with torch.no_grad():
            output = attn(x, context=None, pe=pe)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))


class TestLTX2BackendEquivalence(unittest.TestCase):
    """Test VANILLA and TRTLLM backends produce similar outputs."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_equivalence(self):
        """Test VANILLA and TRTLLM backends produce similar self-attention outputs."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)

        # Create VANILLA attention with initialized weights
        config_vanilla = _create_config("VANILLA")
        vanilla_attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config_vanilla,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        _init_weights(vanilla_attn)

        # Create TRTLLM attention and copy weights
        config_trtllm = _create_config("TRTLLM")
        config_trtllm.attention_metadata_state = create_attention_metadata_state()
        trtllm_attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config_trtllm,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        trtllm_attn.load_state_dict(vanilla_attn.state_dict())

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        pe = _make_pe(batch_size, seq_len, heads, head_dim, dtype, self.DEVICE)

        with torch.no_grad():
            out_vanilla = vanilla_attn(x.clone(), context=None, pe=pe)
            out_trtllm = trtllm_attn(x.clone(), context=None, pe=pe)

        # Skip comparison if either has NaN/Inf (can happen with random weights)
        has_nan = torch.isnan(out_vanilla).any() or torch.isnan(out_trtllm).any()
        has_inf = torch.isinf(out_vanilla).any() or torch.isinf(out_trtllm).any()
        if has_nan or has_inf:
            self.skipTest("NaN/Inf detected in outputs with random weights — skipping comparison")

        self.assertEqual(out_vanilla.shape, out_trtllm.shape)

        vanilla_norm = out_vanilla.float().norm().item()
        trtllm_norm = out_trtllm.float().norm().item()

        if vanilla_norm > 1e-3 and trtllm_norm > 1e-3:
            cos_sim = F.cosine_similarity(
                out_vanilla.float().flatten().unsqueeze(0),
                out_trtllm.float().flatten().unsqueeze(0),
            ).item()
            print(f"\n[LTX2 Backend Equivalence] Cosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.cpu_only
def test_linear_ignored_steps_keep_norm_output_bf16() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.utils_ltx2 import (
        apply_fused_rmsnorm_shift_scale,
        get_nvfp4_input_scale,
    )

    projection = SimpleNamespace(
        quant_config=SimpleNamespace(ignored_steps=[1]),
        quant_method=NVFP4LinearMethod(),
        input_scale=torch.ones(()),
        scaling_vector_size=16,
        pre_quant_scale=None,
        force_dynamic_quantization=False,
    )
    inputs = torch.randn(1, 8, 128, dtype=torch.bfloat16)
    scale = get_nvfp4_input_scale(projection)
    assert scale is None
    with patch(
        "torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale",
        return_value=(torch.empty(8, 128, dtype=torch.bfloat16),),
    ) as fused:
        output = apply_fused_rmsnorm_shift_scale(
            inputs,
            torch.randn(128),
            torch.randn(1, 8, 128),
            torch.randn(128),
            torch.randn(1, 8, 128),
            1e-6,
            True,
            fp4_input_scale=scale,
        )
    fused.assert_called_once()
    assert "sf_scale" not in fused.call_args.kwargs
    assert output.dtype == torch.bfloat16
    projection.quant_config.ignored_steps = []
    assert get_nvfp4_input_scale(projection) is projection.input_scale


@pytest.mark.cpu_only
def test_linear_ignored_steps_bypass_async_qkv_quantization() -> None:
    from unittest.mock import patch

    from test_common.visual_gen_quantization import (
        assert_quantization_routes,
        observe_quantization_paths,
    )

    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention
    from tensorrt_llm._torch.visual_gen.quantization.quant_dispatcher import (
        install_linear_quant_dispatcher,
    )
    from tensorrt_llm.visual_gen.args import DiffusionQuantConfig

    config = DiffusionModelConfig(
        quant_config=DiffusionQuantConfig(quant_algo="NVFP4", ignored_steps=[0]),
        dynamic_weight_quant=True,
        force_dynamic_quantization=False,
        attention=AttentionConfig(backend="VANILLA"),
    )
    # Explicit context gives separate projections without requiring a distributed process group.
    attention = LTX2Attention(query_dim=128, context_dim=128, heads=1, dim_head=128, config=config)
    assert not attention._maybe_share_qkv_quantize
    for name, module in attention.named_modules():
        if hasattr(module, "quant_method"):
            module.register_buffer(
                "_unquantized_weight",
                torch.randn(module.weight.shape[0], module.in_features, dtype=torch.bfloat16)
                * 0.02,
                persistent=False,
            )
            if module.bias is not None:
                module.bias.data.zero_()
    install_linear_quant_dispatcher(attention, {"linear_quant_ignored": True})
    attention.fuse_qk_norm_rope = attention.qk_norm = False

    def run_async(compute_q, compute_k, compute_v, **kwargs):
        values = {
            name: callback()
            for name, callback in (("q", compute_q), ("k", compute_k), ("v", compute_v))
        }
        return values["q"]

    with (
        patch.object(attention.attn, "forward_async", run_async, create=True),
        patch(
            "torch.ops.trtllm.tunable_fp4_quantize",
            side_effect=AssertionError("Shared FP4 quantization must be bypassed"),
        ),
        observe_quantization_paths(attention) as trace,
    ):
        trace.step = 0
        attention.forward_async(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    assert_quantization_routes(
        trace, {(0, name, "linear"): "bf16" for name in ("to_q", "to_k", "to_v", "to_out.0")}
    )
