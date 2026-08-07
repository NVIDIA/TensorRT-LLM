# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for the LTX-2.3 pipeline.

Mirrors ``test_ltx2_pipeline.py`` (same conventions / thresholds) but targets
the LTX-2.3 (``LTX23Pipeline``) native checkpoint. Tests cover:

- Pipeline loading with quantization (FP8, FP8_BLOCK_SCALES) + FP8 weight check
- FP8 vs BF16 single-layer numerical correctness
- FP8 vs BF16 transformer memory comparison
- Attention backend comparison (VANILLA vs TRTLLM), video + audio outputs
- Single-stage variant resolution (no two-stage upsampler/LoRA in Phase-0)

Requires the LTX-2.3 checkpoint. Does NOT require the LTX-2.3 reference code.

Key differences from the LTX-2 template (LTX-2.3 specifics):
- Inputs are ``LTX23Modality`` objects, which add a global ``sigma`` field
  (drives the sigma-dependent text cross-attention K/V modulation).
- ``caption_projection`` is ``nn.Identity`` in LTX-2.3 (the split feature
  extractor projects to inner_dim *before* the connector), so the text
  ``context`` fed to the transformer is already ``cross_attention_dim`` wide,
  not ``caption_channels``.
- The transformer emits ``(video_out, audio_out)`` from an AudioVideo model.
"""

import gc
import os

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, VisualGenArgs

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

# ``test_common`` is part of the TRT-LLM source test harness and is present in
# CI. When running standalone (e.g. against a pip-installed TRT-LLM inside a
# container), it is absent; fall back to the LLM_MODELS_ROOT env var so the
# module still imports and the explicit LTX23_MODEL_PATH override below works.
try:
    from test_common.llm_data import llm_models_root

    _MODELS_ROOT = str(llm_models_root(check=False))
except Exception:  # pragma: no cover - only hit outside the CI harness
    _MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", "")

# Skip non-transformer components. ``skip_components`` focuses the load on the
# transformer; LTX-2.3 native components (audio_vae, vocoder, connectors,
# video decoder) still load from the checkpoint automatically.
SKIP_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
]


_LTX23_BASE = os.path.join(_MODELS_ROOT, "LTX-2.3") if _MODELS_ROOT else ""
_GEMMA3_DEFAULT = (
    os.path.join(_MODELS_ROOT, "gemma", "gemma-3-12b-it") if _MODELS_ROOT else ""
)


# LTX-2.3 ships as a native single-file checkpoint inside its model directory;
# quantization is applied dynamically from the BF16 weights (no separate FP8
# checkpoint file), so a single checkpoint path covers every quant test.
CHECKPOINT_PATH_BF16 = os.environ.get("LTX23_MODEL_PATH", _LTX23_BASE)
GEMMA3_PATH = os.environ.get("LTX23_TEXT_ENCODER_PATH", _GEMMA3_DEFAULT)


def _ltx23_pipeline_config(**overrides):
    """Build pipeline_config with the Gemma3 text_encoder_path LTX-2.3 needs.

    LTX-2.3's tokenizer + text encoder load from a separate Gemma directory
    (not the diffusion checkpoint), so every full-pipeline load needs
    ``text_encoder_path`` set.
    """
    cfg = {"text_encoder_path": GEMMA3_PATH}
    cfg.update(overrides)
    return cfg


def _get_ltx23_transformer_inputs(transformer, device="cuda", dtype=torch.bfloat16):
    """Create test inputs for the LTX-2.3 transformer (LTX23Modality objects).

    Constructs minimal video + audio inputs compatible with the transformer's
    args preprocessor. Unlike LTX-2, the text ``context`` is already
    ``cross_attention_dim`` wide (caption_projection is Identity in LTX-2.3),
    and each modality carries a global ``sigma`` alongside per-token
    ``timesteps``.
    """
    torch.manual_seed(42)
    batch = 1
    n_frames, grid_h, grid_w = 1, 4, 4
    v_patches = n_frames * grid_h * grid_w
    a_patches = 8
    text_len = 8

    cfg = getattr(transformer, "_transformer_config", {})
    in_channels = cfg.get("in_channels", 128)
    audio_in_channels = cfg.get("audio_in_channels", 128)
    # Post-connector context dims (== inner_dim, since caption_projection is
    # Identity in LTX-2.3). Fall back to the checkpoint defaults.
    v_context_dim = cfg.get("cross_attention_dim", 4096)
    a_context_dim = cfg.get("audio_cross_attention_dim", 2048)

    v_positions = torch.zeros(batch, 3, v_patches, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                v_positions[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                v_positions[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                v_positions[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1

    a_positions = torch.zeros(batch, 1, a_patches, 2, device=device)
    for i in range(a_patches):
        a_positions[:, 0, i, :] = torch.tensor([i, i + 1], dtype=torch.float32)

    v_context = torch.randn(batch, text_len, v_context_dim, device=device, dtype=dtype)
    a_context = torch.randn(batch, text_len, a_context_dim, device=device, dtype=dtype)

    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    sigma = torch.tensor([0.5], device=device)
    video = LTX23Modality(
        latent=torch.randn(batch, v_patches, in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
        sigma=sigma,
        positions=v_positions,
        context=v_context,
    )
    audio = LTX23Modality(
        latent=torch.randn(batch, a_patches, audio_in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
        sigma=sigma,
        positions=a_positions,
        context=a_context,
    )
    text_cache = transformer.prepare_text_cache(
        video_context=v_context,
        video_positions=v_positions,
        audio_context=a_context,
        audio_positions=a_positions,
        dtype=dtype,
    )
    return video, audio, text_cache


def _extract_output(output):
    """Extract tensors from (video_out, audio_out) tuple."""
    if isinstance(output, tuple) and len(output) == 2:
        return output
    return output, None


def _find_first_quantizable_linear(transformer):
    """Find the first Linear layer in transformer blocks suitable for testing."""
    for name, module in transformer.named_modules():
        if isinstance(module, Linear) and "blocks" in name:
            return module, name
    return None, None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ltx23_bf16_checkpoint_exists():
    """Check if the LTX-2.3 checkpoint is available locally."""
    if not CHECKPOINT_PATH_BF16 or not os.path.exists(CHECKPOINT_PATH_BF16):
        pytest.skip(
            f"LTX-2.3 checkpoint not found at {CHECKPOINT_PATH_BF16}. "
            "Set LTX23_MODEL_PATH or stage the checkpoint under LLM_MODELS_ROOT/LTX-2.3/."
        )
    return True


# ============================================================================
# Quantization Tests
# ============================================================================


class TestLTX23Quantization:
    """Test LTX-2.3 quantization loading and FP8 weight verification."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_load_with_quantization(self, ltx23_bf16_checkpoint_exists, quant_algo: str):
        """Test loading LTX-2.3 with FP8 quantization and verify FP8 weights."""
        args = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline_config=_ltx23_pipeline_config(),
        )

        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=SKIP_COMPONENTS)

        assert pipeline.pipeline_config.quant_config.quant_algo is not None

        quant_count = 0
        found_fp8 = False
        for name, module in pipeline.transformer.named_modules():
            if isinstance(module, Linear):
                if module.quant_config and module.quant_config.quant_algo:
                    quant_count += 1
                    if "blocks" in name and hasattr(module, "weight") and module.weight is not None:
                        if not found_fp8:
                            assert module.weight.dtype == torch.float8_e4m3fn, (
                                f"Linear {name} should have FP8 weight, got {module.weight.dtype}"
                            )
                            assert hasattr(module, "weight_scale"), (
                                f"Linear {name} missing weight_scale"
                            )
                            found_fp8 = True
                            print(
                                f"\n[{quant_algo}] FP8 layer {name}: weight {module.weight.shape}"
                            )

        print(f"[{quant_algo}] Quantized {quant_count} Linear layers")
        assert quant_count > 0, "No layers were quantized"
        assert found_fp8, f"No FP8 Linear modules found in blocks for {quant_algo}"

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# FP8 Numerical Correctness Tests
# ============================================================================


class TestLTX23FP8NumericalCorrectness:
    """Test FP8 vs BF16 numerical accuracy at single-layer level."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_single_layer(self, ltx23_bf16_checkpoint_exists, quant_algo: str):
        """Test FP8 vs BF16 numerical accuracy on a single Linear layer.

        1. Use F.linear() with BF16 weights as ground truth reference
        2. Verify BF16 layer matches F.linear exactly
        3. Compare FP8 layer output against reference
        """
        print(f"\n[Compare {quant_algo}] Loading BF16 pipeline...")
        args_bf16 = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )

        print(f"[Compare {quant_algo}] Loading {quant_algo} pipeline...")
        args_fp8 = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )

        linear_bf16, layer_name = _find_first_quantizable_linear(pipeline_bf16.transformer)
        linear_fp8, _ = _find_first_quantizable_linear(pipeline_fp8.transformer)

        assert linear_bf16 is not None, "Could not find a Linear layer in BF16 transformer"
        assert linear_fp8 is not None, "Could not find a Linear layer in FP8 transformer"

        weight_bf16 = linear_bf16.weight.data.clone()
        bias_bf16 = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None

        torch.manual_seed(42)
        hidden_size = linear_bf16.in_features
        batch_seq_len = 1024
        input_tensor = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")
        print(f"[Compare] Layer: {layer_name}, Input shape: {input_tensor.shape}")

        with torch.no_grad():
            expected = F.linear(input_tensor, weight_bf16, bias_bf16)
            result_bf16 = linear_bf16(input_tensor)
            result_fp8 = linear_fp8(input_tensor)

        assert torch.allclose(result_bf16, expected, rtol=1e-5, atol=1e-6), (
            "BF16 layer should match F.linear reference exactly"
        )

        max_diff = torch.max(torch.abs(result_fp8 - expected)).item()
        cos_sim = F.cosine_similarity(
            result_fp8.flatten().float(), expected.flatten().float(), dim=0
        )
        mse = F.mse_loss(result_fp8.flatten().float(), expected.flatten().float())

        print(
            f"\n[{layer_name}] max_diff={max_diff:.6f}, cos_sim={cos_sim.item():.6f}, mse={mse.item():.6f}"
        )

        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"
        assert mse < 1.0, f"MSE too high: {mse.item()}"

        del pipeline_bf16, pipeline_fp8
        torch.cuda.empty_cache()


# ============================================================================
# FP8 Memory Comparison Tests
# ============================================================================


class TestLTX23FP8Memory:
    """Test FP8 memory reduction for LTX-2.3."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_memory_comparison(self, ltx23_bf16_checkpoint_exists):
        """Test FP8 uses ~2x less memory than BF16."""

        def get_module_memory_gb(module):
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_bf16 = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )

        bf16_model_mem = get_module_memory_gb(pipeline_bf16.transformer)
        print(f"\n[BF16] Transformer memory: {bf16_model_mem:.2f} GB")

        del pipeline_bf16
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        args_fp8 = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            quant_config={"quant_algo": "FP8", "dynamic": True},
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )

        fp8_model_mem = get_module_memory_gb(pipeline_fp8.transformer)
        print(f"[FP8] Transformer memory: {fp8_model_mem:.2f} GB")

        model_mem_ratio = bf16_model_mem / fp8_model_mem
        print(f"\n[Comparison] Model memory ratio (BF16/FP8): {model_mem_ratio:.2f}x")

        assert model_mem_ratio > 1.8, f"FP8 should use ~2x less memory, got {model_mem_ratio:.2f}x"

        del pipeline_fp8
        torch.cuda.empty_cache()


# ============================================================================
# Attention Backend Comparison Tests
# ============================================================================


class TestLTX23AttentionBackend:
    """Test VANILLA vs TRTLLM attention backend numerical correctness."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_backend_comparison(self, ltx23_bf16_checkpoint_exists):
        """Test that VANILLA and TRTLLM backends produce similar outputs.

        Load each backend sequentially (two full LTX-2.3 transformers don't fit
        in GPU memory simultaneously). Compares both the video and audio output
        streams (LTX-2.3 is an AudioVideo model).
        """
        print("\n[Attention Backend Test] Loading baseline transformer (VANILLA)...")
        args_baseline = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            attention_config=AttentionConfig(backend="VANILLA"),
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_baseline = PipelineLoader(args_baseline).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )
        transformer_baseline = pipeline_baseline.transformer

        video_input, audio_input, text_cache_baseline = _get_ltx23_transformer_inputs(
            transformer_baseline
        )

        print("[Attention Backend Test] Running VANILLA transformer forward...")
        with torch.no_grad():
            output_baseline = transformer_baseline(
                video=video_input, audio=audio_input, text_cache=text_cache_baseline
            )
        vout_baseline, aout_baseline = _extract_output(output_baseline)
        vout_baseline_cpu = vout_baseline.cpu() if vout_baseline is not None else None
        aout_baseline_cpu = aout_baseline.cpu() if aout_baseline is not None else None

        del pipeline_baseline, transformer_baseline
        gc.collect()
        torch.cuda.empty_cache()

        print("[Attention Backend Test] Loading TRTLLM transformer...")
        args_trtllm = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            attention_config=AttentionConfig(backend="TRTLLM"),
            pipeline_config=_ltx23_pipeline_config(),
        )
        pipeline_trtllm = PipelineLoader(args_trtllm).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )
        transformer_trtllm = pipeline_trtllm.transformer

        print("[Attention Backend Test] Running TRTLLM transformer forward...")
        _, _, text_cache_trtllm = _get_ltx23_transformer_inputs(transformer_trtllm)
        with torch.no_grad():
            output_trtllm = transformer_trtllm(
                video=video_input, audio=audio_input, text_cache=text_cache_trtllm
            )
        vout_trtllm, aout_trtllm = _extract_output(output_trtllm)
        vout_trtllm_cpu = vout_trtllm.cpu() if vout_trtllm is not None else None
        aout_trtllm_cpu = aout_trtllm.cpu() if aout_trtllm is not None else None

        def _compare(name, baseline_cpu, trtllm_cpu):
            if baseline_cpu is None or trtllm_cpu is None:
                return
            assert baseline_cpu.shape == trtllm_cpu.shape, (
                f"{name} output shape mismatch: "
                f"VANILLA={baseline_cpu.shape}, TRTLLM={trtllm_cpu.shape}"
            )
            for backend, out in [("VANILLA", baseline_cpu), ("TRTLLM", trtllm_cpu)]:
                assert not torch.isnan(out).any(), f"{backend} {name} output contains NaN"
                assert not torch.isinf(out).any(), f"{backend} {name} output contains Inf"

            baseline_float = baseline_cpu.float()
            trtllm_float = trtllm_cpu.float()
            max_diff = torch.max(torch.abs(trtllm_float - baseline_float)).item()
            cos_sim = F.cosine_similarity(
                trtllm_float.flatten(), baseline_float.flatten(), dim=0
            ).item()

            print(f"\n{'=' * 60}")
            print(f"TRTLLM vs VANILLA Comparison ({name} Output)")
            print(f"{'=' * 60}")
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Cosine similarity: {cos_sim:.6f}")
            print(f"{'=' * 60}")

            assert cos_sim > 0.99, (
                f"TRTLLM should match VANILLA for {name}: cos_sim={cos_sim:.6f}"
            )
            print(f"\n[PASS] TRTLLM matches VANILLA ({name}): cos_sim={cos_sim:.6f} (>0.99)")

        _compare("Video", vout_baseline_cpu, vout_trtllm_cpu)
        _compare("Audio", aout_baseline_cpu, aout_trtllm_cpu)

        del pipeline_trtllm, transformer_trtllm
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Variant Resolution Unit Tests (no model loading required)
# ============================================================================


class TestLTX23VariantResolution:
    """LTX-2.3 is single-stage in Phase-0: resolve_variant always returns itself."""

    def test_resolve_variant_returns_single_stage(self):
        """Even with two-stage-looking config keys, LTX-2.3 stays single-stage."""
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline

        config = MagicMock()
        config.primary_pretrained_config._name_or_path = ""
        config.extra_attrs = {
            "spatial_upsampler_path": "/fake/upsampler.safetensors",
            "distilled_lora_path": "/fake/lora.safetensors",
        }

        assert LTX23Pipeline.resolve_variant(config) is LTX23Pipeline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
