"""Integration tests for LTX2 pipelines.

Tests cover:
- Pipeline loading (BF16, FP8, FP8_BLOCK_SCALES)
- FP8 weight verification
- FP8 vs BF16 single-layer numerical correctness
- FP8 vs BF16 memory comparison
- Attention backend comparison (VANILLA vs TRTLLM)

Requires LTX-2 checkpoint. Does NOT require the LTX-2 reference code.
"""

import gc
import os

import pytest
import torch
import torch.nn.functional as F
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import AttentionConfig, PipelineComponent, VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

os.environ.setdefault("TLLM_DISABLE_MPI", "1")


_LTX2_BASE = os.path.join(str(llm_models_root(check=True)), "LTX-2")

CHECKPOINT_PATH_BF16 = os.environ.get(
    "LTX2_MODEL_PATH",
    os.path.join(_LTX2_BASE, "ltx-2-19b-dev.safetensors"),
)
CHECKPOINT_PATH_FP8 = os.environ.get(
    "LTX2_MODEL_PATH_FP8",
    os.path.join(_LTX2_BASE, "ltx-2-19b-dev-fp8.safetensors"),
)

# Skip non-transformer components.  VisualGenArgs.skip_components is a
# List[PipelineComponent] validated by Pydantic; LTX2-native components
# (audio_vae, vocoder, connectors, video_encoder) will load from the
# checkpoint automatically.
SKIP_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
]


def _get_ltx2_transformer_inputs(transformer, device="cuda", dtype=torch.bfloat16):
    """Create test inputs for the LTX2 transformer (Modality objects).

    Constructs minimal video + audio Modality inputs compatible with the
    transformer's TransformerArgsPreprocessor.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality

    torch.manual_seed(42)
    batch = 1
    n_frames, grid_h, grid_w = 1, 4, 4
    v_patches = n_frames * grid_h * grid_w
    a_patches = 8
    text_len = 8

    cfg = getattr(transformer, "_transformer_config", {})
    in_channels = cfg.get("in_channels", 128)
    caption_channels = cfg.get("caption_channels", 3840)
    audio_in_channels = cfg.get("audio_in_channels", 128)

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

    video = Modality(
        latent=torch.randn(batch, v_patches, in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
        positions=v_positions,
        context=torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype),
    )
    audio = Modality(
        latent=torch.randn(batch, a_patches, audio_in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
        positions=a_positions,
        context=torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype),
    )
    return video, audio


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
def ltx2_bf16_checkpoint_exists():
    """Check if LTX2 BF16 checkpoint is available locally."""
    if not CHECKPOINT_PATH_BF16 or not os.path.exists(CHECKPOINT_PATH_BF16):
        pytest.skip(
            f"LTX2 BF16 checkpoint not found at {CHECKPOINT_PATH_BF16}. "
            "Set LTX2_MODEL_PATH or stage checkpoint under LLM_MODELS_ROOT/LTX-2/."
        )
    return True


@pytest.fixture
def ltx2_fp8_checkpoint_exists():
    """Check if LTX2 FP8 checkpoint is available locally."""
    if not CHECKPOINT_PATH_FP8 or not os.path.exists(CHECKPOINT_PATH_FP8):
        pytest.skip(
            f"LTX2 FP8 checkpoint not found at {CHECKPOINT_PATH_FP8}. "
            "Set LTX2_MODEL_PATH_FP8 or stage checkpoint under LLM_MODELS_ROOT/LTX-2/."
        )
    return True


# ============================================================================
# Quantization Tests
# ============================================================================


class TestLTX2Quantization:
    """Test LTX2 quantization loading and FP8 weight verification."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_load_with_quantization(self, ltx2_bf16_checkpoint_exists, quant_algo: str):
        """Test loading LTX2 with FP8 quantization and verify FP8 weights."""
        args = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )

        pipeline = PipelineLoader(args).load(skip_warmup=True)

        assert pipeline.model_config.quant_config.quant_algo is not None

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


class TestLTX2FP8NumericalCorrectness:
    """Test FP8 vs BF16 numerical accuracy at single-layer level."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_single_layer(self, ltx2_bf16_checkpoint_exists, quant_algo: str):
        """Test FP8 vs BF16 numerical accuracy on a single Linear layer.

        1. Use F.linear() with BF16 weights as ground truth reference
        2. Verify BF16 layer matches F.linear exactly
        3. Compare FP8 layer output against reference
        """
        print(f"\n[Compare {quant_algo}] Loading BF16 pipeline...")
        args_bf16 = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load(skip_warmup=True)

        print(f"[Compare {quant_algo}] Loading {quant_algo} pipeline...")
        args_fp8 = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load(skip_warmup=True)

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


class TestLTX2FP8Memory:
    """Test FP8 memory reduction for LTX2."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_memory_comparison(self, ltx2_bf16_checkpoint_exists):
        """Test FP8 uses ~2x less memory than BF16."""

        def get_module_memory_gb(module):
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_bf16 = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load(skip_warmup=True)

        bf16_model_mem = get_module_memory_gb(pipeline_bf16.transformer)
        print(f"\n[BF16] Transformer memory: {bf16_model_mem:.2f} GB")

        del pipeline_bf16
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        args_fp8 = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load(skip_warmup=True)

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


class TestLTX2AttentionBackend:
    """Test VANILLA vs TRTLLM attention backend numerical correctness."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_backend_comparison(self, ltx2_bf16_checkpoint_exists):
        """Test that VANILLA and TRTLLM backends produce similar outputs.

        Load each backend sequentially (two full LTX2 transformers don't
        fit in GPU memory simultaneously).
        """
        print("\n[Attention Backend Test] Loading baseline transformer (VANILLA)...")
        args_baseline = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend="VANILLA"),
        )
        pipeline_baseline = PipelineLoader(args_baseline).load(skip_warmup=True)
        transformer_baseline = pipeline_baseline.transformer

        video_input, audio_input = _get_ltx2_transformer_inputs(transformer_baseline)

        print("[Attention Backend Test] Running VANILLA transformer forward...")
        with torch.no_grad():
            output_baseline = transformer_baseline(video=video_input, audio=audio_input)
        vout_baseline, aout_baseline = _extract_output(output_baseline)
        vout_baseline_cpu = vout_baseline.cpu() if vout_baseline is not None else None

        del pipeline_baseline, transformer_baseline
        gc.collect()
        torch.cuda.empty_cache()

        print("[Attention Backend Test] Loading TRTLLM transformer...")
        args_trtllm = VisualGenArgs(
            checkpoint_path=CHECKPOINT_PATH_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend="TRTLLM"),
        )
        pipeline_trtllm = PipelineLoader(args_trtllm).load(skip_warmup=True)
        transformer_trtllm = pipeline_trtllm.transformer

        print("[Attention Backend Test] Running TRTLLM transformer forward...")
        with torch.no_grad():
            output_trtllm = transformer_trtllm(video=video_input, audio=audio_input)
        vout_trtllm, aout_trtllm = _extract_output(output_trtllm)
        vout_trtllm_cpu = vout_trtllm.cpu() if vout_trtllm is not None else None

        if vout_baseline_cpu is not None and vout_trtllm_cpu is not None:
            assert vout_baseline_cpu.shape == vout_trtllm_cpu.shape, (
                f"Video output shape mismatch: "
                f"VANILLA={vout_baseline_cpu.shape}, TRTLLM={vout_trtllm_cpu.shape}"
            )

            for name, out in [("VANILLA", vout_baseline_cpu), ("TRTLLM", vout_trtllm_cpu)]:
                assert not torch.isnan(out).any(), f"{name} video output contains NaN"
                assert not torch.isinf(out).any(), f"{name} video output contains Inf"

            v_baseline_float = vout_baseline_cpu.float()
            v_trtllm_float = vout_trtllm_cpu.float()

            max_diff = torch.max(torch.abs(v_trtllm_float - v_baseline_float)).item()
            cos_sim = F.cosine_similarity(
                v_trtllm_float.flatten(), v_baseline_float.flatten(), dim=0
            ).item()

            print(f"\n{'=' * 60}")
            print("TRTLLM vs VANILLA Comparison (Video Output)")
            print(f"{'=' * 60}")
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Cosine similarity: {cos_sim:.6f}")
            print(f"{'=' * 60}")

            assert cos_sim > 0.99, (
                f"TRTLLM should produce similar results to VANILLA: cos_sim={cos_sim:.6f}"
            )
            print(f"\n[PASS] TRTLLM backend matches VANILLA: cos_sim={cos_sim:.6f} (>0.99)")

        del pipeline_trtllm, transformer_trtllm
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Batch Support Unit Tests (no model loading required)
# ============================================================================


class TestLTX2BatchSupport:
    """Test batch support logic without loading the full pipeline."""

    def test_video_pixel_shape_batch_propagation(self):
        """VideoPixelShape(batch=N) propagates through VideoLatentShape."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import (
            VideoLatentShape,
            VideoPixelShape,
        )

        for batch_size in [1, 2, 4]:
            pixel_shape = VideoPixelShape(
                batch=batch_size, frames=9, height=512, width=768, fps=24.0
            )
            video_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
            assert video_shape.batch == batch_size
            torch_shape = video_shape.to_torch_shape()
            assert torch_shape[0] == batch_size

    def test_prompt_normalization(self):
        """forward() normalizes str prompt to List[str] and computes batch_size."""
        # Simulate the normalization logic from forward()
        for prompt_input, expected_batch in [
            ("a cat", 1),
            (["a cat"], 1),
            (["a cat", "a dog"], 2),
        ]:
            prompt = prompt_input
            if isinstance(prompt, str):
                prompt = [prompt]
            assert len(prompt) == expected_batch

    def test_negative_prompt_expansion(self):
        """Negative prompt is expanded to match batch_size."""
        # Simulate the negative prompt expansion logic from forward()
        for neg_input, batch_size, expected_len in [
            ("bad quality", 1, 1),
            ("bad quality", 3, 3),
            (["bad quality"], 3, 3),
            (["bad 1", "bad 2", "bad 3"], 3, 3),
        ]:
            negative_prompt = neg_input
            if isinstance(negative_prompt, str):
                neg_prompt_list = [negative_prompt] * batch_size
            else:
                neg_prompt_list = list(negative_prompt)
                if len(neg_prompt_list) == 1 and batch_size > 1:
                    neg_prompt_list = neg_prompt_list * batch_size
            assert len(neg_prompt_list) == expected_len

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_latent_shape_matches_batch(self):
        """Latents created from VideoLatentShape have correct batch dim."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import (
            VideoLatentShape,
            VideoPixelShape,
        )

        batch_size = 2
        pixel_shape = VideoPixelShape(batch=batch_size, frames=9, height=512, width=768, fps=24.0)
        video_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
        latents = torch.randn(video_shape.to_torch_shape(), device="cuda", dtype=torch.float32)
        assert latents.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
