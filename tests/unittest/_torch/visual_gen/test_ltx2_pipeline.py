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
import json
import os

import pytest
import torch
import torch.nn.functional as F
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2_two_stages as ltx2_two_stages
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2_FORCE_ONE_STAGE_ENV
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, CacheDiTConfig, VisualGenArgs

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

# Skip non-transformer components.  ``skip_components`` is an internal
# escape hatch on ``PipelineLoader.load`` used by unit tests to focus on
# the transformer; LTX-2 native components (audio_vae, vocoder,
# connectors, video_encoder) load from the checkpoint automatically.
SKIP_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
]


_LTX2_BASE = os.path.join(str(llm_models_root(check=True)), "LTX-2")
_GEMMA3_DEFAULT = os.path.join(str(llm_models_root(check=True)), "gemma", "gemma-3-12b-it")


CHECKPOINT_PATH_BF16 = os.environ.get(
    "LTX2_MODEL_PATH",
    os.path.join(_LTX2_BASE, "ltx-2-19b-dev.safetensors"),
)
CHECKPOINT_PATH_FP8 = os.environ.get(
    "LTX2_MODEL_PATH_FP8",
    os.path.join(_LTX2_BASE, "ltx-2-19b-dev-fp8.safetensors"),
)
GEMMA3_PATH = os.environ.get("LTX2_TEXT_ENCODER_PATH", _GEMMA3_DEFAULT)


def _ltx2_pipeline_config(**overrides):
    """Build pipeline_config with the Gemma3 text_encoder_path LTX-2 needs.

    LTX-2's tokenizer + text encoder are loaded from a separate Gemma
    directory (not the diffusion checkpoint), so every full-pipeline
    load needs ``text_encoder_path`` set. Tests can pass extra keys via
    ``overrides`` (e.g. ``spatial_upsampler_path`` for two-stage).
    """
    cfg = {"text_encoder_path": GEMMA3_PATH}
    cfg.update(overrides)
    return cfg


def _write_minimal_ltx2_native_checkpoint(tmp_path):
    import safetensors.torch

    checkpoint_path = tmp_path / "ltx-2-19b-dev.safetensors"
    safetensors.torch.save_file(
        {"__metadata_marker__": torch.zeros(1)},
        str(checkpoint_path),
        metadata={"config": json.dumps({"transformer": {"_class_name": "LTX2"}})},
    )
    return checkpoint_path


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

    v_context = torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype)
    a_context = torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype)

    video = Modality(
        latent=torch.randn(batch, v_patches, in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
        positions=v_positions,
        context=v_context,
    )
    audio = Modality(
        latent=torch.randn(batch, a_patches, audio_in_channels, device=device, dtype=dtype),
        timesteps=torch.tensor([0.5], device=device),
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
            model=CHECKPOINT_PATH_BF16,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline_config=_ltx2_pipeline_config(),
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
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx2_pipeline_config(),
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )

        print(f"[Compare {quant_algo}] Loading {quant_algo} pipeline...")
        args_fp8 = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline_config=_ltx2_pipeline_config(),
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
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx2_pipeline_config(),
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
            pipeline_config=_ltx2_pipeline_config(),
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
            model=CHECKPOINT_PATH_BF16,
            attention_config=AttentionConfig(backend="VANILLA"),
            pipeline_config=_ltx2_pipeline_config(),
        )
        pipeline_baseline = PipelineLoader(args_baseline).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )
        transformer_baseline = pipeline_baseline.transformer

        video_input, audio_input, text_cache_baseline = _get_ltx2_transformer_inputs(
            transformer_baseline
        )

        print("[Attention Backend Test] Running VANILLA transformer forward...")
        with torch.no_grad():
            output_baseline = transformer_baseline(
                video=video_input, audio=audio_input, text_cache=text_cache_baseline
            )
        vout_baseline, aout_baseline = _extract_output(output_baseline)
        vout_baseline_cpu = vout_baseline.cpu() if vout_baseline is not None else None

        del pipeline_baseline, transformer_baseline
        gc.collect()
        torch.cuda.empty_cache()

        print("[Attention Backend Test] Loading TRTLLM transformer...")
        args_trtllm = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            attention_config=AttentionConfig(backend="TRTLLM"),
            pipeline_config=_ltx2_pipeline_config(),
        )
        pipeline_trtllm = PipelineLoader(args_trtllm).load(
            skip_warmup=True, skip_components=SKIP_COMPONENTS
        )
        transformer_trtllm = pipeline_trtllm.transformer

        print("[Attention Backend Test] Running TRTLLM transformer forward...")
        _, _, text_cache_trtllm = _get_ltx2_transformer_inputs(transformer_trtllm)
        with torch.no_grad():
            output_trtllm = transformer_trtllm(
                video=video_input, audio=audio_input, text_cache=text_cache_trtllm
            )
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


# ============================================================================
# Two-Stage Pipeline Unit Tests (no model loading required)
# ============================================================================


class TestTwoStageLoRAHelpers:
    """Test LoRA delta loading and application without checkpoints."""

    def test_bf16_weight_snapshot_gate_uses_cuda_free_memory(self, monkeypatch):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _should_save_bf16_weights,
        )

        gib = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (116 * gib, 180 * gib))
        assert _should_save_bf16_weights()

        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (115 * gib, 180 * gib))
        assert not _should_save_bf16_weights()

        def _raise_mem_query_error():
            raise RuntimeError("mem_get_info failed")

        monkeypatch.setattr(torch.cuda, "mem_get_info", _raise_mem_query_error)
        assert not _should_save_bf16_weights()

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert not _should_save_bf16_weights()

    def test_bf16_weight_snapshot_saved_when_requested(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
            _restore_lora_state,
            _subtract_dense_lora_deltas,
        )

        linear = torch.nn.Linear(32, 32, bias=False).bfloat16()
        original_weight = linear.weight.data.clone()
        deltas = {"weight": torch.randn(32, 32) * 0.1}

        applied, saved_state, snapshot_required = _apply_lora_deltas(
            linear,
            deltas,
            sign=1.0,
            save_bf16_weights=True,
        )

        assert applied == 1
        assert snapshot_required == 1
        assert "weight" in saved_state
        assert torch.allclose(saved_state["weight"], original_weight)
        assert not torch.allclose(linear.weight.data, original_weight)

        removed = _subtract_dense_lora_deltas(linear, deltas, saved_state)
        assert removed == 0

        _restore_lora_state(linear, saved_state)
        assert torch.allclose(linear.weight.data, original_weight)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_and_remove_deltas_bf16(self):
        """Merge then unmerge in BF16 should leave weights approximately unchanged."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
            _subtract_dense_lora_deltas,
        )

        device = "cuda"
        linear = torch.nn.Linear(64, 64, bias=False).to(device).bfloat16()
        original_weight = linear.weight.data.clone()

        delta = torch.randn(64, 64, device=device) * 0.01
        deltas = {"weight": delta}

        applied, saved_state, snapshot_required = _apply_lora_deltas(linear, deltas, sign=1.0)
        assert applied == 1, "Expected one parameter to be modified"
        assert saved_state == {}, "BF16 weights should not be snapshotted by default"
        assert snapshot_required == 0
        assert not torch.allclose(linear.weight.data, original_weight), (
            "Weights should have changed after applying delta"
        )

        removed = _subtract_dense_lora_deltas(linear, deltas, saved_state)
        assert removed == 1, "Expected one dense parameter to be unmerged"
        drift = (linear.weight.data.float() - original_weight.float()).abs().max().item()
        assert drift < 0.05, f"bf16 merge/unmerge drift too large: {drift:.2e}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_skips_missing_params(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
        )

        device = "cuda"
        linear = torch.nn.Linear(8, 8, bias=False).to(device)
        original_weight = linear.weight.data.clone()

        deltas = {"nonexistent_param.weight": torch.randn(8, 8, device=device)}
        applied, _, snapshot_required = _apply_lora_deltas(linear, deltas, sign=1.0)
        assert applied == 0
        assert snapshot_required == 0
        assert torch.allclose(linear.weight.data, original_weight)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_round_drift_bounded(self):
        """After N merge+unmerge rounds the drift stays bounded."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
            _subtract_dense_lora_deltas,
        )

        device = "cuda"
        model = torch.nn.Linear(64, 64, bias=False).to(device).bfloat16()
        original = model.weight.data.clone()

        deltas = {"weight": torch.randn(64, 64, device=device) * 0.01}
        rounds = 10
        for _ in range(rounds):
            _, saved_state, snapshot_required = _apply_lora_deltas(model, deltas, sign=1.0)
            assert saved_state == {}, "BF16 weights should not be snapshotted by default"
            assert snapshot_required == 0
            _subtract_dense_lora_deltas(model, deltas, saved_state)

        drift = (model.weight.data.float() - original.float()).abs().max().item()
        assert drift < 0.1, f"bf16 drift after {rounds} rounds too large: {drift:.2e}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp32_state_not_saved_and_subtract_restores(self):
        """FP32 weights restore by subtraction, even when BF16 snapshots are enabled."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
            _subtract_dense_lora_deltas,
        )

        device = "cuda"
        linear = torch.nn.Linear(32, 32, bias=False).to(device).float()
        original_weight = linear.weight.data.clone()

        deltas = {"weight": torch.randn(32, 32, device=device) * 0.1}
        _, saved_state, snapshot_required = _apply_lora_deltas(
            linear,
            deltas,
            sign=1.0,
            save_bf16_weights=True,
        )

        assert saved_state == {}, "FP32 weights should not use the BF16 snapshot path"
        assert snapshot_required == 0
        assert not torch.allclose(linear.weight.data, original_weight)

        removed = _subtract_dense_lora_deltas(linear, deltas, saved_state)
        assert removed == 1
        assert torch.allclose(linear.weight.data, original_weight), (
            "FP32 LoRA subtraction should restore weights"
        )


class TestTwoStageLoRAFileLoading:
    """Test LoRA loading from synthetic safetensors files."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_lora_deltas_shape(self):
        import tempfile

        import safetensors.torch

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _load_lora_deltas,
        )

        device = "cuda"
        rank, in_f, out_f = 4, 16, 16
        tensors = {
            "model.diffusion_model.linear.weight.lora_A.weight": torch.randn(rank, in_f),
            "model.diffusion_model.linear.weight.lora_B.weight": torch.randn(out_f, rank),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            safetensors.torch.save_file(tensors, tmp.name)
            tmp_path = tmp.name

        try:
            module = torch.nn.Module()
            module.linear = torch.nn.Linear(in_f, out_f, bias=False)
            module = module.to(device)

            deltas = _load_lora_deltas(tmp_path, module, strength=1.0)
            assert "linear.weight" in deltas
            assert deltas["linear.weight"].shape == (out_f, in_f)
        finally:
            os.unlink(tmp_path)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lora_alpha_scaling(self):
        """Verify alpha/rank scaling is applied correctly."""
        import tempfile

        import safetensors.torch

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _load_lora_deltas,
        )

        rank, in_f, out_f = 4, 8, 8
        alpha_val = 8.0
        A = torch.randn(rank, in_f)
        B = torch.randn(out_f, rank)

        tensors = {
            "model.diffusion_model.w.lora_A.weight": A,
            "model.diffusion_model.w.lora_B.weight": B,
            "model.diffusion_model.w.alpha": torch.tensor(alpha_val),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            safetensors.torch.save_file(tensors, tmp.name)
            tmp_path = tmp.name

        try:
            module = torch.nn.Module()
            module.register_parameter("w", torch.nn.Parameter(torch.zeros(out_f, in_f)))

            deltas = _load_lora_deltas(tmp_path, module, strength=1.0)
            expected_scale = 1.0 * alpha_val / rank
            expected_delta = (B.float() @ A.float()) * expected_scale
            assert torch.allclose(deltas["w"], expected_delta, atol=1e-5), (
                "LoRA alpha scaling mismatch"
            )
        finally:
            os.unlink(tmp_path)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lora_strength_scaling(self):
        """Verify strength parameter scales the delta correctly."""
        import tempfile

        import safetensors.torch

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _load_lora_deltas,
        )

        rank, in_f, out_f = 4, 8, 8
        A = torch.randn(rank, in_f)
        B = torch.randn(out_f, rank)

        tensors = {
            "model.diffusion_model.w.lora_A.weight": A,
            "model.diffusion_model.w.lora_B.weight": B,
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            safetensors.torch.save_file(tensors, tmp.name)
            tmp_path = tmp.name

        try:
            module = torch.nn.Module()
            module.register_parameter("w", torch.nn.Parameter(torch.zeros(out_f, in_f)))

            deltas_s1 = _load_lora_deltas(tmp_path, module, strength=1.0)
            deltas_s05 = _load_lora_deltas(tmp_path, module, strength=0.5)

            ratio = deltas_s1["w"] / deltas_s05["w"]
            assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-5), (
                "strength=1.0 should produce 2x the delta of strength=0.5"
            )
        finally:
            os.unlink(tmp_path)


class TestTwoStagePipelineVariantResolution:
    """Test that LTX2Pipeline.resolve_variant selects the correct class."""

    def test_resolve_variant_returns_two_stage_when_configured(self, monkeypatch):
        """When both upsampler and LoRA paths are set, resolve_variant returns TwoStages."""
        monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            LTX2TwoStagesPipeline,
        )

        config = MagicMock()
        config.primary_pretrained_config._name_or_path = ""
        config.extra_attrs = {
            "spatial_upsampler_path": "/fake/upsampler.safetensors",
            "distilled_lora_path": "/fake/lora.safetensors",
        }

        result = LTX2Pipeline.resolve_variant(config)
        assert result is LTX2TwoStagesPipeline

    def test_resolve_variant_honors_force_one_stage_env(self, monkeypatch):
        """The env knob should prevent promotion even when two-stage paths exist."""
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        monkeypatch.setenv(LTX2_FORCE_ONE_STAGE_ENV, "1")
        config = MagicMock()
        config.primary_pretrained_config._name_or_path = ""
        config.extra_attrs = {
            "spatial_upsampler_path": "/fake/upsampler.safetensors",
            "distilled_lora_path": "/fake/lora.safetensors",
        }

        result = LTX2Pipeline.resolve_variant(config)
        assert result is LTX2Pipeline

    def test_resolve_variant_returns_base_without_two_stage_config(self):
        """Without upsampler/LoRA paths, resolve_variant returns base LTX2Pipeline."""
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        config = MagicMock()
        config.primary_pretrained_config._name_or_path = ""
        config.extra_attrs = {}

        result = LTX2Pipeline.resolve_variant(config)
        assert result is LTX2Pipeline

    def test_resolve_variant_requires_both_paths(self):
        """Only one of upsampler/LoRA paths should NOT trigger two-stage."""
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        config = MagicMock()
        config.primary_pretrained_config._name_or_path = ""
        config.extra_attrs = {"spatial_upsampler_path": "/fake/upsampler.safetensors"}

        result = LTX2Pipeline.resolve_variant(config)
        assert result is LTX2Pipeline


class TestLTX2ForceOneStageEnv:
    """Test force-one-stage env-var behavior during LTX2 variant selection."""

    def test_two_stage_auxiliary_paths_are_discovered_by_default(self, tmp_path, monkeypatch):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            LTX2TwoStagesPipeline,
        )

        monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
        checkpoint_path = _write_minimal_ltx2_native_checkpoint(tmp_path)
        upsampler_path = checkpoint_path.parent / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        lora_path = checkpoint_path.parent / "ltx-2-19b-distilled-lora-384.safetensors"
        upsampler_path.touch()
        lora_path.touch()

        args = VisualGenArgs(model=str(checkpoint_path))
        config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_path), args=args)

        assert LTX2Pipeline.resolve_variant(config) is LTX2TwoStagesPipeline
        assert config.extra_attrs["spatial_upsampler_path"] == str(upsampler_path)
        assert config.extra_attrs["distilled_lora_path"] == str(lora_path)

    def test_force_one_stage_env_skips_auto_discovery(self, tmp_path, monkeypatch):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        monkeypatch.setenv(LTX2_FORCE_ONE_STAGE_ENV, "1")
        checkpoint_path = _write_minimal_ltx2_native_checkpoint(tmp_path)
        upsampler_path = checkpoint_path.parent / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        lora_path = checkpoint_path.parent / "ltx-2-19b-distilled-lora-384.safetensors"
        upsampler_path.touch()
        lora_path.touch()

        args = VisualGenArgs(model=str(checkpoint_path))
        config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_path), args=args)

        assert LTX2Pipeline.resolve_variant(config) is LTX2Pipeline
        assert "spatial_upsampler_path" not in config.extra_attrs
        assert "distilled_lora_path" not in config.extra_attrs

    def test_force_one_stage_env_prevents_promotion_with_explicit_auxiliary_paths(
        self, tmp_path, monkeypatch
    ):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        monkeypatch.setenv(LTX2_FORCE_ONE_STAGE_ENV, "1")
        checkpoint_path = _write_minimal_ltx2_native_checkpoint(tmp_path)

        args = VisualGenArgs(
            model=str(checkpoint_path),
            pipeline_config={
                "spatial_upsampler_path": "/fake/upsampler.safetensors",
                "distilled_lora_path": "/fake/lora.safetensors",
            },
        )
        config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_path), args=args)

        assert config.extra_attrs["spatial_upsampler_path"] == "/fake/upsampler.safetensors"
        assert config.extra_attrs["distilled_lora_path"] == "/fake/lora.safetensors"
        assert LTX2Pipeline.resolve_variant(config) is LTX2Pipeline

    def test_cache_dit_config_prevents_promotion_with_explicit_auxiliary_paths(
        self, tmp_path, monkeypatch
    ):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
        checkpoint_path = _write_minimal_ltx2_native_checkpoint(tmp_path)

        args = VisualGenArgs(
            model=str(checkpoint_path),
            cache_config=CacheDiTConfig(),
            pipeline_config={
                "spatial_upsampler_path": "/fake/upsampler.safetensors",
                "distilled_lora_path": "/fake/lora.safetensors",
            },
        )
        config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_path), args=args)

        assert config.cache_backend == "cache_dit"
        assert config.extra_attrs["spatial_upsampler_path"] == "/fake/upsampler.safetensors"
        assert config.extra_attrs["distilled_lora_path"] == "/fake/lora.safetensors"
        assert LTX2Pipeline.resolve_variant(config) is LTX2Pipeline


class TestTwoStageUpsamplerBuildingBlocks:
    """Test upsampler components without checkpoints (random weights)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_latent_upsampler_spatial_2x(self):
        """LatentUpsampler doubles spatial dims while preserving temporal."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.upsampler import LatentUpsampler

        device = "cuda"
        model = (
            LatentUpsampler(
                in_channels=16,
                mid_channels=32,
                num_blocks_per_stage=1,
                dims=3,
                spatial_upsample=True,
                temporal_upsample=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        x = torch.randn(1, 16, 4, 8, 8, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 16, 4, 16, 16), f"Expected (1,16,4,16,16), got {y.shape}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_latent_upsampler_configurator_default(self):
        """Default configurator creates 128-channel upsampler matching LTX-2 spec."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.upsampler import (
            LatentUpsamplerConfigurator,
        )

        device = "cuda"
        model = LatentUpsamplerConfigurator.from_config({})
        model = model.to(device).to(torch.bfloat16)

        assert model.in_channels == 128
        assert model.mid_channels == 512

        x = torch.randn(1, 128, 2, 4, 4, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 128, 2, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_upsample_video_helper(self):
        """upsample_video performs un-normalize → upsample → re-normalize."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.upsampler import (
            LatentUpsampler,
            upsample_video,
        )

        device = "cuda"
        ch = 16
        upsampler = (
            LatentUpsampler(
                in_channels=ch,
                mid_channels=32,
                num_blocks_per_stage=1,
                dims=3,
                spatial_upsample=True,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        # Minimal mock of per-channel stats
        stats = torch.nn.Module()
        stats.register_buffer("std-of-means", torch.ones(ch, device=device))
        stats.register_buffer("mean-of-means", torch.zeros(ch, device=device))

        def _un_normalize(self, x):
            std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
            mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
            return x * std + mean

        def _normalize(self, x):
            std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
            mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
            return (x - mean) / std

        import types

        stats.un_normalize = types.MethodType(_un_normalize, stats)
        stats.normalize = types.MethodType(_normalize, stats)

        x = torch.randn(1, ch, 4, 8, 8, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y = upsample_video(x, stats, upsampler)
        assert y.shape == (1, ch, 4, 16, 16), f"Expected (1,{ch},4,16,16), got {y.shape}"
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"


class TestTwoStageFP4Helpers:
    """Test NVFP4 quantization helpers used in LoRA merge on quantized weights."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pad_up(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import _pad_up

        assert _pad_up(100, 128) == 128
        assert _pad_up(128, 128) == 128
        assert _pad_up(129, 128) == 256
        assert _pad_up(0, 128) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_scale_packed_detection(self):
        """_is_fp8_scale_packed distinguishes standard vs packed layouts."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _is_fp8_scale_packed,
        )

        device = "cuda"
        # Standard float32 grid is NOT packed
        standard_scale = torch.randn(256, 4, device=device)
        assert not _is_fp8_scale_packed(standard_scale, 256, 512)

        # Wrong dtype is NOT packed
        wrong_dtype = torch.randn(256, 1, device=device, dtype=torch.float32)
        assert not _is_fp8_scale_packed(wrong_dtype, 256, 128)


# ============================================================================
# Two-Stage Pipeline Loading Tests (requires checkpoints)
# ============================================================================

UPSAMPLER_PATH = os.environ.get(
    "LTX2_UPSAMPLER_CHECKPOINT",
    os.path.join(_LTX2_BASE, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
)
LORA_PATH = os.environ.get(
    "LTX2_LORA_CHECKPOINT",
    os.path.join(_LTX2_BASE, "ltx-2-19b-distilled-lora-384.safetensors"),
)

_HAS_TWO_STAGE_ASSETS = (
    os.path.exists(CHECKPOINT_PATH_BF16)
    and os.path.exists(UPSAMPLER_PATH)
    and os.path.exists(LORA_PATH)
)


@pytest.fixture
def ltx2_two_stage_assets_exist():
    """Check if all two-stage assets (checkpoint + upsampler + LoRA) are available."""
    if not _HAS_TWO_STAGE_ASSETS:
        pytest.skip(
            f"Two-stage assets not found. Need checkpoint at {CHECKPOINT_PATH_BF16}, "
            f"upsampler at {UPSAMPLER_PATH}, and LoRA at {LORA_PATH}."
        )
    return True


class TestLTX2TwoStageLoRAHelpers:
    """Test LTX-2 two-stage distilled LoRA helpers without model loading."""

    def test_bf16_snapshot_gate_uses_preload_memory(self, monkeypatch):
        """Pre-load memory is the whole-pipeline budget and wins over current memory."""
        queried_devices = []

        def fake_get_free_gpu_memory_gib(device=None):
            queried_devices.append(device)
            return 1.0

        monkeypatch.setattr(
            ltx2_two_stages,
            "_get_free_gpu_memory_gib",
            fake_get_free_gpu_memory_gib,
        )

        assert ltx2_two_stages._should_save_bf16_weights(
            device="cuda:1",
            preload_free_gib=120.0,
            threshold_gib=115.0,
        )
        assert not ltx2_two_stages._should_save_bf16_weights(
            device="cuda:1",
            preload_free_gib=110.0,
            threshold_gib=115.0,
        )
        assert queried_devices == []

    def test_bf16_snapshot_gate_fallback_passes_device(self, monkeypatch):
        """Fallback free-memory query must target the transformer's CUDA device."""
        queried_devices = []

        def fake_get_free_gpu_memory_gib(device=None):
            queried_devices.append(device)
            return 116.0

        monkeypatch.setattr(
            ltx2_two_stages,
            "_get_free_gpu_memory_gib",
            fake_get_free_gpu_memory_gib,
        )

        assert ltx2_two_stages._should_save_bf16_weights(
            device="cuda:1",
            preload_free_gib=None,
            threshold_gib=115.0,
        )
        assert queried_devices == ["cuda:1"]

    def test_persistent_bf16_cache_reuses_weight_storage(self):
        """Persistent BF16 cache swaps between the same original and merged tensors."""

        class TinyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.bfloat16,
                    )
                )

        module = TinyModule()
        original = module.weight.detach().clone()
        delta = torch.full_like(module.weight, 0.5)

        cache = ltx2_two_stages._PersistentLoRAWeightCache.build(
            module,
            {"weight": delta},
        )
        assert cache.applied_count == 1
        assert cache.precision_counts() == {"bf16": 1}

        original_ptr = module.weight.data_ptr()
        cache.bind_merged()
        merged_ptr = module.weight.data_ptr()
        assert merged_ptr != original_ptr
        assert torch.allclose(module.weight, original + delta)

        cache.bind_original()
        assert module.weight.data_ptr() == original_ptr
        assert torch.equal(module.weight, original)

        cache.bind_merged()
        assert module.weight.data_ptr() == merged_ptr
        assert torch.allclose(module.weight, original + delta)

        cache.bind_original()
        assert module.weight.data_ptr() == original_ptr
        assert torch.equal(module.weight, original)

    def test_persistent_fp8_cache_reuses_weight_and_scale_storage(self):
        """Persistent FP8 cache swaps both quantized weight and scale storage."""

        class TinyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Module()
                bf16_weight = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=torch.bfloat16,
                )
                weight_scale = torch.tensor(0.25, dtype=torch.float32)
                self.proj.weight = torch.nn.Parameter(
                    (bf16_weight.float() / weight_scale).to(torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.proj.weight_scale = torch.nn.Parameter(
                    weight_scale,
                    requires_grad=False,
                )

        module = TinyModule()
        original_weight = module.proj.weight.detach().clone()
        original_scale = module.proj.weight_scale.detach().clone()
        delta = torch.full((2, 2), 0.5, dtype=torch.bfloat16)

        cache = ltx2_two_stages._PersistentLoRAWeightCache.build(
            module,
            {"proj.weight": delta},
        )
        assert cache.applied_count == 1
        assert cache.precision_counts() == {"fp8": 1}

        original_weight_ptr = module.proj.weight.data_ptr()
        original_scale_ptr = module.proj.weight_scale.data_ptr()
        cache.bind_merged()
        merged_weight_ptr = module.proj.weight.data_ptr()
        merged_scale_ptr = module.proj.weight_scale.data_ptr()
        assert merged_weight_ptr != original_weight_ptr
        assert merged_scale_ptr != original_scale_ptr
        assert module.proj.weight.dtype == torch.float8_e4m3fn

        cache.bind_original()
        assert module.proj.weight.data_ptr() == original_weight_ptr
        assert module.proj.weight_scale.data_ptr() == original_scale_ptr
        assert torch.equal(module.proj.weight, original_weight)
        assert torch.equal(module.proj.weight_scale, original_scale)

        cache.bind_merged()
        assert module.proj.weight.data_ptr() == merged_weight_ptr
        assert module.proj.weight_scale.data_ptr() == merged_scale_ptr

        cache.bind_original()
        assert module.proj.weight.data_ptr() == original_weight_ptr
        assert module.proj.weight_scale.data_ptr() == original_scale_ptr

    def test_persistent_fp4_cache_swaps_quant_method_and_weight_storage(self, monkeypatch):
        """Persistent FP4 cache binds merged BF16 weight and restores packed state."""

        class TinyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(
                    2,
                    2,
                    bias=False,
                    dtype=torch.bfloat16,
                    skip_create_weights_in_init=True,
                    reduce_output=False,
                )
                self.linear.weight = torch.nn.Parameter(
                    torch.zeros((2, 1), dtype=torch.uint8),
                    requires_grad=False,
                )
                self.linear.weight_scale = torch.nn.Parameter(
                    torch.ones((128, 4), dtype=torch.uint8),
                    requires_grad=False,
                )
                self.linear.weight_scale_2 = torch.nn.Parameter(
                    torch.ones((), dtype=torch.float32),
                    requires_grad=False,
                )
                self.linear.quant_method = object()

        module = TinyModule()
        original_quant_method = module.linear.quant_method
        original_weight_ptr = module.linear.weight.data_ptr()
        original_weight = module.linear.weight.detach().clone()
        delta = torch.full((2, 2), 0.5, dtype=torch.bfloat16)

        def fake_dequantize_fp4_weight(
            packed_weight,
            interleaved_scale,
            weight_scale_2,
            out_features,
            in_features,
        ):
            assert packed_weight.data_ptr() == original_weight_ptr
            assert out_features == 2
            assert in_features == 2
            return torch.ones((2, 2), dtype=torch.bfloat16)

        monkeypatch.setattr(
            ltx2_two_stages,
            "_dequantize_fp4_weight",
            fake_dequantize_fp4_weight,
        )

        cache = ltx2_two_stages._PersistentLoRAWeightCache.build(
            module,
            {"linear.weight": delta},
        )
        assert cache.applied_count == 1
        assert cache.precision_counts() == {"fp4": 1}

        cache.bind_merged()
        merged_weight_ptr = module.linear.weight.data_ptr()
        assert merged_weight_ptr != original_weight_ptr
        assert module.linear.weight.dtype == torch.bfloat16
        assert torch.equal(module.linear.weight, torch.ones_like(delta) + delta)
        assert isinstance(module.linear.quant_method, ltx2_two_stages.UnquantizedLinearMethod)

        cache.bind_original()
        assert module.linear.weight.data_ptr() == original_weight_ptr
        assert torch.equal(module.linear.weight, original_weight)
        assert module.linear.quant_method is original_quant_method

        cache.bind_merged()
        assert module.linear.weight.data_ptr() == merged_weight_ptr
        cache.bind_original()
        assert module.linear.weight.data_ptr() == original_weight_ptr

    def test_apply_lora_deltas_rolls_back_dense_weights_on_failure(self):
        """A later merge failure must not leave earlier dense weights LoRA-merged."""

        class TinyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.good = torch.nn.Parameter(
                    torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.bfloat16,
                    )
                )
                self.bad = torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.bfloat16))

        module = TinyModule()
        original_good = module.good.detach().clone()
        deltas = {
            "good": torch.full_like(module.good, 0.5),
            "bad": torch.ones((2, 2), dtype=torch.bfloat16),
        }

        with pytest.raises(RuntimeError, match="missing bad.weight_scale"):
            ltx2_two_stages._apply_lora_deltas(
                module,
                deltas,
                sign=1.0,
                save_bf16_weights=False,
            )

        assert torch.equal(module.good, original_good)


class TestLTX2TwoStagePipelineLoading:
    """Test two-stage pipeline loading via PipelineLoader."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loads_as_two_stage_pipeline(self, ltx2_two_stage_assets_exist):
        """When upsampler + LoRA paths are set, PipelineLoader yields LTX2TwoStagesPipeline."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            LTX2TwoStagesPipeline,
        )

        args = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx2_pipeline_config(
                spatial_upsampler_path=UPSAMPLER_PATH,
                distilled_lora_path=LORA_PATH,
            ),
        )

        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=SKIP_COMPONENTS)
        try:
            assert isinstance(pipeline, LTX2TwoStagesPipeline), (
                f"Expected LTX2TwoStagesPipeline, got {type(pipeline).__name__}"
            )
            assert pipeline.spatial_upsampler is not None, "Spatial upsampler should be loaded"
            assert len(pipeline._distilled_lora_deltas) > 0, (
                "Distilled LoRA deltas should be loaded"
            )

            print(f"\n[Two-Stage] Loaded {len(pipeline._distilled_lora_deltas)} LoRA deltas")
            print(f"[Two-Stage] Upsampler type: {type(pipeline.spatial_upsampler).__name__}")
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_two_stage_lora_deltas_match_transformer(self, ltx2_two_stage_assets_exist):
        """Loaded LoRA deltas should match transformer parameter names."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            _apply_lora_deltas,
            _subtract_dense_lora_deltas,
        )

        args = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx2_pipeline_config(
                spatial_upsampler_path=UPSAMPLER_PATH,
                distilled_lora_path=LORA_PATH,
            ),
        )

        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=SKIP_COMPONENTS)
        try:
            applied, saved_state, snapshot_required = _apply_lora_deltas(
                pipeline.transformer,
                pipeline._distilled_lora_deltas,
                sign=1.0,
            )
            total = len(pipeline._distilled_lora_deltas)
            match_rate = applied / total * 100

            print(f"\n[Two-Stage] LoRA apply rate: {match_rate:.1f}% ({applied}/{total})")
            assert match_rate > 99.0, f"Expected >99% LoRA match rate, got {match_rate:.1f}%"

            assert saved_state == {}, "BF16 checkpoint should not snapshot weights by default"
            assert snapshot_required == 0

            # Verify dense unmerge by subtraction
            removed = _subtract_dense_lora_deltas(
                pipeline.transformer,
                pipeline._distilled_lora_deltas,
                saved_state,
            )
            assert removed == applied, (
                f"Unmerge applied {removed} deltas, but merge applied {applied}"
            )
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_two_stage_with_quantization(self, ltx2_two_stage_assets_exist, quant_algo: str):
        """Two-stage pipeline loads correctly with FP8 quantization."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            LTX2TwoStagesPipeline,
        )

        args = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            pipeline_config=_ltx2_pipeline_config(
                spatial_upsampler_path=UPSAMPLER_PATH,
                distilled_lora_path=LORA_PATH,
            ),
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )

        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=SKIP_COMPONENTS)
        try:
            assert isinstance(pipeline, LTX2TwoStagesPipeline)
            assert pipeline.pipeline_config.quant_config.quant_algo is not None

            quant_count = sum(
                1
                for _, m in pipeline.transformer.named_modules()
                if isinstance(m, Linear) and m.quant_config and m.quant_config.quant_algo
            )
            print(f"\n[Two-Stage {quant_algo}] Quantized {quant_count} Linear layers")
            assert quant_count > 0, "No layers were quantized"
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()


class TestTwoStageCommonWarmupShapes:
    """Test LTX2TwoStagesPipeline.common_warmup_shapes property."""

    def test_warmup_shapes_defined(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
            LTX2TwoStagesPipeline,
        )

        shapes = LTX2TwoStagesPipeline.common_warmup_shapes.fget(None)
        assert shapes is not None
        assert len(shapes) > 0
        for shape in shapes:
            assert len(shape) == 3, f"Warmup shape should be (H, W, F), got {shape}"
            h, w, f = shape
            assert h % 64 == 0, f"Warmup height {h} not divisible by 64"
            assert w % 64 == 0, f"Warmup width {w} not divisible by 64"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
