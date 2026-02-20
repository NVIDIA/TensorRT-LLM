# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for FLUX pipelines.

Tests cover:
- Pipeline loading (FLUX.1 and FLUX.2)
- Quantization (FP8, FP8_BLOCK_SCALES)
- Single-layer numerical correctness (F.linear reference)
- Full transformer E2E numerical correctness
- Memory usage comparison
- Attention backend comparison (VANILLA vs TRTLLM)
- Multi-GPU parallelism (Ulysses sequence parallelism, 2+ GPUs)
"""

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionArgs, PipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "You shall set LLM_MODELS_ROOT env or be able to access scratch.trt_llm_data to run this test"
    )
    return str(root)


# Checkpoint paths for integration tests
FLUX1_CHECKPOINT_PATH = os.environ.get(
    "FLUX1_MODEL_PATH",
    os.path.join(_llm_models_root(), "FLUX.1-dev"),
)
FLUX2_CHECKPOINT_PATH = os.environ.get(
    "FLUX2_MODEL_PATH",
    os.path.join(_llm_models_root(), "FLUX.2-dev"),
)
SKIP_COMPONENTS = ["text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2", "scheduler"]
# When skip_components includes tokenizer, warmup must be disabled (warmup calls _encode_prompt)
PIPELINE_NO_WARMUP = PipelineConfig(warmup_steps=0, enable_torch_compile=False)


def _get_flux_transformer_inputs(transformer, device="cuda", dtype=torch.bfloat16):
    """Create test inputs appropriate for a FLUX.1 transformer.

    Inspects the transformer's config to determine the correct input shapes.
    Note: Generates 3-axis position IDs (FLUX.1 only). FLUX.2 uses 4-axis IDs.
    """
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 64
    text_seq_len = 32

    config = transformer.config
    in_channels = getattr(config, "in_channels", 64)
    joint_attention_dim = getattr(config, "joint_attention_dim", 4096)
    pooled_projection_dim = getattr(config, "pooled_projection_dim", 768)

    hidden_states = torch.randn(batch_size, seq_len, in_channels, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, joint_attention_dim, device=device, dtype=dtype
    )
    pooled_projections = torch.randn(batch_size, pooled_projection_dim, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    guidance = torch.tensor([3.5], device=device, dtype=dtype)
    img_ids = torch.zeros(batch_size, seq_len, 3, device=device)
    txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=device)

    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        guidance=guidance,
        img_ids=img_ids,
        txt_ids=txt_ids,
    )


def _extract_transformer_output(output):
    """Extract tensor from transformer output (dict or tuple)."""
    if isinstance(output, dict):
        return output["sample"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _find_first_quantizable_linear(transformer):
    """Find the first Linear layer in transformer blocks suitable for testing."""
    # Try attention QKV in first transformer block
    if hasattr(transformer, "transformer_blocks") and len(transformer.transformer_blocks) > 0:
        block = transformer.transformer_blocks[0]
        if hasattr(block, "attn") and hasattr(block.attn, "qkv_proj"):
            return block.attn.qkv_proj, "transformer_blocks.0.attn.qkv_proj"
    # Try single transformer blocks
    if (
        hasattr(transformer, "single_transformer_blocks")
        and len(transformer.single_transformer_blocks) > 0
    ):
        block = transformer.single_transformer_blocks[0]
        if hasattr(block, "attn") and hasattr(block.attn, "qkv_proj"):
            return block.attn.qkv_proj, "single_transformer_blocks.0.attn.qkv_proj"
    # Fallback: first Linear in any block
    for name, module in transformer.named_modules():
        if isinstance(module, Linear) and "blocks" in name:
            return module, name
    return None, None


@pytest.fixture
def flux1_checkpoint_exists():
    """Check if FLUX.1 checkpoint is available locally."""
    if not FLUX1_CHECKPOINT_PATH or not os.path.exists(FLUX1_CHECKPOINT_PATH):
        pytest.skip(
            f"FLUX.1 checkpoint not found at {FLUX1_CHECKPOINT_PATH}. "
            "Set FLUX1_MODEL_PATH or stage checkpoint under LLM_MODELS_ROOT."
        )
    return True


@pytest.fixture
def flux2_checkpoint_exists():
    """Check if FLUX.2 checkpoint is available locally."""
    if not FLUX2_CHECKPOINT_PATH or not os.path.exists(FLUX2_CHECKPOINT_PATH):
        pytest.skip(
            f"FLUX.2 checkpoint not found at {FLUX2_CHECKPOINT_PATH}. "
            "Set FLUX2_MODEL_PATH or stage checkpoint under LLM_MODELS_ROOT."
        )
    return True


# =============================================================================
# Pipeline Loading Tests
# =============================================================================


class TestFluxPipelineLoading:
    """Integration tests for FLUX pipeline loading."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_flux1_pipeline_basic(self, flux1_checkpoint_exists):
        """Test loading FLUX.1 pipeline."""
        args = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )

        pipeline = PipelineLoader(args).load()

        assert pipeline is not None
        assert hasattr(pipeline, "transformer")
        assert pipeline.transformer is not None
        assert pipeline.model_config.attention.backend == "VANILLA"

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_flux2_pipeline_basic(self, flux2_checkpoint_exists):
        """Test loading FLUX.2 pipeline."""
        args = DiffusionArgs(
            checkpoint_path=FLUX2_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )

        pipeline = PipelineLoader(args).load()

        assert pipeline is not None
        assert hasattr(pipeline, "transformer")
        assert pipeline.transformer is not None

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM"])
    def test_load_flux1_with_attention_backend(self, flux1_checkpoint_exists, backend: str):
        """Test loading FLUX.1 with different attention backends."""
        args = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend=backend),
            pipeline=PIPELINE_NO_WARMUP,
        )

        pipeline = PipelineLoader(args).load()

        assert pipeline.model_config.attention.backend == backend

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Quantization Tests
# =============================================================================


class TestFluxQuantization:
    """Test FLUX quantization loading and FP8 weight verification."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_load_flux1_with_quantization(self, flux1_checkpoint_exists, quant_algo: str):
        """Test loading FLUX.1 with FP8 quantization and verify FP8 weights."""
        args = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline=PIPELINE_NO_WARMUP,
        )

        pipeline = PipelineLoader(args).load()

        assert pipeline.model_config.quant_config.quant_algo is not None

        # Count quantized Linear layers and verify FP8 weights
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_load_flux2_with_quantization(self, flux2_checkpoint_exists, quant_algo: str):
        """Test loading FLUX.2 with FP8 quantization and verify FP8 weights."""
        args = DiffusionArgs(
            checkpoint_path=FLUX2_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline=PIPELINE_NO_WARMUP,
        )

        pipeline = PipelineLoader(args).load()

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


# =============================================================================
# FP8 Numerical Correctness Tests
# =============================================================================


class TestFluxFP8NumericalCorrectness:
    """Test FP8 vs BF16 numerical accuracy at single-layer and full-transformer levels."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_single_layer(self, flux1_checkpoint_exists, quant_algo: str):
        """Test FP8 vs BF16 numerical accuracy on a single Linear layer.

        Pattern (matching Wan test_fp8_vs_bf16_numerical_correctness):
        1. Use F.linear() with BF16 weights as ground truth reference
        2. Verify BF16 layer matches F.linear exactly
        3. Compare FP8 layer output against reference
        4. Check max_diff, cosine_similarity, mse_loss
        """
        # Load BF16 pipeline (reference)
        print(f"\n[Compare {quant_algo}] Loading BF16 pipeline...")
        args_bf16 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        # Load FP8 pipeline
        print(f"[Compare {quant_algo}] Loading {quant_algo} pipeline...")
        args_fp8 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()

        # Get matching Linear layers from both pipelines
        linear_bf16, layer_name = _find_first_quantizable_linear(pipeline_bf16.transformer)
        linear_fp8, _ = _find_first_quantizable_linear(pipeline_fp8.transformer)

        assert linear_bf16 is not None, "Could not find a Linear layer in BF16 transformer"
        assert linear_fp8 is not None, "Could not find a Linear layer in FP8 transformer"

        # Get BF16 weights for F.linear reference
        weight_bf16 = linear_bf16.weight.data.clone()
        bias_bf16 = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None

        # Create test input (2D for FP8 kernel compatibility)
        torch.manual_seed(42)
        hidden_size = linear_bf16.in_features
        batch_seq_len = 1024

        input_tensor = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")
        print(f"[Compare] Layer: {layer_name}, Input shape: {input_tensor.shape}")

        # Compute reference output: F.linear (ground truth)
        with torch.no_grad():
            expected = F.linear(input_tensor, weight_bf16, bias_bf16)

        # Compute BF16 layer output
        with torch.no_grad():
            result_bf16 = linear_bf16(input_tensor)

        # Compute FP8 output
        with torch.no_grad():
            result_fp8 = linear_fp8(input_tensor)

        # Verify BF16 layer matches F.linear reference
        assert torch.allclose(result_bf16, expected, rtol=1e-5, atol=1e-6), (
            "BF16 layer should match F.linear reference exactly"
        )

        # Compare FP8 vs reference
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_full_transformer_e2e(self, flux1_checkpoint_exists, quant_algo: str):
        """End-to-end test: Compare full FLUX.1 transformer FP8 vs BF16 output.

        Runs the entire transformer (19 dual + 38 single blocks) and compares outputs.
        Errors accumulate across layers, so uses relaxed tolerances vs single-layer test.
        """
        # Load BF16 transformer (reference)
        print("\n[E2E] Loading BF16 transformer...")
        args_bf16 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()
        transformer_bf16 = pipeline_bf16.transformer

        # Load FP8 transformer
        print(f"[E2E] Loading {quant_algo} transformer...")
        args_fp8 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()
        transformer_fp8 = pipeline_fp8.transformer

        # Create test inputs
        inputs = _get_flux_transformer_inputs(transformer_bf16)

        # Run both transformers
        print("[E2E] Running BF16 transformer forward...")
        with torch.no_grad():
            output_bf16 = transformer_bf16(**inputs)

        print(f"[E2E] Running {quant_algo} transformer forward...")
        inputs_fp8 = {k: v.clone() for k, v in inputs.items()}
        with torch.no_grad():
            output_fp8 = transformer_fp8(**inputs_fp8)

        # Extract outputs
        output_bf16 = _extract_transformer_output(output_bf16)
        output_fp8 = _extract_transformer_output(output_fp8)

        assert output_bf16.shape == output_fp8.shape, (
            f"Output shape mismatch: BF16={output_bf16.shape}, FP8={output_fp8.shape}"
        )

        # Check for NaN/Inf
        assert not torch.isnan(output_bf16).any(), "BF16 output contains NaN"
        assert not torch.isinf(output_bf16).any(), "BF16 output contains Inf"
        assert not torch.isnan(output_fp8).any(), f"{quant_algo} output contains NaN"
        assert not torch.isinf(output_fp8).any(), f"{quant_algo} output contains Inf"

        # Compare numerical accuracy
        output_bf16_float = output_bf16.float()
        output_fp8_float = output_fp8.float()

        max_diff = torch.max(torch.abs(output_fp8_float - output_bf16_float)).item()
        mean_diff = torch.mean(torch.abs(output_fp8_float - output_bf16_float)).item()

        cos_sim = F.cosine_similarity(
            output_fp8_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()

        mse = F.mse_loss(output_fp8_float, output_bf16_float).item()
        rel_error = mean_diff / (output_bf16_float.abs().mean().item() + 1e-8)

        num_dual = len(transformer_bf16.transformer_blocks)
        num_single = len(transformer_bf16.single_transformer_blocks)

        print(f"\n{'=' * 60}")
        print(f"END-TO-END TRANSFORMER COMPARISON ({quant_algo} vs BF16)")
        print(f"{'=' * 60}")
        print(f"Number of layers: {num_dual} dual + {num_single} single")
        print(f"Output shape: {output_bf16.shape}")
        print("")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        print(f"Cosine similarity: {cos_sim:.6f}")
        print(f"MSE loss: {mse:.6f}")
        print("")
        print(f"BF16 output range: [{output_bf16_float.min():.4f}, {output_bf16_float.max():.4f}]")
        print(
            f"{quant_algo} output range: [{output_fp8_float.min():.4f}, {output_fp8_float.max():.4f}]"
        )
        print(f"{'=' * 60}")

        assert cos_sim > 0.95, (
            f"Cosine similarity too low for full transformer: {cos_sim:.6f} (expected >0.95)"
        )
        assert rel_error < 0.15, f"Relative error too high: {rel_error:.6f} (expected <0.15)"

        print(f"\n[PASS] {quant_algo} full transformer output matches BF16 within tolerance!")
        print(f"  Cosine similarity: {cos_sim:.4f} (>0.95)")
        print(f"  Relative error: {rel_error:.4f} (<0.15)")

        del pipeline_bf16, pipeline_fp8, transformer_bf16, transformer_fp8
        torch.cuda.empty_cache()


# =============================================================================
# FP8 Memory Comparison Tests
# =============================================================================


class TestFluxFP8Memory:
    """Test FP8 memory reduction for FLUX models."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_memory_comparison(self, flux1_checkpoint_exists):
        """Test FP8 uses ~2x less memory than BF16 (matching Wan test)."""

        def get_module_memory_gb(module):
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3

        # Load BF16
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_bf16 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        bf16_model_mem = get_module_memory_gb(pipeline_bf16.transformer)
        bf16_peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n[BF16] Transformer memory: {bf16_model_mem:.2f} GB")
        print(f"[BF16] Peak memory: {bf16_peak_mem:.2f} GB")

        del pipeline_bf16
        torch.cuda.empty_cache()

        # Load FP8
        torch.cuda.reset_peak_memory_stats()

        args_fp8 = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": "FP8", "dynamic": True},
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()

        fp8_model_mem = get_module_memory_gb(pipeline_fp8.transformer)
        fp8_peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n[FP8] Transformer memory: {fp8_model_mem:.2f} GB")
        print(f"[FP8] Peak memory: {fp8_peak_mem:.2f} GB")

        # Verify memory savings
        model_mem_ratio = bf16_model_mem / fp8_model_mem
        peak_mem_ratio = bf16_peak_mem / fp8_peak_mem

        print(f"\n[Comparison] Model memory ratio (BF16/FP8): {model_mem_ratio:.2f}x")
        print(f"[Comparison] Peak memory ratio (BF16/FP8): {peak_mem_ratio:.2f}x")

        # FP8 should use ~2x less memory
        assert model_mem_ratio > 1.8, f"FP8 should use ~2x less memory, got {model_mem_ratio:.2f}x"

        del pipeline_fp8
        torch.cuda.empty_cache()


# =============================================================================
# Attention Backend Comparison Tests
# =============================================================================


class TestFluxAttentionBackend:
    """Test VANILLA vs TRTLLM attention backend numerical correctness."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_backend_comparison(self, flux1_checkpoint_exists):
        """Test that VANILLA and TRTLLM backends produce similar outputs.

        FLUX uses joint self-attention (same seq_len for Q and KV), so both
        VANILLA and TRTLLM backends should work. This test verifies numerical
        consistency between them.
        """
        # Run VANILLA first, save output, then free before loading TRTLLM
        # (two full transformers don't fit in GPU memory simultaneously)
        print("\n[Attention Backend Test] Loading baseline transformer (VANILLA)...")
        args_baseline = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend="VANILLA"),
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()
        transformer_baseline = pipeline_baseline.transformer

        inputs = _get_flux_transformer_inputs(transformer_baseline)

        print("[Attention Backend Test] Running VANILLA transformer forward...")
        with torch.no_grad():
            output_baseline = transformer_baseline(**inputs)
        output_baseline = _extract_transformer_output(output_baseline).cpu()

        del pipeline_baseline, transformer_baseline
        gc.collect()
        torch.cuda.empty_cache()

        # Load and run TRTLLM backend
        print("[Attention Backend Test] Loading TRTLLM transformer...")
        args_trtllm = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend="TRTLLM"),
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_trtllm = PipelineLoader(args_trtllm).load()
        transformer_trtllm = pipeline_trtllm.transformer

        print("[Attention Backend Test] Running TRTLLM transformer forward...")
        with torch.no_grad():
            output_trtllm = transformer_trtllm(**inputs)
        output_trtllm = _extract_transformer_output(output_trtllm).cpu()

        assert output_baseline.shape == output_trtllm.shape, (
            f"Output shape mismatch: VANILLA={output_baseline.shape}, TRTLLM={output_trtllm.shape}"
        )

        # Check for NaN/Inf
        for name, output in [("VANILLA", output_baseline), ("TRTLLM", output_trtllm)]:
            assert not torch.isnan(output).any(), f"{name} output contains NaN"
            assert not torch.isinf(output).any(), f"{name} output contains Inf"

        # Compare
        output_baseline_float = output_baseline.float()
        output_trtllm_float = output_trtllm.float()

        max_diff = torch.max(torch.abs(output_trtllm_float - output_baseline_float)).item()
        mean_diff = torch.mean(torch.abs(output_trtllm_float - output_baseline_float)).item()
        cos_sim = F.cosine_similarity(
            output_trtllm_float.flatten(), output_baseline_float.flatten(), dim=0
        ).item()
        mse = F.mse_loss(output_trtllm_float, output_baseline_float).item()

        print(f"\n{'=' * 60}")
        print("TRTLLM vs VANILLA Comparison")
        print(f"{'=' * 60}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Cosine similarity: {cos_sim:.6f}")
        print(f"MSE loss: {mse:.6f}")
        print(f"{'=' * 60}")

        assert cos_sim > 0.99, (
            f"TRTLLM should produce similar results to VANILLA: cos_sim={cos_sim:.6f}"
        )

        print(f"\n[PASS] TRTLLM backend matches VANILLA: cos_sim={cos_sim:.6f} (>0.99)")

        del pipeline_trtllm, transformer_trtllm
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# End-to-End Pipeline Tests (vs HuggingFace Reference)
# =============================================================================


class TestFluxE2E:
    """End-to-end pipeline tests: full generation compared to HuggingFace reference."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux1_e2e_vs_hf(self, flux1_checkpoint_exists):
        """Full FLUX.1 pipeline (all components) generates image matching HF reference."""
        from diffusers import FluxPipeline as HFFluxPipeline

        # 1. Generate HF reference image
        hf_pipe = HFFluxPipeline.from_pretrained(
            FLUX1_CHECKPOINT_PATH, torch_dtype=torch.bfloat16
        ).to("cuda")
        hf_result = hf_pipe(
            prompt="a tiny astronaut hatching from an egg on the moon",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42),
        )
        hf_image = np.array(hf_result.images[0])  # PIL -> (H, W, 3) uint8
        del hf_pipe
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Load TRT-LLM pipeline (full, no skip_components)
        args = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            pipeline=PipelineConfig(),
        )
        pipeline = PipelineLoader(args).load()

        # 3. Generate native image
        result = pipeline.forward(
            prompt="a tiny astronaut hatching from an egg on the moon",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42,
        )
        native_image = result.image.cpu().numpy()  # (H, W, 3) uint8

        # 4. Compute PSNR
        mse = ((hf_image.astype(float) - native_image.astype(float)) ** 2).mean()
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        print(f"\n[E2E FLUX.1] PSNR: {psnr:.2f} dB")

        assert psnr > 20.0, f"PSNR too low: {psnr:.2f} dB (expected >20 dB)"

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux2_e2e_vs_hf(self, flux2_checkpoint_exists):
        """Full FLUX.2 pipeline (all components) generates image matching HF reference."""
        from diffusers import Flux2Pipeline as HFFlux2Pipeline

        # 1. Generate HF reference image
        hf_pipe = HFFlux2Pipeline.from_pretrained(
            FLUX2_CHECKPOINT_PATH, torch_dtype=torch.bfloat16
        ).to("cuda")
        hf_result = hf_pipe(
            prompt="a tiny astronaut hatching from an egg on the moon",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42),
        )
        hf_image = np.array(hf_result.images[0])  # PIL -> (H, W, 3) uint8
        del hf_pipe
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Load TRT-LLM pipeline (full, no skip_components)
        args = DiffusionArgs(
            checkpoint_path=FLUX2_CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            pipeline=PipelineConfig(),
        )
        pipeline = PipelineLoader(args).load()

        # 3. Generate native image
        result = pipeline.forward(
            prompt="a tiny astronaut hatching from an egg on the moon",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42,
        )
        native_image = result.image.cpu().numpy()  # (H, W, 3) uint8

        # 4. Compute PSNR
        mse = ((hf_image.astype(float) - native_image.astype(float)) ** 2).mean()
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        print(f"\n[E2E FLUX.2] PSNR: {psnr:.2f} dB")

        # from HF is expected (~15 dB) compared to FLUX.1 (~32 dB).
        assert psnr > 20.0, f"PSNR too low: {psnr:.2f} dB (expected >20 dB)"

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Multi-GPU Parallelism Tests (Ulysses sequence parallelism)
# =============================================================================


def _setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed process group for multi-GPU tests."""
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_ulysses_worker(rank, world_size, checkpoint_path, inputs_cpu, return_dict):
    """Worker function for Ulysses multi-GPU test.

    Must be module-level for multiprocessing.spawn() pickling.
    """
    try:
        _setup_distributed(rank, world_size)

        from tensorrt_llm._torch.visual_gen.config import DiffusionArgs, ParallelConfig
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        # Load pipeline with Ulysses parallelism
        args = DiffusionArgs(
            checkpoint_path=checkpoint_path,
            device=f"cuda:{rank}",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            parallel=ParallelConfig(dit_ulysses_size=world_size),
            pipeline={"warmup_steps": 0, "enable_torch_compile": False},
        )
        pipeline = PipelineLoader(args).load()

        # Load inputs on this GPU
        inputs = {k: v.to(f"cuda:{rank}") for k, v in inputs_cpu.items()}

        # Run transformer forward
        with torch.no_grad():
            output = pipeline.transformer(**inputs)

        sample = _extract_transformer_output(output)

        # Only rank 0 stores the result
        if rank == 0:
            return_dict["output"] = sample.cpu()
            return_dict["shape"] = list(sample.shape)

        del pipeline
        torch.cuda.empty_cache()

    except Exception as e:
        return_dict[f"error_{rank}"] = str(e)
        raise
    finally:
        _cleanup_distributed()


class TestFluxParallelism:
    """Ulysses sequence parallelism tests for FLUX (requires 2+ GPUs)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() < 2,
        reason="Ulysses parallel test requires at least 2 GPUs",
    )
    def test_ulysses_2gpu_correctness(self, flux1_checkpoint_exists):
        """Test Ulysses (ulysses_size=2) correctness against single-GPU baseline.

        Similar pattern to WAN's test_cfg_2gpu_correctness:
        1. Load single-GPU reference pipeline, run forward
        2. Spawn 2-GPU Ulysses workers, run same forward
        3. Compare outputs (PSNR > 30 dB)
        """

        print("\n" + "=" * 80)
        print("ULYSSES SEQUENCE PARALLELISM (ulysses_size=2) CORRECTNESS TEST")
        print("=" * 80)

        # Load single-GPU reference
        print("\n[1/3] Loading single-GPU reference (ulysses_size=1) on GPU 0...")
        args_baseline = DiffusionArgs(
            checkpoint_path=FLUX1_CHECKPOINT_PATH,
            device="cuda:0",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            pipeline=PIPELINE_NO_WARMUP,
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()

        # Create test inputs (seq_len must be divisible by 2 for Ulysses)
        print("\n[2/3] Creating test inputs...")
        inputs = _get_flux_transformer_inputs(
            pipeline_baseline.transformer, device="cuda:0", dtype=torch.bfloat16
        )

        # Run single-GPU reference
        with torch.no_grad():
            ref_output = pipeline_baseline.transformer(**inputs)
        ref_sample = _extract_transformer_output(ref_output)
        print(f"  Reference output shape: {ref_sample.shape}")
        print(f"  Reference range: [{ref_sample.min():.4f}, {ref_sample.max():.4f}]")

        # Store inputs on CPU for workers
        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

        # Cleanup baseline
        del pipeline_baseline
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

        # Run Ulysses parallel (2 GPUs)
        print("\n[3/3] Running Ulysses (ulysses_size=2) across 2 GPUs...")
        manager = mp.Manager()
        return_dict = manager.dict()

        mp.spawn(
            _run_ulysses_worker,
            args=(2, FLUX1_CHECKPOINT_PATH, inputs_cpu, return_dict),
            nprocs=2,
            join=True,
        )

        # Check for errors
        for i in range(2):
            assert f"error_{i}" not in return_dict, (
                f"Rank {i} failed: {return_dict.get(f'error_{i}')}"
            )

        ulysses_sample = return_dict["output"].to("cuda:0")
        print(f"  Ulysses output shape: {ulysses_sample.shape}")
        print(f"  Ulysses range: [{ulysses_sample.min():.4f}, {ulysses_sample.max():.4f}]")

        # Compare outputs
        assert ref_sample.shape == ulysses_sample.shape, (
            f"Shape mismatch: ref={ref_sample.shape}, ulysses={ulysses_sample.shape}"
        )

        mse = ((ref_sample.float() - ulysses_sample.float()) ** 2).mean().item()
        ref_range = (ref_sample.max() - ref_sample.min()).float().item()
        psnr = 10 * np.log10(ref_range**2 / mse) if mse > 0 else float("inf")

        print(f"\n  MSE: {mse:.6e}")
        print(f"  PSNR: {psnr:.2f} dB")

        # Ulysses should be nearly identical (only BF16 rounding from all-to-all)
        assert psnr > 30.0, f"PSNR too low: {psnr:.2f} dB (expected >30 dB)"

        del ref_sample, ulysses_sample
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
