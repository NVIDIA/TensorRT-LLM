# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for Wan 2.1 T2V pipeline against HuggingFace reference.

Verifies >= 0.99 cosine similarity on decoded video frames (T, H, W, C) after
full denoising + VAE decode.

Models:
  - Wan2.1-T2V-1.3B-Diffusers   (480x832,  33 frames)
  - Wan2.1-T2V-14B-Diffusers    (720x1280, 33 frames)

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_pipeline.py -v -s -k 1_3b
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_pipeline.py -v -s -k 14b

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/1.3b \\
    DIFFUSION_MODEL_PATH_WAN21_14B=/path/to/14b \\
        pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_pipeline.py -v -s
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if set in env, assert when it's set but not a valid path."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return str(root)


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or os.path.join(_llm_models_root(), default_name)


WAN21_1_3B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_1_3B", "Wan2.1-T2V-1.3B-Diffusers")
WAN21_14B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_14B", "Wan2.1-T2V-14B-Diffusers")

# ============================================================================
# Test constants
# ============================================================================

PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""
NUM_STEPS = 4
SEED = 42
COS_SIM_THRESHOLD = 0.99


# ============================================================================
# Helpers
# ============================================================================


def _load_trtllm_pipeline(checkpoint_path: str):
    """Load TRTLLM WanPipeline without torch.compile or warmup."""
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
    )
    return PipelineLoader(args).load(skip_warmup=True)


def _load_hf_pipeline(checkpoint_path: str):
    """Load HuggingFace diffusers pipeline (auto-detects class from model_index.json)."""
    hf_pipe = DiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    hf_pipe = hf_pipe.to("cuda")
    hf_pipe.set_progress_bar_config(disable=True)
    return hf_pipe


def _capture_trtllm_video(
    pipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Run full TRTLLM pipeline including VAE decode; return (T, H, W, C) float in [0, 1]."""
    with torch.no_grad():
        result = pipeline.forward(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    video = result.video  # (T, H, W, C) uint8
    return video.float() / 255.0


def _capture_hf_video(
    hf_pipe,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Run HF pipeline with output_type='np'; return (T, H, W, C) float in [0, 1]."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = hf_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
    )
    frames = output.frames  # (1, T, H, W, C) numpy float32 in [0, 1]
    if isinstance(frames, np.ndarray):
        return torch.from_numpy(frames[0]).float()
    return frames[0].float()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened to 1D, cast to float32 on CPU)."""
    a_flat = a.float().cpu().reshape(-1)
    b_flat = b.float().cpu().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()


def _assert_pipeline_matches_hf(
    checkpoint_path: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    model_label: str,
) -> None:
    """Run TRTLLM and HF pipelines sequentially, compare decoded video output."""
    # --- TRTLLM ---
    trtllm_pipe = _load_trtllm_pipeline(checkpoint_path)
    trtllm_video = _capture_trtllm_video(
        trtllm_pipe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
    )
    del trtllm_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- HF reference ---
    hf_pipe = _load_hf_pipeline(checkpoint_path)
    hf_video = _capture_hf_video(
        hf_pipe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
    )
    del hf_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compare ---
    assert trtllm_video.numel() == hf_video.numel(), (
        f"{model_label}: element count mismatch — "
        f"TRTLLM {trtllm_video.shape} ({trtllm_video.numel()}) vs "
        f"HF {hf_video.shape} ({hf_video.numel()})"
    )

    cos_sim = _cosine_similarity(trtllm_video, hf_video)
    print(f"\n  {model_label} cosine similarity: {cos_sim:.6f}")
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"{model_label}: cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"TRTLLM pipeline output diverges from the HuggingFace reference. "
        f"Video shapes — TRTLLM: {trtllm_video.shape}, HF: {hf_video.shape}."
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan21_1_3B_PipelineCorrectness:
    """Wan2.1-T2V-1.3B correctness vs HuggingFace reference (480x832, 33 frames)."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN21_1_3B_PATH,
            height=480,
            width=832,
            num_frames=9,
            guidance_scale=5.0,
            model_label="Wan2.1-T2V-1.3B",
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan21_14B_PipelineCorrectness:
    """Wan2.1-T2V-14B correctness vs HuggingFace reference (720x1280, 33 frames)."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN21_14B_PATH,
            height=720,
            width=1280,
            num_frames=9,
            guidance_scale=5.0,
            model_label="Wan2.1-T2V-14B",
        )


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestWanBatchGeneration:
    """Batch generation tests for WAN T2V pipeline.

    Tests that passing a list of prompts produces batched output
    and matches sequential generation with the same seeds.
    """

    @pytest.fixture(scope="class")
    def wan21_full_pipeline(self):
        """Load full Wan 2.1 pipeline (all components) for batch tests."""
        if not WAN21_1_3B_PATH or not os.path.exists(WAN21_1_3B_PATH):
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH_WAN21_1_3B.")

        args = VisualGenArgs(
            checkpoint_path=WAN21_1_3B_PATH,
            device="cuda",
            dtype="bfloat16",
            torch_compile=TorchCompileConfig(enable_torch_compile=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        yield pipeline
        del pipeline
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_prompt_backward_compat(self, wan21_full_pipeline):
        """Single prompt returns (B, T, H, W, C) for backward compatibility."""
        result = wan21_full_pipeline.forward(
            prompt="a cat walking",
            height=480,
            width=832,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 1 and H == 480 and W == 832 and C == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_prompt_shape(self, wan21_full_pipeline):
        """List of prompts returns (B, T, H, W, C)."""
        prompts = ["a sunset over mountains", "a cat on a roof"]
        result = wan21_full_pipeline.forward(
            prompt=prompts,
            height=480,
            width=832,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 2 and H == 480 and W == 832 and C == 3


# =============================================================================
# Combined Optimization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan21T2VCombinedOptimizations:
    """FP8 + TeaCache + TRTLLM attention combined on Wan 2.1 T2V (1.3B, 480x832)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_teacache_trtllm(self):
        if not os.path.exists(WAN21_1_3B_PATH):
            pytest.skip(f"Checkpoint not found: {WAN21_1_3B_PATH}")
        args = VisualGenArgs(
            checkpoint_path=WAN21_1_3B_PATH,
            device="cuda",
            dtype="bfloat16",
            torch_compile=TorchCompileConfig(enable_torch_compile=False),
            quant_config={"quant_algo": "FP8", "dynamic": True},
            attention=AttentionConfig(backend="TRTLLM"),
            cache=TeaCacheConfig(teacache_thresh=0.2),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt="a cat sitting on a windowsill",
                    negative_prompt="",
                    height=480,
                    width=832,
                    num_frames=9,
                    num_inference_steps=10,
                    guidance_scale=5.0,
                    seed=42,
                )
            assert result.video.dim() == 5
            B, _T, H, W, C = result.video.shape
            assert B == 1 and H == 480 and W == 832 and C == 3

            assert pipeline.cache_accelerator is not None
            assert pipeline.cache_accelerator.is_enabled()
            stats = pipeline.cache_accelerator.get_stats()
            assert stats["cached_steps"] > 0, f"No TeaCache hits with FP8+TRTLLM. Stats: {stats}"
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()


# =============================================================================
# Quantization / dtype feature tests (transformer only, module-scoped fixtures)
# =============================================================================

_SKIP_AUX = ["text_encoder", "vae", "tokenizer", "scheduler"]


def _make_wan21_t2v(quant_config=None):
    if not os.path.exists(WAN21_1_3B_PATH):
        pytest.skip(f"Checkpoint not found: {WAN21_1_3B_PATH}")
    kwargs = dict(
        checkpoint_path=WAN21_1_3B_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=_SKIP_AUX,
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
    )
    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    return PipelineLoader(VisualGenArgs(**kwargs)).load(skip_warmup=True)


@pytest.fixture(scope="module")
def wan21_t2v_bf16():
    pipeline = _make_wan21_t2v()
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_t2v_fp8():
    pipeline = _make_wan21_t2v({"quant_algo": "FP8", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_t2v_fp8_block():
    pipeline = _make_wan21_t2v({"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_t2v_nvfp4():
    pipeline = _make_wan21_t2v({"quant_algo": "NVFP4", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def _transformer_inputs(device: str = "cuda"):
    torch.manual_seed(42)
    return (
        torch.randn(1, 16, 1, 64, 64, dtype=torch.bfloat16, device=device),
        torch.tensor([500], dtype=torch.long, device=device),
        torch.randn(1, 128, 4096, dtype=torch.bfloat16, device=device),
    )


def _is_fp32_layernorm_param(name: str) -> bool:
    if not name.endswith((".weight", ".bias")):
        return False
    if ".norm" in name and "blocks." in name:
        return any(p in name.split(".") for p in ("norm1", "norm2", "norm3"))
    if name in ("norm_out.weight", "norm_out.bias"):
        return True
    if name.startswith("condition_embedder.") and ".norm" in name:
        return True
    return False


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan21T2VPipelineFeatures:
    """Quantization loading, dtype layout, numerical accuracy, and memory for Wan 2.1 T2V."""

    def test_parameter_dtypes(self, wan21_t2v_bf16):
        """BF16 pipeline: CUDA tensors, FP32 LayerNorms, BF16 everything else."""
        bf16_count = 0
        for name, param in wan21_t2v_bf16.transformer.named_parameters():
            assert param.device.type == "cuda", f"{name} not on CUDA"
            if _is_fp32_layernorm_param(name):
                assert param.dtype == torch.float32, f"{name}: expected float32, got {param.dtype}"
            elif "scale" not in name.lower():
                assert param.dtype == torch.bfloat16, (
                    f"{name}: expected bfloat16, got {param.dtype}"
                )
                bf16_count += 1
        assert bf16_count > 0, "No BF16 parameters found"

    def test_fp8_weights_loaded(self, wan21_t2v_fp8):
        """FP8 transformer blocks have float8_e4m3fn weights and weight_scale."""
        try:
            if not hasattr(torch.ops, "tensorrt_llm"):
                pytest.skip("tensorrt_llm torch ops not available")
            _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
            _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"FP8 quantization ops not available: {e}")
        for name, module in wan21_t2v_fp8.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"{name}: expected float8_e4m3fn, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                return
        pytest.fail("No FP8 Linear found in transformer blocks")

    def test_fp8_block_scales_weights_loaded(self, wan21_t2v_fp8_block):
        """FP8_BLOCK_SCALES transformer blocks have float8_e4m3fn weights and weight_scale."""
        try:
            if not hasattr(torch.ops, "tensorrt_llm"):
                pytest.skip("tensorrt_llm torch ops not available")
            _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
            _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"FP8 quantization ops not available: {e}")
        for name, module in wan21_t2v_fp8_block.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"{name}: expected float8_e4m3fn, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                return
        pytest.fail("No FP8_BLOCK_SCALES Linear found in transformer blocks")

    def test_nvfp4_weights_loaded(self, wan21_t2v_nvfp4):
        """NVFP4 transformer blocks have packed FP4 weights with two-level scale."""
        if torch.cuda.get_device_capability(0) < (10, 0):
            pytest.skip("NVFP4 requires SM>=10.0 (Blackwell+)")
        try:
            _ = torch.ops.trtllm.fp4_quantize
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"fp4_quantize op not available: {e}")
        from tensorrt_llm.quantization.utils import fp4_utils

        for name, module in wan21_t2v_nvfp4.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == fp4_utils.float4_e2m1x2, (
                    f"{name}: expected float4_e2m1x2, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                assert hasattr(module, "weight_scale_2"), f"{name}: missing weight_scale_2"
                return
        pytest.fail("No NVFP4 Linear found in transformer blocks")

    def test_fp8_single_layer_accuracy(self, wan21_t2v_bf16, wan21_t2v_fp8):
        """FP8 qkv_proj output matches BF16 F.linear reference (cos_sim > 0.99)."""
        linear_bf16 = wan21_t2v_bf16.transformer.blocks[0].attn1.qkv_proj
        linear_fp8 = wan21_t2v_fp8.transformer.blocks[0].attn1.qkv_proj

        weight = linear_bf16.weight.data.clone()
        bias = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None
        x = torch.randn(
            1024,
            linear_bf16.in_features,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(42),
        )

        with torch.no_grad():
            ref = torch.nn.functional.linear(x, weight, bias)
            fp8_out = linear_fp8(x)

        cos_sim = torch.nn.functional.cosine_similarity(
            fp8_out.flatten().float(), ref.flatten().float(), dim=0
        ).item()
        mse = torch.nn.functional.mse_loss(fp8_out.float(), ref.float()).item()
        print(f"\n  FP8 qkv_proj: cos_sim={cos_sim:.6f}, mse={mse:.6f}")
        assert cos_sim > 0.99, f"cos_sim too low: {cos_sim:.6f}"
        assert mse < 1.0, f"MSE too high: {mse:.6f}"

    def test_fp8_memory_savings(self, wan21_t2v_bf16, wan21_t2v_fp8):
        """FP8 transformer uses ~2x less parameter memory than BF16."""

        def _mem_gb(pipeline):
            return (
                sum(p.numel() * p.element_size() for p in pipeline.transformer.parameters())
                / 1024**3
            )

        bf16_gb = _mem_gb(wan21_t2v_bf16)
        fp8_gb = _mem_gb(wan21_t2v_fp8)
        ratio = bf16_gb / fp8_gb
        print(f"\n  BF16={bf16_gb:.3f} GB, FP8={fp8_gb:.3f} GB, ratio={ratio:.2f}x")
        assert ratio > 1.8, f"Expected ~2x savings, got {ratio:.2f}x"

    @pytest.mark.parametrize(
        "quant_name,pipe_fixture",
        [
            ("FP8", "wan21_t2v_fp8"),
            ("FP8_BLOCK_SCALES", "wan21_t2v_fp8_block"),
        ],
    )
    def test_fp8_e2e_accuracy(
        self, wan21_t2v_bf16, wan21_t2v_fp8, wan21_t2v_fp8_block, quant_name, pipe_fixture
    ):
        """FP8/FP8_BLOCK_SCALES full-transformer output close to BF16 (cos_sim > 0.99)."""
        quant_pipeline = wan21_t2v_fp8 if pipe_fixture == "wan21_t2v_fp8" else wan21_t2v_fp8_block
        hs, ts, enc = _transformer_inputs()

        with torch.no_grad():
            out_bf16 = wan21_t2v_bf16.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()
            out_quant = quant_pipeline.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()

        assert not torch.isnan(out_bf16).any(), "BF16 output contains NaN"
        assert not torch.isinf(out_bf16).any(), "BF16 output contains Inf"
        assert not torch.isnan(out_quant).any(), f"{quant_name} output contains NaN"
        assert not torch.isinf(out_quant).any(), f"{quant_name} output contains Inf"

        cos_sim = torch.nn.functional.cosine_similarity(
            out_quant.flatten(), out_bf16.flatten(), dim=0
        ).item()
        mse = torch.nn.functional.mse_loss(out_quant, out_bf16).item()
        print(
            f"\n  {quant_name} E2E ({len(wan21_t2v_bf16.transformer.blocks)} layers): "
            f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
        )
        assert cos_sim > 0.99, f"cos_sim too low: {cos_sim:.6f}"

    def test_nvfp4_e2e_accuracy(self, wan21_t2v_bf16, wan21_t2v_nvfp4):
        """NVFP4 full-transformer output close to BF16 (cos_sim > 0.95)."""
        if torch.cuda.get_device_capability(0) < (10, 0):
            pytest.skip("NVFP4 requires SM>=10.0 (Blackwell+)")
        try:
            _ = torch.ops.trtllm.fp4_quantize
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"fp4_quantize op not available: {e}")

        hs, ts, enc = _transformer_inputs()

        with torch.no_grad():
            out_bf16 = wan21_t2v_bf16.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()
            out_nvfp4 = wan21_t2v_nvfp4.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()

        assert not torch.isnan(out_nvfp4).any(), "NVFP4 output contains NaN"
        assert not torch.isinf(out_nvfp4).any(), "NVFP4 output contains Inf"

        cos_sim = torch.nn.functional.cosine_similarity(
            out_nvfp4.flatten(), out_bf16.flatten(), dim=0
        ).item()
        mse = torch.nn.functional.mse_loss(out_nvfp4, out_bf16).item()
        print(
            f"\n  NVFP4 E2E ({len(wan21_t2v_bf16.transformer.blocks)} layers): "
            f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
        )
        assert cos_sim > 0.95, f"NVFP4 cos_sim too low: {cos_sim:.6f}"
