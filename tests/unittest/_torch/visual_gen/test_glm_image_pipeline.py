# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.models.glm_image import GlmImagePipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, TorchCompileConfig, VisualGenArgs

# ============================================================================
# Test constants
# ============================================================================

# Canonical HuggingFace Hub ID for GlmImage.
GLM_IMAGE_HF_ID = "zai-org/GLM-Image"


def _llm_models_root() -> Optional[str]:
    """Return the LLM_MODELS_ROOT path if it resolves to an existing directory."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return str(root) if root.exists() else None


def _resolve_glm_checkpoint() -> str:
    """Resolve the GlmImage checkpoint reference.

    Resolution order:
    1. ``GLM_IMAGE_MODEL_PATH`` env var (explicit local path or HF Hub ID).
    2. ``<LLM_MODELS_ROOT>/GLM-Image`` staged checkpoint, when present locally.
    3. The canonical HF Hub ID — the pipeline loader downloads it on demand.
    """
    explicit = os.environ.get("GLM_IMAGE_MODEL_PATH")
    if explicit:
        return explicit
    root = _llm_models_root()
    if root is not None:
        staged = os.path.join(root, "GLM-Image")
        if os.path.isdir(staged):
            return staged
    return GLM_IMAGE_HF_ID


# Plain HF Hub ID by default; the pipeline loader downloads it on first use.
# Point GLM_IMAGE_MODEL_PATH / LLM_MODELS_ROOT at a local checkpoint to avoid
# re-downloading.
GLM_IMAGE_PATH = _resolve_glm_checkpoint()

# GlmImage takes a plain text prompt and derives the latent grid from the
# explicit height/width (each must be divisible by 32). 256 == 8 * 32.
PROMPT = "A dinosaur walking through the jungle"
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 30
SEED = 42
GUIDANCE_SCALE = 1.5
COS_SIM_THRESHOLD = 0.99


# ============================================================================
# Helpers
# ============================================================================


def _load_trtllm_pipeline(checkpoint_path: str, skip_components=None, **kwargs):
    """Load TRTLLM GlmImage pipeline without torch.compile or warmup."""

    args = VisualGenArgs(
        model=checkpoint_path, torch_compile_config=TorchCompileConfig(enable=False), **kwargs
    )
    return PipelineLoader(args).load(skip_warmup=True, skip_components=skip_components)


def _load_hf_pipeline(checkpoint_path: str):
    """Load HuggingFace diffusers pipeline (auto-detects class from model_index.json)."""
    hf_pipe = DiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    hf_pipe = hf_pipe.to("cuda")
    hf_pipe.set_progress_bar_config(disable=True)
    return hf_pipe


def _teardown_pipeline(pipe) -> None:
    """Release a pipeline reference and reclaim GPU memory."""
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def _capture_trtllm_image(
    pipeline,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Run full TRTLLM pipeline including VAE decode; return the raw pipeline output."""
    with torch.no_grad():
        return pipeline.forward(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )


def _capture_hf_image(
    hf_pipe,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Run HF pipeline with output_type='np'; return the raw pipeline output."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return hf_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
    )


def _to_image_tensor(images) -> torch.Tensor:
    """Convert (B, H, W, C) images to a float (H, W, C) tensor in [0, 1]."""

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images[0]
    if images.dtype == torch.uint8:
        return images.float() / 255.0
    return images.float()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened to 1D, cast to float32 on CPU)."""
    a_flat = a.float().cpu().reshape(-1)
    b_flat = b.float().cpu().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()


def _assert_pipeline_matches_hf(
    checkpoint_path: str,
    height: int,
    width: int,
    model_label: str,
) -> None:
    """Run TRTLLM and HF pipelines sequentially, compare decoded image output."""
    # --- TRTLLM ---
    trtllm_pipe = None
    try:
        trtllm_pipe = _load_trtllm_pipeline(checkpoint_path)
        trtllm_output = _capture_trtllm_image(
            trtllm_pipe,
            prompt=PROMPT,
            height=height,
            width=width,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
        )
        trtllm_image = _to_image_tensor(trtllm_output.image)
    finally:
        _teardown_pipeline(trtllm_pipe)

    # --- HF reference ---
    hf_pipe = None
    try:
        hf_pipe = _load_hf_pipeline(checkpoint_path)
        hf_output = _capture_hf_image(
            hf_pipe,
            prompt=PROMPT,
            height=height,
            width=width,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
        )
        hf_image = _to_image_tensor(hf_output.images)
    finally:
        _teardown_pipeline(hf_pipe)

    # --- Compare ---
    assert trtllm_image.numel() == hf_image.numel(), (
        f"{model_label}: element count mismatch — "
        f"TRTLLM {trtllm_image.shape} ({trtllm_image.numel()}) vs "
        f"HF {hf_image.shape} ({hf_image.numel()})"
    )

    cos_sim = _cosine_similarity(trtllm_image, hf_image)
    print(f"\n  {model_label} cosine similarity: {cos_sim:.6f}")
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"{model_label}: cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"TRTLLM pipeline output diverges from the HuggingFace reference. "
        f"Image shapes — TRTLLM: {trtllm_image.shape}, HF: {hf_image.shape}."
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
class TestGlmImagePipelineCorrectness:
    """GlmImage T2I correctness vs HuggingFace reference (256x256)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=GLM_IMAGE_PATH,
            height=HEIGHT,
            width=WIDTH,
            model_label="GlmImage-T2I",
        )


@pytest.mark.integration
class TestGlmImageGeneration:
    """Generation shape tests for the GlmImage pipeline.

    Validates that single-prompt generation returns a (B, H, W, C) image
    batch, matching the current pipeline contract.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_prompt(self):
        """Single prompt returns a (B, H, W, C) image batch."""

        pipe = None
        try:
            pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH)
            result = _capture_trtllm_image(
                pipe,
                prompt=PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
            )
            images = _to_image_tensor(result.image)
            # _to_image_tensor drops the batch dim, so the remaining tensor is (H, W, C).
            assert images.ndim == 3, f"Expected 3D (H,W,C), got {images.ndim}D"
            H, W, C = images.shape
            assert H == HEIGHT and W == WIDTH and C == 3
        finally:
            _teardown_pipeline(pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_prompts(self):
        """A list of prompts returns one image per prompt in a single batched forward."""

        prompts = [PROMPT, "A neon city skyline reflected in a river at night"]
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH)
            with torch.no_grad():
                result = pipe.forward(
                    prompt=prompts,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=torch.Generator(device="cuda").manual_seed(SEED),
                )
            image = result.image
            assert image.shape[0] == len(prompts), (
                f"Expected batch dim {len(prompts)}, got {image.shape[0]}"
            )
            assert tuple(image.shape[1:]) == (HEIGHT, WIDTH, 3), (
                f"Expected (H,W,C)=({HEIGHT},{WIDTH},3), got {tuple(image.shape[1:])}"
            )
        finally:
            _teardown_pipeline(pipe)


def test_image_conditioning_not_supported():
    """Passing a condition image raises NotImplementedError (I2I lands in a follow-up MR)."""
    # __new__ skips __init__; the guard fires before any self/model access, so no GPU needed.
    pipe = GlmImagePipeline.__new__(GlmImagePipeline)
    with pytest.raises(NotImplementedError, match="image-to-image"):
        pipe.forward(prompt=PROMPT, image=torch.zeros(1))


# =============================================================================
# Quantization Optimization Tests
# =============================================================================


@pytest.mark.integration
class TestGlmImageQuantizationOptimizations:
    """FP8 + TRTLLM attention on GlmImage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_trtllm_attention(self):
        pipe_args = {
            "quant_config": {"quant_algo": "FP8", "dynamic": True},
            "attention_config": AttentionConfig(backend="TRTLLM"),
        }

        pipe = None
        try:
            pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, **pipe_args)
            result = _capture_trtllm_image(
                pipe,
                prompt=PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
            )

            images = _to_image_tensor(result.image)
            assert images.ndim == 3, f"Expected 3D (H,W,C), got {images.ndim}D"
            H, W, C = images.shape
            assert H == HEIGHT and W == WIDTH and C == 3
        finally:
            _teardown_pipeline(pipe)


# =============================================================================
# Quantization / dtype feature tests (transformer only)
# =============================================================================

_SKIP_AUX = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
    PipelineComponent.VISION_LANGUAGE_ENCODER,
]


def _assert_fp8_blocks_quantized(pipe) -> None:
    """Assert at least one transformer-block Linear is FP8 (float8_e4m3fn) with a weight_scale."""
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Linear) and "transformer_blocks." in name:
            assert module.weight.dtype == torch.float8_e4m3fn, (
                f"{name}: expected float8_e4m3fn, got {module.weight.dtype}"
            )
            assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
            return
    pytest.fail("No FP8 Linear found in transformer blocks")


def _assert_nvfp4_blocks_quantized(pipe) -> None:
    """Assert at least one transformer-block Linear is NVFP4 (packed FP4) with a two-level scale."""
    from tensorrt_llm.quantization.utils import fp4_utils

    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Linear) and "transformer_blocks." in name:
            assert module.weight.dtype == fp4_utils.float4_e2m1x2, (
                f"{name}: expected float4_e2m1x2, got {module.weight.dtype}"
            )
            assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
            assert hasattr(module, "weight_scale_2"), f"{name}: missing weight_scale_2"
            return
    pytest.fail("No NVFP4 Linear found in transformer blocks")


def _skip_if_no_fp8_ops() -> None:
    try:
        if not hasattr(torch.ops, "tensorrt_llm"):
            pytest.skip("tensorrt_llm torch ops not available")
        _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
        _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
    except (AttributeError, RuntimeError) as e:
        pytest.skip(f"FP8 quantization ops not available: {e}")


def _skip_if_no_nvfp4_ops() -> None:
    if torch.cuda.get_device_capability(0) < (10, 0):
        pytest.skip("NVFP4 requires SM>=10.0 (Blackwell+)")
    try:
        _ = torch.ops.trtllm.fp4_quantize
    except (AttributeError, RuntimeError) as e:
        pytest.skip(f"fp4_quantize op not available: {e}")


def _transformer_inputs(transformer, device: str = "cuda", dtype=torch.bfloat16, seed: int = 42):
    """Build a minimal valid GlmImage transformer forward batch."""
    config = transformer.config
    in_channels = config.in_channels
    text_embed_dim = config.text_embed_dim
    codebook_size = config.prior_vq_quantizer_codebook_size

    g = torch.Generator(device=device).manual_seed(seed)
    batch_size = 1
    height = width = 32
    seq_len = 6
    return {
        "hidden_states": torch.randn(
            batch_size, in_channels, height, width, device=device, dtype=dtype, generator=g
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, seq_len, text_embed_dim, device=device, dtype=dtype, generator=g
        ),
        "prior_token_id": torch.randint(
            0, min(codebook_size, 64), size=(batch_size,), device=device, generator=g
        ),
        "prior_token_drop": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "timestep": torch.randint(0, 1000, size=(batch_size,), device=device, generator=g),
        "target_size": torch.tensor(
            [[height, width]] * batch_size, dtype=torch.float32, device=device
        ),
        "crop_coords": torch.tensor([[0, 0]] * batch_size, dtype=torch.float32, device=device),
    }


def _run_transformer(transformer, inputs: dict) -> torch.Tensor:
    """Run the transformer on cloned inputs; return the float32 sample output."""
    cloned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = transformer(**cloned, return_dict=False)[0]
    return out.float()


@pytest.mark.integration
class TestGlmImagePipelineFeatures:
    """Quantization loading, dtype layout, numerical accuracy, and memory for GlmImage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parameter_dtypes(self):
        """BF16 pipeline: every transformer param is on CUDA; all non-scale params are BF16."""
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, skip_components=_SKIP_AUX)
            bf16_count = 0
            for name, param in pipe.transformer.named_parameters():
                assert param.device.type == "cuda", f"{name} not on CUDA"
                if "scale" not in name.lower():
                    assert param.dtype == torch.bfloat16, (
                        f"{name}: expected bfloat16, got {param.dtype}"
                    )
                    bf16_count += 1
            assert bf16_count > 0, "No BF16 parameters found"
        finally:
            _teardown_pipeline(pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_weights_loaded(self):
        """FP8 transformer blocks have float8_e4m3fn weights and weight_scale."""
        _skip_if_no_fp8_ops()
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "FP8", "dynamic": True},
            )
            _assert_fp8_blocks_quantized(pipe)
        finally:
            _teardown_pipeline(pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_block_scales_weights_loaded(self):
        """FP8_BLOCK_SCALES transformer blocks have float8_e4m3fn weights and weight_scale."""
        _skip_if_no_fp8_ops()
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            )
            _assert_fp8_blocks_quantized(pipe)
        finally:
            _teardown_pipeline(pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nvfp4_weights_loaded(self):
        """NVFP4 transformer blocks have packed FP4 weights with a two-level scale."""
        _skip_if_no_nvfp4_ops()
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "NVFP4", "dynamic": True},
            )
            _assert_nvfp4_blocks_quantized(pipe)
        finally:
            _teardown_pipeline(pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_single_layer_accuracy(self):
        """FP8 qkv_proj output matches the BF16 F.linear reference (cos_sim > 0.99)."""
        _skip_if_no_fp8_ops()
        bf16_pipe = None
        fp8_pipe = None
        try:
            bf16_pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, skip_components=_SKIP_AUX)
            fp8_pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "FP8", "dynamic": True},
            )
            linear_bf16 = bf16_pipe.transformer.transformer_blocks[0].attn1.qkv_proj
            linear_fp8 = fp8_pipe.transformer.transformer_blocks[0].attn1.qkv_proj

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
                ref = F.linear(x, weight, bias)
                fp8_out = linear_fp8(x)

            cos_sim = F.cosine_similarity(
                fp8_out.flatten().float(), ref.flatten().float(), dim=0
            ).item()
            mse = F.mse_loss(fp8_out.float(), ref.float()).item()
            print(f"\n  FP8 qkv_proj: cos_sim={cos_sim:.6f}, mse={mse:.6f}")
            assert cos_sim > 0.99, f"cos_sim too low: {cos_sim:.6f}"
            assert mse < 1.0, f"MSE too high: {mse:.6f}"
        finally:
            if fp8_pipe is not None:
                _teardown_pipeline(fp8_pipe)
            if bf16_pipe is not None:
                _teardown_pipeline(bf16_pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_memory_savings(self):
        """FP8 transformer uses less parameter memory than BF16."""
        _skip_if_no_fp8_ops()

        def _mem_gb(pipe):
            return (
                sum(p.numel() * p.element_size() for p in pipe.transformer.parameters()) / 1024**3
            )

        bf16_pipe = None
        fp8_pipe = None
        try:
            bf16_pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, skip_components=_SKIP_AUX)
            fp8_pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "FP8", "dynamic": True},
            )
            bf16_gb = _mem_gb(bf16_pipe)
            fp8_gb = _mem_gb(fp8_pipe)
            ratio = bf16_gb / fp8_gb
            print(f"\n  BF16={bf16_gb:.3f} GB, FP8={fp8_gb:.3f} GB, ratio={ratio:.2f}x")
            assert ratio > 1.9, f"Expected FP8 memory reduction, got {ratio:.2f}x"
        finally:
            if fp8_pipe is not None:
                _teardown_pipeline(fp8_pipe)
            if bf16_pipe is not None:
                _teardown_pipeline(bf16_pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_e2e_accuracy(self, quant_algo):
        """FP8 / FP8_BLOCK_SCALES full-transformer output close to BF16 (cos_sim > 0.99)."""
        _skip_if_no_fp8_ops()
        bf16_pipe = None
        quant_pipe = None
        try:
            bf16_pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, skip_components=_SKIP_AUX)
            quant_pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": quant_algo, "dynamic": True},
            )
            inputs = _transformer_inputs(bf16_pipe.transformer)
            out_bf16 = _run_transformer(bf16_pipe.transformer, inputs)
            out_quant = _run_transformer(quant_pipe.transformer, inputs)

            assert not torch.isnan(out_bf16).any(), "BF16 output contains NaN"
            assert not torch.isinf(out_bf16).any(), "BF16 output contains Inf"
            assert not torch.isnan(out_quant).any(), f"{quant_algo} output contains NaN"
            assert not torch.isinf(out_quant).any(), f"{quant_algo} output contains Inf"

            cos_sim = F.cosine_similarity(out_quant.flatten(), out_bf16.flatten(), dim=0).item()
            mse = F.mse_loss(out_quant, out_bf16).item()
            print(
                f"\n  {quant_algo} E2E ({len(bf16_pipe.transformer.transformer_blocks)} layers): "
                f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
            )
            # Block-scales is a coarser (per-block) scheme than per-tensor FP8, so it
            # tolerates a slightly lower cosine-similarity floor.
            min_cos_sim = 0.975 if quant_algo == "FP8_BLOCK_SCALES" else 0.99
            assert cos_sim > min_cos_sim, f"{quant_algo} cos_sim too low: {cos_sim:.6f}"
        finally:
            if quant_pipe is not None:
                _teardown_pipeline(quant_pipe)
            if bf16_pipe is not None:
                _teardown_pipeline(bf16_pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nvfp4_e2e_accuracy(self):
        """NVFP4 full-transformer output close to BF16 (cos_sim > 0.90)."""
        _skip_if_no_nvfp4_ops()
        bf16_pipe = None
        nvfp4_pipe = None
        try:
            bf16_pipe = _load_trtllm_pipeline(GLM_IMAGE_PATH, skip_components=_SKIP_AUX)
            nvfp4_pipe = _load_trtllm_pipeline(
                GLM_IMAGE_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "NVFP4", "dynamic": True},
            )
            inputs = _transformer_inputs(bf16_pipe.transformer)
            out_bf16 = _run_transformer(bf16_pipe.transformer, inputs)
            out_nvfp4 = _run_transformer(nvfp4_pipe.transformer, inputs)

            assert not torch.isnan(out_nvfp4).any(), "NVFP4 output contains NaN"
            assert not torch.isinf(out_nvfp4).any(), "NVFP4 output contains Inf"

            cos_sim = F.cosine_similarity(out_nvfp4.flatten(), out_bf16.flatten(), dim=0).item()
            mse = F.mse_loss(out_nvfp4, out_bf16).item()
            print(
                f"\n  NVFP4 E2E ({len(bf16_pipe.transformer.transformer_blocks)} layers): "
                f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
            )
            # NOTE: AdaLayerNorm modulation linears hurt accuracy at nvfp4, consider adding omissions later
            assert cos_sim > 0.90, f"NVFP4 cos_sim too low: {cos_sim:.6f}"
        finally:
            if nvfp4_pipe is not None:
                _teardown_pipeline(nvfp4_pipe)
            if bf16_pipe is not None:
                _teardown_pipeline(bf16_pipe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
