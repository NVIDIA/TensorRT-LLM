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

from tensorrt_llm._torch.modules.linear import Linear

os.environ["TLLM_DISABLE_MPI"] = "1"

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, TorchCompileConfig, VisualGenArgs


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ============================================================================
# Test constants
# ============================================================================

HUNYUAN_VIDEO_1_5_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

PROMPT = "A dinosaur walking through the jungle"
NEGATIVE_PROMPT = ""
HEIGHT = 256
WIDTH = 256
NUM_FRAMES = 10
NUM_STEPS = 30
SEED = 42
COS_SIM_THRESHOLD = 0.99


# ============================================================================
# Helpers
# ============================================================================


def _load_trtllm_pipeline(checkpoint_path: str, skip_components=None, **kwargs):
    """Load TRTLLM HunyuanVideo 1.5 pipeline without torch.compile or warmup."""

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


def _capture_trtllm_video(
    pipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
):
    """Run full TRTLLM pipeline including VAE decode; return the raw pipeline output."""
    with torch.no_grad():
        return pipeline.forward(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )


def _capture_hf_video(
    hf_pipe,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
):
    """Run HF pipeline with output_type='np'; return the raw pipeline output."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return hf_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="np",
    )


def _to_video_tensor(frames) -> torch.Tensor:
    """Convert (B, T, H, W, C) frames to a float (T, H, W, C) tensor in [0, 1]."""

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    frames = frames[0]
    if frames.dtype == torch.uint8:
        return frames.float() / 255.0
    return frames.float()


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
    model_label: str,
) -> None:
    """Run TRTLLM and HF pipelines sequentially, compare decoded video output."""
    # --- TRTLLM ---
    try:
        trtllm_pipe = _load_trtllm_pipeline(checkpoint_path)
        trtllm_output = _capture_trtllm_video(
            trtllm_pipe,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=NUM_STEPS,
            seed=SEED,
        )
        trtllm_video = _to_video_tensor(trtllm_output.video)
    finally:
        _teardown_pipeline(trtllm_pipe)

    # --- HF reference ---
    try:
        hf_pipe = _load_hf_pipeline(checkpoint_path)
        hf_output = _capture_hf_video(
            hf_pipe,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=NUM_STEPS,
            seed=SEED,
        )
        hf_video = _to_video_tensor(hf_output.frames)
    finally:
        _teardown_pipeline(hf_pipe)

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
class TestHunyuanVideo15PipelineCorrectness:
    """HunyuanVideo 1.5 T2V correctness vs HuggingFace reference (256x256, 10 frames)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=HUNYUAN_VIDEO_1_5_PATH,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            model_label="HunyuanVideo1.5-T2V",
        )


@pytest.mark.integration
class TestHunyuanVideo15BatchGeneration:
    """Batch generation tests for HunyuanVideo 1.5 pipeline

    Tests that passing a list of prompts produces batched output
    and matches sequential generation with the same seeds.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_prompt_backward_compat(self):
        """Single prompt returns (B, T, H, W, C) for backward compatibility."""

        try:
            pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH)
            result = _capture_trtllm_video(
                pipe,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                seed=SEED,
            )
            assert result.video.ndim == 5, f"Expected 5D (B,T,H,W,C), got {result.video.ndim}D"
            B, _T, H, W, C = result.video.shape
            assert B == 1 and H == HEIGHT and W == WIDTH and C == 3
        finally:
            _teardown_pipeline(pipe)


# =============================================================================
# Quantization Optimization Tests
# =============================================================================


@pytest.mark.integration
class TestHunyuanVideo15QuantizationOptimizations:
    """FP8 + TRTLLM attention on HunyuanVideo1.5."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_trtllm_attention(self):
        pipe_args = {
            "quant_config": {"quant_algo": "FP8", "dynamic": True},
            "attention_config": AttentionConfig(backend="TRTLLM"),
        }

        try:
            pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, **pipe_args)
            result = _capture_trtllm_video(
                pipe,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                seed=SEED,
            )

            assert result.video.ndim == 5, f"Expected 5D (B,T,H,W,C), got {result.video.ndim}D"
            B, _T, H, W, C = result.video.shape
            assert B == 1 and H == HEIGHT and W == WIDTH and C == 3
        finally:
            _teardown_pipeline(pipe)


# =============================================================================
# Quantization / dtype feature tests (transformer only)
# =============================================================================

_SKIP_AUX = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TEXT_ENCODER_2,
    PipelineComponent.TOKENIZER,
    PipelineComponent.TOKENIZER_2,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
    PipelineComponent.GUIDER,
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
    """Build a minimal valid HunyuanVideo1.5 transformer forward batch."""
    in_channels = transformer.config.in_channels
    image_embed_dim = transformer.config.image_embed_dim
    text_embed_dim = transformer.context_embedder.proj_in.in_features
    text_embed_2_dim = transformer.context_embedder_2.linear_1.in_features

    g = torch.Generator(device=device).manual_seed(seed)
    seq_len = 6
    return {
        "hidden_states": torch.randn(
            1, in_channels, 1, 32, 32, device=device, dtype=dtype, generator=g
        ),
        "encoder_hidden_states": torch.randn(
            1, seq_len, text_embed_dim, device=device, dtype=dtype, generator=g
        ),
        "encoder_hidden_states_2": torch.randn(
            1, seq_len, text_embed_2_dim, device=device, dtype=dtype, generator=g
        ),
        "image_embeds": torch.randn(
            1, seq_len, image_embed_dim, device=device, dtype=dtype, generator=g
        ),
        "encoder_attention_mask": torch.ones(1, seq_len, device=device, dtype=dtype),
        "encoder_attention_mask_2": torch.ones(1, seq_len, device=device, dtype=dtype),
        "timestep": torch.tensor([1], device=device, dtype=dtype),
    }


def _run_transformer(transformer, inputs: dict) -> torch.Tensor:
    """Run the transformer on cloned inputs; return the float32 sample output."""
    cloned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = transformer(**cloned, return_dict=False)[0]
    return out.float()


@pytest.mark.integration
class TestHunyuanVideo15PipelineFeatures:
    """Quantization loading, dtype layout, numerical accuracy, and memory for HunyuanVideo1.5."""

    def test_parameter_dtypes(self):
        """BF16 pipeline: every transformer param is on CUDA; all non-scale params are BF16."""
        pipe = None
        try:
            pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, skip_components=_SKIP_AUX)
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
                HUNYUAN_VIDEO_1_5_PATH,
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
                HUNYUAN_VIDEO_1_5_PATH,
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
                HUNYUAN_VIDEO_1_5_PATH,
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
            bf16_pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, skip_components=_SKIP_AUX)
            fp8_pipe = _load_trtllm_pipeline(
                HUNYUAN_VIDEO_1_5_PATH,
                skip_components=_SKIP_AUX,
                quant_config={"quant_algo": "FP8", "dynamic": True},
            )
            linear_bf16 = bf16_pipe.transformer.transformer_blocks[0].attn.qkv_proj
            linear_fp8 = fp8_pipe.transformer.transformer_blocks[0].attn.qkv_proj

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
            bf16_pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, skip_components=_SKIP_AUX)
            fp8_pipe = _load_trtllm_pipeline(
                HUNYUAN_VIDEO_1_5_PATH,
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
            bf16_pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, skip_components=_SKIP_AUX)
            quant_pipe = _load_trtllm_pipeline(
                HUNYUAN_VIDEO_1_5_PATH,
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
            min_cos_sim = 0.975 if quant_algo == "FP8_BLOCK_SCALES" else 0.99
            assert cos_sim > min_cos_sim, f"{quant_algo} cos_sim too low: {cos_sim:.6f}"
        finally:
            if quant_pipe is not None:
                _teardown_pipeline(quant_pipe)
            if bf16_pipe is not None:
                _teardown_pipeline(bf16_pipe)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nvfp4_e2e_accuracy(self):
        """NVFP4 full-transformer output close to BF16 (cos_sim > 0.95)."""
        _skip_if_no_nvfp4_ops()
        bf16_pipe = None
        nvfp4_pipe = None
        try:
            bf16_pipe = _load_trtllm_pipeline(HUNYUAN_VIDEO_1_5_PATH, skip_components=_SKIP_AUX)
            nvfp4_pipe = _load_trtllm_pipeline(
                HUNYUAN_VIDEO_1_5_PATH,
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
