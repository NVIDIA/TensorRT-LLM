# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pipeline tests for LTX-2.3. Requires the LTX-2.3 checkpoint."""

import gc
import os

import pytest
import torch
import torch.nn.functional as F
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, VisualGenArgs

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

_MODELS_ROOT = str(llm_models_root(check=False))

# Load transformer only; other native components still come from the checkpoint.
SKIP_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
]

CHECKPOINT_PATH_BF16 = os.environ.get("LTX23_MODEL_PATH", os.path.join(_MODELS_ROOT, "LTX-2.3"))
GEMMA3_PATH = os.environ.get(
    "LTX23_TEXT_ENCODER_PATH", os.path.join(_MODELS_ROOT, "gemma", "gemma-3-12b-it")
)


def _positions(*sizes, device):
    grids = torch.meshgrid(*[torch.arange(s) for s in sizes], indexing="ij")
    idx = torch.stack([g.flatten() for g in grids]).float()
    return torch.stack([idx, idx + 1], dim=-1).unsqueeze(0).to(device)


def _ltx23_transformer_inputs(transformer, device="cuda", dtype=torch.bfloat16):
    """Minimal video + audio LTX23Modality inputs plus the matching text cache."""
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    torch.manual_seed(42)
    frames, grid, a_patches, text_len = 1, 4, 8, 8
    cfg = getattr(transformer, "_transformer_config", {})

    v_positions = _positions(frames, grid, grid, device=device)
    a_positions = _positions(a_patches, device=device)
    v_context = torch.randn(
        1, text_len, cfg.get("cross_attention_dim", 4096), device=device, dtype=dtype
    )
    a_context = torch.randn(
        1, text_len, cfg.get("audio_cross_attention_dim", 2048), device=device, dtype=dtype
    )

    sigma = torch.tensor([0.5], device=device)
    timesteps = torch.tensor([0.5], device=device)
    video = LTX23Modality(
        latent=torch.randn(
            1, frames * grid * grid, cfg.get("in_channels", 128), device=device, dtype=dtype
        ),
        timesteps=timesteps,
        sigma=sigma,
        positions=v_positions,
        context=v_context,
    )
    audio = LTX23Modality(
        latent=torch.randn(
            1, a_patches, cfg.get("audio_in_channels", 128), device=device, dtype=dtype
        ),
        timesteps=timesteps,
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


def _ltx23_modality(fill):
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    return LTX23Modality(
        latent=torch.full((1, 4, 8), fill),
        timesteps=torch.full((1,), fill),
        sigma=torch.full((1,), fill),
        positions=torch.full((1, 3, 4, 2), fill),
        context=torch.full((1, 2, 8), fill),
        context_mask=torch.full((1, 2), fill),
    )


def _ltx23_text_conditioning(fill):
    from tensorrt_llm._torch.visual_gen.models.ltx23.text_conditioning_ltx23 import (
        LTX23TextConditioning,
    )

    pair = (torch.full((1, 2), fill), torch.full((1, 2), fill))
    return LTX23TextConditioning(
        video_context=torch.full((1, 2, 8), fill),
        video_mask=torch.full((1, 2), fill),
        video_pe=pair,
        video_cross_pe=pair,
        audio_context=torch.full((1, 2, 4), fill),
        audio_mask=torch.full((1, 2), fill),
        audio_pe=pair,
        audio_cross_pe=pair,
    )


def test_cuda_graph_runner_tracks_all_ltx23_input_state():
    """Every step-varying LTX-2.3 input tensor must reach the captured graph.

    sigma changes on every denoise step, so a runner that drops it from the
    graph key or from the input copy replays a stale prompt modulation.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import _LTX23CUDAGraphRunner

    modality = _ltx23_modality(1.0)
    static = _LTX23CUDAGraphRunner._clone_value(modality)
    fields = ("latent", "timesteps", "sigma", "positions", "context", "context_mask")

    assert static is not modality
    for field in fields:
        assert getattr(static, field).data_ptr() != getattr(modality, field).data_ptr()

    updated = _ltx23_modality(2.0)
    assert _LTX23CUDAGraphRunner._copy_value(static, updated) is static
    for field in fields:
        torch.testing.assert_close(getattr(static, field), getattr(updated, field))

    runner = object.__new__(_LTX23CUDAGraphRunner)
    labels = {label for label, _ in runner._key_parts_for("video", modality)}
    assert {"video.latent", "video.timesteps", "video.sigma", "video.context"} <= labels

    static = _LTX23CUDAGraphRunner._clone_value(_ltx23_text_conditioning(1.0))
    updated = _ltx23_text_conditioning(2.0)
    _LTX23CUDAGraphRunner._copy_value(static, updated)
    for field in ("video_context", "video_mask", "audio_context", "audio_mask"):
        torch.testing.assert_close(getattr(static, field), getattr(updated, field))
    for field in ("video_pe", "video_cross_pe", "audio_pe", "audio_cross_pe"):
        for static_pe, updated_pe in zip(getattr(static, field), getattr(updated, field)):
            torch.testing.assert_close(static_pe, updated_pe)


@pytest.fixture
def ltx23_bf16_checkpoint_exists():
    if not os.path.exists(CHECKPOINT_PATH_BF16):
        pytest.skip(
            f"LTX-2.3 checkpoint not found at {CHECKPOINT_PATH_BF16}. "
            "Set LTX23_MODEL_PATH or stage it under LLM_MODELS_ROOT/LTX-2.3/."
        )
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention_backend_comparison(ltx23_bf16_checkpoint_exists):
    """TRTLLM attention matches VANILLA on both output streams.

    The backends load sequentially: two full LTX-2.3 transformers do not fit in
    GPU memory at once.
    """

    def _load(backend):
        args = VisualGenArgs(
            model=CHECKPOINT_PATH_BF16,
            attention_config=AttentionConfig(backend=backend),
            pipeline_config={"text_encoder_path": GEMMA3_PATH},
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=SKIP_COMPONENTS)
        return pipeline, pipeline.transformer

    pipeline, transformer = _load("VANILLA")
    video, audio, text_cache = _ltx23_transformer_inputs(transformer)
    with torch.no_grad():
        vout, aout = transformer(video=video, audio=audio, text_cache=text_cache)
    baseline = (vout.cpu(), aout.cpu())

    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()

    pipeline, transformer = _load("TRTLLM")
    _, _, text_cache = _ltx23_transformer_inputs(transformer)
    with torch.no_grad():
        vout, aout = transformer(video=video, audio=audio, text_cache=text_cache)
    trtllm = (vout.cpu(), aout.cpu())

    for name, ref, out in zip(("video", "audio"), baseline, trtllm):
        assert ref.shape == out.shape, (
            f"{name} shape mismatch: VANILLA={ref.shape}, TRTLLM={out.shape}"
        )
        assert torch.isfinite(out).all(), f"TRTLLM {name} output is not finite"
        cos_sim = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        assert cos_sim > 0.99, f"TRTLLM {name} should match VANILLA: cos_sim={cos_sim:.6f}"

    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()
