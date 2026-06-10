# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for Cosmos3VFMTransformer.

Unit tests load architecture params from ``transformer/config.json`` in the
Cosmos3-Nano checkpoint (random weights). Integration tests load full weights.

Run unit tests:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_transformer.py -v -s -k Unit

Run all:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_transformer.py -v -s

Override checkpoint:
    DIFFUSION_MODEL_PATH_COSMOS3=/path/to/Cosmos3-Nano \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_transformer.py -v -s
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    DiffusionPipelineConfig,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = pytest.mark.cosmos3


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _llm_models_root() -> str:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch/trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return str(root)


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or os.path.join(_llm_models_root(), default_name)


COSMOS3_NANO_PATH = _checkpoint("DIFFUSION_MODEL_PATH_COSMOS3", "Cosmos3-Nano")

DEVICE = "cuda"
DTYPE = torch.bfloat16

COSMOS3_FP8_QUANT_CONFIG = {
    "quant_algo": "FP8",
    "dynamic": True,
    "ignore": ["language_model.*", "vae2llm", "llm2vae", "time_embedder.*"],
}

_SKIP_AUX = [
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
    PipelineComponent.TOKENIZER,
]


def _transformer_config_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "transformer", "config.json")


def _require_checkpoint() -> str:
    if not COSMOS3_NANO_PATH or not os.path.isdir(COSMOS3_NANO_PATH):
        pytest.skip(f"Checkpoint not found: {COSMOS3_NANO_PATH}")
    config_path = _transformer_config_path(COSMOS3_NANO_PATH)
    if not os.path.isfile(config_path):
        pytest.skip(f"Transformer config not found: {config_path}")
    return COSMOS3_NANO_PATH


def _load_model_config(checkpoint_dir: str) -> DiffusionModelConfig:
    """Build DiffusionModelConfig from ``checkpoint_dir/transformer/config.json``."""
    args = VisualGenArgs(
        model=checkpoint_dir,
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    return DiffusionPipelineConfig.from_pretrained(checkpoint_dir, args=args).primary_model_config


def _init_all_weights(model: torch.nn.Module, std: float = 0.02) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                param.fill_(1.0)
            elif param.numel() > 0:
                torch.nn.init.normal_(param, mean=0.0, std=std)


def _build_random_weight_model(model_config: DiffusionModelConfig) -> Cosmos3VFMTransformer:
    """Instantiate on CUDA with random weights; keep fp32 RoPE/time embed buffers."""
    model = Cosmos3VFMTransformer(model_config=model_config).to(DEVICE).eval()
    _init_all_weights(model)
    model.post_load_weights()
    return model


def _cosmos3_inputs(
    device: str,
    *,
    batch: int = 1,
    channels: int = 16,
    t: int = 1,
    h: int = 8,
    w: int = 8,
    text_len: int = 32,
    max_text_len: int = 64,
    dtype: torch.dtype = DTYPE,
):
    torch.manual_seed(42)
    hidden_states = torch.randn(batch, channels, t, h, w, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float32)
    text_ids = torch.randint(1, 1000, (batch, max_text_len), device=device, dtype=torch.long)
    text_mask = torch.zeros(batch, max_text_len, device=device, dtype=torch.long)
    text_mask[:, :text_len] = 1
    video_shape = (t, h, w)
    return hidden_states, timestep, text_ids, text_mask, video_shape


def _assert_finite_output(out: torch.Tensor, expected_shape: torch.Size) -> None:
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    out_f = out.float()
    assert not torch.isnan(out_f).any()
    assert not torch.isinf(out_f).any()


@pytest.mark.integration
class TestCosmos3Unit:
    """Unit tests — Nano architecture from checkpoint config, random weights."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture(scope="class")
    def cosmos3_model_config(self):
        checkpoint_dir = _require_checkpoint()
        return _load_model_config(checkpoint_dir)

    def test_model_structure(self, cosmos3_model_config):
        cfg = cosmos3_model_config.pretrained_config
        model = Cosmos3VFMTransformer(model_config=cosmos3_model_config)
        assert hasattr(model, "language_model")
        assert hasattr(model, "gen_layers")
        assert len(model.language_model.layers) == cfg.num_hidden_layers
        assert len(model.gen_layers) == cfg.num_hidden_layers
        assert hasattr(model, "vae2llm")
        assert hasattr(model, "llm2vae")
        assert hasattr(model, "time_embedder")
        linear_names = [n for n, m in model.named_modules() if isinstance(m, Linear)]
        assert any("to_q" in n or "qkv_proj" in n for n in linear_names)

    @pytest.mark.high_cuda_memory
    def test_sanity_forward(self, cosmos3_model_config):
        cfg = cosmos3_model_config.pretrained_config
        model = _build_random_weight_model(cosmos3_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out, hs.shape)

    @pytest.mark.high_cuda_memory
    def test_reset_cache(self, cosmos3_model_config):
        cfg = cosmos3_model_config.pretrained_config
        model = _build_random_weight_model(cosmos3_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        with torch.inference_mode():
            out1 = model(
                hidden_states=hs,
                timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
            out2 = model(
                hidden_states=hs,
                timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out1, hs.shape)
        _assert_finite_output(out2, hs.shape)

    @pytest.mark.high_cuda_memory
    def test_sanity_forward_i2v_mask(self, cosmos3_model_config):
        cfg = cosmos3_model_config.pretrained_config
        model = _build_random_weight_model(cosmos3_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel, t=2
        )
        noisy_frame_mask = torch.zeros(1, 1, 2, 1, 1, device=DEVICE, dtype=DTYPE)
        noisy_frame_mask[:, :, 0, :, :] = 0.0
        noisy_frame_mask[:, :, 1, :, :] = 1.0
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                noisy_frame_mask=noisy_frame_mask,
            )
        _assert_finite_output(out, hs.shape)


@pytest.mark.integration
class TestCosmos3TransformerCheckpoint:
    """Load Cosmos3-Nano transformer weights and run a single forward step."""

    @pytest.fixture(scope="class")
    def cosmos3_transformer(self):
        checkpoint_dir = _require_checkpoint()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        args = VisualGenArgs(
            model=checkpoint_dir,
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=_SKIP_AUX)
        transformer = pipeline.transformer
        yield transformer
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def test_load_weights_and_forward(self, cosmos3_transformer):
        transformer = cosmos3_transformer
        c = transformer.latent_channel_size
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=c, t=1, h=16, w=16
        )
        transformer.reset_cache()
        with torch.inference_mode():
            out = transformer(
                hidden_states=hs,
                timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out, hs.shape)

    @pytest.mark.parametrize("quant_algo", ["FP8"])
    def test_load_fp8_quantization(self, quant_algo: str):
        checkpoint_dir = _require_checkpoint()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        args = VisualGenArgs(
            model=checkpoint_dir,
            quant_config={**COSMOS3_FP8_QUANT_CONFIG, "quant_algo": quant_algo},
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=_SKIP_AUX)
        try:
            assert pipeline.pipeline_config.quant_config.quant_algo is not None
            transformer = pipeline.transformer
            c = transformer.latent_channel_size
            hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
                DEVICE, channels=c, t=1, h=8, w=8
            )
            transformer.reset_cache()
            with torch.inference_mode():
                out = transformer(
                    hidden_states=hs,
                    timestep=ts,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=video_shape,
                )
            _assert_finite_output(out, hs.shape)
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
