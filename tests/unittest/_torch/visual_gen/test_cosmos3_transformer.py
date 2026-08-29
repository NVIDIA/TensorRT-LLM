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
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    PRETRAINED_CONFIG_COMPAT_DEFAULTS,
    Cosmos3CrossAttention,
    Cosmos3VFMTransformer,
    _normalize_control_weights,
    apply_pretrained_config_compat_defaults,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = [pytest.mark.cosmos3, pytest.mark.usefixtures("disable_cosmos3_guardrails")]


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
_NUM_TRAIN_TIMESTEPS = 1000.0

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


def _enable_audio(
    model_config: DiffusionModelConfig,
    *,
    audio_dim: int = 16,
    audio_latent_fps: float = 24.0,
    temporal_compression_factor: int = 1,
) -> DiffusionModelConfig:
    """Pin the audio (sound) modality on with small, test-friendly dimensions.

    The Cosmos3 checkpoint already enables sound by default; this overrides the
    audio dims so random-weight builds stay light and assertions can rely on a
    known ``audio_dim``. The transformer reads audio attributes via ``sound_*``
    fallbacks (see ``Cosmos3VFMTransformer.__init__``), so we set those legacy
    keys. ``pretrained_config`` is a ``SimpleNamespace``, so attributes can be
    set freely.
    """
    cfg = model_config.pretrained_config
    cfg.sound_gen = True
    cfg.sound_dim = audio_dim
    cfg.sound_latent_fps = audio_latent_fps
    cfg.temporal_compression_factor_sound = temporal_compression_factor
    return model_config


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
        assert all(layer.cross_attention.multi_control_attn is None for layer in model.gen_layers)

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
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out.video, hs.shape)

    @pytest.mark.high_cuda_memory
    def test_sanity_forward_multi_control(self, cosmos3_model_config):
        cfg = cosmos3_model_config.pretrained_config
        model = _build_random_weight_model(cosmos3_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        control_latents = [torch.zeros_like(hs), torch.full_like(hs, 0.05)]
        layer_sequence_lengths = []
        hooks = [
            layer.register_forward_pre_hook(
                lambda _module, args: layer_sequence_lengths.append(args[0].shape[1])
            )
            for layer in model.gen_layers
        ]
        try:
            with torch.inference_mode():
                out = model(
                    hidden_states=hs,
                    timestep=ts / _NUM_TRAIN_TIMESTEPS,
                    raw_timestep=ts,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=video_shape,
                    control_latents=control_latents,
                )
        finally:
            for hook in hooks:
                hook.remove()

        target_tokens = (
            video_shape[0]
            * ((video_shape[1] + model.latent_patch_size - 1) // model.latent_patch_size)
            * ((video_shape[2] + model.latent_patch_size - 1) // model.latent_patch_size)
        )
        assert layer_sequence_lengths == [3 * target_tokens] * len(model.gen_layers)
        assert all(
            layer.cross_attention.multi_control_attn is not None for layer in model.gen_layers
        )
        _assert_finite_output(out.video, hs.shape)

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
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
            out2 = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out1.video, hs.shape)
        _assert_finite_output(out2.video, hs.shape)

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
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                noisy_frame_mask=noisy_frame_mask,
            )
        _assert_finite_output(out.video, hs.shape)


@pytest.mark.integration
class TestCosmos3Audio:
    """Audio (sound) modality — Nano architecture, random weights, audio_gen on.

    Loads the Nano transformer config and flips on the audio modality so the
    audio projection heads and sound-token injection path are exercised without
    needing an audio-capable checkpoint.
    """

    AUDIO_DIM = 16
    T_AUDIO = 8

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def audio_model_config(self):
        # Function-scoped + freshly loaded so we never mutate a config shared
        # with the video-only test classes.
        checkpoint_dir = _require_checkpoint()
        model_config = _load_model_config(checkpoint_dir)
        return _enable_audio(model_config, audio_dim=self.AUDIO_DIM)

    @pytest.fixture
    def cosmos3_model_config_noaudio(self):
        # The Cosmos3 checkpoint enables sound by default, so explicitly disable
        # it to exercise the video-only construction path.
        checkpoint_dir = _require_checkpoint()
        model_config = _load_model_config(checkpoint_dir)
        model_config.pretrained_config.sound_gen = False
        return model_config

    def test_audio_model_structure(self, audio_model_config):
        model = Cosmos3VFMTransformer(model_config=audio_model_config)
        assert model.audio_gen is True
        assert model.audio_dim == self.AUDIO_DIM
        assert hasattr(model, "audio2llm")
        assert hasattr(model, "llm2audio")
        assert hasattr(model, "audio_modality_embed")
        # audio2llm: audio_dim -> hidden_size, llm2audio: hidden_size -> audio_dim
        assert model.audio2llm.in_features == self.AUDIO_DIM
        assert model.audio2llm.out_features == model.hidden_size
        assert model.llm2audio.in_features == model.hidden_size
        assert model.llm2audio.out_features == self.AUDIO_DIM
        assert model.audio_modality_embed.shape == (model.hidden_size,)

    def test_video_only_model_has_no_audio_heads(self, cosmos3_model_config_noaudio):
        model = Cosmos3VFMTransformer(model_config=cosmos3_model_config_noaudio)
        assert model.audio_gen is False
        assert not hasattr(model, "audio2llm")
        assert not hasattr(model, "llm2audio")

    @pytest.mark.high_cuda_memory
    def test_forward_with_audio(self, audio_model_config):
        cfg = audio_model_config.pretrained_config
        model = _build_random_weight_model(audio_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        audio_latents = torch.randn(1, model.audio_dim, self.T_AUDIO, device=DEVICE, dtype=DTYPE)
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                audio_latents=audio_latents,
            )
        # Video velocity is unchanged in shape; audio velocity mirrors the input.
        _assert_finite_output(out.video, hs.shape)
        assert out.audio is not None
        _assert_finite_output(out.audio, torch.Size([1, model.audio_dim, self.T_AUDIO]))

    @pytest.mark.high_cuda_memory
    def test_forward_without_audio_latents_returns_none(self, audio_model_config):
        """An audio-capable model still returns audio=None when no audio is passed."""
        cfg = audio_model_config.pretrained_config
        model = _build_random_weight_model(audio_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out.video, hs.shape)
        assert out.audio is None

    @pytest.mark.high_cuda_memory
    def test_forward_with_audio_multiframe(self, audio_model_config):
        """Audio injection works alongside a multi-frame video sequence."""
        cfg = audio_model_config.pretrained_config
        model = _build_random_weight_model(audio_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel, t=3
        )
        audio_latents = torch.randn(1, model.audio_dim, self.T_AUDIO, device=DEVICE, dtype=DTYPE)
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                audio_latents=audio_latents,
            )
        _assert_finite_output(out.video, hs.shape)
        _assert_finite_output(out.audio, torch.Size([1, model.audio_dim, self.T_AUDIO]))


@pytest.mark.integration
class TestCosmos3Action:
    """Action modality — Nano architecture, random weights, action_gen on."""

    ACTION_DIM = 64
    T_ACTION = 4
    NUM_DOMAINS = 32

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def action_model_config(self):
        checkpoint_dir = _require_checkpoint()
        model_config = _load_model_config(checkpoint_dir)
        cfg = model_config.pretrained_config
        cfg.action_gen = True
        cfg.action_dim = self.ACTION_DIM
        cfg.num_embodiment_domains = self.NUM_DOMAINS
        cfg.sound_gen = False
        return model_config

    @pytest.fixture
    def cosmos3_model_config_noaction(self):
        checkpoint_dir = _require_checkpoint()
        model_config = _load_model_config(checkpoint_dir)
        model_config.pretrained_config.action_gen = False
        model_config.pretrained_config.sound_gen = False
        return model_config

    def test_action_model_structure(self, action_model_config):
        model = Cosmos3VFMTransformer(model_config=action_model_config)
        assert model.action_gen is True
        assert model.action_dim == self.ACTION_DIM
        assert hasattr(model, "action_proj_in")
        assert hasattr(model, "action_proj_out")
        assert hasattr(model, "action_modality_embed")
        assert model.action_modality_embed.shape == (model.hidden_size,)

    def test_video_only_model_has_no_action_heads(self, cosmos3_model_config_noaction):
        model = Cosmos3VFMTransformer(model_config=cosmos3_model_config_noaction)
        assert model.action_gen is False
        assert not hasattr(model, "action_proj_in")
        assert not hasattr(model, "action_proj_out")
        assert not hasattr(model, "action_modality_embed")

    def test_pack_action_rejects_wrong_last_dim(self, action_model_config):
        model = Cosmos3VFMTransformer(model_config=action_model_config)
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim - 1)
        with pytest.raises(ValueError, match="action latent dimension mismatch"):
            model.pack_action(action_latents)

    @pytest.mark.high_cuda_memory
    def test_forward_with_action(self, action_model_config):
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        domain_ids = torch.tensor([7], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                action_latents=action_latents,
                action_domain_ids=domain_ids,
            )
        _assert_finite_output(out.video, hs.shape)
        assert out.action is not None
        _assert_finite_output(out.action, torch.Size([1, self.T_ACTION, model.action_dim]))

    @pytest.mark.high_cuda_memory
    def test_domain_ids_validated_once_per_request(self, action_model_config):
        """The range check reads a device tensor, so it is a blocking sync. It
        belongs on the first step of a request, not on every denoise step."""
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        domain_ids = torch.tensor([7], dtype=torch.long, device=DEVICE)

        calls = []
        real_validate = model.action_proj_in.validate_domain_ids
        model.action_proj_in.validate_domain_ids = lambda ids: (
            calls.append(ids),
            real_validate(ids),
        )[1]

        def run_step():
            with torch.inference_mode():
                model(
                    hidden_states=hs,
                    timestep=ts / _NUM_TRAIN_TIMESTEPS,
                    raw_timestep=ts,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=video_shape,
                    fps=24.0,
                    action_latents=action_latents,
                    action_domain_ids=domain_ids,
                )

        run_step()
        run_step()
        assert len(calls) == 1

        model.reset_cache()
        run_step()
        assert len(calls) == 2

    def test_graph_key_separates_requests_that_differ_only_in_scalars(self, action_model_config):
        """TRT-LLM captures a family of graphs and dispatches by key. fps, the
        action clock and the start offset change the rotary positions without
        changing any tensor shape, so they must discriminate keys or two such
        requests would replay the same graph."""
        from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
            CUDAGraphRunner,
            CUDAGraphRunnerConfig,
        )

        model = Cosmos3VFMTransformer(model_config=action_model_config)
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        model.register_cuda_graph_extra_key_fns(runner)

        base = dict(fps=24.0, action_fps=5.0, action_start_frame_offset=1)
        key = runner.get_graph_key(**base)
        for field, other in (
            ("fps", 16.0),
            ("action_fps", 10.0),
            ("action_start_frame_offset", 0),
        ):
            assert runner.get_graph_key(**{**base, field: other}) != key, field

        # A video-only request keys exactly as before: absent scalars drop out.
        assert runner.get_graph_key(fps=None, action_fps=None) == runner.get_graph_key()

    @pytest.mark.high_cuda_memory
    def test_action_rope_table_built_once_per_request(self, action_model_config):
        """Chunk size, prompt lengths, fps and the frame offset are fixed for a
        request, so the rotary table is too. Rebuilding it per step costs a
        device-to-host sync per batch element plus an H2D copy of the position
        ids -- for identical numbers."""
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        domain_ids = torch.tensor([7], dtype=torch.long, device=DEVICE)

        calls = []
        real = model._compute_action_rope_freqs
        model._compute_action_rope_freqs = lambda *a, **k: (calls.append(1), real(*a, **k))[1]

        def run_step():
            with torch.inference_mode():
                model(
                    hidden_states=hs,
                    timestep=ts / _NUM_TRAIN_TIMESTEPS,
                    raw_timestep=ts,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=video_shape,
                    fps=24.0,
                    action_latents=action_latents,
                    action_domain_ids=domain_ids,
                )

        run_step()
        run_step()
        run_step()
        assert len(calls) == 1

        model.reset_cache()
        run_step()
        assert len(calls) == 2

    @pytest.mark.high_cuda_memory
    def test_forward_with_action_domain_id_out_of_range_raises(self, action_model_config):
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        domain_ids = torch.tensor([self.NUM_DOMAINS], dtype=torch.long, device=DEVICE)
        with (
            torch.inference_mode(),
            pytest.raises(ValueError, match=r"domain_id must be in \[0, \d+\)"),
        ):
            model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                action_latents=action_latents,
                action_domain_ids=domain_ids,
            )

    @pytest.mark.high_cuda_memory
    def test_forward_without_action_latents_returns_none(self, action_model_config):
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel
        )
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out.video, hs.shape)
        assert out.action is None

    @pytest.mark.high_cuda_memory
    def test_forward_with_action_noisy_mask(self, action_model_config):
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel, t=2
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        noisy_mask = torch.ones(1, self.T_ACTION, 1, device=DEVICE, dtype=DTYPE)
        noisy_mask[:, 0, :] = 0.0
        domain_ids = torch.tensor([7], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                action_latents=action_latents,
                action_domain_ids=domain_ids,
                action_noisy_mask=noisy_mask,
            )
        _assert_finite_output(out.video, hs.shape)
        _assert_finite_output(out.action, torch.Size([1, self.T_ACTION, model.action_dim]))

    @pytest.mark.high_cuda_memory
    def test_forward_with_action_multiframe(self, action_model_config):
        cfg = action_model_config.pretrained_config
        model = _build_random_weight_model(action_model_config)
        hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
            DEVICE, channels=cfg.latent_channel, t=3
        )
        action_latents = torch.randn(1, self.T_ACTION, model.action_dim, device=DEVICE, dtype=DTYPE)
        domain_ids = torch.tensor([7], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            out = model(
                hidden_states=hs,
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
                action_latents=action_latents,
                action_domain_ids=domain_ids,
            )
        _assert_finite_output(out.video, hs.shape)
        _assert_finite_output(out.action, torch.Size([1, self.T_ACTION, model.action_dim]))


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
                timestep=ts / _NUM_TRAIN_TIMESTEPS,
                raw_timestep=ts,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
            )
        _assert_finite_output(out.video, hs.shape)

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
                    timestep=ts / _NUM_TRAIN_TIMESTEPS,
                    raw_timestep=ts,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=video_shape,
                )
            _assert_finite_output(out.video, hs.shape)
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()


# --- CPU-only coverage: checkpoint config schema compatibility ---


class _RecordingMultiControlBackend:
    def __init__(self, control_ids=(1, 2)):
        self.calls = []
        self.control_ids = iter(control_ids)

    def forward(self, *, q, k, v, **_kwargs):
        self.calls.append({"q": q.clone(), "k": k.clone(), "v": v.clone()})
        control_id = next(self.control_ids)
        result = torch.empty_like(q)
        result[:, :, :1] = 100 + control_id
        result[:, :, 1:] = 10 if control_id == 1 else 30
        return result


class _FailingMultiControlBackend:
    def forward(self, **_kwargs):
        raise AssertionError("single-control fast path used multi-control attention")


def _mock_cross_attention(backend, *, num_attention_heads=2, head_dim=2) -> Cosmos3CrossAttention:
    attention = Cosmos3CrossAttention.__new__(Cosmos3CrossAttention)
    torch.nn.Module.__init__(attention)
    attention.local_num_attention_heads = num_attention_heads
    attention.local_num_key_value_heads = num_attention_heads
    attention.head_dim = head_dim
    attention.multi_control_attn = backend
    return attention


class TestCosmos3MultiControlAttention:
    def test_ulysses_rank_segments_preserve_global_ranges(self):
        # Post-A2A KV is rank-packed as [text_0, gen_0, text_1, gen_1].
        packed = torch.arange(10.0).reshape(1, 10, 1, 1)
        segments = Cosmos3CrossAttention._ulysses_rank_segments(
            packed,
            batch_idx=0,
            global_start=2,
            global_end=5,
            local_sequence_length=3,
            rank_stride=5,
            rank_offset=2,
            world_size=2,
        )
        torch.testing.assert_close(
            torch.cat(segments, dim=1).flatten(), torch.tensor([4.0, 7.0, 8.0])
        )

    def _run_two_controls(self):
        backend = _RecordingMultiControlBackend()
        attention = _mock_cross_attention(backend)
        q = torch.arange(1.0, 13.0).reshape(1, 3, 2, 2)
        k = q + 20
        v = q + 40
        k_und = torch.arange(101.0, 105.0).reshape(1, 1, 2, 2)
        v_und = torch.arange(201.0, 205.0).reshape(1, 1, 2, 2)
        output = attention._forward_multi_control(
            q,
            k,
            v,
            k_und,
            v_und,
            control_token_sizes=(1, 1),
            control_weights=(0.5, 0.5),
            timestep=None,
            real_text_lens=[1],
        )
        return output, backend.calls

    def test_uniform_target_equation(self):
        output, calls = self._run_two_controls()
        assert len(calls) == 2
        expected = torch.tensor([[[101.0] * 4, [102.0] * 4, [20.0] * 4]])
        torch.testing.assert_close(output, expected)

    def test_control_attention_isolation(self):
        _, calls = self._run_two_controls()
        expected_q = [
            torch.tensor([[[[1.0, 2.0], [9.0, 10.0]], [[3.0, 4.0], [11.0, 12.0]]]]),
            torch.tensor([[[[5.0, 6.0], [9.0, 10.0]], [[7.0, 8.0], [11.0, 12.0]]]]),
        ]
        expected_k = [
            torch.tensor(
                [
                    [
                        [[101.0, 102.0], [21.0, 22.0], [29.0, 30.0]],
                        [[103.0, 104.0], [23.0, 24.0], [31.0, 32.0]],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [[101.0, 102.0], [25.0, 26.0], [29.0, 30.0]],
                        [[103.0, 104.0], [27.0, 28.0], [31.0, 32.0]],
                    ]
                ]
            ),
        ]
        expected_v = [
            torch.tensor(
                [
                    [
                        [[201.0, 202.0], [41.0, 42.0], [49.0, 50.0]],
                        [[203.0, 204.0], [43.0, 44.0], [51.0, 52.0]],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [[201.0, 202.0], [45.0, 46.0], [49.0, 50.0]],
                        [[203.0, 204.0], [47.0, 48.0], [51.0, 52.0]],
                    ]
                ]
            ),
        ]
        for call, query, key, value in zip(calls, expected_q, expected_k, expected_v, strict=True):
            torch.testing.assert_close(call["q"], query)
            torch.testing.assert_close(call["k"], key)
            torch.testing.assert_close(call["v"], value)

    def test_per_batch_text_lengths(self):
        backend = _RecordingMultiControlBackend(control_ids=(1, 1, 2, 2))
        attention = _mock_cross_attention(backend)
        q = torch.arange(1.0, 25.0).reshape(2, 3, 2, 2)
        k = q + 30
        v = q + 60
        k_und = torch.arange(101.0, 117.0).reshape(2, 2, 2, 2)
        v_und = torch.arange(201.0, 217.0).reshape(2, 2, 2, 2)

        output = attention._forward_multi_control(
            q,
            k,
            v,
            k_und,
            v_und,
            control_token_sizes=(1, 1),
            control_weights=(0.5, 0.5),
            timestep=None,
            real_text_lens=[1, 2],
        )

        assert [call["k"].shape[2] for call in backend.calls] == [3, 4, 3, 4]
        torch.testing.assert_close(backend.calls[0]["k"][:, :, 0], k_und[0:1, 0])
        torch.testing.assert_close(backend.calls[1]["k"][:, :, :2], k_und[1:2].transpose(1, 2))
        expected = torch.tensor([[[101.0] * 4, [102.0] * 4, [20.0] * 4]] * 2)
        torch.testing.assert_close(output, expected)

    def test_control_weight_defaults_and_normalization(self):
        assert _normalize_control_weights(2, None) == (0.5, 0.5)
        assert _normalize_control_weights(2, [1.0, 3.0]) == (0.25, 0.75)

    @pytest.mark.parametrize(
        ("weights", "match"),
        [
            ([1.0], "length must match"),
            ([-1.0, 2.0], "finite and non-negative"),
            ([float("nan"), 1.0], "finite and non-negative"),
            ([float("inf"), 1.0], "finite and non-negative"),
            ([0.0, 0.0], "positive sum"),
        ],
    )
    def test_rejects_invalid_control_weights(self, weights, match):
        with pytest.raises(ValueError, match=match):
            _normalize_control_weights(2, weights)

    def test_single_control_uses_existing_attention_path(self):
        attention = _mock_cross_attention(
            _FailingMultiControlBackend(), num_attention_heads=1, head_dim=1
        )
        q = torch.ones(1, 1, 1)
        attention.get_qkv = lambda _hidden_states: (q, q, q)
        attention.apply_qk_norm = lambda query, key: (query, key)
        calls = []

        def _existing_attention(query, key, value, **_kwargs):
            calls.append((query.clone(), key.clone(), value.clone()))
            return query.flatten(2)

        attention._attn_impl = _existing_attention
        attention.to_out = torch.nn.ModuleList([torch.nn.Identity()])
        hidden_states = torch.ones(1, 1, 1)
        k_und = torch.zeros(1, 1, 1, 1)
        v_und = torch.zeros(1, 1, 1, 1)
        freqs_cos = torch.ones(1, 1, 1, 1)
        freqs_sin = torch.zeros(1, 1, 1, 1)

        output = attention(
            hidden_states,
            k_und,
            v_und,
            freqs_cos,
            freqs_sin,
        )

        assert len(calls) == 1
        assert calls[0][1].shape[1] == 2
        torch.testing.assert_close(output, hidden_states)


# Distinguishes "attribute absent" from "attribute present and None".
_OMITTED = object()


class TestConfigCompatDefaults:
    """Newer diffusers conversions omit fields older ones carried explicitly."""

    def test_new_schema_gets_defaults(self):
        config = SimpleNamespace(hidden_size=64, rope_axes_dim=[4, 2, 2])
        apply_pretrained_config_compat_defaults(config)
        for key, value in PRETRAINED_CONFIG_COMPAT_DEFAULTS.items():
            assert getattr(config, key) == value

    def test_old_schema_untouched(self):
        # Every field deliberately differs from its compat default, so an
        # overwrite of any one of them fails its assertion.
        config = SimpleNamespace(
            position_embedding_type="rope_3d",
            max_position_embeddings=12345,
            temporal_compression_factor_sound=7,
        )
        apply_pretrained_config_compat_defaults(config)
        assert config.position_embedding_type == "rope_3d"
        assert config.max_position_embeddings == 12345
        assert config.temporal_compression_factor_sound == 7

    def test_idempotent(self):
        config = SimpleNamespace(hidden_size=64)
        apply_pretrained_config_compat_defaults(config)
        snapshot = vars(config).copy()
        apply_pretrained_config_compat_defaults(config)
        assert vars(config) == snapshot

    @pytest.mark.parametrize("rope_scaling", [_OMITTED, None, {}], ids=["omitted", "none", "empty"])
    def test_rope_type_tolerates_missing_rope_scaling(self, rope_scaling: object) -> None:
        """``rope_axes_dim`` alone is a supported shape, so reading ``rope_type``
        must not fail before ``resolve_rope_axes_dim`` gets to honour it.

        ``omitted`` leaves the attribute off entirely, which is the case the
        ``getattr(..., None)`` guard exists for; ``none``/``empty`` only reach the
        ``or {}`` half.
        """
        from tensorrt_llm._torch.visual_gen.models.cosmos3 import transformer_cosmos3 as tf

        config = SimpleNamespace(
            hidden_size=64,
            head_dim=16,
            rope_axes_dim=[4, 2, 2],
            rope_theta=10000.0,
            max_position_embeddings=128,
        )
        if rope_scaling is not _OMITTED:
            config.rope_scaling = rope_scaling
        assert hasattr(config, "rope_scaling") is (rope_scaling is not _OMITTED)
        apply_pretrained_config_compat_defaults(config)
        # Construct for real: the point is that __init__ reaches the resolver
        # instead of raising AttributeError on the missing block.
        embedding = tf.Qwen3VLTextRotaryEmbedding(SimpleNamespace(pretrained_config=config))
        assert embedding.rope_type == "default"
        assert embedding.mrope_section == [4, 2, 2]


class TestI2V4StepConfigShape:
    """The Image2Video-4Step conversion drops the audio/action towers
    (``sound_dim: null``, no ``action_*`` keys) and carries newer schema
    fields (``qk_norm_for_text``, ``hidden_act``, nested ``rope_theta``).
    The transformer must construct from that exact key set. CPU-only with
    shrunk dimensions; the real 64B shape is covered by the checkpoint
    integration test."""

    def _reduced_i2v_config(self) -> SimpleNamespace:
        # Key set mirrors the checkpoint's transformer/config.json verbatim;
        # only the sizes are reduced (head_dim 8 -> mrope_section sums to 4).
        return SimpleNamespace(
            attention_bias=False,
            attention_dropout=0.0,
            base_fps=16,
            enable_fps_modulation=True,
            head_dim=8,
            hidden_act="silu",
            hidden_size=32,
            intermediate_size=64,
            latent_channel=4,
            latent_patch_size=2,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            patch_latent_dim=16,
            qk_norm_for_text=True,
            rms_norm_eps=1e-6,
            rope_axes_dim=[2, 1, 1],
            rope_scaling={
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "rope_theta": 5000000,
                "rope_type": "default",
            },
            rope_theta=5000000,
            sound_dim=None,
            sound_gen=False,
            sound_latent_fps=25,
            timestep_scale=0.001,
            unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=15000,
            vocab_size=64,
        )

    def test_constructs_without_audio_or_action_towers(self):
        model_config = DiffusionModelConfig(pretrained_config=self._reduced_i2v_config())
        model = Cosmos3VFMTransformer(model_config)

        assert model.audio_gen is False
        assert model.has_action_weights is False
        assert not hasattr(model, "audio2llm")
        assert not hasattr(model, "audio_modality_embed")
        assert model.base_fps == 16
        assert len(model.gen_layers) == 2
