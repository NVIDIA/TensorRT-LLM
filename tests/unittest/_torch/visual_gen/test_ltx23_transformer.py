"""Unit tests for the LTX-2.3 transformer and its model-specific components."""

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

# Reduced vs the checkpoint (3840 caption, 4096/2048 streams) for fast CI.
_CAPTION_CHANNELS = 8
_NUM_STATES = 3
_VIDEO_DIM = 32
_AUDIO_DIM = 2

_VIDEO_ONLY_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=32,
    in_channels=16,
    out_channels=16,
    num_layers=1,
    cross_attention_dim=128,
    caption_channels=64,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 32, 32],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
)

_AUDIO_VIDEO_CONFIG = dict(
    **_VIDEO_ONLY_CONFIG,
    audio_num_attention_heads=4,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=64,
    audio_positional_embedding_max_pos=[4],
    av_ca_timestep_scale_multiplier=1,
)

_FRAMES, _GRID, _AUDIO_PATCHES = 1, 4, 8
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model_config():
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )


def _build_model(model_type_name, config, dtype=torch.bfloat16):
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx23.transformer_ltx23 import LTX23Model

    model = (
        LTX23Model(
            model_type=getattr(LTXModelType, model_type_name),
            model_config=_model_config(),
            **config,
        )
        .to(_DEVICE)
        .to(dtype)
        .eval()
    )
    # TRT-LLM Linear allocates with torch.empty(), so unfilled weights give NaN.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
    return model


def _positions(*sizes, device=_DEVICE):
    grids = torch.meshgrid(*[torch.arange(s) for s in sizes], indexing="ij")
    idx = torch.stack([g.flatten() for g in grids]).float()
    return torch.stack([idx, idx + 1], dim=-1).unsqueeze(0).to(device)


def _modality(n_tokens, in_channels, ctx_dim, positions, dtype, sigma, text_len=8):
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    ctx = torch.randn(1, text_len, ctx_dim, device=_DEVICE, dtype=dtype) * 0.02
    modality = LTX23Modality(
        latent=torch.randn(1, n_tokens, in_channels, device=_DEVICE, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=_DEVICE),
        sigma=torch.tensor([sigma], device=_DEVICE),
        positions=positions,
        context=ctx,
    )
    return modality, ctx


def _inputs(cfg, dtype=torch.bfloat16, sigma=0.5):
    """(video, audio, prepare_text_cache kwargs) at the reduced test dims."""
    v_pos = _positions(_FRAMES, _GRID, _GRID)
    a_pos = _positions(_AUDIO_PATCHES)
    video, v_ctx = _modality(
        _FRAMES * _GRID * _GRID,
        cfg["in_channels"],
        cfg["cross_attention_dim"],
        v_pos,
        dtype,
        sigma,
    )
    audio, a_ctx = _modality(
        _AUDIO_PATCHES,
        cfg["audio_in_channels"],
        cfg["audio_cross_attention_dim"],
        a_pos,
        dtype,
        sigma,
    )
    cache_kwargs = dict(
        video_context=v_ctx,
        video_positions=v_pos,
        audio_context=a_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )
    return video, audio, cache_kwargs


class TestLTX23TextFeatures(unittest.TestCase):
    """Per-token RMS pack and the split video/audio Gemma projections."""

    def _extractor(self, caption=_CAPTION_CHANNELS, states=_NUM_STATES):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23GemmaFeaturesExtractor,
        )

        return LTX23GemmaFeaturesExtractor(
            caption_channels=caption,
            video_dim=_VIDEO_DIM,
            audio_dim=_AUDIO_DIM,
            num_hidden_states=states,
        ).eval()

    def test_pack_is_per_token_rms_normalized(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline

        batch, seq, channels, n = 1, 3, 16, 4
        hidden = [torch.randn(batch, seq, channels) * (10.0 * (i + 1)) for i in range(n)]

        packed = LTX23Pipeline._per_token_rms_pack(hidden, eps=1e-6)
        self.assertEqual(packed.shape, (batch, seq, channels * n))

        rms = packed.view(batch, seq, channels, n).float().pow(2).mean(dim=2).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_projections_apply_modality_rescale(self):
        fe = self._extractor()
        x = torch.randn(2, 4, _CAPTION_CHANNELS * _NUM_STATES)
        with torch.no_grad():
            video, audio = fe(x)

        for out, dim, proj in (
            (video, _VIDEO_DIM, fe.video_aggregate_embed),
            (audio, _AUDIO_DIM, fe.audio_aggregate_embed),
        ):
            scale = math.sqrt(dim / _CAPTION_CHANNELS)
            self.assertTrue(
                torch.allclose(out, F.linear(x * scale, proj.weight, proj.bias), atol=1e-5)
            )
            self.assertFalse(torch.allclose(out, F.linear(x, proj.weight, proj.bias), atol=1e-4))

    def test_from_config_stacks_all_gemma_hidden_states(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23GemmaFeaturesExtractor,
        )

        fe = LTX23GemmaFeaturesExtractor.from_config(
            {
                "transformer": {
                    "caption_channels": 3840,
                    "cross_attention_dim": 4096,
                    "audio_cross_attention_dim": 2048,
                }
            }
        )
        self.assertEqual(fe.video_aggregate_embed.in_features, 3840 * 49)
        self.assertEqual(fe.video_aggregate_embed.out_features, 4096)
        self.assertEqual(fe.audio_aggregate_embed.out_features, 2048)


class TestLTX23Connectors(unittest.TestCase):
    """Two independent 8-layer gated connectors with 128 learnable registers."""

    def _configurators(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23AudioConnectorConfigurator,
            LTX23VideoConnectorConfigurator,
        )

        return (
            (LTX23VideoConnectorConfigurator, 128),
            (LTX23AudioConnectorConfigurator, 64),
        )

    def test_defaults(self):
        for configurator, head_dim in self._configurators():
            conn = configurator.from_config({})
            self.assertEqual(conn.num_attention_heads, 32)
            self.assertEqual(conn.inner_dim, 32 * head_dim)
            self.assertEqual(len(conn.transformer_1d_blocks), 8)
            self.assertEqual(conn.num_learnable_registers, 128)
            self.assertEqual(tuple(conn.learnable_registers.shape), (128, 32 * head_dim))
            for block in conn.transformer_1d_blocks:
                self.assertEqual(block.attn1.to_gate_logits.out_features, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23ModelStructure(unittest.TestCase):
    """9-slot AdaLN plus the sigma-driven prompt AdaLN."""

    def test_adaln_slots_and_prompt_adaln(self):
        model = _build_model("AudioVideo", _AUDIO_VIDEO_CONFIG)
        block = model.transformer_blocks[0]

        for adaln, prompt_adaln, table, prompt_table, dim in (
            (
                model.adaln_single,
                model.prompt_adaln_single,
                block.scale_shift_table,
                block.prompt_scale_shift_table,
                model.inner_dim,
            ),
            (
                model.audio_adaln_single,
                model.audio_prompt_adaln_single,
                block.audio_scale_shift_table,
                block.audio_prompt_scale_shift_table,
                model.audio_inner_dim,
            ),
        ):
            self.assertEqual(adaln.linear.out_features, 9 * dim)
            self.assertEqual(prompt_adaln.linear.out_features, 2 * dim)
            self.assertEqual(tuple(table.shape), (9, dim))
            self.assertEqual(tuple(prompt_table.shape), (2, dim))

        self.assertIsInstance(model.caption_projection, nn.Identity)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23Forward(unittest.TestCase):
    """Full LTX23Model.forward over both streams."""

    def test_audio_video_forward(self):
        torch.manual_seed(42)
        cfg = _AUDIO_VIDEO_CONFIG
        model = _build_model("AudioVideo", cfg)
        video, audio, cache_kwargs = _inputs(cfg)
        text_cache = model.prepare_text_cache(**cache_kwargs)

        with torch.no_grad():
            vout, aout = model(video=video, audio=audio, text_cache=text_cache)

        self.assertEqual(vout.shape, (1, video.latent.shape[1], cfg["out_channels"]))
        self.assertEqual(aout.shape, (1, audio.latent.shape[1], cfg["audio_out_channels"]))
        self.assertTrue(torch.isfinite(vout).all())
        self.assertTrue(torch.isfinite(aout).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23SigmaTextConditioning(unittest.TestCase):
    """Text K/V is modulated by sigma and re-projected every step."""

    def test_sigma_changes_output(self):
        torch.manual_seed(7)
        cfg = _AUDIO_VIDEO_CONFIG
        model = _build_model("AudioVideo", cfg)
        video, audio, cache_kwargs = _inputs(cfg, sigma=0.1)
        text_cache = model.prepare_text_cache(**cache_kwargs)

        hi = torch.tensor([0.9], device=_DEVICE)
        with torch.no_grad():
            vout_lo, aout_lo = model(video=video, audio=audio, text_cache=text_cache)
            vout_hi, aout_hi = model(
                video=replace(video, sigma=hi),
                audio=replace(audio, sigma=hi),
                text_cache=text_cache,
            )

        self.assertGreater((vout_lo.float() - vout_hi.float()).abs().max().item(), 1e-4)
        self.assertGreater((aout_lo.float() - aout_hi.float()).abs().max().item(), 1e-4)


class TestLTX23VideoDecoderChannels(unittest.TestCase):
    """compress_time and compress_space reduce channels in LTX-2.3.

    Built at latent_channels=16 (real checkpoint uses 128), so every width is
    1/8 of the real one and the ratios are unchanged.
    """

    _LATENT = 16

    def _build(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.video_vae_ltx23 import (
            LTX23VideoDecoderConfigurator,
        )

        return LTX23VideoDecoderConfigurator.from_config(
            {
                "vae": {
                    "dims": 3,
                    "latent_channels": self._LATENT,
                    "out_channels": 3,
                    "patch_size": 4,
                    "norm_layer": "pixel_norm",
                    "causal_decoder": False,
                    "timestep_conditioning": False,
                    "spatial_padding_mode": "reflect",
                    "decoder_blocks": [
                        ["res_x", {"num_layers": 4}],
                        ["compress_space", {"multiplier": 2}],
                        ["res_x", {"num_layers": 6}],
                        ["compress_time", {"multiplier": 2}],
                        ["res_x", {"num_layers": 4}],
                        ["compress_all", {"multiplier": 1}],
                        ["res_x", {"num_layers": 2}],
                        ["compress_all", {"multiplier": 2}],
                        ["res_x", {"num_layers": 2}],
                    ],
                }
            }
        )

    def test_conv_in_uses_every_compress_multiplier(self):
        dec = self._build()
        self.assertEqual(dec.conv_in.conv.weight.shape[1], self._LATENT)
        self.assertEqual(dec.conv_in.conv.weight.shape[0], self._LATENT * 8)
        self.assertEqual(dec.up_blocks[5].conv.conv.weight.shape[0], 64)
        self.assertEqual(dec.up_blocks[7].conv.conv.weight.shape[0], 64)


class TestLTX23Vocoder(unittest.TestCase):
    """BigVGAN-v2 generator plus the bandwidth-extension wrapper to 48 kHz."""

    def test_configurator_builds_48khz_bwe(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.audio_vae import (
            LTX23VocoderConfigurator,
            VocoderWithBWE,
        )

        gen_cfg = dict(
            resblock="AMP1",
            stereo=True,
            activation="snakebeta",
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_initial_channel=32,
        )
        voc = LTX23VocoderConfigurator.from_config(
            {
                "vocoder": {
                    "vocoder": dict(gen_cfg),
                    "bwe": dict(
                        gen_cfg,
                        input_sampling_rate=16000,
                        output_sampling_rate=48000,
                        n_fft=32,
                        hop_length=8,
                        num_mels=16,
                    ),
                }
            }
        )

        self.assertIsInstance(voc, VocoderWithBWE)
        self.assertEqual(voc.output_sampling_rate, 48000)
        self.assertEqual(voc.vocoder.output_sampling_rate, 16000)
        self.assertEqual(voc.resampler.ratio, 3)


class TestLTX23PipelineDetection(unittest.TestCase):
    """Native-checkpoint dispatch between LTX2Pipeline and LTX23Pipeline."""

    _LTX23_TRANSFORMER = {
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
        "caption_proj_input_norm": False,
        "cross_attention_adaln": True,
    }

    def test_detection(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import (
            detect_native_ltx_pipeline,
        )

        self.assertEqual(
            detect_native_ltx_pipeline({"transformer": dict(self._LTX23_TRANSFORMER)}),
            "LTX23Pipeline",
        )
        self.assertEqual(detect_native_ltx_pipeline({"transformer": {}}), "LTX2Pipeline")
        with self.assertRaises(ValueError):
            detect_native_ltx_pipeline({"transformer": {"cross_attention_adaln": True}})
