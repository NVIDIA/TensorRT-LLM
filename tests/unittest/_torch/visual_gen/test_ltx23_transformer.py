"""Unit tests for LTX-2.3 ("V2") components.

Bottom-up, one component per test class, mirroring ``test_ltx2_transformer.py``.
These are the sigma-independent, structural pieces that can be checked with
random weights and no checkpoint / no CUDA:

- ``TestLTX23FeatureExtractor``  -> split per-modality projectors
- ``TestLTX23TextPack``          -> per-token RMS pack that feeds them
- ``TestLTX23VideoConnector`` / ``TestLTX23AudioConnector`` -> the two V2
  gated 8-layer connectors (heads/dim/depth/registers/gating), plus
  ``TestLTX23ConnectorConfigOverrides`` for config-key wiring

Later components (transformer block, full LTX23Model forward) get their own
CUDA-gated classes as they are validated.
"""

import unittest

import pytest
import torch


# Reduced dims for a fast, checkpoint-free test.
# Real model: caption_channels=3840, 49 Gemma hidden states -> 188160 in;
# video_dim=4096, audio_dim=2048.
_CAPTION_CHANNELS = 8
_NUM_STATES = 3
_VIDEO_DIM = 16
_AUDIO_DIM = 8


class TestLTX23FeatureExtractor(unittest.TestCase):
    """Split Gemma feature extractor: video/audio projections done pre-connector.

    LTX-2.3 replaces LTX-2's single shared ``aggregate_embed``
    (Linear(3840*49 -> 3840, bias=False)) with two biased projections
    (``video_aggregate_embed`` -> 4096, ``audio_aggregate_embed`` -> 2048).
    """

    def _make(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23GemmaFeaturesExtractor,
        )

        return LTX23GemmaFeaturesExtractor(
            caption_channels=_CAPTION_CHANNELS,
            video_dim=_VIDEO_DIM,
            audio_dim=_AUDIO_DIM,
            num_hidden_states=_NUM_STATES,
        )

    def test_structure(self):
        """Two biased Linears with the split output dims and the flattened input dim."""
        fe = self._make()
        in_features = _CAPTION_CHANNELS * _NUM_STATES

        self.assertTrue(hasattr(fe, "video_aggregate_embed"))
        self.assertTrue(hasattr(fe, "audio_aggregate_embed"))

        self.assertEqual(fe.video_aggregate_embed.in_features, in_features)
        self.assertEqual(fe.video_aggregate_embed.out_features, _VIDEO_DIM)
        self.assertEqual(fe.audio_aggregate_embed.in_features, in_features)
        self.assertEqual(fe.audio_aggregate_embed.out_features, _AUDIO_DIM)

        # LTX-2.3 projections are biased (LTX-2's shared projection was not).
        self.assertIsNotNone(fe.video_aggregate_embed.bias)
        self.assertIsNotNone(fe.audio_aggregate_embed.bias)

    def test_forward_shapes(self):
        """(B, S, C*num_states) -> (video [B,S,video_dim], audio [B,S,audio_dim])."""
        fe = self._make().eval()
        batch, seq = 2, 5
        x = torch.randn(batch, seq, _CAPTION_CHANNELS * _NUM_STATES)

        with torch.no_grad():
            video, audio = fe(x)

        self.assertEqual(video.shape, (batch, seq, _VIDEO_DIM))
        self.assertEqual(audio.shape, (batch, seq, _AUDIO_DIM))

    def test_projections_are_independent(self):
        """Video and audio share the input but are produced by distinct weights."""
        fe = self._make().eval()
        x = torch.randn(1, 4, _CAPTION_CHANNELS * _NUM_STATES)
        with torch.no_grad():
            video, audio = fe(x)
        # Distinct out dims already imply distinct projections; also verify the
        # weight tensors themselves are not aliased.
        self.assertIsNot(
            fe.video_aggregate_embed.weight,
            fe.audio_aggregate_embed.weight,
        )
        self.assertFalse(torch.equal(video[..., :_AUDIO_DIM], audio))

    def test_from_config_uses_49_gemma_states(self):
        """from_config wires the real 49-state input dim and cross-attention dims."""
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23GemmaFeaturesExtractor,
        )

        config = {
            "transformer": {
                "caption_channels": 3840,
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
            }
        }
        fe = LTX23GemmaFeaturesExtractor.from_config(config)
        self.assertEqual(fe.video_aggregate_embed.in_features, 3840 * 49)
        self.assertEqual(fe.video_aggregate_embed.out_features, 4096)
        self.assertEqual(fe.audio_aggregate_embed.out_features, 2048)


class TestLTX23TextPack(unittest.TestCase):
    """Per-token RMS pack of the stacked Gemma hidden states.

    LTX-2.3 uses ``text_encoder_norm_type=per_token_rms`` (vs LTX-2's masked
    min-max) with the norm applied here, before the split projection. The pack
    stacks all hidden states, RMS-normalizes per (token, layer), then flattens
    to feed the feature extractor's ``[out, C*num_states]`` weights.
    """

    def _pack(self, hidden_states, eps=1e-6):
        from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline

        return LTX23Pipeline._per_token_rms_pack(hidden_states, eps=eps)

    def test_output_shape(self):
        """N hidden states of [B,S,C] -> [B, S, C*N]."""
        batch, seq, channels, n = 2, 4, 8, 3
        hidden = [torch.randn(batch, seq, channels) for _ in range(n)]
        packed = self._pack(hidden)
        self.assertEqual(packed.shape, (batch, seq, channels * n))

    def test_per_token_rms_normalized(self):
        """Each (token, layer) hidden vector has ~unit RMS after packing."""
        batch, seq, channels, n = 1, 3, 16, 4
        # Deliberately large/varied magnitudes so normalization is observable.
        hidden = [torch.randn(batch, seq, channels) * (10.0 * (i + 1)) for i in range(n)]

        packed = self._pack(hidden)  # [B, S, C*N]
        # Unflatten back to [B, S, C, N] to measure per-(token, layer) RMS.
        unflat = packed.view(batch, seq, channels, n)
        rms = unflat.float().pow(2).mean(dim=2).sqrt()  # [B, S, N]

        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))


class _ConnectorMixin:
    """Shared structural assertions for the V2 embeddings connectors.

    LTX-2.3 uses two independent connectors, both **8-layer, gated, 128
    learnable registers** (vs LTX-2's single 2-layer, ungated, shared one):

    - video: 32 heads x 128 -> inner_dim 4096
    - audio: 32 heads x  64 -> inner_dim 2048

    We assert on the constructed ``Embeddings1DConnector`` module (built with
    random weights, CPU-only, no forward): head count, inner dim, depth,
    register bank shape, and that gating is actually wired (each block's
    attention gains a ``to_gate_logits`` head, which is ``None`` when ungated).
    """

    #: subclasses set these
    _EXPECTED_HEADS = 32
    _EXPECTED_HEAD_DIM = 128
    _EXPECTED_LAYERS = 8
    _EXPECTED_REGISTERS = 128

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def test_head_count_and_inner_dim(self):
        conn = self._build()
        inner_dim = self._EXPECTED_HEADS * self._EXPECTED_HEAD_DIM
        self.assertEqual(conn.num_attention_heads, self._EXPECTED_HEADS)
        self.assertEqual(conn.inner_dim, inner_dim)

    def test_depth(self):
        conn = self._build()
        self.assertEqual(len(conn.transformer_1d_blocks), self._EXPECTED_LAYERS)

    def test_learnable_registers(self):
        conn = self._build()
        inner_dim = self._EXPECTED_HEADS * self._EXPECTED_HEAD_DIM
        self.assertEqual(conn.num_learnable_registers, self._EXPECTED_REGISTERS)
        self.assertEqual(
            tuple(conn.learnable_registers.shape),
            (self._EXPECTED_REGISTERS, inner_dim),
        )

    def test_gated_attention_is_wired(self):
        """Every block's attention has a per-head gate (``to_gate_logits``)."""
        conn = self._build()
        for block in conn.transformer_1d_blocks:
            gate = block.attn1.to_gate_logits
            self.assertIsNotNone(gate, "connector attention must be gated in LTX-2.3")
            self.assertEqual(gate.out_features, self._EXPECTED_HEADS)


class TestLTX23VideoConnector(_ConnectorMixin, unittest.TestCase):
    """Video connector: 32 x 128 = 4096, 8 layers, gated, 128 registers."""

    _EXPECTED_HEADS = 32
    _EXPECTED_HEAD_DIM = 128

    def _build(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23VideoConnectorConfigurator,
        )

        # Empty config -> exercises the V2 defaults baked into the configurator.
        return LTX23VideoConnectorConfigurator.from_config({})


class TestLTX23AudioConnector(_ConnectorMixin, unittest.TestCase):
    """Audio connector: 32 x 64 = 2048, 8 layers, gated, 128 registers."""

    _EXPECTED_HEADS = 32
    _EXPECTED_HEAD_DIM = 64

    def _build(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23AudioConnectorConfigurator,
        )

        return LTX23AudioConnectorConfigurator.from_config({})


class TestLTX23ConnectorConfigOverrides(unittest.TestCase):
    """The configurators honor checkpoint config keys (not just the defaults)."""

    def _video(self, config):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23VideoConnectorConfigurator,
        )

        return LTX23VideoConnectorConfigurator.from_config(config)

    def test_reads_transformer_subdict(self):
        """Keys are read from the ``transformer`` sub-dict when present."""
        conn = self._video(
            {
                "transformer": {
                    "connector_num_attention_heads": 16,
                    "connector_attention_head_dim": 64,
                    "connector_num_layers": 3,
                    "connector_num_learnable_registers": 64,
                }
            }
        )
        self.assertEqual(conn.num_attention_heads, 16)
        self.assertEqual(conn.inner_dim, 16 * 64)
        self.assertEqual(len(conn.transformer_1d_blocks), 3)
        self.assertEqual(conn.num_learnable_registers, 64)

    def test_gating_can_be_disabled_via_config(self):
        conn = self._video({"transformer": {"connector_apply_gated_attention": False}})
        for block in conn.transformer_1d_blocks:
            self.assertIsNone(block.attn1.to_gate_logits)


class TestLTX23AdaLNSlots(unittest.TestCase):
    """9-slot per-block AdaLN split: MSA [0:3], MLP [3:6], text-cross-attn [6:9].

    This is *the* defining LTX-2.3 transformer change: the per-block scale/shift
    table grows from LTX-2's 6 slots to 9 (the extra 3 drive the text
    cross-attention query shift/scale/gate). ``_get_ada_values`` is the pure
    tensor helper that slices those slots out of the ``[9, D]`` table plus the
    embedded timestep. We verify slot counts, shapes, and exact values without a
    GPU / checkpoint / attention forward.
    """

    def _get(self, table, bsz, timestep, indices):
        from tensorrt_llm._torch.visual_gen.models.ltx23.transformer_ltx23 import (
            _get_ada_values,
        )

        return _get_ada_values(table, bsz, timestep, indices)

    def test_each_group_yields_three_modulators(self):
        """MSA / MLP / text-cross-attn each unpack to 3 tensors of [B, T, D]."""
        batch, seq, dim, num_slots = 2, 5, 4, 9
        table = torch.randn(num_slots, dim)
        timestep = torch.randn(batch, seq, num_slots * dim)

        for indices in (slice(0, 3), slice(3, 6), slice(6, 9)):
            vals = self._get(table, batch, timestep, indices)
            self.assertEqual(len(vals), 3)
            for v in vals:
                self.assertEqual(v.shape, (batch, seq, dim))

    def test_text_cross_attn_slots_are_table_plus_timestep(self):
        """Slots [6:9] = table row + matching timestep chunk (shift, scale, gate)."""
        batch, seq, dim, num_slots = 1, 2, 3, 9
        table = torch.randn(num_slots, dim)
        timestep = torch.randn(batch, seq, num_slots * dim)
        ts = timestep.reshape(batch, seq, num_slots, dim)

        shift, scale, gate = self._get(table, batch, timestep, slice(6, 9))
        self.assertTrue(torch.allclose(shift, table[6][None, None] + ts[:, :, 6, :]))
        self.assertTrue(torch.allclose(scale, table[7][None, None] + ts[:, :, 7, :]))
        self.assertTrue(torch.allclose(gate, table[8][None, None] + ts[:, :, 8, :]))


class TestLTX23AVCrossAttnSlots(unittest.TestCase):
    """5-slot audio<->video cross-attention table: 4 scale/shift + 1 gate.

    Reused byte-for-byte from LTX-2. ``_get_av_ca_ada_values`` splits the
    ``[5, D]`` table into (scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate),
    with the first 4 driven by the scale/shift timestep and the last by the gate
    timestep.
    """

    def _get(self, table, bsz, ss_ts, gate_ts):
        from tensorrt_llm._torch.visual_gen.models.ltx23.transformer_ltx23 import (
            _get_av_ca_ada_values,
        )

        return _get_av_ca_ada_values(table, bsz, ss_ts, gate_ts)

    def test_five_outputs_correct_shapes(self):
        batch, seq, dim = 2, 4, 6
        table = torch.randn(5, dim)
        ss_ts = torch.randn(batch, seq, 4 * dim)
        gate_ts = torch.randn(batch, seq, 1 * dim)

        vals = self._get(table, batch, ss_ts, gate_ts)
        self.assertEqual(len(vals), 5)
        for v in vals:
            self.assertEqual(v.shape, (batch, seq, dim))

    def test_gate_uses_gate_timestep_not_scale_shift(self):
        """The 5th output tracks the gate row + gate timestep specifically."""
        batch, seq, dim = 1, 2, 3
        table = torch.randn(5, dim)
        ss_ts = torch.randn(batch, seq, 4 * dim)
        gate_ts = torch.randn(batch, seq, 1 * dim)

        *_, gate = self._get(table, batch, ss_ts, gate_ts)
        expected_gate = table[4][None, None] + gate_ts.reshape(batch, seq, 1, dim)[:, :, 0, :]
        self.assertTrue(torch.allclose(gate, expected_gate))


# ---------------------------------------------------------------------------
# Transformer *model* structure (CUDA-gated).
#
# The AdaLN-helper tests above check the slot-slicing math in isolation. These
# check that a fully-built ``LTX23Model`` is actually wired that way:
#   * main AdaLN embedding_coefficient 6 -> 9 (adds text-cross-attn slots [6:9])
#   * a NEW ``prompt_adaln_single`` (coeff 2) for sigma-driven text-context K/V
#     modulation (absent entirely in LTX-2)
#   * per-block ``scale_shift_table [9, dim]`` + ``prompt_scale_shift_table [2, dim]``
#   * ``caption_projection`` bypassed to ``nn.Identity`` (projection now happens
#     in the split feature extractor, before the connector)
#   * audio<->video cross-attn tables ([5, dim]) unchanged from LTX-2
#
# Building the model instantiates the fused ``LTX2Attention`` modules, so (like
# LTX-2's own transformer tests) these are CUDA-gated. Reduced dims mirror
# ``test_ltx2_transformer.py`` for a fast build with random weights.
# ---------------------------------------------------------------------------

_V2_VIDEO_ONLY_CONFIG = dict(
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

_V2_AUDIO_VIDEO_CONFIG = dict(
    **_V2_VIDEO_ONLY_CONFIG,
    audio_num_attention_heads=4,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=64,
    audio_positional_embedding_max_pos=[4],
    av_ca_timestep_scale_multiplier=1,
)


def _v2_model_config(backend: str = "VANILLA"):
    """Minimal DiffusionModelConfig, matching test_ltx2_transformer.py."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import AttentionConfig

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _build_ltx23_model(model_type_name: str, config: dict, device: str):
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx23.transformer_ltx23 import LTX23Model

    return LTX23Model(
        model_type=getattr(LTXModelType, model_type_name),
        model_config=_v2_model_config(),
        **config,
    ).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23VideoOnlyModelStructure(unittest.TestCase):
    """VideoOnly LTX-2.3 transformer: 9-slot AdaLN + prompt AdaLN + Identity caption proj."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _build(self):
        return _build_ltx23_model("VideoOnly", _V2_VIDEO_ONLY_CONFIG, self.DEVICE)

    def test_main_adaln_has_9_slots(self):
        """adaln_single emits 9*dim (vs LTX-2's 6*dim): MSA + MLP + text-cross-attn."""
        import torch.nn as nn

        model = self._build()
        d = model.inner_dim
        self.assertEqual(model.adaln_single.linear.out_features, 9 * d)
        # caption projection is bypassed in LTX-2.3 (done before the connector).
        self.assertIsInstance(model.caption_projection, nn.Identity)

    def test_prompt_adaln_single_exists(self):
        """A second AdaLN (coeff 2) for sigma-driven text-context K/V; absent in LTX-2."""
        model = self._build()
        d = model.inner_dim
        self.assertTrue(hasattr(model, "prompt_adaln_single"))
        self.assertEqual(model.prompt_adaln_single.linear.out_features, 2 * d)

    def test_block_scale_shift_tables(self):
        """Per-block main table is [9, dim] and prompt table is [2, dim]."""
        model = self._build()
        d = model.inner_dim
        block = model.transformer_blocks[0]
        self.assertEqual(tuple(block.scale_shift_table.shape), (9, d))
        self.assertEqual(tuple(block.prompt_scale_shift_table.shape), (2, d))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23AudioVideoModelStructure(unittest.TestCase):
    """AudioVideo LTX-2.3 transformer: audio mirrors video; AV cross-attn unchanged."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _build(self):
        return _build_ltx23_model("AudioVideo", _V2_AUDIO_VIDEO_CONFIG, self.DEVICE)

    def test_audio_adaln_has_9_slots_and_prompt(self):
        model = self._build()
        da = model.audio_inner_dim
        self.assertEqual(model.audio_adaln_single.linear.out_features, 9 * da)
        self.assertTrue(hasattr(model, "audio_prompt_adaln_single"))
        self.assertEqual(model.audio_prompt_adaln_single.linear.out_features, 2 * da)

    def test_audio_block_scale_shift_tables(self):
        model = self._build()
        da = model.audio_inner_dim
        block = model.transformer_blocks[0]
        self.assertEqual(tuple(block.audio_scale_shift_table.shape), (9, da))
        self.assertEqual(tuple(block.audio_prompt_scale_shift_table.shape), (2, da))

    def test_av_cross_attention_tables_unchanged(self):
        """Audio<->video cross-attn tables stay [5, dim] (byte-for-byte reuse of LTX-2)."""
        model = self._build()
        block = model.transformer_blocks[0]
        self.assertEqual(
            tuple(block.scale_shift_table_a2v_ca_video.shape), (5, model.inner_dim)
        )
        self.assertEqual(
            tuple(block.scale_shift_table_a2v_ca_audio.shape), (5, model.audio_inner_dim)
        )


# ---------------------------------------------------------------------------
# Transformer forward behavior (CUDA-gated).
#
# The structural classes above only inspect the *built* module. These run a
# real forward on a tiny, random-weight LTX23Model, mirroring LTX-2's
# ``test_*_forward_sanity`` / ``test_video_only_input_to_audio_video_model`` /
# ``TestLTX2TextCache`` — but with the LTX-2.3-specific ``sigma`` field and the
# *inverted* text-cache contract (2.3 K/V is step-varying, so the cache must NOT
# carry a static per-block K/V). This is the first coverage of the whole
# ``LTX23Model.forward`` path (prompt_adaln_single -> _compute_prompt_timestep ->
# 9-slot AdaLN block forward -> prompt-modulated text K/V -> (vx, ax)).
#
# Context is ``cross_attention_dim``-wide (not ``caption_channels``) because
# LTX-2.3's ``caption_projection`` is Identity (projection happens in the split
# feature extractor before the connector).
# ---------------------------------------------------------------------------


def _init_all_weights(model: torch.nn.Module, std: float = 0.02) -> None:
    """Fill random weights (TRT-LLM Linear uses torch.empty(); avoids NaN).

    Mirrors ``test_ltx2_transformer._init_all_weights``: norm weights -> 1.0,
    everything else -> small normal noise.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=std)


def _v2_video_positions(batch, n_frames, grid_h, grid_w, device):
    n = n_frames * grid_h * grid_w
    pos = torch.zeros(batch, 3, n, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                pos[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                pos[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                pos[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1
    return pos


def _v2_audio_positions(batch, a_patches, device):
    pos = torch.zeros(batch, 1, a_patches, 2, device=device)
    for i in range(a_patches):
        pos[:, 0, i, :] = torch.tensor([i, i + 1], dtype=torch.float32)
    return pos


def _build_and_init(model_type_name, config, device, dtype=torch.bfloat16):
    model = _build_ltx23_model(model_type_name, config, device).to(dtype).eval()
    _init_all_weights(model)
    return model


def _v2_video_modality(cfg, device, dtype, *, sigma=0.5, n_frames=1, grid_h=4, grid_w=4, text_len=8):
    """(LTX23Modality, context, positions) for the video stream at reduced dims."""
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    n = n_frames * grid_h * grid_w
    ctx = torch.randn(1, text_len, cfg["cross_attention_dim"], device=device, dtype=dtype) * 0.02
    pos = _v2_video_positions(1, n_frames, grid_h, grid_w, device)
    mod = LTX23Modality(
        latent=torch.randn(1, n, cfg["in_channels"], device=device, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        sigma=torch.tensor([sigma], device=device),
        positions=pos,
        context=ctx,
    )
    return mod, ctx, pos


def _v2_audio_modality(cfg, device, dtype, *, sigma=0.5, a_patches=8, text_len=8):
    """(LTX23Modality, context, positions) for the audio stream at reduced dims."""
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    ctx = torch.randn(
        1, text_len, cfg["audio_cross_attention_dim"], device=device, dtype=dtype
    ) * 0.02
    pos = _v2_audio_positions(1, a_patches, device)
    mod = LTX23Modality(
        latent=torch.randn(1, a_patches, cfg["audio_in_channels"], device=device, dtype=dtype)
        * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        sigma=torch.tensor([sigma], device=device),
        positions=pos,
        context=ctx,
    )
    return mod, ctx, pos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23VideoOnlyForward(unittest.TestCase):
    """VideoOnly forward: (vx, None) with the expected shape and finite values."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_sanity(self):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        cfg = _V2_VIDEO_ONLY_CONFIG
        model = _build_and_init("VideoOnly", cfg, self.DEVICE, dtype)

        video, v_ctx, v_pos = _v2_video_modality(cfg, self.DEVICE, dtype)
        text_cache = model.prepare_text_cache(
            video_context=v_ctx, video_positions=v_pos, dtype=dtype
        )

        with torch.no_grad():
            vout, aout = model(video=video, audio=None, text_cache=text_cache)

        self.assertIsNotNone(vout)
        self.assertIsNone(aout)
        self.assertEqual(vout.shape, (1, video.latent.shape[1], cfg["out_channels"]))
        self.assertTrue(torch.isfinite(vout).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23AudioVideoForward(unittest.TestCase):
    """AudioVideo forward: both streams emit finite outputs of the right shape."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_sanity(self):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        cfg = _V2_AUDIO_VIDEO_CONFIG
        model = _build_and_init("AudioVideo", cfg, self.DEVICE, dtype)

        video, v_ctx, v_pos = _v2_video_modality(cfg, self.DEVICE, dtype)
        audio, a_ctx, a_pos = _v2_audio_modality(cfg, self.DEVICE, dtype)
        text_cache = model.prepare_text_cache(
            video_context=v_ctx,
            video_positions=v_pos,
            audio_context=a_ctx,
            audio_positions=a_pos,
            dtype=dtype,
        )

        with torch.no_grad():
            vout, aout = model(video=video, audio=audio, text_cache=text_cache)

        self.assertIsNotNone(vout)
        self.assertIsNotNone(aout)
        self.assertEqual(vout.shape, (1, video.latent.shape[1], cfg["out_channels"]))
        self.assertEqual(aout.shape, (1, audio.latent.shape[1], cfg["audio_out_channels"]))
        self.assertTrue(torch.isfinite(vout).all())
        self.assertTrue(torch.isfinite(aout).all())

    def test_video_only_input_to_audio_video_model(self):
        """AudioVideo model with ``audio=None`` returns (vx, None) cleanly.

        The pipeline always builds an AudioVideo transformer, and the block
        forward has live ``if audio is not None`` branches, so this exercises
        the video-only input path through the dual-stream model.
        """
        torch.manual_seed(0)
        dtype = torch.bfloat16
        cfg = _V2_AUDIO_VIDEO_CONFIG
        model = _build_and_init("AudioVideo", cfg, self.DEVICE, dtype)

        video, v_ctx, v_pos = _v2_video_modality(cfg, self.DEVICE, dtype)
        text_cache = model.prepare_text_cache(
            video_context=v_ctx, video_positions=v_pos, dtype=dtype
        )

        with torch.no_grad():
            vout, aout = model(video=video, audio=None, text_cache=text_cache)

        self.assertIsNotNone(vout)
        self.assertIsNone(aout)
        self.assertEqual(vout.shape, (1, video.latent.shape[1], cfg["out_channels"]))
        self.assertTrue(torch.isfinite(vout).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLTX23SigmaTextConditioning(unittest.TestCase):
    """The defining LTX-2.3 change: sigma-driven, step-varying text K/V.

    LTX-2 caches a *static* per-block text K/V (its ``TestLTX2TextCache`` asserts
    reusing the cache is correct). LTX-2.3 inverts this: text K/V is modulated by
    a sigma-derived ``prompt_timestep`` and is re-projected every step, so
    ``prepare_text_cache`` must NOT carry a static K/V, and changing only
    ``sigma`` (holding ``timesteps`` fixed) must change the output.
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def test_text_cache_has_no_static_kv(self):
        """Guards against reintroducing LTX-2-style static per-block text K/V."""
        dtype = torch.bfloat16
        cfg = _V2_AUDIO_VIDEO_CONFIG
        model = _build_and_init("AudioVideo", cfg, self.DEVICE, dtype)

        _, v_ctx, v_pos = _v2_video_modality(cfg, self.DEVICE, dtype)
        _, a_ctx, a_pos = _v2_audio_modality(cfg, self.DEVICE, dtype)
        text_cache = model.prepare_text_cache(
            video_context=v_ctx,
            video_positions=v_pos,
            audio_context=a_ctx,
            audio_positions=a_pos,
            dtype=dtype,
        )

        for attr in ("video_kv", "audio_kv", "kv", "video_context_kv", "audio_context_kv"):
            self.assertFalse(
                hasattr(text_cache, attr),
                f"LTX-2.3 text cache must not carry static K/V ({attr})",
            )

    def test_sigma_changes_output(self):
        """Same inputs, different global sigma -> different output.

        Isolates the prompt path: ``timesteps`` are held at 0.5 while ``sigma``
        varies (0.1 vs 0.9). Any output change must come from
        ``prompt_adaln_single`` modulating the text-context K/V.
        """
        torch.manual_seed(7)
        dtype = torch.bfloat16
        cfg = _V2_AUDIO_VIDEO_CONFIG
        model = _build_and_init("AudioVideo", cfg, self.DEVICE, dtype)

        # Build one input set, then clone it with only sigma changed.
        video_lo, v_ctx, v_pos = _v2_video_modality(cfg, self.DEVICE, dtype, sigma=0.1)
        audio_lo, a_ctx, a_pos = _v2_audio_modality(cfg, self.DEVICE, dtype, sigma=0.1)
        text_cache = model.prepare_text_cache(
            video_context=v_ctx,
            video_positions=v_pos,
            audio_context=a_ctx,
            audio_positions=a_pos,
            dtype=dtype,
        )

        from dataclasses import replace

        video_hi = replace(video_lo, sigma=torch.tensor([0.9], device=self.DEVICE))
        audio_hi = replace(audio_lo, sigma=torch.tensor([0.9], device=self.DEVICE))

        with torch.no_grad():
            vout_lo, aout_lo = model(video=video_lo, audio=audio_lo, text_cache=text_cache)
            vout_hi, aout_hi = model(video=video_hi, audio=audio_hi, text_cache=text_cache)

        v_diff = (vout_lo.float() - vout_hi.float()).abs().max().item()
        a_diff = (aout_lo.float() - aout_hi.float()).abs().max().item()
        self.assertGreater(v_diff, 1e-4, "video output should depend on sigma (prompt K/V path)")
        self.assertGreater(a_diff, 1e-4, "audio output should depend on sigma (prompt K/V path)")


# ---------------------------------------------------------------------------
# Video VAE decoder channel recipe (CPU, structural).
#
# LTX-2.3's decoder differs from LTX-2 in that compress_time / compress_space
# ALSO reduce channels by their `multiplier` (LTX-2 only did so for
# compress_all). That makes conv_in = latent_channels * product(all compress
# multipliers) instead of just the compress_all ones. Building with LTX-2's
# recipe gave conv_in = 128*2 = 256; the checkpoint needs 128*8 = 1024.
#
# We validate with a reduced latent width (16) and the real block recipe, so the
# per-block conv widths are cheap to allocate but exercise the exact logic.
# ---------------------------------------------------------------------------

# The real LTX-2.3 checkpoint decoder recipe (stored order).
_LTX23_DECODER_BLOCKS = [
    ["res_x", {"num_layers": 4}],
    ["compress_space", {"multiplier": 2}],
    ["res_x", {"num_layers": 6}],
    ["compress_time", {"multiplier": 2}],
    ["res_x", {"num_layers": 4}],
    ["compress_all", {"multiplier": 1}],
    ["res_x", {"num_layers": 2}],
    ["compress_all", {"multiplier": 2}],
    ["res_x", {"num_layers": 2}],
]


class TestLTX23VideoDecoderChannels(unittest.TestCase):
    """conv_in width + compress-block conv widths match the LTX-2.3 checkpoint.

    Uses latent_channels=16 (real is 128); every channel is 1/8 of the real
    checkpoint, so the *ratios* and block behavior are identical while staying
    cheap to allocate on CPU.
    """

    _LATENT = 16
    _PATCH = 4
    _OUT = 3

    def _build(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.video_vae_ltx23 import (
            LTX23VideoDecoderConfigurator,
        )

        config = {
            "vae": {
                "dims": 3,
                "latent_channels": self._LATENT,
                "out_channels": self._OUT,
                "patch_size": self._PATCH,
                "norm_layer": "pixel_norm",
                "causal_decoder": False,
                "timestep_conditioning": False,
                "spatial_padding_mode": "reflect",
                "decoder_blocks": _LTX23_DECODER_BLOCKS,
            }
        }
        return LTX23VideoDecoderConfigurator.from_config(config)

    def test_conv_in_uses_all_compress_multipliers(self):
        """conv_in: latent -> latent * (2*1*2*2) = latent * 8."""
        dec = self._build()
        w = dec.conv_in.conv.weight
        self.assertEqual(w.shape[1], self._LATENT)  # in
        self.assertEqual(w.shape[0], self._LATENT * 8)  # out (16 -> 128)

    def test_compress_time_and_space_reduce_channels(self):
        """compress_time (up5) and compress_space (up7) halve channels.

        This is the exact LTX-2 vs LTX-2.3 difference: in LTX-2 these conv
        widths would be 2x/4x larger (channels unchanged), mismatching the
        checkpoint. Conv output = in * prod(stride) // multiplier.
        """
        dec = self._build()
        # Flow at latent=16: conv_in 128; up1 compress_all/2 -> 64;
        # up5 compress_time in=64 -> conv out = 64*2//2 = 64;
        # up7 compress_space in=32 -> conv out = 32*4//2 = 64.
        self.assertEqual(dec.up_blocks[5].conv.conv.weight.shape[0], 64)  # compress_time
        self.assertEqual(dec.up_blocks[7].conv.conv.weight.shape[0], 64)  # compress_space

    def test_compress_all_conv_width(self):
        """compress_all (up1, multiplier 2): conv out = in * 8 // 2 = 512 at latent=16."""
        dec = self._build()
        self.assertEqual(dec.up_blocks[1].conv.conv.weight.shape[0], self._LATENT * 32)  # 512

    def test_conv_out_channels(self):
        """Final conv emits out_channels * patch_size**2 (unpatchified later)."""
        dec = self._build()
        self.assertEqual(dec.conv_out.conv.weight.shape[0], self._OUT * self._PATCH**2)  # 48

    def test_res_blocks_preserve_channels(self):
        """res_x blocks (UNetMidBlock3D) keep channels; up0 stays at conv_in width."""
        dec = self._build()
        rb = dec.up_blocks[0].res_blocks[0]
        self.assertEqual(rb.conv1.conv.weight.shape[0], self._LATENT * 8)
        self.assertEqual(rb.conv1.conv.weight.shape[1], self._LATENT * 8)


class TestLTX23FeatureExtractorRescale(unittest.TestCase):
    """The split extractor applies ltx-core's modality-specific ``_rescale_norm``.

    LTX-2.3 scales the per-token-RMS'd features by ``sqrt(out_dim / embedding_dim)``
    *before* each projection (embedding_dim = caption_channels). Video (~1.03x) is
    nearly a no-op, but audio (~0.73x at the real dims) is a 27% factor, so a
    plain ``Linear(x)`` (no rescale) is numerically wrong for audio. This is the
    exact regression a reviewer caught; assert the factor is actually applied.
    """

    def test_rescale_matches_manual_and_differs_from_unscaled(self):
        import math

        import torch.nn.functional as F

        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23GemmaFeaturesExtractor,
        )

        # Dims chosen so both scales are non-trivial and != 1:
        #   v_scale = sqrt(32/8) = 2.0 ,  a_scale = sqrt(2/8) = 0.5
        caption, n_states, vdim, adim = 8, 3, 32, 2
        fe = LTX23GemmaFeaturesExtractor(
            caption_channels=caption,
            video_dim=vdim,
            audio_dim=adim,
            num_hidden_states=n_states,
        ).eval()

        x = torch.randn(2, 4, caption * n_states)
        with torch.no_grad():
            video, audio = fe(x)

        v_scale = math.sqrt(vdim / caption)
        a_scale = math.sqrt(adim / caption)
        exp_v = F.linear(x * v_scale, fe.video_aggregate_embed.weight, fe.video_aggregate_embed.bias)
        exp_a = F.linear(x * a_scale, fe.audio_aggregate_embed.weight, fe.audio_aggregate_embed.bias)
        self.assertTrue(torch.allclose(video, exp_v, atol=1e-5))
        self.assertTrue(torch.allclose(audio, exp_a, atol=1e-5))

        # A no-rescale implementation would equal Linear(x); ensure we differ.
        unscaled_v = F.linear(x, fe.video_aggregate_embed.weight, fe.video_aggregate_embed.bias)
        unscaled_a = F.linear(x, fe.audio_aggregate_embed.weight, fe.audio_aggregate_embed.bias)
        self.assertFalse(torch.allclose(video, unscaled_v, atol=1e-4))
        self.assertFalse(torch.allclose(audio, unscaled_a, atol=1e-4))


class TestLTX23ConnectorRopeDefault(unittest.TestCase):
    """Both connectors default to SPLIT RoPE (ltx-core default), not INTERLEAVED.

    The LTX-2.3 checkpoint sets ``rope_type=split`` explicitly, so this default
    has no runtime effect today; it guards against a future/partial config
    silently falling back to LTX-2's INTERLEAVED default.
    """

    def test_default_rope_type_is_split(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23AudioConnectorConfigurator,
            LTX23VideoConnectorConfigurator,
        )

        for configurator in (
            LTX23VideoConnectorConfigurator,
            LTX23AudioConnectorConfigurator,
        ):
            conn = configurator.from_config({})
            self.assertEqual(conn.rope_type, LTXRopeType.SPLIT)

    def test_explicit_config_overrides_default(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.connector import (
            LTX23VideoConnectorConfigurator,
        )

        conn = LTX23VideoConnectorConfigurator.from_config(
            {"transformer": {"rope_type": "interleaved"}}
        )
        self.assertEqual(conn.rope_type, LTXRopeType.INTERLEAVED)


class TestLTX23Vocoder(unittest.TestCase):
    """BigVGAN-v2 (AMP1) generator + the VocoderWithBWE 48 kHz wiring.

    LTX-2.3 replaces LTX-2's HiFi-GAN (24 kHz) with a BigVGAN AMP1 generator and
    a bandwidth-extension wrapper to 48 kHz. We check (1) a small AMP1 generator
    produces the expected upsampled stereo waveform and reports its sample rate,
    and (2) the configurator assembles a ``VocoderWithBWE`` at 48 kHz from a
    2.3-style nested config (structural; no heavy BWE forward).
    """

    def _small_vocoder(self, output_sampling_rate=16000):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.vocoder_ltx23 import (
            Vocoder,
        )

        return Vocoder(
            resblock_kernel_sizes=[3],
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_initial_channel=32,
            resblock="AMP1",
            output_sampling_rate=output_sampling_rate,
            activation="snakebeta",
        )

    def test_amp1_generator_forward_shape_and_rate(self):
        voc = self._small_vocoder(output_sampling_rate=16000).eval()
        # Stereo mel input: (B, C=2, time, mel_bins=64) -> conv_pre expects 2*64=128.
        batch, time, mel_bins = 1, 8, 64
        mel = torch.randn(batch, 2, time, mel_bins)
        with torch.no_grad():
            wav = voc(mel)
        # Two stereo channels out; time upsampled by prod(upsample_rates)=4.
        self.assertEqual(wav.shape, (batch, 2, time * 4))
        self.assertEqual(voc.output_sampling_rate, 16000)

    def test_bwe_configurator_builds_48khz(self):
        from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.vocoder_ltx23 import (
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
        config = {
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
        voc = LTX23VocoderConfigurator.from_config(config)
        self.assertIsInstance(voc, VocoderWithBWE)
        self.assertEqual(voc.output_sampling_rate, 48000)
        # Inner BigVGAN generator runs at the 16 kHz base rate...
        self.assertEqual(voc.vocoder.output_sampling_rate, 16000)
        # ...and the BWE skip resampler bridges 16 kHz -> 48 kHz (ratio 3).
        self.assertEqual(voc.resampler.ratio, 3)


class TestLTX23PipelineDetection(unittest.TestCase):
    """Registry dispatch: LTX-2.3 text-projection config -> ``LTX23Pipeline``.

    ``_detect_native_ltx_pipeline`` is injected into ``pipeline_registry`` by
    ``apply_patches.py``; skip if the package has not been patched (e.g. running
    against a vanilla install outside the container).
    """

    def _detect(self):
        pytest.importorskip("tensorrt_llm._torch.visual_gen.pipeline_registry")
        from tensorrt_llm._torch.visual_gen import pipeline_registry

        fn = getattr(pipeline_registry, "_detect_native_ltx_pipeline", None)
        if fn is None:
            self.skipTest("registry not patched (apply_patches.py not run)")
        return fn

    _V2_TRANSFORMER = {
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
        "caption_proj_input_norm": False,
        "cross_attention_adaln": True,
    }

    def test_v2_config_detected_as_ltx23(self):
        detect = self._detect()
        self.assertEqual(detect({"transformer": dict(self._V2_TRANSFORMER)}), "LTX23Pipeline")

    def test_plain_config_detected_as_ltx2(self):
        detect = self._detect()
        self.assertEqual(detect({"transformer": {}}), "LTX2Pipeline")

    def test_ambiguous_config_raises(self):
        detect = self._detect()
        # cross_attention_adaln=True but none of the V2 text-projection keys.
        with self.assertRaises(ValueError):
            detect({"transformer": {"cross_attention_adaln": True}})


if __name__ == "__main__":
    unittest.main()
