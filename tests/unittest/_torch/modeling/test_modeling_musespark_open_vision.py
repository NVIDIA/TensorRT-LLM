# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU tests for the MuseSparkOpen NaViT perception tower and its config parsing.

No GPU and no checkpoint. The attention kernel and its metadata are the only
things stubbed; everything else -- config parsing, the window schedule and its
permutation, the 2D RoPE table, the patch layout, the pixel shuffle, every
projection and norm -- is the shipping code.
"""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.configs.musespark_open import (
    MuseSparkOpenConfig,
    MuseSparkOpenVisionConfig,
    layer_uses_full_vision_attention,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.musespark_open_vision import (
    CKPT_TO_MODULE,
    PATCH_VECTORIZATION,
    VISION_PROJECTION_CKPT_KEY,
    MuseSparkOpenNaViTVisionTower,
    MuseSparkOpenVisionAttention,
    _pixel_shuffle_permutation,
    _window_geometry,
    apply_rope_2d,
    precompute_rope_freqs_cis,
    vision_attn_backend,
    vision_rows_for_grid,
)
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear

pytestmark = pytest.mark.cpu_only

# The shipped checkpoint's vision_config, verbatim: knobs split over two levels,
# three of them duplicated, projector_input_dim only at the outer level.
SHIPPED_VISION_CONFIG = {
    "downsample_factor": 2,
    "patch_size": 14,
    "patch_temporal": 2,
    "projector_input_dim": 4096,
    "navit_config": {
        "adapter_type": "simple_mlp",
        "downsample_factor": 2,
        "downsample_strategy": "ps",
        "grid_sample_mode": "bilinear",
        "hidden_act": "gelu",
        "hidden_size": 1536,
        "image_mean": [0.5, 0.5, 0.5],
        "image_size": 448,
        "image_std": [0.5, 0.5, 0.5],
        "intermediate_size": 8960,
        "llm_hidden_size": None,
        "max_num_image_tokens": 4096,
        "max_num_video_tokens": 4096,
        "min_num_image_tokens": 1,
        "mlp_ratio": 4.0,
        "norm_eps": 1e-05,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 50,
        "patch_size": 14,
        "patch_temporal": 2,
        "rope_theta": 10000,
        "sparse_attention_factor": 4,
        "use_cls_token": False,
        "use_rope": True,
        "use_rowwise_vision_projection": True,
        "video_downsample_factor": 2,
        "vision_output_dim": 6144,
    },
}

# A tower small enough to build and run on a CPU in a unit test, with the same
# shape relationships as the shipped one: head_dim divisible by 4,
# hidden_size == vision_output_dim // downsample_factor ** 2, a window smaller
# than the test grids so the sparse path actually splits something, and a layer
# count where the (i + 1) % factor schedule and the last-layer rule both fire.
TINY_VISION_CONFIG = {
    "downsample_factor": 2,
    "patch_size": 2,
    "patch_temporal": 2,
    "projector_input_dim": 48,
    "navit_config": {
        "adapter_type": "simple_mlp",
        "downsample_strategy": "ps",
        "grid_sample_mode": "bilinear",
        "hidden_act": "gelu",
        "hidden_size": 32,
        "image_size": 8,
        "intermediate_size": 64,
        "max_num_image_tokens": 64,
        "max_num_video_tokens": 64,
        "min_num_image_tokens": 1,
        "norm_eps": 1e-05,
        "num_attention_heads": 2,
        "num_channels": 3,
        "num_hidden_layers": 6,
        "rope_theta": 10000,
        "sparse_attention_factor": 4,
        "use_cls_token": False,
        "use_rope": True,
        "use_rowwise_vision_projection": True,
        "video_downsample_factor": 2,
        "vision_output_dim": 128,
    },
}

TINY_DECODER_HIDDEN = 96


def _tiny_model_config() -> ModelConfig:
    text_config = MuseSparkOpenConfig(
        vocab_size=128,
        hidden_size=TINY_DECODER_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=24,
        vision_config=TINY_VISION_CONFIG,
        torch_dtype=torch.float32,
    )
    return ModelConfig(pretrained_config=text_config, attn_backend="VANILLA")


# ---------------------------------------------------------------------------
# Attention stub: the one thing a CPU cannot run.
# ---------------------------------------------------------------------------


def _reference_forward_impl(self, *, q, k, v, attn_metadata, **kwargs):
    """Segment-local SDPA driven by the metadata's cu_seqlens."""
    if k is None and v is None:
        q, k, v = self.split_qkv(q)
    heads, head_dim = self.num_heads, self.head_dim
    q = q.view(-1, heads, head_dim).transpose(0, 1)
    k = k.view(-1, heads, head_dim).transpose(0, 1)
    v = v.view(-1, heads, head_dim).transpose(0, 1)
    out = torch.empty_like(q)
    bounds = attn_metadata.cu_q_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        out[:, start:end] = F.scaled_dot_product_attention(
            q[:, start:end], k[:, start:end], v[:, start:end]
        )
    return out.transpose(0, 1).reshape(-1, heads * head_dim)


def _stub_prepare_attn_metadata(self, seq_lens, attn_metadata):
    seq_lens = [int(s) for s in seq_lens]
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), 0, out=cu[1:])
    return SimpleNamespace(cu_q_seqlens=cu, cu_kv_seqlens=cu, seq_lens=seq_lens)


@pytest.fixture
def tiny_tower(monkeypatch):
    monkeypatch.setattr(
        MuseSparkOpenVisionAttention, "forward_impl", _reference_forward_impl, raising=True
    )
    monkeypatch.setattr(
        MuseSparkOpenNaViTVisionTower,
        "prepare_attn_metadata",
        _stub_prepare_attn_metadata,
        raising=True,
    )
    torch.manual_seed(0)
    tower = MuseSparkOpenNaViTVisionTower(_tiny_model_config())
    # Modules allocate with torch.empty and expect a checkpoint; give them
    # finite values so a shape test is not reading uninitialized memory.
    with torch.no_grad():
        for param in tower.parameters():
            param.normal_(0.0, 0.02)
    return tower


# ---------------------------------------------------------------------------
# Reference helpers, written from the spec rather than from the module.
# ---------------------------------------------------------------------------


def _reference_patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`[T, C, H, W]` -> `[grid_h * grid_w, T * C * p * p]`, laid out `[t][c][py][px]`."""
    t, c, h, w = frames.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    return (
        frames.reshape(t, c, grid_h, patch_size, grid_w, patch_size)
        .permute(2, 4, 0, 1, 3, 5)
        .reshape(grid_h * grid_w, t * c * patch_size * patch_size)
    )


def _make_pixel_values(tower, grid_h, grid_w, *, clips=1, seed=0):
    torch.manual_seed(seed)
    rows = []
    for _ in range(clips):
        frames = torch.randn(
            tower.patch_temporal,
            tower.config.num_channels,
            grid_h * tower.patch_size,
            grid_w * tower.patch_size,
        )
        rows.append(_reference_patchify(frames, tower.patch_size))
    return torch.cat(rows)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestVisionConfigParsing:
    def test_every_contract_field(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert cfg.hidden_size == 1536
        assert cfg.num_hidden_layers == 50
        assert cfg.num_attention_heads == 16
        assert cfg.intermediate_size == 8960
        assert cfg.norm_eps == 1e-5
        assert cfg.patch_size == 14
        assert cfg.image_size == 448
        assert cfg.patch_temporal == 2
        assert cfg.downsample_factor == 2
        assert cfg.video_downsample_factor == 2
        assert cfg.sparse_attention_factor == 4
        assert cfg.rope_theta == 10000.0
        assert isinstance(cfg.rope_theta, float)
        assert cfg.vision_output_dim == 6144
        assert cfg.projector_input_dim == 4096
        assert cfg.use_rowwise_vision_projection is True
        assert cfg.image_mean == [0.5, 0.5, 0.5]
        assert cfg.image_std == [0.5, 0.5, 0.5]
        assert cfg.max_num_image_tokens == 4096
        assert cfg.max_num_video_tokens == 4096
        assert cfg.min_num_image_tokens == 1
        assert cfg.num_channels == 3
        assert cfg.adapter_type == "simple_mlp"

    def test_head_dim_is_derived_not_read(self):
        # There is no head_dim key anywhere in vision_config.
        assert "head_dim" not in SHIPPED_VISION_CONFIG["navit_config"]
        assert "head_dim" not in SHIPPED_VISION_CONFIG
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert cfg.head_dim == 96
        # 2D RoPE spends head_dim // 4 base frequencies per axis.
        assert cfg.head_dim % 4 == 0

    def test_pixel_shuffle_identity_is_asserted(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert cfg.hidden_size == cfg.vision_output_dim // cfg.downsample_factor**2

        # 6145 // 4 still floors to 1536, so the check has to be exact.
        for wrong in (6145, 6148):
            broken = {
                k: dict(v) if isinstance(v, dict) else v for k, v in SHIPPED_VISION_CONFIG.items()
            }
            broken["navit_config"]["vision_output_dim"] = wrong
            with pytest.raises(ValueError, match="vision_output_dim"):
                MuseSparkOpenVisionConfig.from_hf(broken)

    def test_window_side_is_derived_from_image_size(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert cfg.pos_emb_grid_size == 32  # 448 / 14

    def test_duplicated_fields_must_agree(self):
        for field in ("patch_size", "patch_temporal", "downsample_factor"):
            broken = {
                k: dict(v) if isinstance(v, dict) else v for k, v in SHIPPED_VISION_CONFIG.items()
            }
            broken[field] = broken["navit_config"][field] + 1
            with pytest.raises(ValueError, match=field):
                MuseSparkOpenVisionConfig.from_hf(broken)

    def test_outer_only_and_inner_only_fields_both_land(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        # projector_input_dim exists only one level up ...
        assert "projector_input_dim" not in SHIPPED_VISION_CONFIG["navit_config"]
        assert cfg.projector_input_dim == 4096
        # ... and vision_output_dim only inside navit_config.
        assert "vision_output_dim" not in SHIPPED_VISION_CONFIG
        assert cfg.vision_output_dim == 6144

    def test_null_llm_hidden_size_is_dropped(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert getattr(cfg, "llm_hidden_size", None) is None

    def test_sparse_schedule_is_thirteen_of_fifty(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        full = cfg.full_attention_layers
        assert full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49]
        assert len(full) == 13
        assert cfg.num_hidden_layers - len(full) == 37

    def test_schedule_rule_is_i_plus_one(self):
        # Off-by-one guard: layer 0 is windowed, layer 3 is full.
        assert not layer_uses_full_vision_attention(0, 50, 4)
        assert layer_uses_full_vision_attention(3, 50, 4)
        assert layer_uses_full_vision_attention(49, 50, 4)
        assert layer_uses_full_vision_attention(7, 50, 1)  # sparsity disabled

    def test_config_attaches_to_the_text_config(self):
        cfg = MuseSparkOpenConfig(vision_config=SHIPPED_VISION_CONFIG)
        assert isinstance(cfg.vision_config, MuseSparkOpenVisionConfig)
        assert cfg.vision_config.head_dim == 96

    def test_token_ids(self):
        cfg = MuseSparkOpenConfig(vision_config=SHIPPED_VISION_CONFIG)
        assert cfg.image_start_token_id == 200080
        assert cfg.image_end_token_id == 200081
        assert cfg.vid_start_token_id == 200082
        assert cfg.vid_end_token_id == 200083
        assert cfg.vid_frame_separator_token_id == 200087
        assert cfg.image_token_id == 200090
        assert cfg.video_token_id == 200091
        assert cfg.patch_token_id == 200092
        assert cfg.audio_token_id == 200038

    def test_round_trips_through_to_dict(self):
        cfg = MuseSparkOpenConfig(vision_config=SHIPPED_VISION_CONFIG)
        again = MuseSparkOpenConfig(**cfg.to_dict())
        assert again.vision_config.head_dim == 96
        assert again.vision_config.full_attention_layers == cfg.vision_config.full_attention_layers


# ---------------------------------------------------------------------------
# Trap 1: patch vectorization
# ---------------------------------------------------------------------------


class TestPatchVectorization:
    def test_declared_order(self):
        assert PATCH_VECTORIZATION == ("patch_temporal", "num_channels", "patch_y", "patch_x")

    def test_element_mapping_is_t_c_py_px(self):
        patch_size, channels, temporal = 14, 3, 2
        grid_h, grid_w = 2, 3
        # Encode each source position as a unique number so the flattened row
        # can be decoded back and checked slot by slot.
        h, w = grid_h * patch_size, grid_w * patch_size
        idx = torch.arange(temporal * channels * h * w, dtype=torch.float64)
        frames = idx.reshape(temporal, channels, h, w)
        rows = _reference_patchify(frames, patch_size)
        assert rows.shape == (grid_h * grid_w, temporal * channels * patch_size**2)

        for gy, gx, t, c, py, px in ((1, 2, 1, 2, 5, 7), (0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 13, 13)):
            token = gy * grid_w + gx
            slot = ((t * channels + c) * patch_size + py) * patch_size + px
            expected = frames[t, c, gy * patch_size + py, gx * patch_size + px]
            assert rows[token, slot] == expected

    def test_still_image_is_duplicated_not_zero_padded(self):
        patch_size, channels, temporal = 2, 3, 2
        frame = torch.randn(1, channels, 4, 6)
        expanded = frame.expand(temporal, -1, -1, -1)
        rows = _reference_patchify(expanded, patch_size)
        half = channels * patch_size**2
        # The second temporal half repeats the first; a zero-pad would be zeros.
        torch.testing.assert_close(rows[:, half:], rows[:, :half])
        assert rows[:, half:].abs().sum() > 0

    def test_conv1_is_a_linear_of_the_patch_dim(self, tiny_tower):
        assert isinstance(tiny_tower.conv1_linear, Linear)
        assert not isinstance(tiny_tower.conv1_linear, torch.nn.Conv2d)
        assert tiny_tower.patch_dim == 3 * 2 * tiny_tower.patch_size**2
        assert tuple(tiny_tower.conv1_linear.weight.shape) == (
            tiny_tower.hidden_size,
            tiny_tower.patch_dim,
        )
        assert tiny_tower.conv1_linear.bias is None

    def test_shipped_patch_dim_is_1176(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        assert cfg.num_channels * cfg.patch_temporal * cfg.patch_size**2 == 1176

    def test_wrong_patch_dim_is_rejected(self, tiny_tower):
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        bad = torch.randn(16, tiny_tower.patch_dim + 1)
        with pytest.raises(ValueError, match="features per patch"):
            tiny_tower(bad, grid, grid_thw_cpu=grid)


# ---------------------------------------------------------------------------
# Trap 3: 2D RoPE
# ---------------------------------------------------------------------------


class TestRope2D:
    def test_frequency_layout(self):
        dim, theta = 96, 10000.0
        table = precompute_rope_freqs_cis(dim, 5, 7, theta=theta)
        assert table.shape == (5, 7, dim // 2, 2)

        half = dim // 2
        inv = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32)[: half // 2] / half))
        assert inv.numel() == 24  # head_dim // 4 base frequencies per axis

        h, w = 3, 4
        angle = torch.atan2(table[h, w, :, 1], table[h, w, :, 0])
        # Positions are 1-indexed and the width half comes first.
        expected_w = (w + 1) * inv
        expected_h = (h + 1) * inv
        torch.testing.assert_close(
            angle[:24],
            torch.remainder(expected_w + math.pi, 2 * math.pi) - math.pi,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            angle[24:],
            torch.remainder(expected_h + math.pi, 2 * math.pi) - math.pi,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_positions_are_one_indexed(self):
        table = precompute_rope_freqs_cis(8, 2, 2)
        # A 0-indexed table would have cos == 1 / sin == 0 at (0, 0).
        assert not torch.allclose(table[0, 0, :, 1], torch.zeros(4))

    def test_rotation_pairs_are_interleaved(self):
        torch.manual_seed(0)
        seq, heads, dim = 5, 2, 8
        q = torch.randn(seq, heads, dim)
        k = torch.randn(seq, heads, dim)
        freqs = precompute_rope_freqs_cis(dim, 1, seq).reshape(seq, dim // 2, 2)

        q_out, _ = apply_rope_2d(q, k, freqs)

        # Independent complex reference over ADJACENT pairs.
        cos, sin = freqs[..., 0].unsqueeze(1), freqs[..., 1].unsqueeze(1)
        ref = torch.view_as_complex(q.float().reshape(seq, heads, dim // 2, 2)) * torch.polar(
            torch.ones_like(cos), torch.atan2(sin, cos)
        )
        torch.testing.assert_close(q_out, torch.view_as_real(ref).flatten(-2), atol=1e-5, rtol=1e-5)

        # A half-split rotation is a different tensor, so this test can fail.
        q_half = q.reshape(seq, heads, 2, dim // 2)
        half_split = torch.cat(
            [
                q_half[..., 0, :] * cos - q_half[..., 1, :] * sin,
                q_half[..., 0, :] * sin + q_half[..., 1, :] * cos,
            ],
            dim=-1,
        )
        assert not torch.allclose(q_out, half_split, atol=1e-4)

    def test_rotation_is_computed_in_fp32_and_cast_back(self):
        torch.manual_seed(0)
        q = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        freqs = precompute_rope_freqs_cis(8, 2, 2).reshape(4, 4, 2)
        q_out, k_out = apply_rope_2d(q, q, freqs)
        assert q_out.dtype == torch.bfloat16
        assert k_out.dtype == torch.bfloat16

    def test_table_slice_is_row_major(self, tiny_tower):
        grid_h, grid_w = 3, 5
        sliced = tiny_tower._rope_slice(grid_h, grid_w, torch.device("cpu"))
        assert sliced.shape == (grid_h * grid_w, tiny_tower.config.head_dim // 2, 2)
        full = tiny_tower.precomputed_freqs_cis
        # Token (h, w) sits at h * grid_w + w.
        torch.testing.assert_close(sliced[2 * grid_w + 4], full[2, 4])


# ---------------------------------------------------------------------------
# Trap 4: the window schedule and its permutation
# ---------------------------------------------------------------------------


class TestWindowedAttention:
    def test_thirteen_full_thirtyseven_windowed(self):
        cfg = MuseSparkOpenVisionConfig.from_hf(SHIPPED_VISION_CONFIG)
        full = [i for i in range(50) if layer_uses_full_vision_attention(i, 50, 4)]
        windowed = [i for i in range(50) if i not in full]
        assert (len(full), len(windowed)) == (13, 37)
        assert cfg.full_attention_layers == full

    def test_tower_dispatches_each_layer_to_the_right_metadata(self, tiny_tower, monkeypatch):
        seen = []
        original = type(tiny_tower.encoder.layers[0]).forward

        def spy(layer_self, x, attn_metadata, freqs_cis):
            seen.append(tuple(attn_metadata.seq_lens))
            return original(layer_self, x, attn_metadata, freqs_cis)

        monkeypatch.setattr(type(tiny_tower.encoder.layers[0]), "forward", spy, raising=True)
        grid = torch.tensor([[1, 6, 8]], dtype=torch.int32)  # 6 x 8 patches, window 4
        tiny_tower(_make_pixel_values(tiny_tower, 6, 8), grid, grid_thw_cpu=grid)

        full_lens = (48,)  # one clip of 6 * 8 patches
        window_lens = tuple(_window_geometry(6, 8, 4, 4)[1])
        assert len(seen) == 6
        assert [i for i, s in enumerate(seen) if s == full_lens] == [3, 5]
        assert [i for i, s in enumerate(seen) if s == window_lens] == [0, 1, 2, 4]

    @pytest.mark.parametrize("grid_h,grid_w", [(4, 4), (6, 8), (5, 7), (1, 1), (9, 4)])
    def test_permutation_round_trips(self, grid_h, grid_w):
        perm, lengths = _window_geometry(grid_h, grid_w, 4, 4)
        n = grid_h * grid_w
        assert sorted(perm.tolist()) == list(range(n))
        assert sum(lengths) == n

        x = torch.randn(n, 3)
        permuted = x[perm]
        restored = torch.empty_like(permuted)
        restored.index_copy_(0, perm, permuted)
        torch.testing.assert_close(restored, x)

    def test_windows_group_spatial_neighbours(self):
        # A 6x8 grid with a 4x4 window is four windows: 4x4, 4x4, 2x4, 2x4.
        perm, lengths = _window_geometry(6, 8, 4, 4)
        assert lengths == (16, 16, 8, 8)
        first = perm[:16].tolist()
        # Window (0, 0) is rows 0-3 x cols 0-3 of the row-major grid.
        assert first == [r * 8 + c for r in range(4) for c in range(4)]

    def test_padded_rows_are_dropped_not_attended(self):
        perm, lengths = _window_geometry(5, 5, 4, 4)
        assert perm.numel() == 25
        assert lengths == (16, 4, 4, 1)
        assert (perm >= 0).all()

    def test_full_grid_is_one_window(self):
        perm, lengths = _window_geometry(4, 4, 4, 4)
        assert lengths == (16,)
        assert perm.tolist() == list(range(16))


# ---------------------------------------------------------------------------
# The weight mapper's name table
# ---------------------------------------------------------------------------


class TestCheckpointNameMap:
    #: The 808 `vision_model.*` keys collapse to these suffixes.
    CKPT_SUFFIXES = (
        "conv1_linear",
        "ln_pre",
        "ln_post",
        "adapter.c_fc",
        "adapter.c_proj",
        "encoder.layers.{i}.ln_1",
        "encoder.layers.{i}.ln_2",
        "encoder.layers.{i}.attn.q_proj",
        "encoder.layers.{i}.attn.k_proj",
        "encoder.layers.{i}.attn.v_proj",
        "encoder.layers.{i}.attn.out_proj",
        "encoder.layers.{i}.mlp.c_fc",
        "encoder.layers.{i}.mlp.c_proj",
    )

    def test_every_checkpoint_suffix_resolves_to_a_real_submodule(self, tiny_tower):
        for suffix in self.CKPT_SUFFIXES:
            path = suffix.format(i=0)
            for ckpt_tail, module_tail in CKPT_TO_MODULE.items():
                if path.endswith(ckpt_tail):
                    path = path[: -len(ckpt_tail)] + module_tail
                    break
            # Raises AttributeError if the mapper would write into thin air.
            tiny_tower.get_submodule(path)

    def test_capacity_keys_name_both_metadata_objects(self, tiny_tower):
        caps = tiny_tower.get_encoder_attention_metadata_capacity(1024)
        assert set(caps) == {"full_attention", "window_attention"}
        assert all(v > 0 for v in caps.values())

    def test_partial_capacity_override_is_merged_not_required(self, tiny_tower, monkeypatch):
        built = []
        # metadata_cls is set per instance in __init__, so patch the instance.
        monkeypatch.setattr(tiny_tower, "metadata_cls", lambda **kw: built.append(kw))
        tiny_tower.setup_attn_metadata(1024, {"full_attention": 7})
        assert built[0]["max_num_requests"] == 7
        assert built[1]["max_num_requests"] == 1024  # derived, not required
        with pytest.raises(ValueError, match="unknown vision attention-metadata"):
            tiny_tower.setup_attn_metadata(1024, {"attention": 7})

    def test_pos_embed_is_a_bare_parameter(self, tiny_tower):
        assert isinstance(tiny_tower.pos_embed, torch.nn.Parameter)
        assert tuple(tiny_tower.pos_embed.shape) == (
            tiny_tower.config.pos_emb_grid_size**2,
            tiny_tower.hidden_size,
        )

    def test_vision_projection_comes_from_the_modalities_prefix(self, tiny_tower):
        assert VISION_PROJECTION_CKPT_KEY.startswith("modalities.vision_perception.")
        assert not VISION_PROJECTION_CKPT_KEY.startswith("vision_model.")
        tiny_tower.get_submodule("vision_projection")

    def test_qkv_fusion_widths(self, tiny_tower):
        attn = tiny_tower.encoder.layers[0].attn
        hidden = tiny_tower.hidden_size
        # q | k | v concatenated along dim 0 of one column-parallel weight.
        assert tuple(attn.qkv_proj.weight.shape) == (3 * hidden, hidden)
        assert tuple(attn.qkv_proj.bias.shape) == (3 * hidden,)
        assert tuple(attn.o_proj.weight.shape) == (hidden, hidden)

    def test_tensor_count_per_encoder_layer_is_sixteen(self, tiny_tower):
        # 4 projections x (weight + bias) + 2 layernorms x (weight + bias) = 12
        # module tensors; the checkpoint splits qkv into three, so 16 keys.
        per_layer = [
            name
            for name, _ in tiny_tower.encoder.layers[0].named_parameters()
            if not name.endswith("_scale")
        ]
        assert len(per_layer) == 12
        assert len(per_layer) + 4 == 16  # q/k/v weight+bias fuse 6 keys into 2


# ---------------------------------------------------------------------------
# Trap 5: norms and bias placement
# ---------------------------------------------------------------------------


class TestNormsAndBiases:
    def test_layernorms_have_weight_and_bias(self, tiny_tower):
        for norm in (tiny_tower.ln_pre, tiny_tower.ln_post):
            assert isinstance(norm, LayerNorm)
            assert isinstance(norm.weight, torch.nn.Parameter)
            assert isinstance(norm.bias, torch.nn.Parameter)
            assert norm.variance_epsilon == 1e-5
        layer = tiny_tower.encoder.layers[0]
        for norm in (layer.ln_1, layer.ln_2):
            assert isinstance(norm.weight, torch.nn.Parameter)
            assert isinstance(norm.bias, torch.nn.Parameter)
            assert norm.variance_epsilon == 1e-5

    def test_biases_are_present_exactly_where_the_checkpoint_has_them(self, tiny_tower):
        layer = tiny_tower.encoder.layers[0]
        assert layer.attn.qkv_proj.bias is not None
        assert layer.attn.o_proj.bias is not None
        assert layer.mlp.up_proj.bias is not None
        assert layer.mlp.down_proj.bias is not None

        assert tiny_tower.conv1_linear.bias is None
        assert tiny_tower.adapter.c_fc.bias is None
        assert tiny_tower.adapter.c_proj.bias is None
        assert tiny_tower.vision_projection.bias is None

    def test_row_parallel_bias_lands_after_the_reduction(self, tiny_tower):
        # Linear applies a ROW-parallel bias on rank 0 only, before the
        # all-reduce, which sums to exactly one bias on the reduced result.
        # At tp_size 1 there is no reduction and the bias applies directly.
        source = inspect.getsource(type(tiny_tower.encoder.layers[0].attn.o_proj).forward)
        assert "bias = None if (self.tp_rank > 0) else self.bias" in source

    def test_tower_is_unquantized_even_when_the_parent_is_not(self):
        from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

        model_config = _tiny_model_config()
        model_config._frozen = False
        model_config.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
        model_config._frozen = True

        tower = MuseSparkOpenNaViTVisionTower(model_config)
        assert tower.model_config.quant_config is None
        assert tower.model_config.get_quant_config() is None
        for module in tower.modules():
            if isinstance(module, Linear):
                assert not getattr(module, "has_nvfp4", False), module
                assert module.weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# Trap 2 + shape closure
# ---------------------------------------------------------------------------


class TestForward:
    def test_positional_embedding_is_added_before_ln_pre(self, tiny_tower):
        source = inspect.getsource(MuseSparkOpenNaViTVisionTower.forward)
        add = source.index("self._positional_embedding")
        pre = source.index("self.ln_pre")
        assert add < pre

    def test_grid_sample_uses_align_corners_false_and_pixel_centres(self, tiny_tower):
        source = inspect.getsource(MuseSparkOpenNaViTVisionTower._positional_embedding)
        assert "align_corners=False" in source
        coords = tiny_tower._grid_coordinates(4, 4, device=torch.device("cpu"), dtype=torch.float32)
        # linspace(-1 + 1/g, 1 - 1/g, g): centres, not corners.
        assert coords.min() == pytest.approx(-0.75)
        assert coords.max() == pytest.approx(0.75)

    def test_meshgrid_xy_quirk_is_replicated(self, tiny_tower):
        grid_h, grid_w = 2, 3
        coords = tiny_tower._grid_coordinates(
            grid_h, grid_w, device=torch.device("cpu"), dtype=torch.float32
        )
        assert coords.shape == (grid_h * grid_w, 2)
        pos_h = torch.linspace(-1 + 1 / grid_h, 1 - 1 / grid_h, grid_h)
        pos_w = torch.linspace(-1 + 1 / grid_w, 1 - 1 / grid_w, grid_w)
        # indexing="xy" makes the list run width-outer and puts the height
        # coordinate in the slot grid_sample reads as x. Both are deliberate.
        expected = torch.stack(
            [torch.stack([pos_h[j], pos_w[i]]) for i in range(grid_w) for j in range(grid_h)]
        )
        torch.testing.assert_close(coords, expected)

    def test_positional_embedding_output_shape(self, tiny_tower):
        coords = tiny_tower._grid_coordinates(3, 5, device=torch.device("cpu"), dtype=torch.float32)
        emb = tiny_tower._positional_embedding(coords)
        assert emb.shape == (15, tiny_tower.hidden_size)
        assert torch.isfinite(emb).all()

    @pytest.mark.parametrize("grid_h,grid_w", [(4, 4), (6, 8), (2, 2)])
    def test_shape_closure_single_image(self, tiny_tower, grid_h, grid_w):
        pixel_values = _make_pixel_values(tiny_tower, grid_h, grid_w)
        grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int32)
        out = tiny_tower(pixel_values, grid, grid_thw_cpu=grid)
        assert out.shape == (grid_h * grid_w // 4, TINY_DECODER_HIDDEN)
        assert torch.isfinite(out).all()

    def test_shape_closure_matches_the_row_accounting_helper(self, tiny_tower):
        # (clips, grid_h, grid_w) in patch units; the last is a 2-clip video.
        cases = [(1, 4, 4), (1, 6, 8), (2, 4, 6)]
        rows = []
        pixel_values = []
        for t, h, w in cases:
            pixel_values.append(_make_pixel_values(tiny_tower, h, w, clips=t, seed=h + w))
            rows.append(vision_rows_for_grid(t, h, w, tiny_tower.config))
        grid = torch.tensor(cases, dtype=torch.int32)
        out = tiny_tower(torch.cat(pixel_values), grid, grid_thw_cpu=grid)
        assert out.shape == (sum(rows), TINY_DECODER_HIDDEN)
        assert torch.isfinite(out).all()

    def test_mixed_resolutions_pack_without_padding(self, tiny_tower):
        cases = [(1, 4, 4), (1, 2, 6), (1, 8, 4)]
        pixel_values = torch.cat(
            [_make_pixel_values(tiny_tower, h, w, seed=i) for i, (_, h, w) in enumerate(cases)]
        )
        grid = torch.tensor(cases, dtype=torch.int32)
        expected = sum(vision_rows_for_grid(*c, tiny_tower.config) for c in cases)
        out = tiny_tower(pixel_values, grid, grid_thw_cpu=grid)
        assert out.shape == (expected, TINY_DECODER_HIDDEN)

    def test_grid_thw_is_clips_and_patch_units(self, tiny_tower):
        # t counts temporal clips (frames // patch_temporal, so 1
        # for a still image) and h/w are patch counts of the resized canvas, not
        # pixels. Getting this wrong scales the row count by patch_size ** 2.
        cfg = tiny_tower.config
        assert vision_rows_for_grid(1, 4, 4, cfg) == 4
        assert vision_rows_for_grid(3, 4, 4, cfg) == 12
        assert vision_rows_for_grid(1, 2, 6, cfg) == 3
        with pytest.raises(ValueError, match="clip count"):
            vision_rows_for_grid(0, 4, 4, cfg)

    def test_video_clips_each_get_their_own_attention_segment(self, tiny_tower, monkeypatch):
        seen = []
        original = type(tiny_tower.encoder.layers[0]).forward

        def spy(layer_self, x, attn_metadata, freqs_cis):
            seen.append(tuple(attn_metadata.seq_lens))
            return original(layer_self, x, attn_metadata, freqs_cis)

        monkeypatch.setattr(type(tiny_tower.encoder.layers[0]), "forward", spy, raising=True)
        grid = torch.tensor([[3, 4, 4]], dtype=torch.int32)  # a 3-clip video
        out = tiny_tower(_make_pixel_values(tiny_tower, 4, 4, clips=3), grid, grid_thw_cpu=grid)
        assert out.shape == (3 * 4, TINY_DECODER_HIDDEN)
        # Full-attention layers see three 16-patch segments, not one of 48.
        assert seen[3] == (16, 16, 16)

    def test_contract_two_argument_call(self, tiny_tower):
        # The encoder wrapper calls forward(pixel_values, grid_thw); grid_thw_cpu
        # is an opt-in that only removes the device read.
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        with_host = tiny_tower(_make_pixel_values(tiny_tower, 4, 4), grid, grid_thw_cpu=grid)
        without = tiny_tower(_make_pixel_values(tiny_tower, 4, 4), grid)
        torch.testing.assert_close(with_host, without)

    def test_geometry_disagreement_is_loud(self, tiny_tower):
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        pixel_values = _make_pixel_values(tiny_tower, 4, 4)[:-1]
        with pytest.raises(ValueError, match="disagree on geometry"):
            tiny_tower(pixel_values, grid, grid_thw_cpu=grid)

    def test_pixel_shuffle_is_channel_major(self, tiny_tower):
        # `(b r) d -> b (d r)`: feature f of sub-position s lands at f * r^2 + s.
        r2, dim = 4, 3
        x = torch.arange(2 * r2 * dim, dtype=torch.float32).reshape(2 * r2, dim)
        folded = x.reshape(2, r2, dim).transpose(1, 2).reshape(2, dim * r2)
        for b in range(2):
            for f in range(dim):
                for s in range(r2):
                    assert folded[b, f * r2 + s] == x[b * r2 + s, f]

    def test_pixel_shuffle_permutation_groups_2x2_blocks(self):
        perm = _pixel_shuffle_permutation(4, 4, 2)
        assert sorted(perm.tolist()) == list(range(16))
        # First group is the top-left 2x2 block of a row-major 4x4 grid.
        assert perm[:4].tolist() == [0, 1, 4, 5]

    def test_odd_grid_is_rejected(self):
        with pytest.raises(ValueError, match="downsample_factor"):
            _pixel_shuffle_permutation(3, 4, 2)

    def test_adapter_activates_twice(self, tiny_tower):
        torch.manual_seed(1)
        x = torch.randn(4, tiny_tower.config.vision_output_dim)
        out = tiny_tower.adapter(x)
        h = F.gelu(tiny_tower.adapter.c_fc(x))
        torch.testing.assert_close(out, F.gelu(tiny_tower.adapter.c_proj(h)))
        # One GELU only would be a different tensor.
        assert not torch.allclose(out, tiny_tower.adapter.c_proj(h))

    def test_projection_shapes(self, tiny_tower):
        assert tuple(tiny_tower.adapter.c_fc.weight.shape) == (
            tiny_tower.config.projector_input_dim,
            tiny_tower.config.vision_output_dim,
        )
        assert tuple(tiny_tower.adapter.c_proj.weight.shape) == (
            tiny_tower.config.projector_input_dim,
            tiny_tower.config.projector_input_dim,
        )
        assert tuple(tiny_tower.vision_projection.weight.shape) == (
            TINY_DECODER_HIDDEN,
            tiny_tower.config.projector_input_dim,
        )

    def test_output_is_unnormed(self, tiny_tower):
        # The decoder's embed_norm is the single RMSNorm; the tower must not
        # hand back rows that are already unit-RMS.
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        out = tiny_tower(_make_pixel_values(tiny_tower, 4, 4), grid, grid_thw_cpu=grid)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert not torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
        assert "perception_emb_norm" not in inspect.getsource(MuseSparkOpenNaViTVisionTower)


# ---------------------------------------------------------------------------
# Sync discipline
# ---------------------------------------------------------------------------


class TestNoDeviceHostSyncs:
    #: `_host_grid_rows` is the declared host-geometry boundary and `_window_geometry`
    #: only ever sees Python ints and CPU tensors; everything else in the forward
    #: path must be free of device reads.
    BANNED = (".item()", ".cpu()", ".tolist()", "torch.nonzero", "masked_select")

    def test_forward_path_has_no_device_reads(self):
        from tensorrt_llm._torch.models import musespark_open_vision as mod

        members = [
            mod.MuseSparkOpenNaViTVisionTower.forward,
            mod.MuseSparkOpenNaViTVisionTower._positional_embedding,
            mod.MuseSparkOpenNaViTVisionTower._grid_coordinates,
            mod.MuseSparkOpenNaViTVisionTower._rope_slice,
            mod.MuseSparkOpenNaViTVisionTower.prepare_attn_metadata,
            mod.MuseSparkOpenVisionEncoder.forward,
            mod.MuseSparkOpenVisionEncoderLayer.forward,
            mod.MuseSparkOpenVisionAttention.forward,
            mod.MuseSparkOpenVisionAdapter.forward,
            mod.apply_rope_2d,
            mod.precompute_rope_freqs_cis,
        ]
        for member in members:
            source = inspect.getsource(member)
            for banned in self.BANNED:
                assert banned not in source, f"{member.__qualname__} contains {banned}"

    def test_host_geometry_is_confined_to_one_helper(self):
        """`_host_grid_rows` is the only place host geometry is read.

        It reads Python ints out of an already-host tensor, which is why
        `.tolist()` lives here and nowhere else. It must NOT contain
        `.cpu()`: this helper used to fall back to copying a device tensor
        after warning once, so the warning fired on the first forward and every
        later one silently paid the stall. It now refuses a device tensor, and
        the absence of the copy is the property worth pinning -- an earlier
        version of this test required its presence.
        """
        source = inspect.getsource(MuseSparkOpenNaViTVisionTower._host_grid_rows)
        assert ".tolist()" in source
        assert ".cpu()" not in source, "the device-to-host fallback is back"
        assert "raise" in source, "a device tensor must be refused, not copied"


# ---------------------------------------------------------------------------
# Tower attention-backend selection
# ---------------------------------------------------------------------------


class TestVisionAttnBackend:
    """A TRTLLM decoder gets a FLASHINFER tower; everyone else keeps theirs.

    TRTLLM has no fused context-FMHA kernel for the tower's head_dim (96), so
    it falls back to the padded unfused MHA, whose workspace is sized from the
    metadata maxima (`max_num_requests * max_seq_len ** 2`). The window
    metadata declares one request per encoder-budget token, so an 8192-token
    budget requested a 232 GiB workspace during startup encoder profiling and
    the boot OOMed before serving anything. See `vision_attn_backend`.
    """

    def test_trtllm_decoder_gets_flashinfer_tower(self):
        assert vision_attn_backend("TRTLLM") == "FLASHINFER"

    @pytest.mark.parametrize("backend", ["FLASHINFER", "VANILLA"])
    def test_other_decoders_keep_their_backend(self, backend):
        # Negative control: the swap is TRTLLM-specific, not a blanket pin.
        assert vision_attn_backend(backend) == backend

    def test_tower_applies_the_swap_without_mutating_the_caller(self, monkeypatch):
        # Construction is enough: the tower must land on FLASHINFER while the
        # decoder's config still says TRTLLM. Backend classes resolve at
        # Attention.__init__, so stub the resolution to stay CPU-only and
        # independent of whether flashinfer is importable in this environment.
        import tensorrt_llm._torch.attention.attention as attention_mod
        import tensorrt_llm._torch.models.musespark_open_vision as vision_mod
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend

        seen = []
        real_get = get_attention_backend
        real_create = attention_mod.create_attention

        def recording_get(backend_name, *args, **kwargs):
            seen.append(backend_name)
            return real_get("VANILLA")

        def recording_create(backend_name, *args, **kwargs):
            seen.append(backend_name)
            return real_create("VANILLA", *args, **kwargs)

        monkeypatch.setattr(vision_mod, "get_attention_backend", recording_get)
        monkeypatch.setattr(attention_mod, "get_attention_backend", recording_get)
        monkeypatch.setattr(attention_mod, "create_attention", recording_create)

        decoder_config = _tiny_model_config()
        decoder_config.attn_backend = "TRTLLM"
        tower = MuseSparkOpenNaViTVisionTower(decoder_config)

        assert tower.model_config.attn_backend == "FLASHINFER"
        assert decoder_config.attn_backend == "TRTLLM", "caller's config was mutated"
        assert seen, "no attention backend was ever resolved"
        assert set(seen) == {"FLASHINFER"}, (
            f"tower resolved backends {sorted(set(seen))}; every tower-side "
            "resolution must see the swapped backend"
        )
