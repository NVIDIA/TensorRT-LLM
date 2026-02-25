"""Preprocessing unit tests for modeling_nemotron_nano.py."""

import random
from unittest import mock

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    DynamicResolutionImageTiler,
    DynamicResolutionParams,
    NanoV2VLVisionEncoder,
)


def make_tiler(**overrides):
    """Create a DynamicResolutionImageTiler with sensible defaults."""
    defaults = {
        "max_model_len": 131072,
        "patch_size": 16,
        "min_num_patches": 4,
        "max_num_patches": 256,
        "downsample_ratio": 0.5,
        "norm_mean": (0.123, 0.456, 0.789),
        "norm_std": (0.321, 0.654, 0.987),
    }
    defaults.update(overrides)
    return DynamicResolutionImageTiler(**defaults)


def test_tiler_rejects_downsample_ratio_ge_1():
    with pytest.raises(ValueError, match="must be < 1"):
        make_tiler(downsample_ratio=1.0)


def test_tiler_rejects_non_half_reduction():
    with pytest.raises(ValueError, match="Only a reduction factor of 2.0"):
        make_tiler(downsample_ratio=0.25)


def test_tiler_accepts_valid_params():
    tiler = make_tiler(downsample_ratio=0.5)
    assert tiler._downsample_ratio == 2


@pytest.mark.parametrize(
    "img_size, budget, min_patches, expected_ps, expected_emb",
    # The `expected_ps` can be calculated via:
    # 1. `closest_patch_h = round(h / patch_size + 0.5)`. Similar formula for w.
    # 2. `factor = min(sqrt(budget / (closest_patch_h * closest_patch_w)), 1.0)`.
    # 3. `target_h = floor(factor * closest_patch_h)`. Similar formula for w.
    # 4. If `target_h * target_w < min_patches < budget`: scale each dim up by
    #    `sqrt(min_patches / (target_h * target_w))` + ceil.
    # 5. Round to even.
    # Then `expected_emb` can be derived from those 2 target values + budget.
    [
        pytest.param(
            (320, 320),
            1000,
            4,
            (20, 20),
            100,
            id="square_generous",
        ),
        pytest.param(
            (320, 320),
            16,
            4,
            (4, 4),
            4,
            id="tight_budget",
        ),
        pytest.param(
            (200, 200),
            169,
            4,
            (14, 12),
            42,
            id="odd_targets_rounding",
        ),
        # This tests `min_patches=16` is enforced.
        pytest.param(
            (32, 32),
            100,
            16,
            (4, 4),
            4,
            id="min_num_patches_enforced",
        ),
        pytest.param(
            (480, 160),
            100,
            4,
            (18, 4),
            18,
            id="landscape",
        ),
    ],
)
def test_process_media(img_size, budget, min_patches, expected_ps, expected_emb):
    tiler = make_tiler(patch_size=16, min_num_patches=min_patches)
    img = Image.new("RGB", img_size)
    params, token_count = tiler.process_media(img, budget)

    assert params.patch_size == expected_ps
    assert params.num_embeddings == expected_emb
    assert params.num_tiles == 1
    assert params.media is img
    assert token_count == params.patch_size[0] * params.patch_size[1]
    assert params.patch_size[0] % 2 == 0
    assert params.patch_size[1] % 2 == 0


@pytest.mark.parametrize("num_images", [1, 2, 3, 5])
def test_compute_params_multiple_images(num_images):
    rng = random.Random(42)
    tiler = make_tiler(patch_size=16)
    imgs = [Image.new("RGB", (rng.randint(32, 64), rng.randint(32, 64))) for _ in range(num_images)]
    result = tiler.compute_params(imgs, num_tokens_available=1000)
    assert len(result) == num_images
    for params in result:
        assert params.patch_size[0] % 2 == 0
        assert params.patch_size[1] % 2 == 0


def test_compute_params_over_budget_scales_down():
    tiler = make_tiler(patch_size=16)
    imgs = [Image.new("RGB", (256, 256)), Image.new("RGB", (256, 256))]
    result = tiler.compute_params(imgs, num_tokens_available=50)
    total_emb = sum(p.num_embeddings for p in result)
    # After pixel-shuffle, the budget is scaled up by 4, so total token_count <= 50*4
    # but num_embeddings = token_count / 4, so num_embeddings <= 50.
    assert total_emb <= 50


def test_compute_params_raises_on_unconvergeable():
    tiler = make_tiler(patch_size=16)
    imgs = [Image.new("RGB", (64, 64))]
    # Patch process_media to always return a huge token count so it never converges.
    with mock.patch.object(
        tiler,
        "process_media",
        return_value=(
            DynamicResolutionParams(
                media=imgs[0],
                num_tiles=1,
                num_embeddings=999999,
                patch_size=(100, 100),
            ),
            999999,
        ),
    ):
        with pytest.raises(ValueError, match="failed to converge"):
            tiler.compute_params(imgs, num_tokens_available=10)


@pytest.fixture
def vision_encoder():
    """Create a mock NanoV2VLVisionEncoder with required attributes."""
    encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
    encoder.llm_hidden_size = 512
    encoder.video_pruning_ratio = 0.0
    return encoder


def test_forward_dynamic_path(vision_encoder):
    fake_embeds = torch.randn(1, 10, 512)
    vision_encoder.extract_feature_dynamic = mock.MagicMock(return_value=fake_embeds)
    vision_encoder.extract_feature = mock.MagicMock()
    vision_encoder.apply_evs = mock.MagicMock()

    mm_data = {
        "modality_type": "image",
        "image": {
            "pixel_values": torch.randn(1, 100, 768),
            "imgs_sizes": [(32, 32)],
        },
    }
    mm_param = mock.MagicMock()
    mm_param.multimodal_data = mm_data

    NanoV2VLVisionEncoder.forward(vision_encoder, [mm_param])

    vision_encoder.extract_feature_dynamic.assert_called_once()
    vision_encoder.extract_feature.assert_not_called()


def test_forward_fixed_tile_path(vision_encoder):
    fake_embeds = torch.randn(2, 8, 512)
    vision_encoder.extract_feature = mock.MagicMock(return_value=fake_embeds)
    vision_encoder.extract_feature_dynamic = mock.MagicMock()
    vision_encoder.apply_evs = mock.MagicMock(
        return_value=([fake_embeds[:1].reshape(-1, 512)], [None])
    )

    mm_data = {
        "modality_type": "image",
        "image": {
            "pixel_values": torch.randn(2, 3, 512, 512),
            "num_patches": torch.tensor([2]),
        },
    }
    mm_param = mock.MagicMock()
    mm_param.multimodal_data = mm_data

    NanoV2VLVisionEncoder.forward(vision_encoder, [mm_param])

    vision_encoder.extract_feature.assert_called_once()
    vision_encoder.extract_feature_dynamic.assert_not_called()
    vision_encoder.apply_evs.assert_called_once()
