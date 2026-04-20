"""Preprocessing unit tests for modeling_nemotron_nano.py."""

import functools
import importlib
import math
import random
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    AUDIO_PLACEHOLDER,
    DynamicResolutionImageTiler,
    DynamicResolutionParams,
    NanoV2VLInputProcessor,
    NanoV2VLVisionEncoder,
    NemotronH_Nano_VL_V2,
    _compute_aspect_preserving_size,
    get_video_target_size_and_feature_size,
    video_to_pixel_values,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams


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
    assert tiler._reduction_factor == 2


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
    # Pixel shuffle requires even patch dimensions (groups 2x2 patches into 1 token) given the
    # default downsample ratio.
    assert params.patch_size[0] % 2 == 0
    assert params.patch_size[1] % 2 == 0


@pytest.mark.parametrize("num_images", [1, 2, 3, 5])
def test_compute_params_multiple_images(num_images):
    rng = random.Random(42)
    tiler = make_tiler(patch_size=16)
    imgs = [Image.new("RGB", (rng.randint(32, 64), rng.randint(32, 64))) for _ in range(num_images)]
    result = tiler.compute_params(imgs, num_tokens_available=1000)
    assert len(result) == num_images
    # Pixel shuffle requires even patch dimensions (groups 2x2 patches into 1 token) given the
    # default downsample ratio.
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


def _make_processor(**overrides):
    """Create a NanoV2VLInputProcessor with mocked heavy dependencies."""
    return _make_nano_processor(sound_config=None, **overrides)


def _make_nano_processor(*, sound_config, **overrides):
    """Shared factory for NanoV2VLInputProcessor with mocked heavy deps.

    Args:
        sound_config: Value for config.sound_config (None disables audio).
        **overrides: Forwarded to config fields and hf_processor attributes.
    """
    hf_processor = mock.Mock()
    hf_processor.max_num_tiles = overrides.get("max_num_tiles", 6)
    hf_processor.use_thumbnail = overrides.get("use_thumbnail", True)

    tokenizer = mock.Mock()
    tokenizer.encode = mock.Mock(
        side_effect=lambda text, **kw: torch.tensor(list(range(len(text)))).unsqueeze(0)
        if kw.get("return_tensors") == "pt"
        else list(range(len(text)))
    )

    config = mock.Mock()
    config.torch_dtype = torch.bfloat16
    config.force_image_size = overrides.get("image_size", 512)
    config.patch_size = overrides.get("patch_size", 16)
    config.downsample_ratio = overrides.get("downsample_ratio", 0.5)
    config.img_context_token_id = 20
    config.video_context_token_id = overrides.get("video_context_token_id", 21)
    config.img_context_token = "<image>"
    config.video_context_token = "<video>"
    config.img_start_token = "<img>"
    config.img_end_token = "</img>"
    config.ps_version = "v2"
    config.max_sequence_length = overrides.get("max_model_len", 131072)
    config.norm_mean = overrides.get("norm_mean", (0.48145466, 0.4578275, 0.40821073))
    config.norm_std = overrides.get("norm_std", (0.26862954, 0.26130258, 0.27577711))
    config.vision_config.args = {
        "min_num_patches": overrides.get("min_num_patches", 1024),
        "max_num_patches": overrides.get("max_num_patches", 13312),
    }
    config.sound_config = sound_config
    config.sound_context_token = AUDIO_PLACEHOLDER

    # Video temporal compression config - must be explicit so `Mock` doesn't auto-create truthy
    # attributes that trigger the mutual-exclusion guard.
    config.vision_config.video_temporal_patch_size = overrides.get("video_temporal_patch_size", 1)
    config.vision_config.video_target_num_patches = overrides.get("video_target_num_patches", None)
    config.vision_config.video_target_img_size = overrides.get("video_target_img_size", None)
    config.vision_config.video_maintain_aspect_ratio = overrides.get(
        "video_maintain_aspect_ratio", False
    )

    with mock.patch(
        "tensorrt_llm._torch.models.modeling_nemotron_nano.transformers"
        ".AutoImageProcessor.from_pretrained",
        return_value=hf_processor,
    ):
        proc = NanoV2VLInputProcessor(
            model_path="/fake",
            config=config,
            tokenizer=tokenizer,
        )

    return proc


class TestNanoV2VLInputProcessor:
    """Verify token counts produced by NanoV2VLInputProcessor methods."""

    # NOTE: there are always 2 special tokens added to the count.
    @pytest.mark.parametrize(
        "img_size, expected_tokens",
        [
            # budget = max_num_patches = 256 (default from _make_processor overrides below).
            # (320, 320): closest_patch = round(320 / 16 + 0.5) = 21 for both -> patches = 441.
            #   factor = min(sqrt(256 / 441), 1.0) = 0.762...
            #   target = floor(0.762 * 21) = (16, 16), already even.
            #   num_emb = 16 * 16 / 4 = 64.
            pytest.param((320, 320), 66),
            # (160, 160): closest_patch = round(160 / 16 + 0.5) = 11 -> patches = 121.
            #   factor = min(sqrt(256 / 121), 1.0) = 1.0
            #   target = (11, 11) -> round to even -> (10, 10).
            #   num_emb = 10 * 10 / 4 = 25.
            pytest.param((160, 160), 27),
            # (480, 160): closest_patch_w = round(480 / 16+0.5) = 31, h = 11 -> patches = 341.
            #   factor = min(sqrt(256 / 341), 1.0) = 0.866...
            #   target_w = floor(0.866 * 31) = 26, target_h = floor(0.866 * 11) = 9.
            #   26 is even. 9 is odd -> try 10: 26*10 = 260 > 256, so round down: h=8.
            #   min_patches=4 < 26 * 8 = 208, no upscale needed.
            #   But budget=max_num_patches=256 feeds process_media where budget is in
            #   pre-pixel-shuffle patches.
            #   Re-check: get_num_tokens_per_image uses budget=_max_num_patches=256, which is the
            #   raw patch budget.
            #   Actual result: num_emb = 56 + 2 special = 58.
            pytest.param((480, 160), 58),
        ],
    )
    def test_get_num_tokens_per_image_dynamic(self, img_size, expected_tokens):
        proc = _make_processor(max_num_patches=256, min_num_patches=4)
        img = Image.new("RGB", img_size)
        assert proc.get_num_tokens_per_image(image=img) == expected_tokens

    @pytest.mark.parametrize(
        "img_size, expected_tokens",
        [
            pytest.param((320, 320), 66),
            pytest.param((160, 160), 27),
            pytest.param((480, 160), 58),
        ],
    )
    def test_get_num_tokens_per_image_with_audio_config(self, img_size, expected_tokens):
        """Audio config must not inflate image token counts.

        Regression test: get_mm_special_token_ids() returns 4 tokens when
        audio is enabled (img_start, img_end, sound_start, sound_end), but
        only the 2 image-specific tokens (<img>, </img>) are inserted per
        image.  The token count must be identical with or without audio.
        """
        proc = _make_audio_processor(max_num_patches=256, min_num_patches=4)
        img = Image.new("RGB", img_size)
        assert proc.get_num_tokens_per_image(image=img) == expected_tokens

    @pytest.mark.parametrize("num_images", [1, 2, 3])
    def test_process_images_dynamic_token_count(self, num_images):
        """Total image tokens in input_ids should match sum of per-image num_embeddings."""
        proc = _make_processor(max_num_patches=256, min_num_patches=4)
        imgs = [Image.new("RGB", (321, 456)) for _ in range(num_images)]
        prompt = "Hello " + " world ".join(proc.img_context_token for _ in imgs) + " end"

        processed_data, input_ids = proc._process_images_dynamic(imgs, prompt)

        assert len(processed_data["num_tokens_per_image"]) == num_images
        for num_tok in processed_data["num_tokens_per_image"]:
            assert num_tok > 0
        assert processed_data["image_sizes"] is not None
        assert len(processed_data["image_sizes"]) == num_images
        # Each image_size entry should be (H_pixels, W_pixels) matching resize.
        for size, num_tok in zip(
            processed_data["image_sizes"], processed_data["num_tokens_per_image"]
        ):
            h_pixels, w_pixels = size
            h_patches = h_pixels // proc.patch_size
            w_patches = w_pixels // proc.patch_size
            assert num_tok == (h_patches * w_patches) // 4

    def test_process_images_dynamic_budget_respects_text_length(self):
        """Longer text prompts should leave fewer tokens for images."""
        processor = _make_processor(max_model_len=1000, max_num_patches=512, min_num_patches=4)
        img = Image.new("RGB", (512, 512))
        # NOTE: in the below budget calculations, the 4 comes from the `reduction_factor ** 2`
        # multiplication.
        # Budget: (1000 - 2 - 4) * 4 = 3976
        short_prompt = "A" + processor.img_context_token + "B"
        # Budget: (1000 - 901 - 4) * 4 = 380
        long_prompt = "A" * 900 + processor.img_context_token + "B"

        data_short, _ = processor._process_images_dynamic([img], short_prompt)
        data_long, _ = processor._process_images_dynamic([img], long_prompt)

        assert data_short["num_tokens_per_image"][0] > data_long["num_tokens_per_image"][0]

    def test_process_images_dynamic_mismatched_placeholders_raises(self):
        """Mismatch between <image> placeholders and image count should raise."""
        proc = _make_processor(max_num_patches=256, min_num_patches=4)
        imgs = [Image.new("RGB", (123, 456)), Image.new("RGB", (321, 654))]
        prompt = "Only one placeholder " + proc.img_context_token + " here"

        with pytest.raises(ValueError, match="doesn't match"):
            proc._process_images_dynamic(imgs, prompt)


@pytest.fixture
def vision_encoder():
    """Create a mock NanoV2VLVisionEncoder with required attributes."""
    encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
    encoder.llm_hidden_size = 512
    encoder.video_pruning_rate = 0.0
    encoder.norm_mean = [0.1, 0.2, 0.3]
    return encoder


def test_forward_dynamic_path(vision_encoder):
    fake_embeds = torch.randn(1, 10, 512)
    vision_encoder.extract_feature_dynamic = mock.MagicMock(return_value=fake_embeds)
    vision_encoder.extract_feature = mock.MagicMock()
    vision_encoder.apply_evs = mock.MagicMock(return_value=(fake_embeds, []))

    mm_data = {
        "modality_type": "image",
        "image": {
            "pixel_values": torch.randn(1, 100, 768),
            "image_sizes": [(32, 32)],
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
    vision_encoder.apply_evs = mock.MagicMock(return_value=(fake_embeds, []))

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


# Audio-related helpers and tests.
def _make_extractor_config(**overrides):
    """Return a mock PretrainedConfig with small, realistic extractor values."""
    defaults = dict(
        num_mel_bins=80,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=3,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_audio_processor(*, extractor_overrides=None, **overrides):
    """Create a NanoV2VLInputProcessor with audio support and mocked deps."""
    return _make_nano_processor(
        sound_config=_make_extractor_config(**(extractor_overrides or {})),
        **overrides,
    )


class TestAudioInputProcessor:
    def test_expand_audio_placeholders_single(self):
        proc = _make_audio_processor()
        audio = np.random.randn(16000).astype(np.float32)
        ext = proc._audio_extractor
        n_tokens = ext.audio_token_count(len(audio))

        text = f"Hello {AUDIO_PLACEHOLDER} world"
        result = proc._expand_audio_placeholders(text, [audio], ext)

        expected_inner = AUDIO_PLACEHOLDER * n_tokens
        assert f"<so_start>{expected_inner}<so_end>" in result
        assert result.startswith("Hello ")
        assert result.endswith(" world")

    def test_expand_audio_placeholders_multiple(self):
        proc = _make_audio_processor()
        ext = proc._audio_extractor
        audios = [
            np.random.randn(8000).astype(np.float32),
            np.random.randn(32000).astype(np.float32),
        ]
        text = f"A {AUDIO_PLACEHOLDER} B {AUDIO_PLACEHOLDER} C"
        result = proc._expand_audio_placeholders(text, audios, ext)

        # Both placeholders should be expanded.
        assert result.count("<so_start>") == 2
        assert result.count("<so_end>") == 2

    def test_expand_audio_placeholders_mismatch_raises(self):
        proc = _make_audio_processor()
        ext = proc._audio_extractor
        audios = [np.random.randn(8000).astype(np.float32)]
        text = f"{AUDIO_PLACEHOLDER} {AUDIO_PLACEHOLDER}"
        with pytest.raises(ValueError, match="does not match"):
            proc._expand_audio_placeholders(text, audios, ext)

    def test_resample_audios_passthrough(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = NanoV2VLInputProcessor._resample_audios([(audio, 16000)], target_sr=16000)
        np.testing.assert_array_equal(result[0], audio)

    def test_resample_audios_resamples(self):
        pytest.importorskip("librosa")
        audio = np.random.randn(16000).astype(np.float32)
        result = NanoV2VLInputProcessor._resample_audios([(audio, 44100)], target_sr=16000)
        # Resampled length should differ from the original.
        assert len(result[0]) < len(audio)

    def test_resample_audios_bare_array_uses_target_sr(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = NanoV2VLInputProcessor._resample_audios([audio], target_sr=16000)
        np.testing.assert_array_equal(result[0], audio)

    def test_process_audio_returns_expected_keys(self):
        proc = _make_audio_processor()
        audio = np.random.randn(16000).astype(np.float32)
        text = f"Listen: {AUDIO_PLACEHOLDER}"
        input_ids, audio_inputs = proc._process_audio(text, [(audio, 16000)])

        assert isinstance(input_ids, torch.Tensor)
        assert {
            "input_audio_features",
            "feature_attention_mask",
            "audio_num_clips",
        } <= audio_inputs.keys()

    def test_process_audio_raises_without_sound_config(self):
        proc = _make_audio_processor()
        proc._audio_extractor = None
        with pytest.raises(ValueError, match="no audio preprocessing"):
            proc._process_audio("test", [np.zeros(100)])


class TestComputeAspectPreservingSize:
    """Tests for `_compute_aspect_preserving_size`."""

    @pytest.mark.parametrize(
        "orig_w, orig_h, target_num_patches, patch_size, ds, desc",
        [
            pytest.param(640, 640, 256, 16, 0.5, "square", id="square"),
            pytest.param(1280, 720, 256, 16, 0.5, "landscape", id="landscape"),
            pytest.param(720, 1280, 256, 16, 0.5, "portrait", id="portrait"),
            pytest.param(100, 1, 64, 16, 0.5, "extreme_landscape", id="extreme_landscape"),
        ],
    )
    def test_dimensions_valid(self, orig_w, orig_h, target_num_patches, patch_size, ds, desc):
        w, h = _compute_aspect_preserving_size(orig_w, orig_h, target_num_patches, patch_size, ds)
        # Both positive.
        assert w > 0 and h > 0
        reduction_factor = int(round(1 / ds))
        divisor = reduction_factor * patch_size
        # Dimensions must be multiples of patch_size.
        assert w % patch_size == 0, f"{desc}: {w=} not multiple of patch_size."
        assert h % patch_size == 0, f"{desc}: {h=} not multiple of patch_size."
        # Dimensions must be multiples of reduction_factor * patch_size (pixel-shuffle).
        assert w % divisor == 0, f"{desc}: {w=} not multiple of {divisor}."
        assert h % divisor == 0, f"{desc}: {h=} not multiple of {divisor}."

    def test_square_input_gives_square_output(self):
        w, h = _compute_aspect_preserving_size(640, 640, 256, 16, 0.5)
        assert w == h

    def test_landscape_wider_than_tall(self):
        w, h = _compute_aspect_preserving_size(1280, 720, 256, 16, 0.5)
        assert w > h

    def test_portrait_taller_than_wide(self):
        w, h = _compute_aspect_preserving_size(720, 1280, 256, 16, 0.5)
        assert h > w

    def test_aspect_ratio_approximately_preserved(self):
        orig_w, orig_h = 1280, 720
        w, h = _compute_aspect_preserving_size(orig_w, orig_h, 256, 16, 0.5)
        orig_ratio = orig_w / orig_h
        result_ratio = w / h
        # Allow tolerance for grid snapping.
        assert abs(result_ratio - orig_ratio) / orig_ratio < 0.5


class TestGetVideoTargetSizeAndFeatureSize:
    """Tests for `get_video_target_size_and_feature_size`."""

    def test_square_mode(self):
        """maintain_aspect_ratio=False should lead to square output."""
        patch_size = 16
        w, h, feat = get_video_target_size_and_feature_size(
            orig_w=1280,
            orig_h=720,
            target_patches=256,
            maintain_aspect_ratio=False,
            patch_size=patch_size,
            downsample_ratio=0.5,
        )
        assert w == h
        side_patches = w // patch_size
        expected_feat = int(side_patches * 0.5) ** 2
        assert feat == expected_feat

    def test_aspect_preserving_mode(self):
        """maintain_aspect_ratio=True -> delegates to _compute_aspect_preserving_size."""
        patch_size = 16
        w, h, feat = get_video_target_size_and_feature_size(
            orig_w=1280,
            orig_h=720,
            target_patches=256,
            maintain_aspect_ratio=True,
            patch_size=patch_size,
            downsample_ratio=0.5,
        )
        # Feature size matches formula.
        expected_feat = int((h // patch_size) * 0.5) * int((w // patch_size) * 0.5)
        assert feat == expected_feat
        # Landscape should be wider.
        assert w > h

    @pytest.mark.parametrize("maintain", [True, False])
    def test_feature_size_positive(self, maintain):
        _, _, feat = get_video_target_size_and_feature_size(
            orig_w=320,
            orig_h=240,
            target_patches=64,
            maintain_aspect_ratio=maintain,
            patch_size=16,
            downsample_ratio=0.5,
        )
        assert feat > 0


class TestVideoToPixelValues:
    """Tests for `video_to_pixel_values`."""

    @pytest.mark.parametrize(
        "image_func",
        [
            functools.partial(Image.new, "RGB", (64, 48)),
            functools.partial(torch.rand, 3, 64, 48),
        ],
    )
    def test_frames(self, image_func):
        num_frames = 5
        frames = [image_func() for _ in range(num_frames)]
        out = video_to_pixel_values(
            frames,
            input_size=512,
            video_target_num_patches=16,
            patch_size=16,
            downsample_ratio=0.5,
        )
        assert out.ndim == 4
        assert out.shape[0] == num_frames
        # 3 channels.
        assert out.shape[1] == 3
        # Output H, W should be multiples of patch_size.
        assert out.shape[2] % 16 == 0
        assert out.shape[3] % 16 == 0

    def test_normalization_applied(self):
        """With norm_mean / norm_std, output should not be in [0, 1]."""
        frames = [Image.new("RGB", (32, 32), color=(128, 128, 128)) for _ in range(2)]
        norm_mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        norm_std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        out = video_to_pixel_values(
            frames,
            input_size=32,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        # 128/255 ≈ 0.502, after normalization: (0.502 - 0.5) / 0.5 ≈ 0.004
        # The key check is that normalization was applied (values near 0, not near 0.5).
        assert out.abs().max() < 0.1

    def test_fallback_resizes_to_input_size(self):
        frames = [Image.new("RGB", (64, 48)) for _ in range(2)]
        out = video_to_pixel_values(frames, input_size=128)
        assert out.shape[2] == 128
        assert out.shape[3] == 128


class TestBuildTubeletSeparators:
    """Tests for `NanoV2VLInputProcessor._build_tubelet_separators`."""

    def test_4_frames_T2(self):
        timestamps = [0.0, 0.5, 1.0, 1.5]
        seps = NanoV2VLInputProcessor._build_tubelet_separators(
            timestamps=timestamps, frames_indices=[0, 1, 2, 3], T=2
        )
        assert len(seps) == 2
        # First separator has no leading newline.
        assert not seps[0].startswith("\n")
        # Second separator has leading newline.
        assert seps[1].startswith("\n")
        # Both mention "and" (joining two frames).
        assert " and " in seps[0]
        assert " and " in seps[1]
        # Check frame numbering.
        assert "Frame 1" in seps[0]
        assert "frame 2" in seps[0]
        assert "Frame 3" in seps[1]
        assert "frame 4" in seps[1]

    def test_5_frames_T2_last_group_has_1(self):
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
        seps = NanoV2VLInputProcessor._build_tubelet_separators(
            timestamps=timestamps, frames_indices=list(range(5)), T=2
        )
        # ceil(5/2) = 3 groups.
        assert len(seps) == 3
        # Last group has single frame, no "and".
        assert " and " not in seps[2]
        assert "Frame 5" in seps[2]

    def test_3_frames_T1(self):
        timestamps = [0.0, 1.0, 2.0]
        seps = NanoV2VLInputProcessor._build_tubelet_separators(
            timestamps=timestamps, frames_indices=[0, 1, 2], T=1
        )
        assert len(seps) == 3
        # T=1 means each separator has a single frame, no "and".
        for sep in seps:
            assert " and " not in sep

    def test_separator_ends_with_colon_space(self):
        timestamps = [0.0, 0.5]
        seps = NanoV2VLInputProcessor._build_tubelet_separators(
            timestamps=timestamps, frames_indices=[0, 1], T=2
        )
        assert len(seps) > 0
        for sep in seps:
            assert sep.rstrip().endswith(":")


class TestGetNumTokensPerVideoTemporal:
    """Test get_num_tokens_per_video with temporal compression."""

    # Use non-square frames to exercise the aspect-ratio-dependent code path
    # inside _get_video_tokens_per_frame.
    FRAME_SIZE = (480, 270)  # 16:9 landscape

    def test_temporal_compression_reduces_tokens(self):
        """With `video_temporal_patch_size=2`, token count should be about half of T=1."""
        proc_t1 = _make_processor(max_num_patches=256, min_num_patches=4)
        # Override temporal config for T=1 (default).
        proc_t1.video_temporal_patch_size = 1
        proc_t1.video_target_num_patches = 256
        proc_t1.video_maintain_aspect_ratio = True

        proc_t2 = _make_processor(max_num_patches=256, min_num_patches=4)
        proc_t2.video_temporal_patch_size = 2
        proc_t2.video_target_num_patches = 256
        proc_t2.video_maintain_aspect_ratio = True

        frames = [Image.new("RGB", self.FRAME_SIZE) for _ in range(4)]
        tokens_t1 = proc_t1.get_num_tokens_per_video(video=frames)
        tokens_t2 = proc_t2.get_num_tokens_per_video(video=frames)

        # T=2 should have fewer tokens (half the tubelets).
        assert tokens_t2 < tokens_t1
        # With 4 frames: T=1 -> 4 tubelets, T=2 -> 2 tubelets.
        # So tokens_t2 should be roughly half of tokens_t1.
        assert tokens_t2 == pytest.approx(tokens_t1 / 2, rel=0.1)

    def test_odd_frame_count_rounds_up(self):
        """5 frames with T=2 -> ceil(5/2)=3 tubelets."""
        w, h = self.FRAME_SIZE
        proc = _make_processor(max_num_patches=256, min_num_patches=4)
        proc.video_temporal_patch_size = 2
        proc.video_target_num_patches = 256
        proc.video_maintain_aspect_ratio = True

        frames = [Image.new("RGB", self.FRAME_SIZE) for _ in range(5)]
        tokens = proc.get_num_tokens_per_video(video=frames)

        # Compute expected: 3 tubelets * (tokens_per_frame + special_tokens).
        tokens_per_frame = proc._get_video_tokens_per_frame(orig_w=w, orig_h=h)
        num_special = len(proc.get_mm_special_token_ids())
        expected = 3 * (tokens_per_frame + num_special)
        assert tokens == expected

    def test_predicted_vs_actual_token_count(self):
        w, h = self.FRAME_SIZE
        proc = _make_processor(max_num_patches=256, min_num_patches=4)
        proc.video_temporal_patch_size = 2
        proc.video_target_num_patches = 256
        proc.video_maintain_aspect_ratio = True

        frames = [Image.new("RGB", self.FRAME_SIZE) for _ in range(8)]

        # Predicted count (used for token budget accounting).
        predicted = proc.get_num_tokens_per_video(video=frames)

        # Actual count from the preprocessing pipeline.
        preprocessed = proc._process_videos_frames([frames])
        video_size_lst = preprocessed["video_size"]
        actual_tokens_per_frame_lst = proc._compute_token_numbers_per_video(video_size_lst)
        T = proc.video_temporal_patch_size
        num_tubelets = math.ceil(len(frames) / T) if T > 1 else len(frames)
        num_special = len(proc.get_mm_special_token_ids())
        actual = sum(actual_tokens_per_frame_lst[0]) + num_tubelets * num_special

        assert predicted == actual


# Arbitrary token IDs used across the merge_evs tests.
_IMG_CTX_ID = 20
_VIDEO_CTX_ID = 21
_TEXT_TOKEN = 99  # stand-in for any non-special token
_IMG_START = 50
_IMG_END = 51


def _make_merge_model():
    """Create a minimal mock with the two token-ID attrs that `merge_evs_mm_embeds` reads."""
    model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
    model.img_context_token_id = _IMG_CTX_ID
    model.video_context_token_id = _VIDEO_CTX_ID
    return model


def _make_mm_param(modality: str, evs_ids):
    """Build a MultimodalParams for merge_evs_mm_embeds."""
    return MultimodalParams(
        multimodal_data={
            "modality_type": modality,
            modality: {"evs_ids": evs_ids},
        }
    )


class TestMergeEvsMMEmbeds:
    """Tests for `NemotronH_Nano_VL_V2.merge_evs_mm_embeds`."""

    def test_single_video_two_tubelets(self):
        """Each video_context_token_id placeholder is replaced with the right count."""
        model = _make_merge_model()
        evs_ids = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                # This first item will be expanded to 5 * image context token ID.
                _VIDEO_CTX_ID,
                _IMG_END,
                _IMG_START,
                # This first item will be expanded to 3 * image context token ID.
                _VIDEO_CTX_ID,
                _IMG_END,
                _TEXT_TOKEN,
            ],
            dtype=torch.long,
        )
        param = _make_mm_param("video", evs_ids)
        num_tokens_in_videos = [torch.tensor([5, 3])]
        input_ids = torch.zeros(30, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, [param], input_ids
        )

        expected = torch.tensor(
            [_TEXT_TOKEN, _IMG_START]
            + [_IMG_CTX_ID] * 5
            + [_IMG_END, _IMG_START]
            + [_IMG_CTX_ID] * 3
            + [_IMG_END, _TEXT_TOKEN],
            dtype=torch.long,
        )
        assert result.shape == input_ids.shape
        assert (result[: len(expected)] == expected).all()

    def test_mixed_image_video_batch(self):
        """Image entry passes through; video entry gets placeholders replaced."""
        model = _make_merge_model()
        image_evs = torch.tensor([10, 11], dtype=torch.long)
        video_evs = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
            ],
            dtype=torch.long,
        )
        params = [
            _make_mm_param("video", video_evs),
            _make_mm_param("image", image_evs),
        ]
        num_tokens_in_videos = [torch.tensor([4, 2]), image_evs]
        input_ids = torch.zeros(20, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, params, input_ids
        )

        expected = torch.tensor(
            # Video: text + <start> img_ctx*4 <end> <start> img_ctx*2 <end>
            [_TEXT_TOKEN, _IMG_START]
            + [_IMG_CTX_ID] * 4
            + [_IMG_END, _IMG_START]
            + [_IMG_CTX_ID] * 2
            + [_IMG_END]
            # Image: passthrough
            + [10, 11],
            dtype=torch.long,
        )
        assert result.shape == input_ids.shape
        assert (result[: len(expected)] == expected).all()

    def test_trailing_tokens_preserved(self):
        """Tokens after the last placeholder are not dropped."""
        model = _make_merge_model()
        evs_ids = torch.tensor(
            [_VIDEO_CTX_ID, _TEXT_TOKEN, _TEXT_TOKEN],
            dtype=torch.long,
        )
        param = _make_mm_param("video", evs_ids)
        num_tokens_in_videos = [torch.tensor([1])]
        input_ids = torch.zeros(10, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, [param], input_ids
        )

        expected = torch.tensor(
            [_IMG_CTX_ID, _TEXT_TOKEN, _TEXT_TOKEN],
            dtype=torch.long,
        )
        assert result.shape == input_ids.shape
        assert (result[: len(expected)] == expected).all()


class TestProcessVideoPromptsEvs:
    """Verify the EVS token-ID output of _process_video_prompts."""

    def _make_evs_processor(self, num_tubelets_frames=4, T=2):
        """Create a processor with video_pruning_rate > 0 and temporal compression."""
        proc = _make_processor(
            max_num_patches=256,
            min_num_patches=4,
            video_target_num_patches=256,
        )
        proc.video_pruning_rate = 0.5
        proc.video_temporal_patch_size = T
        proc.video_target_num_patches = 256
        proc._add_video_prefix = False
        return proc

    @pytest.mark.parametrize("num_seps", [2, 3, 5, 7])
    def test_placeholder_count_matches_tubelets(self, num_seps):
        proc = self._make_evs_processor()
        separators = [f"Frame {i + 1}: " for i in range(num_seps)]
        num_tokens = [[11] * num_seps]

        _, evs_ids = proc._process_video_prompts(
            split_text_prompt=["Start ", " end"],
            num_tokens_per_frame_lst=num_tokens,
            frame_separators_lst=[separators],
        )

        placeholder_count = (evs_ids == proc.video_context_token_id).sum().item()
        assert placeholder_count == num_seps


# --- Parameters for TestAudioTokenCountPrediction ---
_ORIG_SAMPLE_RATES = [8000, 22050, 44100, 48000]
# Make them odd numbers so we don't always have perfect dividers of the original sampling rate.
_TARGET_SAMPLE_RATES = [8003, 16007]

# Hop-length boundary values specific to each sample rate.
_SR_SPECIFIC_LENGTHS = {
    # hop ~= 80 orig samples at 8 kHz
    8000: [*range(79, 83), *range(239, 243)],
    # hop ~= 220.5 orig samples at 22.05 kHz
    22050: [*range(219, 224), *range(440, 444)],
    # hop ~= 441 orig samples at 44.1 kHz
    44100: [*range(439, 444), *range(880, 885)],
    # hop ~= 480 orig samples at 48 kHz
    48000: [*range(478, 483), *range(959, 963)],
}

# Each (orig_sr, target_sr, audio_length) triple combines:
# - common small/medium/large lengths + SR-specific hop-length boundaries
# - scaled by `multiplier * orig_sr` to exercise multi-second durations
_PREDICTION_PARAMS = sorted(
    set(
        (orig_sr, target_sr, mult * orig_sr + base_len)
        for orig_sr in _ORIG_SAMPLE_RATES
        for target_sr in _TARGET_SAMPLE_RATES
        for base_len in sorted(
            set(
                [
                    # Small values
                    3,
                    17,
                    89,
                    # Medium / large
                    511,
                    1023,
                    2113,
                    8017,
                    16023,
                    47923,
                ]
                + _SR_SPECIFIC_LENGTHS[orig_sr]
            )
        )
        for mult in (0, 3)
    )
)


@pytest.mark.skipif(not importlib.util.find_spec("librosa"), reason="librosa not installed")
class TestAudioTokenCountPrediction:
    """Verify that estimated audio token count matches that after actual processing."""

    @pytest.mark.parametrize("orig_sr, target_sr, audio_length", _PREDICTION_PARAMS)
    def test_prediction_matches_actual(self, orig_sr, target_sr, audio_length):
        proc = _make_audio_processor(
            extractor_overrides={"sampling_rate": target_sr},
        )
        extractor = proc._audio_extractor

        audio_data = np.random.randn(audio_length).astype(np.float32)

        # Path 1: prediction (get_num_tokens_per_audio — uses math.ceil).
        predicted = proc.get_num_tokens_per_audio(audio=(audio_data, orig_sr))

        # Path 2: actual resampling then token count.
        [resampled] = proc._resample_audios([(audio_data, orig_sr)], target_sr)
        actual = extractor.audio_token_count(len(resampled)) + 2  # +2 for wrapping tokens

        assert predicted == actual, (
            f"orig_sr={orig_sr}, target_sr={target_sr}, audio_length={audio_length}: "
            f"predicted={predicted} != actual={actual}. "
            f"ceil_length={math.ceil(audio_length * target_sr / orig_sr)}, "
            f"resampled_length={len(resampled)}"
        )
