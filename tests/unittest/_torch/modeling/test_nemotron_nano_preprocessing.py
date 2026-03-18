"""Preprocessing unit tests for modeling_nemotron_nano.py."""

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
        side_effect=lambda text, **kw: torch.tensor(list(range(len(text))))
        if kw.get("return_tensors") == "pt"
        else list(range(len(text)))
    )

    config = mock.Mock()
    config.torch_dtype = torch.bfloat16
    config.force_image_size = overrides.get("image_size", 512)
    config.patch_size = overrides.get("patch_size", 16)
    config.downsample_ratio = overrides.get("downsample_ratio", 0.5)
    config.img_context_token_id = 20
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


def _make_audio_processor(**overrides):
    """Create a NanoV2VLInputProcessor with audio support and mocked deps."""
    return _make_nano_processor(sound_config=_make_extractor_config(), **overrides)


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
            "audio_feature_lengths",
        } <= audio_inputs.keys()

    def test_process_audio_raises_without_sound_config(self):
        proc = _make_audio_processor()
        proc._audio_extractor = None
        with pytest.raises(ValueError, match="no audio preprocessing"):
            proc._process_audio("test", [np.zeros(100)])
