# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from tensorrt_llm.inputs.multimodal import (
    MultimodalParams,
    MultimodalRuntimeData,
    _compute_mm_masks,
    _find_mm_token_start_pos_from_masks,
    compute_mm_embedding_lengths,
)
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData


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
        side_effect=lambda text, **kw: (
            torch.tensor(list(range(len(text)))).unsqueeze(0)
            if kw.get("return_tensors") == "pt"
            else list(range(len(text)))
        )
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

    # Default the start/end wrapper ID lists to single-element lists.
    # Real production tokenizers register <img>/</img> as added special tokens
    # with single IDs, so this matches reality. Without the override, the
    # mock tokenizer's `list(range(len(text)))` encoding yields multi-token
    # IDs (5 for "<img>", 6 for "</img>") which leak into
    # `get_num_tokens_per_image`'s `+= len(start) + len(end)` accounting and
    # break tests that pre-date the multi-token wrapper fix. Test classes that
    # specifically exercise multi-token wrappers can still override these
    # attributes after construction.
    proc._img_start_token_ids = [500]
    proc._img_end_token_ids = [501]
    proc.image_start_token_id = 500
    proc.image_end_token_id = 501

    # When sound_config is None, the constructor still runs
    # `self._sound_context_token_id = getattr(config, "sound_context_token_id", None)`
    # against the Mock config, which returns a Mock instead of None. That
    # would later sneak into get_mm_token_ids / get_mm_special_token_ids and
    # raise "Mock object cannot be interpreted as an integer" inside
    # torch.tensor. Force the audio attributes to None to match the real
    # no-audio case. (When sound_config is provided, __init__ overwrites
    # these from the tokenizer, so the audio path is unaffected.)
    if sound_config is None:
        proc._sound_context_token_id = None
        proc._sound_start_token_id = None
        proc._sound_end_token_id = None

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

    def test_process_images_dynamic_reserves_non_image_tokens(self):
        """Reserving non-image (video/audio) tokens shrinks the image budget.

        Regression for the mixed-request over-allocation: in a mixed image +
        video/audio request the video/audio placeholders are still single,
        un-expanded tokens when the dynamic image tiler runs, so without
        reserving their real post-expansion cost the image would claim nearly
        all of max_model_len and the sequence would overflow once they expand.
        A positive `reserved_non_image_tokens` must leave fewer tokens for the
        image, exactly as a longer text prompt does.
        """
        processor = _make_processor(max_model_len=1000, max_num_patches=512, min_num_patches=4)
        img = Image.new("RGB", (512, 512))
        prompt = "A" + processor.img_context_token + "B"

        # reserved=0 budget: (1000 - 2 - 4) * 4 = 3976 (image hits its natural cap);
        # reserved=900 budget: (1000 - 2 - 4 - 900) * 4 = 376 (image is squeezed),
        # mirroring how a 900-char-longer text prompt squeezes the image.
        data_none, _ = processor._process_images_dynamic([img], prompt, reserved_non_image_tokens=0)
        data_reserved, _ = processor._process_images_dynamic(
            [img], prompt, reserved_non_image_tokens=900
        )

        assert data_reserved["num_tokens_per_image"][0] < data_none["num_tokens_per_image"][0]

    def test_mixed_modalities_reserves_non_image_budget(self):
        """The mixed path reserves the expanded audio/video cost for the image.

        End-to-end regression for the headline mixed image+audio request: the
        audio placeholder is still a single token when the dynamic image tiler
        runs, so `_process_mixed_modalities` must compute the real expanded
        audio length and hand it to `_prepare_image_modality_data` as the
        non-image reservation. Without it the image would be budgeted against
        nearly all of max_model_len and the sequence could overflow once the
        audio placeholder expands.
        """
        proc = _make_audio_processor(max_model_len=1000, max_num_patches=512, min_num_patches=4)
        img = Image.new("RGB", (512, 512))
        audio = np.random.randn(16000).astype(np.float32)
        mm_data = {"image": [img], "audio": [audio]}
        prompt = f"{proc.img_context_token} then {AUDIO_PLACEHOLDER}"

        # Expected reservation == the audio item's real expanded token count.
        expected_reserved = proc.get_num_tokens_per_audio(audio=audio)
        assert expected_reserved > 1  # audio expands to many tokens, not ~1

        captured = {}
        real_prepare_image = proc._prepare_image_modality_data

        def _spy_prepare_image(images, text_prompt, reserved_non_image_tokens=0):
            captured["reserved"] = reserved_non_image_tokens
            return real_prepare_image(images, text_prompt, reserved_non_image_tokens)

        with (
            mock.patch.object(proc, "_prepare_image_modality_data", side_effect=_spy_prepare_image),
            mock.patch.object(proc, "_prepare_audio_modality_data", return_value={}),
            mock.patch.object(
                proc,
                "_expand_mixed_prompt_text_for_mm",
                return_value=(torch.zeros(1, 4, dtype=torch.long), None),
            ),
            # `compute_mm_embedding_lengths` is now a module-level call; its
            # arguments (the token-id hooks below) are evaluated eagerly before
            # the patched function runs, so stub the hooks to harmless values to
            # keep this test scoped to the image-budget reservation.
            mock.patch.object(proc, "get_mm_token_ids", return_value=None),
            mock.patch.object(proc, "get_mm_special_token_ids", return_value=None),
            mock.patch.object(proc, "get_vocab_size", return_value=1000),
            mock.patch(
                "tensorrt_llm._torch.models.modeling_nemotron_nano.compute_mm_embedding_lengths",
                return_value=[],
            ),
        ):
            proc._process_mixed_modalities(prompt, mm_data)

        assert captured["reserved"] == expected_reserved

    def test_mixed_image_audio_dynamic_tiler_reconciles_post_tile_counts(self):
        """The mixed path must size the image slot from the post-tile actual.

        Headline user-reachable crash: on the dynamic-tiler path a large image
        co-located with a real audio is PLANNED at the full `_max_num_patches`
        cap by `get_num_tokens_per_image`, but `_prepare_image_modality_data`
        TILES it at the reservation-reduced budget. When
        `(max_model_len - text_len - reserved - min_num_patches) * 4 <
        max_num_patches` the actual tiled count is smaller than the planned one,
        so the encoder emits fewer rows than `sum(item.rows)` and the row-count
        check in `mixed_modal_encode.py` trips (or, under `python -O`, the
        mismatch silently corrupts the scatter). The fix reconciles the image
        slot to the actual post-tile count.

        Unlike `test_mixed_modalities_reserves_non_image_budget`, this test does
        NOT mock `_expand_mixed_prompt_text_for_mm` or the embedding-length
        computation: those mocks hide this bug because they discard the
        per-image `num_mm_tokens` that the reconciliation must fix. Instead it
        captures the reconciled `num_mm_tokens` that the real build consumes
        (the single source of truth for both `input_ids` and
        `multimodal_embedding_lengths`) and asserts the image slot equals the
        actual tiled count, not the planned full-cap count.
        """
        proc = _make_audio_processor(max_model_len=120, max_num_patches=512, min_num_patches=4)
        # The lightweight Mock config / mock tokenizer leave these token-id and
        # vocab hooks as Mock objects or as the colliding id 0 (the mock
        # tokenizer encodes every string as `range(len(s))`, so `<so_start>` /
        # `<so_end>` both resolve to 0, which then aliases the first text token).
        # That would break the REAL `compute_mm_embedding_lengths` mask logic for
        # reasons unrelated to the reconciliation under test. Pin them to
        # concrete, distinct ids (clear of the image context id 20 and the
        # `<img>`/`</img>` specials 500/501) so the real embed-mask path runs
        # end-to-end and the only divergence left is the image-count bug itself.
        proc._sound_context_token_id = 30
        proc._sound_start_token_id = 31
        proc._sound_end_token_id = 32
        proc.config.llm_config.vocab_size = 1000

        img = Image.new("RGB", (512, 512))
        audio = np.random.randn(16000).astype(np.float32)
        mm_data = {"image": [img], "audio": [audio]}
        prompt = f"{proc.img_context_token} then {AUDIO_PLACEHOLDER}"

        # The non-image reservation is the real expanded audio cost; with it the
        # image budget is squeezed below the `_max_num_patches` cap so the tiled
        # (actual) count is strictly smaller than the planned full-cap count.
        reserved = proc.get_num_tokens_per_audio(audio=audio)
        actual_data, _ = proc._process_images_dynamic([img], prompt, reserved)
        actual_image_context = actual_data["num_tokens_per_image"][0]

        # The OLD planned count: full `_max_num_patches`-cap tiling, plus the
        # `<img>`/`</img>` start/end specials that `get_num_tokens_per_image`
        # folds into its total.
        planned_total = proc.get_num_tokens_per_image(image=img)
        start_end = len(proc._img_start_token_ids) + len(proc._img_end_token_ids)
        planned_image_context = planned_total - start_end

        # The actual tiled context count must be strictly smaller than the old
        # planned full-cap count, otherwise this budget would not exercise the
        # divergence the fix targets.
        assert actual_image_context < planned_image_context

        # Capture the reconciled `num_mm_tokens` that the real build consumes.
        # It drives BOTH `input_ids` (via `_expand_mixed_prompt_text_for_mm`)
        # and `multimodal_embedding_lengths` (via `compute_mm_embedding_lengths`),
        # so the per-image embed length equals this slot minus start/end.
        captured = {}
        real_lengths = compute_mm_embedding_lengths

        def _spy_lengths(input_ids, num_mm_tokens, **kw):
            captured["num_mm_tokens"] = list(num_mm_tokens)
            return real_lengths(input_ids, num_mm_tokens, **kw)

        with mock.patch(
            "tensorrt_llm._torch.models.modeling_nemotron_nano.compute_mm_embedding_lengths",
            side_effect=_spy_lengths,
        ):
            _, extra = proc._process_mixed_modalities(prompt, mm_data)

        # Locate the image slot in prompt order and read its reconciled count.
        item_order = extra["multimodal_data"]["multimodal_item_order"]
        image_slot = next(i for i, entry in enumerate(item_order) if entry["modality"] == "image")
        reconciled_image_context = captured["num_mm_tokens"][image_slot] - start_end

        # The image slot must be sized from the ACTUAL post-tile count, not the
        # PLANNED full-cap count.
        assert reconciled_image_context == actual_image_context
        assert reconciled_image_context < planned_image_context

        # And the resulting embedding length for the image entry must match the
        # actual encoder rows (the real `compute_mm_embedding_lengths` output).
        embedding_lengths = extra["multimodal_data"]["multimodal_embedding_lengths"]
        assert embedding_lengths[image_slot] == actual_image_context

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


def _wire_bucket_helpers_to_extract_feature(vision_encoder):
    """Make `_encode_*` buckets call `extract_feature*` once per non-empty bucket.

    `forward` now delegates to per-modality bucket helpers, which `MagicMock(spec=...)`
    auto-mocks alongside the encoder. Stub the buckets so they forward to the real
    `extract_feature*` mocks the tests configure — keeping the original
    "did we route through the dynamic/fixed path?" assertions valid.
    """

    def _dynamic(image_data_list):
        return [vision_encoder.extract_feature_dynamic()] if image_data_list else []

    def _fixed(image_data_list):
        return [vision_encoder.extract_feature()] if image_data_list else []

    def _video(video_data_list):
        return []

    vision_encoder._encode_dynamic_image.side_effect = _dynamic
    vision_encoder._encode_fixed_tile.side_effect = _fixed
    vision_encoder._encode_temporal_video.side_effect = _video


def test_forward_dynamic_path(vision_encoder):
    fake_embeds = torch.randn(1, 10, 512)
    vision_encoder.extract_feature_dynamic = mock.MagicMock(return_value=fake_embeds)
    vision_encoder.extract_feature = mock.MagicMock()
    vision_encoder.apply_evs = mock.MagicMock(return_value=([fake_embeds], []))
    _wire_bucket_helpers_to_extract_feature(vision_encoder)

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
    vision_encoder.apply_evs = mock.MagicMock(return_value=([fake_embeds], []))
    _wire_bucket_helpers_to_extract_feature(vision_encoder)

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

    # `torch.compile` uses a thread pool to compile.
    @pytest.mark.threadleak(enabled=False)
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

    def test_call_audio_stashes_evs_ids_when_evs_enabled(self):
        proc = _make_audio_processor()
        proc.video_pruning_rate = 0.5
        audio = np.random.randn(16000).astype(np.float32)
        inputs = {
            "prompt": f"Listen: {AUDIO_PLACEHOLDER}",
            "multi_modal_data": {"audio": [(audio, 16000)]},
        }

        input_ids, extra_inputs = proc(inputs, None)

        evs_ids = extra_inputs["multimodal_data"]["audio"]["evs_ids"]
        assert evs_ids.dtype == torch.int32
        assert evs_ids.tolist() == input_ids

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
_SOUND_CTX_ID = 30
_SOUND_START = 200
_SOUND_END = 201
_TEXT_TOKEN = 99  # stand-in for any non-special token
_IMG_START = 50
_IMG_END = 51


def _make_merge_model():
    """Create a minimal mock with the attrs/helpers that `merge_evs_mm_embeds` reads."""
    model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
    model.img_context_token_id = _IMG_CTX_ID
    model.video_context_token_id = _VIDEO_CTX_ID
    model.sound_context_token_id = _SOUND_CTX_ID
    model._build_evs_adjusted_context_ids = functools.partial(
        NemotronH_Nano_VL_V2._build_evs_adjusted_context_ids, model
    )
    model._refresh_evs_runtime_and_slice_context_ids = functools.partial(
        NemotronH_Nano_VL_V2._refresh_evs_runtime_and_slice_context_ids, model
    )
    return model


def _make_runtime(past_seen_token_num: int, chunk_end_pos: int, prompt_len: int):
    return MultimodalRuntimeData(
        past_seen_token_num=past_seen_token_num,
        chunk_end_pos=chunk_end_pos,
        embed_mask_cumsum=torch.zeros(prompt_len, dtype=torch.int64),
    )


def _make_mm_param(modality: str, evs_ids, runtime=None):
    """Build a MultimodalParams for merge_evs_mm_embeds."""
    return MultimodalParams(
        multimodal_data={
            "modality_type": modality,
            modality: {"evs_ids": evs_ids},
        },
        multimodal_runtime=runtime,
    )


class TestMergeEvsMMEmbeds:
    """Tests for `NemotronH_Nano_VL_V2.merge_evs_mm_embeds`."""

    def test_evs_video_with_text_only_context_raises(self):
        """Reject batches where text-only context chunks would shift EVS writes."""
        params = [_make_mm_param("video", torch.tensor([_VIDEO_CTX_ID], dtype=torch.long))]

        with pytest.raises(ValueError, match="text-only context requests"):
            NemotronH_Nano_VL_V2._validate_evs_context_batch(
                params,
                num_context_requests=2,
            )

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

    # Axis `passthrough_modality`: a non-video entry (image or audio) passes
    # through verbatim while the video entry gets its placeholders replaced.
    @pytest.mark.parametrize(
        "passthrough_modality, video_placeholders, video_token_counts, "
        "passthrough_evs, passthrough_num_tokens, expected_tail",
        [
            pytest.param(
                "image",
                2,
                [4, 2],
                [10, 11],
                [10, 11],
                [10, 11],
                id="image_passthrough",
            ),
            pytest.param(
                "audio",
                1,
                [2],
                [_SOUND_START, _SOUND_CTX_ID, _SOUND_CTX_ID, _SOUND_END],
                None,  # audio slot has no video token-count tensor
                [_SOUND_START, _SOUND_CTX_ID, _SOUND_CTX_ID, _SOUND_END],
                id="audio_passthrough",
            ),
        ],
    )
    def test_mixed_passthrough_batch(
        self,
        passthrough_modality,
        video_placeholders,
        video_token_counts,
        passthrough_evs,
        passthrough_num_tokens,
        expected_tail,
    ):
        model = _make_merge_model()
        video_evs = torch.tensor(
            [_TEXT_TOKEN] + ([_IMG_START, _VIDEO_CTX_ID, _IMG_END] * video_placeholders),
            dtype=torch.long,
        )
        passthrough_tensor = torch.tensor(passthrough_evs, dtype=torch.long)
        params = [
            _make_mm_param("video", video_evs),
            _make_mm_param(passthrough_modality, passthrough_tensor),
        ]
        num_tokens_in_videos = [
            torch.tensor(video_token_counts),
            torch.tensor(passthrough_num_tokens) if passthrough_num_tokens is not None else None,
        ]
        input_ids = torch.zeros(20, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, params, input_ids
        )

        # Video: text + <start> img_ctx*count <end> per placeholder, then the
        # non-video entry passes through verbatim.
        video_expanded = [_TEXT_TOKEN]
        for count in video_token_counts:
            video_expanded += [_IMG_START] + [_IMG_CTX_ID] * count + [_IMG_END]
        expected = torch.tensor(video_expanded + expected_tail, dtype=torch.long)
        assert result.shape == input_ids.shape
        assert (result[: len(expected)] == expected).all()

    def test_single_request_mixed_image_video_audio_evs(self):
        """A mixed single request uses the shared EVS stream and expands video placeholders."""
        model = _make_merge_model()
        evs_ids = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                _IMG_CTX_ID,
                _IMG_CTX_ID,
                _IMG_END,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                _SOUND_START,
                _SOUND_CTX_ID,
                _SOUND_END,
            ],
            dtype=torch.long,
        )
        param = MultimodalParams(
            multimodal_data={
                "modality_type": ["image", "video", "audio"],
                "image": {"evs_ids": evs_ids},
                "video": {"evs_ids": evs_ids},
                "audio": {"evs_ids": evs_ids},
            }
        )
        input_ids = torch.zeros(20, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, [torch.tensor([3])], [param], input_ids
        )

        expected = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                _IMG_CTX_ID,
                _IMG_CTX_ID,
                _IMG_END,
                _IMG_START,
                _IMG_CTX_ID,
                _IMG_CTX_ID,
                _IMG_CTX_ID,
                _IMG_END,
                _SOUND_START,
                _SOUND_CTX_ID,
                _SOUND_END,
            ],
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

    def test_chunked_prefill_uses_current_context_slice(self):
        """Chunked prefill should write only the active context window."""
        model = _make_merge_model()
        evs_ids = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                88,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                77,
            ],
            dtype=torch.long,
        )
        full_context = torch.tensor(
            [_TEXT_TOKEN, _IMG_START]
            + [_IMG_CTX_ID] * 3
            + [_IMG_END, 88, _IMG_START]
            + [_IMG_CTX_ID] * 2
            + [_IMG_END, 77],
            dtype=torch.long,
        )
        runtime = _make_runtime(
            past_seen_token_num=4,
            chunk_end_pos=9,
            prompt_len=len(full_context),
        )
        param = _make_mm_param("video", evs_ids, runtime=runtime)
        num_tokens_in_videos = [torch.tensor([3, 2])]
        generation_tail = torch.tensor([700, 701], dtype=torch.long)
        input_ids = torch.cat([torch.zeros(5, dtype=torch.long), generation_tail.clone()])

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, [param], input_ids
        )

        expected_context_chunk = full_context[4:9]
        assert result.shape == input_ids.shape
        assert (result[: len(expected_context_chunk)] == expected_context_chunk).all()
        assert (result[len(expected_context_chunk) :] == generation_tail).all()
        assert runtime.num_cached_mm_tokens == 2
        assert runtime.num_mm_tokens_in_chunk == 2
        assert runtime.total_embeds_in_request == 5

    def test_chunked_prefill_first_chunk_can_be_shorter_than_full_context(self):
        """A full EVS context longer than input_ids must not be assigned wholesale."""
        model = _make_merge_model()
        evs_ids = torch.tensor(
            [
                _TEXT_TOKEN,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                _IMG_START,
                _VIDEO_CTX_ID,
                _IMG_END,
                _TEXT_TOKEN,
            ],
            dtype=torch.long,
        )
        full_context = torch.tensor(
            [_TEXT_TOKEN, _IMG_START]
            + [_IMG_CTX_ID] * 5
            + [_IMG_END, _IMG_START]
            + [_IMG_CTX_ID] * 4
            + [_IMG_END, _TEXT_TOKEN],
            dtype=torch.long,
        )
        runtime = _make_runtime(
            past_seen_token_num=0,
            chunk_end_pos=4,
            prompt_len=len(full_context),
        )
        param = _make_mm_param("video", evs_ids, runtime=runtime)
        num_tokens_in_videos = [torch.tensor([5, 4])]
        input_ids = torch.zeros(4, dtype=torch.long)

        result = NemotronH_Nano_VL_V2.merge_evs_mm_embeds(
            model, num_tokens_in_videos, [param], input_ids
        )

        assert result.shape == input_ids.shape
        assert (result == full_context[:4]).all()
        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 2
        assert runtime.total_embeds_in_request == 9


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


# ---------------------------------------------------------------------------
# Tokenized+MM fast path tests
# ---------------------------------------------------------------------------


def _make_fast_path_processor(**overrides):
    """Create a processor with predictable single-token special token IDs.

    The default mock tokenizer returns list(range(len(text))), which makes
    multi-char tokens map to multi-element lists.  For expansion tests we
    override the pre-tokenized ID lists so that <img>, </img>, etc. are each
    represented by a single, distinct integer.

    We pick IDs outside the printable-ASCII `ord(c)` range (0-127) so that
    tests which override `tokenizer.encode` with ord-based mocks (e.g. the
    text-level path tests) don't produce spurious img_start/img_end IDs
    from letters in frame separators like "Frame" (`ord('e') == 101`).
    """
    proc = _make_processor(**overrides)
    # Override pre-tokenized special token ID lists to single-element lists.
    proc._img_start_token_ids = [500]
    proc._img_end_token_ids = [501]
    proc._img_context_token_ids = [proc.img_context_token_id]  # 20
    # Collapse the `<video>` placeholder to a single-ID pattern for tests so
    # the existing single-token prompts (e.g. [1, 2, vid_ctx, 3]) still match.
    # Real tokenizers may produce multi-token BPE for `<video>`; the
    # TestExpandVideoPlaceholdersMultiToken suite exercises that case.
    proc._video_placeholder_token_ids = [proc.video_context_token_id]  # 21
    proc.image_start_token_id = 500
    proc.image_end_token_id = 501
    return proc


def _make_fast_path_audio_processor(**overrides):
    """Audio-enabled processor with predictable special token IDs."""
    proc = _make_audio_processor(**overrides)
    proc._img_start_token_ids = [500]
    proc._img_end_token_ids = [501]
    proc._video_placeholder_token_ids = [proc.video_context_token_id]
    proc.image_start_token_id = 500
    proc.image_end_token_id = 501
    # Audio special token IDs.
    proc._sound_context_token_id = 30
    proc._sound_start_token_id = 200
    proc._sound_end_token_id = 201
    return proc


class TestGetTextWithMMPlaceholders:
    @pytest.mark.parametrize(
        "mm_counts, expected",
        [
            pytest.param({"image": 1}, "<image>", id="single_image"),
            pytest.param({"image": 3}, "<image><image><image>", id="multiple_images"),
            pytest.param({"video": 1}, "<video>", id="single_video"),
            pytest.param({}, "", id="empty"),
        ],
    )
    def test_image_video_and_empty(self, mm_counts, expected):
        proc = _make_fast_path_processor()
        assert proc.get_text_with_mm_placeholders(mm_counts) == expected

    def test_single_audio(self):
        # Kept separate: audio requires `_make_fast_path_audio_processor` (a
        # different factory that wires up `sound_config`).
        proc = _make_fast_path_audio_processor()
        assert proc.get_text_with_mm_placeholders({"audio": 1}) == AUDIO_PLACEHOLDER


class TestExpandImagePlaceholders:
    # Placeholder IDs are referenced as literals (20=img_ctx, 500/501=start/end)
    # to keep the parametrize tables readable. See `_make_fast_path_processor`.
    @pytest.mark.parametrize(
        "prompt, num_mm_tokens, expected",
        [
            pytest.param(
                [1, 2, 20, 3, 4],
                [12],  # 1 (start) + 10 (context) + 1 (end)
                [1, 2, 500] + [20] * 10 + [501, 3, 4],
                id="single_image",
            ),
            pytest.param(
                [1, 20, 2, 20, 3],
                [7, 7],  # each: 1 start + 5 context + 1 end
                [1, 500] + [20] * 5 + [501, 2, 500] + [20] * 5 + [501, 3],
                id="multiple_images",
            ),
        ],
    )
    def test_happy_path(self, prompt, num_mm_tokens, expected):
        proc = _make_fast_path_processor()
        assert proc._expand_image_placeholders_in_token_ids(prompt, num_mm_tokens) == expected

    @pytest.mark.parametrize(
        "prompt, num_mm_tokens, match",
        [
            pytest.param([20, 20], [5], "More image placeholder", id="more_placeholders"),
            pytest.param([20], [5, 5], "Expected 2 image placeholders", id="fewer_placeholders"),
        ],
    )
    def test_mismatch_raises(self, prompt, num_mm_tokens, match):
        proc = _make_fast_path_processor()
        with pytest.raises(ValueError, match=match):
            proc._expand_image_placeholders_in_token_ids(prompt, num_mm_tokens)


class TestExpandAudioPlaceholders:
    # Literals: 30=snd_ctx, 200/201=sound_start/end. See `_make_fast_path_audio_processor`.
    @pytest.mark.parametrize(
        "prompt, num_mm_tokens, expected",
        [
            pytest.param(
                [1, 2, 30, 3],
                [10],  # 1 start + 8 context + 1 end
                [1, 2, 200] + [30] * 8 + [201, 3],
                id="single_audio",
            ),
            pytest.param(
                [30, 5, 30],
                [6, 8],
                [200] + [30] * 4 + [201, 5, 200] + [30] * 6 + [201],
                id="multiple_audios",
            ),
        ],
    )
    def test_happy_path(self, prompt, num_mm_tokens, expected):
        proc = _make_fast_path_audio_processor()
        assert proc._expand_audio_placeholders_in_token_ids(prompt, num_mm_tokens) == expected

    def test_mismatch_raises(self):
        proc = _make_fast_path_audio_processor()
        snd_ctx = proc._sound_context_token_id
        with pytest.raises(ValueError, match="More audio placeholder"):
            proc._expand_audio_placeholders_in_token_ids([snd_ctx, snd_ctx], [5])


class TestExpandVideoPlaceholders:
    @staticmethod
    def _make_video_processor():
        """Non-dynamic processor with video_target_num_patches for simple math."""
        proc = _make_fast_path_processor(
            video_target_num_patches=None,
        )
        # With video_target_num_patches=None, image_size=512, patch_size=16,
        # downsample_ratio=0.5: tokens_per_frame = (512/16)^2 * 0.5^2 = 256.
        # video_size = [num_frames, 1, 512, 512]
        # _add_video_prefix should be True for fallback path.
        proc._add_video_prefix = False  # disable to simplify test
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1
        return proc

    def test_single_video_no_metadata(self):
        proc = self._make_video_processor()
        vid_ctx = proc.video_context_token_id  # 21
        img_ctx = proc.img_context_token_id  # 20
        prompt = [1, 2, vid_ctx, 3]
        num_frames = 2

        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}

        # tokens_per_frame = 256, so total MM tokens per video =
        # num_frames * (256 + 2) = 2 * 258 = 516
        total_mm = num_frames * (256 + 2)

        expanded, evs_ids = proc._expand_video_placeholders_in_token_ids(
            prompt, [total_mm], mm_data
        )

        # No EVS -> evs_ids must be None.
        assert evs_ids is None

        # Should start with [1, 2], end with [3].
        assert expanded[:2] == [1, 2]
        assert expanded[-1] == 3

        # Count img_context tokens: should be 2 * 256 = 512.
        ctx_count = expanded.count(img_ctx)
        assert ctx_count == num_frames * 256

        # Count img_start tokens: should be 2 (one per frame).
        start_count = expanded.count(500)
        assert start_count == num_frames

        # Count img_end tokens: should be 2.
        end_count = expanded.count(501)
        assert end_count == num_frames

    def test_missing_mm_data_raises(self):
        proc = self._make_video_processor()
        vid_ctx = proc.video_context_token_id
        with pytest.raises(ValueError, match="requires multi_modal_data"):
            proc._expand_video_placeholders_in_token_ids([vid_ctx], [100], None)


class TestExpandVideoPlaceholdersMultiToken:
    """Exercise the realistic case where `<video>` BPE-decomposes into multiple
    tokens (e.g. [1060, 24073, 1062] for the Nemotron Nano tokenizer). This is
    what production hits — `<video>` is typically NOT registered as an added
    special token, so `video_context_token_id` is never what the tokenizer
    produces from the user's prompt text.

    With length-based dispatch, `len(_video_placeholder_token_ids) > 1` always
    goes through the text-level (decode/split/re-encode) path — token-level
    matching is unreliable here because BPE can merge the placeholder's last
    token with the following character."""

    @staticmethod
    def _make_video_processor_multi_token():
        proc = _make_fast_path_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1
        # Simulate realistic BPE decomposition: "<video>" -> three tokens.
        proc._video_placeholder_token_ids = [1060, 24073, 1062]
        return proc

    def test_text_level_path_when_bpe_merges_trailing_context(self):
        """Simulate the production scenario: `<video>` tokenizes to one pattern
        in isolation (e.g. [1060, 24073, 1062]) but the tokens in the actual
        prompt end with a DIFFERENT ID (e.g. [1060, 24073, 3318]) because BPE
        merged `>` with the following newline/whitespace. The text-level path
        must handle this correctly (length-based dispatch routes all multi-
        token placeholders here)."""
        proc = self._make_video_processor_multi_token()
        img_ctx = proc.img_context_token_id

        # Patch the mock tokenizer to (a) decode back to a text that contains
        # `<video>` and (b) re-encode that text stably. We simulate a tokenizer
        # that produces [1060, 24073, 3318] in context but splits <video> into
        # [1060, 24073, 1062] in isolation.
        def mock_decode(ids, **kw):
            # Deterministic: represent any ID as its string form, and emit the
            # literal string "<video>" whenever we see the "in-context" trio.
            text_parts = []
            i = 0
            while i < len(ids):
                if ids[i : i + 3] == [1060, 24073, 3318]:
                    text_parts.append("<video>\n")
                    i += 3
                else:
                    text_parts.append(f"T{ids[i]} ")
                    i += 1
            return "".join(text_parts)

        def mock_encode(text, **kw):
            # Our test prompt decodes to "T1 T2 <video>\nT3 "; we re-encode by
            # splitting on `<video>` and emitting a marker per token.
            if kw.get("return_tensors") == "pt":
                return torch.tensor(list(range(len(text)))).unsqueeze(0)
            return [ord(c) % 1000 for c in text]

        proc.tokenizer.decode = mock_decode
        proc.tokenizer.encode = mock_encode

        # Prompt contains the "in-context" BPE tokenization [1060, 24073, 3318],
        # NOT the "isolation" pattern [1060, 24073, 1062]. Length-based
        # dispatch (len > 1) routes through the text-level path regardless.
        prompt = [1, 2, 1060, 24073, 3318, 3]
        num_frames = 2
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}
        total_mm = num_frames * (256 + 2)

        expanded, evs_ids = proc._expand_video_placeholders_in_token_ids(
            prompt, [total_mm], mm_data
        )
        # Still no EVS -> evs_ids must be None.
        assert evs_ids is None

        # MM token counts should match.
        assert expanded.count(img_ctx) == num_frames * 256
        assert expanded.count(500) == num_frames  # img_start
        assert expanded.count(501) == num_frames  # img_end

    def test_text_level_count_mismatch_raises(self):
        """If the decoded text has a different number of `<video>` occurrences
        than `mm_data["video"]`, the text-level path must raise a clear error."""
        proc = self._make_video_processor_multi_token()
        # Decoded text has zero `<video>` substrings.
        proc.tokenizer.decode = lambda ids, **kw: "no placeholder here"
        prompt = [1, 2, 3, 4]
        frames = [Image.new("RGB", (512, 512)) for _ in range(2)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}
        with pytest.raises(ValueError, match="contains 0 '<video>' placeholders"):
            proc._expand_video_placeholders_in_token_ids(prompt, [516], mm_data)


class TestExpandVideoPlaceholdersEVS:
    """Tests for the EVS (video pruning) path in the tokenized+MM fast path.

    Mirrors the non-fast-path behavior in `_process_video_prompts`: when
    `video_pruning_rate > 0`, `_expand_video_placeholders_in_token_ids`
    returns a tuple `(expanded_ids, evs_ids_tensor)` where `evs_ids_tensor`
    has one `video_context_token_id` per tubelet (wrapped with
    `<img>`/`</img>`), to be consumed by `merge_evs_mm_embeds` at forward
    time.
    """

    @staticmethod
    def _make_evs_video_processor():
        """Video processor with EVS enabled and a mocked
        `_compute_token_numbers_per_video` so tests don't depend on the
        full EVS retention-count math."""
        proc = _make_fast_path_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0.5
        proc.video_temporal_patch_size = 1
        # Pretend EVS will pre-budget 100 tokens to the first tubelet and 0
        # to the rest (this is the dummy shape the non-fast-path also uses).
        proc._compute_token_numbers_per_video = mock.Mock(return_value=[[100, 0, 0]])
        return proc

    def test_evs_token_level_returns_evs_ids_tensor(self):
        """Token-level match path: the `<video>` subsequence is present in
        `prompt_token_ids`, both `expanded_ids` and `evs_ids` tensors are
        produced in parallel."""
        proc = self._make_evs_video_processor()
        vid_ctx = proc.video_context_token_id  # 21 (single-ID in tests)
        img_ctx = proc.img_context_token_id  # 20
        num_frames = 3
        total_mm = 100 + 2 * num_frames  # 100 feature tokens + start/end per tubelet

        prompt = [1, 2, vid_ctx, 3]
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}

        expanded, evs_ids = proc._expand_video_placeholders_in_token_ids(
            prompt, [total_mm], mm_data
        )

        assert evs_ids is not None
        assert isinstance(evs_ids, torch.Tensor)
        assert evs_ids.dtype == torch.long

        # expanded: 100 img_context tokens in tubelet 1, 0 in tubelets 2/3,
        # plus img_start/end per tubelet.
        assert expanded.count(img_ctx) == 100
        assert expanded.count(500) == num_frames  # img_start per tubelet
        assert expanded.count(501) == num_frames  # img_end per tubelet
        # Surrounding non-MM tokens preserved at head/tail.
        assert expanded[:2] == [1, 2]
        assert expanded[-1] == 3

        # evs_ids: one video_context_token_id per tubelet (3 total),
        # each still wrapped with img_start/end; surrounding text identical.
        evs_list = evs_ids.tolist()
        assert evs_list.count(vid_ctx) == num_frames
        assert evs_list.count(500) == num_frames  # img_start
        assert evs_list.count(501) == num_frames  # img_end
        # evs_ids must have NO img_context_token_id (by design — merge_evs
        # substitutes retained counts at forward time).
        assert img_ctx not in evs_list
        # Surrounding prompt tokens preserved in evs_ids too.
        assert evs_list[:2] == [1, 2]
        assert evs_list[-1] == 3

    def test_evs_text_level_path(self):
        """Text-level path under EVS: multi-token `<video>` placeholder
        dispatches to decode/split/re-encode, which must still produce both
        streams in parallel."""
        proc = self._make_evs_video_processor()
        # Multi-token pattern -> dispatch routes through the text-level path.
        proc._video_placeholder_token_ids = [1060, 24073, 1062]

        # Mock decode/encode to simulate the BPE-merged reality.
        def mock_decode(ids, **kw):
            parts = []
            i = 0
            while i < len(ids):
                if ids[i : i + 3] == [1060, 24073, 3318]:
                    parts.append("<video>\n")
                    i += 3
                else:
                    parts.append(f"T{ids[i]} ")
                    i += 1
            return "".join(parts)

        def mock_encode(text, **kw):
            if kw.get("return_tensors") == "pt":
                return torch.tensor(list(range(len(text)))).unsqueeze(0)
            return [ord(c) % 1000 for c in text]

        proc.tokenizer.decode = mock_decode
        proc.tokenizer.encode = mock_encode

        vid_ctx = proc.video_context_token_id  # 21
        img_ctx = proc.img_context_token_id  # 20
        num_frames = 3

        # Prompt uses the "in-context" pattern (3318), not the "isolation"
        # pattern (1062). Length-based dispatch (len > 1) routes through
        # the text-level path unconditionally.
        prompt = [1, 2, 1060, 24073, 3318, 3]
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}
        total_mm = 100 + 2 * num_frames

        expanded, evs_ids = proc._expand_video_placeholders_in_token_ids(
            prompt, [total_mm], mm_data
        )

        assert evs_ids is not None
        assert isinstance(evs_ids, torch.Tensor)

        # MM counts in both streams must match the EVS-dummy shape.
        assert expanded.count(img_ctx) == 100
        assert expanded.count(500) == num_frames  # img_start
        assert expanded.count(501) == num_frames  # img_end

        evs_list = evs_ids.tolist()
        assert evs_list.count(vid_ctx) == num_frames
        assert evs_list.count(500) == num_frames  # img_start
        assert evs_list.count(501) == num_frames  # img_end
        assert img_ctx not in evs_list

    def test_dispatch_returns_evs_ids_in_mm_data_updates(self):
        """Integration test: with EVS on, the dispatch-level
        `expand_prompt_token_ids_for_mm` packages the evs_ids tensor into
        `{"video": {"evs_ids": tensor}}` in the second slot of the tuple."""
        proc = self._make_evs_video_processor()
        vid_ctx = proc.video_context_token_id
        num_frames = 2
        prompt = [1, vid_ctx, 2]
        # Override mock for 2-frame video.
        proc._compute_token_numbers_per_video = mock.Mock(return_value=[[50, 0]])
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=None)]}
        total_mm = 50 + 2 * num_frames

        _, mm_data_updates = proc.expand_prompt_token_ids_for_mm(
            prompt,
            [total_mm],
            mm_data=mm_data,
        )

        assert mm_data_updates is not None
        assert "video" in mm_data_updates
        assert "evs_ids" in mm_data_updates["video"]
        evs_tensor = mm_data_updates["video"]["evs_ids"]
        assert isinstance(evs_tensor, torch.Tensor)
        assert evs_tensor.dtype == torch.long
        # Expect `num_frames` video_context placeholders in evs_ids.
        assert (evs_tensor == vid_ctx).sum().item() == num_frames


class TestExpandPromptTokenIdsForMM:
    """Integration tests for the top-level dispatch method."""

    def test_dispatches_to_image(self):
        proc = _make_fast_path_processor()
        img_ctx = proc.img_context_token_id
        prompt = [1, img_ctx, 2]
        result, mm_data_updates = proc.expand_prompt_token_ids_for_mm(prompt, [5])
        assert mm_data_updates is None
        assert 500 in result  # img_start_token_id
        assert 501 in result  # img_end_token_id

    def test_dispatches_to_audio(self):
        proc = _make_fast_path_audio_processor()
        snd_ctx = proc._sound_context_token_id
        prompt = [1, snd_ctx, 2]
        result, mm_data_updates = proc.expand_prompt_token_ids_for_mm(prompt, [5])
        assert mm_data_updates is None
        assert 200 in result  # sound_start_token_id
        assert 201 in result  # sound_end_token_id

    def test_evs_dispatch_returns_image_evs_ids_in_mm_data_updates(self):
        proc = _make_fast_path_processor()
        proc.video_pruning_rate = 0.5
        img_ctx = proc.img_context_token_id
        prompt = [1, img_ctx, 2]

        result, mm_data_updates = proc.expand_prompt_token_ids_for_mm(
            prompt,
            [5],
            mm_data={"image": [object()]},
        )

        assert mm_data_updates is not None
        evs_ids = mm_data_updates["image"]["evs_ids"]
        assert isinstance(evs_ids, torch.Tensor)
        assert evs_ids.dtype == torch.long
        assert evs_ids.tolist() == result

    def test_evs_dispatch_returns_audio_evs_ids_in_mm_data_updates(self):
        proc = _make_fast_path_audio_processor()
        proc.video_pruning_rate = 0.5
        snd_ctx = proc._sound_context_token_id
        prompt = [1, snd_ctx, 2]

        result, mm_data_updates = proc.expand_prompt_token_ids_for_mm(
            prompt,
            [5],
            mm_data={"audio": [object()]},
        )

        assert mm_data_updates is not None
        evs_ids = mm_data_updates["audio"]["evs_ids"]
        assert isinstance(evs_ids, torch.Tensor)
        assert evs_ids.dtype == torch.long
        assert evs_ids.tolist() == result

    def test_no_placeholders_returns_unchanged(self):
        proc = _make_fast_path_processor()
        prompt = [1, 2, 3, 4]
        result, mm_data_updates = proc.expand_prompt_token_ids_for_mm(prompt, [])
        assert mm_data_updates is None
        assert result == prompt

    # Axis `modalities`: the ordered modality sequence of the mixed request;
    # the prompt, mm_data, prepared-data mocks, expected blocks, and metadata
    # are all derived from it in the body. `audio` needs the audio-enabled
    # factory; the image-video-image case uses the base factory.
    @pytest.mark.parametrize(
        "make_proc, modalities",
        [
            pytest.param(
                _make_fast_path_audio_processor,
                ["image", "video", "audio"],
                id="image_video_audio",
            ),
            pytest.param(
                _make_fast_path_processor,
                ["image", "video", "image"],
                id="image_video_image",
            ),
        ],
    )
    def test_call_mixed_builds_ordered_metadata(self, make_proc, modalities):
        proc = make_proc(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1
        img_ctx = proc.img_context_token_id
        snd_ctx = proc._sound_context_token_id
        frames = [Image.new("RGB", (512, 512))]

        # One prepared-data payload per DISTINCT modality, sized to its count.
        prepared = {
            "image": {"pixel_values": torch.ones(modalities.count("image"), 3, 2, 2)},
            "video": {"pixel_values": torch.ones(1, 3, 2, 2), "video_size": [[1, 1, 512, 512]]},
            "audio": {"input_audio_features": torch.ones(1, 4, 2)},
        }
        proc._prepare_image_modality_data = mock.Mock(return_value=prepared["image"])
        proc._prepare_video_modality_data = mock.Mock(return_value=prepared["video"])
        proc._prepare_audio_modality_data = mock.Mock(return_value=prepared["audio"])
        proc._get_num_tokens_for_item_order = mock.Mock(return_value=[5, 258, 5])

        placeholders = {
            "image": proc.img_context_token,
            "video": proc.video_context_token,
            "audio": AUDIO_PLACEHOLDER,
        }
        prompt = "".join(placeholders[m] for m in modalities)
        multi_modal_data: dict = {}
        for modality in modalities:
            if modality == "video":
                multi_modal_data.setdefault("video", []).append(
                    SimpleNamespace(frames=frames, metadata=None, audio=None)
                )
            else:
                multi_modal_data.setdefault(modality, []).append(object())
        inputs = {"prompt": prompt, "multi_modal_data": multi_modal_data}

        result, extra_inputs = proc(inputs, None)

        blocks = {
            "image": [500] + [img_ctx] * 3 + [501],
            "video": list(range(9)) + [500] + [img_ctx] * 256 + [501],
            "audio": [200] + [snd_ctx] * 3 + [201],
        }
        expected_result: list = []
        for modality in modalities:
            expected_result += blocks[modality]
        # item_order assigns each occurrence the next per-modality running index.
        seen_counts: dict = {}
        expected_item_order = []
        for modality in modalities:
            idx = seen_counts.get(modality, 0)
            expected_item_order.append({"modality": modality, "index": idx})
            seen_counts[modality] = idx + 1
        expected_modality_type = list(dict.fromkeys(modalities))

        assert result == expected_result
        multimodal_data = extra_inputs["multimodal_data"]
        assert multimodal_data["modality_type"] == expected_modality_type
        assert multimodal_data["multimodal_item_order"] == expected_item_order
        for modality in expected_modality_type:
            assert multimodal_data[modality] is prepared[modality]
        assert multimodal_data["multimodal_embedding_lengths"] == [3, 256, 3]


class TestMixedModalityAudioHoist:
    """The audio hoist: in a MIXED request, a video that carries embedded audio
    is promoted to a first-class top-level `audio` item.

    `_process_mixed_modalities` must, before running the order scanner:
      1. inject a `<sound_context>` placeholder after each video-with-audio in
         the text (the raw mixed prompt does NOT carry it — only the
         single-modality video path injects via `_extract_audio_from_video`), and
      2. register the extracted audio as a top-level `mm_data["audio"]` item,
    so the scanner emits `(video, k), (audio, k)` at the right ranks and
    `compute_mm_embedding_lengths` splits the budget into separate
    vision-only video + audio slots.
    """

    @staticmethod
    def _video_with_audio(num_audio_samples=16000):
        frames = [Image.new("RGB", (512, 512))]
        audio = AudioData(
            samples=np.random.randn(num_audio_samples).astype(np.float32),
            sample_rate=16000,
        )
        return SimpleNamespace(frames=frames, metadata=None, audio=audio)

    def _make_proc(self):
        proc = _make_fast_path_audio_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1
        return proc

    def test_primary_image_video_audio_interleave_order(self):
        # PRIMARY: image -> video(+audio) -> image -> video(+audio). The raw
        # prompt has NO <so_embedding>; the hoist must inject one per video and
        # promote both video audios so the order is the plain interleave with
        # the audio as a first-class slot AFTER each video.
        proc = self._make_proc()

        # vision-only video + audio per-slot counts the (mocked) ruler returns,
        # in prompt order: img, vid, aud, img, vid, aud.
        proc._get_num_tokens_for_item_order = mock.Mock(return_value=[5, 258, 5, 5, 258, 5])
        proc._prepare_image_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(2, 3, 2, 2)}
        )
        proc._prepare_video_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(2, 3, 2, 2), "video_size": [[1, 1, 512, 512]]}
        )
        proc._prepare_audio_modality_data = mock.Mock(
            return_value={"input_audio_features": torch.ones(2, 4, 2)}
        )

        v0 = self._video_with_audio()
        v1 = self._video_with_audio()
        prompt = "<image><video><image><video>"
        mm_data = {"image": [object(), object()], "video": [v0, v1]}

        _, extra = proc({"prompt": prompt, "multi_modal_data": mm_data}, None)
        multimodal_data = extra["multimodal_data"]

        # The hoist promoted both video audios into a first-class audio item, so
        # the order is the plain flat interleave with audio after each video.
        assert multimodal_data["multimodal_item_order"] == [
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
            {"modality": "audio", "index": 0},
            {"modality": "image", "index": 1},
            {"modality": "video", "index": 1},
            {"modality": "audio", "index": 1},
        ]
        assert multimodal_data["modality_type"] == ["image", "video", "audio"]
        # An audio payload is present (built from the hoisted video audios).
        assert "audio" in multimodal_data
        # The audio ruler was invoked for two hoisted audio items.
        prepared_audios = proc._prepare_audio_modality_data.call_args[0][0]
        assert len(prepared_audios) == 2

    def test_video_audio_budget_splits_into_separate_slots(self):
        # The video slot must be VISION-ONLY and the audio its own slot:
        # `_get_num_tokens_for_item_order` is passed video items WITHOUT the
        # embedded-audio budget, and the audio slot carries the audio budget.
        proc = self._make_proc()

        # Spy ruler: capture the item_order it is asked to size so we can assert
        # the (video, k) entry's video item carries NO embedded audio.
        captured = {}

        def spy_ruler(item_order, mm_data):
            captured["item_order"] = list(item_order)
            captured["mm_data_audio"] = mm_data.get("audio")
            # Return a deterministic per-slot count list. The video slot budget
            # is the full `get_num_tokens_per_video` count (256 context + 2 for
            # the `<img>`/`</img>` specials = 258), matching what the per-frame
            # video expansion emits; the audio slot is 4 (start + 2 context + end).
            return [5, 258, 4, 5, 258, 4]

        proc._get_num_tokens_for_item_order = mock.Mock(side_effect=spy_ruler)
        proc._prepare_image_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(2, 3, 2, 2)}
        )
        proc._prepare_video_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(2, 3, 2, 2), "video_size": [[1, 1, 512, 512]]}
        )
        proc._prepare_audio_modality_data = mock.Mock(
            return_value={"input_audio_features": torch.ones(2, 4, 2)}
        )

        v0 = self._video_with_audio()
        v1 = self._video_with_audio()
        prompt = "<image><video><image><video>"
        mm_data = {"image": [object(), object()], "video": [v0, v1]}

        _, extra = proc({"prompt": prompt, "multi_modal_data": mm_data}, None)
        multimodal_data = extra["multimodal_data"]

        # The ruler saw the audio hoisted into mm_data and the audio slots in the
        # order it was asked to size (so the video slots are vision-only).
        order_mods = [m for m, _ in captured["item_order"]]
        assert order_mods == ["image", "video", "audio", "image", "video", "audio"]
        assert captured["mm_data_audio"] is not None
        assert len(captured["mm_data_audio"]) == 2
        # Per-slot embedding lengths keep the vision video + audio split: 256
        # video-context embeds (258 budget minus the 2 `<img>`/`</img>` specials)
        # and 2 audio-context embeds (4 budget minus the 2 sound specials).
        lengths = multimodal_data["multimodal_embedding_lengths"]
        assert len(lengths) == 6

    def test_no_hoist_when_video_has_no_audio(self):
        # A mixed image+video with NO embedded audio must NOT synthesize an audio
        # item or inject <so_embedding> (no phantom slot).
        proc = self._make_proc()
        # Video budget is the full `get_num_tokens_per_video` count (256 context
        # + 2 `<img>`/`</img>` specials = 258), matching the per-frame expansion.
        proc._get_num_tokens_for_item_order = mock.Mock(return_value=[5, 258])
        proc._prepare_image_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(1, 3, 2, 2)}
        )
        proc._prepare_video_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(1, 3, 2, 2), "video_size": [[1, 1, 512, 512]]}
        )
        proc._prepare_audio_modality_data = mock.Mock()

        video_no_audio = SimpleNamespace(
            frames=[Image.new("RGB", (512, 512))], metadata=None, audio=None
        )
        mm_data = {"image": [object()], "video": [video_no_audio]}
        _, extra = proc({"prompt": "<image><video>", "multi_modal_data": mm_data}, None)
        multimodal_data = extra["multimodal_data"]

        assert "audio" not in multimodal_data["modality_type"]
        assert "audio" not in multimodal_data
        proc._prepare_audio_modality_data.assert_not_called()
        assert multimodal_data["multimodal_item_order"] == [
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
        ]


class TestGetNumTokensPerAudio:
    @pytest.mark.parametrize(
        "wrap_with_sample_rate",
        [
            pytest.param(False, id="bare_array"),
            pytest.param(True, id="tuple_with_matching_sample_rate"),
        ],
    )
    def test_bare_array_and_sample_rate_tuple(self, wrap_with_sample_rate):
        proc = _make_audio_processor()
        audio = np.random.randn(16000).astype(np.float32)
        audio_input = (
            (audio, proc._audio_extractor.sampling_rate) if wrap_with_sample_rate else audio
        )
        expected = proc._audio_extractor.audio_token_count(16000) + 2
        assert proc.get_num_tokens_per_audio(audio=audio_input) == expected

    def test_no_audio_config_raises(self):
        # Kept separate: uses `_make_processor` (no sound_config) — fundamentally
        # different setup from the happy-path cases above.
        proc = _make_processor()
        with pytest.raises(ValueError, match="sound_config"):
            proc.get_num_tokens_per_audio(audio=np.zeros(100))


class TestGetNumTokensPerVideoInvariants:
    """A regression test for the exact invariant that was broken before the
    `get_num_tokens_per_video` fix: `find_mm_token_lengths` (via
    `get_num_tokens_per_video`) and `_compute_token_numbers_per_video` must
    report counts that agree on per-video totals, because
    `_find_mm_token_start_pos_from_masks` later asserts
    `len(mm_positions) == sum(num_mm_tokens)`, where `len(mm_positions)` is
    driven by the expansion (built from `_compute_token_numbers_per_video`)
    and `sum(num_mm_tokens)` comes from `find_mm_token_lengths` /
    `get_num_tokens_per_video`.

    Before the fix, `get_num_tokens_per_video`'s EVS branch routed the
    frame through `get_num_tokens_per_image`'s dynamic-tiler logic, which
    returned a multi-block count that the vision encoder never actually
    produces — so `find_mm_token_lengths` over-reported the total and the
    fast path's `_find_mm_token_start_pos_from_masks` assertion failed.

    This test parametrizes over the knobs that gate branching in both
    functions (`video_target_num_patches`, `video_maintain_aspect_ratio`)
    plus the EVS pruning rate and a couple of frame aspect ratios. If
    anyone ever re-introduces a divergence between the two computation
    paths, this test catches it.
    """

    @pytest.mark.parametrize("video_pruning_rate", [0.0, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("video_target_num_patches", [None, 256, 1024])
    @pytest.mark.parametrize("maintain_aspect_ratio", [False, True])
    @pytest.mark.parametrize("frame_dims", [(512, 512), (1920, 1080), (640, 360)])
    def test_get_num_tokens_per_video_agrees_with_compute_token_numbers_per_video(
        self,
        video_pruning_rate,
        video_target_num_patches,
        maintain_aspect_ratio,
        frame_dims,
    ):
        proc = _make_nano_processor(
            sound_config=None,
            video_target_num_patches=video_target_num_patches,
            video_maintain_aspect_ratio=maintain_aspect_ratio,
            video_temporal_patch_size=2,
        )
        proc.video_pruning_rate = video_pruning_rate

        num_frames = 10  # with T=2 -> 5 tubelets
        w, h = frame_dims
        frames = [Image.new("RGB", (w, h)) for _ in range(num_frames)]

        video_size = proc._compute_video_shape_descriptor(frames)
        per_tubelet = proc._compute_token_numbers_per_video([video_size])[0]
        num_tubelets = len(per_tubelet)
        # `get_num_tokens_per_video` hardcodes
        # `num_special_tokens_per_frame = 2` (<img> and </img>) and adds it
        # per tubelet on top of the retained feature tokens.
        expected_total = sum(per_tubelet) + num_tubelets * 2
        actual_total = proc.get_num_tokens_per_video(video=frames)
        assert actual_total == expected_total, (
            f"Drift between get_num_tokens_per_video={actual_total} and "
            f"sum(_compute_token_numbers_per_video) + 2*num_tubelets="
            f"{expected_total} "
            f"(pruning={video_pruning_rate}, "
            f"target_patches={video_target_num_patches}, "
            f"maintain_aspect_ratio={maintain_aspect_ratio}, "
            f"frame_dims={frame_dims})"
        )


class TestMultiTokenWrappersConsistency:
    """End-to-end regression for multi-token BPE wrappers (`<img>`, `</img>`).

    `get_num_tokens_per_image` adds `len(_img_start_token_ids) +
    len(_img_end_token_ids)` so multi-token BPE wrappers are correctly counted.
    For that count to line up at the assertion in `_find_mm_token_start_pos_from_masks`
    (`len(mm_positions) == sum(num_mm_tokens)`), `get_mm_special_token_ids`
    must include **every** ID in those wrapper lists — not just the [0]
    aliases. Otherwise trailing wrapper tokens are seen as plain text and the
    counts diverge.
    """

    @staticmethod
    def _make_multi_token_wrapper_processor():
        """Processor whose <img>/</img> tokenize to multi-element BPE lists.

        Real production tokenizers usually register these as single added
        special tokens, but this exercises the BPE-fallback case the class
        explicitly accounts for in __init__ ("These may be multi-token under
        BPE, so we store the full ID list.").
        """
        proc = _make_processor()
        # Multi-token BPE encodings — chosen to be distinct from
        # img_context_token_id (20), video_context_token_id (21), and any
        # plausible mock-tokenizer text-encoding range.
        proc._img_start_token_ids = [9100, 9101, 9102]
        proc._img_end_token_ids = [9200, 9201]
        proc.image_start_token_id = proc._img_start_token_ids[0]
        proc.image_end_token_id = proc._img_end_token_ids[0]
        return proc

    def test_get_mm_special_token_ids_includes_all_wrapper_tokens(self):
        proc = self._make_multi_token_wrapper_processor()
        special_ids = set(proc.get_mm_special_token_ids().tolist())
        # Every BPE token in the wrapper lists must be present.
        for tok in proc._img_start_token_ids + proc._img_end_token_ids:
            assert tok in special_ids, (
                f"get_mm_special_token_ids missing wrapper token {tok}; "
                f"returned={sorted(special_ids)}"
            )
        # The single-ID alias is no longer authoritative on its own, but it
        # must remain present (it's a member of the full list anyway).
        assert proc.image_start_token_id in special_ids
        assert proc.image_end_token_id in special_ids

    def test_mm_token_start_pos_matches_get_num_tokens_per_image(self):
        """End-to-end invariant: with multi-token wrappers, the count
        reported by get_num_tokens_per_image must equal the number of mm
        positions _find_mm_token_start_pos_from_masks extracts from the expanded prompt.
        Before the get_mm_special_token_ids fix, only the first BPE token of
        each wrapper was masked as special and this assertion was off by
        `(len(start) - 1) + (len(end) - 1)` per image."""
        proc = self._make_multi_token_wrapper_processor()
        img_ctx = proc.img_context_token_id  # 20

        # Build a single-image prompt: [text..., <image>, text...]
        prompt = [1, 2, img_ctx, 3]

        # Use the real per-image count (includes feature tokens + wrappers).
        num_per_image = proc.get_num_tokens_per_image(image=Image.new("RGB", (320, 320)))
        expanded = proc._expand_image_placeholders_in_token_ids(prompt, [num_per_image])

        # Mirror the fast path's mask + start-position computation. The
        # legacy public `find_mm_token_positions` was split into private
        # helpers `_compute_mm_masks` + `_find_mm_token_start_pos_from_masks`;
        # the latter still owns the `mm_positions.numel() == sum(num_mm_tokens)`
        # assertion that triggers when wrappers BPE-split and only the [0]
        # alias is in `mm_special_token_ids`.
        input_ids_tensor = torch.tensor(expanded)
        mm_mask, _embed_mask, special_mask = _compute_mm_masks(
            input_ids=input_ids_tensor,
            vocab_size=None,
            mm_token_ids=proc.get_mm_token_ids(),
            mm_special_token_ids=proc.get_mm_special_token_ids(),
        )
        start_positions, _ = _find_mm_token_start_pos_from_masks(
            mm_mask=mm_mask,
            special_mask=special_mask,
            num_mm_tokens=[num_per_image],
        )

        # _find_mm_token_start_pos_from_masks internally asserts
        #   mm_positions.numel() == sum(num_mm_tokens)
        # so reaching here without an exception is the regression check. Also
        # sanity-check the start position lands at the expected offset.
        assert len(start_positions) == 1
        # The image expansion starts after the two leading prompt tokens.
        assert start_positions[0] == 2


@pytest.mark.skipif(not importlib.util.find_spec("librosa"), reason="librosa not installed")
class TestFastPathVideoWithExtractedAudio:
    """Fast-path-only tests for handling audio extracted from video.

    These exercise the tokenized+MM fast path
    (`_expand_video_placeholders_in_token_ids` and the `video_audio` kwarg of
    `get_num_tokens_per_video`). The slow text path (`_extract_audio_from_video`
    -> `_prepare_audio_features` -> `_expand_audio_placeholders`) is covered by
    `TestAudioInputProcessor`.

    When `--media_io_kwargs '{"video": {"extract_audio": true}}'` is set, the
    video loader stores the extracted stream as `VideoData.audio`.
    The fast-path video branch must append `<so_start><so_embedding>*M<so_end>`
    after the per-video frame tokens so the prompt has placeholder slots for the
    audio embeddings. The audio is hoisted to a first-class `(audio, k)` item and
    its embeddings are placed by the assembly's per-item scatter at forward time.
    `get_num_tokens_per_video` must include those tokens in its count so
    `find_mm_token_lengths` reports a total that matches the actual mm-token
    count in the tokenized prompt (otherwise `_find_mm_token_start_pos_from_masks` asserts
    `sum(num_mm_tokens) == len(mm_positions)` and the request fails).
    """

    @staticmethod
    def _make_video_audio_processor():
        """Audio-enabled fast-path processor with EVS off and predictable counts.

        With metadata-driven frame separators ("Frame 1 sampled at 0.00 seconds: "
        ~ 33 chars), the default mock encoder `list(range(len(text)))` produces
        token IDs that collide with mm-token IDs like img_context_token_id=20 and
        _sound_context_token_id=30 — leading to spurious mm-token matches in the
        expansion. Shift the encoded range to 1000+ so text tokens never overlap
        with any mm-token id used in this suite (20, 21, 30, 200, 201, 500, 501).
        """
        proc = _make_fast_path_audio_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1

        def shifted_encode(text, **kw):
            ids = [1000 + i for i in range(len(text))]
            if kw.get("return_tensors") == "pt":
                return torch.tensor(ids).unsqueeze(0)
            return ids

        proc.tokenizer.encode = mock.Mock(side_effect=shifted_encode)
        return proc

    @staticmethod
    def _video_metadata(num_frames=2):
        """Build video metadata matching the cv2 video loader.
        `frames_indices` length must equal the actual number of frames passed
        to the processor — otherwise `_get_frame_separators` produces a
        separator list that mismatches `_compute_token_numbers_per_video`'s
        output and the strict zip in the per-tubelet loop fails."""
        return {
            "total_num_frames": num_frames,
            "fps": 30.0,
            "duration": num_frames / 30.0,
            "frames_indices": list(range(num_frames)),
        }

    @staticmethod
    def _audio_data(num_samples=16000, sample_rate=16000):
        """Build structured video audio matching `VideoData.audio`."""
        return AudioData(
            samples=np.random.randn(num_samples).astype(np.float32), sample_rate=sample_rate
        )

    def test_fast_path_appends_audio_tokens_after_video(self):
        """When VideoData carries audio, the per-video expansion must end
        with `<so_start>` + `<so_embedding>` * M + `<so_end>`, where M is the
        Parakeet-extractor's audio_token_count for the resampled audio length."""
        proc = self._make_video_audio_processor()
        vid_ctx = proc.video_context_token_id
        img_ctx = proc.img_context_token_id
        num_frames = 2

        prompt = [1, 2, vid_ctx, 3]
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata = self._video_metadata(num_frames=num_frames)
        audio = self._audio_data()
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=metadata, audio=audio)]}

        # Match accounting in get_num_tokens_per_video: per-frame video tokens
        # (256 + <img>/</img>) plus audio M+2 (start/end).
        per_frame = 256 + 2
        audio_total = proc.get_num_tokens_per_audio(
            audio=(audio.samples, audio.sample_rate)
        )  # = M + 2
        total_mm = num_frames * per_frame + audio_total

        expanded, evs_ids = proc._expand_video_placeholders_in_token_ids(
            prompt, [total_mm], mm_data
        )

        assert evs_ids is None  # EVS disabled
        # Surrounding text preserved.
        assert expanded[:2] == [1, 2]
        assert expanded[-1] == 3

        # The audio block is appended AFTER the video frames, so the last 4
        # tokens of the MM region are: <so_start>, ..., <so_context>, <so_end>, 3.
        # Find audio block boundaries.
        sound_start_id = proc._sound_start_token_id
        sound_end_id = proc._sound_end_token_id
        sound_ctx_id = proc._sound_context_token_id

        assert expanded.count(sound_start_id) == 1
        assert expanded.count(sound_end_id) == 1
        # Audio block must form a single contiguous run between start and end.
        start_idx = expanded.index(sound_start_id)
        end_idx = expanded.index(sound_end_id)
        assert end_idx > start_idx
        audio_context_run = expanded[start_idx + 1 : end_idx]
        assert all(t == sound_ctx_id for t in audio_context_run)
        assert len(audio_context_run) == audio_total - 2  # M context tokens
        # Audio block must come AFTER all video frame tokens (verify the last
        # img_end appears before <so_start>).
        last_img_end = max(i for i, t in enumerate(expanded) if t == 501)
        assert last_img_end < start_idx
        # And no audio tokens before the video region.
        assert expanded.index(img_ctx) < start_idx

    def test_fast_path_no_audio_tokens_when_video_has_no_audio(self):
        """Regression check: video without `audio` must not
        emit any audio token IDs (otherwise we'd inject phantom slots that the
        encoder produces no embeddings for)."""
        proc = self._make_video_audio_processor()
        vid_ctx = proc.video_context_token_id
        num_frames = 2

        prompt = [1, vid_ctx, 2]
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata_no_audio = self._video_metadata(num_frames=num_frames)
        mm_data = {
            "video": [SimpleNamespace(frames=frames, metadata=metadata_no_audio, audio=None)]
        }
        total_mm = num_frames * (256 + 2)

        expanded, _ = proc._expand_video_placeholders_in_token_ids(prompt, [total_mm], mm_data)

        assert proc._sound_start_token_id not in expanded
        assert proc._sound_end_token_id not in expanded
        assert proc._sound_context_token_id not in expanded

    def test_get_num_tokens_per_video_includes_audio_when_video_has_audio(self):
        """`get_num_tokens_per_video` must add `M + 2` to its return value when
        the video carries extracted audio. Without this,
        `find_mm_token_lengths` under-reports total mm-tokens and the fast path's
        `_find_mm_token_start_pos_from_masks` assertion fires."""
        proc = self._make_video_audio_processor()
        num_frames = 2
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata = self._video_metadata(num_frames=num_frames)
        audio = self._audio_data()

        video_only = proc.get_num_tokens_per_video(video=frames)
        with_audio = proc.get_num_tokens_per_video(
            video=frames, video_metadata=metadata, video_audio=audio
        )
        audio_alone = proc.get_num_tokens_per_audio(audio=(audio.samples, audio.sample_rate))

        assert with_audio == video_only + audio_alone
        # Without structured audio, the count must
        # match the video-only path exactly (no implicit audio accounting).
        assert proc.get_num_tokens_per_video(video=frames, video_metadata=None) == video_only
        assert (
            proc.get_num_tokens_per_video(video=frames, video_metadata={"fps": 30.0}) == video_only
        )

    def test_expansion_length_matches_get_num_tokens_per_video(self):
        """End-to-end invariant: the number of mm-tokens the fast-path expansion
        emits for a single video must equal what `get_num_tokens_per_video` /
        `find_mm_token_lengths` reports for that video — both with and without
        extracted audio. If these drift, `_find_mm_token_start_pos_from_masks`'s
        `sum(num_mm_tokens) == len(mm_positions)` assertion fails."""
        proc = self._make_video_audio_processor()
        vid_ctx = proc.video_context_token_id
        num_frames = 2
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata = self._video_metadata(num_frames=num_frames)
        audio = self._audio_data()
        mm_data = {"video": [SimpleNamespace(frames=frames, metadata=metadata, audio=audio)]}

        declared = proc.get_num_tokens_per_video(
            video=frames, video_metadata=metadata, video_audio=audio
        )
        prompt = [1, vid_ctx, 2]
        expanded, _ = proc._expand_video_placeholders_in_token_ids(prompt, [declared], mm_data)

        # Count mm-token IDs present in the expansion (mirrors
        # _find_mm_token_start_pos_from_masks's mask, which catches both context tokens and
        # special wrappers via mm_token_ids / mm_special_token_ids).
        mm_token_ids = {
            proc.img_context_token_id,
            proc._sound_context_token_id,
            500,  # <img> from _make_fast_path_audio_processor
            501,  # </img>
            proc._sound_start_token_id,
            proc._sound_end_token_id,
        }
        actual_mm = sum(1 for t in expanded if t in mm_token_ids)
        assert actual_mm == declared


@pytest.mark.skipif(not importlib.util.find_spec("librosa"), reason="librosa not installed")
# `torch.compile` (used inside audio feature extraction) spins up a thread pool;
# disable the leak check as the existing audio tests do (see TestAudioInputProcessor).
@pytest.mark.threadleak(enabled=False)
class TestVideoWithEmbeddedAudioHoist:
    """Light guard that the REAL Nano processor hoists video-embedded audio into
    a first-class top-level audio item, without model weights or a GPU.

    `_prepare_video_modality_data` is vision-only (no nested audio); a single
    video-with-audio request routes through `_process_mixed_modalities`, which
    injects `<sound_context>`, promotes the audio to a standalone audio payload,
    and emits a `(video, 0), (audio, 0)` prompt order. The per-item extractor
    contract over the resulting payload is pinned in test_nano_encode_extractor.py.
    """

    @staticmethod
    def _make_proc():
        # Configure video_target_num_patches so _process_videos_frames uses the
        # real video_to_pixel_values path (the HF image-processor fallback is a
        # Mock here and is not subscriptable).
        proc = _make_fast_path_audio_processor(max_num_patches=256, min_num_patches=4)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1
        proc.video_target_num_patches = 256
        proc.video_maintain_aspect_ratio = True
        return proc

    @staticmethod
    def _video_with_audio(num_frames=2, num_audio_samples=16000):
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata = {
            "total_num_frames": num_frames,
            "fps": 30.0,
            "duration": num_frames / 30.0,
            "frames_indices": list(range(num_frames)),
        }
        audio = AudioData(
            samples=np.random.randn(num_audio_samples).astype(np.float32),
            sample_rate=16000,
        )
        return SimpleNamespace(frames=frames, metadata=metadata, audio=audio)

    def test_prepare_video_modality_data_is_vision_only(self):
        # The video payload no longer nests audio — the audio is hoisted to a
        # top-level item, so the video modality data carries only vision fields.
        proc = self._make_proc()
        video = self._video_with_audio()

        modality_data = proc._prepare_video_modality_data([video])

        assert "pixel_values" in modality_data
        assert "video_size" in modality_data
        assert "audio" not in modality_data, "video payload must not nest audio (hoisted)"

    def test_hoist_injects_sound_context_and_builds_audio_list(self):
        # `_hoist_video_embedded_audio` injects `<sound_context>` after the video
        # placeholder and extends mm_data["audio"] with the video's audio.
        proc = self._make_proc()
        video = self._video_with_audio()
        mm_data = {"video": [video]}

        hoisted_text, hoisted_mm_data = proc._hoist_video_embedded_audio("<video>", mm_data)

        assert hoisted_text == "<video>" + AUDIO_PLACEHOLDER
        assert len(hoisted_mm_data["audio"]) == 1
        # The original mm_data is not mutated.
        assert "audio" not in mm_data

    def test_no_hoist_when_video_has_none(self):
        proc = self._make_proc()
        video = self._video_with_audio()
        video.audio = None
        mm_data = {"video": [video]}

        hoisted_text, hoisted_mm_data = proc._hoist_video_embedded_audio("<video>", mm_data)

        assert hoisted_text == "<video>"
        assert "audio" not in hoisted_mm_data

    def test_video_embedded_audios_empty_without_extractor(self):
        # The shared helper short-circuits to [] when no audio extractor is
        # configured, so the promote/hoist early-outs both see "no audio".
        proc = self._make_proc()
        proc._audio_extractor = None
        mm_data = {"video": [self._video_with_audio()]}

        assert proc._video_embedded_audios(mm_data) == []

    def test_video_embedded_audios_returns_per_video_audio_or_none(self):
        # With an extractor, the helper yields one entry per video: the embedded
        # AudioData or None, preserving order.
        proc = self._make_proc()
        with_audio = self._video_with_audio()
        without_audio = self._video_with_audio()
        without_audio.audio = None
        mm_data = {"video": [with_audio, without_audio]}

        audios = proc._video_embedded_audios(mm_data)

        assert len(audios) == 2
        assert audios[0] is with_audio.audio
        assert audios[1] is None

    def test_promote_is_idempotent(self):
        # Re-promoting already-promoted data is a no-op: the hoisted video copies
        # have `.audio` cleared, so the helper yields all-None and promote
        # early-outs (returns the same object).
        proc = self._make_proc()
        mm_data = {"video": [self._video_with_audio()]}

        promoted = proc.promote_nested_mm_data(mm_data)
        assert proc._video_embedded_audios(promoted) == [None]

        repromoted = proc.promote_nested_mm_data(promoted)
        assert repromoted is promoted

    def test_full_pipeline_emits_video_audio_plan(self):
        # End-to-end through the REAL processor __call__ with the REAL hoist: a
        # single video-with-audio yields a (video, 0), (audio, 0) prompt order
        # and a top-level audio payload. The heavy per-modality prep + token
        # ruler are mocked (orthogonal to the hoist + order scanning under test).
        proc = self._make_proc()
        proc._prepare_video_modality_data = mock.Mock(
            return_value={"pixel_values": torch.ones(2, 3, 2, 2), "video_size": [[1, 1, 512, 512]]}
        )
        proc._prepare_audio_modality_data = mock.Mock(
            return_value={"input_audio_features": torch.ones(2, 4, 2)}
        )
        # Per-slot budgets must equal the multimodal-token count the REAL
        # `_expand_mixed_prompt_text_for_mm` emits (the only un-mocked count path
        # here), so `compute_mm_embedding_lengths` reconciles. For this proc
        # (`video_target_num_patches=256`, 2 frames) the video slot is 136 and the
        # audio slot is 6. The video count is 132 vision tokens (2 tubelets x [64
        # context + 2 `<img>`/`</img>` specials]) plus 4 from this suite's Mock
        # tokenizer encoding the long per-frame separator strings to id ranges
        # that happen to include the context-token ids (a test-tokenizer artifact,
        # not production behavior); the real `get_num_tokens_per_video` /
        # `_compute_token_numbers_per_video` agreement is pinned separately by
        # `test_get_num_tokens_per_video_agrees_with_compute_token_numbers_per_video`.
        proc._get_num_tokens_for_item_order = mock.Mock(return_value=[136, 6])

        video = self._video_with_audio()
        inputs = {"prompt": "<video>", "multi_modal_data": {"video": [video]}}

        _, extra = proc(inputs, None)
        multimodal_data = extra["multimodal_data"]

        assert multimodal_data["modality_type"] == ["video", "audio"]
        assert multimodal_data["multimodal_item_order"] == [
            {"modality": "video", "index": 0},
            {"modality": "audio", "index": 0},
        ]
        # An audio payload was built from the hoisted video audio.
        assert "audio" in multimodal_data
        # The hoisted audio was prepared via the standalone audio path.
        prepared_audios = proc._prepare_audio_modality_data.call_args[0][0]
        assert len(prepared_audios) == 1
        # Two slots (video + audio) in the per-item embedding lengths.
        assert len(multimodal_data["multimodal_embedding_lengths"]) == 2


class TestTokenPathVideoEmbeddedAudioHoist:
    """Close the pre-tokenized (`call_with_token_ids`) path for the video-embedded
    audio hoist, so the feature is complete for BOTH the text and token paths.

    The token-path contract mirrors the text path: the pre-tokenized prompt
    carries a `<sound_context>` placeholder after each video-with-audio (the
    natural analogue of the text path injecting one). `mm_data` still nests the
    audio on the `VideoData` (not yet hoisted). The processor's
    `promote_nested_mm_data` hook must, before `find_mm_token_lengths` and the
    token-path order resolution (the model's `get_mm_item_order`) run:
      1. register the embedded audio as a top-level `mm_data["audio"]` item, and
      2. vision-only-strip `.audio` off the video items,
    so the order scanner emits `(video, k), (audio, k)` and the budget splits
    into a vision-only video slot + a separate audio slot — the SAME per-item
    totals the text path produces (no double-counted audio on the video).
    """

    @staticmethod
    def _make_proc():
        # Shift the mock tokenizer's text-token id range to 1000+ so encoded
        # frame-separator text never collides with the mm-token ids used here
        # (20, 21, 30, 200, 201, 500, 501) — see TestFastPathVideoWithExtractedAudio.
        proc = _make_fast_path_audio_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1

        def shifted_encode(text, **kw):
            ids = [1000 + i for i in range(len(text))]
            if kw.get("return_tensors") == "pt":
                return torch.tensor(ids).unsqueeze(0)
            return ids

        proc.tokenizer.encode = mock.Mock(side_effect=shifted_encode)
        return proc

    @staticmethod
    def _video_with_audio(num_frames=2, num_audio_samples=16000):
        frames = [Image.new("RGB", (512, 512)) for _ in range(num_frames)]
        metadata = {
            "total_num_frames": num_frames,
            "fps": 30.0,
            "duration": num_frames / 30.0,
            "frames_indices": list(range(num_frames)),
        }
        audio = AudioData(
            samples=np.random.randn(num_audio_samples).astype(np.float32),
            sample_rate=16000,
        )
        # A real VideoData (not SimpleNamespace) so find_mm_token_lengths unwraps
        # `.frames`/`.metadata`/`.audio` via its isinstance(item, VideoData) path.
        return VideoData(frames=frames, metadata=metadata, audio=audio)

    def test_video_embedded_audio_resolves_and_expands_via_token_ids(self):
        # A video-with-embedded-audio request through `call_with_token_ids`:
        #   (a) the resolved prompt order contains (video, 0) then (audio, 0), and
        #   (b) the expanded MM-token count == vision-only video budget + the one
        #       audio budget (the SAME total the text path produces).
        proc = self._make_proc()
        vid_ctx = proc.video_context_token_id
        snd_ctx = proc._sound_context_token_id
        img_ctx = proc.img_context_token_id
        so_start = proc._sound_start_token_id
        so_end = proc._sound_end_token_id

        video = self._video_with_audio()
        audio = video.audio
        num_frames = len(video.frames)

        # The vision-only video budget + the one-audio budget the REAL ruler
        # reports once the audio is hoisted off the video (matches the text path).
        vision_video_budget = proc.get_num_tokens_per_video(
            video=video.frames, video_metadata=video.metadata, video_audio=None
        )
        audio_budget = proc.get_num_tokens_per_audio(audio=(audio.samples, audio.sample_rate))

        # The pre-tokenized prompt carries `<video>` then `<sound_context>` (one
        # placeholder per mm item, in prompt order — the token-path contract).
        prompt_token_ids = [1, 2, vid_ctx, snd_ctx, 3]
        # mm_data still NESTS the audio on the VideoData (pre-hoist), modality-major.
        mm_data = {"video": [video]}

        # The dummy-placeholder pass (`_process_multimodal_with_dummy_placeholders`)
        # re-enters the text path on synthetic placeholder text; mock its heavy
        # per-modality prep + ruler so it stays CPU-only and orthogonal to the
        # token-path resolve/expand under test.
        proc._prepare_video_modality_data = mock.Mock(
            return_value={
                "pixel_values": torch.ones(num_frames, 3, 2, 2),
                "video_size": [[1, 1, 512, 512]],
            }
        )
        proc._prepare_audio_modality_data = mock.Mock(
            return_value={"input_audio_features": torch.ones(2, 4, 2)}
        )
        proc._get_num_tokens_for_item_order = mock.Mock(
            return_value=[vision_video_budget, audio_budget]
        )

        # Capture the prompt order the token path resolves (assertion a). The
        # framework `resolve_order` is now metadata-only, so the token-id fast
        # path derives the interleaved order from the model's own
        # `get_mm_item_order` hook; capture there.
        resolved_orders = []
        real_get_order = type(proc).get_mm_item_order

        def capturing_get_order(self, prompt_token_ids, mm):
            order = real_get_order(self, prompt_token_ids, mm)
            resolved_orders.append([tuple(entry) for entry in order])
            return order

        with mock.patch.object(
            type(proc),
            "get_mm_item_order",
            capturing_get_order,
        ):
            expanded_ids, _ = proc.call_with_token_ids(
                {"prompt_token_ids": prompt_token_ids, "multi_modal_data": mm_data},
                None,
            )

        # (a) The token path resolved the prompt order as video-then-audio, each
        # a first-class item.
        assert resolved_orders, "model get_mm_item_order was not called"
        token_path_order = resolved_orders[-1]
        assert token_path_order == [("video", 0), ("audio", 0)]

        # (b) The expansion replaced the two placeholders with exactly the
        # vision-only video budget + the one audio budget — the same total the
        # text path produces. Count the mm-token ids in the expanded stream.
        mm_token_ids = {img_ctx, snd_ctx, so_start, so_end, 500, 501}
        actual_mm = sum(1 for t in expanded_ids if t in mm_token_ids)
        assert actual_mm == vision_video_budget + audio_budget
        # And exactly ONE audio run (no double-injection from the video path).
        assert expanded_ids.count(so_start) == 1
        assert expanded_ids.count(so_end) == 1
        assert expanded_ids.count(snd_ctx) == audio_budget - 2
        # Vision-only video region present; the `<video>` placeholder consumed.
        assert expanded_ids.count(img_ctx) == vision_video_budget - num_frames * 2
        assert expanded_ids.count(vid_ctx) == 0

    def test_no_promotion_when_video_has_no_embedded_audio(self):
        # Regression: a token-path video WITHOUT embedded audio must NOT
        # synthesize a top-level audio item (the hook is a no-op).
        proc = self._make_proc()
        video = self._video_with_audio()
        video.audio = None

        mm_data = {"video": [video]}
        promoted = proc.promote_nested_mm_data(mm_data)

        assert "audio" not in promoted
        # The original mm_data is returned unchanged (no copy needed).
        assert promoted is mm_data

    def test_promotion_rejects_standalone_plus_embedded_audio(self):
        # Mixing standalone audio with video-embedded audio is ambiguous and must
        # raise (mirrors the text-path `_hoist_video_embedded_audio` guard).
        proc = self._make_proc()
        video = self._video_with_audio()
        standalone = AudioData(samples=np.random.randn(16000).astype(np.float32), sample_rate=16000)
        mm_data = {"video": [video], "audio": [(standalone.samples, standalone.sample_rate)]}

        with pytest.raises(NotImplementedError, match="standalone audio"):
            proc.promote_nested_mm_data(mm_data)


class TestTokenPathBakesPromptItemOrder:
    """The pre-tokenized (`call_with_token_ids`) path must bake the REAL
    prompt-item order into `extra_processed_inputs["multimodal_data"]`.

    The dummy-placeholder pass runs the text path on synthetic placeholder text
    built modality-major by `get_text_with_mm_placeholders` (all images, then
    videos, then audios). For a genuinely interleaved prompt that order is
    wrong, so the token-id path must overwrite it with the order resolved from
    the real prompt token IDs. Without that bake, the hashing wrapper inherits
    the stale modality-major dummy order and the chunk layout no longer follows
    the prompt.
    """

    @staticmethod
    def _make_proc():
        proc = _make_fast_path_audio_processor(video_target_num_patches=None)
        proc._add_video_prefix = False
        proc.video_pruning_rate = 0
        proc.video_temporal_patch_size = 1

        def shifted_encode(text, **kw):
            ids = [1000 + i for i in range(len(text))]
            if kw.get("return_tensors") == "pt":
                return torch.tensor(ids).unsqueeze(0)
            return ids

        proc.tokenizer.encode = mock.Mock(side_effect=shifted_encode)
        return proc

    def test_interleaved_image_video_image_bakes_real_prompt_order(self):
        # Real prompt: image, video, image -> [(image,0),(video,0),(image,1)].
        # The modality-major dummy text would resolve [(image,0),(image,1),
        # (video,0)] instead; the token-id bake must win.
        proc = self._make_proc()
        img_ctx = proc.img_context_token_id
        vid_ctx = proc.video_context_token_id

        frames = [Image.new("RGB", (64, 64)) for _ in range(2)]
        metadata = {
            "total_num_frames": 2,
            "fps": 30.0,
            "duration": 2 / 30.0,
            "frames_indices": [0, 1],
        }
        video = VideoData(frames=frames, metadata=metadata, audio=None)
        mm_data = {
            "image": [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
            "video": [video],
        }
        prompt_token_ids = [1, img_ctx, 2, vid_ctx, 3, img_ctx, 4]

        # Keep the dummy-placeholder pass CPU-only and orthogonal to the
        # order-resolution under test. `num_tokens_per_image` is required on the
        # dynamic-tiler path: `_process_mixed_modalities` reconciles the deferred
        # image slots from it (one entry per image; values are irrelevant here).
        proc._prepare_image_modality_data = mock.Mock(
            return_value={
                "pixel_values": torch.ones(1, 3, 2, 2),
                "num_tokens_per_image": [4, 4],
            }
        )
        proc._prepare_video_modality_data = mock.Mock(
            return_value={
                "pixel_values": torch.ones(2, 3, 2, 2),
                "video_size": [[1, 1, 64, 64]],
            }
        )

        _, extra = proc.call_with_token_ids(
            {"prompt_token_ids": prompt_token_ids, "multi_modal_data": mm_data},
            None,
        )

        multimodal_data = extra["multimodal_data"]
        assert multimodal_data["multimodal_item_order"] == [
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
            {"modality": "image", "index": 1},
        ]
