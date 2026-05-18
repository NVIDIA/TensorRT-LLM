"""Unit tests for the deterministic math hooks on `BaseMultimodalInputProcessor`.

`get_num_mm_tokens` (pre-merger encoder attention tokens) and
`spatial_merge_unit` are the single source of truth for the encoder-side
math used by both the hashing path (`get_num_tokens_per_image` /
`get_num_tokens_per_video`) and the dummy-sizing path (the
`BaseMultimodalDummyInputsBuilder` mixin). These tests pin the contract:

* Defaults — ``get_num_mm_tokens`` raises ``NotImplementedError``,
  ``spatial_merge_unit`` is 1.
* Deterministic-first dispatch — ``get_num_tokens_per_image`` returns
  ``get_num_mm_tokens // spatial_merge_unit`` when the subclass has
  implemented the math.
* Legacy fallback — when the subclass has not implemented
  ``get_num_mm_tokens``, ``get_num_tokens_per_image`` delegates to the
  HF processor's ``_get_num_multimodal_tokens`` as before (preserving
  behavior for non-migrated models).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor


def _make_subclass(
    *, deterministic: bool, spatial_merge_unit_value: int = 1, hf_post_merger_tokens: int = 999
):
    """Build a concrete subclass with controlled deterministic + HF behavior.

    ``deterministic=True``: ``get_num_mm_tokens`` returns
        ``width * height * num_frames`` (pre-merger). This makes the
        expected post-merger output predictable in tests.
    ``deterministic=False``: ``get_num_mm_tokens`` falls through to the
        base-class default (NotImplementedError).
    """

    class _Sub(BaseMultimodalInputProcessor):
        # Stub the abstracts that get_num_tokens_per_image doesn't actually
        # touch on the deterministic path. The HF fallback path does touch
        # ``processor`` via the ``get_num_multimodal_tokens`` property, so
        # we wire a MagicMock there.
        @property
        def processor(self):
            return self._processor

        @property
        def tokenizer(self):
            return None

        @property
        def config(self):
            return None

        @property
        def dtype(self):
            return None

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def spatial_merge_unit(self):
            return spatial_merge_unit_value

        if deterministic:

            def get_num_mm_tokens(self, *, width, height, num_frames=1):
                return width * height * num_frames

    instance = _Sub.__new__(_Sub)
    # HF processor mock returning the configured post-merger count.
    hf_processor = SimpleNamespace(
        _get_num_multimodal_tokens=MagicMock(
            return_value={
                "num_image_tokens": [hf_post_merger_tokens],
                "num_video_tokens": [hf_post_merger_tokens],
            }
        )
    )
    instance._processor = hf_processor
    return instance


def test_spatial_merge_unit_defaults_to_one():
    sub = _make_subclass(deterministic=False)
    assert sub.spatial_merge_unit == 1


def test_get_num_mm_tokens_default_raises_not_implemented():
    sub = _make_subclass(deterministic=False)
    with pytest.raises(NotImplementedError):
        sub.get_num_mm_tokens(width=224, height=224)


def test_get_num_tokens_per_image_deterministic_path():
    """Image token count equals ``encoder_tokens // spatial_merge_unit``.

    Confirms the deterministic path is preferred when the subclass has
    implemented ``get_num_mm_tokens``; the HF processor is bypassed.
    """
    sub = _make_subclass(deterministic=True, spatial_merge_unit_value=4)
    image = Image.new("RGB", (8, 6))  # width=8, height=6 -> 48 encoder tokens
    # 48 // 4 = 12
    assert sub.get_num_tokens_per_image(image=image) == 12
    # The HF processor must NOT be consulted on the deterministic path.
    sub._processor._get_num_multimodal_tokens.assert_not_called()


def test_get_num_tokens_per_image_falls_back_to_hf_processor():
    """Without a deterministic impl, the count comes from the HF processor."""
    sub = _make_subclass(deterministic=False, hf_post_merger_tokens=999)
    image = Image.new("RGB", (8, 6))
    assert sub.get_num_tokens_per_image(image=image) == 999
    sub._processor._get_num_multimodal_tokens.assert_called_once()


def test_get_num_tokens_per_video_deterministic_path():
    """Video token count threads ``num_frames`` through the math.

    The deterministic path multiplies the per-frame token count by the
    frame count (and applies any temporal padding inside
    ``get_num_mm_tokens``), then divides by ``spatial_merge_unit``.
    """
    sub = _make_subclass(deterministic=True, spatial_merge_unit_value=4)
    frames = [Image.new("RGB", (8, 6)) for _ in range(3)]
    # Each frame width*height=48, *3 frames=144 encoder tokens, //4 = 36
    assert sub.get_num_tokens_per_video(video=frames) == 36
    sub._processor._get_num_multimodal_tokens.assert_not_called()


def test_get_num_tokens_per_video_falls_back_to_hf_processor():
    sub = _make_subclass(deterministic=False, hf_post_merger_tokens=777)
    frames = [Image.new("RGB", (8, 6)) for _ in range(3)]
    assert sub.get_num_tokens_per_video(video=frames) == 777
    sub._processor._get_num_multimodal_tokens.assert_called_once()
