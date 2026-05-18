"""Unit tests for Qwen2/2.5/3-VL deterministic dummy-input sizing.

Validates the encoder-side ``get_num_mm_tokens`` / ``get_size_with_most_features``
pair on the Qwen-VL family. These tests reach into the InputProcessorBase
classes directly (no model loading) and stub just enough of the HF config so
the math runs.

The contracts under test:

* ``get_num_mm_tokens`` returns encoder attention tokens (pre-merger), the
  same unit as ``encoder_max_num_tokens`` and
  ``AttentionMetadata.max_num_tokens``.
* ``get_size_with_most_features(max_tokens=N)`` returns a media size whose
  token count is the largest value ``<= N`` reachable under the aspect-ratio
  bound.
* The two are invertible: feeding the returned size back through
  ``get_num_mm_tokens`` gives a value within the budget.
"""

from types import SimpleNamespace
from typing import Type

import pytest

from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase


def _make_processor(
    processor_cls: Type,
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
):
    """Construct a processor stub with just the vision_config attrs the math
    touches, bypassing the real __init__ (which loads tokenizers/processors).
    """
    instance = processor_cls.__new__(processor_cls)
    vision_config = SimpleNamespace(
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    instance._config = SimpleNamespace(vision_config=vision_config)
    return instance


_PROCESSORS = [Qwen2VLInputProcessorBase, Qwen3VLInputProcessorBase]


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_spatial_merge_unit_is_merge_size_squared(processor_cls):
    proc = _make_processor(processor_cls, spatial_merge_size=2)
    assert proc.spatial_merge_unit == 4

    proc3 = _make_processor(processor_cls, spatial_merge_size=3)
    assert proc3.spatial_merge_unit == 9


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_num_mm_tokens_matches_manual_grid(processor_cls):
    proc = _make_processor(
        processor_cls, patch_size=16, spatial_merge_size=2, temporal_patch_size=2
    )
    # 224x224 image, patch=16, merge=2 -> resize-to-multiple-of-(16*2=32)
    # 224/32 -> rounds to 224 (already a multiple), grid 14*14 = 196.
    assert proc.get_num_mm_tokens(width=224, height=224) == 14 * 14


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_num_mm_tokens_rounds_to_unit(processor_cls):
    proc = _make_processor(
        processor_cls, patch_size=16, spatial_merge_size=2, temporal_patch_size=2
    )
    # 220x220 -> rounds half-up to nearest multiple of 32 = 224 -> 14*14.
    assert proc.get_num_mm_tokens(width=220, height=220) == 14 * 14


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_num_mm_tokens_temporal_padding(processor_cls):
    proc = _make_processor(
        processor_cls, patch_size=16, spatial_merge_size=2, temporal_patch_size=2
    )
    # Single frame still pads to temporal_patch_size, grid_t = 1.
    single = proc.get_num_mm_tokens(width=224, height=224, num_frames=1)
    three = proc.get_num_mm_tokens(width=224, height=224, num_frames=3)
    # 3 frames -> pad to 4 -> grid_t = 2.
    assert three == 2 * single


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
@pytest.mark.parametrize("budget", [256, 1024, 4096, 8192, 16384, 32768])
def test_invertibility_fits_within_budget(processor_cls, budget):
    proc = _make_processor(processor_cls)
    size = proc.get_size_with_most_features(max_tokens=budget)
    actual = proc.get_num_mm_tokens(
        width=size["width"], height=size["height"], num_frames=size["num_frames"]
    )
    assert actual <= budget, f"Got {actual} tokens for size {size}, budget was {budget}"


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
@pytest.mark.parametrize("budget", [256, 1024, 4096, 8192, 16384, 32768])
def test_invertibility_saturates_budget(processor_cls, budget):
    """When ``budget`` factors cleanly into the encoder's unit grid, the
    returned size should hit it exactly (no wasted budget)."""
    proc = _make_processor(processor_cls)
    size = proc.get_size_with_most_features(max_tokens=budget)
    actual = proc.get_num_mm_tokens(
        width=size["width"], height=size["height"], num_frames=size["num_frames"]
    )
    # Budgets above are all powers of 2 with merge_size=2 squared as a factor,
    # so saturation should be exact.
    assert actual == budget


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_aspect_ratio_is_bounded(processor_cls):
    """Even for very large budgets the returned size should not be an
    extreme stripe (the ``200x`` aspect bound mirrors HF's processor)."""
    proc = _make_processor(processor_cls)
    size = proc.get_size_with_most_features(max_tokens=1_000_000)
    long_edge = max(size["width"], size["height"])
    short_edge = min(size["width"], size["height"])
    assert long_edge / short_edge <= 200, size


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_size_rejects_non_positive_budget(processor_cls):
    proc = _make_processor(processor_cls)
    with pytest.raises(ValueError):
        proc.get_size_with_most_features(max_tokens=0)
    with pytest.raises(ValueError):
        proc.get_size_with_most_features(max_tokens=-1)
