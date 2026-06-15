"""Unit tests for Qwen2/2.5/3-VL deterministic dummy-input sizing.

Validates the encoder-side ``get_num_mm_tokens`` / ``get_size_for_max_tokens``
pair on the Qwen-VL family. These tests reach into the InputProcessorBase
classes directly (no model loading) and stub just enough of the HF config so
the math runs.

The contracts under test:

* ``get_num_mm_tokens`` returns encoder attention tokens (pre-merger), the
  same unit as ``encoder_max_num_tokens`` and
  ``AttentionMetadata.max_num_tokens``.
* ``get_size_for_max_tokens(max_tokens=N)`` returns a media size whose
  token count is the largest value ``<= N`` reachable under the aspect-ratio
  bound.
* The two are invertible: feeding the returned size back through
  ``get_num_mm_tokens`` gives a value within the budget.
* ``get_dummy_mm_data_for_size`` materializes the chosen size into the
  processed encoder tensors directly (no PIL/HF-processor round-trip).
"""

from types import SimpleNamespace
from typing import Type

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase


def _make_processor(
    processor_cls: Type,
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 1 << 30,
):
    """Construct a processor stub with stubbed vision_config attrs.

    Bypasses the real ``__init__`` (which loads tokenizers/processors)
    and pins just the fields the deterministic math reads. ``min_pixels`` /
    ``max_pixels`` stub the HF image processor's ``size`` config that
    ``get_num_mm_tokens`` reads for its ``smart_resize`` clamp; the default
    ``max_pixels`` is generous so the factor-pair tests aren't clamped (the
    clamp itself is covered by ``test_size_capped_at_max_pixels``).
    """
    instance = processor_cls.__new__(processor_cls)
    vision_config = SimpleNamespace(
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=3,
    )
    instance._config = SimpleNamespace(vision_config=vision_config)
    instance._processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            size={"shortest_edge": min_pixels, "longest_edge": max_pixels}
        )
    )
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
    size = proc.get_size_for_max_tokens(max_tokens=budget)
    actual = proc.get_num_mm_tokens(
        width=size["width"], height=size["height"], num_frames=size["num_frames"]
    )
    assert actual <= budget, f"Got {actual} tokens for size {size}, budget was {budget}"


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
@pytest.mark.parametrize("budget", [256, 1024, 4096, 8192, 16384, 32768])
def test_invertibility_saturates_budget(processor_cls, budget):
    """Power-of-2 budgets fit the encoder's unit grid exactly.

    When ``budget`` factors cleanly into the encoder's unit grid, the
    returned size should hit it exactly (no wasted budget).
    """
    proc = _make_processor(processor_cls)
    size = proc.get_size_for_max_tokens(max_tokens=budget)
    actual = proc.get_num_mm_tokens(
        width=size["width"], height=size["height"], num_frames=size["num_frames"]
    )
    # Budgets above are all powers of 2 with merge_size=2 squared as a factor,
    # so saturation should be exact.
    assert actual == budget


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_aspect_ratio_is_bounded(processor_cls):
    """The returned size stays within the model's aspect bound.

    Even for very large budgets the returned size should not be an
    extreme stripe — the ``200x`` aspect bound mirrors HF's processor.
    """
    proc = _make_processor(processor_cls)
    size = proc.get_size_for_max_tokens(max_tokens=1_000_000)
    long_edge = max(size["width"], size["height"])
    short_edge = min(size["width"], size["height"])
    assert long_edge / short_edge <= 200, size


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_size_capped_at_max_pixels(processor_cls):
    """A single image cannot exceed the processor's ``max_pixels``.

    For a budget larger than one image can hold, ``get_size_for_max_tokens``
    must cap at ``max_pixels`` so the chosen size round-trips through
    ``get_num_mm_tokens`` (whose ``smart_resize`` would otherwise clamp it back
    down, leaving the predicted token count short of the budget).
    """
    max_pixels = 512 * 512
    proc = _make_processor(processor_cls, max_pixels=max_pixels)
    size = proc.get_size_for_max_tokens(max_tokens=1_000_000)

    # The chosen image must fit within max_pixels ...
    assert size["width"] * size["height"] <= max_pixels, size
    # ... and round-trip exactly (smart_resize does not clamp it further), so
    # the realized token count is the single-image max, below the huge budget.
    actual = proc.get_num_mm_tokens(
        width=size["width"], height=size["height"], num_frames=size["num_frames"]
    )
    assert 0 < actual < 1_000_000


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_size_rejects_non_positive_budget(processor_cls):
    proc = _make_processor(processor_cls)
    with pytest.raises(ValueError):
        proc.get_size_for_max_tokens(max_tokens=0)
    with pytest.raises(ValueError):
        proc.get_size_for_max_tokens(max_tokens=-1)


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_dummy_mm_data_shapes_match_token_count(processor_cls):
    """Direct tensor build: pixel_values rows and the grid_thw product match ``get_num_mm_tokens`` per image."""
    proc = _make_processor(processor_cls)
    cfg = proc._config.vision_config
    width = height = 224
    per_image = proc.get_num_mm_tokens(width=width, height=height)
    in_dim = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

    data = proc.get_dummy_mm_data_for_size(
        width=width, height=height, num_images=3, dtype=torch.float32
    )
    image = data["image"]
    assert image["pixel_values"].shape == (3 * per_image, in_dim)
    assert image["pixel_values"].dtype == torch.float32
    assert image["image_grid_thw"].shape == (3, 3)
    # Each grid row's product equals the per-image token count.
    grid = image["image_grid_thw"]
    assert int(grid[0].prod().item()) == per_image
    assert torch.equal(grid[0], grid[1]) and torch.equal(grid[1], grid[2])


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_get_dummy_mm_data_single_image_default(processor_cls):
    """Defaults to a single image; grid_thw is ``[1, 3]``."""
    proc = _make_processor(processor_cls)
    data = proc.get_dummy_mm_data_for_size(width=224, height=224, dtype=torch.float16)
    assert data["image"]["image_grid_thw"].shape == (1, 3)
    assert data["image"]["pixel_values"].dtype == torch.float16


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
def test_mm_max_tokens_per_item_is_image_only(processor_cls):
    """Qwen-VL declares only ``image`` (image+video share one ViT), valued at the max single-image token count."""
    proc = _make_processor(processor_cls, max_pixels=512 * 512)
    demand = proc.get_mm_max_tokens_per_item()
    assert set(demand) == {"image"}
    # The declared per-item demand is exactly the max single-image token count.
    cap_size = proc.get_size_for_max_tokens(max_tokens=10**9)
    assert demand["image"] == proc.get_num_mm_tokens(
        width=cap_size["width"], height=cap_size["height"]
    )
    assert demand["image"] > 0


@pytest.mark.parametrize("processor_cls", _PROCESSORS)
@pytest.mark.parametrize("budget", [1024, 4096, 16384])
def test_get_dummy_mm_data_for_tokens_saturates_budget(processor_cls, budget):
    """Agnostic entry: total pre-merger patches are ``<= budget`` and within one image of it (saturates the budget)."""
    proc = _make_processor(processor_cls)
    data = proc.get_dummy_mm_data_for_tokens(
        max_tokens_per_modality={"image": budget}, dtype=torch.float32
    )
    grid = data["image"]["image_grid_thw"]
    total_patches = int(grid.prod(dim=1).sum().item())
    per_image = int(grid[0].prod().item())
    # pixel_values rows == total patches across all batched images.
    assert data["image"]["pixel_values"].shape[0] == total_patches
    # Saturates: within the budget, and adding one more image would exceed it.
    assert total_patches <= budget
    assert total_patches + per_image > budget
