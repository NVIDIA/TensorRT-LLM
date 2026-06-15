# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for Mistral3 (Pixtral) deterministic dummy-input sizing.

Validates the encoder-profiling dummy contract on ``Mistral3InputProcessor``:

* ``get_mm_max_tokens_per_item`` reports the largest single image's ViT patch
  count (``max_image_size``-capped), keyed by ``"image"`` only.
* ``get_size_for_max_tokens`` returns a ``patch * spatial_merge_size``-aligned
  square whose ViT patch count is the largest ``<= max_tokens``.
* ``get_dummy_mm_data_for_tokens`` materializes ``pixel_values`` /
  ``image_sizes`` tensors directly (no PIL / HF-processor round-trip), saturating
  the token budget.

The ViT token unit here is the pre-merge patch count ``(h//patch)*(w//patch)``
-- deliberately *not* ``get_num_mm_tokens`` (which the hashing path uses for the
LLM-side Pixtral count with ``[IMG_BREAK]``/``[IMG_END]`` framing).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_mistral import Mistral3InputProcessor


def _make_processor(*, patch_size=14, spatial_merge_size=2, image_size=1540, num_channels=3):
    """Construct a processor stub with just the geometry the dummy math reads.

    Bypasses the real ``__init__`` (tokenizer/processor loading); the empty
    ``_processor`` forces ``_vision_geometry`` to fall back to ``vision_config``
    (the HF ``mistral3`` path).
    """
    instance = Mistral3InputProcessor.__new__(Mistral3InputProcessor)
    instance._config = SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size=patch_size, image_size=image_size, num_channels=num_channels
        ),
        spatial_merge_size=spatial_merge_size,
    )
    instance._processor = SimpleNamespace()
    instance._dtype = torch.float16
    return instance


def test_mm_max_tokens_per_item_is_image_only():
    proc = _make_processor(patch_size=14, image_size=1540)
    demand = proc.get_mm_max_tokens_per_item()
    assert set(demand) == {"image"}
    # max square = 1540 (a multiple of patch*merge=28); patches = (1540/14)^2.
    assert demand["image"] == (1540 // 14) ** 2 == 110**2


@pytest.mark.parametrize("budget", [1024, 4096, 8192])
def test_get_size_for_max_tokens_fits_and_aligns(budget):
    proc = _make_processor()
    size = proc.get_size_for_max_tokens(max_tokens=budget)
    unit = 14 * 2  # patch * spatial_merge_size
    assert size["width"] == size["height"]
    assert size["width"] % unit == 0
    patches = (size["width"] // 14) * (size["height"] // 14)
    assert patches <= budget
    # Adding one more aligned step would exceed the budget (saturated).
    nxt = size["width"] + unit
    assert (nxt // 14) ** 2 > budget or nxt > 1540


def test_get_size_rejects_non_positive_budget():
    proc = _make_processor()
    with pytest.raises(ValueError):
        proc.get_size_for_max_tokens(max_tokens=0)


@pytest.mark.parametrize("budget", [4096, 8192])
def test_get_dummy_mm_data_for_tokens_shapes_and_saturation(budget):
    proc = _make_processor(num_channels=3)
    data = proc.get_dummy_mm_data_for_tokens(
        max_tokens_per_modality={"image": budget}, dtype=torch.float16
    )
    image = data["image"]
    pv = image["pixel_values"]
    sizes = image["image_sizes"]
    n, c, h, w = pv.shape
    assert c == 3 and pv.dtype == torch.float16
    assert sizes == [[h, w]] * n
    # The batch saturates the budget: total patches <= budget, within one image.
    per_image = (h // 14) * (w // 14)
    assert n * per_image <= budget
    assert n * per_image + per_image > budget


def test_dummy_for_tokens_empty_without_image_budget():
    proc = _make_processor()
    assert proc.get_dummy_mm_data_for_tokens(max_tokens_per_modality={"audio": 1024}) == {}
