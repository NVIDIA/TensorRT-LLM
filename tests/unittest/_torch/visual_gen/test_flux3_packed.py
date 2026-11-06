# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for packing FLUX.3 samples into mode-major tokens."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.flux3.packed import bounds, pack

pytestmark = pytest.mark.cpu_only

CHANNELS = {"image": 8, "image_cond": 12, "text": 24}


def samples(text_lengths: tuple[int, ...]) -> list[dict]:
    """Sample 0 has no reference image; ids encode the token index so slices are traceable."""
    result = []
    for index, text_length in enumerate(text_lengths):
        sample = {}
        for name, length in (
            ("image", 16),
            ("image_cond", 9 if index else 0),
            ("text", text_length),
        ):
            if length:
                tokens = torch.randn(length, CHANNELS[name], dtype=torch.bfloat16)
                ids = torch.zeros(length, 4)
                ids[:, 3 if name == "text" else 1] = torch.arange(length)
                sample[name] = tokens, ids
        result.append(sample)
    return result


def test_bounds_are_consecutive_segments() -> None:
    assert bounds((3, 0, 2)) == ((0, 3), (3, 3), (3, 5))
    assert bounds(()) == ()


def test_packing_preserves_sample_boundaries() -> None:
    inputs = samples((7, 11))
    actual = pack(inputs, CHANNELS)
    assert actual.layout == (("image", (16, 16)), ("image_cond", (0, 9)), ("text", (7, 11)))
    for name, lengths in actual.layout:
        for index, (start, end) in enumerate(bounds(lengths)):
            if end > start:
                torch.testing.assert_close(actual.data[name][start:end], inputs[index][name][0])
                torch.testing.assert_close(actual.ids[name][start:end], inputs[index][name][1])
        assert actual.timesteps[name].shape == (2,)
        assert not actual.timesteps[name].any()
    kwargs = actual.kwargs()
    assert kwargs["layout"] == actual.layout
    assert kwargs["x_image_cond_timesteps"] is actual.timesteps["image_cond"]


def test_absent_modality_is_empty_with_input_dtypes() -> None:
    packed = pack(samples((5,)), CHANNELS)
    assert packed.layout[1] == ("image_cond", (0,))
    assert packed.data["image_cond"].shape == (0, 12)
    assert packed.data["image_cond"].dtype == torch.bfloat16
    assert packed.ids["image_cond"].shape == (0, 4)
    assert packed.ids["image_cond"].dtype == torch.float32
    assert packed.timesteps["image"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "inputs, message",
    [
        pytest.param([], "at least one packed sample", id="no_samples"),
        pytest.param([{"text": None}], "nonempty packed conditioning", id="all_empty"),
        pytest.param(
            [{"video": (torch.zeros(2, 8), torch.zeros(2, 4))}],
            "Unknown FLUX.3 packed modality",
            id="unknown_modality",
        ),
        pytest.param(
            [{"image": (torch.zeros(2, 9), torch.zeros(2, 4))}],
            "Invalid packed FLUX.3 image",
            id="wrong_width",
        ),
        pytest.param(
            [{"image": (torch.zeros(2, 8), torch.zeros(2, 3))}],
            "Invalid packed FLUX.3 image",
            id="wrong_ids",
        ),
    ],
)
def test_invalid_samples_raise(inputs: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pack(inputs, CHANNELS)
