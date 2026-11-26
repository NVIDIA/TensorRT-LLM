# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models import modeling_vila
from tensorrt_llm._torch.models.modeling_hyperclovax import HCXVisionInputProcessor
from tensorrt_llm._torch.models.modeling_llama import Llama4InputProcessor
from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def test_llama4_rescales_decoded_png():
    png = BytesIO()
    image = Image.new("RGB", (4, 8))
    image.save(png, format="PNG")
    adapter = Mock()
    adapter.processor.return_value = {
        "input_ids": torch.tensor([[1]]),
        "pixel_values": torch.zeros(1),
    }
    with Image.open(png) as decoded_png:
        Llama4InputProcessor.call_with_text_prompt(
            adapter, {"multi_modal_data": {"image": [decoded_png, image]}}, SamplingParams()
        )
    assert adapter.processor.call_args.kwargs["do_rescale"] is True


def test_hyperclovax_preserves_rgba_tensor_pixels():
    image = torch.tensor([1.0, 0.5, 0.0, 1.0]).view(4, 1, 1).expand(4, 8, 8)
    adapter = Mock()
    adapter._preprocess.return_value = (torch.tensor([1]), None)
    adapter._post_process.return_value = torch.tensor([1])

    HCXVisionInputProcessor.call_with_text_prompt(
        adapter, {"multi_modal_data": {"image": [image]}}, None
    )

    converted = adapter._preprocess.call_args.args[1][0]
    assert isinstance(converted, Image.Image)
    assert converted.mode == "RGBA"
    assert converted.getpixel((0, 0)) == (255, 127, 0, 255)


def test_vila_decoded_video_pixels(monkeypatch):
    frames = np.arange(2 * 4 * 8 * 3, dtype=np.uint8).reshape(2, 4, 8, 3)
    frames.setflags(write=False)
    monkeypatch.setattr(
        modeling_vila,
        "process_images",
        lambda frames, *args, **kwargs: (torch.stack(tuple(frames)), None),
    )
    video = VideoData(frames=frames, metadata={})
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        pixels, _ = modeling_vila.VilaInputProcessor._preprocess(Mock(), {"video": [video]}, {})
    finally:
        torch.set_default_dtype(previous_dtype)
    assert pixels.dtype == torch.float32
    expected = torch.from_numpy(frames.astype(np.float32) * (1.0 / 255.0)).permute(0, 3, 1, 2)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="uint8"):
        modeling_vila.VilaInputProcessor._preprocess(
            Mock(), {"video": [VideoData(frames=frames.astype(np.float32), metadata={})]}, {}
        )
