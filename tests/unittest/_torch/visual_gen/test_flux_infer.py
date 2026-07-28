# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLUX request orchestration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.visual_gen.models.flux import Flux2Pipeline, FluxPipeline


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
def test_infer_forwards_num_images_per_prompt(
    pipeline_cls: type[FluxPipeline] | type[Flux2Pipeline],
) -> None:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.forward = Mock(return_value="image")
    request = SimpleNamespace(
        prompt="a cat",
        params=SimpleNamespace(
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42,
            max_sequence_length=512,
            num_images_per_prompt=2,
        ),
    )

    result = pipeline.infer(request)

    assert result == "image"
    pipeline.forward.assert_called_once()
    assert pipeline.forward.call_args.kwargs["num_images_per_prompt"] == 2
