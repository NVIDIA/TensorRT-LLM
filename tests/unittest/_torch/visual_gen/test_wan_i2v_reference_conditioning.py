# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Wan I2V reference-role dispatch.

The slot declares two roles, so ``infer`` has to decide which reference is the
first frame and which is the last. Getting that backwards conditions the model
on the wrong frame and still returns a video, so nothing downstream would say
anything was wrong.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams


def _request(*refs) -> SimpleNamespace:
    """A request as the coordinator hands it over: references already bytes."""
    return SimpleNamespace(
        prompt=["a prompt"],
        params=VisualGenParams(image_reference=list(refs), seed=0),
    )


def _forward_kwargs(req) -> dict:
    pipeline = WanImageToVideoPipeline.__new__(WanImageToVideoPipeline)
    pipeline.forward = MagicMock(return_value=object())

    pipeline.infer(req)

    return pipeline.forward.call_args.kwargs


def test_roles_select_the_frames() -> None:
    kwargs = _forward_kwargs(
        _request(
            MediaRef(content=b"last", format="bytes", role="last_frame"),
            MediaRef(content=b"first", format="bytes", role="first_frame"),
        )
    )

    # Declaration order must not decide: the roles do.
    assert kwargs["image"] == b"first"
    assert kwargs["last_image"] == b"last"


def test_an_omitted_role_is_the_first_frame() -> None:
    """Plain I2V leaves ``role`` off, and the slot's other role is optional."""
    kwargs = _forward_kwargs(_request(MediaRef(content=b"only", format="bytes")))

    assert kwargs["image"] == b"only"
    assert kwargs["last_image"] is None
