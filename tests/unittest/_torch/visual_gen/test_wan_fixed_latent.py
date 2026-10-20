# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WanPipeline fixed latent: read lazily, so pipeline construction never calls torch.load."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline


def _bare_pipeline(path):
    # Skip __init__ (it builds the whole model); set only what _get_fixed_latent reads.
    pipe = WanPipeline.__new__(WanPipeline)
    pipe._fixed_latent = None
    pipe._fixed_latent_path = path
    return pipe


@pytest.mark.parametrize("path_set", [True, False])
def test_fixed_latent_loaded_once_on_first_use(tmp_path, path_set):
    latent = torch.randn(1, 16, 3, 4, 4)
    path = tmp_path / "latent.pt"
    torch.save(latent, path)
    pipe = _bare_pipeline(str(path) if path_set else None)

    with mock.patch("torch.load", wraps=torch.load) as load:
        first = pipe._get_fixed_latent()
        second = pipe._get_fixed_latent()

    if not path_set:
        assert first is None and second is None
        load.assert_not_called()
        return
    assert load.call_count == 1
    assert second is first
    torch.testing.assert_close(first, latent, rtol=0, atol=0)


def test_init_reads_only_the_path(tmp_path, monkeypatch):
    path = tmp_path / "latent.pt"
    monkeypatch.setenv("TRTLLM_VIDEO_FIXED_LATENT_PATH", str(path))

    # Run only the fixed-latent part of __init__: stop at the base-class constructor.
    with (
        mock.patch("torch.load") as load,
        mock.patch(
            "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.BasePipeline.__init__",
            side_effect=RuntimeError("stop"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop"):
            WanPipeline(SimpleNamespace(primary_pretrained_config=SimpleNamespace()))
        load.assert_not_called()
