# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Beam configuration loading and sampler row selection."""

import json

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def test_load_pretrained_config_preserves_beam_rope(tmp_path):
    from tensorrt_llm._torch.configs.beam import BeamConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    }
    config = {
        "model_type": "beam",
        "num_hidden_layers": 6,
        "torch_dtype": "bfloat16",
        "rope_parameters": rope,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_pretrained_config(str(tmp_path))
    assert isinstance(loaded, BeamConfig)
    assert loaded.name_or_path == str(tmp_path)
    assert loaded.torch_dtype == torch.bfloat16
    assert loaded.rope_parameters == rope


@pytest.mark.parametrize(
    "spans,expected",
    [
        ([(0, 40)], [39]),
        ([(0, 40), (64, 20), (100, 1), (7, 1)], [39, 59, 60, 61]),
        ([(5, 1), (9, 1), (2, 1)], [0, 1, 2]),
    ],
)
def test_default_rows_are_each_requests_last_row(spans, expected):
    from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (  # noqa: E501
        BeamFullModelNvfp4Sm103Tp4,
    )

    select = BeamFullModelNvfp4Sm103Tp4._select_rows
    assert select(None, None, spans) == expected
    assert select(None, "all", spans) == list(range(sum(n for _, n in spans)))
