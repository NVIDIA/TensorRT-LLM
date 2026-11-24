# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the layer-wise benchmark's first-N-layers truncation."""

import json
import types

from tensorrt_llm.tools.layer_wise_benchmarks import layer_truncation as lt


def _config(n, **extra):
    return types.SimpleNamespace(num_hidden_layers=n, architectures=["X"] * n, **extra)


def test_truncates_top_level_and_text_config():
    text = _config(10, layer_types=tuple(range(10)), full_attn_layers=[4, 8], sizes=[1, 2])
    cfg = _config(10, text_config=text, layer_types=list("abcdefghij"))
    assert lt.truncate_config(cfg, 4)
    assert cfg.num_hidden_layers == 4 and cfg.layer_types == list("abcd")
    assert text.num_hidden_layers == 4 and text.layer_types == (0, 1, 2, 3)
    # Index lists and lists of another length stay; models filter indices themselves.
    assert text.full_attn_layers == [4, 8] and text.sizes == [1, 2]
    assert cfg.architectures == ["X"] * 10
    assert lt.original_num_hidden_layers(text) == 10


def test_text_config_only():
    text = _config(10)
    cfg = types.SimpleNamespace(text_config=text)
    assert lt.truncate_config(cfg, 3)
    assert text.num_hidden_layers == 3 and lt.original_num_hidden_layers(text) == 10


def test_smaller_model_untouched():
    cfg = _config(5, layer_types=["a"] * 5)
    assert not lt.truncate_config(cfg, 8)
    assert cfg.num_hidden_layers == 5 and lt.original_num_hidden_layers(cfg) == 5


def test_filter_keeps_moves_and_drops():
    entries = {
        "language_model.model.layers.0.w": 0,
        "language_model.model.layers.1.w": 1,
        "language_model.model.layers.3.w": 3,
        "model.layers.4.mtp.w": 4,  # stored past the 4 layers: an MTP module
        "layers.1.experts": "q1",  # ModelOpt quantized_layers spelling
        "layers.2.experts": "q2",
        "language_model.model.embed_tokens.weight": "e",
        "vision_tower.encoder.layers.7.w": "v",
    }
    assert lt.filter_layer_keys(entries, 2, 4) == (2, 1)
    assert entries == {
        "language_model.model.layers.0.w": 0,
        "language_model.model.layers.1.w": 1,
        "model.layers.2.mtp.w": 4,
        "layers.1.experts": "q1",
        "language_model.model.embed_tokens.weight": "e",
        "vision_tower.encoder.layers.7.w": "v",
    }


def test_filter_no_op_when_not_smaller():
    entries = {"model.layers.5.w": 5}
    assert lt.filter_layer_keys(entries, 8, 6) == (0, 0)
    assert entries == {"model.layers.5.w": 5}


def test_checkpoint_num_layers(tmp_path):
    assert lt.checkpoint_num_layers(str(tmp_path)) is None
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 93}}))
    assert lt.checkpoint_num_layers(str(tmp_path)) == 93
