# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STS calibration loading and confidence-head resolution tests."""

import json

import pytest

from tensorrt_llm._torch.speculative.dspark_sts import (
    load_sts_temperatures_from_path,
    resolve_confidence_head,
)


@pytest.mark.parametrize("key", ["sts_temperatures", "temperatures"])
def test_loads_native_and_sglang_temperature_keys(tmp_path, key):
    path = tmp_path / "sts.json"
    path.write_text(json.dumps({key: [0.75, 1.0, 1.25]}), encoding="utf-8")

    assert load_sts_temperatures_from_path(str(path)) == [0.75, 1.0, 1.25]


@pytest.mark.parametrize(
    ("payload", "error"),
    [({}, KeyError), ({"temperatures": []}, ValueError)],
)
def test_rejects_missing_or_empty_temperatures(tmp_path, payload, error):
    path = tmp_path / "sts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error):
        load_sts_temperatures_from_path(str(path))


class _Head:
    pass


class _Bare:
    def __init__(self, head):
        stage = type("Stage", (), {})()
        if head is not None:
            stage.confidence_head = head
        self.mtp_layers = [object(), object(), stage]


class _Wrapper:
    def __init__(self, head):
        self.dspark_model = _Bare(head)


@pytest.mark.parametrize("wrap", [lambda head: _Bare(head), lambda head: _Wrapper(head)])
def test_resolves_known_model_layouts(wrap):
    head = _Head()

    assert resolve_confidence_head(wrap(head)) is head
    assert resolve_confidence_head(wrap(None)) is None


def test_unknown_model_layout_has_no_confidence_head():
    assert resolve_confidence_head(object()) is None
