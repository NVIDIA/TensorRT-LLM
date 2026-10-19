# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `tensorrt_llm._utils.get_env_flag`, the strict parser for boolean env switches."""

import pytest

from tensorrt_llm._utils import get_env_flag

pytestmark = pytest.mark.cpu_only

_NAME = "TRTLLM_TEST_GET_ENV_FLAG"


@pytest.mark.parametrize(
    "raw, default, expected",
    [
        (None, None, None),
        (None, True, True),
        ("", False, False),
        ("  ", True, True),
        ("1", False, True),
        ("TRUE", False, True),
        (" yes ", False, True),
        ("On", None, True),
        ("0", True, False),
        ("false", True, False),
        ("No", None, False),
        ("OFF", True, False),
    ],
)
def test_get_env_flag(monkeypatch, raw, default, expected):
    if raw is None:
        monkeypatch.delenv(_NAME, raising=False)
    else:
        monkeypatch.setenv(_NAME, raw)
    assert get_env_flag(_NAME, default) is expected


@pytest.mark.parametrize("raw", ["2", "-1", "enable", "1x", "legacy"])
def test_get_env_flag_rejects_other_values(monkeypatch, raw):
    monkeypatch.setenv(_NAME, raw)
    with pytest.raises(ValueError, match=_NAME):
        get_env_flag(_NAME, True)
