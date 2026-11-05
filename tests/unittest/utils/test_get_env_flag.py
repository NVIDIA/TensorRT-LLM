# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `tensorrt_llm._utils.get_env_flag`, the strict parser for boolean env switches."""

import re
from typing import Optional

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
        ("\tyes\n", False, True),
        ("On", None, True),
        ("0", True, False),
        ("false", True, False),
        ("No", None, False),
        ("OFF", True, False),
    ],
)
def test_get_env_flag(
    monkeypatch: pytest.MonkeyPatch,
    raw: Optional[str],
    default: Optional[bool],
    expected: Optional[bool],
) -> None:
    if raw is None:
        monkeypatch.delenv(_NAME, raising=False)
    else:
        monkeypatch.setenv(_NAME, raw)
    assert get_env_flag(_NAME, default) is expected


@pytest.mark.parametrize("raw", ["2", "y", "t", "1x", "\tenable\n"])
def test_get_env_flag_rejects_other_values(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(_NAME, raw)
    message = f"{_NAME}={raw!r} is not one of 1/true/yes/on or 0/false/no/off"
    with pytest.raises(ValueError, match=re.escape(message)):
        get_env_flag(_NAME, True)
