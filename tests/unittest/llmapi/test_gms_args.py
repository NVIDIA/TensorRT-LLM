# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for GMS prototype config fields on ``TorchLlmArgs``."""

from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import LoadFormat, TorchLlmArgs

_LOGGER_PATH = "tensorrt_llm.llmapi.llm_args.logger"
_DUMMY_MODEL = "/tmp/test-gms-args-nonexistent"


def _capture_warnings(call_args_list):
    """Flatten captured ``logger.warning(fmt, *args)`` calls into strings."""
    rendered = []
    for call in call_args_list:
        args, _kwargs = call
        if not args:
            continue
        fmt, *fmt_args = args
        try:
            rendered.append(fmt % tuple(fmt_args) if fmt_args else fmt)
        except (TypeError, ValueError):
            rendered.append(str(args))
    return rendered


def _warnings_for(keyword: str, call_args_list) -> list[str]:
    """Return only captured warnings whose rendered text mentions keyword."""
    return [m for m in _capture_warnings(call_args_list) if keyword in m]


def _make_args(**overrides) -> TorchLlmArgs:
    """Construct a ``TorchLlmArgs`` with the given overrides."""
    return TorchLlmArgs(model=_DUMMY_MODEL, **overrides)


class TestGmsConfigDefaults:
    def test_defaults(self):
        args = _make_args()
        assert args.gms_config.socket_path is None
        assert args.gms_config.mode == "auto"
        assert args.gms_config.tag == "weights"


class TestGmsConfigValidation:
    @pytest.mark.parametrize("mode", ["auto", "rw", "ro"], ids=["auto", "rw", "ro"])
    def test_accepts_known_modes(self, mode):
        args = _make_args(load_format=LoadFormat.GMS, gms_config={"mode": mode})
        assert args.gms_config.mode == mode

    @pytest.mark.parametrize(
        "bad_mode",
        ["", "AUTO", "Rw", "read", "write", "shadow"],
        ids=[
            "empty",
            "uppercase-auto",
            "mixedcase-rw",
            "read",
            "write",
            "shadow",
        ],
    )
    def test_rejects_unknown_mode(self, bad_mode):
        # GmsConfig.mode is typed as Literal["auto", "rw", "ro"], so Pydantic
        # rejects bad values at field validation with its stable literal_error
        # template. Match on the rendered allowed-values list rather than the
        # field path so the assertion is robust across Pydantic 2.x error
        # path-formatting differences (dot vs arrow).
        with pytest.raises(ValueError, match=r"Input should be 'auto', 'rw' or 'ro'"):
            _make_args(gms_config={"mode": bad_mode})

    @pytest.mark.parametrize(
        "bad_tag",
        ["", " ", "\t", "\n", "   \t  "],
        ids=["empty", "space", "tab", "newline", "mixed-whitespace"],
    )
    def test_rejects_empty_or_whitespace_tag(self, bad_tag):
        with pytest.raises(ValueError, match="gms_config.tag must be a non-empty"):
            _make_args(gms_config={"tag": bad_tag})

    def test_accepts_custom_non_empty_tag(self):
        args = _make_args(load_format=LoadFormat.GMS, gms_config={"tag": "custom"})
        assert args.gms_config.tag == "custom"


class TestGmsCrossFieldWarning:
    @pytest.mark.parametrize(
        "load_format, gms_config, expect_warning",
        [
            pytest.param(
                LoadFormat.GMS, {"socket_path": "/tmp/gms.sock"}, False, id="gms-active-no-warning"
            ),
            pytest.param(
                LoadFormat.AUTO,
                {"socket_path": "/tmp/gms.sock"},
                True,
                id="socket-path-without-gms",
            ),
            pytest.param(LoadFormat.AUTO, {"mode": "rw"}, True, id="mode-without-gms"),
            pytest.param(LoadFormat.AUTO, {"tag": "custom"}, True, id="tag-without-gms"),
        ],
    )
    def test_non_default_config_warns_when_gms_inactive(
        self, load_format, gms_config, expect_warning
    ):
        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(load_format=load_format, gms_config=gms_config)

        for key, value in gms_config.items():
            assert getattr(args.gms_config, key) == value
        relevant = _warnings_for("gms_config", mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant, "expected a warning about ignored gms_config"
        else:
            assert relevant == []


class TestLoadFormatGms:
    def test_gms_enum_present(self):
        assert LoadFormat.GMS.name == "GMS"
        assert LoadFormat.GMS.value == 3

    @pytest.mark.parametrize(
        "raw, expected",
        [
            pytest.param("GMS", LoadFormat.GMS, id="uppercase-string"),
            pytest.param("gms", LoadFormat.GMS, id="lowercase-string"),
            pytest.param(3, LoadFormat.GMS, id="enum-int"),
        ],
    )
    def test_gms_load_format_conversion(self, raw, expected):
        args = _make_args(load_format=raw)
        assert args.load_format == expected

    def test_gms_only_two_axis_defaults(self):
        args = _make_args(load_format=LoadFormat.GMS)
        assert args.checkpoint_format == "HF"
        assert args.load_format == LoadFormat.GMS
