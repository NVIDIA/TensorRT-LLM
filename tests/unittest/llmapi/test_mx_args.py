# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for MX prototype config fields on ``TorchLlmArgs``."""

from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

_LOGGER_PATH = "tensorrt_llm.llmapi.llm_args.logger"
_DUMMY_MODEL = "/tmp/test-mx-args-nonexistent"


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


def _warnings_for(keyword: str, call_args_list) -> list:
    """Return only captured warnings whose rendered text mentions keyword."""
    return [m for m in _capture_warnings(call_args_list) if keyword in m]


def _make_args(**overrides) -> TorchLlmArgs:
    """Construct a ``TorchLlmArgs`` with the given overrides."""
    return TorchLlmArgs(model=_DUMMY_MODEL, **overrides)


class TestDefaults:
    """Default values must match the conventions of the upstream MX library."""

    def test_mx_config_defaults(self):
        args = _make_args()
        assert args.mx_config.server_url is None
        assert args.mx_config.server_query_timeout_s is None
        assert args.mx_config.preshard_strategy == "per_module"

    def test_mx_server_query_timeout_accepts_nonnegative_int(self):
        args = _make_args(checkpoint_format="MX", mx_config={"server_query_timeout_s": 1200})
        assert args.mx_config.server_query_timeout_s == 1200

    def test_mx_server_query_timeout_rejects_negative(self):
        with pytest.raises(ValueError):
            _make_args(checkpoint_format="MX", mx_config={"server_query_timeout_s": -1})

    def test_mx_preshard_strategy_accepts_per_module(self):
        args = _make_args(checkpoint_format="MX", mx_config={"preshard_strategy": "per_module"})
        assert args.mx_config.preshard_strategy == "per_module"

    @pytest.mark.parametrize(
        "bad_value",
        ["", "Per_Module", "PER_MODULE", "global ", "perModule", "all", "none", "1", " "],
    )
    def test_mx_preshard_strategy_rejects_unknown_value(self, bad_value):
        with pytest.raises(ValueError, match="mx_config.preshard_strategy"):
            _make_args(mx_config={"preshard_strategy": bad_value})

    def test_mx_preshard_strategy_rejects_global_until_presharded_load_format_lands(self):
        with pytest.raises(ValueError, match="LoadFormat.PRESHARDED"):
            _make_args(checkpoint_format="MX", mx_config={"preshard_strategy": "global"})


class TestMxServerUrl:
    @pytest.mark.parametrize(
        "checkpoint_format, server_url, expect_warning",
        [
            ("HF", None, False),
            ("MX", "http://mx:8001", False),
            ("HF", "http://mx:8001", True),
        ],
    )
    def test_cross_field_warning(self, checkpoint_format, server_url, expect_warning):
        kwargs = {"checkpoint_format": checkpoint_format}
        if server_url is not None:
            kwargs["mx_config"] = {"server_url": server_url}

        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(**kwargs)

        assert args.mx_config.server_url == server_url

        relevant = _warnings_for("mx_config.server_url", mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant, f"expected a warning about mx_config.server_url; got: {relevant}"
        else:
            assert relevant == [], f"expected no mx_config.server_url warning, got: {relevant}"


class TestMxServerUrlEnvFallback:
    """Honor the upstream ``MODEL_EXPRESS_URL`` env-var fallback for MX."""

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch):
        monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
        yield

    def test_env_populates_when_unset_and_mx(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="MX")
        assert args.mx_config.server_url == "http://from-env:9999"

    def test_explicit_value_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="MX", mx_config={"server_url": "http://explicit:8001"})
        assert args.mx_config.server_url == "http://explicit:8001"

    def test_no_env_no_value_stays_none(self):
        args = _make_args(checkpoint_format="MX")
        assert args.mx_config.server_url is None

    def test_env_ignored_when_not_mx(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="HF")
        assert args.mx_config.server_url is None

    def test_empty_env_does_not_populate(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "")
        args = _make_args(checkpoint_format="MX")
        assert args.mx_config.server_url is None

    def test_resolution_does_not_emit_cross_field_warning(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(checkpoint_format="MX")
        assert args.mx_config.server_url == "http://from-env:9999"
        relevant = _warnings_for("mx_config.server_url", mock_logger.warning.call_args_list)
        assert relevant == [], f"unexpected warnings on env-var fallback: {relevant}"
