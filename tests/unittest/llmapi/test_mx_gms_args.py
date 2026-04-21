# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for MX/GMS prototype config fields on ``TorchLlmArgs``.

Covers the prototype fields added by PR #13045:
- ``mx_server_url``
- ``mx_preshard_strategy``
- ``gms_socket_path``
- ``gms_mode``
- ``gms_tag``

These tests exercise only Pydantic validation behavior — no upstream
``modelexpress`` or ``gpu_memory_service`` library is imported.
"""

from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import LoadFormat, TorchLlmArgs

# tensorrt_llm uses a custom Singleton logger (tensorrt_llm.logger.logger)
# that does NOT route through stdlib logging, so pytest's ``caplog`` does
# not intercept its warnings. We patch ``logger.warning`` directly with a
# MagicMock to inspect calls instead.
_LOGGER_PATH = "tensorrt_llm.llmapi.llm_args.logger"


def _capture_warnings(call_args_list):
    """Flatten captured ``logger.warning(fmt, *args)`` calls into rendered strings."""
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
    """Return only captured warnings whose rendered text mentions ``keyword``."""
    return [m for m in _capture_warnings(call_args_list) if keyword in m]


# A tiny model path is fine — TorchLlmArgs lazily resolves the model;
# none of the validators we exercise here actually touches disk.
_DUMMY_MODEL = "/tmp/test-mx-gms-args-nonexistent"


def _make_args(**overrides) -> TorchLlmArgs:
    """Construct a ``TorchLlmArgs`` with the given overrides."""
    return TorchLlmArgs(model=_DUMMY_MODEL, **overrides)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """Default values must match the conventions of the upstream libraries."""

    def test_mx_server_url_default_none(self):
        args = _make_args()
        assert args.mx_server_url is None

    def test_mx_preshard_strategy_default_per_module(self):
        args = _make_args()
        assert args.mx_preshard_strategy == "per_module"

    def test_gms_socket_path_default_none(self):
        # Default None means "resolve via gpu_memory_service.common.utils.
        # get_socket_path(device, tag) at connect time".
        args = _make_args()
        assert args.gms_socket_path is None

    def test_gms_mode_default_auto(self):
        args = _make_args()
        assert args.gms_mode == "auto"

    def test_gms_tag_default_weights(self):
        # IMPORTANT: must be ``weights``, NOT ``model_weights`` — the GMS
        # library convention uses ``weights`` for model weights and
        # ``kv_cache`` for the KV cache (see GMS_TAGS upstream).
        args = _make_args()
        assert args.gms_tag == "weights"


# ---------------------------------------------------------------------------
# mx_preshard_strategy validator
# ---------------------------------------------------------------------------


class TestMxPreshardStrategy:
    def test_accepts_per_module(self):
        args = _make_args(checkpoint_format="MX", mx_preshard_strategy="per_module")
        assert args.mx_preshard_strategy == "per_module"

    def test_accepts_global(self):
        # 'global' is accepted at config time even though selecting it
        # currently raises NotImplementedError when MX P2P succeeds at
        # runtime (LoadFormat.PRESHARDED isn't wired upstream yet).
        args = _make_args(checkpoint_format="MX", mx_preshard_strategy="global")
        assert args.mx_preshard_strategy == "global"

    @pytest.mark.parametrize(
        "bad_value",
        ["", "Per_Module", "PER_MODULE", "global ", "perModule", "all", "none", "1", " "],
    )
    def test_rejects_unknown_value(self, bad_value):
        with pytest.raises(ValueError, match="mx_preshard_strategy"):
            _make_args(mx_preshard_strategy=bad_value)

    @pytest.mark.parametrize(
        "checkpoint_format, mx_preshard_strategy, expect_warning",
        [
            # Default per_module + default HF: no cross-field warning.
            ("HF", "per_module", False),
            # per_module + MX active: no warning.
            ("MX", "per_module", False),
            # global + MX active: meaningful, no warning.
            ("MX", "global", False),
            # global + HF active: warn that the strategy is ignored.
            ("HF", "global", True),
        ],
    )
    def test_cross_field_warning(self, checkpoint_format, mx_preshard_strategy, expect_warning):
        with patch(_LOGGER_PATH) as mock_logger:
            _make_args(
                checkpoint_format=checkpoint_format, mx_preshard_strategy=mx_preshard_strategy
            )
        relevant = _warnings_for("mx_preshard_strategy", mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant, "expected a warning mentioning mx_preshard_strategy"
            assert any("checkpoint_format" in m for m in relevant), (
                "expected the warning to also mention checkpoint_format"
            )
        else:
            assert relevant == [], f"expected no mx_preshard_strategy warning, got: {relevant}"


# ---------------------------------------------------------------------------
# mx_server_url validator
# ---------------------------------------------------------------------------


class TestMxServerUrl:
    @pytest.mark.parametrize(
        "checkpoint_format, mx_server_url, expect_warning",
        [
            # Default unset: no warning.
            ("HF", None, False),
            # Set with MX: no cross-field warning.
            ("MX", "http://mx:8001", False),
            # Set without MX: warn that the URL is ignored.
            ("HF", "http://mx:8001", True),
        ],
    )
    def test_cross_field_warning(self, checkpoint_format, mx_server_url, expect_warning):
        kwargs = {"checkpoint_format": checkpoint_format}
        if mx_server_url is not None:
            kwargs["mx_server_url"] = mx_server_url

        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(**kwargs)

        # Field is always stored verbatim (warnings are advisory, not rejecting).
        assert args.mx_server_url == mx_server_url

        relevant = _warnings_for("mx_server_url", mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant, f"expected a warning about mx_server_url; got: {relevant}"
        else:
            assert relevant == [], f"expected no mx_server_url warning, got: {relevant}"


# ---------------------------------------------------------------------------
# gms_mode validator (only enforced when load_format == GMS)
# ---------------------------------------------------------------------------


class TestGmsMode:
    @pytest.mark.parametrize("mode", ["auto", "rw", "ro"])
    def test_accepts_known_modes_with_gms(self, mode):
        args = _make_args(load_format=LoadFormat.GMS, gms_mode=mode)
        assert args.gms_mode == mode

    @pytest.mark.parametrize("bad_mode", ["", "AUTO", "Rw", "read", "write", "shadow"])
    def test_rejects_unknown_mode_with_gms(self, bad_mode):
        with pytest.raises(ValueError, match="gms_mode"):
            _make_args(load_format=LoadFormat.GMS, gms_mode=bad_mode)

    def test_unknown_mode_without_gms_load_format_is_lenient(self):
        # Validator only enforces gms_mode values when load_format == GMS.
        # Without GMS, the field is unused so we don't reject odd values.
        args = _make_args(load_format=LoadFormat.AUTO, gms_mode="not-a-mode")
        assert args.gms_mode == "not-a-mode"


# ---------------------------------------------------------------------------
# gms_socket_path warning
# ---------------------------------------------------------------------------


class TestGmsSocketPath:
    @pytest.mark.parametrize(
        "load_format, expect_warning",
        [
            # Set with GMS: no cross-field warning.
            (LoadFormat.GMS, False),
            # Set without GMS: warn that the path is ignored.
            (LoadFormat.AUTO, True),
        ],
    )
    def test_cross_field_warning(self, load_format, expect_warning):
        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(load_format=load_format, gms_socket_path="/tmp/gms.sock")
        assert args.gms_socket_path == "/tmp/gms.sock"
        relevant = _warnings_for("gms_socket_path", mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant, f"expected a warning about gms_socket_path; got: {relevant}"
        else:
            assert relevant == [], f"expected no gms_socket_path warning, got: {relevant}"


# ---------------------------------------------------------------------------
# Composition: two-axis validity
# ---------------------------------------------------------------------------


class TestTwoAxisComposition:
    """Sanity checks for the two-axis (checkpoint_format × LoadFormat) design."""

    def test_pure_trtllm_baseline(self):
        args = _make_args()
        assert args.checkpoint_format == "HF"
        assert args.load_format == LoadFormat.AUTO

    def test_mx_only(self):
        args = _make_args(checkpoint_format="MX", mx_server_url="http://mx:8001")
        assert args.checkpoint_format == "MX"
        assert args.load_format == LoadFormat.AUTO  # default
        assert args.mx_server_url == "http://mx:8001"

    def test_gms_only(self):
        args = _make_args(load_format=LoadFormat.GMS, gms_socket_path="/tmp/gms.sock")
        assert args.checkpoint_format == "HF"  # default
        assert args.load_format == LoadFormat.GMS
        assert args.gms_socket_path == "/tmp/gms.sock"

    def test_mx_plus_gms(self):
        args = _make_args(
            checkpoint_format="MX",
            mx_server_url="http://mx:8001",
            load_format=LoadFormat.GMS,
            gms_socket_path="/tmp/gms.sock",
        )
        assert args.checkpoint_format == "MX"
        assert args.load_format == LoadFormat.GMS
        assert args.mx_server_url == "http://mx:8001"
        assert args.gms_socket_path == "/tmp/gms.sock"


# ---------------------------------------------------------------------------
# LoadFormat enum sanity
# ---------------------------------------------------------------------------


class TestLoadFormatEnum:
    def test_gms_enum_present(self):
        # The prototype adds LoadFormat.GMS = 3.
        assert hasattr(LoadFormat, "GMS")
        assert LoadFormat.GMS.name == "GMS"

    def test_pre_existing_enums_unchanged(self):
        # Sanity — make sure the prototype didn't reshuffle enum values.
        assert LoadFormat.AUTO.value == 0
        assert LoadFormat.DUMMY.value == 1
        assert LoadFormat.VISION_ONLY.value == 2

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0, LoadFormat.AUTO),
            (1, LoadFormat.DUMMY),
            (2, LoadFormat.VISION_ONLY),
            (3, LoadFormat.GMS),
        ],
    )
    def test_int_value_is_converted(self, raw, expected):
        # ``TorchLlmArgs`` accepts raw int load_format values via the
        # ``Union[str, LoadFormat]`` field plus its convert_load_format
        # validator. Pin the new GMS=3 mapping (alongside the pre-existing
        # values) so we catch any future enum-value reshuffle.
        args = _make_args(load_format=raw)
        assert args.load_format == expected


# ---------------------------------------------------------------------------
# gms_tag validator — reject empty / whitespace-only
# ---------------------------------------------------------------------------


class TestGmsTagValidator:
    """``gms_tag`` must be a non-empty string.

    Empty or whitespace-only tags would silently collide with the default
    ``"weights"`` pool key and produce hard-to-debug cross-process state
    corruption inside GMS, so the validator rejects them up-front.
    """

    @pytest.mark.parametrize("bad_tag", ["", " ", "\t", "\n", "   \t  "])
    def test_rejects_empty_or_whitespace(self, bad_tag):
        with pytest.raises(ValueError, match="gms_tag must be a non-empty"):
            _make_args(load_format=LoadFormat.GMS, gms_tag=bad_tag)

    def test_rejects_when_load_format_not_gms(self):
        # Validation is unconditional — even when GMS isn't active we
        # still want a meaningful tag, because the field has no other
        # meaning and an empty value is always a mistake.
        with pytest.raises(ValueError, match="gms_tag must be a non-empty"):
            _make_args(load_format=LoadFormat.AUTO, gms_tag="")

    def test_accepts_default(self):
        args = _make_args(load_format=LoadFormat.GMS)
        assert args.gms_tag == "weights"

    def test_accepts_custom_non_empty(self):
        args = _make_args(load_format=LoadFormat.GMS, gms_tag="kv_cache")
        assert args.gms_tag == "kv_cache"


# ---------------------------------------------------------------------------
# MODEL_EXPRESS_URL env-var fallback (validator-level)
# ---------------------------------------------------------------------------


class TestMxServerUrlEnvFallback:
    """Honor the upstream ``MODEL_EXPRESS_URL`` env-var fallback for MX.

    The fallback fires when ``checkpoint_format='MX'`` and
    ``mx_server_url`` is unset. It resolves at validator time so the
    value ends up on ``llm_args.mx_server_url`` (visible to logging /
    startup metrics) rather than being silently re-read by the loader.
    """

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch):
        # Each test starts from a clean MODEL_EXPRESS_URL state.
        monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
        yield

    def test_env_populates_when_unset_and_mx(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="MX")
        assert args.mx_server_url == "http://from-env:9999"

    def test_explicit_value_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="MX", mx_server_url="http://explicit:8001")
        assert args.mx_server_url == "http://explicit:8001"

    def test_no_env_no_value_stays_none(self):
        args = _make_args(checkpoint_format="MX")
        assert args.mx_server_url is None

    def test_env_ignored_when_not_mx(self, monkeypatch):
        # The env var only applies when MX is the active checkpoint
        # format. Otherwise we'd silently advertise an MX URL on a
        # non-MX config — confusing.
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        args = _make_args(checkpoint_format="HF")
        assert args.mx_server_url is None

    def test_empty_env_does_not_populate(self, monkeypatch):
        # An empty string in the env should NOT be treated as configured.
        monkeypatch.setenv("MODEL_EXPRESS_URL", "")
        args = _make_args(checkpoint_format="MX")
        assert args.mx_server_url is None

    def test_resolution_does_not_emit_cross_field_warning(self, monkeypatch):
        # When the URL comes from the env *with* MX active, the
        # cross-field "is set but checkpoint_format != MX" warning must
        # not fire (because the format DOES match).
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://from-env:9999")
        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(checkpoint_format="MX")
        assert args.mx_server_url == "http://from-env:9999"
        relevant = _warnings_for("mx_server_url", mock_logger.warning.call_args_list)
        assert relevant == [], f"unexpected warnings on env-var fallback: {relevant}"
