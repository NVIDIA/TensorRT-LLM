# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import LoadFormat, TorchLlmArgs

_LOGGER_PATH = "tensorrt_llm.llmapi.llm_args.logger"
_DUMMY_MODEL = "/tmp/test-gms-args-nonexistent"


def _make_args(**overrides) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, **overrides)


def _capture_warnings(call_args_list):
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
    return [m for m in _capture_warnings(call_args_list) if keyword in m]


class TestDefaults:

    def test_gms_socket_path_default_none(self):
        assert _make_args().gms_socket_path is None

    def test_gms_mode_default_auto(self):
        assert _make_args().gms_mode == "auto"

    def test_gms_tag_default_weights(self):
        assert _make_args().gms_tag == "weights"


class TestGmsMode:

    @pytest.mark.parametrize("mode", ["auto", "rw", "ro"])
    def test_accepts_known_modes_with_gms(self, mode):
        assert _make_args(load_format=LoadFormat.GMS, gms_mode=mode).gms_mode == mode

    @pytest.mark.parametrize("bad_mode", ["", "AUTO", "Rw", "read", "write", "shadow"])
    def test_rejects_unknown_mode_with_gms(self, bad_mode):
        with pytest.raises(ValueError, match="gms_mode"):
            _make_args(load_format=LoadFormat.GMS, gms_mode=bad_mode)

    def test_unknown_mode_without_gms_is_lenient(self):
        assert _make_args(load_format=LoadFormat.AUTO,
                          gms_mode="not-a-mode").gms_mode == "not-a-mode"


class TestGmsSocketPath:

    @pytest.mark.parametrize(
        "load_format, expect_warning",
        [
            (LoadFormat.GMS, False),
            (LoadFormat.AUTO, True),
        ],
    )
    def test_cross_field_warning(self, load_format, expect_warning):
        with patch(_LOGGER_PATH) as mock_logger:
            args = _make_args(load_format=load_format,
                              gms_socket_path="/tmp/gms.sock")

        assert args.gms_socket_path == "/tmp/gms.sock"
        relevant = _warnings_for("gms_socket_path",
                                 mock_logger.warning.call_args_list)
        if expect_warning:
            assert relevant
        else:
            assert relevant == []


class TestLoadFormatEnum:

    def test_gms_enum_present(self):
        assert hasattr(LoadFormat, "GMS")
        assert LoadFormat.GMS.name == "GMS"

    def test_pre_existing_values_unchanged(self):
        assert LoadFormat.AUTO.value == 0
        assert LoadFormat.DUMMY.value == 1
        assert LoadFormat.VISION_ONLY.value == 2


class TestComposition:

    def test_gms_only(self):
        args = _make_args(load_format=LoadFormat.GMS,
                          gms_socket_path="/tmp/gms.sock")
        assert args.checkpoint_format == "HF"
        assert args.load_format == LoadFormat.GMS
        assert args.gms_socket_path == "/tmp/gms.sock"
