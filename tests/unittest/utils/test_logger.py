# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the modular logger (auto module detection, per-module filtering)."""

import logging
from io import StringIO

import pytest

from tensorrt_llm.logger import (
    Logger,
    _extract_module,
    _format_module,
    _get_caller_module,
    _parse_module_levels,
)


@pytest.fixture
def capture_log():
    """Capture log output via a temporary StringIO handler."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    singleton = Logger()
    singleton._logger.addHandler(handler)
    old_level = singleton._logger.level
    singleton._logger.setLevel(logging.DEBUG)

    yield stream

    singleton._logger.removeHandler(handler)
    singleton._logger.setLevel(old_level)


class TestExtractModule:
    """Tests for _extract_module which parses module name from __name__."""

    def test_standard_subpackage(self):
        assert _extract_module("tensorrt_llm.runtime.generation") == "runtime"

    def test_torch_subpackage(self):
        assert _extract_module("tensorrt_llm._torch.pyexecutor.foo") == "_torch"

    def test_top_level_module(self):
        assert _extract_module("tensorrt_llm.logger") == "logger"

    def test_no_tensorrt_llm_prefix(self):
        assert _extract_module("__main__") == ""

    def test_bare_tensorrt_llm(self):
        assert _extract_module("tensorrt_llm") == ""

    def test_nested_tensorrt_llm(self):
        # rfind picks the last occurrence
        assert _extract_module("some.pkg.tensorrt_llm.serve.server") == "serve"


class TestFormatModule:
    """Tests for _format_module which produces fixed-width display names."""

    def test_short_name_padded(self):
        result = _format_module("serve")
        assert len(result) == 8
        assert result == "serve   "

    def test_exact_8_char(self):
        result = _format_module("executor")
        assert len(result) == 8
        assert result == "executor"

    def test_abbreviation_applied(self):
        result = _format_module("quantization")
        assert result == "quantize"
        assert len(result) == 8

    def test_deep_gemm_abbreviation(self):
        assert _format_module("deep_gemm") == "deepgemm"

    def test_flash_mla_abbreviation(self):
        assert _format_module("flash_mla") == "flashmla"

    def test_auto_parallel_abbreviation(self):
        assert _format_module("auto_parallel") == "autoprll"

    def test_scaffolding_abbreviation(self):
        assert _format_module("scaffolding") == "scaffold"

    def test_unknown_long_name_truncated(self):
        result = _format_module("very_long_module_name")
        assert len(result) == 8
        assert result == "very_lon"


class TestParseModuleLevels:
    """Tests for _parse_module_levels which parses TLLM_LOG_LEVEL_BY_MODULE."""

    def test_single_group(self):
        result = _parse_module_levels("debug:runtime")
        assert result == {"runtime": 10}

    def test_multiple_modules_one_level(self):
        result = _parse_module_levels("info:serve,_torch")
        assert result == {"serve": 20, "_torch": 20}

    def test_multiple_groups(self):
        result = _parse_module_levels("debug:runtime,_torch;warning:executor")
        assert result == {"runtime": 10, "_torch": 10, "executor": 30}

    def test_empty_string(self):
        assert _parse_module_levels("") == {}

    def test_malformed_group_skipped(self):
        result = _parse_module_levels("badformat")
        assert result == {}

    def test_unknown_level_skipped(self):
        result = _parse_module_levels("invalid_level:runtime")
        assert result == {}

    def test_whitespace_handling(self):
        result = _parse_module_levels(" info : serve , _torch ; debug : runtime ")
        assert result == {"serve": 20, "_torch": 20, "runtime": 10}


class TestGetCallerModule:
    """Tests for _get_caller_module auto-detection."""

    def test_from_non_tensorrt_llm(self):
        # Called from this test file (not under tensorrt_llm package)
        module = _get_caller_module()
        assert module == ""

    def test_cache_consistency(self):
        # Repeated calls from same file should return same result
        m1 = _get_caller_module()
        m2 = _get_caller_module()
        assert m1 == m2


class TestLoggerOutput:
    """Tests for Logger log output with auto module detection."""

    def test_log_contains_severity_tag(self, capture_log):
        singleton = Logger()
        singleton._min_severity = "info"

        singleton.info("test message")

        output = capture_log.getvalue()
        assert "[I]" in output
        assert "test message" in output

    def test_warning_output(self, capture_log):
        singleton = Logger()
        singleton._min_severity = "info"

        singleton.warning("warn msg")

        output = capture_log.getvalue()
        assert "[W]" in output
        assert "warn msg" in output

    def test_log_from_fake_module(self, capture_log):
        """Simulate logging from a tensorrt_llm submodule."""
        singleton = Logger()
        singleton._min_severity = "info"

        # Execute logger.info from a fake module context
        fake_globals = {
            "__name__": "tensorrt_llm.serve.router",
            "__file__": "/fake/tensorrt_llm/serve/router.py",
        }
        code = compile(
            "from tensorrt_llm.logger import logger; logger.info('from serve')",
            "/fake/tensorrt_llm/serve/router.py",
            "exec",
        )
        exec(code, fake_globals)

        output = capture_log.getvalue()
        assert "[serve   ]" in output
        assert "from serve" in output


class TestModuleLevelFiltering:
    """Tests for per-module log level filtering."""

    def test_global_level_filters(self):
        singleton = Logger()
        singleton._min_severity = "warning"
        assert singleton.is_severity_enabled(Logger.WARNING) is True
        assert singleton.is_severity_enabled(Logger.INFO) is False

    def test_module_override_more_verbose(self):
        singleton = Logger()
        singleton._min_severity = "error"
        singleton._module_levels = {"runtime": 10}  # debug

        assert singleton.is_severity_enabled(Logger.INFO, "runtime") is True
        assert singleton.is_severity_enabled(Logger.DEBUG, "runtime") is True

        # Cleanup
        singleton._module_levels = {}

    def test_module_override_less_verbose(self):
        singleton = Logger()
        singleton._min_severity = "info"
        singleton._module_levels = {"serve": 40}  # error only

        assert singleton.is_severity_enabled(Logger.INFO, "serve") is False
        assert singleton.is_severity_enabled(Logger.ERROR, "serve") is True

        # Cleanup
        singleton._module_levels = {}

    def test_unmatched_module_uses_global(self):
        singleton = Logger()
        singleton._min_severity = "warning"
        singleton._module_levels = {"other": 10}

        assert singleton.is_severity_enabled(Logger.INFO, "unknown") is False
        assert singleton.is_severity_enabled(Logger.WARNING, "unknown") is True

        # Cleanup
        singleton._module_levels = {}


class TestLoggerAPI:
    """Tests that Logger exposes the expected API."""

    def test_has_level_property(self):
        singleton = Logger()
        assert hasattr(singleton, "level")
        assert isinstance(singleton.level, str)

    def test_has_set_level(self):
        singleton = Logger()
        assert callable(singleton.set_level)

    def test_has_set_rank(self):
        singleton = Logger()
        assert callable(singleton.set_rank)

    def test_has_rank_property(self):
        singleton = Logger()
        assert hasattr(singleton, "rank")

    def test_has_trt_logger(self):
        singleton = Logger()
        assert hasattr(singleton, "trt_logger")

    def test_log_once_deduplication(self, capture_log):
        singleton = Logger()
        singleton._min_severity = "info"

        singleton.info_once("duplicate msg", key="test_key_unique_12345")
        singleton.info_once("duplicate msg", key="test_key_unique_12345")

        output = capture_log.getvalue()
        assert output.count("duplicate msg") == 1

        # Cleanup
        singleton._appeared_keys.discard("test_key_unique_12345")
