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
import logging
import os
import sys
from typing import Dict, Optional

import tensorrt as trt

try:
    from polygraphy.logger import G_LOGGER
except ImportError:
    G_LOGGER = None

# Numeric ordering for severity comparison.  Lower value = more verbose.
_SEVERITY_NUMERIC = {
    "trace": 0,
    "debug": 10,
    "verbose": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "internal_error": 50,
}

# Map severity tag constants (e.g. "[I]") to level names used in _SEVERITY_NUMERIC.
_TAG_TO_LEVEL_NAME = {
    "[F]": "internal_error",
    "[E]": "error",
    "[W]": "warning",
    "[I]": "info",
    "[V]": "verbose",
    "[D]": "debug",
}


def _extract_module(qualname: str) -> str:
    """Extract the first sub-package after ``tensorrt_llm`` from a dotted module name.

    Examples:
        ``tensorrt_llm.runtime.generation``    -> ``runtime``
        ``tensorrt_llm._torch.pyexecutor.foo`` -> ``_torch``
        ``tensorrt_llm.logger``                -> ``logger``
        ``__main__``                           -> ``""``
    """
    parts = qualname.split(".")
    # Find the last occurrence of "tensorrt_llm" (mirrors the C++ rfind approach).
    idx = -1
    for i, p in enumerate(parts):
        if p == "tensorrt_llm":
            idx = i
    if idx >= 0 and idx + 1 < len(parts):
        return parts[idx + 1]
    return ""


_MODULE_WIDTH = 8

# Abbreviation table for module names exceeding _MODULE_WIDTH characters.
_MODULE_ABBREVIATIONS = {
    "auto_parallel": "autoprll",
    "deep_gemm": "deepgemm",
    "flash_mla": "flashmla",
    "quantization": "quantize",
    "scaffolding": "scaffold",
    "_tensorrt_engine": "trt_engn",
    "visual_gen": "vis_gen",
    "__pycache__": "pycache",
    "tokenizer": "tokenizr",
}


def _format_module(name: str) -> str:
    """Return a fixed-width display string for *name* (``_MODULE_WIDTH`` chars).

    Long names are abbreviated via ``_MODULE_ABBREVIATIONS``;
    short names are right-padded with spaces.
    """
    display = _MODULE_ABBREVIATIONS.get(name, name)
    return display[:_MODULE_WIDTH].ljust(_MODULE_WIDTH)


# Cache: filename -> module name (avoids repeated frame inspection).
_filename_to_module: Dict[str, str] = {}


def _get_caller_module() -> str:
    """Walk the call stack to find the first frame outside logger.py and return its module."""
    frame = sys._getframe(1)
    logger_file = __file__
    while frame is not None:
        if frame.f_code.co_filename != logger_file:
            break
        frame = frame.f_back
    if frame is None:
        return ""
    filename = frame.f_code.co_filename
    # Only cache real file paths (not <string>, <stdin>, etc.)
    if not filename.startswith("<"):
        cached = _filename_to_module.get(filename)
        if cached is not None:
            return cached
    module = _extract_module(frame.f_globals.get("__name__", ""))
    if not filename.startswith("<"):
        _filename_to_module[filename] = module
    return module


def _parse_module_levels(env_value: str) -> Dict[str, int]:
    """Parse ``TLLM_LOG_LEVEL_BY_MODULE`` env-var value.

    Format: ``"level:mod1,mod2;level:mod3,mod4"``
    Example: ``"debug:runtime,_torch;info:serve;warning:executor"``

    Returns a dict mapping module name -> numeric severity threshold.
    """
    result: Dict[str, int] = {}
    for group in env_value.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            print(
                f'[TRT-LLM][WARNING] TLLM_LOG_LEVEL_BY_MODULE: skipping malformed group "{group}"',
                file=sys.stderr,
            )
            continue
        level_str, _, module_list = group.partition(":")
        level_str = level_str.strip().lower()
        numeric = _SEVERITY_NUMERIC.get(level_str)
        if numeric is None:
            print(
                f'[TRT-LLM][WARNING] TLLM_LOG_LEVEL_BY_MODULE: unknown level "{level_str}"',
                file=sys.stderr,
            )
            continue
        for mod in module_list.split(","):
            mod = mod.strip()
            if mod:
                result[mod] = numeric
    return result


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    ENV_VARIABLE = "TLLM_LOG_LEVEL"
    PREFIX = "TRT-LLM"
    DEFAULT_LEVEL = "error"

    INTERNAL_ERROR = "[F]"
    ERROR = "[E]"
    WARNING = "[W]"
    INFO = "[I]"
    VERBOSE = "[V]"
    DEBUG = "[D]"

    def __init__(self):
        environ_severity = os.environ.get(self.ENV_VARIABLE)
        self._set_from_env = environ_severity is not None

        self.rank: Optional[int] = None

        min_severity = environ_severity.lower() if self._set_from_env else self.DEFAULT_LEVEL
        invalid_severity = min_severity not in severity_map
        if invalid_severity:
            min_severity = self.DEFAULT_LEVEL

        self._min_severity = min_severity
        self._trt_logger = trt.Logger(severity_map[min_severity][0])
        self._logger = logging.getLogger(self.PREFIX)
        self._logger.propagate = False
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt="[%(asctime)s] %(message)s", datefmt="%m/%d/%Y-%H:%M:%S")
        )
        self._logger.addHandler(handler)

        # Parse per-module log level overrides.
        self._module_levels: Dict[str, int] = {}
        module_env = os.environ.get("TLLM_LOG_LEVEL_BY_MODULE")
        if module_env:
            self._module_levels = _parse_module_levels(module_env)

        # Set the underlying Python logger to the minimum of all configured
        # levels so that per-module overrides more verbose than the global
        # level are not silently dropped by Python's logging framework.
        global_py_level = severity_map[min_severity][1]
        if self._module_levels:
            # Map our numeric levels to Python logging levels.
            _numeric_to_py = {
                0: logging.DEBUG,
                10: logging.DEBUG,
                20: logging.INFO,
                30: logging.WARNING,
                40: logging.ERROR,
                50: logging.CRITICAL,
            }
            min_module_py = min(
                _numeric_to_py.get(v, logging.DEBUG) for v in self._module_levels.values()
            )
            self._logger.setLevel(min(global_py_level, min_module_py))
        else:
            self._logger.setLevel(global_py_level)

        self._polygraphy_logger = G_LOGGER
        if self._polygraphy_logger is not None:
            self._polygraphy_logger.module_severity = severity_map[min_severity][2]

        # For log_once
        self._appeared_keys = set()

        if invalid_severity:
            self.warning(
                f"Requested log level {environ_severity} is invalid. Using '{self.DEFAULT_LEVEL}' instead"
            )

    @property
    def _global_numeric_level(self) -> int:
        return _SEVERITY_NUMERIC.get(self._min_severity, 0)

    def is_severity_enabled(self, severity_tag: str, module: str = "") -> bool:
        """Check whether a message with *severity_tag* should be emitted.

        If *module* is non-empty and has a per-module override, that takes
        precedence over the global level.
        """
        msg_level = _SEVERITY_NUMERIC.get(_TAG_TO_LEVEL_NAME.get(severity_tag, ""), 0)
        if module and self._module_levels:
            mod_threshold = self._module_levels.get(module)
            if mod_threshold is not None:
                return msg_level >= mod_threshold
        return msg_level >= self._global_numeric_level

    def set_rank(self, rank: int):
        self.rank = rank

    def _func_wrapper(self, severity):
        if severity == self.INTERNAL_ERROR:
            return self._logger.critical
        elif severity == self.ERROR:
            return self._logger.error
        elif severity == self.WARNING:
            return self._logger.warning
        elif severity == self.INFO:
            return self._logger.info
        elif severity == self.VERBOSE or severity == self.DEBUG:
            return self._logger.debug
        else:
            raise AttributeError(f"No such severity: {severity}")

    @property
    def trt_logger(self) -> trt.ILogger:
        return self._trt_logger

    def log(self, severity, *msg):
        module = _get_caller_module()
        if not self.is_severity_enabled(severity, module):
            return
        parts = [f"[{self.PREFIX}]"]
        parts.append(severity)
        if module:
            parts.append(f"[{_format_module(module)}]")
        if self.rank is not None:
            parts.append(f"[RANK {self.rank}]")
        parts.extend(map(str, msg))
        self._func_wrapper(severity)(" ".join(parts))

    def log_once(self, severity, *msg, key):
        assert key is not None, "key is required for log_once"
        if key not in self._appeared_keys:
            self._appeared_keys.add(key)
            self.log(severity, *msg)

    def critical(self, *msg):
        self.log(self.INTERNAL_ERROR, *msg)

    def critical_once(self, *msg, key):
        self.log_once(self.INTERNAL_ERROR, *msg, key=key)

    fatal = critical
    fatal_once = critical_once

    def error(self, *msg):
        self.log(self.ERROR, *msg)

    def error_once(self, *msg, key):
        self.log_once(self.ERROR, *msg, key=key)

    def warning(self, *msg):
        self.log(self.WARNING, *msg)

    def warning_once(self, *msg, key):
        self.log_once(self.WARNING, *msg, key=key)

    def info(self, *msg):
        self.log(self.INFO, *msg)

    def info_once(self, *msg, key):
        self.log_once(self.INFO, *msg, key=key)

    def debug(self, *msg):
        self.log(self.VERBOSE, *msg)

    def debug_once(self, *msg, key):
        self.log_once(self.VERBOSE, *msg, key=key)

    @property
    def level(self) -> str:
        return self._min_severity

    def set_level(self, min_severity):
        if self._set_from_env:
            self.warning(
                f"Logger level already set from environment. Discard new verbosity: {min_severity}"
            )
            return
        self._min_severity = min_severity
        self._trt_logger.min_severity = severity_map[min_severity][0]
        self._logger.setLevel(severity_map[min_severity][1])
        if self._polygraphy_logger is not None:
            self._polygraphy_logger.module_severity = severity_map[min_severity][2]


severity_map = {
    "internal_error": [trt.Logger.INTERNAL_ERROR, logging.CRITICAL],
    "error": [trt.Logger.ERROR, logging.ERROR],
    "warning": [trt.Logger.WARNING, logging.WARNING],
    "info": [trt.Logger.INFO, logging.INFO],
    "verbose": [trt.Logger.VERBOSE, logging.DEBUG],
    "debug": [trt.Logger.VERBOSE, logging.DEBUG],
    "trace": [trt.Logger.VERBOSE, logging.DEBUG],
}

if G_LOGGER is not None:
    g_logger_severity_map = {
        "internal_error": G_LOGGER.CRITICAL,
        "error": G_LOGGER.ERROR,
        "warning": G_LOGGER.WARNING,
        "info": G_LOGGER.INFO,
        "verbose": G_LOGGER.SUPER_VERBOSE,
        "debug": G_LOGGER.SUPER_VERBOSE,
        "trace": G_LOGGER.SUPER_VERBOSE,
    }
    for key, value in g_logger_severity_map.items():
        severity_map[key].append(value)

logger = Logger()


def set_level(min_severity):
    logger.set_level(min_severity)
