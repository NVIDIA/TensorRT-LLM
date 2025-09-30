# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import tensorrt as trt

try:
    from polygraphy.logger import G_LOGGER
except ImportError:
    G_LOGGER = None


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
        self._logger.setLevel(severity_map[min_severity][1])
        self._polygraphy_logger = G_LOGGER
        if self._polygraphy_logger is not None:
            self._polygraphy_logger.module_severity = severity_map[min_severity][2]

        # For log_once
        self._appeared_keys = set()

        if invalid_severity:
            self.warning(
                f"Requested log level {environ_severity} is invalid. Using '{self.DEFAULT_LEVEL}' instead"
            )

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
        parts = [f"[{self.PREFIX}]"]
        if self.rank is not None:
            parts.append(f"[RANK {self.rank}]")
        parts.append(severity)
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
