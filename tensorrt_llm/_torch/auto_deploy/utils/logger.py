# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Singleton(type):
    """Metaclass that ensures only one instance of a class exists."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ADLogger(metaclass=Singleton):
    """Logger for auto_deploy using Python's standard logging module.

    Provides the same API surface as TRT-LLM's Logger class so auto_deploy code
    works identically in both standalone and TRT-LLM-integrated modes.

    Uses the Singleton metaclass to ensure a single logger instance.
    """

    ENV_VARIABLE = "AUTO_DEPLOY_LOG_LEVEL"
    PREFIX = "AUTO-DEPLOY"
    DEFAULT_LEVEL = "info"

    # Severity constants matching TRT-LLM's Logger
    INTERNAL_ERROR = "internal_error"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    VERBOSE = "verbose"
    DEBUG = "debug"

    _SEVERITY_TO_LEVEL = {
        "internal_error": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "verbose": logging.DEBUG,
        "debug": logging.DEBUG,
    }

    def __init__(self):
        self._logger = logging.getLogger("auto_deploy")
        level_str = os.environ.get(self.ENV_VARIABLE, self.DEFAULT_LEVEL).upper()
        self._logger.setLevel(getattr(logging, level_str, logging.INFO))
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(f"[{self.PREFIX}] [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
        self.rank = None
        self._appeared_keys = set()

    def set_rank(self, rank: int):
        self.rank = rank

    def log(self, severity, *msg):
        level = self._SEVERITY_TO_LEVEL.get(severity, logging.INFO)
        parts = []
        if self.rank is not None:
            parts.append(f"[RANK {self.rank}]")
        parts.extend(map(str, msg))
        self._logger.log(level, " ".join(parts))

    def log_once(self, severity, *msg, key):
        if key not in self._appeared_keys:
            self._appeared_keys.add(key)
            self.log(severity, *msg)

    def info(self, *msg):
        self.log(self.INFO, *msg)

    def warning(self, *msg):
        self.log(self.WARNING, *msg)

    def error(self, *msg):
        self.log(self.ERROR, *msg)

    def debug(self, *msg):
        self.log(self.DEBUG, *msg)

    def warning_once(self, *msg, key):
        self.log_once(self.WARNING, *msg, key=key)

    def debug_once(self, *msg, key):
        self.log_once(self.DEBUG, *msg, key=key)

    def set_level(self, level):
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)


ad_logger = ADLogger()
