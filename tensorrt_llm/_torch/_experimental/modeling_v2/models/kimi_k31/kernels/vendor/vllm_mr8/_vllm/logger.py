# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vllm.logger stand-in: init_logger returns a logging.Logger with info_once / warning_once."""
import logging

_seen = set()


class _Logger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.pop("scope", None)
        return msg, kwargs

    def _once(self, level, msg, *args, **kwargs):
        key = (level, msg, args)
        if key in _seen:
            return
        _seen.add(key)
        self.log(level, msg, *args, **kwargs)

    def info_once(self, msg, *args, **kwargs):
        self._once(logging.INFO, msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        self._once(logging.WARNING, msg, *args, **kwargs)

    def debug_once(self, msg, *args, **kwargs):
        self._once(logging.DEBUG, msg, *args, **kwargs)


def init_logger(name):
    return _Logger(logging.getLogger(name), {})
