# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Middleware-order snapshot of a trtllm-serve app."""

from typing import Dict, List

__all__ = ["middleware_snapshot"]


def middleware_snapshot(app) -> List[Dict]:
    """One dict per middleware in ``app.user_middleware``, outermost first.

    Starlette's ``add_middleware`` inserts at position 0, so the last one added
    is outermost and ``app.user_middleware`` already lists outermost first. For
    a class whose module starts with ``tensorrt_llm.`` the keyword names are
    recorded (plus ``expose_headers`` for ``PerfMetricsMiddleware``); for any
    other class only the class name is, because third-party keyword names may
    change between versions.
    """
    snapshot = []
    for middleware in app.user_middleware:
        cls = middleware.cls
        if cls.__module__.startswith("tensorrt_llm."):
            entry = {"class": cls.__name__, "kwargs": sorted(middleware.kwargs)}
            if cls.__name__ == "PerfMetricsMiddleware":
                entry["expose_headers"] = middleware.kwargs["expose_headers"]
        else:
            entry = {"class": cls.__name__}
        snapshot.append(entry)
    return snapshot
