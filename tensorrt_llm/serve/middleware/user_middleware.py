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
"""Middleware that users add with ``trtllm-serve --middleware``."""

import importlib
import inspect
from typing import Sequence


def _apply_fastapi_middlewares(app, middlewares: Sequence[str]) -> None:
    """Import and register middleware objects on a FastAPI app."""
    for middleware in middlewares:
        try:
            module_path, object_name = middleware.rsplit(".", 1)
        except ValueError as e:
            raise ValueError(f"Invalid middleware import path '{middleware}'. "
                             "Expected format: <module>.<object>.") from e

        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             "Must be a class or an async function.")
