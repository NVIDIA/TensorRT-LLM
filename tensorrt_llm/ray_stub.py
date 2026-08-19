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
"""Compatibility shim for ``tensorrt_llm.ray_stub``.

Will be removed once all usages are migrated to
``tensorrt_llm.executor.ray.stub``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

# Bound by attribute rather than with ``from ... import remote``: the target
# module answers every unknown name from a module-level ``__getattr__`` that
# raises ``RuntimeError``, and ``from X import Y`` first probes
# ``hasattr(X, "__path__")``, which only swallows ``AttributeError`` -- so that
# spelling raises while merely importing this file.  ``__getattr__`` is
# forwarded as well: raising for every other name is what the target module is
# for, and re-exporting only ``remote`` would answer ``AttributeError`` here.
from tensorrt_llm.executor.ray import stub as _stub

remote = _stub.remote
__getattr__ = _stub.__getattr__

warnings.warn(
    "tensorrt_llm.ray_stub has moved to tensorrt_llm.executor.ray.stub "
    "and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "remote",
]
