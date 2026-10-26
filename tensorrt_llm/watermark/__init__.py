# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Statistical text watermarking: configuration, PRF, keyed sampling and detection."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import WatermarkConfig
    from .detector import GumbelWatermarkDetector, WatermarkDetection

__all__ = ["WatermarkConfig", "GumbelWatermarkDetector", "WatermarkDetection"]

# Resolved lazily: the config pulls in the LLM API, which the sampler (a consumer of the
# lightweight ``prf`` and ``sampling`` submodules) must not import eagerly.
_LAZY = {
    "WatermarkConfig": ".config",
    "GumbelWatermarkDetector": ".detector",
    "WatermarkDetection": ".detector",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
