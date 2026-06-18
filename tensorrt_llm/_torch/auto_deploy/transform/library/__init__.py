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
"""AutoDeploy's library of transforms.

This file ensures that all publicly listed files/transforms in the library folder are auto-imported
and the corresponding transforms are registered.
"""

import importlib
import logging
import pkgutil

from ..._compat import TRTLLM_AVAILABLE

_logger = logging.getLogger(__name__)

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{module_name}")
        __all__.append(module_name)
    except (ModuleNotFoundError, ImportError, AttributeError) as exc:
        if not TRTLLM_AVAILABLE:
            _logger.debug("Skipping transform %s (standalone mode): %s", module_name, exc)
        else:
            raise
