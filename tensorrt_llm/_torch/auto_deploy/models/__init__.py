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
import importlib
import logging

_logger = logging.getLogger(__name__)

from .factory import *  # noqa: E402, F401, F403

# Import model submodules individually so that modules with transitive TRT-LLM
# dependencies (e.g., eagle needing MTPDecodingConfig) don't prevent others
# from loading in standalone mode.
for _name in ("custom", "eagle", "hf", "nemotron_flash", "patches"):
    try:
        importlib.import_module(f".{_name}", __name__)
    except (ImportError, ModuleNotFoundError) as _exc:
        _logger.debug("Skipping models.%s: %s", _name, _exc)
