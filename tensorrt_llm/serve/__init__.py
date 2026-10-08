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

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .openai_disagg_server import OpenAIDisaggServer
    from .openai_server import OpenAIServer

__all__ = ['OpenAIServer', 'OpenAIDisaggServer']

# Public name -> source module. The servers are loaded on first access so that
# importing a light submodule (``tensorrt_llm.serve.chat_tokenization``,
# ``tensorrt_llm.serve.render``) does not construct both server modules and the
# engine stack behind them.
_LAZY_ATTRS = {
    'OpenAIServer': 'tensorrt_llm.serve.openai_server',
    'OpenAIDisaggServer': 'tensorrt_llm.serve.openai_disagg_server',
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        # Unknown names must raise AttributeError so that
        # ``from tensorrt_llm.serve import <submodule>`` falls back to the
        # import system.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
