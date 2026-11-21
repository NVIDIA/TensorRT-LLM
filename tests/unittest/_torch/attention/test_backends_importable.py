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
"""Guards the attention backend import graph.

``tensorrt_llm._torch.attention.backends`` pulls in the whole ``fmha``
package, so a stale module path in any of its members breaks collection of
every test that touches an attention backend. The executor's model engine
imports the attention backends too (directly and through ``engine.lora``),
so a stale path there takes the whole PyTorch runtime down with it. These
checks are import-only and run on CPU.
"""

import importlib

import pytest

FMHA_MODULES = [
    "tensorrt_llm._torch.attention.backends.fmha.phased",
    "tensorrt_llm._torch.attention.backends.fmha.prims_ts",
    "tensorrt_llm._torch.attention.backends.fmha.utils",
]


def test_attention_backends_package_imports():
    importlib.import_module("tensorrt_llm._torch.attention.backends")


@pytest.mark.parametrize("module_name", FMHA_MODULES)
def test_fmha_module_imports(module_name):
    importlib.import_module(module_name)


def test_engine_lora_imports():
    """``engine.lora`` sits on the model engine's import path."""
    module = importlib.import_module("tensorrt_llm._torch.pyexecutor.engine.lora")
    assert hasattr(module, "AttentionMetadata")


def test_kv_cache_manager_v2_names_resolve():
    """The names the fmha modules import must exist at their source."""
    module = importlib.import_module("tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2")
    assert hasattr(module, "KVCacheManagerV2")
    assert hasattr(module, "Role")
