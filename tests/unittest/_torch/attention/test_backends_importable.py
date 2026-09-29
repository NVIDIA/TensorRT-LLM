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

The same file guards the two forwarding modules left behind at the retired
``_torch.modules.attention`` and ``_torch.attention_backend`` paths: they must
hand back the canonical objects themselves, and their re-export list must not
drift away from the canonical one.
"""

import importlib
import sys

import pytest

FMHA_MODULES = [
    "tensorrt_llm._torch.attention.backends.fmha.phased",
    "tensorrt_llm._torch.attention.backends.fmha.prims_ts",
    "tensorrt_llm._torch.attention.backends.fmha.utils",
]

# Import paths retired by the Attention consolidation and kept alive by a
# definition-free forwarding module for the deprecation window.
SHIM_MODULES = [
    "tensorrt_llm._torch.modules.attention",
    "tensorrt_llm._torch.attention_backend",
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


def test_modules_attention_shim_forwards_canonical_class():
    """The retired module path must hand back the canonical class, not a copy of it."""
    shim = importlib.import_module("tensorrt_llm._torch.modules.attention")
    canonical = importlib.import_module("tensorrt_llm._torch.attention.attention")
    assert shim.Attention is canonical.Attention


def test_attention_backend_shim_forwards_canonical_objects():
    """Every name the retired package root re-exports must be the canonical object.

    Identity rather than equality: callers keep isinstance checks and pickles
    against these classes, and a re-exported copy would silently fail both.
    """
    shim = importlib.import_module("tensorrt_llm._torch.attention_backend")
    canonical = importlib.import_module("tensorrt_llm._torch.attention.backends")
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(canonical, name), name


def test_attention_backend_shim_exports_match_canonical():
    """The shim's ``__all__`` must track the canonical package's.

    The shim was first written while the canonical ``__all__`` still carried
    ``StarAttention``/``StarAttentionMetadata``; those were later deleted, and
    a stale copy of the list only surfaces as an import error on a machine that
    has FlashInfer -- which is exactly the configuration the extra names serve.
    """
    shim = importlib.import_module("tensorrt_llm._torch.attention_backend")
    canonical = importlib.import_module("tensorrt_llm._torch.attention.backends")
    # Compared as sets, not as ordered lists. ``__all__`` ordering binds nothing
    # -- ``from ... import *`` is order-insensitive -- so freezing it would fail
    # CI on a cosmetic reorder of the canonical list while catching no drift a
    # caller could observe. Membership is the contract: a name present on only
    # one side is what actually breaks, or silently stops serving, a caller.
    assert canonical.__all__, "an empty canonical __all__ would make this guard vacuous"
    assert set(shim.__all__) == set(canonical.__all__)


@pytest.mark.parametrize("module_name", SHIM_MODULES)
def test_shim_warns_on_import(module_name):
    """Importing a retired path must warn, and specifically with FutureWarning.

    DeprecationWarning is on Python's stock ignore list outside ``__main__``, so
    it would never reach the out-of-tree callers these modules exist for.

    The warning fires while the module body runs, and an earlier test in this
    process may already have imported the shim, so the entry is dropped from
    ``sys.modules`` to force exactly one fresh execution and put back afterwards.
    Re-running the body is cheap: everything it imports is already loaded.
    """
    parent_name, _, attribute = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    previous = sys.modules.pop(module_name, None)
    try:
        with pytest.warns(FutureWarning, match="has moved to"):
            importlib.import_module(module_name)
    finally:
        if previous is not None:
            sys.modules[module_name] = previous
            setattr(parent, attribute, previous)
