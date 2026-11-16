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
"""The compatibility modules kept at the pre-move PEFT import paths.

Each one must forward every name in its own ``__all__`` to the module that
defines it, as the same object.  The names are read from the module rather than
restated here, so changing the export set changes what is verified.

``lora_manager`` is the interesting case: it is the only one whose names do not
all live in a single module.  It was one file before the move and is three now,
so the test resolves each name to wherever it is actually defined instead of
assuming a single destination -- which is also what stops the shim from
quietly pointing a name at a module that merely re-exports it.
"""

import importlib
import sys
import warnings

import pytest

pytestmark = pytest.mark.cpu_only

FORWARDING_PATHS = (
    "tensorrt_llm.lora_helper",
    "tensorrt_llm.lora_manager",
    "tensorrt_llm.prompt_adapter_manager",
)

CANONICAL_PREFIX = "tensorrt_llm._torch.peft."


@pytest.mark.parametrize("forwarding_path", FORWARDING_PATHS)
def test_published_names_come_from_where_they_are_defined(forwarding_path: str) -> None:
    """Identity against the defining module, not against the shim's target.

    A re-implementation would compare equal and still break ``isinstance`` and
    unpickling for callers that kept the old import path.  Resolving through
    ``__module__`` additionally catches a shim that forwards a name to a module
    that only re-exports it: that spelling resolves, so nothing fails until the
    incidental re-export goes away.
    """
    forwarding = importlib.import_module(forwarding_path)

    published = getattr(forwarding, "__all__", None)
    assert published, f"{forwarding_path} publishes nothing, so it forwards nothing"

    for name in published:
        obj = getattr(forwarding, name)
        home = getattr(obj, "__module__", None)
        assert home is not None, f"{name} carries no __module__ to check"
        assert home.startswith(CANONICAL_PREFIX), f"{name}.__module__ is {home!r}"
        assert getattr(importlib.import_module(home), name) is obj, name


@pytest.mark.parametrize("forwarding_path", FORWARDING_PATHS)
def test_import_warns_once_and_names_the_new_path(forwarding_path: str) -> None:
    """Importing the old path has to say so, and say where to go instead."""
    # A module body runs once per interpreter, so drop it before re-importing.
    sys.modules.pop(forwarding_path, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forwarding = importlib.import_module(forwarding_path)

    about_this_module = [w for w in caught if forwarding_path in str(w.message)]
    assert len(about_this_module) == 1, [str(w.message) for w in caught]

    message = str(about_this_module[0].message)
    for name in forwarding.__all__:
        home = getattr(getattr(forwarding, name), "__module__", "")
        assert home in message, f"the warning does not mention {home}"


def test_the_restricted_unpickle_allowlist_still_accepts_the_old_module_strings() -> None:
    """``find_class`` matches the key exactly, before any import happens.

    So the forwarding modules alone do not cover deserialization: a pickle
    written before the move names the old module, and a canonical-only table
    rejects it without ever reaching the shim.
    """
    from tensorrt_llm.llmapi.serialization import BASE_EXAMPLE_CLASSES

    canonical = "tensorrt_llm._torch.peft.lora.config"
    assert "LoraConfig" in BASE_EXAMPLE_CLASSES[canonical]
    # lora_helper is the module a real pre-move LoraConfig pickle names.
    assert "LoraConfig" in BASE_EXAMPLE_CLASSES["tensorrt_llm.lora_helper"]
    assert "LoraConfig" in BASE_EXAMPLE_CLASSES["tensorrt_llm.lora_manager"]
