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
"""The compatibility modules kept at the pre-move Ray import paths.

Each one must forward every name in its own ``__all__`` to the module it
replaces, as the same object.  The names are read from the module rather than
restated here, so changing the export set changes what is verified.
"""

import importlib
import importlib.util
import sys
import warnings
from types import ModuleType

import pytest

# Two of the modules below import Ray unconditionally, so the whole file needs
# it -- the test list schedules this where Ray is installed, and this keeps the
# file honest anywhere else.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="the modules under test import Ray"
)

# The forwarding module -> the module it forwards to.
FORWARDS: dict[str, str] = {
    "tensorrt_llm._ray_utils": "tensorrt_llm.executor.ray.utils",
    "tensorrt_llm.ray_stub": "tensorrt_llm.executor.ray.stub",
    "tensorrt_llm.executor.ray_executor": "tensorrt_llm.executor.ray.executor",
    "tensorrt_llm.executor.ray_gpu_worker": "tensorrt_llm.executor.ray.gpu_worker",
}


def _import_pair(forwarding_path: str) -> tuple[ModuleType, ModuleType, str]:
    """Import a forwarding module and the module it forwards to.

    Returns:
        The forwarding module, the module it forwards to, and the latter's
        dotted name.
    """
    target_path = FORWARDS[forwarding_path]
    return (
        importlib.import_module(forwarding_path),
        importlib.import_module(target_path),
        target_path,
    )


@pytest.mark.parametrize("forwarding_path", sorted(FORWARDS))
def test_published_names_are_the_same_objects(forwarding_path: str) -> None:
    """Identity, not equality.

    A re-implementation would compare equal and still break ``isinstance`` and
    unpickling for callers that kept the old import path.
    """
    forwarding, target, target_path = _import_pair(forwarding_path)

    published = getattr(forwarding, "__all__", None)
    assert published, f"{forwarding_path} publishes nothing, so it forwards nothing"

    for name in published:
        assert hasattr(target, name), f"{target_path} has no {name!r} to forward to"
        assert getattr(forwarding, name) is getattr(target, name), name


@pytest.mark.parametrize("forwarding_path", sorted(FORWARDS))
def test_objects_report_the_module_that_defines_them(forwarding_path: str) -> None:
    """Forwarding must not rewrite ``__module__``.

    Objects have to keep pointing at where they are defined, or pickles written
    now would record a path that is going away.
    """
    forwarding, target, target_path = _import_pair(forwarding_path)

    for name in forwarding.__all__:
        module = getattr(getattr(target, name), "__module__", None)
        if module is None:  # not every object carries one
            continue
        assert module == target_path, f"{name}.__module__ is {module!r}"


@pytest.mark.parametrize("forwarding_path", sorted(FORWARDS))
def test_import_warns_once_and_names_the_new_path(forwarding_path: str) -> None:
    """Importing the old path has to say so, and say where to go instead."""
    target_path = FORWARDS[forwarding_path]

    # A module body runs once per interpreter, so drop it before re-importing.
    sys.modules.pop(forwarding_path, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(forwarding_path)

    # Deliberately not pinned to a warning category: what matters is that one
    # warning is raised and that it points at the replacement.
    about_this_module = [w for w in caught if forwarding_path in str(w.message)]
    assert len(about_this_module) == 1, [str(w.message) for w in caught]
    assert target_path in str(about_this_module[0].message)
