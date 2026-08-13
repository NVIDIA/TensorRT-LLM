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
"""The compatibility modules kept at the pre-move request-contract paths.

These exist for importers outside this repository; nothing in the tree imports
them any more, so without a test a later cleanup could break them silently.

What is checked is the promise each module makes in its own ``__all__`` -- that
every name it publishes is the *same object* as at the module it forwards to.
The names are read from the module rather than restated here, so changing the
export set changes what is verified.

CPU only.
"""

import contextlib
import importlib
import sys
import warnings
from collections.abc import Iterator
from types import ModuleType

import pytest

# The forwarding module -> the module it forwards to.
FORWARDS: dict[str, str] = {
    "tensorrt_llm.sampling_params": "tensorrt_llm.executor.params.sampling",
    "tensorrt_llm.disaggregated_params": "tensorrt_llm.executor.params.disaggregation",
    "tensorrt_llm.scheduling_params": "tensorrt_llm.executor.params.scheduling",
    "tensorrt_llm.conversation_params": "tensorrt_llm.executor.params.conversation",
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
def test_no_object_claims_the_retired_module(forwarding_path: str) -> None:
    """Forwarding must not rewrite ``__module__``.

    A pickle records the class's module and the restricted unpickler matches
    that string exactly, so nothing may report the path that is going away.
    Names that come from elsewhere -- a ``typing`` alias, say -- keep naming
    wherever they are really defined.
    """
    forwarding, target, _target_path = _import_pair(forwarding_path)

    for name in forwarding.__all__:
        module = getattr(getattr(target, name), "__module__", None)
        assert module != forwarding_path, f"{name}.__module__ impersonates the old path"


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


def test_restricted_unpickle_allowlist_keeps_both_module_keys() -> None:
    """``find_class()`` matches the pickled module string exactly.

    It rejects before the forwarding module is ever imported, so dropping the
    old key would silently stop old artifacts from loading -- the one thing the
    forwarding modules cannot fix on their own.
    """
    from tensorrt_llm.llmapi.serialization import BASE_EXAMPLE_CLASSES

    for forwarding_path, target_path in FORWARDS.items():
        # A contract may be absent from the allowlist entirely, but never on one
        # side only: a target-only entry rejects artifacts written before the move.
        legacy, target = (
            forwarding_path in BASE_EXAMPLE_CLASSES,
            target_path in BASE_EXAMPLE_CLASSES,
        )
        assert legacy == target, (
            f"{forwarding_path} present={legacy} but {target_path} present={target}"
        )


# Fewer than the allowlist lists, because two of its names cannot round-trip:
# GreedyDecodingParams, which no class implements, and SamplingParams, whose
# logprobs_mode field defaults to a LogprobMode member -- a name the allowlist
# does not authorise under any module, so the instance is rejected. Both predate
# the move.
LEGACY_ARTIFACTS = [
    ("tensorrt_llm.sampling_params", "GuidedDecodingParams"),
    ("tensorrt_llm.disaggregated_params", "DisaggregatedParams"),
]


@contextlib.contextmanager
def _class_pickled_as(cls: type, module: str) -> Iterator[None]:
    """Make ``pickle`` write ``module`` as the class's module, then restore.

    ``pickle`` takes the module string from ``cls.__module__``, then checks the
    name resolves back to this same object -- which it does, because the
    forwarding module re-exports it.  The bytes are what a build from before the
    move wrote.
    """
    original = cls.__module__
    cls.__module__ = module
    try:
        yield
    finally:
        cls.__module__ = original


@pytest.mark.parametrize("legacy_module,class_name", LEGACY_ARTIFACTS)
def test_pickle_written_before_the_move_still_loads(legacy_module: str, class_name: str) -> None:
    """An artifact naming the retired module deserializes.

    ``find_class`` matches the module string exactly, imports the forwarding
    module, and must hand back the class at its canonical home.
    """
    from tensorrt_llm.llmapi import serialization

    canonical_module = FORWARDS[legacy_module]
    cls = getattr(importlib.import_module(canonical_module), class_name)

    with _class_pickled_as(cls, legacy_module):
        artifact = serialization.dumps(cls())
    assert legacy_module.encode() in artifact, "the artifact does not name the retired module"

    restored = serialization.loads(artifact, approved_imports=serialization.BASE_EXAMPLE_CLASSES)

    # The same class object, reported at its real home: a forwarding module that
    # rewrote __module__ would round-trip here and still break the next release.
    assert type(restored) is cls
    assert type(restored).__module__ == canonical_module


@pytest.mark.parametrize("legacy_module,class_name", LEGACY_ARTIFACTS)
def test_the_legacy_allowlist_key_is_what_lets_it_load(legacy_module: str, class_name: str) -> None:
    """Dropping the legacy key alone must reject the artifact.

    Otherwise the key could be removed without any test noticing, which is the
    failure the key exists to prevent.
    """
    from tensorrt_llm.llmapi import serialization

    cls = getattr(importlib.import_module(FORWARDS[legacy_module]), class_name)

    with _class_pickled_as(cls, legacy_module):
        artifact = serialization.dumps(cls())

    without_legacy_key = {
        module: names
        for module, names in serialization.BASE_EXAMPLE_CLASSES.items()
        if module != legacy_module
    }
    with pytest.raises(ValueError, match=legacy_module):
        serialization.loads(artifact, approved_imports=without_legacy_key)
