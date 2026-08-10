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
"""Pytest plugin: pure-Python stubs of TensorRT-LLM's compiled modules.

Used by ``scripts/check_test_list.py`` so ``pytest --co`` can import product
code without a compiled ``bindings*.so`` or a prebuilt wheel. Covers the
build-generated Python modules listed in ``_STUB_ROOTS`` plus the
``libth_common.so`` surfaces torch exposes (see ``stub_torch_extensions``).

Factory-first. The factory fabricates modules, classes and values on demand,
and deliberately exposes *empty* introspection surfaces (``dir()`` has no
public non-callable names, ``__members__`` is empty) so that PybindMirror's
field/enum mirroring in ``llm_args`` passes without per-symbol entries.

Add ``_EXPLICIT`` entries only when Check Test List fails because a real
*value* is required at import time. Prefer removing
eager imports in tests/conftest over growing this table.

Collection-only — never a substitute for running tests.
"""

from __future__ import annotations

import sys
import types
from abc import ABCMeta
from importlib.machinery import ModuleSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Escape hatch
# ---------------------------------------------------------------------------
# Keyed by fully qualified name. The factory can fake any *shape*, but not a
# concrete value that product code consumes at import time. ``llm_args``
# evaluates these lookahead getters in a class body to seed pydantic field
# defaults, so they must return the real C++ constants
# (``kDefaultLookaheadDecoding*`` in cpp/include/tensorrt_llm/executor/executor.h).


class LookaheadDecodingConfig:
    """Explicit stub: real defaults consumed by ``llm_args`` at class-body time."""

    @staticmethod
    def get_default_lookahead_decoding_window():
        return 4

    @staticmethod
    def get_default_lookahead_decoding_ngram():
        return 3

    @staticmethod
    def get_default_lookahead_decoding_verification_set():
        return 4


_EXPLICIT: dict[str, object] = {
    "tensorrt_llm.bindings.executor.LookaheadDecodingConfig": LookaheadDecodingConfig,
}

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BINDINGS = "tensorrt_llm.bindings"

# Everything the C++ build generates under tensorrt_llm/ and that product code
# imports at module scope (see .gitignore / setup.py package_data).
_STUB_ROOTS = (
    _BINDINGS,
    "tensorrt_llm.deep_ep",
    "tensorrt_llm.deep_ep_cpp_tllm",
    "tensorrt_llm.deep_gemm",
    "tensorrt_llm.deep_gemm_cpp_tllm",
    "tensorrt_llm.flash_mla",
    "tensorrt_llm.flash_mla_cpp_tllm",
    "tensorrt_llm.pg_utils_bindings",
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
)

# Submodules that must resolve to modules even though they are not lower_case,
# so attribute access does not hit the CapWords "this is a class" rule.
_FORCED_SUBMODULES = (f"{_BINDINGS}.BuildInfo",)

# torch::class_ namespaces registered by libs/libth_common.so, which
# TRT_LLM_NO_LIB_INIT=1 skips loading. Probing an unregistered class raises
# RuntimeError (not AttributeError), so hasattr() checks in product code blow
# up; an empty namespace makes them report "unavailable", which is the truth
# for a no-compile checkout.
_TORCH_CLASS_NAMESPACES = ("trtllm",)


class _StubMeta(ABCMeta):
    """Metaclass for fabricated binding classes.

    Derives from ``ABCMeta`` so product classes can inherit from both a stubbed
    binding type and an ABC without a metaclass conflict.
    """

    def __getattr__(cls, name: str):
        # Never fabricate dunders: Python and pydantic probe them to decide
        # which protocols a type supports.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        children = cls.__dict__.get("_stub_children")
        if children is None:
            children = {}
            type.__setattr__(cls, "_stub_children", children)
        if name not in children:
            # Cached and distinct per name so enum-style members stay usable as
            # dict keys (e.g. the DataType maps built in tensorrt_llm/_utils.py).
            children[name] = _make_stub_class(f"{cls.__name__}.{name}", cls.__module__)
        return children[name]

    @property
    def __members__(cls):
        # PybindMirror.mirror_pybind_enum iterates the C++ members and requires
        # each to exist on the Python enum; an empty mapping trivially passes.
        return {}

    def __iter__(cls):
        # Product code materializes some binding sequences at import time
        # (e.g. tuple(KVCacheIterationStatsDelta._field_names)).
        return iter(())

    def __int__(cls):
        # Stubbed enum members are coerced at import time
        # (e.g. int(BufferKind.DEFAULT) in cute_dsl_custom_ops.py).
        return 0

    def __index__(cls):
        return 0


class _StubBase(metaclass=_StubMeta):
    """Base for fabricated binding classes; only dunders, so ``dir()`` is clean."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_stub_class(f"{type(self).__name__}.{name}", type(self).__module__)

    def __call__(self, *args, **kwargs):
        return False

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __int__(self):
        return 0

    def __index__(self):
        return 0


def _make_stub_class(name: str, module: str) -> type:
    return _StubMeta(name, (_StubBase,), {"__module__": module})


class StubModule(types.ModuleType):
    """Fake (sub)module of a stubbed root, resolving attributes on demand."""

    def __init__(self, fullname: str):
        super().__init__(fullname)
        self.__file__ = f"<bindings stub: {fullname}>"
        self.__package__ = fullname
        self.__path__ = []  # marks this as a package for nested imports
        object.__setattr__(self, "_stub_cache", {})

    def __getattr__(self, name: str):
        cache = object.__getattribute__(self, "_stub_cache")
        if name in cache:
            return cache[name]

        fq = f"{self.__name__}.{name}"
        if fq in _EXPLICIT:
            value = _EXPLICIT[fq]
        elif name.isupper():
            # Module-level constants such as BuildInfo.ENABLE_MULTI_DEVICE:
            # falsy keeps mpi_session and communicator off their mpi4py paths.
            value = 0
        elif name[:1].isupper():
            value = _make_stub_class(name, self.__name__)
        else:
            # Lower case: either a submodule or a free function. A module is
            # both importable and callable, so it covers each case.
            value = StubModule(fq)
            sys.modules[fq] = value

        cache[name] = value
        return value

    def __call__(self, *args, **kwargs):
        return False

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<StubModule {self.__name__}>"


class _StubFinder:
    """Meta path finder fabricating any module under a stubbed root.

    Attribute access alone is not enough: ``import a.b.c`` and
    ``from a.b import c`` go through the import system, which never consults a
    parent module's ``__getattr__``.
    """

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if not any(fullname == root or fullname.startswith(f"{root}.") for root in _STUB_ROOTS):
            return None
        return ModuleSpec(fullname, _StubLoader(), is_package=True)


class _StubLoader:
    @staticmethod
    def create_module(spec):
        return StubModule(spec.name)

    @staticmethod
    def exec_module(module):
        pass


def _real_bindings_present() -> bool:
    """True if a compiled bindings extension is loaded or present on disk.

    Avoids ``importlib.util.find_spec``, which would execute
    ``tensorrt_llm/__init__.py`` before the stub is installed.
    """
    existing = sys.modules.get(_BINDINGS)
    if existing is not None and not isinstance(existing, StubModule):
        return True

    for entry in sys.path:
        pkg = Path(entry) / "tensorrt_llm"
        if not pkg.is_dir():
            continue
        for pattern in ("bindings*.so", "bindings*.pyd", "_bindings*.so"):
            if list(pkg.glob(pattern)):
                return True
    return False


def install_bindings_stub() -> StubModule | None:
    """Install stub modules for every ``_STUB_ROOTS`` entry into ``sys.modules``.

    Returns the ``tensorrt_llm.bindings`` stub, or None when real bindings are
    present (the stub refuses to mask a real install).
    """
    installed = sys.modules.get(_BINDINGS)
    if isinstance(installed, StubModule):
        # Idempotent: reinstalling would orphan the stub classes already
        # captured by imported product modules.
        return installed

    if _real_bindings_present():
        return None

    # Appended, so a real installation's finders always take precedence.
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder())

    for root in _STUB_ROOTS:
        sys.modules.setdefault(root, StubModule(root))

    for sub in _FORCED_SUBMODULES:
        if sub in sys.modules:
            continue
        child = StubModule(sub)
        sys.modules[sub] = child
        parent_name, _, attr = sub.rpartition(".")
        parent = sys.modules.get(parent_name)
        if isinstance(parent, StubModule):
            object.__getattribute__(parent, "_stub_cache")[attr] = child

    return sys.modules[_BINDINGS]


def stub_torch_extensions() -> None:
    """Neutralize the parts of torch that expect ``libth_common.so`` to be loaded."""
    if _real_bindings_present():
        return

    try:
        import torch
    except ImportError:
        return

    for namespace in _TORCH_CLASS_NAMESPACES:
        if namespace not in torch.classes.__dict__:
            setattr(torch.classes, namespace, types.ModuleType(f"torch.classes.{namespace}"))

    # Product modules register fake kernels for C++ ops at import time; without
    # the library the schemas are missing, so make those registrations no-ops.
    register_fake = torch.library.register_fake
    if getattr(register_fake, "_trtllm_collection_stub", False):
        return

    def tolerant_register_fake(op, func=None, /, **kwargs):
        def apply(fn):
            try:
                return register_fake(op, fn, **kwargs)
            except RuntimeError as exc:
                if "does not exist" not in str(exc):
                    raise
                return fn

        return apply(func) if func is not None else apply

    tolerant_register_fake._trtllm_collection_stub = True
    torch.library.register_fake = tolerant_register_fake


# Install on import so `pytest -p stubify_bindings` wins the race with
# conftest collection.
install_bindings_stub()


def pytest_configure(config):
    """Re-assert the stubs early in the pytest session."""
    install_bindings_stub()
    stub_torch_extensions()
