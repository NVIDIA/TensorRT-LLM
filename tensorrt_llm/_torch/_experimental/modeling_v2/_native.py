# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target's native (C++/CUDA) code: where it is built, and how the target loads it.

Native code of a model family lives with its Python code, in ``models/<family>/csrc/``
(the family's ``CMakeLists.txt`` and shared code) and ``models/<family>/<target>/csrc/``
(a target's own), and builds into one torch-op library,
``libth_modeling_v2_<family>.so``, installed next to ``libth_common.so``. Nothing of
it is in the stock libraries, and nothing loads it until routing picks a target of
the family. The API is four names:

* :func:`load_target_ops` -- a target package with native code starts with
  ``native_ops = load_target_ops(__name__)``;
* :func:`load_family_ops` -- the family's shared ops, ``load_family_ops(__name__)``
  from any module of the family;
* :func:`namespace_of` -- the torch namespace a ``fake_ops.py`` registers its fakes
  in, ``namespace_of(__name__)``;
* :class:`NativeLoadError` -- a library that is installed but broken.

``route`` imports a target only on the branch that returns it, so loading is part of
routing, and it follows how the stock code treats its optional native modules
(``flash_mla``, ``deep_ep``): a library this installation does not have, or a target
it did not build (it answers ``torch.ops.mv2_<family>.built_targets()``), is a
:class:`NoTarget`, so ``auto`` keeps the built-in model; a library that fails to
load, or lacks an op a ``fake_ops.py`` declares, is a :class:`NativeLoadError`,
never a fallback. The family's ``fake_ops.py`` registers when the library opens, a
target's when the target is found in it.

The "Native code" section of the modeling_v2 README is the guide for authors.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

from ._router_index import NoTarget

_PACKAGE = __name__.rpartition(".")[0]
_MODELS_PACKAGE = f"{_PACKAGE}.models"

# tensorrt_llm/libs, where build_wheel.py installs libth_common.so and the family libraries.
_LIBS_DIR = Path(__file__).resolve().parents[3] / "libs"

_built_targets: Dict[str, FrozenSet[str]] = {}
_family_ops: Dict[str, Any] = {}


class NativeLoadError(RuntimeError):
    """A family library is installed but cannot be used.

    Raised when the library fails to open or an op a ``fake_ops.py`` declares is
    missing from it. It is deliberately not a :class:`NoTarget`, so ``auto`` does
    not hide a broken installation behind the built-in model.
    """


def namespace_of(module: str) -> str:
    """The torch namespace of the ops a module's ``fake_ops.py`` fakes.

    ``mv2_<family>_<target>`` for a module of a target package, ``mv2_<family>`` for
    a module of the family itself (its shared ops). Target directories are named
    ``<checkpoint>__<sm>__<parallel>``, so the module path tells the two apart.
    """
    return _namespace(*_split(module))


# --- Naming (mirrors cpp/cmake/modules/modeling_v2.cmake) --------------------


def _identifier(name: str) -> str:
    """``name`` with ``__`` collapsed, usable as a C++ or torch identifier."""
    return re.sub("_{2,}", "_", name)


def _namespace(family: str, target: Optional[str] = None) -> str:
    return f"mv2_{family}" if target is None else f"mv2_{family}_{_identifier(target)}"


def _library_name(family: str) -> str:
    return f"libth_modeling_v2_{family}.so"


def _split(module: str) -> Tuple[str, Optional[str]]:
    """``(family, target)`` of a module under ``models.<family>``; ``target`` is None
    for a module of the family itself."""
    prefix = f"{_MODELS_PACKAGE}."
    parts = module[len(prefix) :].split(".") if module.startswith(prefix) else []
    if not parts or not parts[0]:
        raise ValueError(f"{module!r} is not a module of a modeling_v2 family")
    target = parts[1] if len(parts) > 1 and "__" in parts[1] else None
    return parts[0], target


# --- Loading ------------------------------------------------------------------


def _import_fakes(module: str, family: str) -> None:
    """Import a ``fake_ops`` module, if there is one; it registers fakes of ops the
    family library must define."""
    if importlib.util.find_spec(module) is None:
        return
    try:
        importlib.import_module(module)
    except (AttributeError, RuntimeError, ValueError) as error:
        raise NativeLoadError(
            f"{module} does not match the ops in {_library_name(family)}: {error}"
        ) from None


def _open_library(path: Path, family: str) -> FrozenSet[str]:
    """dlopen a family library and return the targets it carries."""
    import torch

    torch.ops.load_library(str(path))
    answer = getattr(torch.ops, _namespace(family)).built_targets()
    return frozenset(t for t in answer.split(",") if t)


def _load_family_library(family: str) -> FrozenSet[str]:
    """Open ``libth_modeling_v2_<family>.so`` once and return the targets it carries."""
    if family in _built_targets:
        return _built_targets[family]
    path = _LIBS_DIR / _library_name(family)
    if not path.is_file():
        raise NoTarget(
            f"this installation has no {path.name}: the build compiles a family's library "
            "only when its CUDA architectures include one of the family's targets"
        )
    try:
        built = _open_library(path, family)
    except (AttributeError, OSError, RuntimeError) as error:
        raise NativeLoadError(f"{path} is installed but cannot be loaded: {error}") from None
    _import_fakes(f"{_MODELS_PACKAGE}.{family}.fake_ops", family)
    _built_targets[family] = built
    return built


def load_target_ops(package: str) -> Any:
    """Load the native code of the target package ``package`` and return its ops
    namespace, ``torch.ops.mv2_<family>_<target>``.

    Call it from the target's ``__init__.py`` with ``__name__``. Raises
    :class:`NoTarget` when this installation does not carry the target
    and :class:`NativeLoadError` when its library is broken.
    """
    family, target = _split(package)
    if target is None or package != f"{_MODELS_PACKAGE}.{family}.{target}":
        raise ValueError(f"{package!r} is not a models.<family>.<target> package")
    built = _load_family_library(family)
    if target not in built:
        carried = ", ".join(sorted(built)) or "none"
        raise NoTarget(
            f"{_library_name(family)} does not carry {target}: this build did not compile "
            f"it (its ARCHS miss the build's, a REQUIRES is unmet, or "
            f"TRTLLM_MODELING_V2_TARGETS excluded it); it carries {carried}"
        )
    _import_fakes(f"{package}.fake_ops", family)
    import torch

    return getattr(torch.ops, _namespace(family, target))


def load_family_ops(module: str) -> Any:
    """Load the native code of the family ``module`` belongs to and return the ops of
    its shared code, ``torch.ops.mv2_<family>``.

    Call it with ``__name__`` from any module of the family, where the op is used;
    after the first call it is a dictionary lookup. Raises like
    :func:`load_target_ops`, whose target check the target's ``__init__.py`` runs
    first.
    """
    ops = _family_ops.get(module)
    if ops is None:
        family = _split(module)[0]
        _load_family_library(family)
        import torch

        ops = _family_ops[module] = getattr(torch.ops, _namespace(family))
    return ops
