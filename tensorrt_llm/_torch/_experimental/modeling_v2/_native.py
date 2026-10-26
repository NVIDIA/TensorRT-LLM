# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target's native (C++/CUDA) code: where it is built, and how the target loads it.

Native code of a model family lives in ``cpp/tensorrt_llm/modeling_v2/<familyCamel>/``
and builds into one torch-op library, ``libth_modeling_v2_<family>.so``, installed
next to ``libth_common.so``. Nothing of it is in the stock libraries, and nothing
loads it until routing picks a target of the family: a target package that has
native code starts with::

    from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

    native_ops = load_target_ops(__name__)  # torch.ops.mv2_<family>_<target>

``route`` imports the target only on the branch that returns it, so loading is part
of routing, and it follows how the stock code treats its optional native modules
(``flash_mla``, ``deep_ep``):

* a library this installation does not have is a feature that is not there:
  :class:`NoTarget`, so ``auto`` keeps the built-in model and ``require`` says why;
* a library that is there but fails to load is an error, :class:`NativeLoadError`,
  never a fallback;
* which targets a library carries it answers itself,
  ``torch.ops.mv2_<family>.built_targets()``, the way ``th_common`` answers feature
  queries; a target it does not carry is :class:`NoTarget`;
* the target's ``fake_ops.py`` registers right after the library opens, which fails
  if an op it declares is missing from the library.

``cpp/tensorrt_llm/modeling_v2/README.md`` is the guide for authors.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

from ._router_index import NoTarget

_PACKAGE = __name__.rpartition(".")[0]
_MODELS_PACKAGE = f"{_PACKAGE}.models"

# tensorrt_llm/libs, where build_wheel.py installs libth_common.so and the family libraries.
_LIBS_DIR = Path(__file__).resolve().parents[3] / "libs"

_built_targets: Dict[str, FrozenSet[str]] = {}


class NativeLoadError(RuntimeError):
    """A family library is installed but cannot be used.

    Raised when the library fails to open or an op the target's ``fake_ops.py``
    declares is missing from it. It is deliberately not a :class:`NoTarget`, so
    ``auto`` does not hide a broken installation behind the built-in model.
    """


# --- Naming (mirrors cpp/cmake/modules/modeling_v2.cmake) --------------------


def camel(name: str) -> str:
    """The C++ directory of a family or target: ``scale_add__sm_10x__default`` ->
    ``scaleAddSm10xDefault``."""
    parts = [p for p in name.split("_") if p]
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def identifier(name: str) -> str:
    """``name`` with ``__`` collapsed, usable as a C++ or torch identifier."""
    while "__" in name:
        name = name.replace("__", "_")
    return name


def torch_namespace(family: str, target: Optional[str] = None) -> str:
    """The torch namespace of a target's ops, or of the family's shared ops."""
    if target is None:
        return f"mv2_{family}"
    return f"mv2_{family}_{identifier(target)}"


def library_name(family: str) -> str:
    """The file name of a family's library."""
    return f"libth_modeling_v2_{family}.so"


def split_target_package(package: str) -> Tuple[str, str]:
    """``(family, target)`` of a target package name."""
    prefix = f"{_MODELS_PACKAGE}."
    if not package.startswith(prefix):
        raise ValueError(f"{package!r} is not a modeling_v2 target package")
    parts = package[len(prefix) :].split(".")
    if len(parts) != 2:
        raise ValueError(f"{package!r} is not a models.<family>.<target> package")
    return parts[0], parts[1]


def namespace_of(package: str) -> str:
    """The torch namespace of the target package ``package``, e.g. for its fake_ops.py."""
    return torch_namespace(*split_target_package(package))


# --- Loading ------------------------------------------------------------------


def _open_library(path: Path, family: str) -> FrozenSet[str]:
    """dlopen a family library and return the targets it carries."""
    import torch

    torch.ops.load_library(str(path))
    answer = getattr(torch.ops, torch_namespace(family)).built_targets()
    return frozenset(t for t in answer.split(",") if t)


def load_family_library(family: str) -> FrozenSet[str]:
    """Open ``libth_modeling_v2_<family>.so`` and return the targets it carries.

    Idempotent, and independent of the device, so it can run anywhere the stock
    libraries load. :func:`load_target_ops` adds the target-level checks.
    """
    if family in _built_targets:
        return _built_targets[family]
    if sys.platform == "win32":
        raise NoTarget("modeling_v2 native code is not built on Windows")
    path = _LIBS_DIR / library_name(family)
    if not path.is_file():
        raise NoTarget(
            f"this installation has no {path.name}: the build compiles a family's library "
            "only when its CUDA architectures include one of the family's targets, and "
            "only with BUILD_MODELING_V2=ON"
        )
    try:
        built = _open_library(path, family)
    except (AttributeError, OSError, RuntimeError) as error:
        raise NativeLoadError(f"{path} is installed but cannot be loaded: {error}") from None
    _built_targets[family] = built
    return built


def load_target_ops(package: str) -> Any:
    """Load the native code of the target package ``package`` and return its ops
    namespace, ``torch.ops.mv2_<family>_<target>``.

    Call it from the target's ``__init__.py`` with ``__name__``. Raises
    :class:`NoTarget` when this installation does not carry the target and
    :class:`NativeLoadError` when its library is broken.
    """
    family, target = split_target_package(package)
    built = load_family_library(family)
    if target not in built:
        carried = ", ".join(sorted(built)) or "none"
        raise NoTarget(
            f"{library_name(family)} does not carry {target}: this build did not compile "
            f"it (its ARCHS miss the build's, a REQUIRES is unmet, or "
            f"TRTLLM_MODELING_V2_TARGETS excluded it); it carries {carried}"
        )

    fake_ops = f"{package}.fake_ops"
    if importlib.util.find_spec(fake_ops) is not None:
        try:
            importlib.import_module(fake_ops)
        except (AttributeError, RuntimeError, ValueError) as error:
            raise NativeLoadError(
                f"{fake_ops} does not match the ops in {library_name(family)}: {error}"
            ) from None
    import torch

    return getattr(torch.ops, torch_namespace(family, target))
