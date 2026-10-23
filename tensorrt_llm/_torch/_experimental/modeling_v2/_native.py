# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target's native (C++/CUDA) code: where it is built, and how the target loads it.

Native code of a model family lives in ``cpp/tensorrt_llm/modeling_v2/<familyCamel>/``
and builds into one library, ``libtrtllm_mv2_<family>.so``, installed next to
``libth_common.so`` together with ``modeling_v2_manifest.json``. Nothing of it is in
the stock libraries, and nothing loads it until routing picks a target of the
family: a target package that has native code starts with::

    from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

    ops = load_target_ops(__name__)  # torch.ops.mv2_<family>_<target>

``route`` imports the target only on the branch that returns it, so loading is part
of routing: a target whose code this wheel did not build raises :class:`NoTarget`
(``auto`` keeps the built-in model, ``require`` reports why), while a library that
is present but unusable raises :class:`NativeLoadError`, which is never a fallback.

The checks, in order:

1. the manifest lists the target as built (otherwise ``NoTarget``);
2. the device's compute capability is covered by an architecture the target was
   built for (otherwise ``NoTarget``);
3. the family library opens from the fixed package path;
4. its contract fingerprint equals the loaded ``libth_common.so``'s -- the two were
   compiled against the same stock headers, definitions, torch and CUDA;
5. the target's ``fake_ops.py`` registers, which fails if an op it declares is
   missing from the library.

``cpp/tensorrt_llm/modeling_v2/README.md`` is the guide for authors.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._router_index import NoTarget

MANIFEST_NAME = "modeling_v2_manifest.json"
MANIFEST_SCHEMA = 1

_PACKAGE = __name__.rpartition(".")[0]
_MODELS_PACKAGE = f"{_PACKAGE}.models"

# tensorrt_llm/libs, where build_wheel.py installs the family libraries.
_LIBS_DIR = Path(__file__).resolve().parents[3] / "libs"

_loaded_libraries: set = set()


class NativeLoadError(RuntimeError):
    """A family library the manifest lists as built cannot be used.

    Raised for a broken or mismatched installation: a missing library, one that
    fails to open, a contract-fingerprint mismatch, an op ``fake_ops.py`` declares
    but the library lacks. It is deliberately not a :class:`NoTarget`, so ``auto``
    does not hide it behind the built-in model.
    """


# --- Naming (mirrors cpp/cmake/modules/modelingV2.cmake) ---------------------


def camel(name: str) -> str:
    """The C++ directory of a family or target: ``scale_add__sm_80__default`` ->
    ``scaleAddSm80Default``."""
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


# --- Architectures ---------------------------------------------------------------


def arch_covers(arch: str, capability: Tuple[int, int]) -> bool:
    """Whether code built for ``arch`` (a manifest entry such as ``100f-real``) runs on
    a device of compute capability ``capability``.

    ``NNa`` runs on exactly NN; ``NN`` on NN and later minor versions of the same
    major version; ``NNf`` on every member of family NN, which this build treats the
    same way.
    """
    name = arch[: -len("-real")] if arch.endswith("-real") else arch
    suffix = name[-1] if name[-1] in "af" else ""
    number = int(name[:-1] if suffix else name)
    major, minor = divmod(number, 10)
    if suffix == "a":
        return capability == (major, minor)
    return capability[0] == major and capability[1] >= minor


# --- Manifest -----------------------------------------------------------------


@lru_cache(maxsize=None)
def read_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """The manifest the build wrote next to the family libraries."""
    path = path or (_LIBS_DIR / MANIFEST_NAME)
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        raise NativeLoadError(
            f"{path} is missing: this installation of TensorRT-LLM has no modeling_v2 "
            "manifest. Rebuild with scripts/build_wheel.py."
        ) from None
    except (OSError, ValueError) as error:
        raise NativeLoadError(f"{path} cannot be read: {error}") from None
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise NativeLoadError(
            f"{path} has schema {manifest.get('schema')!r}; this loader reads schema "
            f"{MANIFEST_SCHEMA}. The Python package and the native libraries are from "
            "different builds."
        )
    return manifest


def available_targets(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every target the manifest knows, with its status, architectures and reason --
    what this installation supports."""
    manifest = read_manifest() if manifest is None else manifest
    rows = []
    for family, entry in sorted(manifest.get("families", {}).items()):
        for target, unit in sorted(entry.get("targets", {}).items()):
            rows.append(
                {
                    "family": family,
                    "target": target,
                    "status": unit["status"],
                    "archs": list(unit["archs"]),
                    "reason": unit["reason"],
                    "library": entry.get("library"),
                }
            )
    return rows


def _family_entry(manifest: Dict[str, Any], family: str) -> Dict[str, Any]:
    if not manifest.get("enabled", False):
        raise NoTarget(
            f"this TensorRT-LLM build has no modeling_v2 libraries: {manifest.get('reason')}"
        )
    entry = manifest.get("families", {}).get(family)
    if entry is None:
        raise NativeLoadError(
            f"modeling_v2 family {family!r} has native code but is not in the manifest; "
            "the Python package and the native libraries are from different builds"
        )
    return entry


def _target_entry(manifest: Dict[str, Any], family: str, target: str) -> Dict[str, Any]:
    entry = _family_entry(manifest, family)
    unit = entry.get("targets", {}).get(target)
    if unit is None:
        raise NativeLoadError(
            f"modeling_v2 target {family}/{target} has native code but is not in the "
            "manifest; the Python package and the native libraries are from different builds"
        )
    if unit["status"] != "built":
        raise NoTarget(
            f"{family}/{target}: native code not built in this TensorRT-LLM build "
            f"({unit['status']}: {unit['reason']})"
        )
    return unit


# --- Loading ------------------------------------------------------------------


def _ops_namespace(namespace: str) -> Any:
    import torch

    return getattr(torch.ops, namespace)


def load_family_library(family: str, manifest: Optional[Dict[str, Any]] = None) -> None:
    """Open ``libtrtllm_mv2_<family>.so`` and check its contract fingerprint.

    Idempotent. Does not look at a device, so it can run anywhere the stock
    libraries load; :func:`load_target_ops` adds the target-level checks.
    """
    manifest = read_manifest() if manifest is None else manifest
    entry = _family_entry(manifest, family)
    library = entry.get("library")
    if library is None:
        raise NoTarget(f"modeling_v2 family {family!r}: no target of it is built in this build")
    path = _LIBS_DIR / library
    if path in _loaded_libraries:
        return
    if not path.is_file():
        raise NativeLoadError(f"{path} is listed in the manifest but missing")

    import torch

    try:
        torch.ops.load_library(str(path))
    except OSError as error:
        raise NativeLoadError(f"{path} cannot be loaded: {error}") from None

    try:
        stock = torch.ops.trtllm.mv2_contract_fingerprint()
    except (AttributeError, RuntimeError) as error:
        raise NativeLoadError(
            f"libth_common.so does not report a modeling_v2 contract fingerprint ({error}); "
            f"it predates {path.name}"
        ) from None
    fingerprint = _ops_namespace(torch_namespace(family)).contract_fingerprint()
    if fingerprint != stock:
        raise NativeLoadError(
            f"{path.name} was built against a different stock contract than the loaded "
            f"libth_common.so (fingerprint {fingerprint} vs {stock}); install libraries "
            "from one build"
        )
    _loaded_libraries.add(path)


def _device_capability() -> Optional[Tuple[int, int]]:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def load_target_ops(package: str, manifest: Optional[Dict[str, Any]] = None) -> Any:
    """Load the native code of the target package ``package`` and return its ops
    namespace, ``torch.ops.mv2_<family>_<target>``.

    Call it from the target's ``__init__.py`` with ``__name__``. Raises
    :class:`NoTarget` when this build or this device cannot run the target and
    :class:`NativeLoadError` when the installation is broken.
    """
    if sys.platform == "win32":
        raise NoTarget("modeling_v2 native code is not built on Windows")
    family, target = split_target_package(package)
    manifest = read_manifest() if manifest is None else manifest
    unit = _target_entry(manifest, family, target)

    capability = _device_capability()
    if capability is not None and not any(arch_covers(a, capability) for a in unit["archs"]):
        raise NoTarget(
            f"{family}/{target}: built for {', '.join(unit['archs'])}, which does not run "
            f"on this sm_{capability[0]}{capability[1]} device"
        )

    load_family_library(family, manifest)

    fake_ops = f"{package}.fake_ops"
    if importlib.util.find_spec(fake_ops) is not None:
        try:
            importlib.import_module(fake_ops)
        except (AttributeError, RuntimeError, ValueError) as error:
            raise NativeLoadError(
                f"{fake_ops} does not match the ops in the family library: {error}"
            ) from None
    return _ops_namespace(unit["namespace"])


def built_families(manifest: Optional[Dict[str, Any]] = None) -> Iterable[str]:
    """Families whose library this build produced."""
    manifest = read_manifest() if manifest is None else manifest
    if not manifest.get("enabled", False):
        return []
    return sorted(
        family
        for family, entry in manifest.get("families", {}).items()
        if entry.get("library") is not None
    )
