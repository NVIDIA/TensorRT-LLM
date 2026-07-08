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

"""Compatibility redirect from TensorRT-LLM's bundled AutoDeploy namespace to LLMC."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
import threading
from collections.abc import Sequence
from types import ModuleType

REDIRECT_ENV_VAR = "TRTLLM_REDIRECT_AD_TO_LLMC"

__all__ = [
    "REDIRECT_ENV_VAR",
    "install_autodeploy_redirect",
    "install_autodeploy_redirect_from_env",
]

_LEGACY_PACKAGE = "tensorrt_llm._torch.auto_deploy"
_TARGET_PACKAGE = "llmc"
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FINDER_MARKER = "_is_llmc_autodeploy_redirect"
_INSTALL_LOCK = threading.Lock()


class _RedirectLoader(importlib.abc.Loader):
    """Load an alias by returning its canonical LLMC module object."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name
        self._canonical_attributes: dict[str, object] | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        del spec
        module = importlib.import_module(self._target_name)
        self._canonical_attributes = {
            name: getattr(module, name)
            for name in (
                "__name__",
                "__loader__",
                "__package__",
                "__spec__",
                "__file__",
                "__cached__",
                "__path__",
            )
            if hasattr(module, name)
        }
        return module

    def exec_module(self, module: ModuleType) -> None:
        if self._canonical_attributes is None:
            raise RuntimeError(f"Redirect loader for {self._target_name!r} was not initialized")

        # Import machinery temporarily applies the alias spec to the canonical
        # module returned by create_module(). Restore its LLMC identity so
        # introspection and pickling continue to use canonical module names.
        module.__dict__.update(self._canonical_attributes)


class _AutoDeployRedirectFinder(importlib.abc.MetaPathFinder):
    """Map the legacy AutoDeploy package prefix to the canonical LLMC prefix."""

    _is_llmc_autodeploy_redirect = True

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname != _LEGACY_PACKAGE and not fullname.startswith(_LEGACY_PACKAGE + "."):
            return None

        target_name = _TARGET_PACKAGE + fullname[len(_LEGACY_PACKAGE) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            raise ModuleNotFoundError(
                f"Cannot redirect {fullname!r}: LLMC module {target_name!r} does not exist",
                name=target_name,
            )

        return importlib.util.spec_from_loader(
            fullname,
            _RedirectLoader(target_name),
            origin=f"redirect:{target_name}",
            is_package=target_spec.submodule_search_locations is not None,
        )


def _redirect_is_enabled() -> bool:
    value = os.environ.get(REDIRECT_ENV_VAR)
    if value is None:
        return False

    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    if normalized in _TRUE_VALUES:
        return True
    raise ValueError(f"{REDIRECT_ENV_VAR} must be a boolean value, got {value!r}")


def install_autodeploy_redirect() -> None:
    """Redirect legacy TensorRT-LLM AutoDeploy imports to canonical LLMC modules.

    The redirect must be installed before any module in the legacy namespace is
    imported. Calling this function more than once is safe.

    Raises:
        RuntimeError: If bundled AutoDeploy modules were imported before the redirect.
    """
    with _INSTALL_LOCK:
        for finder in sys.meta_path:
            if getattr(finder, _FINDER_MARKER, False):
                return

        loaded_legacy_modules = sorted(
            name
            for name in sys.modules
            if name == _LEGACY_PACKAGE or name.startswith(_LEGACY_PACKAGE + ".")
        )
        if loaded_legacy_modules:
            raise RuntimeError(
                "Cannot redirect TensorRT-LLM AutoDeploy after bundled modules were loaded: "
                + ", ".join(loaded_legacy_modules)
            )

        sys.meta_path.insert(0, _AutoDeployRedirectFinder())


def install_autodeploy_redirect_from_env() -> None:
    """Install the AutoDeploy redirect when ``TRTLLM_REDIRECT_AD_TO_LLMC`` is enabled."""
    if _redirect_is_enabled():
        install_autodeploy_redirect()
