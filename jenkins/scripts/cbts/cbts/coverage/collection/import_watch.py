# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic meta-path machinery for running a callback once a named module finishes executing.

Used by the guest bootstrap to activate coverage once the product framework's import
completes, but not specific to that use.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import threading
from types import ModuleType
from typing import Callable, Optional, Sequence

_resolving = threading.local()


class ImportCompletionWatcher(importlib.abc.MetaPathFinder):
    """Meta-path finder that runs a hook once a watched module finishes executing.

    Hands resolution back to ``importlib.util.find_spec`` and swaps the resolved
    loader for one that calls back after ``exec_module`` returns, so a hook keys off
    the actual end of the module body rather than sampling import state. Running on
    the importing thread leaves no window between a module becoming usable and its
    hook having run.

    ``callbacks`` maps module name to hook. Entries are dropped as they fire, so each
    hook runs once and every later import costs one dict miss; the finder stays in
    ``sys.meta_path`` rather than removing itself, which would mutate that list
    mid-import while another thread may be walking it.
    """

    def __init__(self, callbacks: dict[str, Callable[[], None]]) -> None:
        self._callbacks = callbacks

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        # The delegated lookup walks sys.meta_path again and arrives back here; the
        # thread-local flag makes that second pass decline instead of recursing.
        if fullname not in self._callbacks or getattr(_resolving, "active", False):
            return None
        _resolving.active = True
        try:
            spec = importlib.util.find_spec(fullname)
        except (ImportError, ValueError):
            return None
        finally:
            _resolving.active = False
        if spec is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _WatchedLoader(spec.loader, fullname, self.fire)
        return spec

    def fire(self, fullname: str) -> None:
        callback = self._callbacks.pop(fullname, None)
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            print(
                f"[cbts] import hook for {fullname} failed in pid {os.getpid()}: {exc!r}",
                file=sys.stderr,
            )


class _WatchedLoader(importlib.abc.Loader):
    """Loader proxy that calls back once the wrapped ``exec_module`` returns."""

    def __init__(
        self, inner: importlib.abc.Loader, fullname: str, fire: Callable[[str], None]
    ) -> None:
        self._inner = inner
        self._fullname = fullname
        self._fire = fire

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Optional[ModuleType]:
        return self._inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        try:
            self._inner.exec_module(module)
        finally:
            # fire() swallows hook failures; an import must not break over coverage.
            self._fire(self._fullname)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
