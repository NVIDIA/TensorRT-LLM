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
"""Shared PEP 562 hooks for lazily imported packages."""

import importlib
from collections.abc import Callable, Mapping


def lazy_imports(
    package_name: str,
    namespace: dict,
    name_to_module: Mapping[str, str],
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Create lazy attribute and directory hooks for a package.

    Args:
        package_name: Fully qualified package name for relative imports.
        namespace: Package globals, used to cache resolved exports.
        name_to_module: Export names mapped to relative provider modules.

    Returns:
        The package's ``__getattr__`` and ``__dir__`` hooks.
    """

    def __getattr__(name: str) -> object:
        module_name = name_to_module.get(name)
        if module_name is not None:
            try:
                module = importlib.import_module(f".{module_name}", package_name)
            except AttributeError as e:
                # An AttributeError escaping the module is a bug inside it, not a
                # missing name here. Raised as is, hasattr() and getattr() with a
                # default would hide it, and ``from ... import`` would run the
                # module twice (losing the cause unless the name is a submodule).
                raise ImportError(
                    f"cannot import {name!r} from {package_name!r}: importing "
                    f"{package_name}.{module_name} raised {e!r}",
                    name=package_name,
                ) from e
            attr = getattr(module, name)
            namespace[name] = attr  # cache: subsequent access skips __getattr__
            return attr
        # Resolve bare submodules as well as named exports.
        try:
            return importlib.import_module(f".{name}", package_name)
        except ModuleNotFoundError as e:
            # Only translate "no such submodule" into AttributeError; a
            # ModuleNotFoundError raised *inside* an existing submodule
            # (missing dependency) must propagate unchanged.
            if e.name != f"{package_name}.{name}":
                raise
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}") from None
        except AttributeError as e:
            # Same as above: a bug inside the submodule, not a missing name.
            raise ImportError(
                f"cannot import {name!r} from {package_name!r}: importing "
                f"{package_name}.{name} raised {e!r}",
                name=package_name,
            ) from e

    def __dir__() -> list[str]:
        return sorted(set(namespace.get("__all__", ())) | set(namespace))

    return __getattr__, __dir__
