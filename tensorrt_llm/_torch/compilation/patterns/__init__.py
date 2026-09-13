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

from collections.abc import Callable

from torch._inductor.pattern_matcher import PatternMatcherPass

MATCHER_SUBSYSTEM = "torch_compile"

_CustomPassRegistrar = Callable[[list[PatternMatcherPass]], None]
_CUSTOM_PASS_REGISTRARS: dict[str, _CustomPassRegistrar] = {}


def register_custom_pass_registrar(name: str,
                                   registrar: _CustomPassRegistrar) -> None:
    """Register an optional group of torch.compile pattern passes."""
    existing = _CUSTOM_PASS_REGISTRARS.get(name)
    if existing is not None and existing is not registrar:
        raise ValueError(f"custom pass registrar {name!r} is already registered")
    _CUSTOM_PASS_REGISTRARS[name] = registrar


def append_registered_custom_passes(
        custom_passes: list[PatternMatcherPass]) -> None:
    """Append passes supplied by registered optional integrations."""
    for registrar in _CUSTOM_PASS_REGISTRARS.values():
        registrar(custom_passes)


__all__ = [
    "MATCHER_SUBSYSTEM",
    "append_registered_custom_passes",
    "register_custom_pass_registrar",
]
