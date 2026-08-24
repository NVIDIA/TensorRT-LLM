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
"""Resolution of optional FlashAttention 4 CuTe symbols.

Shared by the backends that consume ``flash_attn.cute.interface`` so the
optional-dependency contract is defined once instead of per import site.
"""

import importlib
from typing import Any, Callable, Optional


def resolve_flash_attn_cute_symbol(
    name: str,
) -> tuple[Optional[Callable[..., Any]], Optional[BaseException]]:
    """Return ``(symbol, None)`` if FlashAttention 4 provides ``name``, else ``(None, error)``.

    Callers keep the returned error and raise it at first use, so an unusable FA4
    degrades to a ``None`` sentinel instead of breaking module import.

    Every failure mode is treated as "FA4 unusable" rather than enumerating
    exception types: a *broken* install -- present, but built against a different
    CuTe DSL -- fails while executing ``flash_attn``'s module body and surfaces as
    whatever that line happens to raise (e.g. ``AttributeError`` for a DSL symbol
    that has since been removed), not as ``ImportError``. This package's
    ``__init__`` imports its backends unconditionally, so an escaping exception
    would take down every importer of ``visual_gen`` -- including ``VisualGen``
    itself, whose model packages pull in ``attention_backend.parallel``.
    """
    try:
        return getattr(importlib.import_module("flash_attn.cute.interface"), name), None
    except Exception as e:  # noqa: BLE001 - any failure means FA4 is unusable here
        return None, e
