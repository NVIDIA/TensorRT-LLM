# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional PrimTS bindings to FlashInfer attention trace templates."""

from importlib import import_module
from types import ModuleType
from typing import Any


_OPTIONAL_TRACE_MODULES = frozenset(
    {
        "flashinfer.trace",
        "flashinfer.trace.templates",
        "flashinfer.trace.templates.attention",
    }
)


def _load_attention_trace_templates() -> ModuleType | None:
    """Load trace templates when the hosting FlashInfer build provides them."""

    # Downstream projects may vendor PrimTS while depending on a different
    # FlashInfer release.  Its trace schemas need not match these wrappers, so
    # keep every binding local to PrimTS and disable tracing for that vendored
    # copy without changing the host package's module or attributes.
    if __package__ != "flashinfer.attention.prims_ts":
        return None

    try:
        return import_module("flashinfer.trace.templates.attention")
    except ModuleNotFoundError as error:
        if error.name not in _OPTIONAL_TRACE_MODULES:
            raise
        return None


_ATTENTION_TRACE_TEMPLATES = _load_attention_trace_templates()


def _get_attention_trace_template(name: str) -> Any | None:
    """Return one optional trace binding without mutating the trace module."""

    if _ATTENTION_TRACE_TEMPLATES is None:
        return None
    return getattr(_ATTENTION_TRACE_TEMPLATES, name, None)
