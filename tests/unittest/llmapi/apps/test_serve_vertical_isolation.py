# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The common serving path must not drag a vertical in behind it.

``openai_protocol`` is imported by components every LLM deployment uses — the
tool parsers, for one — so an import-time edge from it into ``visual_gen``
makes a DeepSeek test case depend on VisualGen. That matters for
coverage-driven CI triggering, where the import graph decides which stages a
change runs, and the cost is the same whether the edge pulls three modules or
three hundred.

Shared declaration-only types therefore live in ``tensorrt_llm.media``, and
anything with a VisualGen resolver behind it is imported inside the function
that needs it.
"""

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import importlib, json, sys
    importlib.import_module({module!r})
    leaked = sorted(
        name for name in sys.modules
        if name == "tensorrt_llm.visual_gen"
        or name.startswith("tensorrt_llm.visual_gen.")
        or name.startswith("tensorrt_llm._torch.visual_gen")
    )
    print(json.dumps(leaked))
    """
)


def _visual_gen_modules_pulled_by(module: str) -> list[str]:
    """Import *module* in a fresh interpreter and report VisualGen fallout.

    A subprocess because the check is about what a module drags in on its own;
    inside pytest the whole suite has already imported half the product.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"probe failed for {module}:\n{proc.stderr[-2000:]}"
    return __import__("json").loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "module",
    [
        "tensorrt_llm.serve.openai_protocol",
        "tensorrt_llm.serve.tool_parser.deepseekv3_parser",
    ],
)
def test_common_serving_module_does_not_import_visual_gen(module):
    leaked = _visual_gen_modules_pulled_by(module)
    assert leaked == [], (
        f"{module} pulls VisualGen in at import time: {leaked}. Shared "
        "declaration-only types belong in tensorrt_llm.media; import anything "
        "with a VisualGen resolver behind it inside the function that uses it."
    )


def test_media_reference_types_are_dependency_free():
    """The shared leaf must not acquire a VisualGen import of its own."""
    assert _visual_gen_modules_pulled_by("tensorrt_llm.media.reference") == []


def test_visual_gen_still_exports_the_shared_types():
    """Moving the types must not move the public API they are reached through."""
    from tensorrt_llm.media.reference import MediaRef as LeafMediaRef
    from tensorrt_llm.visual_gen import MediaRef

    assert MediaRef is LeafMediaRef
