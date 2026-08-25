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
"""The cutlass DSL must stay optional for importing the MoE backend.

``cute_dsl_custom_ops`` defines its runner classes inside an
``if IS_CUTLASS_DSL_AVAILABLE:`` block with no else-branch, so without the DSL
those names do not exist. ``fused_moe_cute_dsl`` is imported eagerly by
``create_moe`` beneath ``_torch.models``, so a module-scope import of one of them
turns a missing *optional* dependency into an ImportError for the whole
model-architecture registry -- every ``LLM(...)`` construction, not just this
backend. See nvbug 6644645.

The check is structural (AST) on purpose: wherever the DSL is installed -- which
includes CI, since ``nvidia-cutlass-dsl`` is an unconditional requirements.txt
entry -- the offending import succeeds, so no runtime probe could tell a fixed
revision from a broken one. Reading the source is also why nothing here imports
the MoE backend: that module is what the bug breaks, so importing it would make
this file error out on the regression it exists to report.
"""

import ast
from pathlib import Path

import tensorrt_llm

_TORCH_ROOT = Path(tensorrt_llm.__file__).parent / "_torch"
PROVIDER = _TORCH_ROOT / "custom_ops" / "cute_dsl_custom_ops.py"
CONSUMER = _TORCH_ROOT / "modules" / "fused_moe" / "fused_moe_cute_dsl.py"


def _parse(path: Path) -> ast.Module:
    assert path.is_file(), f"expected source at {path}"
    return ast.parse(path.read_text(encoding="utf-8"))


def _dsl_gated_names() -> set[str]:
    """Names the provider defines only when the cutlass DSL is importable.

    Derived from the provider rather than hard-coded, so the set cannot drift as
    runners are added, and so the premise that these names really do live under
    the availability guard is enforced here instead of in a separate test.
    """
    names: set[str] = set()
    for node in _parse(PROVIDER).body:
        if (
            isinstance(node, ast.If)
            and getattr(node.test, "id", None) == "IS_CUTLASS_DSL_AVAILABLE"
        ):
            names.update(
                child.name
                for child in node.body
                if isinstance(child, (ast.ClassDef, ast.FunctionDef))
            )
    assert names, "found no conditionally-defined names; the provider's guard must have moved"
    return names


def _module_scope_imports() -> set[str]:
    """Names the MoE backend binds at import time.

    Only top-level statements count -- an import nested in a function body is
    exactly the fix, and one inside ``if``/``try`` is already conditional.
    """
    return {
        alias.name
        for node in _parse(CONSUMER).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def test_no_dsl_gated_name_is_imported_at_module_scope() -> None:
    offenders = sorted(_dsl_gated_names() & _module_scope_imports())
    assert not offenders, (
        f"fused_moe_cute_dsl.py imports {offenders} at module scope, but those "
        "names only exist when the optional cutlass DSL is installed. create_moe "
        "imports this module eagerly under _torch.models, so this breaks every "
        "LLM(...) construction when the DSL is absent. Import them inside the "
        "function that uses them instead."
    )


def test_the_ungated_provider_helper_still_comes_from_module_scope() -> None:
    """Guards against over-correcting into blanket deferral.

    ``GroupedGemmInputsHelper`` sits outside the provider's availability guard,
    so deferring it too would be churn without a reason -- and would make the
    assertion above pass for the wrong reason.
    """
    assert "GroupedGemmInputsHelper" in _module_scope_imports()
