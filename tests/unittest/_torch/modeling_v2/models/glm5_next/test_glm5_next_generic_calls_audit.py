# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``UNCERTIFIED_GENERIC_CALLS`` audit on source fixtures: what it counts as a stock import and which modules it
audits."""

import ast
import textwrap

import pytest

__extra_import_path__ = ["."]
from _glm5_next_generic_calls import (  # noqa: E402
    PACKAGE,
    audited_modules,
    check_names_every_stock_import,
    stock_imports,
)

pytestmark = pytest.mark.cpu_only

_TARGET = "fixture_target"
_ENTRY = f"{PACKAGE}.models.glm5_next.{_TARGET}.ops.attention"


def _imports(source: str) -> set[str]:
    return stock_imports(ast.parse(textwrap.dedent(source)), _TARGET)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "from tensorrt_llm._torch.modules.gated_mlp import GatedMLP",
            {"tensorrt_llm._torch.modules.gated_mlp.GatedMLP"},
        ),
        (
            "import tensorrt_llm._torch.modules.gated_mlp as mlp",
            {"tensorrt_llm._torch.modules.gated_mlp"},
        ),
        ("import tensorrt_llm._torch.modules.gated_mlp", {"tensorrt_llm._torch.modules.gated_mlp"}),
        ("def f():\n    import tensorrt_llm._utils as u", {"tensorrt_llm._utils"}),
        (f"import {_ENTRY} as attention", set()),
        (f"from {_ENTRY} import sparse_mla", set()),
        ("from tensorrt_llm.mapping import Mapping", set()),
        ("import torch\nimport os.path", set()),
    ],
    ids=[
        "from-import",
        "aliased-module-import",
        "module-import",
        "function-local-import",
        "entry-module-import",
        "entry-from-import",
        "non-compute-name",
        "not-tensorrt_llm",
    ],
)
def test_stock_imports(source, expected):
    assert _imports(source) == expected


def _write_target(tmp_path, files: dict[str, str]):
    target_dir = tmp_path / _TARGET
    for name, source in files.items():
        path = target_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    return target_dir


def test_a_module_reaching_stock_code_through_a_module_import_must_declare_it(tmp_path):
    target_dir = _write_target(
        tmp_path,
        {
            "modeling.py": """\
                import tensorrt_llm._torch.modules.gated_mlp as mlp
                UNCERTIFIED_GENERIC_CALLS = ("tensorrt_llm._torch.modules.gated_mlp.GatedMLP",)
                """
        },
    )
    with pytest.raises(AssertionError, match="imported but not in UNCERTIFIED_GENERIC_CALLS"):
        check_names_every_stock_import(target_dir / "modeling.py", _TARGET)


def test_an_undeclared_module_that_imports_stock_code_is_audited_and_fails(tmp_path):
    target_dir = _write_target(
        tmp_path,
        {
            "modeling.py": """\
                from tensorrt_llm._torch.modules.linear import Linear
                UNCERTIFIED_GENERIC_CALLS = ("tensorrt_llm._torch.modules.linear.Linear",)
                """,
            "helpers/undeclared.py": "from tensorrt_llm._torch.modules.gated_mlp import GatedMLP\n",
            "ops/pure.py": "import torch\n",
        },
    )
    modules = audited_modules(_TARGET, family_dir=tmp_path)
    assert modules == [target_dir / "helpers/undeclared.py", target_dir / "modeling.py"]
    check_names_every_stock_import(target_dir / "modeling.py", _TARGET)
    with pytest.raises(AssertionError, match="declares no UNCERTIFIED_GENERIC_CALLS"):
        check_names_every_stock_import(target_dir / "helpers/undeclared.py", _TARGET)
