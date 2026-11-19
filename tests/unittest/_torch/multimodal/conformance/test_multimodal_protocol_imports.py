# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared multimodal interfaces must be usable without a model implementation or router."""

import ast
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts/cbts/rules"]
from modeling_v2_source import imported_names  # noqa: E402

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[5]
_MODELS = "tensorrt_llm._torch.models."
_V2 = "tensorrt_llm._torch.modeling_v2."
# Add the shared composite base when it becomes available.
_GUARDED = (_MODELS + "multimodal",)
_NEUTRAL = {
    _MODELS + "modeling_multimodal_mixin",
    _MODELS + "modeling_multimodal_encoder",
    _MODELS + "modeling_multimodal_utils",
    _MODELS + "modeling_utils",
    "tensorrt_llm._torch.model_config",
    "tensorrt_llm._torch.tensor_lru_cache",
    "tensorrt_llm.inputs.multimodal",
    "tensorrt_llm.llmapi.llm_args",
}


def _forbidden_imports(module: str, root: Path) -> list[str]:
    pending = [(module, module)]
    visited = set()
    violations = []
    while pending:
        name, source = pending.pop()
        if name in visited or name in _NEUTRAL or name.split(".", 1)[0] != "tensorrt_llm":
            continue
        visited.add(name)
        path = root / Path(*name.split("."))
        package = name.rpartition(".")[0]
        if path.is_dir():
            path = path / "__init__.py"
            package = name
        else:
            path = path.with_suffix(".py")
        if not path.is_file():
            continue  # The imported name is a symbol rather than a source module.
        if (
            name.startswith((_MODELS + "modeling_", _V2 + "models."))
            or name == _V2 + "_router_index"
            or (name.startswith(_V2) and name.endswith(".routing"))
        ):
            violations.append(f"{source} imports {name}")
            continue
        imports, _ = imported_names(ast.parse(path.read_text()), package)
        pending.extend((imported, name) for imported in sorted(imports))
    return violations


@pytest.mark.parametrize("module", _GUARDED)
def test_multimodal_imports(module: str) -> None:
    assert not (violations := _forbidden_imports(module, _ROOT)), violations


def test_transitive_type_imports(tmp_path: Path) -> None:
    models = tmp_path / "tensorrt_llm/_torch/models"
    models.mkdir(parents=True)
    (models / "multimodal.py").write_text("from .helper import Config")
    (models / "helper.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from . import modeling_example\n"
        "    from ..modeling_v2._router_index import routes\n"
    )
    (models / "modeling_example.py").touch()
    v2 = models.parent / "modeling_v2"
    v2.mkdir()
    (v2 / "_router_index.py").touch()
    assert set(_forbidden_imports(_GUARDED[0], tmp_path)) == {
        f"{_MODELS}helper imports {_MODELS}modeling_example",
        f"{_MODELS}helper imports {_V2}_router_index",
    }
