# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nothing outside the glm_moe_dsa family reaches its targets, except the router line, and
no target reaches another.

With ``modeling_v2`` off the resolver returns before importing anything, so the
target can change no other model's behaviour as long as no other module imports
it. No shared module imports the target (``_S4_SHARED_IMPORTS`` is empty): the
engine reaches its step-loop hooks through ``runtime_hooks``, and every other
reference fails. Read off disk as text, like test_modeling_v2_claims.py: no
module is imported.

No shared Python or C++ file uses the family's names either (the C++ build adds the family's
folders in a few CMake lines), and no model of the zoo declares step-loop hooks, so each runs
with the engine's all-off default; that check imports the zoo's model modules.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2

pytestmark = pytest.mark.cpu_only

_MODELING_V2 = Path(_modeling_v2.__file__).resolve().parent
_PACKAGE = _MODELING_V2.parents[2]
_FAMILY = _MODELING_V2 / "models" / "glm_moe_dsa"
_ROUTER_INDEX = _MODELING_V2 / "_router_index.py"

# A reference to the family's modules or to the targets' classes.
_REFERENCE = re.compile(
    r"models[./]glm_moe_dsa\b|glm_5_2_nvfp4__sm_103__\w+|ModelingV2Glm52|GlmMoeDsa(Tp8|Tep|Dep)ModelingV2"
)
_ROUTER_LINE = '"GlmMoeDsaForCausalLM": "models.glm_moe_dsa.routing",'

_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa"
# The imports of the target from shared modules: (file, module, name, alias) -> count.
# They are matched on the parsed import, so the formatter may wrap them.
_S4_SHARED_IMPORTS: dict = {}


def _target_imports(path: Path, rel: str):
    """``(key, first line, last line)`` of every import of the target in ``path``."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(errors="ignore"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == _TARGET or node.module.startswith(_TARGET + "."):
                for alias in node.names:
                    key = (rel, node.module, alias.name, alias.asname)
                    out.append((key, node.lineno, node.end_lineno))
    return out


def _shared_files():
    for path in sorted(_PACKAGE.rglob("*.py")):
        if _FAMILY not in path.parents:
            yield path, str(path.relative_to(_PACKAGE.parent))


def test_only_the_router_line_references_the_family():
    offenders = []
    for path, rel in _shared_files():
        lines = path.read_text(errors="ignore").splitlines()
        hits = [n for n, line in enumerate(lines, 1) if _REFERENCE.search(line)]
        if not hits:
            continue
        allowed = set()
        for key, first, last in _target_imports(path, rel):
            if key in _S4_SHARED_IMPORTS:
                allowed.update(range(first, last + 1))
        for lineno in hits:
            line = lines[lineno - 1]
            if path == _ROUTER_INDEX and line.strip() == _ROUTER_LINE:
                continue
            if lineno in allowed:
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "modules outside glm_moe_dsa reference it:\n" + "\n".join(offenders)


def test_each_listed_shared_import_is_present():
    found = Counter()
    for path, rel in _shared_files():
        if "glm_moe_dsa" in path.read_text(errors="ignore"):
            found.update(key for key, _, _ in _target_imports(path, rel))
    assert dict(found) == _S4_SHARED_IMPORTS


def test_the_router_line_is_present():
    lines = [line.strip() for line in _ROUTER_INDEX.read_text().splitlines()]
    assert lines.count(_ROUTER_LINE) == 1


# The family's names; no file outside the family uses them.
_FORK_NAMES = re.compile(r"glm_ll_|\[glm_ll\]|GLM_LL|TRTLLM_STAIRCASE")
_CPP_FORK_NAMES = re.compile(r"glm_ll|GLM_LL|glmMoeDsa")
_REPO = _PACKAGE.parent
_CPP = _REPO / "cpp" / "tensorrt_llm"
_CPP_FAMILY_DIRS = ("kernels/glmMoeDsa", "thop/glmMoeDsa")
# The lines each parent CMakeLists.txt may spend on the family's folder.
_CMAKE_LINES = 3


def test_no_shared_file_names_the_fork():
    offenders = []
    for path, rel in _shared_files():
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if _FORK_NAMES.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "modules outside glm_moe_dsa name the fork:\n" + "\n".join(offenders)


def test_no_shared_cpp_file_names_the_family():
    if not _CPP.is_dir():
        pytest.skip("the C++ sources are not in this tree")
    family = [_CPP / d for d in _CPP_FAMILY_DIRS]
    offenders, cmake_lines = [], Counter()
    for path in sorted(_CPP.rglob("*")):
        if not path.is_file() or path.suffix not in (".cpp", ".cu", ".h", ".txt"):
            continue
        if any(d == path.parent or d in path.parents for d in family):
            continue
        rel = str(path.relative_to(_REPO))
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if not _CPP_FORK_NAMES.search(line):
                continue
            if path.name == "CMakeLists.txt" and path.parent in {d.parent for d in family}:
                cmake_lines[rel] += 1
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "C++ files outside the family's folders name it:\n" + "\n".join(offenders)
    assert all(count <= _CMAKE_LINES for count in cmake_lines.values()), dict(cmake_lines)


def _zoo_model_classes():
    """Class name -> providing module of every model of the zoo (imported lazily)."""
    from tensorrt_llm._torch.models._arch_index import MODEL_CLASS_TO_MODULE

    classes = dict(MODEL_CLASS_TO_MODULE)
    classes.setdefault("DeepseekV3ForCausalLM", "modeling_deepseekv3")
    classes.setdefault("GlmMoeDsaForCausalLM", "modeling_deepseekv3")
    return classes


_ZOO = _zoo_model_classes()


def test_no_zoo_model_declares_runtime_hooks():
    """Read off disk: no model of the zoo names the hooks, so each gets the all-off default."""
    models = _PACKAGE / "_torch" / "models"
    offenders = [
        str(path.relative_to(_PACKAGE.parent))
        for path in sorted(models.rglob("*.py"))
        if "runtime_hooks" in path.read_text(errors="ignore")
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("name", sorted(_ZOO))
def test_zoo_models_declare_no_runtime_hooks(name):
    """Every model class of the zoo, without its constructor, reads as the all-off hooks."""
    import importlib

    from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, runtime_hooks_of

    try:
        providing = importlib.import_module(f"tensorrt_llm._torch.models.{_ZOO[name]}")
    except ModuleNotFoundError as err:
        if err.name and err.name.split(".")[0] != "tensorrt_llm":
            pytest.skip(f"{err.name} is not installed")
        raise
    cls = getattr(providing, name)
    assert runtime_hooks_of(cls.__new__(cls)) is DEFAULT_RUNTIME_HOOKS


# --- the targets ----------------------------------------------------------------------------

_TARGET_DIRS = {
    "tp8": _FAMILY / "glm_5_2_nvfp4__sm_103__tp8",
    "tep": _FAMILY / "glm_5_2_nvfp4__sm_103__tep",
    "dep": _FAMILY / "glm_5_2_nvfp4__sm_103__dep",
}


def _module_file(module: str):
    """The file of a dotted module of the family, or None outside it."""
    if not (module == _TARGET or module.startswith(_TARGET + ".")):
        return None
    base = (
        _FAMILY.joinpath(*module[len(_TARGET) :].lstrip(".").split("."))
        if module != _TARGET
        else _FAMILY
    )
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _package_of(path: Path) -> str:
    parts = path.relative_to(_FAMILY).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return (
        ".".join((_TARGET, *parts[:-1]))
        if path.name != "__init__.py"
        else ".".join((_TARGET, *parts))
    )


def _imports(path: Path):
    """Every family module ``path`` imports, at module level or inside a function."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                package = _package_of(path).split(".")
                base = ".".join(package[: len(package) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            targets = [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            continue
        for name in targets:
            file = _module_file(name)
            if file is not None:
                found.add(file)
    return found


def _reachable(start: Path):
    """Every family file ``start`` imports, transitively, with the parent packages Python
    imports first."""
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for parent in path.parents:
            if parent == _FAMILY.parent:
                break
            if (parent / "__init__.py").is_file() and parent / "__init__.py" not in seen:
                todo.append(parent / "__init__.py")
        todo.extend(_imports(path))
    return seen


@pytest.mark.parametrize(
    "target, other",
    [(t, o) for t in _TARGET_DIRS for o in _TARGET_DIRS if t != o],
)
def test_a_target_imports_nothing_of_the_other(target, other):
    """Read off disk: the import closure of a target's modeling module stays in the target's
    own directory, the family's shared package and the routing module."""
    reached = _reachable(_TARGET_DIRS[target] / "modeling.py")
    assert not [p for p in reached if _TARGET_DIRS[other] in p.parents], sorted(
        str(p.relative_to(_FAMILY)) for p in reached if _TARGET_DIRS[other] in p.parents
    )
    allowed = (_TARGET_DIRS[target], _FAMILY / "shared")
    stray = [p for p in reached if p.parent != _FAMILY and not any(d in p.parents for d in allowed)]
    assert not stray, sorted(str(p.relative_to(_FAMILY)) for p in stray)


def test_the_shared_package_imports_no_target():
    shared = _FAMILY / "shared"
    for path in sorted(shared.rglob("*.py")):
        for imported in _imports(path):
            assert shared in imported.parents or imported.parent == _FAMILY, (path, imported)


def test_the_tep_target_loads_no_kernel_package_of_the_tp8_target():
    """``get_impl`` imports a path's package from the installed target's ``kernel_modules``;
    the tep target names none, so every package it loads is the family's."""
    source = (_TARGET_DIRS["tep"] / "modeling.py").read_text()
    assert "kernel_modules" not in source
    assert "glm_5_2_nvfp4__sm_103__tp8" not in "".join(
        p.read_text() for p in _TARGET_DIRS["tep"].glob("*.py")
    )


def test_the_dep_target_loads_no_kernel_package_of_another_target():
    """The dep target names no ``kernel_modules`` either, and no other target's directory."""
    source = (_TARGET_DIRS["dep"] / "modeling.py").read_text()
    assert "kernel_modules" not in source
    text = "".join(p.read_text() for p in _TARGET_DIRS["dep"].glob("*.py"))
    assert "glm_5_2_nvfp4__sm_103__tp8" not in text
    assert "glm_5_2_nvfp4__sm_103__tep" not in text
