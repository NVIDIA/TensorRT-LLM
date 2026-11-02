# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The boundary between stock code and modeling_v2 native code, read off disk.

modeling_v2 native code lives with the Python code of its family, in
``models/<family>/csrc/`` and ``models/<family>/<target>/csrc/``, and builds one
library per family that loads only when routing picks one of the family's
targets (``_native``). Three things rest on that staying true: ``off`` loading
nothing, a target never depending on a sibling, and CBTS testing a
family-only change with that family's tests alone
(``jenkins/scripts/cbts/cbts/rules/modeling_v2.py``). The tests here guard the
rules that keep it true for per-family libraries. None imports a target or needs a GPU.

The paths are the source checkout's, found from this file: these are claims
about the tree a PR changes, C++ included, not about an installed wheel.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _native

_REPO = Path(__file__).resolve().parents[4]
# What every family shares: the discovery and the common/ macros.
_INFRA = _REPO / "cpp/tensorrt_llm/modeling_v2"
_PY = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2"
_MODELS = _PY / "models"
_CMAKE_MODULE = _REPO / "cpp/cmake/modules/modeling_v2.cmake"
_TEST_DB = _REPO / "tests/integration/test_lists/test-db"

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(not _INFRA.is_dir(), reason="needs the source checkout"),
]

_NATIVE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp")
# A target name's sm segment that names a family of architectures.
_SM_GROUPS = {"10x": ((10, 0), (10, 3))}


def arch_covers(arch: str, capability: tuple[int, int]) -> bool:
    """Whether code built for ``arch`` (``NN``, ``NNa`` or ``NNf``) runs on ``capability``.

    ``NNa`` runs on exactly NN; ``NN`` on NN and later minor versions of the same
    major version; ``NNf`` on every member of family NN, the same way.
    """
    suffix = arch[-1] if arch[-1] in "af" else ""
    major, minor = divmod(int(arch[:-1] if suffix else arch), 10)
    if suffix == "a":
        return capability == (major, minor)
    return capability[0] == major and capability[1] >= minor


# --- Reading the CMake declarations ------------------------------------------


def _calls(text: str, function: str) -> list[list[str]]:
    """Argument tokens of every ``function(...)`` call in a CMakeLists.txt."""
    text = re.sub(r"#.*", "", text)
    return [m.group(1).split() for m in re.finditer(rf"\b{function}\s*\(([^)]*)\)", text)]


def _keyword_values(tokens: list[str], keyword: str) -> list[str]:
    keywords = {"ARCHS", "SOURCES", "DEPS", "REQUIRES"}
    values, active = [], False
    for token in tokens:
        if token in keywords:
            active = token == keyword
        elif active:
            values.append(token)
    return values


def _families() -> dict[str, Path]:
    """family -> its csrc/ directory, from every models/<family>/csrc/CMakeLists.txt."""
    families = {}
    for cmake in sorted(_MODELS.glob("*/csrc/CMakeLists.txt")):
        calls = _calls(cmake.read_text(), "add_modeling_v2_family")
        assert len(calls) == 1 and len(calls[0]) == 1, (
            f"{cmake.relative_to(_REPO)} must call add_modeling_v2_family(<family>) exactly once"
        )
        families[calls[0][0]] = cmake.parent
    return families


def _targets() -> list[tuple[str, str, list[str]]]:
    """(family, target, tokens) of every add_modeling_v2_target call."""
    out = []
    for csrc in _families().values():
        for tokens in _calls((csrc / "CMakeLists.txt").read_text(), "add_modeling_v2_target"):
            out.append((tokens[0], tokens[1], tokens[2:]))
    return out


def _units() -> list[tuple[str, str | None, Path]]:
    """(family, target or None for shared code, csrc directory) of every unit with code."""
    units = [(family, None, csrc) for family, csrc in _families().items()]
    for family, target, _ in _targets():
        csrc = _MODELS / family / target / "csrc"
        if csrc.is_dir():
            units.append((family, target, csrc))
    return units


def _native_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in _NATIVE_SUFFIXES)


# --- Layout ------------------------------------------------------------------


def test_native_code_is_where_the_build_and_the_loader_find_it():
    """Native code and CMake live only in a family's or target's csrc/, every
    native family and target is a Python one, a target with native code loads it,
    and its ARCHS cover the architecture its name promises."""
    for family, csrc in _families().items():
        assert csrc.parent.name == family and (csrc.parent / "__init__.py").is_file(), (
            f"{csrc.relative_to(_REPO)}/CMakeLists.txt declares family {family!r}; "
            f"it belongs in models/{family}/csrc/ of the Python family {family}"
        )

    misplaced = []
    for path in sorted(_MODELS.rglob("*")):
        if not path.is_file() or not (
            path.suffix in _NATIVE_SUFFIXES or path.name == "CMakeLists.txt"
        ):
            continue
        parts = path.relative_to(_MODELS).parts
        if path.name == "CMakeLists.txt":
            ok = len(parts) == 3 and parts[1] == "csrc"
        else:
            ok = (len(parts) >= 3 and parts[1] == "csrc") or (
                len(parts) >= 4 and parts[2] == "csrc"
            )
        if not ok:
            misplaced.append(str(path.relative_to(_REPO)))
    assert not misplaced, (
        "native code outside models/<family>/csrc/ or models/<family>/<target>/csrc/ "
        f"(CMakeLists.txt only in models/<family>/csrc/): {misplaced}"
    )

    declared, with_code = set(), set()
    for family, target, tokens in _targets():
        declared.add((family, target))
        package = _MODELS / family / target / "__init__.py"
        assert package.is_file() and "load_target_ops(__name__)" in package.read_text(), (
            f"native target {family}/{target}: its Python package must start with "
            "native_ops = load_target_ops(__name__)"
        )
        if _keyword_values(tokens, "SOURCES"):
            with_code.add((family, target))
        archs = _keyword_values(tokens, "ARCHS")
        sm = target.split("__")[1][len("sm_") :]
        assert sm in _SM_GROUPS or sm.isdigit(), f"{family}/{target}: unknown sm segment sm_{sm}"
        for capability in _SM_GROUPS[sm] if sm in _SM_GROUPS else (divmod(int(sm), 10),):
            assert any(arch_covers(a, capability) for a in archs), (
                f"{family}/{target} declares ARCHS {archs}, none of which runs on "
                f"sm_{capability[0]}{capability[1]}, the architecture in its name"
            )
    for csrc in sorted(_MODELS.glob("*/*/csrc")):
        unit = (csrc.parent.parent.name, csrc.parent.name)
        assert unit in with_code, f"{csrc.relative_to(_REPO)} holds code no target's SOURCES builds"
    for package in sorted(_MODELS.glob("*/*/__init__.py")):
        unit = (package.parent.parent.name, package.parent.name)
        assert "load_target_ops(" not in package.read_text() or unit in declared, (
            f"{package.relative_to(_REPO)} loads native code no add_modeling_v2_target declares"
        )


# --- What native code may reach -----------------------------------------------

_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_RAW_REGISTRATION = re.compile(r"\bTORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\s*\(")
_OP_DEF = re.compile(r'\bm\.def\(\s*"(\w+)\(')


def test_native_code_stays_in_its_own_unit():
    """Native code may include stock headers, but of modeling_v2 only its own
    target's csrc/, its family's csrc/ and common/; it registers through the
    TRTLLM_MV2_ macros, into its own namespace only; every op it defines has a
    fake in the unit's fake_ops.py (which the loader imports); and every file
    carries NVIDIA's Apache-2.0 header, which the C++ license checker only checks
    under cpp/."""
    python_families = {p.name for p in _MODELS.iterdir() if (p / "__init__.py").is_file()}
    problems = []
    for family, target, csrc in _units():
        allowed = [f"{family}/csrc/", "tensorrt_llm/modeling_v2/common/"]
        if target is not None:
            allowed.append(f"{family}/{target}/csrc/")
        ops = set()
        for path in [*_native_files(csrc), *csrc.glob("CMakeLists.txt")]:
            text = path.read_text(errors="ignore")
            where = str(path.relative_to(_REPO))
            head = text[:1200]
            apache = (
                "SPDX-License-Identifier: Apache-2.0" in head
                or "Apache License, Version 2.0" in head
            )
            if "NVIDIA CORPORATION" not in head or not apache:
                problems.append(f"{where}: no NVIDIA Apache-2.0 license header")
            if path.name == "CMakeLists.txt":
                continue
            for include in _INCLUDE.findall(text):
                head = include.split("/", 1)[0]
                if head in python_families or include.startswith("tensorrt_llm/modeling_v2/"):
                    if not include.startswith(tuple(allowed)):
                        problems.append(f'{where}: "{include}" reaches another target or family')
                elif ".." in include or "/" not in include and (csrc / include).exists():
                    problems.append(
                        f'{where}: "{include}": include it from models/, "{family}/..."'
                    )
            if _RAW_REGISTRATION.search(text):
                problems.append(f"{where}: use the TRTLLM_MV2_LIBRARY_* macros, not TORCH_LIBRARY*")
            ops.update(_OP_DEF.findall(text))
        if ops:
            python_dir = _MODELS / family if target is None else _MODELS / family / target
            fake_ops = python_dir / "fake_ops.py"
            text = fake_ops.read_text() if fake_ops.is_file() else ""
            missing = sorted(op for op in ops if f'::{op}"' not in text)
            if missing:
                problems.append(f"{fake_ops.relative_to(_REPO)}: no fake for {missing}")
    assert not problems, "\n".join(problems)


# --- What stock code may reach ------------------------------------------------

_STOCK_PY_REFERENCE = re.compile(
    r"modeling_v2[./]models\b|torch\.ops\.mv2_|register_fake\(\s*f?[\"']mv2_"
)
_STOCK_CPP_REFERENCE = re.compile(r'#\s*include\s*"tensorrt_llm/modeling_v2/')
# A path into one family's directory; the discovery's models/*/csrc glob names none.
_STOCK_FAMILY_PATH = re.compile(r"_experimental/modeling_v2/models/[a-z]")


def test_stock_does_not_reach_modeling_v2():
    """Stock Python reaches modeling_v2 only through the resolver (no target
    import, op call or fake), and no stock CMake, C++ or packaging file names a
    family: a family reaches the build only through
    cpp/tensorrt_llm/modeling_v2/'s discovery, so adding one edits none of them."""
    offenders = [
        str(path.relative_to(_REPO))
        for path in sorted((_REPO / "tensorrt_llm").rglob("*.py"))
        if not path.is_relative_to(_PY)
        and _STOCK_PY_REFERENCE.search(path.read_text(errors="ignore"))
    ]
    for root in ("cpp/tensorrt_llm", "cpp/include", "cpp/tests", "cpp/cmake"):
        for path in sorted((_REPO / root).rglob("*")):
            if not path.is_file() or path.is_relative_to(_INFRA):
                continue
            # Skip generated cubin sources; they include nothing of interest.
            if path.suffix in _NATIVE_SUFFIXES and path.stat().st_size < (1 << 20):
                pattern = _STOCK_CPP_REFERENCE
            elif path.name == "CMakeLists.txt" or path.suffix == ".cmake":
                pattern = _STOCK_FAMILY_PATH
            else:
                continue
            if pattern.search(path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(_REPO)))
    for name in ("cpp/CMakeLists.txt", "setup.py", "scripts/build_wheel.py"):
        if _STOCK_FAMILY_PATH.search((_REPO / name).read_text()):
            offenders.append(name)
    assert not offenders, f"stock code reaches modeling_v2 targets or families: {offenders}"


# --- Targets load only when routed ----------------------------------------------


def _is_docstring_or_future(node: ast.stmt) -> bool:
    docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
    return docstring or future


def test_native_code_loads_only_when_routing_picks_its_target():
    """``models/__init__.py`` imports only the families' routing modules, a
    family package holds a docstring only, and a routing module imports a target
    only inside ``route``; anything else would load native code on every resolve.
    A target calls exactly the ops it declares: choosing a path by whether an op
    exists (hasattr, or getattr with a default) would mean ``require`` no longer
    promises the gated code ran."""
    models_init = _MODELS / "__init__.py"
    for node in ast.parse(models_init.read_text()).body:
        routing_import = (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
            and "." not in node.module
            and [alias.name for alias in node.names] == ["routing"]
        )
        assert _is_docstring_or_future(node) or routing_import, (
            f"{models_init.relative_to(_REPO)}:{node.lineno} -- imports something besides "
            "the families' routing modules"
        )
    for init in sorted(_MODELS.glob("*/__init__.py")):
        for node in ast.parse(init.read_text()).body:
            assert _is_docstring_or_future(node), (
                f"{init.relative_to(_REPO)}:{node.lineno} -- a family package holds a docstring only"
            )
    for init in sorted(_MODELS.glob("*/ops/__init__.py")):
        assert all(_is_docstring_or_future(node) for node in ast.parse(init.read_text()).body), (
            f"{init.relative_to(_REPO)}: importing unrelated ops must not load a native library"
        )
    for routing in sorted(_MODELS.glob("*/routing.py")):
        for node in ast.parse(routing.read_text()).body:
            assert not (isinstance(node, ast.ImportFrom) and node.level == 1), (
                f"{routing.relative_to(_REPO)}:{node.lineno} -- imports .{node.module} at "
                "module level; a routing module imports a target only inside route"
            )
    probe = re.compile(r"\bhasattr\(\s*torch\.ops\b|\bgetattr\(\s*torch\.ops[\w.]*\s*,[^,()]+,")
    probing = [
        str(p.relative_to(_REPO)) for p in sorted(_PY.rglob("*.py")) if probe.search(p.read_text())
    ]
    assert not probing, f"op probing in {probing}"


# --- Test lists ----------------------------------------------------------------

_MV2_ENTRY = re.compile(r"^\s*-\s+((?:unittest/_torch/modeling_v2|accuracy/modeling_v2)\S*)(.*)$")


def _runs(target: str, options: str, unit: str) -> bool:
    """Whether a test-db entry runs tests under the directory ``unit``: it names
    the directory or something in it, or a directory above it that it does not
    ``--ignore``."""
    if target == unit or target.startswith(unit + "/"):
        return True
    if not unit.startswith(target.rstrip("/") + "/"):
        return False
    ignored = (path.rstrip("/") for path in re.findall(r"--ignore=(\S+)", options))
    return not any(unit == path or unit.startswith(path + "/") for path in ignored)


def test_test_lists_run_every_native_family():
    """Some test-db entry runs each native family's tests, and accuracy files are
    named after a family, since CBTS selects them by family name."""
    entries = [
        (yml.name, *match.groups())
        for yml in sorted(_TEST_DB.glob("*.yml"))
        for match in map(_MV2_ENTRY.match, yml.read_text().splitlines())
        if match
    ]
    for family in _families():
        unit = f"unittest/_torch/modeling_v2/models/{family}"
        accuracy = f"accuracy/modeling_v2/test_{family}.py"
        has_unit_tests = any((_REPO / "tests" / unit).rglob("test_*.py"))
        assert any(
            target.startswith(accuracy) or (has_unit_tests and _runs(target, options, unit))
            for _, target, options in entries
        ), f"family {family!r} has native code but no test-db entry runs {unit} or {accuracy}"
    families = {p.name for p in _MODELS.iterdir() if (p / "__init__.py").is_file()}
    for yml, target, _ in entries:
        match = re.match(r"accuracy/modeling_v2/test_(\w+)\.py", target)
        assert not match or match.group(1) in families, (
            f"{yml}: {target} is named after no modeling_v2 family"
        )


# --- CMake and Python agree ------------------------------------------------------


@pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
def test_cmake_and_python_agree_on_names_and_architectures(tmp_path):
    """The loader and these checks compute identifiers, namespaces and
    architecture coverage in Python; the build computes them in modeling_v2.cmake."""
    names = ["gpt_oss", "minimax_m3", "r1_0528_nvfp4__sm_10x__default"]
    script = tmp_path / "probe.cmake"
    lines = ["cmake_minimum_required(VERSION 3.18)", f'include("{_CMAKE_MODULE}")']
    for name in names:
        lines += [
            f'_modeling_v2_ident(i "{name}")',
            f'message(STATUS "name {name} ${{i}}")',
        ]
    lines += [
        "_modeling_v2_resolve_archs(r DECLARED 80 90a 100f 103a 120f REQUESTED 86 90 103 107)",
        'message(STATUS "archs ${r}")',
    ]
    script.write_text("\n".join(lines) + "\n")
    output = subprocess.run(
        ["cmake", "-P", str(script)], capture_output=True, text=True, check=True
    ).stdout
    for name in names:
        assert f"name {name} {_native._identifier(name)}" in output, output
    assert "archs 90a-real;100f-real;103a-real" in output, output
    # The family namespace the build registers is the loader's.
    assert _native._namespace("gpt_oss", "gpt_oss_120b__sm_10x__default") == (
        "mv2_gpt_oss_gpt_oss_120b_sm_10x_default"
    )
