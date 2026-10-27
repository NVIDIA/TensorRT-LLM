# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The boundary between stock code and modeling_v2 native code, read off disk.

modeling_v2 native code lives with the Python code of its family, in
``models/<family>/csrc/`` and ``models/<family>/<target>/csrc/``, and builds one
library per family that loads only when routing picks one of the family's
targets (``_native``). Three things rest on that staying true: ``off`` loading
nothing, a target never depending on a sibling, and CBTS testing a
family-only change with that family's tests alone
(``jenkins/scripts/cbts/cbts/rules/modeling_v2.py``). Each test here guards one
of the rules that keeps it true; none imports a target or needs a GPU.

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


def test_every_native_family_is_a_python_family():
    for family, csrc in _families().items():
        assert csrc.parent.name == family, (
            f"{csrc.relative_to(_REPO)}/CMakeLists.txt declares family {family!r}; "
            f"it belongs in models/{family}/csrc/"
        )
        assert (csrc.parent / "__init__.py").is_file(), (
            f"native family {family!r} has no Python family package models/{family}/"
        )


def test_native_code_lives_only_in_csrc():
    """In the Python tree, C++/CUDA and CMake live only in a family's or a
    target's csrc/, where the build finds them and these checks read them."""
    offenders = []
    for path in sorted(_MODELS.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        native = path.suffix in _NATIVE_SUFFIXES or path.name == "CMakeLists.txt"
        if not native:
            continue
        parts = path.relative_to(_MODELS).parts
        in_family_csrc = len(parts) >= 3 and parts[1] == "csrc"
        in_target_csrc = len(parts) >= 4 and parts[2] == "csrc"
        if path.name == "CMakeLists.txt":
            ok = len(parts) == 3 and parts[1] == "csrc"
        else:
            ok = in_family_csrc or in_target_csrc
        if not ok:
            offenders.append(str(path.relative_to(_REPO)))
    assert not offenders, (
        "native code outside models/<family>/csrc/ or models/<family>/<target>/csrc/ "
        f"(CMakeLists.txt only in models/<family>/csrc/): {offenders}"
    )


def test_every_native_target_is_a_python_target_that_loads_it():
    with_code = set()
    for family, target, tokens in _targets():
        package = _MODELS / family / target / "__init__.py"
        assert package.is_file(), f"native target {family}/{target} has no Python target package"
        assert "load_target_ops(__name__)" in package.read_text(), (
            f"{package.relative_to(_REPO)} must start by loading its native code: "
            "native_ops = load_target_ops(__name__)"
        )
        if _keyword_values(tokens, "SOURCES"):
            with_code.add((family, target))
            assert (_MODELS / family / target / "csrc").is_dir(), (
                f"{family}/{target} declares SOURCES but has no {target}/csrc/"
            )
    for csrc in sorted(_MODELS.glob("*/*/csrc")):
        family, target = csrc.parent.parent.name, csrc.parent.name
        assert (family, target) in with_code, (
            f"{csrc.relative_to(_REPO)} holds code no add_modeling_v2_target({family} {target} "
            "... SOURCES ...) builds"
        )


def test_python_targets_that_load_native_code_have_it():
    declared = {(family, target) for family, target, _ in _targets()}
    for package in sorted(_MODELS.glob("*/*/__init__.py")):
        if "load_target_ops(" not in package.read_text():
            continue
        family, target = package.parent.parent.name, package.parent.name
        assert (family, target) in declared, (
            f"{package.relative_to(_REPO)} loads native code, but no "
            f"add_modeling_v2_target({family} {target} ...) declares it"
        )


def test_declared_archs_cover_the_sm_in_the_target_name():
    for family, target, tokens in _targets():
        archs = _keyword_values(tokens, "ARCHS")
        sm = target.split("__")[1][len("sm_") :]
        if sm in _SM_GROUPS:
            needed = _SM_GROUPS[sm]
        else:
            assert sm.isdigit(), (
                f"{family}/{target}: sm segment 'sm_{sm}' is neither a number nor in _SM_GROUPS"
            )
            needed = (divmod(int(sm), 10),)
        for capability in needed:
            assert any(arch_covers(a, capability) for a in archs), (
                f"{family}/{target} declares ARCHS {archs}, none of which runs on "
                f"sm_{capability[0]}{capability[1]}, the architecture in its name"
            )


# --- What native code may reach -----------------------------------------------

_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def test_native_code_includes_no_other_target_or_family():
    """Native code may include stock headers, but of modeling_v2 only its own
    target's csrc/, its family's csrc/ and common/: a target never reaches a
    sibling or another family."""
    python_families = {p.name for p in _MODELS.iterdir() if (p / "__init__.py").is_file()}
    problems = []
    for family, target, csrc in _units():
        allowed = [f"{family}/csrc/", "tensorrt_llm/modeling_v2/common/"]
        if target is not None:
            allowed.append(f"{family}/{target}/csrc/")
        for path in _native_files(csrc):
            for include in _INCLUDE.findall(path.read_text()):
                where = f'{path.relative_to(_REPO)}: "{include}"'
                head = include.split("/", 1)[0]
                if head in python_families or include.startswith("tensorrt_llm/modeling_v2/"):
                    if not include.startswith(tuple(allowed)):
                        problems.append(f"{where} reaches another target or family")
                elif ".." in include or "/" not in include and (csrc / include).exists():
                    problems.append(
                        f'{where}: include the family\'s code from models/, "{family}/..."'
                    )
    assert not problems, "\n".join(problems)


_RAW_REGISTRATION = re.compile(r"\bTORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\s*\(")


def test_native_code_registers_only_into_its_own_namespace():
    """Raw TORCH_LIBRARY* would let a file register into trtllm:: or a sibling's
    namespace; the TRTLLM_MV2_ macros register into the unit's own only."""
    offenders = [
        str(path.relative_to(_REPO))
        for _, _, csrc in _units()
        for path in _native_files(csrc)
        if _RAW_REGISTRATION.search(path.read_text())
    ]
    assert not offenders, (
        f"use TRTLLM_MV2_LIBRARY_FRAGMENT / TRTLLM_MV2_LIBRARY_IMPL, not TORCH_LIBRARY*: {offenders}"
    )


_OP_DEF = re.compile(r'\bm\.def\(\s*"(\w+)\(')


def test_every_native_op_has_a_fake():
    """fake_ops.py doubles as the Python side's list of a unit's ops: the loader
    imports it, so an op without a fake is one torch.compile and fake tracing
    cannot see, and a fake without an op fails the load. A target's ops have
    their fakes in the target's fake_ops.py, the family's shared ops in the
    family's."""
    for family, target, native_dir in _units():
        python_dir = _MODELS / family if target is None else _MODELS / family / target
        ops = sorted(
            {op for p in _native_files(native_dir) for op in _OP_DEF.findall(p.read_text())}
        )
        if not ops:
            continue
        fake_ops = python_dir / "fake_ops.py"
        assert fake_ops.is_file(), (
            f"{native_dir.relative_to(_REPO)} defines {ops}, "
            f"but {fake_ops.relative_to(_REPO)} does not exist"
        )
        text = fake_ops.read_text()
        missing = [op for op in ops if f'::{op}"' not in text]
        assert not missing, f"{fake_ops.relative_to(_REPO)} has no fake for {missing}"


# --- What stock code may reach ------------------------------------------------

_STOCK_PY_REFERENCE = re.compile(r"modeling_v2[./]models\b|torch\.ops\.mv2_")
_STOCK_CPP_REFERENCE = re.compile(r'#\s*include\s*"tensorrt_llm/modeling_v2/')
# A path into one family's directory; the discovery's models/*/csrc glob names none.
_STOCK_FAMILY_PATH = re.compile(r"_experimental/modeling_v2/models/[a-z]")


def test_stock_python_does_not_reference_targets():
    offenders = []
    for path in sorted((_REPO / "tensorrt_llm").rglob("*.py")):
        if path.is_relative_to(_PY):
            continue
        if _STOCK_PY_REFERENCE.search(path.read_text(errors="ignore")):
            offenders.append(str(path.relative_to(_REPO)))
    assert not offenders, (
        f"stock code imports or calls modeling_v2 targets: {offenders}. Stock reaches "
        "modeling_v2 only through the resolver; a target hands stock a callable instead."
    )


def test_stock_build_does_not_reference_a_family():
    """A family's native code reaches the build only through
    cpp/tensorrt_llm/modeling_v2/'s discovery: no stock CMake, C++ or packaging
    file names a family, so adding one edits none of them."""
    offenders = []
    for root in ("cpp/tensorrt_llm", "cpp/include", "cpp/tests", "cpp/cmake"):
        for path in sorted((_REPO / root).rglob("*")):
            if not path.is_file() or path.is_relative_to(_INFRA):
                continue
            # Skip generated cubin sources; they include nothing of interest.
            if path.suffix in _NATIVE_SUFFIXES and path.stat().st_size < (1 << 20):
                if _STOCK_CPP_REFERENCE.search(path.read_text(errors="ignore")):
                    offenders.append(str(path.relative_to(_REPO)))
            elif path.name == "CMakeLists.txt" or path.suffix == ".cmake":
                if _STOCK_FAMILY_PATH.search(path.read_text()):
                    offenders.append(str(path.relative_to(_REPO)))
    for name in ("cpp/CMakeLists.txt", "setup.py", "scripts/build_wheel.py"):
        if _STOCK_FAMILY_PATH.search((_REPO / name).read_text()):
            offenders.append(name)
    assert not offenders, f"stock build files name a modeling_v2 family: {offenders}"


def test_native_files_carry_the_license_header():
    """The C++ license checker (jenkins/license_cpp.json, Apache) scans cpp/ only;
    this holds the families' csrc/ to the same header: NVIDIA's copyright and the
    Apache-2.0 license, as SPDX lines or the license text."""
    missing = []
    for _, _, csrc in _units():
        for path in [*_native_files(csrc), *csrc.glob("CMakeLists.txt")]:
            head = path.read_text(errors="ignore")[:1200]
            apache = (
                "SPDX-License-Identifier: Apache-2.0" in head
                or "Apache License, Version 2.0" in head
            )
            if "NVIDIA CORPORATION" not in head or not apache:
                missing.append(str(path.relative_to(_REPO)))
    assert not missing, f"no NVIDIA Apache-2.0 license header: {missing}"


def test_stock_custom_ops_register_no_fakes_for_targets():
    offenders = [
        str(path.relative_to(_REPO))
        for path in sorted((_REPO / "tensorrt_llm/_torch/custom_ops").rglob("*.py"))
        if re.search(r"register_fake\(\s*f?[\"']mv2_", path.read_text())
    ]
    assert not offenders, f"target fakes belong in the target's fake_ops.py: {offenders}"


# --- Targets ------------------------------------------------------------------


def test_no_op_probing():
    """A target calls exactly the ops it declares. Choosing a path by whether an
    op exists means `require` no longer promises the gated code ran."""
    # hasattr, or getattr with a default: an absent op becomes a path choice
    # instead of an error. The loader's two-argument getattr raises instead.
    probe = re.compile(r"\bhasattr\(\s*torch\.ops\b|\bgetattr\(\s*torch\.ops[\w.]*\s*,[^,()]+,")
    offenders = [
        str(path.relative_to(_REPO))
        for path in sorted(_PY.rglob("*.py"))
        if probe.search(path.read_text())
    ]
    assert not offenders, f"op probing in {offenders}"


def _is_docstring_or_future(node: ast.stmt) -> bool:
    docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
    return docstring or future


def test_family_packages_have_no_side_effects():
    """Native code loads per target, when ``route`` imports the target.

    ``models/__init__.py`` imports only the families' routing modules, a family
    package holds a docstring only, and a routing module imports a target only
    inside ``route``. Anything else would load a target's native code on every
    resolve, or for every target of the family.
    """
    for node in ast.parse((_MODELS / "__init__.py").read_text()).body:
        routing_import = (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
            and "." not in node.module
            and [alias.name for alias in node.names] == ["routing"]
        )
        assert _is_docstring_or_future(node) or routing_import, (
            f"{(_MODELS / '__init__.py').relative_to(_REPO)}:{node.lineno} -- "
            "models/__init__.py imports the families' routing modules and nothing else"
        )
    for init in sorted(_MODELS.glob("*/__init__.py")):
        for node in ast.parse(init.read_text()).body:
            assert _is_docstring_or_future(node), (
                f"{init.relative_to(_REPO)}:{node.lineno} -- a family package holds a docstring only"
            )
    for routing in sorted(_MODELS.glob("*/routing.py")):
        for node in ast.parse(routing.read_text()).body:
            assert not (isinstance(node, ast.ImportFrom) and node.level == 1), (
                f"{routing.relative_to(_REPO)}:{node.lineno} -- imports .{node.module} at "
                "module level; a routing module imports a target only inside route"
            )


def test_modeling_v2_defaults_to_off():
    """`off` loads nothing, which is what lets CBTS run only modeling_v2 tests
    for a modeling_v2 change. Changing the default changes that rule too."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    assert TorchLlmArgs.model_fields["modeling_v2"].default == "off"


# --- Test lists ----------------------------------------------------------------

_MV2_ENTRY = re.compile(r"^\s*-\s+((?:unittest/_torch/modeling_v2|accuracy/modeling_v2)\S*)(.*)$")


def _test_db_entries() -> list[tuple[str, str, str]]:
    entries = []
    for yml in sorted(_TEST_DB.glob("*.yml")):
        for line in yml.read_text().splitlines():
            match = _MV2_ENTRY.match(line)
            if match:
                entries.append((yml.name, match.group(1), match.group(2)))
    return entries


def test_family_unit_entries_point_at_tests():
    """A family entry naming a directory with no tests is a collection error in CI."""
    for yml, target, _ in _test_db_entries():
        if target.startswith("unittest/_torch/modeling_v2/models/"):
            path = _REPO / "tests" / target
            assert path.exists() and (path.is_file() or any(path.rglob("test_*.py"))), (
                f"{yml}: {target} names no test file or directory with tests"
            )


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


def test_families_with_native_code_have_tests_a_test_list_runs():
    entries = _test_db_entries()
    for family in _families():
        unit = f"unittest/_torch/modeling_v2/models/{family}"
        accuracy = f"accuracy/modeling_v2/test_{family}.py"
        has_unit_tests = any((_REPO / "tests" / unit).rglob("test_*.py"))
        assert any(
            target.startswith(accuracy) or (has_unit_tests and _runs(target, options, unit))
            for _, target, options in entries
        ), f"family {family!r} has native code but no test-db entry runs {unit} or {accuracy}"


def test_accuracy_entries_name_a_family():
    families = {p.name for p in _MODELS.iterdir() if (p / "__init__.py").is_file()}
    for yml, target, _ in _test_db_entries():
        match = re.match(r"accuracy/modeling_v2/test_(\w+)\.py", target)
        if match:
            assert match.group(1) in families, (
                f"{yml}: {target} is named after no modeling_v2 family; CBTS selects "
                "accuracy gates by family name"
            )


# --- CMake and Python agree ------------------------------------------------------


@pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
def test_cmake_and_python_agree_on_names_and_architectures(tmp_path):
    """The loader and these checks compute identifiers, namespaces and
    architecture coverage in Python; the build computes them in modeling_v2.cmake."""
    names = ["gpt_oss", "minimax_m3", "r1_0528_nvfp4__sm_10x__default"]
    script = tmp_path / "probe.cmake"
    lines = [f'include("{_CMAKE_MODULE}")']
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
