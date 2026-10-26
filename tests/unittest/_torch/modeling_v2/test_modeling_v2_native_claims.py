# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The boundary between stock code and modeling_v2 native code, read off disk.

modeling_v2 native code lives in ``cpp/tensorrt_llm/modeling_v2/`` and builds
one library per family that loads only when routing picks one of the family's
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
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import camel, identifier, torch_namespace

_REPO = Path(__file__).resolve().parents[4]
_NATIVE = _REPO / "cpp/tensorrt_llm/modeling_v2"
_PY = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2"
_MODELS = _PY / "models"
_CMAKE_MODULE = _REPO / "cpp/cmake/modules/modeling_v2.cmake"
_TEST_DB = _REPO / "tests/integration/test_lists/test-db"

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(not _NATIVE.is_dir(), reason="needs the source checkout"),
]

_NATIVE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp")
# Native directories that are not families.
_NON_FAMILY_DIRS = {"common", "catalog"}
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
    """family -> native family directory, from every family CMakeLists.txt."""
    families = {}
    for cmake in sorted(_NATIVE.glob("*/CMakeLists.txt")):
        calls = _calls(cmake.read_text(), "add_modeling_v2_family")
        assert len(calls) == 1 and len(calls[0]) == 1, (
            f"{cmake.relative_to(_REPO)} must call add_modeling_v2_family(<family>) exactly once"
        )
        families[calls[0][0]] = cmake.parent
    return families


def _targets() -> list[tuple[str, str, list[str]]]:
    """(family, target, tokens) of every add_modeling_v2_target call."""
    out = []
    for family_dir in _families().values():
        for tokens in _calls((family_dir / "CMakeLists.txt").read_text(), "add_modeling_v2_target"):
            out.append((tokens[0], tokens[1], tokens[2:]))
    return out


def _native_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in _NATIVE_SUFFIXES)


def _contract_surface() -> list[str]:
    lines = (_NATIVE / "contractSurface.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# --- Layout ------------------------------------------------------------------


def test_every_native_family_is_a_python_family():
    for family, family_dir in _families().items():
        assert family_dir.name == camel(family), (
            f"family {family!r} must live in {_NATIVE.relative_to(_REPO)}/{camel(family)}"
        )
        assert (_MODELS / family / "__init__.py").is_file(), (
            f"native family {family!r} has no Python family package models/{family}/"
        )


def test_every_native_target_is_a_python_target_that_loads_it():
    declared = set()
    for family, target, _ in _targets():
        declared.add((family, target))
        package = _MODELS / family / target / "__init__.py"
        assert package.is_file(), f"native target {family}/{target} has no Python target package"
        assert "load_target_ops(__name__)" in package.read_text(), (
            f"{package.relative_to(_REPO)} must start by loading its native code: "
            "native_ops = load_target_ops(__name__)"
        )
    for family, family_dir in _families().items():
        for sub in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            if sub.name == "shared":
                continue
            assert any(camel(t) == sub.name for f, t in declared if f == family), (
                f"{sub.relative_to(_REPO)} is not the directory of any add_modeling_v2_target "
                f"of family {family!r}"
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


def test_native_code_includes_only_its_own_code_and_the_contract_surface():
    surface = _contract_surface()
    problems = []
    for family, family_dir in _families().items():
        for unit_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            own = f"tensorrt_llm/modeling_v2/{family_dir.name}/{unit_dir.name}/"
            allowed = (own, "tensorrt_llm/modeling_v2/common/")
            if unit_dir.name != "shared":
                allowed += (f"tensorrt_llm/modeling_v2/{family_dir.name}/shared/",)
            for path in _native_files(unit_dir):
                for include in _INCLUDE.findall(path.read_text()):
                    where = f'{path.relative_to(_REPO)}: "{include}"'
                    if include.startswith("tensorrt_llm/modeling_v2/"):
                        if not include.startswith(allowed):
                            problems.append(f"{where} reaches another target or family")
                    elif include.startswith("tensorrt_llm/"):
                        if not any(fnmatch.fnmatch(include, p) for p in surface):
                            problems.append(f"{where} is not in contractSurface.txt")
                    elif ".." in include:
                        problems.append(f"{where} leaves its directory")
    assert not problems, "\n".join(problems)


def test_contract_surface_patterns_name_existing_headers():
    for pattern in _contract_surface():
        assert any((_REPO / root).glob(pattern) for root in ("cpp/include", "cpp")), (
            f"contractSurface.txt pattern {pattern!r} matches no header"
        )


_RAW_REGISTRATION = re.compile(r"\bTORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\s*\(")


def test_native_code_registers_only_into_its_own_namespace():
    """Raw TORCH_LIBRARY* would let a file register into trtllm:: or a sibling's
    namespace; the TRTLLM_MV2_ macros register into the unit's own only."""
    offenders = [
        str(path.relative_to(_REPO))
        for path in _native_files(_NATIVE)
        if not path.is_relative_to(_NATIVE / "common")
        and _RAW_REGISTRATION.search(path.read_text())
    ]
    assert not offenders, (
        f"use TRTLLM_MV2_LIBRARY_FRAGMENT / TRTLLM_MV2_LIBRARY_IMPL, not TORCH_LIBRARY*: {offenders}"
    )


_OP_DEF = re.compile(r'\bm\.def\(\s*"(\w+)\(')


def test_every_native_op_has_a_fake():
    """fake_ops.py doubles as the Python side's list of a target's ops: the
    loader imports it, so an op without a fake is one torch.compile and fake
    tracing cannot see, and a fake without an op fails the load."""
    for family, target, _ in _targets():
        target_dir = _families()[family] / camel(target)
        ops = sorted(
            {op for p in _native_files(target_dir) for op in _OP_DEF.findall(p.read_text())}
        )
        if not ops:
            continue
        fake_ops = _MODELS / family / target / "fake_ops.py"
        assert fake_ops.is_file(), f"{family}/{target} defines {ops} but has no fake_ops.py"
        text = fake_ops.read_text()
        missing = [op for op in ops if f'::{op}"' not in text]
        assert not missing, f"{fake_ops.relative_to(_REPO)} has no fake for {missing}"


# --- What stock code may reach ------------------------------------------------

_STOCK_PY_REFERENCE = re.compile(r"modeling_v2[./]models\b|torch\.ops\.mv2_")
_STOCK_CPP_REFERENCE = re.compile(r'#\s*include\s*"tensorrt_llm/modeling_v2/')
_STOCK_CMAKE_REFERENCE = re.compile(r"modeling_v2/\w")


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


def test_stock_native_code_does_not_reference_targets():
    offenders = []
    for root in ("cpp/tensorrt_llm", "cpp/include", "cpp/tests"):
        for path in sorted((_REPO / root).rglob("*")):
            if not path.is_file() or path.is_relative_to(_NATIVE):
                continue
            # Skip generated cubin sources; they include nothing of interest.
            if path.suffix in _NATIVE_SUFFIXES and path.stat().st_size < (1 << 20):
                if _STOCK_CPP_REFERENCE.search(path.read_text(errors="ignore")):
                    offenders.append(str(path.relative_to(_REPO)))
            elif path.name == "CMakeLists.txt":
                if _STOCK_CMAKE_REFERENCE.search(path.read_text()):
                    offenders.append(str(path.relative_to(_REPO)))
    assert not offenders, (
        f"stock native code reaches into cpp/tensorrt_llm/modeling_v2: {offenders}"
    )


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


def test_family_packages_have_no_side_effects():
    """Native code loads per target; a family package that imported anything
    would load it for every target of the family, or for none of them."""
    for init in [_MODELS / "__init__.py", *sorted(_MODELS.glob("*/__init__.py"))]:
        body = ast.parse(init.read_text()).body
        for node in body:
            docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            assert docstring or future, (
                f"{init.relative_to(_REPO)}:{node.lineno} -- a family package holds a docstring only"
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


def test_test_lists_keep_family_tests_in_their_own_entries():
    """CBTS selects one family's tests by entry; an entry covering models/ from
    above would hide a family's unit tests inside the catalog's."""
    models = "unittest/_torch/modeling_v2/models"
    for yml, target, options in _test_db_entries():
        if target in ("unittest/_torch/modeling_v2", "unittest/_torch/modeling_v2/"):
            assert f"--ignore={models}" in options, (
                f"{yml}: {target} must carry --ignore={models}; list each family on its own"
            )
        assert target != models, f"{yml}: list each family under {models}/ on its own"


def test_family_unit_entries_point_at_tests():
    """A family entry naming a directory with no tests is a collection error in CI."""
    for yml, target, _ in _test_db_entries():
        if target.startswith("unittest/_torch/modeling_v2/models/"):
            path = _REPO / "tests" / target
            assert path.exists() and (path.is_file() or any(path.rglob("test_*.py"))), (
                f"{yml}: {target} names no test file or directory with tests"
            )


def test_families_with_native_code_have_test_entries():
    targets = {target for _, target, _ in _test_db_entries()}
    for family in _families():
        unit = f"unittest/_torch/modeling_v2/models/{family}"
        accuracy = f"accuracy/modeling_v2/test_{family}.py"
        assert any(t == unit or t.startswith((unit + "/", accuracy)) for t in targets), (
            f"family {family!r} has native code but no test-db entry under {unit} or {accuracy}"
        )


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
    """The loader and CBTS compute directory, namespace and architecture
    coverage in Python; the build computes them in modeling_v2.cmake."""
    names = ["fixture", "minimax_m3", "scale_add__sm_10x__default", "r1_0528_nvfp4__sm_10x__default"]
    script = tmp_path / "probe.cmake"
    lines = [f'include("{_CMAKE_MODULE}")']
    for name in names:
        lines += [
            f'_modeling_v2_camel(c "{name}")',
            f'_modeling_v2_ident(i "{name}")',
            f'message(STATUS "name {name} ${{c}} ${{i}}")',
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
        assert f"name {name} {camel(name)} {identifier(name)}" in output, output
    assert "archs 90a-real;100f-real;103a-real" in output, output
    # The family namespace the build registers is the loader's.
    assert torch_namespace("fixture", "scale_add__sm_10x__default") == (
        "mv2_fixture_scale_add_sm_10x_default"
    )
