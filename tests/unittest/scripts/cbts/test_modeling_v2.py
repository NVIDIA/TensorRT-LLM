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
"""ModelingV2Rule selects the tests of the targets and families a change can reach."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, YAMLIndex
from cbts.rules import modeling_v2, modeling_v2_source
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule

pytestmark = pytest.mark.cpu_only

_SRC = "tensorrt_llm/_torch/_experimental/modeling_v2"
_ABS = "tensorrt_llm._torch._experimental.modeling_v2"
_UNIT = "unittest/_torch/modeling_v2"
_ACC = "accuracy/modeling_v2"


def _routing(*targets: str) -> str:
    """A `routing.py` whose route returns a class from each of `targets`."""
    returns = "".join(
        f"    if config == {target!r}:\n"
        f"        from .{target}.modeling import {target.upper()}\n\n"
        f"        return {target.upper()}\n"
        for target in targets
    )
    return (
        f'from ..._router_index import routes\n\n\n@routes("Arch")\ndef route(config):\n{returns}'
    )


# Family `fam` has targets a and b, family `other` has target c: the
# directories their routing.py returns classes from.
_SOURCES = {
    "__init__.py": "from ._router_index import NoTarget\n",
    "_router_index.py": "def families():\n    from . import models\n",
    "_native.py": "import importlib\n\n\ndef load(m):\n    importlib.import_module(m)\n",
    "catalog/__init__.py": "",
    "catalog/_op.py": "",
    "catalog/norm/__init__.py": "",
    "catalog/norm/rmsnorm.py": "from .._op import OpWrapper\n",
    "catalog/gemm/__init__.py": "",
    "catalog/gemm/mm.py": "from .._op import OpWrapper\n",
    "catalog/moe/__init__.py": "",
    "catalog/moe/runner.py": "",
    "catalog/quant/__init__.py": "",
    "catalog/quant/fp4.py": "from .._op import OpWrapper\n",
    "models/__init__.py": "from .fam import routing\nfrom .other import routing\n",
    "models/fam/__init__.py": "",
    "models/fam/routing.py": _routing("a", "b"),
    "models/fam/shapes.py": "",
    "models/fam/gemm_lib.py": "",
    # Family code no target imports; only a unit test outside the family does.
    "models/fam/reference.py": "",
    "models/fam/a/__init__.py": "",
    # A relative import of family code, and an absolute import of a catalog
    # entry inside a function.
    "models/fam/a/modeling.py": (
        "from ..ops.norm import fused\n\n\ndef f():\n"
        f"    from {_ABS}.catalog.norm.rmsnorm import rmsnorm\n"
    ),
    "models/fam/a/csrc/k.cu": "",
    "models/fam/ops/__init__.py": "",
    "models/fam/ops/norm/__init__.py": "",
    # Three dots from models/fam/ops/norm reach models/fam.
    "models/fam/ops/norm/fused.py": "from ...shapes import SHAPE\n",
    "models/fam/b/__init__.py": "",
    "models/fam/b/modeling.py": (
        "import importlib\n\n"
        f"import {_ABS}.models.fam.pkg.sub\n\n"
        "from ..kernels.loader import load\n\n"
        '_GEMM = importlib.import_module("..gemm_lib", __package__)\n'
    ),
    "models/fam/kernels/__init__.py": "",
    "models/fam/kernels/loader.py": (
        "import importlib\n\n\ndef load(name):\n"
        '    return importlib.import_module(f".{name}", __package__)\n'
    ),
    "models/fam/kernels/moe.py": "",
    "models/fam/pkg/__init__.py": "",
    "models/fam/pkg/sub.py": "",
    "models/fam/tools/__init__.py": "",
    "models/fam/tools/dump.py": "",
    # Holds a modeling file but no route returns from it, so it is family code.
    "models/fam/wip/modeling.py": "",
    "models/fam/csrc/k.cu": "",
    "models/fam/data.json": "",
    "models/other/__init__.py": "",
    "models/other/routing.py": _routing("c"),
    "models/other/c/__init__.py": "",
    "models/other/c/modeling.py": f"from {_ABS}.catalog.gemm import mm\n",
}
_TEST_FILES = {
    "integration/defs/accuracy/modeling_v2/fam/a/__init__.py": "",
    "integration/defs/accuracy/modeling_v2/fam/b/__init__.py": "",
    "integration/defs/accuracy/modeling_v2/other/c/__init__.py": "",
    f"{_UNIT}/norm/test_rmsnorm.py": "",
    f"{_UNIT}/gemm/test_mm.py": "",
    f"{_UNIT}/quant/test_fp4.py": "",
    # Unit tests outside the mirror directory that import catalog code: one
    # directly, one through a helper it finds by bare name on sys.path.
    f"{_UNIT}/graph/test_graph_fp4.py": f"from {_ABS}.catalog.quant.fp4 import fp4\n",
    f"{_UNIT}/_fp4_helper.py": f"from {_ABS}.catalog.quant import fp4\n",
    f"{_UNIT}/activation/test_act_fp4.py": "import _fp4_helper\n",
    f"{_UNIT}/models/fam/test_routing.py": "",
    f"{_UNIT}/models/fam/_helpers.py": f"from {_ABS}.models.fam import reference\n",
    f"{_UNIT}/graph/test_graph_reference.py": "from _helpers import reference\n",
    f"{_UNIT}/models/fam/ops/test_ops.py": "",
    f"{_UNIT}/models/fam/ops/norm/test_fused.py": "",
    f"{_UNIT}/models/fam/a/test_a.py": "",
    f"{_UNIT}/models/fam/b/test_b.py": "",
    f"{_UNIT}/models/other/c/test_c.py": "",
}

_TREE = f"{_UNIT}/test_modeling_v2_routing.py"
_A = {f"{_UNIT}/models/fam/a/test_a.py", f"{_ACC}/fam/a/test_accuracy.py::TestA::test_gsm8k"}
_B = {f"{_UNIT}/models/fam/b/test_b.py", f"{_ACC}/fam/b/test_accuracy.py::TestB::test_gsm8k[tp4]"}
_C = {f"{_UNIT}/models/other/c/test_c.py", f"{_ACC}/other/c/test_accuracy.py::TestC::test_gsm8k"}
_FUSED = f"{_UNIT}/models/fam/ops/norm/test_fused.py"
# The family-level unit tests: test files directly in the family directory and
# the directories that are not targets.
_FAMILY_LEVEL = {
    f"{_UNIT}/models/fam/test_routing.py",
    f"{_UNIT}/models/fam/ops/test_ops.py",
    _FUSED,
}
_FAMILY = _A | _B | _FAMILY_LEVEL
_NORM = f"{_UNIT}/norm/test_rmsnorm.py"
_GEMM = f"{_UNIT}/gemm/test_mm.py"
_FP4 = {
    f"{_UNIT}/quant/test_fp4.py",
    f"{_UNIT}/graph/test_graph_fp4.py",
    f"{_UNIT}/activation/test_act_fp4.py",
}
_REFERENCE = f"{_UNIT}/graph/test_graph_reference.py"
_EVERYTHING = _FAMILY | _C | _FP4 | {_NORM, _GEMM, _REFERENCE, f"{_UNIT}/moe/test_runner.py"}


def _rule(repo: Path) -> ModelingV2Rule:
    for path, text in _SOURCES.items():
        (repo / _SRC / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / _SRC / path).write_text(text)
    for path, text in _TEST_FILES.items():
        (repo / "tests" / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / "tests" / path).write_text(text)
    index = YAMLIndex()
    index.blocks = [Block("l0_b300", 0, {}, sorted(_EVERYTHING | {_TREE}))]
    return ModelingV2Rule(index, {}, repo_root=repo)


def _kept(rule: ModelingV2Rule, *changed: str) -> set[str]:
    result = rule.apply(PRInputs(changed_files=list(changed), diffs={}))
    return {entry for waives in result.block_filters[("l0_b300", 0)].values() for entry in waives}


@pytest.mark.parametrize(
    ("changed", "selected"),
    (
        pytest.param(f"{_SRC}/models/fam/a/modeling.py", _A, id="target-file-runs-its-target"),
        pytest.param(f"{_SRC}/models/fam/a/csrc/k.cu", _A, id="target-native-runs-its-target"),
        pytest.param(
            f"{_SRC}/models/fam/ops/norm/fused.py",
            {_FUSED} | _A,
            id="family-code-runs-its-mirror-unit-dir-and-importing-targets",
        ),
        pytest.param(
            f"{_SRC}/models/fam/shapes.py",
            _FAMILY_LEVEL | _A,
            id="family-code-reached-transitively-through-a-relative-import",
        ),
        pytest.param(
            f"{_SRC}/models/fam/gemm_lib.py",
            _FAMILY_LEVEL | _B,
            id="constant-import-module-counts-as-an-import",
        ),
        pytest.param(
            f"{_SRC}/models/fam/kernels/moe.py",
            _FAMILY_LEVEL | _B,
            id="computed-import-module-imports-its-whole-directory",
        ),
        pytest.param(
            f"{_SRC}/models/fam/pkg/__init__.py",
            _FAMILY_LEVEL | _B,
            id="importing-a-submodule-runs-its-package-init",
        ),
        pytest.param(
            f"{_SRC}/models/fam/__init__.py",
            _FAMILY | {_REFERENCE},
            id="family-init-runs-the-family-and-unit-tests-importing-it",
        ),
        pytest.param(f"{_SRC}/models/fam/routing.py", _FAMILY, id="routing-runs-the-family"),
        pytest.param(
            f"{_SRC}/models/fam/tools/dump.py", _FAMILY, id="family-code-no-target-imports"
        ),
        pytest.param(
            f"{_SRC}/models/fam/reference.py",
            _FAMILY | {_REFERENCE},
            id="family-code-runs-unit-tests-importing-it-through-a-helper",
        ),
        pytest.param(f"{_SRC}/models/fam/wip/modeling.py", _FAMILY, id="unrouted-dir-is-family"),
        pytest.param(f"{_SRC}/models/fam/csrc/k.cu", _FAMILY, id="family-native-runs-the-family"),
        pytest.param(f"{_SRC}/models/fam/data.json", _FAMILY, id="family-data-runs-the-family"),
        pytest.param(
            f"{_SRC}/catalog/norm/rmsnorm.py",
            {_NORM} | _A,
            id="catalog-entry-runs-its-category-and-importing-targets",
        ),
        pytest.param(
            f"{_SRC}/catalog/quant/fp4.py",
            _FP4,
            id="catalog-entry-runs-unit-tests-importing-it-directly-or-through-a-helper",
        ),
        pytest.param(
            f"{_SRC}/catalog/gemm/__init__.py",
            {_GEMM} | _C,
            id="catalog-category-init-runs-its-category-and-importing-targets",
        ),
        pytest.param(
            f"{_SRC}/catalog/moe/runner.py",
            _EVERYTHING,
            id="catalog-category-without-unit-dir-runs-everything",
        ),
        *(
            pytest.param(path, _EVERYTHING, id=f"{path.rpartition('/')[2]}-runs-everything")
            for path in (
                f"{_SRC}/catalog/_op.py",
                f"{_SRC}/catalog/__init__.py",
                f"{_SRC}/_router_index.py",
                f"{_SRC}/_native.py",
                f"{_SRC}/__init__.py",
                f"{_SRC}/models/__init__.py",
                "cpp/tensorrt_llm/modeling_v2/common/torchLibrary.h",
                "cpp/cmake/modules/modeling_v2.cmake",
            )
        ),
    ),
)
def test_change_selects_what_it_reaches(tmp_path: Path, changed: str, selected: set[str]) -> None:
    assert _kept(_rule(tmp_path), changed) == selected | {_TREE}


def test_changes_union(tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    target_and_family = (f"{_SRC}/models/fam/a/modeling.py", f"{_SRC}/models/other/routing.py")
    assert _kept(rule, *target_and_family) == _A | _C | {_TREE}
    target_and_shared = (f"{_SRC}/models/fam/a/modeling.py", f"{_SRC}/catalog/_op.py")
    assert _kept(rule, *target_and_shared) == _EVERYTHING | {_TREE}


@pytest.mark.parametrize(
    ("deleted", "selected"),
    (
        ("catalog/norm/rmsnorm.py", {_NORM} | _A),
        ("models/fam/shapes.py", _FAMILY_LEVEL | _A),
        ("models/fam/kernels/moe.py", _FAMILY_LEVEL | _B),
    ),
    ids=(
        "deleted-catalog-entry-runs-its-importers",
        "deleted-family-code-runs-its-importers",
        "deleted-family-code-under-a-computed-import-runs-its-importers",
    ),
)
def test_deleted_module_runs_what_imported_it(
    tmp_path: Path, deleted: str, selected: set[str]
) -> None:
    """CI reads the PR head, where a deleted file is gone but its importers remain."""
    rule = _rule(tmp_path)
    (tmp_path / _SRC / deleted).unlink()
    assert _kept(rule, f"{_SRC}/{deleted}") == selected | {_TREE}


def test_moved_module_runs_what_imported_either_path(tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    (tmp_path / _SRC / "catalog/norm/rmsnorm.py").rename(
        tmp_path / _SRC / "catalog/gemm/rmsnorm.py"
    )
    moved = (f"{_SRC}/catalog/norm/rmsnorm.py", f"{_SRC}/catalog/gemm/rmsnorm.py")
    assert _kept(rule, *moved) == {_NORM, _GEMM} | _A | {_TREE}


def test_target_depends_on_its_family_init(tmp_path: Path) -> None:
    """Target c imports only the catalog, yet importing it runs models/other/__init__.py."""
    rule = _rule(tmp_path)
    for path, text in (
        ("models/other/routing.py", _routing("c", "e")),
        ("models/other/shared.py", ""),
        ("models/other/e/__init__.py", ""),
        ("models/other/e/modeling.py", "from ..shared import X\n"),
    ):
        (tmp_path / _SRC / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / _SRC / path).write_text(text)
    (tmp_path / "tests/integration/defs" / _ACC / "other/e").mkdir(parents=True)
    e = {f"{_UNIT}/models/other/e/test_e.py", f"{_ACC}/other/e/test_accuracy.py::TestE"}
    rule.yaml_index.blocks[0].tests.extend(sorted(e))
    assert _kept(rule, f"{_SRC}/models/other/__init__.py") == _C | e | {_TREE}


def test_aliased_import_module_counts_as_an_import(tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    for path, text in (
        ("models/fam/kern_a.py", ""),
        ("models/fam/kern_b.py", ""),
        (
            "models/fam/b/loaders.py",
            "import importlib as il\nfrom importlib import import_module as im\n\n"
            '_A = im("..kern_a", __package__)\n'
            f'_B = il.import_module("{_ABS}.models.fam.kern_b")\n',
        ),
    ):
        (tmp_path / _SRC / path).write_text(text)
    for kernel in ("kern_a", "kern_b"):
        assert _kept(rule, f"{_SRC}/models/fam/{kernel}.py") == _FAMILY_LEVEL | _B | {_TREE}


def test_a_rank_job_test_reaches_its_launcher(tmp_path: Path) -> None:
    """A multi-GPU test reaches the code its `_<entry>_op_matrix.py` launcher imports."""
    _rule(tmp_path)
    comm = f"{_UNIT}/models/fam/ops/comm"
    test = f"{comm}/test_fused_op_matrix.py"
    for path, text in (
        (test, 'import _rank_job\n\n\ndef test_fused():\n    _rank_job.run("fam_fused", 4)\n'),
        (f"{comm}/_fam_fused_op_matrix.py", f"from {_ABS}.models.fam.ops.norm import fused\n"),
    ):
        (tmp_path / "tests" / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "tests" / path).write_text(text)
    index = YAMLIndex()
    index.blocks = [Block("l0_b300", 0, {}, sorted(_EVERYTHING | {_TREE, test}))]
    rule = ModelingV2Rule(index, {}, repo_root=tmp_path)
    assert test in _kept(rule, f"{_SRC}/models/fam/ops/norm/fused.py")


def test_targets_come_from_routing(tmp_path: Path) -> None:
    """An accuracy directory neither makes nor is needed for a target."""
    rule = _rule(tmp_path)
    accuracy = tmp_path / "tests/integration/defs" / _ACC
    (accuracy / "fam/wip").mkdir()
    (accuracy / "fam/a/__init__.py").unlink()
    (accuracy / "fam/a").rmdir()
    assert _kept(rule, f"{_SRC}/models/fam/wip/modeling.py") == _FAMILY | {_TREE}
    assert _kept(rule, f"{_SRC}/models/fam/a/modeling.py") == _A | {_TREE}


@pytest.mark.parametrize(
    ("routing", "catalog"),
    (
        (None, {_NORM} | _FAMILY),
        ("def route(:\n", _EVERYTHING),
        (_routing("a", "b") + "    return make()\n", {_NORM} | _FAMILY),
        ("def route(config):\n    from .a.modeling import A\n\n    return A\n", {_NORM} | _FAMILY),
    ),
    ids=("missing", "unparsable", "unknown-return", "no-routes-function"),
)
def test_unreadable_routing_widens_to_the_family(
    tmp_path: Path, routing: str | None, catalog: set[str]
) -> None:
    """A family whose targets are unknown is tested whole, also through its importers."""
    rule = _rule(tmp_path)
    path = tmp_path / _SRC / "models/fam/routing.py"
    if routing is None:
        path.unlink()
    else:
        path.write_text(routing)
    assert _kept(rule, f"{_SRC}/models/fam/a/modeling.py") == _FAMILY | {_TREE}
    assert _kept(rule, f"{_SRC}/catalog/norm/rmsnorm.py") == catalog | {_TREE}
    assert _kept(rule, f"{_SRC}/models/other/c/modeling.py") == _C | {_TREE}


@pytest.mark.parametrize(
    ("routing", "targets"),
    (
        (
            """
            from ..._router_index import routes
            from .b import modeling as b

            @routes("Arch")
            def route(config):
                def fallback():
                    return config

                if config:
                    from .a.modeling import A as Picked

                    return Picked
                from .b.sub.modeling import B
                return B
            """,
            {"a", "b"},
        ),
        ('@router.routes("Arch")\ndef route(config):\n    from .a import A\n    return A\n', {"a"}),
        ("@routes('Arch')\ndef route(config):\n    return make(config)\n", None),
        ("@routes('Arch')\ndef route(config):\n    return\n", None),
        (
            "from ..a.m import A\n@routes('Arch')\ndef route(c):\n    from .a.m import A\n    return A\n",
            None,
        ),
        (
            "from .b.m import A\n@routes('Arch')\ndef route(c):\n    from .a.m import A\n    return A\n",
            None,
        ),
        ("def route(config):\n    from .a.modeling import A\n    return A\n", None),
    ),
    ids=(
        "nested-and-aliased",
        "attribute-decorator",
        "call",
        "bare-return",
        "name-also-bound-elsewhere",
        "name-bound-to-two-directories",
        "no-routes-function",
    ),
)
def test_routed_targets(routing: str, targets: set[str] | None) -> None:
    assert modeling_v2_source.routed_targets(ast.parse(dedent(routing))) == targets


def test_ci_selection_check_reads_imports_with_the_rule_parser() -> None:
    """The CPU check of the tree and the rule share one reading of imports and routing."""
    repo = Path(__file__).resolve().parents[4]
    path = repo / "tests/unittest/_torch/modeling_v2/test_modeling_v2_ci_selection.py"
    spec = importlib.util.spec_from_file_location("modeling_v2_ci_selection", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    assert Path(check._source.__file__).resolve() == Path(modeling_v2_source.__file__).resolve()
    assert modeling_v2.imported_names is modeling_v2_source.imported_names
    assert modeling_v2.routed_targets is modeling_v2_source.routed_targets


def test_docs_are_not_claimed(tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    assert rule.apply(PRInputs(changed_files=[f"{_SRC}/models/fam/a/README.md"], diffs={})) is None


@pytest.mark.parametrize(
    ("changed", "selected"),
    (
        (f"{_SRC}/models/fam/a/modeling.py", _A),
        (f"{_SRC}/models/fam/shapes.py", _FAMILY),
        (f"{_SRC}/catalog/norm/rmsnorm.py", _EVERYTHING),
    ),
    ids=("target-needs-no-imports", "family-code-runs-the-family", "catalog-runs-everything"),
)
def test_unparsable_file_widens_import_based_selection(
    tmp_path: Path, changed: str, selected: set[str]
) -> None:
    rule = _rule(tmp_path)
    (tmp_path / _SRC / "models/other/broken.py").write_text("def broken(:\n")
    assert _kept(rule, changed) == selected | {_TREE}


def test_tree_is_parsed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _rule(tmp_path)
    parsed = []
    parse = ast.parse
    monkeypatch.setattr(
        modeling_v2.ast, "parse", lambda *a, **k: parsed.append(1) or parse(*a, **k)
    )
    # The catalog entry builds the import graph before the targets of `other` are read.
    _kept(rule, f"{_SRC}/models/fam/shapes.py", f"{_SRC}/catalog/norm/rmsnorm.py")
    _kept(rule, f"{_SRC}/models/fam/gemm_lib.py", f"{_SRC}/models/other/c/modeling.py")
    unit_tests = [path for path in _TEST_FILES if path.startswith(_UNIT)]
    assert len(parsed) == sum(path.endswith(".py") for path in _SOURCES) + len(unit_tests)


@pytest.fixture(scope="module")
def repository_cpu_rule() -> ModelingV2Rule:
    repo = Path(__file__).resolve().parents[4]
    index = YAMLIndex.load(repo / "tests/integration/test_lists/test-db")
    index.blocks = [block for block in index.blocks if block.yaml_stem == "l0_cpu"]
    return ModelingV2Rule(index, {}, repo_root=repo)


@pytest.mark.parametrize(
    "family", ["deepseek_v3", "gpt_oss", "glm_moe_dsa", "kimi_k3_vl", "qwen4_exp"]
)
def test_repository_cpu_selection_keeps_model_tests_in_their_family(
    repository_cpu_rule: ModelingV2Rule, family: str
) -> None:
    result = repository_cpu_rule.apply(
        PRInputs(changed_files=[f"{_SRC}/models/{family}/routing.py"], diffs={})
    )
    kept = {
        entry
        for blocks in result.block_filters.values()
        for entries in blocks.values()
        for entry in entries
    }
    expected_family = f"{_UNIT}/models/{family}/"
    family_tests = {entry for entry in kept if entry.startswith(f"{_UNIT}/models/")}
    assert family_tests
    assert all(entry.startswith(expected_family) for entry in family_tests), family_tests
    assert f"{_UNIT}/test_modeling_v2_routing.py" in kept
    expected_tests = {
        entry
        for block in repository_cpu_rule.yaml_index.blocks
        for entry in block.tests
        if entry.startswith(expected_family)
    }
    assert expected_tests <= kept


def test_repository_target_change_runs_only_that_target(
    repository_cpu_rule: ModelingV2Rule,
) -> None:
    target = "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4"
    assert repository_cpu_rule._prefixes_for(f"{_SRC}/models/{target}/modeling.py") == {
        f"{_UNIT}/models/{target}",
        f"{_ACC}/{target}",
    }


def test_repository_catalog_change_runs_unit_tests_importing_it(
    repository_cpu_rule: ModelingV2Rule,
) -> None:
    selected = repository_cpu_rule._prefixes_for(f"{_SRC}/catalog/quantization/fp4_quantize.py")
    assert {
        f"{_UNIT}/quantization",
        f"{_UNIT}/gemm/test_modeling_v2_nvfp4_gemm.py",
    } <= selected
