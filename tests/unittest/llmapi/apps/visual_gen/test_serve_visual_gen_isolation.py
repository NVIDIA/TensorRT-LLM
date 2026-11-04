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
"""Import isolation of the VisualGen serving package, and where its tests live.

CBTS, the CI test selector in ``jenkins/scripts/cbts/``, runs only VisualGen test
entries for a pull request whose changed files are all under
``tensorrt_llm/serve/api/visual_gen/``. That is safe only while no other serving
role, CLI entry point or test outside a ``visual_gen/`` path depends on the
package. The checks here fail as soon as that stops being true:

- the static import graph of ``tensorrt_llm/``: the only references to the
  package from outside it are the edges in ``ALLOWED_EDGES`` (importing one of
  the VisualGen models from ``tensorrt_llm.serve.openai_protocol`` counts, since
  that re-export loads the package);
- at run time, no non-VisualGen serving role and no CLI entry point loads the
  package or the media helpers it uses; the VISUAL_GEN role does (the control
  that proves the probe sees them);
- every test module that exercises VisualGen serving lives under a path that
  contains ``visual_gen/``, or is listed in ``ALLOWED_NON_VG_TESTS`` with a
  reason;
- test-db entries that name a ``visual_gen`` directory keep the trailing slash
  the selector matches on.

This file lives under a ``visual_gen/`` path so that VisualGen-only pull
requests run it. Changes to its allowlists are reviewed by the runtime team and
the owners of ``jenkins/scripts/cbts``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest
import yaml

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.rules.visual_gen import _entry_is_vg

# The L0 CPU stage invokes registered files with `-m cpu_only`; the probes run in
# subprocesses and leave no threads behind to account for.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REPO_ROOT = Path(__file__).resolve().parents[5]
TESTS_ROOT = REPO_ROOT / "tests"
TEST_DB_DIR = TESTS_ROOT / "integration" / "test_lists" / "test-db"
THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()

PACKAGE = "tensorrt_llm.serve.api.visual_gen"
PACKAGE_DIR = "tensorrt_llm/serve/api/visual_gen"

# Modules only the VISUAL_GEN role may load: the serving package and the media
# helpers its handlers import at module level.
WATCHED_MODULES = (PACKAGE, "tensorrt_llm.media.encoding", "tensorrt_llm.media.tensor_payload")

# The VisualGen request and response models that tensorrt_llm.serve.openai_protocol
# re-exports lazily under their old names.
VISUAL_GEN_PROTOCOL_MODELS = frozenset(
    {
        "ImageGenerationRequest",
        "ImageEditRequest",
        "ImageObject",
        "ImageGenerationResponse",
        "MediaReferenceItem",
        "VideoGenerationRequest",
        "VideoJob",
        "VideoJobList",
    }
)
# The module that re-exports them lazily under their old names; importing one of them from
# here loads the VisualGen package, so such an import counts as a reference to it.
OPENAI_PROTOCOL_MODULE = "tensorrt_llm.serve.openai_protocol"

# Every reference the scans below look for contains one of these substrings: the
# package's dotted and slashed names, the role, or a VisualGen model's name. A file
# without any of them is skipped before parsing.
_SCAN_NEEDLES = ("visual_gen", "VISUAL_GEN", *sorted(VISUAL_GEN_PROTOCOL_MODELS))

EdgeKey = Tuple[str, str, str, str]  # (file, scope, kind, target)


def _openai_protocol_file() -> str:
    """The module that re-exports the VisualGen models: a module today, a package later."""
    as_package = "tensorrt_llm/serve/openai_protocol/__init__.py"
    if (REPO_ROOT / as_package).exists():
        return as_package
    return "tensorrt_llm/serve/openai_protocol.py"


# Every reference to the package from outside it, with the reason it is allowed.
# Kinds: "import" (an import statement, `if TYPE_CHECKING:` included, and an import
# of a VisualGen model from tensorrt_llm.serve.openai_protocol), "importlib"
# (importlib.import_module or __import__ with a literal name) and "string" (any
# other string literal naming a module of the package, such as a routing table).
ALLOWED_EDGES: Dict[EdgeKey, str] = {
    (
        "tensorrt_llm/serve/openai_server.py",
        "OpenAIServer.register_visual_gen_routes",
        "importlib",
        f"{PACKAGE}.api_router",
    ): "the composition root: the one place that loads the VisualGen router, for the VISUAL_GEN "
    "role only",
    (
        "tensorrt_llm/serve/_base/routing.py",
        "<module>",
        "string",
        f"{PACKAGE}.api_router",
    ): "the VisualGen entry of the routing table, which names router modules by dotted path",
    (
        _openai_protocol_file(),
        "<module>",
        "string",
        f"{PACKAGE}.protocol",
    ): "the lazy re-export of the VisualGen models under their old names: a name table, never "
    "an import",
    (
        "tensorrt_llm/serve/scripts/benchmark_visual_gen.py",
        "<module>",
        "import",
        f"{PACKAGE}.metrics",
    ): "the VisualGen benchmark client",
    (
        "tensorrt_llm/serve/scripts/benchmark_visual_gen.py",
        "<module>",
        "import",
        f"{PACKAGE}.protocol",
    ): "the VisualGen benchmark client",
}

# Test modules outside a visual_gen/ path that match a criterion below without
# exercising VisualGen serving, with the reason.
ALLOWED_NON_VG_TESTS: Dict[str, str] = {
    "tests/unittest/disaggregated/test_disagg_internal_auth.py": (
        "passes ServerRole.VISUAL_GEN as a role value to validate_subagent_affinity; starts no "
        "VisualGen serving"
    ),
    "tests/unittest/serve/_base/test_routing.py": (
        "checks that role_group_of maps ServerRole.VISUAL_GEN to its role group; starts no "
        "VisualGen serving"
    ),
}


# ---------------------------------------------------------------------------
# Shared AST helpers
# ---------------------------------------------------------------------------


def _is_in_package(module: str) -> bool:
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    """Ids of the string constants that are docstrings; documentation is not a reference."""
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _is_import_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "import_module"
    return isinstance(func, ast.Name) and func.id in ("import_module", "__import__")


def _literal_import_argument(node: ast.Call) -> Optional[ast.Constant]:
    """The literal module name of an importlib.import_module / __import__ call, if any."""
    if not _is_import_call(node.func) or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first
    return None


def _resolve_relative(package: str, level: int, module: Optional[str]) -> Optional[str]:
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(base) if base else None


def _module_name(file: str) -> str:
    """Dotted module name of a repo-relative ``.py`` path."""
    parts = file[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Static import graph (the first check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One reference from a module outside the package to a module inside it."""

    file: str  # repo-relative path of the referencing file
    scope: str  # enclosing definitions, dotted; "<module>" at module level
    kind: str  # "import", "importlib" or "string"
    target: str  # the referenced module
    lineno: int

    @property
    def key(self) -> EdgeKey:
        return (self.file, self.scope, self.kind, self.target)


class _EdgeCollector(ast.NodeVisitor):
    def __init__(self, file: str, package: str, skip: Set[int]) -> None:
        self._file = file
        self._package = package
        self._skip = skip  # ids of string constants that are not references
        self._scope: List[str] = []
        self.edges: List[Edge] = []

    def _add(self, kind: str, target: str, lineno: int) -> None:
        scope = ".".join(self._scope) or "<module>"
        self.edges.append(Edge(self._file, scope, kind, target, lineno))

    def _visit_definition(self, node) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_definition
    visit_AsyncFunctionDef = _visit_definition
    visit_ClassDef = _visit_definition

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_in_package(alias.name):
                self._add("import", alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve_relative(self._package, node.level, node.module)
        if module is None:
            return
        if _is_in_package(module):
            self._add("import", module, node.lineno)
            return
        for alias in node.names:
            # `from tensorrt_llm.serve.api import visual_gen` imports the package.
            if _is_in_package(f"{module}.{alias.name}"):
                self._add("import", f"{module}.{alias.name}", node.lineno)
            # The VisualGen models re-exported by openai_protocol load the package on access.
            elif alias.name in VISUAL_GEN_PROTOCOL_MODELS and module == OPENAI_PROTOCOL_MODULE:
                self._add("import", f"{module}.{alias.name}", node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        literal = _literal_import_argument(node)
        if literal is not None:
            if _is_in_package(literal.value):
                self._add("importlib", literal.value, node.lineno)
            self._skip.add(id(literal))  # recorded as this call's edge, not again as a string
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if id(node) in self._skip:
            return
        if isinstance(node.value, str) and _is_in_package(node.value):
            self._add("string", node.value, node.lineno)


def edges_in_source(source: str, file: str) -> List[Edge]:
    """Every reference to the package in the source of the module at ``file``."""
    tree = ast.parse(source)
    module = _module_name(file)
    package = module if file.endswith("__init__.py") else module.rpartition(".")[0]
    collector = _EdgeCollector(file, package, _docstring_nodes(tree))
    collector.visit(tree)
    return collector.edges


def edges_into_package(repo_root: Path) -> List[Edge]:
    """Every reference to the package from a module under ``tensorrt_llm/`` outside it."""
    edges: List[Edge] = []
    for path in sorted((repo_root / "tensorrt_llm").rglob("*.py")):
        file = path.relative_to(repo_root).as_posix()
        if file.startswith(PACKAGE_DIR + "/"):
            continue
        source = path.read_text(errors="replace")
        if not any(needle in source for needle in _SCAN_NEEDLES):  # a reference always contains one
            continue
        edges.extend(edges_in_source(source, file))
    return edges


def _unexpected_edge(edge: Edge) -> str:
    verb = "names" if edge.kind == "string" else "imports"
    return (
        f"{edge.file}:{edge.lineno} ({edge.scope}) {verb} `{edge.target}`. Code outside "
        f"{PACKAGE_DIR}/ must not import VisualGen serving: CBTS runs only VisualGen tests for "
        "changes under that package, so an LLM, embedding or disaggregated path that imports it "
        "would lose its tests. Reach VisualGen through the composition root's role-gated load. "
        f"If this edge is needed, add it to ALLOWED_EDGES in {THIS_FILE} with a reason; the "
        "runtime team and the owners of jenkins/scripts/cbts review that change."
    )


def _stale_edge(key: EdgeKey) -> str:
    file, scope, kind, target = key
    return (
        f"allowed edge {file} ({scope}, {kind}) `{target}` no longer exists; remove it from "
        f"ALLOWED_EDGES in {THIS_FILE}."
    )


def test_no_module_outside_the_package_imports_visual_gen_serving() -> None:
    found: Dict[EdgeKey, Edge] = {}
    for edge in edges_into_package(REPO_ROOT):
        found.setdefault(edge.key, edge)  # equal (file, scope, kind, target) count once

    problems = [
        _unexpected_edge(edge) for key, edge in sorted(found.items()) if key not in ALLOWED_EDGES
    ]
    problems += [_stale_edge(key) for key in ALLOWED_EDGES if key not in found]
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Runtime checks per role and per CLI entry point (the second check)
# ---------------------------------------------------------------------------

# Each probe runs in a fresh interpreter: the pytest process has already imported
# VisualGen modules through other tests, so sys.modules there proves nothing.
_PROBE_PRELUDE = """
import json
import sys

WATCHED = tuple(json.loads(sys.argv[1]))


def loaded():
    return sorted(
        name for name in sys.modules if any(name == w or name.startswith(w + ".") for w in WATCHED)
    )
"""

# Builds every non-VisualGen app through the fixture factory, one after another,
# and records the watched modules present after each build and its first request.
_ROLE_PROBE = (
    _PROBE_PRELUDE
    + """
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[2])  # tests/: the fixture factory and its fakes, nothing else
tmp_root = sys.argv[3]
steps = json.loads(sys.argv[4])

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from test_common.serve.app_factory import (
    AggConfig,
    CoordinatorConfig,
    DisaggConfig,
    isolated_serve_env,
)

configs = {
    "the default LLM role": ("openai_server", AggConfig()),
    "the CONTEXT role": ("openai_server", AggConfig(role=ServerRole.CONTEXT)),
    "the GENERATION role": ("openai_server", AggConfig(role=ServerRole.GENERATION)),
    "the MM_ENCODER role": ("openai_server", AggConfig(role=ServerRole.MM_ENCODER)),
    "the EMBEDDING role": ("openai_server", AggConfig(role=ServerRole.EMBEDDING)),
    "the disaggregated frontend with a local coordinator": ("disagg_server", DisaggConfig()),
    "the disaggregated frontend in fleet mode": ("disagg_server", DisaggConfig(fleet=True)),
    "the fleet coordinator": ("coordinator_server", CoordinatorConfig()),
}
report = {"importing the fixture factory": {"loaded": loaded(), "health": None}}
for step in steps:
    kind, config = configs[step]
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir, isolated_serve_env(Path(tmp_dir)) as env:
        if kind == "openai_server":
            app = env.build_openai_server(config).app
            client = env.client(app)
        elif kind == "disagg_server":
            # The disaggregated lifespan connects to its workers; /health answers without it.
            app = env.build_disagg_server(config).app
            client = env.client(app, lifespan=False)
        else:
            app = env.build_coordinator_server(config).app
            client = env.client(app, lifespan=False)
        # Every app serves /health; the request is sent unconditionally rather than after
        # finding the route in app.routes, where FastAPI 0.139 nests included routers. A
        # handler that imports VisualGen at request time only shows up after this call.
        status = client.get("/health").status_code
    report["starting " + step] = {"loaded": loaded(), "health": status}
print(json.dumps(report))
"""
)

# Runs every trtllm-serve subcommand's --help and trtllm-render --help.
_CLI_PROBE = (
    _PROBE_PRELUDE
    + """
import contextlib
import io

report = {}
from tensorrt_llm.commands.serve import main

report["importing tensorrt_llm.commands.serve"] = loaded()
with contextlib.redirect_stdout(io.StringIO()):
    main(["--help"], standalone_mode=False)
report["running trtllm-serve --help"] = loaded()
for name in sorted(main.commands):
    with contextlib.redirect_stdout(io.StringIO()):
        main([name, "--help"], standalone_mode=False)
    report["running trtllm-serve " + name + " --help"] = loaded()

from tensorrt_llm.serve.render.__main__ import main as render_main

with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()):
    render_main(["--help"])
report["running trtllm-render --help"] = loaded()
print(json.dumps(report))
"""
)

# The control: the VISUAL_GEN app must load the watched modules.
_VISUAL_GEN_PROBE = (
    _PROBE_PRELUDE
    + """
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[2])
tmp_root = sys.argv[3]

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from test_common.serve.app_factory import AggConfig, isolated_serve_env

with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir, isolated_serve_env(Path(tmp_dir)) as env:
    config = AggConfig(role=ServerRole.VISUAL_GEN, model_name="fake-visual-gen")
    app = env.build_openai_server(config).app
    status = env.client(app).get("/health").status_code
print(json.dumps({"health": status, "loaded": loaded()}))
"""
)

NON_VISUAL_GEN_STEPS = (
    "the default LLM role",
    "the CONTEXT role",
    "the GENERATION role",
    "the MM_ENCODER role",
    "the EMBEDDING role",
    "the disaggregated frontend with a local coordinator",
    "the disaggregated frontend in fleet mode",
    "the fleet coordinator",
)


def _run_in_fresh_interpreter(code: str, *args: str) -> dict:
    """Run ``code`` in a new interpreter and return the JSON it prints last."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _runtime_problem(step: str, module: str) -> str:
    return (
        f"{step[:1].upper()}{step[1:]} imported `{module}`, which only the VISUAL_GEN role may "
        f"load. CBTS runs only VisualGen tests for changes under {PACKAGE_DIR}/, so this path "
        "would no longer be tested by them. Find the import with "
        "test_no_module_outside_the_package_imports_visual_gen_serving (this file) or "
        "`python -X importtime`, and load VisualGen code only from the composition root's "
        "VISUAL_GEN branch."
    )


def test_non_visual_gen_roles_do_not_load_visual_gen_serving(tmp_path: Path) -> None:
    report = _run_in_fresh_interpreter(
        _ROLE_PROBE,
        json.dumps(WATCHED_MODULES),
        str(TESTS_ROOT),
        str(tmp_path),
        json.dumps(NON_VISUAL_GEN_STEPS),
    )
    # Every step ran: a probe that skipped a role would pass vacuously.
    expected_steps = (
        "importing the fixture factory",
        *("starting " + s for s in NON_VISUAL_GEN_STEPS),
    )
    assert tuple(report) == expected_steps
    # Every app answered its first request: the request path ran, not only construction.
    unanswered = [
        step for step in expected_steps[1:] if not isinstance(report[step]["health"], int)
    ]
    assert not unanswered, (
        f"no /health response after {unanswered}; the probe skipped the request path"
    )
    problems = [
        _runtime_problem(step, module)
        for step, result in report.items()
        for module in result["loaded"]
    ]
    assert not problems, "\n".join(problems)


def test_cli_entry_points_do_not_load_visual_gen_serving() -> None:
    report = _run_in_fresh_interpreter(_CLI_PROBE, json.dumps(WATCHED_MODULES))
    assert "running trtllm-serve serve --help" in report
    assert "running trtllm-render --help" in report
    problems = [
        _runtime_problem(step, module) for step, modules in report.items() for module in modules
    ]
    assert not problems, "\n".join(problems)


def test_visual_gen_role_loads_visual_gen_serving(tmp_path: Path) -> None:
    report = _run_in_fresh_interpreter(
        _VISUAL_GEN_PROBE, json.dumps(WATCHED_MODULES), str(TESTS_ROOT), str(tmp_path)
    )
    assert report["health"] == 200
    for module in (f"{PACKAGE}.api_router", "tensorrt_llm.media.encoding"):
        assert module in report["loaded"], (
            f"Building the VISUAL_GEN app did not load `{module}`: the probe the other tests in "
            "this file rely on no longer sees VisualGen serving. Fix the probe or the fixture "
            "factory before trusting them."
        )


# ---------------------------------------------------------------------------
# VisualGen serving tests live under visual_gen/ paths (the third check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    lineno: int
    criterion: str


def _last_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def visual_gen_serving_criteria(source: str) -> List[Match]:
    """Why a test module counts as a VisualGen serving test; empty when it does not.

    A test module exercises VisualGen serving when, at any scope, it imports the
    package or a submodule, names one of them in a string (a patch target), uses
    ``ServerRole.VISUAL_GEN`` or ``launch_visual_gen_server`` as a name or
    attribute (not inside a string), or imports one of the VisualGen models from
    ``tensorrt_llm.serve.openai_protocol``. Docstrings are not references.
    """
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    matches: List[Match] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_in_package(alias.name):
                    matches.append(Match(node.lineno, f"imports {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _is_in_package(node.module):
                matches.append(Match(node.lineno, f"imports {node.module}"))
                continue
            for alias in node.names:
                name = f"{node.module}.{alias.name}"
                if _is_in_package(name):
                    matches.append(Match(node.lineno, f"imports {name}"))
                elif (
                    node.module == OPENAI_PROTOCOL_MODULE
                    and alias.name in VISUAL_GEN_PROTOCOL_MODELS
                ):
                    matches.append(
                        Match(
                            node.lineno,
                            f"imports {alias.name} from tensorrt_llm.serve.openai_protocol",
                        )
                    )
        elif isinstance(node, ast.Constant):
            if id(node) in skip or not isinstance(node.value, str):
                continue
            if _is_in_package(node.value):
                matches.append(Match(node.lineno, f"names {node.value} in a string"))
        elif isinstance(node, ast.Attribute):
            if node.attr == "VISUAL_GEN" and _last_name(node.value) == "ServerRole":
                matches.append(Match(node.lineno, "uses ServerRole.VISUAL_GEN"))
            elif node.attr == "launch_visual_gen_server":
                matches.append(Match(node.lineno, "uses launch_visual_gen_server"))
        elif isinstance(node, ast.Name) and node.id == "launch_visual_gen_server":
            matches.append(Match(node.lineno, "uses launch_visual_gen_server"))
    return sorted(set(matches), key=lambda match: (match.lineno, match.criterion))


def is_test_module(path: Path) -> bool:
    """Pytest's default ``python_files``; the repository's pytest.ini files keep it."""
    return path.suffix == ".py" and (
        path.name.startswith("test_") or path.name.endswith("_test.py")
    )


def visual_gen_serving_tests(tests_root: Path, repo_root: Path) -> Dict[str, List[Match]]:
    """Repo-relative path -> matches, for every test module that exercises VisualGen serving.

    Helper modules (``conftest.py``, ``tests/test_common/**``, anything not named
    like a test) are not scanned: they run only through a test module, whose own
    path decides the selection.
    """
    found: Dict[str, List[Match]] = {}
    for path in sorted(tests_root.rglob("*.py")):
        if not is_test_module(path):
            continue
        source = path.read_text(errors="replace")
        if not any(needle in source for needle in _SCAN_NEEDLES):
            continue
        matches = visual_gen_serving_criteria(source)
        if matches:
            found[path.relative_to(repo_root).as_posix()] = matches
    return found


def _misplaced_test(path: str, match: Match) -> str:
    return (
        f"{path} exercises VisualGen serving ({match.criterion} at line {match.lineno}) but its "
        "path does not contain visual_gen/, so CBTS does not run it for VisualGen-only pull "
        "requests. Move it under a directory whose path contains visual_gen/ "
        "(tests/unittest/serve/api/visual_gen/ for serve tests, tests/unittest/_torch/visual_gen/ "
        "for runtime tests) and update its entry in tests/integration/test_lists/test-db/l0_cpu.yml, "
        "or, if it does not exercise VisualGen serving, add it to ALLOWED_NON_VG_TESTS in "
        f"{THIS_FILE} with a reason."
    )


def test_visual_gen_serving_tests_live_under_visual_gen_paths() -> None:
    found = visual_gen_serving_tests(TESTS_ROOT, REPO_ROOT)
    # The scan sees the VisualGen serving tests that do exist; otherwise it is broken.
    assert any(_entry_is_vg(path) for path in found), "no VisualGen serving test found at all"

    problems = []
    for path, matches in sorted(found.items()):
        if _entry_is_vg(path) or path in ALLOWED_NON_VG_TESTS:
            continue
        problems.append(_misplaced_test(path, matches[0]))
    for path in ALLOWED_NON_VG_TESTS:
        if path not in found:
            problems.append(
                f"{path} is listed in ALLOWED_NON_VG_TESTS but matches no criterion of a "
                f"VisualGen serving test (or no longer exists); remove it from "
                f"ALLOWED_NON_VG_TESTS in {THIS_FILE}."
            )
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Test-db entries keep the trailing slash (the fourth check)
# ---------------------------------------------------------------------------


def iter_test_db_entries(yml: Path) -> Iterator[str]:
    """The ``tests:`` entries of every block of one test-db YAML file."""
    data = yaml.safe_load(yml.read_text()) or {}
    for blocks in data.values():
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for entry in block.get("tests") or []:
                if isinstance(entry, str):
                    yield entry


def _entry_path(entry: str) -> str:
    """The test path of an entry: the text before the first space and before ``::``."""
    words = entry.split()
    return words[0].split("::")[0] if words else ""


def names_visual_gen_directory(entry: str) -> bool:
    """The entry's test path has a component named exactly ``visual_gen``."""
    return "visual_gen" in _entry_path(entry).split("/")


def with_trailing_slash(entry: str) -> str:
    """The entry with ``/`` appended to its test path, options and node ids kept."""
    path = _entry_path(entry)
    return entry.replace(path, path + "/", 1)


def test_visual_gen_test_db_entries_keep_the_trailing_slash() -> None:
    problems = []
    checked = 0
    for yml in sorted(TEST_DB_DIR.glob("*.yml")):
        for entry in iter_test_db_entries(yml):
            if not names_visual_gen_directory(entry):
                continue
            checked += 1
            if _entry_is_vg(entry):
                continue
            problems.append(
                f"{yml.relative_to(REPO_ROOT).as_posix()}: test-db entry `{entry}` names a "
                "visual_gen directory without the trailing slash. CBTS keeps an entry for "
                "VisualGen-only pull requests only if its text contains visual_gen/ "
                "(_VG_ENTRY_PATTERNS in jenkins/scripts/cbts/cbts/rules/visual_gen.py). Write "
                f"`{with_trailing_slash(entry)}`."
            )
    assert checked, "no test-db entry names a visual_gen directory; the scan is broken"
    assert not problems, "\n".join(problems)
