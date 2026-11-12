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
"""Import directions between serving layers, written as tests.

Each pull request that creates a package under ``tensorrt_llm/serve`` adds its
contract here. A contract forbids imports by module prefix; a baseline allowance
records the imports that existed when the contract was written and only shrinks.
"""

import ast
import textwrap
from pathlib import Path

import pytest
from test_common.serve.layering import (
    Contract,
    build_import_graph,
    check_contracts,
    stale_allowances,
)

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

CONTRACTS = (
    Contract(
        name="render core has no HTTP or executor dependency",
        importers=(
            "tensorrt_llm.serve.render.chat",
            "tensorrt_llm.serve.render.resources",
            "tensorrt_llm.serve.render.fingerprint",
            "tensorrt_llm.serve.render.prompt_types",
        ),
        forbidden=("fastapi", "starlette", "tensorrt_llm.executor"),
        fix=(
            "Keep HTTP code in serve/render/api_router.py or _http.py and pass engine data "
            "in through RenderResources."
        ),
    ),
    Contract(
        name="gRPC frontends do not reach into serve",
        importers=("tensorrt_llm.grpc",),
        forbidden=("tensorrt_llm.serve",),
        allowed=frozenset(
            {
                (
                    "tensorrt_llm.grpc.openengine.server",
                    "tensorrt_llm.serve._frontend_processes",
                    name,
                )
                for name in (
                    "FrontendStartupCancelled",
                    "_init_multi_frontend_mode",
                    "_signal_frontend_ready",
                    "_spawn_attached_frontends",
                    "_terminate_attached_frontends",
                )
            }
            | {
                (
                    "tensorrt_llm.grpc.openengine.server",
                    "tensorrt_llm.serve._perf_metrics_config",
                    "resolve_perf_metrics_config",
                )
            }
        ),
        fix=(
            "gRPC frontends use serving code only through public entry points; add a public "
            "API in tensorrt_llm/llmapi instead. The baseline only shrinks: delete an "
            "allowance when its import goes away."
        ),
    ),
    Contract(
        name="serve/_base/ does not import serve/api/",
        importers=("tensorrt_llm.serve._base",),
        forbidden=("tensorrt_llm.serve.api",),
        fix=(
            "serve/_base/ is shared infrastructure and must not depend on endpoint "
            "packages; pass what it needs in from the composition root."
        ),
    ),
    Contract(
        name="serve/api/ and serve/_base/ do not import the composition root",
        importers=("tensorrt_llm.serve.api", "tensorrt_llm.serve._base"),
        forbidden=("tensorrt_llm.serve.openai_server", "tensorrt_llm.serve.app"),
        fix=(
            "Endpoint packages and serve/_base/ must not import the composition root "
            "(serve/openai_server.py or serve/app.py); the root imports them and injects "
            "what they need. Move the shared helper to serve/_base/ instead."
        ),
    ),
)


@pytest.fixture(scope="module")
def edges():
    return build_import_graph(REPO_ROOT / "tensorrt_llm", "tensorrt_llm")


def test_contracts_hold(edges):
    assert check_contracts(edges, CONTRACTS) == []


def test_no_stale_allowances(edges):
    assert stale_allowances(edges, CONTRACTS) == []


_API_PREFIX = "tensorrt_llm.serve.api"


def _api_package_of(module: str):
    """The endpoint package of a module under ``tensorrt_llm.serve.api``.

    The package is the first two components below ``api`` for ``openai`` and
    ``ops`` (``openai.chat``, ``ops.system``) and the first one otherwise
    (``anthropic``, ``visual_gen``).
    """
    below = module.split(".")[3:]
    if below and below[0] in ("openai", "ops"):
        return tuple(below[:2])
    return tuple(below[:1])


def _under_api(module: str) -> bool:
    return module == _API_PREFIX or module.startswith(_API_PREFIX + ".")


def test_api_packages_do_not_import_each_other(edges):
    problems = [
        f"serve/api/ packages do not import each other: {edge.importer} (line {edge.lineno}) "
        f"imports `{edge.imported}`. Share code through serve/_base/ or inject it from the "
        "composition root."
        for edge in edges
        if _under_api(edge.importer)
        and _under_api(edge.imported)
        and not edge.type_checking
        and _api_package_of(edge.importer) != _api_package_of(edge.imported)
    ]
    assert problems == []


def test_api_init_imports_nothing():
    tree = ast.parse((REPO_ROOT / "tensorrt_llm" / "serve" / "api" / "__init__.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports == [], (
        "serve/api/__init__.py imports nothing: attach_routers imports routers by role; an "
        "import here would load them for every role and break the VisualGen CI selection."
    )


# ---------------------------------------------------------------------------
# Self-test on a synthetic package: the graph must see every import form.
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_package(tmp_path):
    pkg = tmp_path / "layerpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "inner.py").write_text("VALUE = 1\n")
    (pkg / "top.py").write_text(
        textwrap.dedent(
            """
            from typing import TYPE_CHECKING

            import os
            from json import dumps, loads

            from .inner import VALUE

            if TYPE_CHECKING:
                import socket

            def late():
                import importlib

                importlib.import_module("collections.abc")
                from ..sibling import thing  # noqa: F401
                return VALUE
            """
        )
    )
    return pkg


def test_import_graph_records_every_form(synthetic_package):
    edges = build_import_graph(synthetic_package, "layerpkg")
    by_imported = {edge.imported: edge for edge in edges}

    assert by_imported["os"].names == ()
    assert not by_imported["os"].in_function and not by_imported["os"].type_checking
    assert by_imported["json"].names == ("dumps", "loads")
    assert by_imported["layerpkg.inner"].names == ("VALUE",)  # relative import resolved
    assert by_imported["socket"].type_checking
    assert by_imported["importlib"].in_function
    assert by_imported["collections.abc"].names == ()  # importlib.import_module literal
    assert by_imported["collections.abc"].in_function
    assert by_imported["sibling"].names == ("thing",)  # level-2 relative from layerpkg.top


def test_contract_violation_and_allowance(synthetic_package):
    edges = build_import_graph(synthetic_package, "layerpkg")
    contract = Contract(
        name="top does not import json",
        importers=("layerpkg.top",),
        forbidden=("json",),
        fix="Use the serializer helper instead.",
    )
    problems = check_contracts(edges, [contract])
    assert len(problems) == 2  # one per from-import name
    assert "contract 'top does not import json'" in problems[0]
    assert "Use the serializer helper instead." in problems[0]

    allowed = Contract(
        name="top does not import json",
        importers=("layerpkg.top",),
        forbidden=("json",),
        allowed=frozenset({("layerpkg.top", "json", "dumps"), ("layerpkg.top", "json", "loads")}),
        fix="Use the serializer helper instead.",
    )
    assert check_contracts(edges, [allowed]) == []
    assert stale_allowances(edges, [allowed]) == []

    stale = Contract(
        name="top does not import json",
        importers=("layerpkg.top",),
        forbidden=("json",),
        allowed=frozenset({("layerpkg.top", "json", "gone")}),
        fix="Use the serializer helper instead.",
    )
    messages = stale_allowances(edges, [stale])
    assert len(messages) == 1 and "the baseline only shrinks" in messages[0]


def test_parent_package_import_of_a_forbidden_module(synthetic_package):
    # Both forms record the edge against the parent package, with the forbidden
    # submodule only as an imported name.
    (synthetic_package / "via_parent.py").write_text(
        textwrap.dedent(
            """
            from layerpkg import inner, top
            from . import inner as relative_inner
            """
        )
    )
    edges = build_import_graph(synthetic_package, "layerpkg")
    contract = Contract(
        name="no inner",
        importers=("layerpkg.via_parent",),
        forbidden=("layerpkg.inner",),
        fix="Import VALUE through layerpkg.top.",
    )
    problems = check_contracts(edges, [contract])
    assert len(problems) == 2  # `top` from the same statement is not flagged
    assert "(line 2) imports `layerpkg.inner`." in problems[0]  # from layerpkg import inner
    assert "(line 3) imports `layerpkg.inner`." in problems[1]  # from . import inner

    allowed = Contract(
        name="no inner",
        importers=("layerpkg.via_parent",),
        forbidden=("layerpkg.inner",),
        allowed=frozenset({("layerpkg.via_parent", "layerpkg", "inner")}),
        fix="Import VALUE through layerpkg.top.",
    )
    assert check_contracts(edges, [allowed]) == []
    assert stale_allowances(edges, [allowed]) == []


def test_type_checking_imports_are_skipped_by_default(synthetic_package):
    edges = build_import_graph(synthetic_package, "layerpkg")
    contract = Contract(
        name="no socket",
        importers=("layerpkg",),
        forbidden=("socket",),
        fix="unused",
    )
    assert check_contracts(edges, [contract]) == []
    including = Contract(
        name="no socket",
        importers=("layerpkg",),
        forbidden=("socket",),
        include_type_checking=True,
        fix="unused",
    )
    assert len(check_contracts(edges, [including])) == 1
