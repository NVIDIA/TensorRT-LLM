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
"""Ratchet against new ``object.__new__(<server class>)`` test sites.

Tests that build a server with ``object.__new__`` bypass the constructor and pin
whatever attribute set the handler happens to read; they break or silently
mistest when handlers move. New tests build servers with ``isolated_serve_env``
from ``tests/test_common/serve/app_factory.py`` instead. The existing sites stay
until the change that moves each handler rewrites them, so this test holds a
per-file baseline that may only shrink.

The scan counts AST call nodes, not text, so a docstring or comment that
mentions ``object.__new__(OpenAIServer)`` does not count.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"

SERVER_CLASSES = ("OpenAIServer", "OpenAIDisaggServer", "CoordinatorServer")

# Per-file counts of the sites that existed when this ratchet landed.
# The baseline only shrinks: rewrite a site on the factory, lower its count.
BASELINE = {
    "tests/unittest/disaggregated/test_coordinator_worker.py": 2,
    "tests/unittest/disaggregated/test_disagg_internal_auth.py": 3,
    "tests/unittest/disaggregated/test_disagg_openai_client.py": 1,
    "tests/unittest/disaggregated/test_disagg_responses_api.py": 2,
    "tests/unittest/disaggregated/test_openai_disagg_server.py": 3,
    "tests/unittest/disaggregated/test_openai_disagg_service.py": 2,
    "tests/unittest/disaggregated/test_openai_server_info.py": 1,
    "tests/unittest/disaggregated/test_openai_server_stream_errors.py": 1,
    "tests/unittest/disaggregated/test_router.py": 2,
    "tests/unittest/executor/test_spec_decode_stats_payload.py": 1,
    "tests/unittest/llmapi/apps/_test_trtllm_serve_profile.py": 1,
    "tests/unittest/llmapi/apps/test_anthropic_routes.py": 13,
    "tests/unittest/llmapi/apps/test_context_length_error.py": 1,
    "tests/unittest/llmapi/apps/test_harmony_parsing.py": 1,
    "tests/unittest/llmapi/apps/test_openai_server_iteration_stats.py": 1,
    "tests/unittest/llmapi/apps/test_render_callers.py": 2,
    "tests/unittest/llmapi/apps/test_responses_store_replay.py": 1,
    "tests/unittest/llmapi/apps/test_responses_streaming_events.py": 1,
    "tests/unittest/llmapi/apps/test_serving_extensions.py": 1,
    "tests/unittest/llmapi/test_llm.py": 2,
    "tests/unittest/llmapi/test_rl_control_auth.py": 1,
    "tests/unittest/llmapi/test_runtime_control.py": 1,
}


def _class_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def count_new_sites(tree: ast.AST) -> int:
    """Call nodes that build a server class with ``__new__``."""
    sites = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "__new__":
            continue
        owner = _class_name(node.func.value)
        if owner == "object":
            if node.args and _class_name(node.args[0]) in SERVER_CLASSES:
                sites += 1
        elif owner in SERVER_CLASSES:
            sites += 1
    return sites


def scan_tests(tests_root: Path = TESTS_ROOT):
    counts = {}
    for path in sorted(tests_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        sites = count_new_sites(tree)
        if sites:
            counts[str(path.relative_to(REPO_ROOT))] = sites
    return counts


def test_no_new_object_new_sites():
    counts = scan_tests()
    problems = []
    for file, sites in sorted(counts.items()):
        baseline = BASELINE.get(file, 0)
        if sites > baseline:
            problems.append(
                f"{file} builds a server with object.__new__ ({sites} sites, baseline "
                f"{baseline}); use `isolated_serve_env` from "
                "tests/test_common/serve/app_factory.py instead; see "
                "tensorrt_llm/serve/AGENTS.md."
            )
    for file, baseline in sorted(BASELINE.items()):
        sites = counts.get(file, 0)
        if sites < baseline:
            problems.append(
                f"{file} now has {sites} object.__new__ sites; lower BASELINE[{file!r}] to "
                f"{sites} in tests/unittest/serve/test_no_object_new.py; the baseline only "
                "shrinks."
            )
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Self-test on synthetic sources: the scan counts calls, not text.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ('"""object.__new__(OpenAIServer) in a docstring"""', 0),
        ("# object.__new__(OpenAIServer) in a comment\nx = 1", 0),
        ("server = object.__new__(OpenAIServer)", 1),
        ("server = object.__new__(mod.OpenAIDisaggServer)", 1),
        ("server = CoordinatorServer.__new__(CoordinatorServer)", 1),
        ("thing = object.__new__(SomethingElse)", 0),
    ],
)
def test_scan_counts_call_nodes(source, expected):
    assert count_new_sites(ast.parse(source)) == expected
