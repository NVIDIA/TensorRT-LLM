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
per-file baseline of sites that may only shrink. A site is named by its
enclosing function or class and the server class it builds, so a site added in
another function fails even when one is removed from the same file.

The scan counts AST call nodes, not text, so a docstring or comment that
mentions ``object.__new__(OpenAIServer)`` does not count. A name bound by
``from ... import OpenAIServer as Server`` counts as ``OpenAIServer``.
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"

SERVER_CLASSES = ("OpenAIServer", "OpenAIDisaggServer", "CoordinatorServer")

# Per-file sites that existed when this ratchet landed, as
# "<enclosing function or class> <server class>" (see find_new_sites), repeated
# when one scope builds the class twice. The baseline only shrinks: rewrite a
# site on the factory, delete its entry.
BASELINE = {
    "tests/unittest/disaggregated/test_coordinator_worker.py": [
        "test_coordinator_keep_alive_timeout_is_passed_to_uvicorn CoordinatorServer",
        "test_coordinator_tcp_listener_uses_the_prebound_socket CoordinatorServer",
    ],
    "tests/unittest/disaggregated/test_disagg_internal_auth.py": [
        "test_aggregated_worker_ignores_affinity_header OpenAIServer",
        "test_worker_affinity_survives_wire_roundtrip OpenAIServer",
        "test_worker_rejects_protected_fields_without_cache_transceiver_config OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_disagg_openai_client.py": [
        "test_worker_clock_calibration_uses_global_clock OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_disagg_responses_api.py": [
        "test_a_context_worker_with_postprocessing_workers_returns_the_prompt OpenAIServer",
        "test_a_models_failure_is_an_error_response OpenAIDisaggServer",
    ],
    "tests/unittest/disaggregated/test_openai_disagg_server.py": [
        "test_create_client_does_not_register_with_server_metrics_collector OpenAIDisaggServer",
        "test_http_cluster_storage_request_is_proxied_to_coordinator OpenAIDisaggServer",
        "test_server_keep_alive_timeout_is_passed_to_uvicorn OpenAIDisaggServer",
    ],
    "tests/unittest/disaggregated/test_openai_disagg_service.py": [
        "test_create_chat_response_sets_prompt_token_ids_for_context_only OpenAIServer",
        "test_subagent_affinity_http_worker_adp_placement.make_worker OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_openai_server_info.py": [
        "_make_server OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_openai_server_stream_errors.py": [
        "_bare_server OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_paired_request_cancellation.py": [
        "test_http_failure_reaches_paired_worker_abort OpenAIDisaggServer",
        "test_http_failure_reaches_paired_worker_abort OpenAIServer",
    ],
    "tests/unittest/disaggregated/test_router.py": [
        "test_chat_harmony_preserves_original_tool_conversion_error OpenAIServer",
        "test_gpt_oss_router_tokens_match_chat_harmony_server_input OpenAIServer",
    ],
    "tests/unittest/executor/test_spec_decode_stats_payload.py": [
        "TestServerOptIn._server OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/_test_trtllm_serve_profile.py": [
        "server_with_mock_generator OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_anthropic_routes.py": [
        "_batch_client OpenAIServer",
        "_chat_stream OpenAIServer",
        "_count_tokens_client OpenAIServer",
        "_handle OpenAIDisaggServer",
        "_make_route_client OpenAIDisaggServer",
        "_make_route_client OpenAIServer",
        "test_await_disconnected_returns_immediately_for_a_synthetic_request OpenAIServer",
        "test_batch_routes_are_registered_on_the_standard_server OpenAIServer",
        "test_count_tokens_honours_the_server_chat_template OpenAIServer",
        "test_create_batch_at_capacity_is_429_not_400 OpenAIServer",
        "test_results_of_an_unfinished_batch_are_400_not_404.scenario OpenAIServer",
        "test_standard_and_disagg_register_messages_route OpenAIDisaggServer",
        "test_standard_and_disagg_register_messages_route OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_context_length_error.py": [
        "_disagg_server OpenAIDisaggServer",
    ],
    "tests/unittest/llmapi/apps/test_openai_server_iteration_stats.py": [
        "_make_server OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_render_callers.py": [
        "TestCountTokens._client OpenAIServer",
        "TestServerInfoFingerprint.test_the_first_call_hashes_off_the_event_loop OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_responses_store_replay.py": [
        "_server_up_to_preprocess OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_responses_streaming_events.py": [
        "_server_streaming OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_serving_extensions.py": [
        "TestOpenAIChatIntegration.chat_client OpenAIServer",
    ],
    "tests/unittest/llmapi/apps/test_tool_parsers.py": [
        "TestConfigureParserSpecialTokenDecoding.test_the_responses_route_applies_it OpenAIServer",
    ],
    "tests/unittest/llmapi/test_kv_hints.py": [
        "test_http_handlers_forward_hints OpenAIServer",
    ],
    "tests/unittest/llmapi/test_llm.py": [
        "test_openai_completion_list_prompt_stream_reuses_stream_metadata.run_request OpenAIServer",
        "test_startup_metrics_by_server_info OpenAIServer",
    ],
    "tests/unittest/llmapi/test_rl_control_auth.py": [
        "_make_server OpenAIServer",
    ],
    "tests/unittest/llmapi/test_runtime_control.py": [
        "_make_server OpenAIServer",
    ],
    "tests/unittest/llmapi/test_serve_multi_frontend_helpers.py": [
        "_attached_server OpenAIServer",
        "test_on_launcher_lost_fails_fast_then_shuts_down OpenAIServer",
        "test_only_the_launcher_registers_the_server OpenAIServer",
        "test_set_steady_clock_offset OpenAIServer",
    ],
    "tests/unittest/llmapi/test_serve_multi_frontend_route_gate.py": [
        "_llm_server_routes OpenAIServer",
    ],
}


def _class_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _server_class_names(tree: ast.AST) -> dict[str, str]:
    """Local name -> server class, with the names bound by ``import ... as``."""
    names = {server_class: server_class for server_class in SERVER_CLASSES}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in SERVER_CLASSES and alias.asname:
                    names[alias.asname] = alias.name
    return names


def _built_server_class(node: ast.AST, names: dict[str, str]) -> str:
    """The server class that a ``__new__`` call node builds, or "" for any other node."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return ""
    if node.func.attr != "__new__":
        return ""
    owner = node.func.value
    if _class_name(owner) == "object":
        if not node.args:
            return ""
        owner = node.args[0]
    if isinstance(owner, ast.Name):
        return names.get(owner.id, "")
    if isinstance(owner, ast.Attribute) and owner.attr in SERVER_CLASSES:
        return owner.attr
    return ""


def find_new_sites(tree: ast.AST) -> list[str]:
    """``"<enclosing function or class> <server class>"`` per ``__new__`` call that builds one.

    The scope is the dotted chain of enclosing ``def``/``class`` names, or
    ``<module>`` at module level; the list is sorted.
    """
    names = _server_class_names(tree)
    sites = []
    stack = [(tree, "")]
    while stack:
        node, scope = stack.pop()
        server_class = _built_server_class(node, names)
        if server_class:
            sites.append(f"{scope or '<module>'} {server_class}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        stack.extend((child, scope) for child in ast.iter_child_nodes(node))
    return sorted(sites)


def scan_tests(tests_root: Path = TESTS_ROOT) -> dict[str, list[str]]:
    sites_by_file = {}
    for path in sorted(tests_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        sites = find_new_sites(tree)
        if sites:
            sites_by_file[str(path.relative_to(REPO_ROOT))] = sites
    return sites_by_file


def ratchet_problems(
    sites_by_file: dict[str, list[str]], baseline: dict[str, list[str]]
) -> list[str]:
    """Sites missing from ``baseline``, then baseline entries with no site left, per file."""
    problems = []
    for file in sorted(set(sites_by_file) | set(baseline)):
        live = Counter(sites_by_file.get(file, []))
        recorded = Counter(baseline.get(file, []))
        for site in sorted((live - recorded).elements()):
            problems.append(
                f"{file}: '{site}' builds a server with object.__new__ and is not in the "
                "baseline; use `isolated_serve_env` from "
                "tests/test_common/serve/app_factory.py instead; see "
                "tensorrt_llm/serve/AGENTS.md."
            )
        for site in sorted((recorded - live).elements()):
            problems.append(
                f"{file}: '{site}' no longer builds a server with object.__new__; delete it "
                f"from BASELINE[{file!r}] in tests/unittest/serve/test_no_object_new.py; the "
                "baseline only shrinks."
            )
    return problems


def test_no_new_object_new_sites():
    problems = ratchet_problems(scan_tests(), BASELINE)
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
        ("from m import OpenAIServer as Server\nserver = object.__new__(Server)", 1),
        ("from m import OpenAIDisaggServer as S\nserver = S.__new__(S)", 1),
        ("from m import SomethingElse as Server\nthing = object.__new__(Server)", 0),
    ],
)
def test_scan_counts_call_nodes(source, expected):
    assert len(find_new_sites(ast.parse(source))) == expected


def test_scan_names_sites_by_scope_and_class():
    source = (
        "from tensorrt_llm.serve.openai_server import OpenAIServer as Server\n"
        "SERVER = object.__new__(OpenAIServer)\n"
        "class TestServer:\n"
        "    def test_build(self):\n"
        "        return object.__new__(Server)\n"
    )
    assert find_new_sites(ast.parse(source)) == [
        "<module> OpenAIServer",
        "TestServer.test_build OpenAIServer",
    ]


def test_ratchet_flags_a_replaced_site():
    # Removing one site and adding one in another function keeps the file's
    # total; the ratchet still reports both.
    file = "tests/unittest/test_example.py"
    before = find_new_sites(ast.parse("def test_old():\n    object.__new__(OpenAIServer)\n"))
    after = find_new_sites(ast.parse("def test_new():\n    object.__new__(OpenAIServer)\n"))
    assert ratchet_problems({file: after}, {file: after}) == []
    problems = ratchet_problems({file: after}, {file: before})
    assert len(problems) == 2, problems
    assert "'test_new OpenAIServer' builds a server" in problems[0]
    assert "'test_old OpenAIServer' no longer builds a server" in problems[1]
