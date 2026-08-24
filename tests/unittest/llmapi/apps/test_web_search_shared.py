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
"""Offline tests for the endpoint-neutral web search core.

Web search is a server tool on every API that offers it, so the provider
selection, budget and result rendering are shared. These tests pin the parts
that no endpoint should have to reimplement.
"""

import asyncio

import pytest

from tensorrt_llm.serve.web_search import (
    SearchOutcome,
    WebSearchConfig,
    WebSearchResult,
    WebSearchSession,
    WebSearchToolSpec,
    load_web_search_config,
    resolve_web_search,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in list(os_environ_names()):
        monkeypatch.delenv(name, raising=False)


def os_environ_names():
    import os

    return [n for n in os.environ if n.startswith("TRTLLM_WEB_SEARCH")]


def _spec(**kwargs):
    kwargs.setdefault("name", "web_search")
    return WebSearchToolSpec(**kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_disabled_by_default():
    assert load_web_search_config().enabled is False


def test_neutral_env_selects_the_provider(monkeypatch):
    monkeypatch.setenv("TRTLLM_WEB_SEARCH", "wikipedia")
    assert load_web_search_config().provider == "wikipedia"


def test_numeric_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TRTLLM_WEB_SEARCH_MAX_RESULTS", "7")
    assert load_web_search_config().max_results == 7


# ---------------------------------------------------------------------------
# Resolution, shared by every endpoint
# ---------------------------------------------------------------------------


def test_no_tool_means_no_config():
    config, error = resolve_web_search(None)
    assert config is None and error is None


def test_disabled_provider_is_an_error_not_a_silent_drop(monkeypatch):
    """Dropping it would let the model answer from memory as if it searched."""
    config, error = resolve_web_search(_spec())
    assert config is None
    assert "not supported by this server" in error


def test_enabled_provider_resolves(monkeypatch):
    monkeypatch.setenv("TRTLLM_WEB_SEARCH", "wikipedia")
    config, error = resolve_web_search(_spec())
    assert error is None
    assert config.provider == "wikipedia"


def test_client_may_lower_the_budget(monkeypatch):
    monkeypatch.setenv("TRTLLM_WEB_SEARCH", "wikipedia")
    monkeypatch.setenv("TRTLLM_WEB_SEARCH_MAX_USES", "5")
    config, _ = resolve_web_search(_spec(max_uses=2))
    assert config.max_uses == 2


def test_client_may_not_raise_the_budget(monkeypatch):
    """Otherwise a prompt could make the server issue unbounded traffic."""
    monkeypatch.setenv("TRTLLM_WEB_SEARCH", "wikipedia")
    monkeypatch.setenv("TRTLLM_WEB_SEARCH_MAX_USES", "5")
    config, _ = resolve_web_search(_spec(max_uses=500))
    assert config.max_uses == 5


def test_domain_filters_are_carried_through(monkeypatch):
    monkeypatch.setenv("TRTLLM_WEB_SEARCH", "wikipedia")
    config, _ = resolve_web_search(
        _spec(allowed_domains=["a.example"], blocked_domains=["b.example"])
    )
    assert config.allowed_domains == ("a.example",)
    assert config.blocked_domains == ("b.example",)


# ---------------------------------------------------------------------------
# The session: budget and failure handling
# ---------------------------------------------------------------------------


def _session(monkeypatch, results=None, error=None, max_uses=2):
    config = WebSearchConfig(provider="wikipedia", max_uses=max_uses)
    session = WebSearchSession(config)

    async def fake_run(query, cfg):
        if error is not None:
            from tensorrt_llm.serve.web_search import WebSearchError

            raise WebSearchError(error)
        return results or []

    monkeypatch.setattr("tensorrt_llm.serve.web_search.run_web_search", fake_run)
    return session


def test_session_counts_searches(monkeypatch):
    session = _session(monkeypatch, results=[WebSearchResult("u", "t")])
    asyncio.run(session.run("one"))
    assert session.searches_done == 1
    assert session.exhausted is False


def test_session_stops_at_the_budget(monkeypatch):
    session = _session(monkeypatch, results=[], max_uses=1)
    asyncio.run(session.run("one"))
    assert session.exhausted is True
    outcome = asyncio.run(session.run("two"))
    assert outcome.ok is False
    assert "budget" in outcome.error
    assert session.searches_done == 1


def test_session_reports_a_failure_rather_than_raising(monkeypatch):
    """A failed search should cost the turn, not the request."""
    session = _session(monkeypatch, error="provider exploded")
    outcome = asyncio.run(session.run("one"))
    assert outcome.ok is False
    assert "provider exploded" in outcome.as_model_text()


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


def test_results_render_for_the_model():
    outcome = SearchOutcome(
        query="trtllm", results=[WebSearchResult("https://x.example", "Title", "Snippet")]
    )
    text = outcome.as_model_text()
    assert "trtllm" in text and "https://x.example" in text and "Snippet" in text


def test_empty_results_say_so():
    assert "No results" in SearchOutcome(query="nothing").as_model_text()
