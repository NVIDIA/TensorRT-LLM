# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin cache manager's placeholder-draft requests under forced acceptance (CPU).

Under forced acceptance a request's first generation step after a context step on this server, or after a
disaggregated handoff, keeps its own accepted-token count, and every later generation row is forced and counted;
as in the engine's parallel-draft modes, that step's drafts count as placeholders. ``placeholder_draft_requests``
decides it per request from the scheduled requests; ``RubinKdaDsaHybridCacheManager`` marks each step's requests
(``placeholder_requests``) and forgets a released one.

The cases: chunked prefill (only the step after the last chunk), a disaggregated first step after a request that
only receives its KV, the manager's step and release.
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

# The family's modules import its kernels, which need the CuTe DSL.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="this environment lacks cutlass"
)
cpu = pytest.mark.cpu_only


def _manager_module():
    return importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager"
    )


_NO_HANDOFF = object()


def _request(request_id, *, handoff=_NO_HANDOFF, receiving=False, prompt_len=64):
    """A scheduled request: ``handoff`` is the drafts a context server handed over (by default, no handoff)."""
    return SimpleNamespace(
        py_request_id=request_id,
        context_phase_params=(
            None if handoff is _NO_HANDOFF else SimpleNamespace(draft_tokens=handoff)
        ),
        is_disagg_generation_init_state=receiving,
        py_prompt_len=prompt_len,
    )


def _batch(context=(), generation=()):
    return SimpleNamespace(context_requests=list(context), generation_requests=list(generation))


def _placeholders(pending, **batch):
    return _manager_module().placeholder_draft_requests(pending, _batch(**batch))


@cpu
def test_chunked_prefill_exempts_only_the_step_after_the_last_chunk():
    """Two context chunks, then generation steps: the first generation step keeps its own count, later ones are
    forced; a request still in its chunks has no generation row."""
    pending = {}
    assert _placeholders(pending, context=[_request(1)]) == frozenset()
    assert _placeholders(pending, context=[_request(1), _request(2)]) == frozenset()
    assert _placeholders(pending, generation=[_request(1)], context=[_request(2)]) == {1}
    assert _placeholders(pending, generation=[_request(1), _request(2)]) == {2}
    assert _placeholders(pending, generation=[_request(1), _request(2)]) == frozenset()


@pytest.mark.parametrize("handoff", [[11, 12, 13, 14]], ids=["full"])
@cpu
def test_a_disaggregated_first_step_is_exempt_whatever_drafts_it_brought(handoff):
    """A generation server: the request receives its KV (scheduled with the context requests, never marked),
    then its first generation step keeps its own count and its later steps are forced."""
    pending = {}
    receiving = [_request(5, handoff=handoff, receiving=True)]
    assert _placeholders(pending, context=receiving) == frozenset()
    assert 5 not in pending
    first = [_request(5, handoff=handoff), _request(6, handoff=handoff)]
    assert _placeholders(pending, generation=first) == {5, 6}
    assert _placeholders(pending, generation=first) == frozenset()


def _rubin_manager(monkeypatch):
    mod = _manager_module()
    manager = mod.RubinKdaDsaHybridCacheManager.__new__(mod.RubinKdaDsaHybridCacheManager)
    manager.first_step_pending = {}
    manager.placeholder_requests = frozenset()
    base = mod.RubinKdaDsaHybridCacheManager.__mro__[1]
    released = []
    monkeypatch.setattr(base, "prepare_resources", lambda self, batch: None)
    monkeypatch.setattr(
        base, "free_resources", lambda self, request, pin=False: released.append(request)
    )
    return manager, released


@cpu
def test_the_manager_marks_each_step_and_forgets_a_released_request(monkeypatch):
    manager, released = _rubin_manager(monkeypatch)
    first = _request(5, handoff=[])
    manager.prepare_resources(_batch(generation=[first]))
    assert manager.placeholder_requests == {5}
    manager.prepare_resources(_batch(generation=[first]))
    assert manager.placeholder_requests == frozenset()
    manager.free_resources(first)
    assert manager.first_step_pending == {} and released == [first]
