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
"""What a context handoff must carry before its request may resume generating.

A handoff short of its first tokens has to be refused while the request is
still unschedulable. Once it is batched, nothing downstream can tell: the
model engine substitutes the prompt's trailing token and the sampler treats
the step as ordinary generation, so the client is served a wrong answer that
reports success.

Both methods under test live in PyExecutor, whose module imports torch; the
methods themselves do not, so they are compiled out of the source file and
handed fakes for the few names they use. That keeps this on the CPU lane with
the rest of the disaggregation contract tests.
"""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PY_EXECUTOR = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "py_executor.py"
)

_REFUSE = "_refuse_incomplete_disagg_gen_handoffs"
_PREPARE = "_prepare_disagg_gen_transmission_complete"
# The loops reach the refusal through the disagg coordinator, not directly.
_REFUSE_CALL = "self.disagg.refuse_incomplete_gen_handoffs"


class _State(Enum):
    DISAGG_GENERATION_TRANS_COMPLETE = "trans_complete"
    GENERATION_IN_PROGRESS = "in_progress"
    DISAGG_TRANS_ERROR = "trans_error"


class _ResourceManagerType(Enum):
    SEQ_SLOT_MANAGER = "seq_slot"
    KV_CACHE_MANAGER = "kv_cache"


def _method(name):
    """The named PyExecutor method, as source AST."""
    tree = ast.parse(_PY_EXECUTOR.read_text(), filename=str(_PY_EXECUTOR))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyExecutor"
    )
    return next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_lines(name, callee):
    """Source lines on which the named method calls ``callee``."""
    return sorted(
        node.lineno
        for node in ast.walk(_method(name))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
    )


def _lift(name):
    """The named PyExecutor method as a plain function, decorators dropped."""
    fn = ast.parse(ast.unparse(_method(name))).body[0]
    fn.decorator_list = []
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {
        "LlmRequestState": _State,
        "ResourceManagerType": _ResourceManagerType,
        "ScheduledRequests": lambda: SimpleNamespace(context_requests_last_chunk=[]),
        "logger": SimpleNamespace(error=lambda *a, **k: None),
    }
    exec(compile(ast.fix_missing_locations(module), str(_PY_EXECUTOR), "exec"), namespace)
    return namespace[name]


class _Request:
    """Enough of LlmRequest to be routed by state, which is the whole point.

    ``is_disagg_generation_transmission_complete`` derives from ``state`` on
    the real request, so a refusal is what keeps the request out of the
    downstream loop -- a static flag here would hide exactly that.
    """

    def __init__(self, request_id, first_gen_tokens, beam_width=1):
        self.py_request_id = request_id
        self.state = _State.DISAGG_GENERATION_TRANS_COMPLETE
        self.context_phase_params = SimpleNamespace(
            first_gen_tokens=first_gen_tokens, draft_tokens=None
        )
        self.py_beam_width = beam_width
        self.prompt_len = 8
        self.context_current_position = 0
        self.decoding_iter = 0
        self.py_decoding_iter = 0
        self.py_kv_transfer_start_time = 1.0
        self.py_kv_transfer_timed_out = False
        self.py_draft_tokens = None
        self.tokens = []

    @property
    def is_disagg_generation_transmission_complete(self):
        return self.state is _State.DISAGG_GENERATION_TRANS_COMPLETE

    def add_new_token(self, token, beam):
        self.tokens.append(token)


class _Executor(SimpleNamespace):
    """Records the three mutations that must not reach a refused request."""

    def __init__(self, active_requests=(), enable_spec_decode=False):
        seq_slots_given = []
        sampler_states_created = []
        kda_slots_seeded = []
        transfer_errors_checked = []
        resource_managers = {
            _ResourceManagerType.SEQ_SLOT_MANAGER: SimpleNamespace(
                prepare_resources=lambda requests: seq_slots_given.extend(
                    requests.context_requests_last_chunk
                )
            ),
            _ResourceManagerType.KV_CACHE_MANAGER: SimpleNamespace(
                seed_kda_replay_caches_for_disagg_gen=kda_slots_seeded.extend
            ),
        }
        super().__init__(
            active_requests=list(active_requests),
            seq_slots_given=seq_slots_given,
            sampler_states_created=sampler_states_created,
            kda_slots_seeded=kda_slots_seeded,
            transfer_errors_checked=transfer_errors_checked,
            _check_cache_transfer_errors=transfer_errors_checked.append,
            resource_manager=SimpleNamespace(resource_managers=resource_managers),
            _setup_sampler_step=lambda requests: sampler_states_created.extend(
                requests.context_requests_last_chunk
            ),
            model_engine=SimpleNamespace(
                enable_spec_decode=enable_spec_decode, max_total_draft_tokens=0
            ),
            # Non-None: the refusal is a disagg-only concern and returns early
            # without one.
            kv_cache_transceiver=SimpleNamespace(commit_blocks_for_reuse=lambda req: None),
            _update_sampler_state_for_disagg_gen_request=lambda *a: True,
            _maybe_prepend_logprobs_and_logits=lambda *a: None,
        )


@pytest.mark.parametrize("missing", [None, []])
def test_a_handoff_without_a_first_token_fails_only_its_own_request(missing):
    """An empty list is how a missing first token actually arrives.

    ``AuxBuffer.fill_slot`` builds ``first_gen_tokens`` from
    ``request.get_last_tokens()``, a list, so a None check alone lets ``[]``
    through to ``first_gen_tokens[beam]``. And it must fail one request, not
    every request the transfer poll happens to sweep.
    """
    bad = _Request(1, missing)
    good = _Request(2, [42])

    _lift(_REFUSE)(_Executor(active_requests=[bad, good]))

    assert bad.state is _State.DISAGG_TRANS_ERROR
    assert good.state is _State.DISAGG_GENERATION_TRANS_COMPLETE


def test_a_handoff_short_of_the_beam_count_is_refused():
    """Four beams need four first tokens; three is the same missing token."""
    short = _Request(3, [7, 8, 9], beam_width=4)

    _lift(_REFUSE)(_Executor(active_requests=[short]))

    assert short.state is _State.DISAGG_TRANS_ERROR


def test_a_refused_handoff_is_swept_rather_than_left_in_the_error_state():
    """Marking it is not failing it: something has to reap the state.

    The request left the transfer manager when its handoff landed, so neither
    transfer poll visits it again. Without the sweep it holds its slot and its
    client waits for a response that is never sent.
    """
    executor = _Executor(active_requests=[_Request(1, None)])

    _lift(_REFUSE)(executor)

    assert executor.transfer_errors_checked == ["generation requests"]


def test_a_loop_head_that_refuses_nothing_does_not_sweep():
    """The sweep takes the response path's locks; it is not a per-iteration cost."""
    executor = _Executor(active_requests=[_Request(1, [42])])

    _lift(_REFUSE)(executor)

    assert executor.transfer_errors_checked == []


def test_a_refused_handoff_is_never_prepared_for_generation():
    """The load-bearing one: refusal has to precede every mutation it guards.

    Before the fix the refusal lived inside ``_PREPARE``, after the seq slot,
    the sampler step and the KDA seed had already run for the whole
    transfer-complete group -- so these three assertions fail, and the request
    then rode the batch into the forward and the sampler.
    """
    bad = _Request(1, [])
    good = _Request(2, [42])
    batch = SimpleNamespace(generation_requests=[bad, good])
    executor = _Executor(active_requests=[bad, good], enable_spec_decode=True)

    # The order the executor runs them in: refuse at the loop head, then
    # prepare whatever the scheduler was still allowed to batch.
    _lift(_REFUSE)(executor)
    _lift(_PREPARE)(executor, batch)

    assert bad not in executor.seq_slots_given
    assert bad not in executor.sampler_states_created
    assert bad.py_request_id not in executor.kda_slots_seeded
    assert bad.state is _State.DISAGG_TRANS_ERROR
    assert bad.context_current_position == 0

    # The refusal did not take its batch neighbour down with it.
    assert good in executor.seq_slots_given
    assert good.state is _State.GENERATION_IN_PROGRESS
    assert good.context_current_position == good.prompt_len
    assert good.tokens == [42]


@pytest.mark.parametrize(
    "loop, schedules",
    [
        ("_prepare_and_schedule_batch", "self._schedule"),
        ("_executor_loop_pp", "self._pp_schedule_and_propagate"),
    ],
)
def test_the_refusal_runs_at_every_loop_head_before_scheduling(loop, schedules):
    """Where the refusal is called is the whole guarantee, and no CPU test can
    drive a real executor loop to observe it -- so read it off the source.

    Both loop heads must sweep: ``_prepare_and_schedule_batch`` serves the
    non-overlap and overlap loops, ``_executor_loop_pp`` schedules inline. A
    handoff landing anywhere in iteration N is therefore refused no later than
    the head of N+1, and ahead of the scheduling call in that same iteration,
    so no scheduler ever sees a request whose first tokens are missing.
    """
    refusals = _call_lines(loop, _REFUSE_CALL)
    scheduling = _call_lines(loop, schedules)

    assert refusals, f"{loop} does not refuse incomplete handoffs"
    assert scheduling, f"{loop} no longer schedules through {schedules}"
    assert max(refusals) < min(scheduling)


@pytest.mark.parametrize("missing", [None, []])
def test_an_unrefused_short_handoff_fails_loudly_rather_than_quietly(missing):
    """No second, silent line of defence inside ``_PREPARE``.

    Before the fix ``_PREPARE`` swallowed a short handoff and returned, leaving
    the half-initialised request in the batch the forward and the sampler were
    handed -- a substituted token reported as success. Raising is the correct
    failure for a producer that reaches ``_PREPARE`` without being refused.
    """
    bypassed = _Request(1, missing)
    batch = SimpleNamespace(generation_requests=[bypassed])

    with pytest.raises((IndexError, TypeError)):
        _lift(_PREPARE)(_Executor(), batch)
