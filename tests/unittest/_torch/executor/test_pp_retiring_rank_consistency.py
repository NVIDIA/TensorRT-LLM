# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Every pipeline stage must derive the same retiring-request count.

The ADP router subtracts retiring requests from each rank's reported load
(nvbug-6627795). Under pipeline parallelism that correction is only safe if every
stage agrees on *which* requests are retiring: each rank pops from its own copy of
the waiting queue and applies the router's decision locally, so a per-stage
disagreement means the stages admit different numbers of requests and diverge --
a hang, not a wrong number.

Before this change only the last stage marked ``GENERATION_TO_COMPLETE`` for
generation requests (``_executor_loop_pp``), which is why the correction was gated
off whenever ``pp_size > 1``. ``_forward_step_inter_pp`` now makes the same call at
the structurally identical point, so the counts agree.

The marking is safe to replicate because the predicate
(``LlmRequest::willCompleteNextIteration``) is pure arithmetic on token counts, and
those counts are replicated to every stage in the same iteration: the last stage's
sample state is ring-broadcast and applied on all ranks by
``_handle_executed_batch``, with the per-iteration batch count itself ring-broadcast
from rank 0.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import count_retiring_requests

pytestmark = pytest.mark.cpu_only

# (num_generated_tokens, max_new_tokens): the replicated token counts every stage
# holds for one request. The first two retire on the next iteration, the rest do
# not -- so a stage that skips the marking under-counts by exactly two.
REPLICATED_TOKEN_COUNTS = [(15, 16), (7, 8), (3, 16), (1, 8), (15, 64)]


class _StageLocalRequest:
    """One pipeline stage's own object for a request.

    Each stage has a distinct ``LlmRequest`` instance; only the token counts are
    replicated. Mirroring that here is the point of the test: the marking must be
    reproducible from the replicated fields alone, with no reference to sampler
    state that lives on the last stage.
    """

    def __init__(
        self, num_generated_tokens: int, max_new_tokens: int, tokens_per_iteration: int = 1
    ):
        self.num_generated_tokens = num_generated_tokens
        self.max_new_tokens = max_new_tokens
        self.tokens_per_iteration = tokens_per_iteration
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.exclude_last_generation_logits = True

    def will_complete_next_iteration(self) -> bool:
        # Mirrors LlmRequest::willCompleteNextIteration (llmRequest.h): pure
        # arithmetic on counts that are identical on every stage. No EOS check, no
        # stop words, no sampler state.
        return self.num_generated_tokens + self.tokens_per_iteration >= self.max_new_tokens

    def set_exclude_last_generation_logits(self, value: bool) -> None:
        self.exclude_last_generation_logits = value


def _stage_local_batch():
    return [_StageLocalRequest(generated, limit) for generated, limit in REPLICATED_TOKEN_COUNTS]


def _mark(requests) -> None:
    """Run the real marking method against a stub ``self``.

    The method reads nothing off ``self``, which is exactly why it can be called
    from both loops without any stage-local context.
    """
    PyExecutor._update_generation_requests_that_will_complete_next_iteration(
        SimpleNamespace(), requests
    )


EXPECTED_RETIRING = 2


def test_the_batch_actually_contains_retiring_requests():
    """Anti-vacuity: without this the consistency assertions hold trivially at 0."""
    batch = _stage_local_batch()
    _mark(batch)
    assert count_retiring_requests(batch) == EXPECTED_RETIRING
    assert EXPECTED_RETIRING < len(batch)


@pytest.mark.parametrize("pp_size", [2, 4])
def test_every_stage_derives_the_same_retiring_count(pp_size):
    """All stages mark: the counts agree, which is what makes the correction safe."""
    stages = [_stage_local_batch() for _ in range(pp_size)]
    for batch in stages:
        _mark(batch)

    counts = [count_retiring_requests(batch) for batch in stages]
    assert counts == [EXPECTED_RETIRING] * pp_size

    # Stronger than the count: the same *requests* are marked on every stage, so
    # each rank subtracts the same load, not merely the same amount of it.
    marked = {
        tuple(
            i for i, req in enumerate(batch) if req.state == LlmRequestState.GENERATION_TO_COMPLETE
        )
        for batch in stages
    }
    assert len(marked) == 1


@pytest.mark.parametrize("pp_size", [2, 4])
def test_last_stage_only_marking_is_detectably_inconsistent(pp_size):
    """Negative control: the pre-change behaviour must fail this test's premise.

    If only the last stage marks, the stages disagree by exactly the number of
    retiring requests. Without this case a test that merely asserts "the counts
    agree" would keep passing after a regression that removes the marking from
    *every* stage.
    """
    stages = [_stage_local_batch() for _ in range(pp_size)]
    _mark(stages[-1])

    counts = [count_retiring_requests(batch) for batch in stages]
    assert counts[-1] == EXPECTED_RETIRING
    assert counts[:-1] == [0] * (pp_size - 1)
    assert len(set(counts)) > 1


def test_marking_is_idempotent_across_repeated_stage_passes():
    """A stage that marks twice in an iteration must not drift from one that marks once.

    ``_forward_step_inter_pp`` runs once per micro-batch, and a request can appear
    in consecutive micro-batches, so the operation has to be a no-op on an
    already-marked request.
    """
    once, twice = _stage_local_batch(), _stage_local_batch()
    _mark(once)
    _mark(twice)
    _mark(twice)
    assert count_retiring_requests(once) == count_retiring_requests(twice)


def test_already_complete_requests_are_left_alone():
    """GENERATION_COMPLETE is terminal; re-marking it would resurrect a torn-down
    request into the retiring set on one stage only."""
    batch = _stage_local_batch()
    batch[0].state = LlmRequestState.GENERATION_COMPLETE
    _mark(batch)
    assert batch[0].state == LlmRequestState.GENERATION_COMPLETE
    assert count_retiring_requests(batch) == EXPECTED_RETIRING - 1


def _calls_in(func) -> set:
    """Names of every method called on ``self`` inside ``func``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_inter_pp_forward_marks_retiring_requests():
    """The structural half of the fix, and the part a unit test can pin.

    The stage-local arithmetic above is only reached if the non-last-stage path
    actually calls the marking method. This is the assertion that fails if that
    call is dropped -- i.e. it is what makes the seat-headroom PP cell honest.
    """
    calls = _calls_in(PyExecutor._forward_step_inter_pp)
    assert "_update_generation_requests_that_will_complete_next_iteration" in calls
    # Same point in the sequence as the last stage: right after the state update.
    assert "_update_request_states" in calls


def test_marking_stays_paired_with_the_state_update_on_both_paths():
    """Both loops must mark, and neither may mark without first updating state.

    The count is only rank-consistent if the two operations stay adjacent: marking
    before ``_update_request_states`` would read token counts from before this
    micro-batch and disagree with a stage that marks after.
    """
    for func in (PyExecutor._forward_step_inter_pp, PyExecutor._executor_loop_pp):
        source = inspect.getsource(func)
        assert "_update_generation_requests_that_will_complete_next_iteration" in source, (
            f"{func.__name__} does not mark retiring generation requests; the ADP "
            "router's admission correction would then diverge between stages"
        )
        assert source.index("_update_request_states") < source.index(
            "_update_generation_requests_that_will_complete_next_iteration"
        )
