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
"""The guided decoder must not return to its caller with host functions pending.

``CapturableGuidedDecoder`` builds its bitmask from CUDA host functions, and
each is a callback into Python that takes the GIL. A one-engine drafter's next
act is a draft forward whose native extension calls hold the GIL for their
whole duration, so a pending callback can never be serviced: it waits for the
GIL while the extension call waits for work only it can advance, and the rank
spins with no error and no log line.

These tests pin the drain that closes that window at the enqueue sites — the
last statement of ``execute`` / ``execute_draft_batch`` /
``rollback_rejected_batch`` — and the cases where it must NOT happen: while a
CUDA graph is being captured (a host wait is illegal there, and the captured
host nodes do not run until replay), and for the non-capturable decoder, which
runs its build inline and enqueues no callback. They also pin that every
worker-side guided-decoding path goes through the drained methods.
"""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    CapturableGuidedDecoder,
    GuidedDecoder,
    GuidedRequests,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase


class _Worker(SpecWorkerBase):
    """Bare carrier for the method under test.

    ``SpecWorkerBase.__init__`` builds real drafter state; nothing here needs it,
    so the instance skips it and carries only the one attribute the method
    reads. The two abstract members exist to make the class concrete; neither is
    reachable from the method under test, and both raise if that ever stops
    being true.
    """

    def __init__(self, guided_decoder):
        self.guided_decoder = guided_decoder

    @property
    def max_draft_len(self):
        raise AssertionError("the drain path must not consult the draft length")

    def _forward_impl(self, *args, **kwargs):
        raise AssertionError("the drain path must not run a draft forward")


def _decoder(cls):
    """A decoder of ``cls`` with no ``__init__`` side effects, calls recorded.

    Everything ``execute`` / ``execute_draft_batch`` / ``rollback_rejected_batch``
    touch is a recorded mock, so the tests run without a GPU and observe only
    the ordering the methods impose.
    """
    decoder = object.__new__(cls)
    decoder.stream = MagicMock(name="stream")
    decoder.token_event = MagicMock(name="token_event")
    decoder.bitmask_event = MagicMock(name="bitmask_event")
    decoder.requests = []
    decoder.max_num_draft_tokens = 1
    decoder.queue = Queue()
    # Skip state as __init__ / add_batch leave it: a batch with a guided
    # request, so the drain tests exercise the normal path.
    decoder._no_guided_requests = False
    decoder._step_skip = None
    decoder._cur_step_eager = False
    decoder._prev_step_eager = False
    for name in (
        "fetch_batch",
        "init_disagg_gen_requests",
        "build",
        "copy_bitmask",
        "apply_bitmask",
        "fetch_draft_batch",
        "fetch_accepted_batch",
        "rollback_rejected_tokens",
        "rollback_draft_tokens",
        "add_accepted_batch",
    ):
        setattr(decoder, name, MagicMock(name=name))
    return decoder


def _run(fn, *, capturing):
    """Run ``fn`` with CUDA stream plumbing stubbed out (no GPU needed)."""
    with (
        patch("torch.cuda.stream"),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
    ):
        return fn()


def _call(decoder, method):
    if method == "execute":
        return lambda: decoder.execute(MagicMock(name="logits"))
    if method == "execute_draft_batch":
        return lambda: decoder.execute_draft_batch(MagicMock(name="logits"), draft_step=0)
    assert method == "rollback_rejected_batch"
    return lambda: decoder.rollback_rejected_batch(MagicMock(name="num_accepted_tokens"))


_ENQUEUE_METHODS = ["execute", "execute_draft_batch", "rollback_rejected_batch"]


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_capturable_decoder_drains_when_not_capturing(method):
    """Eager path: every enqueue method ends with a host wait that releases the GIL."""
    decoder = _decoder(CapturableGuidedDecoder)
    _run(_call(decoder, method), capturing=False)

    decoder.bitmask_event.synchronize.assert_called_once_with()


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_capturable_decoder_drain_is_the_last_step(method):
    """Order matters: draining before the host functions are enqueued is a no-op."""
    decoder = _decoder(CapturableGuidedDecoder)
    calls = []
    decoder.bitmask_event.record.side_effect = lambda *a, **k: calls.append("record")
    decoder.bitmask_event.synchronize.side_effect = lambda *a, **k: calls.append("synchronize")

    _run(_call(decoder, method), capturing=False)

    assert calls == ["record", "synchronize"], calls


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_no_drain_while_capturing_a_cuda_graph(method):
    """A host wait inside capture is illegal, and the host nodes wait for replay."""
    decoder = _decoder(CapturableGuidedDecoder)
    _run(_call(decoder, method), capturing=True)

    decoder.bitmask_event.synchronize.assert_not_called()


def test_no_drain_for_the_non_capturable_decoder():
    """The base decoder builds inline and enqueues no callback, so it owes no drain."""
    decoder = _decoder(GuidedDecoder)
    _run(lambda: decoder.execute(MagicMock(name="logits")), capturing=False)

    decoder.bitmask_event.synchronize.assert_not_called()


def _unguided_eager_step(decoder, *, prev_step_eager=True):
    """State after ``add_batch`` for a batch without guided requests."""
    decoder._no_guided_requests = True
    decoder._step_skip = None
    decoder._cur_step_eager = False
    decoder._prev_step_eager = prev_step_eager


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_unguided_eager_step_after_eager_step_skips(method):
    """No guided request and nothing pending from a graph step: no grammar work, no drain."""
    decoder = _decoder(CapturableGuidedDecoder)
    _unguided_eager_step(decoder)
    _run(_call(decoder, method), capturing=False)

    decoder.bitmask_event.synchronize.assert_not_called()
    decoder.build.assert_not_called()
    decoder.apply_bitmask.assert_not_called()
    decoder.add_accepted_batch.assert_not_called()
    decoder.fetch_accepted_batch.assert_not_called()
    decoder.rollback_rejected_tokens.assert_not_called()


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_first_eager_step_after_graph_step_still_drains(method):
    """Host nodes of the previous (captured) step may still be pending, so keep the drain."""
    decoder = _decoder(CapturableGuidedDecoder)
    _unguided_eager_step(decoder, prev_step_eager=False)
    _run(_call(decoder, method), capturing=False)

    decoder.bitmask_event.synchronize.assert_called_once_with()


_CAPTURED_CALLS = {
    "execute": (
        "fetch_batch",
        "init_disagg_gen_requests",
        "build",
        "copy_bitmask",
        "apply_bitmask",
    ),
    "execute_draft_batch": (
        "fetch_draft_batch",
        "rollback_rejected_tokens",
        "build",
        "rollback_draft_tokens",
        "copy_bitmask",
        "apply_bitmask",
    ),
    "rollback_rejected_batch": (
        "add_accepted_batch",
        "fetch_accepted_batch",
        "rollback_rejected_tokens",
    ),
}


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_no_skip_while_capturing(method):
    """A captured step never skips: its host nodes run at replay and need their queue entry."""
    decoder = _decoder(CapturableGuidedDecoder)
    decoder.queue.put("captured-step")
    _unguided_eager_step(decoder)
    _run(_call(decoder, method), capturing=True)

    assert decoder._step_skip is None
    assert decoder._cur_step_eager is False
    assert list(decoder.queue.queue) == ["captured-step"]
    for name in _CAPTURED_CALLS[method]:
        getattr(decoder, name).assert_called_once()
    decoder.bitmask_event.record.assert_called_once_with()


def _add_batch(decoder, *, guided):
    """Submit one batch through the real ``add_batch``; only the request snapshot is stubbed."""
    params = object() if guided else None
    requests = SimpleNamespace(requests=[SimpleNamespace(guided_decoding_params=params)])
    with patch.object(GuidedRequests, "from_scheduled_requests", return_value=requests):
        decoder.add_batch(MagicMock(name="scheduled_requests"))


def _eager_execute(decoder):
    _run(_call(decoder, "execute"), capturing=False)


def _decoder_after_an_eager_step():
    decoder = _decoder(CapturableGuidedDecoder)
    _add_batch(decoder, guided=False)
    _eager_execute(decoder)  # The first eager step keeps the normal path.
    decoder.build.reset_mock()
    decoder.apply_bitmask.reset_mock()
    decoder.bitmask_event.synchronize.reset_mock()
    return decoder


def test_add_batch_unguided_batch_after_eager_step_skips():
    decoder = _decoder_after_an_eager_step()
    _add_batch(decoder, guided=False)
    _eager_execute(decoder)

    decoder.build.assert_not_called()
    decoder.apply_bitmask.assert_not_called()
    decoder.bitmask_event.synchronize.assert_not_called()


def test_add_batch_guided_batch_after_eager_step_runs_the_grammar():
    decoder = _decoder_after_an_eager_step()
    _add_batch(decoder, guided=True)
    _eager_execute(decoder)

    decoder.build.assert_called_once()
    decoder.apply_bitmask.assert_called_once()
    decoder.bitmask_event.synchronize.assert_called_once_with()


_BATCH_ADD_METHODS = {
    "add_draft_batch": lambda decoder: CapturableGuidedDecoder.add_draft_batch(
        decoder, torch.zeros(1), torch.ones(1), draft_step=0
    ),
    # The fixture mocks ``add_accepted_batch`` for the rollback tests, so call the real one.
    "add_accepted_batch": lambda decoder: CapturableGuidedDecoder.add_accepted_batch(
        decoder, torch.ones(1)
    ),
}


def _batch_add_decoder():
    decoder = _decoder(CapturableGuidedDecoder)
    decoder.requests = [object()]
    decoder.new_tokens = MagicMock(name="new_tokens")
    decoder.num_accepted_tokens = MagicMock(name="num_accepted_tokens")
    return decoder


@pytest.mark.parametrize("method", _BATCH_ADD_METHODS)
def test_batch_add_on_a_skipped_step_copies_nothing(method):
    decoder = _batch_add_decoder()
    _unguided_eager_step(decoder)
    _run(lambda: _BATCH_ADD_METHODS[method](decoder), capturing=False)

    decoder.new_tokens.__getitem__.assert_not_called()
    decoder.num_accepted_tokens.__getitem__.assert_not_called()
    decoder.token_event.record.assert_not_called()


@pytest.mark.parametrize("method", _BATCH_ADD_METHODS)
def test_batch_add_on_a_guided_step_copies_and_records(method):
    decoder = _batch_add_decoder()
    _run(lambda: _BATCH_ADD_METHODS[method](decoder), capturing=False)

    decoder.num_accepted_tokens.__getitem__.return_value.copy_.assert_called_once()
    if method == "add_draft_batch":
        decoder.new_tokens.__getitem__.return_value.copy_.assert_called_once()
    decoder.token_event.record.assert_called_once_with()


def test_skipped_step_consumes_only_its_own_queue_entry():
    """A skipped step drops the newest entry; older ones belong to pending graph host nodes."""
    decoder = _decoder(CapturableGuidedDecoder)
    decoder.queue.put("graph-step")
    decoder.queue.put("this-step")
    _unguided_eager_step(decoder)

    _run(lambda: decoder.execute(MagicMock(name="logits")), capturing=False)
    _run(
        lambda: decoder.execute_draft_batch(MagicMock(name="logits"), draft_step=0), capturing=False
    )

    assert list(decoder.queue.queue) == ["graph-step"]


def test_worker_executes_the_guided_decoder():
    """The worker helper delegates to ``execute``, which drains internally."""
    decoder = MagicMock(name="guided_decoder")
    worker = _Worker(decoder)
    logits = object()

    worker._execute_guided_decoder_if_present(logits)

    decoder.execute.assert_called_once_with(logits)


def test_no_guided_decoder_is_a_no_op():
    """The overwhelmingly common path must not touch CUDA at all."""
    worker = _Worker(None)
    with patch("torch.cuda.is_current_stream_capturing") as capturing:
        worker._execute_guided_decoder_if_present(object())
    capturing.assert_not_called()


def test_skip_drafting_routes_through_the_guided_helper():
    """``skip_drafting`` must route through the shared guided helper (not call
    ``guided_decoder.execute`` directly) so every guard in the helper covers it."""
    worker = _Worker(MagicMock(name="guided_decoder"))
    batch_size = 3
    attn_metadata = MagicMock(num_seqs=batch_size, num_contexts=1)
    logits = torch.zeros((batch_size, 8))
    with (
        patch.object(_Worker, "_execute_guided_decoder_if_present") as helper,
        patch.object(
            _Worker,
            "_sample_tokens_for_batch",
            return_value=torch.arange(batch_size, dtype=torch.int),
        ),
    ):
        out = worker.skip_drafting(
            input_ids=None,
            position_ids=None,
            hidden_states=None,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=MagicMock(name="spec_metadata"),
            draft_model=None,
        )

    helper.assert_called_once_with(logits)
    assert out["new_tokens"].tolist() == [[0], [1], [2]]
