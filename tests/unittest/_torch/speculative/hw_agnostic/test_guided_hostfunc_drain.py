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

Replayed graphs' host functions are drained before a multimodal encoder runs,
and an eager forward with nothing constrained enqueues none.
"""

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch import hostfunc
from tensorrt_llm._torch.models import modeling_multimodal_utils
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
    decoder.active = True
    decoder.max_num_draft_tokens = 1
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


@pytest.mark.parametrize("method", _ENQUEUE_METHODS)
def test_inactive_forward_enqueues_nothing(method):
    """An eager forward with no constrained request launches no host function."""
    decoder = _decoder(CapturableGuidedDecoder)
    decoder.active = False

    result = _run(_call(decoder, method), capturing=False)

    for name in ("fetch_batch", "fetch_draft_batch", "fetch_accepted_batch", "build"):
        getattr(decoder, name).assert_not_called()
    decoder.bitmask_event.record.assert_not_called()
    decoder.bitmask_event.synchronize.assert_not_called()
    assert result in (None, [])


@pytest.mark.parametrize(
    "is_cuda_graph,constrained",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_add_batch_activates_for_cuda_graphs_and_constrained_batches(is_cuda_graph, constrained):
    """A graph replays the guided work it captured, whatever the batch holds."""
    decoder = object.__new__(CapturableGuidedDecoder)
    decoder.max_num_draft_tokens = 1
    decoder.queue = Queue()
    snapshot = MagicMock(name="guided_requests")
    snapshot.valid_requests.side_effect = lambda: iter([object()] if constrained else [])

    with patch.object(GuidedRequests, "from_scheduled_requests", return_value=snapshot):
        decoder.add_batch(MagicMock(name="scheduled_requests"), is_cuda_graph=is_cuda_graph)

    expected = is_cuda_graph or constrained
    assert decoder.active == expected
    assert decoder.queue.qsize() == int(expected)


@pytest.mark.parametrize(
    "captured,capturing,expect_sync",
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_drain_captured_hostfuncs(captured, capturing, expect_sync):
    """Synchronize only when a graph holds host functions, and never inside capture."""
    handles = {1} if captured else set()
    with (
        patch.object(hostfunc, "HOSTFUNC_USER_DATA_HANDLES", handles),
        patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        patch("torch.cuda.synchronize") as synchronize,
    ):
        hostfunc.drain_captured_hostfuncs()

    assert synchronize.call_count == int(expect_sync)


def _multimodal_param(embedding=None):
    param = MagicMock(name="multimodal_param")
    param.encoder_event = None
    param.multimodal_runtime = None
    param.multimodal_data = {} if embedding is None else {"multimodal_embedding": embedding}
    return param


def test_multimodal_encoder_runs_after_the_drain():
    calls = []
    param = _multimodal_param()

    def encode(params):
        calls.append("encode")
        return torch.zeros(1, 4)

    with (
        patch.object(
            modeling_multimodal_utils,
            "drain_captured_hostfuncs",
            side_effect=lambda: calls.append("drain"),
        ),
        patch.object(
            modeling_multimodal_utils, "_get_uncached_multimodal_params", return_value=[param]
        ),
    ):
        modeling_multimodal_utils.get_multimodal_embeddings(encode, [param])

    assert calls == ["drain", "encode"]


def test_no_drain_when_every_embedding_is_cached():
    param = _multimodal_param(embedding=torch.zeros(1, 4))
    with (
        patch.object(modeling_multimodal_utils, "drain_captured_hostfuncs") as drain,
        patch.object(modeling_multimodal_utils, "_get_uncached_multimodal_params", return_value=[]),
    ):
        modeling_multimodal_utils.get_multimodal_embeddings(MagicMock(name="encoder"), [param])

    drain.assert_not_called()


def test_no_drain_for_the_non_capturable_decoder():
    """The base decoder builds inline and enqueues no callback, so it owes no drain."""
    decoder = _decoder(GuidedDecoder)
    _run(lambda: decoder.execute(MagicMock(name="logits")), capturing=False)

    decoder.bitmask_event.synchronize.assert_not_called()


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
