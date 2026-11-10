# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logits rows of context requests, as the sampler and the logits post-processors read them.

Without the context-logits plan (always the case when a request has a
per-request logits processor), a step that gathers context logits projects every
context token. The requested context logits are stored from those rows before
any post-processor runs, leaving one sampling row per context request, and a
post-processor edits the row its request is sampled from.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.context_logits_plan import (
    context_logits_rows,
    store_context_logits_from_full_rows,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

VOCAB = 4

pytestmark = pytest.mark.cpu_only


def _context(chunk_size, *, returns_context_logits=False, processors=None, last_chunk=True):
    return SimpleNamespace(
        py_request_id=chunk_size,
        context_chunk_size=chunk_size,
        prepopulated_prompt_len=0,
        py_prompt_logprobs=None,
        py_context_logits_explicit=True,
        py_result=Mock(),
        py_return_context_logits=returns_context_logits,
        py_logits_post_processors=processors,
        is_last_context_chunk=last_chunk,
        get_tokens=lambda beam: [1, 2, 3],
    )


def _generation(processors=None):
    return SimpleNamespace(
        py_request_id=99,
        py_beam_width=1,
        py_logits_post_processors=processors,
        get_beam_width_by_iter=lambda for_next_iteration: 1,
        get_tokens=lambda beam: [1, 2, 3, 4],
    )


def _rows(count):
    # Row i holds i in every column, so a row's value names its position.
    return torch.arange(count, dtype=torch.float32)[:, None].repeat(1, VOCAB)


def _runner():
    runner = object.__new__(DecoderRunner)
    runner._mapping = SimpleNamespace(is_last_pp_rank=lambda: True, enable_attention_dp=False)
    return runner


def _batch(context=(), generation=()):
    context, generation = list(context), list(generation)
    return SimpleNamespace(
        context_requests=context,
        generation_requests=generation,
        all_requests=lambda: context + generation,
    )


def _recorder(seen):
    def processor(req_id, logits, token_ids, stream_ptr, client_id):
        seen.append((req_id, logits[:, 0, 0].tolist()))
        logits.fill_(-1.0)

    return processor


@pytest.mark.parametrize("handled", [False, True])
def test_rows_follow_whether_the_request_returns_context_logits(handled):
    assert context_logits_rows(_context(5), context_logits_handled=handled) == 1
    expected = 1 if handled else 5
    assert (
        context_logits_rows(
            _context(5, returns_context_logits=True), context_logits_handled=handled
        )
        == expected
    )


def _stored_rows(request):
    (stored,), kwargs = request.py_result.append_context_logits.call_args
    assert kwargs == {"reusable": True}
    return stored[:, 0].tolist()


def test_requested_chunks_are_stored_and_one_row_per_context_request_is_left():
    # Context rows 0-2 belong to the requester, 3-4 and 5 to the others; 6-7 are
    # generation rows.
    requester = _context(3, returns_context_logits=True)
    others = [_context(2), _context(1)]

    kept = store_context_logits_from_full_rows([requester, *others], _rows(8))

    assert _stored_rows(requester) == [0, 1, 2]
    assert all(not other.py_result.append_context_logits.called for other in others)
    assert kept[:, 0].tolist() == [2, 4, 5, 6, 7]


def test_editing_a_sampling_row_leaves_the_stored_context_logits_alone():
    requester = _context(2, returns_context_logits=True)

    kept = store_context_logits_from_full_rows([requester], _rows(2))
    kept.fill_(-1.0)

    assert _stored_rows(requester) == [0, 1]


def test_fewer_rows_than_context_tokens_are_refused():
    with pytest.raises(ValueError, match="fewer than the 4 scheduled context tokens"):
        store_context_logits_from_full_rows([_context(4)], _rows(2))


def test_a_processor_edits_the_row_its_request_is_sampled_from():
    seen = []
    requester = _context(3, returns_context_logits=True, processors=[_recorder(seen)])
    other = _context(2, processors=[_recorder(seen)])
    generation = _generation(processors=[_recorder(seen)])
    # As the sampler reads them: the requester's 3 rows, the other's last row,
    # then the generation row.
    logits = _rows(5)

    _runner()._execute_logit_post_processors(
        _batch(context=[requester, other], generation=[generation]), {"logits": logits}
    )

    assert seen == [(3, [2.0]), (2, [3.0]), (99, [4.0])]
    # The requester's earlier prompt rows stay as the context logits returned.
    assert logits[:, 0].tolist() == [0.0, 1.0, -1.0, -1.0, -1.0]


def test_stored_context_logits_leave_one_row_per_context_request():
    seen = []
    requester = _context(3, returns_context_logits=True, processors=[_recorder(seen)])
    generation = _generation(processors=[_recorder(seen)])

    _runner()._execute_logit_post_processors(
        _batch(context=[requester], generation=[generation]),
        {"logits": _rows(2), "context_logits_handled": True},
    )

    assert seen == [(3, [0.0]), (99, [1.0])]


@pytest.mark.parametrize("last_chunk", [False, True])
def test_only_the_last_context_chunk_is_processed(last_chunk):
    # get_tokens(0) holds more than the original prompt both during chunked
    # prefill and after a pause folds generated tokens into the prompt, so only
    # the chunk bookkeeping decides.
    seen = []
    request = _context(1, processors=[_recorder(seen)], last_chunk=last_chunk)

    _runner()._execute_logit_post_processors(_batch(context=[request]), {"logits": _rows(1)})

    assert seen == ([(1, [0.0])] if last_chunk else [])
