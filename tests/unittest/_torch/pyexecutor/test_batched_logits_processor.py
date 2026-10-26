# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine's logits post-processor walk with ``LLM(batched_logits_processor=...)``.

Requests that set ``SamplingParams.apply_batched_logits_processor`` are collected
as the walk visits them and handed to the batched processor in one call, after
their own processors, with views into the step's logits tensor.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed as distributed
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

VOCAB = 8


class _Recorder:
    """Stands in for a BatchedLogitsProcessor: records each call and adds ``mark`` to every row."""

    def __init__(self, mark=1.0):
        self.calls = []
        self.mark = mark

    def __call__(self, req_ids, logits, token_ids, stream_ptr, client_ids):
        self.calls.append((req_ids, logits, token_ids, stream_ptr, client_ids))
        for rows in logits:
            rows += self.mark


def _request(
    req_id,
    *,
    batched=False,
    processors=None,
    beam_width=1,
    admitted_beam_width=None,
    tokens=(1, 2, 3),
    orig_prompt_len=None,
):
    tokens = list(tokens)
    return SimpleNamespace(
        py_request_id=req_id,
        py_client_id=100 + req_id,
        py_logits_post_processors=processors,
        py_apply_batched_logits_processor=batched,
        py_beam_width=admitted_beam_width or beam_width,
        py_prompt_len=len(tokens),
        py_orig_prompt_len=len(tokens) if orig_prompt_len is None else orig_prompt_len,
        get_beam_width_by_iter=lambda for_next_iteration: beam_width,
        get_tokens=lambda beam: list(tokens),
    )


def _engine(processor, *, attention_dp=False, vocab_size=None):
    engine = object.__new__(DecoderRunner)
    engine._mapping = SimpleNamespace(
        is_last_pp_rank=lambda: True, enable_attention_dp=attention_dp
    )
    engine._model = SimpleNamespace(lm_head=SimpleNamespace(num_embeddings=vocab_size))
    engine.batched_logits_processor = processor
    engine.start_beam_histories_callable = None
    return engine


def _scheduled(context=(), generation=()):
    return SimpleNamespace(
        context_requests=list(context),
        generation_requests=list(generation),
        all_requests=lambda: [*context, *generation],
    )


def _logits(rows):
    # Row r holds the value r everywhere, so edits are attributable to a row.
    return torch.arange(rows, dtype=torch.float32)[:, None].repeat(1, VOCAB)


def test_flagged_requests_share_one_call_in_scheduling_order():
    processor = _Recorder()
    context = _request(1, batched=True, tokens=(4, 5))
    plain = _request(2)
    generation = _request(3, batched=True, tokens=(6, 7, 8))
    logits = _logits(3)

    _engine(processor)._execute_logit_post_processors(
        _scheduled(context=[context], generation=[plain, generation]), {"logits": logits}
    )

    assert len(processor.calls) == 1
    req_ids, rows, token_ids, stream_ptr, client_ids = processor.calls[0]
    assert req_ids == [1, 3]
    assert client_ids == [101, 103]
    assert token_ids == [[[4, 5]], [[6, 7, 8]]]
    assert [tuple(r.shape) for r in rows] == [(1, 1, VOCAB), (1, 1, VOCAB)]
    assert stream_ptr is None  # the logits are not on a CUDA device
    # The callback's in-place edits land in the step's logits; the other request's row is untouched.
    assert torch.equal(logits[:, 0], torch.tensor([1.0, 1.0, 3.0]))


def test_per_request_processors_run_before_the_batched_one():
    order = []

    def own(req_id, rows, token_ids, stream_ptr, client_id):
        order.append(("own", req_id, rows[0, 0, 0].item()))
        rows.fill_(5.0)

    batched = _Recorder()
    request = _request(1, batched=True, processors=[own])
    logits = _logits(1)

    _engine(batched)._execute_logit_post_processors(
        _scheduled(generation=[request]), {"logits": logits}
    )

    assert order == [("own", 1, 0.0)]
    assert batched.calls[0][0] == [1]
    assert torch.equal(logits[0], torch.full((VOCAB,), 6.0))  # 5 from `own`, then +1


def test_beam_rows_and_histories_reach_the_batched_processor():
    processor = _Recorder()
    first = _request(1, batched=True, beam_width=2, admitted_beam_width=4, tokens=(1, 2))
    second = _request(2, batched=True, beam_width=1, admitted_beam_width=4, tokens=(3,))
    logits = _logits(8)
    histories = SimpleNamespace(wait=lambda: [[[7], [8]]])

    _engine(processor)._execute_logit_post_processors(
        _scheduled(generation=[first, second]),
        {"logits": logits},
        pending_beam_histories=([first], histories),
    )

    req_ids, rows, token_ids, _, _ = processor.calls[0]
    assert req_ids == [1, 2]
    # Live beams only, and the prompt plus each beam's own hypothesis.
    assert [tuple(r.shape) for r in rows] == [(2, 1, VOCAB), (1, 1, VOCAB)]
    assert token_ids == [[[1, 2, 7], [1, 2, 8]], [[3]]]
    # The second request's rows start at the admission width, not the live one.
    assert torch.equal(logits[:, 0], torch.tensor([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0]))


def test_an_unfinished_context_chunk_is_not_handed_over():
    processor = _Recorder()
    chunk = _request(1, batched=True, tokens=(1, 2, 3), orig_prompt_len=2)

    _engine(processor)._execute_logit_post_processors(
        _scheduled(context=[chunk]), {"logits": _logits(1)}
    )

    assert processor.calls == []


def test_ragged_verification_is_refused_for_batched_only_requests():
    processor = _Recorder()
    request = _request(1, batched=True)
    request.py_verify_len = 2

    with pytest.raises(RuntimeError, match="DSpark ragged verification"):
        _engine(processor)._execute_logit_post_processors(
            _scheduled(generation=[request]), {"logits": _logits(4)}
        )

    assert processor.calls == []


def test_without_opted_in_requests_the_processor_is_not_called():
    processor = _Recorder()
    request = _request(1, processors=[lambda req_id, rows, tokens, stream_ptr, client_id: None])

    _engine(processor)._execute_logit_post_processors(
        _scheduled(generation=[request]), {"logits": _logits(1)}
    )

    assert processor.calls == []


def test_the_flag_without_a_registered_processor_is_skipped_with_a_warning(monkeypatch):
    # LLM refuses it earlier; a direct caller gets a warning, not a failed step.
    import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner as runner_module

    warnings = []
    monkeypatch.setattr(
        runner_module.logger, "warning_once", lambda msg, key=None: warnings.append(key)
    )
    logits = _logits(1)

    _engine(None)._execute_logit_post_processors(
        _scheduled(generation=[_request(1, batched=True)]), {"logits": logits}
    )

    assert torch.equal(logits, _logits(1))
    assert warnings == ["batched_logits_processor_unregistered"]


def test_beam_histories_start_for_batched_only_requests():
    started = []
    engine = _engine(_Recorder())
    engine.start_beam_histories_callable = lambda requests: started.append(requests) or "pending"
    beam = _request(1, batched=True, beam_width=2)
    single = _request(2, batched=True, beam_width=1)
    plain = _request(3, beam_width=2)

    result = engine._start_beam_histories(_scheduled(generation=[beam, single, plain]))

    assert result == ([beam], "pending")
    assert started == [[beam]]


def test_sharded_logits_are_gathered_for_the_batched_processor(monkeypatch):
    calls = []

    def allgather(tensor, mapping, dim=-1):
        calls.append(tensor)
        return torch.zeros(tensor.shape[0], 32)

    monkeypatch.setattr(distributed, "allgather", allgather)
    processor = _Recorder()
    shard = torch.ones(1, 8)

    _engine(processor)._execute_logit_post_processors(
        _scheduled(generation=[_request(1, batched=True)]),
        {"logits": shard, "logits_vocab_shard": True},
    )

    assert calls == [shard]
    assert [tuple(r.shape) for r in processor.calls[0][1]] == [(1, 1, 32)]


def test_the_flag_is_refused_without_a_batched_processor_on_the_llm():
    llm = SimpleNamespace(
        args=SimpleNamespace(batched_logits_processor=None, enable_chunked_prefill=False),
        _check_one_model_speculative_sampling=lambda params: None,
    )
    params = SamplingParams(apply_batched_logits_processor=True)

    with pytest.raises(ValueError, match="batched_logits_processor"):
        BaseLLM._check_arguments(llm, 3, params, is_gen_only=True)

    llm.args.batched_logits_processor = _Recorder()
    BaseLLM._check_arguments(llm, 3, params, is_gen_only=True)
