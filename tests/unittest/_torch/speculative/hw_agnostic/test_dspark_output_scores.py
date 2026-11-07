# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for target-score selection and speculative result bookkeeping."""

import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative import spec_sampler_base as sampler_ops
from tensorrt_llm._torch.speculative.logprobs import (
    GATHER_IDS_KEY,
    HIDDEN_STATES_KEY,
    raw_logprobs,
    target_row_indices,
    token_logprobs,
)
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
)
from tensorrt_llm.sampling_params import LogprobMode

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dspark", [False, True])
def test_model_exports_original_states_without_changing_worker_output(dspark: bool) -> None:
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    gather_ids = torch.tensor([2, 3, 4])
    metadata = types.SimpleNamespace(gather_ids=gather_ids, is_layer_capture=lambda layer: False)
    worker_output = {"new_tokens": torch.tensor([[1, 2]])}

    class Worker:
        def target_logits(self, states, *args):
            assert torch.equal(states, hidden[gather_ids])
            return states.clone()

        def __call__(self, **kwargs):
            # A guided mask edits the sampling scores, not the original states.
            kwargs["logits"].fill_(float("-inf"))
            return worker_output

    model = types.SimpleNamespace(
        spec_decode_fast_paths=False,
        spec_worker=Worker(),
        spec_config=types.SimpleNamespace(
            spec_dec_mode=types.SimpleNamespace(is_dspark=lambda: dspark)
        ),
        model=lambda **kwargs: hidden,
        lm_head=None,
        logits_processor=None,
        draft_model=None,
        layer_idx=-1,
    )
    outputs = SpecDecOneEngineForCausalLM.forward(
        model,
        attn_metadata=types.SimpleNamespace(padded_num_tokens=None),
        input_ids=torch.arange(5),
        spec_metadata=metadata,
    )
    assert outputs is worker_output
    if dspark:
        assert outputs[HIDDEN_STATES_KEY] is hidden
        assert outputs[GATHER_IDS_KEY] is gather_ids
        assert torch.isfinite(hidden).all()
    else:
        assert HIDDEN_STATES_KEY not in outputs


def test_projection_runs_the_original_target_head_on_every_selected_row() -> None:
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    calls = []

    def project(hidden, head, metadata, all_rows):
        calls.append((head, metadata, all_rows))
        return hidden @ head

    head = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    model = types.SimpleNamespace(
        lm_head=head, logits_processor=types.SimpleNamespace(forward=project)
    )
    actual = SpecDecOneEngineForCausalLM.project_speculative_logits(model, states)
    assert calls[0][0] is head and calls[0][1:] == (None, True)
    torch.testing.assert_close(actual, states @ head)


@pytest.mark.parametrize(
    "tp_size,gather_output,ordinary_head,ordinary_processor,expected",
    [
        (1, False, True, True, True),
        (4, True, True, True, True),
        (4, False, True, True, False),
        (1, False, False, True, False),
        (1, False, True, False, False),
    ],
)
def test_capability_rejects_sharded_or_omitted_heads_before_projection(
    tp_size: int,
    gather_output: bool,
    ordinary_head: bool,
    ordinary_processor: bool,
    expected: bool,
) -> None:
    if ordinary_head:
        head = LMHead.__new__(LMHead)
        torch.nn.Module.__init__(head)
        head.tp_size = tp_size
        head.tp_mode = TensorParallelMode.COLUMN
        head.gather_output = gather_output
    else:
        head = torch.nn.Identity()
    processor = LogitsProcessor() if ordinary_processor else torch.nn.Identity()
    model = types.SimpleNamespace(lm_head=head, logits_processor=processor)
    supported = SpecDecOneEngineForCausalLM.supports_speculative_output_scores(model)
    assert supported is expected
    sampler = _sampler()
    if supported:
        sampler.logit_projection = lambda states: states
        sampler.validate_request(_request(py_return_log_probs=True))
    else:
        assert sampler.logit_projection is None
        with pytest.raises(ValueError, match="logprobs"):
            sampler.validate_request(_request(py_return_log_probs=True))


@pytest.mark.parametrize("ragged", [False, True])
def test_target_rows_keep_shuffled_requests_and_executed_windows(ragged: bool) -> None:
    windows = torch.tensor([1, 1, 2, 4, 1], dtype=torch.int32) if ragged else None
    rows, valid = target_row_indices(
        torch.tensor([4, 2, 1]),
        num_contexts=2,
        num_generations=3,
        width=4,
        verify_lens=windows,
    )
    if ragged:
        assert rows.tolist() == [[8, 8, 8, 8], [2, 3, 2, 2], [1, 1, 1, 1]]
        assert valid.tolist() == [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, False, False],
        ]
    else:
        assert rows.tolist() == [[10, 11, 12, 13], [2, 3, 4, 5], [1, 1, 1, 1]]


@pytest.mark.parametrize("top_k", [0, 1, 3])
def test_raw_scores_match_temperature_one_target_distribution(top_k: int) -> None:
    logits = torch.tensor([[[1001.0, 1000.0, 999.0, 998.0], [-3.0, -1.0, -2.0, -4.0]]])
    original = logits.clone()
    tokens = torch.tensor([[3, 2]])
    scores = raw_logprobs(logits, tokens, top_k)
    expected = logits.log_softmax(-1)
    torch.testing.assert_close(
        scores.sampled, expected.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    )
    expected_top, indices = expected.topk(top_k, dim=-1)
    torch.testing.assert_close(scores.top_values, expected_top)
    assert torch.equal(scores.top_indices, indices)
    assert scores.ranks.tolist() == [[3, 1]]
    assert torch.equal(logits, original)


def test_formatter_keeps_sampled_token_and_only_committed_prefix() -> None:
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]] * 3])
    scores = raw_logprobs(logits, torch.tensor([[3, 1, 2]]), 2)
    entries = token_logprobs(scores, 0, [3], top_k=2, simple_format=False)
    assert len(entries) == 1
    assert list(entries[0]) == [3, 0, 1]
    assert entries[0][3].rank == 4
    assert entries[0][0].rank == 1
    assert (
        token_logprobs(scores, 0, [3], top_k=0, simple_format=True)
        == scores.sampled[0, :1].tolist()
    )


def _request(**overrides: object) -> types.SimpleNamespace:
    fields = dict(
        py_return_log_probs=False,
        py_return_context_logits=False,
        py_return_generation_logits=False,
        py_logprobs_mode=LogprobMode.RAW,
        py_num_logprobs=0,
        py_logprobs_simple_format=False,
        sampling_config=None,
        context_chunk_size=0,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _sampler() -> SpecSampler:
    sampler = SpecSampler.__new__(SpecSampler)
    sampler._copy_to_host = lambda tensor: tensor.clone()
    return sampler


def test_projection_is_not_run_for_ordinary_requests() -> None:
    sampler = _sampler()
    scheduled = types.SimpleNamespace(context_requests=[])
    assert sampler._stage_requested_logits(scheduled, {}, [_request()], 2) == (None, [])


def test_output_validation_requires_a_bound_projection_and_raw_mode() -> None:
    sampler = _sampler()
    request = _request(py_return_log_probs=True)
    with pytest.raises(ValueError, match="logprobs"):
        sampler.validate_request(request)
    sampler.logit_projection = lambda tensor: tensor
    sampler.validate_request(request)
    sampler.validate_request(_request(py_return_context_logits=True))
    with pytest.raises(ValueError, match="Processed log probabilities"):
        sampler.validate_request(
            _request(py_return_log_probs=True, py_logprobs_mode=LogprobMode.PROCESSED)
        )
    with pytest.raises(ValueError, match="return_generation_logits"):
        sampler.validate_request(_request(py_return_generation_logits=True))


def test_processed_scores_are_rejected_before_submission() -> None:
    params = types.SimpleNamespace(logprobs=0, logprobs_mode=LogprobMode.PROCESSED)
    assert sampler_ops.one_model_sampling_rejection_reason(params) == (
        sampler_ops.UNSUPPORTED_PROCESSED_LOGPROBS_MSG
    )


def test_staging_selects_original_context_chunks_and_requested_target_rows() -> None:
    sampler = _sampler()
    projections = []

    def project(states: torch.Tensor) -> torch.Tensor:
        projections.append(states.clone())
        return states

    sampler.logit_projection = project
    context_chunks = []
    first = _request(
        py_return_context_logits=True,
        context_chunk_size=2,
        py_result=types.SimpleNamespace(append_context_logits=context_chunks.append),
    )
    final = _request(context_chunk_size=3, py_return_log_probs=True)
    generation = _request(py_return_log_probs=True, py_num_logprobs=2)
    scheduled = types.SimpleNamespace(
        context_requests=[first, final], context_requests_chunking=[first]
    )
    hidden = torch.arange(11 * 5, dtype=torch.float32).reshape(11, 5) / 7
    outputs = {
        HIDDEN_STATES_KEY: hidden,
        GATHER_IDS_KEY: torch.tensor([1, 4, 5, 6, 7, 8, 9, 10]),
        "new_tokens_lens": torch.tensor([1, 1, 2, 2]),
        "new_tokens": torch.tensor([[0, -1, -1], [3, -1, -1], [0, 1, -1], [2, 4, -1]]),
    }
    scores, indices = sampler._stage_requested_logits(
        scheduled, outputs, [final, _request(), generation], 2
    )
    assert indices == [0, 2]
    assert len(context_chunks) == 1
    assert torch.equal(context_chunks[0], hidden[:2])
    assert torch.equal(projections[1], hidden[torch.tensor([4, 4, 4, 8, 9, 10])])
    expected = hidden[[4, 8, 9]].log_softmax(-1)
    torch.testing.assert_close(scores.sampled[0, 0], expected[0, 3])
    torch.testing.assert_close(
        scores.sampled[1, :2], expected[1:].gather(-1, torch.tensor([[2], [4]])).flatten()
    )


def test_update_appends_scores_for_stop_truncated_prefix_and_shuffled_slots(monkeypatch) -> None:
    sampler = _sampler()
    sampler.max_accepted_path_len = 3
    sampler.max_seq_len = 100
    sampler.draft_len = 2
    recorded = [[], []]
    requests = [
        _request(
            py_return_log_probs=True,
            py_seq_slot=slot,
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            py_decoding_iter=0,
            py_result=types.SimpleNamespace(append_log_probs=recorded[index].append),
        )
        for index, slot in enumerate([3, 0])
    ]
    tokens = torch.zeros((3, 4, 1), dtype=torch.int32)
    tokens[:, 3, 0] = torch.tensor([2, 1, 0])
    tokens[:, 0, 0] = torch.tensor([1, 0, 3])
    host = SampleStateTensorsSpec(
        new_tokens=tokens,
        new_tokens_lens=torch.tensor([2, 0, 0, 3]),
        next_draft_tokens=torch.zeros((4, 2), dtype=torch.int32),
    )
    scores = raw_logprobs(
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        torch.tensor([[2, 1, 0], [1, 0, 3]]),
        0,
    )
    state = SampleStateSpec(
        requests=requests,
        device=host,
        host=host,
        sampler_event=types.SimpleNamespace(synchronize=lambda: None),
        runtime_draft_len=2,
        draft_lens=[2, 2],
        raw_logprobs=scores,
        logprob_request_indices=[0, 1],
    )
    monkeypatch.setattr(
        sampler_ops,
        "add_token",
        lambda request, values, beam_idx, step: values[step][request.py_seq_slot][beam_idx],
    )
    monkeypatch.setattr(
        sampler_ops, "handle_stop_criteria", lambda request, token, **kwargs: token == 2
    )
    sampler.update_requests(state)
    assert len(recorded[0][0][0]) == 1
    assert len(recorded[1][0][0]) == 2
    assert list(recorded[0][0][0][0]) == [2]
    assert list(recorded[1][0][0][1]) == [0]
