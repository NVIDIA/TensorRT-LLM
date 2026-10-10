# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-model spec worker's target-logits hook and the engine gather of vocabulary-sharded logits."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed as distributed
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only


class _RecordingLogitsProcessor:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def forward(self, *args):
        self.calls.append(args)
        return self.logits


def test_default_target_logits_are_the_logits_processors():
    logits = torch.randn(3, 16)
    processor = _RecordingLogitsProcessor(logits)
    hidden, lm_head, attn_metadata = torch.randn(3, 8), object(), object()

    out = SpecWorkerBase.target_logits(
        object(), hidden, lm_head, processor, attn_metadata, object(), object()
    )

    assert out is logits
    assert len(processor.calls) == 1
    args = processor.calls[0]
    assert args[0] is hidden and args[1] is lm_head and args[2] is attn_metadata
    assert args[3] is True


def _generation_request(post_processors, tokens=(1, 2, 3)):
    return SimpleNamespace(
        py_request_id=7,
        py_logits_post_processors=post_processors,
        py_beam_width=1,
        get_beam_width_by_iter=lambda for_next_iteration: 1,
        get_tokens=lambda beam: list(tokens),
    )


def _engine(vocab_size=None, attention_dp=False):
    engine = object.__new__(DecoderRunner)
    engine.mapping = SimpleNamespace(is_last_pp_rank=lambda: True, enable_attention_dp=attention_dp)
    engine.model = SimpleNamespace(lm_head=SimpleNamespace(num_embeddings=vocab_size))
    return engine


def _scheduled(*requests):
    return SimpleNamespace(
        context_requests=[], generation_requests=list(requests), all_requests=lambda: list(requests)
    )


def _recording_allgather(monkeypatch, vocab):
    calls = []

    def allgather(tensor, mapping, dim=-1):
        calls.append((tensor, mapping, dim))
        return torch.zeros(tensor.shape[0], vocab, dtype=torch.bfloat16)

    monkeypatch.setattr(distributed, "allgather", allgather)
    return calls


def test_sharded_logits_are_gathered_for_post_processors(monkeypatch):
    calls = _recording_allgather(monkeypatch, vocab=32)
    seen = []
    request = _generation_request(
        [lambda req_id, rows, tokens, s, c: seen.append(tuple(rows.shape))]
    )
    shard = torch.ones(1, 8, dtype=torch.bfloat16)
    engine = _engine()

    engine._execute_logit_post_processors(
        _scheduled(request), {"logits": shard, "logits_vocab_shard": True}
    )

    assert len(calls) == 1
    assert calls[0][0] is shard and calls[0][1] is engine.mapping and calls[0][2] == -1
    assert seen == [(1, 1, 32)]  # the post-processor sees the whole vocabulary


def test_sharded_logits_without_post_processors_are_not_gathered(monkeypatch):
    calls = _recording_allgather(monkeypatch, vocab=32)

    _engine()._execute_logit_post_processors(
        _scheduled(_generation_request(None)),
        {"logits": torch.ones(1, 8, dtype=torch.bfloat16), "logits_vocab_shard": True},
    )

    assert calls == []


def test_full_logits_are_never_gathered(monkeypatch):
    calls = _recording_allgather(monkeypatch, vocab=32)
    seen = []
    request = _generation_request(
        [lambda req_id, rows, tokens, s, c: seen.append(tuple(rows.shape))]
    )

    _engine()._execute_logit_post_processors(_scheduled(request), {"logits": torch.ones(1, 32)})

    assert calls == []
    assert seen == [(1, 1, 32)]


def test_sharded_logits_drop_the_vocab_padding(monkeypatch):
    # 4 ranks x 8 columns hold a 30-token vocabulary: the gather's last 2 columns are TP padding.
    _recording_allgather(monkeypatch, vocab=32)
    seen = []
    request = _generation_request(
        [lambda req_id, rows, tokens, s, c: seen.append(tuple(rows.shape))]
    )

    _engine(vocab_size=30)._execute_logit_post_processors(
        _scheduled(request),
        {"logits": torch.ones(1, 8, dtype=torch.bfloat16), "logits_vocab_shard": True},
    )

    assert seen == [(1, 1, 30)]


def test_sharded_logits_under_attention_dp_raise(monkeypatch):
    calls = _recording_allgather(monkeypatch, vocab=32)
    request = _generation_request([lambda *args: None])

    with pytest.raises(RuntimeError, match="attention DP"):
        _engine(attention_dp=True)._execute_logit_post_processors(
            _scheduled(request),
            {"logits": torch.ones(1, 8, dtype=torch.bfloat16), "logits_vocab_shard": True},
        )
    assert calls == []


def test_the_spec_model_hands_the_workers_target_logits_to_its_forward():
    hidden = torch.randn(4, 8)
    gather_ids = torch.tensor([1, 3])
    calls = {}

    class Worker:
        def target_logits(self, *args):
            calls["target_logits"] = args
            return "worker logits"

        def __call__(self, **kwargs):
            calls["forward"] = kwargs
            return "worker outputs"

    model = object.__new__(SpecDecOneEngineForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = lambda **kwargs: hidden
    model.layer_idx = -1
    model.spec_worker = Worker()
    model.lm_head, model.logits_processor, model.draft_model = "lm_head", "processor", "drafter"
    spec_metadata = SimpleNamespace(is_layer_capture=lambda layer: False, gather_ids=gather_ids)
    attn_metadata = SimpleNamespace(padded_num_tokens=None, num_tokens=4)

    out = SpecDecOneEngineForCausalLM.forward(
        model, attn_metadata, input_ids=torch.arange(4), spec_metadata=spec_metadata
    )

    assert out == "worker outputs"
    rows, *rest = calls["target_logits"]
    assert torch.equal(rows, hidden[gather_ids])
    assert rest == ["lm_head", "processor", attn_metadata, spec_metadata, "drafter"]
    assert calls["forward"]["logits"] == "worker logits"
