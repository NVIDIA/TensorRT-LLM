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
"""When the DFlash/DSpark worker selects tokens vocab-parallel, and when not."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.speculative import vocab_parallel as vp
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB = 4096


def _lm_head(tp_size=2, **overrides):
    head = SimpleNamespace(
        mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=0),
        tp_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        padding_size=0,
        has_any_quant=False,
        out_features=VOCAB // tp_size,
        num_embeddings=VOCAB,
        bias=None,
    )
    for key, value in overrides.items():
        setattr(head, key, value)
    return head


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv(vp.VOCAB_PARALLEL_COMM_ENV_VAR, "nccl")
    spec_config = SimpleNamespace(
        max_draft_len=6,
        max_total_draft_tokens=6,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        draft_is_embedded_in_target=False,
        attention_backend="TRTLLM",
    )
    return DSparkWorker(spec_config, Mapping())


def test_target_head_follows_the_switch(worker, monkeypatch):
    lm_head = _lm_head()
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "0")
    assert worker.target_vocab_parallel_head(lm_head) is None
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "1")
    head = worker.target_vocab_parallel_head(lm_head)
    assert isinstance(head, vp.VocabParallelHead)
    assert head.shard_width == VOCAB // 2 and head.vocab_size == VOCAB
    assert worker.target_vocab_parallel_head(lm_head) is head  # cached
    worker.guided_decoder = object()
    assert worker.target_vocab_parallel_head(lm_head) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tp_mode=TensorParallelMode.ROW),
        dict(gather_output=False),
        dict(padding_size=3),
        dict(has_any_quant=True),
        dict(out_features=VOCAB // 2 - 2),
        dict(mapping=Mapping(world_size=1, tp_size=1, rank=0), out_features=VOCAB),
    ],
)
def test_unsupported_heads_keep_gathered_logits(worker, monkeypatch, overrides):
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "1")
    assert worker.target_vocab_parallel_head(_lm_head(**overrides)) is None


def _metadata(greedy, rejection=False, mode=AdvancedSamplingMode.FULL):
    return SimpleNamespace(
        is_all_greedy_sample=greedy,
        use_rejection_sampling=rejection,
        advanced_sampling_mode=mode,
        wants_advanced_draft_sampling=rejection and not greedy,
    )


def test_target_selection_covers_greedy_and_temperature_only(worker, monkeypatch):
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "1")
    applies = worker._vocab_parallel_target_selection_applies
    no_filters = AdvancedSamplingMode.NO_TOPK_NO_TOPP
    assert applies(_metadata(True))
    assert applies(_metadata(True, rejection=True))
    monkeypatch.setenv(vp.VOCAB_PARALLEL_SAMPLING_ENV_VAR, "0")
    assert not applies(_metadata(False, mode=no_filters))
    monkeypatch.setenv(vp.VOCAB_PARALLEL_SAMPLING_ENV_VAR, "1")
    assert applies(_metadata(False, mode=no_filters))
    assert not applies(_metadata(False, mode=AdvancedSamplingMode.FULL))
    assert not applies(_metadata(False, mode=AdvancedSamplingMode.NO_TOPK))
    assert not applies(_metadata(False, rejection=True, mode=no_filters))


def _draft_model(markov=True, contiguous=True):
    w1 = torch.randn(VOCAB, 64, device="cuda").to(torch.bfloat16)
    w2 = torch.randn(VOCAB, 64, device="cuda").to(torch.bfloat16)
    if not contiguous:
        w2 = torch.randn(64, VOCAB, device="cuda").to(torch.bfloat16).t()
    return SimpleNamespace(
        lm_head=_lm_head(),
        has_markov_head=markov,
        markov_w1=w1 if markov else None,
        markov_w2=w2 if markov else None,
    )


def test_draft_head_selection(worker, monkeypatch):
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "1")
    model = _draft_model()
    assert worker._draft_vocab_parallel_head(model, _metadata(True)) is not None
    assert worker._vocab_parallel_markov_weights(model) == (model.markov_w1, model.markov_w2)
    # Non-greedy target sampling still drafts greedily (no rejection sampling).
    assert worker._draft_vocab_parallel_head(model, _metadata(False)) is not None
    assert worker._draft_vocab_parallel_head(model, _metadata(False, rejection=True)) is None
    assert (
        worker._draft_vocab_parallel_head(_draft_model(contiguous=False), _metadata(True)) is None
    )
    plain = _draft_model(markov=False)
    assert worker._draft_vocab_parallel_head(plain, _metadata(True)) is not None
    assert worker._vocab_parallel_markov_weights(plain) == (None, None)
    monkeypatch.setenv(vp.VOCAB_PARALLEL_ENV_VAR, "0")
    worker._vocab_parallel_heads.clear()
    assert worker._draft_vocab_parallel_head(model, _metadata(True)) is None
