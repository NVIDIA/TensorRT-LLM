# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
from tensorrt_llm._torch.speculative import eagle3
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.utils import (
    get_num_extra_kv_tokens,
    get_num_spec_layers,
    update_spec_config_from_model_config,
)
from tensorrt_llm.llmapi import MTPDecodingConfig


def _shared_kv_spec_config(**kwargs) -> MTPDecodingConfig:
    spec_config = MTPDecodingConfig(
        max_draft_len=kwargs.pop("max_draft_len", 3),
        speculative_model="/tmp/gemma4-assistant",
        mtp_eagle_one_model=True,
        **kwargs,
    )
    spec_config._use_shared_kv_cache = True
    return spec_config


def test_external_checkpoint_does_not_imply_shared_kv_cache():
    spec_config = MTPDecodingConfig(
        max_draft_len=3,
        speculative_model="/tmp/assistant",
        mtp_eagle_one_model=True,
    )
    model_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        num_nextn_predict_layers=1,
    )

    update_spec_config_from_model_config(spec_config, model_config)

    assert not spec_config._use_shared_kv_cache
    assert get_num_spec_layers(spec_config) == 1
    assert get_num_extra_kv_tokens(spec_config) == 2
    assert should_use_separate_draft_kv_cache(spec_config)


@pytest.mark.parametrize("one_model,expected", [(True, True), (False, False)])
def test_gemma4_config_sets_shared_kv_cache_for_one_model_only(
    one_model,
    expected,
):
    spec_config = MTPDecodingConfig(
        max_draft_len=3,
        speculative_model="/tmp/gemma4-assistant",
        mtp_eagle_one_model=one_model,
    )
    model_config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        num_nextn_predict_layers=1,
    )

    update_spec_config_from_model_config(spec_config, model_config)

    assert spec_config._use_shared_kv_cache is expected


def test_external_shared_kv_uses_no_draft_kv_cache():
    spec_config = _shared_kv_spec_config()

    assert spec_config._use_shared_kv_cache
    assert get_num_spec_layers(spec_config) == 0
    assert get_num_extra_kv_tokens(spec_config) == 0
    assert not should_use_separate_draft_kv_cache(spec_config)


def test_external_shared_kv_builds_draft_from_external_config(monkeypatch):
    draft_config = object()
    expected_model = object()
    monkeypatch.setattr(
        modeling_speculative.AutoModelForCausalLM,
        "from_config",
        lambda config: expected_model,
    )

    model_config = SimpleNamespace(spec_config=_shared_kv_spec_config())
    assert (
        modeling_speculative.get_draft_model(
            model_config,
            draft_config,
            lm_head=None,
            model=None,
        )
        is expected_model
    )


def test_shared_kv_alias_setup_rebinds_target_model():
    calls = []
    draft_model = SimpleNamespace(
        shares_target_kv_cache=True,
        load_weights_from_target_model=lambda target: calls.append(target),
    )
    model = SimpleNamespace(draft_model=draft_model)

    modeling_speculative.SpecDecOneEngineForCausalLM.setup_aliases(model)

    assert calls == [model]

    model.draft_model = SimpleNamespace(shares_target_kv_cache=True)
    modeling_speculative.SpecDecOneEngineForCausalLM.setup_aliases(model)


def test_external_shared_kv_worker_uses_config_and_supports_guided_decoding():
    worker = MTPEagleWorker(_shared_kv_spec_config())
    guided_decoder = object()

    assert worker._uses_external_shared_target_kv
    assert worker.set_guided_decoder(guided_decoder)
    assert worker.guided_decoder is guided_decoder
    worker.set_draft_model(SimpleNamespace(model=SimpleNamespace()))


def test_external_shared_kv_draft_loop_applies_guided_decoding(monkeypatch):
    draft_metadata = SimpleNamespace(
        update_shared_kv_draft_lengths=Mock(),
    )

    class FakeFlashInferAttentionMetadata:
        def __init__(self):
            self.seq_lens_cuda = torch.tensor([2, 2], dtype=torch.int32)

        def get_draft_metadata(self):
            return draft_metadata

    monkeypatch.setattr(eagle3, "FlashInferAttentionMetadata", FakeFlashInferAttentionMetadata)
    worker = MTPEagleWorker(_shared_kv_spec_config(max_draft_len=2))
    guided_decoder = SimpleNamespace(
        add_draft_batch=Mock(),
        execute_draft_batch=Mock(),
    )
    worker.set_guided_decoder(guided_decoder)

    sampled_tokens = [
        torch.tensor([41, 42], dtype=torch.int32),
        torch.tensor([51, 52], dtype=torch.int32),
    ]
    monkeypatch.setattr(
        worker,
        "sample_draft_tokens",
        lambda *args, **kwargs: sampled_tokens.pop(0),
    )
    draft_model = SimpleNamespace(
        forward_draft_step=lambda **kwargs: (torch.zeros(2, 4), kwargs["recurrent_hidden_states"])
    )
    attn_metadata = FakeFlashInferAttentionMetadata()
    spec_metadata = SimpleNamespace(
        batch_indices_cuda=torch.arange(2),
        runtime_draft_len=2,
        subseq_all_rank_num_tokens=None,
    )
    accepted_tokens = torch.tensor(
        [[10, 11, 12], [20, 21, 22]],
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.ones(2, dtype=torch.long)

    next_draft_tokens = worker._forward_external_shared_target_kv_draft_loop(
        position_ids=torch.arange(4, dtype=torch.int32),
        hidden_states=torch.arange(8, dtype=torch.float32).unsqueeze(1),
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        draft_model=draft_model,
        accepted_tokens=accepted_tokens,
        num_accepted_tokens=num_accepted_tokens,
        num_contexts=1,
        batch_size=2,
    )

    assert torch.equal(
        next_draft_tokens,
        torch.tensor([[41, 51], [42, 52]], dtype=torch.int32),
    )
    assert [
        call.kwargs["draft_step"] for call in guided_decoder.add_draft_batch.call_args_list
    ] == [0, 1]
    assert [
        call.kwargs["draft_step"] for call in guided_decoder.execute_draft_batch.call_args_list
    ] == [0, 1]


@pytest.mark.parametrize(
    "num_accepted_tokens, expected_hidden_rows",
    [
        ([1, 1, 1], [1, 2, 6]),
        ([1, 2, 3], [1, 3, 8]),
        ([1, 4, 4], [1, 5, 9]),
    ],
)
def test_external_shared_kv_selects_last_accepted_target_state(
    num_accepted_tokens,
    expected_hidden_rows,
):
    accepted_tokens = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.int32,
    )
    accepted_counts = torch.tensor(num_accepted_tokens, dtype=torch.long)
    hidden_states = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    position_ids = torch.arange(10, dtype=torch.int32).unsqueeze(0)

    draft_ids, recurrent_hidden, draft_positions = (
        MTPEagleWorker._prepare_external_shared_target_kv_draft_inputs(
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=accepted_counts,
            hidden_states=hidden_states,
            position_ids=position_ids,
            sequence_lengths=torch.tensor([2, 4, 4]),
            num_contexts=1,
            batch_indices=torch.arange(3),
        )
    )

    expected_tokens = accepted_tokens[
        torch.arange(3),
        accepted_counts - 1,
    ]
    assert torch.equal(draft_ids, expected_tokens)
    assert torch.equal(
        recurrent_hidden.squeeze(1),
        torch.tensor(expected_hidden_rows, dtype=torch.float32),
    )
    assert torch.equal(
        draft_positions,
        torch.tensor(expected_hidden_rows, dtype=torch.int32).unsqueeze(0) + 1,
    )


def test_gemma4_target_forward_dispatches_one_model_worker():
    hidden_states = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    worker_calls = []

    def spec_worker(**kwargs):
        worker_calls.append(kwargs)
        return {"logits": kwargs["logits"], "new_tokens": torch.tensor([[7]])}

    model = SimpleNamespace(
        layer_idx=-1,
        config=SimpleNamespace(final_logit_softcapping=None),
        model=lambda **kwargs: hidden_states,
        logits_processor=SimpleNamespace(forward=lambda selected, *args: selected),
        lm_head=object(),
        spec_worker=spec_worker,
        draft_model=object(),
    )
    spec_metadata = SimpleNamespace(
        gather_ids=torch.tensor([2]),
        is_layer_capture=lambda layer_idx: False,
    )
    attn_metadata = SimpleNamespace(padded_num_tokens=None)

    outputs = Gemma4ForCausalLM.forward(
        model,
        attn_metadata=attn_metadata,
        input_ids=torch.tensor([1, 2, 3]),
        position_ids=torch.tensor([[0, 1, 2]]),
        spec_metadata=spec_metadata,
    )

    assert torch.equal(outputs["new_tokens"], torch.tensor([[7]]))
    assert len(worker_calls) == 1
    assert torch.equal(worker_calls[0]["hidden_states"], hidden_states)
    assert torch.equal(worker_calls[0]["logits"], hidden_states[[2]])
    assert worker_calls[0]["draft_model"] is model.draft_model
