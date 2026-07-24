# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.interface import (
    DraftModelCapabilities,
    needs_external_draft_weights,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._torch.speculative.utils import (
    get_num_draft_kv_layers,
    get_num_extra_kv_tokens,
    get_num_spec_layers,
)
from tensorrt_llm.llmapi import MTPDecodingConfig


def _shared_kv_capabilities() -> DraftModelCapabilities:
    return DraftModelCapabilities.external_shared_target_kv()


def _shared_kv_spec_config(**kwargs) -> MTPDecodingConfig:
    spec_config = MTPDecodingConfig(
        max_draft_len=kwargs.pop("max_draft_len", 3),
        speculative_model="/tmp/gemma4-assistant",
        mtp_eagle_one_model=True,
        **kwargs,
    )
    spec_config._draft_model_capabilities = _shared_kv_capabilities()
    return spec_config


def test_external_shared_kv_capability_separates_module_and_kv_counts():
    spec_config = _shared_kv_spec_config()

    assert needs_external_draft_weights(spec_config)
    assert get_num_spec_layers(spec_config) == 1
    assert get_num_draft_kv_layers(spec_config) == 0
    assert get_num_extra_kv_tokens(spec_config) == 0
    assert not should_use_separate_draft_kv_cache(spec_config)


def test_embedded_one_model_mtp_does_not_load_external_weights():
    spec_config = _shared_kv_spec_config()
    spec_config._draft_model_capabilities = None

    assert not needs_external_draft_weights(spec_config)


def test_external_shared_kv_worker_rejects_unverified_modes():
    spec_config = _shared_kv_spec_config(
        use_dynamic_tree=True,
        dynamic_tree_max_topK=2,
    )
    worker = MTPEagleWorker(spec_config)

    with pytest.raises(ValueError, match="linear draft path"):
        worker.set_draft_model(SimpleNamespace(model=SimpleNamespace()))


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


def test_gemma4_target_forward_still_captures_hidden_states_without_worker():
    model = SimpleNamespace(
        layer_idx=-1,
        config=SimpleNamespace(final_logit_softcapping=None),
        model=lambda **kwargs: torch.tensor([[1.0, 2.0]]),
        logits_processor=SimpleNamespace(forward=lambda hidden_states, *args: hidden_states),
        lm_head=object(),
        spec_worker=None,
    )
    captured = []
    spec_metadata = SimpleNamespace(
        is_layer_capture=lambda layer_idx: layer_idx == -1,
        maybe_capture_hidden_states=lambda layer_idx, hidden_states: captured.append(
            (layer_idx, hidden_states.clone())
        ),
    )
    attn_metadata = SimpleNamespace(padded_num_tokens=None)

    output = Gemma4ForCausalLM.forward(
        model,
        attn_metadata=attn_metadata,
        input_ids=torch.tensor([1]),
        spec_metadata=spec_metadata,
    )

    assert torch.equal(output, torch.tensor([[1.0, 2.0]]))
    assert len(captured) == 1
    assert captured[0][0] == -1
    assert torch.equal(captured[0][1], torch.tensor([[1.0, 2.0]]))
