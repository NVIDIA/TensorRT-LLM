# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.interface import (
    should_use_separate_draft_kv_cache,
    uses_shared_kv_cache,
)
from tensorrt_llm._torch.speculative.utils import (
    get_num_extra_kv_tokens,
    get_num_spec_layers,
    update_spec_config_from_loaded_model,
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

    assert not uses_shared_kv_cache(spec_config)
    assert get_num_spec_layers(spec_config) == 1
    assert get_num_extra_kv_tokens(spec_config) == 2
    assert should_use_separate_draft_kv_cache(spec_config)


def test_loaded_draft_capability_updates_runtime_spec_config():
    spec_config = MTPDecodingConfig(
        max_draft_len=3,
        speculative_model="/tmp/gemma4-assistant",
        mtp_eagle_one_model=True,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(num_nextn_predict_layers=1),
        draft_config=None,
        draft_model=SimpleNamespace(shares_target_kv_cache=True),
    )

    update_spec_config_from_loaded_model(spec_config, model)

    assert uses_shared_kv_cache(spec_config)


def test_external_shared_kv_uses_no_draft_kv_cache():
    spec_config = _shared_kv_spec_config()

    assert uses_shared_kv_cache(spec_config)
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


def test_external_shared_kv_worker_requires_draft_model_capability():
    worker = MTPEagleWorker(_shared_kv_spec_config())

    with pytest.raises(ValueError, match="shares_target_kv_cache=True"):
        worker.set_draft_model(SimpleNamespace(model=SimpleNamespace()))


def test_external_shared_kv_worker_rejects_unverified_modes():
    spec_config = _shared_kv_spec_config(
        use_dynamic_tree=True,
        dynamic_tree_max_topK=2,
    )
    worker = MTPEagleWorker(spec_config)

    with pytest.raises(ValueError, match="linear draft path"):
        worker.set_draft_model(
            SimpleNamespace(
                model=SimpleNamespace(),
                shares_target_kv_cache=True,
            )
        )


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
