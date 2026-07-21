# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.speculative.drafting_loops import Gemma4AssistantDraftingLoopWrapper
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter


class _DummyGemma4Assistant(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="gemma4_assistant")
        self.model_config = None
        self.model = SimpleNamespace()
        self.calls = []

    def forward(self, input_ids, position_ids, attn_metadata, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "kv_lens": attn_metadata.kv_lens_cuda.clone(),
            }
        )
        logits = torch.zeros((input_ids.shape[0], 8))
        logits[:, len(self.calls)] = 1
        return logits


def test_gemma4_drafting_loop_keeps_position_and_target_kv_length():
    draft_model = _DummyGemma4Assistant()
    wrapper = Gemma4AssistantDraftingLoopWrapper(
        max_draft_len=3,
        max_total_draft_tokens=3,
        draft_model=draft_model,
    )
    wrapper.sample = lambda logits: logits.argmax(dim=-1)

    attn_metadata = SimpleNamespace(
        num_seqs=2,
        kv_lens_cuda=torch.tensor([7, 11]),
    )
    spec_metadata = object.__new__(Eagle3SpecMetadata)
    spec_metadata.gather_ids = torch.tensor([0, 1])
    spec_metadata.hidden_states_read_indices = torch.tensor([4, 8])
    spec_metadata.hidden_states_write_indices = torch.tensor([5, 9])

    outputs = wrapper(
        input_ids=torch.tensor([2, 3]),
        position_ids=torch.tensor([[6, 10]]),
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
    )

    assert outputs["new_draft_tokens"].tolist() == [[1, 1], [2, 2], [3, 3]]
    assert len(draft_model.calls) == 3
    assert all(call["position_ids"].tolist() == [[6, 10]] for call in draft_model.calls)
    assert all(call["kv_lens"].tolist() == [7, 11] for call in draft_model.calls)
    assert spec_metadata.hidden_states_read_indices.tolist() == [5, 9]


def test_gemma4_drafter_records_target_hidden_state_offset():
    drafter = object.__new__(ModelDrafter)
    drafter.spec_resource_manager = SimpleNamespace(
        draft_hidden_state_offsets={},
        seq_lens={4: 10},
        slot_manager=SimpleNamespace(get_slot=lambda request_id: 4),
    )
    draft_request = SimpleNamespace()
    drafter._create_generation_request = lambda request, tokens: draft_request
    request = SimpleNamespace(
        py_request_id=17,
        py_last_context_chunk=(4, 10),
        py_prompt_len=10,
        py_num_accepted_draft_tokens=3,
    )

    assert (
        drafter._create_gemma4_assistant_request(request, [1, 2], is_first_draft=True)
        is draft_request
    )
    assert drafter.spec_resource_manager.draft_hidden_state_offsets[17] == 9

    drafter._create_gemma4_assistant_request(request, [1, 2], is_first_draft=False)
    assert drafter.spec_resource_manager.draft_hidden_state_offsets[17] == 3


def test_gemma4_cuda_graph_warmup_uses_one_token_generation_request():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = True
    engine.model_is_wrapped = True
    engine.model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_assistant"))
    spec_resource_manager = object.__new__(Eagle3ResourceManager)
    spec_resource_manager.is_first_draft = True
    resource_manager = SimpleNamespace(
        get_resource_manager=lambda resource_type: spec_resource_manager
    )
    request = SimpleNamespace(py_is_first_draft=True, py_draft_tokens=[1])
    batch = SimpleNamespace(generation_requests=[request])

    engine._update_draft_inference_state_for_warmup(
        batch, is_first_draft=True, resource_manager=resource_manager
    )

    assert not spec_resource_manager.is_first_draft
    assert not request.py_is_first_draft
    assert request.py_draft_tokens == []


def test_gemma4_target_forward_captures_speculative_hidden_states():
    model = SimpleNamespace(
        layer_idx=-1,
        config=SimpleNamespace(final_logit_softcapping=None),
        model=lambda **kwargs: torch.tensor([[1.0, 2.0]]),
        logits_processor=SimpleNamespace(forward=lambda hidden_states, *args: hidden_states),
        lm_head=object(),
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
