# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal-import CPU regression for inference-state serving reprojection.

Use the genuine serving method, context staging, LogitsProcessor and CPU LMHead
with grad-enabled Parameters at the exact unquantized F.linear boundary seen
in the model failure; this is not a GPU/TP/model qualification.
No framework imports, globals, types or AST bodies are substituted.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.speculative.logprobs import GATHER_IDS_KEY, HIDDEN_STATES_KEY
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _projection() -> types.SimpleNamespace:
    with torch.device("cpu"):
        head = LMHead(
            num_embeddings=3,
            embedding_dim=2,
            dtype=torch.float32,
            mapping=Mapping(tp_size=1),
            reduce_output=False,
            use_custom_cublas_mm=False,
        )
    assert type(head) is LMHead and head.weight.device.type == "cpu"
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]], device="cpu"))
    assert head.weight.requires_grad and not head.weight.is_inference()
    model = types.SimpleNamespace(lm_head=head, logits_processor=LogitsProcessor())
    model.project_speculative_logits = types.MethodType(
        SpecDecOneEngineForCausalLM.project_speculative_logits, model
    )
    return model


def _inference_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]], device="cpu")
        gather = torch.arange(3, dtype=torch.long, device="cpu")
    assert hidden.is_inference() and hidden[:2].is_inference()
    return hidden, gather


def _flags() -> tuple[bool, bool]:
    return torch.is_grad_enabled(), torch.is_inference_mode_enabled()


def test_real_projection_primitive_reproduces_original_backward_save_failure() -> None:
    model = _projection()
    hidden, _ = _inference_states()
    with torch.enable_grad():
        before = _flags()
        with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
            model.logits_processor.forward(hidden[:2], model.lm_head, None, True)
        assert _flags() == before == (True, False)
    assert model.lm_head.weight.grad is None


@pytest.mark.parametrize("caller_grad", [True, False])
def test_real_projector_accepts_inference_view_and_restores_caller_flags(caller_grad: bool) -> None:
    model = _projection()
    hidden, _ = _inference_states()
    seen = []

    def observe(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append((_flags(), inputs[0].is_inference(), inputs[0].data_ptr()))

    handle = model.lm_head.register_forward_pre_hook(observe)
    view = hidden[:2]
    with torch.set_grad_enabled(caller_grad):
        before = _flags()
        actual = model.project_speculative_logits(view)
        assert _flags() == before == (caller_grad, False)
    handle.remove()
    with torch.no_grad():
        expected = hidden[:2] @ model.lm_head.weight.T
    torch.testing.assert_close(actual, expected)
    assert seen == [((False, False), True, view.data_ptr())]
    assert not actual.requires_grad and actual.grad_fn is None
    assert not actual.is_inference()
    assert model.lm_head.weight.grad is None and model.lm_head.weight.requires_grad


@pytest.mark.parametrize("caller_grad", [True, False])
def test_actual_context_staging_uses_guarded_bound_projection_on_inference_view(
    caller_grad: bool,
) -> None:
    model = _projection()
    hidden, gather = _inference_states()
    seen = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: seen.append((_flags(), inputs[0].is_inference()))
    )
    context = PyResult(
        prompt_len=2,
        max_new_tokens=1,
        use_device_memory=False,
        return_context_logits=True,
        use_chunked_generation_logits=False,
    )
    request = types.SimpleNamespace(
        py_return_context_logits=True,
        py_context_logits_explicit=True,
        py_prompt_logprobs=None,
        prepopulated_prompt_len=0,
        context_chunk_size=2,
        context_current_position=0,
        orig_prompt_len=2,
        py_client_id=17,
        py_result=context,
    )
    scheduled = types.SimpleNamespace(context_requests=[request])
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.logit_projection = model.project_speculative_logits
    outputs = {HIDDEN_STATES_KEY: hidden, GATHER_IDS_KEY: gather}
    with torch.set_grad_enabled(caller_grad):
        before = _flags()
        scores, indices = sampler._stage_requested_logits(scheduled, outputs, [], 1)
        assert _flags() == before == (caller_grad, False)
    handle.remove()
    assert scores is None and indices == []
    assert seen == [((False, False), True)]
    with torch.no_grad():
        expected = hidden[:2] @ model.lm_head.weight.T
    actual = context.context_logits
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and actual.grad_fn is None
    assert model.lm_head.weight.requires_grad and model.lm_head.weight.grad is None


def test_actual_generation_staging_keeps_raw_scores_out_of_autograd() -> None:
    model = _projection()
    hidden, gather = _inference_states()
    seen = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: seen.append((_flags(), inputs[0].is_inference()))
    )
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.logit_projection = model.project_speculative_logits
    sampler._copy_to_host = lambda tensor: tensor.clone()
    request = types.SimpleNamespace(py_return_log_probs=True, py_num_logprobs=1)
    scheduled = types.SimpleNamespace(context_requests=[], context_requests_chunking=[])
    outputs = {
        HIDDEN_STATES_KEY: hidden,
        GATHER_IDS_KEY: gather,
        "new_tokens": torch.tensor([[0, 1]], device="cpu"),
        "new_tokens_lens": torch.tensor([2], device="cpu"),
    }
    with torch.enable_grad():
        before = _flags()
        scores, indices = sampler._stage_requested_logits(scheduled, outputs, [request], 1)
        assert _flags() == before == (True, False)
    handle.remove()
    assert indices == [0] and seen == [((False, False), False)]
    with torch.no_grad():
        expected = (hidden[:2] @ model.lm_head.weight.T).log_softmax(-1)
    torch.testing.assert_close(scores.sampled, expected[[0, 1], [0, 1]].reshape(1, 2))
    assert not scores.sampled.requires_grad and scores.sampled.grad_fn is None
    assert model.lm_head.weight.grad is None


def test_projection_preserves_preexisting_inference_mode() -> None:
    model = _projection()
    hidden, _ = _inference_states()
    outside = _flags()
    with torch.inference_mode():
        before = _flags()
        actual = model.project_speculative_logits(hidden[:2])
        assert _flags() == before == (False, True)
        assert actual.is_inference() and not actual.requires_grad
    assert _flags() == outside
    with torch.no_grad():
        expected = hidden[:2] @ model.lm_head.weight.T
    torch.testing.assert_close(actual, expected)


def test_projection_preserves_normal_input_values_without_autograd_graph() -> None:
    model = _projection()
    hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0]], device="cpu")
    assert not hidden.is_inference()
    with torch.enable_grad():
        before = _flags()
        actual = model.project_speculative_logits(hidden)
        assert _flags() == before == (True, False)
    with torch.no_grad():
        expected = hidden @ model.lm_head.weight.T
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and actual.grad_fn is None and not actual.is_inference()
    assert model.lm_head.weight.grad is None


def test_projection_preserves_original_exception_and_restores_caller_flags() -> None:
    model = _projection()
    hidden, _ = _inference_states()
    failure = RuntimeError("deliberate real-head hook failure")

    def fail(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        assert _flags() == (False, False)
        raise failure

    handle = model.lm_head.register_forward_pre_hook(fail)
    with torch.enable_grad():
        before = _flags()
        with pytest.raises(RuntimeError) as observed:
            model.project_speculative_logits(hidden[:2])
        assert observed.value is failure and _flags() == before == (True, False)
    handle.remove()
