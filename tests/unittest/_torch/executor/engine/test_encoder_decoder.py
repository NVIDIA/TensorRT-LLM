# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.encoder import EncoderPreparedInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


def _scheduled_encoder_requests(*requests: object) -> ScheduledRequests:
    scheduled_requests = ScheduledRequests()
    scheduled_requests.encoder_requests = list(requests)
    return scheduled_requests


def test_token_encoder_preparation_preserves_request_order_and_position_offset() -> None:
    prepared = EncoderPreparedInputs({}, sequence_lengths=[2, 3])
    runner = object.__new__(EncoderDecoderRunner)
    runner._model = SimpleNamespace(position_id_offset=4)
    runner._config = SimpleNamespace(max_num_tokens=8)
    runner._prepare_packed_token_inputs = Mock(return_value=prepared)
    resource_manager = object()
    requests = [
        SimpleNamespace(encoder_tokens=[11, 12], py_request_id=7),
        SimpleNamespace(encoder_tokens=[21, 22, 23], py_request_id=9),
    ]

    actual = runner._prepare_token_inputs(requests, resource_manager=resource_manager)

    assert actual is prepared
    runner._prepare_packed_token_inputs.assert_called_once_with(
        input_ids=[11, 12, 21, 22, 23],
        position_ids=[4, 5, 4, 5, 6],
        sequence_lengths=[2, 3],
        request_ids=[7, 9],
        resource_manager=resource_manager,
    )


def test_encoder_decoder_rejects_mixed_token_and_feature_batch() -> None:
    runner = object.__new__(EncoderDecoderRunner)
    scheduled_requests = _scheduled_encoder_requests(
        SimpleNamespace(py_encoder_input_features=torch.empty(1)),
        SimpleNamespace(py_encoder_input_features=None),
    )

    with pytest.raises(ValueError, match="cannot share one batch"):
        runner.prepare_inputs(
            scheduled_requests,
            resource_manager=None,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
        )


def test_encoder_decoder_rejects_unhandled_model_inputs() -> None:
    runner = object.__new__(EncoderDecoderRunner)

    with pytest.raises(NotImplementedError, match="position_ids"):
        runner.prepare_inputs(
            ScheduledRequests(),
            resource_manager=None,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
            position_ids=object(),
        )


def test_token_encoder_stack_applies_shared_embedding_scale_and_positions() -> None:
    embedding = Mock(side_effect=lambda input_ids: input_ids.to(torch.float32).unsqueeze(1))
    expected = torch.tensor([[2.0], [4.0], [6.0]])
    encoder = Mock(return_value=expected)
    runner = object.__new__(EncoderDecoderRunner)
    runner._model = SimpleNamespace(
        encoder=encoder,
        model=SimpleNamespace(shared_embedding=embedding, embed_scale=2.0),
    )
    metadata = object()

    actual = runner._forward_encoder_stack(
        {
            "encoder_input_ids": torch.tensor([1, 2, 3]),
            "encoder_position_ids": torch.tensor([[4, 5, 6]]),
            "encoder_attn_metadata": metadata,
        }
    )

    assert actual is expected
    encoder.assert_called_once()
    torch.testing.assert_close(
        encoder.call_args.kwargs["hidden_states"],
        torch.tensor([[2.0], [4.0], [6.0]]),
    )
    assert encoder.call_args.kwargs["attn_metadata"] is metadata
    torch.testing.assert_close(encoder.call_args.kwargs["position_ids"], torch.tensor([4, 5, 6]))


def test_feature_encoder_stack_uses_feature_model_contract() -> None:
    expected = torch.arange(6).reshape(3, 2)
    encoder = Mock(return_value=expected)
    runner = object.__new__(EncoderDecoderRunner)
    runner._model = SimpleNamespace(encoder=encoder)
    features = torch.arange(12).reshape(3, 4)
    metadata = object()

    actual = runner._forward_encoder_stack(
        {
            "input_features": features,
            "encoder_attn_metadata": metadata,
        }
    )

    assert actual is expected
    encoder.assert_called_once_with(input_features=features, attn_metadata=metadata)


@pytest.mark.parametrize("feature_mode", [False, True])
def test_graph_execution_restores_variant_specific_output_layout(feature_mode: bool) -> None:
    key = (2, 8, 4)
    prepared = EncoderPreparedInputs(
        {"input": object()},
        sequence_lengths=[2, 3],
        graph_key=key,
    )
    graph_output = torch.arange(16).reshape(8, 2)
    graph_runner = SimpleNamespace(
        feature_mode=feature_mode,
        get_graph_pool=Mock(return_value=object()),
        restore_encoder_decoder_output=Mock(return_value=torch.full((5, 2), -1)),
    )
    runner = object.__new__(EncoderDecoderRunner)
    runner._encoder_cuda_graph_runner = graph_runner
    runner._execute_encoder_cuda_graph = Mock(return_value=graph_output)

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder.with_shared_pool",
        return_value=nullcontext(),
    ) as shared_pool:
        actual = runner._execute_prepared(prepared)

    if feature_mode:
        torch.testing.assert_close(actual, graph_output[:5])
        assert actual.data_ptr() != graph_output.data_ptr()
        shared_pool.assert_not_called()
        graph_runner.restore_encoder_decoder_output.assert_not_called()
    else:
        torch.testing.assert_close(actual, torch.full((5, 2), -1))
        shared_pool.assert_called_once_with(graph_runner.get_graph_pool.return_value)
        graph_runner.restore_encoder_decoder_output.assert_called_once_with(
            key, graph_output, prepared.kwargs
        )


def test_encoder_decoder_forward_returns_hidden_states_with_prepared_lengths() -> None:
    prepared = EncoderPreparedInputs({}, sequence_lengths=[2, 3])
    hidden_states = torch.arange(10).reshape(5, 2)
    runner = object.__new__(EncoderDecoderRunner)
    runner.prepare_inputs = Mock(return_value=prepared)
    runner._execute_prepared = Mock(return_value=hidden_states)
    scheduled_requests = ScheduledRequests()
    resource_manager = object()

    outputs = runner.forward(
        scheduled_requests,
        resource_manager=resource_manager,
        cuda_graph_lora_manager=None,
        runtime_draft_len=0,
        gather_context_logits=False,
    )

    assert outputs == {
        "encoder_hidden_states": hidden_states,
        "encoder_seq_lens": [2, 3],
    }
    runner.prepare_inputs.assert_called_once_with(
        scheduled_requests,
        resource_manager=resource_manager,
        cuda_graph_lora_manager=None,
        runtime_draft_len=0,
    )
    runner._execute_prepared.assert_called_once_with(prepared)
