# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM
from tensorrt_llm._torch.pyexecutor.engine.runners import encoder_decoder as encoder_decoder_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder import EncoderPreparedInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import (
    CrossAttentionInputs,
    EncoderDecoderRunner,
    EncoderStage,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


def _scheduled_encoder_requests(*requests: object) -> ScheduledRequests:
    scheduled_requests = ScheduledRequests()
    scheduled_requests.encoder_requests = list(requests)
    return scheduled_requests


def test_token_encoder_preparation_preserves_request_order_and_position_offset() -> None:
    prepared = EncoderPreparedInputs({}, sequence_lengths=[2, 3])
    runner = object.__new__(EncoderStage)
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
    runner = object.__new__(EncoderStage)
    scheduled_requests = _scheduled_encoder_requests(
        SimpleNamespace(py_encoder_input_features=torch.empty(1)),
        SimpleNamespace(py_encoder_input_features=None),
    )

    with pytest.raises(ValueError, match="cannot share one batch"):
        runner.prepare_inputs(
            scheduled_requests,
            resource_manager=None,
        )


def test_token_encoder_stack_applies_shared_embedding_scale_and_positions() -> None:
    embedding = Mock(side_effect=lambda input_ids: input_ids.to(torch.float32).unsqueeze(1))
    expected = torch.tensor([[2.0], [4.0], [6.0]])
    encoder = Mock(return_value=expected)
    runner = object.__new__(EncoderStage)
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
    runner = object.__new__(EncoderStage)
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
    runner = object.__new__(EncoderStage)
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
    runner = object.__new__(EncoderStage)
    runner.prepare_inputs = Mock(return_value=prepared)
    runner._execute_prepared = Mock(return_value=hidden_states)
    scheduled_requests = ScheduledRequests()
    resource_manager = object()

    outputs = runner.forward(
        ScheduledInputs(batch=scheduled_requests),
        resource_manager=resource_manager,
    )

    assert outputs == {
        "encoder_hidden_states": hidden_states,
        "encoder_seq_lens": [2, 3],
    }
    runner.prepare_inputs.assert_called_once_with(
        scheduled_requests,
        resource_manager=resource_manager,
    )
    runner._execute_prepared.assert_called_once_with(prepared)


def test_encoder_decoder_runner_builds_stage_before_decoder() -> None:
    model = object()
    config = object()
    encoder_config = object()
    mapping = object()
    dist = object()
    input_processor = object()
    shapes = frozenset({(1, 16)})

    order = Mock()
    with (
        patch.object(encoder_decoder_module, "EncoderStage") as stage_type,
        patch.object(DecoderRunner, "__init__", return_value=None) as decoder_init,
    ):
        order.attach_mock(stage_type, "stage")
        order.attach_mock(decoder_init, "decoder")
        stage_type.return_value._encoder_graph_shapes = shapes
        runner = EncoderDecoderRunner(
            model,
            config,
            encoder_config=encoder_config,
            mapping=mapping,
            dist=dist,
            moe_load_balancer=None,
            input_processor=input_processor,
        )

    assert runner._encoder_stage is stage_type.return_value
    assert runner._encoder_graph_shapes is shapes
    assert order.mock_calls == [
        call.stage(model, encoder_config, mapping=mapping, dist=dist, moe_load_balancer=None),
        call.decoder(
            model,
            config,
            mapping=mapping,
            dist=dist,
            moe_load_balancer=None,
            input_processor=input_processor,
        ),
    ]


def test_encoder_decoder_runner_releases_stage_before_decoder_graphs() -> None:
    runner = object.__new__(EncoderDecoderRunner)
    calls = Mock()
    runner._encoder_stage = calls.stage
    runner._release_decoder_graphs = calls.decoder

    runner.release_graphs()

    assert calls.mock_calls == [
        call.stage.release_graphs(),
        call.decoder(),
    ]


@dataclass
class _GraphRunnerConfig:
    is_encoder_decoder: bool = False
    enable_encoder_decoder_mixed_cuda_graph: bool = False


@pytest.mark.parametrize(
    ("encoder_graph_shapes", "cuda_graph_config", "mixed_enabled", "expected"),
    [
        (frozenset({(1, 16)}), object(), True, True),
        (frozenset(), object(), True, False),
        (frozenset({(1, 16)}), None, True, False),
        (frozenset({(1, 16)}), object(), False, False),
    ],
)
def test_encoder_decoder_graph_runner_config_gates_mixed_graphs(
    encoder_graph_shapes: frozenset,
    cuda_graph_config: object,
    mixed_enabled: bool,
    expected: bool,
) -> None:
    runner = object.__new__(EncoderDecoderRunner)
    runner._encoder_graph_shapes = encoder_graph_shapes
    runner._config = SimpleNamespace(
        cuda_graph_config=cuda_graph_config,
        enable_encoder_decoder_mixed_cuda_graph=mixed_enabled,
    )

    with patch.object(
        DecoderRunner, "_cuda_graph_runner_config", return_value=_GraphRunnerConfig()
    ):
        config = runner._cuda_graph_runner_config()

    assert config.is_encoder_decoder
    assert config.enable_encoder_decoder_mixed_cuda_graph is expected


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_encoder_decoder_release_context_frees_cross_kv(fails: bool) -> None:
    runner = object.__new__(EncoderDecoderRunner)
    requests = [object(), object()]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    cross_kv_cache_manager = Mock()
    resource_manager = Mock()
    resource_manager.get_resource_manager.side_effect = lambda key: (
        cross_kv_cache_manager if key == ResourceManagerType.CROSS_KV_CACHE_MANAGER else None
    )

    with (
        patch.object(
            DecoderRunner, "_release_batch_context", return_value=nullcontext(batch)
        ) as release_decoder,
        pytest.raises(RuntimeError) if fails else nullcontext(),
        runner._release_batch_context(batch, resource_manager) as released,
    ):
        assert released is batch
        cross_kv_cache_manager.free_resources.assert_not_called()
        if fails:
            raise RuntimeError("capture failure")

    release_decoder.assert_called_once_with(batch, resource_manager)
    assert cross_kv_cache_manager.free_resources.call_args_list == [
        call(request) for request in requests
    ]


def _dummy_request_runner(
    runner_type: type[DecoderRunner], pretrained_config: SimpleNamespace
) -> DecoderRunner:
    runner = object.__new__(runner_type)
    runner.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    runner._config = SimpleNamespace(
        max_seq_len=64,
        max_draft_loop_tokens=0,
        spec_config=None,
        use_mrope=False,
        max_beam_width=1,
    )
    runner.model = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=pretrained_config)
    )
    runner._get_draft_kv_cache_manager = Mock(return_value=None)
    return runner


def _dummy_request_resources() -> tuple[Mock, Mock]:
    kv_cache_manager = Mock(max_seq_len=64)
    kv_cache_manager.get_num_available_tokens.return_value = 40
    managers = {
        ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
        ResourceManagerType.CROSS_KV_CACHE_MANAGER: Mock(max_seq_len=24),
    }
    resource_manager = Mock()
    resource_manager.get_resource_manager.side_effect = managers.get
    return resource_manager, kv_cache_manager


@pytest.mark.parametrize(
    ("runner_type", "expected_tokens", "expected_kwargs"),
    [
        (DecoderRunner, 32, {}),
        (EncoderDecoderRunner, 16, {"encoder_output_lens": [24]}),
    ],
)
def test_longest_dummy_request_applies_runner_limits(
    runner_type: type[DecoderRunner], expected_tokens: int, expected_kwargs: dict
) -> None:
    runner = _dummy_request_runner(
        runner_type, SimpleNamespace(max_position_embeddings=32, max_target_positions=16)
    )
    resource_manager, kv_cache_manager = _dummy_request_resources()
    longest = object()
    kv_cache_manager.add_dummy_requests.return_value = [longest]

    actual = runner._add_longest_dummy_request(resource_manager, [], 3, None, 0, None)

    assert actual is longest
    kv_cache_manager.add_dummy_requests.assert_called_once_with(
        request_ids=[2],
        token_nums=[expected_tokens],
        is_gen=True,
        max_num_draft_tokens=0,
        kv_reserve_draft_tokens=0,
        use_mrope=False,
        max_beam_width=1,
        draft_kv_cache_manager=None,
        capture_sampling_params=None,
        **expected_kwargs,
    )


def test_longest_dummy_request_failure_frees_filler_requests() -> None:
    runner = _dummy_request_runner(DecoderRunner, SimpleNamespace())
    draft_kv_cache_manager = Mock()
    draft_kv_cache_manager.get_num_available_tokens.return_value = 40
    runner._get_draft_kv_cache_manager = Mock(return_value=draft_kv_cache_manager)
    resource_manager, kv_cache_manager = _dummy_request_resources()
    kv_cache_manager.add_dummy_requests.return_value = None
    fillers = [object(), object()]

    assert runner._add_longest_dummy_request(resource_manager, fillers, 3, None, 0, None) is None
    assert kv_cache_manager.free_resources.call_args_list == [call(r) for r in fillers]
    assert draft_kv_cache_manager.free_resources.call_args_list == [call(r) for r in fillers]


def test_mixed_warmup_request_orders_context_generation_and_longest_rows() -> None:
    runner = _dummy_request_runner(EncoderDecoderRunner, SimpleNamespace())
    runner.get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    resource_manager, kv_cache_manager = _dummy_request_resources()
    kv_cache_manager.get_num_free_blocks.return_value = 8
    contexts = [SimpleNamespace(), SimpleNamespace()]
    generations = [SimpleNamespace()]
    longest = SimpleNamespace()
    kv_cache_manager.add_dummy_requests.side_effect = [contexts, generations]
    longest_inputs = []

    def add_longest(resources: object, requests: list, *args: object, **kwargs: object) -> object:
        longest_inputs.append(list(requests))
        return longest

    with (
        patch.object(runner, "_add_longest_dummy_request", side_effect=add_longest),
        patch.object(
            runner, "_finish_cuda_graph_warmup_request", side_effect=lambda result, *_: result
        ) as finish,
    ):
        batch = runner._create_mixed_cuda_graph_warmup_request(resource_manager, 4, 0, [5, 7], 3)

    assert longest_inputs == [contexts + generations]
    assert batch.context_requests_last_chunk == contexts
    assert batch.generation_requests == [*generations, longest]
    finish.assert_called_once_with(batch, resource_manager, 4, 0)
    for request in contexts:
        assert request.state == LlmRequestState.CONTEXT_INIT
        assert request.context_current_position == 0
        assert request.context_chunk_size == 3
        assert request.cached_tokens == 0
        assert request.py_batch_idx is None
    context_call, generation_call = kv_cache_manager.add_dummy_requests.call_args_list
    assert context_call.args == ([0, 1],)
    assert context_call.kwargs["is_gen"] is False
    assert context_call.kwargs["token_nums"] == [3, 3]
    assert context_call.kwargs["encoder_output_lens"] == [5, 7]
    assert generation_call.args == ([2],)
    assert generation_call.kwargs["is_gen"] is True
    assert generation_call.kwargs["token_nums"] == [ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM]
    assert generation_call.kwargs["encoder_output_lens"] == [24]


def test_cross_attention_inputs_project_context_and_repeat_generation_rows() -> None:
    runner = Mock()
    inputs = CrossAttentionInputs(runner)
    encoder_output = torch.zeros(3, 2)
    inputs.add_context_request(
        SimpleNamespace(
            py_request_id=1,
            encoder_output_len=3,
            py_encoder_output=encoder_output,
            py_skip_cross_kv_projection=False,
        )
    )
    inputs.add_context_request(
        SimpleNamespace(py_request_id=2, encoder_output_len=4, py_skip_cross_kv_projection=True)
    )
    inputs.add_context_request(
        SimpleNamespace(
            py_request_id=3,
            encoder_output_len=6,
            py_skip_cross_kv_projection=False,
            is_dummy=True,
        )
    )
    inputs.add_generation_request(SimpleNamespace(py_request_id=4, encoder_output_len=5), repeat=2)
    attn_metadata = object()
    resource_manager = object()

    built = inputs.build(attn_metadata, resource_manager)

    assert built is runner._prepare_enc_dec_cross_attn_inputs.return_value
    runner._prepare_enc_dec_cross_attn_inputs.assert_called_once_with(
        [encoder_output], [3, 0, 0, 0, 0], [0, 4, 6, 5, 5], attn_metadata, resource_manager
    )
    with pytest.raises(RuntimeError, match="has no encoder output"):
        CrossAttentionInputs(runner).add_context_request(
            SimpleNamespace(
                py_request_id=5, encoder_output_len=3, py_skip_cross_kv_projection=False
            )
        )


@pytest.mark.parametrize(
    ("promoted_context_request_ids", "eligible", "taken"),
    [
        (frozenset(), True, True),
        (frozenset(), False, False),
        (frozenset({7}), True, False),
    ],
    ids=["taken", "ineligible", "promoted_context"],
)
def test_input_fast_path_clears_staged_request_ids_when_not_taken(
    promoted_context_request_ids: frozenset[int], eligible: bool, taken: bool
) -> None:
    runner = object.__new__(EncoderDecoderRunner)
    staged_request_ids = object()
    runner._encoder_decoder_staged_request_ids = staged_request_ids
    runner._can_use_encoder_decoder_input_fast_path = Mock(return_value=eligible)
    runner._prepare_encoder_decoder_inputs_fast = Mock()

    result = runner._prepare_inputs_fast_path(
        ScheduledRequests(),
        Mock(),
        object.__new__(TrtllmAttentionMetadata),
        None,
        None,
        None,
        promoted_context_request_ids=promoted_context_request_ids,
        enable_spec_decode=False,
    )

    if taken:
        assert result is runner._prepare_encoder_decoder_inputs_fast.return_value
        assert runner._encoder_decoder_staged_request_ids is staged_request_ids
    else:
        assert result is None
        assert runner._encoder_decoder_staged_request_ids is None
        runner._prepare_encoder_decoder_inputs_fast.assert_not_called()
