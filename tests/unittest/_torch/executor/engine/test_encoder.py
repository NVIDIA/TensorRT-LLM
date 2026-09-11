# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.engine.runners import encoder as encoder_module
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder import (
    EncoderConfigMixin,
    EncoderPreparedInputs,
    EncoderRunner,
    EncoderRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import (
    EncoderDecoderRunner,
    EncoderDecoderRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import EncodeCudaGraphConfig

pytestmark = pytest.mark.cpu_only


def _encoder_config(
    graph_config: EncodeCudaGraphConfig | None,
    *,
    declares_feature_spec: bool,
    tp_size: int = 1,
    encoder_decoder: bool = False,
) -> tuple[EncoderConfigMixin, tuple[tuple[int, ...], torch.dtype, int]]:
    feature_spec = ((480_000,), torch.float32, 1_500)

    class _Model:
        model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(),
        )

        if declares_feature_spec:

            def encoder_graph_spec(self) -> tuple[tuple[int, ...], torch.dtype, int]:
                return feature_spec

    config_type = EncoderDecoderRunnerConfig if encoder_decoder else EncoderRunnerConfig
    kwargs = dict(
        model=_Model(),
        mapping=SimpleNamespace(tp_size=tp_size),
        graph_config=graph_config,
        max_batch_size=8,
        max_num_tokens=8 * 1_500,
        max_seq_len=1_500,
        max_beam_width=1,
        without_logits=False,
        attention_backend=TrtllmAttention,
        attention_runtime_features=AttentionRuntimeFeatures(),
        enable_autotuner=False,
        draft_model=False,
    )
    if encoder_decoder:
        kwargs["is_encoder_decoder"] = True
    return config_type.create(**kwargs), feature_spec


@pytest.mark.parametrize(
    ("graph_config", "declares_feature_spec", "tp_size", "expected"),
    [
        (EncodeCudaGraphConfig(batch_sizes=[1, 2]), True, 1, "feature"),
        (
            EncodeCudaGraphConfig(batch_sizes=[1], num_tokens=[1_500], seq_lens=[1_500]),
            False,
            1,
            "token",
        ),
        (None, True, 1, "disabled"),
        (EncodeCudaGraphConfig(batch_sizes=[1]), True, 2, "disabled"),
    ],
)
def test_encoder_config_resolves_model_graph_contract(
    graph_config: EncodeCudaGraphConfig | None,
    declares_feature_spec: bool,
    tp_size: int,
    expected: str,
) -> None:
    config, feature_spec = _encoder_config(
        graph_config,
        declares_feature_spec=declares_feature_spec,
        tp_size=tp_size,
        encoder_decoder=True,
    )

    if expected == "feature":
        assert (config.feature_shape, config.feature_dtype, config.fixed_seq_len) == feature_spec
        assert config.cuda_graph_enabled
    else:
        assert (config.feature_shape, config.feature_dtype, config.fixed_seq_len) == (
            None,
            None,
            None,
        )
        assert config.cuda_graph_enabled == (expected == "token")


def test_encoder_decoder_token_graph_config_requires_token_and_sequence_buckets() -> None:
    with pytest.raises(ValueError, match="num_tokens/max_num_token and seq_lens/max_seq_len"):
        _encoder_config(
            EncodeCudaGraphConfig(batch_sizes=[1, 2]),
            declares_feature_spec=False,
            encoder_decoder=True,
        )

    with pytest.raises(ValueError, match="seq_lens/max_seq_len unset"):
        _encoder_config(
            EncodeCudaGraphConfig(batch_sizes=[1, 2], num_tokens=[1_500]),
            declares_feature_spec=False,
            encoder_decoder=True,
        )


def test_encoder_only_incomplete_graph_config_warns_and_stays_eager() -> None:
    with patch("tensorrt_llm._torch.pyexecutor.engine.runners.encoder.logger.warning") as warning:
        config, _ = _encoder_config(
            EncodeCudaGraphConfig(batch_sizes=[1, 2]),
            declares_feature_spec=False,
        )

    assert not config.is_encoder_decoder
    assert not config.cuda_graph_enabled
    assert warning.call_count == 1
    assert "stays eager" in warning.call_args.args[0]


def test_encoder_only_attention_metadata_uses_runner_cache_indirection() -> None:
    cache_indirection = object()
    metadata = SimpleNamespace(
        block_ids_per_seq=object(),
        kv_block_ids_per_seq=object(),
    )
    runner = object.__new__(EncoderRunner)
    runner._model = SimpleNamespace(model_config=object())
    runner._config = SimpleNamespace(
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=TrtllmAttention,
        attention_runtime_features=AttentionRuntimeFeatures(),
    )
    runner._deps = SimpleNamespace(
        mapping=object(),
        cache_indirection=cache_indirection,
    )
    runner._encoder_config = SimpleNamespace(is_encoder_decoder=False)

    with patch.object(
        encoder_module,
        "build_attention_metadata",
        return_value=metadata,
    ) as build_metadata:
        actual = runner._create_attention_metadata()

    assert actual is metadata
    assert build_metadata.call_args.kwargs["cache_indirection"] is cache_indirection
    assert metadata.block_ids_per_seq is None
    assert metadata.kv_block_ids_per_seq is None


def test_encoder_decoder_attention_metadata_omits_decoder_cache_indirection() -> None:
    metadata = SimpleNamespace(
        block_ids_per_seq=object(),
        kv_block_ids_per_seq=object(),
    )
    runner = object.__new__(EncoderDecoderRunner)
    runner._model = SimpleNamespace(model_config=object())
    runner._config = SimpleNamespace(
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=TrtllmAttention,
        attention_runtime_features=AttentionRuntimeFeatures(),
    )
    runner._deps = SimpleNamespace(
        mapping=object(),
        cache_indirection=object(),
    )
    runner._encoder_config = SimpleNamespace(is_encoder_decoder=True)

    with patch.object(
        encoder_module,
        "build_attention_metadata",
        return_value=metadata,
    ) as build_metadata:
        runner._create_attention_metadata()

    assert build_metadata.call_args.kwargs["cache_indirection"] is None


def test_encoder_runner_collects_scheduled_inputs_without_losing_request_boundaries() -> None:
    requests = [
        SimpleNamespace(
            get_tokens=lambda _: [11, 12],
            py_multi_item_part_lens=[1, 1],
        ),
        SimpleNamespace(
            get_tokens=lambda _: [21, 22, 23],
            py_multi_item_part_lens=[2, 1],
        ),
    ]
    scheduled_requests = ScheduledRequests()
    scheduled_requests.context_requests_last_chunk = requests

    assert EncoderRunner._collect_scheduled_inputs(scheduled_requests) == (
        [11, 12, 21, 22, 23],
        [2, 3],
        [[1, 1], [2, 1]],
    )


def test_encoder_runner_rejects_partial_multi_item_metadata() -> None:
    requests = [
        SimpleNamespace(get_tokens=lambda _: [11], py_multi_item_part_lens=[1]),
        SimpleNamespace(get_tokens=lambda _: [21], py_multi_item_part_lens=None),
    ]
    scheduled_requests = ScheduledRequests()
    scheduled_requests.context_requests_last_chunk = requests

    with pytest.raises(ValueError, match="provided for all requests or for none"):
        EncoderRunner._collect_scheduled_inputs(scheduled_requests)


@pytest.mark.parametrize("reserved_name", ["input_ids", "seq_lens", "attn_metadata"])
def test_encoder_runner_rejects_model_inputs_owned_by_the_runner(reserved_name: str) -> None:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(enabled=False)

    with pytest.raises(ValueError, match=reserved_name):
        runner.prepare_inputs(
            ScheduledRequests(),
            resource_manager=None,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
            **{reserved_name: object()},
        )


def test_encoder_runner_forwards_model_inputs_to_eager_preparation() -> None:
    expected = EncoderPreparedInputs({}, sequence_lengths=[2])
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(enabled=False)
    runner._collect_scheduled_inputs = Mock(return_value=([11, 12], [2], None))
    runner._prepare_encoder_batch = Mock(return_value=expected)
    scheduled_requests = ScheduledRequests()
    token_type_ids = torch.tensor([0, 1])

    actual = runner.prepare_inputs(
        scheduled_requests,
        resource_manager=None,
        cuda_graph_lora_manager=None,
        runtime_draft_len=0,
        token_type_ids=token_type_ids,
    )

    assert actual is expected
    runner._collect_scheduled_inputs.assert_called_once_with(scheduled_requests)
    runner._prepare_encoder_batch.assert_called_once_with(
        [11, 12],
        [2],
        multi_item_part_lens=None,
        model_inputs={"token_type_ids": token_type_ids},
    )


def test_encoder_runner_rejects_model_inputs_for_cuda_graph_execution() -> None:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(enabled=True)

    with pytest.raises(NotImplementedError, match="token_type_ids"):
        runner.prepare_inputs(
            ScheduledRequests(),
            resource_manager=None,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
            token_type_ids=object(),
        )


def test_encoder_runner_forwards_eager_model_inputs_and_gathers_logits() -> None:
    metadata = SimpleNamespace(on_update_kv_lens=Mock())
    model_forward = Mock(return_value={"logits": torch.arange(12).reshape(3, 4)})
    runner = object.__new__(EncoderRunner)
    runner._deps = SimpleNamespace(model_forward=model_forward)
    runner._config = SimpleNamespace(without_logits=False)
    inputs = {
        "input_ids": torch.tensor([1, 2, 3]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "token_type_ids": torch.tensor([0, 1, 1]),
        "attn_metadata": metadata,
    }

    outputs = runner._forward_step(
        inputs,
        gather_ids=torch.tensor([2, 0]),
        gather_context_logits=False,
    )

    torch.testing.assert_close(outputs["logits"], torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3]]))
    metadata.on_update_kv_lens.assert_called_once_with()
    model_forward.assert_called_once_with(**inputs, return_context_logits=True)


def test_encoder_capture_runs_warmup_then_capture_and_restores_phase() -> None:
    events: list[bool] = []

    @contextmanager
    def allow_capture():
        yield

    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(
        enabled=True,
        is_warmup_only=False,
        allow_capture=allow_capture,
    )
    runner._run_encoder_capture_pass = Mock(
        side_effect=lambda *_: events.append(runner._encoder_cuda_graph_runner.is_warmup_only)
    )

    runner._capture_encoder_cuda_graphs(Mock(), Mock())

    assert events == [True, False]
    assert not runner._encoder_cuda_graph_runner.is_warmup_only


def test_token_graph_capture_and_replay_enter_moe_iteration_context() -> None:
    load_balancer = object()
    context_balancers: list[object | None] = []
    capture_forward_calls: list[dict[str, object]] = []

    @contextmanager
    def moe_context(balance: object | None):
        context_balancers.append(balance)
        yield

    class _GraphRunner:
        feature_mode = False
        is_warmup_only = False

        @staticmethod
        def needs_capture(key: tuple[int, int, int]) -> bool:
            return True

        @staticmethod
        def capture(key, forward, inputs):
            capture_forward_calls.append(inputs)
            return forward(inputs)

        @staticmethod
        def replay(key, inputs):
            return {"logits": torch.tensor([[2.0]])}

    runner = object.__new__(EncoderRunner)
    runner._deps = SimpleNamespace(moe_load_balancer=load_balancer)
    runner._encoder_cuda_graph_runner = _GraphRunner()
    prepared = EncoderPreparedInputs(
        {"input_ids": object()},
        sequence_lengths=[1],
        graph_key=(1, 1, 1),
    )
    forward = Mock(return_value={"logits": torch.tensor([[1.0]])})

    with patch.object(encoder_module, "MoeLoadBalancerIterContext", side_effect=moe_context):
        outputs = runner._execute_encoder_cuda_graph(prepared, forward)

    torch.testing.assert_close(outputs["logits"], torch.tensor([[2.0]]))
    assert capture_forward_calls == [prepared.kwargs]
    assert context_balancers == [load_balancer, load_balancer]
    forward.assert_called_once_with(prepared.kwargs)


def _warmup_runner(*, world_size: int) -> EncoderRunner:
    runner = object.__new__(EncoderRunner)
    runner._deps = SimpleNamespace(
        dist=SimpleNamespace(world_size=world_size),
        mapping=SimpleNamespace(dwdp_enabled=False),
    )
    runner._encoder_cuda_graph_runner = SimpleNamespace(
        enabled=True,
        build_capture_sequence_lengths=Mock(return_value=[1]),
    )
    runner._prepare_encoder_batch = Mock(
        return_value=EncoderPreparedInputs({}, sequence_lengths=[1])
    )
    return runner


def test_encoder_warmup_recovers_from_oom_when_not_distributed() -> None:
    runner = _warmup_runner(world_size=1)
    runner._execute_prepared = Mock(side_effect=[torch.OutOfMemoryError("OOM"), None])

    with patch("torch.cuda.empty_cache") as empty_cache, patch("torch.cuda.synchronize"):
        runner._run_warmup_shapes([(2, 16, 8), (1, 8, 8)])

    assert runner._execute_prepared.call_count == 2
    empty_cache.assert_called_once_with()


def test_encoder_warmup_oom_is_fatal_when_distributed() -> None:
    runner = _warmup_runner(world_size=2)
    error = torch.OutOfMemoryError("OOM")
    runner._execute_prepared = Mock(side_effect=error)

    with patch("torch.cuda.empty_cache"), patch("torch.cuda.synchronize"):
        with pytest.raises(torch.OutOfMemoryError) as raised:
            runner._run_warmup_shapes([(2, 16, 8), (1, 8, 8)])

    assert raised.value is error
    runner._execute_prepared.assert_called_once_with(runner._prepare_encoder_batch.return_value)


def test_encoder_release_clears_owned_graph_backend() -> None:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = Mock()

    runner.release_graph()

    runner._encoder_cuda_graph_runner.clear.assert_called_once_with()
