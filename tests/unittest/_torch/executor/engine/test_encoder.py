# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import EncoderCUDAGraphRunner
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
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PackedInputs
from tensorrt_llm.llmapi.llm_args import EncodeCudaGraphConfig, EncodeExtraInputSpec

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
    runner._mapping = object()
    runner._buffers = SimpleNamespace(cache_indirection=cache_indirection)
    runner._attn_metadata = None
    runner._encoder_config = SimpleNamespace(is_encoder_decoder=False)

    with patch.object(
        encoder_module,
        "build_attention_metadata",
        return_value=metadata,
    ) as build_metadata:
        actual = runner._setup_attention_metadata()

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
    runner._mapping = object()
    runner._encoder_config = SimpleNamespace(is_encoder_decoder=True)

    with patch.object(
        encoder_module,
        "build_attention_metadata",
        return_value=metadata,
    ) as build_metadata:
        runner._create_attention_metadata()

    assert build_metadata.call_args.kwargs["cache_indirection"] is None


def test_multi_item_batch_stays_eager_even_when_a_graph_is_available() -> None:
    """`maybe_get_cuda_graph` hands back a captured graph before it rejects multi-item
    scoring, and its rejection sits behind `_capture_allowed`, so the runner must refuse
    the batch itself or the replay would silently drop `multi_item_part_lens`."""
    runner = object.__new__(EncoderRunner)

    @contextmanager
    def pad_batch(inputs, batch_size):
        yield dict(inputs)

    graph_runner = SimpleNamespace(
        pad_batch=pad_batch,
        # A captured hit: what a runtime batch matching a warmed key would get.
        maybe_get_cuda_graph=Mock(return_value=(object(), (1, 2, 2))),
        is_encoder_decoder=False,
    )
    runner._encoder_cuda_graph_runner = graph_runner
    inputs = {
        "input_ids": [11, 12],
        "seq_lens": [2],
        "multi_item_part_lens": [[1, 1]],
    }

    assert runner._prepare_encoder_graph_inputs(inputs, object()) is None
    graph_runner.maybe_get_cuda_graph.assert_not_called()


@pytest.mark.parametrize(
    ("input_ids", "sequence_lengths", "multi_item_part_lens", "message"),
    [
        ([1, 2], [], None, "at least one request"),
        ([1, 2], [1], None, "sum of seq_lens"),
        ([1, 2], [1, 1], [[1]], "provided for all requests or for none"),
        # A negative length passes the sum check and would reach attention metadata.
        ([1, 2], [3, -1], None, "must not be negative"),
        # An empty entry passes the cardinality check and would IndexError later.
        ([1, 2], [1, 1], [[1], []], "entries must not be empty"),
    ],
)
def test_packed_inputs_reject_inconsistent_request_boundaries(
    input_ids: list[int],
    sequence_lengths: list[int],
    multi_item_part_lens: list[list[int]] | None,
    message: str,
) -> None:
    runner = object.__new__(EncoderRunner)

    with pytest.raises(ValueError, match=message):
        runner.prepare_inputs(PackedInputs(input_ids, sequence_lengths, multi_item_part_lens))


@pytest.mark.parametrize(
    "reserved_name",
    [
        "input_ids",
        "seq_lens",
        "multi_item_part_lens",
        "attn_metadata",
        "return_context_logits",
    ],
)
def test_encoder_runner_rejects_model_inputs_owned_by_the_runner(reserved_name: str) -> None:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(enabled=False)

    with pytest.raises(ValueError, match=reserved_name):
        runner.prepare_inputs(PackedInputs([11, 12], [2], model_inputs={reserved_name: object()}))


def test_encoder_runner_forwards_model_inputs_to_eager_preparation() -> None:
    expected = EncoderPreparedInputs({}, sequence_lengths=[2])
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(enabled=False)
    runner._prepare_encoder_batch = Mock(return_value=expected)
    token_type_ids = torch.tensor([0, 1])

    actual = runner.prepare_inputs(
        PackedInputs([11, 12], [2], model_inputs={"token_type_ids": token_type_ids})
    )

    assert actual is expected
    runner._prepare_encoder_batch.assert_called_once_with(
        [11, 12],
        [2],
        multi_item_part_lens=None,
        model_inputs={"token_type_ids": token_type_ids},
        allow_cuda_graph=False,
    )


def _graph_enabled_runner(*, attention_backend=TrtllmAttention) -> EncoderRunner:
    """An EncoderRunner whose graph runner is enabled but has no declared extra inputs."""
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(
        enabled=True,
        extra_input_specs=[],
        supports_metadata_type=EncoderCUDAGraphRunner.supports_metadata_type,
    )
    runner._config = SimpleNamespace(attention_backend=attention_backend)
    runner._prepare_encoder_batch = Mock(
        return_value=EncoderPreparedInputs({}, sequence_lengths=[2])
    )
    return runner


def test_encoder_runner_rejects_undeclared_tensor_input_when_graph_is_usable() -> None:
    runner = _graph_enabled_runner()

    with pytest.raises(ValueError, match="not declared"):
        runner.prepare_inputs(
            PackedInputs([11, 12], [2], model_inputs={"token_type_ids": torch.tensor([0, 1])})
        )


def test_multi_item_batch_skips_extra_input_validation() -> None:
    """A multi-item batch always runs eagerly, so undeclared tensor inputs pass through."""
    runner = _graph_enabled_runner()
    token_type_ids = torch.tensor([0, 1])

    runner.prepare_inputs(
        PackedInputs([11, 12], [2], [[1, 1]], model_inputs={"token_type_ids": token_type_ids})
    )

    assert runner._prepare_encoder_batch.call_args.kwargs["allow_cuda_graph"] is False


def test_unsupported_attention_backend_skips_extra_input_validation() -> None:
    """A backend that cannot replay encoder graphs keeps every call eager."""
    runner = _graph_enabled_runner(attention_backend=SimpleNamespace(Metadata=AttentionMetadata))

    runner.prepare_inputs(
        PackedInputs([11, 12], [2], model_inputs={"token_type_ids": torch.tensor([0, 1])})
    )

    assert runner._prepare_encoder_batch.call_args.kwargs["allow_cuda_graph"] is False


def test_encoder_graph_supports_only_trtllm_attention_metadata() -> None:
    class _TrtllmSubclassMetadata(TrtllmAttentionMetadata):
        pass

    assert EncoderCUDAGraphRunner.supports_metadata_type(TrtllmAttentionMetadata)
    assert EncoderCUDAGraphRunner.supports_metadata_type(_TrtllmSubclassMetadata)
    assert not EncoderCUDAGraphRunner.supports_metadata_type(AttentionMetadata)


def test_encoder_runner_forwards_eager_model_inputs_and_gathers_logits() -> None:
    metadata = SimpleNamespace(on_update_kv_lens=Mock())
    model_forward = Mock(return_value={"logits": torch.arange(12).reshape(3, 4)})
    runner = object.__new__(EncoderRunner)
    runner._model_caller = model_forward
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
    runner._moe_load_balancer = load_balancer
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
    runner._dist = SimpleNamespace(world_size=world_size)
    runner._mapping = SimpleNamespace(dwdp_enabled=False)
    runner._encoder_cuda_graph_runner = SimpleNamespace(
        enabled=True,
        build_capture_sequence_lengths=Mock(return_value=[1]),
        extra_input_specs=[],
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


def _capture_input_runner(extra_input_specs: list[EncodeExtraInputSpec]) -> EncoderRunner:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = SimpleNamespace(extra_input_specs=extra_input_specs)
    runner._prepare_encoder_batch = Mock(
        return_value=EncoderPreparedInputs({}, sequence_lengths=[3, 5])
    )
    return runner


def test_capture_inputs_build_zero_stand_ins_for_declared_extra_inputs() -> None:
    runner = _capture_input_runner(
        [
            EncodeExtraInputSpec(name="feat", shape=("batch_size", 40), dtype="int32"),
            EncodeExtraInputSpec(name="embeds", shape=("num_tokens", 768), dtype="float16"),
        ]
    )

    # 8 tokens across 2 requests, so swapping the two symbolic sizes would fail.
    runner._prepare_capture_inputs([3, 5])

    input_ids, sequence_lengths = runner._prepare_encoder_batch.call_args.args
    model_inputs = runner._prepare_encoder_batch.call_args.kwargs["model_inputs"]
    assert input_ids == [0] * 8
    assert sequence_lengths == [3, 5]
    assert set(model_inputs) == {"feat", "embeds"}
    for name, shape, dtype in (
        ("feat", (2, 40), torch.int32),
        ("embeds", (8, 768), torch.float16),
    ):
        stand_in = model_inputs[name]
        assert stand_in.shape == shape
        assert stand_in.dtype == dtype
        assert not stand_in.any()


def test_capture_inputs_without_declared_extra_inputs_pass_no_model_inputs() -> None:
    runner = _capture_input_runner([])

    runner._prepare_capture_inputs([3, 5])

    assert runner._prepare_encoder_batch.call_args.kwargs["model_inputs"] is None


def test_encoder_release_clears_owned_graph_backend() -> None:
    runner = object.__new__(EncoderRunner)
    runner._encoder_cuda_graph_runner = Mock()

    runner.release_graphs()

    runner._encoder_cuda_graph_runner.clear.assert_called_once_with()
