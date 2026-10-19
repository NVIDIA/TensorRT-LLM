# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU validation of service-side beam sampling graph capture ranges."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import (
    BeamSearchCudaGraphCaptureConfig,
    BeamSearchCudaGraphConfig,
    TorchLlmArgs,
)

pytestmark = pytest.mark.cpu_only


def test_disabled_by_default() -> None:
    assert TorchLlmArgs.model_fields["beam_search_cuda_graph_config"].default is None


@pytest.mark.parametrize("widths,tokens", [([4], 8), ([2, 3, 4], 3), ([2, 4], 5), ([2, 4, 8], 2)])
def test_capture_config_round_trip(widths: list[int], tokens: int) -> None:
    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=widths, max_capture_tokens=tokens)]
    )
    assert BeamSearchCudaGraphConfig.model_validate_json(config.model_dump_json()) == config
    assert config.batch_sizes is None and config.max_batch_size == 0
    assert config.max_graphs == 192
    assert config.max_static_input_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    "changes",
    [
        {"beam_width_array": []},
        {"beam_width_array": [0]},
        {"beam_width_array": [4, 2]},
        {"max_capture_tokens": 0},
        {"early_stopping": 3},
        {"batch_sizes": [1]},
        {"beam_widths": [4]},
        {"max_tokens": 8},
        {"unknown": True},
    ],
)
def test_invalid_capture_config_rejected(changes: dict) -> None:
    with pytest.raises(ValidationError):
        BeamSearchCudaGraphCaptureConfig(
            **{**dict(beam_width_array=[4], max_capture_tokens=8), **changes}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"capture_configs": []},
        {"batch_sizes": []},
        {"batch_sizes": [0]},
        {"max_batch_size": -1},
        {"batch_sizes": [1, 2], "max_batch_size": 4},
        {"max_graphs": 0},
        {"max_static_input_bytes": -1},
        {"profiles": []},
        {"enable_padding": True},
        {"unknown": True},
    ],
)
def test_invalid_batch_or_resource_policy_rejected(changes: dict) -> None:
    with pytest.raises(ValidationError):
        BeamSearchCudaGraphConfig(
            **{
                **dict(capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)]),
                **changes,
            }
        )


@pytest.mark.parametrize(
    "options,capacity,expected",
    [
        ({}, 4, (1, 2, 3, 4)),
        ({}, 2048, (*range(1, 33), 64, 128)),
        ({"max_batch_size": 3}, 64, (1, 2, 3)),
        ({"max_batch_size": 40}, 64, (*range(1, 33), 40)),
        ({"batch_sizes": [4, 1, 4, 2]}, 64, (1, 2, 4)),
        ({"batch_sizes": [2, 4], "max_batch_size": 4}, 64, (2, 4)),
    ],
)
def test_batch_policy(options: dict, capacity: int, expected: tuple[int, ...]) -> None:
    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)], **options
    )
    assert config.resolve_batch_sizes(capacity) == expected
    if config.batch_sizes is not None:
        assert config.max_batch_size == max(expected)
    assert BeamSearchCudaGraphConfig.model_validate_json(config.model_dump_json()) == config


def test_automatic_batches_do_not_mutate_shared_config() -> None:
    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)]
    )
    before = config.model_dump_json()
    assert config.resolve_batch_sizes(2) == (1, 2)
    assert config.resolve_batch_sizes(4) == (1, 2, 3, 4)
    assert config.model_dump_json() == before


@pytest.mark.parametrize("options", [{"max_batch_size": 3}, {"batch_sizes": [1, 3]}])
def test_explicit_batches_cannot_exceed_executor_capacity(options: dict) -> None:
    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)], **options
    )
    with pytest.raises(ValueError, match="max_batch_size"):
        config.resolve_batch_sizes(2)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_beam_width": 2}, "max_beam_width"),
        ({"max_batch_size": 1}, "max_batch_size"),
        ({"max_seq_len": 8}, "max_seq_len"),
        ({"tensor_parallel_size": 2}, "single GPU"),
        ({"pipeline_parallel_size": 2}, "single GPU"),
        ({"context_parallel_size": 2}, "single GPU"),
        ({"sampler_force_async_worker": True}, "synchronous"),
        ({"encode_only": True}, "synchronous"),
    ],
)
def test_llm_capacity_and_execution_mode_validation(overrides: dict, match: str) -> None:
    config = BeamSearchCudaGraphConfig(
        batch_sizes=[1, 2],
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)],
    )
    args = TorchLlmArgs.model_construct(
        **{
            **dict(
                beam_search_cuda_graph_config=config,
                max_beam_width=4,
                max_batch_size=2,
                max_seq_len=16,
            ),
            **overrides,
        }
    )
    with pytest.raises(ValueError, match=match):
        args.validate_beam_search_cuda_graph_config()


def test_llm_accepts_supported_configuration_without_batch_declaration() -> None:
    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)]
    )
    args = TorchLlmArgs.model_construct(
        beam_search_cuda_graph_config=config, max_beam_width=4, max_batch_size=2, max_seq_len=16
    )
    assert args.validate_beam_search_cuda_graph_config() is args
    assert config.batch_sizes is None


def test_sampler_factory_preserves_capture_combinations_and_resource_limits() -> None:
    from tensorrt_llm._torch.pyexecutor._util import create_torch_sampler_args

    config = BeamSearchCudaGraphConfig(
        batch_sizes=[2, 1, 2],
        capture_configs=[
            dict(beam_width_array=[4], max_capture_tokens=8),
            dict(beam_width_array=[2, 3, 4], max_capture_tokens=3),
            dict(beam_width_array=[2, 4], max_capture_tokens=4, early_stopping=0),
            dict(beam_width_array=[2, 3, 4], max_capture_tokens=2, early_stopping=2),
        ],
        max_graphs=12,
        max_static_input_bytes=123456,
    )
    args = create_torch_sampler_args(
        max_seq_len=16,
        speculative_config=None,
        max_beam_width=4,
        disable_overlap_scheduler=False,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
        max_num_sequences=4,
        beam_search_cuda_graph_config=config,
    )
    captures = args.beam_search_graph_profiles
    assert [capture.beam_widths for capture in captures] == [
        (4,) * 8,
        (2, 3, 4),
        (2, 4, 4, 4),
        (2, 3),
    ]
    assert [capture.early_stopping for capture in captures] == [1, 1, 0, 2]
    assert all(capture.batch_sizes == (1, 2) for capture in captures)
    assert all(not capture.return_probs for capture in captures)
    assert args.beam_search_graph_max_graphs == 12
    assert args.beam_search_graph_max_static_input_bytes == 123456


def test_instantiation_resolves_capture_batches_without_changing_slot_capacity(monkeypatch) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor import _util

    config = BeamSearchCudaGraphConfig(
        capture_configs=[dict(beam_width_array=[4], max_capture_tokens=8)]
    )
    llm_args = TorchLlmArgs.model_construct(beam_search_cuda_graph_config=config)
    monkeypatch.setattr(_util, "confidential_compute_enabled", lambda: False)
    monkeypatch.setattr(_util, "resolve_max_num_sequences", lambda *args, **kwargs: 8)
    captured = {}

    class FactoryReached(Exception):
        pass

    def record(**kwargs):
        captured.update(kwargs)
        raise FactoryReached

    monkeypatch.setattr(_util, "create_torch_sampler_args", record)
    with pytest.raises(FactoryReached):
        _util.instantiate_sampler(
            SimpleNamespace(max_seq_len=16),
            llm_args,
            None,
            max_batch_size=2,
            max_beam_width=4,
            mm_encoder_only=False,
            speculative_config=None,
        )
    assert captured["max_num_sequences"] == 8
    resolved = captured["beam_search_cuda_graph_config"]
    assert resolved.batch_sizes == [1, 2] and resolved.max_batch_size == 2
    assert config.batch_sizes is None and config.max_batch_size == 0
