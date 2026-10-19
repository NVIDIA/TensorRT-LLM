# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU validation of opt-in beam sampling graph configuration."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import (
    BeamSearchCudaGraphConfig,
    BeamSearchCudaGraphProfile,
    TorchLlmArgs,
)

pytestmark = pytest.mark.cpu_only


def test_disabled_by_default() -> None:
    assert TorchLlmArgs.model_fields["beam_search_cuda_graph_config"].default is None


@pytest.mark.parametrize("widths,tokens", [([4], 8), ([2, 3, 4], 3)])
def test_explicit_profile_round_trip(widths: list[int], tokens: int) -> None:
    config = BeamSearchCudaGraphConfig(
        profiles=[dict(batch_sizes=[1, 2], beam_widths=widths, max_tokens=tokens)]
    )
    assert BeamSearchCudaGraphConfig.model_validate_json(config.model_dump_json()) == config
    assert config.max_graphs == 192
    assert config.max_static_input_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_sizes": []},
        {"batch_sizes": [0]},
        {"beam_widths": []},
        {"beam_widths": [-1]},
        {"beam_widths": [2, 4], "max_tokens": 3},
        {"max_tokens": 0},
        {"early_stopping": 3},
        {"unknown": True},
    ],
)
def test_invalid_profile_rejected(changes: dict) -> None:
    with pytest.raises(ValidationError):
        BeamSearchCudaGraphProfile(
            **{**dict(batch_sizes=[1], beam_widths=[4], max_tokens=8), **changes}
        )


@pytest.mark.parametrize(
    "changes",
    [{"profiles": []}, {"max_graphs": 0}, {"max_static_input_bytes": -1}, {"unknown": True}],
)
def test_invalid_resource_policy_rejected(changes: dict) -> None:
    with pytest.raises(ValidationError):
        BeamSearchCudaGraphConfig(
            **{**dict(profiles=[dict(batch_sizes=[1], beam_widths=[4], max_tokens=8)]), **changes}
        )


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
        profiles=[dict(batch_sizes=[1, 2], beam_widths=[4], max_tokens=8)]
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


def test_llm_accepts_explicit_supported_configuration() -> None:
    config = BeamSearchCudaGraphConfig(
        profiles=[dict(batch_sizes=[1, 2], beam_widths=[4], max_tokens=8)]
    )
    args = TorchLlmArgs.model_construct(
        beam_search_cuda_graph_config=config, max_beam_width=4, max_batch_size=2, max_seq_len=16
    )
    assert args.validate_beam_search_cuda_graph_config() is args


def test_sampler_factory_preserves_profiles_and_resource_limits() -> None:
    from tensorrt_llm._torch.pyexecutor._util import create_torch_sampler_args

    config = BeamSearchCudaGraphConfig(
        profiles=[
            dict(batch_sizes=[1, 2, 1], beam_widths=[4], max_tokens=8),
            dict(batch_sizes=[1], beam_widths=[2, 3, 4], max_tokens=3),
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
    assert args.beam_search_graph_profiles[0].beam_widths == (4,) * 8
    assert args.beam_search_graph_profiles[0].batch_sizes == (1, 2)
    assert args.beam_search_graph_profiles[1].beam_widths == (2, 3, 4)
    assert not args.beam_search_graph_profiles[1].return_probs
    assert args.beam_search_graph_max_graphs == 12
    assert args.beam_search_graph_max_static_input_bytes == 123456
