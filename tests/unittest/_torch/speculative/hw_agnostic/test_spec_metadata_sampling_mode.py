# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for propagating advanced_sampling_mode into speculative metadata."""

import pytest

from tensorrt_llm._torch.speculative.utils import get_spec_metadata
from tensorrt_llm.llmapi.llm_args import (
    AdvancedSamplingMode,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
)


class _ModelConfig:
    num_hidden_layers = 32
    hidden_size = 128
    vocab_size = 1024
    torch_dtype = None


def _eagle3_one_model(mode):
    return Eagle3DecodingConfig(
        max_draft_len=2,
        speculative_model="/nonexistent",
        eagle3_one_model=True,
        advanced_sampling_mode=mode,
    )


def _mtp(mode):
    return MTPDecodingConfig(num_nextn_predict_layers=1, advanced_sampling_mode=mode)


@pytest.mark.parametrize("mode", list(AdvancedSamplingMode))
@pytest.mark.parametrize("make_config", [_eagle3_one_model, _mtp], ids=["eagle3_one_model", "mtp"])
def test_metadata_carries_the_configured_sampling_mode(make_config, mode):
    config = make_config(mode)
    metadata = get_spec_metadata(config, _ModelConfig(), max_num_requests=4, max_num_tokens=64)
    if metadata is None:
        pytest.skip("mode does not build one-model metadata")
    assert metadata.advanced_sampling_mode == mode, (
        f"{type(metadata).__name__} kept {metadata.advanced_sampling_mode} for a config "
        f"asking for {mode}; validate_request would admit min_p while the buffers stay "
        f"empty and the dispatcher picks the wrong backend"
    )


def test_fused_is_not_silently_downgraded():
    metadata = get_spec_metadata(
        _eagle3_one_model(AdvancedSamplingMode.FUSED),
        _ModelConfig(),
        max_num_requests=4,
        max_num_tokens=64,
    )
    assert metadata is not None
    assert metadata.advanced_sampling_mode.is_fused
