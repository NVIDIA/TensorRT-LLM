# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``get_spec_metadata`` must carry advanced_sampling_mode onto the metadata.

The mode has two readers that must agree:

* ``SpecSampler.validate_request`` reads it off the **config**, and decides whether a
  min_p request is admitted at all;
* ``populate_sampling_params_for_one_model`` (via ``fill_min_p``) and the sampling
  dispatcher read it off the **metadata**, and decide whether the buffers are filled and
  which backend runs.

A disagreement is silent: the request is admitted, its min_p buffers stay at the neutral
sentinel, and the dispatcher routes to a backend that takes no min_p argument, so the
filter is dropped without an error. Neither the op-level nor the buffer-level tests can
see it, because each constructs its own metadata and therefore asserts on a mode the
factory may never produce.

These tests exercise the factory, which is the only place that disagreement can appear.
"""

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


def test_universal_is_not_silently_downgraded():
    """The one direction that loses min_p, stated on its own.

    UNIVERSAL on the config and FULL on the metadata is the combination that admits a
    min_p request and then drops the filter, so it is asserted separately from the
    parametrized sweep rather than left to be inferred from it.
    """
    metadata = get_spec_metadata(
        _eagle3_one_model(AdvancedSamplingMode.UNIVERSAL),
        _ModelConfig(),
        max_num_requests=4,
        max_num_tokens=64,
    )
    assert metadata is not None
    assert metadata.advanced_sampling_mode.is_universal
