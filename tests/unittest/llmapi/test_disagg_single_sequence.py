# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated context-only and generation-only requests carry one sequence."""

from unittest.mock import Mock

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _llm() -> BaseLLM:
    llm = object.__new__(BaseLLM)
    llm._encode_only = False
    llm._executor = Mock()
    llm._executor.is_shutdown.return_value = False
    llm._prepare_sampling_params = lambda sampling_params: sampling_params
    return llm


@pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
@pytest.mark.parametrize("fields", [{"n": 2}, {"n": 1, "best_of": 2}])
def test_multi_sequence_disagg_request_is_rejected(request_type, fields):
    llm = _llm()
    sampling_params = SamplingParams(temperature=0.8, end_id=0, **fields)

    with pytest.raises(ValueError, match="return one sequence"):
        llm.generate_async(
            [1, 2, 3],
            sampling_params,
            disaggregated_params=DisaggregatedParams(request_type=request_type),
        )

    llm._executor.generate_async.assert_not_called()
