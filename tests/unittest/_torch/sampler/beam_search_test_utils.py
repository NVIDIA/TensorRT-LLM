# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.sampling_params import SamplingParams


def make_beam_request(beam_width_array: list[int] | None, max_beam_width: int = 4) -> LlmRequest:
    """A beam-search request carrying ``beam_width_array`` (VBWS)."""
    sampling_params = SamplingParams(
        n=max_beam_width,
        best_of=max_beam_width,
        use_beam_search=True,
        beam_width_array=beam_width_array,
    )
    return LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=16,
        input_tokens=[1, 2, 3],
        end_id=-1,
        sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
        is_streaming=False,
    )


def set_beam_iteration(request: LlmRequest, iteration: int) -> None:
    """Hold both decoding counters at ``iteration``.

    The Python width schedule is indexed with ``py_decoding_iter`` and the C++
    binding with ``decoding_iter``; ``_handle_responses`` copies the former into
    the latter once a step's responses are handled, which is the state every
    non-overlap step is in. Tests that walk the schedule set both, so they
    exercise the same iteration on either side -- see
    test_vbws_width_follows_python_iter_when_response_handling_lags for the
    overlap case, where the two differ.
    """
    request.py_decoding_iter = iteration
    request.decoding_iter = iteration
