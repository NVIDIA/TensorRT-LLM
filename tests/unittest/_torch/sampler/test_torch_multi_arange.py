# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from itertools import product
from typing import Iterable, Optional

import numpy as np
import pytest
import torch
from utils.util import assert_no_cuda_sync, force_ampere

from tensorrt_llm._torch.pyexecutor.sampling_utils import (ACCEPT_SYNC_COMPUTE,
                                                           torch_multi_arange)

BASE_CASES = [
    (None, [], None, []),
    ([], [], None, []),
    (None, [], [], []),
    ([], [], [], []),
    (None, [1], None, [0]),
    (None, [-1], None, []),
    (None, [3], None, [0, 1, 2]),
    (None, [-3], None, []),
    ([-5], [-3], None, [-5, -4]),
    ([-5], [-2], [2], [-5, -3]),
    ([-5], [-1], [2], [-5, -3]),
    ([-5], [-3], [3], [-5]),
    ([-3], [-5], None, []),
    ([-3], [-5], [-1], [-3, -4]),
    ([-3], [-5], [-3], [-3]),
    ([-3], [-5], [1], []),
    ([-5], [-3], [-2], []),
    ([-3], [2], None, [-3, -2, -1, 0, 1]),
    ([-3], [2], [2], [-3, -1, 1]),
    ([-3], [3], [2], [-3, -1, 1]),
    ([2], [5], None, [2, 3, 4]),
    ([2], [5], [2], [2, 4]),
    ([2], [6], [2], [2, 4]),
]


def _build_multi_arange_case() -> tuple[Iterable, Iterable, Iterable, Iterable]:
    gen = np.random.default_rng(seed=42)
    cases = [
        BASE_CASES[i] for i in gen.choice(len(BASE_CASES), 128)
        if len(BASE_CASES[i][3]) > 0
    ]
    starts = [
        val for case in cases
        for val in (case[0] if case[0] is not None else [0] * len(case[1]))
    ]
    ends = [val for case in cases for val in case[1]]
    steps = [
        val for case in cases
        for val in (case[2] if case[2] is not None else [1] * len(case[1]))
    ]
    expected = [val for case in cases for val in case[3]]
    return starts, ends, steps, expected


@force_ampere
@pytest.mark.parametrize(
    "device, allow_sync, dtype, starts, ends, steps, expected",
    [
        pytest.param(device, allow_sync, dtype, starts, ends, steps, expected)
        for (dtype,
             (starts, ends, steps, expected), device, allow_sync) in product(
                 [
                     torch.int32,
                     torch.int64,
                 ],
                 BASE_CASES + [_build_multi_arange_case()],
                 [
                     "cpu",
                     "cuda",
                 ],
                 [False, True],
             ) if device == "cuda" or allow_sync
    ],
)
def test_torch_multi_arange(
    device: str,
    allow_sync: bool,
    dtype: torch.dtype,
    starts: Optional[Iterable],
    ends: Iterable,
    steps: Optional[Iterable],
    expected: Iterable,
):
    torch_device = torch.device(device)

    def _make_tensor(data: Iterable) -> torch.Tensor:
        return torch.tensor(data, device=torch_device, dtype=dtype)

    def _maybe_make_tensor(data: Optional[Iterable]) -> Optional[torch.Tensor]:
        if data is None:
            return None
        return _make_tensor(data)

    starts_tensor = _maybe_make_tensor(starts)
    ends_tensor = _make_tensor(ends)
    steps_tensor = _maybe_make_tensor(steps)
    expected_tensor = _make_tensor(expected)

    extra_args = {}
    extra_args["output_length"] = ACCEPT_SYNC_COMPUTE
    if device != "cpu":
        # Pre-allocates a large chunk of memory, because PyTorch caching memory allocator
        # can sync otherwise.
        buf = torch.ones((2**30, ), device=device)
        del buf
        if not allow_sync:
            extra_args["output_length"] = expected_tensor.numel()
        # Warmup to avoid syncs due to lazy loading of kernels
        _ = torch_multi_arange(
            ends_tensor,
            starts=starts_tensor,
            steps=steps_tensor,
            **extra_args,
        )

    with torch.cuda.Stream():
        with assert_no_cuda_sync() if not allow_sync else nullcontext():
            result = torch_multi_arange(
                ends_tensor,
                starts=starts_tensor,
                steps=steps_tensor,
                **extra_args,
            )

    torch.testing.assert_close(result, expected_tensor)
