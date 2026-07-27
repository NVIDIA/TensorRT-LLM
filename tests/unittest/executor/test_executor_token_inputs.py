# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.sampling_params import SamplingParams


class _FakeExecutor(GenerationExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[GenerationRequest] = []

    def submit(self, request: GenerationRequest) -> GenerationResult:
        self.submitted.append(request)
        return MagicMock()

    def abort_request(self, request_id: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_generate_async_accepts_numpy_token_ids(dtype: type[np.integer]) -> None:
    executor = _FakeExecutor()
    token_ids = np.array([1, 2, 3], dtype=dtype)

    executor.generate_async(token_ids, SamplingParams(max_tokens=1))

    assert executor.submitted[0].prompt_token_ids == token_ids.tolist()


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_generate_treats_numpy_token_ids_as_unbatched(dtype: type[np.integer]) -> None:
    executor = _FakeExecutor()
    token_ids = np.array([1, 2, 3], dtype=dtype)

    result = executor.generate(token_ids, SamplingParams(max_tokens=1))

    assert len(executor.submitted) == 1
    result.result.assert_called_once_with()
