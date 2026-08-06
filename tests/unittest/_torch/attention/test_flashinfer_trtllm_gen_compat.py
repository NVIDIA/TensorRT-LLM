# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest

from tensorrt_llm._torch.attention_backend.fmha import flashinfer_trtllm_gen


@pytest.fixture(autouse=True)
def reset_decode_num_args() -> None:
    flashinfer_trtllm_gen._trtllm_gen_decode_num_args = None


def test_decode_compat_uses_legacy_signature() -> None:
    calls = []

    def run_func(*args: object) -> None:
        calls.append(args)

    flashinfer_trtllm_gen._run_trtllm_gen_decode_compat(run_func, *range(31))
    flashinfer_trtllm_gen._run_trtllm_gen_decode_compat(run_func, *range(31))

    assert [len(args) for args in calls] == [31, 31]


def test_decode_compat_detects_and_caches_block_sparse_signature() -> None:
    calls = []

    def run_func(*args: object) -> None:
        calls.append(args)
        if len(args) == 31:
            raise TypeError("Mismatched number of arguments. Expected 33 but got 31 arguments")

    flashinfer_trtllm_gen._run_trtllm_gen_decode_compat(run_func, *range(31))
    flashinfer_trtllm_gen._run_trtllm_gen_decode_compat(run_func, *range(31))

    assert [len(args) for args in calls] == [31, 33, 33]
    assert calls[1][-2:] == (False, None)
    assert calls[2][-2:] == (False, None)


def test_decode_compat_does_not_hide_unrelated_type_errors() -> None:
    def run_func(*args: object) -> None:
        raise TypeError("kernel launch failed")

    with pytest.raises(TypeError, match="kernel launch failed"):
        flashinfer_trtllm_gen._run_trtllm_gen_decode_compat(run_func, *range(31))
