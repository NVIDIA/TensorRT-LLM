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
from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.llm_request import reported_cached_tokens


def _request(gen_only: bool, cached_tokens: int, ctx_usage=None, prepopulated_prompt_len: int = 0):
    return SimpleNamespace(
        is_generation_only_request=lambda: gen_only,
        cached_tokens=cached_tokens,
        prepopulated_prompt_len=prepopulated_prompt_len,
        py_disaggregated_params=SimpleNamespace(ctx_usage=ctx_usage)
        if ctx_usage is not None
        else None,
    )


def test_context_request_keeps_engine_value():
    assert reported_cached_tokens(_request(False, 320)) == 320


def test_generation_only_without_ctx_usage_reports_local_reuse():
    # The generation path first-sets cached_tokens to the full prompt; that is
    # transferred KV, not a prefix-cache hit. Only the prefix this worker
    # reused from its own cache instead of transferring counts.
    assert reported_cached_tokens(_request(True, 4000)) == 0
    assert reported_cached_tokens(_request(True, 4000, prepopulated_prompt_len=256)) == 256


def test_generation_only_adopts_ctx_usage_dict():
    ctx = {"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 96}}
    assert reported_cached_tokens(_request(True, 4000, ctx)) == 96


def test_generation_only_adopts_ctx_usage_object():
    ctx = SimpleNamespace(
        prompt_tokens=4000, prompt_tokens_details=SimpleNamespace(cached_tokens=64)
    )
    assert reported_cached_tokens(_request(True, 4000, ctx)) == 64


def test_generation_only_ctx_usage_without_details_falls_back():
    assert (
        reported_cached_tokens(
            _request(True, 4000, {"prompt_tokens": 4000}, prepopulated_prompt_len=128)
        )
        == 128
    )


def test_generation_only_ctx_usage_wins_over_local_reuse():
    ctx = {"prompt_tokens_details": {"cached_tokens": 96}}
    assert reported_cached_tokens(_request(True, 4000, ctx, prepopulated_prompt_len=256)) == 96
