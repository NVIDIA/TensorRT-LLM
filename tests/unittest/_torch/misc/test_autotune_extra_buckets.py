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
"""TRTLLM_AUTOTUNE_EXTRA_TOKEN_BUCKETS: extra exact-M tuning buckets."""

import pytest

from tensorrt_llm._torch import utils


@pytest.fixture
def extra(monkeypatch):
    def _set(value):
        monkeypatch.setenv("TRTLLM_AUTOTUNE_EXTRA_TOKEN_BUCKETS", value)
        utils.autotune_extra_token_buckets.cache_clear()

    yield _set
    utils.autotune_extra_token_buckets.cache_clear()


def test_unset_is_power_of_two(extra):
    extra("")
    assert utils.autotune_extra_token_buckets() == ()
    assert utils.last_positive_power_of_2_or_extra(1536) == 1024
    assert utils.get_last_power_of_2_or_extra_num_tokens_buckets(
        1664
    ) == utils.get_last_power_of_2_num_tokens_buckets(1664)


def test_extra_buckets(extra):
    extra("1536,1280")
    assert utils.autotune_extra_token_buckets() == (1280, 1536)
    assert utils.last_positive_power_of_2_or_extra(1536) == 1536
    assert utils.last_positive_power_of_2_or_extra(1500) == 1280
    assert utils.last_positive_power_of_2_or_extra(1100) == 1024
    assert utils.last_positive_power_of_2_or_extra(2048) == 2048
    buckets = utils.get_last_power_of_2_or_extra_num_tokens_buckets(1664)
    assert buckets[-3:] == (1024, 1280, 1536)
    assert 2048 not in buckets


def test_bad_value(extra):
    extra("0,12")
    with pytest.raises(ValueError):
        utils.autotune_extra_token_buckets()
