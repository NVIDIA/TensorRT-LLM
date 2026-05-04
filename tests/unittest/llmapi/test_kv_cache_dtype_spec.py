# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm.llmapi.llm_args import parse_kv_cache_dtype_spec


def test_uniform_string():
    result = parse_kv_cache_dtype_spec("fp8", num_layers=4)
    assert result == {0: "fp8", 1: "fp8", 2: "fp8", 3: "fp8"}


def test_range_syntax():
    result = parse_kv_cache_dtype_spec("0-3:fp8,4-35:nvfp4", num_layers=36)
    for i in range(4):
        assert result[i] == "fp8"
    for i in range(4, 36):
        assert result[i] == "nvfp4"


def test_single_index_syntax():
    result = parse_kv_cache_dtype_spec("2:fp8", num_layers=4)
    assert result[2] == "fp8"
    for i in (0, 1, 3):
        assert result[i] == "auto"


def test_dict_input_passthrough():
    spec = {0: "fp8", 35: "nvfp4"}
    result = parse_kv_cache_dtype_spec(spec, num_layers=36)
    assert result[0] == "fp8"
    assert result[35] == "nvfp4"
    assert result[1] == "auto"


def test_descending_range_raises():
    with pytest.raises(ValueError, match="start.*must not exceed end"):
        parse_kv_cache_dtype_spec("5-2:fp8", num_layers=36)


def test_out_of_range_layer_raises():
    with pytest.raises(ValueError, match="out of range"):
        parse_kv_cache_dtype_spec("0-40:fp8", num_layers=36)


def test_missing_colon_in_segment_raises():
    # The full spec has a colon (triggers range parsing), but one segment is missing it.
    with pytest.raises(ValueError, match="Invalid per-layer dtype segment"):
        parse_kv_cache_dtype_spec("0-3,4-35:fp8", num_layers=36)


def test_dict_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        parse_kv_cache_dtype_spec({100: "fp8"}, num_layers=36)
