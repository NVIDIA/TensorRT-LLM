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

import importlib.util
from pathlib import Path

import pytest


_SOURCE_PATH = Path(__file__).resolve().parents[3] / "tensorrt_llm/math_utils.py"
_SPEC = importlib.util.spec_from_file_location("math_utils_under_test", _SOURCE_PATH)
_MATH_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MATH_UTILS)


@pytest.mark.parametrize(
    ("value", "divisor", "expected"),
    [
        (0, 4, 0),
        (8, 4, 2),
        (9, 4, 3),
        (1, 4, 1),
    ],
)
def test_ceil_div_rounds_up_to_next_integer_multiple(value, divisor, expected):
    assert _MATH_UTILS.ceil_div(value, divisor) == expected
