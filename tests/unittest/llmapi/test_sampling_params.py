# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm.sampling_params import MIN_SAMPLING_TEMPERATURE, SamplingParams


def test_temperature_none_unchanged():
    assert SamplingParams().temperature is None


def test_temperature_zero_kept_for_greedy():
    # 0.0 means greedy decoding and must not be clamped.
    assert SamplingParams(temperature=0.0).temperature == 0.0


@pytest.mark.parametrize("tiny", [1e-12, 1e-6, MIN_SAMPLING_TEMPERATURE / 2])
def test_tiny_positive_temperature_clamped(tiny):
    assert SamplingParams(temperature=tiny).temperature == MIN_SAMPLING_TEMPERATURE


@pytest.mark.parametrize("temp", [MIN_SAMPLING_TEMPERATURE, 0.5, 1.0, 2.0])
def test_normal_temperature_unchanged(temp):
    assert SamplingParams(temperature=temp).temperature == temp


def test_negative_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-1.0)
