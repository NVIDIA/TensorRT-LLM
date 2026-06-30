# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm.runtime import model_runner_cpp as mrc


def test_map_fsd_divergence_type_strings():
    assert mrc._map_fsd_divergence_type("js") == 0
    assert mrc._map_fsd_divergence_type("JSD") == 0
    assert mrc._map_fsd_divergence_type("kl") == 1
    assert mrc._map_fsd_divergence_type("tv") == 2
    assert mrc._map_fsd_divergence_type("reverse_kl") == 3


def test_map_fsd_divergence_type_invalid():
    with pytest.raises(ValueError):
        mrc._map_fsd_divergence_type("unknown")
    with pytest.raises(ValueError):
        mrc._map_fsd_divergence_type(4)
    with pytest.raises(TypeError):
        mrc._map_fsd_divergence_type(0.5)


def test_normalize_fsd_param():
    values = mrc._normalize_fsd_param(0.1, 3, "fsd_threshold", mrc._validate_fsd_threshold)
    assert values == [0.1, 0.1, 0.1]

    values = mrc._normalize_fsd_param([0.1, 0.2], 2, "fsd_threshold", mrc._validate_fsd_threshold)
    assert values == [0.1, 0.2]

    with pytest.raises(ValueError):
        mrc._normalize_fsd_param([0.1], 2, "fsd_threshold", mrc._validate_fsd_threshold)

    with pytest.raises(ValueError):
        mrc._validate_fsd_threshold(-0.1)
