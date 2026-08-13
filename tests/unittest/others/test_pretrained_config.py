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
from utils.runtime_defaults import assert_runtime_defaults_are_parsed_correctly

from tensorrt_llm.models.modeling_utils import PretrainedConfig

pytestmark = pytest.mark.cpu_only


def test_pretrained_config_parses_runtime_defaults_correctly():
    assert_runtime_defaults_are_parsed_correctly(
        PretrainedConfig.create_runtime_defaults)
