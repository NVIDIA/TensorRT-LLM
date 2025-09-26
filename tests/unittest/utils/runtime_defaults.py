# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Callable

from pytest import raises

from tensorrt_llm.bindings.executor import RuntimeDefaults


def assert_runtime_defaults_are_parsed_correctly(
    produce_defaults: Callable[[dict | None], RuntimeDefaults | None],
    *,
    strict_keys=True,
) -> None:
    # No Defaults
    assert produce_defaults(None) is None

    # Full Defaults
    defaults = produce_defaults({
        "max_attention_window": [2],
        "sink_token_length": 2
    })
    assert isinstance(defaults, RuntimeDefaults)
    assert defaults.max_attention_window == [2]
    assert defaults.sink_token_length == 2

    # Partial Defaults
    defaults = produce_defaults({"sink_token_length": 2})
    assert isinstance(defaults, RuntimeDefaults)
    assert defaults.sink_token_length == 2

    # Invalid Keys
    if strict_keys:
        with raises(TypeError):
            produce_defaults({"fake": 1})
    else:
        assert isinstance(produce_defaults({"fake": 1}), RuntimeDefaults)
