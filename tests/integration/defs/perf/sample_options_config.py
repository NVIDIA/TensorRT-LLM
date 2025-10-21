# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# -*- coding: utf-8 -*-
"""
Sample options config for trtllm-bench perf tests
"""


def get_sample_options_config(model_label: str) -> dict:
    """
    Return the sample options config corresponding to the model label.
    Args:
        model_label: model label from self._config.to_string()
    Returns:
        dict: sample options config
    """
    base_config = {
        'top_k': 4,
        'top_p': 0.5,
        'temperature': 0.5,
    }
    return base_config
