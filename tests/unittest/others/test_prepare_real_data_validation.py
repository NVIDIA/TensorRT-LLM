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
"""Boundary validation for trtllm-bench real-dataset config keys (issue-agnostic).

The --dataset-*-key CLI options are user input, so a missing key must raise a
ValueError (which survives `python -O`) rather than a bare AssertionError.
"""

import pytest

from tensorrt_llm.bench.dataset.prepare_real_data import DatasetConfig


def _config(**overrides):
    base = dict(name="d", split="train", output_key=None)
    base.update(overrides)
    return DatasetConfig(**base)


def test_missing_input_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="input-key"):
        _config(input_key="missing").get_input({"text": "hi"})


def test_missing_prompt_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="prompt-key"):
        _config(prompt_key="missing").get_prompt({"text": "hi"})


def test_missing_image_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dataset-image-key"):
        _config(image_key="missing").get_images({"text": "hi"})


def test_missing_output_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="output-key"):
        _config(output_key="missing").get_output({"text": "hi"})


def test_present_key_is_returned() -> None:
    assert _config(input_key="text").get_input({"text": "hi"}) == "hi"
