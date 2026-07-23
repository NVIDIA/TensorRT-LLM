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

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def test_layerwise_loading_defaults_to_false_and_is_prototype():
    args = TorchLlmArgs(model="gpt2")

    assert args.enable_hf_layerwise_loading is False
    field = TorchLlmArgs.model_fields["enable_hf_layerwise_loading"]
    assert field.json_schema_extra["status"] == "prototype"


def test_layerwise_loading_accepts_true_and_round_trips():
    args = TorchLlmArgs(model="gpt2", enable_hf_layerwise_loading=True)

    data = args.model_dump()
    assert data["enable_hf_layerwise_loading"] is True
    assert TorchLlmArgs(**data).enable_hf_layerwise_loading is True
