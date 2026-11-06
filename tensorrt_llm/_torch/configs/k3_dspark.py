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

from transformers.configuration_utils import PretrainedConfig


# The MLA-backboned DSpark drafter (Inferact/Kimi-K3-DSpark) ships a config.json
# with model_type "k3_dspark", no auto_map and no modeling code, so
# AutoConfig.from_pretrained cannot resolve it. Same workaround as LagunaConfig:
# the fields are plain attributes, and MLADSparkForCausalLM reads them directly.
class K3DsparkConfig(PretrainedConfig):
    model_type = "k3_dspark"
