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
"""Simulation mode configuration."""

from typing import Literal

from pydantic import Field, NonNegativeFloat

from tensorrt_llm.llmapi.utils import StrictBaseModel


class PredictorConfig(StrictBaseModel):
    """Configuration for the batch time predictor."""

    name: Literal["constant"] = Field(
        default="constant",
        description="Predictor type. 'constant' returns fixed time per batch.")

    constant_prefill_time_ms: NonNegativeFloat = Field(
        default=10.0,
        description="Fixed prefill (context) batch time in milliseconds.")

    constant_decode_time_ms: NonNegativeFloat = Field(
        default=5.0,
        description="Fixed decode (generation) batch time in milliseconds.")


class SimConfig(StrictBaseModel):
    """Simulation mode configuration.

    When set on TorchLlmArgs, enables GPU-free simulation: the real
    scheduler runs but model forward is replaced with predicted timing.
    """

    predictor: PredictorConfig = Field(
        default_factory=PredictorConfig,
        description="Time predictor configuration.")
