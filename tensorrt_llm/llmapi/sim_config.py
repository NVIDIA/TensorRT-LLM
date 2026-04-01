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

from typing import Any, Literal, Optional

from pydantic import Field, NonNegativeFloat, PrivateAttr, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel


class PredictorConfig(StrictBaseModel):
    """Configuration for the batch time predictor."""

    name: Literal["constant", "aiconfigurator"] = Field(
        default="constant",
        description="Predictor type. 'constant' returns fixed time per batch. "
        "'aiconfigurator' uses analytical per-operation predictions.")

    # --- Constant predictor fields ---
    constant_prefill_time_ms: NonNegativeFloat = Field(
        default=10.0,
        description="Fixed prefill (context) batch time in milliseconds. "
        "Only used when name='constant'.")

    constant_decode_time_ms: NonNegativeFloat = Field(
        default=5.0,
        description="Fixed decode (generation) batch time in milliseconds. "
        "Only used when name='constant'.")

    # --- AIConfigurator predictor fields ---
    device_name: Optional[str] = Field(
        default=None,
        description="AIC system name for target hardware, e.g. 'h100_sxm', "
        "'a100_sxm', 'b200_sxm'. Required when name='aiconfigurator'.")

    database_path: Optional[str] = Field(
        default=None,
        description="Custom path to AIC systems/ directory. "
        "Defaults to the bundled database in the aiconfigurator package.")

    backend_version: Optional[str] = Field(
        default=None,
        description="TRT-LLM version for AIC database lookup, e.g. '1.2.0rc5'. "
        "Required when name='aiconfigurator'.")

    prefill_scale_factor: float = Field(
        default=1.0, gt=0,
        description="Multiplicative correction factor for prefill predictions.")

    decode_scale_factor: float = Field(
        default=1.0, gt=0,
        description="Multiplicative correction factor for decode predictions.")

    @model_validator(mode='after')
    def validate_aiconfigurator_fields(self):
        if self.name == "aiconfigurator":
            if not self.device_name:
                raise ValueError(
                    "device_name is required when name='aiconfigurator'")
            if not self.backend_version:
                raise ValueError(
                    "backend_version is required when name='aiconfigurator'")
        return self


class SimConfig(StrictBaseModel):
    """Simulation mode configuration.

    When set on TorchLlmArgs, enables GPU-free simulation: the real
    scheduler runs but model forward is replaced with predicted timing.
    """

    predictor: PredictorConfig = Field(
        default_factory=PredictorConfig,
        description="Time predictor configuration.")

    _clock: Any = PrivateAttr(default=None)
