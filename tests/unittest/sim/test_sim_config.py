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
"""Unit tests for SimConfig and PredictorConfig."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.sim_config import PredictorConfig, SimConfig


class TestPredictorConfig:

    def test_defaults(self):
        config = PredictorConfig()
        assert config.name == "constant"
        assert config.constant_prefill_time_ms == 10.0
        assert config.constant_decode_time_ms == 5.0

    def test_custom_values(self):
        config = PredictorConfig(
            constant_prefill_time_ms=20.0,
            constant_decode_time_ms=8.0)
        assert config.constant_prefill_time_ms == 20.0
        assert config.constant_decode_time_ms == 8.0

    def test_zero_times_allowed(self):
        config = PredictorConfig(
            constant_prefill_time_ms=0.0,
            constant_decode_time_ms=0.0)
        assert config.constant_prefill_time_ms == 0.0
        assert config.constant_decode_time_ms == 0.0

    def test_negative_prefill_time_rejected(self):
        with pytest.raises(ValidationError):
            PredictorConfig(constant_prefill_time_ms=-1.0)

    def test_negative_decode_time_rejected(self):
        with pytest.raises(ValidationError):
            PredictorConfig(constant_decode_time_ms=-1.0)

    def test_invalid_predictor_name_rejected(self):
        with pytest.raises(ValidationError):
            PredictorConfig(name="nonexistent")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            PredictorConfig(bogus_field=True)


class TestSimConfig:

    def test_defaults(self):
        config = SimConfig()
        assert config.predictor.name == "constant"
        assert config.predictor.constant_prefill_time_ms == 10.0

    def test_custom_predictor(self):
        config = SimConfig(
            predictor=PredictorConfig(constant_prefill_time_ms=50.0))
        assert config.predictor.constant_prefill_time_ms == 50.0

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            SimConfig(invalid_field=True)

    def test_model_dump_roundtrip(self):
        config = SimConfig(
            predictor=PredictorConfig(
                constant_prefill_time_ms=25.0,
                constant_decode_time_ms=12.0))
        data = config.model_dump()
        restored = SimConfig(**data)
        assert restored.predictor.constant_prefill_time_ms == 25.0
        assert restored.predictor.constant_decode_time_ms == 12.0
