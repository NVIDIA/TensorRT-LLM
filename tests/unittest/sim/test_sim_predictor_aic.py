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
"""Unit tests for AIConfiguratorPredictor."""

import os

import pytest

from tensorrt_llm._torch.pyexecutor.sim_predictor import (
    InferTimePredictor,
    SimBatch,
    SimBatchRequest,
)

AIC_SYSTEMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "slop", "aiconfigurator",
    "src", "aiconfigurator", "systems")

# Skip all tests if AIC database is not available
pytestmark = pytest.mark.skipif(
    not os.path.isdir(AIC_SYSTEMS_DIR),
    reason="AIConfigurator systems directory not found")


@pytest.fixture(scope="module")
def aic_predictor():
    from tensorrt_llm._torch.pyexecutor.sim_predictor_aic import (
        AIConfiguratorPredictor,
    )
    return AIConfiguratorPredictor(
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_name="h100_sxm",
        backend_version="1.2.0rc5",
        database_path=AIC_SYSTEMS_DIR,
    )


def _prefill_batch(num_requests=1, input_len=128):
    reqs = [SimBatchRequest(input_length=input_len, past_kv_length=0)
            for _ in range(num_requests)]
    return SimBatch(
        num_context_requests=num_requests,
        num_context_tokens=num_requests * input_len,
        num_generation_requests=0,
        num_generation_tokens=0,
        requests=reqs)


def _decode_batch(num_requests=4, past_kv_len=128):
    reqs = [SimBatchRequest(input_length=1, past_kv_length=past_kv_len)
            for _ in range(num_requests)]
    return SimBatch(
        num_context_requests=0,
        num_context_tokens=0,
        num_generation_requests=num_requests,
        num_generation_tokens=num_requests,
        requests=reqs)


class TestAIConfiguratorPredictor:

    def test_is_infer_time_predictor(self, aic_predictor):
        assert isinstance(aic_predictor, InferTimePredictor)

    def test_prefill_returns_positive_time(self, aic_predictor):
        t = aic_predictor.predict(_prefill_batch())
        assert t > 0, f"Expected positive prefill time, got {t}"

    def test_decode_returns_positive_time(self, aic_predictor):
        t = aic_predictor.predict(_decode_batch())
        assert t > 0, f"Expected positive decode time, got {t}"

    def test_larger_batch_takes_longer(self, aic_predictor):
        t1 = aic_predictor.predict(_decode_batch(num_requests=1))
        t4 = aic_predictor.predict(_decode_batch(num_requests=4))
        assert t4 > t1, f"bs=4 ({t4:.4f}s) should be slower than bs=1 ({t1:.4f}s)"

    def test_longer_sequence_takes_longer(self, aic_predictor):
        t_short = aic_predictor.predict(_prefill_batch(input_len=64))
        t_long = aic_predictor.predict(_prefill_batch(input_len=512))
        assert t_long > t_short, (
            f"isl=512 ({t_long:.4f}s) should be slower than isl=64 ({t_short:.4f}s)")

    def test_empty_requests_returns_zero(self, aic_predictor):
        b = SimBatch(num_context_requests=0, num_context_tokens=0,
                     num_generation_requests=0, num_generation_tokens=0,
                     requests=[])
        assert aic_predictor.predict(b) == 0.0

    def test_scale_factors(self):
        from tensorrt_llm._torch.pyexecutor.sim_predictor_aic import (
            AIConfiguratorPredictor,
        )
        base = AIConfiguratorPredictor(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_name="h100_sxm", backend_version="1.2.0rc5",
            database_path=AIC_SYSTEMS_DIR)
        scaled = AIConfiguratorPredictor(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_name="h100_sxm", backend_version="1.2.0rc5",
            database_path=AIC_SYSTEMS_DIR,
            prefill_scale_factor=2.0)

        batch = _prefill_batch()
        t_base = base.predict(batch)
        t_scaled = scaled.predict(batch)
        assert abs(t_scaled - 2.0 * t_base) < 1e-6, (
            f"Scaled ({t_scaled:.6f}) should be 2x base ({t_base:.6f})")


class TestAIConfiguratorPredictorErrors:

    def test_invalid_device_raises(self):
        from tensorrt_llm._torch.pyexecutor.sim_predictor_aic import (
            AIConfiguratorPredictor,
        )
        with pytest.raises(ValueError, match="Failed to load AIC database"):
            AIConfiguratorPredictor(
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_name="nonexistent_gpu",
                backend_version="1.2.0rc5",
                database_path=AIC_SYSTEMS_DIR)
