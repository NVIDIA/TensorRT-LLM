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
"""Unit tests for InferTimePredictor and ConstantPredictor."""

from tensorrt_llm._torch.pyexecutor.sim_predictor import (
    ConstantPredictor,
    InferTimePredictor,
    SimBatch,
)


def _prefill_batch(num_requests=1, num_tokens=128):
    return SimBatch(
        num_context_requests=num_requests,
        num_context_tokens=num_tokens,
        num_generation_requests=0,
        num_generation_tokens=0)


def _decode_batch(num_requests=4):
    return SimBatch(
        num_context_requests=0,
        num_context_tokens=0,
        num_generation_requests=num_requests,
        num_generation_tokens=num_requests)


def _mixed_batch(num_ctx=1, ctx_tokens=128, num_gen=3):
    return SimBatch(
        num_context_requests=num_ctx,
        num_context_tokens=ctx_tokens,
        num_generation_requests=num_gen,
        num_generation_tokens=num_gen)


class TestSimBatch:

    def test_prefill_batch_is_prefill(self):
        assert _prefill_batch().is_prefill is True

    def test_decode_batch_is_not_prefill(self):
        assert _decode_batch().is_prefill is False

    def test_mixed_batch_is_prefill(self):
        assert _mixed_batch().is_prefill is True

    def test_empty_batch(self):
        b = SimBatch(num_context_requests=0, num_context_tokens=0,
                     num_generation_requests=0, num_generation_tokens=0)
        assert b.is_prefill is False


class TestConstantPredictor:

    def test_prefill_returns_prefill_time(self):
        p = ConstantPredictor(prefill_time_ms=10.0, decode_time_ms=5.0)
        assert p.predict(_prefill_batch()) == 0.01

    def test_decode_returns_decode_time(self):
        p = ConstantPredictor(prefill_time_ms=10.0, decode_time_ms=5.0)
        assert p.predict(_decode_batch()) == 0.005

    def test_mixed_batch_returns_prefill_time(self):
        p = ConstantPredictor(prefill_time_ms=10.0, decode_time_ms=5.0)
        assert p.predict(_mixed_batch()) == 0.01

    def test_zero_times(self):
        p = ConstantPredictor(prefill_time_ms=0.0, decode_time_ms=0.0)
        assert p.predict(_prefill_batch()) == 0.0
        assert p.predict(_decode_batch()) == 0.0

    def test_large_times(self):
        p = ConstantPredictor(prefill_time_ms=1000.0, decode_time_ms=500.0)
        assert p.predict(_prefill_batch()) == 1.0
        assert p.predict(_decode_batch()) == 0.5

    def test_is_infer_time_predictor(self):
        p = ConstantPredictor(prefill_time_ms=1.0, decode_time_ms=1.0)
        assert isinstance(p, InferTimePredictor)
