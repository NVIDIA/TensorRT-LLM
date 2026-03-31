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
"""Batch time predictors for simulation mode."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SimBatch:
    """Lightweight batch description for the time predictor.

    Decoupled from ScheduledRequests so predictors can be tested
    without TRT-LLM internals.
    """

    num_context_requests: int
    num_context_tokens: int
    num_generation_requests: int
    num_generation_tokens: int

    @property
    def is_prefill(self) -> bool:
        """True if this batch contains any context (prefill) requests."""
        return self.num_context_requests > 0


class InferTimePredictor(ABC):
    """Abstract base class for batch execution time predictors."""

    @abstractmethod
    def predict(self, batch: SimBatch) -> float:
        """Predict batch execution time in seconds.

        Args:
            batch: Description of the batch to predict timing for.

        Returns:
            Predicted execution time in seconds. Return 0.0 for instant.
        """


class ConstantPredictor(InferTimePredictor):
    """Returns a fixed time per batch, based on whether it is prefill or decode.

    Args:
        prefill_time_ms: Fixed time in milliseconds for prefill batches.
        decode_time_ms: Fixed time in milliseconds for decode batches.
    """

    def __init__(self, prefill_time_ms: float, decode_time_ms: float):
        self._prefill_s = prefill_time_ms / 1000.0
        self._decode_s = decode_time_ms / 1000.0

    def predict(self, batch: SimBatch) -> float:
        if batch.is_prefill:
            return self._prefill_s
        return self._decode_s
