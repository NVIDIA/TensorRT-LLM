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
"""Simulated clock for simulation mode."""


class SimClock:
    """Accumulates predicted iteration times for simulation mode.

    Not a singleton — owned by SimModelEngine as an instance attribute.
    Phase 4 will extend this to record per-iteration breakdown.
    """

    def __init__(self):
        self._total_time_s: float = 0.0
        self._num_iterations: int = 0

    def step(self, duration_s: float) -> None:
        """Advance clock by one iteration's predicted duration."""
        self._total_time_s += duration_s
        self._num_iterations += 1

    @property
    def total_time_s(self) -> float:
        return self._total_time_s

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    def reset(self) -> None:
        self._total_time_s = 0.0
        self._num_iterations = 0
