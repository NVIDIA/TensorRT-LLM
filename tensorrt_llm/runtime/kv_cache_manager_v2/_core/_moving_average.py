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

import time


class MovingAverage:
    __slots__ = ("decay", "avg", "weight", "num_updates")
    decay: float
    avg: float
    weight: float
    num_updates: int

    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.avg = 0.0
        self.weight = 0.0
        self.num_updates = 0

    def update(self, value: int | float) -> float:
        self.weight = 1.0 + self.decay * self.weight
        self.avg += (value - self.avg) / self.weight
        self.num_updates += 1
        return self.avg

    @property
    def value(self) -> float:
        return self.avg


class Average:
    __slots__ = ("sum", "count")
    sum: float
    count: int

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: int | float) -> None:
        self.sum += value
        self.count += 1

    @property
    def value(self) -> float:
        return self.sum / self.count


class TimeWeightedAverage:
    # Time-weighted lifetime mean of a piecewise-constant value. Each call to
    # update() closes the prior segment by adding current_value * (now - t_last)
    # to the integral, then sets current_value to the new sample. value reports
    # integral / (now - t_init), including the still-open segment up to now.
    # This matches the natural "average history length across the cache's life"
    # semantic and is immune to commit-driven bursts of update() calls — many
    # rapid samples contribute proportionally little because little wall time
    # elapses between them.
    __slots__ = ("integral", "current_value", "t_init", "t_last")
    integral: float
    current_value: float
    t_init: float
    t_last: float

    def __init__(self, initial_value: int | float):
        self.integral = 0.0
        self.current_value = float(initial_value)
        self.t_init = time.monotonic()
        self.t_last = self.t_init

    def update(self, value: int | float) -> None:
        t_now = time.monotonic()
        self.integral += self.current_value * (t_now - self.t_last)
        self.current_value = float(value)
        self.t_last = t_now

    @property
    def value(self) -> float:
        t_now = time.monotonic()
        integral = self.integral + self.current_value * (t_now - self.t_last)
        elapsed = t_now - self.t_init
        if elapsed <= 0.0:
            return self.current_value
        return integral / elapsed
