# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Timer:

    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag):
        self._start_times[tag] = time.time()

    def stop(self, tag) -> float:
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag) -> float:
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self):
        self._start_times.clear()
        self._total_elapsed_times.clear()

    def summary(self):
        print('Profile Results')
        for tag, elapsed_time in self._total_elapsed_times.items():
            print(f' - {tag.ljust(30, ".")}: {elapsed_time:.6f} (sec)')


_default_timer = Timer()


def start(tag):
    _default_timer.start(tag)


def stop(tag):
    return _default_timer.stop(tag)


def elapsed_time_in_sec(tag):
    return _default_timer.elapsed_time_in_sec(tag)


def reset():
    _default_timer.reset()


def summary():
    _default_timer.summary()
