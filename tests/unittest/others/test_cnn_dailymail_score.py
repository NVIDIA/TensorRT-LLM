# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from types import SimpleNamespace

from tensorrt_llm.evaluate.cnn_dailymail import CnnDailymail


def test_compute_score_with_no_beams_returns_zero() -> None:
    # When outputs[0].outputs is empty the beam loop never runs; rouge1 was
    # then returned while unbound, raising UnboundLocalError. It should now
    # return the 0.0 default.
    evaluator = CnnDailymail.__new__(CnnDailymail)  # skip heavy __init__
    empty_request = SimpleNamespace(outputs=[])
    assert evaluator.compute_score([empty_request], references=[]) == 0.0
