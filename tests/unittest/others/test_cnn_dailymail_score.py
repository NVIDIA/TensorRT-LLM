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

import pytest

from tensorrt_llm.evaluate.cnn_dailymail import CnnDailymail

# The CPU CI stage runs pytest with -m cpu_only and skips files without this marker.
pytestmark = pytest.mark.cpu_only


def _evaluator(rouge1: float = 0.0) -> CnnDailymail:
    # Skip __init__ (it downloads the dataset and the rouge metric) and stub
    # the only dependency compute_score touches.
    evaluator = CnnDailymail.__new__(CnnDailymail)
    evaluator.rouge = SimpleNamespace(compute=lambda predictions, references: {"rouge1": rouge1})
    return evaluator


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [SimpleNamespace(outputs=[])],
    ],
)
def test_compute_score_degenerate_input_returns_zero(outputs) -> None:
    # Previously: IndexError on an empty request list, UnboundLocalError on a
    # request with no beams. Both now return the 0.0 default.
    assert _evaluator().compute_score(outputs, references=[]) == 0.0


def test_compute_score_returns_rouge1_percentage() -> None:
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(text="a summary")])]
    score = _evaluator(rouge1=0.4321).compute_score(outputs, references=["a summary"])
    assert score == pytest.approx(43.21)
