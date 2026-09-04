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
"""Load STS calibration and resolve the DSpark confidence head.

``apply_sts`` computes ``confidence[r][j] = sigmoid(logit[r][j] / T[j])``. The
planner consumes the cumulative product ``survival[r][j]``, in which
per-position calibration error compounds geometrically, so the table must be
fitted against the survival. Collection and fitting live in offline tooling and
do not run in serving.
"""

import json
from typing import List

__all__ = [
    "load_sts_temperatures_from_path",
    "resolve_confidence_head",
]


def load_sts_temperatures_from_path(path: str) -> List[float]:
    """Read a temperature vector, accepting either spelling of the key.

    TRT-LLM calibration artifacts use ``sts_temperatures``; SGLang uses
    ``temperatures``. The vectors are interchangeable.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("sts_temperatures", "temperatures"):
        if key in payload:
            temps = payload[key]
            break
    else:
        raise KeyError(
            f"{path} has neither 'sts_temperatures' nor 'temperatures'; "
            f"found keys {sorted(payload)}"
        )
    if not temps:
        raise ValueError(f"{path} carries an empty temperature vector")
    return [float(t) for t in temps]


def resolve_confidence_head(draft_model):
    """Find the confidence head across the known draft-model layouts.

    Two layouts exist: the bare ``DSparkDraftModel`` (stages under
    ``.mtp_layers``, head on the last stage) and the ``DSparkForCausalLM``
    wrapper (bare model under ``.dspark_model``). Written once and unit-tested
    against both; returns None when no head is found.
    """
    inner = getattr(draft_model, "dspark_model", draft_model)
    stages = getattr(inner, "mtp_layers", None)
    if not stages:
        return None
    return getattr(stages[-1], "confidence_head", None)
