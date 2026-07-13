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
"""Unit tests for the --smoothquant_alpha override (quantize_by_modelopt)."""

import copy

import pytest

# quantize_by_modelopt imports modelopt/torch at module load; skip cleanly where absent.
pytest.importorskip("modelopt")
from tensorrt_llm.quantization.quantize_by_modelopt import override_smoothquant_alpha


def _int8_sq_cfg():
    """Minimal stand-in for an INT8_SMOOTHQUANT preset.

    Carries an ``algorithm`` with the ModelOpt default alpha=1.0, plus a nested
    ``quant_cfg`` to catch accidental mutation.
    """
    return {
        "quant_cfg": {"*weight_quantizer": {"num_bits": 8, "axis": 0}},
        "algorithm": {"method": "smoothquant", "alpha": 1.0},
    }


def test_none_is_noop():
    # Unset flag -> ModelOpt default is preserved and the same object is returned.
    cfg = _int8_sq_cfg()
    out = override_smoothquant_alpha(cfg, "int8_sq", None)
    assert out is cfg
    assert out["algorithm"]["alpha"] == 1.0


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.85, 1.0])
def test_override_applied_for_int8_sq(alpha):
    # int8_sq + a value -> algorithm rewritten to that alpha.
    out = override_smoothquant_alpha(_int8_sq_cfg(), "int8_sq", alpha)
    assert out["algorithm"] == {"method": "smoothquant", "alpha": alpha}


@pytest.mark.parametrize("qformat", ["fp8", "int4_awq", "w4a8_awq"])
def test_noop_for_non_int8_sq(qformat):
    # The flag only affects int8_sq; other formats are untouched.
    cfg = _int8_sq_cfg()
    out = override_smoothquant_alpha(cfg, qformat, 0.5)
    assert out is cfg
    assert out["algorithm"]["alpha"] == 1.0


def test_does_not_mutate_input():
    # The shared ModelOpt preset dict must not be mutated by the override (deep-copy).
    cfg = _int8_sq_cfg()
    snapshot = copy.deepcopy(cfg)
    override_smoothquant_alpha(cfg, "int8_sq", 0.5)
    assert cfg == snapshot


def test_out_of_range_alpha_not_validated_here():
    # Range validation is intentionally CLI-only (--smoothquant_alpha rejects values
    # outside [0, 1] via parser.error); the helper applies whatever it is given.
    # Locks in that design decision so validation is never duplicated in this layer.
    out = override_smoothquant_alpha(_int8_sq_cfg(), "int8_sq", 2.0)
    assert out["algorithm"] == {"method": "smoothquant", "alpha": 2.0}
