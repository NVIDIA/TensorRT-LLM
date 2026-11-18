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
"""Storage dtype of the DFlash draft KV layers (``speculative/dflash_kv_storage.py``), on the host.

The module is loaded from its file (it imports nothing of the package). The tests pin:

* the ``TRTLLM_DFLASH_FP8_DRAFT_KV`` reader: ``"1"`` after stripping whitespace, nothing else;
* the compute dtype of a pool: the pool's own, except FP8, whose K/V are computed in the
  activation dtype;
* the store cast: the same tensor for the pool's own dtype; for FP8 E4M3 a saturating cast
  (+-inf and every value past 448 clamped, NaN kept) that rounds every one of the 65,536 BF16
  bit patterns to nearest even, as a float64 reference over the E4M3 grid does.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_PATH = _ROOT / "tensorrt_llm/_torch/speculative/dflash_kv_storage.py"


def _load():
    name = "_dflash_kv_storage_under_test"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STORAGE = _load()


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), (" 1 ", True), ("1\n", True), ("0", False), ("", False), ("true", False)],
)
def test_knob_reader(value, expected):
    assert STORAGE.fp8_draft_kv_requested({STORAGE.FP8_DRAFT_KV_ENV: value}) is expected


def test_knob_reader_unset_and_environment(monkeypatch):
    assert STORAGE.fp8_draft_kv_requested({}) is False
    monkeypatch.delenv(STORAGE.FP8_DRAFT_KV_ENV, raising=False)
    assert STORAGE.fp8_draft_kv_requested() is False
    monkeypatch.setenv(STORAGE.FP8_DRAFT_KV_ENV, "1")
    assert STORAGE.fp8_draft_kv_requested() is True
    assert STORAGE.FP8_DRAFT_KV_ENV == "TRTLLM_DFLASH_FP8_DRAFT_KV"


@pytest.mark.parametrize(
    "cache_dtype,expected",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
def test_compute_dtype(cache_dtype, expected):
    assert STORAGE.kv_compute_dtype(cache_dtype, torch.bfloat16) == expected
    assert STORAGE.is_storage_only_dtype(cache_dtype) is (cache_dtype == torch.float8_e4m3fn)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_store_in_the_own_dtype_is_the_same_tensor(dtype):
    x = torch.zeros(3, 4).to(dtype)
    assert STORAGE.to_kv_storage(x, dtype) is x


def test_store_refuses_other_dtypes():
    with pytest.raises(ValueError, match="does not store"):
        STORAGE.to_kv_storage(torch.zeros(2, dtype=torch.bfloat16), torch.float16)


def test_store_saturates_and_keeps_nan():
    values = [449.0, 464.0, 480.0, 1.0e4, math.inf, -449.0, -1.0e4, -math.inf, 448.0, -448.0]
    x = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    stored = STORAGE.to_kv_storage(x, torch.float8_e4m3fn).float()
    assert stored.tolist() == [448.0] * 5 + [-448.0] * 3 + [448.0, -448.0]
    nan = STORAGE.to_kv_storage(torch.tensor([math.nan], dtype=torch.bfloat16), torch.float8_e4m3fn)
    assert bool(nan.float().isnan().all())


def _e4m3_grid() -> torch.Tensor:
    """Every finite non-negative E4M3 value in float64, in increasing code order (0..0x7E)."""
    codes = torch.arange(0x7F, dtype=torch.int16).to(torch.uint8)
    return codes.view(torch.float8_e4m3fn).to(torch.float64)


def _reference_e4m3_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even E4M3 codes of finite float64 ``x`` after clamping to +-448."""
    grid = _e4m3_grid()
    magnitude = x.abs().clamp(max=448.0)
    upper = torch.searchsorted(grid, magnitude).clamp(max=grid.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    exact = grid[upper] == magnitude
    below = magnitude - grid[lower]
    above = grid[upper] - magnitude
    # On a tie the even code (even mantissa) wins.
    pick_upper = (above < below) | ((above == below) & (upper % 2 == 0))
    code = torch.where(exact | pick_upper, upper, lower)
    sign = torch.where(torch.signbit(x), 0x80, 0)
    return (code | sign).to(torch.uint8)


def test_store_rounds_every_bf16_pattern_to_nearest_even():
    bits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    stored = STORAGE.to_kv_storage(x, torch.float8_e4m3fn).view(torch.uint8)
    wide = x.to(torch.float64)
    finite = torch.isfinite(wide)
    expected = _reference_e4m3_codes(torch.where(finite, wide, torch.zeros_like(wide)))
    infinite = torch.isinf(wide)
    expected = torch.where(infinite, torch.where(wide > 0, 0x7E, 0xFE).to(torch.uint8), expected)
    checked = ~torch.isnan(wide)
    assert torch.equal(stored[checked], expected[checked])
    assert bool(stored[~checked].view(torch.float8_e4m3fn).float().isnan().all())
    assert int(checked.sum()) == (1 << 16) - 254
