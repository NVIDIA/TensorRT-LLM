# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cache dtypes the drafter window attention ops take, on the host only (CPU torch).

``ops/spec/dflash_window_attention.py`` and ``dflash_window_attention_wide.py`` are loaded
from their files with Triton and the op catalog stubbed (the kernels are not run). The tests
pin which operand / cache dtype pairs the ops' checks take: BF16 or FP16 operands over caches
of their own dtype, and BF16 operands over FP8 E4M3 caches (the FP8 draft KV pool), which the
kernels widen to BF16 on load; never FP8 operands, an FP8 cache under FP16 operands, or K and V
caches of different dtypes.
"""

from __future__ import annotations

import enum
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_SPEC = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/spec"
)
_PACKAGE = "_musespark_open_tep_window_attention_fp8_under_test"
_CATALOG = "tensorrt_llm._torch._experimental.modeling_v2.catalog._op"
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


def _triton_stub() -> tuple[types.ModuleType, types.ModuleType]:
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton = types.ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 << max(0, int(n) - 1).bit_length()
    triton.language = language
    return triton, language


def _load():
    triton, language = _triton_stub()
    catalog = types.ModuleType(_CATALOG)
    catalog.Arch = enum.Enum("Arch", "SM_107")
    catalog.OpWrapper = type("OpWrapper", (), {})
    stubs = {"triton": triton, "triton.language": language, _CATALOG: catalog}
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_SPEC)]
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    sys.modules[_PACKAGE] = package
    try:
        single = importlib.import_module(f"{_PACKAGE}.dflash_window_attention")
        wide = importlib.import_module(f"{_PACKAGE}.dflash_window_attention_wide")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return single, wide


SINGLE, WIDE = _load()
WINDOW = (2047, 2047)


@pytest.mark.parametrize(
    "dtype,kv_dtype,taken",
    [
        (BF16, None, True),
        (BF16, BF16, True),
        (BF16, FP8, True),
        (torch.float16, torch.float16, True),
        (torch.float16, FP8, False),
        (BF16, torch.float16, False),
        (FP8, FP8, False),
    ],
)
def test_geometry_dtype_pairs(dtype, kv_dtype, taken):
    reason = SINGLE.geometry_unsupported_reason(8, 4, 2, 128, dtype, WINDOW, kv_dtype)
    assert (reason is None) is taken, reason


def _call(q_dtype, k_dtype, v_dtype=None, batch=2, m=8):
    q = torch.zeros(batch, m, 4, 128).to(q_dtype)
    k_cache = torch.zeros(3, 64, 2, 128).to(k_dtype)
    v_cache = torch.zeros(3, 64, 2, 128).to(v_dtype or k_dtype)
    lengths = torch.full((batch,), 40, dtype=torch.int32)
    slots = torch.arange(batch, dtype=torch.int32)
    return q, k_cache, v_cache, lengths, slots, WINDOW


@pytest.mark.parametrize("check", ["single", "wide"])
@pytest.mark.parametrize("cache_dtype", [BF16, FP8])
def test_bf16_operands_over_bf16_or_fp8_caches_pass_the_dtype_checks(check, cache_dtype):
    reason_of = SINGLE.unsupported_reason if check == "single" else WIDE.wide_unsupported_reason
    # The host tensors stop the checks at the device test, past every dtype check.
    assert reason_of(*_call(BF16, cache_dtype)) == "all tensors must be on the GPU"


@pytest.mark.parametrize("check", ["single", "wide"])
@pytest.mark.parametrize(
    "q_dtype,k_dtype,v_dtype,match",
    [
        (FP8, FP8, FP8, "operands must share"),
        (torch.float16, FP8, FP8, "operands must share"),
        (BF16, torch.float16, torch.float16, "operands must share"),
        (BF16, FP8, BF16, "dtypes differ"),
        (BF16, BF16, FP8, "dtypes differ"),
    ],
)
def test_other_dtype_pairs_are_refused(check, q_dtype, k_dtype, v_dtype, match):
    reason_of = SINGLE.unsupported_reason if check == "single" else WIDE.wide_unsupported_reason
    reason = reason_of(*_call(q_dtype, k_dtype, v_dtype))
    assert reason is not None and match in reason, reason


def test_narrow_cache_dtypes():
    assert SINGLE.NARROW_CACHE_DTYPES == {BF16: (FP8,)}
    assert SINGLE.SUPPORTED_DTYPES == (BF16, torch.float16)
