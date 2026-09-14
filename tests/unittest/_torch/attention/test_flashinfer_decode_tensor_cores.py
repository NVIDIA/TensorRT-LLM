# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``TRTLLM_FI_DECODE_TENSOR_CORES``.

``_use_tensor_cores`` decides whether a pure-decode plan gets a tensor-core
wrapper, and such a wrapper does not run FlashInfer's split-K decode kernel: it
builds a synthetic ``qo_indptr`` and dispatches the paged-prefill kernel. The
GQA-ratio arm fires for any model with at least 4 query heads per KV head,
which is most of them, so those models run the prefill kernel on every layer at
pure decode. That arm is FlashInfer's own perf heuristic rather than a
correctness requirement, and the knob exists to measure the alternative.

These tests pin what the knob may and may not touch, because the risk in a
switch like this is not that it fails to work -- it is that it silently changes
something else: the FP8 arm or the speculative-decode path, neither of which
has a split-K decode kernel to fall back to, or a model whose operator never
set the variable.

Pure dispatch logic, so no GPU is needed.
"""

import os
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import (
    FI_DECODE_TENSOR_CORES_ENV,
    FlashInferAttentionMetadata,
    PlanParams,
    decode_tensor_cores_override,
)
from tensorrt_llm.functional import AttentionMaskType


def plan(
    num_heads: int,
    num_kv_heads: int,
    kv_dtype=torch.bfloat16,
    q_len_per_req: int = 1,
) -> PlanParams:
    """A decode PlanParams; only the fields the decision reads vary."""
    return PlanParams(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        q_dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
        attention_mask_type=AttentionMaskType.causal,
        q_len_per_req=q_len_per_req,
    )


def use_tensor_cores(plan_params: PlanParams) -> bool:
    """``_use_tensor_cores`` reads only ``plan_params``, never ``self``."""
    return FlashInferAttentionMetadata._use_tensor_cores(None, plan_params)


@pytest.fixture(autouse=True)
def _unset_env():
    """Never inherit the operator's setting: it would invert every assertion."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
        yield


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,expected",
    [
        (128, 16, True),  # ratio 8: a high-GQA decode shape
        (32, 8, True),  # ratio 4: exactly at the threshold
        (24, 8, False),  # ratio 3: below it
        (16, 16, False),  # MHA
    ],
)
@pytest.mark.parametrize("unset_as", [None, "", "auto", "AUTO"])
def test_default_is_unchanged(num_heads, num_kv_heads, expected, unset_as):
    """Unset, empty and ``auto`` are all exactly the pre-existing heuristic."""
    if unset_as is not None:
        os.environ[FI_DECODE_TENSOR_CORES_ENV] = unset_as
    assert decode_tensor_cores_override() is None
    assert use_tensor_cores(plan(num_heads, num_kv_heads)) is expected


@pytest.mark.parametrize("off", ["0", "false", "off", "OFF", " 0 "])
def test_off_selects_the_decode_kernel(off):
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = off
    assert use_tensor_cores(plan(128, 16)) is False


@pytest.mark.parametrize("on", ["1", "true", "on", "ON", " 1 "])
def test_on_forces_tensor_cores_below_the_ratio(on):
    """The A/B is symmetric: it can force the kernel the heuristic would skip."""
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = on
    assert use_tensor_cores(plan(24, 8)) is True
    assert use_tensor_cores(plan(128, 16)) is True


@pytest.mark.parametrize("bad", ["yes", "2", "no", "disable", "None"])
def test_an_unrecognised_value_raises(bad):
    """Loud, not ignored. A typo that silently kept the heuristic would make
    both legs of the A/B measure the same thing and report a 0% difference."""
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = bad
    with pytest.raises(ValueError, match=FI_DECODE_TENSOR_CORES_ENV):
        decode_tensor_cores_override()


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("value", ["0", "1"])
def test_fp8_kv_cache_always_keeps_tensor_cores(kv_dtype, value):
    """The FP8 arm is not part of the experiment and must not move.

    There is no non-tensor-core decode kernel for an FP8 KV cache, so honouring
    ``0`` here would not select a different kernel -- it would select a path
    that does not exist.
    """
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = value
    assert use_tensor_cores(plan(128, 16, kv_dtype=kv_dtype)) is True
    # Also for a ratio that would not have asked for tensor cores anyway.
    assert use_tensor_cores(plan(16, 16, kv_dtype=kv_dtype)) is True


@pytest.mark.parametrize("q_len_per_req", [2, 4])
def test_multi_token_query_ignores_an_off_override(q_len_per_req):
    """Speculative decode has no split-K decode kernel either.

    This is the case the knob has to leave alone: with ``q_len_per_req > 1``
    the heuristic stands whatever the variable says.
    """
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = "0"
    assert use_tensor_cores(plan(128, 16, q_len_per_req=q_len_per_req)) is True
    # And a shape the heuristic already declined stays declined -- the
    # multi-token guard restores the heuristic, it does not force cores on.
    assert use_tensor_cores(plan(24, 8, q_len_per_req=q_len_per_req)) is False


def test_decision_is_read_per_call_not_cached_at_import():
    """The A/B sets the variable per server launch, so an import-time read
    would give both legs the same answer and quietly measure nothing."""
    os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
    assert use_tensor_cores(plan(128, 16)) is True
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = "0"
    assert use_tensor_cores(plan(128, 16)) is False
    os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
    assert use_tensor_cores(plan(128, 16)) is True
