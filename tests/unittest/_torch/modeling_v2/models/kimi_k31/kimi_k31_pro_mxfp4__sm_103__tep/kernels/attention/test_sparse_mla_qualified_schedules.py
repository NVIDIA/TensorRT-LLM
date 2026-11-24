# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The D576 sparse MLA allowlist is the schedule policy's image over the plans the family requests (CPU).

``mla_decode`` carries only the kernel code of ``QUALIFIED_SCHEDULES`` and rejects a plan whose policy schedule is not
one of them. Enumerating ``sparse_policy`` over the plans the tep target makes -- 96 query heads, Top-K 2048, one
query row per token, 148-160 SMs; ``native`` E4M3 rows at power-of-two capacities of 32 rows or more (the ShardQ
prefill rounds its row count up to one, ``_shardq_plan_rows``) -- every schedule the policy picks must be qualified,
and every qualified schedule must be one it picks: the 2CTA M128 kernel without a KV split.

Smaller power-of-two plans pick schedules the package does not carry (Q64 keeps-MMA-AB up to 8 rows, 2CTA with 4 KV
splits at 16), so such a plan is rejected.
"""

import functools

import pytest

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the sparse MLA package imports its CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.attention import (  # noqa: E402, E501
    mla_decode_plan as mla_decode,
)

pytestmark = pytest.mark.cpu_only

_SM_COUNTS = range(148, 161)
#: The smallest plan the ShardQ prefill makes (``_shardq_plan_rows``).
_MIN_PLAN_ROWS = 32
_POW2_ROWS = [1 << j for j in range(_MIN_PLAN_ROWS.bit_length() - 1, 21)]
_ROWS = {
    "native": _POW2_ROWS,
}


@functools.lru_cache(maxsize=1)
def _policy_schedules():
    """``({schedule key: (KV format, SM count, rows) of its first pick}, [plans the plan-time guard rejects])`` over
    the family's plans."""
    capacity = max(256, (mla_decode.SPARSE_MLA_TOP_K + 127) // 128 * 128)
    picked, rejected = {}, []
    for kv_format, rows_list in _ROWS.items():
        for sm_count in _SM_COUNTS:
            for rows in rows_list:
                profile = mla_decode.select_sparse_mla_profile(
                    rows=rows,
                    heads=mla_decode.SPARSE_MLA_HEADS,
                    query_length=1,
                    capacity=capacity,
                    dtype="fp8",
                    sm_count=sm_count,
                    rope_dim=64,
                )
                key = mla_decode._schedule_key(kv_format, profile)
                picked.setdefault(key, (kv_format, sm_count, rows))
                try:
                    mla_decode._check_schedule(kv_format, profile, rows, sm_count)
                except NotImplementedError as error:
                    rejected.append((kv_format, sm_count, rows, str(error)[:200]))
    return picked, rejected


def test_the_formats_are_the_packages() -> None:
    assert set(_ROWS) == set(mla_decode.KV_CACHE_FORMATS)


def test_every_policy_schedule_is_qualified() -> None:
    picked, rejected = _policy_schedules()
    assert not rejected, f"{len(rejected)} plans rejected, first: {rejected[:3]}"
    assert set(picked) <= mla_decode.QUALIFIED_SCHEDULES


def test_every_qualified_schedule_is_picked() -> None:
    never = mla_decode.QUALIFIED_SCHEDULES - set(_policy_schedules()[0])
    assert not never, f"allowlist entries the policy never picks: {sorted(never, key=str)}"


@pytest.mark.parametrize("rows", [1 << j for j in range(_MIN_PLAN_ROWS.bit_length() - 1)])
def test_plans_below_the_floor_are_rejected(rows) -> None:
    capacity = max(256, (mla_decode.SPARSE_MLA_TOP_K + 127) // 128 * 128)
    for sm_count in _SM_COUNTS:
        profile = mla_decode.select_sparse_mla_profile(
            rows=rows,
            heads=mla_decode.SPARSE_MLA_HEADS,
            query_length=1,
            capacity=capacity,
            dtype="fp8",
            sm_count=sm_count,
            rope_dim=64,
        )
        with pytest.raises(NotImplementedError, match="not one of the qualified schedules"):
            mla_decode._check_schedule("native", profile, rows, sm_count)
