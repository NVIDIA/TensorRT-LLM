# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM paged_mqa_logits_schedule entry: the schedule is integers, compared exactly.

Every cell runs at the device's SM count, the count the logits kernel launches its CTAs with.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa.paged_mqa_logits_schedule import (  # noqa: E501
    paged_mqa_logits_schedule as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "paged_mqa_logits_schedule requires a CUDA device"


def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _build(spec) -> dict:
    sms = _sms()
    return dict(
        context_lens=torch.tensor(spec["context_lens"], dtype=torch.int32, device="cuda"),
        block_kv=spec["block_kv"],
        num_sms=sms,
        out=torch.full((sms + 1, 2), -7, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec)
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert out is args["out"]
    assert torch.equal(out, ref), f"schedule differs:\n{out.tolist()}\nexpected\n{ref.tolist()}"


def test_the_partition_covers_every_split_once():
    """Independent of the reference: consecutive starts tile the requests' splits."""
    lens = [3_000, 70_000, 129, 40_000]
    args = _build(dict(context_lens=lens, block_kv=64))
    out = op(**args).tolist()
    splits = [-(-n // 256) for n in lens]
    flat = [(r, s) for r, count in enumerate(splits) for s in range(count)]
    position = {unit: i for i, unit in enumerate(flat)}
    starts = [position.get(tuple(row), len(flat)) for row in out]
    assert starts[0] == 0 and starts[-1] == len(flat)
    assert all(a <= b for a, b in zip(starts, starts[1:]))


def test_guard_refuses_another_sm_count():
    args = _build(op.CELLS[0].spec)
    sms = args["num_sms"] - 1
    with pytest.raises(AssertionError, match="SM count"):
        with validating(op):
            op(**dict(args, num_sms=sms, out=torch.empty(sms + 1, 2, dtype=torch.int32).cuda()))


def test_guard_refuses_two_dimensional_lengths():
    args = _build(op.CELLS[4].spec)
    with pytest.raises(AssertionError, match="num_requests"):
        with validating(op):
            op(**dict(args, context_lens=args["context_lens"].view(2, 2)))
