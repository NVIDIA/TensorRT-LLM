# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The low-M Hyper-Connections entries of the targets' trunk (`ops/residual/hc_lowm.py`): the split-path predicate and
the Lamport workspace decision against their restated rules, and the un-finalized MoE combine against a float64
reference (fp32 sum of at most top_k + 1 terms, one bf16 rounding: one bf16 step of the row's largest element)."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.distributed.ops import AllReduceStrategy
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lowm import (  # noqa: E501
    hc_lamport_workspace,
    hc_split_enabled,
    moe_unfinalized,
    moe_unfinalized_finalize,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize("deterministic", ["0", "1"], ids=["default", "force-deterministic"])
@pytest.mark.parametrize(
    "cell", hc_split_enabled.CELLS, ids=[c.why[:44] for c in hc_split_enabled.CELLS]
)
def test_split_enabled_cells(cell, deterministic, certifying, monkeypatch):  # noqa: F811
    certifying(hc_split_enabled)
    monkeypatch.setenv("FORCE_DETERMINISTIC", deterministic)
    s = cell.spec
    mega = s.get("mega_mhc")
    got = hc_split_enabled(s["n"], s["hidden"], s["mult"], mega)
    assert got is hc_split_enabled.reference(s["n"], s["hidden"], s["mult"], mega)
    if deterministic == "1":
        assert got is False  # stock keeps every boundary on mhc_fused_hc's pinned tactic


@pytest.mark.parametrize(
    "cell", hc_lamport_workspace.CELLS, ids=[c.why[:44] for c in hc_lamport_workspace.CELLS]
)
def test_lamport_workspace_cells(cell, certifying):  # noqa: F811
    certifying(hc_lamport_workspace)
    s = cell.spec
    mapping = Mapping(world_size=s["tp"], tp_size=s["tp"], rank=0, enable_attention_dp=s["adp"])
    strategy = getattr(AllReduceStrategy, s["strategy"])
    got = hc_lamport_workspace(mapping, strategy)
    want = hc_lamport_workspace.reference(mapping, strategy)
    kind = None if got is None else ("mnnvl" if getattr(got, "mnnvl", False) else "ipc")
    assert kind == want


def _parts(spec, cell):
    gen = gen_for(cell)
    m, k, h = spec["m"], spec["k"], spec["h"]
    rows = m * k // 2 + 3
    fc2 = torch.randn(rows, h, generator=gen, device=DEV).bfloat16()
    perm = torch.randint(-1, rows, (m, k), generator=gen, device=DEV, dtype=torch.int32)
    if spec.get("none"):
        perm.fill_(-1)
    scales = torch.rand(m, k, generator=gen, device=DEV)
    shared = torch.randn(m, h, generator=gen, device=DEV).bfloat16()
    return moe_unfinalized(shared, fc2, scales, perm)


@pytest.mark.parametrize(
    "cell", moe_unfinalized_finalize.CELLS, ids=[c.why[:44] for c in moe_unfinalized_finalize.CELLS]
)
def test_finalize_cells(cell, certifying):  # noqa: F811
    certifying(moe_unfinalized_finalize)
    parts = _parts(cell.spec, cell)
    got = moe_unfinalized_finalize(parts)
    assert got.dtype == torch.bfloat16 and got.shape == parts.shared.shape
    assert_within_ulp(got, moe_unfinalized_finalize.reference(parts), element_ulp=2.0, rms_ulp=0.2)
    if cell.spec.get("none"):
        assert torch.equal(got, parts.shared)


def test_finalize_rejections():
    parts = _parts(dict(m=2, k=2, h=64), type("C", (), {"why": "rejections"})())
    moe_unfinalized_finalize.is_valid(parts)
    bad = moe_unfinalized(parts.shared, parts.fc2, parts.scales[:, :1], parts.perm)
    with pytest.raises(AssertionError):
        moe_unfinalized_finalize.is_valid(bad)
