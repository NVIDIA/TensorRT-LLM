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
"""specDSA host side without a GPU: the large-batch CTA schedule and the knob tables of the two specs.

The CTA schedule (``large_batch_op._schedule``) cuts the rows' concatenated stream units over the 148 CTAs. The FP8 spec
must keep the even split (and the aligned split below 80 rows) exactly, since the schedule is part of what the FP8
kernels were validated with. The FP4 spec moves the cuts with ``_balance_cuts``: a CTA that streams pieces of several
rows pays a fresh start of the selection for every row it touches, so stream units plus ``_SEG_UNITS`` per row segment
are even.

The knob tables (``megakernel_op._KNOBS``, ``large_batch_op._KNOBS`` / ``_VARIANTS`` / ``_TAIL_KNOBS``) are what the
kernel objects are built from: the FP8 entries stay the ones of the first specDSA version (identical generated code),
the FP4 entries add the knobs of the FP4 kernels. Building a kernel object needs CuTe DSL (not a GPU) and runs every
constructor check of its knob set.

The comparisons against ``SPEC_DSA_PR_HEAD_ROOT`` (a checkout of the specDSA version before the FP4 spec was tuned) skip
without it; the independent reference of the large-batch test and the structural checks always run.
"""

import importlib
import importlib.util
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import large_batch_op as op
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import megakernel_op
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import DS_V4_FLASH as FLASH
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import (
    DS_V4_FLASH_FP4 as FLASH_FP4,
)
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import NUM_SMS

from . import test_cute_dsl_spec_dsa_large_batch as H

UNIT, MIN_UNITS = op.SCHED_UNIT, op.SCHED_MIN_UNITS
N_MIN, N_MAX = FLASH.min_index_len, FLASH.max_index_len
# The shortest row of MIN_UNITS stream units: from here a row costs its length and can be cut; a shorter row costs
# MIN_UNITS and is never split.
LONG_MIN = (MIN_UNITS - 1) * UNIT + 1
EDGES = [512, 513, 32512, 32513, 32767, 32768, 32769, 65535, 65536, 131071, 262143, 262144]
BATCHES = [1, 2, 3, 7, 16, 32, 47, 48, 64, 79, 80, 100, 127, 128, 148, 149, 256, 777, 1024]
# Mixtures of row lengths (every mixture exists for every batch size); the "long_*" ones have no row below LONG_MIN.
MIXES = [
    "max",
    "min",
    "edges",
    "short",
    "uniform",
    "log",
    "one_long",
    "one_short",
    "stair",
    "long_uniform",
    "long_edges",
    "long_log",
    "long_stair",
    "one_long_rest_long",
]
LONG_MIXES = ["max", "long_uniform", "long_edges", "long_log", "long_stair", "one_long_rest_long"]


def _mix(kind, batch):
    rng = np.random.RandomState(zlib.crc32(f"{kind}-{batch}".encode()))
    if kind == "max":
        ns = [N_MAX] * batch
    elif kind == "min":
        ns = [N_MIN] * batch
    elif kind == "edges":
        ns = [EDGES[i % len(EDGES)] for i in range(batch)]
    elif kind == "short":
        ns = rng.randint(N_MIN, N_MIN + 3000, batch)
    elif kind == "uniform":
        ns = rng.randint(N_MIN, N_MAX + 1, batch)
    elif kind == "log":
        ns = np.exp(rng.uniform(np.log(N_MIN), np.log(N_MAX), batch))
    elif kind == "one_long":
        ns = [N_MAX] + [N_MIN] * (batch - 1)
    elif kind == "one_short":
        ns = [N_MIN] + [N_MAX] * (batch - 1)
    elif kind == "stair":
        ns = np.linspace(N_MIN, N_MAX, batch)
    elif kind == "long_uniform":
        ns = rng.randint(LONG_MIN, N_MAX + 1, batch)
    elif kind == "long_edges":
        long_edges = [n for n in EDGES if n >= LONG_MIN]
        ns = [long_edges[i % len(long_edges)] for i in range(batch)]
    elif kind == "long_log":
        ns = np.exp(rng.uniform(np.log(LONG_MIN), np.log(N_MAX), batch))
    elif kind == "long_stair":
        ns = np.linspace(LONG_MIN, N_MAX, batch)
    elif kind == "one_long_rest_long":
        ns = [N_MAX] + [LONG_MIN] * (batch - 1)
    else:
        raise ValueError(kind)
    return [int(min(max(n, N_MIN), N_MAX)) for n in ns]


def _units(ns):
    return -(-np.asarray(ns, dtype=np.int64) // UNIT)


def _positions(ns, sched, cost_space=False):
    """Position of every schedule entry in the concatenated stream units of the rows.

    The real units of a row are ``ceil(n / UNIT)``; with ``cost_space`` a row shorter than MIN_UNITS units counts
    MIN_UNITS (the cost the schedule balances).
    """
    units = _units(ns)
    work = np.concatenate([[0], np.cumsum(np.maximum(units, MIN_UNITS) if cost_space else units)])
    return work[sched[:, 0].astype(np.int64)] + sched[:, 1]


def _check_schedule(sched, ns, num_ctas=NUM_SMS):
    """A valid schedule: what the kernels accept (their schedule verdict) and a partition of the stream units.

    Entry s = (row, unit offset in the row) where CTA s starts, entry ``num_ctas`` = (batch, 0). The verdict of the
    kernels refuses an entry outside the batch, an offset beyond the row's units, entries that go backwards, and an end
    entry other than (batch, 0).
    """
    batch = len(ns)
    units = _units(ns)
    assert sched.shape == (num_ctas + 1, 2) and sched.dtype == np.int32
    row, off = sched[:, 0].astype(np.int64), sched[:, 1].astype(np.int64)
    assert (row[0], off[0]) == (0, 0) and (row[-1], off[-1]) == (batch, 0)
    assert ((row >= 0) & (row <= batch)).all() and (off >= 0).all()
    inside = row < batch
    assert (off[~inside] == 0).all()
    u_of = units[np.minimum(row, batch - 1)]
    assert (off[inside] < np.maximum(u_of[inside], 1)).all(), "offset beyond the row's units"
    ahead = (row[1:] > row[:-1]) | ((row[1:] == row[:-1]) & (off[1:] >= off[:-1]))
    assert ahead.all(), "entries go backwards"
    assert (off[inside & (u_of < MIN_UNITS)] == 0).all(), (
        "a row of fewer than MIN_UNITS units is split"
    )
    # every stream unit is read by exactly one CTA
    pos = _positions(ns, sched)
    assert pos[0] == 0 and pos[-1] == units.sum() and (np.diff(pos) >= 0).all()
    cover = np.zeros(int(units.sum()), dtype=np.int32)
    for s in range(num_ctas):
        cover[pos[s] : pos[s + 1]] += 1
    assert (cover == 1).all(), "a stream unit is read by none or by two CTAs"


def _cta_cost(ns, sched, seg_units):
    """Per CTA, the modeled cost the FP4 cuts balance: its units (a short row counts MIN_UNITS) plus ``seg_units`` per
    row segment it touches; 0 for a CTA without work."""
    pos = _positions(ns, sched, cost_space=True)
    units = pos[1:] - pos[:-1]
    segs = (sched[1:, 0].astype(np.int64) - sched[:-1, 0]) + (sched[1:, 1] > 0)
    return units + seg_units * np.where(units > 0, segs, 0)


# --------------------------------------------------------------------------------------------------------------------
# the reference copy of the specDSA version before the FP4 tuning (skips without it)


def _load_pr_head(name):
    root = os.environ.get("SPEC_DSA_PR_HEAD_ROOT")
    if not root:
        pytest.skip(
            "SPEC_DSA_PR_HEAD_ROOT (a checkout of the specDSA version before the FP4 tuning) is not set"
        )
    pkg_dir = Path(root) / "tensorrt_llm/_torch/attention/kernels/blackwell/spec_dsa"
    if not (pkg_dir / "__init__.py").exists():
        pytest.skip(f"no specDSA package under {root}")
    alias = "_spec_dsa_pr_head"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[alias] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{alias}.{name}")


# --------------------------------------------------------------------------------------------------------------------
# the CTA schedule


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("spec", [FLASH, FLASH_FP4], ids=["fp8", "fp4"])
def test_schedule_is_a_valid_partition(spec, batch):
    """Every batch size, every mixture: monotone entries the kernels accept, every stream unit read once."""
    seg = op._seg_units(spec, batch)
    for kind in MIXES:
        ns = _mix(kind, batch)
        _check_schedule(op._schedule(ns, NUM_SMS, seg), ns)


@pytest.mark.parametrize("num_ctas", [1, 2, 3, 17, 64, NUM_SMS])
@pytest.mark.parametrize("seg", [0, 14, 38])
def test_schedule_for_other_cta_counts(num_ctas, seg):
    for batch in (1, 2, 5, 40, 80, 200):
        for kind in ("uniform", "edges", "one_long", "long_log", "stair"):
            ns = _mix(kind, batch)
            _check_schedule(op._schedule(ns, num_ctas, seg), ns, num_ctas)


@pytest.mark.parametrize("seg", [0, 1, 14, 28, 41, 500])
@pytest.mark.parametrize("num_ctas", [1, 2, 7, NUM_SMS])
def test_balance_cuts_are_monotone_and_cover_the_total(num_ctas, seg):
    for batch in (1, 2, 9, 100, 1024):
        for kind in ("uniform", "edges", "short", "one_long", "long_log"):
            ns = _mix(kind, batch)
            prefix = np.concatenate([[0], np.cumsum(np.maximum(_units(ns), MIN_UNITS))])
            cuts = op._balance_cuts(prefix, num_ctas, seg)
            assert cuts.shape == (num_ctas + 1,) and cuts.dtype == np.int64
            assert cuts[0] == 0 and cuts[-1] == prefix[-1]
            assert (np.diff(cuts) >= 0).all() and (cuts >= 0).all() and (cuts <= prefix[-1]).all()
            if (
                seg == 0
            ):  # without a row segment cost: every CTA gets floor or ceil of the even share
                quota = int(prefix[-1]) // num_ctas
                assert set(np.diff(cuts).tolist()) <= {quota, quota + 1}


def test_balance_cuts_small_cases():
    """Hand-computed: a cut inside the cost jump of a row start moves to that start; otherwise it shifts by the cost."""
    # two rows of 100 units, 2 CTAs, a row segment costs 50: the balanced split is one row per CTA (the even split too)
    assert op._balance_cuts(np.array([0, 100, 200]), 2, 50).tolist() == [0, 100, 200]
    # three rows of 100, 2 CTAs, a segment costs 10: CTA 0 = row 0 + half of row 1 (2 rows), CTA 1 = the rest (2 rows)
    cuts = op._balance_cuts(np.array([0, 100, 200, 300]), 2, 10)
    assert cuts.tolist() == [0, 150, 300]
    # one row: no row start inside, the even split
    assert op._balance_cuts(np.array([0, 1000]), 4, 40).tolist() == [0, 250, 500, 750, 1000]


@pytest.mark.parametrize("batch", BATCHES)
def test_fp8_schedule_matches_the_independent_reference(batch):
    """The FP8 spec keeps the even split (aligned below 80 rows) exactly, with and without the spec's cost argument."""
    assert op._seg_units(FLASH, batch) == 0
    for kind in MIXES:
        ns = _mix(kind, batch)
        want = [list(r) for r in H._schedule_reference(ns, NUM_SMS)]
        assert op._schedule(ns, NUM_SMS).tolist() == want, kind
        assert op._schedule(ns, NUM_SMS, op._seg_units(FLASH, batch)).tolist() == want, kind


def test_fp8_schedule_is_the_pr_head_schedule():
    """The same arrays (values and dtype) as the schedule of the version before the FP4 tuning, on a grid of batch
    sizes 1 to 160 and larger ones, and the aligned helper on its own."""
    pr = _load_pr_head("large_batch_op")
    n_cases = 0
    for batch in list(range(1, 161)) + [200, 256, 333, 512, 777, 1024]:
        for kind in MIXES:
            ns = _mix(kind, batch)
            want = pr._schedule(ns, NUM_SMS)
            for got in (
                op._schedule(ns, NUM_SMS),
                op._schedule(ns, NUM_SMS, op._seg_units(FLASH, batch)),
            ):
                assert got.dtype == want.dtype and np.array_equal(got, want), (batch, kind)
            n_cases += 1
            if batch < op.SCHED_ALIGN_BATCH and batch <= NUM_SMS:
                units = _units(ns)
                assert np.array_equal(
                    op._aligned_schedule(units, NUM_SMS), pr._aligned_schedule(units, NUM_SMS)
                )
                assert op._busiest_cta_units(units, want) == pr._busiest_cta_units(units, want)
    assert n_cases > 2000


@pytest.mark.parametrize("batch", BATCHES)
def test_fp4_schedule_balances_units_plus_segment_cost(batch):
    """The busiest CTA (units + the segment cost of every row it touches) is at most one segment cost plus rounding
    above the even share of the total (below 80 rows: and the 3 % slack of the aligned split); a row too short to be cut
    (it costs 128 units) can add up to 127 units."""
    seg = op._seg_units(FLASH_FP4, batch)
    assert seg > 0
    slack = 1.0 + (op.SCHED_ALIGN_SLACK_PCT / 100 if batch < op.SCHED_ALIGN_BATCH else 0.0)
    for kind in MIXES:
        ns = _mix(kind, batch)
        sched = op._schedule(ns, NUM_SMS, seg)
        cost = np.maximum(_units(ns), MIN_UNITS)
        share = (
            cost.sum() + seg * (batch - 1)
        ) / NUM_SMS  # the even share of units + segment costs
        cta = _cta_cost(ns, sched, seg)
        extra = 0 if kind in LONG_MIXES else MIN_UNITS - 1
        assert cta.max() <= (share + seg + 1.5) * slack + extra, (kind, int(cta.max()), share)


@pytest.mark.parametrize("batch", [b for b in BATCHES if b >= op.SCHED_ALIGN_BATCH])
def test_fp4_cuts_lower_the_busiest_cta_of_mixed_long_rows(batch):
    """From 80 rows of at least 128 units: the busiest CTA of the segment-aware cuts costs at most one segment more than
    the even split's (the even split has at least the even share), at least one segment less when the lengths differ
    (the even split cuts at arbitrary units, and a CTA that touches many rows pays for each), and the CTAs with work are
    within two segment costs of each other."""
    seg = op._seg_units(FLASH_FP4, batch)
    for kind in LONG_MIXES:
        ns = _mix(kind, batch)
        balanced = _cta_cost(ns, op._schedule(ns, NUM_SMS, seg), seg)
        even = _cta_cost(ns, op._schedule(ns, NUM_SMS, 0), seg)
        assert balanced.max() <= even.max() + seg + 1, (kind, int(balanced.max()), int(even.max()))
        if kind in ("long_uniform", "long_edges", "long_log", "long_stair"):
            assert balanced.max() + seg <= even.max(), (kind, int(balanced.max()), int(even.max()))
        if batch <= NUM_SMS:
            assert balanced.max() - balanced.min() <= 2 * seg + 3, (
                kind,
                balanced.min(),
                balanced.max(),
            )


def test_segment_cost_table():
    """No cost for the FP8 spec; for FP4 the points (batch size, units) joined linearly and clamped at both ends."""
    assert op._SEG_UNITS.keys() <= {FLASH_FP4.kernel_key}
    assert all(op._seg_units(FLASH, b) == 0 for b in (0, 1, 47, 48, 148, 1024, 5000))
    points = op._SEG_UNITS[FLASH_FP4.kernel_key]
    assert [b for b, _ in points] == sorted({b for b, _ in points})
    for b, units in points:
        assert op._seg_units(FLASH_FP4, b) == units
    assert (
        op._seg_units(FLASH_FP4, 0) == points[0][1]
        and op._seg_units(FLASH_FP4, 4000) == points[-1][1]
    )
    for (b0, u0), (b1, u1) in zip(points[:-1], points[1:]):
        for b in range(b0, b1 + 1):
            want = u0 + (u1 - u0) * (b - b0) / (b1 - b0)
            assert abs(op._seg_units(FLASH_FP4, b) - want) <= 0.5, b
        assert min(u0, u1) <= op._seg_units(FLASH_FP4, (b0 + b1) // 2) <= max(u0, u1)
    assert [op._seg_units(FLASH_FP4, b) for b in (64, 82, 100, 114, 128, 192, 256)] == [
        14,
        19,
        24,
        31,
        38,
        33,
        28,
    ]


# --------------------------------------------------------------------------------------------------------------------
# the knob tables


def _kernel(spec, batch):
    """The megakernel object of a batch size (CuTe DSL import, no GPU: building it runs the constructor checks)."""
    return megakernel_op._engine(torch.device("cuda", 0), batch, spec)["kernel"]


def _desc_params_pr_head(kernel, batch_size, row):
    """``_desc_params`` of the version before the FP4 tuning."""
    shared = batch_size > int(kernel.nteams)
    zlast = shared and (
        (kernel.tsz and row == batch_size - 1) or (kernel.tsz2 and row >= batch_size - 2)
    )
    zc = -1
    if shared and kernel.tsz01 and (row < 2 or (kernel.tszall and row < batch_size - 2)):
        zc = (kernel.ns - 2) if row < 2 else (kernel.ns - 2 - row // 2)
    return (row * int(kernel.drot)) % kernel.ns, bool(zlast), int(zc)


def test_megakernel_knob_tables():
    for spec in (FLASH, FLASH_FP4):
        table = megakernel_op._KNOBS[spec.kernel_key]
        assert (
            sorted(table)
            == list(megakernel_op.supported_batch_sizes(spec))
            == list(range(1, spec.megakernel_max_batch + 1))
        )
    fp8, fp4 = megakernel_op._KNOBS[FLASH.kernel_key], megakernel_op._KNOBS[FLASH_FP4.kernel_key]
    assert sorted(fp8) == list(range(1, 48)) and sorted(fp4) == list(range(1, 21))
    # the FP8 sets have none of the FP4 knobs (the generated code of the FP8 kernels is the first version's)
    assert all("tcta" not in k and "drot" not in k for k in fp8.values())
    # one to four rows: the same sets for both specs; from five rows the FP4 sets add the shared-team knobs
    for b in range(1, 5):
        assert fp4[b] == fp8[b], b
    for b in range(5, 21):
        k = fp4[b]
        assert (k["team4"], k["nteams"], k["tcta"], k["b1l_upub"], k["rs24_dry"]) == (
            True,
            4,
            2,
            3,
            False,
        ), b
        assert k["drot"] > 0 and k.get("ring", False) == (b >= 9), b
        # everything else is the FP8 set of the batch size
        assert {x: v for x, v in k.items() if x not in megakernel_op._FP4_SHARED} == {
            x: v for x, v in fp8[b].items() if x not in megakernel_op._FP4_SHARED
        }, b


def _fp8_teams(b):
    """Teams of the FP8 knob sets: one per row up to 4 rows, two shared teams above."""
    return b if b <= 4 else 2


def test_megakernel_kernel_objects_build_for_every_batch_size():
    """Each knob set passes the constructor checks of its kernel; the FP4 sets use the FP4 front end."""
    pytest.importorskip("cutlass.cute")
    for spec in (FLASH, FLASH_FP4):
        for b in megakernel_op.supported_batch_sizes(spec):
            k = _kernel(spec, b)
            assert k.nrows == b and k.fp4 == (spec is FLASH_FP4)
            assert k.ns % 2 == 0 and k.ns <= 148
            assert k.tcta == (2 if spec is FLASH_FP4 and b >= 5 else 0)
            assert k.nteams == (4 if spec is FLASH_FP4 and b >= 5 else _fp8_teams(b))
            # CTAs: stream + teams + T servers = the grid
            assert k.ns + k.team_ctas * k.nteams + k.tcta == 148


def test_desc_params_of_the_fp8_spec_are_unchanged():
    """The rotation / early-threshold parameters of every row of every FP8 batch size are the old ones; the FP4
    spec's rows rotate over 8 distinct offsets and never use the early-threshold variants (dedicated T servers)."""
    pytest.importorskip("cutlass.cute")
    for b in megakernel_op.supported_batch_sizes(FLASH):
        k = _kernel(FLASH, b)
        for r in range(b):
            assert megakernel_op._desc_params(k, b, r) == _desc_params_pr_head(k, b, r), (b, r)
    for b in range(5, 21):
        k = _kernel(FLASH_FP4, b)
        params = [megakernel_op._desc_params(k, b, r) for r in range(b)]
        assert len({p[0] for p in params}) == min(b, 8), b
        assert all(p[0] == (r % 8) * int(k.drot) % k.ns for r, p in enumerate(params))
        assert all((p[1], p[2]) == (False, -1) for p in params), b
    for b in range(1, 5):
        k = _kernel(FLASH_FP4, b)
        assert [megakernel_op._desc_params(k, b, r) for r in range(b)] == [
            _desc_params_pr_head(k, b, r) for r in range(b)
        ]


def test_fp8_knob_tables_are_the_pr_head_tables():
    pr_mo = _load_pr_head("megakernel_op")
    assert megakernel_op._KNOBS[FLASH.kernel_key] == pr_mo._KNOBS[FLASH.kernel_key]
    assert megakernel_op._BASE == pr_mo._BASE and megakernel_op._CONFIGS == pr_mo._CONFIGS
    pr_lb = _load_pr_head("large_batch_op")
    assert op._KNOBS[FLASH.kernel_key] == pr_lb._KNOBS[FLASH.kernel_key]
    assert op._KERNEL_CONFIG == pr_lb._KERNEL_CONFIG


def test_large_batch_knob_tables():
    """FP8: the one knob set, no variants, no tail knobs. FP4: its knobs, a variant from 60 rows, exit_early."""
    assert op._KNOBS[FLASH.kernel_key] == op._KERNEL_CONFIG
    assert FLASH.kernel_key not in op._VARIANTS and FLASH.kernel_key not in op._TAIL_KNOBS
    assert FLASH.kernel_key not in op._SEG_UNITS
    fp4 = op._KNOBS[FLASH_FP4.kernel_key]
    assert fp4["kv_stages"] == 8 and fp4["merge1"] == 2 and fp4["trig0"] == 1536
    assert all(fp4[k] for k in ("fast_verdict", "sel_fast", "cmp_fast", "sel_hand2"))
    # the knobs the FP8 set does not have are exactly the FP4 additions
    added = {k for k in fp4 if k not in op._KERNEL_CONFIG}
    assert added == {"merge1", "fast_verdict", "sel_fast", "cmp_fast", "sel_hand2"}
    variants = op._VARIANTS[FLASH_FP4.kernel_key]
    firsts = [first for first, _ in variants]
    assert firsts == sorted(set(firsts)) and firsts[0] > FLASH_FP4.megakernel_max_batch
    assert all(first <= FLASH_FP4.max_batch for first in firsts)
    for _, over in variants:
        assert set(over) <= set(fp4), "a variant overrides a knob of the kernel"
        assert all(over[k] != fp4[k] for k in over), (
            "a variant that changes nothing is a second compile for nothing"
        )
    assert op._TAIL_KNOBS[FLASH_FP4.kernel_key] == {"exit_early": True}


def test_large_batch_kernel_objects_build():
    """The indexer kernel and the attention tail of both specs, and every variant, pass their constructor checks."""
    pytest.importorskip("cutlass.cute")
    lb, _ = op._kernel_modules()
    for spec in (FLASH, FLASH_FP4):
        knobs = op._KNOBS[spec.kernel_key]
        lb.IndexerKernel(num_sms=NUM_SMS, config=spec, **knobs)
        for _, over in op._VARIANTS.get(spec.kernel_key, ()):
            lb.IndexerKernel(num_sms=NUM_SMS, config=spec, **{**knobs, **over})
        lb.AttentionTail(config=spec, **op._TAIL_KNOBS.get(spec.kernel_key, {}))
