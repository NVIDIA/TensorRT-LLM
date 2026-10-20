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
"""Tests of the top-p nucleus search of the fused rejection-sampling kernels (`ftp.fast_topp_nucleus`)
against the float64 oracle `Nucleus` of ``topp_rs_test_utils.py``.

The draft and verify kernels that use the nucleus are tested in ``test_fast_topp_rejection_sampling.py``.
"""

import dataclasses

import pytest
import torch
from topp_rs_test_utils import (
    DEV,
    KINDS,
    Nucleus,
    V,
    diff_mass,
    kernel_kept,
    make_rows,
    scaled,
    support_diff_ok,
)

from tensorrt_llm._torch.speculative import fast_topp_rejection_sampling as ftp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


_TEMPS = (1.0, 0.6, 1.7)
_PS = (0.95, 0.5, 0.99, 0.999, 1e-3, 1.0, 0.0)
# Relative tolerance of the kernels' kept mass Z and total mass S against the oracle.
_MASS_TOL = 1e-5


def _check_nucleus(logits, temps, tps, cfg=None):
    """Check the kernels' nucleus record of each row against the oracle.

    No row may fall back (status 0), every kept-set must lie within `_NUC_MASS_TOL` of the oracle's
    target, and the kept mass (of rows with the oracle's kept-set) and the total mass must match within
    `_MASS_TOL` plus the rounding of ``z``. Returns: (number of rows whose kept-set differs from the
    oracle's, worst target error)."""
    kept, rec = kernel_kept(logits, temps, tps, cfg)
    status = rec[:, 3]
    assert bool((status == 0).all()), status.nonzero().flatten().tolist()
    nuc = Nucleus(logits, temps, tps)
    same = (kept == nuc.kept).all(dim=1)
    bad = [
        r
        for r in (~same).nonzero().flatten().tolist()
        if not support_diff_ok(nuc, r, kept[r], nuc.kept[r])
    ]
    assert not bad, [
        (r, float(temps[r]), float(tps[r]), int(kept[r].sum()), int(nuc.kept[r].sum()),
         float(nuc.gap[r]), diff_mass(nuc, r, kept[r], nuc.kept[r]))
        for r in bad
    ]  # fmt: skip
    zrel = ((rec[:, 1].double() - nuc.mass).abs() / nuc.mass)[same]
    srel = (rec[:, 2].double() - nuc.s).abs() / nuc.s
    # Weights are exp(z - max) with z rounded to float32: |z| * 2^-24 relative error each.
    tol = _MASS_TOL + 6e-8 * float(nuc.z.abs().max())
    assert float(zrel.max()) <= tol, float(zrel.max())
    assert float(srel.max()) <= tol, float(srel.max())
    return int((~same).sum())


# Runs the single-program kernel for every row, without the chunk-parallel pipeline.
SINGLE_PROGRAM = dataclasses.replace(ftp.DEFAULT_NUCLEUS, fast=False)


@pytest.mark.parametrize("fast", [True, False], ids=["pipeline", "single_program"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("kind", KINDS + ["zipf6", "zipf2.5", "zipf1.6", "zipf1.4", "zipf1.2"])
def test_nucleus_matches_oracle(kind, bf16, fast):
    """The nucleus (pivot, kept mass, total mass) matches the float64 oracle for every row kind, with
    per-row temperature and top-p (0 to 1, tiny and near one), both from the chunk-parallel pipeline
    and from the single-program kernel alone. Exact ties (constant, two-level and bf16 rows) form one
    group, and no row may fall back (status 0)."""
    gen = torch.Generator(device=DEV).manual_seed(1000 + 7 * len(kind) + 100 * int(bf16))
    combos = [(t, p) for t in _TEMPS for p in _PS] * 2
    logits = make_rows(kind, len(combos), V, gen, bf16)
    temps = torch.tensor([t for t, _ in combos], dtype=torch.float32, device=DEV)
    tps = torch.tensor([p for _, p in combos], dtype=torch.float32, device=DEV)
    differing = _check_nucleus(logits, temps, tps, None if fast else SINGLE_PROGRAM)
    print(
        f"nucleus {kind} bf16={bf16} fast={fast}: {logits.shape[0]} rows, {differing} within rounding"
    )


@pytest.mark.parametrize(
    "kind,min_share",
    [("zipf6", 1.0), ("zipf2.5", 1.0), ("zipf1.8", 1.0), ("peaked", 0.9), ("zipf1.6", 0.95)],
)
def test_pipeline_resolves_peaked_rows_without_the_fallback(kind, min_share):
    """Rows whose nucleus holds up to some hundreds of tokens are resolved by the chunk-parallel
    pipeline (no row reaches the single-program kernel), which is what the step latency relies on.
    (``peaked`` rows sit on a Gaussian background that carries about half of the mass: the nucleus of
    some of them holds tens of thousands of tokens.)"""
    gen = torch.Generator(device=DEV).manual_seed(21)
    rows = 96
    logits = make_rows(kind, rows, V, gen)
    temps = torch.ones(rows, device=DEV)
    tps = torch.full((rows,), 0.95, device=DEV)
    rec, path = ftp.fast_topp_nucleus(logits, temps, tps, return_path=True)
    assert bool((rec[:, 3] == 0).all())
    # Path codes 1, 7 and 8 (see `ftp.fast_topp_nucleus`) and 9 (the maximum alone reaches the target)
    # are the pipeline's.
    share = float(((path == 1) | (path == 7) | (path == 8) | (path == 9)).float().mean())
    print(
        f"{kind}: {share:.3f} of the rows resolved by the pipeline, codes {path.unique().tolist()}"
    )
    assert share >= min_share


def test_single_token_rows_are_resolved_from_the_stats():
    """A row whose maximum alone reaches the target needs no pass over it: the nucleus is the tokens
    tied with the maximum (code 9), here one, two and three of them."""
    gen = torch.Generator(device=DEV).manual_seed(31)
    rows = 12
    x = torch.randn(rows, V, generator=gen, device=DEV)
    ties = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]
    for r, n in enumerate(ties):
        idx = torch.randperm(V, generator=gen, device=DEV)[:n]
        x[r, idx] = 40.0  # the same float32 value: tied
    temps = torch.tensor([1.0, 0.6] * 6, device=DEV)
    tps = torch.tensor([0.001, 0.2] * 6, device=DEV)
    rec, path = ftp.fast_topp_nucleus(x, temps, tps, return_path=True)
    assert bool((rec[:, 3] == 0).all()) and bool((path == 9).all()), path.tolist()
    kept = scaled(x, temps) >= rec[:, 0:1]
    assert kept.sum(1).tolist() == ties
    assert rec[:, 1].tolist() == [float(n) for n in ties]  # the tied maxima weigh exp(0) = 1 each
    assert torch.equal(kept, Nucleus(x, temps, tps).kept)


@pytest.mark.parametrize(
    "cfg",
    [
        ftp.NucleusConfig(tile=4096, bins=64, cap=256, cap_small=32, iters=6, warps=8),
        ftp.NucleusConfig(tile=2048, bins=32, cap=128, cap_small=32, iters=6, warps=4),
        # Other chunkings of the stats and of the histogram / gather kernels, the middle sort and the
        # zone-only gather (``zone_small``).
        ftp.NucleusConfig(
            tile=4096,
            bins=64,
            cap=256,
            cap_small=32,
            iters=6,
            warps=8,
            hist_bins=128,
            chunk_cap=64,
            hg_chunk=4096,
            stats_chunk=8192,
            cap_mid=128,
            zone_small=32,
        ),
        ftp.NucleusConfig(
            tile=2048,
            bins=32,
            cap=128,
            cap_small=32,
            iters=6,
            warps=4,
            hist_bins=64,
            chunk_cap=32,
            hg_chunk=16384,
            stats_chunk=2048,
            cap_mid=64,
            zone_small=16,
        ),
        # The bin-narrowed resolve (``bin_min``) on short candidate lists, with few bins and slabs.
        ftp.NucleusConfig(
            tile=4096,
            bins=64,
            cap=256,
            cap_small=16,
            iters=6,
            warps=8,
            hist_bins=128,
            chunk_cap=64,
            hg_chunk=4096,
            stats_chunk=4096,
            cap_mid=32,
            bin_min=16,
            sub_bins=8,
            sub_slab=32,
        ),
        ftp.NucleusConfig(
            tile=2048,
            bins=32,
            cap=128,
            cap_small=8,
            iters=6,
            warps=4,
            hist_bins=64,
            chunk_cap=32,
            hg_chunk=8192,
            stats_chunk=2048,
            cap_mid=16,
            zone_small=24,
            bin_min=8,
            sub_bins=4,
            sub_slab=8,
        ),
    ],
)
def test_nucleus_refinement_matches_oracle(cfg):
    """Other launch configurations (`ftp.NucleusConfig`) give the oracle's nucleus too. Few bins and a
    small zone cap force several refinement iterations (and the zone-overflow branch); the answer must
    not change."""
    gen = torch.Generator(device=DEV).manual_seed(77)
    for kind in ("peaked", "flat", "very_flat", "heavy", "two_level", "uniform01"):
        logits = make_rows(kind, 12, V, gen)
        temps = torch.tensor([1.0, 0.6, 1.7] * 4, dtype=torch.float32, device=DEV)
        tps = torch.tensor([0.95] * 6 + [0.5, 0.99, 0.999, 0.9, 0.7, 0.2], device=DEV)
        _check_nucleus(logits, temps, tps, cfg)


def test_nucleus_tie_group_across_tiles():
    """A pivot group of tens of thousands of equal values spanning many tiles (and the whole
    vocabulary for a constant row): every member is kept."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    x = torch.full((4, V), 1.0, device=DEV)
    x[0, : V // 3] = 2.0  # top group = a third of the vocabulary
    x[1, : V // 100] = 6.0  # a small top group; the pivot group is the big bottom one
    x[2, ::7] = 3.0
    x[3] = torch.randn(V, generator=gen, device=DEV).to(torch.bfloat16).float()
    temps = torch.ones(4, device=DEV)
    tps = torch.tensor([0.5, 0.95, 0.95, 0.95], device=DEV)
    kept, rec = kernel_kept(x, temps, tps)
    assert bool((rec[:, 3] == 0).all())
    nuc = Nucleus(x, temps, tps)
    assert bool((kept == nuc.kept).all())
    assert int(kept[0].sum()) == int((x[0] == 2.0).sum())  # p = 0.5 is reached inside the top group
    assert int(kept[1].sum()) == V  # p = 0.95 needs the tied bottom group: all of it is kept
    assert int(kept[2].sum()) == V


def test_clustered_candidates_need_the_full_size_sort():
    """One token at the maximum and thousands within a 1e-3 range of ``z`` below it: the z-range
    split of the candidate list puts almost all of them into one bin, so the bin's sort is the
    full-size one. The nucleus must still be exact (the pivot lies inside the cluster)."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    sizes = [1500, 2500, 3500, 3900, 2000, 1200]
    x = torch.full((len(sizes), V), -60.0, device=DEV)
    x[:, 0] = 0.0
    for r, n in enumerate(sizes):
        idx = torch.randperm(V - 1, generator=gen, device=DEV)[:n] + 1
        x[r, idx] = -3.0 + torch.rand(n, generator=gen, device=DEV) * 1e-3
    temps = torch.ones(len(sizes), device=DEV)
    tps = torch.full((len(sizes),), 0.95, device=DEV)
    _check_nucleus(x, temps, tps)
    rec, path = ftp.fast_topp_nucleus(x, temps, tps, return_path=True)
    assert bool((rec[:, 3] == 0).all())
    assert bool(((path == 1) | (path == 8)).all()), path.tolist()  # resolved without the fallback


@pytest.mark.parametrize(
    "kind", ["zipf2.5", "zipf1.6", "zipf1.5", "zipf1.4", "very_flat", "medium"]
)
def test_bin_narrowed_resolve_agrees_with_the_full_sort(kind):
    """Narrowing a long candidate list to one z-bin before sorting finds the pivot the full sort finds:
    the same ``z_pivot`` (the pivot is a stored logit; only a cumulative mass within float rounding of the
    target could move it) and the same kept mass up to the rounding of the sums."""
    gen = torch.Generator(device=DEV).manual_seed(41)
    rows = 96
    logits = make_rows(kind, rows, V, gen)
    temps = torch.tensor([1.0, 0.6, 1.7] * (rows // 3), dtype=torch.float32, device=DEV)
    tps = torch.tensor([0.95, 0.5, 0.99, 0.999] * (rows // 4), dtype=torch.float32, device=DEV)
    full_sort = dataclasses.replace(ftp.DEFAULT_NUCLEUS, bin_min=ftp.DEFAULT_NUCLEUS.cap)
    narrowed = dataclasses.replace(ftp.DEFAULT_NUCLEUS, bin_min=ftp.DEFAULT_NUCLEUS.small)
    rec_a, path_a = ftp.fast_topp_nucleus(logits, temps, tps, full_sort, return_path=True)
    rec_b, path_b = ftp.fast_topp_nucleus(logits, temps, tps, narrowed, return_path=True)
    assert bool((rec_a[:, 3] == 0).all()) and bool((rec_b[:, 3] == 0).all())
    same_pivot = rec_a[:, 0] == rec_b[:, 0]
    # A different pivot is allowed only where the two nuclei differ by float rounding of the mass.
    for r in (~same_pivot).nonzero().flatten().tolist():
        ka = scaled(logits[r : r + 1], temps[r : r + 1]) >= rec_a[r, 0]
        kb = scaled(logits[r : r + 1], temps[r : r + 1]) >= rec_b[r, 0]
        nuc = Nucleus(logits[r : r + 1], temps[r : r + 1], tps[r : r + 1])
        assert support_diff_ok(nuc, 0, ka[0], kb[0]), (kind, r, float(tps[r]), float(temps[r]))
    rel = ((rec_a[:, 1] - rec_b[:, 1]).abs() / rec_a[:, 1])[same_pivot]
    assert float(rel.max()) < 5e-6, float(rel.max())


@pytest.mark.parametrize("kind", ["peaked", "zipf1.6", "very_flat", "medium"])
def test_records_are_bitwise_reproducible(kind):
    """A row's nucleus record is a pure function of its logits (integer atomics only, fixed-order float
    sums): repeated calls, a different batch order and the row alone in a smaller batch give
    bit-identical ``(z_pivot, Z, S, status)``. Tensor-parallel ranks hold the same logits and rely on
    this to find the same nucleus (and so draw the same tokens)."""
    gen = torch.Generator(device=DEV).manual_seed(61)
    rows = 48
    logits = make_rows(kind, rows, V, gen)
    temps = torch.tensor([1.0, 0.6, 1.7] * (rows // 3), dtype=torch.float32, device=DEV)
    tps = torch.tensor([0.95, 0.5, 0.99, 0.999] * (rows // 4), dtype=torch.float32, device=DEV)
    base = ftp.fast_topp_nucleus(logits, temps, tps)
    for _ in range(3):
        assert torch.equal(base, ftp.fast_topp_nucleus(logits, temps, tps))
    reverse = ftp.fast_topp_nucleus(logits.flip(0).contiguous(), temps.flip(0), tps.flip(0))
    assert torch.equal(base, reverse.flip(0))
    for r in (0, 7, 23, 47):
        alone = ftp.fast_topp_nucleus(
            logits[r : r + 1].contiguous(), temps[r : r + 1], tps[r : r + 1]
        )
        assert torch.equal(base[r : r + 1], alone), r
