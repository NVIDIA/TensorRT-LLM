# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-cluster K-split ("stream-K") tactics of the Kimi K3 MXFP8 Rubin GEMM.

``TRTLLM_K3_DENSE_STREAMK=1`` adds ``("streamk", ..., cluster_k)`` tactics to
``CuteDSLKimiK3MXFP8RubinLinear``: ``cluster_k`` CTAs of one cluster (its z dimension) compute
one output tile from disjoint K slices and reduce their FP32 partials through distributed
shared memory, in the order of the K rank. Only the FP32 summation order differs from the
unsplit GEMM: the result agrees with an FP64 reference to about one BF16 ulp, is bitwise
deterministic from run to run, and does not depend on PDL.
"""

import math

import pytest
import torch
from test_kimi_k3_mxfp8_split_k_rubin import _k3_inputs, _rubin_available, _tactics

from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo
from tensorrt_llm._torch.cute_dsl_kernels.rubin.streamk_layout import (
    mailbox_bytes,
    mailbox_floats,
    select_epilogue_subtile,
)

# (N, K, M): K3 decode shapes. TP8 latent down / fused_a (28 / 17 weight-row tiles), g_proj at
# a few tokens (partial N tile: most epilogue subtiles have no valid column), KDA qkvg, the
# DEP16 latent down at the token counts of the rank, and K = 3584 (28 K tiles).
GEMM_CASES = [
    (3584, 7168, 6),
    (2112, 7168, 12),
    (1536, 7168, 24),
    (6144, 7168, 12),
    (7168, 3584, 48),
    (3584, 7168, 128),
    (2112, 7168, 256),  # weight rows that do not fill the last tile, many tokens
]
# Tactics checked per shape: each distinct tactic is a kernel compile.
MAX_TACTICS = 8

_NEEDS_RUBIN = pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_available(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)


@pytest.fixture
def occupancy(monkeypatch):
    """Max active clusters by cluster size of a 212-SM device (cudaOccupancyMaxActiveClusters)."""
    table = {1: 212, 2: 106, 3: 60, 4: 46, 5: 40, 6: 28, 7: 24, 8: 22}
    monkeypatch.setattr(cdo, "get_max_activate_clusters", lambda size: table[size], raising=False)
    return table


def test_streamk_flag_parsing(monkeypatch):
    monkeypatch.delenv("TRTLLM_K3_DENSE_STREAMK", raising=False)
    assert not cdo._k3_dense_streamk_enabled(), "in-cluster K split must be opt-in"
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    assert cdo._k3_dense_streamk_enabled()
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "0")
    assert not cdo._k3_dense_streamk_enabled()
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "yes")
    with pytest.raises(ValueError):
        cdo._k3_dense_streamk_enabled()


def test_streamk_factors_fit_one_wave_and_the_cluster(occupancy):
    # TP8 latent down at a few tokens: 28 tiles of 1-CTA clusters (56 K tiles). Cluster sizes up
    # to 8 CTAs; 28 clusters of 6 fit (28), of 7 (24) and 8 (22) do not; the grid is capped at
    # 192 CTAs (28 x 7 = 196).
    assert cdo._streamk_factors(56, 28, 1, 64) == [2, 3, 4, 5, 6]
    # 2-CTA clusters: 14 tiles; 2 x 4 = 8 CTAs is the largest cluster (2 x 3 = 6 fits 28 clusters).
    assert cdo._streamk_factors(56, 14, 2, 64) == [2, 3, 4]
    # Many tiles: the grid bound (192 CTAs) and the occupancy of the cluster size drop splits.
    assert cdo._streamk_factors(56, 48, 1, 64) == [2, 3]
    assert cdo._streamk_factors(56, 100, 1, 64) == []
    # Few tiles: every size up to the cluster limit fits one wave.
    assert cdo._streamk_factors(56, 17, 1, 64) == [2, 3, 4, 5, 6, 7, 8]
    # Each K rank streams at least 4 K tiles: 12 K tiles allow 2 and 3 splits only.
    assert cdo._streamk_factors(12, 12, 1, 64) == [2, 3]
    assert cdo._streamk_factors(7, 12, 1, 64) == []


def test_streamk_factors_bound_the_mailbox(occupancy):
    # The mailbox (64 KB at most) takes shared memory from the pipeline stages: wide tiles with
    # many K ranks do not leave enough stages and are not offered.
    assert cdo._streamk_factors(56, 17, 1, 128) == [2, 3, 4, 5, 8]
    assert cdo._streamk_factors(56, 17, 1, 64) == [2, 3, 4, 5, 6, 7, 8]


def test_streamk_epilogue_subtile_keeps_the_mailbox_small():
    # (K ranks, tile width) -> (subtile width, mailbox KB); every rank owns a subtile when it can.
    expected = {
        (2, 64): (32, 16),
        (3, 64): (16, 32),
        (4, 64): (16, 24),
        (2, 128): (32, 32),
        (3, 128): (16, 48),  # 32-wide subtiles would need 64 KB
        (4, 128): (32, 48),
        (8, 128): (16, 56),
    }
    for (ck, tile_n), (epi_n, kb) in expected.items():
        assert select_epilogue_subtile(ck, tile_n, 32) == epi_n, (ck, tile_n)
        assert mailbox_floats(ck, epi_n, tile_n) * 4 == kb * 1024, (ck, tile_n)
        assert mailbox_bytes(ck, tile_n) == kb * 1024, (ck, tile_n)
    # Never wider than the base epilogue subtile; a 64-wide tile cannot be cut finer than 16.
    assert select_epilogue_subtile(2, 128, 16) == 16
    assert select_epilogue_subtile(8, 64, 32) == 16


@_NEEDS_RUBIN
def test_streamk_tactics_exist_only_with_the_flag(monkeypatch):
    inputs = _k3_inputs(3584, 7168, 12)
    runner_cls = cdo.CuteDSLKimiK3MXFP8RubinLinear
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "0")
    runner = runner_cls(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    assert not any(t[0] == "streamk" for t in _tactics(runner, inputs))
    key_off = runner.unique_id()
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    tactics = [t for t in _tactics(runner, inputs) if t[0] == "streamk"]
    assert tactics and all(len(t) == 9 and t[6] == "static" and t[7] == "m" for t in tactics)
    assert runner.unique_id() != key_off, "autotuner cache entries must not mix"
    assert runner_cls.streamk_supported
    for other in (cdo.CuteDSLBlockScaledRubinLinear, cdo.CuteDSLMXFP8RubinLinear):
        assert not other.streamk_supported, other.__name__


def _dequant(q, sf, rows, k):
    """FP8 [rows, k] with R128c4 E8M0 scales [ceil(rows/128), k/128, 32, 4, 4] -> FP64."""
    mt, kt = (rows + 127) // 128, k // 128
    scale = sf.view(torch.uint8).reshape(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4)
    scale = torch.exp2(scale.reshape(mt * 128, kt * 4)[:rows].to(torch.float64) - 127.0)
    return q.to(torch.float64) * scale.repeat_interleave(32, dim=1)


def _reference(inputs):
    a, w, a_sf, w_sf = inputs[:4]
    m, k = a.shape
    return _dequant(a, a_sf, m, k) @ _dequant(w, w_sf, w.shape[0], k).t()


def _ulp_bound(ref: torch.Tensor) -> torch.Tensor:
    """BF16 half-ulp of each reference value, floored at 2**-9 of the largest magnitude."""
    mag = ref.abs().clamp_min(ref.abs().max() * 2.0**-9)
    return torch.exp2(torch.floor(torch.log2(mag)) - 8.0)


@_NEEDS_RUBIN
@pytest.mark.parametrize("n,k,m", GEMM_CASES)
def test_streamk_matches_fp64_reference_and_is_deterministic(monkeypatch, n, k, m):
    inputs = _k3_inputs(n, k, m)
    ref = _reference(inputs)
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactics = _tactics(runner, inputs)
    streamk = [t for t in tactics if t[0] == "streamk"]
    assert streamk, f"no in-cluster K-split tactic for N={n} K={k} M={m}"
    base = runner(inputs, tactic=next(t for t in tactics if t[0] == "base"))
    # Spread over the offered tile shapes and splits: first, last, and an even sample between.
    step = max(1, len(streamk) // MAX_TACTICS)
    checked = list(dict.fromkeys([*streamk[::step][: MAX_TACTICS - 1], streamk[-1]]))
    # BF16 rounds the FP32 sum once; summation-order error (~1e-7 relative) only matters at a
    # rounding boundary, so the output stays within one rounding step of the exact value.
    bound = 2.0 * _ulp_bound(ref)
    base_err = (base.double() - ref).abs()
    assert bool((base_err <= bound).all()), "the unsplit GEMM must meet the same bound"
    for tactic in checked:
        out = runner(inputs, tactic=tactic)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        err = (out.double() - ref).abs()
        assert bool((err <= bound).all()), f"{tactic}: max error {err.max().item():.3g}"
        for _ in range(5):
            assert torch.equal(runner(inputs, tactic=tactic), out), f"{tactic}: not deterministic"


@_NEEDS_RUBIN
@pytest.mark.parametrize("n,k,m", [(3584, 7168, 6), (3584, 7168, 128)])
def test_streamk_is_bitwise_across_pdl_and_prewait(monkeypatch, n, k, m):
    inputs = _k3_inputs(n, k, m)
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    streamk = [t for t in _tactics(runner, inputs) if t[0] == "streamk"]
    assert streamk
    for tactic in (streamk[0], streamk[-1]):
        monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "0")
        out = runner(inputs, tactic=tactic)
        monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
        monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "100000")
        assert torch.equal(runner(inputs, tactic=tactic), out), f"{tactic}: PDL changed the output"
        # Weights of the first K tiles load before the PDL wait; the reserve takes shared memory
        # from the A/B stages (the mailbox of the K split comes out of the same stages).
        for stages, reserve_kb in ((2, 0), (4, 8)):
            monkeypatch.setenv("TRTLLM_K3_GEMM_PREWAIT_STAGES", str(stages))
            monkeypatch.setenv("TRTLLM_K3_GEMM_PREWAIT_RESERVE_KB", str(reserve_kb))
            assert torch.equal(runner(inputs, tactic=tactic), out), (
                f"{tactic}: weight prefetch before the PDL wait changed the output"
            )
            assert any(
                isinstance(key, tuple)
                and key[:1] == ("streamk",)
                and "prewait" in key
                and key[key.index("prewait") + 1] == stages
                for key in runner.__class__.kernel_cache
            ), f"{tactic}: no staged kernel compiled for {stages} stages"
        monkeypatch.delenv("TRTLLM_K3_GEMM_PREWAIT_STAGES")
        monkeypatch.delenv("TRTLLM_K3_GEMM_PREWAIT_RESERVE_KB")
        monkeypatch.delenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS")


@_NEEDS_RUBIN
def test_streamk_rejects_more_splits_than_k_tiles(monkeypatch):
    inputs = _k3_inputs(1536, 512, 6)  # 4 K tiles
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactic = ("streamk", (128, 64, 128), (128, 64, 64), (1, 1), True, False, "static", "m", 6)
    with pytest.raises(ValueError, match="K tiles"):
        runner(inputs, tactic=tactic)


@_NEEDS_RUBIN
def test_streamk_unsplit_remainder_slices_cover_k_once(monkeypatch):
    """K tiles that the split does not divide (56 tiles / 3 ranks: 19 + 19 + 18)."""
    n, k, m = 3584, 7168, 6
    inputs = _k3_inputs(n, k, m)
    ref = _reference(inputs)
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactics = [
        t
        for t in _tactics(runner, inputs)
        if t[0] == "streamk" and t[-1] == 3 and math.prod(t[3]) == 1
    ]
    assert tactics, "no 3-way split of the 56 K tiles"
    out = runner(inputs, tactic=tactics[0])
    assert bool(((out.double() - ref).abs() <= 2.0 * _ulp_bound(ref)).all())


def _tactic(swap, cluster, tile_n, cluster_k):
    """("streamk", ...) tactic of a layout, as ``_streamk_tactics`` builds it."""
    tiler_m = 128 * math.prod(cluster)
    return (
        "streamk",
        (tiler_m, tile_n, 128),
        (tiler_m, tile_n, 64),
        cluster,
        swap,
        False,
        "static",
        "m",
        cluster_k,
    )


def _streamk_runner(monkeypatch):
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    return cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)


def _assert_matches_reference(runner, inputs, ref, tactic, repeats=3):
    """Within one BF16 rounding step of the FP64 reference and bitwise stable run to run."""
    out = runner(inputs, tactic=tactic)
    err = (out.double() - ref).abs()
    assert bool((err <= 2.0 * _ulp_bound(ref)).all()), f"{tactic}: max error {err.max().item():.3g}"
    for _ in range(repeats):
        assert torch.equal(runner(inputs, tactic=tactic), out), f"{tactic}: not deterministic"
    return out


# Every layout the tactic generator can offer: (cluster shape, CTA tile width, K ranks); the
# cluster of a split is (cluster shape) x K ranks CTAs, at most 8.
LAYOUTS = [
    (cluster, tile_n, cluster_k)
    for cluster in ((1, 1), (2, 1))
    for tile_n in (64, 128)
    for cluster_k in range(2, 9)
    if math.prod(cluster) * cluster_k <= 8
]


@_NEEDS_RUBIN
@pytest.mark.parametrize("swap", [True, False])
def test_streamk_every_layout_matches_reference(monkeypatch, swap):
    """All 20 layouts x swap, with a partial last weight tile and a partial token tile.

    N = 2080: 16 full 128-row weight tiles plus 32 rows (swapped: partial kernel M); as the
    kernel N it leaves 32 valid columns in the last 64- and 128-wide tile. M = 24 swapped
    (partial N tile of 24 columns) or 136 (a full and a partial row tile). Grids larger than
    one wave of the device (up to 66 clusters of 8 CTAs) must also work.
    """
    n, k, m = 2080, 7168, (24 if swap else 136)
    inputs = _k3_inputs(n, k, m)
    ref = _reference(inputs)
    runner = _streamk_runner(monkeypatch)
    for cluster, tile_n, cluster_k in LAYOUTS:
        _assert_matches_reference(runner, inputs, ref, _tactic(swap, cluster, tile_n, cluster_k))


# M around the epilogue subtile boundaries (16 and 32 columns) and the tile widths: the number
# of subtiles that hold a valid column, hence the mailbox arrivals every owner expects, changes
# with M, while the autotuner only validates the M of its tuning bucket.
M_SWEEP = [
    *range(1, 41),
    *(47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256),
]
SWEEP_LAYOUTS = [
    ((1, 1), 64, 2),
    ((1, 1), 64, 3),
    ((1, 1), 64, 8),  # 4 subtiles of 16 columns: ranks 4..7 only push
    ((1, 1), 128, 3),
    ((1, 1), 128, 8),
    ((2, 1), 128, 4),
]


@_NEEDS_RUBIN
def test_streamk_m_sweep_over_subtile_boundaries(monkeypatch):
    n, k = 1536, 7168
    runner = _streamk_runner(monkeypatch)
    for m in M_SWEEP:
        inputs = _k3_inputs(n, k, m)
        ref = _reference(inputs)
        for cluster, tile_n, cluster_k in SWEEP_LAYOUTS:
            tactic = _tactic(True, cluster, tile_n, cluster_k)
            try:
                _assert_matches_reference(runner, inputs, ref, tactic, repeats=1)
            except AssertionError as error:
                raise AssertionError(f"M={m}: {error}") from error


@_NEEDS_RUBIN
@pytest.mark.parametrize(
    "swap,cluster,tile_n,cluster_k,m",
    [
        (True, (1, 1), 64, 8, 6),  # ranks 1..7 only push (one valid subtile)
        (True, (1, 1), 128, 5, 12),
        (True, (2, 1), 64, 4, 24),
        (False, (1, 1), 128, 2, 128),
    ],
)
def test_streamk_is_stable_under_back_to_back_pdl_launches(
    monkeypatch, swap, cluster, tile_n, cluster_k, m
):
    """Thousands of graph-launched PDL GEMMs with a memory-bound kernel on a second stream.

    Pushers exit right after issuing their st.async stores and the owner arms its mailbox
    barriers while peers may already push; any lost or mis-ordered push would hang the kernel
    (the test then times out) or leave a different output.
    """
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "100000")
    inputs = _k3_inputs(1536, 7168, m)
    ref = _reference(inputs)
    runner = _streamk_runner(monkeypatch)
    tactic = _tactic(swap, cluster, tile_n, cluster_k)
    first = _assert_matches_reference(runner, inputs, ref, tactic)
    torch.cuda.synchronize()

    launches, replays = 32, 150
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [runner(inputs, tactic=tactic) for _ in range(launches)]
    side = torch.cuda.Stream()
    src = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(replays):
        with torch.cuda.stream(side):
            dst.copy_(src)
        graph.replay()
    torch.cuda.synchronize()
    expected = first.view(torch.int16)
    mismatches = sum(int((out.view(torch.int16) != expected).sum()) for out in outs)
    assert mismatches == 0, f"{tactic}: {mismatches} mismatching output values"


@_NEEDS_RUBIN
def test_streamk_is_chosen_and_run_by_the_autotuner(monkeypatch):
    """The op path of the model: tune once, then run the cached tactic.

    TP8 latent down at 6 tokens has 28 weight tiles, so the unsplit kernel streams the whole of
    K with 28 CTAs; the in-cluster split roughly halves its time and must win the tuning.
    """
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    n, k, m = 3584, 7168, 6
    inputs = _k3_inputs(n, k, m)
    ref = _reference(inputs)
    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    args = (inputs[0], inputs[1], inputs[2], inputs[3], torch.bfloat16, True, True)
    with autotune(skip_dynamic_tuning_buckets=True):
        tuned = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(*args)
    cached = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(*args)
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    _, best = AutoTuner.get().choose_one(
        "trtllm::cute_dsl_mxfp8_gemm_rubin", [runner], runner.__class__.tuning_config, inputs
    )
    assert best[0] == "streamk", f"the autotuner chose {best}"
    bound = 2.0 * _ulp_bound(ref)
    for out in (tuned, cached):
        assert bool(((out.double() - ref).abs() <= bound).all())
    assert torch.equal(tuned, cached)
