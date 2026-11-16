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
"""C3 gate: the 2-CTA QKV split-K with the packed decode prep in its epilogue (``tep8_wide64_splitk``
PREP_FOLD, ``ops/gemm/splitk_gemm_2cta_prep``) against the two launches it replaces.

Bits: every output -- the raw projection, ``qkv_out``, ``gate_out``, the E4M3 query buffer, every
byte of the FP8 K/V pool, the bmm1 / bmm2 scale words and ``cu_seqlens`` -- equals what
``splitk_gemm_2cta`` (reduce-scatter, prestage) followed by ``multi_request_prep_append`` (packed
kernel) writes, byte for byte, including zero, tiny (subnormal squares), huge (overflowing squares,
saturating E4M3), infinite and NaN heads, both QKV layouts, with and without RoPE, int32 / int64
positions, split_k 4 (served) and 2, at 65..128 rows; and from a CUDA graph. Timing: a 62-layer chain
of distinct weights, 2-CTA + packed prep vs the fused launch, us per layer ([C3] lines; the gate wants
>= 1.2 us per layer at 96 and 128 rows).
"""

import importlib
import subprocess
import sys

import pytest
import torch

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires an sm107 device",
)

H, G, D, PAGE = 16, 2, 64, 32
N, K = 2 * (H + G) * D, 8192
EPS = 1e-5
Q_SCALE = 43.7840518911 / D**0.5


def _module(name: str):
    return importlib.import_module(f"{_TARGET}.{name}")


def _inputs(
    rows: int,
    rows_per_seq: int,
    seed: int,
    permuted: bool,
    rope: bool,
    pos64: bool,
    eps: float = EPS,
):
    """The projection operands (with special rows) and the prep's operands of one call."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    seqs = rows // rows_per_seq
    x = torch.randn(rows, K, device="cuda", generator=gen)
    w = torch.randn(N, K, device="cuda", generator=gen) * 0.02
    w[:, 0] = torch.randn(N, device="cuda", generator=gen) * 4.0
    special = {
        0: 0.0,  # zero heads
        1: 2.0**60,  # huge heads: saturating E4M3
        2: 2.0**-60,  # tiny heads: subnormal squares
        3: 2.0**100,  # overflowing squares
        4: 3.0e38,  # infinite outputs
        5: float("nan"),  # NaN heads
        6: 2.0**-70,  # subnormal squares (add.rn.ftz flushes them: visible with eps 0)
    }
    for row, value in special.items():
        if row < rows:
            x[row] = 0.0
            x[row, 0] = value
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    pages_per_seq = 16
    num_pages = seqs * pages_per_seq * 2 + 3
    lens = torch.randint(
        rows_per_seq, pages_per_seq * PAGE, (seqs,), device="cuda", generator=gen
    ).to(torch.int32)
    perm = torch.randperm(num_pages, device="cuda", generator=gen)[: seqs * 2 * pages_per_seq]
    tables = perm.view(seqs, 2, pages_per_seq).to(torch.int32).contiguous()
    pool = torch.randint(
        0, 255, (num_pages, G, PAGE, D), dtype=torch.uint8, device="cuda", generator=gen
    )
    cos_sin = torch.rand(1024, 2, D // 2, device="cuda", generator=gen) * 2.0 - 1.0
    positions = (
        lens.long()[:, None] - rows_per_seq + torch.arange(rows_per_seq, device="cuda")
    ).reshape(-1)
    positions = positions.to(torch.int64 if pos64 else torch.int32).view(1, -1)
    prep = dict(
        cos_sin=cos_sin if rope else None,
        position_ids=positions if rope else None,
        num_q_heads=H,
        num_kv_heads=G,
        eps=eps,
        q_scale=Q_SCALE,
        qkv_permuted=permuted,
        num_seqs=seqs,
        kv_pool=pool.view(torch.float8_e4m3fn),
        block_tables=tables,
        seq_lens=lens,
        kv_scale_orig_quant=torch.tensor([0.75], device="cuda"),
        kv_scale_quant_orig=torch.tensor([1.0 / 0.75], device="cuda"),
        q_fp8=torch.full((rows * H * D + 64,), 0x5A, dtype=torch.uint8, device="cuda"),
        bmm1_scale=torch.zeros(2, device="cuda"),
        bmm2_scale=torch.zeros(1, device="cuda"),
        cu_seqlens=torch.full((seqs + 1,), -1, dtype=torch.int32, device="cuda"),
        bmm1_host_scale=0.125,
        o_scale=1.0,
    )
    return x, w, prep


_MUTABLE = ("kv_pool", "q_fp8", "bmm1_scale", "bmm2_scale", "cu_seqlens")


def _clone(prep: dict) -> dict:
    return {k: (v.clone() if k in _MUTABLE else v) for k, v in prep.items()}


def _reference(x, w, prep, split_k: int):
    """2-CTA (reduce-scatter, prestage) then the packed prep, as served."""
    two = _module("ops.gemm.splitk_gemm_2cta")
    mr = _module("ops.attention.multi_request_prep_append")
    mine = _clone(prep)
    out = torch.empty(x.shape[0], N, dtype=torch.bfloat16, device="cuda")
    two.run_splitk_2cta(x, w, out, True, split_k, rs=True, prestage=True)
    qkv_out, gate = mr.multi_request_prep_append(out, **mine, enable_pdl=True, packed_min_rows=33)
    return out, qkv_out, gate, mine


def _fused(x, w, prep, split_k: int):
    fold = _module("ops.gemm.splitk_gemm_2cta_prep")
    mine = _clone(prep)
    rows = x.shape[0]
    out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
    qkv_out = torch.empty(rows, (H + 2 * G) * D, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(rows, H * D, dtype=torch.bfloat16, device="cuda")
    fold.run_splitk_2cta_prep_fold(
        x, w, out, True, split_k, True, qkv_out=qkv_out, gate_out=gate, **mine
    )
    return out, qkv_out, gate, mine


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return int((a != b).sum())


def _compare(ref, fused) -> dict:
    out_r, qkv_r, gate_r, mine_r = ref
    out_f, qkv_f, gate_f, mine_f = fused
    counts = {
        "raw": _differ(out_r, out_f),
        "qkv_out": _differ(qkv_r, qkv_f),
        "gate": _differ(gate_r, gate_f),
    }
    for name in _MUTABLE:
        counts[name] = _differ(mine_r[name], mine_f[name])
    return counts


_SMOKE = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
import test_modeling_v2_qkv_prep_fold as t
for split_k in (4, 2):
    x, w, prep = t._inputs(128, 8, 7, False, True, False)
    t._fused(x, w, prep, split_k)
    torch.cuda.synchronize()
    print("SMOKE OK split_k", split_k, flush=True)
"""


@pytest.fixture(scope="module")
def fold_runs():
    """One fused launch per split_k in a subprocess first: a launch that hangs (a reduce-scatter
    byte count gone wrong) fails here within the timeout instead of hanging the session."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SMOKE, here],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("[C3] the fused launch did not finish within 900 s (hang?)")
    print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        pytest.fail(f"[C3] the fused launch failed:\n{proc.stderr[-6000:]}")
    return True


CASES = [
    # (rows, rows per sequence)
    (72, 8),
    (96, 8),
    (128, 8),
    (97, 1),
]
VARIANTS = [
    # (QKV permuted, RoPE, int64 positions)
    (False, True, False),
    (True, True, False),
    (False, False, False),
    (True, True, True),
]


@pytest.mark.parametrize("permuted, rope, pos64", VARIANTS)
@pytest.mark.parametrize("split_k", [4, 2])
@pytest.mark.parametrize("rows, rows_per_seq", CASES)
def test_c3_fold_bits(fold_runs, rows, rows_per_seq, split_k, permuted, rope, pos64):
    x, w, prep = _inputs(rows, rows_per_seq, 31 * rows + split_k, permuted, rope, pos64)
    ref = _reference(x, w, prep, split_k)
    fused = _fused(x, w, prep, split_k)
    torch.cuda.synchronize()
    counts = _compare(ref, fused)
    tag = f"rows {rows} (Q {rows_per_seq}) split_k {split_k} permuted {permuted} rope {rope} pos64 {pos64}"
    print(f"[C3] bits {tag}: bytes differing {counts}", flush=True)
    # The pool's untouched bytes, the query buffer's tail and the cu_seqlens of a Q=1 step are
    # compared too (both sides start from the same bytes).
    assert all(v == 0 for v in counts.values()), f"{tag}: {counts}"
    # The written K/V bytes moved at all (the comparison is not of two untouched pools).
    assert _differ(prep["kv_pool"], fused[3]["kv_pool"]) > 0


@pytest.mark.parametrize("split_k", [4, 2])
def test_c3_fold_bits_eps0(fold_runs, split_k):
    """eps 0: the norm's in-lane sum is the only thing between a head of subnormal squares and its
    output, so the fold's add.rn.ftz must be the prep's (a non-ftz add keeps them)."""
    x, w, prep = _inputs(96, 8, 77 + split_k, False, True, False, eps=0.0)
    counts = _compare(_reference(x, w, prep, split_k), _fused(x, w, prep, split_k))
    print(f"[C3] bits eps 0 rows 96 split_k {split_k}: bytes differing {counts}", flush=True)
    assert all(v == 0 for v in counts.values()), counts


def test_c3_fold_graph_replays(fold_runs):
    """Two fused launches captured in one CUDA graph (the second under PDL behind the first) write
    the eager bytes on replay."""
    x, w, prep = _inputs(96, 8, 5, False, True, False)
    eager = _fused(x, w, prep, 4)
    fold = _module("ops.gemm.splitk_gemm_2cta_prep")
    mine = _clone(prep)
    out = torch.empty(96, N, dtype=torch.bfloat16, device="cuda")
    qkv_out = torch.empty(96, (H + 2 * G) * D, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(96, H * D, dtype=torch.bfloat16, device="cuda")
    run = lambda: fold.run_splitk_2cta_prep_fold(  # noqa: E731
        x, w, out, True, 4, True, qkv_out=qkv_out, gate_out=gate, **mine
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
        run()
    for name in _MUTABLE:
        mine[name].copy_(prep[name])
    out.zero_()
    qkv_out.zero_()
    gate.zero_()
    graph.replay()
    torch.cuda.synchronize()
    counts = _compare(eager, (out, qkv_out, gate, mine))
    print(f"[C3] graph replay: bytes differing {counts}", flush=True)
    assert all(v == 0 for v in counts.values()), counts


def _per_layer(run, layers: int) -> float:
    """us per layer of ``run(l)`` for l in 0..layers-1, from a CUDA graph (PDL as served)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run(0)
        run(1)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(layers):
            run(layer)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (10 * layers))
    return sorted(samples)[len(samples) // 2]


@pytest.mark.parametrize("rows", [96, 128])
def test_c3_fold_timing(fold_runs, rows, capsys):
    """62 layers of distinct weights: 2-CTA + packed prep vs the fused launch (and each alone)."""
    two = _module("ops.gemm.splitk_gemm_2cta")
    mr = _module("ops.attention.multi_request_prep_append")
    fold = _module("ops.gemm.splitk_gemm_2cta_prep")
    layers = 62
    x, _, prep = _inputs(rows, 8, 11, False, True, False)
    gen = torch.Generator(device="cuda").manual_seed(12)
    weights = [
        (torch.randn(N, K, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        for _ in range(layers)
    ]
    mine = _clone(prep)
    out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
    qkv_out = torch.empty(rows, (H + 2 * G) * D, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(rows, H * D, dtype=torch.bfloat16, device="cuda")

    def two_cta(layer):
        two.run_splitk_2cta(x, weights[layer], out, True, 4, rs=True, prestage=True)

    def prep_only(layer):
        mr.multi_request_prep_append(out, **mine, enable_pdl=True, packed_min_rows=33)

    def unfused(layer):
        two_cta(layer)
        prep_only(layer)

    def fused(layer):
        fold.run_splitk_2cta_prep_fold(
            x, weights[layer], out, True, 4, True, qkv_out=qkv_out, gate_out=gate, **mine
        )

    times = {
        "2cta+prep": _per_layer(unfused, layers),
        "fused": _per_layer(fused, layers),
        "2cta": _per_layer(two_cta, layers),
        "prep": _per_layer(prep_only, layers),
    }
    saved = times["2cta+prep"] - times["fused"]
    verdict = "PASS" if saved >= 1.2 else "FAIL"
    with capsys.disabled():
        print(
            f"\n[C3] timing {rows} rows ({layers} layers, us/layer): "
            + ", ".join(f"{k} {v:6.2f}" for k, v in times.items())
            + f"; saved {saved:+.2f} us/layer = {saved * layers / 1e3:+.3f} ms/step "
            f"(gate >= 1.2: {verdict})",
            flush=True,
        )
