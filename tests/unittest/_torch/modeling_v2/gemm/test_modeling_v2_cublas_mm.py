# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry.

Beyond the cells, the DeepSeek-V4.1 section sweeps every row count of each of
that target's calls, replays them under CUDA graphs at the row counts the target
captures, and keeps the controls that show the gates see TF32 and bf16-partial
split-K arithmetic. The router's bf16 -> fp32 call has its own file,
test_modeling_v2_cublas_mm_router_bf16.py.
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


def _build(spec, seed):
    """Materialize one cell.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target.
    `w_std`, when a cell gives one, scales the weight before rounding.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    weight = torch.randn(n, k, generator=g, device="cuda")
    if spec.get("w_std") is not None:
        weight = weight * spec["w_std"]
    mat_b = weight.to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    with validating(op):
        out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    op.compare(out, op.reference(mat_a, mat_b, bias, out_dtype))


def _ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_guard_refuses_a_row_major_mat_b() -> None:
    """Read column-major regardless; a row-major b is a different product."""
    a, _ = _ok()
    b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_strided_mat_a() -> None:
    a = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")[:, :64]
    _, b = _ok()
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    """Accepted by the op, and the result is silently wrong."""
    a, b = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(op):
            op(a, b, bias)


def test_guard_refuses_a_bias_on_fp32_inputs() -> None:
    """The op takes it and never adds it."""
    a, b = _ok(torch.float32)
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="silently ignored"):
        with validating(op):
            op(a, b, bias)


# ---------------------------------------------------------------------------
# DeepSeek-V4.1. The cells above hold each call at the ends of its row range;
# what follows keeps the measured range certified: every M swept, CUDA-graph
# replay at the row counts the target captures, and the controls showing the
# gates can see the failures they are there for.
#
# The core target's calls pass bias=None, a contiguous activation and the .t()
# view of a contiguous [N, K] parameter:
#   fp32 [T, 5120] x [5120, 1024]    compressor [kv || score], ratio-2 KV source
#   fp32 [T, 5120] x [5120, 384]     router gate on fp32 inputs
#   fp32 [5G, 5120] x [5120, 128]    DSpark drafter MoE gate (128 routed experts)
#   fp32 [G, 256] x [256, 129280]    DSpark Markov head: the fp32 copy of the
#                                    gathered W1 row times W2 [129280, 256]
#   bf16 [T, 5120] x [5120, 512]     compressor projection, ratio-1 KV source
#   bf16 [n, 512] x [512, 128]       index-K projection of the compressed latent
#   bf16 [T, 5120] x [5120, 32]      indexer weights_proj
# T is the step's tokens, at most max_num_tokens (16384 at the largest envelope
# engine); weights_proj also runs in the re-index layers of the decoder
# bounded-replay pass, whose rows reach 128 x 256; the routed MoE splits its
# rows into chunks of at most 8192 before the router; G is the step's
# generation requests, at most 256.
#
# The bf16 gate's atol at these calls, per K (rtol 1.6e-2). With unit-variance
# operands the outputs have std sqrt(K), 72 at K = 5120, and near a cancellation
# only the absolute fp32 accumulation error is left, in the op and in the IEEE
# fp32 reference alike. Measured on sm_103 against a float64 product at
# M = N_max over 16 seeds: the op's error beyond half an output ulp reached
# 1.8e-3 at K = 5120 (4.1e-5 at K = 512), and the fp32 reference's own reached
# 1.1e-3 (8.6e-5); one element of 8.4M in a 16384-row draw at K = 5120 sat
# 1.1e-3 from the reference. So K = 5120 takes 4e-3, 5.6e-5 of the output std.
_DSV41_BF16_ATOL = {512: 1e-3, 5120: 4e-3}
_BF16_RTOL, _BF16_ATOL = 1.6e-2, 1e-3
# Max |op - float64 product| per K for unit-variance fp32 operands. Measured on
# sm_103: at most 1.56e-3 at K = 5120 (every M to 16384, outputs up to ~400) and
# 1.05e-4 at K = 256; a TF32 product of the same operands sits 0.09..0.11
# (K = 5120, M >= 32) and 0.022..0.027 (K = 256) away. K = 256 takes the lower
# gate so the TF32 control still lands more than 4x outside it.
_FP32_GATE = {5120: 5e-3, 256: 1e-3}
# The capture set. G ranges over every CUDA-graph batch size an envelope engine
# (max_batch_size up to 256) captures: every G 1..32, plus TRT-LLM's padded
# default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 is a subset. A
# captured step is generation-only, a decode step (next_n 1) or a DSpark
# target-verify step (next_n 6), and the row count each call sees is:
#   G     every call, decode step;
#   6G    the calls fed by the step's tokens (compressors, weights_proj, router,
#         and index-K at the ratio-1 source), verify step;
#   3G    index-K at the ratio-2 source, verify step (its compressed-row
#         reservation: 3 rows per verify request);
#   4G, 24G  the router on dep4's gathered rows (decode, verify);
#   5G, G    the DSpark gate and Markov head (the drafter runs inside the
#            target's decode graphs).
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)


def _rows(*factors: int) -> tuple[int, ...]:
    return tuple(sorted({b * f for b in _GSET for f in factors}))


def _fp32_err(out: torch.Tensor, mat_a: torch.Tensor, mat_b: torch.Tensor) -> float:
    """Max |out - float64 product|; the entry's fp32 reference is that product."""
    return (out.double() - op.reference(mat_a, mat_b)).abs().max().item()


def _tf32_mm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """torch's fp32 matmul with TF32 forced on: the control product."""
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = True
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        return mat_a @ mat_b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def _bf16_violations(
    out: torch.Tensor, ref: torch.Tensor, atol: float = _BF16_ATOL
) -> torch.Tensor:
    """Elements outside the bf16 gate, as a 0-d device tensor (no host sync)."""
    o, r = out.float(), ref.float()
    return ((o - r).abs() > atol + _BF16_RTOL * r.abs()).sum()


@pytest.mark.parametrize("k, n, m", [(5120, 384, 1005), (5120, 128, 40), (256, 129280, 40)])
def test_dsv41_fp32_compare_rejects_tf32(k: int, n: int, m: int) -> None:
    """A TF32 product of the fp32 cells' operands fails `compare`, even at a
    quarter of its error, so a TF32 path cannot pass unseen. The op passes.

    M = 40 at the DSpark shapes: torch's TF32 matmul at M = 1 and K = 5120 took
    a GEMV path that is not TF32 (4.4e-5 from float64).
    """
    torch.manual_seed(10 + k + n)
    mat_a = torch.randn(m, k, device="cuda")
    mat_b = torch.randn(n, k, device="cuda").t()
    ref = op.reference(mat_a, mat_b)
    op.compare(op(mat_a, mat_b), ref)
    quarter = (ref + (_tf32_mm(mat_a, mat_b).double() - ref) / 4).float()
    with pytest.raises(AssertionError, match="exceeds"):
        op.compare(quarter, ref)
    assert _fp32_err(_tf32_mm(mat_a, mat_b), mat_a, mat_b) > 4 * _FP32_GATE[k]


# The DeepSeek-V4.1 vision tower and aligner. Every call is bf16 in and out,
# with a contiguous activation and the .t() view of a contiguous [N, K]
# parameter, and the checkpoint's bf16 bias fused in the epilogue where it has
# one. P = patches of one request, all its images in one call: at most 17037
# over the 900 MMMU val documents. C = aligner rows per request: at most 1902;
# the sweep goes to 2048. Operands: unit-normal activations and bias, N(0, 1/K)
# weights (a trained layer's scale, so outputs are O(1)). The reference is the
# float64 product plus the bias, rounded once to bf16, and the gate is rtol
# 1.6e-2, atol 1e-3 plus the fraction of elements bit-identical to it.
_VIS_CELLS = {
    # name: (K, N, has_bias, M_max)
    "patch_embed": (588, 1024, True, 17037),
    "vit_wqkv": (1024, 3072, True, 17037),
    "vit_wo": (1024, 1024, True, 17037),
    "vit_w1": (1024, 5632, False, 17037),
    "vit_w2": (2816, 1024, False, 17037),
    "aligner_w1": (9216, 5120, True, 2048),
    "aligner_w2": (5120, 5120, True, 2048),
}
# The one excursion the sweep finds, and what `is_valid` refuses: at K 588,
# N 1024 and M 2305..2432, cuBLASLt's heuristic picks a split-K kernel whose
# partial sums are stored in bf16, so each partial is rounded before the sum.
_PATCH_BAD_ROWS = range(2305, 2433)
# cuBLASLt's split-K reduce with bf16 partial sums.
_BF16_PARTIALS = "splitKreduce_kernel<32, 16, int, __nv_bfloat16"
# Floor on the fraction of elements bit-identical to the rounded-once reference.
# Measured over every M of every vision cell outside the band: >= 0.9946.
_VIS_MIN_EXACT = 0.99


def _vis_operands(k: int, n: int, has_bias: bool, rows: int, seed: int):
    torch.manual_seed(seed)
    mat_a = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
    mat_b = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16).t()
    bias = torch.randn(n, device="cuda").to(torch.bfloat16) if has_bias else None
    return mat_a, mat_b, bias


def _vis_ref(mat_a: torch.Tensor, mat_b: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """float64 product plus bias, rounded once to bf16."""
    ref = mat_a.double() @ mat_b.double()
    if bias is not None:
        ref = ref + bias.double()
    return ref.to(torch.bfloat16)


def _bit_exact(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.view(torch.int16) == ref.view(torch.int16)).float().mean())


def test_guard_refuses_the_patch_embed_split_k_band() -> None:
    """bf16 K = 588, N = 1024 at M 2305..2432: the op returns a plausible wrong
    answer (shown unguarded first), and `is_valid` refuses exactly that band.
    """
    k, n, has_bias, _ = _VIS_CELLS["patch_embed"]
    mat_a, mat_b, bias = _vis_operands(k, n, has_bias, 2433, 61)
    ref = _vis_ref(mat_a, mat_b, bias)
    for m in (2305, 2368, 2432):
        out = op(mat_a[:m], mat_b, bias)
        outside = int(_bf16_violations(out, ref[:m])) / out.numel()
        assert _bit_exact(out, ref[:m]) < 0.9 and outside > 1e-3, (m, outside)
        with pytest.raises(AssertionError, match="split-K"):
            with validating(op):
                op(mat_a[:m], mat_b, bias)
    for m in (2304, 2433):
        with validating(op):
            out = op(mat_a[:m], mat_b, bias)
        assert int(_bf16_violations(out, ref[:m])) == 0, m


def test_dsv41_patch_embed_band_padded_and_unbiased() -> None:
    """At M 2305..2432 a call padded with zero rows to 2433 is in the gate on its
    first M rows. Dropping the bias does not help.
    """
    k, n, has_bias, _ = _VIS_CELLS["patch_embed"]
    mat_a, mat_b, bias = _vis_operands(k, n, has_bias, 2433, 60)
    ref = _vis_ref(mat_a, mat_b, bias)
    for m in (2305, 2368, 2432):
        padded = torch.zeros(2433, k, dtype=torch.bfloat16, device="cuda")
        padded[:m] = mat_a[:m]
        with validating(op):
            out = op(padded, mat_b, bias)[:m]
        exact = _bit_exact(out, ref[:m])
        assert int(_bf16_violations(out, ref[:m])) == 0 and exact >= _VIS_MIN_EXACT, (m, exact)
        exact_nb = _bit_exact(op(mat_a[:m], mat_b), _vis_ref(mat_a[:m], mat_b, None))
        print(f"\n[patch M={m}] padded to 2433: bit-exact {exact:.4f}; no bias: {exact_nb:.4f}")
        assert exact_nb < 0.9


def _capture_and_replay(k: int, n: int, dtype: torch.dtype, rows) -> None:
    """Warm up eagerly, capture one call per row count, replay on two fresh
    inputs written into the captured input buffer with the output poisoned:
    every replay is bitwise equal to an eager call on the same input, passes
    the call's gate, and differs from the previous replay (a stale replay
    would repeat it)."""
    weight = torch.randn(n, k, device="cuda").to(dtype)
    mat_b = weight.t()
    for m in rows:
        mat_a = torch.randn(m, k, device="cuda").to(dtype)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(mat_a, mat_b)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(mat_a, mat_b)
        previous = None
        for _ in range(2):
            mat_a.copy_(torch.randn(m, k, device="cuda").to(dtype))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, op(mat_a, mat_b)), (k, n, m, "replay != eager")
            assert previous is None or not torch.equal(out, previous), (k, n, m, "stale replay")
            previous = out.clone()
            if dtype == torch.float32:
                err = _fp32_err(out, mat_a, mat_b)
                assert err <= _FP32_GATE[k], (k, n, m, err)
            else:
                torch.testing.assert_close(
                    out, op.reference(mat_a, mat_b), rtol=_BF16_RTOL, atol=_DSV41_BF16_ATOL[k]
                )
        del graph, out


# Every compressor and indexer call is captured at the union G, 3G, 6G (97 row
# counts, to 1536; 3G is extra coverage except at index-K); the fp32 router also
# at 4G and 24G (133, to 6144). The DSpark gate at every M 1..160 plus 5G (174,
# to 1280), the Markov head at every M 1..160 plus G (162, to 256).
_DSV41_GRAPH_CELLS = (
    (5120, 1024, torch.float32, _rows(1, 3, 6)),
    (5120, 384, torch.float32, _rows(1, 3, 6, 4, 24)),
    (5120, 512, torch.bfloat16, _rows(1, 3, 6)),
    (512, 128, torch.bfloat16, _rows(1, 3, 6)),
    (5120, 32, torch.bfloat16, _rows(1, 3, 6)),
    (5120, 128, torch.float32, tuple(sorted(set(range(1, 161)) | set(_rows(5))))),
    (256, 129280, torch.float32, tuple(sorted(set(range(1, 161)) | set(_GSET)))),
)
assert len(_GSET) == 46 and max(_GSET) == 256
assert [len(c[3]) for c in _DSV41_GRAPH_CELLS] == [97, 133, 97, 97, 97, 174, 162]
assert [max(c[3]) for c in _DSV41_GRAPH_CELLS] == [1536, 6144, 1536, 1536, 1536, 1280, 256]


@pytest.mark.parametrize(
    "k, n, dtype, rows", _DSV41_GRAPH_CELLS, ids=[f"{c[0]}x{c[1]}" for c in _DSV41_GRAPH_CELLS]
)
def test_dsv41_cuda_graph_replay(k: int, n: int, dtype: torch.dtype, rows) -> None:
    torch.manual_seed(12 + k + n)
    _capture_and_replay(k, n, dtype, rows)
    print(
        f"\n[capture K={k} N={n} {str(dtype).removeprefix('torch.')}] {len(rows)} captures, "
        f"{2 * len(rows)} replays bitwise = eager, in the gate"
    )


def _sweep(mat_a, mat_b, bias, rows, record) -> dict:
    """op on the first m rows of mat_a for every m in `rows`, calling
    record(m, out) after each; returns {m: names of the kernels that call launched}.

    Rows run largest first, so the caching allocator serves every later output
    from blocks it already holds. Each call is bracketed by two FillFunctor
    marker kernels, so the profiler's kernel stream splits into per-call sets;
    record() launches no fill kernel. Unguarded: the vision sweep walks the
    patch embed's refused band on purpose.
    """
    marker = torch.zeros(1, device="cuda")
    for m in (rows[0], rows[-1]):  # handle and workspace set up before the profiler starts
        op(mat_a[:m], mat_b, bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for m in rows:
            marker.fill_(1.0)
            out = op(mat_a[:m], mat_b, bias)
            marker.fill_(2.0)
            record(m, out)
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    per_call, cur, inside = [], [], False
    for e in evs:
        if "FillFunctor" in e.name:
            if inside:
                per_call.append(tuple(cur))
            cur, inside = [], not inside
        elif inside:
            cur.append(e.name)
    assert len(per_call) == len(rows), (len(per_call), len(rows))
    assert all(1 <= len(k) <= 2 for k in per_call), "a call launched no kernel or more than two"
    return dict(zip(rows, per_call))


def _bands(kernels: dict) -> list:
    """(lo, hi, names) runs of consecutive M that launched the same kernels."""
    ms = sorted(kernels)
    bands, lo = [], ms[0]
    for prev, m in zip(ms, ms[1:] + [None]):
        if m is None or kernels[m] != kernels[lo]:
            bands.append((lo, prev, kernels[lo]))
            lo = m
    return bands


def _ranges(ms) -> str:
    """'a..b c..d' runs of consecutive integers in the sorted list ms."""
    runs, lo, prev = [], None, None
    for m in list(ms) + [None]:
        if lo is not None and (m is None or m != prev + 1):
            runs.append(f"{lo}..{prev}")
            lo = None
        if m is not None and lo is None:
            lo = m
        prev = m
    return " ".join(runs)


# cuBLAS picks its kernel by M (none of these (M, K, N) is in the op's own
# algorithm table): cublasGemmEx's default heuristic for fp32, cuBLASLt's for
# bf16. On sm_103 the kernel changes every 16 to 64 rows over most of the range,
# with split-K reduce kernels coming and going, so every M in 1..N_max is run:
# that covers each band's edges and interior and every runtime-dumped row count.
# GEMM rows are independent, so one reference at N_max rows serves every M as
# its first M rows. Each bf16 call also holds the fraction of elements
# bit-identical to the rounded-once reference to a floor at every M: a
# bf16-partial split-K kernel collapses it (0.685 or less at the patch embed).
_DSV41_SWEEP_CELLS = {
    # name: (K, N, dtype, N_max)
    "compressor_ratio2_fp32": (5120, 1024, torch.float32, 16384),
    "router_gate_fp32": (5120, 384, torch.float32, 8192),
    "dspark_gate_fp32": (5120, 128, torch.float32, 8192),
    "dspark_markov_head_fp32": (256, 129280, torch.float32, 8192),
    "compressor_ratio1_bf16": (5120, 512, torch.bfloat16, 16384),
    "index_k_proj_bf16": (512, 128, torch.bfloat16, 16384),
    "indexer_weights_proj_bf16": (5120, 32, torch.bfloat16, 32768),
}
_CORE_MIN_EXACT = 0.99
# Cells whose every M returns exactly the first M rows of the N_max call
# (asserted; at the other cells some M do not, and nothing is claimed).
_PREFIX_BITWISE_CELLS = {"index_k_proj_bf16"}


@pytest.mark.parametrize("cell", list(_DSV41_SWEEP_CELLS))
def test_dsv41_every_row_count(cell: str) -> None:
    """Every M in 1..N_max at one core call, held to the reference; no M takes a
    bf16-partial split-K kernel; at a bf16 call every M keeps the bit-exact
    floor. The kernel band map and the M whose rows are bitwise those of the
    N_max call are reported (every M, asserted, at _PREFIX_BITWISE_CELLS)."""
    k, n, dtype, n_max = _DSV41_SWEEP_CELLS[cell]
    torch.manual_seed(30 + list(_DSV41_SWEEP_CELLS).index(cell))
    mat_a = torch.randn(n_max, k, device="cuda").to(dtype)
    mat_b = torch.randn(n, k, device="cuda").to(dtype).t()
    if dtype == torch.float32:
        # The float64 product rounded to fp32: that moves it by at most half an
        # fp32 ulp of the output (< 3.1e-5 at the |out| <= 512 seen here),
        # which is under 3.1% of the tightest gate.
        ref = op.reference(mat_a, mat_b).float()
        gate = _FP32_GATE[k]
    else:
        ref = op.reference(mat_a, mat_b)
        ref_bits = ref.view(torch.int16)
    worst = torch.zeros(n_max + 1, device="cuda")
    exact = torch.zeros(n_max + 1, device="cuda")
    differs = torch.zeros(n_max + 1, device="cuda")
    last = {}

    def record(m: int, out: torch.Tensor) -> None:
        if dtype == torch.float32:
            worst[m] = (out - ref[:m]).abs().max()
        else:
            worst[m] = _bf16_violations(out, ref[:m], _DSV41_BF16_ATOL[k])
            exact[m] = (out.view(torch.int16) == ref_bits[:m]).sum()
        if m == n_max:
            last["out"] = out
        else:
            differs[m] = (out != last["out"][:m]).any()

    kernels = _sweep(mat_a, mat_b, None, list(range(n_max, 0, -1)), record)
    out = last["out"]
    assert out.shape == (n_max, n) and out.dtype == dtype
    bands = _bands(kernels)
    worst, exact, differs = worst.cpu(), exact.cpu(), differs.cpu()
    distinct = {name for names in kernels.values() for name in names}
    reduces = sorted({x.split("(")[0][:90] for x in distinct if "educe" in x})
    bf16_partials = sorted(
        m for m, names in kernels.items() if any(_BF16_PARTIALS in x for x in names)
    )
    if dtype == torch.float32:
        bad = [(m, float(worst[m])) for m in range(1, n_max + 1) if worst[m] > gate]
        summary = (
            f"max |op - float64| {float(worst.max()):.3e} at M={int(worst.argmax())}, gate {gate:g}"
        )
    else:
        bad = [(m, int(worst[m])) for m in range(1, n_max + 1) if worst[m] > 0]
        summary = (
            "0 elements outside the bf16 gate" if not bad else f"{len(bad)} row counts outside"
        )
        frac = [float(exact[m]) / (m * n) for m in range(1, n_max + 1)]
        low = [m for m in range(1, n_max + 1) if frac[m - 1] < _CORE_MIN_EXACT]
        m_min = 1 + min(range(n_max), key=frac.__getitem__)
        summary += f"; min bit-exact {frac[m_min - 1]:.4f} at M={m_min}, floor {_CORE_MIN_EXACT}"
    same = [m for m in range(1, n_max) if not differs[m]] + [n_max]
    other = [m for m in range(1, n_max) if differs[m]]
    print(
        f"\n[{cell} K={k} N={n}] every M 1..{n_max}: {summary}; "
        f"{len(bands)} kernel bands, {len(distinct)} distinct kernels; "
        f"M with bf16-partial split-K: {len(bf16_partials)}"
    )
    print(f"  split-K reduce kernels seen: {reduces if reduces else 'none'}")
    print(
        f"  M whose rows are bitwise the first M rows of the M={n_max} call: {len(same)} of {n_max}"
    )
    print(f"  M whose rows differ in bits from that call: {_ranges(other) if other else 'none'}")
    print("  band edges: " + " ".join(f"{lo}..{hi}" for lo, hi, _ in bands))
    assert not bad, f"{len(bad)} row counts outside the gate, first {bad[:10]}"
    assert not bf16_partials, f"bf16-partial split-K at M {_ranges(bf16_partials)}"
    if dtype != torch.float32:
        assert not low, f"bit-exact fraction below {_CORE_MIN_EXACT} at M {_ranges(low)}"
    if cell in _PREFIX_BITWISE_CELLS:
        assert not other, f"rows differ from the N_max call at M {_ranges(other)}"

    # Control: the gate rejects a kernel that dropped the last 16 of the K
    # products -- the reference without them is outside it at M = N_max.
    if dtype == torch.float32:
        wrong = ref - (mat_a[:, -16:].double() @ mat_b[-16:, :].double()).float()
        miss = float((out - wrong).abs().max())
        print(f"  control (last 16 K products dropped): max |op - wrong| {miss:.3g}, gate {gate:g}")
        assert miss > 4 * gate
    else:
        wrong = (ref.float() - mat_a[:, -16:].float() @ mat_b[-16:, :].float()).to(dtype)
        outside = int(_bf16_violations(out, wrong, _DSV41_BF16_ATOL[k])) / out.numel()
        print(f"  control (last 16 K products dropped): {outside:.4f} of elements outside")
        assert outside > 0.01
        # Control for the bit-exact floor: a bf16-partial split-K emulated at
        # M = N_max -- the K sum split into two halves, each partial rounded to
        # bf16, then summed and rounded -- lands below the floor.
        h = k // 2
        with true_fp32_matmul():
            p1 = (mat_a[:, :h].float() @ mat_b[:h].float()).to(dtype)
            p2 = (mat_a[:, h:].float() @ mat_b[h:].float()).to(dtype)
        emu = (p1.float() + p2.float()).to(dtype)
        emu_exact = float((emu.view(torch.int16) == ref_bits).float().mean())
        print(f"  control (2-way split-K, bf16 partials): bit-exact {emu_exact:.4f}")
        assert emu_exact < _CORE_MIN_EXACT


@pytest.mark.parametrize("cell", list(_VIS_CELLS))
def test_dsv41_vision_every_row_count(cell: str) -> None:
    """Every M in 1..M_max at one vision call, held to the rounded-once float64
    reference; the patch embed's bf16-partial band is exactly where it was."""
    k, n, has_bias, m_max = _VIS_CELLS[cell]
    mat_a, mat_b, bias = _vis_operands(k, n, has_bias, m_max, 50 + list(_VIS_CELLS).index(cell))
    ref = _vis_ref(mat_a, mat_b, bias)
    ref_bits = ref.view(torch.int16)
    viol = torch.zeros(m_max + 1, device="cuda")
    exact = torch.zeros(m_max + 1, device="cuda")

    def record(m: int, out: torch.Tensor) -> None:
        viol[m] = _bf16_violations(out, ref[:m])
        exact[m] = (out.view(torch.int16) == ref_bits[:m]).sum()

    kernels = _sweep(mat_a, mat_b, bias, list(range(m_max, 0, -1)), record)
    bands = _bands(kernels)
    viol, exact = viol.cpu(), exact.cpu()
    frac = {m: float(exact[m]) / (m * n) for m in range(1, m_max + 1)}
    bf16_partials = sorted(
        m for m, names in kernels.items() if any(_BF16_PARTIALS in x for x in names)
    )
    distinct = {name for names in kernels.values() for name in names}
    print(
        f"\n[vision {cell} K={k} N={n} bias={has_bias}] every M 1..{m_max}: "
        f"{len(bands)} kernel bands, {len(distinct)} distinct kernels; "
        f"M with bf16-partial split-K: {_ranges(bf16_partials) if bf16_partials else 'none'}"
    )
    bad_rows = set(_PATCH_BAD_ROWS) if cell == "patch_embed" else set()
    assert set(bf16_partials) == bad_rows, "the bf16-partial split-K band moved"
    good = [m for m in range(1, m_max + 1) if m not in bad_rows]
    worst_good = min(frac[m] for m in good)
    print(
        f"  outside that band: {sum(int(viol[m]) for m in good)} elements outside the bf16 gate, "
        f"min bit-exact {worst_good:.4f}"
    )
    assert all(viol[m] == 0 for m in good), [(m, int(viol[m])) for m in good if viol[m]][:10]
    assert worst_good >= _VIS_MIN_EXACT, worst_good
    if bad_rows:
        # The excursion `is_valid` refuses: every M in the band leaves the gate
        # and falls far below the bit-exact floor.
        worst_bad = max(frac[m] for m in bad_rows)
        out_bad = min(int(viol[m]) / (m * n) for m in bad_rows)
        print(
            f"  M {min(bad_rows)}..{max(bad_rows)}: bit-exact at most {worst_bad:.4f}, "
            f"at least {out_bad:.4f} of elements outside the bf16 gate at every M"
        )
        assert worst_bad < 0.9 and out_bad > 1e-3


# The prototype's whole-tower CUDA graph ran at the golden cases' sizes:
# P = 1517, 4680, 5325, 8772 (C = 182, 546, 601, 986); the MMMU maxima
# (P 17037, C 1902) are captured too.
_VIS_GRAPH_ROWS = {"vit": (1517, 4680, 5325, 8772, 17037), "aligner": (182, 546, 601, 986, 1902)}


@pytest.mark.parametrize("cell", list(_VIS_CELLS))
def test_dsv41_vision_cuda_graph_replay(cell: str) -> None:
    """Each vision call captured at the prototype's sizes; replay on new data is
    bitwise equal to eager and in the gate."""
    k, n, has_bias, _ = _VIS_CELLS[cell]
    rows = _VIS_GRAPH_ROWS["aligner" if cell.startswith("aligner") else "vit"]
    _, mat_b, bias = _vis_operands(k, n, has_bias, 1, 70 + list(_VIS_CELLS).index(cell))
    for m in rows:
        mat_a = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(mat_a, mat_b, bias)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(mat_a, mat_b, bias)
        previous = None
        for _ in range(2):
            mat_a.copy_(torch.randn(m, k, device="cuda").to(torch.bfloat16))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, op(mat_a, mat_b, bias)), (cell, m, "replay != eager")
            assert int(_bf16_violations(out, _vis_ref(mat_a, mat_b, bias))) == 0, (cell, m)
            assert previous is None or not torch.equal(out, previous), (cell, m, "stale replay")
            previous = out.clone()
        del graph, out
