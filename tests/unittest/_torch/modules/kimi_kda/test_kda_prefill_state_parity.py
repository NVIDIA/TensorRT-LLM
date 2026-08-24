# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State parity tests: optimized KDA prefill dispatch vs FLA, kernel level.

Complements test_kda_prefill_op.py (module outputs, no cache) with the cases
the executor runtime actually exercises through KDAKernelDispatch:

  * final_state parity in the pool's V-first [N, H, V, K] layout — K == V ==
    128 for Kimi K3, so a layout mix-up is invisible to shape checks and only
    caught numerically;
  * a LARGE carried initial_state — the K4 sign error fixed by dev-tech
    commit e45ae259 (NV = U - W@S, was U + W@S) is masked by tiny/zero
    initial states and random inputs, and only blows up when a real-magnitude
    state is carried in;
  * non-64-aligned varlen (single-seq pad path and multi-seq masked path);
  * fresh input tensors on every call — the runtime never reuses tensor
    objects, so this exercises per-call wrapper rebuild instead of the
    benchmark-style stable-object path, and would catch both stale-cache
    corruption and per-call pinning leaks.
"""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402

NUM_HEADS = 96
HEAD_K_DIM = 128
LOWER_BOUND = -5.0


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


@pytest.fixture(scope="module")
def dispatch_pair():
    optimized = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    assert optimized.prefill_kernel_path == "optimized"
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    assert reference.prefill_kernel_path == "fla"
    return optimized, reference


@pytest.fixture(scope="module")
def gate_params():
    torch.manual_seed(0)
    a_log = torch.randn(NUM_HEADS, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = torch.randn(NUM_HEADS * HEAD_K_DIM, dtype=torch.float32, device="cuda") * 0.1
    return a_log, dt_bias


def _make_inputs(batch: int, total_t: int, h0_scale, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, k = NUM_HEADS, HEAD_K_DIM

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (
            torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda").to(dtype) * scale
        )

    q = rnd(batch, total_t, h, k)
    key = rnd(batch, total_t, h, k)
    v = rnd(batch, total_t, h, k)
    g = rnd(batch, total_t, h, k)
    beta = rnd(batch, total_t, h, dtype=torch.float32)
    h0 = None
    if h0_scale is not None:
        # Pool V-first [N, H, V, K] layout, deliberately non-symmetric in the
        # last two dims (row-index modulation) so a [K,V]<->[V,K] transpose
        # mix-up cannot cancel out.
        h0 = rnd(batch, h, k, k, dtype=torch.float32, scale=h0_scale)
        h0 = h0 * torch.linspace(0.5, 1.5, k, device="cuda").view(1, 1, k, 1)
    return q, key, v, g, beta, h0


def _run(dispatch, gate_params, q, k, v, g, beta, h0, cu):
    a_log, dt_bias = gate_params
    return dispatch.prefill_chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_K_DIM**-0.5,
        initial_state=h0.clone() if h0 is not None else None,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu.clone() if cu is not None else None,
    )


def _assert_close(name, actual, expected):
    actual, expected = actual.float(), expected.float()
    cos = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    rel = ((actual - expected).norm() / (expected.norm() + 1e-12)).item()
    assert cos > 0.999 and rel < 3e-2, f"{name}: cos={cos:.6f} rel_l2={rel:.3e}"


CASES = [
    # (label, batch, eqlen T or None, varlen seq lens or None, h0_scale)
    ("eqlen_b2_t256_no_state", 2, 256, None, None),
    ("eqlen_b2_t256_large_state", 2, 256, None, 1.0),
    ("eqlen_b1_t300_pad_large_state", 1, 300, None, 1.0),
    ("varlen_aligned_large_state", 1, None, [128, 256, 192], 1.0),
    ("varlen_nonaligned_large_state", 1, None, [100, 257, 64], 1.0),
    ("varlen_single_nonaligned_large_state", 1, None, [300], 1.0),
]


@pytest.mark.parametrize("label,batch,eqlen_t,lens,h0_scale", CASES, ids=[c[0] for c in CASES])
@torch.no_grad()
def test_state_parity(dispatch_pair, gate_params, label, batch, eqlen_t, lens, h0_scale):
    optimized, reference = dispatch_pair
    if lens is not None:
        total_t = sum(lens)
        cu = torch.tensor(
            [0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.long, device="cuda"
        )
        n_seqs = len(lens)
    else:
        total_t, cu, n_seqs = eqlen_t, None, batch
    case_idx = [c[0] for c in CASES].index(label)
    q, k, v, g, beta, h0 = _make_inputs(
        1 if lens is not None else batch, total_t, h0_scale, seed=100 + case_idx
    )
    if h0 is not None and lens is not None:
        h0 = h0[:1].expand(n_seqs, -1, -1, -1).contiguous() * torch.linspace(
            0.5, 1.5, n_seqs, device="cuda"
        ).view(n_seqs, 1, 1, 1)

    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, h0, cu)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, h0, cu)
    _assert_close(f"{label}/out", out_opt, out_ref)
    _assert_close(f"{label}/state", state_opt, state_ref)


@torch.no_grad()
def test_fresh_tensors_every_call(dispatch_pair, gate_params):
    """Same shape, new tensor objects + contents per call. Catches stale
    id-keyed cache hits (wrong data); a persistently climbing allocation
    here would indicate per-call activation pinning."""
    optimized, reference = dispatch_pair
    for i in range(8):
        q, k, v, g, beta, h0 = _make_inputs(2, 256, 1.0, seed=1000 + i)
        out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, h0, None)
        out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, h0, None)
        _assert_close(f"iter{i}/out", out_opt, out_ref)
        _assert_close(f"iter{i}/state", state_opt, state_ref)


# Eval-scale coverage (2026-07-23): GSM8K on the optimized prefill scored
# 68.99 vs 97.01 on the FLA path while every existing unit test passed —
# the gap is the eval regime: varlen batches packing dozens of
# mixed-length sequences, and chunked prefill chaining a sequence's state
# across calls. These tests parity-check exactly those two regimes.

_EVAL_SCALE_LENS = [(97 * (i + 3)) % 911 + 45 for i in range(24)]


@pytest.mark.parametrize("h0_scale", [None, 1.0], ids=["no_state", "state"])
@torch.no_grad()
def test_eval_scale_packed_varlen_parity(dispatch_pair, gate_params, h0_scale):
    """24 packed mixed-length sequences (45..955 tokens), like one
    max_num_tokens=8192 eval context batch."""
    optimized, reference = dispatch_pair
    lens = _EVAL_SCALE_LENS
    n_seqs, total_t = len(lens), sum(lens)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.long, device="cuda"
    )
    q, k, v, g, beta, h0 = _make_inputs(1, total_t, h0_scale, seed=4242)
    if h0 is not None:
        h0 = h0[:1].expand(n_seqs, -1, -1, -1).contiguous() * torch.linspace(
            0.5, 1.5, n_seqs, device="cuda"
        ).view(n_seqs, 1, 1, 1)
    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, h0, cu)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, h0, cu)
    _assert_close("evalscale/out", out_opt, out_ref)
    _assert_close("evalscale/state", state_opt, state_ref)


# GSM8K-shaped coverage (2026-07-23, second regression pass): the stream
# fix lifted GSM8K 68.99 -> 77.79, still below the 97.01 FLA control, and
# the partial-score trajectory declined with completion length — the
# residual defect skews toward LONG sequences (8-shot GSM8K prompts are
# ~1-1.2k tokens = 16-20 chunks/seq; the eval-scale test above caps at
# 955). These cases pin per-seq NT at 16-20. The weak-gate variants
# (g_scale=0.05) reduce the per-chunk decay so state accumulates across
# many chunks — accumulation-order/precision defects that strong random
# gates (effectively local memory) cannot surface.

_LONG_MIXED_LENS = [1150, 1200, 980, 1100, 1279, 1024, 1216, 1090]


@pytest.mark.parametrize("g_scale", [1.0, 0.05], ids=["g_normal", "g_weak"])
@pytest.mark.parametrize(
    "lens,h0_scale",
    [([1150], None), ([1150], 1.0), (_LONG_MIXED_LENS, None), (_LONG_MIXED_LENS, 1.0)],
    ids=["single_long", "single_long_state", "mixed_long", "mixed_long_state"],
)
@torch.no_grad()
def test_long_sequence_parity(dispatch_pair, gate_params, lens, h0_scale, g_scale):
    optimized, reference = dispatch_pair
    n_seqs, total_t = len(lens), sum(lens)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.long, device="cuda"
    )
    q, k, v, g, beta, h0 = _make_inputs(1, total_t, h0_scale, seed=9000 + len(lens))
    g = (g.float() * g_scale).to(g.dtype)
    if h0 is not None:
        h0 = h0[:1].expand(n_seqs, -1, -1, -1).contiguous() * torch.linspace(
            0.5, 1.5, n_seqs, device="cuda"
        ).view(n_seqs, 1, 1, 1)
    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, h0, cu)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, h0, cu)
    _assert_close("long/out", out_opt, out_ref)
    _assert_close("long/state", state_opt, state_ref)


# Poisoned-scratch FINAL-STATE parity (2026-07-23, review follow-up on the
# fused_k123 beta-guard lines ~1520): the store guard (t < ci_eos) leaves
# k_scaled/kg/q_scaled scratch rows past each seq's end UNWRITTEN for
# partial final chunks. K4 is supposed to neutralize those rows via its
# per-seq bounded TMA (token extent = cu[s+1], hardware zero-fill); these
# cases PROVE that with garbage actually present in the stale rows: a
# preceding call with the SAME buffer-cache key (same B/T/NT/n_seqs) and
# huge-magnitude inputs fills the scratch, then the victim call's final
# state is asserted against FLA. Every eval prompt ends in a partial
# chunk and the state feeds all decode steps, so state corruption here is
# invisible to output-only checks.
#
# Victim/poison pairs share (T_total, NT_total, n_seqs) so they hit the
# same _buf_cache entry; the poison split is chosen so its guarded stores
# leave different rows written than the victim's layout expects.

_POISON_STATE_CASES = [
    # (victim lens, poison lens) — equal sum and equal total chunk count
    ([8191, 1], [8127, 65]),
    ([8000, 150, 42], [7999, 129, 64]),
    ([1186, 30], [1150, 66]),
]


def _nt(lens):
    return sum((n + 63) // 64 for n in lens)


@pytest.mark.parametrize(
    "victim,poison", _POISON_STATE_CASES, ids=["v8191_1", "v8000_150_42", "v1186_30"]
)
@torch.no_grad()
def test_final_state_parity_poisoned_scratch(dispatch_pair, gate_params, victim, poison):
    optimized, reference = dispatch_pair
    assert (
        sum(victim) == sum(poison) and _nt(victim) == _nt(poison) and len(victim) == len(poison)
    ), "pairs must share the buffer-cache key"
    total_t = sum(victim)

    # Poison pass: same buffer-cache key, huge magnitudes.
    q, k, v, g, beta, _ = _make_inputs(1, total_t, None, seed=31)
    cu_p = torch.tensor(
        [0] + torch.cumsum(torch.tensor(poison), 0).tolist(), dtype=torch.long, device="cuda"
    )
    _run(optimized, gate_params, q * 100, k * 100, v * 1000, g, beta * 10, None, cu_p)
    torch.cuda.synchronize()

    # Victim pass: fresh inputs, partial final chunks per seq.
    q, k, v, g, beta, h0 = _make_inputs(1, total_t, 1.0, seed=32)
    n_seqs = len(victim)
    h0 = h0[:1].expand(n_seqs, -1, -1, -1).contiguous() * torch.linspace(
        0.5, 1.5, n_seqs, device="cuda"
    ).view(n_seqs, 1, 1, 1)
    cu_v = torch.tensor(
        [0] + torch.cumsum(torch.tensor(victim), 0).tolist(), dtype=torch.long, device="cuda"
    )
    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, h0, cu_v)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, h0, cu_v)
    _assert_close("poisoned/out", out_opt, out_ref)
    _assert_close("poisoned/state", state_opt, state_ref)


def _run_headcount_case(dispatch_pair, heads):
    optimized, reference = dispatch_pair
    h, k = heads, HEAD_K_DIM
    gen = torch.Generator(device="cuda").manual_seed(4711 + heads)
    a_log = torch.randn(h, generator=gen, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = torch.randn(h * k, generator=gen, dtype=torch.float32, device="cuda") * 0.1
    lens = [1150, 731, 1024, 987]
    total_t = sum(lens)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.long, device="cuda"
    )

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda").to(dtype)

    q = rnd(1, total_t, h, k)
    key = rnd(1, total_t, h, k)
    v = rnd(1, total_t, h, k)
    g = rnd(1, total_t, h, k)
    beta = rnd(1, total_t, h, dtype=torch.float32)
    h0 = rnd(len(lens), h, k, k, dtype=torch.float32) * torch.linspace(
        0.5, 1.5, k, device="cuda"
    ).view(1, 1, k, 1)

    def run(dispatch):
        return dispatch.prefill_chunk_kda(
            q=q.clone(),
            k=key.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=a_log,
            dt_bias=dt_bias,
            scale=k**-0.5,
            initial_state=h0.clone(),
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            cu_seqlens=cu.clone(),
        )

    out_opt, state_opt = run(optimized)
    out_ref, state_ref = run(reference)
    _assert_close(f"h{heads}/out", out_opt, out_ref)
    _assert_close(f"h{heads}/state", state_opt, state_ref)


@pytest.mark.parametrize("heads", [6, 12], ids=["tp16_heads6", "tp8_heads12"])
@torch.no_grad()
def test_long_sequence_parity_tp_headcount(dispatch_pair, heads):
    """Per-rank head count under tensor parallelism: the e2e eval runs
    tp_size=16 -> H = 96/16 = 6 heads per rank, but every other test here
    compiles the DSL kernels at H=96 (H is a compile-time specializer).
    Parity-check the H=6 / H=12 compiles on a GSM8K-shaped batch."""
    _run_headcount_case(dispatch_pair, heads)


@torch.no_grad()
def test_headcount_recompile_parity(dispatch_pair):
    """Cross-head-count compile-cache isolation, order-explicit: run two
    different per-rank head counts back-to-back in the same process and
    parity-check the SECOND one. Regression test for the K4 persistent
    kernel cache key losing H (0e44bf64a6 follow-up): s_ct bakes the
    [H, K, V] state shape/strides at compile time, so reusing the
    first-compiled head count's kernel for a different H misaddresses
    every (seq, head) state tile. The tp_headcount params above only catch
    this via pytest execution order; this test pins the order even when
    run in isolation. Heads 8 then 4 avoid cache hits from other tests."""
    _run_headcount_case(dispatch_pair, 8)
    _run_headcount_case(dispatch_pair, 4)


@torch.no_grad()
def test_chunked_continuation_parity(dispatch_pair, gate_params):
    """Chunked prefill: run [3000] as [2048] then [952] with the first
    call's final state carried into the second (what the runtime's
    chunked-prefill continuation does through the pool), and compare the
    second chunk's output and final state against a single full-sequence
    reference run."""
    optimized, reference = dispatch_pair
    total_t, split = 3000, 2048
    q, k, v, g, beta, _ = _make_inputs(1, total_t, None, seed=777)
    cu_full = torch.tensor([0, total_t], dtype=torch.long, device="cuda")
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, None, cu_full)

    cu1 = torch.tensor([0, split], dtype=torch.long, device="cuda")
    cu2 = torch.tensor([0, total_t - split], dtype=torch.long, device="cuda")
    sl1 = slice(0, split)
    sl2 = slice(split, total_t)
    out1, state1 = _run(
        optimized, gate_params, q[:, sl1], k[:, sl1], v[:, sl1], g[:, sl1], beta[:, sl1], None, cu1
    )
    out2, state2 = _run(
        optimized,
        gate_params,
        q[:, sl2],
        k[:, sl2],
        v[:, sl2],
        g[:, sl2],
        beta[:, sl2],
        state1,
        cu2,
    )
    _assert_close("chunked/out_chunk1", out1, out_ref[:, sl1])
    _assert_close("chunked/out_chunk2", out2, out_ref[:, sl2])
    _assert_close("chunked/state", state2, state_ref)
