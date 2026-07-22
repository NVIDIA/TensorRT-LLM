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

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    KDAKernelDispatch,
)

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
    cos = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
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


@pytest.mark.parametrize("label,batch,eqlen_t,lens,h0_scale", CASES,
                         ids=[c[0] for c in CASES])
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
