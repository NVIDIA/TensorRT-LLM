# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from einops import rearrange, repeat

from utils.torch_ref import selective_state_update_ref  # noqa: F401 (pure-Python reference)

from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import \
    replay_selective_state_update
from tensorrt_llm._torch.modules.mamba.selective_state_update import \
    selective_state_update
from tensorrt_llm._torch.modules.mamba.softplus import softplus as softplus_fn

# ---------------------------------------------------------------------------
# TODO: Add a test that exercises the full conv1d → precompute → main PDL chain
# (external + internal PDL) to catch data-race bugs from chained PDL ordering.
# The current tests only exercise the incremental kernel in isolation (no conv1d
# predecessor), so they cannot detect bugs where main kernel reads conv1d
# outputs before conv1d completes.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configs derived from NVIDIA-Nemotron-3-Super-120B-A12B Mamba2 parameters
# (nheads=128, headdim=64, d_state=128, ngroups=8) with TP split applied:
#   TP=8: nheads=16, ngroups=1   — primary production config
#   TP=4: nheads=32, ngroups=2   — exercises ngroups>1 (grouped B/C path)
# ---------------------------------------------------------------------------
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),   # TP=8 production config
    (32, 64, 128, 2),   # TP=4, ngroups>1 (more heads than B/C groups)
]


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("paged_cache", [False, True],
                         ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize("T", [6, 10, 16, 27, 32, 55],
                         ids=["T6", "T10", "T16", "T27", "T32", "T55"])
def test_replay_selective_state_update(nheads, head_dim, d_state, ngroups,
                                            state_dtype, paged_cache, T):
    """
    Verify that:
      replay_selective_state_update(state0, old_caches, k, new_x, ...)
    produces the same output as:
      selective_state_update(state_after_k_old_tokens, new_x, ...)
    and writes state_after_k_old_tokens back to the state tensor.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16   # input activations are bf16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device,
                                           dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, head_dim, d_state) with stride(-2)=0, stride(-1)=0  [tie_hdim]
    A_base = -torch.rand(nheads, device=device) - 0.5   # float32, negative
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, head_dim) with stride(-1)=0  [tie_hdim]
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial SSM state (cache_size slots)
    state0 = torch.randn(cache_size, nheads, head_dim, d_state,
                         device=device, dtype=state_dtype)

    # Old inputs: T tokens per batch request
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)   # stride(-1)=0
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # -------------------------------------------------------------------
    # Capture intermediate SSM states using selective_state_update.
    # -------------------------------------------------------------------
    states_buffer_f32 = torch.zeros(cache_size, T, nheads, head_dim, d_state,
                                    device=device, dtype=torch.float32)
    cache_idx_for_capture = (state_batch_indices if paged_cache else
                             torch.arange(batch, device=device, dtype=torch.int32))
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1, dt1, A, B1, C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )

    # -------------------------------------------------------------------
    # Build cache tensors for the incremental kernel.
    # old_x: (cache, T, nheads, dim) bf16 — single-buffered
    # old_B: (cache, 2, T, ngroups, dstate) bf16 — double-buffered
    # old_dt_proc: (cache, 2, nheads, T) fp32 — double-buffered, T contiguous
    # old_cumAdt: (cache, 2, nheads, T) fp32 — double-buffered, T contiguous
    # cache_buf_idx: random 0s and 1s to verify indexing correctness
    # -------------------------------------------------------------------
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Fill each slot's READ buffer (indexed by cache_buf_idx) with step 1's data.
    # The OTHER buffer has random garbage to catch indexing bugs.
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots] = x1

    # Compute processed dt and cumAdt for step 1
    dt1_proc = dt1_base.float() + dt_bias_base.float()[None, None, :]
    dt1_proc = torch.where(dt1_proc > 20.0, dt1_proc, torch.log1p(torch.exp(dt1_proc)))
    cumAdt1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)

    # Write to each slot's read buffer based on its cache_buf_idx
    slot_indices = (state_batch_indices.tolist() if paged_cache
                    else list(range(cache_size)))
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        batch_idx = i  # maps slot back to the batch index
        old_B[slot, buf] = B1[batch_idx]
        old_dt_proc[slot, buf] = dt1_proc[batch_idx].T  # (T, nheads) → (nheads, T)
        old_cumAdt[slot, buf] = cumAdt1[batch_idx].T    # (T, nheads) → (nheads, T)

    # -------------------------------------------------------------------
    # Main loop: test each k (number of old tokens replayed)
    # -------------------------------------------------------------------
    for k in range(T + 1):
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        # Reference
        ref_state_f32 = state0.float().clone()
        if k > 0:
            ref_state_f32[slots] = states_buffer_f32[slots, k - 1]

        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state_f32,
            x2, dt2, A, B2, C2,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=(state_batch_indices if paged_cache else None),
            out=ref_out,
        )

        # Incremental kernel
        test_state = state0.clone()
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        # cache_buf_idx stays at its random values — each slot reads from its own buffer

        replay_selective_state_update(
            test_state,
            old_x.clone(),
            old_B.clone(),
            old_dt_proc.clone(),
            old_cumAdt.clone(),
            cache_buf_idx.clone(),
            prev_tokens,
            x=x2, dt=dt2, A=A, B=B2, C=C2,
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
        )

        # ---------------------------------------------------------------
        # Precision analysis: incremental kernel vs baselines
        #
        # CONTEXT: The incremental kernel uses bf16 tl.dot (tensor cores,
        # fp32 accumulator) for four key matmuls.  The reference kernel
        # used in this test (selective_state_update) and the flashinfer
        # baseline both use fp32 element-wise multiply-accumulate with
        # no tensor cores.  This section documents exactly where and why
        # precision differs, so that tolerance choices are justified.
        #
        # PREFILL EQUIVALENCE: The prefill path (ssd_chunk_scan) performs
        # identical bf16 tl.dot operations with the same fp32→bf16 input
        # casts (see ssd_chunk_state.py:353, ssd_chunk_scan.py:502-554,
        # ssd_bmm.py:258).  Our kernel matches prefill precision exactly.
        # The baseline decode kernels are actually MORE precise than
        # prefill, not the other way around.
        #
        # INPUT DTYPES: All varying inputs (x, B, C, dt) are bf16 from
        # in_proj/conv1d output.  Fixed parameters (A, dt_bias, D) are
        # fp32.  State can be fp32 or bf16.
        #
        # WHERE PRECISION DIFFERS (our kernel vs baseline):
        #
        # 1. State update (replay): dB_scaled = coeff * B, then
        #    state += tl.dot(old_x.bf16, dB_scaled.bf16).
        #    The .to(bf16) cast on dB_scaled is round-to-nearest-even
        #    (not truncation — NVIDIA hardware default).  This loses the
        #    dt_bias-derived lower mantissa bits that fp32 dB_scaled
        #    carried: dt_proc = softplus(dt_bf16 + dt_bias_fp32) has
        #    more than 7 mantissa bits from dt_bias, and A_fp32
        #    contributes full precision to coeff = exp(cumAdt).  The
        #    product coeff * B is fp32 with ~23 stored bits, but B is
        #    bf16 (7 bits), so the extra precision is a precise scaling
        #    of a coarse value.  The bf16 cast drops those lower bits.
        #    The baseline keeps dB in fp32 and multiplies by x (also
        #    bf16→fp32) element-wise, preserving the dt_bias/A bits
        #    through the multiply.
        #
        # 2. Output (C @ state^T): state is fp32, cast to bf16 for
        #    tl.dot.  Same round-to-nearest-even.  Baseline does
        #    element-wise fp32 multiply of state * C, both loaded as
        #    fp32, then fp32 sum-reduce.
        #
        # 3. Output (CB_scaled @ x): CB_scaled is fp32 (from precompute),
        #    cast to bf16 for tl.dot.  x is bf16.  Baseline computes
        #    this implicitly per-token in fp32.
        #
        # 4. Decay (state *= exp(cumAdt)): Identical precision in both
        #    kernels.  A is fp32, cumAdt is fp32 cumsum, exp is fp32,
        #    multiply is fp32.  No bf16 involvement.  Our single-multiply
        #    approach (cumsum then exp) may be marginally more precise
        #    than the baseline's sequential multiply (exp then multiply
        #    repeatedly) for large k.
        #
        # ERROR CHARACTERISTICS:
        # - Max absolute error: ~1.0 for T≤16, ~2.0 for T=32-55.
        #   Does NOT grow meaningfully with T — errors are per-element
        #   bf16 rounding, not accumulating.  The sqrt(k) growth from
        #   summing random-sign errors over dstate=128 dot products
        #   explains the ~2.0 at larger T.
        # - Mean absolute error: ~0.014 (tiny), independent of T.
        # - Affected elements: <0.02% exceed 0.5 absolute error.
        # - Identical for fp32 and bf16 state: the error comes from
        #   bf16 casts of tl.dot INPUTS, not from state storage dtype.
        #   For bf16 state, the baseline's fp32 intermediate advantage
        #   is erased by state→bf16 store truncation at each step.
        #
        # FUTURE PRECISION OPTIONS (if needed):
        # - TF32 tl.dot: pass fp32 inputs instead of casting to bf16.
        #   Triton uses TF32 tensor cores (10 mantissa bits vs bf16's
        #   7) on Ampere+/Blackwell.  ~8x reduction in per-element
        #   rounding error.  Moderate speed cost.  Could apply
        #   selectively to replay tl.dot (where state precision matters
        #   across steps) while keeping bf16 for output tl.dots.
        # - Philox stochastic rounding for state store: required for
        #   bf16 state in production to avoid systematic rounding bias
        #   over many decode steps.  FlashInfer already implements
        #   this (rand_seed/philox_rounds kwargs).  Orthogonal to the
        #   tl.dot input precision.
        # - Philox for tl.dot input casts: stochastically round
        #   dB_scaled/state/CB_scaled to bf16 before tl.dot.  Unusual
        #   but would reduce systematic dot-product bias.
        # ---------------------------------------------------------------
        torch.testing.assert_close(test_out, ref_out, rtol=2e-2, atol=1.0,
                                   msg=f"Output mismatch at k={k}")

        expected_state = (state0[slots] if k == 0
                          else states_buffer_f32[slots, k - 1].to(state_dtype))
        torch.testing.assert_close(test_state[slots], expected_state,
                                   rtol=2e-2, atol=1.0,
                                   msg=f"State mismatch at k={k}")


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("paged_cache", [False, True],
                         ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
def test_replay_selective_state_update_philox(nheads, head_dim, d_state,
                                                    ngroups, paged_cache, T):
    """
    Verify that Philox stochastic rounding produces correct results.

    Runs our kernel twice with identical inputs: once without rounding
    (fp16 state, deterministic), once with rounding (fp16 state, Philox).
    The outputs should be nearly identical — stochastic rounding only
    perturbs the state by ±1 fp16 ULP, which barely affects output.
    Also verifies the state dtype remains fp16.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device,
                                           dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(cache_size, nheads, head_dim, d_state,
                         device=device, dtype=state_dtype)

    # Cache tensors
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(cache_size, 2, nheads, T, device=device,
                               dtype=torch.float32)
    old_cumAdt = torch.randn(cache_size, 2, nheads, T, device=device,
                              dtype=torch.float32)
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # New token inputs
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((cache_size,), T // 2, device=device,
                             dtype=torch.int32)

    common_kwargs = dict(
        x=x, dt=dt, A=A, B=B, C=C,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=state_batch_indices,
    )

    # --- Run without rounding (deterministic fp16 state store) ---
    state_nornd = state0.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_nornd,
        old_x.clone(), old_B.clone(), old_dt_proc.clone(), old_cumAdt.clone(),
        cache_buf_idx.clone(), prev_tokens,
        out=out_nornd, **common_kwargs,
    )

    # --- Run with Philox rounding ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rnd = state0.clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rnd,
        old_x.clone(), old_B.clone(), old_dt_proc.clone(), old_cumAdt.clone(),
        cache_buf_idx.clone(), prev_tokens,
        out=out_rnd, rand_seed=rand_seed, philox_rounds=10,
        **common_kwargs,
    )

    # Outputs should be nearly identical — rounding only perturbs the
    # post-replay state by ±1 ULP before the output phase reads it.
    torch.testing.assert_close(out_rnd, out_nornd, rtol=2e-2, atol=1.0,
                               msg="Output diverged with Philox rounding")

    # State should remain fp16
    assert state_rnd.dtype == torch.float16

    # States should differ by at most 1 fp16 ULP per element.
    # fp16 ULP depends on magnitude: up to 0.5 for values near 512.
    # Use rtol to account for magnitude-dependent ULP.
    slots = state_batch_indices if paged_cache else slice(None)
    torch.testing.assert_close(state_rnd[slots], state_nornd[slots],
                               rtol=2e-3, atol=0.2,
                               msg="State diverged with Philox rounding")


def test_philox_rounding_unbiased():
    """
    Verify that Philox stochastic rounding is unbiased.

    Runs the incremental kernel with fp32 state (capturing the true fp32
    post-replay state) and with fp16 state + Philox rounding.  Compares the
    rounding residual (fp16_state.float() - fp32_state) against deterministic
    rounding (fp32_state.to(fp16).float() - fp32_state).

    Deterministic round-to-nearest-even has a systematic positive bias on
    the residual.  Philox stochastic rounding should be unbiased: the mean
    residual should be near zero.

    Uses a large batch (16) for ~2M state elements — plenty of statistics.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 16, 6
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Use fp32 initial state so replay produces non-fp16-representable values
    state0 = torch.randn(batch, nheads, head_dim, d_state,
                         device=device, dtype=torch.float32)

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(batch, 2, nheads, T, device=device,
                               dtype=torch.float32)
    old_cumAdt = torch.randn(batch, 2, nheads, T, device=device,
                              dtype=torch.float32)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt_val = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((batch,), T, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x, dt=dt_val, A=A, B=B, C=C,
        D=D, dt_bias=dt_bias, dt_softplus=True,
    )

    # 1. fp32 state — captures true post-replay state
    state_fp32 = state0.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_fp32,
        old_x.clone(), old_B.clone(), old_dt_proc.clone(), old_cumAdt.clone(),
        cache_buf_idx.clone(), prev_tokens,
        out=out_fp32, **common_kwargs,
    )

    # 2. fp16 state with Philox rounding
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    state_rnd = state0.to(torch.float16).clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rnd,
        old_x.clone(), old_B.clone(), old_dt_proc.clone(), old_cumAdt.clone(),
        cache_buf_idx.clone(), prev_tokens,
        out=out_rnd, rand_seed=rand_seed, philox_rounds=10,
        **common_kwargs,
    )

    # Compute rounding residuals where fp32 state has non-zero values
    fp32_vals = state_fp32.flatten()
    stochastic_residual = state_rnd.float().flatten() - fp32_vals
    deterministic_residual = fp32_vals.to(torch.float16).float() - fp32_vals

    # Only consider elements where rounding matters (non-zero residual possible)
    nonzero_mask = deterministic_residual.abs() > 0
    n_nonzero = nonzero_mask.sum().item()
    assert n_nonzero > 1000, f"Too few roundable elements: {n_nonzero}"

    stoch_mean = stochastic_residual[nonzero_mask].mean().item()
    determ_mean = deterministic_residual[nonzero_mask].mean().item()

    # Stochastic rounding should be less biased than deterministic.
    # With ~millions of elements, the stochastic mean should be very close to 0.
    # Deterministic round-to-nearest-even has a small but systematic bias.
    assert abs(stoch_mean) < abs(determ_mean) or abs(stoch_mean) < 1e-5, (
        f"Stochastic rounding appears biased: stoch_mean={stoch_mean:.6f}, "
        f"determ_mean={determ_mean:.6f}, n_elements={n_nonzero}"
    )
