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

import math

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import repeat

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    REPLAY_WORK_CACHE_BUF_IDX,
    REPLAY_WORK_CACHE_SLOT,
    REPLAY_WORK_ITEM_WIDTH,
    REPLAY_WORK_PNAT,
    REPLAY_WORK_POSITION_IN_DECODE_BATCH,
)
from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
    _stochastic_round_int8_packed,
    _stochastic_round_int16_packed,
    replay_selective_state_update,
)
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._utils import get_sm_version


def _make_replay_work_items(prev_tokens, cache_buf_idx, T, max_window, batch,
                            state_batch_indices, device, explicit_order=None):
    """Build the replay metadata consumed by persistent_main."""
    position_in_decode_batch = torch.arange(batch,
                                            device=device,
                                            dtype=torch.int32)
    if state_batch_indices is not None:
        cache_slot = state_batch_indices[:batch].to(torch.int32)
    else:
        cache_slot = position_in_decode_batch
    cache_slot_long = cache_slot.to(torch.long)
    pnat = prev_tokens[cache_slot_long].to(torch.int32)
    active_cache_buf_idx = cache_buf_idx[cache_slot_long].to(torch.int32)
    write_mask = (pnat + T) > max_window
    n_writes = write_mask.sum().to(torch.int32).reshape(1)

    if explicit_order is None:
        order = torch.argsort((~write_mask).to(torch.int32),
                              stable=True).to(torch.long)
    else:
        order = torch.tensor(explicit_order, device=device, dtype=torch.long)

    replay_work_items = torch.empty(batch,
                                    REPLAY_WORK_ITEM_WIDTH,
                                    device=device,
                                    dtype=torch.int32)
    replay_work_items[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = (
        position_in_decode_batch[order])
    replay_work_items[:, REPLAY_WORK_CACHE_SLOT] = cache_slot[order]
    replay_work_items[:, REPLAY_WORK_PNAT] = pnat[order]
    replay_work_items[:, REPLAY_WORK_CACHE_BUF_IDX] = active_cache_buf_idx[order]
    return n_writes, replay_work_items.contiguous()


# Philox stochastic rounding uses PTX cvt.rs.f16x2.f32 which requires sm >= 100.
_skip_pre_sm100 = pytest.mark.skipif(
    get_sm_version() < 100, reason="Philox stochastic rounding needs sm >= 100"
)

# Configs derived from NVIDIA-Nemotron-3-Super-120B-A12B Mamba2 parameters
# (nheads=128, headdim=64, d_state=128, ngroups=8) with TP split applied:
#   TP=8: nheads=16, ngroups=1   — primary production config
#   TP=4: nheads=32, ngroups=2   — exercises ngroups>1 (grouped B/C path)
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),  # TP=8 production config
    (32, 64, 128, 2),  # TP=4, ngroups>1 (more heads than B/C groups)
]

# Quantized state dtypes and their representable-magnitude limits (== QUANT_MAX
# in the kernel).  fp8_e4m3fn cells require SM 89+ for the fp32↔fp8 cvt PTX
# instructions; SR variants of fp16/fp8 additionally need SM 100+.
_QUANT_MAX_BY_DTYPE = {
    torch.int8: 127.0,
    torch.int16: 32767.0,
    torch.float8_e4m3fn: 448.0,
}


def _quantize_state(state_fp32: torch.Tensor, state_dtype: torch.dtype, quant_max: float):
    """Quantize fp32 state to (state_quant, decode_scale) using the same
    per-(head, dim) channel scheme the kernel does on store.  decode_scale =
    max_abs_per_channel / quant_max (= 1/encode_scale).
    """
    amax = state_fp32.abs().amax(dim=-1)  # (cache, nheads, head_dim)
    encode_scale = quant_max / amax.clamp(min=1e-30)
    decode_scale = 1.0 / encode_scale
    scaled = state_fp32 * encode_scale.unsqueeze(-1)
    if state_dtype == torch.float8_e4m3fn:
        # Native cast does RN at the fp8 grid; explicit round() would destroy
        # sub-integer precision (matches the kernel's fp8 RN path).
        state_quant = scaled.clamp(-quant_max, quant_max).to(state_dtype)
    else:
        state_quant = scaled.round().clamp(-quant_max, quant_max).to(state_dtype)
    return state_quant, decode_scale


def _dequantize_state(state_quant: torch.Tensor, decode_scale: torch.Tensor):
    return state_quant.to(torch.float32) * decode_scale.unsqueeze(-1)


def _maybe_skip_dtype(state_dtype, use_sr):
    """Skip on insufficient SM.  fp8 e4m3fn (any) needs SM 89+; fp16/fp8 SR
    needs SM 100+; int8/int16 (RN or SR) runs anywhere."""
    if state_dtype == torch.float8_e4m3fn and get_sm_version() < 89:
        pytest.skip("fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)")
    if use_sr and state_dtype in (torch.float16, torch.float8_e4m3fn) and get_sm_version() < 100:
        pytest.skip(f"{state_dtype} stochastic rounding requires SM 100+ (Blackwell B200+)")


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "state_dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.float8_e4m3fn,
    ],
    ids=["fp16", "bf16", "fp32", "int8", "int16", "fp8"],
)
@pytest.mark.parametrize("paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize(
    "T", [6, 10, 16, 27, 32, 55], ids=["T6", "T10", "T16", "T27", "T32", "T55"]
)
@pytest.mark.parametrize(
    "write_checkpoint,rectangle_for_nowrite",
    [
        (True, False),    # write path (rectangle_for_nowrite is ignored)
        (False, False),   # nowrite path via replay-style kernels
        (False, True),    # nowrite path via dedicated rectangle kernels
    ],
    ids=["write", "no_write_replay", "no_write_rectangle"],
)
@pytest.mark.parametrize(
    "mode",
    ["persistent_dynamic", "persistent_main"],
    ids=["persistent_dynamic", "persistent_main"],
)
def test_replay_selective_state_update(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T,
    write_checkpoint, rectangle_for_nowrite, mode,
):
    """
    Verify that:
      replay_selective_state_update(state0, old_caches, k, new_x, ...)
    produces the same output as:
      selective_state_update(state_after_k_old_tokens, new_x, ...)
    and writes state_after_k_old_tokens back to the state tensor.

    Quantized state dtypes (int8/int16/fp8) follow the same flow with
    a per-(head, dim) channel decode-scale tensor; comparison is done
    via dequant(state, scales) against the fp32 reference.
    """
    _maybe_skip_dtype(state_dtype, use_sr=False)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

    batch = 2
    device = "cuda"
    dtype = torch.bfloat16  # input activations are bf16
    assert nheads % ngroups == 0

    # Cache T-axis size (max_window).  Use the kernel's BLOCK_SIZE_T as the
    # ceiling — this is what the wrapper allows and enables PNAT-aware writes
    # at [PNAT, PNAT+T) for no-replay-write mode.  For T=6 that's 16 (production
    # max_window); for larger T it scales with np2(T).
    max_window = max(triton.next_power_of_2(T), 16)

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, head_dim, d_state) with stride(-2)=0, stride(-1)=0  [tie_hdim]
    A_base = -torch.rand(nheads, device=device) - 0.5  # float32, negative
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, head_dim) with stride(-1)=0  [tie_hdim]
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial SSM state (cache_size slots).  Quantized dtypes need a separate
    # init: derive scales from a fp32 source so the quantized state isn't
    # garbage on dequant.  ref_input_state is what the fp32 reference run
    # sees — for non-quant it's state0 (cast to fp32 inside reference); for
    # quant it's the lossy dequant of state0 (matches what the kernel sees
    # internally on load).
    if is_quantized:
        state0_fp32 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)
        ref_input_state = _dequantize_state(state0, state0_scales)
    else:
        state0 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state0_scales = None
        ref_input_state = state0.float()

    # Old inputs: up to `max_window` tokens per batch request, so the test
    # loop can probe PNAT > T-1 (which the prior T-token setup couldn't
    # reach).  step1_T = max_window covers the full PNAT range we sweep.
    step1_T = max_window
    x1 = torch.randn(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, step1_T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)  # stride(-1)=0
    B1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)

    # Capture intermediate SSM states using selective_state_update across
    # all step1_T positions — gives us reference states for k ∈ [0, step1_T].
    states_buffer_f32 = torch.zeros(
        cache_size, step1_T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    out1 = torch.zeros(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_input_state.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=step1_T,
        out=out1,
        disable_state_update=True,
    )

    # Build cache tensors for the replay kernel.
    # old_x: (cache, max_window, nheads, dim) bf16 — single-buffered
    # old_B: (cache, 2, max_window, ngroups, dstate) bf16 — double-buffered
    # old_dt: (cache, 2, nheads, max_window) fp32 — double-buffered, T contiguous
    # old_dA_cumsum: (cache, 2, nheads, max_window) fp32 — double-buffered, T contiguous
    # cache_buf_idx: random 0s and 1s to verify indexing correctness
    old_x = torch.zeros(cache_size, max_window, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, max_window, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(cache_size, 2, nheads, max_window, device=device, dtype=torch.float32)
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Fill each slot's active buffer (= cache_buf_idx) with step 1's data at
    # positions [0:step1_T) = [0:max_window).  Whole buffer covered so PNAT
    # values up to max_window are exercised.  Inactive buffer has random
    # garbage to catch indexing bugs.
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots, :step1_T] = x1

    # Compute processed dt and dA_cumsum for step 1
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1, dim=1)

    # Write to each slot's active buffer based on its cache_buf_idx
    slot_indices = state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        batch_idx = i  # maps slot back to the batch index
        old_B[slot, buf, :step1_T] = B1[batch_idx]
        old_dt[slot, buf, :, :step1_T] = dt1[batch_idx].T  # (step1_T, nheads) → (nheads, step1_T)
        old_dA_cumsum[slot, buf, :, :step1_T] = dA_cumsum1[batch_idx].T

    # Main loop: test each k (number of old tokens replayed).
    #   write_checkpoint=False (nowrite): k ∈ [0, max_window-T] — new tokens
    #     append at [k, k+T) of the active buffer; need k+T ≤ max_window.
    #   write_checkpoint=True  (write):   k ∈ [max_window-T+1, max_window] —
    #     new tokens land in the staging buffer at [0, T); k > max_window-T
    #     captures the overflow case that triggers a checkpoint write in production.
    # Combined sweep covers the full k ∈ [0, max_window] with the
    # appropriate boundary handling per mode.
    if write_checkpoint:
        k_lo = max(0, max_window - T + 1)
        k_hi = max_window + 1  # exclusive
    else:
        k_lo = 0
        k_hi = max_window - T + 1  # exclusive
    for k in range(k_lo, k_hi):
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        # Reference (fp32, starting from the same lossy-or-not state the
        # kernel sees).
        ref_state_f32 = ref_input_state.clone()
        if k > 0:
            ref_state_f32[slots] = states_buffer_f32[slots, k - 1]

        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state_f32,
            x2,
            dt2,
            A,
            B2,
            C2,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=(state_batch_indices if paged_cache else None),
            out=ref_out,
        )

        # Replay kernel — clone caches into mutable working copies that we
        # can inspect AFTER the call to verify cache postconditions.
        test_state = state0.clone()
        test_scales = state0_scales.clone() if is_quantized else None
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        old_x_w = old_x.clone()
        old_B_w = old_B.clone()
        old_dt_w = old_dt.clone()
        old_dA_cumsum_w = old_dA_cumsum.clone()
        # cache_buf_idx stays at its random values — each slot reads from its own buffer

        # Persistent_main consumes write-first work items. For pure-write or
        # pure-nowrite cases here, all slots have the same status, so the
        # work-item order is identity.
        n_writes_t, replay_work_items_t = _make_replay_work_items(
            prev_tokens, cache_buf_idx, T, max_window, batch,
            state_batch_indices, device,
        )
        replay_selective_state_update(
            test_state,
            old_x_w,
            old_B_w,
            old_dt_w,
            old_dA_cumsum_w,
            cache_buf_idx.clone(),
            prev_tokens,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=test_out,
            n_writes=n_writes_t,
            replay_work_items=replay_work_items_t,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            state_scales=test_scales,
            write_checkpoint=write_checkpoint,
            rectangle_for_nowrite=rectangle_for_nowrite,
            mode=mode,
        )

        # Tolerance rationale: the replay kernel uses bf16 tl.dot for four
        # matmuls (dB_scaled @ old_x, C @ state, CB_scaled @ x, and C @ B in
        # precompute).  The reference (selective_state_update) and flashinfer
        # baseline use fp32 element-wise MACs.  The bf16 input casts lose the
        # dt_bias/A-derived bits that the baselines keep — per-element rounding,
        # not accumulating.  Prefill (ssd_chunk_scan) does identical bf16 tl.dot
        # casts, so we match prefill precision exactly.  Empirical: max ~1.0 at
        # T<=16, ~2.0 at T=32-55; mean ~0.014; <0.02% of elements exceed 0.5.
        # State dtype (fp16/bf16/fp32) doesn't shift the error — bf16 dot
        # inputs dominate, not state storage.
        #
        # Quantized states add a per-element state quant error eps that
        # propagates through C @ state in the output dot.  With dstate=128
        # and C ~ N(0,1), the output channel std from this noise is roughly
        # eps * sqrt(128/3) ≈ 6.5 * eps.  Stack with the bf16 baseline:
        #   out_atol = bf16_atol + 6.5 * eps_max
        # where eps_max is the worst-case per-element error at the
        # post-replay SSM state magnitude (T=55 → amax ≈ 23).
        #
        # Per-element error (eps_max for T=55):
        #   int8     (uniform grid):  amax/(2*127)        ≈ 0.091
        #   int16    (uniform grid):  amax/(2*32767)      ≈ 3.5e-4
        #   fp8_e4m3 (variable grid): amax/16             ≈ 1.44 (worst-case
        #                             cell at top of channel; smaller for
        #                             smaller-magnitude elements)
        out_atol = (
            {torch.int8: 1.6, torch.int16: 1.05, torch.float8_e4m3fn: 4.0}[state_dtype]
            if is_quantized else 1.0
        )
        out_rtol = (
            {torch.int8: 2e-2, torch.int16: 2e-2, torch.float8_e4m3fn: 5e-2}[state_dtype]
            if is_quantized else 2e-2
        )
        out_diff = (test_out.float() - ref_out.float()).abs()
        out_max = out_diff.max().item()
        out_mean = out_diff.mean().item()
        try:
            torch.testing.assert_close(
                test_out, ref_out, rtol=out_rtol, atol=out_atol,
                msg=f"Output mismatch at k={k}",
            )
        except AssertionError:
            print(
                f"k={k}  out:  max={out_max:.4f}  mean={out_mean:.4f}  "
                f"nan={torch.isnan(test_out).any().item()}  "
                f"inf={torch.isinf(test_out).any().item()}"
            )
            raise

        # State expectation depends on write_checkpoint:
        #   True  → kernel writes the post-replay SSM state; expect the
        #           selective_state_update reference's state at step k-1.
        #   False → kernel skips the HBM store; state must be UNCHANGED
        #           from the input (state0; for quant, scales also unchanged).
        if is_quantized:
            if write_checkpoint:
                # Compare via dequant against the fp32 reference state.
                expected_fp32 = (
                    ref_input_state[slots] if k == 0 else states_buffer_f32[slots, k - 1]
                )
                actual_fp32 = _dequantize_state(test_state[slots], test_scales[slots])
                # State diff = bf16_replay_error + quant_error (per element).
                # The bf16 component is the SAME error source the non-quant
                # test absorbs in its atol=1.0 baseline (replay's tl.dot is
                # bf16-input fp32-accum; per-element error ~ 2^-7 * amax,
                # empirically ≤ ~0.2 at T=55 amax≈23).  Quant adds:
                #   int8: amax/(2*127) ≈ 0.091 worst-case
                #   int16: amax/(2*32767) ≈ 3.5e-4 (negligible vs bf16)
                #   fp8_e4m3 (variable grid): amax/16 ≈ 1.44 worst-case
                # Atol = bf16_baseline (1.0) + quant_eps_max.
                state_atol = {
                    torch.int8: 1.1, torch.int16: 1.0, torch.float8_e4m3fn: 2.5,
                }[state_dtype]
                state_rtol = {
                    torch.int8: 5e-2, torch.int16: 2e-2, torch.float8_e4m3fn: 1e-1,
                }[state_dtype]
                try:
                    torch.testing.assert_close(
                        actual_fp32, expected_fp32,
                        rtol=state_rtol, atol=state_atol,
                        msg=f"State mismatch at k={k} dtype={state_dtype}",
                    )
                except AssertionError:
                    diff = (actual_fp32 - expected_fp32).abs()
                    print(
                        f"k={k}  state(dequant): max={diff.max().item():.4f}  "
                        f"mean={diff.mean().item():.4f}"
                    )
                    raise
                # Scales sanity (fp32, finite, positive).
                assert test_scales.dtype == torch.float32
                assert torch.isfinite(test_scales[slots]).all(), (
                    f"state_scales has non-finite values at k={k}"
                )
                assert (test_scales[slots] > 0).all(), (
                    f"state_scales has non-positive values at k={k}"
                )
            else:
                # No write: raw quant state and scales unchanged.  Use
                # torch.equal for byte-level equality (dtype-agnostic; works
                # for int8 / int16 / fp8 alike).
                assert torch.equal(test_state[slots], state0[slots]), (
                    f"Quant state changed at k={k} write_checkpoint=False"
                )
                assert torch.equal(test_scales[slots], state0_scales[slots]), (
                    f"State scales changed at k={k} write_checkpoint=False"
                )
        else:
            if write_checkpoint:
                expected_state = (
                    state0[slots] if k == 0 else states_buffer_f32[slots, k - 1].to(state_dtype)
                )
            else:
                expected_state = state0[slots]
            state_diff = (test_state[slots].float() - expected_state.float()).abs()
            state_max = state_diff.max().item()
            state_mean = state_diff.mean().item()
            try:
                torch.testing.assert_close(
                    test_state[slots],
                    expected_state,
                    rtol=2e-2,
                    atol=1.0 if write_checkpoint else 0.0,
                    msg=f"State mismatch at k={k} (write_checkpoint={write_checkpoint})",
                )
            except AssertionError:
                print(
                    f"k={k}  state: max={state_max:.4f}  mean={state_mean:.4f}  "
                    f"nan={torch.isnan(test_state).any().item()}  "
                    f"inf={torch.isinf(test_state).any().item()}"
                )
                raise

        # --- Cache postconditions ---
        # Compute step 2's processed values (what the kernel should have
        # stored at [write_offset : write_offset+T) of write_buf):
        #   write_buf    = (1 - active_buf) if write_checkpoint else active_buf
        #   write_offset = 0                if write_checkpoint else k
        # Untouched cache regions must equal their pre-call snapshots
        # (old_x / old_B / old_dt / old_dA_cumsum captured before the call).
        dt2_proc = F.softplus(dt2_base.float() + dt_bias_base.float()[None, None, :])  # (B,T,H)
        dA_cumsum2 = torch.cumsum(A_base.float()[None, None, :] * dt2_proc, dim=1)
        write_offset = 0 if write_checkpoint else k

        for batch_idx, slot in enumerate(slot_indices):
            active = cache_buf_idx[slot].item()
            wb = (1 - active) if write_checkpoint else active

            # --- old_x (single-buffered): write at [write_offset : +T) of slot ---
            written_x = old_x_w[slot, write_offset : write_offset + T]
            torch.testing.assert_close(
                written_x, x2[batch_idx], rtol=0, atol=0,
                msg=f"old_x written region wrong at k={k} write={write_checkpoint}",
            )
            # Untouched ranges of old_x[slot]
            if write_offset > 0:
                torch.testing.assert_close(
                    old_x_w[slot, :write_offset], old_x[slot, :write_offset],
                    rtol=0, atol=0,
                    msg=f"old_x [0:{write_offset}) modified at k={k} write={write_checkpoint}",
                )
            if write_offset + T < max_window:
                torch.testing.assert_close(
                    old_x_w[slot, write_offset + T:], old_x[slot, write_offset + T:],
                    rtol=0, atol=0,
                    msg=f"old_x [{write_offset+T}:) modified at k={k} write={write_checkpoint}",
                )

            # --- old_B (double-buffered): write at write_buf, [write_offset:+T) ---
            torch.testing.assert_close(
                old_B_w[slot, wb, write_offset : write_offset + T],
                B2[batch_idx], rtol=0, atol=0,
                msg=f"old_B written region wrong at k={k} write={write_checkpoint}",
            )
            # Other-buffer (= 1-wb) untouched
            torch.testing.assert_close(
                old_B_w[slot, 1 - wb], old_B[slot, 1 - wb],
                rtol=0, atol=0,
                msg=f"old_B inactive buffer modified at k={k} write={write_checkpoint}",
            )

            # --- old_dt (double-buffered, fp32, layout (heads, T)): ---
            torch.testing.assert_close(
                old_dt_w[slot, wb, :, write_offset : write_offset + T],
                dt2_proc[batch_idx].T,
                rtol=1e-4, atol=1e-4,
                msg=f"old_dt written region wrong at k={k} write={write_checkpoint}",
            )
            torch.testing.assert_close(
                old_dt_w[slot, 1 - wb], old_dt[slot, 1 - wb],
                rtol=0, atol=0,
                msg=f"old_dt inactive buffer modified at k={k} write={write_checkpoint}",
            )

            # --- old_dA_cumsum (double-buffered, fp32, layout (heads, T)): ---
            # WRITE: fresh staging buf starts from 0, store per-step cumsum.
            # NOWRITE: append at offset k of active buf — values are continuous
            # from the start of the buffer, so add the prefix at position k-1
            # (matches the kernel's cross-step continuity fix).
            if write_checkpoint or k == 0:
                expected_dAcs = dA_cumsum2[batch_idx].T
            else:
                prefix = old_dA_cumsum[slot, wb, :, k - 1]  # (heads,)
                expected_dAcs = dA_cumsum2[batch_idx].T + prefix[:, None]
            torch.testing.assert_close(
                old_dA_cumsum_w[slot, wb, :, write_offset : write_offset + T],
                expected_dAcs,
                rtol=1e-4, atol=1e-4,
                msg=f"old_dA_cumsum written region wrong at k={k} write={write_checkpoint}",
            )
            torch.testing.assert_close(
                old_dA_cumsum_w[slot, 1 - wb], old_dA_cumsum[slot, 1 - wb],
                rtol=0, atol=0,
                msg=f"old_dA_cumsum inactive buf modified at k={k} write={write_checkpoint}",
            )


@pytest.mark.parametrize(
    "scenario,pnat_per_slot_list,explicit_order,rectangle_for_nowrite",
    [
        # All-write: every slot has PNAT triggering write
        # (PNAT + T > max_window). No explicit order needed.
        ("all_write", [12, 13, 14, 15], None, False),
        # All-nowrite: every slot fits in the window.
        ("all_nowrite", [3, 4, 5, 6], None, False),
        # Mixed PNATs with HAND-CODED work-item order.  The auto-computed order
        # for these PNATs would be [2, 3, 0, 1] — same as the explicit
        # value — so this scenario is functionally redundant with the
        # auto variant ONLY if `_make_replay_work_items` (the test
        # helper) is itself correct.  Keeping a hand-coded copy guards
        # against a buggy helper: the kernel still gets a known-good order
        # and the scenario would still pass even if the helper regressed.
        ("mixed_explicit", [3, 10, 12, 16], [2, 3, 0, 1], False),
        ("mixed_explicit_rect", [3, 10, 12, 16], [2, 3, 0, 1], True),
        # Mixed PNATs with AUTO-COMPUTED work-item order via `_make_replay_work_items`
        # — production-shaped flow. Different PNAT layout from the explicit
        # case ([1, 3, 0, 2]) so the kernel sees a distinct order.
        ("mixed_auto", [3, 12, 10, 15], None, False),
        ("mixed_auto_rect", [3, 12, 10, 15], None, True),
    ],
    ids=[
        "all_write", "all_nowrite",
        "mixed_explicit", "mixed_explicit_rect",
        "mixed_auto", "mixed_auto_rect",
    ],
)
@pytest.mark.parametrize("mode", ["persistent_main", "persistent_dynamic"], ids=["pm", "pd"])
def test_replay_selective_state_update_scenarios(
    scenario, pnat_per_slot_list, explicit_order, rectangle_for_nowrite, mode,
):
    """
    Combined scenarios test covering both kernel modes (persistent_main,
    persistent_dynamic) across a representative mix of write/nowrite
    layouts:

      - all_write / all_nowrite: every slot on one branch — verifies the
        empty-half early-return on pm and the all-uniform per-slot dispatch
        on pd.
      - mixed_explicit: hand-coded work-item order, bypassing the test's
        `_make_replay_work_items` helper.  Guards against a buggy helper:
        if the auto-computation regressed, the auto scenarios would still
        pass with the broken value, but this one runs against a known-good
        order and would still detect the kernel-side issue.
      - mixed_auto: unsorted PNATs, work-item order auto-computed via the
        helper — the production-shaped flow.

    Setup mirrors test_replay_selective_state_update_sorted_dispatch
    (same fixed seeds, same input shapes) so the reference state evolution
    is identical and we can compare per-slot output and HBM-state
    postconditions to the same reference.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    T = 6
    max_window = 16
    batch = 4
    device = "cuda"
    dtype = torch.bfloat16

    pnat_per_slot = torch.tensor(pnat_per_slot_list, device=device, dtype=torch.int32)
    pnat_means_write = (pnat_per_slot + T > max_window).tolist()

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=dtype
    )
    ref_input_state = state0.float()

    step1_T = max_window
    x1 = torch.randn(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, step1_T, nheads, device=device, dtype=dtype)
    dt1_input = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)

    states_buffer_f32 = torch.zeros(
        batch, step1_T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = torch.arange(batch, device=device, dtype=torch.int32)
    out1 = torch.zeros(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_input_state.clone(),
        x1, dt1_input, A, B1, C1,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=step1_T,
        out=out1,
        disable_state_update=True,
    )

    old_x = torch.zeros(batch, max_window, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, max_window, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, max_window, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(
        batch, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (batch,), device=device, dtype=torch.int32)

    old_x[:, :step1_T] = x1
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_processed, dim=1)
    for i in range(batch):
        buf = cache_buf_idx[i].item()
        old_B[i, buf, :step1_T] = B1[i]
        old_dt[i, buf, :, :step1_T] = dt1_processed[i].T
        old_dA_cumsum[i, buf, :, :step1_T] = dA_cumsum1[i].T

    torch.manual_seed(123)
    x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    ref_state_f32 = ref_input_state.clone()
    for i in range(batch):
        k_i = pnat_per_slot[i].item()
        if k_i > 0:
            ref_state_f32[i] = states_buffer_f32[i, k_i - 1]
    ref_state_after_replay = ref_state_f32.clone()

    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32, x2, dt2, A, B2, C2,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=None, out=ref_out,
    )

    test_state = state0.clone()
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    n_writes_t, replay_work_items = _make_replay_work_items(
        pnat_per_slot, cache_buf_idx, T, max_window, batch, None, device,
        explicit_order=explicit_order,
    )
    replay_selective_state_update(
        test_state,
        old_x.clone(), old_B.clone(), old_dt.clone(), old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        pnat_per_slot,
        x=x2, dt=dt2, A=A, B=B2, C=C2,
        out=test_out,
        n_writes=n_writes_t,
        replay_work_items=replay_work_items,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=None,
        mode=mode,
        rectangle_for_nowrite=rectangle_for_nowrite,
    )

    torch.testing.assert_close(
        test_out.float(), ref_out.float(),
        atol=1.0, rtol=0.05,
        msg=f"Output mismatch (scenario={scenario})",
    )

    for i in range(batch):
        if pnat_means_write[i]:
            torch.testing.assert_close(
                test_state[i].float(), ref_state_after_replay[i].float(),
                atol=1.0, rtol=0.05,
                msg=f"Write slot {i}: state mismatch (scenario={scenario})",
            )
        else:
            torch.testing.assert_close(
                test_state[i], state0[i], rtol=0, atol=0,
                msg=f"Nowrite slot {i}: state HBM modified (scenario={scenario})",
            )


@pytest.mark.parametrize(
    "scenario,pnat_per_slot_list,n_writes_expected,work_item_order",
    [
        # Mixed batch with write-first work-item order.
        ("mixed_sorted", [3, 10, 12, 16], 2, [2, 3, 0, 1]),
        # Boundary: n_writes=batch. The nowrite half has an empty slot range
        # and the persistent loop must do no work.
        ("all_write_noskip", [12, 13, 14, 15], 4, [0, 1, 2, 3]),
        # Boundary: n_writes=0. The write half has an empty slot range.
        ("all_nowrite_noskip", [3, 4, 5, 6], 0, [0, 1, 2, 3]),
    ],
    ids=["mixed_sorted", "all_write_noskip", "all_nowrite_noskip"],
)
@pytest.mark.parametrize("rectangle_for_nowrite", [True, False], ids=["rect", "norect"])
@pytest.mark.parametrize("mode", ["persistent_main", "persistent_dynamic"], ids=["pm", "pd"])
def test_replay_selective_state_update_persistent_main_device_n_writes(
    scenario, pnat_per_slot_list, n_writes_expected, work_item_order,
    rectangle_for_nowrite, mode,
):
    """
    Persistent_main with the device-tensor n_writes plumbing.

    The kernel reads `n_writes` from device memory at entry, so CUDA graphs
    can reuse the same captured pointer while updating the value between
    replays. Both persistent_main halves launch even when one half has no
    slots.

    Verifies:
      1. Kernel reads device n_writes correctly (output matches reference).
      2. Empty-half launches don't corrupt state (n_writes=0 / =batch).
      3. replay_work_items still work through the device-n_writes path.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    T = 6
    max_window = 16
    batch = 4
    device = "cuda"
    dtype = torch.bfloat16

    pnat_per_slot = torch.tensor(pnat_per_slot_list, device=device, dtype=torch.int32)
    pnat_means_write = (pnat_per_slot + T > max_window).tolist()
    write_count = sum(pnat_means_write)
    assert write_count == n_writes_expected, (
        f"test setup error: expected {n_writes_expected} writes, got {write_count}"
    )

    # Device-tensor n_writes. Caller mutates between iters in CUDA-graph
    # benchmarking; we only run one iter here so a single fill is enough.
    n_writes = torch.tensor([n_writes_expected],
                            device=device,
                            dtype=torch.int32)

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=dtype
    )
    ref_input_state = state0.float()

    step1_T = max_window
    x1 = torch.randn(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, step1_T, nheads, device=device, dtype=dtype)
    dt1_input = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, step1_T, ngroups, d_state, device=device, dtype=dtype)

    states_buffer_f32 = torch.zeros(
        batch, step1_T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = torch.arange(batch, device=device, dtype=torch.int32)
    out1 = torch.zeros(batch, step1_T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_input_state.clone(),
        x1, dt1_input, A, B1, C1,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=step1_T,
        out=out1,
        disable_state_update=True,
    )

    old_x = torch.zeros(batch, max_window, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, max_window, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, max_window, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(
        batch, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (batch,), device=device, dtype=torch.int32)
    _n_writes_check, replay_work_items = _make_replay_work_items(
        pnat_per_slot, cache_buf_idx, T, max_window, batch, None, device,
        explicit_order=work_item_order,
    )
    torch.testing.assert_close(_n_writes_check, n_writes)

    old_x[:, :step1_T] = x1
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_processed, dim=1)
    for i in range(batch):
        buf = cache_buf_idx[i].item()
        old_B[i, buf, :step1_T] = B1[i]
        old_dt[i, buf, :, :step1_T] = dt1_processed[i].T
        old_dA_cumsum[i, buf, :, :step1_T] = dA_cumsum1[i].T

    torch.manual_seed(123)
    x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    ref_state_f32 = ref_input_state.clone()
    for i in range(batch):
        k_i = pnat_per_slot[i].item()
        if k_i > 0:
            ref_state_f32[i] = states_buffer_f32[i, k_i - 1]
    ref_state_after_replay = ref_state_f32.clone()

    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32, x2, dt2, A, B2, C2,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=None, out=ref_out,
    )

    test_state = state0.clone()
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        test_state,
        old_x.clone(), old_B.clone(), old_dt.clone(), old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        pnat_per_slot,
        x=x2, dt=dt2, A=A, B=B2, C=C2,
        out=test_out,
        n_writes=n_writes,
        replay_work_items=replay_work_items,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=None,
        mode=mode,
        rectangle_for_nowrite=rectangle_for_nowrite,
    )

    torch.testing.assert_close(
        test_out.float(), ref_out.float(),
        atol=1.0, rtol=0.05,
        msg=f"Output mismatch (scenario={scenario})",
    )

    for i in range(batch):
        if pnat_means_write[i]:
            torch.testing.assert_close(
                test_state[i].float(), ref_state_after_replay[i].float(),
                atol=1.0, rtol=0.05,
                msg=f"Write slot {i}: state mismatch (scenario={scenario})",
            )
        else:
            torch.testing.assert_close(
                test_state[i], state0[i], rtol=0, atol=0,
                msg=f"Nowrite slot {i}: state HBM modified (scenario={scenario})",
            )


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn],
    ids=["fp16", "int8", "int16", "fp8"],
)
@pytest.mark.parametrize("paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
@pytest.mark.parametrize("mode", ["persistent_main", "persistent_dynamic"], ids=["pm", "pd"])
def test_replay_selective_state_update_philox(
    state_dtype, nheads, head_dim, d_state, ngroups, paged_cache, T, mode,
):
    """
    Verify that Philox stochastic rounding produces correct results across
    all SR-supported state dtypes (fp16, int8, int16, fp8_e4m3fn).

    Runs our kernel twice with identical inputs — once without rand_seed
    (deterministic RN), once with rand_seed (Philox SR) — and confirms:
      - Outputs are within bf16-dot tolerance (state perturbation ≤ 1 ULP).
      - State dtype is preserved.
      - State difference is bounded by ~1 ULP of the chosen grid.
    """
    _maybe_skip_dtype(state_dtype, use_sr=True)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
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

    if is_quantized:
        state0_fp32 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)
    else:
        state0 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state0_scales = None

    # Cache tensors
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # New token inputs
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    # max_window is old_x.shape[1] per the wrapper convention; the philox
    # test sets old_x = (cache_size, T, ...) so max_window = T here.
    _max_window_philox = T
    _n_writes_philox, _replay_work_items_philox = _make_replay_work_items(
        prev_tokens, cache_buf_idx, T, _max_window_philox, batch,
        state_batch_indices, device,
    )

    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        n_writes=_n_writes_philox,
        replay_work_items=_replay_work_items_philox,
        mode=mode,
    )

    # --- Run without rounding (deterministic RN store) ---
    state_no_round = state0.clone()
    scales_no_round = state0_scales.clone() if is_quantized else None
    out_no_round = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_no_round,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_no_round,
        state_scales=scales_no_round,
        **common_kwargs,
    )

    # --- Run with Philox rounding ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rounded = state0.clone()
    scales_rounded = state0_scales.clone() if is_quantized else None
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        state_scales=scales_rounded,
        **common_kwargs,
    )

    # Outputs should be nearly identical — rounding only perturbs the
    # post-replay SSM state by ±1 ULP before the output phase reads it.
    # Out_atol = bf16_baseline + 6.5 * per_elem_ULP_after_dequant:
    #   non-quant fp16: fp16 ULP at typical magnitude is tiny → 1.0
    #   int8:  amax/127 ≈ 23/127 → 6.5*0.18 ≈ 1.2 + bf16_baseline
    #   int16: amax/32767 ≈ 7e-4 → ~bf16_baseline only
    #   fp8:   amax/14 ≈ 23/14 → 6.5*1.6 ≈ 10.7 + bf16_baseline
    out_atol = (
        {torch.int8: 1.5, torch.int16: 1.0, torch.float8_e4m3fn: 6.0}[state_dtype]
        if is_quantized else 1.0
    )
    out_rtol = (
        {torch.int8: 2e-2, torch.int16: 2e-2, torch.float8_e4m3fn: 5e-2}[state_dtype]
        if is_quantized else 2e-2
    )
    torch.testing.assert_close(
        out_rounded, out_no_round, rtol=out_rtol, atol=out_atol,
        msg=f"Output diverged with Philox rounding ({state_dtype})",
    )

    # State dtype preserved.
    assert state_rounded.dtype == state_dtype

    # State diff between RN and SR is bounded by 1 quant cell per element.
    # Per-channel decode_scale varies by 10x+ across channels (amax depends
    # on randn extremes), so a single flat atol can't bound it accurately —
    # use per-channel ULP-aware comparison.
    slots = state_batch_indices if paged_cache else slice(None)
    if is_quantized:
        rounded_fp32 = _dequantize_state(state_rounded[slots], scales_rounded[slots])
        no_round_fp32 = _dequantize_state(state_no_round[slots], scales_no_round[slots])
        diff = (rounded_fp32 - no_round_fp32).abs()
        # Per-element bound = max(decode_scale_no_round, decode_scale_rounded).
        # decode_scale is shape (cache, nheads, dim); broadcast over dstate.
        scale_bound = torch.maximum(
            scales_no_round[slots], scales_rounded[slots]
        ).unsqueeze(-1)
        # int8 / int16: 1 cell after dequant = decode_scale exactly.
        # fp8_e4m3: variable grid; the largest cell within a channel scaled
        # to fit ±448 is at the channel's max-magnitude element, where the
        # cell is ~32x larger than the average.  Bound = decode_scale * 32.
        # Apply a 1.5x slack pad for floating-point compare quirks at the
        # exact-cell boundary.
        cell_pad = (
            32.0 if state_dtype == torch.float8_e4m3fn else 1.0
        )
        bound = scale_bound * (cell_pad * 1.5)
        if not (diff <= bound).all():
            offenders = (diff > bound).sum().item()
            n_total = diff.numel()
            pytest.fail(
                f"State RN-SR diff exceeds 1 cell per element for "
                f"{offenders}/{n_total} elements ({state_dtype}).  "
                f"max_diff={diff.max().item():.4g}, "
                f"max_bound={bound.max().item():.4g}."
            )
    else:
        # fp16 ULP depends on magnitude — rtol absorbs that.
        torch.testing.assert_close(
            state_rounded[slots],
            state_no_round[slots],
            rtol=2e-3,
            atol=0.2,
            msg=f"State diverged with Philox rounding ({state_dtype})",
        )


@pytest.mark.parametrize(
    "state_dtype",
    [torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn],
    ids=["fp16", "int8", "int16", "fp8"],
)
def test_philox_rounding_unbiased(state_dtype):
    """
    Verify that Philox stochastic rounding is unbiased across all
    SR-supported state dtypes (fp16, int8, int16, fp8_e4m3fn).

    Captures the true fp32 post-replay SSM state by running with fp32 storage,
    then runs the kernel with the target dtype + Philox SR.  Compares the
    SR rounding residual against the deterministic-RN residual: SR should
    have mean residual closer to zero than RN, since RN has a systematic
    round-to-nearest-even bias and SR is unbiased by construction.

    Uses a large batch (16) for ~2M state elements — plenty of statistics.
    """
    _maybe_skip_dtype(state_dtype, use_sr=True)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

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

    # fp32 reference state — replay produces values that don't fit cleanly
    # in the target dtype's grid, exposing the rounding bias.
    state0_fp32 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt_val = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((batch,), T, device=device, dtype=torch.int32)

    # max_window = old_x.shape[1] = T
    _n_writes_unb, _replay_work_items_unb = _make_replay_work_items(
        prev_tokens, cache_buf_idx, T, T, batch, None, device,
    )
    common_kwargs = dict(
        x=x, dt=dt_val, A=A, B=B, C=C, D=D, dt_bias=dt_bias, dt_softplus=True,
        n_writes=_n_writes_unb, replay_work_items=_replay_work_items_unb,
    )

    # 1. fp32 state — captures true post-replay fp32 state.
    state_fp32 = state0_fp32.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_fp32,
        old_x.clone(), old_B.clone(), old_dt.clone(), old_dA_cumsum.clone(),
        cache_buf_idx.clone(), prev_tokens, out=out_fp32, **common_kwargs,
    )

    # 2. Target dtype + Philox SR.  For quant we also need scales (derived
    # from the same per-channel amax used by the kernel on store).
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    if is_quantized:
        state_rounded, scales_rounded = _quantize_state(state0_fp32, state_dtype, quant_max)
    else:
        state_rounded = state0_fp32.to(state_dtype)
        scales_rounded = None
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rounded,
        old_x.clone(), old_B.clone(), old_dt.clone(), old_dA_cumsum.clone(),
        cache_buf_idx.clone(), prev_tokens, out=out_rounded,
        rand_seed=rand_seed, philox_rounds=10,
        state_scales=scales_rounded,
        **common_kwargs,
    )

    # Compute residuals.  For non-quant: stochastic_residual = SR(fp32) -
    # fp32, deterministic_residual = RN(fp32) - fp32.  For quant: dequant
    # both, comparing in fp32.
    if is_quantized:
        fp32_vals = state_fp32.flatten()
        stochastic_residual = (
            _dequantize_state(state_rounded, scales_rounded).flatten() - fp32_vals
        )
        # Deterministic reference: do the same per-channel quant on the
        # captured fp32 state, then dequant.  This is what the kernel would
        # have produced with rand_seed=None.
        det_quant, det_scales = _quantize_state(state_fp32, state_dtype, quant_max)
        deterministic_residual = (
            _dequantize_state(det_quant, det_scales).flatten() - fp32_vals
        )
    else:
        fp32_vals = state_fp32.flatten()
        stochastic_residual = state_rounded.float().flatten() - fp32_vals
        deterministic_residual = fp32_vals.to(state_dtype).float() - fp32_vals

    # Only consider elements where rounding matters (non-zero residual possible).
    nonzero_mask = deterministic_residual.abs() > 0
    num_nonzero = nonzero_mask.sum().item()
    assert num_nonzero > 1000, f"Too few roundable elements: {num_nonzero}"

    stochastic_mean = stochastic_residual[nonzero_mask].mean().item()
    stochastic_std = stochastic_residual[nonzero_mask].std().item()
    deterministic_mean = deterministic_residual[nonzero_mask].mean().item()

    # SE-based bias check.  An unbiased estimator's sample mean has standard
    # error SE = std / sqrt(n).  We require |sr_mean| < K*SE (K=4 ≈ ~3.2e-5
    # one-sided false-positive rate).  This auto-calibrates per dtype:
    #   * int16: residual std ~1e-4 → SE ~9e-8 (very tight bound)
    #   * int8:  residual std ~3e-2 → SE ~2e-5
    #   * fp8:   residual std ~1e-1 → SE ~9e-5 (loosest, magnitude-driven)
    # The previous fixed-1e-5 threshold was below SE for int8/fp8 and would
    # always fail by chance.  Note the |sr|<|det| fallback was also dropped:
    # on Gaussian (symmetric) inputs RN's bias is ~0 by symmetry, so SR vs RN
    # is just two unbiased estimators racing — unreliable as a unbias test.
    se_sr = stochastic_std / (num_nonzero ** 0.5)
    K = 4
    assert abs(stochastic_mean) < K * se_sr, (
        f"SR mean exceeds {K}*SE (likely biased) ({state_dtype}): "
        f"stochastic_mean={stochastic_mean:.3e}, "
        f"SE={se_sr:.3e} (K*SE={K * se_sr:.3e}), "
        f"deterministic_mean={deterministic_mean:.3e} (for reference), "
        f"n_elements={num_nonzero}"
    )


# HEADS_PER_BLOCK > 1 test.  The default heuristic only picks HPB > 1 at large
# total_heads (>= 256-512), which the main test with batch=2 never reaches.
# This test overrides _heads_per_block to exercise multi-head precompute tiles.
#
# Beyond OUTPUT and STATE checks, this also asserts the kernel's WRITE
# CONTRACT on every cache buffer (old_x, old_B, old_dt, old_dA_cumsum)
# across a sweep of PNATs covering nowrite (PNAT=0,1,T,max_window-T-1,
# max_window-T) and write (PNAT=max_window-T+1, max_window-1) paths.  The
# old_dA_cumsum check is the one that originally hid the
# continuous-across-nowrite bug — direct verification prevents regression.
# Both `rectangle_for_nowrite` arms are exercised explicitly (don't rely on
# tuning).
#
# Configs: (nheads=16, ngroups=1) and (nheads=32, ngroups=2) both have
# heads_per_group=16.  The heuristic caps HPB at min(2|4, hpg), so HPB=2, 4.
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
@pytest.mark.parametrize("rectangle_nowrite", [True, False], ids=["rect", "norect"])
@pytest.mark.parametrize("mode", ["persistent_main", "persistent_dynamic"], ids=["pm", "pd"])
def test_replay_heads_per_block(
    nheads,
    head_dim,
    d_state,
    ngroups,
    state_dtype,
    T,
    heads_per_block,
    rectangle_nowrite,
    mode,
):
    """
    Verify replay_selective_state_update produces correct results when
    _heads_per_block > 1.

    In addition to the output + state checks, this verifies the full write
    contract: per-slot the kernel touches the correct staging/active buffer
    at the correct offset for old_x, old_B, old_dt, old_dA_cumsum; leaves
    untouched regions and the other buffer byte-identical to pre-call; and
    (for nowrite) preserves the dA_cumsum prefix continuity by adding the
    pre-call old_dA_cumsum[slot, active_buf, head, PNAT-1] value.
    """
    # PDL flags use wrapper defaults; trimming the parametrize keeps this
    # suite fast.  Coverage of {launch_with_pdl, use_internal_pdl} variations
    # lives in the dedicated correctness tests above (test_replay_selective_state_update).
    batch = 8
    device = "cuda"
    dtype = torch.bfloat16

    if nheads % heads_per_block != 0:
        pytest.skip(f"nheads ({nheads}) not divisible by heads_per_block ({heads_per_block})")
    if heads_per_block > nheads // ngroups:
        pytest.skip(
            f"heads_per_block ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})"
        )

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # max_window = 2*next_pow2(T) so the PNAT=T nowrite case is always
    # valid (requires max_window >= 2T) and we have headroom for the
    # full PNAT sweep below.
    max_window = max(2 * triton.next_power_of_2(T), 16)

    cache_size = batch
    state0 = torch.randn(cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype)

    # Generate enough fill data to cover max_window steps (needed for the
    # write-slot replay reference, which walks up to max_window-1 steps).
    x1 = torch.randn(batch, max_window, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, max_window, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, max_window, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, max_window, ngroups, d_state, device=device, dtype=dtype)

    states_buffer_f32 = torch.zeros(
        cache_size, max_window, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = torch.arange(batch, device=device, dtype=torch.int32)
    out1 = torch.zeros(batch, max_window, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=max_window,
        out=out1,
        disable_state_update=True,
    )

    # Pre-fill cache buffers.  old_x is single-buffer (no dbuf dim); old_B,
    # old_dt, old_dA_cumsum are double-buffered.  Initialize BOTH buffers
    # with controlled random data so "outside write range / other buffer
    # unchanged" assertions have well-defined expected values for both.
    old_x_init = torch.randn(
        cache_size, max_window, nheads, head_dim, device=device, dtype=dtype
    )
    old_B_init = torch.randn(
        cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype
    )
    old_dt_init = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    old_dA_cumsum_init = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Capture-step data populates ONE buffer per slot (the active one selected
    # by cache_buf_idx).  Mirrors production: previous step wrote into the
    # now-active buffer; the other buffer holds stale data the kernel must
    # not touch on a nowrite call.
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)

    old_x_init[:] = x1  # single buffer
    for slot in range(cache_size):
        buf = int(cache_buf_idx[slot].item())
        old_B_init[slot, buf] = B1[slot]
        old_dt_init[slot, buf] = dt1_proc[slot].T            # (nheads, max_window)
        old_dA_cumsum_init[slot, buf] = dA_cumsum1[slot].T   # (nheads, max_window)

    # --- PNAT sweep --------------------------------------------------------
    # Cover nowrite (PNAT+T <= max_window) and write (PNAT+T > max_window)
    # paths plus the boundary, with both PNAT=0 (no prefix) and PNAT=T (the
    # smallest prefix-load case the kernel cares about).
    candidate_pnats = [
        0,                       # nowrite, no prefix
        1,                       # nowrite, smallest nontrivial prefix
        T,                       # nowrite, prefix length one stored step
        max_window - T - 1,      # nowrite, largest PNAT just below threshold
        max_window - T,          # nowrite, exactly at threshold
        max_window - T + 1,      # write, smallest PNAT above threshold
        max_window - 1,          # write, maximum
    ]
    seen = set()
    pnat_list = []
    for p in candidate_pnats:
        if 0 <= p < max_window and p not in seen:
            seen.add(p)
            pnat_list.append(p)
    while len(pnat_list) < batch:
        pnat_list.append(0)
    pnat_list = pnat_list[:batch]

    has_write = any((p + T) > max_window for p in pnat_list)
    has_nowrite = any((p + T) <= max_window for p in pnat_list)
    assert has_write and has_nowrite, (
        f"PNAT sweep must cover both write and nowrite: {pnat_list}, "
        f"T={T}, max_window={max_window}"
    )

    prev_tokens = torch.tensor(pnat_list, device=device, dtype=torch.int32)
    pnat_means_write = [(pnat_list[i] + T) > max_window for i in range(batch)]

    torch.manual_seed(123)
    x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # Per-step processed dt and per-step cumsum — what the kernel writes to
    # old_dt and the cumsum-from-zero portion of old_dA_cumsum.
    dt2_proc = F.softplus(dt2_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum2_step = torch.cumsum(A_base.float()[None, None, :] * dt2_proc, dim=1)

    # Reference state walk identical to the original test.
    ref_state_f32 = state0.float().clone()
    for slot in range(batch):
        if pnat_list[slot] > 0:
            ref_state_f32[slot] = states_buffer_f32[slot, pnat_list[slot] - 1]
    ref_state_after_replay = ref_state_f32.clone()
    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32, x2, dt2, A, B2, C2,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=None, out=ref_out,
    )

    # Pre-call snapshots double as the "expected" baseline for untouched
    # regions / untouched buffer.  Kernel operates on the _test copies.
    test_state = state0.clone()
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    state_pre = test_state.clone()
    old_x_pre = old_x_init.clone()
    old_B_pre = old_B_init.clone()
    old_dt_pre = old_dt_init.clone()
    old_dA_cumsum_pre = old_dA_cumsum_init.clone()
    cache_buf_idx_pre = cache_buf_idx.clone()

    old_x_test = old_x_pre.clone()
    old_B_test = old_B_pre.clone()
    old_dt_test = old_dt_pre.clone()
    old_dA_cumsum_test = old_dA_cumsum_pre.clone()
    cache_buf_idx_test = cache_buf_idx_pre.clone()

    n_writes_t, replay_work_items_t = _make_replay_work_items(
        prev_tokens, cache_buf_idx_test, T, max_window, batch, None, device,
    )

    replay_selective_state_update(
        test_state,
        old_x_test,
        old_B_test,
        old_dt_test,
        old_dA_cumsum_test,
        cache_buf_idx_test,
        prev_tokens,
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=test_out,
        n_writes=n_writes_t,
        replay_work_items=replay_work_items_t,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
        mode=mode,
        rectangle_for_nowrite=rectangle_nowrite,
        _heads_per_block=heads_per_block,
    )

    # ---------------- Output + state checks (existing coverage) -------------
    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=2e-2,
        atol=1.0,
        msg=f"Output mismatch with HPB={heads_per_block}, T={T}, "
        f"nheads={nheads}, ngroups={ngroups}, state_dtype={state_dtype}, "
        f"rect={rectangle_nowrite}, pnats={pnat_list}",
    )

    for slot in range(batch):
        if pnat_means_write[slot]:
            torch.testing.assert_close(
                test_state[slot].float(), ref_state_after_replay[slot].float(),
                rtol=2e-2, atol=1.0,
                msg=(
                    f"Write slot {slot} (PNAT={pnat_list[slot]}): state mismatch "
                    f"(HPB={heads_per_block}, T={T}, nheads={nheads}, "
                    f"ngroups={ngroups}, state_dtype={state_dtype}, "
                    f"rect={rectangle_nowrite})"
                ),
            )
        else:
            torch.testing.assert_close(
                test_state[slot], state_pre[slot],
                rtol=0, atol=0,
                msg=(
                    f"Nowrite slot {slot} (PNAT={pnat_list[slot]}): state HBM "
                    f"modified (HPB={heads_per_block}, T={T}, nheads={nheads}, "
                    f"ngroups={ngroups}, state_dtype={state_dtype}, "
                    f"rect={rectangle_nowrite})"
                ),
            )

    # ---------------- New checks: kernel write contract ---------------------
    # For each slot: decide active/staging buffer, write offset, build the
    # expected per-buffer tensor element-wise, compare against the actual
    # kernel-modified tensor.  The expected_* tensors are clones of the
    # pre-call snapshot with only [target_buf, write_offset:write_end]
    # overwritten — so the full-slot equality compares implicitly assert
    # the "other buffer untouched" and "outside write range untouched"
    # contracts.
    for slot in range(batch):
        pnat = pnat_list[slot]
        is_write = pnat_means_write[slot]
        active_buf = int(cache_buf_idx_pre[slot].item())
        staging_buf = 1 - active_buf

        if is_write:
            target_buf = staging_buf
            write_offset = 0
        else:
            target_buf = active_buf
            write_offset = pnat
        write_end = write_offset + T

        # ----- old_x (single-buffer) -----
        expected_old_x_slot = old_x_pre[slot].clone()
        expected_old_x_slot[write_offset:write_end] = x2[slot]
        torch.testing.assert_close(
            old_x_test[slot], expected_old_x_slot,
            rtol=0, atol=0,
            msg=(
                f"old_x slot {slot} (PNAT={pnat}, is_write={is_write}, "
                f"write_offset={write_offset}): mismatch "
                f"(HPB={heads_per_block}, T={T}, nheads={nheads}, "
                f"ngroups={ngroups}, rect={rectangle_nowrite})"
            ),
        )

        # ----- old_B (double-buffer) -----
        expected_old_B_slot = old_B_pre[slot].clone()
        expected_old_B_slot[target_buf, write_offset:write_end] = B2[slot]
        torch.testing.assert_close(
            old_B_test[slot], expected_old_B_slot,
            rtol=0, atol=0,
            msg=(
                f"old_B slot {slot} (PNAT={pnat}, is_write={is_write}, "
                f"target_buf={target_buf}, write_offset={write_offset}): "
                f"mismatch (HPB={heads_per_block}, T={T}, nheads={nheads}, "
                f"ngroups={ngroups}, rect={rectangle_nowrite})"
            ),
        )

        # ----- old_dt (double-buffer (cache, 2, nheads, max_window)) -----
        # Kernel writes per-head processed dt at [target_buf, :, write_offset:write_end].
        # softplus on chip vs F.softplus host: small ULP diff possible; use
        # tight but non-zero tolerance.
        expected_old_dt_slot = old_dt_pre[slot].clone()
        expected_old_dt_slot[target_buf, :, write_offset:write_end] = dt2_proc[slot].T
        torch.testing.assert_close(
            old_dt_test[slot], expected_old_dt_slot,
            rtol=1e-5, atol=1e-5,
            msg=(
                f"old_dt slot {slot} (PNAT={pnat}, is_write={is_write}, "
                f"target_buf={target_buf}, write_offset={write_offset}): "
                f"mismatch (HPB={heads_per_block}, T={T}, nheads={nheads}, "
                f"ngroups={ngroups}, rect={rectangle_nowrite})"
            ),
        )

        # ----- old_dA_cumsum (double-buffer (cache, 2, nheads, max_window)) -----
        # WRITE: per-step cumsum starting from 0 (fresh staging buf).
        # NOWRITE: continuous — cumsum offset by the prefix value at
        # old_dA_cumsum_pre[slot, active_buf, head, PNAT-1] (or 0 if PNAT=0).
        # This is the direct regression assertion for the bug just fixed.
        expected_old_dAcs_slot = old_dA_cumsum_pre[slot].clone()
        step_cumsum = dA_cumsum2_step[slot].T  # (nheads, T)
        if is_write:
            expected_old_dAcs_slot[target_buf, :, write_offset:write_end] = step_cumsum
        else:
            if pnat > 0:
                prefix = old_dA_cumsum_pre[slot, active_buf, :, pnat - 1]
            else:
                prefix = torch.zeros(nheads, device=device, dtype=torch.float32)
            expected_old_dAcs_slot[target_buf, :, write_offset:write_end] = (
                step_cumsum + prefix[:, None]
            )
        torch.testing.assert_close(
            old_dA_cumsum_test[slot], expected_old_dAcs_slot,
            rtol=1e-5, atol=1e-5,
            msg=(
                f"old_dA_cumsum slot {slot} (PNAT={pnat}, is_write={is_write}, "
                f"target_buf={target_buf}, write_offset={write_offset}): "
                f"mismatch (HPB={heads_per_block}, T={T}, nheads={nheads}, "
                f"ngroups={ngroups}, rect={rectangle_nowrite})"
            ),
        )


# HPB > 1 multi-step test.  Production chains decode steps; bugs in
# buffer ordering or stale cache values accumulate across steps and can
# be invisible in a single-step test.
#
# Divergent per-slot acceptance: slot 0 accepts more tokens each step,
# slot 1 accepts a smaller fixed count.  This forces the write/nowrite
# mask to differ between slots on multiple steps (n_writes in {0, 1, 2}
# within an 8-step run and the work-item order hits both identity [0,1]
# and swapped [1,0] — exercising the kernel's per-slot dispatch).
# `rectangle_nowrite` forces the rectangle vs non-rectangle nowrite path
# (via `mode="persistent_main"` + `rectangle_for_nowrite=…` kwargs)
# rather than relying on the tuning table's mode pick.
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
@pytest.mark.parametrize("paged_cache", [False, True], ids=["contig", "paged"])
@pytest.mark.parametrize("rectangle_nowrite", [True, False], ids=["rect", "norect"])
@pytest.mark.parametrize("mode", ["persistent_main", "persistent_dynamic"], ids=["pm", "pd"])
def test_replay_heads_per_block_multistep(
    nheads, head_dim, d_state, ngroups, state_dtype, T, heads_per_block,
    paged_cache, rectangle_nowrite, mode,
):
    """
    Chain N decode steps with HPB > 1 and verify each step's output matches
    a fresh reference.  A bug that mixes up WRITE/READ buffers, writes wrong
    data to cache, or reuses stale cache values would accumulate across steps.
    Per-slot acceptance diverges so write/nowrite masks differ across slots;
    the n_writes=1 case (mixed batch) is exercised.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    n_steps = 8

    if nheads % heads_per_block != 0:
        pytest.skip(f"nheads ({nheads}) not divisible by HPB ({heads_per_block})")
    if heads_per_block > nheads // ngroups:
        pytest.skip(f"HPB ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})")

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    all_x = []
    all_dt = []
    all_B = []
    all_C = []
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        all_x.append(torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype))
        dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        all_dt.append(repeat(dt_base, "b t h -> b t h p", p=head_dim))
        all_B.append(torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype))
        all_C.append(torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype))

    torch.manual_seed(999)
    state_init = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    # max_window = 2*np2(T) so the buffer has slack for several nowrite
    # steps before overflow — required for the divergent-acceptance pattern
    # to produce a chain of mixed write/nowrite steps within n_steps=8.
    # For T=6 this is 16, for T=16 this is 32.
    max_window = max(triton.next_power_of_2(2 * T), 16)

    # Per-slot acceptance counts.  Slot 0 advances by 6 per step, slot 1 by
    # 4 — divergence (PNAT trajectories desync, write_mask varies between
    # slots, n_writes hits 0/1/2 within the 8-step run).  Values constant
    # across T because they exercise the kernel's per-slot dispatch
    # independent of T; acc <= T is the only requirement.
    accepted_per_slot = [6, 4]
    accepted_tensor = torch.tensor(accepted_per_slot, device=device, dtype=torch.int32)

    # Per-slot reference: each slot's reference state advances by only its
    # `accepted` tokens per step, not all T.  selective_state_update doesn't
    # natively support per-slot variable T, so run it once per slot per step
    # with a single-slot batch view.
    ref_state = state_init.float().clone()
    ref_outs = []
    for step in range(n_steps):
        out_step = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        for s_local in range(batch):
            acc = accepted_per_slot[s_local]
            if acc == 0:
                continue
            c_idx = (
                state_batch_indices[s_local].item()
                if state_batch_indices is not None else s_local
            )
            s_state = ref_state[c_idx:c_idx + 1].clone()
            s_x = all_x[step][s_local:s_local + 1, :acc].contiguous()
            s_dt = all_dt[step][s_local:s_local + 1, :acc].contiguous()
            s_B = all_B[step][s_local:s_local + 1, :acc].contiguous()
            s_C = all_C[step][s_local:s_local + 1, :acc].contiguous()
            s_out = torch.zeros(1, acc, nheads, head_dim, device=device, dtype=dtype)
            selective_state_update(
                s_state, s_x, s_dt, A, s_B, s_C,
                D=D, dt_bias=dt_bias, dt_softplus=True,
                state_batch_indices=torch.tensor([0], device=device, dtype=torch.int32),
                out=s_out,
            )
            out_step[s_local, :acc] = s_out[0]
            ref_state[c_idx] = s_state[0]
        ref_outs.append(out_step)

    test_state = state_init.clone()
    old_x = torch.zeros(cache_size, max_window, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.zeros(cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.zeros(cache_size, 2, nheads, max_window, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.zeros(cache_size, 2, nheads, max_window, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # Per-active-slot PNAT tracker, advanced by `accepted` (not T) per step.
    pnat_active = torch.zeros(batch, device=device, dtype=torch.int32)

    for step in range(n_steps):
        prev_tokens = torch.zeros(cache_size, device=device, dtype=torch.int32)
        if state_batch_indices is not None:
            prev_tokens[state_batch_indices.long()] = pnat_active
        else:
            prev_tokens[:] = pnat_active
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        n_writes_t, replay_work_items_t = _make_replay_work_items(
            prev_tokens, cache_buf_idx, T, max_window, batch,
            state_batch_indices, device,
        )

        replay_selective_state_update(
            test_state,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            prev_tokens,
            x=all_x[step],
            dt=all_dt[step],
            A=A,
            B=all_B[step],
            C=all_C[step],
            out=test_out,
            n_writes=n_writes_t,
            replay_work_items=replay_work_items_t,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            _heads_per_block=heads_per_block,
            mode=mode,
            rectangle_for_nowrite=rectangle_nowrite,
        )

        # PNAT update uses `accepted` (per-slot), not T.  Write step resets
        # PNAT to `accepted` of this step (new buffer starts fresh); nowrite
        # appends `accepted` to current PNAT.
        write_mask = (pnat_active + T) > max_window
        new_pnat_active = torch.where(
            write_mask, accepted_tensor, pnat_active + accepted_tensor,
        )
        pnat_active = new_pnat_active
        cache_active_idx = (
            state_batch_indices.long() if state_batch_indices is not None
            else torch.arange(batch, device=device)
        )
        write_slots = cache_active_idx[write_mask]
        cache_buf_idx[write_slots] = 1 - cache_buf_idx[write_slots]

        # Per-slot output comparison: only the first `accepted` output tokens
        # of each slot are meaningful (the rest are produced from "candidate"
        # tokens that wouldn't be accepted in production).
        for s_local in range(batch):
            acc = accepted_per_slot[s_local]
            if acc == 0:
                continue
            torch.testing.assert_close(
                test_out[s_local, :acc],
                ref_outs[step][s_local, :acc],
                rtol=2e-2,
                atol=2.0,
                msg=f"Output mismatch at step {step}, slot {s_local} "
                f"(acc={acc}) with HPB={heads_per_block}, T={T}, "
                f"nheads={nheads}, ngroups={ngroups}, state_dtype={state_dtype}, "
                f"paged_cache={paged_cache}, rectangle={rectangle_nowrite}",
            )


# ----- SR grid-bracket tests (fp8 and fp16) -----
#
# Verify that each PTX SR output lands on the destination dtype's grid as
# a bracket neighbour of the fp32 input.  Catches byte-order traps in the
# inline-asm source-register specifier:
#   * fp8: cvt.rs.satfinite.e4m3x4.f32 with pack=4, asm "{$4,$3,$2,$1}"
#   * fp16: cvt.rs.f16x2.f32 with pack=2, asm "$0, $2, $1, $3"
# The unbiased test (test_philox_rounding_unbiased) wouldn't catch a
# shuffle: outputs that are still on-grid but swapped within a pack still
# average correctly.  Only the per-element bracket check exposes it.
#
# Both kernels are inline copies of the production helpers — kept here so
# the test exercises the exact PTX form independent of wrapper changes.


@triton.jit
def _packed_int8_sr_kernel(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + (offs // 4))
    y = _stochastic_round_int8_packed(x, rand, offs)
    tl.store(out_ptr + offs, y.to(tl.int8))


@triton.jit
def _packed_int16_sr_kernel(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + (offs // 2))
    y = _stochastic_round_int16_packed(x, rand, offs)
    tl.store(out_ptr + offs, y.to(tl.int16))


def _bitrev_int(x: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (x & 1)
        x >>= 1
    return out


def _rand_words(rand: torch.Tensor) -> list[int]:
    return [int(v) & 0xFFFFFFFF for v in rand.cpu().tolist()]


def test_packed_int_sr_matches_reference():
    device = "cuda"
    n = 1024
    offs = torch.arange(n, device=device, dtype=torch.float32)
    x = ((offs % 37) - 18.0) + (((offs * 13.0) % 97.0) + 0.3) / 128.0

    torch.manual_seed(42)
    rand_i8 = torch.randint(-(2**31), 2**31, (n // 4,), device=device, dtype=torch.int32)
    out_i8 = torch.empty(n, device=device, dtype=torch.int8)
    _packed_int8_sr_kernel[(1,)](x, rand_i8, out_i8, BLOCK=n)

    x_cpu = x.cpu().tolist()
    rand_i8_words = _rand_words(rand_i8)
    ref_i8 = []
    for i, value in enumerate(x_cpu):
        word = rand_i8_words[i // 4]
        low = word & 0x0000FFFF
        high = (word >> 16) & 0x0000FFFF
        pos = i & 3
        if pos == 0:
            rand16 = low
        elif pos == 1:
            rand16 = _bitrev_int(low, 16)
        elif pos == 2:
            rand16 = high
        else:
            rand16 = _bitrev_int(high, 16)
        ref_i8.append(math.floor(value + rand16 / float(1 << 16)))

    torch.testing.assert_close(
        out_i8.cpu().to(torch.int16),
        torch.tensor(ref_i8, dtype=torch.int16),
        rtol=0,
        atol=0,
    )

    rand_i16 = torch.randint(-(2**31), 2**31, (n // 2,), device=device, dtype=torch.int32)
    out_i16 = torch.empty(n, device=device, dtype=torch.int16)
    _packed_int16_sr_kernel[(1,)](x, rand_i16, out_i16, BLOCK=n)

    rand_i16_words = _rand_words(rand_i16)
    ref_i16 = []
    for i, value in enumerate(x_cpu):
        word = rand_i16_words[i // 2]
        rand_bits = word if (i & 1) == 0 else _bitrev_int(word, 32)
        rand24 = rand_bits & 0x00FFFFFF
        ref_i16.append(math.floor(value + rand24 / float(1 << 24)))

    torch.testing.assert_close(
        out_i16.cpu(),
        torch.tensor(ref_i16, dtype=torch.int16),
        rtol=0,
        atol=0,
    )


@triton.jit
def _bracket_kernel_fp8(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + offs)
    y = tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )
    tl.store(out_ptr + offs, y)


@triton.jit
def _bracket_kernel_fp16(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + offs)
    y = tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )
    tl.store(out_ptr + offs, y)


_BRACKET_KERNEL = {
    torch.float8_e4m3fn: _bracket_kernel_fp8,
    torch.float16: _bracket_kernel_fp16,
}


def _build_finite_grid(dtype: torch.dtype, device: str) -> torch.Tensor:
    """Reinterpret all bit patterns of ``dtype`` as floats; return sorted
    unique finite values (drops ±inf, NaNs)."""
    if dtype == torch.float8_e4m3fn:
        ints = torch.arange(256, dtype=torch.uint8, device=device)
        full = ints.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == torch.float16:
        # int16 view of all 65536 patterns (covers fp16 normals + subnormals
        # + ±inf + NaN; we filter to finite below).
        ints = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
        full = ints.view(torch.float16).to(torch.float32)
    else:
        raise ValueError(f"Unsupported bracket-test dtype: {dtype}")
    return full[torch.isfinite(full)].sort()[0].unique()


def _build_bracket_inputs(dtype: torch.dtype, n: int, device: str) -> torch.Tensor:
    """Test inputs spanning the dtype's grid range.  Includes on-grid points
    so we exercise the no-rounding case; for fp8 also includes overflow to
    test saturation (PTX `cvt.rs.satfinite.e4m3x4.f32` clamps in-op).

    fp16 inputs are kept inside the finite range — `cvt.rs.f16x2.f32` does
    NOT have a `satfinite` modifier and produces ±inf for OOR inputs (not
    a saturate-to-±max).  The kernel only ever sees in-range fp32 state in
    practice (state_amax is always ≪ fp16_max), so the test mirrors that.
    """
    grid = _build_finite_grid(dtype, device)
    g_min, g_max = grid[0].item(), grid[-1].item()
    x = torch.empty(n, device=device, dtype=torch.float32)
    if dtype == torch.float8_e4m3fn:
        # 1.5x range exercises saturation; satfinite handles it in-op.
        x.uniform_(g_min * 1.5, g_max * 1.5)
    else:  # fp16: four magnitude bands, all within finite range.
        x[: n // 4].uniform_(-1.0, 1.0)
        x[n // 4 : n // 2].uniform_(-100, 100)
        x[n // 2 : 3 * n // 4].uniform_(-1000, 1000)
        x[3 * n // 4 :].uniform_(g_min * 0.99, g_max * 0.99)
    return x, grid


@_skip_pre_sm100
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float8_e4m3fn, torch.float16],
    ids=["fp8", "fp16"],
)
def test_sr_grid_bracket(state_dtype):
    """Verify SR PTX outputs each lie on the destination grid as a bracket
    neighbour of the fp32 input."""
    device = "cuda"
    n = 1024  # multiple of both pack=4 (fp8) and pack=2 (fp16)

    torch.manual_seed(42)
    x, grid_finite = _build_bracket_inputs(state_dtype, n, device)
    g_min, g_max = grid_finite[0].item(), grid_finite[-1].item()

    # Bracket [lo, hi] in the destination grid for each input.  For
    # out-of-range inputs the bracket is the saturating endpoint pair.
    x_clamped = x.clamp(g_min, g_max)
    idx = torch.searchsorted(grid_finite, x_clamped, right=False).clamp(
        min=1, max=len(grid_finite) - 1
    )
    lo = grid_finite[idx - 1]
    hi = grid_finite[idx]
    # For x exactly on grid, idx points at it; lo = grid[i-1], hi = x — the
    # bracket allows out==hi (=x) which is what RN-on-grid produces.

    kernel = _BRACKET_KERNEL[state_dtype]

    for seed in range(4):
        torch.manual_seed(seed)
        # int32 for raw random bits — PTX takes the bit pattern, sign
        # interpretation doesn't matter.
        rand = torch.randint(-(2**31), 2**31, (n,), device=device, dtype=torch.int32)
        out = torch.empty(n, device=device, dtype=state_dtype)
        kernel[(1,)](x, rand, out, BLOCK=n)
        out_fp32 = out.to(torch.float32)

        on_grid = (out_fp32 == lo) | (out_fp32 == hi)
        if not on_grid.all():
            offenders = ~on_grid
            n_off = offenders.sum().item()
            sample = (
                x[offenders][:5].tolist(),
                lo[offenders][:5].tolist(),
                hi[offenders][:5].tolist(),
                out_fp32[offenders][:5].tolist(),
            )
            pytest.fail(
                f"{state_dtype} SR output not on grid bracket for {n_off}/{n} "
                f"elements (seed={seed}).  x={sample[0]} lo={sample[1]} "
                f"hi={sample[2]} out={sample[3]}.  Likely the PTX byte-order "
                "bug (cvt.rs source-register order)."
            )
