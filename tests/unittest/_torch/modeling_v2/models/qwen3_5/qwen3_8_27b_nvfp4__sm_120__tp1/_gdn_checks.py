# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and checks shared by this target's Gated-DeltaNet entry tests.

* `q8_pool`: the quantized state pool `[S, 48, 260, 128]` (int16 rows), compact or as the cache manager
  v2 hands a layer its view (every GDN layer's state of a slot in one page, so the slot
  stride is 48 layers' states), with chosen slots holding quantized random states.
* `decode_inputs`: q/k/v/z views of one `[N, 16384]` in_proj_qkvz block and a/b views of
  one `[N, 96]` in_proj_ba block, as the decode step hands them over.
* `assert_q8_states_close` / `assert_q8_rows_close`: two quantized pools against each other
  on the named slots, in the band two independent quantizations of (nearly) the same
  fp32 state allow -- element by element, or per state row.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm._gdn_state_q8 import (  # noqa: E501
    Q8_QMAX,
    dequantize_slots,
    q8_state_rows,
    store_slots,
)

DEV = "cuda"
#: The checkpoint's Gated DeltaNet at tp1.
H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
ROWS8 = q8_state_rows(V, K)
#: GDN layers whose states the cache manager v2 coalesces into one slot page.
LAYERS = 48
#: int16 rows: round-to-nearest of `h / scale` errs by at most half a code, i.e. 0.5 per
#: unit of the row's scale (`amax / 32767`).
Q8_ELEM_BOUND = 0.5


def q8_pool(slots: int, layout: str, filled, gen: torch.Generator, std: float = 0.2):
    """`(backing, pool)`: an int8-typed pool view `[slots, 48, 260, 128]` (`layout` "compact"
    or "coalesced") whose `filled` slots hold quantized `std * randn` states; every other
    byte is zero (a zero state)."""
    if layout == "coalesced":
        backing = torch.zeros(slots, LAYERS, HV, ROWS8, K, dtype=torch.int8, device=DEV)
        pool = backing[:, 0]
    else:
        assert layout == "compact", layout
        backing = torch.zeros(slots, HV, ROWS8, K, dtype=torch.int8, device=DEV)
        pool = backing
    filled = sorted(set(filled))
    states = torch.randn(len(filled), HV, V, K, generator=gen, device=DEV) * std
    store_slots(pool, filled, states)
    return backing, pool


def scales_of(pool: torch.Tensor, slots) -> torch.Tensor:
    """fp32 row scales `[len(slots), HV, V]` of the named slots."""
    return torch.stack(
        [pool[s, :, 2 * V :].contiguous().view(torch.float32).reshape(HV, V) for s in slots]
    )


def assert_q8_states_close(got_pool, want_pool, slots, *, fp32_slack: float = 1e-6) -> None:
    """The named slots' dequantized states, element by element.

    Each pool is within one int16 rounding of its own fp32 state (`scale / 2` per element, the
    row's scale), and the two fp32 states differ by at most `fp32_slack` (absolute: the reorder
    of one step's fp32 sums, ~2^-20 of states of magnitude < 4), so the pools differ by at most
    the sum. The row scales (`amax / 32767`) then agree to 1e-5 relative plus
    `fp32_slack / 32767`."""
    slots = [s for s in slots if s >= 0]
    got, want = dequantize_slots(got_pool, slots), dequantize_slots(want_pool, slots)
    s_got, s_want = scales_of(got_pool, slots), scales_of(want_pool, slots)
    bound = (s_got + s_want)[..., None] * Q8_ELEM_BOUND * 1.001 + fp32_slack
    worst = ((got - want).abs() - bound).amax().item()
    assert worst <= 0, f"state off by {worst:.3e} past the int16 band"
    torch.testing.assert_close(s_got, s_want, rtol=1e-5, atol=fp32_slack / Q8_QMAX)


def assert_q8_rows_close(got_pool, want_pool, slots, *, row_rel: float, abs_slack: float) -> None:
    """The named slots' dequantized states, per value row (L2 over K): each pool is
    within `sqrt(K) * scale / 2` of its own fp32 row, and the two fp32 rows
    differ by at most `row_rel * |row| + abs_slack` (the chunked kernel's reordered sum
    against the sequential recurrence)."""
    slots = [s for s in slots if s >= 0]
    got, want = dequantize_slots(got_pool, slots), dequantize_slots(want_pool, slots)
    s_got, s_want = scales_of(got_pool, slots), scales_of(want_pool, slots)
    err = (got - want).pow(2).sum(-1).sqrt()
    norm = want.pow(2).sum(-1).sqrt()
    bound = math.sqrt(K) * (s_got + s_want) * Q8_ELEM_BOUND * 1.001 + row_rel * norm + abs_slack
    worst = (err - bound).amax().item()
    assert worst <= 0, f"a state row off by {worst:.3e} past the band"


def assert_untouched(pool, before, named) -> None:
    """Every slot the call did not name is bitwise what it was."""
    named = {s for s in named if s >= 0}
    others = [s for s in range(pool.shape[0]) if s not in named]
    idx = torch.tensor(others, device=pool.device)
    assert torch.equal(pool[idx], before[idx]), "a slot the call did not name was written"


def decode_inputs(rows: int, gen: torch.Generator) -> dict:
    """The decode step's operands as views of its projections: q, k `[1, N, 16, 128]`,
    v `[1, N, 48, 128]` and z `[N, 48, 128]` of one `[N, 16384]` bf16 block; a, b
    `[N, 48]` of one `[N, 96]` block; the layer's A_log, dt_bias (fp32) and gated-norm
    weight (bf16)."""
    qkvz = (torch.randn(rows, QKVZ_DIM, generator=gen, device=DEV) * 0.7).to(torch.bfloat16)
    ba = (torch.randn(rows, 2 * HV, generator=gen, device=DEV) * 1.5).to(torch.bfloat16)
    A = torch.exp(torch.rand(HV, generator=gen, device=DEV) * 6.0 - 5.3)
    return dict(
        qkvz=qkvz,
        ba=ba,
        q=qkvz[:, :KEY_DIM].view(1, rows, H, K),
        k=qkvz[:, KEY_DIM : 2 * KEY_DIM].view(1, rows, H, K),
        v=qkvz[:, 2 * KEY_DIM : CONV_DIM].view(1, rows, HV, V),
        z=qkvz[:, CONV_DIM:].view(rows, HV, V),
        a=ba[:, HV:],
        b=ba[:, :HV],
        A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, generator=gen, device=DEV) * 0.5 - 1.0).float(),
        norm_weight=(1.0 + 0.1 * torch.randn(V, generator=gen, device=DEV)).to(torch.bfloat16),
    )


def scattered_slots(rows: int, slots: int, gen: torch.Generator, pad: bool = False) -> list[int]:
    """`rows` distinct slots of a `slots`-row pool in random order; row 1 a pad (-1) when
    `pad`."""
    picked = torch.randperm(slots, generator=gen, device=DEV)[:rows].tolist()
    if pad and rows > 1:
        picked[1] = -1
    return picked


def coalesced_slots(rows: int) -> list[int]:
    """Slots of the coalesced cells: slot 100 (past 2^31 bytes at 48 layers' stride) first,
    then low slots."""
    assert 100 * LAYERS * HV * ROWS8 * K > 2**31
    return [100] + [3 * i + 1 for i in range(rows - 1)]
