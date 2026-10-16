# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_fwd catalog entry."""

from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_fwd import (
    causal_conv1d_fwd,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

assert torch.cuda.is_available(), "causal_conv1d_fwd requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

WIDTH = 4  # Kimi K3's short_conv_kernel_size
# Kimi K3's packed q | k | v conv channels per rank: 3 x 128 x heads (6 at TP16, 24 at TP4).
K3_DIMS = (2304, 9216)
NUM_SLOTS = 16
SLOT_PAD = 40  # extra elements per slot: the cache manager's slot stride is wider than the state


class _Pool:
    """A conv-state pool laid out as a cache manager's: [slots, dim, width - 1] views of padded slots."""

    def __init__(self, dim: int, dtype=torch.bfloat16) -> None:
        self.dim = dim
        stride = dim * (WIDTH - 1) + SLOT_PAD
        self.storage = torch.randn(NUM_SLOTS, stride, dtype=dtype, device="cuda")
        self.states = self.storage[:, : dim * (WIDTH - 1)].view(NUM_SLOTS, dim, WIDTH - 1)
        assert self.states.stride(0) == stride


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    states: torch.Tensor,
    seq_lens: List[int],
    slots: List[int],
    has_init: List[bool],
    silu: bool,
):
    """fp32 per-sequence reference: (output [dim, T] in x.dtype, {slot: new state [dim, width - 1]})."""
    out = x.clone()
    new_states = {}
    start = 0
    for length, slot, init in zip(seq_lens, slots, has_init):
        seq = x[:, start : start + length].float()
        if slot == PAD_SLOT_ID:
            start += length
            continue
        hist = states[slot].float() if init else torch.zeros(x.shape[0], WIDTH - 1, device="cuda")
        full = torch.cat([hist, seq], dim=1)
        y = sum(weight[:, k : k + 1].float() * full[:, k : k + length] for k in range(WIDTH))
        if bias is not None:
            y = y + bias.float().unsqueeze(1)
        if silu:
            y = y * torch.sigmoid(y)
        out[:, start : start + length] = y.to(x.dtype)
        new_states[slot] = full[:, -(WIDTH - 1) :].to(states.dtype)
        start += length
    return out, new_states


def _run(dim, seq_lens, slots, has_init, silu=True, use_bias=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    x = torch.randn(dim, total, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(dim, WIDTH, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    bias = torch.randn(dim, generator=gen, device="cuda").to(torch.bfloat16) if use_bias else None
    pool = _Pool(dim)
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int32, device="cuda")
    idx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    init = torch.tensor(has_init, dtype=torch.bool, device="cuda")
    ref_out, ref_states = _reference(x, weight, bias, pool.states, seq_lens, slots, has_init, silu)
    storage_before = pool.storage.clone()
    causal_conv1d_fwd(x, weight, bias, pool.states, qsl, idx, init, silu, PAD_SLOT_ID)
    return x, ref_out, pool, storage_before, ref_states


def _check(x, ref_out, pool, storage_before, ref_states) -> None:
    # fp32 accumulation and one rounding on both sides; the summation order can move a value by
    # one bf16 ulp.
    torch.testing.assert_close(x, ref_out, rtol=2.0**-7, atol=1e-2)
    for slot in range(NUM_SLOTS):
        if slot in ref_states:
            # The new state is the raw inputs (or kept history), copied: bit for bit.
            assert torch.equal(pool.states[slot], ref_states[slot]), f"slot {slot} state"
        else:
            assert torch.equal(pool.storage[slot], storage_before[slot]), f"slot {slot} touched"
    # The slot padding past each state is never written.
    pad = pool.storage[:, pool.dim * (WIDTH - 1) :]
    assert torch.equal(pad, storage_before[:, pool.dim * (WIDTH - 1) :])


def test_kimi_k3_varlen_prefill() -> None:
    # Kimi K3's KDA prefill conv: packed q | k | v channels, width 4, SiLU, no bias, a varlen batch
    # mixing fresh sequences and continuations (chunked prefill or a reused prefix), sequences
    # shorter than the state, scattered slots of a padded pool.
    for dim in K3_DIMS:
        seq_lens = [1, 2, 3, 17, 300, 64]
        slots = [5, 0, 11, 3, 15, 8]
        has_init = [True, False, True, False, True, True]
        _check(*_run(dim, seq_lens, slots, has_init, seed=dim))


def test_chunked_prefill_continuation() -> None:
    # A sequence prefilled in three chunks, each continuing from the state the previous one left at
    # the slot, equals the sequence in one call: the slot carries the conv history across calls.
    dim = K3_DIMS[0]
    whole_x, _, whole_pool, _, _ = _run(dim, [181], [6], [False], seed=1)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(dim, 181, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(dim, WIDTH, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    pool = _Pool(dim)
    outs = []
    start = 0
    for i, length in enumerate((64, 1, 116)):
        chunk = x[:, start : start + length].contiguous()
        causal_conv1d_fwd(
            chunk,
            weight,
            None,
            pool.states,
            torch.tensor([0, length], dtype=torch.int32, device="cuda"),
            torch.tensor([6], dtype=torch.int32, device="cuda"),
            torch.tensor([i > 0], dtype=torch.bool, device="cuda"),
            True,
            PAD_SLOT_ID,
        )
        outs.append(chunk)
        start += length
    assert torch.equal(torch.cat(outs, dim=1), whole_x)
    assert torch.equal(pool.states[6], whole_pool.states[6])

    # Negative control: the same chunks in the wrong order raise nothing and give other numbers.
    pool = _Pool(dim)
    swapped = []
    for i, (lo, hi) in enumerate(((0, 64), (65, 181), (64, 65))):
        chunk = x[:, lo:hi].contiguous()
        causal_conv1d_fwd(
            chunk,
            weight,
            None,
            pool.states,
            torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda"),
            torch.tensor([6], dtype=torch.int32, device="cuda"),
            torch.tensor([i > 0], dtype=torch.bool, device="cuda"),
            True,
            PAD_SLOT_ID,
        )
        swapped.append(chunk)
    assert not torch.equal(swapped[1], whole_x[:, 65:181])


def test_padded_slot_is_skipped() -> None:
    # A sequence whose slot is PAD_SLOT_ID (a CUDA-graph padding row) is skipped: its columns of x
    # and every slot of the pool are left as they were.
    dim = K3_DIMS[0]
    x, ref_out, pool, before, ref_states = _run(
        dim, [9, 33, 5], [2, PAD_SLOT_ID, 7], [False, True, True], seed=2
    )
    _check(x, ref_out, pool, before, ref_states)


def test_bias_and_identity_activation() -> None:
    dim = 768
    for silu in (False, True):
        _check(*_run(dim, [40, 3], [1, 4], [True, False], silu=silu, use_bias=True, seed=3))


def test_channel_last_with_out() -> None:
    # A token-major x (channels contiguous) with an explicit, non-overlapping `out` takes the
    # channel-last kernel and gives the channel-major kernel's numbers; x itself is not written.
    dim = K3_DIMS[0]
    seq_lens, slots, has_init = [17, 130, 2], [3, 9, 12], [True, False, True]
    _, ref_out, _, pool_before, _ = _run(dim, seq_lens, slots, has_init, seed=4)
    gen = torch.Generator(device="cuda").manual_seed(4)
    x_src = torch.randn(dim, sum(seq_lens), generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(dim, WIDTH, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    x_tm = x_src.t().contiguous().t()  # [dim, T] view of a token-major buffer
    assert x_tm.stride(0) == 1
    out = torch.empty(sum(seq_lens), dim, dtype=torch.bfloat16, device="cuda").t()
    pool = _Pool(dim)
    pool.storage.copy_(pool_before)  # the initial states the reference read
    starts = torch.tensor([0, 17, 147, 149], dtype=torch.int32, device="cuda")
    causal_conv1d_fwd(
        x_tm,
        weight,
        None,
        pool.states,
        starts,
        torch.tensor(slots, dtype=torch.int32, device="cuda"),
        torch.tensor(has_init, dtype=torch.bool, device="cuda"),
        True,
        PAD_SLOT_ID,
        out,
    )
    assert torch.equal(x_tm, x_src)
    torch.testing.assert_close(out, ref_out, rtol=2.0**-7, atol=1e-2)


def test_cuda_graph_replay() -> None:
    dim = K3_DIMS[0]
    seq_lens, slots, has_init = [33, 64], [1, 2], [True, False]
    x, _, pool, _, _ = _run(dim, seq_lens, slots, has_init, seed=5)
    weight = torch.randn(dim, WIDTH, device="cuda").to(torch.bfloat16)
    qsl = torch.tensor([0, 33, 97], dtype=torch.int32, device="cuda")
    idx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    init = torch.tensor(has_init, dtype=torch.bool, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        causal_conv1d_fwd(x, weight, None, pool.states, qsl, idx, init, True, PAD_SLOT_ID)
    for seed in (6, 7):
        src = torch.randn(
            dim, 97, generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda"
        )
        state0 = torch.randn_like(pool.storage)
        x.copy_(src.to(torch.bfloat16))
        pool.storage.copy_(state0)
        graph.replay()
        replayed_x, replayed_pool = x.clone(), pool.storage.clone()
        x.copy_(src.to(torch.bfloat16))
        pool.storage.copy_(state0)
        causal_conv1d_fwd(x, weight, None, pool.states, qsl, idx, init, True, PAD_SLOT_ID)
        assert torch.equal(replayed_x, x) and torch.equal(replayed_pool, pool.storage)


def test_rejects_out_of_contract() -> None:
    dim = 768
    x = torch.randn(dim, 10, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(dim, WIDTH, dtype=torch.bfloat16, device="cuda")
    pool = _Pool(dim)
    qsl = torch.tensor([0, 10], dtype=torch.int32, device="cuda")
    idx = torch.tensor([0], dtype=torch.int32, device="cuda")
    init = torch.tensor([False], dtype=torch.bool, device="cuda")
    bad_calls = {
        "int64 query_start_loc": (x, weight, None, pool.states, qsl.long(), idx, init),
        "int64 cache_indices": (x, weight, None, pool.states, qsl, idx.long(), init),
        "int has_initial_state": (x, weight, None, pool.states, qsl, idx, init.int()),
        "weight rows": (x, weight[:-1].contiguous(), None, pool.states, qsl, idx, init),
        "fp32 states": (x, weight, None, pool.states.float(), qsl, idx, init),
        "bias dtype": (x, weight, torch.zeros(dim, device="cuda"), pool.states, qsl, idx, init),
    }
    for name, args in bad_calls.items():
        try:
            causal_conv1d_fwd(*args, True, PAD_SLOT_ID)
        except RuntimeError:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")
