# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F3 on a GPU: REJ_MC's window forms (``minimax_m31_chain_speculative_sampling_mc_win`` / ``_slots_win``) return
the stock op's tokens and counts for offset = base[k] + add, and write window = base + add bit for bit (the int64
add, wrap-around included), eagerly and from a CUDA graph. ``ops/sampling/m31_chain_spec_mc`` takes a pending window
only for its accept offset; every decline writes the window before the stock op could read the offset, and the slot
override leaves a window it does not take to the plain override."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_chain_spec_mc as entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

K, V = 7, 200064
SLOTS, KMAX = 5, 8
CASES = ("realistic", "same", "far", "ties", "onehot")
I64 = dict(dtype=torch.int64, device="cuda")


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("REJ_MC is built for sm_107 here")
    assert entry.window_ops_built(), "this build has no REJ_MC window ops"
    assert hasattr(torch.ops.trtllm, "chain_speculative_sampling")
    pytest.importorskip("flashinfer.sampling")


def _probs(logits, top_p=0.95):
    import flashinfer.sampling

    p = torch.softmax(logits, -1)
    return (
        flashinfer.sampling.top_p_renorm_probs(p.view(-1, V).contiguous(), top_p)
        .view(logits.shape)
        .contiguous()
    )


def _make(case, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rl(shape, scale=3.0):
        return torch.randn(*shape, V, generator=g, device="cuda") * scale

    dl = rl((1, K))
    if case == "realistic":
        tl = torch.cat([dl, rl((1, 1))], 1) + rl((1, K + 1), 1.0)
    elif case == "same":
        tl = torch.cat([dl, rl((1, 1))], 1)
    elif case == "far":
        tl = rl((1, K + 1))
        tl[:, :, :8] += 20
    elif case == "ties":
        dl = torch.round(dl * 2) / 2
        tl = torch.round(torch.cat([dl, rl((1, 1))], 1) * 2) / 2
    else:
        tl = torch.full((1, K + 1, V), -1e4, device="cuda")
        tl[:, :, 1234] = 0
    dp, tp = _probs(dl), _probs(tl)
    ids = torch.multinomial(dp.view(-1, V), 1, generator=g).view(1, K).to(torch.int32).contiguous()
    seeds = torch.randint(0, 2**31, (1,), generator=g, **I64)
    return dp, ids, tp, seeds


def _window(seed, wrap):
    """R1's operands: offsets[:K + 1] and first * SPAN; ``wrap`` puts the bases where the add overflows int64."""
    g = torch.Generator(device="cuda").manual_seed(seed + 99)
    if wrap:
        base = (2**63 - 1) - torch.randint(0, 2**20, (K + 1,), generator=g, **I64)
    else:
        base = torch.randint(0, 2**40, (K + 1,), generator=g, **I64)
    return base.contiguous(), (K + 1) * (2**40 + 12345)


def _stock(dp, ids, tp, seeds, offset):
    return torch.ops.trtllm.chain_speculative_sampling(dp, ids, tp, seeds, offset.contiguous())


def _same(got, ref, what):
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), what


def _slots(dp, slot=3):
    buf = torch.full((SLOTS, KMAX, V), 0.5, device="cuda")
    buf[slot, :K] = dp[0]
    return buf, torch.tensor([slot], **I64)


@pytest.mark.parametrize("wrap", [False, True], ids=["plain", "wraps"])
@pytest.mark.parametrize("case", CASES)
def test_window_ops_are_the_plain_ops_at_the_added_offset(case, wrap):
    _gpu()
    ws = entry.prebuild(torch.device("cuda"))
    g = entry.ctas_per_row(1, entry.sm_count(torch.device("cuda")))
    for seed in range(3):
        dp, ids, tp, seeds = _make(case, 100 * seed + 7)
        base, add = _window(seed, wrap)
        want = base + add
        ref = _stock(dp, ids, tp, seeds, want[K : K + 1])
        win = torch.full((K + 1,), -1, **I64)
        got = family_ops().minimax_m31_chain_speculative_sampling_mc_win(
            dp, ids, tp, seeds, win[K : K + 1], ws, g, base, add, win, K
        )
        _same(got, ref, (case, wrap, seed))
        assert torch.equal(win, want), (case, wrap, seed)
        buf, slot_ids = _slots(dp)
        win2 = torch.full((K + 1,), -1, **I64)
        got2 = family_ops().minimax_m31_chain_speculative_sampling_mc_slots_win(
            buf, slot_ids, K, ids, tp, seeds, win2[K : K + 1], ws, g, base, add, win2, K
        )
        _same(got2, ref, ("slots", case, wrap, seed))
        assert torch.equal(win2, want), ("slots", case, wrap, seed)


def test_window_op_in_a_cuda_graph():
    _gpu()
    ws = entry.prebuild(torch.device("cuda"))
    g = entry.ctas_per_row(1, entry.sm_count(torch.device("cuda")))
    dp, ids, tp, seeds = (t.clone() for t in _make("realistic", 1))
    base, add = _window(1, False)
    base = base.clone()
    win = torch.full((K + 1,), -1, **I64)
    op = family_ops().minimax_m31_chain_speculative_sampling_mc_win
    op(dp, ids, tp, seeds, win[K : K + 1], ws, g, base, add, win, K)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(dp, ids, tp, seeds, win[K : K + 1], ws, g, base, add, win, K)
    for step, case in enumerate(CASES):
        fresh = _make(case, 50 + step)
        for dst, src in zip((dp, ids, tp, seeds), fresh):
            dst.copy_(src)
        base.copy_(_window(60 + step, step % 2 == 1)[0])
        graph.replay()
        torch.cuda.synchronize()
        want = base + add
        _same(out, _stock(dp, ids, tp, seeds, want[K : K + 1]), case)
        assert torch.equal(win, want), case


def test_the_override_takes_only_its_accept_offset_and_flushes_on_decline():
    _gpu()
    dp, ids, tp, seeds = _make("realistic", 3)
    base, add = _window(3, True)
    want = base + add
    ref = _stock(dp, ids, tp, seeds, want[K : K + 1])
    # Taken: the window written by the launch.
    win = torch.full((K + 1,), -1, **I64)
    entry.defer_window(win, base, add)
    _same(entry.chain_spec_mc(dp, ids, tp, seeds, win[K : K + 1]), ref, "taken")
    torch.cuda.synchronize()
    assert torch.equal(win, want) and entry._WINDOW[0] is None
    # Declined (a non-contiguous draft): None, and the window written before returning.
    wide = torch.zeros(1, K, V + 4, device="cuda")
    wide[..., :V] = dp
    win = torch.full((K + 1,), -1, **I64)
    entry.defer_window(win, base, add)
    assert entry.chain_spec_mc(wide[..., :V], ids, tp, seeds, win[K : K + 1]) is None
    torch.cuda.synchronize()
    assert torch.equal(win, want) and entry._WINDOW[0] is None
    # Another offset: the pending window is neither used nor written; flush_window writes it.
    win = torch.full((K + 1,), -1, **I64)
    entry.defer_window(win, base, add)
    _same(entry.chain_spec_mc(dp, ids, tp, seeds, want[K : K + 1].clone()), ref, "other offset")
    torch.cuda.synchronize()
    assert torch.all(win == -1) and entry._WINDOW[0] is not None
    entry.flush_window(win)
    torch.cuda.synchronize()
    assert torch.equal(win, want) and entry._WINDOW[0] is None


def test_the_slot_override_leaves_a_window_it_declines_to_the_plain_one():
    _gpu()
    dp, ids, tp, seeds = _make("far", 4)
    base, add = _window(4, False)
    want = base + add
    ref = _stock(dp, ids, tp, seeds, want[K : K + 1])
    buf, slot_ids = _slots(dp)
    win = torch.full((K + 1,), -1, **I64)
    entry.defer_window(win, base, add)
    _same(
        entry.chain_spec_mc_slots(buf, slot_ids, K, ids, tp, seeds, win[K : K + 1]),
        ref,
        "slots taken",
    )
    torch.cuda.synchronize()
    assert torch.equal(win, want) and entry._WINDOW[0] is None
    odd = torch.full(
        (SLOTS, KMAX, V + 2), 0.5, device="cuda"
    )  # a row width the slot op does not take
    odd[3, :K, :V] = dp[0]
    win = torch.full((K + 1,), -1, **I64)
    entry.defer_window(win, base, add)
    assert entry.chain_spec_mc_slots(odd, slot_ids, K, ids, tp, seeds, win[K : K + 1]) is None
    assert entry._WINDOW[0] is not None
    _same(entry.chain_spec_mc(dp, ids, tp, seeds, win[K : K + 1]), ref, "then the plain override")
    torch.cuda.synchronize()
    assert torch.equal(win, want) and entry._WINDOW[0] is None
