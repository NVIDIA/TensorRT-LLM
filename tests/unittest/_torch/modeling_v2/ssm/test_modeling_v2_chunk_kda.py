# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the chunk_kda catalog entry: Kimi K3's KDA prefill below four 64-token chunks."""

from typing import Dict, List

import _kda_reference as kr
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.chunk_kda import chunk_kda
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_prefill import kda_prefill
from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices

assert torch.cuda.is_available(), "chunk_kda requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

HEAD_DIM = kr.HEAD_DIM
_TIME_MAJOR = ("q", "k", "v", "g", "beta")


def _call(inp: Dict, cu_seqlens: torch.Tensor, initial_state, **overrides):
    """Kimi K3's layer's call (``prefill_chunk_kda``'s FLA path): o and the final V-first states."""
    args = dict(
        scale=kr.SCALE,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=kr.LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
    )
    args.update(overrides)
    with torch.inference_mode():
        return chunk_kda(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], **args)


def _states(heads: int, fresh: List[bool], gen: torch.Generator) -> torch.Tensor:
    """Dense fp32 V-first initial states: zero rows for fresh sequences (the layer zeroes them)."""
    rows = [
        torch.zeros(heads, HEAD_DIM, HEAD_DIM, device="cuda")
        if f
        else torch.randn(heads, HEAD_DIM, HEAD_DIM, generator=gen, device="cuda") * 0.1
        for f in fresh
    ]
    return torch.stack(rows)


def _slice(inp: Dict, sl: slice) -> Dict:
    return {name: t[:, sl].contiguous() if name in _TIME_MAJOR else t for name, t in inp.items()}


def test_kimi_k3_short_varlen_prefill() -> None:
    # Kimi K3's layer routes a prefill batch below four chunks here, at its TP16 rank slice (6 heads)
    # and TP4 (24): fresh sequences and continuations, lengths not chunk multiples, one token.
    cases = (
        (6, 0, [100, 37], [True, False]),
        (24, 1, [150], [False]),
        (6, 2, [1], [True]),
        (6, 3, [17, 5, 64], [False, True, False]),
    )
    for heads, seed, seq_lens, fresh in cases:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        inp = kr.kimi_k3_inputs(heads, sum(seq_lens), gen)
        cu_seqlens = kr.cu_seqlens_of(seq_lens)
        assert prepare_chunk_indices(cu_seqlens, kr.CHUNK).shape[0] < 4
        h0 = _states(heads, fresh, gen)
        h0_before = h0.clone()
        o, ht = _call(inp, cu_seqlens, h0)
        assert o.shape == inp["v"].shape and o.dtype == inp["q"].dtype
        assert ht.shape == h0.shape and ht.dtype == torch.float32
        assert torch.equal(h0, h0_before), "initial_state was written"
        start = 0
        for i, length in enumerate(seq_lens):
            sl = slice(start, start + length)
            ref_o, ref_s = kr.recurrent_reference(
                inp["q"][0, sl],
                inp["k"][0, sl],
                inp["v"][0, sl],
                inp["g"][0, sl],
                inp["beta"][0, sl],
                h0[i],
                inp["A_log"],
                inp["dt_bias"],
            )
            # Chunked evaluation with bf16 operands against an fp64 recurrence: within 2e-2 of each
            # sequence's largest output, 1e-2 of its state's largest entry.
            kr.assert_scaled_close(o[0, sl], ref_o, atol=2e-2)
            kr.assert_scaled_close(ht[i], ref_s, atol=1e-2)
            start += length


def test_continuation_equals_one_call() -> None:
    # A 192-token sequence prefilled as 128 + 64 (the second call starting from the first's final
    # state) matches one 192-token call; the halves in the wrong order leave another state.
    heads = 6
    gen = torch.Generator(device="cuda").manual_seed(4)
    inp = kr.kimi_k3_inputs(heads, 192, gen)
    zero = torch.zeros(1, heads, HEAD_DIM, HEAD_DIM, device="cuda")
    o_one, h_one = _call(inp, kr.cu_seqlens_of([192]), zero)
    first, second = _slice(inp, slice(0, 128)), _slice(inp, slice(128, 192))
    o_a, h_a = _call(first, kr.cu_seqlens_of([128]), zero)
    o_b, h_b = _call(second, kr.cu_seqlens_of([64]), h_a)
    # The split falls on a chunk boundary and the recurrence carries fp32 across chunks either way.
    kr.assert_scaled_close(torch.cat([o_a, o_b], dim=1), o_one, atol=2e-3)
    kr.assert_scaled_close(h_b, h_one, atol=1e-2)
    # Negative control: the second part first raises nothing and ends elsewhere.
    _, h_x = _call(second, kr.cu_seqlens_of([64]), zero)
    _, h_y = _call(first, kr.cu_seqlens_of([128]), h_x)
    assert (h_y - h_one).abs().max() > 1e-2


def test_agrees_with_kda_prefill_from_four_chunks() -> None:
    # From four chunks up the layer runs ssm/kda_prefill instead; on a batch either path can take
    # (four chunks) the two agree within the tolerance each has against the fp64 reference.
    heads = 6
    seq_lens = [100, 64, 37]
    gen = torch.Generator(device="cuda").manual_seed(5)
    inp = kr.kimi_k3_inputs(heads, sum(seq_lens), gen)
    cu_seqlens = kr.cu_seqlens_of(seq_lens)
    h0 = _states(heads, [True, False, False], gen)
    o_fla, h_fla = _call(inp, cu_seqlens, h0)
    pool = h0.clone()
    o_op = kda_prefill(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        pool,
        torch.arange(len(seq_lens), dtype=torch.int32, device="cuda"),
        kr.SCALE,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, kr.CHUNK),
        chunk_size=kr.CHUNK,
        safe_gate=True,
        lower_bound=kr.LOWER_BOUND,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
    )
    start = 0
    for i, length in enumerate(seq_lens):
        sl = slice(start, start + length)
        kr.assert_scaled_close(o_fla[0, sl], o_op[0, sl], atol=2e-2)
        kr.assert_scaled_close(h_fla[i], pool[i], atol=1e-2)
        start += length


def test_none_initial_state_is_zero_and_calls_repeat() -> None:
    # The layer passes initial_state=None when no sequence of the batch continues: that equals
    # zero states, and a repeated call (FLA's kernels autotune once per process) repeats bit for bit.
    heads = 6
    seq_lens = [90, 70]
    gen = torch.Generator(device="cuda").manual_seed(6)
    inp = kr.kimi_k3_inputs(heads, sum(seq_lens), gen)
    cu_seqlens = kr.cu_seqlens_of(seq_lens)
    o_none, h_none = _call(inp, cu_seqlens, None)
    o_zero, h_zero = _call(inp, cu_seqlens, _states(heads, [True, True], gen))
    assert torch.equal(o_none, o_zero) and torch.equal(h_none, h_zero)
    o_again, h_again = _call(inp, cu_seqlens, None)
    assert torch.equal(o_again, o_none) and torch.equal(h_again, h_none)


def test_rejects_out_of_contract() -> None:
    heads = 6
    gen = torch.Generator(device="cuda").manual_seed(7)
    inp = kr.kimi_k3_inputs(heads, 80, gen)
    cu_seqlens = kr.cu_seqlens_of([40, 40])
    h0 = _states(heads, [True, True], gen)
    two_rows = {
        name: t.view(2, 40, *t.shape[2:]) if name in _TIME_MAJOR else t for name, t in inp.items()
    }
    bad = {
        "a batch of two rows with cu_seqlens": (two_rows, dict(initial_state=None)),
        "one initial state for two sequences": (inp, dict(initial_state=h0[:1])),
        "lower_bound below -5": (inp, dict(initial_state=h0, lower_bound=-6.0)),
        "safe_gate without lower_bound": (inp, dict(initial_state=h0, lower_bound=None)),
    }
    for name, (args, overrides) in bad.items():
        initial_state = overrides.pop("initial_state")
        try:
            _call(args, cu_seqlens, initial_state, **overrides)
        except ValueError:
            continue
        raise AssertionError(f"{name}: the call accepted an out-of-contract input")
    try:
        _call(inp, cu_seqlens, h0.to(torch.bfloat16))
    except AssertionError as e:
        assert "float32" in str(e)
    else:
        raise AssertionError("a bf16 initial_state was accepted")
