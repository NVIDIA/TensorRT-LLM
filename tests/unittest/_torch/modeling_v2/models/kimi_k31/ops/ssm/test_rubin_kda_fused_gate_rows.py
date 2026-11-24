# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's fused forget gate against its f_b GEMM path at the generation rows it serves (sm_107).

With ``kda_verify_fused_fb`` helix8 passes the f_a columns of its stacked projection and the f_b
weight to rubin_kda_mtp_verify (``fa`` / ``w_fb``) instead of the gate ``rubin_linear`` (cuBLAS
``F.linear``) computes from them, with the launch it has today: PDL with the early release and the
``mxfp8_exact`` output, on the v2 kernel within one wave of (head, request) tiles. The fused call
must write today's bytes: the E4M3 codes, the scale bytes of every written row and every pool word
the verify writes, over chained rounds with mixed acceptance,

* eagerly at k5 for 2 to 17 requests (split_v 2 up to 5 requests, one CTA per tile above, the
  one-wave edge), and at 4 and 9 requests for k4, k6 and k7;
* in CUDA graph replays of [f_b GEMM -> verify] against [verify with f_a], the outputs poisoned and
  the pools restored before each replay.

``test_rubin_kda_fused_gate_linear.py`` covers the same contract at the targets' cells against the
plain launch.
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import stacked_columns
from _rubin_kda_verify_inputs import MUTATED, K, bits, clone_mutated, scale_bytes, verify_args

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (
    rubin_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)

#: helix8's KDA heads per rank.
H = 12
#: (requests, num_spec): k5 over helix8's served requests, and the other draft lengths at b4 / b9.
CASES = [(b, 5) for b in (2, 3, 4, 5, 6, 9, 13, 17)] + [(b, k) for b in (4, 9) for k in (4, 6, 7)]
ROUNDS = 4
REPLAYS = 20


def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _case(requests: int, num_spec: int, seed: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """helix8's verify arguments (the stacked views, ``mxfp8_exact``, PDL with the early release)
    with the f_b operands: the f_a columns of the same stacked projection rows and a BF16 f_b
    weight ``[H * 128, 128]``."""
    args = verify_args(
        dict(
            N=requests,
            H=H,
            num_spec=num_spec,
            pool=2 * requests + 3,
            stacked=True,
            out="mxfp8_exact",
            use_pdl=True,
            early_trigger=True,
        ),
        seed,
    )
    x_q = args["x_q"]
    rows = x_q.shape[1]
    f_a = torch.as_strided(
        x_q, (rows, K), (x_q.stride(1), 1), x_q.storage_offset() + stacked_columns(H)["f_a"]
    )
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    w_fb = (torch.randn(H * K, K, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    return args, f_a, w_fb


def _call(args: dict, f_a: torch.Tensor, w_fb: torch.Tensor, fused: bool):
    """helix8's verify: [f_b GEMM -> verify] today, or the verify with f_a and the f_b weight."""
    if fused:
        gate = dict(g=None, fa=f_a.unsqueeze(0), w_fb=w_fb)
    else:
        gate = dict(
            g=rubin_linear(f_a, w_fb, False).view(1, f_a.shape[0], H, K), fa=None, w_fb=None
        )
    return rubin_kda_mtp_verify(**dict(args, **gate))


def _assert_same(want, want_args: dict, got, got_args: dict, rows: int, where: str) -> None:
    assert torch.equal(bits(got[0]), bits(want[0])), f"{where}: E4M3 codes"
    assert torch.equal(scale_bytes(got[1], rows, H), scale_bytes(want[1], rows, H)), (
        f"{where}: scales"
    )
    for name in MUTATED:
        assert torch.equal(bits(got_args[name]), bits(want_args[name])), f"{where}: {name}"


@pytest.mark.parametrize("requests,num_spec", CASES)
@torch.inference_mode()
def test_fused_gate_writes_the_f_b_gemm_paths_bytes(requests, num_spec) -> None:
    if H * requests > _sms():
        pytest.skip("past one wave of tiles the two-CTA kernel serves the rows")
    args, f_a, w_fb = _case(requests, num_spec, seed=17 * requests + num_spec)
    today, fused = clone_mutated(args), clone_mutated(args)
    rows = requests * (num_spec + 1)
    gen = torch.Generator(device="cpu").manual_seed(requests)
    slots = args["ssm_state_indices"].long()
    for rnd in range(ROUNDS):
        want = _call(today, f_a, w_fb, False)
        got = _call(fused, f_a, w_fb, True)
        torch.cuda.synchronize()
        _assert_same(want, today, got, fused, rows, f"round {rnd}")
        accepted = torch.randint(0, num_spec + 1, (requests,), generator=gen).to(torch.int32)
        today["num_accepted_tokens"][slots] = accepted.cuda()
        fused["num_accepted_tokens"][slots] = accepted.cuda()


@pytest.mark.parametrize("requests", (2, 9, 17))
@torch.inference_mode()
def test_fused_gate_graph_replays_write_the_f_b_gemm_paths_bytes(requests) -> None:
    if H * requests > _sms():
        pytest.skip("past one wave of tiles the two-CTA kernel serves the rows")
    args, f_a, w_fb = _case(requests, 5, seed=300 + requests)
    rows = requests * 6
    source = clone_mutated(args)
    arms = {}
    for fused in (False, True):
        _call(clone_mutated(args), f_a, w_fb, fused)  # compile and tune outside the capture
        torch.cuda.synchronize()
        live = clone_mutated(args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _call(live, f_a, w_fb, fused)
        arms[fused] = (graph, live, out)
    for rep in range(REPLAYS):
        for graph, live, out in arms.values():
            for name in MUTATED:
                live[name].copy_(source[name])
            out[0].view(torch.uint8).fill_(0x7F)  # E4M3 NaN: every code must be rewritten
            out[1].view(torch.uint8).fill_(0xFF)
            graph.replay()
        torch.cuda.synchronize()
        _, today_live, today_out = arms[False]
        _, fused_live, fused_out = arms[True]
        _assert_same(today_out, today_live, fused_out, fused_live, rows, f"replay {rep}")
    eager = clone_mutated(source)
    want = _call(eager, f_a, w_fb, False)
    torch.cuda.synchronize()
    _, fused_live, fused_out = arms[True]
    _assert_same(want, eager, fused_out, fused_live, rows, "graph vs eager")
