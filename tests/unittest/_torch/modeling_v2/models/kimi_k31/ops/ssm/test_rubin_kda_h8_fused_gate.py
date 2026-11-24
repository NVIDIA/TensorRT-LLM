# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 verify variants of rubin_kda_mtp_verify at helix8's shapes (sm_107).

Above 32 verify rows at 12 KDA heads the verify can write the MXFP8 operand of o_proj. Two
variants must reproduce the plain one (the full `rubin_kda_fab` kernel, then the two-CTA MXFP8
verify fed its forget gate) bit for bit: E4M3 codes, UE8M0 scales, committed state, conv windows
and replay caches, over chained rounds with mixed acceptance, for an FP32 and a BF16 state:

* the fused forget gate: `rubin_kda_fab` in f_a mode, then the v2 kernel with `fa` / `w_fb` and
  its MXFP8 epilogue (`quantize_output`, optionally a PDL launch);
* the MXFP8 two-CTA kernel as a PDL launch with the early release (`mxfp8_pdl_early_trigger`).
"""

import pytest
import torch
from _rubin_kda_verify_inputs import (
    MUTATED,
    K,
    bits,
    clone_mutated,
    fab_args,
    scale_bytes,
    verify_args,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_fab import (
    rubin_kda_fab,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)

HEADS = 12
# helix8 GEN b16k5 (96 rows) and b24k5 (144 rows); 33 rows, the smallest MXFP8 verify.
SHAPES = [(16, 5), (24, 5), (11, 2)]
STATES = [torch.float32, torch.bfloat16]


def _layer(batch, num_spec, state_dtype, seed):
    spec = dict(N=batch, H=HEADS, num_spec=num_spec, pool=2 * batch, out="mxfp8")
    args = verify_args(dict(spec, state_dtype=str(state_dtype).removeprefix("torch.")), seed)
    rows = args["x_q"].shape[1]
    fab = fab_args(
        dict(rows=rows, heads=HEADS, hidden=7168, fa_only=False, multi_tile=rows > 128), seed
    )
    return args, fab


def _plain(args, fab, **flags):
    beta, gate = rubin_kda_fab(**fab)
    rows = beta.shape[0]
    return rubin_kda_mtp_verify(
        **dict(args, g=gate.view(1, rows, HEADS, K), beta=beta.view(1, rows, HEADS), **flags)
    )


def _fused_gate(args, fab, use_pdl):
    bfa = rubin_kda_fab(**dict(fab, fa_only=True))
    rows = bfa.shape[0]
    return rubin_kda_mtp_verify(
        **dict(
            args,
            g=None,
            fa=bfa[:, :K].view(1, rows, K),
            w_fb=fab["w_fb"],
            beta=bfa[:, K : K + HEADS].view(1, rows, HEADS),
            use_pdl=use_pdl,
        )
    )


def _rounds(batch, num_spec, state_dtype, run_a, run_b):
    args, fab = _layer(batch, num_spec, state_dtype, seed=100 * batch + num_spec)
    a, b = clone_mutated(args), clone_mutated(args)
    rows = batch * (num_spec + 1)
    gen = torch.Generator(device="cpu").manual_seed(3)
    with validating(rubin_kda_fab, rubin_kda_mtp_verify):
        for rnd in range(4):
            (codes_a, sf_a), (codes_b, sf_b) = run_a(a, fab), run_b(b, fab)
            torch.cuda.synchronize()
            assert torch.equal(bits(codes_a), bits(codes_b)), f"round {rnd} E4M3"
            assert torch.equal(scale_bytes(sf_a, rows, HEADS), scale_bytes(sf_b, rows, HEADS)), (
                f"round {rnd} scales"
            )
            for name in MUTATED:
                assert torch.equal(bits(a[name]), bits(b[name])), f"round {rnd} {name}"
            if rnd == 0:  # the verify committed a new state
                assert not torch.equal(bits(a["recurrent_state"]), bits(args["recurrent_state"]))
            slots = args["ssm_state_indices"].long()
            accepted = torch.randint(0, num_spec + 1, (batch,), generator=gen).to(torch.int32)
            a["num_accepted_tokens"][slots] = accepted.cuda()  # shared by a and b


@pytest.mark.parametrize("state_dtype", STATES, ids=["fp32-state", "bf16-state"])
@pytest.mark.parametrize("use_pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("batch,num_spec", SHAPES)
@torch.inference_mode()
def test_fused_gate_mxfp8_is_bitwise_the_plain_verify(batch, num_spec, use_pdl, state_dtype):
    _rounds(
        batch,
        num_spec,
        state_dtype,
        lambda args, fab: _fused_gate(args, fab, use_pdl),
        _plain,
    )


@pytest.mark.parametrize("state_dtype", STATES, ids=["fp32-state", "bf16-state"])
@pytest.mark.parametrize("batch,num_spec", SHAPES)
@torch.inference_mode()
def test_mxfp8_pdl_early_release_is_bitwise_the_plain_launch(batch, num_spec, state_dtype):
    _rounds(
        batch,
        num_spec,
        state_dtype,
        lambda args, fab: _plain(args, fab, use_pdl=True, mxfp8_pdl_early_trigger=True),
        _plain,
    )


@pytest.mark.parametrize("batch,num_spec", [(16, 5)])
@torch.inference_mode()
def test_fused_gate_mxfp8_cuda_graph(batch, num_spec):
    """The fused-gate MXFP8 verify replays from a CUDA graph with the eager bits."""
    args, fab = _layer(batch, num_spec, torch.float32, seed=9)
    eager, graphed, warm = clone_mutated(args), clone_mutated(args), clone_mutated(args)
    _fused_gate(warm, fab, use_pdl=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes_g, sf_g = _fused_gate(graphed, fab, use_pdl=True)
    graph.replay()
    codes_e, sf_e = _fused_gate(eager, fab, use_pdl=True)
    torch.cuda.synchronize()
    rows = batch * (num_spec + 1)
    assert torch.equal(bits(codes_g), bits(codes_e))
    assert torch.equal(scale_bytes(sf_g, rows, HEADS), scale_bytes(sf_e, rows, HEADS))
    for name in MUTATED:
        assert torch.equal(bits(graphed[name]), bits(eager[name])), name
