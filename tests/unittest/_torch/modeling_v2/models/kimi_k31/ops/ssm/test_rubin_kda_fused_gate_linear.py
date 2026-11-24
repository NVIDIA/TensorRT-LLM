# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused forget gate of rubin_kda_mtp_verify against the targets' separate f_b GEMM (sm_107).

The tep8 / helix8 targets run the verify with ``fa`` / ``w_fb`` as a PDL launch with the early
trigger where the step's ``heads x requests`` tiles fit one wave. That call must reproduce the
plain one, ``rubin_linear`` (cuBLAS ``F.linear``) on the ``f_a`` columns of the stacked
projection followed by the verify without PDL, bit for bit: the BF16 output (tep8) or the
``mxfp8_exact`` codes and scales (helix8), every committed state and cache, over chained rounds
with mixed acceptance.
"""

import pytest
import torch
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

HEADS = 12
# (requests, num_spec, output): TEP8 b1k7 / b4k7 and helix8 b2k5 / b9k5 / b16k5 (one wave of tiles).
CASES = [
    (1, 7, "bf16"),
    (4, 7, "bf16"),
    (2, 5, "mxfp8_exact"),
    (9, 5, "mxfp8_exact"),
    (16, 5, "mxfp8_exact"),
]


def _layer(batch, num_spec, seed):
    args = verify_args(
        dict(N=batch, H=HEADS, num_spec=num_spec, pool=2 * batch, stacked=True), seed
    )
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    w_fb = (torch.randn(HEADS * K, K, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    # f_a: the K columns after q / k / v / g of the stacked projection the views come from.
    f_a = args["beta"][0].as_strided(
        (args["beta"].shape[1], K), (args["beta"].stride(1), 1), args["beta"].storage_offset() - K
    )
    return args, w_fb, f_a


def _call(args, w_fb, f_a, out, fused):
    rows = f_a.shape[0]
    exact = out == "mxfp8_exact"
    if fused:
        gate = dict(g=None, fa=f_a.unsqueeze(0), w_fb=w_fb, use_pdl=True, early_trigger=True)
    else:
        g = rubin_linear(f_a, w_fb, False).view(1, rows, HEADS, K)
        gate = dict(g=g, fa=None, w_fb=None, use_pdl=False, early_trigger=False)
    return rubin_kda_mtp_verify(**dict(args, **gate, quantize_output=exact, mxfp8_exact=exact))


@pytest.mark.parametrize("batch,num_spec,out", CASES)
@torch.inference_mode()
def test_fused_gate_is_bitwise_the_separate_f_b_gemm(batch, num_spec, out):
    args, w_fb, f_a = _layer(batch, num_spec, seed=31 * batch + num_spec)
    plain, fused = clone_mutated(args), clone_mutated(args)
    rows = batch * (num_spec + 1)
    gen = torch.Generator(device="cpu").manual_seed(5)
    for rnd in range(4):
        a, b = _call(plain, w_fb, f_a, out, False), _call(fused, w_fb, f_a, out, True)
        torch.cuda.synchronize()
        if out == "bf16":
            assert torch.equal(bits(a), bits(b)), f"round {rnd} output"
        else:
            assert torch.equal(bits(a[0]), bits(b[0])), f"round {rnd} E4M3"
            assert torch.equal(scale_bytes(a[1], rows, HEADS), scale_bytes(b[1], rows, HEADS)), (
                f"round {rnd} scales"
            )
        for name in MUTATED:
            assert torch.equal(bits(plain[name]), bits(fused[name])), f"round {rnd} {name}"
        slots = args["ssm_state_indices"].long()
        accepted = torch.randint(0, num_spec + 1, (batch,), generator=gen).to(torch.int32)
        plain["num_accepted_tokens"][slots] = accepted.cuda()
        fused["num_accepted_tokens"][slots] = accepted.cuda()
