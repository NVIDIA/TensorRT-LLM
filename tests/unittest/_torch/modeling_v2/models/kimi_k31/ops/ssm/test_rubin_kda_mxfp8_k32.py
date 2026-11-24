# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (sm_107): the KDA verify's ``mxfp8_exact`` output is bitwise helix8's o_proj operand.

Without ``kda_oproj_quant_k32`` the helix8 verify writes the BF16 core and ``rubin_mxfp8_linear``
quantizes it with ``torch.ops.trtllm.mxfp8_quantize(x, True, 32)``. With it the verify epilogue
writes that MXFP8 operand itself (``mxfp8_exact``, 32-value scales, on the kernel the BF16 output
takes: v2 within one wave of tiles, two-CTA above). Bitwise checks:

* the entry on the helix8 cells (b16 k5 = 96 rows, b24 k5, the stacked views, b1 k7) and on planted
  output-norm weights and gates that drive the outputs through zeros, denormals, the tiny-scale bands
  and the top binades: E4M3 codes and the scale bytes of every written row equal the quantization of
  the BF16 output, and every tensor the call writes in place equals the BF16 call's;
* the epilogue's code (``mxfp8_exact_store``) on chosen rows (``mxfp8_k32_rows``: every finite
  bf16 block maximum, signed zeros, denormal blocks, ``448 * 2^k`` maxima and their neighbours,
  infinite maxima): codes and the whole scale buffer equal ``mxfp8_quantize``'s.
"""

import pytest
import torch
from _rubin_kda_verify_inputs import MUTATED, bits, clone_mutated, verify_args

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_mxfp8_k32_rows import (
    mxfp8_k32_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: sm_107",
)

CELLS = (
    dict(N=16, H=12, num_spec=5),
    dict(N=24, H=12, num_spec=5),
    dict(N=16, H=12, num_spec=5, stacked=True),
    dict(N=1, H=12, num_spec=7, stacked=True, early_trigger=True),
)


def todays_operand(core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    codes, scales = torch.ops.trtllm.mxfp8_quantize(core, True, 32)
    return codes.view(torch.uint8), scales.view(torch.uint8)


def written_slots(rows: int, heads: int, device) -> torch.Tensor:
    """R128c4 offsets of the scale slots of rows ``< rows`` (the epilogue leaves padding rows)."""
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(4 * heads, device=device).view(1, -1)
    off = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * 512 * heads
    )
    return off.reshape(-1)


def plant(args: dict, seed: int) -> dict:
    """Output-norm weights over four 32-column regimes (fp32 denormal to 2^-127, the 448 * 2^-127
    band, the top binades, plain) and gates down to sigmoid ~ 1e-35 on some rows."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    exps = torch.cat(
        [
            torch.randint(-149, -126, (32,), generator=gen),
            torch.randint(-126, -108, (32,), generator=gen),
            torch.randint(100, 124, (32,), generator=gen),
            torch.randint(-4, 4, (32,), generator=gen),
        ]
    )
    sign = torch.where(torch.rand(128, generator=gen) < 0.25, -1.0, 1.0)
    weight = torch.ldexp(sign, exps).to(torch.float32)
    args = dict(args, onorm_weight=weight.to(args["onorm_weight"].device))
    g = args["onorm_g"]
    rows = g.shape[1]
    pick = torch.randperm(rows, generator=gen)[: rows // 4].to(g.device)
    g[0, pick] = torch.full_like(g[0, pick], -80.0)
    return args


@pytest.mark.parametrize("planted", (False, True), ids=("plain", "planted"))
@pytest.mark.parametrize(
    "spec", CELLS, ids=lambda s: f"N{s['N']}k{s['num_spec']}{'s' if s.get('stacked') else ''}"
)
@torch.inference_mode()
def test_verify_k32_is_todays_operand(spec, planted) -> None:
    args = verify_args(dict(spec, out="bf16"), seed=11)
    if planted:
        args = plant(args, seed=5)
    a_today, a_k32 = clone_mutated(args), clone_mutated(args)
    core = op(**a_today)
    rows, heads = core.shape[1], spec["H"]
    want_codes, want_scales = todays_operand(core.view(rows, heads * 128))
    a_k32.update(quantize_output=True, mxfp8_exact=True)
    got_codes, got_scales = op(**a_k32)
    assert torch.equal(got_codes.view(torch.uint8), want_codes), "E4M3 codes"
    slots = written_slots(rows, heads, core.device)
    assert torch.equal(got_scales.view(torch.uint8)[slots], want_scales[slots]), "scale bytes"
    for name in MUTATED:
        assert torch.equal(bits(a_k32[name]), bits(a_today[name])), name


def sweep_rows(heads: int = 12, seed: int = 3) -> torch.Tensor:
    """bf16 rows [T, heads * 128]: one 32-value block per finite bf16 magnitude and sign of the max
    (the rest uniform below it by bits, random signs), then named blocks (signed zeros, denormal
    blocks, maxima at 448 * 2^k and one bf16 step either side, infinite maxima)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mags = torch.arange(0, 0x7F80, dtype=torch.int64)
    blocks = []
    for sign in (0, 0x8000):
        other = (torch.rand(mags.numel(), 32, generator=gen) * (mags.view(-1, 1) + 1)).long()
        other = torch.minimum(other, mags.view(-1, 1))
        other |= torch.randint(0, 2, other.shape, generator=gen) << 15
        other[torch.arange(mags.numel()), torch.arange(mags.numel()) % 32] = mags | sign
        blocks.append(other)
    named = [
        torch.zeros(32, dtype=torch.int64),
        torch.full((32,), 0x8000),
        torch.full((32,), 0x0001),
    ]
    named.append(torch.tensor([0x7F80 if i == 4 else 0x3F80 + i for i in range(32)]))
    named.append(torch.tensor([0xFF80 if i == 30 else 0x0001 + i for i in range(32)]))
    for m in range(-133, 127):
        v = (
            int(
                torch.tensor(1.75 * 2.0**m, dtype=torch.float32)
                .to(torch.bfloat16)
                .view(torch.int16)
            )
            & 0xFFFF
        )
        for d in (-1, 0, 1):
            b = torch.full((32,), ((v + d) >> 1) & 0x7FFF, dtype=torch.int64)
            b[(m + 200) % 32] = v + d
            named.append(b)
    blocks.append(torch.stack(named))
    allb = torch.cat(blocks)
    pad = (-allb.shape[0]) % (4 * heads)
    allb = torch.cat([allb, torch.zeros(pad, 32, dtype=torch.int64)])
    return allb.to(torch.int16).view(-1, heads * 128).view(torch.bfloat16)


@torch.inference_mode()
def test_epilogue_code_is_todays_quantization_on_chosen_rows() -> None:
    x = sweep_rows().cuda()
    got_codes, got_scales = mxfp8_k32_rows(x)
    want_codes, want_scales = todays_operand(x)
    assert torch.equal(got_codes.view(torch.uint8), want_codes), "E4M3 codes"
    assert torch.equal(got_scales, want_scales), "scale bytes (padding rows zero on both)"
