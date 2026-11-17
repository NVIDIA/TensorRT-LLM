# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for a mixed context step's generation-row quantize on the generation stream
(TRTLLM_GLM52_CTX_GEN_SIDE_QUANT, `PrefillTarget._layer_attention`).

o_proj's MXF8 operand `o8 = (q8, sf)` has two writers:
- the fp8 context MHA's MXF8-O epilogue writes rows `[0, tc)` and their block scales;
- `cute_dsl_act_quant_mxf8_rows` writes the generation rows `[tc, tc + g)` and the padding
  scale rows up to the next multiple of 128.
With the switch on they run on different streams at the same time. When `tc % 128 != 0`
they share the last 128-row scale block, whose bytes interleave rows. Each case compares
every operand and scale byte against the serial order (MHA, then the quantize on one stream),
over 100 repetitions. The generation stream is launched before the MHA, and a varying spin
moves the quantize to before, inside and after the MHA.

The generation row counts cover MTP1/3/5 widths: 2, 4 and 6 rows per request.
"""

import os
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernels need an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_prefill as mha_entry,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_act_quant_mxf8_rows as rows_entry,
)

mha = mha_entry.cute_dsl_mha_prefill
quant_rows = rows_entry.cute_dsl_act_quant_mxf8_rows
HEADS, DIM = mha_entry.NUM_HEADS, mha_entry.HEAD_DIM
WIDTH = HEADS * DIM
REPS = int(os.environ.get("GEN_SIDE_QUANT_REPS", "100"))

# (context rows per request, generation rows): tc % 128 in {0, 1, 77, 127}, several requests,
# and generation rows of MTP1 / MTP3 / MTP5 batches (2 / 4 / 6 rows per request).
CASES = [
    ([640], 2),
    ([641], 46),
    ([717], 28),
    ([767], 54),
    ([300, 213, 128], 64),  # tc = 641
    ([1000, 999, 1000, 1000, 999, 1000, 1000, 1001], 6),  # tc = 7999 (% 128 = 63)
    ([7550, 9], 92),  # tc = 7559 (% 128 = 7), 23 MTP1 requests
]


def _cu(lengths) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()


def _pow2(amax: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2((amax.float() / 448.0).clamp_min(2.0**-126))))


def _inputs(new, gen, seed):
    """The MHA's fp8 operands as the target quantizes them (as in test_modeling_v2_cute_dsl_mha_prefill), and the
    generation rows' bf16 attention output with block magnitudes from 2^-10 to 2^5."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = sum(new)
    q = torch.randn(t, HEADS, DIM, generator=g, device="cuda") * 3.0
    k = torch.randn(t, HEADS, DIM, generator=g, device="cuda") * 0.7
    v = torch.randn(t, HEADS, DIM, generator=g, device="cuda")
    s_q = _pow2(q.abs().amax())
    s_k, s_v = _pow2(k.abs().amax(dim=(0, 2))), _pow2(v.abs().amax(dim=(0, 2)))
    args = dict(
        q=(q / s_q).to(torch.float8_e4m3fn),
        k=(k / s_k.view(1, -1, 1)).to(torch.float8_e4m3fn),
        v=(v / s_v.view(1, -1, 1)).to(torch.float8_e4m3fn),
        out=torch.full((t, HEADS, DIM), float("nan"), dtype=torch.bfloat16, device="cuda"),
        cu_seqlens=_cu(new),
        max_seqlen=max(new),
        cu_seqlens_k=None,
        head_scales=mha_entry.fp8_head_scales(float(s_q), s_k, s_v).cuda(),
    )
    x = torch.randn(gen, WIDTH, generator=g, device="cuda")
    exp = torch.randint(-10, 6, (gen, WIDTH // 32), generator=g, device="cuda").float()
    x = (x * torch.exp2(exp).repeat_interleave(32, dim=1)).to(torch.bfloat16)
    return args, x


_HANDLES: dict = {}


def _handles():
    if not _HANDLES:
        dev = torch.cuda.current_device()
        _HANDLES["mha"] = mha_entry.compile(dev, fp8=True, q8=True)
        _HANDLES["rows"] = rows_entry.compile(dev)
    return _HANDLES["mha"], _HANDLES["rows"]


def _o8(t_o):
    q8 = torch.full((t_o, WIDTH), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.full((rows_entry.sf_bytes(t_o, WIDTH),), 0x33, dtype=torch.uint8, device="cuda")
    return q8, sf


def _serial(args, x, tc):
    h_mha, h_rows = _handles()
    q8, sf = _o8(tc + int(x.shape[0]))
    mha(**args, compiled=h_mha, q8_out=(q8, sf))
    quant_rows(x, q8, sf, tc, h_rows)
    torch.cuda.synchronize()
    return q8, sf


def _concurrent(args, x, tc, spin_cycles, gen_stream):
    """As `PrefillTarget._layer_attention` with the switch on: the generation stream waits for the main stream
    (o8 is allocated there), then quantizes while the main stream runs the context MHA; the main stream joins
    it before o_proj would read o8. The generation side is enqueued first; `spin_cycles` delays it."""
    h_mha, h_rows = _handles()
    main = torch.cuda.current_stream()
    q8, sf = _o8(tc + int(x.shape[0]))
    gen_stream.wait_stream(main)
    with torch.cuda.stream(gen_stream):
        if spin_cycles:
            torch.cuda._sleep(spin_cycles)
        quant_rows(x, q8, sf, tc, h_rows)
    mha(**args, compiled=h_mha, q8_out=(q8, sf))
    main.wait_stream(gen_stream)
    torch.cuda.synchronize()
    return q8, sf


@pytest.mark.parametrize("new,gen", CASES, ids=[f"tc{sum(n)}_g{g}" for n, g in CASES])
def test_concurrent_writers_give_the_serial_bytes(new, gen) -> None:
    tc = sum(new)
    args, x = _inputs(new, gen, zlib.crc32(repr((new, gen)).encode()) % 2**31)
    want_q8, want_sf = _serial(args, x, tc)
    # A non-degenerate reference: the MHA rows and the generation rows were both written.
    assert not bool((want_q8[:tc] == 0x5A).all()), "the MHA wrote no operand bytes"
    assert not bool((want_q8[tc:] == 0x5A).all()), "the quantize wrote no operand bytes"
    assert not bool((want_sf == 0x33).any()), (
        "a scale byte up to the padded row count was not written"
    )
    gen_stream = torch.cuda.Stream()
    # Delays from none to well past the MHA (tens of microseconds at these row counts).
    spins = [0, 1_000, 5_000, 20_000, 50_000, 100_000, 200_000]
    for rep in range(REPS):
        q8, sf = _concurrent(args, x, tc, spins[rep % len(spins)], gen_stream)
        assert torch.equal(q8, want_q8), (
            f"rep {rep}: {int((q8 != want_q8).sum())} operand bytes differ from the serial order"
        )
        assert torch.equal(sf, want_sf), (
            f"rep {rep}: {int((sf != want_sf).sum())} scale bytes differ from the serial order"
        )
