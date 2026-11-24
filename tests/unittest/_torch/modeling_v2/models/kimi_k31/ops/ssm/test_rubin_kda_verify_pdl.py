# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (sm_107): helix8's KDA verify as f_b's PDL dependent writes the plain launch's bytes.

helix8 computes the forget-gate logits with a cuBLAS GEMM over the f_a columns of its fused
projection (f_b) and, with ``kda_verify_pdl``, launches the verify (``mxfp8_exact`` output, on
the v2 kernel within one wave of (head, request) tiles) as that GEMM's PDL dependent. The PDL
variant only moves the load phase's pool, replay-cache, slot and weight loads before
``griddepcontrol.wait``. Behind a real f_b GEMM on the stacked projection views of helix8 k5
(12 heads; split_v 2 up to 5 requests, one CTA per tile above), every byte the PDL launch writes
is compared with the plain launch's:

* eagerly at 1 to 17 requests: the E4M3 codes, the scale bytes of every written row and every
  tensor the call writes in place (state, conv windows, replay caches);
* in CUDA graph replays of [f_b GEMM -> verify], the outputs poisoned and the pools restored
  before each replay, against each other and against the eager plain launch;
* above one wave of tiles (the two-CTA kernel) the launcher refuses ``mxfp8_exact`` with
  ``use_pdl``.
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import stacked_columns
from _rubin_kda_verify_inputs import MUTATED, K, bits, clone_mutated, verify_args

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: sm_107",
)

#: helix8's KDA heads per rank and its draft length.
H = 12
NUM_SPEC = 5
#: Requests per step: split_v 2 (up to 64 tiles), one CTA per tile, and the one-wave edge.
REQUESTS = (1, 2, 4, 5, 6, 9, 16, 17)
REPLAYS = 50


def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _case(requests: int, seed: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """helix8's verify arguments (the stacked views, ``mxfp8_exact``) with the f_b operands: the
    f_a columns of the same stacked projection rows and a BF16 f_b weight ``[H * 128, 128]``."""
    args = verify_args(
        dict(N=requests, H=H, num_spec=NUM_SPEC, stacked=True, out="mxfp8_exact"), seed
    )
    x_q = args["x_q"]
    rows = x_q.shape[1]
    f_a = torch.as_strided(
        x_q, (rows, K), (x_q.stride(1), 1), x_q.storage_offset() + stacked_columns(H)["f_a"]
    )
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    w_fb = (torch.randn(H * K, K, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    return args, f_a, w_fb


def _chain(args: dict, f_a: torch.Tensor, w_fb: torch.Tensor, use_pdl: bool):
    """helix8's [f_b -> verify]: the cuBLAS forget-gate GEMM, then the verify on its output."""
    g = torch.nn.functional.linear(f_a, w_fb)
    return op(**dict(args, g=g.view(1, -1, H, K), use_pdl=use_pdl))


def _written_scales(scales: torch.Tensor, rows: int) -> torch.Tensor:
    """The R128c4 scale bytes of rows ``< rows`` (the epilogue leaves the padding rows)."""
    r = torch.arange(rows, device=scales.device).view(-1, 1)
    c = torch.arange(4 * H, device=scales.device).view(1, -1)
    off = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * 512 * H
    return scales.view(torch.uint8)[off.reshape(-1)]


def _assert_same(want, want_args: dict, got, got_args: dict, rows: int) -> None:
    assert torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8)), "E4M3 codes"
    assert torch.equal(_written_scales(got[1], rows), _written_scales(want[1], rows)), "scales"
    for name in MUTATED:
        assert torch.equal(bits(got_args[name]), bits(want_args[name])), name


@pytest.mark.parametrize("requests", REQUESTS)
@torch.inference_mode()
def test_pdl_launch_writes_the_plain_launch_bytes(requests) -> None:
    if H * requests > _sms():
        pytest.skip("past one wave of tiles the two-CTA kernel serves the rows")
    args, f_a, w_fb = _case(requests, seed=100 + requests)
    rows = requests * (NUM_SPEC + 1)
    plain, pdl = clone_mutated(args), clone_mutated(args)
    want = _chain(plain, f_a, w_fb, False)
    got = _chain(pdl, f_a, w_fb, True)
    torch.cuda.synchronize()
    _assert_same(want, plain, got, pdl, rows)


@pytest.mark.parametrize("requests", (2, 4, 9, 17))
@torch.inference_mode()
def test_pdl_graph_replays_write_the_plain_launch_bytes(requests) -> None:
    if H * requests > _sms():
        pytest.skip("past one wave of tiles the two-CTA kernel serves the rows")
    args, f_a, w_fb = _case(requests, seed=200 + requests)
    rows = requests * (NUM_SPEC + 1)
    source = clone_mutated(args)
    arms = {}
    for use_pdl in (False, True):
        _chain(clone_mutated(args), f_a, w_fb, use_pdl)  # compile outside the capture
        torch.cuda.synchronize()
        live = clone_mutated(args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _chain(live, f_a, w_fb, use_pdl)
        arms[use_pdl] = (graph, live, out)
    for _ in range(REPLAYS):
        for graph, live, out in arms.values():
            for name in MUTATED:
                live[name].copy_(source[name])
            out[0].view(torch.uint8).fill_(0x7F)  # E4M3 NaN: every code must be rewritten
            out[1].view(torch.uint8).fill_(0xFF)
            graph.replay()
        torch.cuda.synchronize()
        _, plain_live, plain_out = arms[False]
        _, pdl_live, pdl_out = arms[True]
        _assert_same(plain_out, plain_live, pdl_out, pdl_live, rows)
    eager = clone_mutated(source)
    want = _chain(eager, f_a, w_fb, False)
    torch.cuda.synchronize()
    _, pdl_live, pdl_out = arms[True]
    _assert_same(want, eager, pdl_out, pdl_live, rows)


@torch.inference_mode()
def test_two_cta_rows_refuse_the_pdl_launch() -> None:
    args, f_a, w_fb = _case(_sms() // H + 1, seed=7)
    with pytest.raises(ValueError, match="PDL launch takes it on the v2 kernel only"):
        _chain(args, f_a, w_fb, True)
