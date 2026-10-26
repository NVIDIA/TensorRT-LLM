# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the unfused sparse layers' separate MXFP8 index projection (``fuse_qkv_index_projection=False``) of the
tp2 target equals the index columns of the fused five-way projection on the same checkpoint tensors.

Its weight rows are the five-way weight's index rows byte for byte, its block scales are applied (not dropped), and
on every engine MXFP8 backend its output is bitwise the five-way GEMM's index columns.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    weights as tw,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mxfp8_linear import (  # noqa: E501
    m3_mxfp8_linear as op,
)

assert torch.cuda.is_available(), "the MXFP8 index projection requires a CUDA device"

TP = 2
HIDDEN = 6144
BLOCK = 32
# Text-model rows of the attention projections (64 query / 4 KV heads, 4 index heads, head and index dim 128): at TP2
# the five-way projection is 32 q + 2 k + 2 v + 2 index-q heads + the replicated index-k = 4992 rows per rank.
ROWS = {"q": 64 * 128, "k": 4 * 128, "v": 4 * 128, "index_q": 4 * 128, "index_k": 128}
_FLAGS = {
    "native": (False, False, False, False),
    "flashinfer": (False, False, True, False),
    "tuned": (True, False, False, False),
}


def _checkpoint(seed: int, lo: int = 121, hi: int = 133):
    """E4M3 weights and UE8M0 block scales per projection; the scales spread over 2^(lo-127) .. 2^(hi-128) (default
    2^-6 .. 2^5; the stress 115..140 spans 2^-12 .. 2^+12)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {}
    for name, rows in ROWS.items():
        w = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
        s = torch.randint(
            lo, hi, (rows, HIDDEN // BLOCK), generator=g, device="cuda", dtype=torch.uint8
        )
        out[name] = (w.cpu(), s.cpu())
    return out


def _ns(rank: int) -> SimpleNamespace:
    return SimpleNamespace(tp=TP, rank=rank, dev=torch.device("cuda"), head_dim=128, index_dim=128)


def _packs(ck, rank):
    d = _ns(rank)
    names = ("q", "k", "v", "index_q", "index_k")
    w5 = tw._qkv_sparse_w(tuple(ck[n][0] for n in names), d)
    s5 = tw._qkv_sparse_s(
        tuple(ck[n][1] for n in names), d
    )  # UE8M0 bytes, as the checkpoint stores them
    wi = tw._index_qk_w((ck["index_q"][0], ck["index_k"][0]), d)
    si = tw._index_qk_s((ck["index_q"][1], ck["index_k"][1]), d)
    return w5, s5.view(torch.uint8), wi, si.view(torch.uint8)


@pytest.mark.parametrize("rank", [0, TP - 1])
def test_weight_rows_are_the_five_way_index_rows(rank) -> None:
    ck = _checkpoint(1234 + rank)
    w5, _, wi, _ = _packs(ck, rank)
    n = wi.shape[0]
    assert w5.shape == (4992, HIDDEN), "the TP2 five-way projection's rows per rank"
    assert n == 2 * 128 + 128, "this rank's 2 index-q heads and the replicated index-k"
    assert wi.dtype == torch.float8_e4m3fn and wi.shape == (n, HIDDEN)
    assert torch.equal(wi.view(torch.uint8), w5[-n:].view(torch.uint8))


@pytest.mark.parametrize("scales", [(121, 133), (115, 140)], ids=["narrow", "stress"])
@pytest.mark.parametrize("path", list(_FLAGS))
@pytest.mark.parametrize("rows", [1, 4, 8, 32, 64, 256])
def test_unfused_output_is_bitwise_the_five_way_index_columns(path, rows, scales) -> None:
    if path == "flashinfer" and getattr(torch.ops.trtllm, "flashinfer_mm_mxfp8", None) is None:
        pytest.skip("this build has no trtllm::flashinfer_mm_mxfp8")
    ck = _checkpoint(77, *scales)
    w5, s5, wi, si = _packs(ck, 0)
    n = wi.shape[0]
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    out5 = op(x, w5, s5, alpha, torch.bfloat16, *_FLAGS[path])
    outi = op(x, wi, si, alpha, torch.bfloat16, *_FLAGS[path])
    assert outi.shape == (rows, n)
    assert torch.equal(outi, out5[:, -n:]), (
        f"{path}: max |diff| {(outi.float() - out5[:, -n:].float()).abs().max().item()}"
    )


def test_block_scales_are_applied() -> None:
    """The output follows the op's reference with the checkpoint scales and differs from a unit-scale (dropped)
    projection by far more than rounding."""
    ck = _checkpoint(5)
    _, _, wi, si = _packs(ck, 0)
    unit = tw._index_qk_s(
        (torch.full_like(ck["index_q"][1], 127), torch.full_like(ck["index_k"][1], 127)), _ns(0)
    ).view(torch.uint8)
    x = torch.randn(8, HIDDEN, device="cuda").to(torch.bfloat16)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    out = op(x, wi, si, alpha, torch.bfloat16, *_FLAGS["native"])
    ref = op.reference(x, wi, si, alpha, torch.bfloat16, *_FLAGS["native"])
    dropped = op(x, wi, unit, alpha, torch.bfloat16, *_FLAGS["native"])
    scale = ref.float().abs().mean()
    # The engine GEMM rounds once to BF16 after an fp32 accumulation in its own order: within 2 BF16 ulps (2^-7
    # relative) of the reference per element.
    assert bool(
        ((out.float() - ref.float()).abs() <= 2.0**-7 * ref.float().abs() + 1e-3 * scale).all()
    )
    assert (out.float() - dropped.float()).abs().mean() > 0.5 * scale
