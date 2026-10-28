# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness test: Triton MLA context-K concat vs torch.cat.

Run on a GPU box:  python tests/unittest/_torch/modules/test_mla_ctx_k_triton.py
"""
import pytest
import torch

from tensorrt_llm._torch.modules.mla_ctx_k_triton import mla_concat_k

D_NOPE, D_ROPE, D_V = 128, 64, 128


def _reference(k_nope, k_pe, heads):
    return torch.cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)


def _make_inputs(tokens, heads, dtype, strided):
    """Mimic mla.py: k_nope is a view of the kv_b_proj output split, k_pe a
    slice of the latent cache row when `strided`."""
    dev = "cuda"

    def rnd(*shape):
        x = torch.randn(*shape, device=dev, dtype=torch.float32) * 4
        return x.to(dtype)

    if strided:
        kv = rnd(tokens, heads * (D_NOPE + D_V))
        k_nope = kv.split([heads * D_NOPE, heads * D_V], -1)[0].view(tokens, heads, D_NOPE)
        lat = rnd(tokens, 512 + D_ROPE)
        k_pe = lat[:, -D_ROPE:].view(tokens, 1, D_ROPE)
    else:
        k_nope = rnd(tokens, heads, D_NOPE)
        k_pe = rnd(tokens, 1, D_ROPE)
    return k_nope, k_pe


def _fp8_ok():
    return hasattr(torch, "float8_e4m3fn")


CASES = [(1, 96), (7, 96), (1000, 96), (4097, 128), (333, 64), (65536, 96)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("tokens,heads", CASES)
def test_equal(tokens, heads, strided, dtype):
    k_nope, k_pe = _make_inputs(tokens, heads, dtype, strided)
    if dtype == torch.float8_e4m3fn:  # torch.equal / cat on fp8: compare raw bytes
        ref = _reference(k_nope, k_pe, heads).view(torch.uint8)
        out = mla_concat_k(k_nope, k_pe).view(torch.uint8)
    else:
        ref = _reference(k_nope, k_pe, heads)
        out = mla_concat_k(k_nope, k_pe)
    assert out.is_contiguous()
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_cuda_graph_capture():
    tokens, heads = 4093, 96
    k_nope, k_pe = _make_inputs(tokens, heads, torch.bfloat16, True)
    mla_concat_k(k_nope, k_pe)  # compile before capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mla_concat_k(k_nope, k_pe)
    k_nope.copy_(torch.randn_like(k_nope))  # new data must flow through replay
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, _reference(k_nope, k_pe, heads))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a GPU"
    dtypes = [torch.bfloat16, torch.float16] + ([torch.float8_e4m3fn] if _fp8_ok() else [])
    for dt in dtypes:
        for strided in (False, True):
            for t, h in CASES:
                test_equal(t, h, strided, dt)
                print(f"OK {dt} strided={strided} T={t} H={h}")
    test_cuda_graph_capture()
    print("OK cuda graph")
    print("ALL PASSED")
