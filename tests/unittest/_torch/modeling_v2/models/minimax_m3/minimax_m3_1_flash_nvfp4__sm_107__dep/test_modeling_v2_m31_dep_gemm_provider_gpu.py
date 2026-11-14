# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_GEMM_PROVIDER (gemm_provider.py) at 65..128 decode rows: qkv [9856, 6144], o_proj [6144, 8192], the shared
gate_up [6144, 6144] and the shared down [6144, 3072] on cuBLASLt at one pinned algorithm (DEP_GEMM_CUBLASLT); o_proj
and the shared gate_up on CuteDSLMXFP8RubinLinear at split-K 2 without it.

* the route: only those shapes, only those rows, phase 1 with DEP_GEMM_CUBLASLT off, nothing with the switch off;
* on an SM107 GPU: the output is close to the pinned full-K tactic (rel-L2 <= 1e-2 on split-K 2, <= 1e-4 on cuBLASLt),
  deterministic, and a CUDA graph replays it bit for bit; the hand-off GEMM returns it and the pinned tactic runs where
  it declines; with DEP_GEMM_CUBLASLT off o_proj and the shared gate_up are split-K 2's output.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_rubin_mxfp8
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    gemm_provider,
    merge_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod

SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
needs_sm107 = pytest.mark.skipif(not SM107, reason="needs an SM107 GPU")

CUBLASLT = (gemm_provider.CUBLASLT, gemm_provider.CUBLASLT_ALGO66)
RUBIN = (gemm_provider.RUBIN, gemm_provider.RUBIN_SPLIT_K2)
ROUTED = {
    "qkv_proj": ((9856, 6144), CUBLASLT),
    "o_proj": ((6144, 8192), CUBLASLT),
    "shared_gate_up": ((6144, 6144), CUBLASLT),
    "shared_down": ((6144, 3072), CUBLASLT),
}


def test_the_route_covers_four_shapes_at_the_decode_rows(monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_CUBLASLT", True)
    for (n, k), expected in ROUTED.values():
        assert gemm_provider.route(96, n, k) == expected
        assert (
            gemm_provider.route(65, n, k) == expected and gemm_provider.route(128, n, k) == expected
        )
        assert gemm_provider.route(64, n, k) is None and gemm_provider.route(129, n, k) is None
    assert gemm_provider.route(96, 6144, 4096) is None
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", False)
    assert all(gemm_provider.route(96, n, k) is None for (n, k), _ in ROUTED.values())


def test_without_cublaslt_the_route_is_phase_1(monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_CUBLASLT", False)
    assert gemm_provider.route(96, 9856, 6144) is None
    assert gemm_provider.route(96, 6144, 3072) is None
    assert gemm_provider.route(96, 6144, 6144) == RUBIN
    assert gemm_provider.route(96, 6144, 8192) == RUBIN


def _operands(n, k, rows, seed):
    from flashinfer import mxfp8_quantize

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w_mx, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    a_mx, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    return a_mx, a_sf, w_mx, w_sf


@needs_sm107
@torch.inference_mode()
@pytest.mark.parametrize("rows", (72, 96, 128))
@pytest.mark.parametrize("name", list(ROUTED))
def test_the_provider_is_close_to_the_pinned_tactic_and_deterministic(name, rows, monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_CUBLASLT", True)
    (n, k), routed = ROUTED[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=rows + n % 97)
    first = gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    assert first is not None and first.shape == (rows, n) and first.dtype == torch.bfloat16
    again = gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    assert torch.equal(first.view(torch.int16), again.view(torch.int16)), (
        f"{routed[0]} must be deterministic"
    )
    with pin.engaged_for("test DEP_GEMM_PROVIDER"):
        pinned = pin.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    assert pinned is not None
    rel = ((first.float() - pinned.float()).norm() / pinned.float().norm()).item()
    bound = 1e-4 if routed == CUBLASLT else 1e-2
    assert rel <= bound, f"{routed[0]}: rel-L2 {rel:.3e} vs the pinned full-K tactic"


@needs_sm107
@torch.inference_mode()
@pytest.mark.parametrize("name", ["o_proj", "qkv_proj"])
def test_a_cuda_graph_replays_the_provider_output(name, monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_CUBLASLT", True)
    (n, k), _ = ROUTED[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, 96, seed=3)
    eager = gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


def _method():
    method = MXFP8LinearMethod()
    method.__class__ = merge_mxfp8.DepMergeMxfp8LinearMethod
    method.use_cutlass = True
    method.backend = "trtllm"
    method.use_native_autotuner = False
    method._flashinfer_autotuned = False
    method.tune_decode_graph_backends = True
    return method


@needs_sm107
@torch.inference_mode()
@pytest.mark.parametrize("rows,provider", [(96, True), (48, False)])
def test_the_hand_off_gemm_takes_the_provider_where_it_routes(rows, provider, monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", True)
    (n, k), _ = ROUTED["o_proj"]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=11)
    module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
    x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
    # The slot's previous hand-off comes back at teardown even if setup or apply raises before consuming this one.
    monkeypatch.setattr(merge_mxfp8.SLOT, "pending", None)
    merge_mxfp8.SLOT.publish(x, a_mx, a_sf)
    token = _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.set(True)
    try:
        with pin.engaged_for("test DEP_GEMM_PROVIDER"):
            out = _method().apply(module, x, None)
            expected = (
                gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
                if provider
                else pin.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
            )
    finally:
        _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.reset(token)
    assert merge_mxfp8.SLOT.pending is None
    assert expected is not None
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@needs_sm107
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(6144, 6144), (6144, 8192)])
def test_without_cublaslt_o_proj_and_the_shared_gate_up_are_split_k2(n, k, monkeypatch):
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_CUBLASLT", False)
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, 96, seed=5)
    out = gemm_provider.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    rubin = m31_rubin_mxfp8.run(a_mx, a_sf, w_mx, w_sf, gemm_provider.RUBIN_SPLIT_K2, "test")
    assert out is not None and rubin is not None
    assert torch.equal(out.view(torch.int16), rubin.view(torch.int16))
    # Split-K 2 against an independent full-K GEMM: the pinned tactic.
    with pin.engaged_for("test DEP_GEMM_PROVIDER"):
        pinned = pin.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    assert pinned is not None
    rel = ((out.float() - pinned.float()).norm() / pinned.float().norm()).item()
    assert rel <= 1e-2, f"split-K 2: rel-L2 {rel:.3e} vs the pinned full-K tactic"
