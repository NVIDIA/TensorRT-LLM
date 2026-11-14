# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 drafter his-layout GEMV provider: parity with the stock projections and CUDA-graph replay."""

import importlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- register native ops before evaluating skips
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires SM107 and the TP8 GEMV provider",
)


class WindowAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def uses_nccl_symmetric_memory_window(self):
        return True

    def forward(self, x, all_reduce_params=None):
        self.calls.append(all_reduce_params)
        return x


@pytest.mark.parametrize("rows", [7, 8])
def test_row_forward_reaches_gemv_and_replays(rows, monkeypatch):
    mxd = importlib.import_module(f"{PKG}.drafter_mxfp8")
    gemv = importlib.import_module(f"{PKG}.kernels.mx_gemv.op")
    n, k = mxd.HIS_SHAPES["o_proj"]
    linear = Linear(k, n, bias=False, dtype=torch.bfloat16).cuda()
    g = torch.Generator(device="cuda").manual_seed(161)
    codes = (torch.randn(n, k, device="cuda", generator=g) * 16).to(torch.float8_e4m3fn)
    with torch.no_grad():
        linear.weight.copy_(codes.to(torch.bfloat16))
    linear.tp_mode, linear.reduce_output, linear.tp_size = TensorParallelMode.ROW, True, 8
    linear.all_reduce = WindowAR()
    draft = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=linear))])
    )
    assert mxd.route_his_layout(draft, [rows]) == (1, 0)
    assert linear._m31_hislayout_row_forward
    x = torch.randn(rows, k, device="cuda", generator=g).to(torch.bfloat16)
    params = SimpleNamespace(enable_allreduce=False)
    w8, sf, _ = linear._m31_mxdraft
    expected = gemv.m3_mx_gemv_bf16(x, w8, sf, True, mxd.HIS_CFGS[(n, k)])
    assert expected is not None

    def bypassed(*args, **kwargs):
        raise AssertionError("the ROW dispatch bypassed the his-layout GEMV")

    monkeypatch.setattr(linear.quant_method, "apply", bypassed)
    got = linear(x, all_reduce_params=params)
    assert linear.all_reduce.calls[-1] is params
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear(x, all_reduce_params=params)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = linear(x, all_reduce_params=params)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [7, 8])
def test_fused_mlp_matches_activation_quantization_and_replays(rows):
    mxd = importlib.import_module(f"{PKG}.drafter_mxfp8")
    gemv = importlib.import_module(f"{PKG}.kernels.mx_gemv.op")
    family = PKG.rsplit(".", 1)[0]
    native = importlib.import_module(f"{family}.ops.native_swiglu")
    g = torch.Generator(device="cuda").manual_seed(162)
    projections = {}
    for name in ("gate_up_proj", "down_proj"):
        n, k = mxd.HIS_SHAPES[name]
        linear = Linear(k, n, bias=False, dtype=torch.bfloat16).cuda()
        codes = (torch.randn(n, k, device="cuda", generator=g) * 16).to(torch.float8_e4m3fn)
        with torch.no_grad():
            linear.weight.copy_(codes.to(torch.bfloat16))
        projections[name] = linear
    down = projections["down_proj"]
    down.tp_mode, down.reduce_output, down.tp_size = TensorParallelMode.ROW, True, 8
    down.all_reduce = WindowAR()
    mlp = SimpleNamespace(**projections, split_gate_up=False, forward=lambda *a, **kw: None)
    draft = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    assert mxd.route_his_layout(draft, [rows]) == (1, 1)
    _, width = mxd.HIS_SHAPES["gate_up_proj"]
    x = torch.randn(rows, width, device="cuda", generator=g).to(torch.bfloat16)
    params = SimpleNamespace(enable_allreduce=False)
    w8, sf, _, cfg = mlp._m31_hislayout_mlp
    gate_up = mlp.gate_up_proj(x)
    quantized, scales = torch.ops.trtllm.mxfp8_quantize(
        native.swiglu_oai(gate_up).contiguous(), True
    )
    expected = gemv.m3_mx_gemv(quantized.view(rows, w8.shape[1]), scales, w8, sf, True, cfg)
    assert expected is not None
    got = mlp.forward(x, final_all_reduce_params=params)
    assert down.all_reduce.calls[-1] is params
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mlp.forward(x, final_all_reduce_params=params)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mlp.forward(x, final_all_reduce_params=params)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
