# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod, get_quant_method
from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight, quant_bf16_to_mxfp8
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def test_quant_dequant_roundtrip_is_close():
    torch.manual_seed(0)
    out_features, in_features = 64, 128  # in_features divisible by 32
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    w_e4m3, scale_ue8m0 = quant_bf16_to_mxfp8(w, block_size=32)
    assert w_e4m3.dtype == torch.float8_e4m3fn
    assert scale_ue8m0.dtype == torch.uint8
    assert scale_ue8m0.shape == (out_features, in_features // 32)

    w_deq = dequant_mxfp8_weight(w_e4m3, scale_ue8m0, block_size=32)
    assert w_deq.shape == (out_features, in_features)
    # MXFP8 is coarse; check relative error of the reconstructed matmul output.
    x = torch.randn(8, in_features, dtype=torch.bfloat16)
    ref = x.float() @ w.float().t()
    got = x.float() @ w_deq.float().t()
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-6)
    assert rel < 0.1, f"relative error too high: {rel}"


def test_mxfp8_dispatch_returns_mxfp8_method():
    """get_quant_method must dispatch QuantAlgo.MXFP8 to MXFP8LinearMethod.

    This is a pure dispatch check; no CUDA required.
    """
    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    method = get_quant_method(qc)
    assert isinstance(method, MXFP8LinearMethod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 Linear load path requires CUDA")
def test_mxfp8_linear_reference_matches_dequant():
    """End-to-end Linear MXFP8 forward (whichever path is active) vs the
    out-of-module dequant reference. Uses norm-relative tolerance because the
    CUTLASS path's per-element error is larger than 2% but the aggregate
    output is still aligned (this is fundamental to MXFP8's coarse 32-element
    block scaling, not a kernel bug).
    """
    torch.manual_seed(0)
    out_f, in_f = 128, 256
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    w_e4m3, scale = quant_bf16_to_mxfp8(w, 32)

    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    lin = Linear(
        in_features=in_f, out_features=out_f, bias=False, dtype=torch.bfloat16, quant_config=qc
    ).cuda()
    # Mirror the checkpoint key naming (`weight_scale_inv`).
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])

    x = torch.randn(4, in_f, dtype=torch.bfloat16, device="cuda")
    got = lin(x)
    w_deq = dequant_mxfp8_weight(w_e4m3, scale, 32).cuda()
    ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.1, f"rel err {rel} (got={got.dtype}, ref={ref.dtype})"


def _mxfp8_cutlass_op_available():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    return hasattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm")


@pytest.mark.skipif(
    not _mxfp8_cutlass_op_available(), reason="MXFP8xMXFP8 GEMM op not compiled or sm < 100"
)
def test_mxfp8_linear_cutlass_matches_reference():
    """End-to-end CUTLASS path: must agree with the dequant reference."""
    torch.manual_seed(0)
    out_f, in_f = 256, 512
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    w_e4m3, scale = quant_bf16_to_mxfp8(w, 32)
    x = torch.randn(16, in_f, dtype=torch.bfloat16, device="cuda")

    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    lin = Linear(
        in_features=in_f, out_features=out_f, bias=False, dtype=torch.bfloat16, quant_config=qc
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])

    got = lin(x)
    w_deq = dequant_mxfp8_weight(w_e4m3, scale, 32).cuda()
    ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.05, f"CUTLASS vs reference rel err {rel}"


CKPT = "/home/scratch.fredw_sw/workspace/hidden_trail/minimax-m3-preview_vv1"


@pytest.mark.skipif(not os.path.isdir(CKPT), reason="MXFP8 ckpt not present")
def test_load_real_mxfp8_dense_layer():
    from safetensors import safe_open

    # Find the shard holding layer 0 q_proj.
    candidate_names = [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    name_w = name_s = None
    w = s = None
    for f in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            keys = set(h.keys())
            for cand in candidate_names:
                if cand in keys:
                    name_w, name_s = cand, cand + "_scale_inv"
                    w, s = h.get_tensor(name_w), h.get_tensor(name_s)
                    break
            if w is not None:
                break
    assert w is not None and s is not None, f"q_proj weight not found in {CKPT}"
    assert w.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.uint8
    # Scale layout is [out_features, in_features/32].
    assert s.shape[1] == w.shape[1] // 32
    w_deq = dequant_mxfp8_weight(w, s, 32)
    assert torch.isfinite(w_deq).all()
