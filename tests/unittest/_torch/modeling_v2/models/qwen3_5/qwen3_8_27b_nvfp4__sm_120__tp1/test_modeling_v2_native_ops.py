# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's native ops: registered in its own torch namespace, with fakes.

Importing the target package loads ``libth_modeling_v2_qwen3_5.so`` through
``_native.load_target_ops``, as routing does: a pass here means the family library
was built for this GPU, packaged, opened, carries the target, and registered the
fakes of ``fake_ops.py``. The numerics of each op are certified by its entry's test
under ``gemm/``; this checks the registration and the fakes' output shapes, which
torch.compile tracing relies on.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5 import (
    qwen3_8_27b_nvfp4__sm_120__tp1 as target,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3_5 family library is built for sm_120f only"
)

_OPS = ("nvfp4_w4a16_gemv", "nvfp4_w4a16_gemv_conv", "nvfp4_gate_up_swiglu_nvfp4")


def test_ops_live_in_the_target_namespace():
    assert namespace_of(target.__name__) == "mv2_qwen3_5_qwen3_8_27b_nvfp4_sm_120_tp1"
    assert target.native_ops is getattr(torch.ops, namespace_of(target.__name__))
    for name in _OPS:
        assert hasattr(target.native_ops, name), name
        assert not hasattr(torch.ops.trtllm, name), f"{name} is still registered in trtllm"


def test_fakes_match_the_ops_output_shapes():
    ops = target.native_ops
    m, k, n = 3, 5120, 16384
    with FakeTensorMode():
        x = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
        sf = torch.empty(n, k // 16, dtype=torch.uint8, device="cuda")
        out = ops.nvfp4_w4a16_gemv(x, w, sf, 1.0)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16

        conv_dim = 10240
        state = torch.empty(4, conv_dim, 3, dtype=torch.bfloat16, device="cuda")
        conv_w = torch.empty(conv_dim, 4, dtype=torch.bfloat16, device="cuda")
        idx = torch.empty(m, dtype=torch.int32, device="cuda")
        out = ops.nvfp4_w4a16_gemv_conv(x, w, sf, 1.0, state, conv_w, idx, conv_dim, -1)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16

        # gate_up of hidden 5120 -> 2 x 17408 at 41 rows: fp4 [m, n/4], SF over ceil128(m) rows.
        m, n = 41, 34816
        act = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
        weight = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
        act_sf = torch.empty(128 * (k // 16), dtype=torch.uint8, device="cuda")
        weight_sf = torch.empty(n * (k // 16), dtype=torch.uint8, device="cuda")
        scale = torch.empty(1, dtype=torch.float32, device="cuda")
        fp4, out_sf = ops.nvfp4_gate_up_swiglu_nvfp4(
            act, weight, act_sf, weight_sf, scale, scale, 2
        )
        assert fp4.shape == (m, n // 4) and fp4.dtype == torch.uint8
        assert out_sf.shape == (128 * ((n // 2 // 16 + 3) // 4) * 4,)
