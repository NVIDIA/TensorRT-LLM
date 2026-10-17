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
"""Attribution probes for compute-sanitizer initcheck reports.

Run under ``compute-sanitizer --tool initcheck`` with
``PYTORCH_NO_CUDA_MEMORY_CACHING=1``.
Each probe produces one fully written tensor and then reads it once with an
ordinary PyTorch conversion kernel, so pytest's per-test output attributes
every report to one producer: native producers (H2D copy, elementwise,
cuBLAS GEMMs, the native oracle tail) and the default attention_output_tail. A report under
a native probe concerns how that producer's writes are tracked; a report
under the attention_output_tail probe would mean attention_output_tail left part of its partial
unwritten.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _read(y):
    return float(y.float().abs().sum())


def test_probe_h2d_copy_then_read():
    x = torch.randn(2048, 512, dtype=torch.bfloat16).cuda()
    torch.cuda.synchronize()
    assert _read(x) > 0


def test_probe_elementwise_then_read():
    x = torch.randn(2048, 512, device="cuda").bfloat16()
    torch.cuda.synchronize()
    assert _read(x) > 0


def test_probe_cublas_linear_then_read():
    x = torch.randn(2048, 6144, device="cuda").bfloat16()
    w = torch.randn(576, 6144, device="cuda").bfloat16()
    y = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    assert _read(y) > 0


def test_probe_cublas_linear_small_m_then_read():
    x = torch.randn(6, 2048, device="cuda").bfloat16()
    w = torch.randn(6144, 2048, device="cuda").bfloat16()
    y = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    assert _read(y) > 0


def test_probe_cublas_bmm_then_read():
    a = torch.randn(8, 6, 512, device="cuda").bfloat16()
    b = torch.randn(8, 512, 256, device="cuda").bfloat16()
    y = torch.bmm(a, b)
    torch.cuda.synchronize()
    assert _read(y) > 0


def _tail_inputs(rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    w_uv = (torch.randn(8, 256, 512, generator=gen, device="cuda") * 0.02).bfloat16()
    w_o = (torch.randn(6144, 2048, generator=gen, device="cuda") * 0.02).bfloat16()
    lat = torch.randn(rows, 8, 512, generator=gen, device="cuda").bfloat16()
    return lat, w_uv, w_o


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_probe_native_tail_then_read(rows):
    """The native oracle tail (bmm into the o_proj input buffer, then
    F.linear), read once by a conversion kernel."""
    import tail_reference

    lat, w_uv, w_o = _tail_inputs(rows)
    native = tail_reference.native_attn_tail(lat, w_uv, w_o, rows)
    torch.cuda.synchronize()
    assert _read(native) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="attention_output_tail kernels target SM100/SM103",
)
@pytest.mark.parametrize("rows,out_rows", [(1, 1), (6, 6), (6, 8), (8, 8)])
def test_probe_attn_tail_then_read(rows, out_rows):
    """The default attention_output_tail (PDL on), its partial read once by a conversion
    kernel: every element of the partial, padded rows included, must be
    written by attention_output_tail."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
        attn_tail,
    )

    lat, w_uv, w_o = _tail_inputs(rows)
    out = attn_tail(lat, w_uv, w_o, rows, out_rows=out_rows)
    torch.cuda.synchronize()
    assert _read(out) > 0
