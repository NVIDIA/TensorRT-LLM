# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family SiTU kernel is ``trtllm::situ_and_mul``'s, launched the same way, bit for bit.

* source (CPU): the Triton kernel and the op's launch code are V1's token for token
  (``_torch/modules/situ.py``, unchanged since the reference 084745cde4).
* GPU: ``mv2_kimi_k3_vl::situ_and_mul`` equals ``trtllm::situ_and_mul`` bitwise on Kimi K3's
  widths and caps, other dtypes, a row-strided input, and under CUDA graph replay.
"""

import importlib

import pytest
import torch

__extra_import_path__ = ["../.."]
from _k3_source_parity import REPO, function_dump

_KERNELS = REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kernels/k3r_situ"
_V1 = REPO / "tensorrt_llm/_torch/modules/situ.py"
_OP = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_situ.op"

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the Triton SiTU kernel needs an SM90+ GPU here",
)


@pytest.mark.cpu_only
def test_kernel_is_the_v1_kernel():
    assert function_dump(_KERNELS / "situ_and_mul_kernel.py", "situ_and_mul_kernel") == (
        function_dump(_V1, "situ_and_mul_kernel")
    )


@pytest.mark.cpu_only
def test_launch_is_the_v1_launch():
    # The one local edit: PDL is the family's SM107 launch constant, not get_env_enable_pdl().
    edits = [("get_env_enable_pdl() and", "USE_PDL and")]
    assert function_dump(_KERNELS / "op.py", "situ_and_mul") == function_dump(
        _V1, "situ_and_mul", edits
    )


def _ops():
    importlib.import_module(_OP)
    import tensorrt_llm._torch.modules.situ  # noqa: F401 — registers trtllm::situ_and_mul

    return torch.ops.mv2_kimi_k3_vl.situ_and_mul, torch.ops.trtllm.situ_and_mul


_CASES = [
    *(
        (rows, width, torch.bfloat16, 4.0, 25.0)
        for width in (768, 3072, 4224, 16896)
        for rows in (1, 8, 64, 2048)
    ),
    (33, 6144, torch.bfloat16, 2.5, 7.0),
    (33, 6144, torch.bfloat16, 1.0, None),
    *((17, 1536, dtype, 4.0, 25.0) for dtype in (torch.float16, torch.float32)),
]


@_gpu
@pytest.mark.parametrize("rows,width,dtype,beta,linear_beta", _CASES)
def test_bitwise_equal_to_v1(rows, width, dtype, beta, linear_beta):
    fam, v1 = _ops()
    gen = torch.Generator(device="cuda").manual_seed(rows * 131 + width)
    x = torch.randn(rows, width, generator=gen, device="cuda", dtype=dtype) * 4.0
    assert torch.equal(fam(x, beta, linear_beta), v1(x, beta, linear_beta))


@_gpu
def test_bitwise_equal_on_a_row_strided_input():
    fam, v1 = _ops()
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)[:, :768]
    assert not x.is_contiguous()
    assert torch.equal(fam(x, 4.0, 25.0), v1(x, 4.0, 25.0))


@_gpu
def test_cuda_graph_replay_matches_v1():
    fam, v1 = _ops()
    x = torch.randn(8, 768, dtype=torch.bfloat16, device="cuda")
    fam(x, 4.0, 25.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fam(x, 4.0, 25.0)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(out, v1(x, 4.0, 25.0))
