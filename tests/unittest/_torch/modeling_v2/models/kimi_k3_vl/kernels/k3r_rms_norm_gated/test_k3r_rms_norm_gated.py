# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family gated RMSNorm is ``trtllm::rms_norm_gated_token_major`` as the reference runs it.

* source (CPU): the kernels, their launcher and the op are those of
  ``_torch/modules/mamba/layernorm_gated.py`` at the reference 084745cde4 token for token, pinned
  by digest (``_k3_source_parity.function_digest`` of ``git show 084745cde4:<that file>``). Main
  has since moved the multi-row kernel's weight load before the PDL wait and added an early PDL
  trigger; the math is the same, the PDL order is the reference's here.
* GPU: ``mv2_kimi_k3_vl::rms_norm_gated_token_major`` equals today's V1 op bitwise (same math),
  on the multi-row kernel (KDA's sigmoid gate in a column slice, several head counts and token
  counts), the one-row fallback, the FP8 output and under CUDA graph replay.
"""

import importlib

import pytest
import torch

__extra_import_path__ = ["../.."]
from _k3_source_parity import REPO, function_digest

_KERNELS = (
    REPO
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kernels/k3r_rms_norm_gated"
)
_OP = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_rms_norm_gated.op"
)

#: function -> digest of 084745cde4:tensorrt_llm/_torch/modules/mamba/layernorm_gated.py.
_REFERENCE = {
    "_rms_norm_gated_fwd_multirow_kernel": "048d1c56a7ed9c8d0c52fd1b7b5f0576cc30974cc0f34d98d97bb898b50c0d76",
    "_layer_norm_fwd_1pass_kernel": "3d90cad0abf718992b41847a2e28c3d7f0ded55da065366b0fad3ca3ae0010d7",
    "_layer_norm_fwd": "6b22c6df638e8b105e997d1457d0823f3091e84f46742ff0eba6e1664128f6a6",
    "_multirow_pdl": "6f1dbd1d8942719eab78dc6222df203fed9e319d91f8be0fb7cd704c423bfd92",
    "_pdl_device_policy": "1012626a2e5164d24270ce1aa1d1fb327dd09a1fa75dc55db1374466a332cbfd",
    "_multirow_gated_rmsnorm_eligible": "6eead4c8efaf7e80b4f165b8250573ed23ff7ccf0e0bdcee56e315907c9ba3e7",
}
#: The one local edit (reversed before hashing): PDL is the family's SM107 launch constant
#: (``ops/common/sm107_launch.py``) instead of a read of ``TRTLLM_ENABLE_PDL``.
_LOCAL_EDITS = {
    "_pdl_device_policy": [
        (
            "allowed = sm_version >= 90 and USE_PDL",
            'allowed = sm_version >= 90 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"',
        )
    ],
}
_REFERENCE_OP = "b8a11cac2fd1b34e85ac68365f88488fd928c4abae6c0cac39f6d288991cdb10"

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the Triton gated RMSNorm needs an SM90+ GPU here",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_kernels_are_the_reference_kernels(name):
    edits = _LOCAL_EDITS.get(name, ())
    assert function_digest(_KERNELS / "rms_norm_gated_kernel.py", name, edits) == _REFERENCE[name]


@pytest.mark.cpu_only
def test_launch_is_the_reference_launch():
    assert function_digest(_KERNELS / "op.py", "rms_norm_gated_token_major") == _REFERENCE_OP


def _ops():
    importlib.import_module(_OP)
    import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401 — registers the V1 op

    return (
        torch.ops.mv2_kimi_k3_vl.rms_norm_gated_token_major,
        torch.ops.trtllm.rms_norm_gated_token_major,
    )


def _inputs(tokens, heads, width, gen, extra=0, head_strided=False):
    x = torch.randn(tokens * heads, width, generator=gen, device="cuda", dtype=torch.bfloat16)
    if head_strided:
        z = torch.randn(tokens, width, heads, generator=gen, device="cuda", dtype=torch.bfloat16)
        z = z.transpose(1, 2)
    else:
        wide = torch.randn(
            tokens, heads * width + extra, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        z = wide[:, extra:].view(tokens, heads, width) if extra else wide.view(tokens, heads, width)
    weight = torch.randn(width, generator=gen, device="cuda", dtype=torch.bfloat16)
    return x, z, weight


_CASES = [
    # KDA: 128-wide heads, sigmoid gate, the gate a column slice of the qkvg projection.
    *(
        (tokens, heads, 128, "sigmoid", 3 * heads * 128, False)
        for heads in (6, 12, 24, 96)
        for tokens in (1, 7, 64, 1024)
    ),
    *((48, 4, width, "silu", 0, False) for width in (32, 64, 256)),
    # one-row fallback: not a power of two, wider than 256, or a head-strided gate
    (32, 4, 96, "sigmoid", 0, False),
    (32, 2, 512, "sigmoid", 0, False),
    (16, 6, 128, "sigmoid", 0, True),
]


@_gpu
@pytest.mark.parametrize("tokens,heads,width,gate,extra,head_strided", _CASES)
def test_bitwise_equal_to_v1(tokens, heads, width, gate, extra, head_strided):
    fam, v1 = _ops()
    gen = torch.Generator(device="cuda").manual_seed(tokens * 7 + heads * 131 + width)
    x, z, weight = _inputs(tokens, heads, width, gen, extra, head_strided)
    got = fam(x, z, weight, 1e-5, None, gate)
    assert torch.equal(got, v1(x, z, weight, 1e-5, None, gate))


@_gpu
@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_fp8_output_bitwise_equal_to_v1(scale):
    fam, v1 = _ops()
    gen = torch.Generator(device="cuda").manual_seed(3)
    x, z, weight = _inputs(64, 6, 128, gen)
    fp8_scale = torch.tensor([scale], device="cuda", dtype=torch.float32)
    got = fam(x, z, weight, 1e-5, fp8_scale, "sigmoid")
    want = v1(x, z, weight, 1e-5, fp8_scale, "sigmoid")
    assert got.dtype == torch.float8_e4m3fn
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@_gpu
def test_cuda_graph_replay_matches_v1():
    fam, v1 = _ops()
    gen = torch.Generator(device="cuda").manual_seed(5)
    x, z, weight = _inputs(4, 96, 128, gen, extra=3 * 96 * 128)
    fam(x, z, weight, 1e-5, None, "sigmoid")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fam(x, z, weight, 1e-5, None, "sigmoid")
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(out, v1(x, z, weight, 1e-5, None, "sigmoid"))
