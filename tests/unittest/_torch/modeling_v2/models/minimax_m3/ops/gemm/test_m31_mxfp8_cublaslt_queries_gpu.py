# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's cuBLASLt MXFP8 configuration queries and the peer-access helper.

``minimax_m31_mxfp8_cublaslt_heuristic`` returns the heuristic's first configuration for (M, N, K) as the nine
attributes the pinned GEMM takes plus its workspace bytes; ``minimax_m31_mxfp8_cublaslt_algo_check`` reports the bytes
a configuration needs for (M, N, K), or -1. The heuristic's configuration must pass the check with the same bytes and
run in ``minimax_m31_mxfp8_cublaslt_gemm`` within the FP32-accumulation band of the dequantized product.
``minimax_m31_sp_addnorm_enable_peer`` enables peer access and accepts an already-enabled peer.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

__extra_import_path__ = ["../.."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the MXFP8 cuBLASLt GEMM runs on SM10x",
)

#: The serving workspace limit of the pinned GEMM (kWorkspaceBytes).
WORKSPACE_BYTES = 1 << 20
#: (M, N, K) of the prefill linears the pins serve.
SHAPES = ((1024, 6144, 4096), (2048, 12288, 6144), (4096, 6144, 1536))


def _dequant(values, swizzled_sf, rows, k):
    """values [rows, k] E4M3 times their UE8M0 block scales (un-swizzling mxfp8's 128x4 layout)."""
    cols = k // 32
    padded_cols = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=values.device).view(-1, 1)
    c = torch.arange(cols, device=values.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    sf = swizzled_sf.view(torch.uint8).reshape(-1)[index].to(torch.int32)
    scale = torch.exp2((sf - 127).to(torch.float32)).repeat_interleave(32, dim=1)
    return values.to(torch.float32) * scale


@pytest.mark.parametrize("m,n,k", SHAPES, ids=str)
def test_the_heuristic_configuration_checks_and_runs(m, n, k) -> None:
    ops = family_ops()
    config = list(ops.minimax_m31_mxfp8_cublaslt_heuristic(m, n, k))
    assert len(config) == 10, config
    algo, workspace = config[:9], config[9]
    assert 0 <= workspace <= WORKSPACE_BYTES
    assert ops.minimax_m31_mxfp8_cublaslt_algo_check(m, n, k, algo) == workspace
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    w, w_sf = mxfp8(n, k, g)
    act, act_sf = mxfp8(m, k, g)
    out = ops.minimax_m31_mxfp8_cublaslt_gemm(act, act_sf, w, w_sf, algo)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref = _dequant(act, act_sf, m, k) @ _dequant(w, w_sf, n, k).t()
    # One bf16 rounding of an FP32 accumulation: within 2^-7 of each row's largest magnitude.
    row_max = ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    assert bool(((out.float() - ref).abs() <= row_max * 2.0**-7).all())


def test_the_check_rejects_a_shape_the_configuration_cannot_serve() -> None:
    """A configuration recorded for one K, asked about a K it cannot tile, is rejected (-1) or needs no more than the
    serving workspace: the check never reports bytes past the limit."""
    ops = family_ops()
    algo = list(ops.minimax_m31_mxfp8_cublaslt_heuristic(1024, 6144, 4096))[:9]
    got = ops.minimax_m31_mxfp8_cublaslt_algo_check(1024, 6144, 4096 + 32, algo)
    assert got == -1 or 0 <= got <= WORKSPACE_BYTES


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_enable_peer_accepts_an_already_enabled_peer() -> None:
    ops = family_ops()
    with torch.cuda.device(0):
        ops.minimax_m31_sp_addnorm_enable_peer(1)
        ops.minimax_m31_sp_addnorm_enable_peer(1)
