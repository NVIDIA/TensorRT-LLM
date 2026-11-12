# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 glue's FP8 weight read (the K3 low-M GEMM folded in) against the kernel calls it
stands for: the same operands and, on every row count, the same bits.

The attention projections and the MoE MLPs of the helix8 targets all run through it. With the
K3 low-M GEMM folded in (no generic low-M GEMM), a swapped and post-loaded projection must hold:

* ``layers.prepare_fp8_block_weight`` of the checkpoint codes and grid (``weight``,
  ``weight_scale``);
* on SM107, ``lowm_weight_scale`` = ``k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout`` of
  that scale (None elsewhere, where the low-M GEMM reads ``weight_scale``);

and its forward on BF16 rows must be ``ops.gemm.k3_fp8w_lowm_gemm`` on that low-M scale wherever
``k3_fp8w_lowm_gemm.can_run`` accepts the rows, else ``layers.fp8_block_scale_gemm`` without a
generic low-M GEMM.
"""

import importlib

import pytest
import torch
from torch import nn

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs a block-scaled FP8 GPU (SM100 / SM103 / SM107)",
)

ROWS = (1, 2, 3, 8, 16, 17, 31, 32, 64, 97, 100, 128, 129, 256, 1000)
# K3's FP8 projection shapes: q_a + kv_a fused, q_b, kv_b, o (MLA), KDA o / q, the latent down /
# up and the shared expert's gate_up / down.
SHAPES = [
    (2112, 7168),
    (1152, 1536),
    (2304, 512),
    (7168, 1024),
    (4096, 7168),
    (3584, 7168),
    (7168, 3584),
    (6144, 7168),
    (7168, 3072),
]


def _modules():
    layers = importlib.import_module(f"{_FAMILY}.ops.gemm.sm107_linear")
    helix8 = importlib.import_module(f"{_FAMILY}.kimi_k3_nvfp4__sm_107__helix8_moeep8.modeling")
    lowm_kernel = importlib.import_module(f"{_FAMILY}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
    lowm_op = importlib.import_module(f"{_FAMILY}.ops.gemm.k3_fp8w_lowm_gemm").k3_fp8w_lowm_gemm
    return layers, helix8, lowm_kernel, lowm_op


def _checkpoint_fp8(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = torch.nn.functional.pad(weight, (0, -k % 128, 0, -n % 128)).view(
        -(-n // 128), 128, -(-k // 128), 128
    )
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return (weight / expanded).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("n,k", SHAPES)
@torch.inference_mode()
def test_folded_weight_read_matches_its_kernel_calls(n, k):
    layers, helix8, lowm_kernel, lowm_op = _modules()
    codes, scale = _checkpoint_fp8(n, k, seed=n * 7 + k)

    parent = nn.Module()
    source = layers.Fp8BlockScaleLinear(k, n, device="cuda")
    source.load(codes.clone(), scale.clone())
    parent.o_proj = source
    assert helix8._swap_linear_to_fp8_weight_read(parent, "o_proj", (layers.Fp8BlockScaleLinear,))
    got = parent.o_proj
    got.post_load_weights()
    assert not hasattr(got, "_low_m_max") and not hasattr(got, "_low_m_weight_scale")

    weight, weight_scale = layers.prepare_fp8_block_weight(codes.clone(), scale.clone())
    rubin = helix8._rubin_mxfp8_is_available()
    lowm_weight_scale = None
    if rubin and n % lowm_kernel.N_ALIGN == 0 and k % 128 == 0:
        lowm_weight_scale = lowm_kernel.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)
    assert torch.equal(got.weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(got.weight_scale, weight_scale)
    assert (got.lowm_weight_scale is None) == (lowm_weight_scale is None)
    if lowm_weight_scale is not None:
        assert torch.equal(got.lowm_weight_scale, lowm_weight_scale)
    lowm_scale = lowm_weight_scale if rubin else weight_scale

    gen = torch.Generator(device="cuda").manual_seed(n + k)
    served_low_m = 0
    for m in ROWS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        low_m = lowm_scale is not None and lowm_kernel.can_run(x, weight, lowm_scale)
        if low_m:
            served_low_m += 1
            want = lowm_op(x, weight, lowm_scale)
        else:
            want = layers.fp8_block_scale_gemm(x, weight, weight_scale)
        # The low-M kernel the glue reports for these rows is the one forward runs.
        assert not got._runs_low_m_kernel(m) or low_m, m
        assert torch.equal(got(x), want), m
    # A full last row tile (N % 128 == 0): the low-M GEMM serves the small row counts.
    if n % 128 == 0:
        assert served_low_m > 0
