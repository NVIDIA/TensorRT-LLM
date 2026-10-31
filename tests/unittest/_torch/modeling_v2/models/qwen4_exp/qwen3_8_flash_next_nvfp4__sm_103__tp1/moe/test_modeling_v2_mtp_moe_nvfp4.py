# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the NVFP4 copy of the draft layer's routed experts
(`TRTLLM_MODELING_V2_MTP_MOE_NVFP4`).

`weights.mtp_experts_to_nvfp4` requantizes FP8 128x128 block-scaled experts to NVFP4 in the
trunk's trtllm-gen layout, `modeling._mtp_moe_nvfp4` adds the runner's scalars and
`modeling._mtp_routed_nvfp4` runs them. Checked against the FP8 block-scale path the draft
layer runs by default, on the same random experts, activations and router: the NVFP4 output
must stay inside a relative-error band of the FP8 one, keep its magnitude (a wrong runner
scalar moves the norm, which unbiased quantization noise does not), and leave no FP8 copy
behind. A layout error is O(1) relative, far outside the band.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as m,
)

assert torch.cuda.is_available(), "the NVFP4 draft MoE requires a CUDA device"

DEV = "cuda"
E, H, INTER, K = 512, 2560, 640, 10
BLK = 128
#: A trunk calibration to derive the static draft scales from: the checkpoint's last trunk layer
#: (FC1 input, FC2 input amax), the layer that feeds the draft.
TRUNK_AMAX = (11.1875, 57.75)
#: Relative RMS of NVFP4 - FP8 over the FP8 output, the floor on each row's cosine, and the
#: band on the output norm ratio. Measured here: 0.239-0.251 RMS, row cosine >= 0.961, at every
#: row count; on the checkpoint's real draft experts 0.222-0.231 and >= 0.964 (bench). That is
#: the NVFP4 chain's own noise -- four e2m1 roundings (activation, FC1 weight, FC1 output, FC2
#: weight) at ~0.1 each, compounded by the SwiGLU product -- not a bias: it is flat across
#: activation scales 0.5-4x, and FP8 sits at 0.07 from the fp32 reference of the same weights.
REL_RMS_BAND = 0.3
COS_FLOOR = 0.95
NORM_RATIO_BAND = 0.1


def _fp8_blocks(rows: int, cols: int, gen: torch.Generator):
    """`[E, rows, cols]` e4m3 codes and `[E, rows/128, cols/128]` fp32 scales of a random
    expert stack whose 128x128 blocks span 16x in magnitude, as trained experts' do."""
    w = torch.randn(E, rows, cols, generator=gen, device=DEV) * cols**-0.5
    spread = torch.exp2(
        torch.randint(-2, 3, (E, rows // BLK, 1, cols // BLK, 1), generator=gen, device=DEV)
    ).float()
    blocks = w.view(E, rows // BLK, BLK, cols // BLK, BLK) * spread
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True) / 448.0
    codes = (blocks / scale).to(torch.float8_e4m3fn).view(E, rows, cols)
    return codes, scale.view(E, rows // BLK, cols // BLK).contiguous()


@pytest.fixture(scope="module")
def experts():
    gen = torch.Generator(device=DEV).manual_seed(20261022)
    fc1_w, fc1_s = _fp8_blocks(2 * INTER, H, gen)
    fc2_w, fc2_s = _fp8_blocks(H, INTER, gen)
    router = (torch.randn(E, H, generator=gen, device=DEV) * H**-0.5).to(torch.bfloat16)
    fp8 = SimpleNamespace(fc1_w=fc1_w, fc1_s=fc1_s, fc2_w=fc2_w, fc2_s=fc2_s, router=router.t())
    p = "l1_mtp_"
    w = {p + k: getattr(fp8, k).clone() for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")}
    w["l0_e_isc1"] = torch.full((E,), TRUNK_AMAX[0] / 2688.0, device=DEV)
    w["l0_e_isc2"] = torch.full((E,), TRUNK_AMAX[1] / 2688.0, device=DEV)
    core = SimpleNamespace(
        w=w, mtp_layer_idx=1, num_layers=1, num_experts=E, topk=K, moe_inter=INTER
    )
    fields = m._mtp_moe_nvfp4(core)
    nvfp4 = SimpleNamespace(router=fp8.router, **fields)
    return core, fp8, nvfp4


def _fp8_routed(fp8, x):
    """The FP8 block-scale draft path, call for call as `_mtp_moe_block` makes it."""
    logits = m.cublas_mm(x, fp8.router)
    ids, wts = renorm_moe_routing_op(logits, topk=K, output_dtype=torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    return fp8_block_scale_moe_runner(
        routing_logits=None,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=fp8.fc1_w,
        gemm1_weights_scale=fp8.fc1_s,
        gemm2_weights=fp8.fc2_w,
        gemm2_weights_scale=fp8.fc2_s,
        num_experts=E,
        top_k=K,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=m._ROUTING_METHOD_RENORM,
        topk_weights=wts,
        topk_ids=ids,
        act_type=m._ACT_TYPE_SWIGLU,
        gemm1_clamp_limit=None,
    )


def test_fp8_copies_freed_and_layout(experts):
    core, _, nvfp4 = experts
    for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
        assert core.w[f"l1_mtp_{k}"].numel() == 0, k
    assert nvfp4.fc1_w.shape == (E, 2 * INTER, H // 2) and nvfp4.fc1_w.dtype == torch.uint8
    assert nvfp4.fc1_s.shape == (E, 2 * INTER, H // 16) and nvfp4.fc1_s.dtype == torch.float8_e4m3fn
    assert nvfp4.fc2_w.shape == (E, H, INTER // 2) and nvfp4.fc2_s.shape == (E, H, INTER // 16)
    for s in (nvfp4.out1_scale, nvfp4.out1_gate_scale, nvfp4.out2_scale):
        assert s.shape == (E,) and s.dtype == torch.float32 and torch.isfinite(s).all()
    assert nvfp4.moe_gscale.shape == (1,)


@pytest.mark.parametrize("rows", [1, 4, 32, 128, 1024])
def test_nvfp4_tracks_fp8(experts, rows):
    core, fp8, nvfp4 = experts
    gen = torch.Generator(device=DEV).manual_seed(rows)
    x = torch.randn(rows, H, generator=gen, device=DEV).to(torch.bfloat16)
    ref = _fp8_routed(fp8, x).float()
    out = m._mtp_routed_nvfp4(core, nvfp4, x)
    assert out.shape == (rows, H) and out.dtype == torch.bfloat16
    out = out.float()
    rel = ((out - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=1).min().item()
    ratio = (out.norm() / ref.norm()).item()
    print(f"rows={rows}: rel RMS {rel:.4f}, min row cosine {cos:.4f}, norm ratio {ratio:.4f}")
    assert rel < REL_RMS_BAND, rel
    assert cos > COS_FLOOR, cos
    assert abs(ratio - 1.0) < NORM_RATIO_BAND, ratio
