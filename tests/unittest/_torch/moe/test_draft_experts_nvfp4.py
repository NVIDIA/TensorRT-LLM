# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time NVFP4 quantization of the Qwen3.5 MTP layer's BF16 experts for the TRTLLM-gen FP4 MoE."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)


def _mod():
    from tensorrt_llm._torch.modules import draft_experts_nvfp4 as m

    return m


def _bf16_moe_reference(x, router_logits, w1, w3, w2, top_k):
    probs = torch.softmax(router_logits.float(), dim=-1)
    topv, topi = probs.topk(top_k, dim=-1)
    topv = topv / topv.sum(-1, keepdim=True)
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(top_k):
            e = int(topi[t, j])
            h = x[t].float()
            a = torch.nn.functional.silu(w1[e].float() @ h) * (w3[e].float() @ h)
            out[t] += topv[t, j] * (w2[e].float() @ a)
    return out


def test_nvfp4_e2m1_rounding_grid():
    """Round to nearest, ties to the even code (cvt.rn.satfinite.e2m1x2): 0.25 -> 0, 0.75 -> 1, 1.25 -> 1,
    1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4, above 5 -> 6; negative values keep the sign, -0 stays 0."""
    m = _mod()
    vals = [0.0, 0.24, 0.25, 0.26, 0.75, 1.25, 1.3, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, -0.75, -0.1]
    grid = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, -1.0, 0.0]
    w = torch.zeros((2, 16), device="cuda")
    w[0, :15] = torch.tensor(vals, device="cuda")
    w[0, 15] = 6.0  # block amax 6 with weight_scale_2 = 1 -> block scale exactly 1
    w[1, :15] = torch.tensor(vals, device="cuda") * 0.5
    w[1, 15] = 3.0  # block amax 3 -> block scale 0.5 (exact in E4M3): same codes, half the values
    packed, sf, ws2 = m.quantize_nvfp4(w, amax=6.0 * 448.0)
    assert float(ws2) == 1.0
    assert sf.view(torch.float8_e4m3fn).float()[:, 0].tolist() == [1.0, 0.5]
    got = m.dequantize_nvfp4(packed, sf, ws2)
    assert got[0, :15].tolist() == grid, got[0]
    assert got[1, :15].tolist() == [v * 0.5 for v in grid], got[1]
    assert got[:, 15].tolist() == [6.0, 3.0]
    # -0.1 rounds to -0 (code 8, the sign kept as the hardware conversion does), -0.75 to -1 (code 8 | 2)
    codes = torch.stack([packed[0] & 0x0F, packed[0] >> 4], dim=-1).view(-1)
    assert int(codes[14]) == 0x8 and int(codes[13]) == 0xA and int(codes[0]) == 0


def test_nvfp4_matches_trtllm_fp4_quantize():
    """Same codes and block scales as trtllm::fp4_quantize (linear scale layout) for the same global scale."""
    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(21)
    w = (torch.randn((512, 4096), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w[3, 100] = 0.9  # an outlier block
    w[5, :64] = 0  # all-zero blocks
    packed, sf, ws2 = m.quantize_nvfp4(w)
    gs = (1.0 / ws2).reshape(1).float().cuda()
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, False)
    sf_ref = sf_ref.view(torch.uint8).view(512, 256)
    assert torch.equal(sf.view(torch.uint8), sf_ref)
    same = (packed == q_ref.view(torch.uint8)).float().mean().item()
    assert same > 0.9999, same  # reciprocal rounding can move rare boundary values by one code
    ours = m.dequantize_nvfp4(packed, sf, ws2)
    ref = m.dequantize_nvfp4(q_ref.view(torch.uint8), sf_ref, ws2)
    assert (ours - ref).abs().max().item() <= ours.abs().max().item() * 0.2
    rel = (ours - w.float()).norm() / w.float().norm()
    assert rel < 0.12, float(rel)


@pytest.mark.parametrize("tokens,via_backend", [(1, False), (28, True)])
def test_nvfp4_trtllm_gen_moe_vs_bf16(tokens, via_backend, monkeypatch):
    """BF16 experts converted to NVFP4 and loaded (through the ConfigurableMoE or its backend) into the trtllm-gen
    FP4 MoE match the BF16 reference within NVFP4 error."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import (
        MoEWeightLoadingMode,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )

    m = _mod()
    monkeypatch.setattr(m, "DRAFT_MOE_NVFP4_FC1_AMAX", 16.0)
    monkeypatch.setattr(m, "DRAFT_MOE_NVFP4_FC2_AMAX", 64.0)
    E, H, inter_dim, K = 64, 4096, 512, 10
    g = torch.Generator(device="cuda").manual_seed(6)
    w1 = (torch.randn((E, inter_dim, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w3 = (torch.randn((E, inter_dim, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, inter_dim), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    weights = {
        "gate_up_proj": torch.cat([w1, w3], dim=1).transpose(1, 2).contiguous(),
        "down_proj": w2.transpose(1, 2).contiguous(),
    }
    cfg = PretrainedConfig()
    cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.torch_dtype = (
        E,
        H,
        inter_dim,
        torch.bfloat16,
    )
    moe = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=K),
        reduce_results=False,
        model_config=ModelConfig(pretrained_config=cfg, moe_backend="TRTLLM", max_num_tokens=8192),
        override_quant_config=m.quant_config_for(),
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        dtype=torch.bfloat16,
    )
    m.wrap_experts_loader(moe)
    moe.cuda()  # as the model loader: parameters on the GPU before load_weights
    (getattr(moe, "backend", moe) if via_backend else moe).load_weights([weights])
    if hasattr(moe, "post_load_weights"):
        moe.post_load_weights()
    backend = getattr(moe, "backend", moe)
    assert backend.has_nvfp4
    assert float(backend.fc31_input_scale) == pytest.approx(2688.0 / 16.0)
    assert float(backend.fc2_input_scale) == pytest.approx(2688.0 / 64.0)
    x = torch.randn((tokens, H), generator=g, device="cuda").to(torch.bfloat16)
    logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode(), autotune():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = moe.forward(x, logits).float()
    torch.cuda.synchronize()
    ref = _bf16_moe_reference(x, logits, w1, w3, w2, K)
    rel = (out - ref).norm() / ref.norm()
    # Four NVFP4 roundings compound to ~0.23; a wrong layout / scale gives ~1.
    assert rel < 0.30, float(rel)
