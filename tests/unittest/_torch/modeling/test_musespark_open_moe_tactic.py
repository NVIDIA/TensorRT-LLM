# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_FI_BF16_MOE_TACTIC: the pinned FlashInfer BF16 MoE tactic gives the same bits as the default."""

import pytest
import torch

E, TOPK, H, INTER = 256, 8, 4096, 4096


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="TRTLLM-Gen MoE needs SM100-family GPUs",
)
def test_flashinfer_bf16_moe_tactic_pin_bitwise(monkeypatch):
    pytest.importorskip("flashinfer")
    from flashinfer.autotuner import AutoTuner

    from tensorrt_llm._torch.moe.fused_moe import moe_op_backend as mob

    g = torch.Generator(device="cuda").manual_seed(2)
    w1 = (torch.randn((E, H // 64, 2 * INTER, 64), device="cuda", generator=g) * 0.02).bfloat16()
    w2 = (torch.randn((E, INTER // 64, H, 64), device="cuda", generator=g) * 0.02).bfloat16()
    x = torch.randn((4096, H), device="cuda", generator=g).bfloat16()
    pop = torch.randn((E,), device="cuda", generator=g) * 1.5
    u = torch.rand((4096, E), device="cuda", generator=g).clamp_(1e-9, 1 - 1e-9)
    tid = torch.topk(pop[None] - torch.log(-torch.log(u)), TOPK, dim=-1).indices.int()
    tw = torch.softmax(torch.rand((4096, TOPK), device="cuda", generator=g), -1).bfloat16()

    def run(be):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = be.run_bf16_moe(
                None,
                None,
                x,
                w1,
                w2,
                E,
                TOPK,
                None,
                None,
                INTER,
                0,
                E,
                None,
                be._trtllmgen_routing_method_type.MiniMax2,
                topk_weights=tw,
                topk_ids=tid,
                gated_act_type=0,
                use_shuffled_weight=True,
                weight_layout=2,
                do_finalize=False,
            )
            torch.cuda.synchronize()
        rows = out[0][out[2].long().flatten().clamp_min(0)].view(torch.int16)
        return rows, [e.key for e in prof.key_averages() if "mm_Bfloat16" in e.key]

    monkeypatch.delenv("TRTLLM_FI_BF16_MOE_TACTIC", raising=False)
    ref, names = run(mob.FlashinferOpBackend())
    assert any("t128x64x128" in n for n in names), names  # untuned fallback: tile 64
    monkeypatch.setenv("TRTLLM_FI_BF16_MOE_TACTIC", "2048:128:1")
    try:
        rows, names = run(mob.FlashinferOpBackend())
    finally:
        tuner = AutoTuner.get()
        tuner.__dict__.pop("choose_one", None)
        tuner.__dict__.pop("_trtllm_bf16_moe_tactic_pin", None)
    assert torch.equal(rows, ref)
    assert any("t128x128x64" in n for n in names), names
