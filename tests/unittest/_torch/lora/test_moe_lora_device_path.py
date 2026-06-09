# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the *device path* of routed-expert MoE LoRA in
`torch.ops.trtllm.fused_moe`.

The device path (opted into for the per-request schema via
`TLLM_MOE_LORA_USE_DEVICE_PATH=1`) performs the per-token pointer expansion,
problem building, and grouped GEMMs entirely on the CUDA stream via the new
on-device kernels, instead of the legacy host-pointer LoRA path. This test
checks device-path eager correctness vs. both the legacy host path and an fp32
PyTorch reference, exercising the pointer-expand / problem-builder /
grouped-GEMM kernels.

It requires a CUDA GPU and the built `trtllm::fused_moe` op.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora, reference_swiglu_moe_lora

_TRTLLM_AVAILABLE = hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_moe")

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_AVAILABLE,
    reason="Requires CUDA and built TensorRT-LLM C++ extension (torch.ops.trtllm.fused_moe).",
)


@pytest.fixture(autouse=True)
def _isolate_moe_runner_cache():
    """Give every test a fresh cached FusedMoeRunner and release device scratch
    afterward.

    The device path is selected per-runner at construction from
    TLLM_MOE_LORA_USE_DEVICE_PATH, and the runner is cached at module level by
    MoERunner. Clearing the cache before each test forces a fresh runner that
    re-reads the env var; clearing + empty_cache afterward releases the
    per-runner device scratch so it cannot alias later allocations.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    MoERunner.runner_dict.clear()
    yield
    MoERunner.runner_dict.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# Adapters drawn from N(0, 1) blow up the SwiGLU intermediate at these shapes;
# scale them down so the legitimate output stays O(1)-O(10) and the bf16 noise
# stays well under the tolerance (see the rationale in test_moe_lora_op.py).
_LORA_SCALE = 0.25
_RTOL = 5e-2
_ATOL = 1.0


def _build_base_inputs(
    num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device, seed=0
):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w3_w1 = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device) * 0.02
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1)
    return x, w3_w1, w2, topk_ids.to(torch.int32), topk_scores.to(torch.float32)


def _make_adapter_set(num_experts, rank, hidden_size, inter_size, dtype, device, base_seed):
    """Three scaled per-expert adapters (fc1/gate-side, gated/up-side, fc2)."""

    def _scaled(*args, seed):
        a = make_per_expert_lora(*args, dtype=dtype, device=device, seed=seed)
        a["A"].mul_(_LORA_SCALE)
        a["B"].mul_(_LORA_SCALE)
        return a

    fc1 = _scaled(num_experts, rank, hidden_size, inter_size, seed=base_seed + 0)
    gated = _scaled(num_experts, rank, hidden_size, inter_size, seed=base_seed + 1)
    fc2 = _scaled(num_experts, rank, inter_size, hidden_size, seed=base_seed + 2)
    return {"fc1": fc1, "gated": gated, "fc2": fc2}


def _per_request_kwargs(num_tokens, adapters, rank):
    """Single-request per-request schema covering all tokens with one adapter."""
    fc1, gated, fc2 = adapters["fc1"], adapters["gated"], adapters["fc2"]
    return dict(
        fc1_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        fc1_lora_weight_ptrs=torch.tensor(
            [[fc1["A"].data_ptr(), fc1["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        fc2_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        fc2_lora_weight_ptrs=torch.tensor(
            [[fc2["A"].data_ptr(), fc2["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        gated_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        gated_lora_weight_ptrs=torch.tensor(
            [[gated["A"].data_ptr(), gated["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        host_request_types=torch.zeros(1, dtype=torch.int32, device="cpu"),
        host_context_lengths=torch.tensor([num_tokens], dtype=torch.int32, device="cpu"),
        lora_max_low_rank=rank,
    )


def _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, output_dtype, lora_kwargs):
    common = dict(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_scores,
        fc1_expert_weights=w3_w1,
        fc1_expert_biases=None,
        fc2_expert_weights=w2,
        fc2_expert_biases=None,
        output_dtype=output_dtype,
        quant_scales=[],
    )
    common.update(lora_kwargs)
    return torch.ops.trtllm.fused_moe(**common)[0]


def _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters):
    return reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=adapters["fc1"]["A"],
        fc1_b=adapters["fc1"]["B"],
        gated_a=adapters["gated"]["A"],
        gated_b=adapters["gated"]["B"],
        fc2_a=adapters["fc2"]["A"],
        fc2_b=adapters["fc2"]["B"],
    )


@requires_cuda_and_op
def test_device_path_eager_matches_host_and_reference(monkeypatch):
    """Per-request schema on the device path (env-var opt-in) must match both
    the legacy host path and the fp32 PyTorch reference. Exercises the on-device
    pointer-expand / problem-builder / grouped-GEMM kernels in eager mode.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapters = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=300
    )
    lora_kwargs = _per_request_kwargs(num_tokens, adapters, rank)

    # Host path (device path env explicitly disabled), fresh runner.
    monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", "0")
    MoERunner.runner_dict.clear()
    try:
        out_host = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(lora_kwargs))
    finally:
        MoERunner.runner_dict.clear()

    # Device path (env opt-in), fresh runner.
    monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", "1")
    MoERunner.runner_dict.clear()
    try:
        out_device = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(lora_kwargs))
    finally:
        MoERunner.runner_dict.clear()

    out_ref = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters)

    assert torch.isfinite(out_device).all()
    torch.testing.assert_close(out_device, out_ref, rtol=_RTOL, atol=_ATOL)
    # Host vs device path are different reduction orders but should agree
    # within the same bf16 tolerance.
    torch.testing.assert_close(out_device, out_host, rtol=_RTOL, atol=_ATOL)
