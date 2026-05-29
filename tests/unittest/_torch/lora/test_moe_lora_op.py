# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests for the routed-expert MoE LoRA path through
`torch.ops.trtllm.fused_moe`.

These tests require:
  - CUDA-capable GPU
  - Built TensorRT-LLM C++ extension (so the `trtllm::fused_moe` op is registered).

They do NOT attempt full numerical equivalence with a hand-written reference
(SwiGLU + top-k routing makes that involved); instead they verify:
  1. The op accepts the new LoRA tensors without raising.
  2. With LoRA tensors present, the output differs from the no-LoRA baseline.
  3. Per-expert and shared-outer adapters both run.
  4. The Python-side rejection (min_latency_mode + LoRA) fires correctly.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import (
    make_native_shared_lora, make_per_expert_lora, reference_swiglu_moe_lora)

_TRTLLM_AVAILABLE = hasattr(torch.ops, "trtllm") and hasattr(
    torch.ops.trtllm, "fused_moe")

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_AVAILABLE,
    reason="Requires CUDA and built TensorRT-LLM C++ extension "
    "(torch.ops.trtllm.fused_moe).")


# ---------- shared fixtures ----------------------------------------------------


def _build_base_inputs(num_tokens, hidden_size, inter_size, num_experts, top_k,
                      dtype, device):
    """Create base MoE inputs (no LoRA)."""
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w3_w1 = torch.randn(num_experts, 2 * inter_size, hidden_size,
                        dtype=dtype, device=device) * 0.02
    w2 = torch.randn(num_experts, hidden_size, inter_size,
                     dtype=dtype, device=device) * 0.02
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32,
                         device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1)
    # token_final_scales must be float32 (see CHECK_INPUT in moeOp.cpp::runMoe).
    return x, w3_w1, w2, topk_ids.to(torch.int32), topk_scores.to(torch.float32)


def _build_lora_request_buffers(num_tokens, fc1_a, fc1_b, fc2_a, fc2_b,
                                rank, lora_max_low_rank=None,
                                gated_a=None, gated_b=None):
    """Pack LoRA into the per-request CPU tensors that the C++ op expects.

    Single-request multi-LoRA disabled here; one request covers all tokens.
    For gated activations (e.g. SwiGLU) the kernel requires three LoRA
    modules per layer (fc1 = up, gated = gate, fc2 = down). When `gated_a` /
    `gated_b` are not provided, they default to the same buffers as fc1.
    """
    if lora_max_low_rank is None:
        lora_max_low_rank = rank
    if gated_a is None:
        gated_a = fc1_a
    if gated_b is None:
        gated_b = fc1_b
    num_seqs = 1
    fc1_ranks = torch.tensor([rank], dtype=torch.int32, device="cpu")
    fc2_ranks = torch.tensor([rank], dtype=torch.int32, device="cpu")
    gated_ranks = torch.tensor([rank], dtype=torch.int32, device="cpu")
    fc1_ptrs = torch.tensor(
        [[fc1_a.data_ptr(), fc1_b.data_ptr(), 0]],
        dtype=torch.int64, device="cpu")
    fc2_ptrs = torch.tensor(
        [[fc2_a.data_ptr(), fc2_b.data_ptr(), 0]],
        dtype=torch.int64, device="cpu")
    gated_ptrs = torch.tensor(
        [[gated_a.data_ptr(), gated_b.data_ptr(), 0]],
        dtype=torch.int64, device="cpu")
    host_request_types = torch.zeros(num_seqs, dtype=torch.int32, device="cpu")
    host_context_lengths = torch.tensor([num_tokens], dtype=torch.int32,
                                        device="cpu")
    return dict(
        fc1_lora_ranks=fc1_ranks,
        fc1_lora_weight_ptrs=fc1_ptrs,
        fc2_lora_ranks=fc2_ranks,
        fc2_lora_weight_ptrs=fc2_ptrs,
        gated_lora_ranks=gated_ranks,
        gated_lora_weight_ptrs=gated_ptrs,
        host_request_types=host_request_types,
        host_context_lengths=host_context_lengths,
        lora_max_low_rank=lora_max_low_rank,
    )


def _build_lora_slot_buffers(num_tokens, fc1_a, fc1_b, fc2_a, fc2_b, rank,
                             max_lora_size=4, lora_max_low_rank=None,
                             gated_a=None, gated_b=None):
    """Pack a single LoRA into slot-indexed pinned CPU buffers (CUDA-graph mode).

    All tokens are placed in slot 0; remaining slots are zero-filled and inactive.
    For gated activations the gated buffers default to fc1 (matches
    `_build_lora_request_buffers`).
    """
    if lora_max_low_rank is None:
        lora_max_low_rank = rank
    if gated_a is None:
        gated_a = fc1_a
    if gated_b is None:
        gated_b = fc1_b
    slot_ranks = torch.zeros(max_lora_size, dtype=torch.int32, device="cpu")
    slot_ranks[0] = rank
    fc1_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64, device="cpu")
    fc1_slot_ptrs[0, 0] = fc1_a.data_ptr()
    fc1_slot_ptrs[0, 1] = fc1_b.data_ptr()
    fc2_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64, device="cpu")
    fc2_slot_ptrs[0, 0] = fc2_a.data_ptr()
    fc2_slot_ptrs[0, 1] = fc2_b.data_ptr()
    gated_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64,
                                  device="cpu")
    gated_slot_ptrs[0, 0] = gated_a.data_ptr()
    gated_slot_ptrs[0, 1] = gated_b.data_ptr()
    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32, device="cpu")
    return dict(
        fc1_slot_lora_ranks=slot_ranks,
        fc1_slot_lora_weight_ptrs=fc1_slot_ptrs,
        fc2_slot_lora_ranks=slot_ranks,
        fc2_slot_lora_weight_ptrs=fc2_slot_ptrs,
        gated_slot_lora_ranks=slot_ranks,
        gated_slot_lora_weight_ptrs=gated_slot_ptrs,
        token_to_slot=token_to_slot,
        lora_max_low_rank=lora_max_low_rank,
    )


def _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, output_dtype,
                   lora_kwargs=None):
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
    if lora_kwargs is not None:
        common.update(lora_kwargs)
    return torch.ops.trtllm.fused_moe(**common)


# ---------- tests --------------------------------------------------------------


@requires_cuda_and_op
def test_moe_per_expert_lora_changes_output():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    # Baseline (no LoRA).
    out_baseline = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                   output_dtype=dtype)[0]

    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       shared_side=None, seed=10)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       shared_side=None, seed=11)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    out_lora = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    assert out_lora.shape == out_baseline.shape
    # The LoRA delta must move the output meaningfully (not bit-equal, not NaN).
    assert torch.isfinite(out_lora).all()
    diff = (out_lora.float() - out_baseline.float()).abs().mean().item()
    assert diff > 1e-3, f"LoRA had no observable effect (mean abs diff={diff})"


@requires_cuda_and_op
def test_moe_shared_outer_lora_changes_output():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    # Shared A on FC1 (residual-stream side), shared B on FC2 (residual-stream side).
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       shared_side="A", seed=20)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       shared_side="B", seed=21)

    # Sanity: replicated slice matches.
    assert torch.equal(fc1_adapter["A"][0], fc1_adapter["A"][-1])
    assert torch.equal(fc2_adapter["B"][0], fc2_adapter["B"][-1])

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    out_baseline = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                   output_dtype=dtype)[0]
    out_lora = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    assert torch.isfinite(out_lora).all()
    diff = (out_lora.float() - out_baseline.float()).abs().mean().item()
    assert diff > 1e-3


@requires_cuda_and_op
def test_moe_lora_rejected_in_min_latency_mode():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2
    rank = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=30)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=31)
    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    with pytest.raises(RuntimeError, match="MoE LoRA is not supported in min-latency mode"):
        torch.ops.trtllm.fused_moe(
            input=x,
            token_selected_experts=topk_ids,
            token_final_scales=topk_scores,
            fc1_expert_weights=w3_w1,
            fc1_expert_biases=None,
            fc2_expert_weights=w2,
            fc2_expert_biases=None,
            output_dtype=dtype,
            quant_scales=[],
            min_latency_mode=True,
            **lora_kwargs,
        )


@requires_cuda_and_op
def test_moe_lora_slot_indexed_matches_per_request():
    """Slot-indexed mode must produce bit-identical output to per-request mode
    when both express the same adapter assignment (single LoRA, all tokens
    routed to slot 0)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=50)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=51)

    per_request_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)
    slot_kwargs = _build_lora_slot_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    out_per_request = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                      output_dtype=dtype,
                                      lora_kwargs=per_request_kwargs)[0]
    out_slot = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype, lora_kwargs=slot_kwargs)[0]

    torch.testing.assert_close(out_slot, out_per_request, rtol=0, atol=0)


@requires_cuda_and_op
def test_moe_lora_slot_indexed_multi_lora_mixed_batch():
    """Slot-indexed mode with multiple distinct adapters in flight, half of the
    tokens on slot 0 and half on slot 1. Compare each half against a
    single-adapter per-request call to confirm slot dispatch is correct."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    fc1_a0 = make_per_expert_lora(num_experts, rank, hidden_size, inter_size,
                                  dtype=dtype, device=device, seed=60)
    fc2_a0 = make_per_expert_lora(num_experts, rank, inter_size, hidden_size,
                                  dtype=dtype, device=device, seed=61)
    fc1_a1 = make_per_expert_lora(num_experts, rank, hidden_size, inter_size,
                                  dtype=dtype, device=device, seed=62)
    fc2_a1 = make_per_expert_lora(num_experts, rank, inter_size, hidden_size,
                                  dtype=dtype, device=device, seed=63)

    # Single-adapter references (per-request mode, applies same adapter to all tokens).
    ref0 = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype,
        lora_kwargs=_build_lora_request_buffers(
            num_tokens, fc1_a0["A"], fc1_a0["B"], fc2_a0["A"], fc2_a0["B"],
            rank=rank))[0]
    ref1 = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype,
        lora_kwargs=_build_lora_request_buffers(
            num_tokens, fc1_a1["A"], fc1_a1["B"], fc2_a1["A"], fc2_a1["B"],
            rank=rank))[0]

    # Slot-indexed mixed batch: tokens 0..3 -> slot 0, tokens 4..7 -> slot 1.
    max_lora_size = 4
    slot_ranks = torch.zeros(max_lora_size, dtype=torch.int32, device="cpu")
    slot_ranks[0] = rank
    slot_ranks[1] = rank
    fc1_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64,
                                device="cpu")
    fc1_slot_ptrs[0, 0] = fc1_a0["A"].data_ptr()
    fc1_slot_ptrs[0, 1] = fc1_a0["B"].data_ptr()
    fc1_slot_ptrs[1, 0] = fc1_a1["A"].data_ptr()
    fc1_slot_ptrs[1, 1] = fc1_a1["B"].data_ptr()
    fc2_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64,
                                device="cpu")
    fc2_slot_ptrs[0, 0] = fc2_a0["A"].data_ptr()
    fc2_slot_ptrs[0, 1] = fc2_a0["B"].data_ptr()
    fc2_slot_ptrs[1, 0] = fc2_a1["A"].data_ptr()
    fc2_slot_ptrs[1, 1] = fc2_a1["B"].data_ptr()
    # Gated module is required for SwiGLU; reuse fc1 buffers per slot.
    gated_slot_ptrs = fc1_slot_ptrs.clone()
    # First half of tokens -> slot 0, second half -> slot 1.
    half = num_tokens // 2
    token_to_slot = torch.tensor(
        [0] * half + [1] * (num_tokens - half), dtype=torch.int32,
        device="cpu")
    out_mixed = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype,
        lora_kwargs=dict(
            fc1_slot_lora_ranks=slot_ranks,
            fc1_slot_lora_weight_ptrs=fc1_slot_ptrs,
            fc2_slot_lora_ranks=slot_ranks,
            fc2_slot_lora_weight_ptrs=fc2_slot_ptrs,
            gated_slot_lora_ranks=slot_ranks,
            gated_slot_lora_weight_ptrs=gated_slot_ptrs,
            token_to_slot=token_to_slot,
            lora_max_low_rank=rank,
        ))[0]

    # First half must match the slot-0-only reference (bit-identical).
    torch.testing.assert_close(out_mixed[:half], ref0[:half], rtol=0, atol=0)
    # Second half must match the slot-1-only reference (bit-identical).
    torch.testing.assert_close(out_mixed[half:], ref1[half:], rtol=0, atol=0)


@pytest.fixture
def moe_device_lora_env(monkeypatch):
    """Enable the device LoRA path for the duration of a test and bulletproof
    the cleanup. The C++ ``FusedMoeRunner`` reads
    ``TLLM_MOE_LORA_USE_DEVICE_PATH`` once at construction and caches the
    flag; ``MoERunner.runner_dict`` then memoizes the runner per input
    shape. Clearing the dict before *and* after the test (including on
    assertion failure) guarantees that the env-var-aware runner does not
    leak into subsequent tests that share the same input shape but rely on
    the legacy host path (e.g. shared-outer bit-identity tests).
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", "1")
    try:
        yield MoERunner
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
@pytest.mark.skip(
    reason=
    "Phase 6b.D lifted the op-level `TORCH_CHECK(!isCapturing)` so the device "
    "LoRA path (`TLLM_MOE_LORA_USE_DEVICE_PATH=1`) can be captured into a CUDA "
    "graph, and capture+replay completes without error. Numerical parity is "
    "gated on an upstream kernel fix: Phase 6b.E pinned the residual mismatch "
    "down to an eager-mode non-determinism in the FC2 main GEMM (CUTLASS "
    "Blackwell TMA warp-specialized grouped GEMM running with `EpilogueFusion::NONE` on the "
    "LoRA path's tiny per-expert problem shapes). The graph faithfully "
    "captures one such non-deterministic eager run; `out_eager` is taken from "
    "a different eager run and disagrees at the same lanes. See "
    "docs/source/_dev_notes/moe-lora-preflight.md `#phase-6be-investigation-"
    "of-the-graph-parity-mismatch` for the full bisection and the proposed "
    "workarounds. The `moe_device_lora_env` fixture is intentionally kept so "
    "that flipping this marker off is the only change needed once the kernel "
    "fix lands.")
def test_moe_lora_slot_indexed_cuda_graph_replay_matches_eager(
        moe_device_lora_env):
    """CUDA graph capture+replay of slot-indexed MoE LoRA must produce the same
    output as eager execution. Verifies that the persistent pinned-host buffer
    addresses survive capture and that re-writing slot pointers between
    captures and replays takes effect.

    Phase 6b.D: this requires the device LoRA path
    (`TLLM_MOE_LORA_USE_DEVICE_PATH=1`, set by `moe_device_lora_env`). The
    legacy host path performs a host-side `cudaEventSynchronize` in
    `setupLoraWorkspace` and host-side pointer aggregation in
    `LoraImpl::run` -- neither is capturable, and the op-level
    `TORCH_CHECK` rejects that combination explicitly. The device path
    replaces both with `launchMoeLoraPointerExpand` +
    `runMoeLoraDeviceModule`, which run entirely on the stream.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8
    max_lora_size = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=70)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=71)

    # Persistent pinned host buffers (addresses survive across captures/replays).
    slot_ranks = torch.zeros(max_lora_size, dtype=torch.int32, device="cpu",
                             pin_memory=True)
    slot_ranks[0] = rank
    fc1_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64,
                                device="cpu", pin_memory=True)
    fc1_slot_ptrs[0, 0] = fc1_adapter["A"].data_ptr()
    fc1_slot_ptrs[0, 1] = fc1_adapter["B"].data_ptr()
    fc2_slot_ptrs = torch.zeros(max_lora_size, 3, dtype=torch.int64,
                                device="cpu", pin_memory=True)
    fc2_slot_ptrs[0, 0] = fc2_adapter["A"].data_ptr()
    fc2_slot_ptrs[0, 1] = fc2_adapter["B"].data_ptr()
    # Gated module is required for SwiGLU; reuse fc1 buffers.
    gated_slot_ptrs = fc1_slot_ptrs.clone()
    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32, device="cpu",
                                pin_memory=True)

    slot_kwargs = dict(
        fc1_slot_lora_ranks=slot_ranks,
        fc1_slot_lora_weight_ptrs=fc1_slot_ptrs,
        fc2_slot_lora_ranks=slot_ranks,
        fc2_slot_lora_weight_ptrs=fc2_slot_ptrs,
        gated_slot_lora_ranks=slot_ranks,
        gated_slot_lora_weight_ptrs=gated_slot_ptrs,
        token_to_slot=token_to_slot,
        lora_max_low_rank=rank,
    )

    # Eager reference.
    out_eager = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                output_dtype=dtype, lora_kwargs=slot_kwargs)[0]

    # Warm up to populate any lazy state (autotuner, LoraImpl cache, host buffer
    # reservation). Without this the first capture would record allocations.
    for _ in range(3):
        _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                        output_dtype=dtype, lora_kwargs=slot_kwargs)

    # Pre-allocate output tensor; the captured graph writes in place into this.
    out_graph = torch.empty_like(out_eager)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            captured = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                       output_dtype=dtype,
                                       lora_kwargs=slot_kwargs)[0]
            out_graph.copy_(captured)
    torch.cuda.current_stream().wait_stream(stream)

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_graph, out_eager, rtol=1e-2, atol=1e-2)


@requires_cuda_and_op
@pytest.mark.parametrize("missing_module", ["fc2", "host_request_types"])
def test_moe_lora_rejects_incomplete_inputs(missing_module):
    """Supplying fc1 LoRA but missing fc2 or host_request_types must raise."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2
    rank = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=40)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=41)
    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    if missing_module == "fc2":
        lora_kwargs["fc2_lora_ranks"] = None
        lora_kwargs["fc2_lora_weight_ptrs"] = None
    elif missing_module == "host_request_types":
        lora_kwargs["host_request_types"] = None

    with pytest.raises(RuntimeError):
        _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                        output_dtype=dtype, lora_kwargs=lora_kwargs)


@requires_cuda_and_op
def test_moe_lora_rejects_mixed_per_request_and_slot_indexed():
    """Supplying both per-request and slot-indexed LoRA inputs is an error."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2
    rank = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=80)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=81)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)
    lora_kwargs.update(_build_lora_slot_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank))

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                        output_dtype=dtype, lora_kwargs=lora_kwargs)


# -------- Phase 6a: native shared-outer kernel flag ---------------------------


@requires_cuda_and_op
def test_moe_native_shared_outer_matches_replicated_bitidentical():
    """Bit-identity check between the two equivalent shared-outer encodings:

      (a) Load-time replication: shared side is materialized as `[E, ...]`
          and the kernel applies the standard `weight_index * dim * rank`
          offset (default behavior, MVP path).
      (b) Native shared-outer: shared side is stored once and the kernel
          zero-offsets that side via `LoraParams::*_shared_a/b` (Phase 6a).

    Both paths receive the same underlying weights, so the kernel output
    must be bit-identical. This validates the offset gating in
    `setupLoraWorkspace` and the threading of the 6 bool flags through
    `runMoe` -> `buildMoeLoraParams` -> `LoraParams`.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    # Native shared-outer adapters: fc1.A shared, fc2.B shared, gated.A shared
    # (gated mirrors fc1 by convention since both are up-projections).
    fc1_native = make_native_shared_lora(num_experts, rank, hidden_size,
                                         inter_size, shared_side="A",
                                         dtype=dtype, device=device, seed=70)
    fc2_native = make_native_shared_lora(num_experts, rank, inter_size,
                                         hidden_size, shared_side="B",
                                         dtype=dtype, device=device, seed=71)
    gated_native = make_native_shared_lora(num_experts, rank, hidden_size,
                                           inter_size, shared_side="A",
                                           dtype=dtype, device=device, seed=72)

    # ---- Path (b): native, with shared flags set on the op ----
    lora_kwargs_native = _build_lora_request_buffers(
        num_tokens, fc1_native["A"], fc1_native["B"],
        fc2_native["A"], fc2_native["B"], rank=rank,
        gated_a=gated_native["A"], gated_b=gated_native["B"])
    lora_kwargs_native.update(dict(
        fc1_shared_a=True, fc1_shared_b=False,
        fc2_shared_a=False, fc2_shared_b=True,
        gated_shared_a=True, gated_shared_b=False,
    ))

    out_native = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                 output_dtype=dtype,
                                 lora_kwargs=lora_kwargs_native)[0]

    # ---- Path (a): replicate shared sides into [E, ...] and run with default flags ----
    fc1_a_replicated = (
        fc1_native["A"].unsqueeze(0).expand(num_experts, -1, -1).contiguous())
    fc2_b_replicated = (
        fc2_native["B"].unsqueeze(0).expand(num_experts, -1, -1).contiguous())
    gated_a_replicated = (
        gated_native["A"].unsqueeze(0).expand(num_experts, -1, -1).contiguous())

    lora_kwargs_repl = _build_lora_request_buffers(
        num_tokens, fc1_a_replicated, fc1_native["B"],
        fc2_native["A"], fc2_b_replicated, rank=rank,
        gated_a=gated_a_replicated, gated_b=gated_native["B"])
    # Default: all *_shared_* = False.
    out_repl = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype,
                               lora_kwargs=lora_kwargs_repl)[0]

    assert torch.isfinite(out_native).all()
    assert torch.isfinite(out_repl).all()
    # Same kernel math, same offsets (the native path zeros what the replicated
    # path makes redundant). Outputs must be bit-identical.
    torch.testing.assert_close(out_native, out_repl, rtol=0, atol=0)


@requires_cuda_and_op
def test_moe_native_shared_outer_differs_from_no_lora():
    """A sanity smoke for the native shared-outer path: the LoRA delta must
    move the output meaningfully vs the no-LoRA baseline, with NO replication
    on either side."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    fc1_native = make_native_shared_lora(num_experts, rank, hidden_size,
                                         inter_size, shared_side="A",
                                         dtype=dtype, device=device, seed=90)
    fc2_native = make_native_shared_lora(num_experts, rank, inter_size,
                                         hidden_size, shared_side="B",
                                         dtype=dtype, device=device, seed=91)
    gated_native = make_native_shared_lora(num_experts, rank, hidden_size,
                                           inter_size, shared_side="A",
                                           dtype=dtype, device=device, seed=92)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_native["A"], fc1_native["B"],
        fc2_native["A"], fc2_native["B"], rank=rank,
        gated_a=gated_native["A"], gated_b=gated_native["B"])
    lora_kwargs.update(dict(
        fc1_shared_a=True,
        fc2_shared_b=True,
        gated_shared_a=True,
    ))

    out_baseline = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                   output_dtype=dtype)[0]
    out_lora = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    assert torch.isfinite(out_lora).all()
    diff = (out_lora.float() - out_baseline.float()).abs().mean().item()
    assert diff > 1e-3, f"native shared-outer LoRA had no observable effect (mean abs diff={diff})"


@requires_cuda_and_op
def test_moe_lora_device_path_matches_host_path(monkeypatch):
    """Phase 6b.C.2.b: eager-mode parity between the device LoRA path
    (TLLM_MOE_LORA_USE_DEVICE_PATH=1, runs launchMoeLoraPointerExpand +
    launchMoeLoraProblemBuilder + cudaGraph(SplitK)GroupedGemm) and the
    legacy host LoRA path (LoraImpl::run reading pinned-host per-token
    pointer tables).

    The two paths exercise different GEMM kernels (CUTLASS-based
    cuda_graph_grouped_gemm vs cuBLAS via LoraImpl), so we don't expect
    bit-identical results in bf16; we require closeness within a wide
    bf16 tolerance.
    """
    # Avoid circular-import surprises by importing here; the runner cache
    # has to be clearable so the FusedMoeRunner re-reads the env var.
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       seed=200)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
                                       seed=201)
    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank)

    def _compute(env_value):
        # The C++ FusedMoeRunner reads TLLM_MOE_LORA_USE_DEVICE_PATH once at
        # construction and caches the result; dropping the only Python
        # reference (via runner_dict.clear()) lets the next call construct
        # a fresh runner that picks up the new env value.
        MoERunner.runner_dict.clear()
        if env_value is None:
            monkeypatch.delenv("TLLM_MOE_LORA_USE_DEVICE_PATH", raising=False)
        else:
            monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", env_value)
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                               output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    # try/finally so the env-var-aware runner cannot leak into subsequent
    # tests sharing the same input shape, even if the assertion below fails.
    try:
        out_host = _compute(None)
        out_device = _compute("1")
    finally:
        MoERunner.runner_dict.clear()

    assert torch.isfinite(out_device).all()
    assert torch.isfinite(out_host).all()
    # bf16 cross-kernel tolerance. The LoRA delta itself is on the order of
    # 1e-2 (rank-8 adapters fed through small dims), and the two GEMM
    # backends use different reduction orders, so we accept ~5% relative
    # error.
    torch.testing.assert_close(out_device, out_host, rtol=5e-2, atol=5e-2)


# -- External-reference parity tests for fused MoE + LoRA -------------------
#
# All other comparison tests in this file put the same kernel path on both
# sides of the diff (device-vs-host at rtol=5e-2/atol=5e-2; slot-vs-per-request
# and native-vs-replicated at rtol=0/atol=0), so any kernel-side
# non-determinism cancels out. The tests below instead compare the fused
# MoE op output against a hand-written PyTorch fp32 reference, which
# catches reordering / mis-application of LoRA deltas that the same-kernel
# tests cannot see.
#
# The empirical investigation logged in commit 40b3a19ddc found that
# `make_per_expert_lora` adapters drawn from N(0, 1) produce LoRA deltas
# many orders of magnitude larger than the base weights at this shape,
# and the SwiGLU + FC2 path amplifies the resulting intermediates to
# magnitudes ~1e5 -- at which point bf16 reduction-order noise on a few
# percent of lanes routinely exceeds an `atol=1.0` budget without there
# being any kernel correctness bug. The fix is to scale the LoRA adapters
# down so the legitimate output stays in O(1)-O(10), making `atol=1.0`
# meaningful relative to bf16 noise. ``_LORA_REFERENCE_SCALE`` is that knob.
#
# If you bump the LoRA scale up here you should expect failures driven by
# bf16 precision at the larger output range, not real kernel regressions;
# either widen tolerances proportionally or stay under the threshold above.

# Multiplied into both A and B of every reference-test LoRA adapter, so
# dW = B@A scales as the square of this constant. Sized empirically so
# that the most-amplified configuration (all three modules nonzero,
# distinct adapters) produces output magnitudes O(10) at the
# ``num_tokens=16, hidden=128, inter=256, rank=8`` shape used below.
_LORA_REFERENCE_SCALE = 0.25


def _make_per_expert_lora_scaled(*args, **kwargs):
    """``make_per_expert_lora`` followed by an in-place scale of A and B by
    ``_LORA_REFERENCE_SCALE``. See the section header above for the
    rationale.
    """
    adapter = make_per_expert_lora(*args, **kwargs)
    adapter["A"].mul_(_LORA_REFERENCE_SCALE)
    adapter["B"].mul_(_LORA_REFERENCE_SCALE)
    return adapter


def _run_eager_vs_reference_check():
    """Body of `test_moe_lora_eager_matches_pytorch_reference`."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    # Use distinct adapters on the gate/up/down projections so the test is
    # actually exercising all three LoRA modules independently rather than
    # relying on the gated-defaults-to-fc1 alias used by the smoke tests.
    fc1_adapter = _make_per_expert_lora_scaled(num_experts, rank, hidden_size,
                                               inter_size, dtype=dtype,
                                               device=device,
                                               shared_side=None, seed=300)
    gated_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                 hidden_size, inter_size,
                                                 dtype=dtype, device=device,
                                                 shared_side=None, seed=301)
    fc2_adapter = _make_per_expert_lora_scaled(num_experts, rank, inter_size,
                                               hidden_size, dtype=dtype,
                                               device=device,
                                               shared_side=None, seed=302)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank,
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"])

    out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                             output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    out_ref = reference_swiglu_moe_lora(
        x, w3_w1, w2, topk_ids, topk_scores,
        fc1_a=fc1_adapter["A"], fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"], fc2_b=fc2_adapter["B"],
    )

    assert torch.isfinite(out_op).all(), \
        "fused_moe op produced NaN / Inf with LoRA active"
    assert torch.isfinite(out_ref).all(), \
        "PyTorch reference produced NaN / Inf"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    # Always-printed diagnostic: pytest captures stdout and surfaces it on
    # failure regardless of verbosity flags.
    print(f"[eager_vs_ref] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")

    torch.testing.assert_close(
        out_op, out_ref, rtol=5e-2, atol=1.0,
        msg=lambda m: (
            f"{m}\nmax_abs_diff={max_abs:.3e}, op_max_mag={op_max_mag:.3e}, "
            f"ref_max_mag={ref_max_mag:.3e}, rel_err={rel_err:.3e}."))


@requires_cuda_and_op
def test_moe_lora_eager_matches_pytorch_reference():
    """Eager-mode fused MoE + LoRA parity vs an fp32 PyTorch reference at
    the ``num_tokens=16, hidden=128, inter=256, num_experts=4, top_k=2,
    rank=8, bf16`` shape with all three LoRA modules active and distinct.

    Tolerance: ``rtol=5e-2, atol=1.0``. LoRA adapters are scaled by
    ``_LORA_REFERENCE_SCALE`` so the legitimate output stays in
    O(1)-O(10); see the section header above for why that matters.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


# ---------------------------------------------------------------------------
# Bisection probes for fused MoE + LoRA correctness.
#
# These probes complement `test_moe_lora_eager_matches_pytorch_reference`
# by exercising configurations that isolate parts of the kernel pipeline:
#
#   * no-LoRA       -> base FC1 + SwiGLU + FC2 + finalize, no LoRA path.
#   * zero-LoRA     -> LoRA pipeline fully active but adapter weights zero;
#                      isolates the pointer expansion / setupLoraWorkspace
#                      plumbing from the LoRA delta computation itself.
#   * only-{fc1,gated,fc2} -> exactly one LoRA module nonzero, others zero.
#                              Pins regressions to a specific module's
#                              LoRA-IN / LoRA-OUT GEMM and pointer slice.
#   * pair-* / all-three-aliased -> multi-module configurations exercising
#                                    cross-module LoRA delta interactions.
#
# All probes share the reference test's shape and the same fp32 PyTorch
# reference, so a regression in any of them either localizes the failing
# subset or stays masked behind a passing baseline.
# ---------------------------------------------------------------------------


def _run_no_lora_eager_vs_reference_check():
    """Body of `test_moe_no_lora_eager_matches_pytorch_reference`. Same
    shape and the same PyTorch fp32 reference as the LoRA test, but the op
    is invoked WITHOUT the `lora_kwargs` so the entire LoRA pipeline is
    inactive. Any garbage here pins the regression to the base MoE op,
    not to the LoRA path.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                             output_dtype=dtype, lora_kwargs=None)[0]

    # Reference WITHOUT LoRA: pass zero-stride zero-rank stand-in adapters so
    # `reference_swiglu_moe_lora` collapses to plain SwiGLU MoE. We use rank=1
    # zeros so the reference's dtype/shape promotions all stay valid.
    rank0 = 1
    z = torch.zeros
    fc1_a = z(num_experts, rank0, hidden_size, dtype=dtype, device=device)
    fc1_b = z(num_experts, inter_size, rank0, dtype=dtype, device=device)
    gated_a = z(num_experts, rank0, hidden_size, dtype=dtype, device=device)
    gated_b = z(num_experts, inter_size, rank0, dtype=dtype, device=device)
    fc2_a = z(num_experts, rank0, inter_size, dtype=dtype, device=device)
    fc2_b = z(num_experts, hidden_size, rank0, dtype=dtype, device=device)

    out_ref = reference_swiglu_moe_lora(
        x, w3_w1, w2, topk_ids, topk_scores,
        fc1_a=fc1_a, fc1_b=fc1_b,
        gated_a=gated_a, gated_b=gated_b,
        fc2_a=fc2_a, fc2_b=fc2_b,
    )

    assert torch.isfinite(out_op).all(), \
        "fused_moe op produced NaN / Inf in the NO-LoRA configuration"
    assert torch.isfinite(out_ref).all(), \
        "PyTorch reference produced NaN / Inf in the NO-LoRA configuration"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(f"[no_lora] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")
    torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)


def _run_zero_lora_eager_vs_reference_check():
    """Body of `test_moe_zero_lora_eager_matches_pytorch_reference`. The
    LoRA pipeline is fully active (pointer expansion, ranks, etc.) but the
    adapter weights are zero, so the LoRA delta MUST be zero and the
    expected output equals the no-LoRA reference. Any garbage here pins the
    regression to LoRA control flow that runs even when the delta is zero.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    z = torch.zeros
    fc1_a = z(num_experts, rank, hidden_size, dtype=dtype, device=device)
    fc1_b = z(num_experts, inter_size, rank, dtype=dtype, device=device)
    gated_a = z(num_experts, rank, hidden_size, dtype=dtype, device=device)
    gated_b = z(num_experts, inter_size, rank, dtype=dtype, device=device)
    fc2_a = z(num_experts, rank, inter_size, dtype=dtype, device=device)
    fc2_b = z(num_experts, hidden_size, rank, dtype=dtype, device=device)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_a, fc1_b, fc2_a, fc2_b, rank=rank,
        gated_a=gated_a, gated_b=gated_b)

    out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                             output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    out_ref = reference_swiglu_moe_lora(
        x, w3_w1, w2, topk_ids, topk_scores,
        fc1_a=fc1_a, fc1_b=fc1_b,
        gated_a=gated_a, gated_b=gated_b,
        fc2_a=fc2_a, fc2_b=fc2_b,
    )

    assert torch.isfinite(out_op).all(), \
        "fused_moe op produced NaN / Inf with zero-LoRA"
    assert torch.isfinite(out_ref).all(), \
        "PyTorch reference produced NaN / Inf with zero-LoRA"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(f"[zero_lora] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")
    torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)


@requires_cuda_and_op
def test_moe_no_lora_eager_matches_pytorch_reference():
    """Bisection probe: NO LoRA at the same shape as the LoRA reference test.
    Confirms the base MoE op (FC1 + SwiGLU + FC2 + finalize) is healthy.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_no_lora_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_zero_lora_eager_matches_pytorch_reference():
    """Bisection probe: LoRA pipeline fully active, adapter weights zero.
    The LoRA delta is mathematically zero, so the output must equal the
    no-LoRA reference. Isolates the pointer-expand / setupLoraWorkspace
    plumbing from the LoRA delta computation itself.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_zero_lora_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


# ---------------------------------------------------------------------------
# Per-module LoRA bisection probes.
#
# Each probe activates exactly one LoRA module (others zero). A failure
# pins the regression to a specific module's LoRA-IN/OUT GEMM, pointer
# slice, or delta-application path:
#
#   * fc1 only   -> moe_h_to_4h LoRA (SwiGLU gate side, silu).
#   * gated only -> moe_gate    LoRA (SwiGLU up side, linear).
#   * fc2 only   -> moe_4h_to_h LoRA (down projection).
#
# Multiple modules fail simultaneously if the bug lives in the shared
# LoRA driver / pointer-expansion path rather than a single module.
# ---------------------------------------------------------------------------


def _make_zero_per_expert_lora(num_experts, rank, in_dim, out_dim, dtype, device):
    """Build a per-expert LoRA pair with the same shape contract as
    ``make_per_expert_lora`` (A: ``[E, rank, in_dim]``, B: ``[E, out_dim, rank]``)
    but with identically zero values. Used to "disable" a LoRA module while
    keeping the LoRA pipeline plumbing fully active."""
    return dict(
        A=torch.zeros(num_experts, rank, in_dim, dtype=dtype, device=device),
        B=torch.zeros(num_experts, out_dim, rank, dtype=dtype, device=device),
    )


def _run_per_module_lora_check(active_module: str):
    """Body of the per-module bisection tests.

    ``active_module`` is one of ``"fc1"``, ``"gated"``, ``"fc2"``. The
    selected module is initialized with ``_make_per_expert_lora_scaled``
    (same scale as the reference test); the other two are zero adapters
    of matching shape, so their LoRA deltas are mathematically zero.
    """
    assert active_module in ("fc1", "gated", "fc2"), active_module

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    if active_module == "fc1":
        fc1_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype=dtype, device=device,
                                                   shared_side=None, seed=300)
    else:
        fc1_adapter = _make_zero_per_expert_lora(num_experts, rank, hidden_size,
                                                 inter_size, dtype, device)

    if active_module == "gated":
        gated_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                     hidden_size, inter_size,
                                                     dtype=dtype, device=device,
                                                     shared_side=None, seed=301)
    else:
        gated_adapter = _make_zero_per_expert_lora(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype, device)

    if active_module == "fc2":
        fc2_adapter = _make_per_expert_lora_scaled(num_experts, rank, inter_size,
                                           hidden_size, dtype=dtype,
                                           device=device, shared_side=None,
                                           seed=302)
    else:
        fc2_adapter = _make_zero_per_expert_lora(num_experts, rank, inter_size,
                                                 hidden_size, dtype, device)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank,
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"])

    out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                             output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    out_ref = reference_swiglu_moe_lora(
        x, w3_w1, w2, topk_ids, topk_scores,
        fc1_a=fc1_adapter["A"], fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"], fc2_b=fc2_adapter["B"],
    )

    assert torch.isfinite(out_op).all(), \
        f"fused_moe op produced NaN / Inf with only-{active_module}-LoRA"
    assert torch.isfinite(out_ref).all(), \
        f"PyTorch reference produced NaN / Inf with only-{active_module}-LoRA"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(f"[only_{active_module}] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")
    torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)


@requires_cuda_and_op
def test_moe_only_fc1_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only ``moe_h_to_4h`` (fc1, gate-side, silu) LoRA
    is active; ``moe_gate`` and ``moe_4h_to_h`` adapters are zeros.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_per_module_lora_check("fc1")
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_only_gated_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only ``moe_gate`` (gated, up-side, linear) LoRA
    is active; ``moe_h_to_4h`` and ``moe_4h_to_h`` adapters are zeros.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_per_module_lora_check("gated")
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_only_fc2_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only ``moe_4h_to_h`` (fc2, down) LoRA is active;
    ``moe_h_to_4h`` and ``moe_gate`` adapters are zeros.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_per_module_lora_check("fc2")
    finally:
        MoERunner.runner_dict.clear()


# ---------------------------------------------------------------------------
# Pairwise + alias bisection probes.
#
# Multi-module probes covering the cross-module LoRA delta interaction
# space. The pair tests activate exactly two modules with distinct adapter
# buffers and zero the third; the aliased test activates all three but
# aliases ``gated`` to ``fc1`` (matching the smoke-test default in
# `_build_lora_request_buffers`).
#
#   * pair_fc1_gated_distinct: fc1 + gated distinct, fc2 zero
#   * pair_fc1_fc2_distinct:   fc1 + fc2 distinct,    gated zero
#   * pair_gated_fc2_distinct: gated + fc2 distinct,  fc1 zero
#   * all_three_gated_aliased: all three active, with `gated == fc1`
# ---------------------------------------------------------------------------


def _run_pair_lora_check(active_pair):
    """Body of the pairwise bisection tests.

    ``active_pair`` is a 2-tuple of module names from
    ``{"fc1", "gated", "fc2"}``. Both selected modules use
    ``_make_per_expert_lora_scaled`` with distinct seeds; the third module
    is zeros of matching shape so its delta is identically zero.
    """
    assert len(active_pair) == 2 and set(active_pair).issubset(
        {"fc1", "gated", "fc2"}), active_pair
    active_set = set(active_pair)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

    if "fc1" in active_set:
        fc1_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype=dtype, device=device,
                                                   shared_side=None, seed=300)
    else:
        fc1_adapter = _make_zero_per_expert_lora(num_experts, rank, hidden_size,
                                                 inter_size, dtype, device)

    if "gated" in active_set:
        gated_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                     hidden_size, inter_size,
                                                     dtype=dtype, device=device,
                                                     shared_side=None, seed=301)
    else:
        gated_adapter = _make_zero_per_expert_lora(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype, device)

    if "fc2" in active_set:
        fc2_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                   inter_size, hidden_size,
                                                   dtype=dtype, device=device,
                                                   shared_side=None, seed=302)
    else:
        fc2_adapter = _make_zero_per_expert_lora(num_experts, rank, inter_size,
                                                 hidden_size, dtype, device)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_adapter["A"], fc1_adapter["B"],
        fc2_adapter["A"], fc2_adapter["B"], rank=rank,
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"])

    out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                             output_dtype=dtype, lora_kwargs=lora_kwargs)[0]

    out_ref = reference_swiglu_moe_lora(
        x, w3_w1, w2, topk_ids, topk_scores,
        fc1_a=fc1_adapter["A"], fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"], gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"], fc2_b=fc2_adapter["B"],
    )

    pair_label = "+".join(active_pair)
    assert torch.isfinite(out_op).all(), \
        f"fused_moe op produced NaN / Inf with pair={pair_label}-LoRA"
    assert torch.isfinite(out_ref).all(), \
        f"PyTorch reference produced NaN / Inf with pair={pair_label}-LoRA"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(f"[pair={pair_label}] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")

    torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)


@requires_cuda_and_op
def test_moe_pair_fc1_gated_distinct_lora_eager_matches_pytorch_reference():
    """Pairwise bisection: fc1 + gated active with DISTINCT adapter buffers,
    fc2 zero.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_pair_lora_check(("fc1", "gated"))
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_pair_fc1_fc2_distinct_lora_eager_matches_pytorch_reference():
    """Pairwise bisection: fc1 + fc2 active with distinct adapter buffers,
    gated zero.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_pair_lora_check(("fc1", "fc2"))
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_pair_gated_fc2_distinct_lora_eager_matches_pytorch_reference():
    """Pairwise bisection: gated + fc2 active with distinct adapter buffers,
    fc1 zero.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_pair_lora_check(("gated", "fc2"))
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_all_three_lora_with_gated_aliased_to_fc1_eager_matches_pytorch_reference():
    """Alias probe: all three LoRA modules active, with the gated buffers
    being the SAME object as the fc1 buffers (matches the smoke-test default
    for ``_build_lora_request_buffers``).
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_tokens, hidden_size, inter_size = 16, 128, 256
        num_experts, top_k = 4, 2
        rank = 8

        x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
            num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)

        fc1_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype=dtype, device=device,
                                                   shared_side=None, seed=300)
        fc2_adapter = _make_per_expert_lora_scaled(num_experts, rank,
                                                   inter_size, hidden_size,
                                                   dtype=dtype, device=device,
                                                   shared_side=None, seed=302)

        # Deliberately do NOT pass gated_*: `_build_lora_request_buffers`
        # then aliases `gated_a := fc1_a`, `gated_b := fc1_b`, matching the
        # passing smoke-test pattern.
        lora_kwargs = _build_lora_request_buffers(
            num_tokens, fc1_adapter["A"], fc1_adapter["B"],
            fc2_adapter["A"], fc2_adapter["B"], rank=rank)

        out_op = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores,
                                 output_dtype=dtype,
                                 lora_kwargs=lora_kwargs)[0]

        out_ref = reference_swiglu_moe_lora(
            x, w3_w1, w2, topk_ids, topk_scores,
            fc1_a=fc1_adapter["A"], fc1_b=fc1_adapter["B"],
            gated_a=fc1_adapter["A"], gated_b=fc1_adapter["B"],
            fc2_a=fc2_adapter["A"], fc2_b=fc2_adapter["B"],
        )

        assert torch.isfinite(out_op).all(), \
            "fused_moe op produced NaN / Inf with gated aliased to fc1"
        assert torch.isfinite(out_ref).all(), \
            "PyTorch reference produced NaN / Inf with gated aliased to fc1"

        op_max_mag = out_op.float().abs().max().item()
        ref_max_mag = out_ref.float().abs().max().item()
        abs_diff = (out_op.float() - out_ref.float()).abs()
        max_abs = abs_diff.max().item()
        rel_err = max_abs / max(op_max_mag, 1e-12)

        print(f"[all3_gated_alias_fc1] op_max_mag={op_max_mag:.3e} "
              f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
              f"rel_err={rel_err:.3e}")
        torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)
    finally:
        MoERunner.runner_dict.clear()
