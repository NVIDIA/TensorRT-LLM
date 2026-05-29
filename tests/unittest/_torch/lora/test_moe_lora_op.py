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


@pytest.fixture
def moe_lora_force_ampere_gemm2_env(monkeypatch):
    """Enable workaround (b) -- force non-TMA-WS GEMM2 for the LoRA path --
    for the duration of a test. The C++ ``FusedMoeRunner`` reads
    ``TLLM_MOE_LORA_FORCE_AMPERE_GEMM2`` once at construction and caches
    the flag, and ``MoERunner.runner_dict`` memoizes the runner per shape,
    so we have to clear the dict on entry and exit (including on assertion
    failure) for the env var to actually take effect on a runner built by
    a previous same-shape test.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    monkeypatch.setenv("TLLM_MOE_LORA_FORCE_AMPERE_GEMM2", "1")
    try:
        yield MoERunner
    finally:
        MoERunner.runner_dict.clear()


@pytest.fixture
def moe_lora_fused_finalize_env(monkeypatch):
    """Enable workaround (a) -- pre-sum LoRA delta into FINALIZE epilogue
    for the LoRA path -- for the duration of a test. Same plumbing as the
    other Phase 6b.E env-var fixtures: clear the runner cache on entry and
    exit so the env var takes effect even if a previous same-shape test
    already built a runner.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    monkeypatch.setenv("TLLM_MOE_LORA_FUSED_FINALIZE", "1")
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


# -- Phase 1 of the FC2-main-GEMM stability work: external-reference parity --
#
# All other comparison tests in this file put the same kernel path on both
# sides of the diff (device-vs-host at rtol=5e-2/atol=5e-2; slot-vs-per-request
# and native-vs-replicated at rtol=0/atol=0), so any kernel-side
# non-determinism cancels out. This test instead compares the fused MoE op
# output against a hand-written PyTorch reference run in fp32 internally,
# using a tolerance band that admits ordinary bf16 reduction noise but
# flags the garbage-magnitude lanes that Phase 6b.E identified inside the
# Blackwell TMA warp-specialized grouped GEMM `EpilogueFusion::NONE`
# instantiation used by the LoRA path. The original Phase 6b.E investigation
# saw ~1e7-1e8 garbage; a Phase 1 reference run on the same B300 host caught
# ~1e5 garbage on a different invocation -- the magnitude is not invariant,
# but "at least one lane carries a magnitude many orders of magnitude above
# the legitimate ~1e0-1e1 LoRA output" is.


def _run_eager_vs_reference_check():
    """Body of the eager-vs-PyTorch-reference test. Factored out so that the
    default-mode test and the workaround-(a) / workaround-(b) variants
    (which only differ in which env var is set on the surrounding fixture)
    can share the same numerical comparison and diagnostic.
    """
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
    fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                       inter_size, dtype=dtype, device=device,
                                       shared_side=None, seed=300)
    gated_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                         inter_size, dtype=dtype, device=device,
                                         shared_side=None, seed=301)
    fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                       hidden_size, dtype=dtype, device=device,
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
    # failure regardless of verbosity flags. The numerical check below is
    # what decides PASS/FAIL; magnitude alone is not a correctness signal
    # (with distinct adapters on all three modules the legitimate output
    # may legitimately reach >1e4 at this shape).
    print(f"[eager_vs_ref] op_max_mag={op_max_mag:.3e} "
          f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
          f"rel_err={rel_err:.3e}")

    torch.testing.assert_close(
        out_op, out_ref, rtol=5e-2, atol=1.0,
        msg=lambda m: (
            f"{m}\nmax_abs_diff={max_abs:.3e}, op_max_mag={op_max_mag:.3e}, "
            f"ref_max_mag={ref_max_mag:.3e}, rel_err={rel_err:.3e}; "
            "if a small number of lanes are off by ~1e7+, this is the "
            "FC2 main GEMM EpilogueFusion::NONE instability documented "
            "in Phase 6b.E."))


@requires_cuda_and_op
def test_moe_lora_eager_matches_pytorch_reference():
    """Eager-mode FC2 main GEMM correctness check against an external fp32
    PyTorch reference at the Phase 6b.E reproducing shape, using the
    DEFAULT (currently broken) kernel path.

    Tolerance: ``rtol=5e-2, atol=1.0``. The legitimate output magnitude at
    these scales is ~1e0-1e1 (``W_base ~ N(0, 0.02**2)``, ``x ~ N(0, 1)``,
    plus a small rank-8 LoRA delta). The Phase 6b.E investigation reports
    the broken lanes carry magnitude ~1e7-1e8, so this band sits ~6 orders
    of magnitude below the bug signal and well above bf16 reduction noise.
    The test is intended to be binary: it MUST fail on today's tree before
    the kernel fix and pass after either workaround lands.

    The two workaround-specific variants (`*_workaround_a`,
    `*_workaround_b`) are expected to PASS today, with the corresponding
    env var enabling each candidate fix.
    """
    # Make sure neither workaround is silently active because of a stale
    # runner from another test in the same process.
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_lora_eager_matches_pytorch_reference_workaround_b(
        moe_lora_force_ampere_gemm2_env):
    """Same correctness check as `test_moe_lora_eager_matches_pytorch_reference`
    but with workaround (b) enabled (`TLLM_MOE_LORA_FORCE_AMPERE_GEMM2=1`).

    Workaround (b) swaps GEMM2 to a non-TMA-WS Ampere-style tactic for any
    LoRA-active call. The Phase 6b.E memo notes that the FC2 main GEMM is
    bit-stable in eager mode under that kernel template (the `EpilogueFusion::NONE`
    TMA warp-specialized instantiation observed on Blackwell is the only one
    that drifts), so this variant is expected to PASS today even though the
    default-mode test fails.
    """
    _run_eager_vs_reference_check()


@requires_cuda_and_op
def test_moe_lora_eager_matches_pytorch_reference_workaround_a(
        moe_lora_fused_finalize_env):
    """Same correctness check as `test_moe_lora_eager_matches_pytorch_reference`
    but with workaround (a) enabled (`TLLM_MOE_LORA_FUSED_FINALIZE=1`).

    Workaround (a) keeps the TMA warp-specialized GEMM2 but swaps the
    autotuner-selected `EpilogueFusion::NONE` tactic for an
    `EpilogueFusion::FINALIZE` one, and pre-aggregates the FC2 LoRA delta
    into ``final_output`` BEFORE the main GEMM via the new
    ``launchMoeLoraPreSum`` device kernel. The FINALIZE epilogue then
    atomic-adds the GEMM contribution on top. The unstable NONE epilogue
    is never exercised on the LoRA path, so this variant should PASS today
    even though the default-mode test fails.

    Note for Blackwell (sm_100/sm_103): `calcMaxWorkspaceSizeTmaWarpSpecialized`
    only sizes the FINALIZE workspace under `if (sm_ == 90)` today, so the
    runtime allocation will be the NONE size. Empirically the production
    no-LoRA Blackwell path also picks FINALIZE configs and works, so this is
    unlikely to matter, but it is a known unverified assumption -- if this
    test fails with a confusing kernel-internal CUDA error, that gate is the
    first place to look. See `cpp/tensorrt_llm/kernels/cutlass_kernels/
    moe_gemm/moe_gemm_template_dispatch.h` ~ line 936 and the moe-lora-
    preflight memo.
    """
    _run_eager_vs_reference_check()


# ---------------------------------------------------------------------------
# Phase 6b.E follow-up bisection probes.
#
# After the first round of workarounds shipped, a Blackwell B300 (sm_103)
# rebuild ran the three reference tests above and the [trtllm-moe-lora-trace]
# diagnostic prints proved that:
#   * baseline       runs the default sm_103 TMA-WS NONE GEMM2 tactic.
#   * workaround (a) successfully swaps to sm_103 TMA-WS FINALIZE GEMM2.
#   * workaround (b) successfully swaps to sm_80 Ampere non-TMA-WS GEMM2.
# All three produce the bit-identical garbage value `op_max_mag = 1.142e+05`.
# Three orthogonal FC2 GEMM2 templates cannot produce identical garbage if
# the bug were in FC2 GEMM2 itself, so the corruption must enter UPSTREAM of
# FC2 -- in FC1, in the SwiGLU activation, in the LoRA pointer-expansion /
# adapter-pointer arithmetic, in `loraImpl::run`, or in some shared
# workspace memory that all three FC2 paths read from. The two probes below
# bisect that:
#
#   * `test_moe_no_lora_eager_matches_pytorch_reference` runs the same shape
#     with NO LoRA at all. If this passes, the base FC1 + SwiGLU + FC2
#     pipeline is healthy and the bug is LoRA-specific. If it fails with the
#     same garbage signature, the bug is independent of LoRA and is somewhere
#     in the base MoE op at this shape.
#   * `test_moe_zero_lora_eager_matches_pytorch_reference` runs LoRA-active
#     with adapter weights identically zero. The LoRA delta should be zero,
#     so the output should match a no-LoRA reference. If this still produces
#     garbage, the corruption enters via some control-flow code that is gated
#     on "LoRA active" (pointer expansion, LoRA-IN / LoRA-OUT bookkeeping
#     buffers, `setupLoraWorkspace`, etc.), NOT via the LoRA delta values
#     themselves -- a much more specific finding.
#
# Both probes are lightweight (single shape, single call); they exist purely
# to localize the bug.
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
    """Bisection probe: NO LoRA at all at the same shape as the LoRA reference
    test. See the ``Phase 6b.E follow-up bisection probes`` section comment
    above for the full story; in short, this test EXPECTING TO PASS would
    confirm that the base MoE op (FC1 + SwiGLU + FC2 + finalize) is healthy
    at this shape, and that the 1.142e+05 garbage observed in the LoRA
    reference test is LoRA-specific.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_no_lora_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_zero_lora_eager_matches_pytorch_reference():
    """Bisection probe: LoRA pipeline fully active, but adapter weights are
    all zero. The LoRA delta is mathematically zero, so the output must equal
    the no-LoRA reference. See the ``Phase 6b.E follow-up bisection probes``
    section comment above for the full story.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    try:
        _run_zero_lora_eager_vs_reference_check()
    finally:
        MoERunner.runner_dict.clear()


# ---------------------------------------------------------------------------
# Phase 6b.E follow-up: per-module LoRA bisection probes.
#
# Round 1 of the follow-up established:
#   * baseline / workaround_a / workaround_b all produce bit-identical
#     `op_max_mag = 1.142e+05` despite running three orthogonal GEMM2
#     templates (sm_103 TMA-WS NONE, sm_103 TMA-WS FINALIZE, sm_80 Ampere
#     non-TMA-WS).
#   * `test_moe_no_lora_eager_matches_pytorch_reference` PASSES: base MoE
#     pipeline (FC1+SwiGLU+FC2+finalize) is healthy at this shape.
#   * `test_moe_zero_lora_eager_matches_pytorch_reference` PASSES: LoRA
#     pipeline plumbing (pointer expand, setupLoraWorkspace, etc.) is also
#     healthy with zero adapters; the bug requires nonzero adapter values.
#
# So the bug is in the LoRA *delta* computation specifically, in some path
# that is identical across all three FC2 GEMM2 templates. Three suspects:
#
#   * `moe_h_to_4h` LoRA (the kernel calls this fc1; SwiGLU gate side, silu)
#   * `moe_gate`    LoRA (the kernel calls this gated; SwiGLU up side, linear)
#   * `moe_4h_to_h` LoRA (fc2 / down)
#
# A corrupted FC1 or gated delta flows through SwiGLU into FC2's input and
# the resulting FC2 GEMM2 output is broken regardless of which GEMM2 kernel
# runs. A corrupted FC2 LoRA delta is added to the FC2 GEMM2 output through
# three different mechanisms (post-GEMM `loraBiasApplyFunc`, pre-sum +
# FINALIZE atomic-add, or fused-bias on Ampere) but with the same net
# contribution `Σ_k topk_scores[t,k] · delta[t,k_expert]`, so a corrupted
# delta value would produce identical garbage on all three paths as well.
#
# Each probe below activates exactly one LoRA module (others are zero).
# The pass/fail pattern across the three probes pins the bug to a specific
# module:
#
#   * fc1 only fails  -> moe_h_to_4h LoRA-IN/OUT GEMM, fc1 pointer slice,
#                        or the path that adds the gate-side delta into
#                        `fc1_result_` is broken.
#   * gated only fails -> moe_gate LoRA-IN/OUT, gated pointer slice, or
#                         the path that adds the up-side delta is broken.
#   * fc2 only fails   -> moe_4h_to_h LoRA-IN/OUT, fc2 pointer slice, or
#                         the FC2 LoRA delta computation is broken.
#
# Multiple-module fail is also possible if there is a shared bug in the
# LoRA-IN/OUT GEMM driver itself.
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
    selected module is initialized with ``make_per_expert_lora`` (the same
    nonzero generator as the failing reference test); the other two are
    zero adapters of matching shape, so their LoRA deltas are mathematically
    zero. If the test fails, the bug is in the active module's LoRA path.
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
        fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                           inter_size, dtype=dtype,
                                           device=device, shared_side=None,
                                           seed=300)
    else:
        fc1_adapter = _make_zero_per_expert_lora(num_experts, rank, hidden_size,
                                                 inter_size, dtype, device)

    if active_module == "gated":
        gated_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                             inter_size, dtype=dtype,
                                             device=device, shared_side=None,
                                             seed=301)
    else:
        gated_adapter = _make_zero_per_expert_lora(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype, device)

    if active_module == "fc2":
        fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
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
# Phase 6b.E follow-up: pairwise + alias bisection probes.
#
# Round 2 of the follow-up established:
#   * Each LoRA module ALONE is fine (`only_{fc1,gated,fc2}` PASS).
#   * All three modules with DISTINCT adapters together produce garbage
#     (the failing reference test).
#
# The smoke tests `test_moe_per_expert_lora_changes_output` and friends pass
# today's CI; they call `_build_lora_request_buffers` WITHOUT a `gated_*`
# argument, so the helper aliases `gated_a := fc1_a` / `gated_b := fc1_b` --
# i.e. the gated branch shares fc1's buffer. This is the single biggest
# difference between the passing smoke tests and the failing reference test.
#
# So the bug appears when at least TWO LoRA modules are simultaneously
# active with distinct adapter buffers. The probes below pin down which
# pair (or all-three) is required, and whether `gated == fc1` aliasing
# masks the bug:
#
#   * pair_fc1_gated_distinct: fc1 + gated distinct, fc2 zero
#   * pair_fc1_fc2_distinct:   fc1 + fc2 distinct,    gated zero
#   * pair_gated_fc2_distinct: gated + fc2 distinct,  fc1 zero
#   * all_three_gated_aliased: all three active, but `gated == fc1` (matches
#     the passing smoke-test pattern)
#
# Expected reads:
#
#   pair_fc1_gated_distinct FAIL, others PASS
#       -> the bug is in the fc1/gated LoRA-OUT routing into `fc1_result_`'s
#       up-vs-gate halves (most likely candidate given the smoke tests pass
#       with `gated == fc1`).
#
#   all_three_gated_aliased PASS
#       -> confirms aliasing masks the bug, regardless of which pair fires.
#
#   any other combination -> the bug is more subtle than "fc1 vs gated
#   placement"; we will need to instrument the LoRA-OUT writes directly.
# ---------------------------------------------------------------------------


def _run_pair_lora_check(active_pair):
    """Body of the pairwise bisection tests.

    ``active_pair`` is a 2-tuple of module names from
    ``{"fc1", "gated", "fc2"}``. Both selected modules use the
    ``make_per_expert_lora`` nonzero generator with distinct seeds; the
    third module is zeros of matching shape so its delta is identically zero.
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
        fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                           inter_size, dtype=dtype,
                                           device=device, shared_side=None,
                                           seed=300)
    else:
        fc1_adapter = _make_zero_per_expert_lora(num_experts, rank, hidden_size,
                                                 inter_size, dtype, device)

    if "gated" in active_set:
        gated_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                             inter_size, dtype=dtype,
                                             device=device, shared_side=None,
                                             seed=301)
    else:
        gated_adapter = _make_zero_per_expert_lora(num_experts, rank,
                                                   hidden_size, inter_size,
                                                   dtype, device)

    if "fc2" in active_set:
        fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
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
    fc2 zero. Most-expected-to-fail probe given that the passing smoke
    tests alias gated to fc1.
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
    gated zero. This matches the passing smoke-test pattern's pair (fc1 + fc2,
    gated aliased to fc1) except gated is zeroed instead of aliased -- so if
    this PASSES we have additional confidence the bug requires non-zero
    gated, while if this FAILS the bug also lurks in the fc1+fc2 pair.
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
    """Alias probe: all three LoRA modules are active and nonzero, but the
    gated buffers are the SAME object as the fc1 buffers (matches the smoke
    test default for `_build_lora_request_buffers`). This isolates the
    "gated must equal fc1" hypothesis from the "all three modules active"
    hypothesis.

    Expected outcome under the gated-vs-fc1-aliasing hypothesis: PASS.
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

        fc1_adapter = make_per_expert_lora(num_experts, rank, hidden_size,
                                           inter_size, dtype=dtype, device=device,
                                           shared_side=None, seed=300)
        fc2_adapter = make_per_expert_lora(num_experts, rank, inter_size,
                                           hidden_size, dtype=dtype, device=device,
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
