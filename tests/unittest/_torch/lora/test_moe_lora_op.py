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
  3. Per-expert adapters run and match a hand-written PyTorch reference.
  4. The Python-side rejection (min_latency_mode + LoRA) fires correctly.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora, reference_swiglu_moe_lora

_TRTLLM_AVAILABLE = hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_moe")

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_AVAILABLE,
    reason="Requires CUDA and built TensorRT-LLM C++ extension (torch.ops.trtllm.fused_moe).",
)


# ---------- shared fixtures ----------------------------------------------------


def _build_base_inputs(num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device):
    """Create base MoE inputs (no LoRA)."""
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w3_w1 = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device) * 0.02
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1)
    # token_final_scales must be float32 (see CHECK_INPUT in moeOp.cpp::runMoe).
    return x, w3_w1, w2, topk_ids.to(torch.int32), topk_scores.to(torch.float32)


def _build_lora_request_buffers(
    num_tokens, fc1_a, fc1_b, fc2_a, fc2_b, rank, lora_max_low_rank=None, gated_a=None, gated_b=None
):
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
        [[fc1_a.data_ptr(), fc1_b.data_ptr(), 0]], dtype=torch.int64, device="cpu"
    )
    fc2_ptrs = torch.tensor(
        [[fc2_a.data_ptr(), fc2_b.data_ptr(), 0]], dtype=torch.int64, device="cpu"
    )
    gated_ptrs = torch.tensor(
        [[gated_a.data_ptr(), gated_b.data_ptr(), 0]], dtype=torch.int64, device="cpu"
    )
    host_request_types = torch.zeros(num_seqs, dtype=torch.int32, device="cpu")
    host_context_lengths = torch.tensor([num_tokens], dtype=torch.int32, device="cpu")
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


def _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, output_dtype, lora_kwargs=None):
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
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    # Baseline (no LoRA).
    out_baseline = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype)[0]

    fc1_adapter = make_per_expert_lora(
        num_experts,
        rank,
        hidden_size,
        inter_size,
        dtype=dtype,
        device=device,
        seed=10,
    )
    fc2_adapter = make_per_expert_lora(
        num_experts,
        rank,
        inter_size,
        hidden_size,
        dtype=dtype,
        device=device,
        seed=11,
    )

    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
    )

    out_lora = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
    )[0]

    assert out_lora.shape == out_baseline.shape
    # The LoRA delta must move the output meaningfully (not bit-equal, not NaN).
    assert torch.isfinite(out_lora).all()
    diff = (out_lora.float() - out_baseline.float()).abs().mean().item()
    assert diff > 1e-3, f"LoRA had no observable effect (mean abs diff={diff})"


@requires_cuda_and_op
def test_moe_lora_rejected_in_min_latency_mode():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2
    rank = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    fc1_adapter = make_per_expert_lora(
        num_experts, rank, hidden_size, inter_size, dtype=dtype, device=device, seed=30
    )
    fc2_adapter = make_per_expert_lora(
        num_experts, rank, inter_size, hidden_size, dtype=dtype, device=device, seed=31
    )
    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
    )

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
@pytest.mark.parametrize("missing_module", ["fc2", "host_request_types"])
def test_moe_lora_rejects_incomplete_inputs(missing_module):
    """Supplying fc1 LoRA but missing fc2 or host_request_types must raise."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2
    rank = 4

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    fc1_adapter = make_per_expert_lora(
        num_experts, rank, hidden_size, inter_size, dtype=dtype, device=device, seed=40
    )
    fc2_adapter = make_per_expert_lora(
        num_experts, rank, inter_size, hidden_size, dtype=dtype, device=device, seed=41
    )
    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
    )

    if missing_module == "fc2":
        lora_kwargs["fc2_lora_ranks"] = None
        lora_kwargs["fc2_lora_weight_ptrs"] = None
    elif missing_module == "host_request_types":
        lora_kwargs["host_request_types"] = None

    with pytest.raises(RuntimeError):
        _call_fused_moe(
            x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
        )


# -- External-reference parity tests for fused MoE + LoRA -------------------
#
# The tests below compare the fused
# MoE op output against a hand-written PyTorch fp32 reference, which
# catches reordering / mis-application of LoRA deltas that the same-kernel
# tests cannot see.
#
# Empirically, `make_per_expert_lora` adapters drawn from N(0, 1) produce LoRA deltas
# many orders of magnitude larger than the base weights at this shape,
# and the SwiGLU + FC2 path amplifies the resulting intermediates to
# magnitudes ~1e5 -- at which point bf16 reduction-order noise on a few
# percent of lanes routinely exceeds an `atol=1.0` budget without there
# being any kernel correctness bug. The fix is to scale the LoRA adapters
# down so the legitimate output stays in O(1)-O(10), making `atol=1.0`
# meaningful relative to bf16 noise. `_LORA_REFERENCE_SCALE` is that knob.
#
# If you bump the LoRA scale up here you should expect failures driven by
# bf16 precision at the larger output range, not real kernel regressions;
# either widen tolerances proportionally or stay under the threshold above.

# Multiplied into both A and B of every reference-test LoRA adapter, so
# dW = B@A scales as the square of this constant. Sized empirically so
# that the most-amplified configuration (all three modules nonzero,
# distinct adapters) produces output magnitudes O(10) at the
# `num_tokens=16, hidden=128, inter=256, rank=8` shape used below.
_LORA_REFERENCE_SCALE = 0.25


def _make_per_expert_lora_scaled(*args, **kwargs):
    """`make_per_expert_lora` followed by an in-place scale of A and B by
    `_LORA_REFERENCE_SCALE`. See the section header above for the
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
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    # Use distinct adapters on the gate/up/down projections so the test is
    # actually exercising all three LoRA modules independently rather than
    # relying on the gated-defaults-to-fc1 alias used by the smoke tests.
    fc1_adapter = _make_per_expert_lora_scaled(
        num_experts,
        rank,
        hidden_size,
        inter_size,
        dtype=dtype,
        device=device,
        seed=300,
    )
    gated_adapter = _make_per_expert_lora_scaled(
        num_experts,
        rank,
        hidden_size,
        inter_size,
        dtype=dtype,
        device=device,
        seed=301,
    )
    fc2_adapter = _make_per_expert_lora_scaled(
        num_experts,
        rank,
        inter_size,
        hidden_size,
        dtype=dtype,
        device=device,
        seed=302,
    )

    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
    )

    out_op = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
    )[0]

    out_ref = reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=fc1_adapter["A"],
        fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"],
        fc2_b=fc2_adapter["B"],
    )

    assert torch.isfinite(out_op).all(), "fused_moe op produced NaN / Inf with LoRA active"
    assert torch.isfinite(out_ref).all(), "PyTorch reference produced NaN / Inf"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    # Always-printed diagnostic: pytest captures stdout and surfaces it on
    # failure regardless of verbosity flags.
    print(
        f"[eager_vs_ref] op_max_mag={op_max_mag:.3e} "
        f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
        f"rel_err={rel_err:.3e}"
    )

    torch.testing.assert_close(
        out_op,
        out_ref,
        rtol=5e-2,
        atol=1.0,
        msg=lambda m: (
            f"{m}\nmax_abs_diff={max_abs:.3e}, op_max_mag={op_max_mag:.3e}, "
            f"ref_max_mag={ref_max_mag:.3e}, rel_err={rel_err:.3e}."
        ),
    )


@requires_cuda_and_op
def test_moe_lora_eager_matches_pytorch_reference():
    """Eager-mode fused MoE + LoRA parity vs an fp32 PyTorch reference at
    the `num_tokens=16, hidden=128, inter=256, num_experts=4, top_k=2,
    rank=8, bf16` shape with all three LoRA modules active and distinct.

    Tolerance: `rtol=5e-2, atol=1.0`. LoRA adapters are scaled by
    `_LORA_REFERENCE_SCALE` so the legitimate output stays in
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
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    out_op = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=None
    )[0]

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
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=fc1_a,
        fc1_b=fc1_b,
        gated_a=gated_a,
        gated_b=gated_b,
        fc2_a=fc2_a,
        fc2_b=fc2_b,
    )

    assert torch.isfinite(out_op).all(), (
        "fused_moe op produced NaN / Inf in the NO-LoRA configuration"
    )
    assert torch.isfinite(out_ref).all(), (
        "PyTorch reference produced NaN / Inf in the NO-LoRA configuration"
    )

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(
        f"[no_lora] op_max_mag={op_max_mag:.3e} "
        f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
        f"rel_err={rel_err:.3e}"
    )
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
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    z = torch.zeros
    fc1_a = z(num_experts, rank, hidden_size, dtype=dtype, device=device)
    fc1_b = z(num_experts, inter_size, rank, dtype=dtype, device=device)
    gated_a = z(num_experts, rank, hidden_size, dtype=dtype, device=device)
    gated_b = z(num_experts, inter_size, rank, dtype=dtype, device=device)
    fc2_a = z(num_experts, rank, inter_size, dtype=dtype, device=device)
    fc2_b = z(num_experts, hidden_size, rank, dtype=dtype, device=device)

    lora_kwargs = _build_lora_request_buffers(
        num_tokens, fc1_a, fc1_b, fc2_a, fc2_b, rank=rank, gated_a=gated_a, gated_b=gated_b
    )

    out_op = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
    )[0]

    out_ref = reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=fc1_a,
        fc1_b=fc1_b,
        gated_a=gated_a,
        gated_b=gated_b,
        fc2_a=fc2_a,
        fc2_b=fc2_b,
    )

    assert torch.isfinite(out_op).all(), "fused_moe op produced NaN / Inf with zero-LoRA"
    assert torch.isfinite(out_ref).all(), "PyTorch reference produced NaN / Inf with zero-LoRA"

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(
        f"[zero_lora] op_max_mag={op_max_mag:.3e} "
        f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
        f"rel_err={rel_err:.3e}"
    )
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
    `make_per_expert_lora` (A: `[E, rank, in_dim]`, B: `[E, out_dim, rank]`)
    but with identically zero values. Used to "disable" a LoRA module while
    keeping the LoRA pipeline plumbing fully active."""
    return dict(
        A=torch.zeros(num_experts, rank, in_dim, dtype=dtype, device=device),
        B=torch.zeros(num_experts, out_dim, rank, dtype=dtype, device=device),
    )


def _run_per_module_lora_check(active_module: str):
    """Body of the per-module bisection tests.

    `active_module` is one of `"fc1"`, `"gated"`, `"fc2"`. The
    selected module is initialized with `_make_per_expert_lora_scaled`
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
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    if active_module == "fc1":
        fc1_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            hidden_size,
            inter_size,
            dtype=dtype,
            device=device,
            seed=300,
        )
    else:
        fc1_adapter = _make_zero_per_expert_lora(
            num_experts, rank, hidden_size, inter_size, dtype, device
        )

    if active_module == "gated":
        gated_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            hidden_size,
            inter_size,
            dtype=dtype,
            device=device,
            seed=301,
        )
    else:
        gated_adapter = _make_zero_per_expert_lora(
            num_experts, rank, hidden_size, inter_size, dtype, device
        )

    if active_module == "fc2":
        fc2_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            inter_size,
            hidden_size,
            dtype=dtype,
            device=device,
            seed=302,
        )
    else:
        fc2_adapter = _make_zero_per_expert_lora(
            num_experts, rank, inter_size, hidden_size, dtype, device
        )

    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
    )

    out_op = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
    )[0]

    out_ref = reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=fc1_adapter["A"],
        fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"],
        fc2_b=fc2_adapter["B"],
    )

    assert torch.isfinite(out_op).all(), (
        f"fused_moe op produced NaN / Inf with only-{active_module}-LoRA"
    )
    assert torch.isfinite(out_ref).all(), (
        f"PyTorch reference produced NaN / Inf with only-{active_module}-LoRA"
    )

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(
        f"[only_{active_module}] op_max_mag={op_max_mag:.3e} "
        f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
        f"rel_err={rel_err:.3e}"
    )
    torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)


@requires_cuda_and_op
def test_moe_only_fc1_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only `moe_h_to_4h` (fc1, gate-side, silu) LoRA
    is active; `moe_gate` and `moe_4h_to_h` adapters are zeros.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    MoERunner.runner_dict.clear()
    try:
        _run_per_module_lora_check("fc1")
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_only_gated_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only `moe_gate` (gated, up-side, linear) LoRA
    is active; `moe_h_to_4h` and `moe_4h_to_h` adapters are zeros.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    MoERunner.runner_dict.clear()
    try:
        _run_per_module_lora_check("gated")
    finally:
        MoERunner.runner_dict.clear()


@requires_cuda_and_op
def test_moe_only_fc2_lora_eager_matches_pytorch_reference():
    """Per-module bisection: only `moe_4h_to_h` (fc2, down) LoRA is active;
    `moe_h_to_4h` and `moe_gate` adapters are zeros.
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
# aliases `gated` to `fc1` (the default in `_build_lora_request_buffers`,
# which reuses the fc1 buffers for the gated module).
#
#   * pair_fc1_gated_distinct: fc1 + gated distinct, fc2 zero
#   * pair_fc1_fc2_distinct:   fc1 + fc2 distinct,    gated zero
#   * pair_gated_fc2_distinct: gated + fc2 distinct,  fc1 zero
#   * all_three_gated_aliased: all three active, with `gated == fc1`
# ---------------------------------------------------------------------------


def _run_pair_lora_check(active_pair):
    """Body of the pairwise bisection tests.

    `active_pair` is a 2-tuple of module names from
    `{"fc1", "gated", "fc2"}`. Both selected modules use
    `_make_per_expert_lora_scaled` with distinct seeds; the third module
    is zeros of matching shape so its delta is identically zero.
    """
    assert len(active_pair) == 2 and set(active_pair).issubset({"fc1", "gated", "fc2"}), active_pair
    active_set = set(active_pair)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k = 4, 2
    rank = 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    if "fc1" in active_set:
        fc1_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            hidden_size,
            inter_size,
            dtype=dtype,
            device=device,
            seed=300,
        )
    else:
        fc1_adapter = _make_zero_per_expert_lora(
            num_experts, rank, hidden_size, inter_size, dtype, device
        )

    if "gated" in active_set:
        gated_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            hidden_size,
            inter_size,
            dtype=dtype,
            device=device,
            seed=301,
        )
    else:
        gated_adapter = _make_zero_per_expert_lora(
            num_experts, rank, hidden_size, inter_size, dtype, device
        )

    if "fc2" in active_set:
        fc2_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            inter_size,
            hidden_size,
            dtype=dtype,
            device=device,
            seed=302,
        )
    else:
        fc2_adapter = _make_zero_per_expert_lora(
            num_experts, rank, inter_size, hidden_size, dtype, device
        )

    lora_kwargs = _build_lora_request_buffers(
        num_tokens,
        fc1_adapter["A"],
        fc1_adapter["B"],
        fc2_adapter["A"],
        fc2_adapter["B"],
        rank=rank,
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
    )

    out_op = _call_fused_moe(
        x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
    )[0]

    out_ref = reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=fc1_adapter["A"],
        fc1_b=fc1_adapter["B"],
        gated_a=gated_adapter["A"],
        gated_b=gated_adapter["B"],
        fc2_a=fc2_adapter["A"],
        fc2_b=fc2_adapter["B"],
    )

    pair_label = "+".join(active_pair)
    assert torch.isfinite(out_op).all(), (
        f"fused_moe op produced NaN / Inf with pair={pair_label}-LoRA"
    )
    assert torch.isfinite(out_ref).all(), (
        f"PyTorch reference produced NaN / Inf with pair={pair_label}-LoRA"
    )

    op_max_mag = out_op.float().abs().max().item()
    ref_max_mag = out_ref.float().abs().max().item()
    abs_diff = (out_op.float() - out_ref.float()).abs()
    max_abs = abs_diff.max().item()
    rel_err = max_abs / max(op_max_mag, 1e-12)

    print(
        f"[pair={pair_label}] op_max_mag={op_max_mag:.3e} "
        f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
        f"rel_err={rel_err:.3e}"
    )

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
    being the same object as the fc1 buffers (the default in
    `_build_lora_request_buffers`).
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
            num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
        )

        fc1_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            hidden_size,
            inter_size,
            dtype=dtype,
            device=device,
            seed=300,
        )
        fc2_adapter = _make_per_expert_lora_scaled(
            num_experts,
            rank,
            inter_size,
            hidden_size,
            dtype=dtype,
            device=device,
            seed=302,
        )

        # Deliberately do NOT pass gated_*: `_build_lora_request_buffers`
        # then aliases `gated_a := fc1_a`, `gated_b := fc1_b`, matching the
        # passing smoke-test pattern.
        lora_kwargs = _build_lora_request_buffers(
            num_tokens,
            fc1_adapter["A"],
            fc1_adapter["B"],
            fc2_adapter["A"],
            fc2_adapter["B"],
            rank=rank,
        )

        out_op = _call_fused_moe(
            x, w3_w1, w2, topk_ids, topk_scores, output_dtype=dtype, lora_kwargs=lora_kwargs
        )[0]

        out_ref = reference_swiglu_moe_lora(
            x,
            w3_w1,
            w2,
            topk_ids,
            topk_scores,
            fc1_a=fc1_adapter["A"],
            fc1_b=fc1_adapter["B"],
            gated_a=fc1_adapter["A"],
            gated_b=fc1_adapter["B"],
            fc2_a=fc2_adapter["A"],
            fc2_b=fc2_adapter["B"],
        )

        assert torch.isfinite(out_op).all(), (
            "fused_moe op produced NaN / Inf with gated aliased to fc1"
        )
        assert torch.isfinite(out_ref).all(), (
            "PyTorch reference produced NaN / Inf with gated aliased to fc1"
        )

        op_max_mag = out_op.float().abs().max().item()
        ref_max_mag = out_ref.float().abs().max().item()
        abs_diff = (out_op.float() - out_ref.float()).abs()
        max_abs = abs_diff.max().item()
        rel_err = max_abs / max(op_max_mag, 1e-12)

        print(
            f"[all3_gated_alias_fc1] op_max_mag={op_max_mag:.3e} "
            f"ref_max_mag={ref_max_mag:.3e} max_abs_diff={max_abs:.3e} "
            f"rel_err={rel_err:.3e}"
        )
        torch.testing.assert_close(out_op, out_ref, rtol=5e-2, atol=1.0)
    finally:
        MoERunner.runner_dict.clear()
