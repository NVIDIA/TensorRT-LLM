# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential correctness test for ``fused_expand_group_quant_fp8``.

The DeepGemm pre-GEMM pipeline used to be::

    moe_permute_op(...)  # writes the expanded buffer
    masked_index_copy_group_quant_fp8(expanded)  # re-reads it, group-quants FP8

``fused_expand_group_quant_fp8`` collapses both: instead of re-reading the
expanded (topk-materialized) buffer, it reads the *compact* source activations
directly via the permutation map, re-deriving the source row analytically
(``source_row = perm_to_unperm[i] % num_source_tokens``), and applies the
identical group-wise FP8 quantization.

Because the eliminated "expand" step is a pure data copy (no arithmetic) and the
new kernel runs the exact same quant op-sequence over the same input group, the
result is **bit-identical** -- there is no floating-point reordering, and the
outputs are FP8 bytes + packed int32 scales, so there is no FMA/rounding
ambiguity. This test therefore asserts strict equality.

To make the "expanded buffer == compact source, reordered" invariant exact, the
source ``x`` is **fp32**: ``moe_permute_op`` produces an fp32 expanded buffer, so
the old kernel (reading the fp32 expanded buffer) and the new kernel (reading
fp32 ``x``) load identical values. This is precisely the off-by-one-prone index
encoding the kernel relies on (``unpermuted_idx = k_rank * num_tokens +
token_id``, k_rank slow), so the test directly guards it against the real maps.

Run as::

    pytest tests/unittest/_torch/modules/fused_moe/test_deepgemm_fused_expand_quant.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm.*)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import (
    fused_expand_group_quant_fp8,
    masked_index_copy_group_quant_fp8,
    preprocess_after_permute,
)
from tensorrt_llm._utils import get_sm_version

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="Requires CUDA SM90+ (DeepGemm MoE pre-GEMM kernels)",
)

GROUP_SIZE = 128


@dataclass(frozen=True)
class ExpandQuantShape:
    name: str
    num_source_tokens: int
    hidden: int
    num_experts: int
    top_k: int


# Anchors mirror the PR-2 microbench sweep (study-plan note): the model anchor
# (32 x 8 x 4096 x 128), the small corner, and a medium batch. ``hidden`` is a
# multiple of the 128 quant group size (required by both kernels).
SHAPES = [
    ExpandQuantShape("anchor", num_source_tokens=32, hidden=4096, num_experts=128, top_k=8),
    ExpandQuantShape("anchor_h7168", num_source_tokens=32, hidden=7168, num_experts=128, top_k=8),
    ExpandQuantShape("small", num_source_tokens=1, hidden=512, num_experts=8, top_k=4),
    ExpandQuantShape("medium_batch", num_source_tokens=64, hidden=7168, num_experts=128, top_k=8),
    ExpandQuantShape("topk4", num_source_tokens=16, hidden=4096, num_experts=64, top_k=4),
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _alloc_outputs(num_experts: int, m_max: int, hidden: int, *, device: str):
    """Allocate the (output_q, output_s) buffer pair the way DeepGemm forward()
    does, zero-initialized so untouched padding compares equal across runs."""
    output_q = torch.zeros((num_experts, m_max, hidden), dtype=torch.float8_e4m3fn, device=device)
    m_padded = _align(m_max, 4)
    scale_k = _ceil_div(hidden, GROUP_SIZE)
    scale_k_padded = _align(scale_k, 4)
    output_s = torch.zeros(
        (num_experts, scale_k_padded // 4, m_padded), dtype=torch.int32, device=device
    )
    return output_q, output_s


@skip_unsupported
def test_skip_data_expand_omits_unused_outputs() -> None:
    num_rows, hidden, num_experts, top_k = 4, 512, 8, 4
    x = torch.randn((num_rows, hidden), device="cuda", dtype=torch.float32)
    token_selected_experts = torch.arange(top_k, device="cuda", dtype=torch.int32).repeat(
        num_rows, 1
    )
    token_final_scales = torch.full(
        (num_rows, top_k), 1.0 / top_k, device="cuda", dtype=torch.float32
    )

    outputs = torch.ops.trtllm.moe_permute_op(
        x,
        token_selected_experts,
        token_final_scales,
        None,
        None,
        None,
        input_sf=None,
        num_experts_on_rank=num_experts,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        min_latency_mode=False,
        use_fp8_block_scaling=False,
        skip_data_expand=True,
    )

    permuted_row_to_unpermuted_row_tensor = outputs[0]
    permuted_data_tensor = outputs[2]
    permuted_token_final_scales_tensor = outputs[4]
    assert permuted_row_to_unpermuted_row_tensor.numel() == num_rows * top_k
    assert permuted_data_tensor.shape == (0, hidden)
    assert permuted_token_final_scales_tensor.shape == (0,)


@skip_unsupported
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_fused_expand_quant_matches_unfused(shape: ExpandQuantShape) -> None:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(1234)

    num_rows = shape.num_source_tokens
    hidden = shape.hidden
    num_experts = shape.num_experts
    top_k = shape.top_k

    tp_size, tp_rank, ep_size, ep_rank = 1, 0, 1, 0
    cluster_size, cluster_rank = 1, 0
    num_experts_per_node = num_experts

    # Per-row distinct top-k experts with positive normalized weights.
    logits = torch.randn((num_rows, num_experts), device=device, dtype=torch.float32, generator=gen)
    topk_vals, topk_ids = logits.topk(top_k, dim=-1)
    token_selected_experts = topk_ids.to(torch.int32)
    token_final_scales = torch.softmax(topk_vals, dim=-1).to(torch.float32)

    # fp32 source: makes the expanded buffer an exact copy of x, so the old and
    # new kernels read bit-identical values (see module docstring).
    x = torch.randn((num_rows, hidden), device=device, dtype=torch.float32, generator=gen)

    # Real permutation maps. skip_data_expand defaults to False here, so the
    # expanded buffer (permuted_data_tensor) IS populated -- it is the input the
    # old (baseline) kernel reads from.
    (
        permuted_row_to_unpermuted_row_tensor,
        _permuted_token_selected_experts_tensor,
        permuted_data_tensor,
        expert_first_token_offset_tensor,
        _permuted_token_final_scales_tensor,
        _unpermuted_row_to_permuted_row_tensor,
    ) = torch.ops.trtllm.moe_permute_op(
        x,
        token_selected_experts,
        token_final_scales,
        None,  # fc1_expert_weights
        None,  # fc2_expert_weights
        None,  # quant_scales
        input_sf=None,
        num_experts_on_rank=num_experts_per_node,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        min_latency_mode=False,
        use_fp8_block_scaling=False,
    )

    num_expanded = num_rows * top_k
    assert permuted_data_tensor.shape[0] == num_expanded
    # The expanded buffer must equal x reordered for the bit-exact premise to
    # hold; moe_permute_op produces it in fp32.
    assert permuted_data_tensor.dtype == torch.float32

    _masked_m, token_to_expert_map = preprocess_after_permute(
        expert_first_token_offset_tensor, permuted_data_tensor.shape[0]
    )

    m_max = _align(num_rows, 128)

    # ---- Old path: quantize from the pre-expanded buffer ----
    out_q_old, out_s_old = _alloc_outputs(num_experts, m_max, hidden, device=device)
    masked_index_copy_group_quant_fp8(
        out_q_old,
        out_s_old,
        permuted_data_tensor,
        expert_first_token_offset_tensor,
        token_to_expert_map,
        group_size=GROUP_SIZE,
    )

    # ---- New path: fused expand + quant, reading x via the perm map ----
    out_q_new, out_s_new = _alloc_outputs(num_experts, m_max, hidden, device=device)
    fused_expand_group_quant_fp8(
        out_q_new,
        out_s_new,
        x,
        permuted_row_to_unpermuted_row_tensor,
        expert_first_token_offset_tensor,
        token_to_expert_map,
        experts_per_token=top_k,
        group_size=GROUP_SIZE,
    )

    # Strict bit-exact: FP8 quantized bytes and packed int32 scales must match
    # exactly. Compare FP8 via its int8 byte view (torch.equal does not operate
    # on float8 dtypes directly).
    q_equal = torch.equal(out_q_old.view(torch.int8), out_q_new.view(torch.int8))
    s_equal = torch.equal(out_s_old, out_s_new)
    if not (q_equal and s_equal):
        q_mismatch = (out_q_old.view(torch.int8) != out_q_new.view(torch.int8)).sum().item()
        s_mismatch = (out_s_old != out_s_new).sum().item()
        pytest.fail(
            f"[{shape.name}] fused expand+quant differs from baseline: "
            f"{q_mismatch} FP8 byte mismatches, "
            f"{s_mismatch} int32 scale mismatches"
        )


@skip_unsupported
@pytest.mark.parametrize("kernel", ["masked_index_copy", "fused_expand"], ids=lambda k: k)
def test_group_quant_fp8_under_torch_compile(kernel: str) -> None:
    """Both group-quant kernels have to survive TorchInductor.

    Inductor binds the host-side Python ``float`` args (``fp8_max``, ``eps``) as
    fp64 scalars. Unpinned, ``_absmax / fp8_max`` promotes ``output_s`` to
    float64 and the UE8M0 exponent ``bitcast`` to int32 fails to compile
    ("Cannot bitcast data-type of size 64 to data-type of size 32"). Both
    kernels pin the two scalars to fp32.

    Note ``eps`` matters as much as ``fp8_max`` here: it feeds ``_absmax``, which
    propagates into ``output_s``, so pinning ``fp8_max`` alone still compiles to
    an fp64 ``output_s``.

    On the validated torch/Inductor builds, these scalars are bound as fp64: the
    generated kernel signature emits ``'fp8_max': 'fp64'`` on torch 2.11
    (sm_120) and torch 2.12 (sm_121), both with triton 3.6.0. This guard ensures
    the kernels remain valid when Inductor uses that scalar specialization.

    ``fullgraph=True`` keeps a future graph break from turning this into a silent
    pass that never reaches Inductor.

    The permutation maps come from the real ops rather than hand-built tensors:
    Triton compiles and caches a separate kernel per argument dtype signature, so
    synthesizing them risks exercising a specialization that never runs in
    production. Building them outside ``run()`` keeps them clear of the compiled
    region, so ``fullgraph=True`` is unaffected.
    """
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(4321)

    num_rows, hidden, num_experts, top_k = 8, 512, 2, 2
    m_max = _align(num_rows, 128)

    x = torch.randn((num_rows, hidden), device=device, dtype=torch.float32, generator=gen)
    logits = torch.randn((num_rows, num_experts), device=device, dtype=torch.float32, generator=gen)
    topk_vals, topk_ids = logits.topk(top_k, dim=-1)

    (
        perm_to_unperm,
        _permuted_token_selected_experts,
        expanded,
        start_offsets,
        _permuted_token_final_scales,
        _unpermuted_row_to_permuted_row,
    ) = torch.ops.trtllm.moe_permute_op(
        x,
        topk_ids.to(torch.int32),
        torch.softmax(topk_vals, dim=-1).to(torch.float32),
        None,  # fc1_expert_weights
        None,  # fc2_expert_weights
        None,  # quant_scales
        input_sf=None,
        num_experts_on_rank=num_experts,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        min_latency_mode=False,
        use_fp8_block_scaling=False,
    )
    _masked_m, row_indices = preprocess_after_permute(start_offsets, expanded.shape[0])

    def run():
        out_q, out_s = _alloc_outputs(num_experts, m_max, hidden, device=device)
        if kernel == "masked_index_copy":
            masked_index_copy_group_quant_fp8(
                out_q, out_s, expanded, start_offsets, row_indices, group_size=GROUP_SIZE
            )
        else:
            fused_expand_group_quant_fp8(
                out_q,
                out_s,
                x,
                perm_to_unperm,
                start_offsets,
                row_indices,
                experts_per_token=top_k,
                group_size=GROUP_SIZE,
            )
        return out_q, out_s

    eager_q, eager_s = run()
    compiled_q, compiled_s = torch.compile(run, dynamic=True, fullgraph=True)()
    torch.cuda.synchronize()

    assert torch.equal(eager_q.view(torch.int8), compiled_q.view(torch.int8)), (
        f"[{kernel}] compiled FP8 output differs from eager"
    )
    assert torch.equal(eager_s, compiled_s), f"[{kernel}] compiled int32 scales differ from eager"
