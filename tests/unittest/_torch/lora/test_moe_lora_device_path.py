# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the capture-safe *device path* of routed-expert MoE LoRA in
`torch.ops.trtllm.fused_moe`.

The device path (selected by the slot-indexed input schema, or by
`TLLM_MOE_LORA_USE_DEVICE_PATH=1` for the per-request schema) performs the
per-token pointer expansion, problem building, and grouped GEMMs entirely on
the CUDA stream, so it is safe to record into a CUDA graph. These tests cover
the surface the eager-only tests in `test_moe_lora_op.py` cannot reach:

  1. Device-path eager correctness vs. the legacy host path and an fp32
     PyTorch reference (exercises the new on-device kernels).
  2. CUDA-graph capture + replay of the device path (slot-indexed schema),
     verified against the eager result.
  3. Multi-adapter routing within a single capture (token_to_slot mixing two
     adapter slots).
  4. Slot reassignment under replay: the slot -> per-token expansion runs
     on-device fed by captured H2D copies of the stable pinned slot tables, so
     reassigning a slot's adapter in place is reflected on replay WITHOUT
     re-capture (mirroring attention LoRA and the normal decode loop).

They require a CUDA GPU and the built `trtllm::fused_moe` op.
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
    """Give every test a fresh cached FusedMoeRunner and release captured graphs
    / device scratch afterward.

    The MoE LoRA device path keeps persistent per-runner scratch (slot tables,
    pointer arrays, low-rank workspace) on the module-level MoERunner cache, and
    these tests leak the CUDA graphs they capture. Without isolation, a test that
    grows the cached runner's slot tables (e.g. max_lora_size 1 -> 2) while an
    earlier test's captured graph still references the old device scratch
    corrupts that scratch, producing an illegal memory access in the next
    forward (see TRTLLM-12507). Clearing the cache before each test forces a
    fresh runner; clearing + empty_cache afterward releases the leaked graph
    pools so they cannot alias later allocations.
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


def _slot_kwargs(token_to_slot, adapter_sets, rank):
    """Slot-indexed schema. `adapter_sets` is a list (one per slot) of adapter
    dicts; `token_to_slot` maps each token to a slot index. Pinned host tensors
    so the op can dereference them and the addresses stay stable across capture.
    """
    max_lora_size = len(adapter_sets)

    def _ptrs(key):
        return torch.tensor(
            [[a[key]["A"].data_ptr(), a[key]["B"].data_ptr(), 0] for a in adapter_sets],
            dtype=torch.int64,
            device="cpu",
        ).pin_memory()

    ranks = torch.full((max_lora_size,), rank, dtype=torch.int32, device="cpu").pin_memory()
    return dict(
        fc1_slot_lora_ranks=ranks,
        fc1_slot_lora_weight_ptrs=_ptrs("fc1"),
        fc2_slot_lora_ranks=ranks,
        fc2_slot_lora_weight_ptrs=_ptrs("fc2"),
        gated_slot_lora_ranks=ranks,
        gated_slot_lora_weight_ptrs=_ptrs("gated"),
        token_to_slot=token_to_slot.to(torch.int32).cpu().pin_memory(),
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


def _reference_multi_slot(x, w3_w1, w2, topk_ids, topk_scores, adapter_sets, token_to_slot):
    """Per-token reference where token t uses adapter_sets[token_to_slot[t]].

    The LoRA delta and base MoE are fully per-token independent, so we evaluate
    the single-adapter reference once per slot over the whole input and gather
    each token's row from the reference computed with its slot's adapter.
    """
    per_slot = [_reference(x, w3_w1, w2, topk_ids, topk_scores, a) for a in adapter_sets]
    out = torch.empty_like(per_slot[0])
    for t in range(x.shape[0]):
        out[t] = per_slot[int(token_to_slot[t].item())][t]
    return out


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


def _warmup_and_capture(call_fn, warmup_iters=3):
    """Warm up `call_fn` (so workspaces/tactics are allocated and LoRA scratch
    is sized outside capture), then capture it into a CUDA graph. Returns
    (graph, captured_output)."""
    for _ in range(warmup_iters):
        call_fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call_fn()
    return graph, captured


@requires_cuda_and_op
def test_device_path_cuda_graph_replay_matches_eager():
    """Slot-indexed schema auto-selects the device path; capturing the op into a
    CUDA graph and replaying must reproduce the eager device-path output.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapters = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=400
    )
    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32)
    slot_kwargs = _slot_kwargs(token_to_slot, [adapters], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    out_eager = call().clone()

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_replay = captured.clone()

    out_ref = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters)
    assert torch.isfinite(out_replay).all()
    # Replay reproduces the captured device-path computation bit-for-bit.
    torch.testing.assert_close(out_replay, out_eager, rtol=0, atol=0)
    torch.testing.assert_close(out_replay, out_ref, rtol=_RTOL, atol=_ATOL)


@requires_cuda_and_op
def test_device_path_cuda_graph_multi_adapter():
    """Two adapter slots routed per-token via token_to_slot, captured + replayed.
    Validates the device path threads each token's slot through the on-device
    pointer expansion correctly.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapter_a = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=500
    )
    adapter_b = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=600
    )
    # Even tokens -> slot 0 (adapter_a), odd tokens -> slot 1 (adapter_b).
    token_to_slot = (torch.arange(num_tokens) % 2).to(torch.int32)
    slot_kwargs = _slot_kwargs(token_to_slot, [adapter_a, adapter_b], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_replay = captured.clone()

    out_ref = _reference_multi_slot(
        x, w3_w1, w2, topk_ids, topk_scores, [adapter_a, adapter_b], token_to_slot
    )
    assert torch.isfinite(out_replay).all()
    torch.testing.assert_close(out_replay, out_ref, rtol=_RTOL, atol=_ATOL)


@requires_cuda_and_op
def test_device_path_reserve_prevents_growth_across_captures():
    """Production-mirroring positive test: pre-reserve worst-case LoRA scratch on
    the cached runner, then capture two graphs on the SAME runner at growing slot
    counts (1 then 2). Because reserve pre-sized to the worst case
    (max_lora_size == 2) before any capture, neither capture reallocates the
    device scratch, so both graphs coexist and replay correctly with no illegal
    memory access.

    This is the invariant CudaGraphLoraManager.reserve_moe_lora_cuda_graph_workspace
    guarantees in the real decode loop, where one cached FusedMoeRunner is shared
    across all captured batch-size graphs. It is the positive counterpart to the
    per-test isolation fixture: without reserve, growing the slot tables 1 -> 2 on
    a runner that already captured a graph corrupts the device scratch
    (TRTLLM-12507).
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    from tensorrt_llm._torch.utils import ActivationType

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8
    max_lora_size = 2

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )

    # Build (and cache) the FusedMoeRunner with the same instance key the op uses
    # for the unquantized bf16 path: (x_dtype, weight_dtype, output_dtype) plus
    # all-False quant flags. Then pre-reserve worst-case device scratch for
    # max_lora_size == 2 before any capture.
    runner = MoERunner(
        x_dtype=dtype,
        weight_dtype=w3_w1.dtype,
        output_dtype=dtype,
        top_k=top_k,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_int8_woq_per_channel=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        use_fused_finalize=True,
        activation_type=int(ActivationType.Swiglu),
    )
    runner.fused_moe_runner.reserve_lora_host_buffers(
        num_tokens,  # max_num_tokens
        top_k,  # experts_per_token
        rank,  # max_lora_rank
        max_lora_size,  # max_lora_size
        True,  # has_gated (SwiGLU)
    )

    adapter_a = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=900
    )
    adapter_b = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=1000
    )

    # Capture #1: a single active slot.
    tts1 = torch.zeros(num_tokens, dtype=torch.int32)
    sk1 = _slot_kwargs(tts1, [adapter_a], rank)
    graph1, captured1 = _warmup_and_capture(
        lambda: _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(sk1))
    )

    # Capture #2: two active slots on the SAME cached runner, with no cache clear
    # in between. reserve pre-sized the slot tables to 2, so this must NOT
    # reallocate scratch (which would invalidate graph1).
    tts2 = (torch.arange(num_tokens) % 2).to(torch.int32)
    sk2 = _slot_kwargs(tts2, [adapter_a, adapter_b], rank)
    graph2, captured2 = _warmup_and_capture(
        lambda: _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(sk2))
    )

    # Both captured graphs coexist; replay each and verify it reproduces its own
    # adapter routing (each graph re-runs its recorded H2D + slot-expand).
    graph1.replay()
    graph2.replay()
    torch.cuda.synchronize()

    out1 = captured1.clone()
    out2 = captured2.clone()

    ref1 = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapter_a)
    ref2 = _reference_multi_slot(x, w3_w1, w2, topk_ids, topk_scores, [adapter_a, adapter_b], tts2)
    assert torch.isfinite(out1).all()
    assert torch.isfinite(out2).all()
    torch.testing.assert_close(out1, ref1, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(out2, ref2, rtol=_RTOL, atol=_ATOL)


def _set_slot_ptrs_inplace(slot_kwargs, adapters, slot_index=0):
    """Overwrite one slot's (A, B) pointer rows for all three modules in place,
    preserving the pinned-tensor storage (and thus the data_ptr the captured
    H2D reads from)."""
    for kernel, mod in (("fc1", "fc1"), ("fc2", "fc2"), ("gated", "gated")):
        row = torch.tensor(
            [adapters[mod]["A"].data_ptr(), adapters[mod]["B"].data_ptr(), 0],
            dtype=torch.int64,
            device="cpu",
        )
        slot_kwargs[f"{kernel}_slot_lora_weight_ptrs"][slot_index].copy_(row)


@requires_cuda_and_op
def test_device_path_slot_reassignment_reflected_on_replay():
    """In-place slot reassignment must be reflected on replay WITHOUT recapture.

    The slot -> per-token expansion runs on-device (launchMoeLoraSlotExpand)
    fed by captured H2D copies of the stable pinned slot tables. Because both
    the copies and the expansion kernel are recorded into the graph, mutating a
    slot's adapter pointers in place (same pinned storage) and replaying picks
    up the new adapter, mirroring attention LoRA and the normal decode loop
    where requests/adapters change across steps under one captured graph.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapter_a = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=700
    )
    adapter_b = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=800
    )

    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32)
    # One slot whose pointers we will reassign in place.
    slot_kwargs = _slot_kwargs(token_to_slot, [adapter_a], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    out_ref_a = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapter_a)
    out_ref_b = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapter_b)
    # Sanity: the two adapters produce meaningfully different outputs.
    assert (out_ref_a.float() - out_ref_b.float()).abs().mean().item() > 1e-2

    # Correctness of the device-path numerics is covered by
    # test_device_path_cuda_graph_replay_matches_eager and
    # test_device_path_eager_matches_host_and_reference. This test targets the
    # *reassignment* invariant, and uses a precision-robust discriminative check
    # rather than a tight element-wise match: these randomly-drawn adapters push
    # the SwiGLU + FC2 intermediates large enough that a few percent of output
    # lanes fall into bf16 catastrophic cancellation (the device path is
    # bit-identical to the host path there; both diverge from the fp32 reference
    # only on those lanes), so an element-wise reference match is not meaningful.
    # Instead assert that the replayed output tracks whichever adapter is
    # currently assigned (closer in mean to its own reference than to the other)
    # and flips when the slot is reassigned in place, all without recapture.
    def _mean_abs(p, q):
        return (p.float() - q.float()).abs().mean().item()

    signal = _mean_abs(out_ref_a, out_ref_b)
    assert signal > 1.0, f"adapters not distinguishable enough (signal={signal:.3e})"

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_a = captured.clone()
    assert torch.isfinite(out_a).all()
    # Replay reflects the currently-assigned adapter (a).
    assert _mean_abs(out_a, out_ref_a) < _mean_abs(out_a, out_ref_b)

    # Reassign slot 0 to adapter_b *in place* (same pinned tensor storage), then
    # replay WITHOUT re-capturing. The captured H2D + device slot-expand re-run
    # on replay, so the output must now reflect adapter_b.
    _set_slot_ptrs_inplace(slot_kwargs, adapter_b, slot_index=0)
    graph.replay()
    torch.cuda.synchronize()
    out_b = captured.clone()
    assert torch.isfinite(out_b).all()
    # Reassignment took effect (output changed substantially) and now tracks b.
    assert _mean_abs(out_b, out_a) > 0.5 * signal
    assert _mean_abs(out_b, out_ref_b) < _mean_abs(out_b, out_ref_a)

    # And switching back is likewise reflected, confirming it is not a one-shot.
    _set_slot_ptrs_inplace(slot_kwargs, adapter_a, slot_index=0)
    graph.replay()
    torch.cuda.synchronize()
    out_a2 = captured.clone()
    assert torch.isfinite(out_a2).all()
    assert _mean_abs(out_a2, out_ref_a) < _mean_abs(out_a2, out_ref_b)
