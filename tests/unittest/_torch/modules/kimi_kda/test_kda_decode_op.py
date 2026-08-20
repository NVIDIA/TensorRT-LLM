# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA decode op."""

import copy

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import _kda_decode  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (  # noqa: E402
    KimiKDACachedState,
    KimiKDALinearAttention,
)

# 73: deliberately odd and > 64 to cover non-power-of-two batched decode.
BATCH_SIZE = 73
NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HIDDEN_SIZE = 7168
SUPPORTED_HEADS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96)
COMPACT_WORK_THRESHOLD = 144


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _make_attention_pair() -> tuple[KimiKDALinearAttention, KimiKDALinearAttention]:
    common = {
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
        "rms_norm_eps": 1e-5,
        "dtype": torch.bfloat16,
    }
    optimized = KimiKDALinearAttention(**common).to("cuda")
    reference = KimiKDALinearAttention(**common, use_optimized_decode=False).to("cuda")
    reference.load_state_dict(optimized.state_dict())

    assert optimized.decode_kernel_path == "optimized"
    assert reference.decode_kernel_path == "fla"
    assert reference.prefill_kernel_path == optimized.prefill_kernel_path
    return optimized, reference


def _make_cache(batch_size: int = BATCH_SIZE) -> KimiKDACachedState:
    projection_size = NUM_HEADS * HEAD_DIM
    return KimiKDACachedState(
        conv_state_q=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_k=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_v=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        recurrent_state=(
            torch.randn(
                batch_size,
                NUM_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.05
        ),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, BATCH_SIZE])
def test_optimized_decode_matches_fla_reference(batch_size: int) -> None:
    torch.manual_seed(0)
    optimized, reference = _make_attention_pair()
    hidden_states = (
        torch.randn(
            batch_size,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    initial_cache = _make_cache(batch_size)

    actual_output, actual_cache = optimized.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )
    expected_output, expected_cache = reference.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    _assert_close(actual_output, expected_output)
    _assert_close(actual_cache.recurrent_state, expected_cache.recurrent_state)
    _assert_close(actual_cache.conv_state_q, expected_cache.conv_state_q)
    _assert_close(actual_cache.conv_state_k, expected_cache.conv_state_k)
    _assert_close(actual_cache.conv_state_v, expected_cache.conv_state_v)
    assert optimized.decode_kernel_source()


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch_size", "slot_gap"),
    [(3, 0), (3, 4096), (1, 0)],
    ids=["many-dense-pool", "many-strided-pool", "compact-dense-pool"],
)
def test_optimized_decode_updates_indexed_recurrent_state_pool_in_place(
    batch_size: int,
    slot_gap: int,
) -> None:
    torch.manual_seed(1)
    optimized, _ = _make_attention_pair()
    hidden_states = (
        torch.randn(
            batch_size,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    initial_cache = _make_cache(batch_size)

    local_output, local_cache = optimized.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    slots = batch_size + 3
    slot_indices = torch.arange(
        slots - 1,
        slots - batch_size - 1,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    dense_slot_stride = NUM_HEADS * HEAD_DIM * HEAD_DIM
    state_storage = torch.randn(
        slots * (dense_slot_stride + slot_gap),
        dtype=torch.float32,
        device="cuda",
    )
    state_pool = state_storage.as_strided(
        (slots, NUM_HEADS, HEAD_DIM, HEAD_DIM),
        (dense_slot_stride + slot_gap, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    state_pool.mul_(0.2)
    assert state_pool.is_contiguous() is (slot_gap == 0)
    state_pool.index_copy_(0, slot_indices.long(), initial_cache.recurrent_state)
    unselected_indices = torch.arange(3, device="cuda")
    unselected_before = state_pool.index_select(0, unselected_indices).clone()
    indexed_cache = KimiKDACachedState(
        conv_state_q=initial_cache.conv_state_q.clone(),
        conv_state_k=initial_cache.conv_state_k.clone(),
        conv_state_v=initial_cache.conv_state_v.clone(),
        recurrent_state=state_pool,
    )

    indexed_output, indexed_cache = optimized.forward_decode(
        hidden_states,
        indexed_cache,
        ssm_state_indices=slot_indices,
    )

    assert indexed_cache.recurrent_state is state_pool
    torch.testing.assert_close(indexed_output, local_output, rtol=0, atol=0)
    torch.testing.assert_close(
        state_pool.index_select(0, slot_indices.long()),
        local_cache.recurrent_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state_pool.index_select(0, unselected_indices),
        unselected_before,
        rtol=0,
        atol=0,
    )


def _make_direct_decode_args(
    batch_size: int,
    num_heads: int,
    *,
    indexed_state: bool,
) -> dict:
    device = torch.device("cuda")
    projection_size = num_heads * HEAD_DIM
    slots = batch_size + 1 if indexed_state else batch_size
    state = torch.zeros(
        slots,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    indices = (
        torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) if indexed_state else None
    )
    return {
        "x_q": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "x_k": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "x_v": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "w_q_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "w_k_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "w_v_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "bias_q": None,
        "bias_k": None,
        "bias_v": None,
        "cs_q": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "cs_k": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "cs_v": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "A_log": torch.zeros(num_heads, dtype=torch.float32, device=device),
        "g": torch.zeros(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "dt_bias": torch.zeros(projection_size, dtype=torch.float32, device=device),
        "beta": torch.zeros(1, batch_size, num_heads, dtype=torch.bfloat16, device=device),
        "state": state,
        "onorm_g": torch.zeros(
            1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        "onorm_weight": torch.ones(HEAD_DIM, dtype=torch.float32, device=device),
        "out": torch.empty(
            batch_size,
            1,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        "ssm_state_indices": indices,
        "cu_seqlens": torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        "lower_bound": -5.0,
    }


def _profile_decode_backend(kwargs: dict) -> str:
    _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
        torch.cuda.synchronize()

    kernel_names = [
        event.key
        for event in prof.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    has_compact = any("kda_decode_fusion_compact_heads_kernel" in name for name in kernel_names)
    has_many = any("kda_decode_fusion_many_heads_kernel" in name for name in kernel_names)
    assert has_compact != has_many, kernel_names
    return "compact" if has_compact else "many"


@torch.no_grad()
@pytest.mark.parametrize("num_heads", SUPPORTED_HEADS)
def test_sm103_selector_dispatches_each_supported_head_at_boundary(num_heads: int) -> None:
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("compact-head selector sweep is tuned only for SM103")

    compact_batch = COMPACT_WORK_THRESHOLD // num_heads
    compact_args = _make_direct_decode_args(
        compact_batch,
        num_heads,
        indexed_state=False,
    )
    many_args = _make_direct_decode_args(
        compact_batch + 1,
        num_heads,
        indexed_state=False,
    )
    assert _profile_decode_backend(compact_args) == "compact"
    assert _profile_decode_backend(many_args) == "many"


@torch.no_grad()
def test_selector_preserves_legacy_compact_heads_off_sm103() -> None:
    if torch.cuda.get_device_capability(0) == (10, 3):
        pytest.skip("non-SM103 fallback requires a different Blackwell target")
    # Off SM103 the H==2 legacy rule dispatches the compact kernel; the
    # SM103-only selector must not change that.
    args = _make_direct_decode_args(1, 2, indexed_state=False)
    assert _profile_decode_backend(args) == "compact"


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch_size", "indexed_state", "expected_backend"),
    [(1, False, "compact"), (2, True, "many")],
)
def test_sm103_selector_is_cuda_graph_safe(
    batch_size: int,
    indexed_state: bool,
    expected_backend: str,
) -> None:
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("compact-head selector sweep is tuned only for SM103")

    args = _make_direct_decode_args(
        batch_size,
        96,
        indexed_state=indexed_state,
    )
    assert _profile_decode_backend(args) == expected_backend

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured_output = _kda_decode.run_kda_decode_fusion_cuda(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert captured_output is args["out"]
    assert torch.isfinite(captured_output).all()
