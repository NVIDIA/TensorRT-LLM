# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA decode op."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention, _kda_decode  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.kimi_kda_test_utils import (  # noqa: E402
    KimiKDAReference,
    KimiKDATestCachedState,
    get_production_decode_kernel_path,
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


def _make_attention_pair(
    *, finalize_decode_weights: bool = True
) -> tuple[KimiKDALinearAttention, KimiKDAReference]:
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
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=common["rms_norm_eps"],
        linear_attn_config={
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": common["use_full_rank_gate"],
            "gate_lower_bound": common["gate_lower_bound"],
        },
    )
    optimized = KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        optimized.dt_bias.zero_()
    reference = KimiKDAReference(**common).to("cuda")
    reference.load_state_dict(optimized.state_dict())
    if finalize_decode_weights:
        optimized.finalize_decode_weights()

    assert get_production_decode_kernel_path(optimized) == "optimized"
    return optimized, reference


def _make_cache(batch_size: int = BATCH_SIZE) -> KimiKDATestCachedState:
    projection_size = NUM_HEADS * HEAD_DIM
    return KimiKDATestCachedState(
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


def _run_production_decode(
    attention: KimiKDALinearAttention,
    hidden_states: torch.Tensor,
    initial_cache: KimiKDATestCachedState,
    *,
    conv_pool: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    slot_indices: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    include_metadata: bool = True,
) -> tuple[torch.Tensor, KimiKDATestCachedState]:
    batch_size = hidden_states.shape[0]
    projection_size = NUM_HEADS * HEAD_DIM
    if slot_indices is None:
        slot_indices = torch.arange(batch_size, device="cuda", dtype=torch.long)
    if conv_pool is None:
        conv_pool = torch.cat(
            [
                initial_cache.conv_state_q[:, :, 1:],
                initial_cache.conv_state_k[:, :, 1:],
                initial_cache.conv_state_v[:, :, 1:],
            ],
            dim=1,
        ).clone()
    if state_pool is None:
        state_pool = initial_cache.recurrent_state.clone()

    metadata = (
        SimpleNamespace(
            _arange_buffer=torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)
        )
        if include_metadata
        else None
    )
    core = attention.forward_decode(
        hidden_states.squeeze(1),
        conv_pool,
        state_pool,
        slot_indices,
        metadata,
        ssm_state_indices=ssm_state_indices,
    )
    output = attention._project_output(core)
    selected_conv = conv_pool.index_select(0, slot_indices)
    return output.unsqueeze(1), KimiKDATestCachedState(
        conv_state_q=torch.cat(
            [initial_cache.conv_state_q[:, :, 1:2], selected_conv[:, :projection_size]], dim=-1
        ),
        conv_state_k=torch.cat(
            [
                initial_cache.conv_state_k[:, :, 1:2],
                selected_conv[:, projection_size : 2 * projection_size],
            ],
            dim=-1,
        ),
        conv_state_v=torch.cat(
            [initial_cache.conv_state_v[:, :, 1:2], selected_conv[:, 2 * projection_size :]],
            dim=-1,
        ),
        recurrent_state=state_pool.index_select(0, slot_indices),
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

    actual_output, actual_cache = _run_production_decode(
        optimized, hidden_states, copy.deepcopy(initial_cache)
    )
    expected_output, expected_cache = reference.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    _assert_close(actual_output, expected_output)
    _assert_close(actual_cache.recurrent_state, expected_cache.recurrent_state)
    _assert_close(actual_cache.conv_state_q, expected_cache.conv_state_q)
    _assert_close(actual_cache.conv_state_k, expected_cache.conv_state_k)
    _assert_close(actual_cache.conv_state_v, expected_cache.conv_state_v)


@torch.no_grad()
@pytest.mark.parametrize(
    ("fallback_case", "batch_size"),
    (
        ("unfused-projections", 1),
        ("unfused-projections", BATCH_SIZE),
        ("missing-metadata", 1),
        ("capture-before-staging", 1),
    ),
    ids=(
        "unfused-projections-b1",
        "unfused-projections-odd-large-batch",
        "missing-metadata",
        "capture-before-staging",
    ),
)
def test_decode_fallback_matches_fla_reference(monkeypatch, fallback_case, batch_size):
    """Cover missing fused projections/metadata and capture-safe staging fallback."""
    torch.manual_seed(2)
    finalize_weights = fallback_case != "unfused-projections"
    optimized, reference = _make_attention_pair(finalize_decode_weights=finalize_weights)
    if fallback_case == "capture-before-staging":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

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
    actual_output, actual_cache = _run_production_decode(
        optimized,
        hidden_states,
        copy.deepcopy(initial_cache),
        include_metadata=fallback_case != "missing-metadata",
    )
    expected_output, expected_cache = reference.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    _assert_close(actual_output, expected_output)
    _assert_close(actual_cache.recurrent_state, expected_cache.recurrent_state)
    _assert_close(actual_cache.conv_state_q, expected_cache.conv_state_q)
    _assert_close(actual_cache.conv_state_k, expected_cache.conv_state_k)
    _assert_close(actual_cache.conv_state_v, expected_cache.conv_state_v)


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

    local_output, local_cache = _run_production_decode(
        optimized, hidden_states, copy.deepcopy(initial_cache)
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
    conv_pool = torch.randn(
        slots,
        3 * NUM_HEADS * HEAD_DIM,
        CONV_KERNEL_SIZE - 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    conv_pool.index_copy_(
        0,
        slot_indices.long(),
        torch.cat(
            [
                initial_cache.conv_state_q[:, :, 1:],
                initial_cache.conv_state_k[:, :, 1:],
                initial_cache.conv_state_v[:, :, 1:],
            ],
            dim=1,
        ),
    )

    indexed_output, indexed_cache = _run_production_decode(
        optimized,
        hidden_states,
        initial_cache,
        conv_pool=conv_pool,
        state_pool=state_pool,
        slot_indices=slot_indices.long(),
        ssm_state_indices=slot_indices,
    )

    torch.testing.assert_close(indexed_output, local_output, rtol=0, atol=0)
    torch.testing.assert_close(
        indexed_cache.recurrent_state,
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


@torch.no_grad()
@pytest.mark.parametrize("num_heads", [2, 32], ids=["compact-heads", "many-heads"])
def test_decode_reads_row_strided_projection_slices(num_heads: int) -> None:
    """Fused-projection views match packed inputs with direct W-1 updates."""
    torch.manual_seed(2)
    batch_size = 5
    args = _make_direct_decode_args(batch_size, num_heads, indexed_state=True)
    projection_size = num_heads * HEAD_DIM
    slots = args["state"].shape[0]
    initial_conv_pool = torch.randn(
        slots,
        3 * projection_size,
        CONV_KERNEL_SIZE - 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for name in ("g", "beta", "onorm_g"):
        args[name].normal_(std=0.01)

    def clone_args(conv_pool: torch.Tensor) -> dict:
        cloned = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
            if name not in ("cs_q", "cs_k", "cs_v")
        }
        cloned.update(
            cs_q=conv_pool[:, :projection_size],
            cs_k=conv_pool[:, projection_size : 2 * projection_size],
            cs_v=conv_pool[:, 2 * projection_size :],
            update_conv_cache=True,
        )
        return cloned

    packed_conv_pool = initial_conv_pool.clone()
    packed_args = clone_args(packed_conv_pool)
    packed_output = _kda_decode.run_kda_decode_fusion_cuda(**packed_args)

    # One wide row exercises every supported per-token row stride at once.
    row_width = 5 * projection_size + num_heads
    projection = torch.empty(
        batch_size,
        row_width,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for section, name in enumerate(("x_q", "x_k", "x_v", "onorm_g", "g")):
        projection[:, section * projection_size : (section + 1) * projection_size] = args[
            name
        ].view(batch_size, projection_size)
    projection[:, 5 * projection_size :] = args["beta"].view(batch_size, num_heads)

    qkvg = (
        projection[:, : 4 * projection_size]
        .unflatten(-1, (4, num_heads, HEAD_DIM))
        .permute(1, 0, 2, 3)
    )
    strided_g = (
        projection[:, 4 * projection_size : 5 * projection_size]
        .unflatten(-1, (num_heads, HEAD_DIM))
        .unsqueeze(0)
    )
    strided_beta = projection[:, 5 * projection_size :].unsqueeze(0)
    assert not qkvg[0:1].is_contiguous()
    assert not strided_g.is_contiguous()
    assert not strided_beta.is_contiguous()

    strided_conv_pool = initial_conv_pool.clone()
    strided_args = clone_args(strided_conv_pool)
    strided_args.update(
        x_q=qkvg[0:1],
        x_k=qkvg[1:2],
        x_v=qkvg[2:3],
        onorm_g=qkvg[3:4],
        g=strided_g,
        beta=strided_beta,
    )
    strided_output = _kda_decode.run_kda_decode_fusion_cuda(**strided_args)

    torch.testing.assert_close(strided_output, packed_output, rtol=0, atol=0)
    torch.testing.assert_close(strided_args["state"], packed_args["state"], rtol=0, atol=0)
    torch.testing.assert_close(strided_conv_pool, packed_conv_pool, rtol=0, atol=0)
    torch.testing.assert_close(strided_conv_pool[0], initial_conv_pool[0], rtol=0, atol=0)


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
