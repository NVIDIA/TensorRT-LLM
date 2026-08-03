# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A/B coverage for direct KDA prefill recurrent-state pool access."""

import os
import statistics

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402

HEAD_DIM = 128
LOWER_BOUND = -5.0
PERF_ENV = "TLLM_RUN_KDA_PREFILL_POOL_PERF"
RUN_PERF = os.environ.get(PERF_ENV, "0")
if RUN_PERF not in ("0", "1"):
    raise ValueError(f"{PERF_ENV} must be 0 or 1, got {RUN_PERF!r}")


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


@pytest.fixture(scope="module")
def dispatch() -> KDAKernelDispatch:
    result = KDAKernelDispatch(
        use_optimized_prefill=True,
        use_optimized_decode=False,
    )
    assert result.prefill_kernel_path == "optimized"
    return result


def _make_state_pool(layout: str, slots: int, heads: int) -> torch.Tensor:
    inner = heads * HEAD_DIM * HEAD_DIM
    if layout == "dense":
        return torch.empty(
            slots,
            heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
    if layout == "cpp_envelope":
        slot_stride = inner + 256
        storage = torch.empty(slots * slot_stride, dtype=torch.float32, device="cuda")
        return torch.as_strided(
            storage,
            (slots, heads, HEAD_DIM, HEAD_DIM),
            (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
    if layout == "v2_page_scale":
        page_index_scale = 3
        raw = torch.empty(
            slots * page_index_scale,
            heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        return raw.as_strided(
            (slots, heads, HEAD_DIM, HEAD_DIM),
            (raw.stride(0) * page_index_scale, *raw.stride()[1:]),
        )
    raise ValueError(f"Unknown state-pool layout: {layout}")


def _seed_state_pool(pool: torch.Tensor, seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(
        pool.shape,
        generator=generator,
        dtype=torch.float32,
        device=pool.device,
    )
    value_axis = torch.linspace(0.5, 1.5, HEAD_DIM, dtype=torch.float32, device=pool.device).view(
        1, 1, HEAD_DIM, 1
    )
    key_axis = torch.linspace(-0.25, 0.25, HEAD_DIM, dtype=torch.float32, device=pool.device).view(
        1, 1, 1, HEAD_DIM
    )
    pool.copy_(values * value_axis + key_axis)


def _make_inputs(heads: int, lens: list[int], seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_tokens = sum(lens)

    def _random(*shape, dtype=torch.bfloat16):
        return torch.randn(
            *shape,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).to(dtype)

    q = _random(1, total_tokens, heads, HEAD_DIM)
    k = _random(1, total_tokens, heads, HEAD_DIM)
    v = _random(1, total_tokens, heads, HEAD_DIM)
    g = _random(1, total_tokens, heads, HEAD_DIM)
    beta = _random(1, total_tokens, heads, dtype=torch.float32)
    A_log = _random(heads, dtype=torch.float32) * 0.5
    dt_bias = _random(heads * HEAD_DIM, dtype=torch.float32) * 0.1
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    return q, k, v, g, beta, A_log, dt_bias, cu_seqlens


def _run_legacy(
    dispatch: KDAKernelDispatch,
    inputs,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
) -> torch.Tensor:
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = inputs
    initial_state = state_pool.index_select(0, state_indices)
    initial_state[~has_initial_states] = 0
    output, final_state = dispatch.prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=initial_state,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
    )
    assert final_state is not None
    state_pool.index_copy_(0, state_indices, final_state)
    return output


def _run_indexed(
    dispatch: KDAKernelDispatch,
    inputs,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
) -> torch.Tensor:
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = inputs
    assert dispatch.can_use_indexed_prefill(
        state_pool=state_pool,
        state_indices=state_indices,
        has_initial_states=has_initial_states,
        cu_seqlens=cu_seqlens,
        num_tokens=q.shape[1],
    )
    output, final_state = dispatch.prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        state_pool=state_pool,
        state_indices=state_indices,
        has_initial_states=has_initial_states,
    )
    assert final_state is None
    return output


@pytest.mark.parametrize(
    "layout",
    ["dense", "cpp_envelope", "v2_page_scale"],
)
@torch.no_grad()
def test_indexed_state_pool_matches_legacy(
    dispatch: KDAKernelDispatch,
    layout: str,
) -> None:
    heads, slots = 4, 9
    inputs = _make_inputs(heads, [64, 128, 96], seed=1234)
    state_indices = torch.tensor([7, 2, 5], dtype=torch.int64, device="cuda")
    state_modes = (
        torch.tensor([False, False, False], device="cuda"),
        torch.tensor([True, True, True], device="cuda"),
        torch.tensor([True, False, True], device="cuda"),
    )

    for case_idx, has_initial_states in enumerate(state_modes):
        legacy_pool = _make_state_pool(layout, slots, heads)
        indexed_pool = _make_state_pool(layout, slots, heads)
        _seed_state_pool(legacy_pool, seed=100 + case_idx)
        indexed_pool.copy_(legacy_pool)
        untouched = indexed_pool.clone()

        legacy_output = _run_legacy(
            dispatch,
            inputs,
            legacy_pool,
            state_indices,
            has_initial_states,
        ).clone()
        indexed_output = _run_indexed(
            dispatch,
            inputs,
            indexed_pool,
            state_indices,
            has_initial_states,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(indexed_output, legacy_output, rtol=0, atol=0)
        torch.testing.assert_close(
            indexed_pool.index_select(0, state_indices),
            legacy_pool.index_select(0, state_indices),
            rtol=0,
            atol=0,
        )
        unselected = torch.ones(slots, dtype=torch.bool, device="cuda")
        unselected[state_indices] = False
        assert torch.equal(indexed_pool[unselected], untouched[unselected])


@pytest.mark.parametrize("has_initial_state", [False, True], ids=["fresh", "carried"])
@torch.no_grad()
def test_indexed_state_pool_matches_saturated_e2e_schedule(
    dispatch: KDAKernelDispatch,
    has_initial_state: bool,
) -> None:
    """Cover the H=96 persistent schedule on the executor's CUDA stream."""
    heads, slots = 96, 6
    inputs = _make_inputs(heads, [1150, 1200], seed=5678)
    state_indices = torch.tensor([5, 3], dtype=torch.int64, device="cuda")
    has_initial_states = torch.full((2,), has_initial_state, dtype=torch.bool, device="cuda")
    legacy_pool = _make_state_pool("dense", slots, heads)
    indexed_pool = _make_state_pool("dense", slots, heads)
    _seed_state_pool(legacy_pool, seed=9)
    indexed_pool.copy_(legacy_pool)

    legacy_output = _run_legacy(
        dispatch,
        inputs,
        legacy_pool,
        state_indices,
        has_initial_states,
    ).clone()
    torch.cuda.synchronize()

    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = inputs
    execution_stream = torch.cuda.Stream()
    with torch.cuda.stream(execution_stream):
        torch.cuda._sleep(1 << 28)
        streamed_inputs = (
            q * 1.0,
            k * 1.0,
            v * 1.0,
            g * 1.0,
            beta * 1.0,
            A_log,
            dt_bias,
            cu_seqlens,
        )
        indexed_output = _run_indexed(
            dispatch,
            streamed_inputs,
            indexed_pool,
            state_indices,
            has_initial_states,
        )
        indexed_output = indexed_output * 1.0
        indexed_state = indexed_pool.index_select(0, state_indices) * 1.0
    torch.cuda.synchronize()

    torch.testing.assert_close(indexed_output, legacy_output, rtol=0, atol=0)
    torch.testing.assert_close(
        indexed_state,
        legacy_pool.index_select(0, state_indices),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_indexed_state_pool_reuses_compile_across_batch_sizes(
    dispatch: KDAKernelDispatch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup must cover full and underfilled context batches."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops

    heads, slots = 5, 8
    warmup_inputs = _make_inputs(heads, [64, 192], seed=6001)
    warmup_pool = _make_state_pool("dense", slots, heads)
    _seed_state_pool(warmup_pool, seed=61)
    _run_indexed(
        dispatch,
        warmup_inputs,
        warmup_pool,
        torch.tensor([6, 2], dtype=torch.int64, device="cuda"),
        torch.tensor([True, False], device="cuda"),
    )
    torch.cuda.synchronize()

    inputs = _make_inputs(heads, [64, 64, 128], seed=6002)
    state_indices = torch.tensor([7, 3, 1], dtype=torch.int64, device="cuda")
    has_initial_states = torch.tensor([True, False, True], device="cuda")
    legacy_pool = _make_state_pool("dense", slots, heads)
    indexed_pool = _make_state_pool("dense", slots, heads)
    _seed_state_pool(legacy_pool, seed=62)
    indexed_pool.copy_(legacy_pool)

    legacy_output = _run_legacy(
        dispatch,
        inputs,
        legacy_pool,
        state_indices,
        has_initial_states,
    ).clone()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("changing only the runtime context batch size recompiled a CuTe kernel")

    monkeypatch.setattr(cute_dsl_kimi_k3_custom_ops.cute, "compile", unexpected_compile)

    indexed_output = _run_indexed(
        dispatch,
        inputs,
        indexed_pool,
        state_indices,
        has_initial_states,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(indexed_output, legacy_output, rtol=0, atol=0)
    torch.testing.assert_close(indexed_pool, legacy_pool, rtol=0, atol=0)


@torch.no_grad()
def test_indexed_state_pool_removes_full_state_copies(
    dispatch: KDAKernelDispatch,
) -> None:
    from torch.profiler import ProfilerActivity, profile

    heads, slots = 4, 9
    inputs = _make_inputs(heads, [64, 128, 96], seed=4321)
    state_indices = torch.tensor([7, 2, 5], dtype=torch.int64, device="cuda")
    has_initial_states = torch.tensor([True, False, True], device="cuda")
    legacy_pool = _make_state_pool("dense", slots, heads)
    indexed_pool = _make_state_pool("dense", slots, heads)
    _seed_state_pool(legacy_pool, seed=5)
    indexed_pool.copy_(legacy_pool)

    _run_legacy(dispatch, inputs, legacy_pool, state_indices, has_initial_states)
    _run_indexed(dispatch, inputs, indexed_pool, state_indices, has_initial_states)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as legacy_profile:
        _run_legacy(dispatch, inputs, legacy_pool, state_indices, has_initial_states)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as indexed_profile:
        _run_indexed(dispatch, inputs, indexed_pool, state_indices, has_initial_states)
    torch.cuda.synchronize()

    legacy_counts = {event.key: event.count for event in legacy_profile.key_averages()}
    indexed_counts = {event.key: event.count for event in indexed_profile.key_averages()}
    assert legacy_counts.get("aten::index_select", 0) >= 1
    assert legacy_counts.get("aten::index_copy_", 0) >= 1
    assert indexed_counts.get("aten::index_select", 0) == 0
    assert indexed_counts.get("aten::index_copy_", 0) == 0
    assert legacy_counts.get("aten::contiguous", 0) >= (
        indexed_counts.get("aten::contiguous", 0) + 2
    )


def _measure_ms(fn, iterations: int = 20) -> list[float]:
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


@pytest.mark.skipif(
    RUN_PERF == "0",
    reason=f"Set {PERF_ENV}=1 to run the KDA prefill A/B microbenchmark.",
)
@torch.no_grad()
def test_indexed_state_pool_is_faster(
    dispatch: KDAKernelDispatch,
) -> None:
    # Match one K3 attention-DP context from the 8K/1K E2E workload. Each
    # rank owns all 96 KDA heads, so this also matches its 6 MiB state row.
    heads, slots = 96, 4
    inputs = _make_inputs(heads, [8192], seed=2026)
    state_indices = torch.tensor([3], dtype=torch.int64, device="cuda")
    has_initial_states = torch.ones(1, dtype=torch.bool, device="cuda")
    legacy_pool = _make_state_pool("dense", slots, heads)
    indexed_pool = _make_state_pool("dense", slots, heads)
    _seed_state_pool(legacy_pool, seed=8)
    indexed_pool.copy_(legacy_pool)

    legacy_output = _run_legacy(
        dispatch,
        inputs,
        legacy_pool,
        state_indices,
        has_initial_states,
    ).clone()
    indexed_output = _run_indexed(
        dispatch,
        inputs,
        indexed_pool,
        state_indices,
        has_initial_states,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(indexed_output, legacy_output, rtol=0, atol=0)
    torch.testing.assert_close(indexed_pool, legacy_pool, rtol=0, atol=0)

    for _ in range(3):
        _run_legacy(dispatch, inputs, legacy_pool, state_indices, has_initial_states)
        _run_indexed(dispatch, inputs, indexed_pool, state_indices, has_initial_states)
    torch.cuda.synchronize()

    legacy_samples = _measure_ms(
        lambda: _run_legacy(dispatch, inputs, legacy_pool, state_indices, has_initial_states)
    )
    indexed_samples = _measure_ms(
        lambda: _run_indexed(dispatch, inputs, indexed_pool, state_indices, has_initial_states)
    )
    torch.testing.assert_close(indexed_pool, legacy_pool, rtol=0, atol=0)
    legacy_ms = statistics.median(legacy_samples)
    indexed_ms = statistics.median(indexed_samples)
    speedup = legacy_ms / indexed_ms
    state_mib = state_indices.numel() * heads * HEAD_DIM * HEAD_DIM * 4 / 2**20
    print(
        f"KDA prefill state-pool A/B: legacy={legacy_ms:.3f} ms, "
        f"indexed={indexed_ms:.3f} ms, speedup={speedup:.3f}x, "
        f"state={state_mib:.1f} MiB"
    )
    assert indexed_ms < legacy_ms
