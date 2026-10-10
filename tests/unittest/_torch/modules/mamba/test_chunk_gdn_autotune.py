# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cold autotuning must not advance the caller's recurrent state repeatedly."""

from collections.abc import Callable
from typing import Literal

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("state_mode", ["indexed_zero", "indexed_nonzero", "dense", "none"])
@pytest.mark.parametrize("use_gate", [False, True])
@torch.inference_mode()
def test_chunk_gdn_cold_autotune_preserves_initial_state(
    monkeypatch: pytest.MonkeyPatch,
    state_mode: Literal["indexed_zero", "indexed_nonzero", "dense", "none"],
    use_gate: bool,
) -> None:
    from tensorrt_llm._torch.modules.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )

    # Clear only the tuning decisions, leaving the compiled kernel cache intact.
    # Disable disk tuning results so this test also exercises a cold call in CI.
    autotuner = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    monkeypatch.setattr(autotuner, "cache", {})
    monkeypatch.setattr(autotuner, "cache_results", False)

    torch.manual_seed(2026)
    lengths = [17, 65]
    heads, key_heads, dim = 4, 2, 64
    total = sum(lengths)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.05

    k = randn(1, total, key_heads, dim)
    w = randn(1, total, heads, dim)
    u = randn(1, total, heads, dim)
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int64)
    g = None
    if use_gate:
        positions = torch.cat([torch.arange(n, device="cuda") % 64 + 1 for n in lengths])
        g = (-0.02 * positions.float()).view(1, total, 1).expand(-1, -1, heads).contiguous()

    indexed = state_mode.startswith("indexed")
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32) if indexed else None
    initial = None
    if state_mode != "none":
        slots = 5 if indexed else len(lengths)
        initial = torch.randn(slots, heads, dim, dim, device="cuda") * 0.1
        if state_mode == "indexed_zero":
            initial[indices.long()] = 0

    def run() -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None]:
        state = None if initial is None else initial.clone()
        result = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=state,
            initial_state_indices=indices,
            inplace_indexed_state_update=indexed,
            output_final_state=True,
            cu_seqlens=cu,
        )
        torch.cuda.synchronize()
        return result, state

    cold, cold_pool = run()
    warm, warm_pool = run()
    for cold_tensor, warm_tensor in zip(cold, warm):
        torch.testing.assert_close(cold_tensor, warm_tensor, rtol=0, atol=0)

    # The first chunk of each sequence must see exactly the original state.
    for seq, chunk_offset in enumerate((0, 1)):
        expected = torch.zeros(heads, dim, dim, device="cuda")
        if initial is not None:
            expected = initial[int(indices[seq]) if indexed else seq].transpose(-1, -2)
        torch.testing.assert_close(cold[0][0, chunk_offset], expected.to(k.dtype), rtol=0, atol=0)

    if indexed:
        torch.testing.assert_close(cold_pool, warm_pool, rtol=0, atol=0)
        torch.testing.assert_close(cold[2], cold_pool[indices.long()], rtol=0, atol=0)
        torch.testing.assert_close(cold_pool[[0, 2, 4]], initial[[0, 2, 4]], rtol=0, atol=0)
        assert not torch.equal(cold_pool[indices.long()], initial[indices.long()])
    elif initial is not None:
        torch.testing.assert_close(cold_pool, initial, rtol=0, atol=0)
        torch.testing.assert_close(warm_pool, initial, rtol=0, atol=0)


@torch.inference_mode()
def test_chunk_gdn_autotune_memory_scales_with_active_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.modules.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )

    autotuner = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    monkeypatch.setattr(autotuner, "cache", {})
    monkeypatch.setattr(autotuner, "cache_results", False)

    def benchmark(fn: Callable[[], object], quantiles: list[float]) -> list[float]:
        # Exercise every candidate without including Triton's timing/flush
        # buffers in the state-backup memory measurement.
        fn()
        return [1.0] * len(quantiles)

    monkeypatch.setattr(autotuner, "do_bench", benchmark)
    state = torch.zeros(8192, 4, 64, 64, device="cuda", dtype=torch.float32)
    indices = torch.tensor([8190, 1], device="cuda", dtype=torch.int32)
    k = torch.full((2, 17, 2, 64), 0.01, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(2, 17, 4, 64, device="cuda", dtype=torch.bfloat16)
    u = torch.ones_like(w)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        initial_state=state,
        initial_state_indices=indices,
        inplace_indexed_state_update=True,
        output_final_state=True,
    )
    torch.cuda.synchronize()
    extra_peak = torch.cuda.max_memory_allocated() - before

    # A 512 MiB pool with two active slots must not require a second pool.
    assert extra_peak < 8 * 1024 * 1024, f"Autotuning allocated {extra_peak} extra bytes"
    assert torch.count_nonzero(h) == 0
    torch.testing.assert_close(v_new, u, rtol=0, atol=0)
    expected = torch.full_like(final, float(k[0, 0, 0, 0]) * 17)
    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[indices.long()], expected, rtol=0, atol=0)
    assert torch.count_nonzero(state[[0, 8191]]) == 0
