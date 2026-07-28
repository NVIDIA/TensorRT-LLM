# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused KDA decode conv-window step.

``kda_conv_state_decode_step`` replaces four ATen passes over the layer's
short-convolution pool (indexed gather, strided repack, concat, indexed
scatter) with one indexed pass that stages the history columns and rolls the
pool in place. In place is the risk: the pass writes window columns it is
concurrently reading, so these tests check bit-exact agreement with the ATen
sequence both for a single step and across a long run of steps, where a
one-column drift would accumulate instead of cancelling.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_conv_state import kda_conv_state_decode_step

SECTIONS = 3  # q | k | v


def _skip_without_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("fused KDA conv-state step requires a CUDA device")


def _reference_step(
    conv_pool: torch.Tensor, slot_indices: torch.Tensor, x_new: torch.Tensor, staging: torch.Tensor
) -> None:
    """The ATen sequence the fused pass replaces."""
    batch, dim = staging.shape[1], staging.shape[2]
    width = conv_pool.shape[-1]
    cs = conv_pool.index_select(0, slot_indices)
    staging.copy_(cs.view(batch, SECTIONS, dim, width)[:, :, :, 1:].permute(1, 0, 2, 3))
    conv_pool.index_copy_(0, slot_indices, torch.cat([cs[:, :, 1:], x_new.unsqueeze(-1)], dim=-1))


def _make_case(dim: int, width: int, slots: int, batch: int, seed: int, strided_input: bool):
    torch.manual_seed(seed)
    device = "cuda"
    pool = torch.randn(slots, SECTIONS * dim, width, dtype=torch.bfloat16, device=device)
    slot_indices = torch.randperm(slots, device=device)[:batch].long()
    if strided_input:
        # The runtime feeds a column slice of the fused in-projection output,
        # so the input is row-strided rather than contiguous.
        wide = torch.randn(batch, (SECTIONS + 1) * dim, dtype=torch.bfloat16, device=device)
        x_new = wide[:, : SECTIONS * dim]
    else:
        x_new = torch.randn(batch, SECTIONS * dim, dtype=torch.bfloat16, device=device)
    staging = torch.empty(SECTIONS, batch, dim, width - 1, dtype=torch.bfloat16, device=device)
    return pool, slot_indices, x_new, staging


@pytest.mark.parametrize(
    "dim,width,slots,batch,strided_input",
    [
        (12288, 4, 32, 16, True),
        (12288, 4, 32, 4, True),
        (768, 4, 8, 5, False),
        (640, 3, 4, 4, False),
    ],
)
def test_single_step_matches_aten(dim, width, slots, batch, strided_input):
    """One fused pass reproduces the four-pass ATen sequence exactly."""
    _skip_without_gpu()
    pool, slot_indices, x_new, staging = _make_case(
        dim, width, slots, batch, seed=0xC0FFEE, strided_input=strided_input
    )
    ref_pool = pool.clone()
    ref_staging = torch.empty_like(staging)
    _reference_step(ref_pool, slot_indices, x_new, ref_staging)

    kda_conv_state_decode_step(pool, slot_indices, x_new, staging)

    torch.testing.assert_close(staging, ref_staging, rtol=0, atol=0)
    torch.testing.assert_close(pool, ref_pool, rtol=0, atol=0)


def test_state_evolution_over_many_steps():
    """The in-place roll must not drift over a long decode run.

    Each step re-draws the admitted slot set, so a slot both leaves and
    re-enters the batch while its window keeps advancing — the pattern a
    column-aliasing bug shows up in only after several steps.
    """
    _skip_without_gpu()
    dim, width, slots, batch = 1024, 4, 12, 6
    pool, _, _, staging = _make_case(dim, width, slots, batch, seed=7, strided_input=False)
    ref_pool = pool.clone()
    ref_staging = torch.empty_like(staging)

    generator = torch.Generator(device="cuda").manual_seed(11)
    for step in range(64):
        slot_indices = torch.randperm(slots, device="cuda")[:batch].long()
        x_new = torch.randn(
            batch, SECTIONS * dim, dtype=torch.bfloat16, device="cuda", generator=generator
        )

        _reference_step(ref_pool, slot_indices, x_new, ref_staging)
        kda_conv_state_decode_step(pool, slot_indices, x_new, staging)

        torch.testing.assert_close(
            staging, ref_staging, rtol=0, atol=0, msg=f"staging diverged at step {step}"
        )
        torch.testing.assert_close(
            pool, ref_pool, rtol=0, atol=0, msg=f"pool diverged at step {step}"
        )


def test_untouched_slots_are_preserved():
    """Slots outside the admitted batch keep their windows byte for byte."""
    _skip_without_gpu()
    dim, width, slots, batch = 1024, 4, 16, 5
    pool, slot_indices, x_new, staging = _make_case(
        dim, width, slots, batch, seed=3, strided_input=False
    )
    before = pool.clone()
    kda_conv_state_decode_step(pool, slot_indices, x_new, staging)

    admitted = torch.zeros(slots, dtype=torch.bool, device=pool.device)
    admitted[slot_indices] = True
    torch.testing.assert_close(pool[~admitted], before[~admitted], rtol=0, atol=0)


def test_cuda_graph_capture_replay():
    """The step is capture-safe: no allocation, no host sync in the launch."""
    _skip_without_gpu()
    dim, width, slots, batch = 1024, 4, 8, 4
    pool, slot_indices, x_new, staging = _make_case(
        dim, width, slots, batch, seed=5, strided_input=False
    )
    # Warm up outside capture so Triton's JIT runs before the graph is taped.
    kda_conv_state_decode_step(pool, slot_indices, x_new, staging)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            kda_conv_state_decode_step(pool, slot_indices, x_new, staging)
    torch.cuda.current_stream().wait_stream(stream)

    ref_pool = pool.clone()
    ref_staging = torch.empty_like(staging)
    _reference_step(ref_pool, slot_indices, x_new, ref_staging)

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(staging, ref_staging, rtol=0, atol=0)
    torch.testing.assert_close(pool, ref_pool, rtol=0, atol=0)
