# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests and benchmarks for block table <-> ragged tensor conversions.

Tests two implementations of bidirectional conversion between:
- Block table format: [max_batch_size, max_blocks_per_seq] padded 2D tensor
- Ragged format: flat 1D cache_loc + cumulative offset vector cu_num_blocks

Implementations:
1. PyTorch reference (tensorized, loop-free)
2. Triton kernels (one program per sequence row)
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.utils.block_table_ragged  # noqa: F401

ops = torch.ops.auto_deploy

# ========================================================================================
# Test helpers
# ========================================================================================


def _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device="cuda"):
    """Create randomized test data for block table / ragged conversion tests.

    Returns:
        block_table: [max_batch_size, max_blocks_per_seq] with random page IDs.
        num_blocks: [max_batch_size] with random valid lengths in [1, max_blocks_per_seq].
    """
    num_blocks = torch.randint(
        1, max_blocks_per_seq + 1, (max_batch_size,), device=device, dtype=torch.int32
    )

    block_table = torch.randint(
        1, 10000, (max_batch_size, max_blocks_per_seq), device=device, dtype=torch.int32
    )

    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    block_table[:num_sequences][pad_mask] = 0

    return block_table, num_blocks


def _make_extra_idx(
    max_batch_size, num_sequences, max_val=10000, valid_fraction=0.5, device="cuda"
):
    """Create extra_idx tensor with a mix of valid values and -1 sentinels."""
    extra_idx = torch.full((max_batch_size,), -1, device=device, dtype=torch.int32)
    if num_sequences > 0:
        valid_mask = torch.rand(num_sequences, device=device) < valid_fraction
        valid_values = torch.randint(1, max_val, (num_sequences,), device=device, dtype=torch.int32)
        extra_idx[:num_sequences] = torch.where(
            valid_mask, valid_values, torch.tensor(-1, device=device, dtype=torch.int32)
        )
    return extra_idx


# ========================================================================================
# Correctness tests
# ========================================================================================


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 1), (4, 8), (32, 64), (128, 256), (512, 4096)],
)
def test_block_table_to_ragged(max_batch_size, max_blocks_per_seq):
    """Test block_table -> ragged: Triton matches PyTorch reference (exact int match)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)

    max_capacity = max_batch_size * max_blocks_per_seq

    # --- PyTorch reference ---
    cache_loc_ref = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_ref = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_ref = ops.block_table_to_ragged_torch(
        bt, num_blocks, cache_loc_ref, cu_ref, num_sequences
    )

    # --- Triton ---
    cache_loc_tri = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_tri = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_tri = ops.block_table_to_ragged_triton(
        bt, num_blocks, cache_loc_tri, cu_tri, num_sequences
    )

    # --- Verify ---
    assert total_ref == total_tri, f"Total mismatch: {total_ref} vs {total_tri}"
    torch.testing.assert_close(
        cu_tri[: num_sequences + 1], cu_ref[: num_sequences + 1], rtol=0, atol=0
    )
    torch.testing.assert_close(cache_loc_tri[:total_ref], cache_loc_ref[:total_ref], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 1), (4, 8), (32, 64), (128, 256)],
)
def test_ragged_to_block_table(max_batch_size, max_blocks_per_seq):
    """Test ragged -> block_table: Triton matches PyTorch reference (exact int match)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_src, num_blocks_src = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    max_capacity = max_batch_size * max_blocks_per_seq
    cache_loc = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_num_blocks = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(
        block_table_src, num_blocks_src, cache_loc, cu_num_blocks, num_sequences
    )

    # --- PyTorch reference (ragged -> block_table) ---
    bt_ref = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
    nb_ref = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    ops.ragged_to_block_table_torch(cache_loc, cu_num_blocks, bt_ref, nb_ref, num_sequences)

    # --- Triton (ragged -> block_table) ---
    bt_tri = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
    ops.ragged_to_block_table_triton(cache_loc, cu_num_blocks, bt_tri, num_sequences)

    # --- Verify ---
    torch.testing.assert_close(bt_tri[:num_sequences], bt_ref[:num_sequences], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 1), (4, 8), (32, 64), (128, 256)],
)
def test_ragged_to_block_table_strided(max_batch_size, max_blocks_per_seq):
    """Test ragged -> strided block_table: kernel writes correctly into non-contiguous slice."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_src, num_blocks_src = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    max_capacity = max_batch_size * max_blocks_per_seq
    cache_loc = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_num_blocks = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(
        block_table_src, num_blocks_src, cache_loc, cu_num_blocks, num_sequences
    )

    # Contiguous reference
    bt_ref = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
    ops.ragged_to_block_table_triton(cache_loc, cu_num_blocks, bt_ref, num_sequences)

    # Strided output: [B, 2, M] tensor, write into [:, 0, :] slice (stride [2*M, 1])
    sentinel = -1
    strided_buf = torch.full(
        (max_batch_size, 2, max_blocks_per_seq), sentinel, device=device, dtype=torch.int32
    )
    strided_slice = strided_buf[:, 0, :]
    assert strided_slice.stride() == (2 * max_blocks_per_seq, 1)
    ops.ragged_to_block_table_triton(cache_loc, cu_num_blocks, strided_slice, num_sequences)

    # Verify: strided slice matches contiguous reference
    torch.testing.assert_close(
        strided_buf[:num_sequences, 0, :], bt_ref[:num_sequences], rtol=0, atol=0
    )
    # Verify: interleaved rows ([:, 1, :]) are untouched
    torch.testing.assert_close(
        strided_buf[:, 1, :],
        torch.full_like(strided_buf[:, 1, :], sentinel),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 1), (4, 8), (32, 64), (128, 256)],
)
def test_roundtrip(max_batch_size, max_blocks_per_seq):
    """Test block_table -> ragged -> block_table roundtrip (exact recovery)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_orig, num_blocks_orig = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )

    max_capacity = max_batch_size * max_blocks_per_seq

    # Forward: block_table -> ragged (Triton)
    cache_loc = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_num_blocks = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_triton(
        block_table_orig, num_blocks_orig, cache_loc, cu_num_blocks, num_sequences
    )

    # Inverse: ragged -> block_table (Triton)
    block_table_recovered = torch.zeros(
        max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32
    )
    ops.ragged_to_block_table_triton(cache_loc, cu_num_blocks, block_table_recovered, num_sequences)

    # Verify exact recovery of the active region
    torch.testing.assert_close(
        block_table_recovered[:num_sequences],
        block_table_orig[:num_sequences],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq, num_sequences",
    [(8, 16, 3), (32, 64, 10), (128, 256, 1)],
)
def test_partial_batch(max_batch_size, max_blocks_per_seq, num_sequences):
    """Test with num_sequences < max_batch_size (partial batch)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)

    max_capacity = max_batch_size * max_blocks_per_seq

    # --- PyTorch reference ---
    cache_loc_ref = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_ref = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_ref = ops.block_table_to_ragged_torch(
        bt, num_blocks, cache_loc_ref, cu_ref, num_sequences
    )

    # --- Triton ---
    cache_loc_tri = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_tri = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_tri = ops.block_table_to_ragged_triton(
        bt, num_blocks, cache_loc_tri, cu_tri, num_sequences
    )

    assert total_ref == total_tri
    torch.testing.assert_close(cache_loc_tri[:total_ref], cache_loc_ref[:total_ref], rtol=0, atol=0)

    # Also roundtrip the partial batch
    bt_recovered = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
    ops.ragged_to_block_table_triton(cache_loc_tri, cu_tri, bt_recovered, num_sequences)
    torch.testing.assert_close(bt_recovered[:num_sequences], bt[:num_sequences], rtol=0, atol=0)


# ========================================================================================
# Correctness tests -- adjust block index (append / remove / no-op)
# ========================================================================================


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 4), (4, 8), (32, 64), (128, 256)],
)
def test_adjust_append_block_table(max_batch_size, max_blocks_per_seq):
    """Test append (delta=+1) to block_table: Triton matches PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, device=device)
    delta = (extra_idx[:num_sequences] >= 0).to(torch.int32)
    delta_full = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta_full[:num_sequences] = delta

    bt_ref = bt.clone()
    nb_ref = num_blocks.clone()
    ops.adjust_block_table_torch(bt_ref, nb_ref, extra_idx, delta_full, num_sequences)

    bt_tri = bt.clone()
    nb_tri = num_blocks.clone()
    ops.adjust_block_table_triton(bt_tri, nb_tri, extra_idx, delta_full, num_sequences)

    torch.testing.assert_close(bt_tri[:num_sequences], bt_ref[:num_sequences], rtol=0, atol=0)
    torch.testing.assert_close(nb_tri[:num_sequences], nb_ref[:num_sequences], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(1, 4), (4, 8), (32, 64), (128, 256)],
)
def test_adjust_append_ragged(max_batch_size, max_blocks_per_seq):
    """Test append (delta=+1) to ragged: Triton matches PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_src, num_blocks_src = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    buffer_capacity = max_batch_size * max_blocks_per_seq + num_sequences

    cache_loc_base = torch.zeros(buffer_capacity, device=device, dtype=torch.int32)
    cu_base = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(
        block_table_src, num_blocks_src, cache_loc_base, cu_base, num_sequences
    )

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, device=device)
    delta = (extra_idx[:num_sequences] >= 0).to(torch.int32)
    delta_full = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta_full[:num_sequences] = delta

    cl_ref = cache_loc_base.clone()
    cu_ref = cu_base.clone()
    total_ref = ops.adjust_ragged_torch(cl_ref, cu_ref, extra_idx, delta_full, num_sequences)

    cl_tri = cache_loc_base.clone()
    cu_tri = cu_base.clone()
    total_tri = ops.adjust_ragged_triton(cl_tri, cu_tri, extra_idx, delta_full, num_sequences)

    assert total_ref == total_tri, f"Total mismatch: {total_ref} vs {total_tri}"
    torch.testing.assert_close(
        cu_tri[: num_sequences + 1], cu_ref[: num_sequences + 1], rtol=0, atol=0
    )
    torch.testing.assert_close(cl_tri[:total_ref], cl_ref[:total_ref], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64)],
)
def test_adjust_all_zero_delta(max_batch_size, max_blocks_per_seq):
    """Test adjust with all delta=0 (no-op)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    extra_idx = torch.full((max_batch_size,), -1, device=device, dtype=torch.int32)
    delta = torch.zeros(max_batch_size, device=device, dtype=torch.int32)

    bt_before = bt.clone()
    nb_before = num_blocks.clone()
    ops.adjust_block_table_triton(bt, num_blocks, extra_idx, delta, num_sequences)
    torch.testing.assert_close(bt[:num_sequences], bt_before[:num_sequences], rtol=0, atol=0)
    torch.testing.assert_close(
        num_blocks[:num_sequences], nb_before[:num_sequences], rtol=0, atol=0
    )

    max_capacity = max_batch_size * max_blocks_per_seq
    cache_loc = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt_before, nb_before, cache_loc, cu, num_sequences)
    cl_before = cache_loc.clone()
    cu_before = cu.clone()
    ops.adjust_ragged_triton(cache_loc, cu, extra_idx, delta, num_sequences)
    old_total = int(cu_before[num_sequences].item())
    torch.testing.assert_close(cache_loc[:old_total], cl_before[:old_total], rtol=0, atol=0)
    torch.testing.assert_close(
        cu[: num_sequences + 1], cu_before[: num_sequences + 1], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64)],
)
def test_adjust_all_append(max_batch_size, max_blocks_per_seq):
    """Test adjust with all delta=+1 (every sequence gets one append)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = torch.ones(max_batch_size, device=device, dtype=torch.int32)

    bt_ref, nb_ref = bt.clone(), num_blocks.clone()
    bt_tri, nb_tri = bt.clone(), num_blocks.clone()
    ops.adjust_block_table_torch(bt_ref, nb_ref, extra_idx, delta, num_sequences)
    ops.adjust_block_table_triton(bt_tri, nb_tri, extra_idx, delta, num_sequences)
    torch.testing.assert_close(bt_tri[:num_sequences], bt_ref[:num_sequences], rtol=0, atol=0)
    torch.testing.assert_close(nb_tri[:num_sequences], nb_ref[:num_sequences], rtol=0, atol=0)

    max_capacity = max_batch_size * max_blocks_per_seq + num_sequences
    cache_loc_base = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_base = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt, num_blocks, cache_loc_base, cu_base, num_sequences)

    cl_ref, cu_ref = cache_loc_base.clone(), cu_base.clone()
    cl_tri, cu_tri = cache_loc_base.clone(), cu_base.clone()
    total_ref = ops.adjust_ragged_torch(cl_ref, cu_ref, extra_idx, delta, num_sequences)
    total_tri = ops.adjust_ragged_triton(cl_tri, cu_tri, extra_idx, delta, num_sequences)
    assert total_ref == total_tri
    torch.testing.assert_close(cl_tri[:total_ref], cl_ref[:total_ref], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_append_roundtrip(max_batch_size, max_blocks_per_seq):
    """Test: adjust block_table then to ragged == to ragged then adjust (append only)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, device=device)
    delta = (extra_idx[:num_sequences] >= 0).to(torch.int32)
    delta_full = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta_full[:num_sequences] = delta
    max_capacity = max_batch_size * max_blocks_per_seq + num_sequences

    bt_a = bt.clone()
    nb_a = num_blocks.clone()
    ops.adjust_block_table_torch(bt_a, nb_a, extra_idx, delta_full, num_sequences)
    cl_a = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_a = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_a = ops.block_table_to_ragged_torch(bt_a, nb_a, cl_a, cu_a, num_sequences)

    cl_b = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_b = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt, num_blocks, cl_b, cu_b, num_sequences)
    total_b = ops.adjust_ragged_torch(cl_b, cu_b, extra_idx, delta_full, num_sequences)

    assert total_a == total_b, f"Total mismatch: {total_a} vs {total_b}"
    torch.testing.assert_close(cl_a[:total_a], cl_b[:total_b], rtol=0, atol=0)
    torch.testing.assert_close(cu_a[: num_sequences + 1], cu_b[: num_sequences + 1], rtol=0, atol=0)


# ========================================================================================
# Correctness tests -- adjust with removal (delta=-1) and mixed delta
# ========================================================================================


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_remove_block_table(max_batch_size, max_blocks_per_seq):
    """Test removal (delta=-1) on block_table: Triton matches PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(min=2)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta = torch.full((max_batch_size,), -1, device=device, dtype=torch.int32)

    bt_ref = bt.clone()
    nb_ref = num_blocks.clone()
    ops.adjust_block_table_torch(bt_ref, nb_ref, extra_idx, delta, num_sequences)

    bt_tri = bt.clone()
    nb_tri = num_blocks.clone()
    ops.adjust_block_table_triton(bt_tri, nb_tri, extra_idx, delta, num_sequences)

    torch.testing.assert_close(nb_tri[:num_sequences], nb_ref[:num_sequences], rtol=0, atol=0)
    assert (nb_ref[:num_sequences] == num_blocks[:num_sequences] - 1).all()


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_remove_ragged(max_batch_size, max_blocks_per_seq):
    """Test removal (delta=-1) on ragged: Triton matches PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(min=2)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    max_capacity = max_batch_size * max_blocks_per_seq
    cache_loc_base = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_base = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt, num_blocks, cache_loc_base, cu_base, num_sequences)

    extra_idx = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta = torch.full((max_batch_size,), -1, device=device, dtype=torch.int32)

    cl_ref = cache_loc_base.clone()
    cu_ref = cu_base.clone()
    total_ref = ops.adjust_ragged_torch(cl_ref, cu_ref, extra_idx, delta, num_sequences)

    cl_tri = cache_loc_base.clone()
    cu_tri = cu_base.clone()
    total_tri = ops.adjust_ragged_triton(cl_tri, cu_tri, extra_idx, delta, num_sequences)

    assert total_ref == total_tri, f"Total mismatch: {total_ref} vs {total_tri}"
    torch.testing.assert_close(
        cu_tri[: num_sequences + 1], cu_ref[: num_sequences + 1], rtol=0, atol=0
    )
    torch.testing.assert_close(cl_tri[:total_ref], cl_ref[:total_ref], rtol=0, atol=0)


def _make_mixed_delta(num_sequences, max_batch_size, device="cuda"):
    """Create a delta tensor with a mix of -1, 0, +1 values."""
    choices = torch.tensor([-1, 0, 1], device=device, dtype=torch.int32)
    indices = torch.randint(0, 3, (num_sequences,), device=device)
    delta = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    delta[:num_sequences] = choices[indices]
    return delta


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_mixed_delta_block_table(max_batch_size, max_blocks_per_seq):
    """Test mixed delta (-1, 0, +1) on block_table: Triton matches PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(min=2, max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = _make_mixed_delta(num_sequences, max_batch_size, device=device)

    bt_ref = bt.clone()
    nb_ref = num_blocks.clone()
    ops.adjust_block_table_torch(bt_ref, nb_ref, extra_idx, delta, num_sequences)

    bt_tri = bt.clone()
    nb_tri = num_blocks.clone()
    ops.adjust_block_table_triton(bt_tri, nb_tri, extra_idx, delta, num_sequences)

    torch.testing.assert_close(bt_tri[:num_sequences], bt_ref[:num_sequences], rtol=0, atol=0)
    torch.testing.assert_close(nb_tri[:num_sequences], nb_ref[:num_sequences], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_mixed_delta_ragged(max_batch_size, max_blocks_per_seq):
    """Test mixed delta (-1, 0, +1) on ragged: Triton matches PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(min=2, max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    max_capacity = max_batch_size * max_blocks_per_seq + num_sequences
    cache_loc_base = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_base = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt, num_blocks, cache_loc_base, cu_base, num_sequences)

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = _make_mixed_delta(num_sequences, max_batch_size, device=device)

    cl_ref = cache_loc_base.clone()
    cu_ref = cu_base.clone()
    total_ref = ops.adjust_ragged_torch(cl_ref, cu_ref, extra_idx, delta, num_sequences)

    cl_tri = cache_loc_base.clone()
    cu_tri = cu_base.clone()
    total_tri = ops.adjust_ragged_triton(cl_tri, cu_tri, extra_idx, delta, num_sequences)

    assert total_ref == total_tri, f"Total mismatch: {total_ref} vs {total_tri}"
    torch.testing.assert_close(
        cu_tri[: num_sequences + 1], cu_ref[: num_sequences + 1], rtol=0, atol=0
    )
    torch.testing.assert_close(cl_tri[:total_ref], cl_ref[:total_ref], rtol=0, atol=0)


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(4, 8), (32, 64), (128, 256)],
)
def test_adjust_mixed_roundtrip(max_batch_size, max_blocks_per_seq):
    """Test: adjust block_table then to ragged == to ragged then adjust (mixed delta)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    num_blocks[:num_sequences] = num_blocks[:num_sequences].clamp(min=2, max=max_blocks_per_seq - 1)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks[:num_sequences].unsqueeze(1)
    bt[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = _make_mixed_delta(num_sequences, max_batch_size, device=device)
    max_capacity = max_batch_size * max_blocks_per_seq + num_sequences

    bt_a = bt.clone()
    nb_a = num_blocks.clone()
    ops.adjust_block_table_torch(bt_a, nb_a, extra_idx, delta, num_sequences)
    cl_a = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_a = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    total_a = ops.block_table_to_ragged_torch(bt_a, nb_a, cl_a, cu_a, num_sequences)

    cl_b = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_b = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(bt, num_blocks, cl_b, cu_b, num_sequences)
    total_b = ops.adjust_ragged_torch(cl_b, cu_b, extra_idx, delta, num_sequences)

    assert total_a == total_b, f"Total mismatch: {total_a} vs {total_b}"
    torch.testing.assert_close(cl_a[:total_a], cl_b[:total_b], rtol=0, atol=0)
    torch.testing.assert_close(cu_a[: num_sequences + 1], cu_b[: num_sequences + 1], rtol=0, atol=0)


# ========================================================================================
# Performance benchmarks
# ========================================================================================


def _benchmark_fn(fn, num_warmup=10, num_runs=100):
    """Benchmark a callable using torch.cuda.Event timing.

    Returns:
        Average latency in microseconds.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(num_runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    avg_us = sum(latencies_ms) / len(latencies_ms) * 1000.0
    return avg_us


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(64, 128), (256, 256), (512, 512), (1024, 4096)],
)
def test_benchmark_block_table_to_ragged(max_batch_size, max_blocks_per_seq):
    """Benchmark block_table -> ragged: Triton vs PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    bt, num_blocks = _make_test_data(max_batch_size, max_blocks_per_seq, num_sequences, device)
    max_capacity = max_batch_size * max_blocks_per_seq

    def run_torch():
        cl = torch.zeros(max_capacity, device=device, dtype=torch.int32)
        cu = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
        ops.block_table_to_ragged_torch(bt, num_blocks, cl, cu, num_sequences)

    def run_triton():
        cl = torch.zeros(max_capacity, device=device, dtype=torch.int32)
        cu = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
        ops.block_table_to_ragged_triton(bt, num_blocks, cl, cu, num_sequences)

    torch_us = _benchmark_fn(run_torch)
    triton_us = _benchmark_fn(run_triton)

    speedup = torch_us / triton_us if triton_us > 0 else float("inf")
    print(
        f"\n  block_table -> ragged  [{max_batch_size}x{max_blocks_per_seq}]"
        f"\n    PyTorch: {torch_us:8.1f} us"
        f"\n    Triton:  {triton_us:8.1f} us"
        f"\n    Speedup: {speedup:8.2f}x"
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(64, 128), (256, 256), (512, 512)],
)
def test_benchmark_ragged_to_block_table(max_batch_size, max_blocks_per_seq):
    """Benchmark ragged -> block_table: Triton vs PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_src, num_blocks_src = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    max_capacity = max_batch_size * max_blocks_per_seq
    cache_loc = torch.zeros(max_capacity, device=device, dtype=torch.int32)
    cu_num_blocks = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(
        block_table_src, num_blocks_src, cache_loc, cu_num_blocks, num_sequences
    )

    def run_torch():
        bt = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
        nb = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        ops.ragged_to_block_table_torch(cache_loc, cu_num_blocks, bt, nb, num_sequences)

    def run_triton():
        bt = torch.zeros(max_batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
        ops.ragged_to_block_table_triton(cache_loc, cu_num_blocks, bt, num_sequences)

    torch_us = _benchmark_fn(run_torch)
    triton_us = _benchmark_fn(run_triton)

    speedup = torch_us / triton_us if triton_us > 0 else float("inf")
    print(
        f"\n  ragged -> block_table  [{max_batch_size}x{max_blocks_per_seq}]"
        f"\n    PyTorch: {torch_us:8.1f} us"
        f"\n    Triton:  {triton_us:8.1f} us"
        f"\n    Speedup: {speedup:8.2f}x"
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(64, 128), (256, 256), (512, 512)],
)
def test_benchmark_adjust_block_table(max_batch_size, max_blocks_per_seq):
    """Benchmark adjust block_table: Triton vs PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_base, num_blocks_base = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    num_blocks_base[:num_sequences] = num_blocks_base[:num_sequences].clamp(
        min=2, max=max_blocks_per_seq - 1
    )
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks_base[:num_sequences].unsqueeze(1)
    block_table_base[:num_sequences][pad_mask] = 0

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = _make_mixed_delta(num_sequences, max_batch_size, device=device)

    def run_torch():
        bt = block_table_base.clone()
        nb = num_blocks_base.clone()
        ops.adjust_block_table_torch(bt, nb, extra_idx, delta, num_sequences)

    def run_triton():
        bt = block_table_base.clone()
        nb = num_blocks_base.clone()
        ops.adjust_block_table_triton(bt, nb, extra_idx, delta, num_sequences)

    torch_us = _benchmark_fn(run_torch)
    triton_us = _benchmark_fn(run_triton)

    speedup = torch_us / triton_us if triton_us > 0 else float("inf")
    print(
        f"\n  adjust block_table  [{max_batch_size}x{max_blocks_per_seq}]"
        f"\n    PyTorch: {torch_us:8.1f} us"
        f"\n    Triton:  {triton_us:8.1f} us"
        f"\n    Speedup: {speedup:8.2f}x"
    )


@pytest.mark.parametrize(
    "max_batch_size, max_blocks_per_seq",
    [(64, 128), (256, 256), (512, 512)],
)
def test_benchmark_adjust_ragged(max_batch_size, max_blocks_per_seq):
    """Benchmark adjust ragged: Triton vs PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    num_sequences = max_batch_size

    block_table_src, num_blocks_src = _make_test_data(
        max_batch_size, max_blocks_per_seq, num_sequences, device
    )
    num_blocks_src[:num_sequences] = num_blocks_src[:num_sequences].clamp(min=2)
    col = torch.arange(max_blocks_per_seq, device=device)
    pad_mask = col.unsqueeze(0) >= num_blocks_src[:num_sequences].unsqueeze(1)
    block_table_src[:num_sequences][pad_mask] = 0

    buffer_capacity = max_batch_size * max_blocks_per_seq + num_sequences
    cache_loc_base = torch.zeros(buffer_capacity, device=device, dtype=torch.int32)
    cu_base = torch.zeros(max_batch_size + 1, device=device, dtype=torch.int32)
    ops.block_table_to_ragged_torch(
        block_table_src, num_blocks_src, cache_loc_base, cu_base, num_sequences
    )

    extra_idx = _make_extra_idx(max_batch_size, num_sequences, valid_fraction=1.0, device=device)
    delta = _make_mixed_delta(num_sequences, max_batch_size, device=device)

    def run_torch():
        cl = cache_loc_base.clone()
        cu = cu_base.clone()
        ops.adjust_ragged_torch(cl, cu, extra_idx, delta, num_sequences)

    def run_triton():
        cl = cache_loc_base.clone()
        cu = cu_base.clone()
        ops.adjust_ragged_triton(cl, cu, extra_idx, delta, num_sequences)

    torch_us = _benchmark_fn(run_torch)
    triton_us = _benchmark_fn(run_triton)

    speedup = torch_us / triton_us if triton_us > 0 else float("inf")
    print(
        f"\n  adjust ragged  [{max_batch_size}x{max_blocks_per_seq}]"
        f"\n    PyTorch: {torch_us:8.1f} us"
        f"\n    Triton:  {triton_us:8.1f} us"
        f"\n    Speedup: {speedup:8.2f}x"
    )
