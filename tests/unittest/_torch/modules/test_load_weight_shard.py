# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for presharded-aware `load_weight_shard`.

Covers the GMS RO failure mode: a module whose weights were zero-copy
materialized (already this rank's slice) must not be re-sharded on later
`load_weights` / `ModelLoader.reload` invocations.
"""

import torch

from tensorrt_llm._torch.modules.linear import (
    TensorParallelMode,
    load_weight_shard,
)


class _FakeModule:
    """Minimal stand-in for a TP-aware module without the full Linear import cost."""

    def __init__(self, presharded: bool = False):
        self._weights_presharded = presharded


def _make_weight(shape):
    return torch.arange(int(torch.tensor(shape).prod()),
                        dtype=torch.float32).reshape(shape)


class TestPreshardedGate:

    def test_presharded_false_slices(self):
        """Default path slices full checkpoint weight by rank."""
        weight = _make_weight((8, 4))
        shard = load_weight_shard(
            weight,
            tensor_parallel_size=2,
            tensor_parallel_rank=1,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        assert shard.shape == (4, 4)
        torch.testing.assert_close(shard, weight[4:, :])

    def test_presharded_true_passes_through(self):
        """Presharded module skips re-sharding regardless of tp args."""
        weight = _make_weight((4, 4))
        module = _FakeModule(presharded=True)
        shard = load_weight_shard(
            weight,
            tensor_parallel_size=2,
            tensor_parallel_rank=1,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            module=module,
        )
        assert shard.shape == weight.shape
        torch.testing.assert_close(shard, weight)

    def test_module_without_flag_slices(self):
        """`_weights_presharded` defaulting to False is respected via getattr."""

        class _BareModule:
            pass

        weight = _make_weight((8, 4))
        module = _BareModule()
        shard = load_weight_shard(
            weight,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            module=module,
        )
        assert shard.shape == (4, 4)
        torch.testing.assert_close(shard, weight[:4, :])

    def test_presharded_true_row_mode(self):
        weight = _make_weight((4, 4))
        module = _FakeModule(presharded=True)
        shard = load_weight_shard(
            weight,
            tensor_parallel_size=4,
            tensor_parallel_rank=2,
            tensor_parallel_mode=TensorParallelMode.ROW,
            module=module,
        )
        torch.testing.assert_close(shard, weight)

    def test_module_none_preserves_legacy_behavior(self):
        """Callers that do not pass module=... continue to re-shard."""
        weight = _make_weight((8, 4))
        shard = load_weight_shard(
            weight,
            tensor_parallel_size=4,
            tensor_parallel_rank=0,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        assert shard.shape == (2, 4)
        torch.testing.assert_close(shard, weight[:2, :])
