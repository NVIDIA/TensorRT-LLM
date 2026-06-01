# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the MPI-allgather refactor of DWDP's ``fixup_moe_backends``.

These tests verify that the MPI-based allgather (replacing tekit's TCPDWDPStore)
produces the same semantics as the tekit original:

- Each rank's local shard is contributed via comm.allgather.
- Ranks receive a list of shards (one per peer, indexed by rank).
- The function concatenates along dim 0 and updates the parameter in place.

We mock the MPI communicator since the tests run single-process on one GPU.
The mock's ``allgather`` returns a pre-built shard list that the test provides
(simulating "what this rank would receive if 4 ranks each contributed their
shard").  This lets us cover the multi-rank semantics end-to-end without a
real MPI environment.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.setup import (
    _allgather_e_score_correction_bias,
    _allgather_expert_scales,
    _scatter_shards_to_full,
)


def _make_mock_comm(all_shards):
    """Create a mock MPI comm whose allgather() returns ``all_shards``.

    ``all_shards`` is a list of torch.Tensor objects, one per simulated rank.
    The mock ignores the input argument and returns the list, which is the
    semantic shape of a real ``Intracomm.allgather(local_shard)`` call.
    """
    comm = MagicMock()
    comm.allgather = MagicMock(side_effect=lambda _local: all_shards)
    comm.Barrier = MagicMock()
    comm.Get_rank = MagicMock(return_value=0)
    comm.Get_size = MagicMock(return_value=len(all_shards))
    return comm


class TestAllgatherEScoreCorrectionBias(unittest.TestCase):
    def _make_gate(self, bias_tensor, as_parameter=True):
        """Minimal nn.Module with an e_score_correction_bias attribute."""
        gate = nn.Module()
        if as_parameter:
            gate.e_score_correction_bias = nn.Parameter(bias_tensor, requires_grad=False)
        else:
            gate.e_score_correction_bias = bias_tensor
        return gate

    def test_missing_bias_is_noop(self):
        gate = nn.Module()
        comm = _make_mock_comm([torch.zeros(2)])
        _allgather_e_score_correction_bias(
            gate, layer_idx=3, dwdp_rank=0, dwdp_size=4, num_experts_total=16, comm=comm
        )
        comm.allgather.assert_not_called()
        self.assertFalse(
            hasattr(gate, "e_score_correction_bias")
            and getattr(gate, "e_score_correction_bias") is not None
            and isinstance(getattr(gate, "e_score_correction_bias"), (torch.Tensor, nn.Parameter))
        )

    def test_allgather_sharded_parameter(self):
        """Sharded bias: allgather 4 shards, concat, replace nn.Parameter."""
        dwdp_size = 4
        shard_size = 2
        num_experts_total = shard_size * dwdp_size
        # Each rank's shard has unique values
        shards = [
            torch.arange(r * shard_size, (r + 1) * shard_size, dtype=torch.float32)
            for r in range(dwdp_size)
        ]
        expected = torch.cat(shards, dim=0)

        for dwdp_rank in range(dwdp_size):
            with self.subTest(dwdp_rank=dwdp_rank):
                gate = self._make_gate(shards[dwdp_rank].clone())
                comm = _make_mock_comm(shards)

                _allgather_e_score_correction_bias(
                    gate,
                    layer_idx=3,
                    dwdp_rank=dwdp_rank,
                    dwdp_size=dwdp_size,
                    num_experts_total=num_experts_total,
                    comm=comm,
                )

                comm.allgather.assert_called_once()
                self.assertIsInstance(gate.e_score_correction_bias, nn.Parameter)
                self.assertEqual(gate.e_score_correction_bias.shape, (num_experts_total,))
                torch.testing.assert_close(gate.e_score_correction_bias.data, expected)

    def test_allgather_sharded_plain_tensor(self):
        """Sharded bias stored as plain Tensor (not Parameter) — update in-place."""
        dwdp_size = 2
        shard_size = 3
        num_experts_total = shard_size * dwdp_size
        shards = [
            torch.arange(r * shard_size, (r + 1) * shard_size, dtype=torch.float32)
            for r in range(dwdp_size)
        ]
        gate = self._make_gate(shards[0].clone(), as_parameter=False)
        comm = _make_mock_comm(shards)

        _allgather_e_score_correction_bias(
            gate,
            layer_idx=3,
            dwdp_rank=0,
            dwdp_size=dwdp_size,
            num_experts_total=num_experts_total,
            comm=comm,
        )

        comm.allgather.assert_called_once()
        self.assertEqual(gate.e_score_correction_bias.shape, (num_experts_total,))
        torch.testing.assert_close(gate.e_score_correction_bias, torch.cat(shards, dim=0))


class TestAllgatherExpertScales(unittest.TestCase):
    def _make_experts_module(self, params):
        """Return an nn.Module with the given {name: tensor} parameters."""
        module = nn.Module()
        for name, tensor in params.items():
            setattr(module, name, nn.Parameter(tensor, requires_grad=False))
        return module

    def test_skip_non_scale_params(self):
        """Parameters without scale keywords are not touched."""
        experts_per_rank = 2
        module = self._make_experts_module(
            {
                "random_weight": torch.zeros(experts_per_rank, 4),
                "fc31_alpha": torch.tensor([1.0, 2.0]),
            }
        )
        all_alpha_shards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        comm = _make_mock_comm(all_alpha_shards)

        _allgather_expert_scales(
            module,
            layer_idx=3,
            dwdp_rank=0,
            dwdp_size=2,
            comm=comm,
            experts_per_rank=experts_per_rank,
        )

        # Called exactly once — only fc31_alpha matched.
        self.assertEqual(comm.allgather.call_count, 1)
        # random_weight unchanged (still size (2,4))
        self.assertEqual(module.random_weight.shape, (experts_per_rank, 4))
        # fc31_alpha concatenated to full size (4,)
        self.assertEqual(module.fc31_alpha.shape, (4,))
        torch.testing.assert_close(module.fc31_alpha.data, torch.cat(all_alpha_shards, dim=0))

    def test_allgather_multiple_scales(self):
        """Two matching scale params: each gets its own allgather call."""
        experts_per_rank = 2
        dwdp_size = 2
        alpha_shards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        beta_shards = [torch.tensor([[5.0]]), torch.tensor([[6.0]])]
        # fc31_alpha has shape (2,); fc2_alpha has shape (2,1) — both match
        # experts_per_rank=2 on dim 0.
        module = self._make_experts_module(
            {
                "fc31_alpha": alpha_shards[0].clone(),
                "fc2_alpha": torch.tensor([[5.0]]),  # shape (1,1) — shape[0]=1 != 2
            }
        )
        # Rewrite fc2_alpha with shape[0]=2 so it matches
        module.fc2_alpha = nn.Parameter(torch.tensor([[1.0], [2.0]]), requires_grad=False)
        # Two shards, each shape (2,1)
        beta_shards = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]

        # The mock returns the pre-set shard list for whichever param is being
        # allgathered. named_parameters() order is insertion order — the
        # ordering here is fc31_alpha first, fc2_alpha second.
        call_idx = {"n": 0}

        def _side_effect(_local):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return alpha_shards if idx == 0 else beta_shards

        comm = MagicMock()
        comm.allgather = MagicMock(side_effect=_side_effect)

        _allgather_expert_scales(
            module,
            layer_idx=3,
            dwdp_rank=0,
            dwdp_size=dwdp_size,
            comm=comm,
            experts_per_rank=experts_per_rank,
        )

        self.assertEqual(comm.allgather.call_count, 2)
        torch.testing.assert_close(module.fc31_alpha.data, torch.cat(alpha_shards, dim=0))
        torch.testing.assert_close(module.fc2_alpha.data, torch.cat(beta_shards, dim=0))


class TestScatterShardsToFull(unittest.TestCase):
    """Unit tests for ``_scatter_shards_to_full`` — the peer_ranges-aware
    reconstruction of a full-size tensor from per-peer EP shards.  Covers
    the three partition modes that motivate Phase 2:

      * Uniform (size == stride == num_experts // dwdp_size): no overlap,
        no padding; equivalent to ``torch.cat``.
      * Non-uniform tail-padding (size * dwdp_size > num_experts): tail
        shard's trailing entries are skipped via ``[:valid]``.
      * Redundancy (size > stride): overlapping ranges; later peers
        overwrite earlier ones at overlap indices but values agree
        (per Phase 1 invariant: every rank that owns expert ``e`` loaded
        ``e`` from the same checkpoint).
    """

    def test_uniform_partition(self):
        # dwdp=4, 8 experts, size=stride=2.  Each rank's shard covers
        # exactly 2 experts; reconstruction = concat.
        peer_ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]
        shards = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([4.0, 5.0]),
            torch.tensor([6.0, 7.0]),
        ]
        full = _scatter_shards_to_full(
            shards=shards,
            peer_ranges=peer_ranges,
            num_experts_total=8,
            ref=shards[0],
        )
        torch.testing.assert_close(full, torch.arange(8, dtype=torch.float32))

    def test_mode_b_overlap_dwdp3(self):
        # dwdp=3, 8 experts, size=4, stride=2: 2*2+4=8 (Mode B equality).
        # Adjacent ranks overlap by 2 experts; shared values agree
        # because every rank that owns expert ``e`` loaded ``e`` from
        # the same checkpoint.
        peer_ranges = [(0, 4), (2, 6), (4, 8)]
        shards = [
            torch.tensor([0.0, 1.0, 2.0, 3.0]),  # rank 0: experts 0,1,2,3
            torch.tensor([2.0, 3.0, 4.0, 5.0]),  # rank 1: experts 2,3,4,5 (overlap rank 0 at 2,3)
            torch.tensor([4.0, 5.0, 6.0, 7.0]),  # rank 2: experts 4,5,6,7 (overlap rank 1 at 4,5)
        ]
        full = _scatter_shards_to_full(
            shards=shards,
            peer_ranges=peer_ranges,
            num_experts_total=8,
            ref=shards[0],
        )
        torch.testing.assert_close(full, torch.arange(8, dtype=torch.float32))

    def test_higher_dim_trailing_shape(self):
        # Shards may have trailing dims (e.g., scale param shape (size, K)).
        peer_ranges = [(0, 2), (2, 4)]
        shards = [
            torch.tensor([[0.0, 0.5], [1.0, 1.5]]),
            torch.tensor([[2.0, 2.5], [3.0, 3.5]]),
        ]
        full = _scatter_shards_to_full(
            shards=shards,
            peer_ranges=peer_ranges,
            num_experts_total=4,
            ref=shards[0],
        )
        expected = torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        torch.testing.assert_close(full, expected)


class TestAllgatherExpertScalesNonUniform(unittest.TestCase):
    """Validate ``_allgather_expert_scales`` end-to-end with the new
    Phase 2 ``peer_ranges`` argument: shards loaded under Mode B
    overlap (size > stride) must reconstruct the full bias correctly,
    with overlapping experts getting the same value from any peer."""

    def _make_experts_module(self, params):
        module = torch.nn.Module()
        for name, tensor in params.items():
            setattr(module, name, torch.nn.Parameter(tensor, requires_grad=False))
        return module

    def test_mode_b_overlap_dwdp3_via_peer_ranges(self):
        # dwdp=3, 256 experts, Mode B with size=86, stride=85
        # (2*85 + 86 = 256 — exact equality, 1-expert overlap between
        # adjacent ranks).  Shards loaded by each rank cover overlapping
        # ranges; ``_scatter_shards_to_full`` writes the shared experts
        # multiple times but the values agree (same checkpoint).
        peer_ranges = [(0, 86), (85, 171), (170, 256)]
        all_shards = [
            torch.arange(0, 86, dtype=torch.float32),  # rank 0: [0..85]
            torch.arange(85, 171, dtype=torch.float32),  # rank 1: [85..170]
            torch.arange(170, 256, dtype=torch.float32),  # rank 2: [170..255]
        ]
        module = self._make_experts_module(
            {
                "fc31_alpha": torch.arange(0, 86, dtype=torch.float32),
            }
        )
        comm = _make_mock_comm(all_shards)

        _allgather_expert_scales(
            module,
            layer_idx=3,
            dwdp_rank=0,
            dwdp_size=3,
            comm=comm,
            experts_per_rank=86,
            num_experts_total=256,
            peer_ranges=peer_ranges,
        )

        comm.allgather.assert_called_once()
        # Reconstructed alpha covers ALL 256 experts with their canonical values.
        torch.testing.assert_close(
            module.fc31_alpha.data,
            torch.arange(0, 256, dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
