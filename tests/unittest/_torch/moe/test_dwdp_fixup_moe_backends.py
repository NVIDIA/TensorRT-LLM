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
"""Unit tests for the small-parameter exchange of DWDP's setup and ``fixup_moe_backends``.

``_exchange_expert_scalars`` gathers every rank's small per-expert parameters and gate bias in one
``comm.allgather`` and puts them together along the expert partition, and checks the per-layer
ones; ``fixup_moe_backends`` installs them.  A mock comm returns every simulated rank's report.
"""

import unittest
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.setup import (
    _collect_expert_scalars,
    _exchange_expert_scalars,
    _scatter_shards_to_full,
    fixup_moe_backends,
)
from tensorrt_llm._torch.modules.dwdp.specs import ExpertPartition
from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4CuteDslFusedMoEMethod

pytestmark = pytest.mark.cpu_only

LAYER = 3


class _Experts(nn.Module):
    """An MoE backend: an expert weight plus the given small parameters."""

    def __init__(self, num_experts: int, rows: int, params: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w3_w1_weight = nn.Parameter(torch.zeros(rows, 4), requires_grad=False)
        for name, tensor in params.items():
            setattr(self, name, nn.Parameter(tensor, requires_grad=False))


def _make_model(
    experts: nn.Module, bias: torch.Tensor | None = None, bias_as_parameter: bool = True
) -> nn.Module:
    """A model whose layer ``LAYER`` is an MoE layer; its gate holds ``bias`` when given one."""
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module() for _ in range(LAYER + 1)])
    mlp = model.layers[LAYER].mlp = nn.Module()
    mlp.experts, mlp.gate = experts, nn.Module()
    if bias is not None and bias_as_parameter:
        bias = nn.Parameter(bias, requires_grad=False)
    if bias is not None:
        mlp.gate.e_score_correction_bias = bias
    return model


def _make_rank_models(
    partition: ExpertPartition,
    params_per_rank: list[dict[str, torch.Tensor]],
    biases: list[torch.Tensor | None] | None = None,
    bias_as_parameter: bool = True,
) -> list[nn.Module]:
    """One model per DWDP rank, holding that rank's shards."""
    models = []
    for rank in range(partition.dwdp_size):
        experts = _Experts(partition.num_experts, partition.count_of(rank), params_per_rank[rank])
        bias = None if biases is None else biases[rank]
        models.append(_make_model(experts, bias, bias_as_parameter))
    return models


def _exchange_and_fixup(
    models: list[nn.Module], rank: int, partition: ExpertPartition
) -> tuple[MagicMock, nn.Module]:
    """Run the exchange and the fixup on ``rank``; return its comm and model.

    The comm's allgather returns every rank's report, this rank's as passed in.
    """
    reports = [(None, _collect_expert_scalars(m, [LAYER], partition.num_experts)) for m in models]
    comm = MagicMock()
    comm.allgather.side_effect = lambda local: [
        local if r == rank else report for r, report in enumerate(reports)
    ]
    model = models[rank]
    scalars = _exchange_expert_scalars(comm, model, [LAYER], partition)
    fixup_moe_backends(model, [LAYER], partition.num_experts, scalars)
    return comm, model


class TestAllgatherEScoreCorrectionBias(unittest.TestCase):
    def test_missing_bias_is_noop(self) -> None:
        partition = ExpertPartition.from_counts([2, 2, 2, 2])
        params = [{"fc31_alpha": torch.ones(2)} for _ in range(4)]
        models = _make_rank_models(partition, params)

        comm, model = _exchange_and_fixup(models, 0, partition)

        # One exchange for every layer, also without a bias.
        comm.allgather.assert_called_once()
        self.assertFalse(hasattr(model.layers[LAYER].mlp.gate, "e_score_correction_bias"))

    def test_allgather_sharded_parameter(self) -> None:
        """Sharded bias: allgather 4 shards, concat, replace nn.Parameter."""
        dwdp_size = 4
        shard_size = 2
        num_experts_total = shard_size * dwdp_size
        partition = ExpertPartition.strided(
            num_experts=num_experts_total, dwdp_size=dwdp_size, size=shard_size, stride=shard_size
        )
        # Each rank's shard has unique values
        shards = [
            torch.arange(r * shard_size, (r + 1) * shard_size, dtype=torch.float32)
            for r in range(dwdp_size)
        ]
        expected = torch.cat(shards, dim=0)
        params = [{"fc31_alpha": torch.ones(shard_size)} for _ in range(dwdp_size)]

        for dwdp_rank in range(dwdp_size):
            with self.subTest(dwdp_rank=dwdp_rank):
                models = _make_rank_models(partition, params, [s.clone() for s in shards])

                comm, model = _exchange_and_fixup(models, dwdp_rank, partition)

                comm.allgather.assert_called_once()
                gate = model.layers[LAYER].mlp.gate
                self.assertIsInstance(gate.e_score_correction_bias, nn.Parameter)
                self.assertEqual(gate.e_score_correction_bias.shape, (num_experts_total,))
                torch.testing.assert_close(gate.e_score_correction_bias.data, expected)

    def test_allgather_sharded_plain_tensor(self) -> None:
        """Sharded bias stored as plain Tensor (not Parameter) — update in-place."""
        dwdp_size = 2
        shard_size = 3
        num_experts_total = shard_size * dwdp_size
        partition = ExpertPartition.strided(
            num_experts=num_experts_total, dwdp_size=dwdp_size, size=shard_size, stride=shard_size
        )
        shards = [
            torch.arange(r * shard_size, (r + 1) * shard_size, dtype=torch.float32)
            for r in range(dwdp_size)
        ]
        params = [{"fc31_alpha": torch.ones(shard_size)} for _ in range(dwdp_size)]
        models = _make_rank_models(
            partition, params, [s.clone() for s in shards], bias_as_parameter=False
        )

        comm, model = _exchange_and_fixup(models, 0, partition)

        comm.allgather.assert_called_once()
        bias = model.layers[LAYER].mlp.gate.e_score_correction_bias
        self.assertNotIsInstance(bias, nn.Parameter)
        self.assertEqual(bias.shape, (num_experts_total,))
        torch.testing.assert_close(bias, torch.cat(shards, dim=0))


class TestAllgatherExpertScales(unittest.TestCase):
    def test_skip_non_scale_params(self) -> None:
        """Parameters without scale keywords are not touched."""
        experts_per_rank = 2
        partition = ExpertPartition.from_counts([experts_per_rank, experts_per_rank])
        all_alpha_shards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        params = [
            {"random_weight": torch.zeros(experts_per_rank, 4), "fc31_alpha": shard.clone()}
            for shard in all_alpha_shards
        ]
        models = _make_rank_models(partition, params)

        comm, model = _exchange_and_fixup(models, 0, partition)

        # Called exactly once — only fc31_alpha matched.
        comm.allgather.assert_called_once()
        _status, reported = comm.allgather.call_args.args[0]
        self.assertEqual(sorted(reported[LAYER][0]), ["fc31_alpha"])
        experts = model.layers[LAYER].mlp.experts
        # random_weight unchanged (still size (2,4))
        self.assertEqual(experts.random_weight.shape, (experts_per_rank, 4))
        # fc31_alpha concatenated to full size (4,)
        self.assertEqual(experts.fc31_alpha.shape, (4,))
        torch.testing.assert_close(experts.fc31_alpha.data, torch.cat(all_alpha_shards, dim=0))

    def test_allgather_multiple_scales(self) -> None:
        """Two matching scale params, one with a trailing dim: both put together."""
        partition = ExpertPartition.from_counts([2, 2])
        alpha_shards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        # fc2_alpha has shape (2, 1) — dim 0 is the rank's expert count too.
        beta_shards = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]
        params = [
            {"fc31_alpha": alpha.clone(), "fc2_alpha": beta.clone()}
            for alpha, beta in zip(alpha_shards, beta_shards)
        ]
        models = _make_rank_models(partition, params)

        comm, model = _exchange_and_fixup(models, 0, partition)

        self.assertEqual(comm.allgather.call_count, 1)
        experts = model.layers[LAYER].mlp.experts
        torch.testing.assert_close(experts.fc31_alpha.data, torch.cat(alpha_shards, dim=0))
        torch.testing.assert_close(experts.fc2_alpha.data, torch.cat(beta_shards, dim=0))


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
    """Validate ``_exchange_expert_scalars`` end-to-end on a redundant partition: shards loaded
    under Mode B overlap (size > stride) must reconstruct the full parameter correctly, with
    overlapping experts getting the same value from any peer."""

    def test_mode_b_overlap_dwdp3_via_peer_ranges(self) -> None:
        # dwdp=3, 256 experts, Mode B with size=86, stride=85
        # (2*85 + 86 = 256 — exact equality, 1-expert overlap between
        # adjacent ranks).  Shards loaded by each rank cover overlapping
        # ranges; ``_scatter_shards_to_full`` writes the shared experts
        # multiple times but the values agree (same checkpoint).
        partition = ExpertPartition.strided(num_experts=256, dwdp_size=3, size=86, stride=85)
        self.assertEqual(partition.peer_ranges, [(0, 86), (85, 171), (170, 256)])
        params = [
            {"fc31_alpha": torch.arange(start, end, dtype=torch.float32)}
            for start, end in partition.ranges
        ]
        models = _make_rank_models(partition, params)

        comm, model = _exchange_and_fixup(models, 0, partition)

        comm.allgather.assert_called_once()
        # Reconstructed alpha covers ALL 256 experts with their canonical values.
        torch.testing.assert_close(
            model.layers[LAYER].mlp.experts.fc31_alpha.data,
            torch.arange(0, 256, dtype=torch.float32),
        )


def _alphas(partition: ExpertPartition) -> list[dict[str, torch.Tensor]]:
    """Each rank's ``fc31_alpha`` shard: the ids of the experts it stores."""
    return [{"fc31_alpha": torch.arange(s, e, dtype=torch.float32)} for s, e in partition.ranges]


class TestExchangeRefusesReportsThatDoNotFit(unittest.TestCase):
    def test_every_rank_raises_the_same_error(self) -> None:
        """The verdict is made from the gathered reports, the same on every rank."""
        uneven = ExpertPartition.from_counts([3, 2])
        # Experts 2-3 are stored on ranks 0 and 1, experts 4-5 on ranks 1 and 2.
        overlap = ExpertPartition.strided(num_experts=8, dwdp_size=3, size=4, stride=2)
        two_copies = _alphas(overlap)
        two_copies[1]["fc31_alpha"][0] = -1.0
        alphas = _alphas(uneven)
        scales = [dict(a, fc31_input_scale=torch.tensor(s)) for a, s in zip(alphas, (1.0, 0.5))]
        other_names = [alphas[0], dict(alphas[1], fc2_alpha=torch.ones(2))]
        bias = "e_score_correction_bias"
        cases = {
            # case: (partition, small parameters per rank, bias per rank, offending name)
            "two values of an overlapping expert": (overlap, two_copies, None, "fc31_alpha"),
            "input scales that differ": (uneven, scales, None, "fc31_input_scale"),
            "other parameter names": (uneven, other_names, None, "fc2_alpha"),
            "a bias neither full nor per expert": (uneven, alphas, [torch.ones(3)] * 2, bias),
            "a bias on one rank only": (uneven, alphas, [torch.ones(3), None], bias),
        }
        for case, (partition, params, biases, name) in cases.items():
            with self.subTest(case):
                messages = set()
                for rank in range(partition.dwdp_size):
                    models = _make_rank_models(partition, params, biases)
                    with self.assertRaisesRegex(RuntimeError, name) as caught:
                        _exchange_and_fixup(models, rank, partition)
                    messages.add(str(caught.exception))
                self.assertEqual(len(messages), 1, messages)


def _bump(tensor: torch.Tensor, ulps: int) -> torch.Tensor:
    """``tensor`` moved by ``ulps`` units in the last place (positive values within a binade)."""
    ints = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return (tensor.view(ints) + ulps).view(tensor.dtype)


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


class TestPerLayerActivationScale(unittest.TestCase):
    """An NVFP4 AWQ layer's ``fc31_act_scale`` (``[1, hidden_size]``, one for every expert).

    Each rank takes it from the first expert it loads and casts it to the module dtype, so the
    ranks' copies may differ by one unit in the last place; every rank then installs rank 0's.
    """

    HIDDEN = 8
    REFERENCE = torch.full((1, HIDDEN), 1.5, dtype=torch.bfloat16)

    @staticmethod
    def _models(
        partition: ExpertPartition,
        act_scales: list[torch.Tensor],
        input_scales: list[torch.Tensor] | None = None,
    ) -> list[nn.Module]:
        """One model per rank: per-expert alphas, the given act scales and input scales."""
        params = _alphas(partition)
        for rank, entry in enumerate(params):
            entry["fc31_act_scale"] = act_scales[rank].clone()
            scale = torch.tensor(1.0) if input_scales is None else input_scales[rank]
            entry["fc31_input_scale"] = scale.clone()
        return _make_rank_models(partition, params)

    def test_one_ulp_apart_ends_with_rank_0s_copy_on_every_rank(self) -> None:
        partition = ExpertPartition.balanced(num_experts=7, dwdp_size=3)  # 3 + 2 + 2
        act_scales = [self.REFERENCE, _bump(self.REFERENCE, 1), _bump(self.REFERENCE, -1)]
        for rank in range(partition.dwdp_size):
            with self.subTest(rank=rank):
                models = self._models(partition, act_scales)
                param = models[rank].layers[LAYER].mlp.experts.fc31_act_scale

                comm, model = _exchange_and_fixup(models, rank, partition)

                comm.allgather.assert_called_once()
                experts = model.layers[LAYER].mlp.experts
                self.assertIs(experts.fc31_act_scale, param)  # copied in place
                self.assertTrue(_same_bytes(experts.fc31_act_scale.data, self.REFERENCE))
                torch.testing.assert_close(experts.fc31_alpha.data, torch.arange(7.0))

    def test_two_ulps_apart_raise_on_every_rank(self) -> None:
        partition = ExpertPartition.balanced(num_experts=7, dwdp_size=3)
        act_scales = [self.REFERENCE, _bump(self.REFERENCE, 1), _bump(self.REFERENCE, 2)]
        messages = set()
        for rank in range(partition.dwdp_size):
            models = self._models(partition, act_scales)
            pattern = r"fc31_act_scale is a per-layer activation scale, but ranks \[2\]"
            with self.assertRaisesRegex(RuntimeError, pattern) as caught:
                _exchange_and_fixup(models, rank, partition)
            messages.add(str(caught.exception))
        self.assertEqual(len(messages), 1, messages)

    def test_input_scales_one_ulp_apart_still_raise(self) -> None:
        """The alphas fold the zero-dimensional input scales in, so those stay bitwise."""
        partition = ExpertPartition.balanced(num_experts=7, dwdp_size=3)
        act_scales = [self.REFERENCE] * 3
        one = torch.tensor(1.0)
        for rank in range(partition.dwdp_size):
            with self.subTest(rank=rank):
                models = self._models(partition, act_scales, [one, one, _bump(one, 1)])
                with self.assertRaisesRegex(RuntimeError, r"fc31_input_scale is per layer"):
                    _exchange_and_fixup(models, rank, partition)

    def test_one_expert_per_rank_keeps_the_act_scale_per_layer(self) -> None:
        """With one expert per rank a ``[1, hidden_size]`` parameter has a per-expert shape: its
        name decides."""
        partition = ExpertPartition.balanced(num_experts=4, dwdp_size=4)
        act_scales = [self.REFERENCE, _bump(self.REFERENCE, 1), self.REFERENCE, self.REFERENCE]

        _, model = _exchange_and_fixup(self._models(partition, act_scales), 1, partition)

        experts = model.layers[LAYER].mlp.experts
        self.assertEqual(tuple(experts.fc31_act_scale.shape), (1, self.HIDDEN))
        self.assertTrue(_same_bytes(experts.fc31_act_scale.data, self.REFERENCE))
        torch.testing.assert_close(experts.fc31_alpha.data, torch.arange(4.0))


class _ConfigurableMoE(nn.Module):
    """ConfigurableMoE's shape: the backend holds the weights, the wrapper delegates to it."""

    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.backend = backend

    @property
    def w3_w1_weight(self) -> nn.Parameter:
        return self.backend.w3_w1_weight


class TestFixupConfigurableMoE(unittest.TestCase):
    def test_both_modules_see_every_expert_and_the_quant_scales_the_full_alphas(self) -> None:
        """K3's routed experts: a CuteDSL backend in a wrapper; the gate bias is full-sized."""
        partition = ExpertPartition.balanced(num_experts=7, dwdp_size=2)  # 4 + 3
        bias, models = torch.linspace(-1.0, 1.0, 7), []
        for start, end in partition.ranges:
            alpha = torch.arange(start, end, dtype=torch.float32)
            block = torch.zeros(end - start, 1)
            params = dict(fc31_alpha=alpha, fc2_alpha=alpha + 0.5, w3_w1_weight_scale=block)
            params.update(w2_weight_scale=block.clone(), fc31_input_scale=torch.tensor(1.0))
            params.update(fc2_input_scale=torch.tensor(1.0))
            backend = _Experts(partition.num_experts, end - start, params)
            NVFP4CuteDslFusedMoEMethod().setup_quant_scales(backend)
            models.append(_make_model(_ConfigurableMoE(backend), bias.clone()))

        _, model = _exchange_and_fixup(models, 1, partition)

        wrapper = model.layers[LAYER].mlp.experts
        self.assertEqual(wrapper.expert_size_per_partition, 7)
        self.assertEqual(wrapper.backend.expert_size_per_partition, 7)
        scales = wrapper.backend.quant_scales
        torch.testing.assert_close(scales.fc1_global, torch.arange(7.0))
        torch.testing.assert_close(scales.fc2_global, torch.arange(7.0) + 0.5)
        torch.testing.assert_close(model.layers[LAYER].mlp.gate.e_score_correction_bias.data, bias)


if __name__ == "__main__":
    unittest.main()
