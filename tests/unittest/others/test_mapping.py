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
import unittest
from unittest.mock import patch

import pytest

from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _force_mpi_topology_mapping():
    with patch("tensorrt_llm.mapping.mpi_disabled", return_value=False):
        yield


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "attention_dp", "shares_requests"),
    [
        (1, 4, True, True),  # helix + attention DP at tp 1: one DP group
        (2, 2, True, False),  # attention DP across the 2 original tp ranks
        (2, 2, False, True),  # helix without attention DP
    ],
)
def test_repurposed_helix_mapping_for_full_rows(tp_size, cp_size, attention_dp,
                                                shares_requests):
    world = tp_size * cp_size
    helix = Mapping(
        world_size=world,
        rank=world - 1,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_config={"cp_type": "HELIX"},
        enable_attention_dp=attention_dp,
    )
    repurposed = helix.repurpose_helix_cp_to_tp()
    assert repurposed.helix_repurposed_cp_size == cp_size
    assert repurposed.tp_group_shares_requests is shares_requests
    assert Mapping.from_dict(
        repurposed.to_dict()).helix_repurposed_cp_size == cp_size

    full_rows = repurposed.mapping_for_full_rows()
    if attention_dp and shares_requests:
        assert full_rows is not repurposed
        assert not full_rows.enable_attention_dp
        assert (full_rows.tp_size, full_rows.tp_rank) == (world,
                                                          repurposed.tp_rank)
        assert repurposed.enable_attention_dp  # the source mapping is untouched
    else:
        assert full_rows is repurposed


@pytest.mark.parametrize(("world_size", "pp_size"), [(1, 1), (2, 2)])
def test_plain_attention_dp_at_tp1_keeps_attention_dp(world_size, pp_size):
    # Not helix: attention DP at tp_size 1 must not be read as a shared TP group,
    # or the LM head and the vocab embedding disagree on their TP mode.
    mapping = Mapping(world_size=world_size,
                      rank=0,
                      tp_size=1,
                      pp_size=pp_size,
                      enable_attention_dp=True)
    assert not mapping.tp_group_shares_requests
    assert mapping.mapping_for_full_rows() is mapping


def _fields(mapping):
    return {
        **mapping.to_dict(), "tp_group": mapping.tp_group,
        "pp_group": mapping.pp_group
    }


@pytest.mark.parametrize(
    ("world_size", "rank", "tp_size", "pp_size", "dp_size", "tp_group"),
    [
        (4, 3, 4, 1, 2, [2, 3]),
        (8, 5, 4, 2, 2, [4, 5]),  # groups stay inside one PP stage
        (8, 6, 4, 2, 4, [6]),
    ],
)
def test_split_tp_uses_contiguous_groups(world_size, rank, tp_size, pp_size,
                                         dp_size, tp_group):
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      enable_attention_dp=True)

    split = mapping.split_tp(dp_size)

    assert (split.tp_size, split.pp_size) == (tp_size // dp_size,
                                              pp_size * dp_size)
    assert split.tp_group == tp_group
    assert not split.enable_attention_dp
    assert mapping.split_tp(dp_size,
                            enable_attention_dp=True).enable_attention_dp


def test_split_tp_with_context_parallel_needs_one_rank_per_group():
    mapping = Mapping(world_size=4, rank=3, tp_size=2, cp_size=2)

    assert mapping.split_tp(2).tp_group == [3]
    with pytest.raises(NotImplementedError, match="one rank per group"):
        mapping.split_tp(1)
    with pytest.raises(ValueError, match="must divide"):
        mapping.split_tp(3)


# Each caller below used to build its own sub-mapping. These are the old
# constructions, copied as they were, to check that split_tp gives the same
# mapping field by field.
def _old_overridden_tp_mapping(mapping, overridden_tp_size, keep_adp):
    pp_size = mapping.pp_size * mapping.tp_size // overridden_tp_size
    kwargs = {
        "enable_attention_dp": mapping.enable_attention_dp
    } if keep_adp else {}
    return Mapping(world_size=overridden_tp_size * pp_size,
                   rank=mapping.rank,
                   gpus_per_node=mapping.gpus_per_node,
                   tp_size=overridden_tp_size,
                   pp_size=pp_size,
                   **kwargs)


def _gated_mlp_mapping(mapping, overridden_tp_size):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    config = ModelConfig(mapping=mapping, skip_create_weights_in_init=True)
    return GatedMLP(hidden_size=8,
                    intermediate_size=16,
                    bias=False,
                    config=config,
                    overridden_tp_size=overridden_tp_size).down_proj.mapping


def _mlp_mapping(mapping, overridden_tp_size):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP
    config = ModelConfig(mapping=mapping, skip_create_weights_in_init=True)
    return MLP(hidden_size=8,
               intermediate_size=16,
               bias=False,
               config=config,
               overridden_tp_size=overridden_tp_size).down_proj.mapping


@pytest.mark.parametrize("module", [_gated_mlp_mapping, _mlp_mapping],
                         ids=["gated_mlp", "mlp"])
@pytest.mark.parametrize(
    ("tp_size", "pp_size", "rank", "enable_attention_dp", "overridden_tp_size"),
    [(4, 1, 3, False, 2), (4, 2, 5, True, 2), (4, 1, 2, True, 1),
     (2, 2, 3, False, 1)],
)
def test_mlp_overridden_tp_mapping_is_unchanged(module, tp_size, pp_size, rank,
                                                enable_attention_dp,
                                                overridden_tp_size):
    mapping = Mapping(world_size=tp_size * pp_size,
                      rank=rank,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      enable_attention_dp=enable_attention_dp)

    old = _old_overridden_tp_mapping(mapping, overridden_tp_size, keep_adp=True)
    # The all-reduce workspace needs a GPU; only the mapping is checked here.
    with patch("tensorrt_llm._torch.distributed.AllReduce"):
        new = module(mapping, overridden_tp_size)
    assert _fields(new) == _fields(old)


@pytest.mark.parametrize(("tp_size", "pp_size", "rank"), [(4, 1, 3), (2, 2, 3)])
def test_qwen3vl_merger_mapping_is_unchanged(tp_size, pp_size, rank):
    from types import SimpleNamespace

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3vl import \
        Qwen3VLVisionPatchMerger
    mapping = Mapping(world_size=tp_size * pp_size,
                      rank=rank,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      enable_attention_dp=True)
    pretrained_config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=8,
                                      spatial_merge_size=2,
                                      out_hidden_size=8),
        text_config=SimpleNamespace(rms_norm_eps=1e-6, dtype=None))
    config = ModelConfig(pretrained_config=pretrained_config,
                         mapping=mapping,
                         skip_create_weights_in_init=True)

    merger = Qwen3VLVisionPatchMerger(config)

    old = _old_overridden_tp_mapping(mapping, 1, keep_adp=False)
    assert _fields(merger.linear_fc2.mapping) == _fields(old)


@pytest.mark.parametrize(
    ("tp_size", "pp_size", "cp_size", "rank"), [(2, 1, 1, 1), (2, 2, 1, 3),
                                                (2, 1, 2, 3), (1, 1, 4, 2)])
def test_kimi_vision_mapping_is_unchanged(tp_size, pp_size, cp_size, rank):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_kimi_k25 import \
        _get_vision_tp_mapping
    mapping = Mapping(world_size=tp_size * pp_size * cp_size,
                      rank=rank,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      cp_size=cp_size)
    attn_ranks = pp_size * tp_size * cp_size
    old = Mapping(world_size=attn_ranks,
                  rank=rank,
                  gpus_per_node=mapping.gpus_per_node,
                  tp_size=1,
                  pp_size=attn_ranks)

    # 3 heads do not split over the TP ranks, so the tower is replicated.
    new = _get_vision_tp_mapping(ModelConfig(mapping=mapping), num_heads=3)

    assert _fields(new) == _fields(old)


@pytest.mark.parametrize(("tp_size", "pp_size", "rank"), [(4, 1, 3), (2, 2, 3)])
def test_glm5_next_attention_mapping_is_unchanged(tp_size, pp_size, rank):
    from tensorrt_llm._torch.models.modeling_glm5_next import \
        glm5_next_attention_mapping
    mapping = Mapping(world_size=tp_size * pp_size,
                      rank=rank,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      enable_attention_dp=True)
    old = Mapping(world_size=pp_size * tp_size,
                  tp_size=1,
                  pp_size=pp_size * tp_size,
                  rank=rank,
                  gpus_per_node=mapping.gpus_per_node,
                  enable_attention_dp=True)

    assert _fields(glm5_next_attention_mapping(mapping)) == _fields(old)


@pytest.mark.parametrize(("enable_attention_dp", "dp_size"), [(False, 2),
                                                              (True, 1),
                                                              (True, 2)])
def test_encoder_mapping_is_unchanged(enable_attention_dp, dp_size):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.multimodal_encoder_data_parallel import \
        EncoderParallelLayout
    from tensorrt_llm.llmapi.llm_args import MultimodalConfig
    mapping = Mapping(world_size=4,
                      rank=3,
                      tp_size=4,
                      enable_attention_dp=enable_attention_dp)
    old = Mapping(world_size=4,
                  rank=3,
                  gpus_per_node=mapping.gpus_per_node,
                  tp_size=4 // dp_size,
                  pp_size=dp_size)
    config = ModelConfig(
        mapping=mapping,
        multimodal_config=MultimodalConfig(encoder_data_parallel_size=dp_size))

    new = EncoderParallelLayout.from_config(config).mapping

    assert _fields(new) == _fields(old)


class TestMapping(unittest.TestCase):

    def test_mapping(self):
        m = Mapping(world_size=8, rank=0, tp_size=8)
        self.assertEqual(len(m.tp_groups), 1)
        self.assertEqual(len(m.pp_groups), 8)
        self.assertEqual(m.tp_group, [0, 1, 2, 3, 4, 5, 6, 7])

        m = Mapping(world_size=8, rank=0, tp_size=4, pp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(m.tp_group, [0, 1, 2, 3])
        self.assertEqual(m.pp_group, [0, 4])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 4)

        m = Mapping(world_size=8, rank=6, tp_size=2, pp_size=4)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(m.tp_group, [6, 7])
        self.assertEqual(m.pp_group, [0, 2, 4, 6])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 0)

        m = Mapping(world_size=2, rank=0, cp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(len(m.cp_groups), 1)
        self.assertEqual(m.tp_group, [0])
        self.assertEqual(m.pp_group, [0])
        self.assertEqual(m.cp_group, [0, 1])

        m = Mapping(world_size=8, rank=3, tp_size=2, pp_size=2, cp_size=2)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(len(m.cp_groups), 4)
        self.assertEqual(m.tp_group, [1, 3])
        self.assertEqual(m.pp_group, [3, 7])
        self.assertEqual(m.cp_group, [2, 3])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertTrue(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 7)
        self.assertEqual(m.next_pp_rank(), 7)
        self.assertEqual(m.prev_cp_rank(), 2)
        self.assertEqual(m.next_cp_rank(), 2)

        m = Mapping(world_size=16, rank=9, tp_size=2, pp_size=2, cp_size=4)
        self.assertEqual(m.tp_group, [9, 13])
        self.assertEqual(m.pp_group, [1, 9])
        self.assertEqual(m.cp_group, [8, 9, 10, 11])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertFalse(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 1)
        self.assertEqual(m.next_pp_rank(), 1)
        self.assertEqual(m.prev_cp_rank(), 8)
        self.assertEqual(m.next_cp_rank(), 10)
