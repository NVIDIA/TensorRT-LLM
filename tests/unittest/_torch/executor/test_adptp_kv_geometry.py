# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP groups (``TRTLLM_ADPTP_ATTN_TP``): KV-cache and attention head geometry, CPU only (torch + the
tensorrt_llm package; no CUDA device).

MSO shapes: 128 query / 16 KV heads, head_dim 64. Under DP2TP4 (8 ranks, attention DP, t=4) each rank's attention
holds 32 query / 4 KV heads, exactly TEP4's split, and every KV-head consumer (the attention module, KVCacheManager V1
and V2, the static size estimates, the bindings model config, the KvCacheCreator) must agree on it: the paged views
are shaped by the manager's per-rank KV heads (M3.1 E11 job 941231 failed when KVCacheManagerV2 kept every head
while the attention group held its share). With the switch unset every value equals the stock attention-DP one.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch import adp_groups
from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 128, 16, 64, 8192
GROUPED = {adp_groups.ENV: "4"}
UNSET = {adp_groups.ENV: ""}


def adp_mapping(rank, world=8):
    return Mapping(world_size=world, tp_size=world, moe_ep_size=world, rank=rank, gpus_per_node=4,
                   enable_attention_dp=True)


def tp_mapping(rank, world=4):
    return Mapping(world_size=world, tp_size=world, moe_ep_size=world, rank=rank, gpus_per_node=4)


def _per_rank(heads, tp):
    return (heads + tp - 1) // tp


def _model_config(mapping):
    pretrained = SimpleNamespace(num_attention_heads=Q_HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
                                 hidden_size=HIDDEN, num_hidden_layers=4)
    quant_mode = SimpleNamespace(has_fp8_kv_cache=lambda: True, has_fp4_kv_cache=lambda: False,
                                 has_int8_kv_cache=lambda: False)
    return SimpleNamespace(pretrained_config=pretrained, mapping=mapping,
                           quant_config=SimpleNamespace(quant_mode=quant_mode),
                           get_num_attention_layers=lambda: pretrained.num_hidden_layers)


@pytest.fixture
def clean_env():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(adp_groups.ENV, None)
        os.environ.pop(adp_groups.LEGACY_ENV, None)
        yield


class TestHeadSplit:
    def test_grouped_manager_heads_match_tep4(self, clean_env):
        with mock.patch.dict(os.environ, GROUPED):
            for world in (8, 16, 32):  # DP2/4/8 TP4
                for rank in range(world):
                    m = adp_mapping(rank, world)
                    assert adp_groups.group_attn_tp(m) == 4
                    assert kvm2.kv_head_tp_size(m) == 4
                    assert _per_rank(KV_HEADS, kvm2.kv_head_tp_size(m)) == 4
        # TEP4 (plain TP4, no attention DP): the same split, the switch is ignored without attention DP.
        for env in (GROUPED, UNSET):
            with mock.patch.dict(os.environ, env):
                assert kvm2.kv_head_tp_size(tp_mapping(1)) == 4

    def test_switch_unset_is_stock(self, clean_env):
        for env in ({}, UNSET, {adp_groups.ENV: "1"}):
            with mock.patch.dict(os.environ, env):
                for rank in range(8):
                    m = adp_mapping(rank)
                    assert kvm2.kv_head_tp_size(m) == 1  # stock: 1 if enable_attention_dp
                    assert adp_groups.dp_size(m) == m.tp_size

    def test_static_cache_size_estimates(self, clean_env):
        """Per-token KV bytes of the static estimates (V1 and V2): the grouped ADP rank equals a TEP4 rank; unset
        equals the stock ADP rank (every head)."""
        def v1(m):
            return KVCacheManager.get_cache_size_per_token(_model_config(m), m, num_layers=4)

        def v2(m):
            sizes, _ = kvm2._get_static_cache_size_layer_components(_model_config(m), m, 4)
            return sizes

        with mock.patch.dict(os.environ, GROUPED):
            assert v1(adp_mapping(5)) == v1(tp_mapping(1))
            assert v2(adp_mapping(5)) == v2(tp_mapping(1))
            grouped_v1, grouped_v2 = v1(adp_mapping(5)), v2(adp_mapping(5))
        with mock.patch.dict(os.environ, UNSET):
            stock_v1, stock_v2 = v1(adp_mapping(5)), v2(adp_mapping(5))
            no_adp = Mapping(world_size=1, tp_size=1, rank=0, gpus_per_node=4)
            assert stock_v1 == v1(no_adp)
            assert stock_v2 == v2(no_adp)
        assert stock_v1 == 4 * grouped_v1
        assert [s for s in stock_v2] == [4 * s for s in grouped_v2]


class TestKvCacheCreator:
    def _creator(self, mapping):
        c = object.__new__(_util.KvCacheCreator)
        c._mapping = mapping
        return c

    def test_dflash_layout_heads(self, clean_env):
        """The unified target/DFlash-draft KV layout (rank-local KV heads) follows the group: a grouped rank holds
        a TEP4 rank's 4 heads (the drafter shares the target's TP4-in-group layout); unset = every head (stock)."""
        def heads(m):
            return self._creator(m)._dflash_kv_layout(_model_config(m), torch.bfloat16)[0]

        with mock.patch.dict(os.environ, GROUPED):
            assert heads(adp_mapping(6)) == 4 == heads(tp_mapping(2))
        with mock.patch.dict(os.environ, UNSET):
            assert heads(adp_mapping(6)) == KV_HEADS

    def test_dp_units(self, clean_env):
        with mock.patch.dict(os.environ, GROUPED):
            assert adp_groups.dp_size(adp_mapping(3)) == 2
            assert adp_groups.dp_size(adp_mapping(3, 32)) == 8
        with mock.patch.dict(os.environ, UNSET):
            assert adp_groups.dp_size(adp_mapping(3)) == 8


class _FakeAllReduce:
    """Stands in for distributed.AllReduce (its workspace needs CUDA IPC); records the mapping it reduces over."""

    def __init__(self, mapping, strategy=None, dtype=None, **kwargs):
        self.mapping = mapping


class TestAttentionModule:
    """The built-in Attention under groups: tp = group, dp = groups, the sub-mappings are plain TP over the group and
    o_proj all-reduces over the group's 4 ranks (CPU: the backend and the all-reduce workspace are stubbed)."""

    def _attention(self, mapping):
        import tensorrt_llm._torch.attention.attention as attention_mod
        import tensorrt_llm._torch.distributed as dist_mod
        from tensorrt_llm._torch.model_config import ModelConfig

        config = ModelConfig(mapping=mapping, skip_create_weights_in_init=True)
        backend = mock.MagicMock()
        backend.support_fused_qkv.return_value = True
        with mock.patch.object(dist_mod, "AllReduce", _FakeAllReduce), \
                mock.patch.object(attention_mod, "create_attention", return_value=backend) as create:
            attn = attention_mod.Attention(hidden_size=HIDDEN, num_attention_heads=Q_HEADS,
                                           num_key_value_heads=KV_HEADS, max_position_embeddings=4096, bias=False,
                                           layer_idx=0, dtype=torch.bfloat16, config=config, head_dim=HEAD_DIM)
        # the backend (and its KV-cache view) is built with the per-rank head counts
        args = create.call_args.args
        assert (args[2], args[4]) == (attn.num_heads, attn.num_key_value_heads)
        return attn

    def test_grouped_rank_is_a_tp4_rank(self, clean_env):
        with mock.patch.dict(os.environ, GROUPED):
            for world in (8, 32):
                for rank in (0, 5, world - 1):
                    attn = self._attention(adp_mapping(rank, world))
                    group = list(range(rank // 4 * 4, rank // 4 * 4 + 4))
                    assert attn.tp_size == 4
                    assert attn.tp_rank == rank % 4
                    assert attn.num_heads == Q_HEADS // 4
                    assert attn.num_key_value_heads == KV_HEADS // 4
                    assert attn.mapping_o.tp_size == 4
                    assert not attn.mapping_o.enable_attention_dp
                    assert list(attn.mapping_o.tp_group) == group
                    assert attn.o_proj.all_reduce is not None
                    assert list(attn.o_proj.all_reduce.mapping.tp_group) == group
                    assert attn.num_key_value_heads == _per_rank(KV_HEADS, kvm2.kv_head_tp_size(adp_mapping(rank, world)))

    def test_grouped_matches_tep4_rank(self, clean_env):
        with mock.patch.dict(os.environ, GROUPED):
            g = self._attention(adp_mapping(6))
            t = self._attention(tp_mapping(2))
        assert (g.tp_size, g.tp_rank, g.num_heads, g.num_key_value_heads, g.q_size, g.kv_size) == \
            (t.tp_size, t.tp_rank, t.num_heads, t.num_key_value_heads, t.q_size, t.kv_size)

    def test_unset_is_stock_adp(self, clean_env):
        with mock.patch.dict(os.environ, UNSET):
            attn = self._attention(adp_mapping(5))
            assert attn.tp_size == 1
            assert attn.num_heads == Q_HEADS
            assert attn.num_key_value_heads == KV_HEADS
            assert attn.mapping_o.tp_size == 1
            assert attn.mapping_o.enable_attention_dp
            # stock ADP: o_proj's all-reduce is over a 1-rank TP group (a no-op)
            assert attn.o_proj.all_reduce is None or attn.o_proj.all_reduce.mapping.tp_size == 1
            assert attn.num_key_value_heads == _per_rank(KV_HEADS, kvm2.kv_head_tp_size(adp_mapping(5)))
