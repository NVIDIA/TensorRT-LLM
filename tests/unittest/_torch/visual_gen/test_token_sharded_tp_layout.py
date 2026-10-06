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
"""CPU tests for token-sharded TP as a layout: the SequenceSharder mode and the converter.

No process group: the sharder uses a simulated helper whose plan is already cached, so
``shard`` runs no collective (``gather`` is covered by the gloo tests in
``multi_gpu/test_token_sharded_tp_collectives.py``).
"""

import pytest
import torch
import torch.nn as nn
from token_sharded_tp_test_utils import padded_rows, simulated_helper

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.modules.rms_norm import RMSNormTPAware
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import (
    TokenShardedAdapter,
    TokenShardedColumn,
    TokenShardedMLP,
    TokenShardedRow,
    classify,
    convert_to_token_sharded_tp,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import TokenShardPlan
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm.visual_gen.args import ParallelConfig

pytestmark = pytest.mark.cpu_only

_SHAPES = [(1, 6, 2), (1, 5, 2), (2, 8, 2), (2, 9, 4), (3, 7, 2), (2, 9, 3)]


def _token_sharded(batch, seq, tp, rank):
    plan = TokenShardPlan.build(batch, seq, tp, rank)
    sharder = SequenceSharder(size=1, rank=0, group=None)
    sharder.use_token_sharded_tp(simulated_helper(plan))
    return sharder, plan


def _sample_ids(batch, seq, dim=4):
    """[B, S, dim] whose every value is its sample index + 1 (pad rows stay 0)."""
    return (torch.arange(batch, dtype=torch.float32) + 1).view(batch, 1, 1).expand(batch, seq, dim)


# =============================================================================
# Stream and per-token tables: this rank's rows, as whole-sample groups
# =============================================================================


@pytest.mark.parametrize("batch,seq,tp", _SHAPES)
def test_shard_gives_each_rank_its_rows_as_sample_groups(batch, seq, tp):
    x = torch.randn(batch, seq, 4)
    ids = _sample_ids(batch, seq)
    for rank in range(tp):
        sharder, plan = _token_sharded(batch, seq, tp, rank)
        rows = padded_rows(x, plan)[plan.row_start : plan.row_start + plan.local_rows]
        got = sharder.shard(x, dim=1)
        assert got.dim() == 3 and got.shape[-1] == 4
        assert got.shape[0] * got.shape[1] == plan.local_rows
        assert torch.equal(got.reshape(-1, 4), rows)
        # Each [g, D] group holds rows of one sample (or that sample's padding), so a
        # per-sample [n, 1, D] table broadcasts over the shard as over [B, S, D].
        for group in sharder.shard(ids, dim=1):
            assert len(set(group[group != 0].tolist())) <= 1


@pytest.mark.parametrize("batch,seq,tp", _SHAPES)
def test_shard_per_token_table(batch, seq, tp):
    temb = torch.randn(batch, seq, 6, 8)
    for rank in range(tp):
        sharder, plan = _token_sharded(batch, seq, tp, rank)
        x_loc = sharder.shard(torch.zeros(batch, seq, 8), dim=1)
        got = sharder.shard(temb, dim=1, expected_seq_len=seq)
        rows = padded_rows(temb, plan)[plan.row_start : plan.row_start + plan.local_rows]
        assert got.shape == (*x_loc.shape[:2], 6, 8)
        assert torch.equal(got.reshape(-1, 6, 8), rows)


@pytest.mark.parametrize("batch,seq,tp", _SHAPES)
def test_shard_per_sample_matches_each_group(batch, seq, tp):
    table = torch.randn(batch, 6, 8)
    ids = _sample_ids(batch, seq)
    for rank in range(tp):
        sharder, _ = _token_sharded(batch, seq, tp, rank)
        groups = sharder.shard(ids, dim=1)
        got = sharder.shard_per_sample(table)
        assert got.shape == (groups.shape[0], 6, 8)
        for j, group in enumerate(groups):
            sample = int(group.max().item()) - 1  # this group's sample (all rows non-pad: max)
            if sample >= 0:
                assert torch.equal(got[j], table[sample])


def test_attention_inputs_stay_whole():
    """RoPE tables, positions, masks and attention-side K/V are not sharded: attention
    sees all tokens after the column projection's all-gather."""
    sharder, _ = _token_sharded(2, 8, 2, 1)
    cos, sin = torch.randn(8, 16), torch.randn(8, 16)
    rope = sharder.shard_rope((cos, sin), seq_len=8, seq_dim=0)
    assert rope[0] is cos and rope[1] is sin
    ids = torch.randn(8, 3)
    assert sharder.shard_attention_input(ids, dim=0) is ids


def test_mode_flags_keep_their_sequence_parallel_meaning():
    sharder, _ = _token_sharded(2, 8, 2, 0)
    assert sharder.token_sharded_tp is True
    # Ulysses-only code (padding to multiples of size, K/V sharding) keys on these.
    assert sharder.is_active is False
    assert sharder.size == 1


def test_token_sharded_mode_shards_only_the_sequence_dim():
    sharder, _ = _token_sharded(2, 8, 2, 0)
    assert sharder.shard(None, dim=1) is None
    other = torch.randn(2, 5, 4)  # a field whose seq axis is not the stream's
    assert sharder.shard(other, dim=1, expected_seq_len=8) is other
    with pytest.raises(ValueError, match="dim=1"):
        sharder.shard(torch.randn(8, 3), dim=0)


def test_tables_follow_the_stream_plan():
    """Only the token stream (no expected_seq_len) selects the plan; a per-token table must
    match it rather than re-plan the forward."""
    sharder, _ = _token_sharded(2, 8, 2, 0)
    sharder.shard(torch.randn(2, 8, 4), dim=1)
    with pytest.raises(ValueError, match="token stream"):
        sharder.shard(torch.randn(4, 8, 6, 4), dim=1, expected_seq_len=8)


# =============================================================================
# Sequence mode (Ulysses / Ring / Attention2D) and inactive sharders are unchanged
# =============================================================================


def test_sequence_mode_shards_attention_inputs_too():
    sharder = SequenceSharder(size=2, rank=1, group=None)
    t = torch.arange(24.0).view(2, 4, 3)
    assert torch.equal(sharder.shard_attention_input(t, dim=1), sharder.shard(t, dim=1))
    table = torch.randn(2, 6, 8)
    assert sharder.shard_per_sample(table) is table
    assert sharder.token_sharded_tp is False


def test_inactive_sharder_passes_everything_through():
    sharder = SequenceSharder(size=1, rank=0, group=None)
    t = torch.randn(2, 4, 3)
    assert sharder.shard(t, dim=1) is t
    assert sharder.shard_attention_input(t, dim=1) is t
    assert sharder.shard_per_sample(t) is t
    assert sharder.gather(t, dim=1) is t
    assert sharder.token_sharded_tp is False


def test_token_sharded_mode_excludes_sequence_parallelism():
    plan = TokenShardPlan.build(1, 8, 2, 0)
    sharder = SequenceSharder(size=2, rank=0, group=None)
    with pytest.raises(ValueError, match="cannot be combined with sequence parallelism"):
        sharder.use_token_sharded_tp(simulated_helper(plan))


# =============================================================================
# The converter: which TP modules become token-sharded adapters
# =============================================================================
#
# Real TP modules need a CUDA device mesh; these toy blocks use real TRT-LLM modules built
# at TP=1 and given the TP metadata they would have at tp_size=2 (the GPU tests convert real
# Wan blocks).

_TP = 2


def _all_reduce():
    """An AllReduce module without a communicator (only its type matters here)."""
    ar = AllReduce.__new__(AllReduce)
    nn.Module.__init__(ar)
    return ar


def _as_tp(linear, mode, reduce_output=False):
    linear.tp_size, linear.tp_mode, linear.reduce_output = _TP, mode, reduce_output
    linear.all_reduce = _all_reduce() if reduce_output else None
    return linear


def _attention(qkv_mode, self_attention=False):
    attn = Attention(
        hidden_size=64,
        num_attention_heads=4,
        qkv_mode=qkv_mode,
        config=DiffusionModelConfig(),
        separate_qkv_is_self_attention=self_attention,
    )
    for name in ("qkv_proj", "to_q", "to_k", "to_v"):
        if hasattr(attn, name):
            _as_tp(getattr(attn, name), TensorParallelMode.COLUMN)
    _as_tp(attn.to_out[0], TensorParallelMode.ROW, reduce_output=True)
    return attn


def _mlp():
    mlp = MLP(
        hidden_size=64, intermediate_size=128, bias=True, dtype=torch.bfloat16, config=ModelConfig()
    )
    # The up-projection keeps the dormant AllReduce a default-built COLUMN Linear has.
    _as_tp(mlp.up_proj, TensorParallelMode.COLUMN).all_reduce = _all_reduce()
    _as_tp(mlp.down_proj, TensorParallelMode.ROW, reduce_output=True)
    return mlp


def _tp_aware_norm():
    norm = RMSNormTPAware(hidden_size=64, eps=1e-6)
    norm.allreduce = _all_reduce()  # reduces over heads within each token: stays
    return norm


class _WanLikeBlock(nn.Module):
    """Self-attention (fused QKV), cross-attention, an FFN, a TP-aware QK norm and a column
    projection of the image embeddings (Wan I2V's add_k_proj)."""

    def __init__(self):
        super().__init__()
        self.attn1 = _attention(QKVMode.FUSE_QKV)
        self.attn1.norm_q = _tp_aware_norm()
        self.attn2 = _attention(QKVMode.SEPARATE_QKV)
        self.add_k_proj = _as_tp(Linear(64, 64, dtype=torch.bfloat16), TensorParallelMode.COLUMN)
        self.ffn = _mlp()


class _Model(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks or [_WanLikeBlock(), _WanLikeBlock()])


_WAN_LIKE = {
    "attn1.qkv_proj": "column",
    "attn1.to_out.0": "row",
    "attn2.to_q": "column",
    "attn2.to_out.0": "row",
    "ffn": "mlp",
}


def _helper():
    return simulated_helper(TokenShardPlan.build(1, 8, _TP, 0))


def test_rules_pick_the_boundary_modules():
    assert classify(_Model()) == {
        f"blocks.{i}.{name}": kind for i in range(2) for name, kind in _WAN_LIKE.items()
    }


def test_self_attention_with_separate_projections_gathers_q_k_v():
    block = nn.Module()
    block.attn = _attention(QKVMode.SEPARATE_QKV, self_attention=True)
    assert classify(_Model(block)) == {
        "blocks.0.attn.to_q": "column",
        "blocks.0.attn.to_k": "column",
        "blocks.0.attn.to_v": "column",
        "blocks.0.attn.to_out.0": "row",
    }


def test_convert_swaps_classes_in_place():
    model = _Model()
    keys = set(model.state_dict())
    ids = {name: id(m) for name, m in model.named_modules() if not name.endswith("all_reduce")}
    convert_to_token_sharded_tp(model, _helper())
    b0, b1 = model.blocks
    row = b0.attn1.to_out[0]
    assert isinstance(row, Linear) and isinstance(row, TokenShardedRow)
    assert (row.reduce_output, row.all_reduce, row.use_fused_gemm_allreduce) == (False, None, False)
    assert isinstance(b0.attn1.qkv_proj, TokenShardedColumn)
    assert isinstance(b0.attn2.to_q, TokenShardedColumn)
    assert not isinstance(b0.attn2.to_k, TokenShardedColumn)  # reads the encoder states
    assert not isinstance(b0.add_k_proj, TokenShardedColumn)
    assert isinstance(b0.ffn, MLP) and isinstance(b0.ffn, TokenShardedMLP)
    assert type(b0.ffn.down_proj) is Linear  # converted with its MLP
    assert (b0.ffn.down_proj.reduce_output, b0.ffn.down_proj.all_reduce) == (False, None)
    assert b0.attn1.norm_q.allreduce is not None
    # Same objects and state-dict keys; one adapter class per base class.
    assert set(model.state_dict()) == keys
    assert all(id(model.get_submodule(name)) == i for name, i in ids.items())
    assert type(b0.attn1.to_out[0]) is type(b1.attn1.to_out[0]) is type(b0.attn2.to_out[0])
    with pytest.raises(ValueError, match="already token-sharded"):
        convert_to_token_sharded_tp(model, _helper())


def test_unknown_all_reduce_is_rejected_unless_listed():
    block = _WanLikeBlock()
    block.custom = nn.Module()
    block.custom.allreduce = _all_reduce()
    with pytest.raises(ValueError, match=r"blocks\.0\.custom\.allreduce"):
        classify(_Model(block))
    assert classify(_Model(block), exceptions={"custom": "keep"}) == {
        f"blocks.0.{name}": kind for name, kind in _WAN_LIKE.items()
    }


def test_exceptions_override_the_rules():
    got = classify(
        _Model(_WanLikeBlock()), exceptions={"attn2.to_q": "keep", "add_k_proj": "column"}
    )
    want = dict(_WAN_LIKE, add_k_proj="column")
    del want["attn2.to_q"]
    assert got == {f"blocks.0.{name}": kind for name, kind in want.items()}


def test_blocks_of_a_container_must_convert_alike():
    odd = _WanLikeBlock()
    odd.attn2 = _attention(QKVMode.SEPARATE_QKV, self_attention=True)
    with pytest.raises(ValueError, match="convert differently"):
        classify(_Model(_WanLikeBlock(), odd))


def test_block_without_row_projections_is_rejected():
    block = nn.Module()
    block.proj = _as_tp(Linear(64, 64, dtype=torch.bfloat16), TensorParallelMode.COLUMN)
    with pytest.raises(ValueError, match="no all-reducing row-parallel"):
        classify(_Model(block))


def test_adapter_classes_convert_like_their_kind():
    """An exception (or registry entry) naming an adapter class prepares the module as the
    kind does: a row stops all-reducing."""
    model = _Model(_WanLikeBlock())
    convert_to_token_sharded_tp(model, _helper(), exceptions={"attn1.to_out.0": TokenShardedRow})
    row = model.blocks[0].attn1.to_out[0]
    assert isinstance(row, TokenShardedRow)
    assert (row.reduce_output, row.all_reduce) == (False, None)


class _JointProjection(nn.Module):
    """A custom module with its own all-reduce, as a model package might have."""

    def __init__(self):
        super().__init__()
        self.allreduce = _all_reduce()


class _TokenShardedJoint(TokenShardedAdapter):
    @classmethod
    def prepare(cls, module, tp, name):
        module.allreduce = None  # reduce-scatters in forward instead


def test_registered_adapters_prepare_their_module(monkeypatch):
    from tensorrt_llm._torch.visual_gen.parallel import token_sharded_modules

    monkeypatch.setitem(token_sharded_modules._REGISTRY, _JointProjection, _TokenShardedJoint)
    block = _WanLikeBlock()
    block.joint = _JointProjection()
    model = _Model(block)
    convert_to_token_sharded_tp(model, _helper())
    assert isinstance(model.blocks[0].joint, _TokenShardedJoint)
    assert model.blocks[0].joint.allreduce is None


def _row_as_column(block):
    return {"attn1.to_out.0": "column"}


def _column_as_row(block):
    return {"add_k_proj": "row"}


def _down_proj_for_another_tp_size(block):
    block.ffn.down_proj.tp_size = 4
    return {}


def _row_built_for_fused_gemm_all_reduce(block):
    block.attn1.to_out[0].use_fused_gemm_allreduce = True
    return {}


@pytest.mark.parametrize(
    "misbuild,error",
    [
        (_row_as_column, "column-parallel Linear"),
        (_column_as_row, "row-parallel Linear"),
        (_down_proj_for_another_tp_size, "tp_size=4"),
        (_row_built_for_fused_gemm_all_reduce, "fused GEMM"),
    ],
)
def test_conversion_rejects_projections_built_for_another_layout(misbuild, error):
    block = _WanLikeBlock()
    exceptions = misbuild(block)
    with pytest.raises(ValueError, match=error):
        convert_to_token_sharded_tp(_Model(block), _helper(), exceptions=exceptions)


class _ToyDiT(BaseDiffusionModel):
    _supports_token_sharded_tp = True

    def __init__(self, model_config):
        super().__init__(model_config)
        self.sharder = SequenceSharder(size=1, rank=0, group=None)
        self.blocks = nn.ModuleList([_WanLikeBlock()])
        self._apply_tp_layout()


@pytest.mark.parametrize("layout", [None, "replicated"])
def test_plain_tp_model_is_left_unchanged(layout):
    model = _ToyDiT(DiffusionModelConfig(parallel=ParallelConfig(tp_size=_TP, tp_layout=layout)))
    assert model.sharder.token_sharded_tp is False
    assert not any(isinstance(m, TokenShardedAdapter) for m in model.modules())
    assert model.blocks[0].attn1.to_out[0].reduce_output is True
    model.check_tp_layout_applied()


class _ForgetfulDiT(BaseDiffusionModel):
    """Declares support but never calls _apply_tp_layout()."""

    _supports_token_sharded_tp = True

    def __init__(self, model_config, with_sharder=True):
        super().__init__(model_config)
        if with_sharder:
            self.sharder = SequenceSharder(size=1, rank=0, group=None)
        self.blocks = nn.ModuleList([_WanLikeBlock()])


def test_a_supporting_model_must_apply_the_layout():
    config = DiffusionModelConfig(parallel=ParallelConfig(tp_size=_TP, tp_layout="token_sharded"))
    with pytest.raises(RuntimeError, match="_apply_tp_layout"):
        _ForgetfulDiT(config).check_tp_layout_applied()
    with pytest.raises(AttributeError, match="sharder"):
        _ForgetfulDiT(config, with_sharder=False)._apply_tp_layout()


# =============================================================================
# Attention: a self-attention whose q and k would cover different tokens
# =============================================================================


class _GatheringProjection(nn.Module):
    """Stands in for a converted projection: returns all tokens (here: twice the shard)."""

    def forward(self, x):
        return torch.cat([x, x], dim=1)


def test_self_attention_rejects_q_and_k_over_different_tokens():
    attn = Attention(
        hidden_size=64,
        num_attention_heads=4,
        qkv_mode=QKVMode.SEPARATE_QKV,
        config=DiffusionModelConfig(),
    )
    attn.to_q, attn.to_k, attn.to_v = nn.Identity(), nn.Identity(), nn.Identity()
    x = torch.randn(1, 4, 64)
    q, k, v = attn.get_qkv(x)
    assert q.shape == k.shape == v.shape == x.shape
    attn.to_q = _GatheringProjection()  # to_q converted, to_k / to_v not: a misbinding
    with pytest.raises(ValueError, match="separate_qkv_is_self_attention"):
        attn.get_qkv(x)
    # Cross-attention: k / v read the encoder states, so their token count differs.
    q, k, _ = attn.get_qkv(x, encoder_hidden_states=torch.randn(1, 3, 64))
    assert (q.shape[1], k.shape[1]) == (8, 3)
