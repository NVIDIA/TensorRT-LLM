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
"""CPU tests for the token-sharded TP helper (no process group, no GPU).

Plans are simulated per rank: every rank's shard of a full tensor is checked against
the same operation on the full tensor, sliced by that rank's rows.
"""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from token_sharded_tp_test_utils import padded_rows, simulated_helper, swizzle_ref, unswizzle_ref

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    TokenShardedTP,
    TokenShardPlan,
    regroup_swizzled_sf,
    swizzled_sf_numel,
)
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.visual_gen.args import ParallelConfig

pytestmark = pytest.mark.cpu_only


# =============================================================================
# Helpers
# =============================================================================


def rank_slice(plan: TokenShardPlan) -> slice:
    return slice(plan.row_start, plan.row_start + plan.local_rows)


# =============================================================================
# A1. Plan invariants
# =============================================================================

_PLAN_BATCHES = [1, 2, 3, 4]
_PLAN_SEQS = [1, 5, 7, 8, 9, 12, 105, 4680, 42525, 75600]
_PLAN_TPS = [2, 3, 4, 6, 8]


@pytest.mark.parametrize("tp", _PLAN_TPS)
@pytest.mark.parametrize("batch", _PLAN_BATCHES)
def test_plan_invariants(batch, tp):
    gen = torch.Generator().manual_seed(batch * 100 + tp)
    for seq in _PLAN_SEQS:
        d = math.gcd(tp, batch)
        t_prime = tp // d
        covered = 0
        for rank in range(tp):
            p = TokenShardPlan.build(batch, seq, tp, rank)
            s_pad, m, g = p.padded_seq_len, p.local_rows, p.rows_per_entry
            assert tp * m == batch * s_pad
            assert s_pad % t_prime == 0 and 0 <= s_pad - seq < t_prime
            assert len(p.entry_batch) == batch // d
            assert p.is_padded == (batch * seq % tp != 0)
            assert p.num_tokens == batch * seq
            assert p.row_start == rank * m and m % g == 0
            if seq <= 105:
                rows = torch.arange(m)
            else:
                rows = torch.randint(0, m, (1000,), generator=gen)
            entry = torch.tensor(p.entry_batch)[rows // g]
            assert torch.equal(entry, (p.row_start + rows) // s_pad), (batch, seq, tp, rank)
            segs = p.local_segments()
            assert len(segs) <= batch + 1
            assert sum(s1 - s0 for _, s0, s1 in segs) == m
            flat = [b * s_pad + s for b, s0, s1 in segs for s in (s0, s1 - 1)]
            assert flat[0] == p.row_start and flat[-1] == p.row_start + m - 1
            covered += m
        assert covered == batch * pad_up(seq, t_prime)


def test_plan_build_rejects_invalid():
    with pytest.raises(ValueError):
        TokenShardPlan.build(0, 8, 2, 0)
    with pytest.raises(ValueError):
        TokenShardPlan.build(1, 0, 2, 0)
    with pytest.raises(ValueError):
        TokenShardPlan.build(1, 8, 2, 2)
    with pytest.raises(ValueError):
        TokenShardPlan.build(1, 8, 2, -1)


# =============================================================================
# A2. Per-shard tables: shard result == full-tensor result sliced by rows
# =============================================================================

# (B, S, tp): unpadded, rank straddling a sample boundary, interior padding,
# tail padding (B=1) with a fully padded rank, per-row tables.
_ROW_LOCAL_CASES = [
    (1, 8, 2),
    (2, 8, 4),
    (2, 12, 3),
    (2, 5, 3),
    (2, 9, 4),
    (3, 7, 4),
    (1, 5, 4),
    (4, 6, 8),
]
_D = 256


def _row_local_inputs(batch, seq, dtype):
    gen = torch.Generator().manual_seed(batch * 1000 + seq)
    x = torch.randn(batch, seq, _D, generator=gen).to(dtype)
    y = torch.randn(batch, seq, _D, generator=gen).to(dtype)
    table = torch.randn(batch, 6, _D, generator=gen)  # per-sample modulation [B, 6, D]
    per_token = torch.randn(batch, seq, 6, _D, generator=gen)  # per-token modulation
    return x, y, table, per_token


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,seq,tp", _ROW_LOCAL_CASES)
def test_tables_match_full_rows(batch, seq, tp, dtype):
    x, _, table, per_token = _row_local_inputs(batch, seq, dtype)
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank)
        sp = simulated_helper(p)
        rows = rank_slice(p)
        # per_sample_table expanded per local row == the per-row sample table.
        tab = sp.per_sample_table(table)
        assert tab.shape[0] == len(p.entry_batch)
        per_row = tab.repeat_interleave(p.rows_per_entry, dim=0)
        sample_of_row = torch.arange(batch).repeat_interleave(p.padded_seq_len)[rows]
        assert torch.equal(per_row, table[sample_of_row])
        # shard == the global padded stream sliced (zeros on pad rows), tables included.
        assert torch.equal(sp.shard(per_token), padded_rows(per_token, p)[rows])
        assert torch.equal(sp.shard(x), padded_rows(x, p)[rows])


# =============================================================================
# A3. Scaling-factor layout
# =============================================================================


def test_swizzle_ref_roundtrip():
    lin = torch.randint(0, 255, (300, 5), dtype=torch.uint8)
    assert torch.equal(unswizzle_ref(swizzle_ref(lin), 300, 5), lin)


def _shard_sf_buffers(lin_real: torch.Tensor, batch: int, seq: int, tp: int):
    """Per-rank swizzled SF buffers as the fused LN op / fp4_quantize leave them.

    Token-pad rows carry garbage and the 128-row tile padding is uninitialized: both are
    poisoned with 0xAB so a regroup that reads them fails the comparison.
    """
    sf_cols = lin_real.shape[-1]
    p0 = TokenShardPlan.build(batch, seq, tp, 0)
    lin_pad = torch.full((batch, p0.padded_seq_len, sf_cols), 0xAB, dtype=torch.uint8)
    lin_pad[:, :seq] = lin_real.view(batch, seq, sf_cols)
    lin_pad = lin_pad.view(batch * p0.padded_seq_len, sf_cols)
    bufs = []
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank)
        bufs.append(swizzle_ref(lin_pad[rank_slice(p)], pad_value=0xAB))
    return torch.cat(bufs), p0


# (B, S, tp, sf_cols)
_SF_CASES = [
    (1, 300, 1, 8),  # tp = 1
    (2, 256, 2, 320),  # m = 256: fast path
    (2, 300, 2, 320),  # m = 300: regroup
    (1, 75600 // 8, 8, 12),  # m % 128 != 0
    (2, 200, 4, 5),  # sf_cols % 4 != 0 (K = 80)
    (2, 105, 3, 5),  # straddling rank, m = 70
    (1, 509, 4, 8),  # padded B = 1, m = 128: zero-copy prefix
    (1, 1001, 8, 8),  # padded B = 1, m % 128 != 0
    (1, 1000, 8, 8),  # unpadded B = 1, m % 128 != 0
    (2, 105, 4, 8),  # padded B = 2: interior drop
    (2, 255, 4, 8),  # padded B = 2 with m = 128: must not take the zero-copy path
    (3, 255, 6, 5),  # padded B = 3 with m = 128 (straddling ranks)
    (2, 511, 4, 20),  # padded B = 2 with m = 256
    (3, 7, 4, 5),  # padded B = 3
    (1, 5, 4, 4),  # a fully padded rank
    (4, 131, 8, 20),
]


@pytest.mark.parametrize("batch,seq,tp,sf_cols", _SF_CASES)
def test_regroup_swizzled_sf(batch, seq, tp, sf_cols):
    gen = torch.Generator().manual_seed(batch * seq + tp)
    lin = torch.randint(0, 255, (batch * seq, sf_cols), dtype=torch.uint8, generator=gen)
    sf_cat, plan = _shard_sf_buffers(lin, batch, seq, tp)
    got = regroup_swizzled_sf(sf_cat, plan, sf_cols)
    assert got.numel() == swizzled_sf_numel(batch * seq, sf_cols)
    assert torch.equal(unswizzle_ref(got, batch * seq, sf_cols), lin)
    fast = plan.local_rows % 128 == 0 and (not plan.is_padded or batch == 1)
    assert (got.data_ptr() == sf_cat.data_ptr()) == fast


def test_regroup_zero_copy_for_padded_single_sample():
    lin = torch.randint(0, 255, (509, 8), dtype=torch.uint8)
    sf_cat, plan = _shard_sf_buffers(lin, 1, 509, 4)
    assert plan.is_padded and plan.local_rows == 128
    got = regroup_swizzled_sf(sf_cat, plan, 8)
    assert got.data_ptr() == sf_cat.data_ptr() and got.is_contiguous()


# =============================================================================
# A6. Capability gate and from_model_config validation
# =============================================================================


class _PlainDiT(BaseDiffusionModel):
    pass


class _SpTpDiT(BaseDiffusionModel):
    _supports_token_sharded_tp = True


def test_capability_gate():
    cfg = DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_layout="token_sharded"))
    with pytest.raises(ValueError, match="not implemented for _PlainDiT"):
        _PlainDiT(cfg)
    _SpTpDiT(cfg)
    for layout in (None, "replicated"):
        _PlainDiT(DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_layout=layout)))


def test_wan_opts_in():
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    assert WanTransformer3DModel._supports_token_sharded_tp is True
    assert BaseDiffusionModel._supports_token_sharded_tp is False


def _stub_config(flag, tp_size=2, ulysses=1, ring=1, attn2d=(1, 1), cache_backend=None):
    vgm = SimpleNamespace(
        tp_size=tp_size,
        seq_size=ulysses * ring * attn2d[0] * attn2d[1],
        ulysses_size=ulysses,
        ring_size=ring,
        attn2d_row_size=attn2d[0],
        attn2d_col_size=attn2d[1],
    )
    return SimpleNamespace(
        parallel=SimpleNamespace(token_sharded_tp=flag),
        visual_gen_mapping=vgm,
        cache_backend=cache_backend,
    )


def test_from_model_config_validation():
    for flag in (None, False):
        assert TokenShardedTP.from_model_config(_stub_config(flag)) is None
    assert TokenShardedTP.from_model_config(SimpleNamespace()) is None
    with pytest.raises(ValueError, match=r"tp_size > 1 and seq_size == 1 \(got tp_size=1,"):
        TokenShardedTP.from_model_config(_stub_config(True, tp_size=1))
    no_vgm = _stub_config(True)
    no_vgm.visual_gen_mapping = None
    with pytest.raises(ValueError, match="needs a VisualGenMapping"):
        TokenShardedTP.from_model_config(no_vgm)
    for kwargs in (dict(ulysses=2), dict(ring=2), dict(attn2d=(2, 1))):
        with pytest.raises(ValueError, match=r"seq_size == 1 \(got tp_size=2, seq_size=2"):
            TokenShardedTP.from_model_config(_stub_config(True, **kwargs))
    with pytest.raises(ValueError, match="does not support cache_backend='cache_dit'"):
        TokenShardedTP.from_model_config(_stub_config(True, cache_backend="cache_dit"))


def test_helper_requires_a_group():
    with pytest.raises(ValueError, match="needs a torch.distributed TP process group; got None"):
        TokenShardedTP(None)


def test_plan_before_begin_raises():
    sp = TokenShardedTP.__new__(TokenShardedTP)
    sp._plan = None
    with pytest.raises(RuntimeError, match=r"begin\(batch_size, seq_len\) must be called"):
        _ = sp.plan


def test_shape_errors():
    """Wrongly shaped inputs raise before any collective, naming the expected shape."""
    sp = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))  # m = 8, unpadded
    with pytest.raises(ValueError, match=r"shard: expected a \[B=2, S=8, \.\.\.\] tensor"):
        sp.shard(torch.zeros(2, 9, 4))
    with pytest.raises(ValueError, match=r"shard: expected a \[B=2, S=8"):
        sp.shard(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"reduce_scatter: expected 16 rows for the current"):
        sp.reduce_scatter(torch.zeros(15, 4))
    # Forgetting shard(): the full [B * S, D] or [B, S, D] is not this rank's [m, D].
    with pytest.raises(ValueError, match=r"all_gather: expected this rank's \[8, K\] rows"):
        sp.all_gather(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"unshard: expected this rank's \[8, K\] rows"):
        sp.unshard(torch.zeros(2, 8, 4))
    with pytest.raises(ValueError, match="payload must be uint8"):
        sp.all_gather(Fp4QuantizedTensor(torch.zeros(16, 2, dtype=torch.uint8), torch.zeros(512)))
    with pytest.raises(ValueError, match=r"pad_row_input: expected a \[B=2, S=8, K\] or"):
        sp.pad_row_input(torch.zeros(2 * 9, 4))
    sp = simulated_helper(TokenShardPlan.build(1, 7, 2, 0))  # padded: S_pad = 8
    with pytest.raises(ValueError, match=r"expected 7 \(B \* S\) or 8 \(B \* S_pad\) rows"):
        sp.reduce_scatter(torch.zeros(6, 4))
    with pytest.raises(ValueError, match=r"pad_row_input: expected"):
        sp.pad_row_input(torch.zeros(8, 4))  # the padded stream is not accepted


def test_fp4_gather_rejects_side_car_and_dynamic_scale():
    sp = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))
    payload = torch.zeros(8, 64, dtype=torch.uint8)
    sf = torch.zeros(swizzled_sf_numel(8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="per-rank dynamic scale"):
        sp.all_gather(Fp4QuantizedTensor(payload, sf, reciprocal_scale=torch.ones(1)))
    with pytest.raises(ValueError, match="unquantized_hidden_states side-car"):
        sp.all_gather(
            Fp4QuantizedTensor(payload, sf, unquantized_hidden_states=torch.zeros(8, 128))
        )


@pytest.mark.parametrize("batch,seq,tp", list(itertools.product([1, 2, 3], [5, 8], [2, 4])))
def test_padding_roundtrip(batch, seq, tp):
    """_add_padding then _drop_padding is the identity on [B * S, K]."""
    sp = simulated_helper(TokenShardPlan.build(batch, seq, tp, 0))
    t = torch.randn(batch * seq, 3)
    padded = sp._add_padding(t)
    assert padded.shape[0] == batch * sp.plan.padded_seq_len
    assert torch.equal(sp._drop_padding(padded), t)
    assert torch.equal(sp._add_padding(t.view(batch, seq, 3)), padded)


# =============================================================================
# A7. A Wan block rejects a modulation table that does not match its hidden states
# =============================================================================


def test_wan_block_rejects_a_global_table_on_a_shard():
    """Under token-sharded TP a block sees [n, g, D] sample groups; a global [B, 6, D]
    table (instead of the sharder's shard_per_sample) must raise, not mix samples."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanBlock

    cfg = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=4, attention_head_dim=16, ffn_dim=128, eps=1e-6
        )
    )
    block = WanBlock(cfg, 0)
    x = torch.zeros(1, 8, 64)  # one sample group of 8 rows (CFG B=2 at TP2)
    rope = (torch.zeros(8, 16), torch.zeros(8, 16))
    with pytest.raises(ValueError, match="modulation table"):
        block(x, torch.zeros(2, 4, 64), torch.zeros(2, 6, 64), *rope)  # global per-sample
    with pytest.raises(ValueError, match="modulation table"):
        block(x, torch.zeros(2, 4, 64), torch.zeros(1, 16, 6, 64), *rope)  # wrong per-token
