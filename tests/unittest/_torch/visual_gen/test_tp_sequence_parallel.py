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
"""CPU tests for the TP sequence-parallel helper (no process group, no GPU).

Plans are simulated per rank: every rank's shard of a full tensor is checked against
the same operation on the full tensor, sliced by that rank's rows.
"""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tp_sequence_parallel_test_utils import (
    padded_rows,
    simulated_helper,
    swizzle_ref,
    unswizzle_ref,
)

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.tp_sequence_parallel import (
    RowNorm,
    TokenShardPlan,
    TPSequenceParallel,
    apply_residual,
    apply_row_norm,
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
# A2. Row-local ops: shard result == full-tensor result sliced by rows
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
        # shard_rows == the global padded stream sliced (zeros on pad rows).
        assert torch.equal(sp.shard_rows(per_token), padded_rows(per_token, p)[rows])
        assert torch.equal(sp.shard(x), padded_rows(x, p)[rows])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,seq,tp", _ROW_LOCAL_CASES)
def test_residual_matches_full_rows(batch, seq, tp, dtype):
    x, y, table, per_token = _row_local_inputs(batch, seq, dtype)
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank)
        sp = simulated_helper(p)
        rows = rank_slice(p)
        x_full, y_full = padded_rows(x, p), padded_rows(y, p)
        x_loc, y_loc = sp.shard(x), sp.shard(y)
        # Ungated: plain add.
        assert torch.equal(sp.residual(x_loc, y_loc), (x_full + y_full)[rows])
        # Per-sample gate: full tensor with the global [B, D] table (g = S_pad).
        gate = table[:, 2]
        ref = apply_residual(x_full, y_full, gate)[rows]
        assert torch.equal(sp.residual(x_loc, y_loc, sp.per_sample_table(gate)), ref)
        # Per-token gate (g = 1).
        gate_tok = per_token[:, :, 2]
        ref = apply_residual(x_full, y_full, padded_rows(gate_tok, p))[rows]
        assert torch.equal(sp.residual(x_loc, y_loc, sp.shard_rows(gate_tok)), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,seq,tp", _ROW_LOCAL_CASES)
def test_row_norm_matches_full_rows(batch, seq, tp, dtype):
    x, _, table, per_token = _row_local_inputs(batch, seq, dtype)
    gen = torch.Generator().manual_seed(7)
    weight, bias = torch.randn(_D, generator=gen), torch.randn(_D, generator=gen)
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank)
        sp = simulated_helper(p)
        rows = rank_slice(p)
        x_full, x_loc = padded_rows(x, p), sp.shard(x)
        # AdaLN with a per-sample table.
        full = RowNorm(scale=table[:, 1], shift=table[:, 0])
        loc = RowNorm(
            scale=sp.per_sample_table(table[:, 1]), shift=sp.per_sample_table(table[:, 0])
        )
        assert torch.equal(sp.norm(x_loc, loc), apply_row_norm(x_full, full)[rows])
        # AdaLN with a per-token table.
        full = RowNorm(
            scale=padded_rows(per_token[:, :, 1], p), shift=padded_rows(per_token[:, :, 0], p)
        )
        loc = RowNorm(
            scale=sp.shard_rows(per_token[:, :, 1]), shift=sp.shard_rows(per_token[:, :, 0])
        )
        assert torch.equal(sp.norm(x_loc, loc), apply_row_norm(x_full, full)[rows])
        # Affine.
        spec = RowNorm(eps=1e-5, weight=weight, bias=bias)
        assert torch.equal(sp.norm(x_loc, spec), apply_row_norm(x_full, spec)[rows])
        # Identity.
        assert torch.equal(sp.norm(x_loc, RowNorm(identity=True)), x_full[rows])


def test_row_norm_matches_wan_eager_math():
    """Eager apply_row_norm == Wan's eager norm1/norm2/norm3 expressions.

    Wan: ``norm1(x.float()) * (1 + scale) + shift`` and ``norm2(x.float()).to(x.dtype)``,
    where ``LayerNorm.forward`` is fp32 layer_norm with fp32 weight/bias (weight=1 and
    bias=0 buffers when has_weights/has_bias is False). That module is maybe_compile'd,
    so bitwise equality is not guaranteed; compare with a tight tolerance.
    """
    gen = torch.Generator().manual_seed(0)
    b, s, eps = 2, 40, 1e-6
    x = torch.randn(b, s, _D, generator=gen).to(torch.bfloat16)
    temb = torch.randn(b, 6, _D, generator=gen)
    shift, scale = temb[:, 0:1], temb[:, 1:2]  # [B, 1, D] like chunk(6, dim=1)
    ones, zeros = torch.ones(_D), torch.zeros(_D)
    ref = F.layer_norm(x.float(), (_D,), ones, zeros, eps) * (1 + scale) + shift
    ref = ref.to(x.dtype).reshape(b * s, _D)
    got = apply_row_norm(
        x.reshape(b * s, _D), RowNorm(eps=eps, scale=scale.squeeze(1), shift=shift.squeeze(1))
    )
    torch.testing.assert_close(got.float(), ref.float(), rtol=1e-6, atol=1e-6)

    weight, bias = torch.randn(_D, generator=gen), torch.randn(_D, generator=gen)
    ref = F.layer_norm(x.float(), (_D,), weight, bias, eps).to(x.dtype).reshape(b * s, _D)
    got = apply_row_norm(x.reshape(b * s, _D), RowNorm(eps=eps, weight=weight, bias=bias))
    torch.testing.assert_close(got.float(), ref.float(), rtol=1e-6, atol=1e-6)


def test_row_local_ops_reject_mismatched_table():
    x = torch.randn(10, _D)
    with pytest.raises(ValueError, match="does not divide"):
        apply_residual(x, x, torch.randn(3, _D))
    with pytest.raises(ValueError, match="does not divide"):
        apply_row_norm(x, RowNorm(scale=torch.randn(4, _D), shift=torch.randn(4, _D)))


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
# A5. RowNorm validation
# =============================================================================


def test_row_norm_validation():
    t = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="scale and shift"):
        RowNorm(scale=t)
    with pytest.raises(ValueError, match="scale and shift"):
        RowNorm(shift=t)
    with pytest.raises(ValueError, match="weight and bias"):
        RowNorm(weight=t[0])
    with pytest.raises(ValueError, match="identity=True excludes"):
        RowNorm(identity=True, weight=t[0], bias=t[0])
    with pytest.raises(ValueError, match="identity=True excludes"):
        RowNorm(identity=True, scale=t, shift=t)
    with pytest.raises(ValueError, match="identity=True excludes"):
        RowNorm(identity=True, module=torch.nn.Identity())
    RowNorm(identity=True, quant_scale=torch.ones(1))  # identity + quantize is allowed


def test_row_norm_module_is_used_on_eager_path():
    """RowNorm.module replaces F.layer_norm on the eager path (same kernel as the model)."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(12, _D, generator=gen).to(torch.bfloat16)
    scale, shift = torch.randn(2, _D, generator=gen), torch.randn(2, _D, generator=gen)
    calls = []

    class Spy(torch.nn.Module):
        def forward(self, h):
            calls.append(h.dtype)
            return F.layer_norm(h, (_D,), None, None, 1e-6)

    got = apply_row_norm(x, RowNorm(scale=scale, shift=shift, module=Spy()))
    assert calls == [torch.float32]
    assert torch.equal(got, apply_row_norm(x, RowNorm(scale=scale, shift=shift)))


# =============================================================================
# A6. Capability gate and from_model_config validation
# =============================================================================


class _PlainDiT(BaseDiffusionModel):
    pass


class _SpTpDiT(BaseDiffusionModel):
    _supports_tp_sequence_parallel = True


def test_capability_gate():
    cfg = DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_sequence_parallel=True))
    with pytest.raises(ValueError, match="not implemented for _PlainDiT"):
        _PlainDiT(cfg)
    _SpTpDiT(cfg)
    for flag in (None, False):
        _PlainDiT(
            DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_sequence_parallel=flag))
        )


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
        parallel=SimpleNamespace(tp_sequence_parallel=flag),
        visual_gen_mapping=vgm,
        cache_backend=cache_backend,
    )


def test_from_model_config_validation():
    for flag in (None, False):
        assert TPSequenceParallel.from_model_config(_stub_config(flag)) is None
    assert TPSequenceParallel.from_model_config(SimpleNamespace()) is None
    with pytest.raises(ValueError, match=r"tp_size > 1 and seq_size == 1 \(got tp_size=1,"):
        TPSequenceParallel.from_model_config(_stub_config(True, tp_size=1))
    no_vgm = _stub_config(True)
    no_vgm.visual_gen_mapping = None
    with pytest.raises(ValueError, match="needs a VisualGenMapping"):
        TPSequenceParallel.from_model_config(no_vgm)
    for kwargs in (dict(ulysses=2), dict(ring=2), dict(attn2d=(2, 1))):
        with pytest.raises(ValueError, match=r"seq_size == 1 \(got tp_size=2, seq_size=2"):
            TPSequenceParallel.from_model_config(_stub_config(True, **kwargs))
    with pytest.raises(ValueError, match="does not support cache_backend='cache_dit'"):
        TPSequenceParallel.from_model_config(_stub_config(True, cache_backend="cache_dit"))


def test_helper_requires_a_group():
    with pytest.raises(ValueError, match="needs a torch.distributed TP process group; got None"):
        TPSequenceParallel(None)


def test_plan_before_begin_raises():
    sp = TPSequenceParallel.__new__(TPSequenceParallel)
    sp._plan = None
    with pytest.raises(RuntimeError, match=r"begin\(batch_size, seq_len\) must be called"):
        _ = sp.plan


def test_shape_errors():
    """Wrongly shaped inputs raise before any collective, naming the expected shape."""
    sp = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))  # m = 8, unpadded
    with pytest.raises(ValueError, match=r"shard: expected a \[B=2, S=8, \.\.\.\] tensor"):
        sp.shard(torch.zeros(2, 9, 4))
    with pytest.raises(ValueError, match=r"shard_rows: expected a \[B=2, S=8"):
        sp.shard_rows(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"reduce_scatter: expected 16 rows for the current"):
        sp.reduce_scatter(torch.zeros(15, 4))
    # Forgetting shard(): the full [B * S, D] or [B, S, D] is not this rank's [m, D].
    with pytest.raises(ValueError, match=r"all_gather: expected this rank's \[8, K\] rows"):
        sp.all_gather(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"unshard: expected this rank's \[8, K\] rows"):
        sp.unshard(torch.zeros(2, 8, 4))
    with pytest.raises(ValueError, match="payload must be uint8"):
        sp.all_gather(Fp4QuantizedTensor(torch.zeros(16, 2, dtype=torch.uint8), torch.zeros(512)))
    with pytest.raises(ValueError, match=r"row_linear: expected a \[B=2, S=8, K\] or"):
        sp.row_linear(lambda a: a, torch.zeros(2 * 9, 4))
    sp = simulated_helper(TokenShardPlan.build(1, 7, 2, 0))  # padded: S_pad = 8
    with pytest.raises(ValueError, match=r"expected 7 \(B \* S\) or 8 \(B \* S_pad\) rows"):
        sp.reduce_scatter(torch.zeros(6, 4))
    with pytest.raises(ValueError, match=r"row_linear: expected"):
        sp.row_linear(lambda a: a, torch.zeros(8, 4))  # the padded stream is not accepted


@pytest.mark.parametrize("batch,seq,tp", [(2, 16, 2), (2, 16, 4), (4, 8, 2)])
def test_global_table_rejected_on_shard(batch, seq, tp):
    """A global [B, D] table (the documented trap) raises instead of mixing samples."""
    x = torch.randn(batch, seq, _D)
    table = torch.randn(batch, _D)
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank)
        assert len(p.entry_batch) != batch  # the global table has the wrong entry count
        sp = simulated_helper(p)
        x_loc = sp.shard(x)
        with pytest.raises(ValueError, match="gate has 2 entries|gate has 4 entries"):
            sp.residual(x_loc, x_loc, table)
        with pytest.raises(ValueError, match="RowNorm.scale has .* per_sample_table"):
            sp.norm(x_loc, RowNorm(scale=table, shift=table))
        # The shard's own tables pass.
        sp.residual(x_loc, x_loc, sp.per_sample_table(table))
        sp.norm(x_loc, RowNorm(scale=sp.per_sample_table(table), shift=sp.per_sample_table(table)))


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
