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
"""CPU tests for the token-sharded TP helper (no process group, no GPU): plan invariants, the
NVFP4 scaling-factor regroup, the FP8 block-scale re-layout and engagement rule, the capability
gate, the helper's input checks, and the gather-mode and reduce-scatter-mode plumbing (the
copy-engine rules, their lifecycles against stand-in states, the traced part of the fused
reduce-scatter against a stand-in op, and the real probes' fallback on a gloo group). Ranks are
simulated from their plans.
"""

import contextlib
import itertools
import math
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from token_sharded_tp_test_utils import padded_rows, simulated_helper, swizzle_ref, unswizzle_ref

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.parallel import token_sharded_modules, token_sharded_tp
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import (
    TokenShardedColumn,
    TokenShardedMLP,
    TokenShardedRow,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    Fp8BlockScaledActivation,
    TokenShardedSequenceSharder,
    TokenShardedTP,
    TokenShardPlan,
    fp8_block_scale_prequant_ok,
    fp8_scale_cols,
    fp8_scales_mn_major,
    regroup_swizzled_sf,
    swizzled_sf_numel,
)
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import ParallelConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _exercise_ce_for_small_correctness_shapes(monkeypatch):
    monkeypatch.setattr(token_sharded_tp, "MIN_CE_REDUCE_SCATTER_ROWS", 0)


@pytest.mark.parametrize(
    "tokens,bias,expected",
    [
        (768, True, False),
        (32764, True, False),
        (32768, True, True),
        (65536, True, True),
        (32768, False, False),
        (65536, False, False),
    ],
)
def test_ce_reduce_scatter_small_m_fallback(tokens, bias, expected, monkeypatch):
    monkeypatch.setattr(token_sharded_tp, "MIN_CE_REDUCE_SCATTER_ROWS", 8192)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    consumer = _fp8_row_linear(bias=bias)
    x = torch.zeros(tokens, 256, dtype=torch.bfloat16)
    for rank in range(4):
        ts = simulated_helper(
            TokenShardPlan.build(1, tokens, 4, rank), reduce_scatter_mode="copy_engine"
        )
        ts._rs_state = SimpleNamespace(effective=True, reason="")
        assert ts.uses_ce_reduce_scatter(consumer, x) is expected


# =============================================================================
# Helpers
# =============================================================================


def rank_slice(plan: TokenShardPlan) -> slice:
    return slice(plan.row_start, plan.row_start + plan.local_rows)


# =============================================================================
# Plan invariants
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
    with pytest.raises(ValueError, match="row_align >= 1"):
        TokenShardPlan.build(1, 8, 2, 0, 0)


@pytest.mark.parametrize("row_align", [4, 64])
@pytest.mark.parametrize("tp", [2, 3, 4, 8])
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_plan_row_align(batch, tp, row_align):
    """Every rank's m is a multiple of row_align; the group / sample invariants still hold."""
    for seq in (1, 5, 7, 8, 12, 105, 418, 420, 4680, 75600):
        d = math.gcd(tp, batch)
        unit = (tp // d) * row_align
        covered = 0
        for rank in range(tp):
            p = TokenShardPlan.build(batch, seq, tp, rank, row_align)
            s_pad, m, g = p.padded_seq_len, p.local_rows, p.rows_per_entry
            assert p.row_align == row_align and m % row_align == 0
            assert tp * m == batch * s_pad
            assert s_pad % unit == 0 and 0 <= s_pad - seq < unit
            assert p.is_padded == (seq % unit != 0)
            assert len(p.entry_batch) == batch // d and m % g == 0 and p.row_start == rank * m
            rows = torch.arange(m)
            entry = torch.tensor(p.entry_batch)[rows // g]
            assert torch.equal(entry, (p.row_start + rows) // s_pad), (batch, seq, tp, rank)
            segs = p.local_segments()
            assert len(segs) <= batch + 1 and sum(s1 - s0 for _, s0, s1 in segs) == m
            covered += m
        assert covered == batch * pad_up(seq, unit)
    # The default alignment is today's plan, field for field.
    assert TokenShardPlan.build(batch, 418, tp, 0, 1) == TokenShardPlan.build(batch, 418, tp, 0)


# =============================================================================
# Scaling-factor layout
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


# =============================================================================
# FP8 block-scale layout: packed-scale re-layout and the gather without a group
# =============================================================================


def _int32(gen, *shape):
    return torch.randint(0, 2**31 - 1, shape, generator=gen).to(torch.int32)


def _mn_major_ref(rows: torch.Tensor) -> torch.Tensor:
    """Pure-Python reference of the MN-major storage ``[P, pad4(M)]`` of row-major ``[M, P]``."""
    m, cols = rows.shape
    buf = torch.full((cols, pad_up(m, 4)), -1, dtype=rows.dtype)
    for r in range(m):
        for c in range(cols):
            buf[c, r] = rows[r, c]
    return buf


def test_fp8_scale_cols():
    assert [fp8_scale_cols(k) for k in (128, 512, 640, 5120, 13824)] == [1, 1, 2, 10, 27]


@pytest.mark.parametrize("cols", [1, 2, 10, 27])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 53, 105, 209, 256])
def test_fp8_scales_mn_major(m, cols):
    rows = _int32(torch.Generator().manual_seed(m * 31 + cols), m, cols)
    got = fp8_scales_mn_major(rows)
    assert got.shape == (m, cols) and got.stride() == (1, pad_up(m, 4))
    assert torch.equal(got, rows)
    storage = got.as_strided((cols, pad_up(m, 4)), (pad_up(m, 4), 1))
    assert torch.equal(storage[:, :m], _mn_major_ref(rows)[:, :m])


def _fp8_shards(batch, seq, k, tp, gen):
    """A global FP8 payload + packed scales and every rank's shard of them, as the quantizer
    returns them (fp8 ``[m, K]``, MN-major int32 ``[m, P]``); token-pad rows are poisoned."""
    cols = fp8_scale_cols(k)
    payload = torch.randint(0, 256, (batch, seq, k), dtype=torch.uint8, generator=gen)
    scales = _int32(gen, batch, seq, cols)
    plans = [TokenShardPlan.build(batch, seq, tp, rank) for rank in range(tp)]
    payload_pad, scales_pad = padded_rows(payload, plans[0]), padded_rows(scales, plans[0])
    real = padded_rows(torch.ones(batch, seq, 1, dtype=torch.bool), plans[0])[:, 0]
    payload_pad[~real] = 0xAB
    scales_pad[~real] = -7
    shards = [
        Fp8BlockScaledActivation(
            payload_pad[rank_slice(p)].view(torch.float8_e4m3fn),
            fp8_scales_mn_major(scales_pad[rank_slice(p)]),
        )
        for p in plans
    ]
    return payload.reshape(batch * seq, k), scales.reshape(batch * seq, cols), plans, shards


def _simulated_gather(shards, rank):
    """Stand-in for the collective seam: what ``all_gather_single`` returns to ``rank``."""

    def gather(x_loc, group_name):
        assert group_name == "simulated"
        pieces = [
            a.fp8.view(torch.uint8) if x_loc.dtype == torch.uint8 else a.scale for a in shards
        ]
        assert torch.equal(x_loc, pieces[rank])  # this rank's own rows, contiguous
        return torch.cat([t.contiguous() for t in pieces])

    return gather


# (B, S, tp, K): unpadded, padded, straddling, m % 4 != 0 (209, 105, 53), P = 1 / 2 / 10 / 27.
_FP8_GATHER_CASES = [
    (1, 8, 2, 128),
    (1, 5, 2, 640),
    (2, 5, 3, 640),
    (3, 7, 4, 5120),
    (2, 9, 4, 13824),
    (1, 418, 2, 640),
    (1, 420, 4, 640),
    (1, 420, 8, 128),
    (2, 256, 2, 640),
]


@pytest.mark.parametrize("batch,seq,tp,k", _FP8_GATHER_CASES)
def test_fp8_all_gather_simulated(batch, seq, tp, k, monkeypatch):
    """all_gather of an FP8 block-scale pair, with the collective simulated: every rank gets
    the global payload and the global scales (per-sample padding dropped) in the MN-major
    layout DeepGEMM reads, checked against a pure-Python per-row reference."""
    gen = torch.Generator().manual_seed(batch * seq + tp + k)
    payload, scales, plans, shards = _fp8_shards(batch, seq, k, tp, gen)
    for rank, plan in enumerate(plans):
        monkeypatch.setattr(token_sharded_tp, "_all_gather_rows", _simulated_gather(shards, rank))
        ts = simulated_helper(plan)
        n, g = len(plan.entry_batch), plan.rows_per_entry
        given = Fp8BlockScaledActivation(shards[rank].fp8.view(n, g, k), shards[rank].scale)
        for got in (ts.all_gather(shards[rank]), ts.gather_input(None, given)):
            assert isinstance(got, Fp8BlockScaledActivation)
            assert got.fp8.dtype == torch.float8_e4m3fn and got.fp8.shape == (batch * seq, k)
            assert torch.equal(got.fp8.view(torch.uint8), payload)
            assert got.scale.dtype == torch.int32 and torch.equal(got.scale, scales)
            assert got.scale.stride() == (1, pad_up(batch * seq, 4))
            expected = _mn_major_ref(scales)
            storage = got.scale.as_strided(expected.shape, (expected.shape[1], 1))
            assert torch.equal(storage[:, : batch * seq], expected[:, : batch * seq])


def test_fp8_all_gather_rejects_bad_pairs():
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))  # m = 8
    fp8 = torch.zeros(8, 256, dtype=torch.float8_e4m3fn)
    sf = torch.zeros(8, 1, dtype=torch.int32)
    with pytest.raises(ValueError, match="payload must be float8_e4m3fn"):
        ts.all_gather(Fp8BlockScaledActivation(fp8.view(torch.uint8), sf))
    with pytest.raises(ValueError, match=r"payload must be float8_e4m3fn \[8, K\]"):
        ts.all_gather(Fp8BlockScaledActivation(fp8[:-1], sf))
    with pytest.raises(ValueError, match=r"scales must be packed UE8M0 int32 \[8, 1\]"):
        ts.all_gather(Fp8BlockScaledActivation(fp8, sf.float()))
    with pytest.raises(ValueError, match=r"scales must be packed UE8M0 int32 \[8, 1\]"):
        ts.all_gather(Fp8BlockScaledActivation(fp8, torch.zeros(8, 2, dtype=torch.int32)))
    # A bare (fp8, scale) tuple is not typed as FP8 here (Linear takes NVFP4 pairs the same way).
    with pytest.raises(TypeError, match="all_gather: expected a tensor"):
        ts.all_gather((fp8, sf))
    with pytest.raises(TypeError, match="gather_input: expected a tensor"):
        ts.gather_input(None, (fp8, sf))


# =============================================================================
# The FP8 block-scale engagement rule and its consumer check
# =============================================================================


def _fp8_block_linear(k=256, n=128, **kwargs):
    return Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        **kwargs,
    )


def test_fp8_block_scale_prequant_rule(monkeypatch):
    bf16 = Linear(256, 128, bias=False, dtype=torch.bfloat16)
    fp8 = _fp8_block_linear()
    assert fp8_block_scale_prequant_ok(None) is False
    assert fp8_block_scale_prequant_ok(bf16) is False
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: False)  # not an SM100 GPU
    assert fp8_block_scale_prequant_ok(fp8) is False
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    assert fp8_block_scale_prequant_ok(fp8) is True
    assert fp8_block_scale_prequant_ok(bf16) is False
    assert fp8_block_scale_prequant_ok(_fp8_block_linear(disable_deep_gemm=True)) is False
    assert (
        fp8_block_scale_prequant_ok(_fp8_block_linear(use_cute_dsl_blockscaling_mm=True)) is False
    )
    assert fp8_block_scale_prequant_ok(_fp8_block_linear(k=192)) is False  # K % 128 != 0
    # A LoraLayer attached at construction (MLP does this unconditionally) is not an active
    # LoRA: the rule still engages; the adapters keep the input dense per call instead.
    with_lora = _fp8_block_linear()
    with_lora.lora = object()
    assert fp8_block_scale_prequant_ok(with_lora) is True

    # An FP8 pair must only reach a consumer the rule accepts (checked before any collective).
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))
    # prequantize=False (an active LoRA at call time) keeps a bf16 input bf16 on the FP8 path.
    monkeypatch.setattr(
        token_sharded_tp, "_all_gather_rows", lambda x, group_name: torch.cat([x, x])
    )
    dense = ts.gather_input(fp8, torch.zeros(8, 256, dtype=torch.bfloat16), prequantize=False)
    assert dense.dtype == torch.bfloat16 and dense.shape == (16, 256)
    act = Fp8BlockScaledActivation(
        torch.zeros(8, 256, dtype=torch.float8_e4m3fn), torch.zeros(8, 1, dtype=torch.int32)
    )
    # ... including a consumer the rule cannot read (not a Linear): no silent pass-through.
    for consumer in (bf16, _fp8_block_linear(disable_deep_gemm=True), object()):
        with pytest.raises(ValueError, match="not an FP8 block-scale Linear on the DeepGEMM path"):
            ts.gather_input(consumer, act)
    TokenShardedTP._check_fp8_consumer(fp8, act)
    TokenShardedTP._check_fp8_consumer(None, act)
    TokenShardedTP._check_fp8_consumer(bf16, torch.zeros(8, 256))


class _Stop(Exception):
    pass


def _mlp_adapter(base):
    """A converted MLP / GatedMLP whose gather_input records ``prequantize`` and stops."""
    cls = type(f"TokenSharded{base.__name__}", (TokenShardedMLP, base), {})  # as the converter
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    seen = []

    def gather_input(consumer, x, *, prequantize=True):
        seen.append(prequantize)
        raise _Stop

    module._token_sharded_tp = SimpleNamespace(
        gather_input=gather_input,
        uses_ce_gather=lambda consumer, x, lora_params: False,
        effective_reduce_scatter_mode="nccl",
        uses_ce_reduce_scatter=lambda consumer, x, lora_params: False,
    )
    module.up_proj = object()
    return module, seen


@pytest.mark.parametrize(
    "base, calls",
    [
        # MLP.forward(x, lora_params=None) takes its LoRA path when lora_params is not None.
        (
            MLP,
            [
                ((), {}, True),
                ((), {"lora_params": None}, True),
                ((), {"lora_params": {}}, False),
                ((), {"lora_params": {"k": 1}}, False),
                (({},), {}, False),
                ((None,), {}, True),
            ],
        ),
        # GatedMLP.forward(x, all_rank_num_tokens, final_all_reduce_params, lora_params) when
        # lora_params is truthy.
        (
            GatedMLP,
            [
                ((), {}, True),
                ((), {"lora_params": {}}, True),
                ((), {"lora_params": {"k": 1}}, False),
                ((None, None, {"k": 1}), {}, False),
                ((None, None, {}), {}, True),
            ],
        ),
    ],
)
def test_mlp_adapter_prequantizes_exactly_when_the_wrapped_forward_skips_lora(base, calls):
    """The FP8 pre-quantize is off for the calls the wrapped forward routes to its LoRA path."""
    module, seen = _mlp_adapter(base)
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    for args, kwargs, prequantize in calls:
        with pytest.raises(_Stop):
            module.forward(x, *args, **kwargs)
        assert seen[-1] is prequantize, (base.__name__, args, kwargs)


# =============================================================================
# Capability gate and helper construction
# =============================================================================


class _PlainDiT(BaseDiffusionModel):
    pass


class _TokenShardedDiT(BaseDiffusionModel):
    _supports_token_sharded_tp = True


def test_capability_gate():
    cfg = DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_layout="token_sharded"))
    with pytest.raises(ValueError, match="not implemented for _PlainDiT"):
        _PlainDiT(cfg)
    _TokenShardedDiT(cfg)
    for layout in (None, "replicated"):
        _PlainDiT(DiffusionModelConfig(parallel=ParallelConfig(tp_size=2, tp_layout=layout)))


def test_helper_requires_a_mapping_and_a_group():
    with pytest.raises(ValueError, match="needs a VisualGenMapping"):
        TokenShardedTP.from_model_config(SimpleNamespace(visual_gen_mapping=None))
    with pytest.raises(ValueError, match="needs a torch.distributed TP process group; got None"):
        TokenShardedTP(None)


def test_plan_before_begin_raises():
    ts = TokenShardedTP.__new__(TokenShardedTP)
    ts._plan = None
    with pytest.raises(RuntimeError, match=r"begin\(batch_size, seq_len\) must be called"):
        _ = ts.plan


def test_shape_errors():
    """Wrongly shaped inputs raise before any collective, naming the expected shape."""
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))  # m = 8, unpadded
    with pytest.raises(ValueError, match=r"shard: expected a \[B=2, S=8, \.\.\.\] tensor"):
        ts.shard(torch.zeros(2, 9, 4))
    with pytest.raises(ValueError, match=r"shard: expected a \[B=2, S=8"):
        ts.shard(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"reduce_scatter: expected 16 rows for the current"):
        ts.reduce_scatter(torch.zeros(15, 4))
    # Forgetting shard(): the full [B * S, D] or [B, S, D] is not this rank's [m, D].
    with pytest.raises(ValueError, match=r"all_gather: expected this rank's \[8, K\] rows"):
        ts.all_gather(torch.zeros(16, 4))
    with pytest.raises(ValueError, match=r"unshard: expected this rank's \[8, K\] rows"):
        ts.unshard(torch.zeros(2, 8, 4))
    with pytest.raises(ValueError, match="payload must be uint8"):
        ts.all_gather(Fp4QuantizedTensor(torch.zeros(16, 2, dtype=torch.uint8), torch.zeros(512)))
    with pytest.raises(ValueError, match=r"pad_row_input: expected a \[B=2, S=8, K\] or"):
        ts.pad_row_input(torch.zeros(2 * 9, 4))
    ts = simulated_helper(TokenShardPlan.build(1, 7, 2, 0))  # padded: S_pad = 8
    with pytest.raises(ValueError, match=r"expected 7 \(B \* S\) or 8 \(B \* S_pad\) rows"):
        ts.reduce_scatter(torch.zeros(6, 4))
    with pytest.raises(ValueError, match=r"pad_row_input: expected"):
        ts.pad_row_input(torch.zeros(8, 4))  # the padded stream is not accepted


def test_fp4_gather_rejects_bad_inputs():
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))
    payload = torch.zeros(8, 64, dtype=torch.uint8)
    sf = torch.zeros(swizzled_sf_numel(8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="per-rank dynamic scale"):
        ts.all_gather(Fp4QuantizedTensor(payload, sf, reciprocal_scale=torch.ones(1)))
    with pytest.raises(ValueError, match="unquantized_hidden_states side-car"):
        ts.all_gather(
            Fp4QuantizedTensor(payload, sf, unquantized_hidden_states=torch.zeros(8, 128))
        )
    with pytest.raises(ValueError, match="payload must be uint8"):
        ts.all_gather(Fp4QuantizedTensor(payload.float(), sf))
    with pytest.raises(ValueError, match="must be 128x4-swizzled"):
        ts.all_gather(Fp4QuantizedTensor(payload, sf[:-1]))
    with pytest.raises(ValueError, match="must be 128x4-swizzled"):
        ts.all_gather(Fp4QuantizedTensor(payload, sf, is_sf_swizzled=False))


@pytest.mark.parametrize("row_align", [1, 4])
@pytest.mark.parametrize("batch,seq,tp", list(itertools.product([1, 2, 3], [5, 8], [2, 4])))
def test_padding_roundtrip(batch, seq, tp, row_align):
    """_add_padding then _drop_padding is the identity on [B * S, K]; shard follows the plan."""
    ts = simulated_helper(TokenShardPlan.build(batch, seq, tp, 0, row_align))
    assert ts.plan.local_rows % row_align == 0
    t = torch.randn(batch * seq, 3)
    padded = ts._add_padding(t)
    assert padded.shape[0] == batch * ts.plan.padded_seq_len
    assert torch.equal(ts._drop_padding(padded), t)
    assert torch.equal(ts._add_padding(t.view(batch, seq, 3)), padded)
    x = t.view(batch, seq, 3)
    for rank in range(tp):
        p = TokenShardPlan.build(batch, seq, tp, rank, row_align)
        assert torch.equal(simulated_helper(p).shard(x), padded_rows(x, p)[rank_slice(p)])


# =============================================================================
# Gather modes: plumbing, the copy-engine rule, its lifecycle and the probe fallback
# =============================================================================

_FAKE_GROUP = SimpleNamespace(group_name="fake_tp_group")


def _fake_group_tp(monkeypatch, **kwargs):
    """A real TokenShardedTP on a stand-in 2-rank group (no torch.distributed init); only
    construction-time behavior (begin() would run a collective)."""
    monkeypatch.setattr(token_sharded_tp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(token_sharded_tp.dist, "get_rank", lambda group: 0)
    return TokenShardedTP(_FAKE_GROUP, **kwargs)


class _FakeCeState:
    """Stand-in for ``token_sharded_ce_gather.CeGatherState`` recording the helper's calls."""

    probe_ok = True

    def __init__(self, group, group_name, *, timeout_ms=600_000):
        self.group, self.group_name, self.timeout_ms = group, group_name, timeout_ms
        self.effective, self.reason = False, "not probed"
        self.ks: set[int] = set()
        self.calls: list[tuple] = []

    def note_consumer(self, k):
        self.ks.add(k)

    def prepare(self, plan):  # starts the forward's slot sequence itself (no begin_forward)
        self.calls.append(("prepare", plan.batch_size, plan.seq_len))
        self.effective = self.probe_ok
        self.reason = "" if self.probe_ok else "fake: no symmetric memory on this group"

    def close(self):
        self.calls.append(("close",))


def _fake_ce_module(monkeypatch, probe_ok):
    """Install a stand-in ``token_sharded_ce_gather`` (registry, state class, validator) and
    record the helper's warnings. Returns ``(module, warnings, validated consumers)``."""
    registry, warnings, validated = {}, [], []
    state_cls = type("FakeCeGatherState", (_FakeCeState,), {"probe_ok": probe_ok})

    def get_state(group_name):
        return registry[group_name]

    def release_state(group_name):  # as the real one: closes and unregisters
        registry.pop(group_name).close()

    mod = SimpleNamespace(
        CeGatherState=state_cls,
        get_state=get_state,
        register_state=lambda state: registry.setdefault(state.group_name, state),
        release_state=release_state,
        validate_fp8_block_consumer=validated.append,
        registry=registry,
    )
    monkeypatch.setattr(token_sharded_tp, "_ce_gather_module", lambda: mod)
    monkeypatch.setattr(token_sharded_tp.logger, "warning", lambda *msg: warnings.append(msg[0]))
    return mod, warnings, validated


def _as_tp(linear, mode, reduce_output=False):
    """Give a TP=1-built Linear the TP metadata it would have at tp_size=2."""
    linear.tp_size, linear.tp_mode, linear.reduce_output = 2, mode, reduce_output
    if reduce_output:
        linear.all_reduce = AllReduce.__new__(AllReduce)
        torch.nn.Module.__init__(linear.all_reduce)
    else:
        linear.all_reduce = None
    return linear


@contextlib.contextmanager
def _single_rank_gloo_group():
    """A one-process gloo group: the smallest group the probe can all-reduce its verdict over."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def test_gather_mode_plumbing(monkeypatch):
    """gather_mode defaults to the module constant (flippable before construction) and is
    validated; the NCCL mode never touches the copy-engine state."""
    assert token_sharded_tp.DEFAULT_GATHER_MODE == "nccl"
    assert token_sharded_tp.GATHER_MODES == ("nccl", "copy_engine")
    tp = _fake_group_tp(monkeypatch)
    assert tp.gather_mode == "nccl" and tp.effective_gather_mode == "nccl"
    tp = _fake_group_tp(monkeypatch, gather_mode="copy_engine")
    assert tp.gather_mode == "copy_engine"
    assert tp.effective_gather_mode == "nccl"  # not probed before the first begin()
    with pytest.raises(ValueError, match="gather_mode must be one of"):
        _fake_group_tp(monkeypatch, gather_mode="rdma")
    monkeypatch.setattr(token_sharded_tp, "DEFAULT_GATHER_MODE", "copy_engine")
    assert _fake_group_tp(monkeypatch).gather_mode == "copy_engine"
    assert _fake_group_tp(monkeypatch, gather_mode="nccl").gather_mode == "nccl"
    mod, warnings, _ = _fake_ce_module(monkeypatch, probe_ok=True)
    tp = _fake_group_tp(monkeypatch, gather_mode="nccl")
    tp.note_consumer(_fp8_block_linear())
    tp.close()
    assert tp._ce_consumers == [] and tp._ce_state is None and mod.registry == {}
    assert warnings == []


@pytest.mark.parametrize("gather_mode", ["nccl", "copy_engine"])
@pytest.mark.parametrize("reduce_scatter_mode", ["nccl", "copy_engine"])
@pytest.mark.parametrize("row_align", [1, 128])
def test_from_model_config_reads_parallel_controls(
    gather_mode, reduce_scatter_mode, row_align, monkeypatch
):
    monkeypatch.setattr(token_sharded_tp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(token_sharded_tp.dist, "get_rank", lambda group: 0)
    vgm = SimpleNamespace(tp_group_pg=_FAKE_GROUP, tp_rank=0)
    parallel = ParallelConfig(
        tp_size=2,
        tp_layout="token_sharded",
        gather_mode=gather_mode,
        reduce_scatter_mode=reduce_scatter_mode,
        row_align=row_align,
    )
    cfg = DiffusionModelConfig(parallel=parallel)
    cfg.visual_gen_mapping = vgm
    tp = TokenShardedTP.from_model_config(cfg)
    assert tp.gather_mode == gather_mode
    assert tp.reduce_scatter_mode == reduce_scatter_mode
    assert tp.row_align == row_align
    # Exercise begin()'s plan creation while keeping the collective/probes out of this CPU test.
    monkeypatch.setattr(tp, "_check_rank_agreement", lambda b, s: None)
    monkeypatch.setattr(tp, "_ce_begin", lambda p: None)
    monkeypatch.setattr(tp, "_rs_begin", lambda p: None)
    plan = tp.begin(1, 5)
    assert plan.row_align == row_align and plan.local_rows % row_align == 0


def test_uses_ce_gather_rule(monkeypatch):
    """The fused path takes an FP8 block-scale consumer's bf16 or FP8 input, with no active
    LoRA, once the probe accepted the mode; everything else stays on gather_input."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    fp8 = _as_tp(_fp8_block_linear(), TensorParallelMode.COLUMN)
    bf16 = _as_tp(Linear(256, 128, bias=False, dtype=torch.bfloat16), TensorParallelMode.COLUMN)
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0), gather_mode="copy_engine")
    x = torch.zeros(8, 256, dtype=torch.bfloat16)
    pair = Fp8BlockScaledActivation(
        torch.zeros(8, 256, dtype=torch.float8_e4m3fn), torch.zeros(8, 1, dtype=torch.int32)
    )
    assert not ts.uses_ce_gather(fp8, x)  # no state yet (begin() has not probed)
    ts._ce_state = SimpleNamespace(effective=False, reason="probe failed")
    assert ts.effective_gather_mode == "nccl" and not ts.uses_ce_gather(fp8, x)
    ts._ce_state = SimpleNamespace(effective=True, reason="")
    assert ts.effective_gather_mode == "copy_engine"
    assert ts.uses_ce_gather(fp8, x) and ts.uses_ce_gather(fp8, pair)
    assert ts.uses_ce_gather(fp8, x.view(1, 8, 256))  # [n, g, K] sample groups
    assert ts.uses_ce_gather(fp8, x, lora_params=None) and ts.uses_ce_gather(fp8, x, {})
    assert not ts.uses_ce_gather(fp8, x, lora_params={"adapter": 1})  # active LoRA
    assert not ts.uses_ce_gather(bf16, x)  # not an FP8 block-scale consumer
    assert not ts.uses_ce_gather(None, x)
    # a row projection (reduce-scatter side) never takes the gather path, FP8 or not
    assert not ts.uses_ce_gather(_as_tp(_fp8_block_linear(), TensorParallelMode.ROW), x)
    assert not ts.uses_ce_gather(_fp8_block_linear(), x)  # no TP metadata at all
    assert not ts.uses_ce_gather(_fp8_block_linear(disable_deep_gemm=True), x)
    assert not ts.uses_ce_gather(fp8, x.float())  # not bf16
    fp4 = Fp4QuantizedTensor(
        torch.zeros(8, 128, dtype=torch.uint8), torch.zeros(swizzled_sf_numel(8, 16))
    )
    assert not ts.uses_ce_gather(fp8, fp4)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: False)
    assert not ts.uses_ce_gather(fp8, x)  # the FP8 rule itself needs the SM100 family
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    # The NCCL mode never engages, whatever a state says.
    nccl = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))
    nccl._ce_state = SimpleNamespace(effective=True, reason="")
    assert nccl.effective_gather_mode == "nccl" and not nccl.uses_ce_gather(fp8, x)


def test_copy_engine_lifecycle_probe_fallback(monkeypatch):
    """A failed probe keeps the NCCL gather (the state warns, the helper stays quiet); the
    group's state is shared by its helpers, sized by every noted consumer whose K the fused
    op can take, and released by the first close()."""
    mod, warnings, validated = _fake_ce_module(monkeypatch, probe_ok=False)
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, gather_mode="copy_engine")
    consumer = _fp8_block_linear()
    ts.note_consumer(consumer)  # at conversion, before any state exists
    ts.note_consumer(Linear(192, 128, bias=False, dtype=torch.bfloat16))  # K % 128 != 0
    assert ts._ce_state is None and mod.registry == {}
    ts.begin(2, 8)
    state = mod.registry["simulated"]
    assert ts._ce_state is state and state.ks == {256}
    assert state.calls == [("prepare", 2, 8)]
    assert ts.effective_gather_mode == "nccl"
    assert not ts.uses_ce_gather(consumer, torch.zeros(8, 256, dtype=torch.bfloat16))
    assert validated == [] and warnings == []  # nothing to validate or say: the state warned
    ts.begin(2, 8)  # the next forward
    assert state.calls[1:] == [("prepare", 2, 8)]
    # A second helper of the same group (Wan2.2's second transformer) shares the state.
    other = simulated_helper(plan, gather_mode="copy_engine")
    other.note_consumer(_fp8_block_linear(k=512))
    other.begin(2, 8)
    assert other._ce_state is state and state.ks == {256, 512}
    other.note_consumer(_fp8_block_linear(k=640))  # noted after the state exists
    assert state.ks == {256, 512, 640}
    # close(): the first releases the group's state, the rest are no-ops, repeats are harmless.
    ts.close()
    assert mod.registry == {} and state.calls[-1] == ("close",) and ts._ce_state is None
    other.close()
    ts.close()
    assert state.calls.count(("close",)) == 1


def test_copy_engine_lifecycle_effective(monkeypatch):
    """An accepted probe validates the FP8 consumers once (not the others) and engages the
    rule; a rejected operand surfaces at begin()."""
    mod, warnings, validated = _fake_ce_module(monkeypatch, probe_ok=True)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, gather_mode="copy_engine")
    fp8 = _as_tp(_fp8_block_linear(), TensorParallelMode.COLUMN)
    bf16 = _as_tp(Linear(256, 128, bias=False, dtype=torch.bfloat16), TensorParallelMode.COLUMN)
    ts.note_consumer(fp8)
    ts.note_consumer(bf16)
    ts.begin(2, 8)
    assert ts.effective_gather_mode == "copy_engine" and warnings == []
    assert validated == [fp8]
    ts.begin(2, 8)
    assert validated == [fp8]  # once
    x = torch.zeros(8, 256, dtype=torch.bfloat16)
    assert ts.uses_ce_gather(fp8, x) and not ts.uses_ce_gather(bf16, x)
    ts.close()
    assert mod.registry == {}

    mod, _, _ = _fake_ce_module(monkeypatch, probe_ok=True)

    def reject(linear):
        raise ValueError("weight_scale is not the UE8M0 layout")

    mod.validate_fp8_block_consumer = reject
    ts = simulated_helper(plan, gather_mode="copy_engine")
    ts.note_consumer(_fp8_block_linear())
    with pytest.raises(ValueError, match="UE8M0"):
        ts.begin(2, 8)
    assert ts._ce_validated is False


def test_adapters_note_their_consumers():
    """prepare() hands the column projections to the helper (their K sizes the pool); the
    NCCL mode keeps the helper free of them."""
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, gather_mode="copy_engine")
    col = _as_tp(_fp8_block_linear(), TensorParallelMode.COLUMN)
    TokenShardedColumn.prepare(col, ts, "blocks.0.attn1.qkv_proj")
    mlp = MLP(
        hidden_size=256,
        intermediate_size=512,
        bias=True,
        dtype=torch.bfloat16,
        config=ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)),
    )
    _as_tp(mlp.up_proj, TensorParallelMode.COLUMN)
    _as_tp(mlp.down_proj, TensorParallelMode.ROW, reduce_output=True)
    TokenShardedMLP.prepare(mlp, ts, "blocks.0.ffn")
    assert ts._ce_consumers == [col, mlp.up_proj]
    nccl = simulated_helper(plan)
    TokenShardedColumn.prepare(col, nccl, "blocks.0.attn1.qkv_proj")
    assert nccl._ce_consumers == []


def test_copy_engine_probe_falls_back_on_gloo(monkeypatch):
    """The real probe on a CPU gloo group: no symmetric memory, so the NCCL gather is kept
    with one warning naming the reason (over repeated forwards and a second helper of the
    group), and close() releases the group's state."""
    warnings = []
    monkeypatch.setattr(token_sharded_tp.logger, "warning", lambda *msg: warnings.append(msg[0]))
    with _single_rank_gloo_group() as group:
        plan = TokenShardPlan.build(2, 8, 2, 0)
        ts = simulated_helper(plan, gather_mode="copy_engine")
        ts.group, ts.group_name = group, group.group_name
        ts.note_consumer(_fp8_block_linear())
        ts.begin(2, 8)
        assert ts.effective_gather_mode == "nccl"
        reason = ts._ce_state.reason
        assert reason and len(warnings) == 1 and reason in warnings[0]
        assert "keeping the NCCL path" in warnings[0]  # the protocol base's one fallback warning
        ts.begin(2, 8)
        other = simulated_helper(plan, gather_mode="copy_engine")
        other.group, other.group_name = group, group.group_name
        other.begin(2, 8)
        assert other._ce_state is ts._ce_state and len(warnings) == 1
        ts.close()
        other.close()
        ceg = token_sharded_tp._ce_gather_module()
        assert token_sharded_tp._registered_ce_state(ceg, group.group_name) is None


# =============================================================================
# Reduce-scatter modes: plumbing, the copy-engine rule, its lifecycle and the probe fallback
# =============================================================================


class _FakeRsState:
    """Stand-in for ``token_sharded_ce_reduce_scatter.CeReduceScatterState`` recording the
    helper's calls."""

    probe_ok = True

    def __init__(self, group, group_name, *, timeout_ms=600_000, per_peer_streams=True):
        self.group, self.group_name, self.timeout_ms = group, group_name, timeout_ms
        self.per_peer_streams = per_peer_streams
        self.effective, self.reason = False, "not probed"
        self.ks: set[int] = set()
        self.ns: set[int] = set()
        self.calls: list[tuple] = []

    def note_row_consumer(self, k_local, out_features=None):
        self.ks.add(k_local)
        if out_features is not None:
            self.ns.add(out_features)

    def prepare(self, plan):  # starts the forward's slot sequence itself (no begin_forward)
        self.calls.append(("prepare", plan.batch_size, plan.seq_len))
        self.effective = self.probe_ok
        self.reason = "" if self.probe_ok else "fake: no symmetric memory on this group"

    def close(self):
        self.calls.append(("close",))


def _fake_rs_module(monkeypatch, probe_ok, warnings=None):
    """Install a stand-in ``token_sharded_ce_reduce_scatter`` (registry, state class, validator)
    and record the helper's warnings (into ``warnings`` when given, so a test with both
    stand-in modules sees every warning in one list). Returns ``(module, warnings, validated
    consumers)``."""
    registry, validated = {}, []
    warnings = [] if warnings is None else warnings
    state_cls = type("FakeCeReduceScatterState", (_FakeRsState,), {"probe_ok": probe_ok})

    def get_rs_state(group_name):
        return registry[group_name]

    def release_rs_state(group_name):  # as the real one: closes and unregisters
        registry.pop(group_name).close()

    mod = SimpleNamespace(
        CeReduceScatterState=state_cls,
        get_rs_state=get_rs_state,
        register_rs_state=lambda state: registry.setdefault(state.group_name, state),
        release_rs_state=release_rs_state,
        validate_fp8_block_row_consumer=validated.append,
        registry=registry,
    )
    monkeypatch.setattr(token_sharded_tp, "_ce_rs_module", lambda: mod)
    monkeypatch.setattr(token_sharded_tp.logger, "warning", lambda *msg: warnings.append(msg[0]))
    return mod, warnings, validated


def _fp8_row_linear(k_local=256, n=128, bias=True, **kwargs):
    """An FP8 block-scale row-parallel Linear as a converted block carries it (tp_size=2,
    its all-reduce stopped)."""
    lin = Linear(
        k_local,
        n,
        bias=bias,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        **kwargs,
    )
    return _as_tp(lin, TensorParallelMode.ROW)


def test_reduce_scatter_mode_plumbing(monkeypatch):
    """reduce_scatter_mode defaults to the module constant (flippable before construction), is
    validated and independent of gather_mode; the NCCL mode never touches the state."""
    assert token_sharded_tp.DEFAULT_REDUCE_SCATTER_MODE == "nccl"
    assert token_sharded_tp.REDUCE_SCATTER_MODES == ("nccl", "copy_engine")
    tp = _fake_group_tp(monkeypatch)
    assert tp.reduce_scatter_mode == "nccl" and tp.effective_reduce_scatter_mode == "nccl"
    tp = _fake_group_tp(monkeypatch, reduce_scatter_mode="copy_engine")
    assert tp.reduce_scatter_mode == "copy_engine" and tp.gather_mode == "nccl"
    assert tp.effective_reduce_scatter_mode == "nccl"  # not probed before the first begin()
    tp = _fake_group_tp(monkeypatch, gather_mode="copy_engine")
    assert tp.reduce_scatter_mode == "nccl"  # the gather mode does not imply the RS mode
    with pytest.raises(ValueError, match="reduce_scatter_mode must be one of"):
        _fake_group_tp(monkeypatch, reduce_scatter_mode="rdma")
    monkeypatch.setattr(token_sharded_tp, "DEFAULT_REDUCE_SCATTER_MODE", "copy_engine")
    assert _fake_group_tp(monkeypatch).reduce_scatter_mode == "copy_engine"
    assert _fake_group_tp(monkeypatch, reduce_scatter_mode="nccl").reduce_scatter_mode == "nccl"
    mod, warnings, _ = _fake_rs_module(monkeypatch, probe_ok=True)
    tp = _fake_group_tp(monkeypatch, reduce_scatter_mode="nccl")
    tp.note_row_consumer(_fp8_row_linear())
    tp.close()
    assert tp._rs_consumers == [] and tp._rs_state is None and mod.registry == {}
    assert warnings == []


def test_uses_ce_reduce_scatter_rule(monkeypatch):
    """The fused path takes an FP8 block-scale row consumer's bf16 input, with no active LoRA,
    once the probe accepted the mode; everything else stays on the module's GEMM and
    reduce_scatter."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    fp8 = _fp8_row_linear()
    bf16 = _as_tp(Linear(256, 128, bias=False, dtype=torch.bfloat16), TensorParallelMode.ROW)
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0), reduce_scatter_mode="copy_engine")
    x = torch.zeros(16, 256, dtype=torch.bfloat16)  # all tokens, [B * S, K_local]
    assert not ts.uses_ce_reduce_scatter(fp8, x)  # no state yet (begin() has not probed)
    # A declined probe (as on a one-rank group, where there is nothing to reduce) keeps NCCL.
    ts._rs_state = SimpleNamespace(effective=False, reason="probe failed")
    assert ts.effective_reduce_scatter_mode == "nccl" and not ts.uses_ce_reduce_scatter(fp8, x)
    ts._rs_state = SimpleNamespace(effective=True, reason="")
    assert ts.effective_reduce_scatter_mode == "copy_engine"
    assert ts.uses_ce_reduce_scatter(fp8, x)
    assert ts.uses_ce_reduce_scatter(fp8, x.view(2, 8, 256))  # [B, S, K_local]
    assert ts.uses_ce_reduce_scatter(fp8, x, lora_params=None)
    assert ts.uses_ce_reduce_scatter(fp8, x, {})
    assert not ts.uses_ce_reduce_scatter(fp8, x, lora_params={"adapter": 1})  # active LoRA
    assert not ts.uses_ce_reduce_scatter(bf16, x)  # not an FP8 block-scale consumer
    assert not ts.uses_ce_reduce_scatter(None, x)
    assert not ts.uses_ce_reduce_scatter(_fp8_row_linear(disable_deep_gemm=True), x)
    assert not ts.uses_ce_reduce_scatter(_fp8_row_linear(use_cute_dsl_blockscaling_mm=True), x)
    assert not ts.uses_ce_reduce_scatter(_fp8_row_linear(k_local=192), x)  # K_local % 128 != 0
    assert not ts.uses_ce_reduce_scatter(fp8, x.float())  # not bf16
    # The fused reduce-scatter quantizes its input itself: pre-quantized inputs stay out.
    pair = Fp8BlockScaledActivation(
        torch.zeros(16, 256, dtype=torch.float8_e4m3fn), torch.zeros(16, 1, dtype=torch.int32)
    )
    assert not ts.uses_ce_reduce_scatter(fp8, pair)
    fp4 = Fp4QuantizedTensor(
        torch.zeros(16, 128, dtype=torch.uint8), torch.zeros(swizzled_sf_numel(16, 16))
    )
    assert not ts.uses_ce_reduce_scatter(fp8, fp4)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: False)
    assert not ts.uses_ce_reduce_scatter(fp8, x)  # the FP8 rule itself needs the SM100 family
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    # The gather's state says nothing about the reduce-scatter, and vice versa.
    ts._ce_state, ts._rs_state = SimpleNamespace(effective=True, reason=""), None
    assert not ts.uses_ce_reduce_scatter(fp8, x) and ts.effective_gather_mode == "nccl"
    # The NCCL mode never engages, whatever a state says.
    nccl = simulated_helper(TokenShardPlan.build(2, 8, 2, 0))
    nccl._rs_state = SimpleNamespace(effective=True, reason="")
    assert nccl.effective_reduce_scatter_mode == "nccl"
    assert not nccl.uses_ce_reduce_scatter(fp8, x)


def _fp8_block_mlp(cls=MLP, **kwargs):
    mlp = cls(
        hidden_size=256,
        intermediate_size=512,
        bias=True,
        dtype=torch.bfloat16,
        config=ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)),
        **kwargs,
    )
    up = mlp.up_proj if hasattr(mlp, "up_proj") else mlp.gate_up_proj
    _as_tp(up, TensorParallelMode.COLUMN)
    _as_tp(mlp.down_proj, TensorParallelMode.ROW, reduce_output=True)
    return mlp


def test_row_adapters_note_consumers():
    """prepare() hands the row projections the fused op can take to the helper (Row and
    MLP.down_proj; a GatedMLP's down_proj never engages -- GatedMLP.forward owns its call -- so
    it is not noted and does not size the pool); the NCCL mode keeps the helper free of them,
    and the column side is untouched."""
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
    row = _as_tp(_fp8_block_linear(), TensorParallelMode.ROW, reduce_output=True)
    TokenShardedRow.prepare(row, ts, "blocks.0.attn1.to_out.0")
    assert row.reduce_output is False and row.all_reduce is None  # the all-reduce is stopped
    mlp, gated = _fp8_block_mlp(), _fp8_block_mlp(GatedMLP)
    TokenShardedMLP.prepare(mlp, ts, "blocks.0.ffn")
    TokenShardedMLP.prepare(gated, ts, "blocks.0.gated_ffn")
    assert ts._rs_consumers == [row, mlp.down_proj]
    assert gated.down_proj.reduce_output is False  # converted (its all-reduce stopped) ...
    assert gated.down_proj not in ts._rs_consumers  # ... but never a fused-op candidate
    assert ts._ce_consumers == []  # the gather is in its NCCL mode
    nccl = simulated_helper(plan)
    TokenShardedRow.prepare(row, nccl, "blocks.0.attn1.to_out.0")
    TokenShardedMLP.prepare(mlp, nccl, "blocks.0.ffn")
    assert nccl._rs_consumers == [] and nccl._ce_consumers == []
    both = simulated_helper(plan, gather_mode="copy_engine", reduce_scatter_mode="copy_engine")
    TokenShardedMLP.prepare(mlp, both, "blocks.0.ffn")
    assert both._ce_consumers == [mlp.up_proj] and both._rs_consumers == [mlp.down_proj]


def test_ce_reduce_scatter_lifecycle_probe_fallback(monkeypatch):
    """A failed probe keeps the NCCL reduce-scatter (the state warns, the helper stays quiet);
    the group's state is shared by its helpers, told the K_local and N of every noted consumer
    the fused op can take (N sizes its pool), and released together with the gather's state by
    the first close()."""
    ce_mod, warnings, ce_validated = _fake_ce_module(monkeypatch, probe_ok=True)
    rs_mod, _, rs_validated = _fake_rs_module(monkeypatch, probe_ok=False, warnings=warnings)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, gather_mode="copy_engine", reduce_scatter_mode="copy_engine")
    col, row = _fp8_block_linear(k=512), _fp8_row_linear()
    ts.note_consumer(col)
    ts.note_row_consumer(row)  # at conversion, before any state exists
    ts.note_row_consumer(_as_tp(Linear(192, 128, dtype=torch.bfloat16), TensorParallelMode.ROW))
    assert ts._rs_state is None and rs_mod.registry == {}
    ts.begin(2, 8)
    state = rs_mod.registry["simulated"]
    assert ts._rs_state is state  # K_local % 128 != 0 is not noted:
    assert state.ks == {256} and state.ns == {128}
    assert state.calls == [("prepare", 2, 8)]
    assert ts.effective_reduce_scatter_mode == "nccl"
    assert not ts.uses_ce_reduce_scatter(row, torch.zeros(16, 256, dtype=torch.bfloat16))
    assert rs_validated == [] and warnings == []  # nothing to validate or say: the state warned
    assert ts.effective_gather_mode == "copy_engine" and ce_validated == [col]  # independent
    ts.begin(2, 8)  # the next forward
    assert state.calls[1:] == [("prepare", 2, 8)]
    # A second helper of the same group (Wan2.2's second transformer) shares the state.
    other = simulated_helper(plan, reduce_scatter_mode="copy_engine")
    other.note_row_consumer(_fp8_row_linear(k_local=512))
    other.begin(2, 8)
    assert other._rs_state is state and state.ks == {256, 512} and state.ns == {128}
    other.note_row_consumer(_fp8_row_linear(k_local=640, n=256))  # after the state exists
    assert state.ks == {256, 512, 640} and state.ns == {128, 256}
    # close(): the first releases both of the group's states, the rest are no-ops, repeats
    # are harmless.
    ts.close()
    assert rs_mod.registry == {} and ce_mod.registry == {}
    assert state.calls[-1] == ("close",) and ts._rs_state is None and ts._ce_state is None
    other.close()
    ts.close()
    assert state.calls.count(("close",)) == 1


def test_ce_reduce_scatter_lifecycle_effective(monkeypatch):
    """An accepted probe validates the FP8 row consumers once (not the others) and engages the
    rule; a rejected operand surfaces at begin()."""
    rs_mod, warnings, validated = _fake_rs_module(monkeypatch, probe_ok=True)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    plan = TokenShardPlan.build(2, 8, 2, 0)
    ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
    fp8 = _fp8_row_linear()
    bf16 = _as_tp(Linear(256, 64, bias=False, dtype=torch.bfloat16), TensorParallelMode.ROW)
    ts.note_row_consumer(fp8)
    ts.note_row_consumer(bf16)
    ts.begin(2, 8)
    assert ts.effective_reduce_scatter_mode == "copy_engine" and warnings == []
    assert validated == [fp8]
    # Only the FP8 block-scale row consumer's N sizes the pool; the bf16 one hands over its
    # K_local alone (it can never engage, so its N must not inflate the symmetric slot).
    state = rs_mod.registry["simulated"]
    assert state.ks == {256} and state.ns == {128}
    ts.begin(2, 8)
    assert validated == [fp8]  # once
    x = torch.zeros(16, 256, dtype=torch.bfloat16)
    assert ts.uses_ce_reduce_scatter(fp8, x) and not ts.uses_ce_reduce_scatter(bf16, x)
    ts.close()
    assert rs_mod.registry == {}

    rs_mod, _, _ = _fake_rs_module(monkeypatch, probe_ok=True)

    def reject(linear):
        raise ValueError("weight_scale is not the UE8M0 layout")

    rs_mod.validate_fp8_block_row_consumer = reject
    ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
    ts.note_row_consumer(_fp8_row_linear())
    with pytest.raises(ValueError, match="UE8M0"):
        ts.begin(2, 8)
    assert ts._rs_validated is False


@pytest.mark.parametrize("batch,seq,tp", [(2, 8, 2), (2, 5, 3), (1, 7, 4)])
def test_ce_gemm_reduce_scatter_traced_part(batch, seq, tp, monkeypatch):
    """The traced part of the fused reduce-scatter: the input is padded per sample and
    quantized once on all rows, the op gets the plan's ints and the group name, and the bias
    is passed on every rank so each destination applies it once to source 0's partial."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    quantized, op_calls = [], []

    def fake_quantize(h):  # the pinned quantizer, shape-only: fp8 [rows, K] + int32 [rows, P]
        quantized.append(h)
        rows, k = h.shape
        return Fp8BlockScaledActivation(
            torch.zeros(rows, k, dtype=torch.float8_e4m3fn),
            torch.zeros(rows, fp8_scale_cols(k), dtype=torch.int32),
        )

    def fake_op(*args):
        op_calls.append(args)
        m, n = args[8] * args[10] // args[7], args[2].shape[0]
        return torch.full((m, n), 7.0, dtype=torch.bfloat16)

    monkeypatch.setattr(token_sharded_tp, "quantize_fp8_block", fake_quantize)
    monkeypatch.setattr(
        torch.ops.trtllm, "token_sharded_fp8_ce_gemm_reduce_scatter", fake_op, raising=False
    )
    k_local, n = 256, 128
    consumer = _fp8_row_linear(k_local, n, bias=True)
    x = torch.randn(batch, seq, k_local, dtype=torch.bfloat16)
    for rank in range(tp):
        plan = TokenShardPlan.build(batch, seq, tp, rank)
        ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
        ts._rs_state = SimpleNamespace(effective=True, reason="")
        assert ts.uses_ce_reduce_scatter(consumer, x)
        out = ts.ce_gemm_reduce_scatter(consumer, x if rank % 2 else x.reshape(-1, k_local))
        assert out.shape == (plan.local_rows, n) and out.dtype == torch.bfloat16
        # The whole stream was padded per sample before the quantize (zero pad rows).
        h = quantized[-1]
        assert h.shape == (plan.padded_rows, k_local)
        assert torch.equal(h, padded_rows(x, plan))
        fp8, sf, w, w_sf, bias, group_name, *ints = op_calls[-1]
        assert fp8.shape == (plan.padded_rows, k_local) and fp8.dtype == torch.float8_e4m3fn
        assert sf.shape == (plan.padded_rows, fp8_scale_cols(k_local)) and sf.dtype == torch.int32
        assert w is consumer.weight and w_sf is consumer.weight_scale
        assert bias is consumer.bias  # every destination adds the bias once (source 0's block)
        assert group_name == "simulated"
        assert ints == [rank, tp, batch, seq, plan.padded_seq_len]
        # The adapter's view of the result: [n, g, N] sample groups.
        assert ts.local_view(out).shape == (len(plan.entry_batch), plan.rows_per_entry, n)
    assert len(op_calls) == tp
    # A Linear without a bias passes None on every rank.
    no_bias = _fp8_row_linear(k_local, n, bias=False)
    ts = simulated_helper(
        TokenShardPlan.build(batch, seq, tp, 0), reduce_scatter_mode="copy_engine"
    )
    ts._rs_state = SimpleNamespace(effective=True, reason="")
    assert not ts.uses_ce_reduce_scatter(no_bias, x)
    ts.ce_gemm_reduce_scatter(no_bias, x)
    assert op_calls[-1][4] is None
    # The padded stream is not accepted as the input (the op pads itself).
    with pytest.raises(ValueError, match="pad_row_input: expected"):
        ts.ce_gemm_reduce_scatter(consumer, torch.zeros(ts.plan.padded_rows + 1, k_local))


@pytest.mark.parametrize("gather_mode", ["nccl", "copy_engine"])
def test_mlp_adapter_fuses_the_reduce_scatter_independently_of_the_gather(gather_mode, monkeypatch):
    """TokenShardedMLP.forward takes the fused GEMM + reduce-scatter for an FP8 block-scale
    down_proj on the NCCL-gather path too (up_proj on the NCCL-gathered input, then
    MLP._act_from_up, then the fused op) and on the copy-engine-gather path; with the
    reduce-scatter in its NCCL mode the NCCL-gather path is MLP.forward whole plus
    reduce_scatter, and an active LoRA keeps it so in every mode."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    plan = TokenShardPlan.build(2, 8, 2, 0)
    calls = []

    def helper(reduce_scatter_mode):
        ts = simulated_helper(plan, gather_mode, reduce_scatter_mode)
        # Accepted probes; prepare() hands the consumers to the states, which take them.
        ts._ce_state = SimpleNamespace(effective=True, reason="", note_consumer=lambda k: None)
        ts._rs_state = SimpleNamespace(
            effective=True, reason="", note_row_consumer=lambda k, out_features=None: None
        )
        ts.gather_input = lambda consumer, x, prequantize=True: (
            calls.append(("gather_input", consumer, prequantize)) or ("gathered", x)
        )
        ts.ce_gather_gemm = lambda consumer, x: (
            calls.append(("ce_gather_gemm", consumer))
            or (torch.ones(16, 256, dtype=torch.bfloat16) * 3)
        )
        ts.ce_gemm_reduce_scatter = lambda consumer, act: (
            calls.append(("ce_gemm_reduce_scatter", consumer, act))
            or torch.zeros(plan.local_rows, 128, dtype=torch.bfloat16)
        )
        ts.reduce_scatter = lambda partial: (
            calls.append(("reduce_scatter", partial))
            or (torch.zeros(plan.local_rows, 128, dtype=torch.bfloat16))
        )
        return ts

    def convert(ts):
        # Hidden 256, intermediate 512 (256 per rank), bias, GELU (CPU-runnable activation).
        mlp = _fp8_block_mlp(activation=torch.nn.functional.gelu)
        token_sharded_modules._convert(mlp, "mlp", ts, "blocks.0.ffn")
        assert isinstance(mlp, TokenShardedMLP) and isinstance(mlp, MLP)
        # The projections' GEMMs cannot run on CPU: stand-ins keyed on the input they get.
        mlp.up_proj.forward = lambda x: (
            calls.append(("up_proj", x)) or (torch.ones(16, 256, dtype=torch.bfloat16) * 2)
        )
        mlp.down_proj.forward = lambda x: (
            calls.append(("down_proj", x)) or torch.zeros(16, 128, dtype=torch.bfloat16)
        )
        return mlp

    x = torch.randn(plan.local_rows, 256, dtype=torch.bfloat16)

    # Copy-engine reduce-scatter: the adapter owns down_proj whatever the gather mode.
    ts = helper("copy_engine")
    mlp = convert(ts)
    out = mlp(x)
    assert out.shape == (len(plan.entry_batch), plan.rows_per_entry, 128)
    names = [c[0] for c in calls]
    if gather_mode == "copy_engine":
        assert names == ["ce_gather_gemm", "ce_gemm_reduce_scatter"]
        up_out = torch.ones(16, 256, dtype=torch.bfloat16) * 3
    else:
        assert names == ["gather_input", "up_proj", "ce_gemm_reduce_scatter"]
        assert calls[0][1:] == (mlp.up_proj, True) and calls[1][1] == ("gathered", x)
        up_out = torch.ones(16, 256, dtype=torch.bfloat16) * 2
    fused = calls[-1]
    assert fused[1] is mlp.down_proj
    assert torch.equal(fused[2], mlp._act_from_up(up_out))  # the activation, bf16
    assert torch.equal(fused[2], mlp.activation(up_out))
    # An active LoRA keeps MLP.forward whole (its forward_lora) and the NCCL reduce-scatter.
    calls.clear()
    monkeypatch.setattr(
        MLP,
        "forward",
        lambda self, h, lora_params=None: (
            calls.append(("MLP.forward", h, lora_params))
            or torch.zeros(16, 128, dtype=torch.bfloat16)
        ),
    )
    mlp(x, {"adapter": 1})
    assert [c[0] for c in calls] == ["gather_input", "MLP.forward", "reduce_scatter"]
    assert calls[0][1:] == (mlp.up_proj, False) and calls[1][2] == {"adapter": 1}
    monkeypatch.undo()
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)

    # NCCL reduce-scatter on the NCCL gather: MLP.forward whole, byte for byte the old path.
    calls.clear()
    ts = helper("nccl")
    mlp = convert(ts)
    monkeypatch.setattr(
        MLP,
        "forward",
        lambda self, h, lora_params=None: (
            calls.append(("MLP.forward", h, lora_params))
            or torch.zeros(16, 128, dtype=torch.bfloat16)
        ),
    )
    mlp(x)
    if gather_mode == "copy_engine":
        assert [c[0] for c in calls] == ["ce_gather_gemm", "down_proj", "reduce_scatter"]
    else:
        assert [c[0] for c in calls] == ["gather_input", "MLP.forward", "reduce_scatter"]
        assert calls[1][1] == ("gathered", x) and calls[1][2] is None
    # Not both FP8 block-scale: MLP.forward keeps its dispatch (no adapter-owned down_proj).
    calls.clear()
    ts = helper("copy_engine")
    mlp = convert(ts)
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: False)  # the FP8 rule is off
    mlp(x)
    assert [c[0] for c in calls] == ["gather_input", "MLP.forward", "reduce_scatter"]


def test_needs_eager_warmup_covers_both_copy_engine_modes():
    """The pipeline's eager warm-up pass runs when either copy-engine mode is configured (both
    pools are sized by the forwards), not in the NCCL modes."""
    plan = TokenShardPlan.build(2, 8, 2, 0)
    for gather_mode, reduce_scatter_mode in itertools.product(("nccl", "copy_engine"), repeat=2):
        ts = simulated_helper(plan, gather_mode, reduce_scatter_mode)
        sharder = TokenShardedSequenceSharder(ts)
        expected = "copy_engine" in (gather_mode, reduce_scatter_mode)
        assert sharder.needs_eager_warmup is expected, (gather_mode, reduce_scatter_mode)


def test_ce_reduce_scatter_probe_falls_back_on_gloo(monkeypatch):
    """The real probe on a CPU gloo group: no symmetric memory, so the NCCL reduce-scatter is
    kept with one warning naming the reason (over repeated forwards and a second helper of
    the group), and close() releases the group's state."""
    warnings = []
    monkeypatch.setattr(token_sharded_tp.logger, "warning", lambda *msg: warnings.append(msg[0]))
    with _single_rank_gloo_group() as group:
        plan = TokenShardPlan.build(2, 8, 2, 0)
        ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
        ts.group, ts.group_name = group, group.group_name
        ts.note_row_consumer(_fp8_row_linear())
        ts.begin(2, 8)
        assert ts.effective_reduce_scatter_mode == "nccl"
        reason = ts._rs_state.reason
        assert reason and len(warnings) == 1 and reason in warnings[0]
        assert "reduce-scatter" in warnings[0]
        ts.begin(2, 8)
        other = simulated_helper(plan, reduce_scatter_mode="copy_engine")
        other.group, other.group_name = group, group.group_name
        other.begin(2, 8)
        assert other._rs_state is ts._rs_state and len(warnings) == 1
        ts.close()
        other.close()
        mod = token_sharded_tp._ce_rs_module()
        assert token_sharded_tp._registered_rs_state(mod, group.group_name) is None


def _shared_adapters(monkeypatch, mode="nccl", *, mixed=False):
    from tensorrt_llm._torch.visual_gen.parallel import token_sharded_shared_input as shared_mod

    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    ts = simulated_helper(TokenShardPlan.build(1, 8, 2, 0), gather_mode=mode)
    ts._ce_state = SimpleNamespace(
        effective=mode == "copy_engine", reason="", note_consumer=lambda k: None
    )
    calls = {"gather": [], "quantize": [], "multi": [], "linear": []}
    quant = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    col = _as_tp(
        Linear(
            128,
            128,
            bias=True,
            dtype=torch.bfloat16,
            quant_config=QuantConfig() if mixed else quant,
        ),
        TensorParallelMode.COLUMN,
    )
    mlp = MLP(
        hidden_size=128,
        intermediate_size=256,
        bias=True,
        activation=torch.nn.functional.gelu,
        dtype=torch.bfloat16,
        config=ModelConfig(quant_config=quant),
    )
    gated = GatedMLP(
        hidden_size=128,
        intermediate_size=256,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(quant_config=quant),
    )
    gated._can_fuse_swiglu_fp8_quant = lambda: False
    gated._apply_activation = (
        lambda h: torch.nn.functional.silu(h.chunk(2, -1)[0]) * h.chunk(2, -1)[1]
    )
    for module in (mlp, gated):
        up = module.up_proj if isinstance(module, MLP) else module.gate_up_proj
        _as_tp(up, TensorParallelMode.COLUMN)
        _as_tp(module.down_proj, TensorParallelMode.ROW)
    for name, module, kind in (
        ("qkv", col, "column"),
        ("mlp", mlp, "mlp"),
        ("gated", gated, "mlp"),
    ):
        token_sharded_modules._convert(module, kind, ts, name)
    for module in (col, mlp, gated):
        for linear in module.modules():
            if isinstance(linear, Linear):
                linear.weight.data.copy_(
                    torch.full(linear.weight.shape, 0.0625).to(linear.weight.dtype)
                )
                if linear.bias is not None:
                    linear.bias.data.fill_(0.25)

    def quantize(x):
        calls["quantize"].append(x)
        x = x.reshape(-1, 128)
        return Fp8BlockScaledActivation(
            x.to(torch.float8_e4m3fn),
            torch.zeros(x.shape[0], fp8_scale_cols(128), dtype=torch.int32),
        )

    def gather(value):
        calls["gather"].append(value)
        if isinstance(value, Fp8BlockScaledActivation):
            return Fp8BlockScaledActivation(
                torch.cat([value.fp8, value.fp8]), torch.cat([value.scale, value.scale])
            )
        return torch.cat([value, value])

    def linear_forward(module, x, *args, **kwargs):
        calls["linear"].append((module, x, args, kwargs))
        x = x.fp8 if isinstance(x, Fp8BlockScaledActivation) else x
        result = x.to(torch.bfloat16) @ module.weight.to(torch.bfloat16).t()
        return result + module.bias if module.bias is not None else result

    def multi(x, sf, weights, scales, *plan):
        calls["multi"].append((weights, scales, plan))
        full = torch.cat([x, x]).to(torch.bfloat16)
        return [full @ weight.to(torch.bfloat16).t() for weight in weights]

    monkeypatch.setattr(shared_mod, "quantize_fp8_block", quantize)
    monkeypatch.setattr(token_sharded_tp, "quantize_fp8_block", quantize)
    monkeypatch.setattr(Linear, "forward", linear_forward)
    monkeypatch.setattr(torch.ops.trtllm, "token_sharded_fp8_ce_gather_gemms", multi)
    ts.all_gather = gather
    ts.reduce_scatter = lambda x: x[: ts.plan.local_rows]
    return ts, (col, mlp, gated), calls


@pytest.mark.parametrize("mode", ["nccl", "copy_engine"])
def test_shared_input_column_mlp_and_gated_mlp(mode, monkeypatch):
    ts, modules, calls = _shared_adapters(monkeypatch, mode)
    x = torch.ones(4, 128, dtype=torch.bfloat16)
    shared = ts.share_input(x, modules)
    outputs = tuple(module(shared) for module in modules)
    assert [tuple(out.shape) for out in outputs] == [(1, 8, 128), (1, 4, 128), (1, 4, 128)]
    assert len(calls["quantize"]) == 1
    assert len(calls["gather"]) == (mode == "nccl")
    assert len(calls["multi"]) == (mode == "copy_engine")
    if mode == "nccl":
        up_projections = (modules[0], modules[1].up_proj, modules[2].gate_up_proj)
        seen = [x for module, x, _, _ in calls["linear"] if module in up_projections]
        assert len(seen) == 3 and all(value is shared.activation for value in seen)
    else:
        assert len(calls["multi"][0][0]) == 3
    # A fresh invocation must recompute even if the same tensor is modified in place.
    x.add_(1)
    again = ts.share_input(x, modules)
    assert not torch.equal(modules[0](again), outputs[0])
    assert len(calls["quantize"]) == 2


def test_shared_input_mixed_quantizers_and_lora_keep_dense_rows(monkeypatch):
    ts, modules, calls = _shared_adapters(monkeypatch, "copy_engine", mixed=True)
    x = torch.ones(4, 128, dtype=torch.bfloat16)
    shared = ts.share_input(x, modules)
    tuple(module(shared) for module in modules)
    assert len(calls["gather"]) == 1
    assert isinstance(shared.activation, torch.Tensor)
    assert calls["quantize"] == calls["multi"] == []
    dense = ts.share_input(x, modules, prequantize=False)
    modules[0](dense, lora_params={"active": True})
    assert calls["linear"][-1][3]["lora_params"] == {"active": True}


def test_shared_input_rejects_wrong_consumers_plans_and_lora(monkeypatch):
    ts, modules, _ = _shared_adapters(monkeypatch, "copy_engine")
    x = torch.ones(4, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at least one"):
        ts.share_input(x, ())
    with pytest.raises(ValueError, match="distinct"):
        ts.share_input(x, (modules[0], modules[0]))
    with pytest.raises(ValueError, match="converted"):
        ts.share_input(x, (torch.nn.Identity(),))
    with pytest.raises(ValueError, match="width"):
        ts.share_input(x[:, :64], modules)
    shared = ts.share_input(x, modules[:1])
    with pytest.raises(ValueError, match="not a consumer"):
        modules[1](shared)
    with pytest.raises(ValueError, match="LoRA"):
        modules[0](shared, lora_params={"active": True})
    with pytest.raises(TypeError, match="keyword-only"):
        modules[0](shared, None)
    ts._plan = TokenShardPlan.build(1, 6, 2, 0)
    with pytest.raises(ValueError, match="token plan"):
        modules[0](shared)


@pytest.mark.parametrize("gather_mode", ["nccl", "copy_engine"])
@pytest.mark.parametrize("fused_reduce", [False, True])
def test_shared_input_mlp_preserves_reduce_scatter_routing(gather_mode, fused_reduce, monkeypatch):
    ts, modules, calls = _shared_adapters(monkeypatch, gather_mode)
    mlp = modules[1]
    x = torch.ones(4, 128, dtype=torch.bfloat16)
    shared = ts.share_input(x, modules)
    reference = mlp(shared)
    ts.reduce_scatter_mode = "copy_engine"
    ts._rs_state = SimpleNamespace(effective=True, reason="")
    decisions, fused = [], []

    def uses_ce(consumer, activation, lora_params=None):
        decisions.append((consumer, activation, lora_params))
        return fused_reduce

    def gemm_reduce(consumer, activation):
        fused.append((consumer, activation))
        return ts.reduce_scatter(consumer(activation))

    monkeypatch.setattr(ts, "uses_ce_reduce_scatter", uses_ce)
    monkeypatch.setattr(ts, "ce_gemm_reduce_scatter", gemm_reduce)
    actual = mlp(shared)
    assert torch.equal(actual, reference)
    assert len(decisions) == 1 and decisions[0][0] is mlp.down_proj
    assert len(fused) == fused_reduce
    assert len(calls["quantize"]) == 1
    assert len(calls["gather"]) == (gather_mode == "nccl")
    assert len(calls["multi"]) == (gather_mode == "copy_engine")


def test_model_config_defaults_keep_nccl_and_unaligned_rows(monkeypatch):
    monkeypatch.setattr(token_sharded_tp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(token_sharded_tp.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(token_sharded_tp, "DEFAULT_GATHER_MODE", "copy_engine")
    monkeypatch.setattr(token_sharded_tp, "DEFAULT_REDUCE_SCATTER_MODE", "copy_engine")
    cfg = SimpleNamespace(
        visual_gen_mapping=SimpleNamespace(tp_group_pg=_FAKE_GROUP, tp_rank=0),
        parallel=ParallelConfig(tp_size=2, tp_layout="token_sharded"),
    )
    tp = TokenShardedTP.from_model_config(cfg)
    assert tp.gather_mode == tp.reduce_scatter_mode == "nccl"
    assert tp.row_align == 1
