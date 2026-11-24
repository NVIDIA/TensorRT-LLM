# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``bf16_linear`` and the ``K31Linear`` / ``K31SituMLP`` parameter holders give the engine modules' bits.

Every certified cell runs through ``bf16_linear`` (its guard on) against the FP32 reference and against the engine
``Linear`` with the same weight (and the same strided input where the target passes a column slice), bit for bit.
The holders are compared bit for bit with the engine ``Linear`` / ``GatedMLP`` (SiTU activation) at the projections'
local shapes for TP1 and the TEP8 / TEP16 / TEP32 / DEP16 shards, and their shard slices with the engine
``Linear``'s at the first, second, middle and last rank of those layouts. The row-parallel sums are in
``test_bf16_linear_tp4.py``, the RMSNorm holder in ``ops/norm/test_rms_norm.py``.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31Linear,
    K31SituMLP,
    bf16_linear,
)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.situ import SituAndMul
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 (sm_107 pending)",
)

BF16 = torch.bfloat16
ROWS = (1, 5, 8, 40, 300, 2048)
_MODE = {"column": TensorParallelMode.COLUMN, "row": TensorParallelMode.ROW, None: None}


def _fill(*tensors: torch.Tensor, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for t in tensors:
        t.data.copy_((torch.randn(t.shape, generator=gen) * 0.02).to(t.dtype))


def _x(rows: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=gen, device="cuda").to(BF16)


@pytest.mark.parametrize("cell", bf16_linear.CELLS, ids=[c.why for c in bf16_linear.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    m, n, k, cublas = spec["m"], spec["n"], spec["k"], spec["cublas"]
    linear = Linear(k, n, bias=False, dtype=BF16, use_custom_cublas_mm=cublas).cuda()
    seed = zlib.crc32(cell.why.encode()) % 2**31
    _fill(linear.weight, seed=seed)
    x = _x(m, spec["row_stride"], seed=seed + 1)[:, :k]
    with validating(bf16_linear):
        actual = bf16_linear(x, linear.weight, cublas)
    assert torch.equal(actual, linear(x)), "not the engine Linear's bits"
    # One BF16 rounding of an FP32 sum taken in another order.
    assert_within_ulp(
        actual, bf16_linear.reference(x, linear.weight, cublas), element_ulp=2.0, rms_ulp=0.5
    )


def test_cublas_guard_refuses_strided_operands() -> None:
    """The cuBLAS branch reads x as dense row-major and weight.t() as dense column-major: the guard refuses a strided
    view of either (the GEMM would give a plausible wrong product)."""
    weight = torch.zeros(64, 256, dtype=BF16, device="cuda")
    x = torch.zeros(4, 256, dtype=BF16, device="cuda")
    with validating(bf16_linear):
        with pytest.raises(AssertionError, match="dense column-major"):
            bf16_linear(x[:, :128].contiguous(), weight[:, :128], True)
        with pytest.raises(AssertionError, match="dense row-major"):
            bf16_linear(x[:, :128], weight[:, :128].contiguous(), True)


# (name, in, out, tp_mode, use_custom_cublas_mm): the projections the targets build, at their local (one-rank) shapes.
_PROJECTIONS = (
    ("fused q_a / kv_a", 7168, 2112, None, True),
    ("DSA indexer wq_b", 1536, 4096, None, True),
    ("TEP8 q_b", 1536, 2304, "column", False),
    ("TEP16 q_b", 1536, 1152, "column", False),
    ("TEP32 q_b", 1536, 576, "column", False),
    ("DEP16 q_b", 1536, 18432, "column", False),
    ("TEP8 kv_b", 512, 3072, "column", False),
    ("DEP16 kv_b", 512, 24576, "column", False),
    ("TEP8 o_proj", 1536, 7168, "row", False),
    ("TEP32 o_proj", 384, 7168, "row", False),
    ("DEP16 o_proj", 12288, 7168, "row", False),
    ("TEP8 output gate", 7168, 1536, "column", False),
)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name, k, n, mode, cublas", _PROJECTIONS, ids=[p[0] for p in _PROJECTIONS])
def test_holder_matches_the_engine_linear(name, k, n, mode, cublas, rows) -> None:
    one = Mapping(world_size=1, tp_size=1, rank=0)
    linear = Linear(
        k,
        n,
        bias=False,
        dtype=BF16,
        mapping=one,
        tensor_parallel_mode=_MODE[mode],
        use_custom_cublas_mm=cublas,
    ).cuda()
    holder = K31Linear(
        k, n, dtype=BF16, mapping=one, tp_mode=mode, use_custom_cublas_mm=cublas
    ).cuda()
    assert holder.weight.shape == linear.weight.shape
    assert dict(holder.named_parameters()).keys() == {"weight"}
    _fill(linear.weight, seed=n * 7 + k)
    holder.weight.data.copy_(linear.weight.data)
    x = _x(rows, k, seed=rows + n)
    assert torch.equal(holder(x), linear(x))
    assert torch.equal(holder(x, reduce=False), linear(x))


@pytest.mark.parametrize("mode", ["column", "row"])
@pytest.mark.parametrize("tp", [8, 16, 32])
@pytest.mark.parametrize("k, n", [(1536, 18432), (512, 24576), (12288, 7168), (7168, 12288)])
def test_load_shard_matches_the_engine_linear(k, n, tp, mode) -> None:
    gen = torch.Generator().manual_seed(tp + k)
    full = torch.randn(n, k, generator=gen).to(BF16)
    for rank in sorted({0, 1, tp // 2, tp - 1}):
        mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
        linear = Linear(
            k,
            n,
            bias=False,
            dtype=BF16,
            mapping=mapping,
            tensor_parallel_mode=_MODE[mode],
            reduce_output=False,
        )
        holder = K31Linear(k, n, dtype=BF16, mapping=mapping, tp_mode=mode, reduce_output=False)
        assert holder.weight.shape == linear.weight.shape
        assert (holder.in_features, holder.out_features) == (
            linear.in_features,
            linear.out_features,
        )
        expected = linear.load_shard(full, device=torch.device("cpu"))
        actual = holder.load_shard(full, device=torch.device("cpu"))
        assert torch.equal(actual, expected), (rank, mode)


@pytest.mark.parametrize("mode", ["column", "row"])
def test_load_shard_refuses_an_uneven_split(mode) -> None:
    mapping = Mapping(world_size=8, tp_size=8, rank=1)
    holder = K31Linear(1536, 2304, dtype=BF16, mapping=mapping, tp_mode=mode, reduce_output=False)
    full = torch.zeros(2304 + 4, 1536 + 4, dtype=BF16)
    with pytest.raises(ValueError, match="does not split evenly"):
        holder.load_shard(full, device=torch.device("cpu"))


@pytest.mark.parametrize("mode", ["column", "row", None])
def test_holder_refuses_cublas_on_a_tp_group(mode) -> None:
    mapping = Mapping(world_size=8, tp_size=8, rank=0)
    with pytest.raises(ValueError, match="cuBLAS branch"):
        K31Linear(1536, 18432, dtype=BF16, mapping=mapping, tp_mode=mode, use_custom_cublas_mm=True)


# (name, local intermediate): the shared experts at attention TP 1 / 8 / 16 / 32 and the dense MLP over 1 / 4 ranks.
_MLPS = (
    ("shared TP1", 6144),
    ("shared TP8", 768),
    ("shared TP16", 384),
    ("shared TP32", 192),
    ("dense TP1", 33792),
    ("dense TP4 subgroup", 8448),
)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name, inter", _MLPS, ids=[m[0] for m in _MLPS])
def test_situ_mlp_matches_the_engine_gated_mlp(name, inter, rows) -> None:
    one = Mapping(world_size=1, tp_size=1, rank=0)
    from tensorrt_llm._torch.model_config import ModelConfig

    mlp = GatedMLP(
        hidden_size=7168,
        intermediate_size=inter,
        bias=False,
        activation=SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True),
        dtype=BF16,
        config=ModelConfig(mapping=one),
        reduce_output=False,
        layer_idx=0,
    ).cuda()
    holder = K31SituMLP(
        7168,
        inter,
        dtype=BF16,
        mapping=one,
        beta=4.0,
        linear_beta=25.0,
        reduce_output=False,
    ).cuda()
    assert dict(holder.named_parameters()).keys() == {"gate_up_proj.weight", "down_proj.weight"}
    _fill(mlp.gate_up_proj.weight, mlp.down_proj.weight, seed=inter + rows)
    holder.gate_up_proj.weight.data.copy_(mlp.gate_up_proj.weight.data)
    holder.down_proj.weight.data.copy_(mlp.down_proj.weight.data)
    x = _x(rows, 7168, seed=rows * 3 + inter)
    expected = mlp(x)
    assert torch.equal(holder(x), expected)
    assert torch.equal(holder(x, reduce=False), expected)
    # The split form the targets use around a planned gate / up GEMM.
    h = holder.activation(holder.gate_up_proj(x))
    assert torch.equal(holder.down_proj(h), expected)


@pytest.mark.parametrize("tp, rank", [(2, 1), (4, 0), (4, 3), (8, 5)])
def test_situ_mlp_subgroup_mapping_matches_the_engine(tp, rank) -> None:
    """``tp_size`` narrows the group as GatedMLP's ``overridden_tp_size``: same local widths and shard ranks."""
    from tensorrt_llm._torch.model_config import ModelConfig

    mapping = Mapping(world_size=8, tp_size=8, rank=rank)
    mlp = GatedMLP(
        hidden_size=7168,
        intermediate_size=33792,
        bias=False,
        activation=SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True),
        dtype=BF16,
        config=ModelConfig(mapping=mapping),
        overridden_tp_size=tp,
        reduce_output=False,
    )
    holder = K31SituMLP(
        7168,
        33792,
        dtype=BF16,
        mapping=mapping,
        beta=4.0,
        linear_beta=25.0,
        tp_size=tp,
        reduce_output=False,
    )
    for ours, theirs in (
        (holder.gate_up_proj, mlp.gate_up_proj),
        (holder.down_proj, mlp.down_proj),
    ):
        assert ours.weight.shape == theirs.weight.shape
        assert (ours.tp_size, ours.tp_rank) == (theirs.tp_size, theirs.tp_rank)
        assert ours.mapping.tp_group == theirs.mapping.tp_group
