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

import math

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


class FakeMapping(Mapping):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, world_size, rank):
        super().__init__(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
        )
        self.tp_rank = rank


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def build_weights(in_features, out_features, quant_algo, bias=True):
    if quant_algo == QuantAlgo.NO_QUANT:
        w = {
            "weight": torch.randn(out_features, in_features)
            * torch.rsqrt(torch.tensor(float(in_features)))
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.FP8:
        fp32_weight = torch.randn(out_features, in_features) * torch.rsqrt(
            torch.tensor(float(in_features))
        )
        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        weight_scale = fp32_weight.abs().max() / max_fp8
        fp8_weight = (fp32_weight / weight_scale).to(torch.float8_e4m3fn)
        w = {
            "weight": fp8_weight,
            "weight_scale": weight_scale,
            "input_scale": torch.tensor(1.0, dtype=torch.float32),
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
        fp32_weight = torch.randn(out_features, in_features) * torch.rsqrt(
            torch.tensor(float(in_features))
        )
        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        # Per-row scale: one scale per output row
        row_max = fp32_weight.abs().amax(dim=1)
        weight_scale = row_max / max_fp8
        fp8_weight = (fp32_weight / weight_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
        w = {
            "weight": fp8_weight,
            "weight_scale": weight_scale,
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
        fp32_weight = torch.randn(out_features, in_features) * torch.rsqrt(
            torch.tensor(float(in_features))
        )
        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        # Per 128-element block scales
        scale_rows = math.ceil(out_features / 128)
        scale_cols = math.ceil(in_features / 128)
        weight_scale = torch.empty(scale_rows, scale_cols, dtype=torch.float32)
        fp8_weight = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        for r in range(scale_rows):
            for c in range(scale_cols):
                r_start, r_end = r * 128, min((r + 1) * 128, out_features)
                c_start, c_end = c * 128, min((c + 1) * 128, in_features)
                block = fp32_weight[r_start:r_end, c_start:c_end]
                block_max = block.abs().max()
                s = block_max / max_fp8
                weight_scale[r, c] = s
                fp8_weight[r_start:r_end, c_start:c_end] = (block / s).to(torch.float8_e4m3fn)
        w = {
            "weight": fp8_weight,
            "weight_scale": weight_scale,
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.NVFP4:
        FP8_MAX, E2M1_MAX = 448.0, 6.0
        scaling_vector_size = 16
        fp32_weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16
        ) * torch.rsqrt(torch.tensor(float(in_features)))
        weight_amax = fp32_weight.abs().max().float()
        weight_scale_2 = weight_amax / (FP8_MAX * E2M1_MAX)
        input_scale = torch.tensor(FP8_MAX * E2M1_MAX, dtype=torch.float32)
        # Quantize weight to FP4 using the TRTLLM op (NVFP4: sfVecSize=16, UE8M0=False)
        global_scale = torch.tensor(FP8_MAX * E2M1_MAX / weight_amax, device="cuda")
        fp4_weight, fp4_weight_scale = torch.ops.trtllm.fp4_quantize(
            fp32_weight,
            global_scale,
            scaling_vector_size,
            sfUseUE8M0=False,
            isSfSwizzledLayout=False,
        )
        fp4_weight_scale = fp4_weight_scale.reshape(
            out_features, in_features // scaling_vector_size
        )
        w = {
            "weight": fp4_weight.cpu(),
            "weight_scale": fp4_weight_scale.cpu(),
            "input_scale": input_scale,
            "weight_scale_2": weight_scale_2,
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.W4A8_NVFP4_FP8:
        scaling_vector_size = 32
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

        # FP4 E2M1: 1.0 = 0b0010. Super-diagonal: M[i, i+1] = 1.
        # Packed pairs: element i+1 sits in byte (i+1)//2, nibble (i+1)%2.
        # This is an easy way to generate synthetic data that will not cause
        # overflows but still requires cross-gpu communication (ie not block diagonal)
        packed_cols = in_features // 2
        raw = torch.zeros(out_features, packed_cols, dtype=torch.uint8)
        for i in range(min(out_features, in_features - 1)):
            j = i + 1
            byte_idx = j // 2
            if j % 2 == 0:
                raw[i, byte_idx] = 0x02  # low nibble
            else:
                raw[i, byte_idx] = 0x20  # high nibble
        fp4_weight = raw.view(fp4_utils.float4_e2m1x2)
        scale_shape = (out_features, in_features // scaling_vector_size)
        fp4_weight_scale = torch.ones(scale_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        input_scale = torch.tensor(1.0, dtype=torch.float32)
        weight_scale_2 = torch.tensor(1.0, dtype=torch.float32)
        w = {
            "weight": fp4_weight,
            "weight_scale": fp4_weight_scale,
            "input_scale": input_scale,
            "weight_scale_2": weight_scale_2,
        }
        if bias:
            w["bias"] = torch.zeros(out_features)
        return [w]
    elif quant_algo in (QuantAlgo.W4A8_MXFP4_FP8, QuantAlgo.W4A8_MXFP4_MXFP8):
        scaling_vector_size = 32
        fp32_weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16
        ) * torch.rsqrt(torch.tensor(float(in_features)))
        # MXFP4: sfVecSize=32, UE8M0=True, no globalScale needed
        fp4_weight, fp4_weight_scale = torch.ops.trtllm.fp4_quantize(
            fp32_weight, None, scaling_vector_size, sfUseUE8M0=True, isSfSwizzledLayout=False
        )
        fp4_weight_scale = fp4_weight_scale.reshape(
            out_features, in_features // scaling_vector_size
        )
        w = {
            "weight": fp4_weight.cpu(),
            "weight_scale": fp4_weight_scale.cpu(),
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.W8A16:
        # Match the existing THOP weight-only linear test: quantize a logical
        # (in_features, out_features) matrix and store checkpoint weight as
        # (out_features, in_features).
        fp32_weight = torch.randn(in_features, out_features) * torch.rsqrt(
            torch.tensor(float(in_features))
        )
        quant_weight, _, weight_scale = (
            torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
                fp32_weight.cpu(), torch.int8
            )
        )
        w = {
            "weight": quant_weight.T.contiguous(),
            "weight_scale": weight_scale,
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo == QuantAlgo.W4A16:
        # INT4 weight-only checkpoint stores the output dimension packed 2:1:
        # quant_weight is (in_features, out_features // 2), so transposed
        # checkpoint weight is (out_features // 2, in_features).
        fp32_weight = torch.randn(in_features, out_features) * torch.rsqrt(
            torch.tensor(float(in_features))
        )
        quant_weight, _, weight_scale = (
            torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
                fp32_weight.cpu(), torch.quint4x2
            )
        )
        w = {
            "weight": quant_weight.T.contiguous(),
            "weight_scale": weight_scale,
        }
        if bias:
            w["bias"] = torch.randn(out_features)
        return [w]
    elif quant_algo in (QuantAlgo.W4A16_AWQ, QuantAlgo.W4A8_AWQ):
        group_size = 128
        dtype = torch.float16 if quant_algo == QuantAlgo.W4A16_AWQ else torch.bfloat16
        # Checkpoint weight is packed along output dim. Use a sparse synthetic
        # super-diagonal so sharded and full GEMMs compare stably.
        raw_weight = torch.zeros(in_features, out_features // 2, dtype=torch.uint8, device="cuda")
        for i in range(min(in_features, out_features - 1)):
            j = i + 1
            byte_idx = j // 2
            if j % 2 == 0:
                raw_weight[i, byte_idx] = 0x01  # low nibble
            else:
                raw_weight[i, byte_idx] = 0x10  # high nibble
        pre_quant_scale = torch.ones(in_features, dtype=dtype, device="cuda")
        scale_dtype = torch.float32 if quant_algo == QuantAlgo.W4A16_AWQ else torch.float16
        weight_scale = torch.ones(
            in_features // group_size, out_features, dtype=scale_dtype, device="cuda"
        )
        w = {
            "weight": raw_weight.T.contiguous(),
            "weight_scale": weight_scale.T.contiguous(),
            "pre_quant_scale": pre_quant_scale,
        }
        if quant_algo == QuantAlgo.W4A8_AWQ:
            w["input_scale"] = torch.tensor(1.0, dtype=torch.float32)
            w["weight_scale_2"] = torch.tensor(1.0, dtype=torch.float32)
        if bias:
            w["bias"] = torch.zeros(out_features)
        return [w]
    else:
        raise NotImplementedError(f"Test does not support QuantAlgo {quant_algo}")


DEFAULT_DTYPES = {
    QuantAlgo.NO_QUANT: torch.float32,
    QuantAlgo.FP8: torch.float32,
    QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN: torch.bfloat16,
    QuantAlgo.FP8_BLOCK_SCALES: torch.bfloat16,
    QuantAlgo.NVFP4: torch.bfloat16,
    QuantAlgo.W4A8_NVFP4_FP8: torch.bfloat16,
    QuantAlgo.W4A8_MXFP4_FP8: torch.bfloat16,
    QuantAlgo.W4A8_MXFP4_MXFP8: torch.bfloat16,
    QuantAlgo.W8A16: torch.float16,
    QuantAlgo.W4A16: torch.float16,
    QuantAlgo.W4A16_AWQ: torch.float16,
    QuantAlgo.W4A8_AWQ: torch.bfloat16,
}


def build_linears(
    in_features,
    out_features,
    world_size,
    quant_algo,
    bias=True,
    dtype=None,
    overrides=None,
    **kwargs,
):
    """Build one Linear per rank, load shared weights.

    Args:
        overrides: Optional list of per-rank override_tp_sharding tuples.
                   Length must equal world_size. Overrides auto tp_sharding.
    """
    weights = build_weights(in_features, out_features, quant_algo, bias=bias)
    if dtype is None:
        dtype = DEFAULT_DTYPES[quant_algo]
    if overrides is not None:
        assert len(overrides) == world_size
    linears = []
    for rank in range(world_size):
        mapping = FakeMapping(world_size, rank)
        if quant_algo in (QuantAlgo.W4A16_AWQ, QuantAlgo.W4A8_AWQ):
            quant_config = QuantConfig(quant_algo=quant_algo, group_size=128, has_zero_point=False)
        else:
            quant_config = QuantConfig(quant_algo=quant_algo)
        override = overrides[rank] if overrides is not None else None
        linear = Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            reduce_output=False,
            override_tp_sharding=override,
            **kwargs,
        )
        linear.load_weights(weights)
        linear.post_load_weights()
        linear.cuda()
        linears.append(linear)
    return linears, weights


def _fused_shard_indices_mapping(shard_keys, ranges):
    mapping = {}
    offset = 0
    for key in shard_keys:
        start, end = ranges[key]
        size = end - start
        mapping[key] = (offset, size)
        offset += size
    return mapping


def _prepare_fused_weights_for_loading(weights, quant_algo):
    if quant_algo == QuantAlgo.NVFP4:
        shared_weight_scale_2 = weights[0]["weight_scale_2"].clone()
        for weight in weights:
            weight["weight_scale"] = weight["weight_scale"].view(torch.float8_e4m3fn)
            weight["weight_scale_2"] = shared_weight_scale_2.clone()


def build_fused_linears(
    in_features,
    sub_out_features,
    world_size,
    quant_algo,
    weight_mode,
    shard_keys,
    overrides=None,
    allow_partial_loading=False,
):
    weights = [
        build_weights(in_features, sub_out_features, quant_algo, bias=True)[0] for _ in shard_keys
    ]
    _prepare_fused_weights_for_loading(weights, quant_algo)
    dtype = DEFAULT_DTYPES[quant_algo]
    if overrides is not None:
        assert len(overrides) == world_size
    linears = []
    for rank in range(world_size):
        mapping = FakeMapping(world_size, rank)
        quant_config = QuantConfig(quant_algo=quant_algo)
        override = overrides[rank] if overrides is not None else None
        shard_indices_mapping = (
            _fused_shard_indices_mapping(shard_keys, override)
            if override is not None and (allow_partial_loading or quant_algo == QuantAlgo.NVFP4)
            else None
        )
        linear = Linear(
            in_features,
            sub_out_features * len(shard_keys),
            bias=True,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            weights_loading_config=WeightsLoadingConfig(weight_mode=weight_mode),
            reduce_output=False,
            override_tp_sharding=override,
            fused_weight_shard_indices_mapping=shard_indices_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        linear.load_weights(weights, allow_partial_loading=allow_partial_loading)
        if allow_partial_loading:
            linear.process_weights_after_loading()
        linear.post_load_weights()
        linear.cuda()
        linears.append(linear)
    return linears, weights


def build_fused_reference(
    in_features, sub_out_features, quant_algo, weight_mode, shard_keys, weights
):
    dtype = DEFAULT_DTYPES[quant_algo]
    mapping = FakeMapping(1, 0)
    quant_config = QuantConfig(quant_algo=quant_algo)
    shard_indices_mapping = _fused_shard_indices_mapping(
        shard_keys, {key: (0, sub_out_features) for key in shard_keys}
    )
    ref = Linear(
        in_features,
        sub_out_features * len(shard_keys),
        bias=True,
        dtype=dtype,
        mapping=mapping,
        quant_config=quant_config,
        weights_loading_config=WeightsLoadingConfig(weight_mode=weight_mode),
        reduce_output=False,
        fused_weight_shard_indices_mapping=shard_indices_mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
    )
    ref.load_weights(weights)
    ref.post_load_weights()
    ref.cuda()
    return ref


def build_reference(in_features, out_features, quant_algo, weights, bias=True, **kwargs):
    """Build a single tp_size=1 linear loaded with the given weights."""
    dtype = DEFAULT_DTYPES.get(quant_algo, torch.float32)
    mapping = FakeMapping(1, 0)
    quant_config = QuantConfig(quant_algo=quant_algo)
    ref = Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=dtype,
        mapping=mapping,
        quant_config=quant_config,
        reduce_output=False,
        **kwargs,
    )
    ref.load_weights(weights)
    ref.post_load_weights()
    ref.cuda()
    return ref


def _legacy_even_slice(total, tp_size, rank):
    shard = total // tp_size
    return rank * shard, (rank + 1) * shard


def _assert_same_storage(actual, expected):
    torch.testing.assert_close(
        actual.detach().cpu().view(torch.uint8),
        expected.detach().cpu().view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _check_fused_weight_reconstruction(linears, weights, shard_keys, per_rank_ranges):
    for rank, linear in enumerate(linears):
        expected_weights = []
        expected_biases = []
        for key in shard_keys:
            start, end = per_rank_ranges[rank][key]
            expected_weights.append(weights[shard_keys.index(key)]["weight"][start:end])
            expected_biases.append(weights[shard_keys.index(key)]["bias"][start:end])

        expected_weight = torch.cat(expected_weights, dim=0)
        expected_bias = torch.cat(expected_biases, dim=0).to(linear.bias.dtype)
        _assert_same_storage(linear.weight, expected_weight)
        torch.testing.assert_close(linear.bias.detach().cpu(), expected_bias)


def _assemble_fused_outputs(outputs, shard_keys, per_rank_ranges):
    per_key_outputs = {key: [] for key in shard_keys}
    for output, ranges in zip(outputs, per_rank_ranges):
        offset = 0
        for key in shard_keys:
            start, end = ranges[key]
            size = end - start
            per_key_outputs[key].append(output[..., offset : offset + size])
            offset += size

    return torch.cat([torch.cat(per_key_outputs[key], dim=-1) for key in shard_keys], dim=-1)


def _check_fused_forward(linears, ref, shard_keys, per_rank_ranges, quant_algo):
    x = torch.randn(2, ref.in_features, device="cuda", dtype=DEFAULT_DTYPES[quant_algo])
    outputs = [linear(x) for linear in linears]
    result = _assemble_fused_outputs(outputs, shard_keys, per_rank_ranges)
    expected = ref(x)
    if quant_algo == QuantAlgo.NO_QUANT:
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.2)


# ── Test parametrizations ──

# Pipeline: unified input → column(in→hidden) → row(hidden→out) → sum
# (in_features, hidden, out_features, tp_size)
PIPELINE_CASES = [
    # even
    (32, 32, 32, 2),
    (32, 64, 32, 4),
    # uneven hidden (column out and row in both split unevenly)
    (32, 10, 32, 3),
    (16, 7, 16, 2),
    (16, 13, 16, 4),
    (8, 5, 8, 3),
]


class TestMLP:
    """Unified input → ColumnParallel → RowParallel(no allreduce) → sum."""

    @pytest.mark.parametrize("in_features,hidden,out_features,tp_size", PIPELINE_CASES)
    def test_pipeline(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            QuantAlgo.NO_QUANT,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            QuantAlgo.NO_QUANT,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda")

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)

        result = sum(partial_outputs)

        w_col = col_weights[0]["weight"].cuda()
        b_col = col_weights[0]["bias"].cuda()
        w_row = row_weights[0]["weight"].cuda()
        b_row = row_weights[0]["bias"].cuda()
        expected = (x @ w_col.t() + b_col) @ w_row.t() + b_row
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("in_features,hidden,out_features,tp_size", PIPELINE_CASES)
    def test_pipeline_no_bias(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            QuantAlgo.NO_QUANT,
            bias=False,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            QuantAlgo.NO_QUANT,
            bias=False,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda")

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)

        result = sum(partial_outputs)

        w_col = col_weights[0]["weight"].cuda()
        w_row = row_weights[0]["weight"].cuda()
        expected = (x @ w_col.t()) @ w_row.t()
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    get_sm_version() < 89,
    reason="FP8 per-tensor is supported on SM 89+ GPUs",
)
class TestFP8QDQMLP:
    """FP8QDQ: unified input → ColumnParallel → RowParallel → sum."""

    @pytest.mark.parametrize(
        "in_features,hidden,out_features,tp_size",
        [
            (32, 32, 32, 2),
            (32, 64, 32, 4),
            (64, 48, 64, 3),
        ],
    )
    def test_pipeline(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            QuantAlgo.FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            QuantAlgo.FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            in_features,
            hidden,
            QuantAlgo.FP8,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            hidden,
            out_features,
            QuantAlgo.FP8,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda")

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)
        result = sum(partial_outputs)

        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


FP8R = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN


@pytest.mark.skipif(
    get_sm_version() != 90,
    reason="FP8 rowwise is supported on Hopper GPUs",
)
class TestFP8RowwiseMLP:
    """FP8 Rowwise: unified input → ColumnParallel → RowParallel → sum."""

    @pytest.mark.parametrize(
        "in_features,hidden,out_features,tp_size",
        [
            (32, 32, 32, 2),
            (32, 64, 32, 4),
            (64, 48, 64, 3),
        ],
    )
    def test_pipeline(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            FP8R,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            FP8R,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            in_features,
            hidden,
            FP8R,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            hidden,
            out_features,
            FP8R,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        # fp8 input bypasses dynamic per-token quantization in both column
        # and row linears. Scale keeps values in fp8 normal range.
        x = (torch.randn(2, in_features, device="cuda") * 0.1).to(torch.float8_e4m3fn)

        # Column outputs bf16; cast to fp8 before row to avoid requantization
        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x).to(torch.float8_e4m3fn)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)
        result = sum(partial_outputs)

        col_ref_out = col_ref(x).to(torch.float8_e4m3fn)
        expected = row_ref(col_ref_out)
        # atol accounts for bf16 accumulation order differences between
        # sharded and full GEMM (max observed diff ~0.008)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


FP8BS = QuantAlgo.FP8_BLOCK_SCALES

# All shard boundaries must be 128-aligned (scale_span=128 assertion in
# load_shard). The tp_size=3 case distributes five 128-blocks as 2,2,1.
FP8BS_PIPELINE_CASES = [
    (256, 256, 256, 2),
    (512, 512, 512, 4),
    (640, 640, 640, 3),
]


@pytest.mark.skipif(
    not (get_sm_version() == 90 or is_sm_100f()),
    reason="FP8 block scales are supported on Hopper and SM 100 family GPUs",
)
class TestFP8BlockScalesMLP:
    """FP8 Block Scales: column → row pipeline."""

    @pytest.mark.parametrize("in_features,hidden,out_features,tp_size", FP8BS_PIPELINE_CASES)
    def test_pipeline(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            FP8BS,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            FP8BS,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            in_features,
            hidden,
            FP8BS,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            hidden,
            out_features,
            FP8BS,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda", dtype=torch.bfloat16)

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)
        result = sum(partial_outputs)

        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not is_sm_100f(),
    reason="This test is only supported on SM 100 family GPUs",
)
class TestNVFP4MLP:
    """NVFP4: column → row pipeline. ROW requires 16-aligned shard boundaries."""

    @pytest.mark.parametrize(
        "in_features,hidden,out_features,tp_size",
        [
            (256, 256, 256, 2),  # even
            (256, 256, 256, 3),  # uneven: ROW shards 16 blocks → 6,5,5
            (256, 256, 256, 4),  # even
        ],
    )
    def test_pipeline(self, in_features, hidden, out_features, tp_size):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            QuantAlgo.NVFP4,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            QuantAlgo.NVFP4,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            in_features,
            hidden,
            QuantAlgo.NVFP4,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            hidden,
            out_features,
            QuantAlgo.NVFP4,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda", dtype=torch.bfloat16)

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)
        result = sum(partial_outputs)

        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not is_sm_100f(),
    reason="This test is only supported on SM 100 family GPUs",
)
class TestW4A8MXFP4FP8MLP:
    """W4A8 MXFP4/FP8: column → row pipeline.

    CUTLASS MXFP8xMXFP4 kernel requires shard dims divisible by 128.
    Uneven test uses explicit overrides with 128-aligned splits.
    """

    def test_pipeline_even(self):
        self._run_pipeline(256, 2)

    def test_pipeline_uneven(self):
        overrides = [(0, 256), (256, 512), (512, 640)]
        self._run_pipeline(640, 3, overrides=overrides)

    def _run_pipeline(self, dim, tp_size, overrides=None):
        col_linears, col_weights = build_linears(
            dim,
            dim,
            tp_size,
            QuantAlgo.W4A8_MXFP4_FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            overrides=overrides,
        )
        row_linears, row_weights = build_linears(
            dim,
            dim,
            tp_size,
            QuantAlgo.W4A8_MXFP4_FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
            overrides=overrides,
        )
        col_ref = build_reference(
            dim,
            dim,
            QuantAlgo.W4A8_MXFP4_FP8,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            dim,
            dim,
            QuantAlgo.W4A8_MXFP4_FP8,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
        partial_outputs = []
        for rank in range(tp_size):
            partial_outputs.append(row_linears[rank](col_linears[rank](x)))
        result = sum(partial_outputs)
        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.2)


@pytest.mark.skipif(
    not is_sm_100f(),
    reason="This test is only supported on SM 100 family GPUs",
)
class TestW4A8NVFP4FP8MLP:
    """W4A8 NVFP4/FP8: column → row pipeline.

    Uses synthetic weights with reinterpreted scale dtype.
    Same 128-aligned override requirement as MXFP4.
    """

    def test_pipeline_even(self):
        self._run_pipeline(256, 2)

    def test_pipeline_uneven(self):
        overrides = [(0, 256), (256, 512), (512, 640)]
        self._run_pipeline(640, 3, overrides=overrides)

    def _run_pipeline(self, dim, tp_size, overrides=None):
        col_linears, col_weights = build_linears(
            dim,
            dim,
            tp_size,
            QuantAlgo.W4A8_NVFP4_FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            overrides=overrides,
        )
        row_linears, row_weights = build_linears(
            dim,
            dim,
            tp_size,
            QuantAlgo.W4A8_NVFP4_FP8,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
            overrides=overrides,
        )
        col_ref = build_reference(
            dim,
            dim,
            QuantAlgo.W4A8_NVFP4_FP8,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            dim,
            dim,
            QuantAlgo.W4A8_NVFP4_FP8,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
        partial_outputs = []
        for rank in range(tp_size):
            partial_outputs.append(row_linears[rank](col_linears[rank](x)))
        result = sum(partial_outputs)
        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.2)


@pytest.mark.skipif(
    not is_sm_100f(),
    reason="This test is only supported on SM 100 family GPUs",
)
class TestW4A8MXFP4MXFP8MLP:
    """W4A8 MXFP4/MXFP8: inherits W4A8MXFP4FP8, uses mxfp8_quantize for activation."""

    def test_pipeline_even(self):
        self._run_pipeline(256, 2)

    def test_pipeline_uneven(self):
        overrides = [(0, 256), (256, 512), (512, 640)]
        self._run_pipeline(640, 3, overrides=overrides)

    def _run_pipeline(self, dim, tp_size, overrides=None):
        algo = QuantAlgo.W4A8_MXFP4_MXFP8
        col_linears, col_weights = build_linears(
            dim,
            dim,
            tp_size,
            algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            overrides=overrides,
        )
        row_linears, row_weights = build_linears(
            dim,
            dim,
            tp_size,
            algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
            overrides=overrides,
        )
        col_ref = build_reference(
            dim,
            dim,
            algo,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            dim,
            dim,
            algo,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
        partial_outputs = []
        for rank in range(tp_size):
            partial_outputs.append(row_linears[rank](col_linears[rank](x)))
        result = sum(partial_outputs)
        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.2)


@pytest.mark.skipif(
    get_sm_version() < 80,
    reason="Weight-only INT8/INT4 is supported on Ampere+ GPUs",
)
class TestWeightOnlyQuantMLP:
    """Weight-only INT8 and INT4 quantization."""

    @pytest.mark.parametrize("quant_algo", [QuantAlgo.W8A16, QuantAlgo.W4A16])
    @pytest.mark.parametrize(
        "in_features,hidden,out_features,tp_size",
        [
            (256, 256, 256, 2),  # even
            (256, 256, 256, 3),  # uneven
        ],
    )
    def test_pipeline(self, in_features, hidden, out_features, tp_size, quant_algo):
        col_linears, col_weights = build_linears(
            in_features,
            hidden,
            tp_size,
            quant_algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            hidden,
            out_features,
            tp_size,
            quant_algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            in_features,
            hidden,
            quant_algo,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            hidden,
            out_features,
            quant_algo,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, in_features, device="cuda", dtype=torch.float16)

        partial_outputs = []
        for rank in range(tp_size):
            col_out = col_linears[rank](x)
            row_out = row_linears[rank](col_out)
            partial_outputs.append(row_out)
        result = sum(partial_outputs)

        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


class _AWQMLPMixin:
    quant_algo = None

    @pytest.mark.parametrize(
        "dim,tp_size",
        [
            (256, 2),  # even
            (640, 3),  # uneven: 128-group shards -> 256,256,128
        ],
    )
    def test_pipeline(self, dim, tp_size):
        quant_algo = self.quant_algo
        col_linears, col_weights = build_linears(
            dim,
            dim,
            tp_size,
            quant_algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_linears, row_weights = build_linears(
            dim,
            dim,
            tp_size,
            quant_algo,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        col_ref = build_reference(
            dim,
            dim,
            quant_algo,
            weights=col_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        row_ref = build_reference(
            dim,
            dim,
            quant_algo,
            weights=row_weights,
            bias=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        x = torch.randn(2, dim, device="cuda", dtype=DEFAULT_DTYPES[quant_algo])
        partial_outputs = []
        for rank in range(tp_size):
            partial_outputs.append(row_linears[rank](col_linears[rank](x)))
        result = sum(partial_outputs)
        expected = row_ref(col_ref(x))
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    get_sm_version() < 80,
    reason="W4A16 AWQ is supported on Ampere+ GPUs",
)
class TestW4A16AWQMLP(_AWQMLPMixin):
    """W4A16 AWQ with grouped scales."""

    quant_algo = QuantAlgo.W4A16_AWQ


@pytest.mark.skipif(
    not (get_sm_version() in (89, 90) or is_sm_100f()),
    reason="W4A8 AWQ is supported on Ada, Hopper, and SM 100 family GPUs",
)
class TestW4A8AWQMLP(_AWQMLPMixin):
    """W4A8 AWQ with grouped scales."""

    quant_algo = QuantAlgo.W4A8_AWQ


FUSED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_FP8


class TestFusedLinearLoading:
    """Fused QKV and Gate/Up loading for legacy even and override uneven TP."""

    @pytest.mark.parametrize(
        "weight_mode,quant_algo,sub_out,tp_size",
        [
            (WeightMode.FUSED_QKV_LINEAR, QuantAlgo.NO_QUANT, 96, 3),
            (WeightMode.FUSED_GATE_UP_LINEAR, QuantAlgo.NO_QUANT, 96, 3),
        ],
    )
    def test_even_no_override(self, weight_mode, quant_algo, sub_out, tp_size):
        shard_keys = weight_mode.shard_keys
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
        )
        ranges = []
        for rank in range(tp_size):
            start, end = _legacy_even_slice(sub_out, tp_size, rank)
            ranges.append({key: (start, end) for key in shard_keys})

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        if quant_algo != QuantAlgo.NO_QUANT:
            for linear in linears:
                assert linear.weight_scale.numel() > 0

    @pytest.mark.parametrize("quant_algo", [QuantAlgo.NO_QUANT])
    @pytest.mark.parametrize(
        "weight_mode",
        [
            WeightMode.FUSED_QKV_LINEAR,
            WeightMode.FUSED_GATE_UP_LINEAR,
        ],
    )
    def test_uneven_override(self, weight_mode, quant_algo):
        shard_keys = weight_mode.shard_keys
        sub_out = 640
        tp_size = 3
        boundaries = [(0, 256), (256, 512), (512, 640)]
        overrides = [{key: boundary for key in shard_keys} for boundary in boundaries]
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
            overrides=overrides,
        )
        ranges = [{key: boundary for key in shard_keys} for boundary in boundaries]

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        if quant_algo != QuantAlgo.NO_QUANT:
            for linear in linears:
                assert linear.weight_scale.numel() > 0

    @pytest.mark.parametrize("quant_algo", [QuantAlgo.NO_QUANT])
    @pytest.mark.parametrize(
        "weight_mode",
        [
            WeightMode.FUSED_QKV_LINEAR,
            WeightMode.FUSED_GATE_UP_LINEAR,
        ],
    )
    def test_uneven_override_partial_loading(self, weight_mode, quant_algo):
        shard_keys = weight_mode.shard_keys
        sub_out = 640
        tp_size = 3
        boundaries = [(0, 256), (256, 512), (512, 640)]
        overrides = [{key: boundary for key in shard_keys} for boundary in boundaries]
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
            overrides=overrides,
            allow_partial_loading=True,
        )
        ranges = [{key: boundary for key in shard_keys} for boundary in boundaries]

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        if quant_algo != QuantAlgo.NO_QUANT:
            for linear in linears:
                assert linear.weight_scale.numel() > 0


@pytest.mark.skipif(
    not is_sm_100f(),
    reason="Fused FP4/NVFP4 loading is supported on SM 100 family GPUs",
)
class TestFusedQuantizedLinearLoading:
    """Quantized fused QKV and Gate/Up loading for override uneven TP."""

    @pytest.mark.parametrize(
        "weight_mode,quant_algo,sub_out,tp_size",
        [
            (WeightMode.FUSED_QKV_LINEAR, FUSED_QUANT_ALGO, 256, 2),
            (WeightMode.FUSED_GATE_UP_LINEAR, FUSED_QUANT_ALGO, 256, 2),
        ],
    )
    def test_even_no_override(self, weight_mode, quant_algo, sub_out, tp_size):
        shard_keys = weight_mode.shard_keys
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
        )
        ranges = []
        for rank in range(tp_size):
            start, end = _legacy_even_slice(sub_out, tp_size, rank)
            ranges.append({key: (start, end) for key in shard_keys})

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        for linear in linears:
            assert linear.weight_scale.numel() > 0

    @pytest.mark.parametrize("quant_algo", [FUSED_QUANT_ALGO])
    @pytest.mark.parametrize(
        "weight_mode",
        [
            WeightMode.FUSED_QKV_LINEAR,
            WeightMode.FUSED_GATE_UP_LINEAR,
        ],
    )
    def test_uneven_override(self, weight_mode, quant_algo):
        shard_keys = weight_mode.shard_keys
        sub_out = 640
        tp_size = 3
        boundaries = [(0, 256), (256, 512), (512, 640)]
        overrides = [{key: boundary for key in shard_keys} for boundary in boundaries]
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
            overrides=overrides,
        )
        ranges = [{key: boundary for key in shard_keys} for boundary in boundaries]

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        for linear in linears:
            assert linear.weight_scale.numel() > 0

    @pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4])
    @pytest.mark.parametrize(
        "weight_mode",
        [
            WeightMode.FUSED_QKV_LINEAR,
            WeightMode.FUSED_GATE_UP_LINEAR,
        ],
    )
    def test_uneven_override_partial_loading(self, weight_mode, quant_algo):
        shard_keys = weight_mode.shard_keys
        sub_out = 640
        tp_size = 3
        boundaries = [(0, 256), (256, 512), (512, 640)]
        overrides = [{key: boundary for key in shard_keys} for boundary in boundaries]
        linears, weights = build_fused_linears(
            256,
            sub_out,
            tp_size,
            quant_algo,
            weight_mode,
            shard_keys,
            overrides=overrides,
            allow_partial_loading=True,
        )
        ranges = [{key: boundary for key in shard_keys} for boundary in boundaries]

        ref = build_fused_reference(256, sub_out, quant_algo, weight_mode, shard_keys, weights)
        _check_fused_weight_reconstruction(linears, weights, shard_keys, ranges)
        _check_fused_forward(linears, ref, shard_keys, ranges, quant_algo)
        for linear in linears:
            assert linear.weight_scale.numel() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
