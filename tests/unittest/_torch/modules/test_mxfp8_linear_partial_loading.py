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
"""Partial (bucketed) weight loading for ``MXFP8LinearMethod``.

The RLHF refit path streams weights bucket by bucket via
``load_weights(allow_partial_loading=True)``, so a fused QKV / gate-up Linear
may see only one of its shards -- or a weight without its scale -- in a given
bucket. Each present shard must be written at its own offset and the result
must be bitwise identical to a single-shot load.

On the CUTLASS path the scales live in the swizzled layout, which
``get_sf_out_offset_128x4`` tiles in 128-row groups -- so an individual fused
shard is a contiguous byte range there only when its bounds happen to be
128-aligned, which q/k/v sizes are not in general. The loader therefore stages
raw scales in a row-major ``[O, K/32]`` buffer (sliceable at any alignment) and
swizzles once in ``process_weights_after_loading``. Partial loading thus
accepts the same shard shapes a full load always did.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
    get_quant_method,
)
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

BLOCK_SIZE = 32
DTYPE = torch.bfloat16

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MXFP8 Linear load path requires CUDA"
)


def _quant_config():
    return QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=BLOCK_SIZE)


# The staged raw scales and the deferred swizzle exist only on the CUTLASS path; elsewhere the
# tests that assert on them would pass without exercising anything.
requires_cutlass = pytest.mark.skipif(
    not torch.cuda.is_available() or not get_quant_method(_quant_config()).use_cutlass,
    reason="needs the CUTLASS MXFP8 path (staged scales, deferred swizzle)",
)


def _make_linear(in_features, out_features, weight_mode=WeightMode.VANILLA, shard_mapping=None):
    return Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=DTYPE,
        quant_config=_quant_config(),
        weights_loading_config=WeightsLoadingConfig(weight_mode=weight_mode),
        fused_weight_shard_indices_mapping=shard_mapping,
    ).cuda()


def _mxfp8(out_features, in_features, seed):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen, dtype=DTYPE)
    return quant_bf16_to_mxfp8(w, BLOCK_SIZE)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit pattern as uint8: these are exact-reload checks, and 0x7F/0xFF are
    NaN in float8_e4m3fn, so torch.equal on the float view would report
    identical buffers as unequal."""
    return t.contiguous().view(torch.uint8)


def _snapshot(lin):
    return _bits(lin.weight.data).clone(), _bits(lin.weight_scale.data).clone()


def _assert_same(lin, expected, label):
    w, s = _snapshot(lin)
    assert torch.equal(w, expected[0]), f"{label}: weight differs from single-shot load"
    assert torch.equal(s, expected[1]), f"{label}: weight_scale differs from single-shot load"


@requires_cuda
def test_vanilla_weight_and_scale_in_separate_buckets():
    """A bucket carrying the weight but not its scale (and vice versa) is legal
    under partial loading and must compose to the single-shot result."""
    out_f, in_f = 256, 256
    w, s = _mxfp8(out_f, in_f, seed=11)

    ref = _make_linear(in_f, out_f)
    ref.load_weights([{"weight": w, "weight_scale_inv": s}])
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f)
    lin.load_weights([{"weight": w}], allow_partial_loading=True)
    lin.load_weights([{"weight_scale_inv": s}], allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, "weight-then-scale")

    lin2 = _make_linear(in_f, out_f)
    lin2.load_weights([{"weight_scale_inv": s}], allow_partial_loading=True)
    lin2.load_weights([{"weight": w}], allow_partial_loading=True)
    lin2.process_weights_after_loading()
    _assert_same(lin2, expected, "scale-then-weight")


@requires_cuda
@pytest.mark.parametrize("order", [("q", "k", "v"), ("v", "k", "q")])
def test_fused_qkv_per_shard_buckets(order):
    """One shard per bucket, in several arrival orders."""
    in_f = 256
    q_size, kv_size = 256, 128
    out_f = q_size + 2 * kv_size
    mapping = {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}

    shards = {
        "q": _mxfp8(q_size, in_f, seed=21),
        "k": _mxfp8(kv_size, in_f, seed=22),
        "v": _mxfp8(kv_size, in_f, seed=23),
    }
    full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]} for n in ("q", "k", "v")]

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    ref.load_weights(full)
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    index = {"q": 0, "k": 1, "v": 2}
    for name in order:
        bucket = [{}, {}, {}]
        bucket[index[name]] = {"weight": shards[name][0], "weight_scale_inv": shards[name][1]}
        lin.load_weights(bucket, allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, f"qkv order={order}")


@requires_cuda
@pytest.mark.parametrize("order", [("gate", "up"), ("up", "gate")])
def test_fused_gate_up_per_shard_buckets(order):
    in_f = 256
    inter = 256  # 128-aligned
    out_f = 2 * inter
    mapping = {"gate": (0, inter), "up": (inter, inter)}
    shards = {"gate": _mxfp8(inter, in_f, seed=41), "up": _mxfp8(inter, in_f, seed=42)}

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_GATE_UP_LINEAR, mapping)
    ref.load_weights(
        [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]} for n in ("gate", "up")]
    )
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_GATE_UP_LINEAR, mapping)
    index = {"gate": 0, "up": 1}
    for name in order:
        bucket = [{}, {}]
        bucket[index[name]] = {"weight": shards[name][0], "weight_scale_inv": shards[name][1]}
        lin.load_weights(bucket, allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, f"gate_up order={order}")


@requires_cutlass
@pytest.mark.parametrize("q_size,kv_size", [(64, 64), (192, 96)])
def test_unaligned_fused_shard_bounds_are_supported(q_size, kv_size):
    """Fused shards whose row bounds are NOT 128-aligned must load correctly.

    The swizzled CUTLASS buffer is tiled in 128-row groups, so such a shard is
    not a contiguous byte range there. Staging raw scales row-major and
    swizzling once at finalize removes that constraint -- which matters for
    e.g. head_dim=64 GQA, where kv_size is a multiple of 64 but not 128.
    Full (non-partial) loading always supported these shapes; this pins that
    partial loading now matches.
    """
    in_f = 256
    out_f = q_size + 2 * kv_size
    mapping = {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}
    shards = {
        n: _mxfp8(sz, in_f, seed=s)
        for n, sz, s in (("q", q_size, 51), ("k", kv_size, 52), ("v", kv_size, 53))
    }
    full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]} for n in ("q", "k", "v")]

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    ref.load_weights(full)
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    for i, name in enumerate(("q", "k", "v")):
        bucket = [{}, {}, {}]
        bucket[i] = {"weight": shards[name][0], "weight_scale_inv": shards[name][1]}
        lin.load_weights(bucket, allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, f"unaligned q={q_size} kv={kv_size}")


@requires_cuda
def test_rlhf_refit_cycle_matches_single_shot():
    """Full update_weights lifecycle: pre_reload_weights, then one bucket per
    shard, then finalize -- twice, with different weights each round.

    ``pre_reload_weights`` re-registers every parameter as empty, so a refit
    must deliver all shards; this pins that a complete bucketed round lands
    exactly where a single-shot load of the same tensors would, and that a
    second round does not inherit state from the first.
    """
    in_f = 256
    q_size, kv_size = 256, 128
    out_f = q_size + 2 * kv_size
    mapping = {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    for round_seed in (61, 71):
        shards = {
            n: _mxfp8(sz, in_f, seed=round_seed + i)
            for i, (n, sz) in enumerate((("q", q_size), ("k", kv_size), ("v", kv_size)))
        }
        full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]} for n in ("q", "k", "v")]

        ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
        ref.load_weights(full)
        expected = _snapshot(ref)

        lin.pre_reload_weights()
        for i, name in enumerate(("q", "k", "v")):
            bucket = [{}, {}, {}]
            bucket[i] = {"weight": shards[name][0], "weight_scale_inv": shards[name][1]}
            lin.load_weights(bucket, allow_partial_loading=True)
        lin.process_weights_after_loading()
        _assert_same(lin, expected, f"refit round seed={round_seed}")


@requires_cuda
def test_staging_buffer_is_released_after_finalize():
    """The raw staging copy is a load-time temporary: it must not outlive the
    finalize that consumes it (mirrors the tmp_* buffers in the MoE methods)."""
    method = get_quant_method(_quant_config())
    if not method.use_cutlass:
        pytest.skip("reference path stores scales directly, no staging copy")

    out_f, in_f = 256, 256
    w, s = _mxfp8(out_f, in_f, seed=91)
    lin = _make_linear(in_f, out_f)

    lin.load_weights([{"weight": w, "weight_scale_inv": s}], allow_partial_loading=True)
    assert hasattr(lin, "tmp_weight_scale_raw"), (
        "staging copy should exist between a partial bucket and finalize"
    )

    lin.process_weights_after_loading()
    assert not hasattr(lin, "tmp_weight_scale_raw"), (
        "staging copy must be released by process_weights_after_loading"
    )

    # A full (non-partial) load finalizes internally, so it must not leak one.
    lin2 = _make_linear(in_f, out_f)
    lin2.load_weights([{"weight": w, "weight_scale_inv": s}])
    assert not hasattr(lin2, "tmp_weight_scale_raw")


@requires_cutlass
def test_repeated_finalize_does_not_reswizzle():
    """process_weights_after_loading is invoked by both the RLHF finalize walk
    and post_load_weights; the pending flag must make the second call a no-op
    (the swizzle is not an involution)."""
    out_f, in_f = 256, 256
    w, s = _mxfp8(out_f, in_f, seed=81)

    lin = _make_linear(in_f, out_f)
    lin.load_weights([{"weight": w, "weight_scale_inv": s}], allow_partial_loading=True)
    lin.process_weights_after_loading()
    first = _snapshot(lin)
    for _ in range(3):
        lin.process_weights_after_loading()
    _assert_same(lin, first, "repeated finalize")


class _FakeTpMapping(Mapping):
    """A tensor-parallel Mapping for one rank, without initializing torch.distributed."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, tp_size, rank):
        super().__init__(world_size=tp_size, rank=rank, tp_size=tp_size)
        self.tp_rank = rank


def _make_row_linear(in_features, out_features, tp_size, rank):
    """A standalone row-parallel MXFP8 Linear: it is built with weights, so its TP sharding is the
    explicit (start, end) bounds a model only gets for a layer created outside the deferred
    ``skip_create_weights_in_init`` path."""
    return Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=DTYPE,
        mapping=_FakeTpMapping(tp_size, rank),
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=False,
        quant_config=_quant_config(),
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.VANILLA),
    ).cuda()


def _staged_scales(lin):
    """The row-major [O, K/32] scales as loaded, before the swizzle."""
    return lin.tmp_weight_scale_raw if hasattr(lin, "tmp_weight_scale_raw") else lin.weight_scale


def _check_row_tp_scale_slices(w, s, tp_size):
    """Each rank of a row-parallel layer gets its K/tp_size weight columns and the matching
    K/(32*tp_size) scale columns."""
    in_f = w.shape[1]
    for rank in range(tp_size):
        lin = _make_row_linear(in_f, w.shape[0], tp_size, rank)
        assert lin.tp_sharding is not None, "the layer should carry explicit TP sharding bounds"
        lin.load_weights([{"weight": w, "weight_scale_inv": s}], allow_partial_loading=True)

        w_cols = slice(rank * in_f // tp_size, (rank + 1) * in_f // tp_size)
        s_cols = slice(w_cols.start // BLOCK_SIZE, w_cols.stop // BLOCK_SIZE)
        assert torch.equal(_bits(lin.weight.data), _bits(w[:, w_cols].cuda())), (
            f"rank {rank} weight"
        )
        assert torch.equal(_bits(_staged_scales(lin).data), _bits(s[:, s_cols].cuda())), (
            f"rank {rank}: scales must be the K/{BLOCK_SIZE} columns that match the weight shard"
        )


@requires_cuda
def test_row_parallel_scales_follow_the_weight_shard():
    """Synthetic weights: the scale slice of a row-parallel layer is in scale columns, not weight
    columns (a scale covers 32 weight columns)."""
    w, s = _mxfp8(256, 1024, seed=101)
    _check_row_tp_scale_slices(w, s, tp_size=2)


def _qwen3_0p6b_layer0_weights():
    """Real weights of Qwen3-0.6B layer 0 (``o_proj`` and ``down_proj`` are its row-parallel
    projections)."""
    from safetensors import safe_open
    from utils.llm_data import llm_models_root

    checkpoint = llm_models_root() / "Qwen3" / "Qwen3-0.6B"
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        pytest.skip(f"Qwen3-0.6B checkpoint not found under {checkpoint}")
    wanted = {
        "o_proj": "model.layers.0.self_attn.o_proj.weight",
        "down_proj": "model.layers.0.mlp.down_proj.weight",
    }
    weights = {}
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for short, name in wanted.items():
                if name in f.keys():
                    weights[short] = f.get_tensor(name).to(DTYPE)
    assert set(weights) == set(wanted), f"missing {set(wanted) - set(weights)} in {checkpoint}"
    return weights


@requires_cutlass
def test_row_parallel_forward_matches_bf16_qwen3_0p6b():
    """End to end on Qwen3-0.6B ``o_proj``: the two row-parallel ranks' partial outputs add up to
    the BF16 projection."""
    bf16 = _qwen3_0p6b_layer0_weights()["o_proj"]
    w, s = quant_bf16_to_mxfp8(bf16, BLOCK_SIZE)
    out_f, in_f = bf16.shape
    tp_size = 2
    torch.manual_seed(0)
    x = torch.randn(16, in_f, dtype=DTYPE, device="cuda") * 0.1

    total = torch.zeros(16, out_f, dtype=torch.float32, device="cuda")
    for rank in range(tp_size):
        lin = _make_row_linear(in_f, out_f, tp_size, rank)
        lin.load_weights([{"weight": w, "weight_scale_inv": s}])
        cols = slice(rank * in_f // tp_size, (rank + 1) * in_f // tp_size)
        total += lin(x[:, cols].contiguous()).float()

    expected = x.float() @ bf16.float().cuda().t()
    rel_err = (total - expected).norm() / expected.norm()
    assert rel_err < 0.1, f"relative error {rel_err:.3f} against the BF16 projection"
