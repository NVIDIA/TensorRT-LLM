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

from tensorrt_llm._torch.modules.linear import (Linear, MXFP8LinearMethod,
                                                WeightMode,
                                                WeightsLoadingConfig,
                                                get_quant_method)
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

BLOCK_SIZE = 32
DTYPE = torch.bfloat16

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MXFP8 Linear load path requires CUDA")


def _quant_config():
    return QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=BLOCK_SIZE)


def _make_linear(in_features, out_features, weight_mode=WeightMode.VANILLA,
                 shard_mapping=None):
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
def test_vanilla_repeated_bucket_is_idempotent():
    """Replaying the same bucket must not change the buffers: the Linear path
    transforms the incoming source, never the destination."""
    out_f, in_f = 256, 256
    w, s = _mxfp8(out_f, in_f, seed=12)

    lin = _make_linear(in_f, out_f)
    lin.load_weights([{"weight": w, "weight_scale_inv": s}], allow_partial_loading=True)
    lin.process_weights_after_loading()
    first = _snapshot(lin)
    for _ in range(3):
        lin.load_weights([{"weight": w, "weight_scale_inv": s}], allow_partial_loading=True)
        lin.process_weights_after_loading()
    _assert_same(lin, first, "replayed bucket")


@requires_cuda
@pytest.mark.parametrize("order", [("q", "k", "v"), ("v", "k", "q"), ("k", "q", "v")])
def test_fused_qkv_per_shard_buckets(order):
    """One shard per bucket, in several arrival orders."""
    in_f = 256
    q_size, kv_size = 256, 128
    out_f = q_size + 2 * kv_size
    mapping = {"q": (0, q_size), "k": (q_size, kv_size),
               "v": (q_size + kv_size, kv_size)}

    shards = {
        "q": _mxfp8(q_size, in_f, seed=21),
        "k": _mxfp8(kv_size, in_f, seed=22),
        "v": _mxfp8(kv_size, in_f, seed=23),
    }
    full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]}
            for n in ("q", "k", "v")]

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    ref.load_weights(full)
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    index = {"q": 0, "k": 1, "v": 2}
    for name in order:
        bucket = [{}, {}, {}]
        bucket[index[name]] = {"weight": shards[name][0],
                               "weight_scale_inv": shards[name][1]}
        lin.load_weights(bucket, allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, f"qkv order={order}")


@requires_cuda
def test_fused_qkv_weight_and_scale_split_across_buckets():
    """Weights and scales for the same shard arriving in different buckets."""
    in_f = 256
    q_size, kv_size = 256, 128
    out_f = q_size + 2 * kv_size
    mapping = {"q": (0, q_size), "k": (q_size, kv_size),
               "v": (q_size + kv_size, kv_size)}
    shards = {n: _mxfp8(sz, in_f, seed=s) for n, sz, s in
              (("q", q_size, 31), ("k", kv_size, 32), ("v", kv_size, 33))}

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    ref.load_weights([{"weight": shards[n][0], "weight_scale_inv": shards[n][1]}
                      for n in ("q", "k", "v")])
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    lin.load_weights([{"weight": shards[n][0]} for n in ("q", "k", "v")],
                     allow_partial_loading=True)
    lin.load_weights([{"weight_scale_inv": shards[n][1]} for n in ("q", "k", "v")],
                     allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, "qkv weights-then-scales")


@requires_cuda
@pytest.mark.parametrize("order", [("gate", "up"), ("up", "gate")])
def test_fused_gate_up_per_shard_buckets(order):
    in_f = 256
    inter = 256  # 128-aligned
    out_f = 2 * inter
    mapping = {"gate": (0, inter), "up": (inter, inter)}
    shards = {"gate": _mxfp8(inter, in_f, seed=41),
              "up": _mxfp8(inter, in_f, seed=42)}

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_GATE_UP_LINEAR, mapping)
    ref.load_weights([{"weight": shards[n][0], "weight_scale_inv": shards[n][1]}
                      for n in ("gate", "up")])
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_GATE_UP_LINEAR, mapping)
    index = {"gate": 0, "up": 1}
    for name in order:
        bucket = [{}, {}]
        bucket[index[name]] = {"weight": shards[name][0],
                               "weight_scale_inv": shards[name][1]}
        lin.load_weights(bucket, allow_partial_loading=True)
    lin.process_weights_after_loading()
    _assert_same(lin, expected, f"gate_up order={order}")


@requires_cuda
def test_partial_loading_is_forwarded_to_mxfp8_method():
    """Guard the dispatcher allowlist in LinearMethodBase.load_weights: if
    MXFP8LinearMethod is not listed there the kwarg is silently dropped and the
    method runs its full-load asserts instead."""
    import inspect

    from tensorrt_llm._torch.modules.linear import LinearMethodBase

    method = get_quant_method(_quant_config())
    assert isinstance(method, MXFP8LinearMethod)
    for fn in (method.load_weights_vanilla,
               method.load_weights_fused_qkv_linear,
               method.load_weights_fused_gate_up_linear):
        assert "allow_partial_loading" in inspect.getfullargspec(fn).args, (
            f"{fn.__name__} must accept allow_partial_loading")

    # There are TWO independent isinstance allowlists on the path and both must
    # list MXFP8LinearMethod: Linear.load_weights asserts partial loading is
    # unsupported, and LinearMethodBase.load_weights decides whether to forward
    # the kwarg at all. Missing the first raises; missing the second silently
    # drops the flag and takes the full-load path.
    for owner, fn in ((Linear, Linear.load_weights),
                      (LinearMethodBase, LinearMethodBase.load_weights)):
        src = inspect.getsource(fn)
        assert "MXFP8LinearMethod" in src, (
            f"{owner.__name__}.load_weights must allow MXFP8LinearMethod "
            f"through its partial-loading allowlist")


@requires_cuda
@pytest.mark.parametrize("q_size,kv_size", [(64, 64), (256, 64), (192, 96)])
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
    mapping = {"q": (0, q_size), "k": (q_size, kv_size),
               "v": (q_size + kv_size, kv_size)}
    shards = {n: _mxfp8(sz, in_f, seed=s) for n, sz, s in
              (("q", q_size, 51), ("k", kv_size, 52), ("v", kv_size, 53))}
    full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]}
            for n in ("q", "k", "v")]

    ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    ref.load_weights(full)
    expected = _snapshot(ref)

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    for i, name in enumerate(("q", "k", "v")):
        bucket = [{}, {}, {}]
        bucket[i] = {"weight": shards[name][0],
                     "weight_scale_inv": shards[name][1]}
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
    mapping = {"q": (0, q_size), "k": (q_size, kv_size),
               "v": (q_size + kv_size, kv_size)}

    lin = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
    for round_seed in (61, 71):
        shards = {n: _mxfp8(sz, in_f, seed=round_seed + i) for i, (n, sz) in
                  enumerate((("q", q_size), ("k", kv_size), ("v", kv_size)))}
        full = [{"weight": shards[n][0], "weight_scale_inv": shards[n][1]}
                for n in ("q", "k", "v")]

        ref = _make_linear(in_f, out_f, WeightMode.FUSED_QKV_LINEAR, mapping)
        ref.load_weights(full)
        expected = _snapshot(ref)

        lin.pre_reload_weights()
        for i, name in enumerate(("q", "k", "v")):
            bucket = [{}, {}, {}]
            bucket[i] = {"weight": shards[name][0],
                         "weight_scale_inv": shards[name][1]}
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

    lin.load_weights([{"weight": w, "weight_scale_inv": s}],
                     allow_partial_loading=True)
    assert hasattr(lin, "tmp_weight_scale_raw"), (
        "staging copy should exist between a partial bucket and finalize")

    lin.process_weights_after_loading()
    assert not hasattr(lin, "tmp_weight_scale_raw"), (
        "staging copy must be released by process_weights_after_loading")

    # A full (non-partial) load finalizes internally, so it must not leak one.
    lin2 = _make_linear(in_f, out_f)
    lin2.load_weights([{"weight": w, "weight_scale_inv": s}])
    assert not hasattr(lin2, "tmp_weight_scale_raw")


@requires_cuda
def test_repeated_finalize_does_not_reswizzle():
    """process_weights_after_loading is invoked by both the RLHF finalize walk
    and post_load_weights; the pending flag must make the second call a no-op
    (the swizzle is not an involution)."""
    out_f, in_f = 256, 256
    w, s = _mxfp8(out_f, in_f, seed=81)

    lin = _make_linear(in_f, out_f)
    lin.load_weights([{"weight": w, "weight_scale_inv": s}],
                     allow_partial_loading=True)
    lin.process_weights_after_loading()
    first = _snapshot(lin)
    for _ in range(3):
        lin.process_weights_after_loading()
    _assert_same(lin, first, "repeated finalize")
