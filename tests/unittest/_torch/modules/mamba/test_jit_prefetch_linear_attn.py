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
"""The GDN / KDA Triton planner must predict the real launches' cache keys.

Runs the real launchers on CUDA tensors laid out as GatedDeltaNet and
KimiKDALinearAttention lay them out, records their Triton launches, and
checks every real cache key is among the planned ones (meta-tensor replay).
"""

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.modules.mamba import jit_prefetch_linear_attn as la

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _keys(calls):
    """Triton cache keys of the given launches, computed as JITFunction.run does."""
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


def _gdn(k_heads=4, v_heads=8, k_dim=128, v_dim=128, fp8=False):
    s = la._GdnShape.__new__(la._GdnShape)
    s.k_heads, s.v_heads, s.k_dim, s.v_dim = k_heads, v_heads, k_dim, v_dim
    s.conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
    s.dtype = torch.bfloat16
    s.qkvz_width = s.conv_dim + v_heads * v_dim
    s.fp8_norm = fp8
    return s


def _kda(heads=12, head_dim=128):
    s = la._KdaShape.__new__(la._KdaShape)
    s.heads, s.head_dim, s.head_k_dim, s.dtype = heads, head_dim, head_dim, torch.bfloat16
    return s


def _provider(shapes):
    p = la.LinearAttnProvider.__new__(la.LinearAttnProvider)
    p.shapes = shapes
    p.max_num_tokens = 8192
    p.device_index = torch.cuda.current_device()
    p._seen = set()
    p._num_sms = None
    p.chunk_size = 64
    p.max_batch_size = 0
    p.kda_staged = False
    return p


def _real_gdn(s, p, d):
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
        extract_transpose_prefill_slice,
        fused_gdn_post_conv,
    )
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    dev = torch.device("cuda")
    t = p + d
    qkvz = torch.randn(t, s.qkvz_width, dtype=s.dtype, device=dev)
    ba = torch.randn(t, 2 * s.v_heads, dtype=s.dtype, device=dev)
    mixed_qkv = qkvz[:, : s.conv_dim]
    z = qkvz[:, s.conv_dim :].view(t, s.v_heads, s.v_dim)
    b, a = ba[:, : s.v_heads], ba[:, s.v_heads :]
    A_log = torch.randn(s.v_heads, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(s.v_heads, dtype=torch.float32, device=dev)
    with jp.shadow_launches() as rec:
        if p > 0:
            src = mixed_qkv[:p] if d > 0 else mixed_qkv
            extract_transpose_prefill_slice(src, src.shape[0], 0, src.shape[1])
            conv_p = torch.empty(p, s.conv_dim, dtype=s.dtype, device=dev).t()
            conv_d = torch.empty(d, s.conv_dim, dtype=s.dtype, device=dev) if d > 0 else None
            fused_gdn_post_conv(conv_p, conv_d, a, b, A_log, dt_bias, s.k_heads, s.k_dim,
                                s.v_heads, s.v_dim)  # fmt: skip
        x = torch.randn(t * s.v_heads, s.v_dim, dtype=s.dtype, device=dev)
        w = torch.randn(s.v_dim, dtype=s.dtype, device=dev)
        scale = torch.ones((), dtype=torch.float32, device=dev) if s.fp8_norm else None
        rms_norm_gated_token_major._init_fn(x, z, w, 1e-6, fp8_scale=scale)
    return list(rec)


def _real_kda(s, t, token_major=False):
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import fused_kda_post_conv
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
        extract_transpose_prefill_slice,
    )
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    dev = torch.device("cuda")
    proj = s.heads * s.head_dim
    # KimiKDALinearAttention._project_packed_conv_input with the fused FP8
    # qkvg projection: [T, 4 * proj], the [q | k | v] columns repacked
    # channel-major; with the BF16 qkvg GEMM on an unaligned token count, a
    # transposed view of the GEMM's token-major rows instead.
    qkvg = torch.randn(t, 4 * proj, dtype=s.dtype, device=dev)
    g = torch.randn(t, proj, dtype=s.dtype, device=dev).view(t, s.heads, s.head_dim)
    x = torch.randn(t * s.heads, s.head_dim, dtype=s.dtype, device=dev)
    w = torch.randn(s.head_dim, dtype=s.dtype, device=dev)
    conv_w = torch.randn(3 * proj, 4, dtype=s.dtype, device=dev)
    query_start_loc = torch.tensor([0, t], dtype=torch.int32, device=dev)
    cache_indices = torch.zeros(1, dtype=torch.int32, device=dev)
    has_init = torch.zeros(1, dtype=torch.bool, device=dev)
    conv_states = torch.zeros(1, 3 * proj, 3, dtype=s.dtype, device=dev)
    with jp.shadow_launches() as rec:
        if token_major:
            packed = qkvg[:, : 3 * proj].contiguous().t()
        else:
            packed = extract_transpose_prefill_slice(qkvg, t, 0, 3 * proj)
        # The real convolution keeps its input's layout (channel-last in,
        # token-major out); it is not a Triton kernel, so it records nothing.
        packed = causal_conv1d_fn(packed, conv_w, query_start_loc=query_start_loc,
                                  cache_indices=cache_indices, has_initial_state=has_init,
                                  conv_states=conv_states, activation="silu")  # fmt: skip
        fused_kda_post_conv(packed, num_heads=s.heads, head_dim=s.head_dim)
        rms_norm_gated_token_major._init_fn(x, g, w, 1e-6, gate_activation="sigmoid")
    return list(rec)


# Prefill-only and mixed batches across the integer classes, plus sizes on
# both sides of the PDL threshold (tokens * heads / 4 vs the SM count).
_BATCHES = [([100], 0), ([1], 0), ([16], 0), ([64, 37], 0), ([100], 7), ([32], 16),
            ([2000], 0), ([4000, 96], 33)]  # fmt: skip


@pytest.mark.parametrize("ctx_lens,gen", _BATCHES)
@pytest.mark.parametrize("fp8", [False, True])
def test_gdn_planned_keys_cover_real_launches(ctx_lens, gen, fp8):
    s = _gdn(fp8=fp8)
    p = _provider([s])
    batch = type("B", (), {"ctx_chunk_lens": ctx_lens, "gen_tokens": gen})()
    planned = _keys(p(batch))
    real = _keys(_real_gdn(s, sum(ctx_lens), gen))
    assert real, "no real launches recorded"
    assert real <= planned, f"unplanned: {sorted(real - planned)}"


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("t", [1, 7, 16, 100, 2000, 6000, 8134])
def test_kda_planned_keys_cover_real_launches(t, token_major):
    s = _kda()
    p = _provider([s])
    batch = type("B", (), {"ctx_chunk_lens": [t], "gen_tokens": 0})()
    planned = _keys(p(batch))
    real = _keys(_real_kda(s, t, token_major))
    names = {
        "_fused_kda_post_conv_kernel",
        "_rms_norm_gated_fwd_multirow_kernel",
    }
    if not token_major:
        names.add("_extract_transpose_prefill_kernel")
    assert {name for name, _ in real} == names, real
    assert real <= planned, f"unplanned: {sorted(real - planned)}"


def test_enumeration_covers_pdl_threshold():
    s = _gdn()
    p = _provider([s])
    pdl = {p.batch_key(b.ctx_chunk_lens, b.gen_tokens)[5] for b in p.enumerate_batches()}
    assert {(True,), (False,)} <= pdl


def _real_chunk_indices(ctx_lens, chunk_size):
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
        compute_extra_chunks_cpu,
        cu_seqlens_to_chunk_indices_offsets_triton,
    )

    cu = torch.tensor([0, *torch.tensor(ctx_lens).cumsum(0).tolist()], dtype=torch.int,
                      device="cuda")  # fmt: skip
    with jp.shadow_launches() as rec:
        cu_seqlens_to_chunk_indices_offsets_triton(
            cu, chunk_size, total_seqlens=sum(ctx_lens),
            extra_chunks=compute_extra_chunks_cpu(ctx_lens, len(ctx_lens), chunk_size),
        )  # fmt: skip
    return list(rec)


# Cached-prefix multi-sequence prefill (Mamba2Metadata.prepare's chunk-index
# kernel), across num_seqs and chunk-count classes.
@pytest.mark.parametrize("ctx_lens", [[64, 64], [100, 37, 200], [1] * 17, [64] * 16, [65] * 33])
def test_chunk_index_kernel_planned(ctx_lens):
    p = _provider([_kda()])
    batch = type("B", (), {"ctx_chunk_lens": ctx_lens, "gen_tokens": 0, "any_ctx_cached": True})()
    planned = _keys(p(batch))
    real = _keys(_real_chunk_indices(ctx_lens, p.chunk_size))
    assert real and real <= planned, f"unplanned: {sorted(real - planned)}"


def test_provider_finds_real_gdn_and_kda_layers():
    """Built from a model's modules (not __new__), as model_engine builds it."""
    from transformers import Qwen3NextConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

    gdn_cfg = Qwen3NextConfig(
        hidden_size=256,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        torch_dtype=torch.bfloat16,
    )
    gdn = Qwen3NextGatedDeltaNet(ModelConfig(pretrained_config=gdn_cfg), None, layer_idx=0)
    kda_cfg = type(
        "Cfg",
        (),
        dict(hidden_size=256, rms_norm_eps=1e-6,
             linear_attn_config=dict(num_heads=4, head_dim=128, short_conv_kernel_size=4,
                                     use_full_rank_gate=True, gate_lower_bound=-5.0)),
    )()  # fmt: skip
    kda = KimiKDALinearAttention(kda_cfg, layer_idx=1)
    model = torch.nn.ModuleList([gdn, kda])

    p = la.LinearAttnProvider(model, max_num_tokens=256, chunk_size=64)
    assert sorted(type(s).__name__ for s in p.shapes) == ["_GdnShape", "_KdaShape"]
    gdn_shape = next(s for s in p.shapes if isinstance(s, la._GdnShape))
    assert (gdn_shape.k_heads, gdn_shape.v_heads, gdn_shape.k_dim, gdn_shape.v_dim) == (
        4,
        8,
        64,
        64,
    )
    kda_shape = next(s for s in p.shapes if isinstance(s, la._KdaShape))
    assert (kda_shape.heads, kda_shape.head_dim) == (4, 128)
    batch = type("B", (), {"ctx_chunk_lens": [100], "gen_tokens": 0})()
    assert p(batch), "no launches planned for a prefill batch"


def _real_kda_reset(s, n):
    """KimiKDALinearAttention.forward_prefill's reset on a BF16 state cache:
    the batch's rows staged into a dense FP32 copy, addressed by position."""
    from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

    pool = torch.zeros(
        n + 7, s.heads, s.head_dim, s.head_k_dim, dtype=torch.bfloat16, device="cuda"
    )
    slots = torch.randperm(n + 7, device="cuda")[:n]
    staged = pool.index_select(0, slots).float()
    arange = torch.arange(64 + 1, dtype=torch.int, device="cuda")
    has_init = torch.zeros(64, dtype=torch.bool, device="cuda")[:n]
    with jp.shadow_launches() as rec:
        reset_recurrent_state_rows(staged, arange[:n], has_init)
    return list(rec)


@pytest.mark.parametrize("n", [1, 2, 3, 16, 17])
def test_kda_staged_reset_enumerated_per_request_count(n):
    """The staged state pool has one row per context request, a constexpr of
    the reset kernel: the enumeration plans every request count up to
    max_batch_size."""
    s = _kda()
    p = _provider([s])
    p.max_batch_size, p.kda_staged = 32, True
    real = _keys(_real_kda_reset(s, n))
    assert {name for name, _ in real} == {"_reset_recurrent_state_rows_kernel"}, real
    enumerated = set()
    for b in p.enumerate_batches():
        enumerated |= _keys(p(b, seen=set()))
    assert real <= enumerated, f"unplanned: {sorted(real - enumerated)}"


def test_kda_reset_not_planned_for_fp32_state():
    p = la.LinearAttnProvider(torch.nn.Module(), max_num_tokens=256, max_batch_size=8,
                              ssm_state_dtype=torch.float32)  # fmt: skip
    p.shapes = [_kda()]
    names = {c.label for b in p.enumerate_batches() for c in p(b, seen=set())}
    assert names and "_reset_recurrent_state_rows_kernel" not in names


def test_enumeration_covers_chunk_index_classes():
    p = _provider([_kda()])
    planned = set()
    for b in p.enumerate_batches():
        planned |= _keys(p(b, seen=set()))
    for lens in ([64, 64], [100, 37, 200], [65] * 33):
        real = _keys(_real_chunk_indices(lens, p.chunk_size))
        assert real <= planned, f"{lens}: unplanned {sorted(real - planned)}"
