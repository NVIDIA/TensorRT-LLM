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
    s.heads, s.head_dim, s.dtype = heads, head_dim, torch.bfloat16
    return s


def _provider(shapes):
    p = la.LinearAttnProvider.__new__(la.LinearAttnProvider)
    p.shapes = shapes
    p.max_num_tokens = 8192
    p.device_index = torch.cuda.current_device()
    p._seen = set()
    p._num_sms = None
    p.chunk_size = 64
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


def _real_kda(s, t):
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    dev = torch.device("cuda")
    g = torch.randn(t, s.heads * s.head_dim, dtype=s.dtype, device=dev).view(t, s.heads, s.head_dim)
    x = torch.randn(t * s.heads, s.head_dim, dtype=s.dtype, device=dev)
    w = torch.randn(s.head_dim, dtype=s.dtype, device=dev)
    with jp.shadow_launches() as rec:
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


@pytest.mark.parametrize("t", [1, 7, 16, 100, 2000, 6000])
def test_kda_planned_keys_cover_real_launches(t):
    s = _kda()
    p = _provider([s])
    batch = type("B", (), {"ctx_chunk_lens": [t], "gen_tokens": 0})()
    planned = _keys(p(batch))
    real = _keys(_real_kda(s, t))
    assert real and real <= planned, f"unplanned: {sorted(real - planned)}"


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


def test_enumeration_covers_chunk_index_classes():
    p = _provider([_kda()])
    planned = set()
    for b in p.enumerate_batches():
        planned |= _keys(p(b, seen=set()))
    for lens in ([64, 64], [100, 37, 200], [65] * 33):
        real = _keys(_real_chunk_indices(lens, p.chunk_size))
        assert real <= planned, f"{lens}: unplanned {sorted(real - planned)}"
