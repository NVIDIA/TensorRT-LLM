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
"""dsa_fp4_decode_prologue must write, bit for bit, what the separate kernels of forward_dsa_proj's packed path
write: the latent-cache concat, the contiguous MLA query, fused_cat_fp4 of the index query and key, and the FP32
head weights."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.decode_prologue import (
    dsa_fp4_decode_prologue,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_LAT = 1536 + 512 + 64 + 128 + 32  # [q_a | kv_a | k_pe | index k | head weights]


def _reference(pq, num_mla_q, kvn, k_pe, k, w, scale, n_heads):
    import tensorrt_llm._torch.custom_ops  # noqa: F401 - registers torch.ops.trtllm

    latent_cache = torch.concat([kvn, k_pe], dim=-1)
    q, iq = pq.split([num_mla_q, n_heads * 128], -1)
    q = q.contiguous()
    iq = iq.contiguous().view(-1, n_heads, 128)
    q4, qs = torch.ops.trtllm.fused_cat_fp4(*iq.split([64, 64], -1))
    k4, ks = torch.ops.trtllm.fused_cat_fp4(*k.split([64, 64], -1))
    return q, latent_cache, q4, qs, k4, ks, w.float() * scale


def _values(shape, gen, dev):
    """BF16 values spanning zero blocks, FP4 midpoints of power-of-two scales, tiny and large magnitudes."""
    x = torch.randn(shape, generator=gen, device=dev)
    mag = torch.tensor([0.0, 1e-30, 1e-3, 1.0, 7.0, 3e4], device=dev)
    pick = torch.randint(0, len(mag), (shape[0], 1), generator=gen, device=dev)
    x = x * mag[pick]
    # Exact E2M1 midpoints of a block whose amax is 6 * 2^e (scale 2^e): 0.25, 0.75, ..., 5.0 times 2^e.
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -2.5, 6.0], device=dev)
    rows = torch.arange(0, shape[0], 3, device=dev)
    x[rows, : len(mid)] = mid * 4.0
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 6, 24, 48, 96, 112, 127])
@pytest.mark.parametrize("heads_tp", [6, 12])
def test_matches_separate_kernels_bitwise(tokens, heads_tp):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(tokens * 100 + heads_tp)
    n_heads, num_mla_q = 32, heads_tp * 192
    pq = _values((tokens, num_mla_q + n_heads * 128), gen, dev)
    lat = _values(
        (tokens, _LAT + 768), gen, dev
    )  # a packed hidden-state GEMM output with gate columns
    kvn = _values((tokens, 512), gen, dev)
    k_pe = lat[:, 2048:2112]
    k = _values((tokens, 128), gen, dev)
    w = lat[:, 2240:2272]
    scale = 128**-0.5 * n_heads**-0.5
    got = dsa_fp4_decode_prologue(pq, num_mla_q, kvn, k_pe, k, w, scale, n_heads)
    assert got is not None
    want = _reference(pq, num_mla_q, kvn, k_pe, k, w, scale, n_heads)
    names = ("q", "latent_cache", "q_fp4", "q_scale", "k_fp4", "k_scale", "weights")
    for name, g, r in zip(names, got, want):
        assert g.shape == r.shape and g.dtype == r.dtype, name
        assert g.is_contiguous(), name
        assert torch.equal(g.view(torch.uint8), r.contiguous().view(torch.uint8)), name


def _k_cache_case(tokens, gen, dev, bs=64, per_token=68, layers=3, blocks=9):
    """A layer view of a multi-layer uint8 indexer K pool (non-contiguous over blocks) and int64 slot mappings of
    ``tokens`` distinct (block, position) slots, every fifth one -1, padded with junk past ``tokens``."""
    pool = torch.randint(0, 256, (blocks, layers, bs * per_token), generator=gen, device=dev).to(
        torch.uint8
    )
    pick = torch.randperm(blocks * bs, generator=gen, device=dev)[:tokens]
    blk, pos = pick // bs, pick % bs
    slot_k = blk * bs * per_token + pos * 64
    slot_s = blk * bs * per_token + bs * 64 + pos * 4
    slot_k[::5] = -1
    junk = torch.full((7,), 3, dtype=torch.int64, device=dev)
    return pool, torch.cat([slot_k, junk]), torch.cat([slot_s, junk])


def _layer_view(pool, layer=1, bs=64, per_token=68):
    return pool[:, layer].view(pool.shape[0], bs, 1, per_token)


@pytest.mark.parametrize("tokens,num_scatter", [(1, 1), (37, 30), (112, 112), (127, 0)])
def test_k_cache_scatter_matches_scatter_op(tokens, num_scatter):
    """With ``k_cache_scatter`` the kernel leaves the pool exactly as indexer_k_cache_scatter_op does on its
    k_fp4 / k_scale outputs, and its tensor outputs are unchanged."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 - registers torch.ops.trtllm

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(tokens + 7 * num_scatter)
    n_heads, num_mla_q = 32, 12 * 192
    pq = _values((tokens, num_mla_q + n_heads * 128), gen, dev)
    kvn = _values((tokens, 512), gen, dev)
    k_pe = _values((tokens, 64), gen, dev)
    k = _values((tokens, 128), gen, dev)
    w = _values((tokens, 32), gen, dev)
    pool, slot_k, slot_s = _k_cache_case(tokens, gen, dev)
    want_pool = pool.clone()
    got = dsa_fp4_decode_prologue(
        pq,
        num_mla_q,
        kvn,
        k_pe,
        k,
        w,
        0.5,
        n_heads,
        k_cache_scatter=(_layer_view(pool), slot_k, slot_s, num_scatter),
    )
    want = dsa_fp4_decode_prologue(pq, num_mla_q, kvn, k_pe, k, w, 0.5, n_heads)
    assert got is not None and want is not None
    for g, r in zip(got, want):
        assert torch.equal(g.view(torch.uint8), r.view(torch.uint8))
    torch.ops.trtllm.indexer_k_cache_scatter_op(
        want[4], want[5], _layer_view(want_pool), slot_k, slot_s, num_scatter
    )
    assert torch.equal(pool, want_pool)


@pytest.mark.parametrize("tokens", [1, 128, 300])
def test_split_projection_matches_packed(tokens):
    """With ``indexer_q`` (the two-GEMM projection) the kernel returns the MLA query tensor itself and the other
    outputs bit for bit as from the packed GEMM output."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(tokens)
    n_heads, num_mla_q = 32, 6 * 192
    pq = _values((tokens, num_mla_q + n_heads * 128), gen, dev)
    kvn = _values((tokens, 512), gen, dev)
    k_pe = _values((tokens, 64), gen, dev)
    k = _values((tokens, 128), gen, dev)
    w = _values((tokens, 32), gen, dev)
    mla_q = pq[:, :num_mla_q].contiguous()
    iq = pq[:, num_mla_q:].contiguous()
    got = dsa_fp4_decode_prologue(mla_q, num_mla_q, kvn, k_pe, k, w, 0.25, n_heads, indexer_q=iq)
    want = dsa_fp4_decode_prologue(pq, num_mla_q, kvn, k_pe, k, w, 0.25, n_heads)
    assert got is not None and want is not None
    assert got[0] is mla_q
    for g, r in zip(got, want):
        assert torch.equal(g.view(torch.uint8), r.view(torch.uint8))
    assert (
        dsa_fp4_decode_prologue(pq, num_mla_q, kvn, k_pe, k, w, 0.25, n_heads, indexer_q=iq) is None
    )


def test_k_cache_scatter_unserved_forms_return_none():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    x = torch.zeros(4, 1152 + 4096, dtype=torch.bfloat16, device=dev)
    kvn = torch.zeros(4, 512, dtype=torch.bfloat16, device=dev)
    k_pe = torch.zeros(4, 64, dtype=torch.bfloat16, device=dev)
    k = torch.zeros(4, 128, dtype=torch.bfloat16, device=dev)
    w = torch.zeros(4, 32, dtype=torch.bfloat16, device=dev)
    pool, slot_k, slot_s = _k_cache_case(4, gen, dev)
    before = pool.clone()
    cache = _layer_view(pool)
    for bad in (
        (cache.view(torch.int8), slot_k, slot_s, 4),
        (cache[:, :, 0], slot_k, slot_s, 4),
        (cache, slot_k.int(), slot_s, 4),
        (cache, slot_k, slot_s[:3], 4),
        (cache, slot_k, slot_s, 5),
    ):
        assert (
            dsa_fp4_decode_prologue(x, 1152, kvn, k_pe, k, w, 1.0, 32, k_cache_scatter=bad) is None
        )
    assert torch.equal(pool, before)


def test_unserved_layouts_return_none():
    dev = torch.device("cuda")
    x = torch.zeros(4, 1152 + 4096, dtype=torch.bfloat16, device=dev)
    kvn = torch.zeros(4, 512, dtype=torch.bfloat16, device=dev)
    k_pe = torch.zeros(4, 64, dtype=torch.bfloat16, device=dev)
    k = torch.zeros(4, 128, dtype=torch.bfloat16, device=dev)
    w = torch.zeros(4, 32, dtype=torch.bfloat16, device=dev)
    assert dsa_fp4_decode_prologue(x, 1152, kvn, k_pe, k, w, 1.0, 32) is not None
    assert dsa_fp4_decode_prologue(x.float(), 1152, kvn, k_pe, k, w, 1.0, 32) is None
    assert dsa_fp4_decode_prologue(x, 1024, kvn, k_pe, k, w, 1.0, 32) is None
    assert dsa_fp4_decode_prologue(x, 1152, kvn[:, ::2], k_pe, k, w, 1.0, 32) is None
    assert dsa_fp4_decode_prologue(x, 1152, kvn, k_pe, k[:3], w, 1.0, 32) is None


def _packed_fp4_mla(gate_rows: int = 0):
    """A DSA MLA with the K3.1 packed projections (BF16 [kv_a | wk | wp (| g)] and [q_b | wq_b]) and a
    position-free MXFP4 indexer."""
    from tensorrt_llm._utils import get_sm_version

    from .test_short_seq_mha import _build_mla, _init_mla_weights, _make_rope_config

    if get_sm_version() < 100:
        pytest.skip("The FP4 indexer requires SM100+")
    mla, _, _, _ = _build_mla(
        _make_rope_config(),
        torch.device("cuda"),
        threshold=0,
        indexer_k_dtype="fp4",
        index_rope_dim=0,
    )
    _init_mla_weights(mla)
    idx = mla.mqa.indexer
    rows = [
        mla.kv_a_proj_with_mqa.weight.data,
        idx.wk.weight.data.to(torch.bfloat16),
        idx.weights_proj.weight.data.to(torch.bfloat16),
    ]
    if gate_rows:
        rows.append(torch.randn(gate_rows, rows[0].shape[1], device="cuda").to(torch.bfloat16))
    mla.dsa_packed_hidden_weight = torch.cat(rows, 0)
    mla.dsa_packed_q_weight = torch.cat([mla.q_b_proj.weight.data, idx.wq_b.weight.data], 0)
    mla.dsa_packed_gemm_plan = None
    mla.dsa_packed_gate_rows = gate_rows
    return mla


@pytest.mark.parametrize("tokens", [1, 37, 127])
@torch.inference_mode()
def test_forward_dsa_proj_fused_prologue_is_bitwise(monkeypatch, tokens):
    """forward_dsa_proj on the packed path returns the same nine tensors with and without the fused kernel."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import module as dsa_module

    mla = _packed_fp4_mla()
    torch.manual_seed(tokens)
    hidden = torch.randn(tokens, mla.kv_a_proj_with_mqa.weight.shape[1], device="cuda").to(
        torch.bfloat16
    )
    pos = torch.arange(tokens, dtype=torch.int32, device="cuda")
    outs = []
    for fused in (False, True):
        monkeypatch.setattr(dsa_module, "_DSA_FUSED_DECODE_PROLOGUE", fused)
        outs.append(dsa_module.forward_dsa_proj(mla, pos, hidden, None))
    assert len(outs[0]) == len(outs[1]) == 9
    for i, (r, g) in enumerate(zip(*outs)):
        assert g.shape == r.shape and g.dtype == r.dtype, i
        assert torch.equal(g.contiguous().view(torch.uint8), r.contiguous().view(torch.uint8)), i


@torch.inference_mode()
def test_forward_dsa_proj_stashes_the_packed_gate():
    """With g_proj rows in the packed hidden-state weight, forward_dsa_proj leaves (input, gate slice)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import module as dsa_module

    mla = _packed_fp4_mla(gate_rows=256)
    hidden = torch.randn(24, mla.kv_a_proj_with_mqa.weight.shape[1], device="cuda").to(
        torch.bfloat16
    )
    pos = torch.arange(24, dtype=torch.int32, device="cuda")
    dsa_module.forward_dsa_proj(mla, pos, hidden, None)
    x, gate = mla.dsa_packed_gate_out
    assert x is hidden and gate.shape == (24, 256)
    packed = mla.dsa_packed_hidden_weight
    full = torch.nn.functional.linear(hidden, packed)
    assert torch.equal(gate, full[:, packed.shape[0] - 256 :])
    torch.testing.assert_close(
        gate,
        torch.nn.functional.linear(hidden, packed[packed.shape[0] - 256 :]),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("tokens", [1, 37, 127, 128, 300])
@torch.inference_mode()
def test_forward_dsa_proj_indexer_early_is_bitwise(monkeypatch, tokens):
    """``_dsa_indexer_early`` (index-key LayerNorm on the aux stream, keys written by the prologue) returns the
    same nine tensors as the default path and leaves the indexer K cache as _update_k_cache does."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa import module as dsa_module

    mla = _packed_fp4_mla()
    torch.manual_seed(tokens)
    dev = torch.device("cuda")
    hidden = torch.randn(tokens, mla.kv_a_proj_with_mqa.weight.shape[1], device=dev).to(
        torch.bfloat16
    )
    pos = torch.arange(tokens, dtype=torch.int32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(tokens)
    pool, slot_k, slot_s = _k_cache_case(tokens, gen, dev)
    pools = [pool, pool.clone()]

    def metadata(p):
        manager = SimpleNamespace(get_indexer_k_cache_buffers=lambda layer_idx: _layer_view(p))
        return SimpleNamespace(
            kv_cache_manager=manager,
            slot_mapping_fp8=slot_k,
            slot_mapping_scale=slot_s,
            num_tokens=tokens,
            num_generations=tokens,
        )

    outs = []
    for early, p in zip((False, True), pools):
        monkeypatch.setattr(mla, "_dsa_indexer_early", early, raising=False)
        md = metadata(p)
        out = dsa_module.forward_dsa_proj(mla, pos, hidden, md)
        assert mla.dsa_k_cache_written is early
        if not early:
            mla.mqa.indexer._update_k_cache(out[5], out[6], md)
        outs.append(out)
    torch.cuda.synchronize()
    assert len(outs[0]) == len(outs[1]) == 9
    for i, (r, g) in enumerate(zip(*outs)):
        assert g.shape == r.shape and g.dtype == r.dtype, i
        assert torch.equal(g.contiguous().view(torch.uint8), r.contiguous().view(torch.uint8)), i
    assert torch.equal(pools[0], pools[1])
