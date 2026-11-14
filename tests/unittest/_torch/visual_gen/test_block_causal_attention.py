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
"""Attention backends over PagedRolloutKVCache against a dense SDPA reference.

Random inputs; the reference keeps its own dense copy of every K/V ever written
and attends over exactly what the model's window allows: the fixed region (the
prompt), the ``WINDOW`` tokens before each causal block, the earlier causal blocks
of the chunk and the causal block itself. Stale tokens the whole-page eviction keeps
resident must not be seen.
"""

import weakref
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from utils.block_causal_reference import exact_reference as _exact_reference
from utils.block_causal_reference import reference_attention

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.attention import Attention
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import PagedRolloutKVCache, RolloutSpec
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.visual_gen.args import AttentionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
PROMPT_CAPACITY = 64
WINDOW = 64
CHUNK = 40  # not a page multiple: exercises partial pages and stale tokens


def spec_for(max_pin_tokens=PROMPT_CAPACITY, causal_blocks=None, window_tokens=WINDOW, chunk=CHUNK):
    """The tests' spec: a denoising step over the whole chunk and four causal blocks of
    a quarter chunk, unless ``causal_blocks`` says otherwise."""
    return RolloutSpec(
        window_tokens=window_tokens,
        max_staged_tokens=chunk,
        causal_blocks=causal_blocks or {chunk: 1, chunk // 4: 4},
        max_pin_tokens=max_pin_tokens,
    )


def make_cache(tokens_per_page, pin_tokens=PROMPT_CAPACITY, spec=None, num_layers=1, dtype=DTYPE):
    """A cache whose pool is sized for ``spec`` (default: the tests' spec with up to
    ``pin_tokens`` pinned). The spec rides on the cache for ``open_with_prompt``."""
    spec = spec or spec_for(max_pin_tokens=pin_tokens)
    cache = PagedRolloutKVCache(
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        tokens_per_page=tokens_per_page,
        pool_tokens=PagedRolloutKVCache.pool_tokens_for(spec, tokens_per_page=tokens_per_page),
    )
    cache.test_spec = spec
    return cache


exact_reference = partial(_exact_reference, window=WINDOW)


@pytest.fixture(params=[32, 64], ids=["tpb32", "tpb64"])
def cache(request):
    mgr = make_cache(request.param)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def make_backend(name, layer_idx=0, dtype=DTYPE):
    if name == "trtllm":
        return TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=dtype,
            attention_metadata_state={},
        )
    if name == "flashinfer":
        return FlashInferAttention(
            layer_idx=layer_idx,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=dtype,
        )
    if name == "cudnn":
        return CuDNNAttention(
            layer_idx=layer_idx,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=dtype,
        )
    return CuTeDSLAttention(
        layer_idx=layer_idx,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
    )


def run(attn, cache, q, k, v, causal_block_size=None):
    """Per-token [T, H, D] in, per-token [T, H, D] out, whichever backend."""
    out = attn.forward(
        q[None],
        k[None],
        v[None],
        batch_size=1,
        seq_len=q.shape[0],
        kv=cache.plan(q.shape[0], causal_block_size),
    )
    return out.reshape(q.shape[0], NUM_HEADS, HEAD_DIM)


def read_kv(cache, layer, positions):
    """Gather ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions`` straight from the pool."""
    buf = cache.kv_buffer(layer)
    page, slot = (torch.from_numpy(x).to(buf.device) for x in cache.locate(positions.cpu()))
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def rand_qkv(n, dtype=DTYPE):
    q = torch.randn(n, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    return q, k, torch.randn_like(k)


def open_with_prompt(cache, prompt_len):
    """Open the cache's spec pinning ``prompt_len`` tokens, write a random prompt at
    position 0 and commit it; returns its K and V."""
    cache.open(cache.test_spec, pin_tokens=prompt_len)
    pk, pv = rand_qkv(prompt_len)[1:]
    cache.write_range(0, 0, pk, pv)
    if prompt_len:
        cache.commit(prompt_len)
    return pk, pv


BACKENDS = [
    "cudnn",
    "trtllm",
    pytest.param(
        "cutedsl",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or get_sm_version() not in (100, 103),
            reason="the CuTe DSL kernel runs on SM100/SM103",
        ),
    ),
    pytest.param(
        "flashinfer",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or get_sm_version() // 10 != 10,
            reason="the TRTLLM-Gen paged kernel runs on SM10x",
        ),
    ),
]


def lacks_kernel(backend, cache, dtype=DTYPE):
    """Whether the backend has no attention kernel for the cache's page size on this GPU."""
    return not make_backend(backend, dtype=dtype).support_kv_cache_page_size(
        cache.tokens_per_page, dtype
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "prompt_len", [0, 17, 64], ids=["no-prompt", "unaligned-prompt", "page-exact-prompt"]
)
def test_rollout_matches_dense_reference(cache, backend, prompt_len):
    torch.manual_seed(0)
    prompt_k, prompt_v = open_with_prompt(cache, prompt_len)
    attn = make_backend(backend)
    if lacks_kernel(backend, cache):
        pytest.skip("no paged attention kernel for this page size")

    history_k, history_v = [], []  # the reference's dense copy of committed generator K/V
    empty = prompt_k.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
    saw_stale = False
    for step in range(8):  # long enough to recycle every resident page
        q, k, v = rand_qkv(CHUNK)
        out = run(attn, cache, q, k, v)
        torch.cuda.synchronize()

        saw_stale |= cache.history_tokens > WINDOW
        hk = torch.cat(history_k) if history_k else empty
        hv = torch.cat(history_v) if history_v else empty
        expected = exact_reference(q, prompt_k, prompt_v, hk, hv, k, v, 0, CHUNK)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")
        if step > 0 or prompt_len > 0:
            chunk_only = reference_attention(q, k, v)
            assert (out.float() - chunk_only.float()).abs().max() > 1e-2, (
                f"step {step}: output equals chunk-only attention, the cached prefix was ignored"
            )

        # The call wrote this chunk's K/V where the next forward expects them.
        positions = torch.arange(cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE)
        k_back, v_back = read_kv(cache, 0, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)

        cache.commit(cache.max_staged_tokens)
        history_k.append(k)
        history_v.append(v)
    assert saw_stale, "test geometry should hold stale tokens at some step"


def check_causal_blocks(attn, cache, prompt_k, prompt_v, hk, hv, num_causal_blocks, chunk):
    """One call cut into causal blocks: causal block i sees the prompt, its own window
    of history, the causal blocks before it and itself. ``hk``/``hv``: all committed history."""
    causal_block_size = chunk // num_causal_blocks
    q, k, v = rand_qkv(chunk)
    out = run(attn, cache, q, k, v, causal_block_size=causal_block_size)
    torch.cuda.synchronize()
    for i in range(num_causal_blocks):
        lo, hi = i * causal_block_size, (i + 1) * causal_block_size
        expected = exact_reference(q, prompt_k, prompt_v, hk, hv, k, v, lo, hi)
        torch.testing.assert_close(
            out[lo:hi], expected, rtol=2e-2, atol=2e-2, msg=f"causal block {i}"
        )
        if i < num_causal_blocks - 1:  # a causal block must not see the causal blocks after it
            leaky = reference_attention(
                q[lo:hi], torch.cat([prompt_k, hk, k]), torch.cat([prompt_v, hv, v])
            )
            assert (out[lo:hi].float() - leaky.float()).abs().max() > 1e-2, (
                f"causal block {i} leaks"
            )
    positions = torch.arange(cache.staging_offset, cache.staging_offset + chunk, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, k)
    torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        run(attn, cache, q, k, v, causal_block_size=7)


@pytest.fixture(params=[(4, 2, 64), (32, 8, 128)], ids=["d64_gqa2", "d128_gqa4"])
def head_shape(request, monkeypatch):
    """Query heads, K/V heads and head_dim for the helpers above: the small default, and
    the Cosmos3 shape (head_dim 128, 4:1 grouping), which selects different kernels."""
    for name, value in zip(("NUM_HEADS", "NUM_KV_HEADS", "HEAD_DIM"), request.param):
        monkeypatch.setitem(globals(), name, value)
    return request.param


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("num_causal_blocks", [1, 4, 8])
@pytest.mark.parametrize("tokens_per_page", [32, 64], ids=["tpb32", "tpb64"])
def test_every_layer_and_library_over_a_multi_layer_cache(
    backend, dtype, num_causal_blocks, tokens_per_page, head_shape
):
    """A 3-layer cache with different K/V in every layer, attended on every layer
    while pages are recycled, with 1, 4 and 8 causal blocks in BF16 and FP16, at two
    head shapes. Each layer must read its own pages. For cuDNN every layer shares one
    graph."""
    torch.manual_seed(11)
    num_layers, prompt_len = 3, 17
    spec = spec_for(max_pin_tokens=prompt_len, causal_blocks={CHUNK: 1, 10: 4, 5: 8})
    cache = make_cache(tokens_per_page, spec=spec, num_layers=num_layers, dtype=dtype)
    try:
        if lacks_kernel(backend, cache, dtype):
            pytest.skip("no paged context kernel for this page size in this build")
        cache.open(spec, pin_tokens=prompt_len)
        prompt = [rand_qkv(prompt_len, dtype)[1:] for _ in range(num_layers)]
        for layer, (pk, pv) in enumerate(prompt):
            cache.write_range(layer, 0, pk, pv)
        cache.commit(prompt_len)
        attns = [make_backend(backend, layer, dtype) for layer in range(num_layers)]
        if backend == "cudnn":
            CuDNNAttention.clear_graph_cache()
        history = [([], []) for _ in range(num_layers)]
        empty = prompt[0][0].new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        size = CHUNK // num_causal_blocks
        saw_stale = False
        for step in range(4):  # recycles pages and leaves stale tokens
            saw_stale |= cache.history_tokens > WINDOW
            for layer, attn in enumerate(attns):
                q, k, v = rand_qkv(CHUNK, dtype)
                out = attn.forward(
                    q[None],
                    k[None],
                    v[None],
                    batch_size=1,
                    seq_len=CHUNK,
                    kv=cache.plan(CHUNK, size),
                ).reshape(CHUNK, NUM_HEADS, HEAD_DIM)
                torch.cuda.synchronize()
                hk = torch.cat(history[layer][0]) if history[layer][0] else empty
                hv = torch.cat(history[layer][1]) if history[layer][1] else empty
                pk, pv = prompt[layer]
                for i in range(num_causal_blocks):
                    lo, hi = i * size, (i + 1) * size
                    expected = exact_reference(q, pk, pv, hk, hv, k, v, lo, hi)
                    torch.testing.assert_close(
                        out[lo:hi],
                        expected,
                        rtol=2e-2,
                        atol=2e-2,
                        msg=f"step {step} layer {layer} causal block {i}",
                    )
                history[layer][0].append(k)
                history[layer][1].append(v)
            cache.commit(CHUNK)
        assert saw_stale, "test geometry should hold stale tokens at some step"
        if backend == "cudnn":
            assert len(CuDNNAttention._graph_cache) == 1, "every layer should share one graph"
    finally:
        cache.shutdown()


@pytest.mark.parametrize("backend", BACKENDS)
def test_causal_blocks_at_any_alignment(cache, backend):
    """The clean pass: one launch, four causal blocks of 10 sharing pages mid-way,
    each with its own window start, over history with stale tokens."""
    torch.manual_seed(3)
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    pk, pv = open_with_prompt(cache, 9)
    history_k, history_v = [], []
    for _ in range(3):  # 120 tokens committed, one page dropped: 88 resident, 24 stale
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.staging_offset, k, v)
        cache.commit(cache.max_staged_tokens)
        history_k.append(k)
        history_v.append(v)
    assert cache.history_tokens > WINDOW, "test geometry should hold stale tokens here"
    check_causal_blocks(
        make_backend(backend),
        cache,
        pk,
        pv,
        torch.cat(history_k),
        torch.cat(history_v),
        num_causal_blocks=4,
        chunk=CHUNK,
    )


def indicator_values(watch, first, count):
    """V for tokens ``[first, first + count)`` of the sequence: 1 in dimension ``d`` of
    K/V head ``h`` for the token ``watch[h][d]``, 0 everywhere else."""
    ids = torch.arange(first, first + count, device=DEVICE)
    return (ids[:, None, None] == watch[None]).to(DTYPE)


@pytest.mark.parametrize("commits, sink", [(0, 0), (1, 0), (3, 0), (3, 12)])
@pytest.mark.parametrize("num_causal_blocks", [1, 4])
@pytest.mark.parametrize("backend", BACKENDS)
def test_each_block_sees_exactly_its_window(cache, backend, num_causal_blocks, commits, sink):
    """Exact visible-key sets, no tolerance games: with q = 0 every visible key gets
    the same weight 1/N, so with V one-hot per watched token the output is
    ``[token visible] / N``. An off-by-one window edge, a stale or evicted token
    read, or a key read twice shows up as a 1, a 0 or a 2 where it should not.
    Covers the window filling (0 and 1 commits) and full (3), and ``sink`` generated
    tokens pinned with the prompt by the first commit."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    torch.manual_seed(7)
    prompt = 9
    fixed = prompt + (sink if commits else 0)
    hist = commits * CHUNK
    total = prompt + hist + CHUNK
    # Watch the prompt, up to the last 79 committed tokens (window edges of every
    # causal block and stale tokens), the whole chunk, and ids that never exist: one per
    # (K/V head, dimension).
    ids = torch.cat(
        [
            torch.arange(prompt),
            torch.arange(prompt + max(0, hist - 79), prompt + hist),
            torch.arange(prompt + hist, total),
        ]
    )
    ids = torch.cat([ids, torch.arange(total, total + NUM_KV_HEADS * HEAD_DIM - ids.numel())])
    watch = ids.to(DEVICE).view(NUM_KV_HEADS, HEAD_DIM)

    def keys(n):
        return torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    cache.open(cache.test_spec, pin_tokens=prompt + sink)
    cache.write_range(0, 0, keys(prompt), indicator_values(watch, 0, prompt))
    cache.commit(prompt)
    for c in range(commits):
        first = prompt + c * CHUNK
        cache.write_range(
            0, cache.staging_offset, keys(CHUNK), indicator_values(watch, first, CHUNK)
        )
        cache.commit(cache.max_staged_tokens)
    assert cache.fixed_tokens == fixed

    size = CHUNK // num_causal_blocks
    q = torch.zeros(CHUNK, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = indicator_values(watch, prompt + hist, CHUNK)
    out = run(make_backend(backend), cache, q, keys(CHUNK), v, causal_block_size=size)
    torch.cuda.synchronize()

    token = torch.arange(CHUNK, device=DEVICE)
    start = token // size * size  # each query's causal block start within the chunk
    end = start + size
    # Ids are sequence positions before any eviction: fixed [0, fixed), then history.
    g = watch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0)[None]  # [1, H, D]
    block_start = (prompt + hist + start)[:, None, None]
    win_start = torch.clamp(block_start - WINDOW, min=fixed)
    visible = (g < fixed) | ((g >= win_start) & (g < (prompt + hist + end)[:, None, None]))
    num_visible = fixed + (block_start - win_start) + size
    seen = out.float() * num_visible
    # One key too many or too few moves a watched value by 1/N of itself, N <= 113;
    # bf16 rounding of 1/N is far below that.
    assert torch.allclose(seen, visible.float(), atol=0.004), (
        f"{(seen.round() != visible.float()).sum().item()} (query, key) pairs wrong"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_tokens_are_neither_written_nor_attended(cache, backend):
    """The plan counts the real tokens: a padded chunk behaves like the unpadded one."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    torch.manual_seed(6)
    pk, pv = open_with_prompt(cache, 9)
    attn = make_backend(backend)
    pad = 8
    q, k, v = rand_qkv(CHUNK + pad)
    out = attn.forward(
        q[None],
        k[None],
        v[None],
        batch_size=1,
        seq_len=CHUNK + pad,
        kv=cache.plan(CHUNK),
    ).reshape(CHUNK + pad, NUM_HEADS, HEAD_DIM)
    torch.cuda.synchronize()
    expected = reference_attention(
        q[:CHUNK], torch.cat([pk, k[:CHUNK]]), torch.cat([pv, v[:CHUNK]])
    )
    torch.testing.assert_close(out[:CHUNK], expected, rtol=2e-2, atol=2e-2)
    assert out[CHUNK:].abs().max().item() == 0.0, "padding rows must be zero"
    positions = torch.arange(cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, k[:CHUNK])
    torch.testing.assert_close(v_back, v[:CHUNK])
    for bad in (0, CHUNK + pad + 1):
        with pytest.raises(ValueError):
            cache.plan(bad)
    with pytest.raises(ValueError, match="exceed"):  # more real tokens than rows
        short = (x[None, : CHUNK - 1] for x in (q, k, v))
        attn.forward(*short, batch_size=1, seq_len=CHUNK - 1, kv=cache.plan(CHUNK))


@pytest.mark.parametrize("backend", BACKENDS)
def test_dirty_steps_overwrite_in_place(cache, backend):
    """Several forwards at the same ``past`` leave only the last K/V in the cache."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    torch.manual_seed(1)
    open_with_prompt(cache, 9)
    attn = make_backend(backend)
    past = cache.staging_offset
    last_k = last_v = None
    for _ in range(4):
        q, last_k, last_v = rand_qkv(CHUNK)
        run(attn, cache, q, last_k, last_v)
    torch.cuda.synchronize()
    assert cache.staging_offset == past, "dirty steps must not advance the window"
    positions = torch.arange(past, past + CHUNK, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, last_k)
    torch.testing.assert_close(v_back, last_v)


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_replay_survives_commit(cache, backend):
    """A forward captured in a CUDA graph stays correct after commit() moves the cache,
    including when pages are recycled: writes land on the new pages and attention
    reads the new lengths, with nothing refreshed between commit and replay."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    torch.manual_seed(5)
    pk, pv = open_with_prompt(cache, 9)
    attn = make_backend(backend)
    history_k, history_v = [], []
    for _ in range(2):
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.staging_offset, k, v)
        cache.commit(cache.max_staged_tokens)
        history_k.append(k)
        history_v.append(v)

    q, k, v = rand_qkv(CHUNK)  # static buffers the graph reads
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # eager warmup creates lengths and metadata before capture
            run(attn, cache, q, k, v)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(attn, cache, q, k, v)

    for step in range(3):  # pages are recycled
        history_k.append(k.clone())
        history_v.append(v.clone())
        cache.commit(
            cache.max_staged_tokens
        )  # no metadata refresh: the kernel reads the cache's tensors
        q2, k2, v2 = rand_qkv(CHUNK)
        q.copy_(q2), k.copy_(k2), v.copy_(v2)
        graph.replay()
        torch.cuda.synchronize()

        hk, hv = torch.cat(history_k), torch.cat(history_v)
        expected = exact_reference(q, pk, pv, hk, hv, k, v, 0, CHUNK)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"replay {step}")
        positions = torch.arange(cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE)
        k_back, v_back = read_kv(cache, 0, positions)
        torch.testing.assert_close(k_back, k, msg=f"replay {step}: K landed on stale pages")
        torch.testing.assert_close(v_back, v, msg=f"replay {step}: V landed on stale pages")


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_runner_keys_cached_forwards(backend):
    """Through VisualGen's CUDAGraphRunner, the way a model runs: with the plan's graph
    key registered as an extra key, forwards of equal tensor shapes but different causal
    block sizes get their own graphs, and commits between forwards replay them (the
    cache's device state needs no recapture)."""
    torch.manual_seed(7)
    cache = make_cache(32)
    attn = make_backend(backend)
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    runner.register_extra_key_fn("kv", lambda *a, kv=None, **k: kv.graph_key if kv else None)

    def step(q, k, v, *, kv):
        out = attn.forward(q[None], k[None], v[None], batch_size=1, seq_len=q.shape[0], kv=kv)
        return out.reshape(q.shape[0], NUM_HEADS, HEAD_DIM)

    graphed = runner.wrap(step)
    try:
        pk, pv = open_with_prompt(cache, 9)
        history_k, history_v = [], []
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        for i in range(6):
            size = CHUNK if i % 2 == 0 else CHUNK // 4  # denoising step, then clean pass
            q, k, v = rand_qkv(CHUNK)
            out = graphed(q, k, v, kv=cache.plan(CHUNK, size))
            torch.cuda.synchronize()
            hk = torch.cat(history_k) if history_k else empty
            hv = torch.cat(history_v) if history_v else empty
            expected = torch.cat(
                [
                    exact_reference(q, pk, pv, hk, hv, k, v, lo, lo + size)
                    for lo in range(0, CHUNK, size)
                ]
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"forward {i}")
            cache.commit(CHUNK)
            history_k.append(k)
            history_v.append(v)
        assert len(runner.graphs) == 2, "one graph per causal block size, replayed across commits"
    finally:
        runner.clear()
        cache.shutdown()


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_survives_reopen_with_a_longer_prefix(backend):
    """A graph captured on a rollout with a short pinned prefix replays correctly on a
    later rollout of the same spec whose prefix spans more pages, so nothing the
    kernels read may depend on the first rollout's prefix."""
    torch.manual_seed(6)
    long_prefix = 512  # outgrows the first rollout's capacity by several kernel tiles
    cache = make_cache(32, pin_tokens=long_prefix)
    attn = make_backend(backend)
    try:
        open_with_prompt(cache, 9)
        q, k, v = rand_qkv(CHUNK)  # static buffers the graph reads
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run(attn, cache, q, k, v)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(attn, cache, q, k, v)
        cache.close()

        pk, pv = open_with_prompt(cache, long_prefix)  # sixteen more resident pages
        history_k, history_v = [], []
        for _ in range(2):
            _, hk, hv = rand_qkv(CHUNK)
            cache.write_range(0, cache.staging_offset, hk, hv)
            cache.commit(cache.max_staged_tokens)
            history_k.append(hk)
            history_v.append(hv)
        assert cache.fixed_tokens == long_prefix
        q2, k2, v2 = rand_qkv(CHUNK)
        q.copy_(q2), k.copy_(k2), v.copy_(v2)
        graph.replay()
        torch.cuda.synchronize()
        expected = exact_reference(
            q, pk, pv, torch.cat(history_k), torch.cat(history_v), k, v, 0, CHUNK
        )
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
        positions = torch.arange(cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE)
        k_back, v_back = read_kv(cache, 0, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)
    finally:
        cache.shutdown()


@pytest.mark.parametrize("num_causal_blocks", [1, 4])
@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_captured_while_the_window_fills(backend, num_causal_blocks):
    """A forward captured on the rollout's first chunk, with no history yet, replays
    correctly through sixteen commits: while the window fills, once it is full, and
    while pages are recycled many times over. Lengths the kernels
    read must come from the cache at replay, not from the moment of capture."""
    torch.manual_seed(8)
    prompt, size = 8, CHUNK // num_causal_blocks
    cache = make_cache(32, pin_tokens=prompt)
    try:
        pk, pv = open_with_prompt(cache, prompt)
        attn = make_backend(backend)
        q, k, v = rand_qkv(CHUNK)  # static buffers the graph reads
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # eager warmup creates lengths and metadata before capture
                run(attn, cache, q, k, v, causal_block_size=size)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(attn, cache, q, k, v, causal_block_size=size)

        history_k, history_v = [], []
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        for step in range(16):
            q2, k2, v2 = rand_qkv(CHUNK)
            q.copy_(q2), k.copy_(k2), v.copy_(v2)
            graph.replay()  # nothing refreshed since the commit, on either backend
            torch.cuda.synchronize()
            hk = torch.cat(history_k) if history_k else empty
            hv = torch.cat(history_v) if history_v else empty
            expected = torch.cat(
                [
                    exact_reference(q, pk, pv, hk, hv, k, v, i * size, (i + 1) * size)
                    for i in range(num_causal_blocks)
                ]
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")
            history_k.append(k.clone())
            history_v.append(v.clone())
            cache.commit(cache.max_staged_tokens)
        assert cache.staging_offset > 3 * cache.resident_tokens, "every page should be recycled"
    finally:
        cache.shutdown()


def test_backend_without_cache_support_refuses_a_cache(cache):
    """A cache routed to a backend that cannot use it raises instead of being ignored."""
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=NUM_HEADS * HEAD_DIM,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=True,
    )
    attn = Attention(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        config=config,
    )
    assert not attn.attn.support_kv_cache()
    assert make_backend("cudnn").support_kv_cache()
    assert make_backend("cutedsl").support_kv_cache()
    assert make_backend("trtllm").support_kv_cache()
    if get_sm_version() // 10 == 10:
        assert make_backend("flashinfer").support_kv_cache()
    open_with_prompt(cache, 9)
    q = torch.zeros(1, CHUNK, NUM_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
    with pytest.raises(NotImplementedError, match="does not support a K/V cache"):
        attn._attn_impl(q, q, q, kv=cache.plan(CHUNK))


def test_quantized_or_sparse_backends_report_no_cache_support():
    """Configurations the cache path cannot run say so up front, so ``Attention`` refuses
    the cache before the backend's forward does."""
    cudnn = make_backend("cudnn")
    cudnn.quant_dtype = "fp8"
    assert not cudnn.support_kv_cache()
    quantized = make_backend("cutedsl")
    quantized.quant_attention_config = SimpleNamespace()
    assert not quantized.support_kv_cache()
    sparse = make_backend("cutedsl")
    sparse.sparse_params = SimpleNamespace()
    assert not sparse.support_kv_cache()
    skip_softmax = make_backend("cutedsl")
    skip_softmax.skip_softmax_threshold_scale = 1.0
    assert not skip_softmax.support_kv_cache()
    for name, value in (("quant_attention_config", SimpleNamespace()), ("sparse_params", object())):
        trtllm = make_backend("trtllm")
        setattr(trtllm, name, value)
        assert not trtllm.support_kv_cache()
    quantized_flashinfer = make_backend("flashinfer")
    quantized_flashinfer.quant_attention_config = SimpleNamespace()
    assert not quantized_flashinfer.support_kv_cache()


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_with_causal_blocks_over_stale_history(cache, backend):
    """Ulysses padding during the clean pass: four causal blocks over history with
    stale tokens, plus padding rows that are neither written nor attended."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    torch.manual_seed(9)
    pk, pv = open_with_prompt(cache, 9)
    history_k, history_v = [], []
    for _ in range(3):
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.staging_offset, k, v)
        cache.commit(cache.max_staged_tokens)
        history_k.append(k)
        history_v.append(v)
    assert cache.history_tokens > WINDOW, "test geometry should hold stale tokens here"
    hk, hv = torch.cat(history_k), torch.cat(history_v)
    pad, size = 6, CHUNK // 4
    q, k, v = rand_qkv(CHUNK + pad)
    out = (
        make_backend(backend)
        .forward(
            q[None],
            k[None],
            v[None],
            batch_size=1,
            seq_len=CHUNK + pad,
            kv=cache.plan(CHUNK, size),
        )
        .reshape(CHUNK + pad, NUM_HEADS, HEAD_DIM)
    )
    torch.cuda.synchronize()
    expected = torch.cat(
        [exact_reference(q, pk, pv, hk, hv, k, v, i * size, (i + 1) * size) for i in range(4)]
    )
    torch.testing.assert_close(out[:CHUNK], expected, rtol=2e-2, atol=2e-2)
    assert out[CHUNK:].abs().max().item() == 0.0, "padding rows must be zero"
    positions = torch.arange(cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, k[:CHUNK])
    torch.testing.assert_close(v_back, v[:CHUNK])


@pytest.mark.parametrize("causal_block_size", [32, 64])
@pytest.mark.parametrize("backend", BACKENDS)
def test_causal_blocks_of_one_and_two_whole_pages(backend, causal_block_size):
    """Causal blocks exactly one page and exactly two pages long, several per chunk, so
    their starts and ends fall on page boundaries, while pages are recycled."""
    torch.manual_seed(10)
    prompt, window, chunk = 8, 96, 128
    spec = spec_for(
        max_pin_tokens=prompt,
        causal_blocks={128: 1, 64: 2, 32: 4},
        window_tokens=window,
        chunk=chunk,
    )
    cache = make_cache(32, spec=spec)
    try:
        pk, pv = open_with_prompt(cache, prompt)
        attn = make_backend(backend)
        history_k, history_v = [], []
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        for step in range(6):
            q, k, v = rand_qkv(chunk)
            out = run(attn, cache, q, k, v, causal_block_size=causal_block_size)
            torch.cuda.synchronize()
            hk = torch.cat(history_k) if history_k else empty
            hv = torch.cat(history_v) if history_v else empty
            keys, values = torch.cat([pk, hk, k]), torch.cat([pv, hv, v])
            pos = torch.arange(keys.shape[0], device=DEVICE)
            expected = []
            for start in range(0, chunk, causal_block_size):
                end = start + causal_block_size
                visible = (pos < prompt) | (
                    (pos >= prompt + hk.shape[0] + start - window)
                    & (pos < prompt + hk.shape[0] + end)
                )
                expected.append(reference_attention(q[start:end], keys[visible], values[visible]))
            torch.testing.assert_close(
                out, torch.cat(expected), rtol=2e-2, atol=2e-2, msg=f"step {step}"
            )
            history_k.append(k)
            history_v.append(v)
            cache.commit(cache.max_staged_tokens)
        assert cache.history_tokens >= window, "the window should be full by now"
    finally:
        cache.shutdown()


@pytest.mark.parametrize("backend", BACKENDS)
def test_cache_path_refusals(cache, backend):
    """What the cache path does not support raises instead of computing something else."""
    if lacks_kernel(backend, cache):
        pytest.skip("no paged context kernel for this page size in this build")
    open_with_prompt(cache, 9)
    attn = make_backend(backend)
    q, k, v = rand_qkv(CHUNK)
    common = dict(seq_len=CHUNK, kv=cache.plan(CHUNK))
    with pytest.raises(ValueError):  # two videos
        attn.forward(
            torch.stack([q, q]), torch.stack([k, k]), torch.stack([v, v]), batch_size=2, **common
        )
    with pytest.raises(NotImplementedError):  # anything but full attention over the cache
        attn.forward(
            q[None],
            k[None],
            v[None],
            batch_size=1,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            **common,
        )
    with pytest.raises(ValueError, match="not declared"):
        cache.plan(CHUNK, 8)


def _assert_attention_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """T1 attention parity against FP32 SDPA with BF16 fused-kernel accumulation."""
    actual_fp32, expected_fp32 = actual.float().flatten(), expected.float().flatten()
    error = actual_fp32 - expected_fp32
    ref_norm = expected_fp32.norm()
    if ref_norm > 0:
        relative_l2 = (error.norm() / ref_norm).item()
        cosine = torch.nn.functional.cosine_similarity(actual_fp32, expected_fp32, dim=0).item()
    else:
        relative_l2 = error.norm().item()
        cosine = float(relative_l2 == 0)
    max_error = error.abs().max().item()
    stats = f"relative_l2={relative_l2:.6g}, cosine={cosine:.8f}, max_abs={max_error:.6g}"
    print(stats)
    assert relative_l2 < 1e-2 and cosine > 0.9999, stats
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2, msg=stats)


@pytest.mark.parametrize("metadata", ["indices", "offsets", "both"])
def test_q_only_rejects_sparse_compaction_before_cache_write(monkeypatch, metadata: str) -> None:
    """The Q-only native op refuses compaction metadata without modifying cached K/V."""
    cache = make_cache(32)
    try:
        open_with_prompt(cache, 9)
        q, k, v = rand_qkv(CHUNK)
        plan = cache.plan(CHUNK)
        cache.write_staged(0, k, v, plan)
        pool = cache.kv_buffer(0)
        before = pool.clone()
        indices = torch.full((NUM_KV_HEADS, 1), 5, dtype=torch.int32, device=DEVICE)
        offsets = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        native_attention = FallbackFmha.attention

        def with_compaction(**kwargs) -> None:
            kwargs["sparse_kv_indices"] = indices if metadata != "offsets" else None
            kwargs["sparse_kv_offsets"] = offsets if metadata != "indices" else None
            native_attention(**kwargs)

        monkeypatch.setattr(FallbackFmha, "attention", staticmethod(with_compaction))
        with pytest.raises(RuntimeError, match="Q-only self-attention.*sparse KV compaction"):
            make_backend("trtllm").forward_paged_kv(q[None], pool, cache.tables(plan))
        torch.cuda.synchronize()
        torch.testing.assert_close(pool, before, rtol=0, atol=0)
    finally:
        cache.shutdown()


@pytest.mark.parametrize("backend", ["trtllm", BACKENDS[3]])
def test_graph_replay_after_larger_plan_warmup(monkeypatch, backend: str) -> None:
    """T1: a captured paged-attention output still matches FP32 SDPA after scratch growth.

    A larger plan crosses the counter's SM-count floor and increases query storage. Weak
    references and saved addresses prove the captured allocation survives without test-owned
    tensor references; allocator churn makes an accidentally released allocation observable.
    """
    torch.manual_seed(15)
    buffers = []

    def record(buffer: torch.Tensor) -> None:
        buffers.append((weakref.ref(buffer), buffer.data_ptr(), buffer.numel()))

    if backend == "flashinfer":
        monkeypatch.setattr(FlashInferAttention, "_trtllm_gen_paged_buffers", {})
        get_buffers = FlashInferAttention._trtllm_gen_buffers

        def recording_buffers(
            cls: type[FlashInferAttention], q: torch.Tensor, batch_size: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            workspace, counters = get_buffers(q, batch_size)
            record(counters)
            return workspace, counters

        monkeypatch.setattr(
            FlashInferAttention, "_trtllm_gen_buffers", classmethod(recording_buffers)
        )
    else:
        monkeypatch.setattr(TrtllmAttention, "_paged_kv_workspaces", {})
        native_attention = FallbackFmha.attention

        def recording_attention(**kwargs) -> None:
            native_attention(**kwargs)
            record(kwargs["workspace"])

        monkeypatch.setattr(FallbackFmha, "attention", staticmethod(recording_attention))

    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    large_chunk = 1 << sm_count.bit_length()
    small_chunk = 8
    small = make_cache(32, spec=spec_for(chunk=small_chunk, causal_blocks={small_chunk: 1}))
    large = make_cache(32, spec=spec_for(chunk=large_chunk, causal_blocks={1: large_chunk}))
    try:
        pk, pv = open_with_prompt(small, 9)
        open_with_prompt(large, 9)
        attention = make_backend(backend)
        q, k, v = rand_qkv(small_chunk)
        plan = small.plan(small_chunk)
        small.write_staged(0, k, v, plan)
        with pytest.raises(RuntimeError, match="first needed during CUDA graph capture"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                attention.forward_paged_kv(q[None], small.kv_buffer(0), small.tables(plan))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run(attention, small, q, k, v)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(attention, small, q, k, v)
        captured_ref, captured_ptr, captured_size = buffers[-1]
        run(attention, large, *rand_qkv(large_chunk), causal_block_size=1)
        assert buffers[-1][2] > captured_size, "larger plan must require a larger allocation"
        assert captured_ref() is not None, "captured scratch allocation was released"
        assert captured_ref().data_ptr() == captured_ptr, "captured scratch storage moved"
        scratch = torch.empty(captured_size, dtype=torch.uint8, device=DEVICE)
        scratch.fill_(0xA5)
        q2, k2, v2 = rand_qkv(small_chunk)
        q.copy_(q2), k.copy_(k2), v.copy_(v2)
        graph.replay()
        torch.cuda.synchronize()
        empty = pk[:0]
        expected = exact_reference(q, pk, pv, empty, empty, k, v, 0, small_chunk)
        _assert_attention_parity(output, expected)
    finally:
        small.shutdown()
        large.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the CuTe DSL kernel runs on SM100/SM103",
)
@pytest.mark.parametrize("layout", ["padded_pages", "nonunit_head", "unaligned", "overlapping"])
def test_cutedsl_rejects_unsupported_pool_strides(layout: str) -> None:
    """Reject unsupported TMA layouts before launching a paged-attention kernel."""
    cache = make_cache(32)
    try:
        open_with_prompt(cache, 9)
        pool = cache.kv_buffer(0)
        shape = pool.shape
        if layout == "padded_pages":
            invalid = pool.new_empty(shape[0] * 2, *shape[1:])[::2]
        elif layout == "nonunit_head":
            invalid = pool.new_empty(*shape[:-1], shape[-1] * 2)[..., ::2]
        elif layout == "unaligned":
            invalid = pool.new_empty(pool.numel() + 1)[1:].view(shape)
            assert invalid.is_contiguous()
        else:
            invalid = pool[:1].expand_as(pool)
        q = rand_qkv(CHUNK)[0]
        with pytest.raises(ValueError, match="K/V pool must be contiguous and 32-byte aligned"):
            make_backend("cutedsl").forward_paged_kv(
                q[None], invalid, cache.tables(cache.plan(CHUNK))
            )
    finally:
        cache.shutdown()
