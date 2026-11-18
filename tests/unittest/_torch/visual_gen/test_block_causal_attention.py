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
"""Attention backends over BlockCausalKVCacheManager against a dense SDPA reference.

Random inputs; the reference keeps its own dense copy of every K/V ever written
and attends over exactly what the model's window allows: the fixed region (the
prompt), the ``WINDOW`` tokens before each causal block, the earlier causal blocks
of the chunk and the causal block itself. Stale tokens the whole-page eviction keeps
resident must not be seen.
"""

import gc
import weakref
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from utils.block_causal_reference import exact_reference as _exact_reference
from utils.block_causal_reference import reference_attention

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention as BaseTrtllmAttention
from tensorrt_llm._torch.visual_gen.attention.attention import Attention
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import BlockCausalKVCacheManager, RolloutSpec
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
    cache = BlockCausalKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        tokens_per_page=tokens_per_page,
        pool_tokens=BlockCausalKVCacheManager.pool_tokens_for(
            spec, tokens_per_page=tokens_per_page
        ),
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
    if name == "cudnn":
        return CuDNNAttention(
            layer_idx=layer_idx,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=dtype,
        )
    return TrtllmAttention(
        layer_idx=layer_idx,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dtype=dtype,
        max_seq_len=PROMPT_CAPACITY + WINDOW + CHUNK + 32,
        attention_metadata_state={},
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
    table = cache.table.long()
    page = table[positions // cache.tokens_per_page]
    slot = positions % cache.tokens_per_page
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


BACKENDS = ["cudnn", "trtllm"]


def lacks_kernel(backend, cache, dtype=DTYPE):
    """Whether this build has no attention kernel for the cache's page size on this GPU."""
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
        # Without a kernel the op would silently drop the prefix, so the backend refuses.
        q, k, v = rand_qkv(CHUNK)
        with pytest.raises(NotImplementedError, match="no fused paged context attention kernel"):
            run(attn, cache, q, k, v)
        return

    history_k, history_v = [], []  # the reference's dense copy of committed generator K/V
    empty = prompt_k.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
    saw_stale = False
    for step in range(8):  # long enough to rotate the table more than once
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


def expected_fmha_library(dtype, num_causal_blocks):
    """The FMHA library the TRTLLM backend should pick for a cached forward. On
    SM100/SM103 FlashInfer's TRTLLM-Gen kernels serve it, except a BF16 context batch
    of at most 4 requests (one per causal block), which goes to the fallback for LLM
    time to first token. Elsewhere only the fallback is available."""
    if get_sm_version() not in (100, 103):
        return "FallbackFmha"
    if dtype == torch.bfloat16 and num_causal_blocks <= 4:
        return "FallbackFmha"
    return "FlashInferTrtllmGenFmha"


def record_fmha_libraries(attn, served):
    """Wrap the backend's FMHA selection so every library it picks lands in ``served``."""
    manager = attn._fmha_manager
    select = manager.select

    def recording_select(*args, **kwargs):
        fmha = select(*args, **kwargs)
        served.add(type(fmha).__name__)
        return fmha

    manager.select = recording_select


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
    through a rotating table, with 1, 4 and 8 causal blocks in BF16 and FP16, at two
    head shapes. Each layer must read its own pages. For TRTLLM the cases also cross the FMHA library
    boundary on SM100/SM103, and the library that served each one is checked. For
    cuDNN every layer shares one graph."""
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
        served = set()
        if backend == "trtllm":
            if get_sm_version() in (100, 103):
                names = {type(lib).__name__ for lib in attns[0]._fmha_manager.fmha_libs}
                assert "FlashInferTrtllmGenFmha" in names, f"FMHA libraries built: {names}"
            for attn in attns:
                record_fmha_libraries(attn, served)
        history = [([], []) for _ in range(num_layers)]
        empty = prompt[0][0].new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        size = CHUNK // num_causal_blocks
        saw_stale = False
        for step in range(4):  # rotates the table and leaves stale tokens
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
        if backend == "trtllm":
            assert served == {expected_fmha_library(dtype, num_causal_blocks)}
        else:
            assert len(CuDNNAttention._graph_cache) == 1, "every layer should share one graph"
    finally:
        cache.shutdown()


@pytest.mark.parametrize("backend", BACKENDS)
def test_causal_blocks_at_any_alignment(cache, backend):
    """The clean pass: one launch, four causal blocks of 10 sharing pages mid-way,
    each with its own window start, over a rotated table with stale tokens."""
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
    including across a page rotation: writes land on the new pages and attention
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

    for step in range(3):  # the third commit rotates the table
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
    correctly through a dozen commits: while the window fills, once it is full, and
    across many rotations of a pool of five 32-token pages. Lengths the kernels
    read must come from the cache at replay, not from the moment of capture."""
    torch.manual_seed(8)
    prompt, size = 8, CHUNK // num_causal_blocks
    cache = make_cache(32, pin_tokens=prompt)
    try:
        pk, pv = open_with_prompt(cache, prompt)
        assert cache.num_pages == 5
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
        for step in range(12):
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
        assert len(history_k) * CHUNK >= 3 * cache.capacity, "the pool should cycle three times"
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
    assert make_backend("trtllm").support_kv_cache()
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
    sage = make_backend("trtllm")
    sage.quant_attention_config = SimpleNamespace()
    assert not sage.support_kv_cache()
    sparse = make_backend("trtllm")
    sparse.sparse_params = SimpleNamespace()
    assert not sparse.support_kv_cache()


def test_trtllm_refuses_a_page_size_without_a_kernel(cache, monkeypatch):
    """Where the build has no paged context kernel for the cache's page size, the TRTLLM
    backend reports it and refuses the cache instead of running the unfused fallback."""
    from tensorrt_llm._torch.visual_gen.attention.backends import trtllm as trtllm_backend

    open_with_prompt(cache, 9)
    attn = make_backend("trtllm")
    monkeypatch.setattr(trtllm_backend, "_paged_context_kernel_exists", lambda *a: False)
    assert not attn.support_kv_cache_page_size(cache.tokens_per_page)
    q, k, v = rand_qkv(CHUNK)
    with pytest.raises(NotImplementedError, match="no fused paged context attention kernel"):
        run(attn, cache, q, k, v)


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_with_causal_blocks_over_stale_history(cache, backend):
    """Ulysses padding during the clean pass: four causal blocks over a rotated table
    with stale tokens, plus padding rows that are neither written nor attended."""
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
    their starts and ends fall on page boundaries, through several rotations."""
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
    if backend == "trtllm":
        attn.sparse_params = object()  # any sparse attention configuration
        with pytest.raises(NotImplementedError, match="sparse"):
            attn.forward(q[None], k[None], v[None], batch_size=1, **common)


@pytest.mark.parametrize("pad", [0, 6], ids=["unpadded", "padded"])
def test_trtllm_reads_fused_qkv_slices_without_a_copy(cache, monkeypatch, pad):
    """q/k/v sliced from one qkv_proj output reach the kernel as that buffer, padded or
    not, and give the same output as separate tensors, which are concatenated."""
    if lacks_kernel("trtllm", cache):
        pytest.skip("no paged context kernel for this page size in this build")
    open_with_prompt(cache, 9)
    attn = make_backend("trtllm")
    rows = CHUNK + pad
    q_dim, kv_dim = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
    fused = torch.randn(1, rows, q_dim + 2 * kv_dim, device=DEVICE, dtype=DTYPE)
    q, k, v = (x.view(1, rows, -1, HEAD_DIM) for x in fused.split([q_dim, kv_dim, kv_dim], dim=-1))
    kernel_inputs = []
    parent_forward = BaseTrtllmAttention.forward

    def recording_forward(self, *args, **kwargs):
        kernel_inputs.append(kwargs["q"])
        return parent_forward(self, *args, **kwargs)

    monkeypatch.setattr(BaseTrtllmAttention, "forward", recording_forward)

    def forward(q, k, v):
        return attn.forward(q, k, v, batch_size=1, seq_len=rows, kv=cache.plan(CHUNK)).reshape(
            rows, NUM_HEADS, HEAD_DIM
        )

    out_fused = forward(q, k, v)
    assert kernel_inputs[-1].data_ptr() == fused.data_ptr(), "expected the qkv_proj buffer"
    # The same tokens again, as separate tensors: staged in place of the first ones.
    out_separate = forward(q.clone(), k.clone(), v.clone())
    assert kernel_inputs[-1].untyped_storage().data_ptr() != fused.untyped_storage().data_ptr()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_fused, out_separate, rtol=0, atol=0)


def test_trtllm_metadata_lives_on_the_cache():
    """The TRTLLM backend keeps the metadata it builds over a cache on that cache and
    holds no reference to the cache itself. A closed cache keeps its metadata, since
    graphs captured over it still read it; shutdown() drops it. A cache dropped without
    shutdown() is reclaimed by the cyclic garbage collector."""
    attn = make_backend("trtllm")
    cache = make_cache(32)
    try:
        open_with_prompt(cache, 9)
        one = attn.metadata.prepare_with_kv_cache(cache, 1, CHUNK)
        four = attn.metadata.prepare_with_kv_cache(cache, 4, CHUNK // 4)
        assert len(cache.attention_metadata) == 2
        assert attn.metadata.prepare_with_kv_cache(cache, 1, CHUNK) is one
        cache.close()
        assert len(cache.attention_metadata) == 2
        open_with_prompt(cache, 9)
        assert attn.metadata.prepare_with_kv_cache(cache, 4, CHUNK // 4) is four
        cache.shutdown()
        assert cache.attention_metadata == {}
    finally:
        if not cache.is_shut_down:
            cache.shutdown()

    dropped = make_cache(32)
    open_with_prompt(dropped, 9)
    attn.metadata.prepare_with_kv_cache(dropped, 1, CHUNK)
    ref = weakref.ref(dropped)
    del dropped
    gc.collect()
    assert ref() is None, "the backend keeps a dropped cache alive"


def test_packed_qkv_matches_separate_tensors_on_the_cache_path():
    """trtllm-gen takes the fused projection [1, S, H + 2 H_kv, D] directly; the
    cache gets strided K/V slices of it. Same output, same cached K/V as the
    separate-tensor call."""
    torch.manual_seed(13)
    attn = make_backend("trtllm")
    outs, kvs = [], []
    q, k, v = rand_qkv(CHUNK)
    packed = torch.cat([q, k, v], dim=1).unsqueeze(0).contiguous()  # [1, S, H + 2 H_kv, D]
    for call in ("separate", "packed"):
        cache = make_cache(32)
        try:
            torch.manual_seed(14)  # the same prompt K/V for both calls
            open_with_prompt(cache, 9)
            if call == "separate":
                out = run(attn, cache, q, k, v, causal_block_size=CHUNK // 4)
            else:
                out = attn.forward(
                    q=packed,
                    k=None,
                    v=None,
                    batch_size=1,
                    seq_len=CHUNK,
                    seq_len_kv=CHUNK,
                    kv=cache.plan(CHUNK, CHUNK // 4),
                )
            torch.cuda.synchronize()
            positions = torch.arange(
                cache.staging_offset, cache.staging_offset + CHUNK, device=DEVICE
            )
            outs.append(out.reshape(CHUNK, -1).clone())
            kvs.append(tuple(x.clone() for x in read_kv(cache, 0, positions)))
        finally:
            cache.shutdown()
    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
    torch.testing.assert_close(kvs[1][0], kvs[0][0], rtol=0, atol=0)
    torch.testing.assert_close(kvs[1][1], kvs[0][1], rtol=0, atol=0)
