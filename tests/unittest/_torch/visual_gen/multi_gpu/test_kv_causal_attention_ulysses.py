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
"""CausalKVCacheManager under Ulysses: two ranks, each holding a head slice of the cache.

Every rank builds the same random full-head q/k/v, feeds the wrapper its token
slice, and checks its token slice of the output against a full-head dense
reference over prompt, resident history and the new tokens. The cache itself is
built with per-rank heads and never learns about the topology. The VisualGen
mesh is built first, as in a real run, so the cache's one-rank mapping coexists
with it.
"""

import functools
import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tensorrt_llm._torch.distributed import all_to_all_4d
from tensorrt_llm._torch.visual_gen.attention_backend import UlyssesAttention
from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.modules.attention import Attention
from tensorrt_llm.visual_gen.args import AttentionConfig

WORLD = 2
NUM_HEADS = 8
NUM_KV_HEADS = 4  # fewer than NUM_HEADS: the wrapper must take its per-tensor path
HEAD_DIM = 64
DTYPE = torch.bfloat16
TPB = 32
PROMPT = 16
WINDOW = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} GPUs",
)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _worker(rank, world_size, test_fn, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def run_distributed(test_fn):
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(_worker, args=(WORLD, test_fn, port), nprocs=WORLD, join=True)
    )


def exact_reference(q, pk, pv, hk, hv, k, v, start, end, window):
    """The new tokens ``[start, end)`` over the prompt, the ``window`` keys before
    ``start`` and the chunk up to ``end``; ``hk``/``hv`` all committed history."""
    keys = torch.cat([pk, hk, k[:end]])
    values = torch.cat([pv, hv, v[:end]])
    pos = torch.arange(keys.shape[0], device=keys.device)
    visible = (pos < pk.shape[0]) | (pos >= pk.shape[0] + hk.shape[0] + start - window)
    return reference_attention(q[start:end], keys[visible], values[visible])


def reference_attention(q, keys, values):
    """Full heads: q [T, H, D], keys/values [S, H_kv, D] restricted to the visible set."""
    rep = q.shape[1] // keys.shape[1]
    kx = keys.repeat_interleave(rep, dim=1)
    vx = values.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float().unsqueeze(0),
        kx.transpose(0, 1).float().unsqueeze(0),
        vx.transpose(0, 1).float().unsqueeze(0),
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def make_backend(name, chunk, prompt=PROMPT, window=WINDOW):
    heads, kv_heads = NUM_HEADS // WORLD, NUM_KV_HEADS // WORLD
    if name == "cudnn":
        return CuDNNAttention(
            layer_idx=0, num_heads=heads, head_dim=HEAD_DIM, num_kv_heads=kv_heads, dtype=DTYPE
        )
    return TrtllmAttention(
        layer_idx=0,
        num_heads=heads,
        head_dim=HEAD_DIM,
        num_kv_heads=kv_heads,
        dtype=DTYPE,
        max_seq_len=prompt + window + chunk + TPB,
        attention_metadata_state={},
    )


def make_cache(chunk, prompt=PROMPT, window=WINDOW):
    return CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS // WORLD,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=TPB,
        fixed_capacity=prompt,
        window_tokens=window,
        chunk_tokens=chunk,
        causal_block_sizes=(chunk, chunk // 4),
    )


def rand_qkv(n):
    q = torch.randn(n, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    return q, k, torch.randn_like(k)


def token_slice(x, rank):
    """This rank's contiguous token shard, batched: [1, T/P, H, D]."""
    per = x.shape[0] // WORLD
    return x[rank * per : (rank + 1) * per][None].contiguous()


def head_slice(x, rank):
    per = x.shape[1] // WORLD
    return x[:, rank * per : (rank + 1) * per]


def to_head_layout(x, rank, group):
    """What the model does for the prompt: token shard in, all tokens of my heads out."""
    return all_to_all_4d(token_slice(x, rank), scatter_dim=2, gather_dim=1, process_group=group)[0]


def make_ulysses(rank, world_size, backend, chunk, prompt=PROMPT, window=WINDOW):
    """The wrapper over this rank's head slice, plus the Ulysses group it uses."""
    vgm = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
    inner = make_backend(backend, chunk, prompt, window)
    return UlyssesAttention(inner_backend=inner, process_group=vgm.ulysses_group), vgm.ulysses_group


def read_kv(cache, positions):
    buf = cache.kv_buffer(0)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_page]
    slot = positions % cache.tokens_per_page
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def forward(attn, cache, q, k, v, rank, causal_block_size=None, seq_len=None):
    """``q``/``k``/``v`` hold all tokens, padded to a multiple of ``WORLD``; ``seq_len``
    is the real count (default: all rows). Returns this rank's rows of the output."""
    per = q.shape[0] // WORLD
    out = attn.forward(
        token_slice(q, rank),
        token_slice(k, rank),
        token_slice(v, rank),
        batch_size=1,
        seq_len=q.shape[0] if seq_len is None else seq_len,
        kv_cache=cache,
        causal_block_size=causal_block_size,
    )
    return out.reshape(per, NUM_HEADS, HEAD_DIM)


def _logic_rollout(rank, world_size, backend):
    torch.manual_seed(0)  # same tensors on every rank
    chunk = 40
    cache = make_cache(chunk)
    attn, group = make_ulysses(rank, world_size, backend, chunk)
    try:
        cache.open(pin_tokens=PROMPT)
        _, pk, pv = rand_qkv(PROMPT)
        cache.write_range(0, 0, to_head_layout(pk, rank, group), to_head_layout(pv, rank, group))
        cache.commit(PROMPT)
        history_k, history_v = [], []
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        per = chunk // world_size
        for step in range(8):  # rotates the table more than once
            q, k, v = rand_qkv(chunk)
            out = forward(attn, cache, q, k, v, rank)
            torch.cuda.synchronize()

            hk = torch.cat(history_k) if history_k else empty
            hv = torch.cat(history_v) if history_v else empty
            expected = exact_reference(q, pk, pv, hk, hv, k, v, 0, chunk, WINDOW)[
                rank * per : (rank + 1) * per
            ]
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")

            positions = torch.arange(cache.past_tokens, cache.past_tokens + chunk, device="cuda")
            k_back, v_back = read_kv(cache, positions)
            torch.testing.assert_close(k_back, head_slice(k, rank), msg=f"step {step}: K")
            torch.testing.assert_close(v_back, head_slice(v, rank), msg=f"step {step}: V")

            cache.commit()
            history_k.append(k)
            history_v.append(v)
        assert cache.history_tokens < 8 * chunk, "window never rotated"
    finally:
        cache.shutdown()


def _logic_causal_blocks(
    rank, world_size, backend, chunk, num_causal_blocks, prompt=PROMPT, window=WINDOW
):
    torch.manual_seed(1)
    cache = make_cache(chunk, prompt, window)
    attn, group = make_ulysses(rank, world_size, backend, chunk, prompt, window)
    try:
        cache.open(pin_tokens=prompt)
        _, pk, pv = rand_qkv(prompt)
        cache.write_range(0, 0, to_head_layout(pk, rank, group), to_head_layout(pv, rank, group))
        cache.commit(prompt)
        history_k, history_v = [], []
        for _ in range(3):  # 120 tokens committed, one page dropped: 88 resident, 24 stale
            _, k, v = rand_qkv(chunk)
            cache.write_range(0, cache.past_tokens, head_slice(k, rank), head_slice(v, rank))
            cache.commit()
            history_k.append(k)
            history_v.append(v)
        assert cache.history_tokens > window, "test geometry should hold stale tokens here"
        hk, hv = torch.cat(history_k), torch.cat(history_v)

        size = chunk // num_causal_blocks
        q, k, v = rand_qkv(chunk)
        out = forward(attn, cache, q, k, v, rank, causal_block_size=size)
        torch.cuda.synchronize()
        per = chunk // world_size
        expected = torch.cat(
            [
                exact_reference(q, pk, pv, hk, hv, k, v, i * size, (i + 1) * size, window)
                for i in range(num_causal_blocks)
            ]
        )[rank * per : (rank + 1) * per]
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    finally:
        cache.shutdown()


def _logic_padded_first_chunk(rank, world_size, backend):
    """A one-block first chunk whose length does not divide by the rank count: the
    model pads it, passes the real ``seq_len``, and the padding is neither
    attended nor written."""
    torch.manual_seed(2)  # same tensors on every rank
    chunk, block = 44, 11  # 11 real tokens on 2 ranks: padded to 12
    cache = make_cache(chunk)
    attn, group = make_ulysses(rank, world_size, backend, chunk)
    try:
        cache.open(pin_tokens=PROMPT)
        _, pk, pv = rand_qkv(PROMPT)
        cache.write_range(0, 0, to_head_layout(pk, rank, group), to_head_layout(pv, rank, group))
        cache.commit(PROMPT)
        q, k, v = rand_qkv(block)
        pad = (-block) % world_size
        padded = [torch.cat([x, torch.randn_like(x[:pad])]) for x in (q, k, v)]
        out = forward(attn, cache, *padded, rank, causal_block_size=block, seq_len=block)
        torch.cuda.synchronize()

        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        full = exact_reference(q, pk, pv, empty, empty, k, v, 0, block, WINDOW)
        full = torch.cat([full, full.new_zeros((pad, NUM_HEADS, HEAD_DIM))])
        per = (block + pad) // world_size
        expected = full[rank * per : (rank + 1) * per]
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

        positions = torch.arange(cache.past_tokens, cache.past_tokens + block, device="cuda")
        k_back, v_back = read_kv(cache, positions)
        torch.testing.assert_close(k_back, head_slice(k, rank))
        torch.testing.assert_close(v_back, head_slice(v, rank))
        cache.commit(block)
        assert cache.history_tokens == block
    finally:
        cache.shutdown()


def _logic_head_count_guard(rank, world_size, backend):
    """A cache built for the wrong head count is refused, not silently written."""
    chunk = 40
    cache = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,  # full count: wrong under Ulysses
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=TPB,
        fixed_capacity=PROMPT,
        window_tokens=WINDOW,
        chunk_tokens=chunk,
        causal_block_sizes=(chunk,),
    )
    attn, _ = make_ulysses(rank, world_size, backend, chunk)
    try:
        cache.open()
        q, k, v = rand_qkv(chunk)
        with pytest.raises(ValueError, match="per rank"):
            forward(attn, cache, q, k, v, rank)
    finally:
        cache.shutdown()


def _logic_per_rank_seq_len_refused(rank, world_size, backend):
    """With a cache, seq_len counts the whole sequence's real tokens; a caller passing
    its per-rank length would have real tokens treated as padding, so it raises."""
    chunk = 40
    cache = make_cache(chunk)
    attn, _ = make_ulysses(rank, world_size, backend, chunk)
    try:
        cache.open()
        q, k, v = rand_qkv(chunk)
        with pytest.raises(ValueError, match="real tokens of the whole sequence"):
            forward(attn, cache, q, k, v, rank, seq_len=chunk // world_size)
    finally:
        cache.shutdown()


def _logic_through_the_attention_module(rank, world_size, backend):
    """The model's path: the shared Attention module builds the Ulysses wrapper and
    must hand the caller's real token count through, padding included."""
    torch.manual_seed(3)
    chunk = 44  # declares blocks of 11, the padded single-frame chunk below
    vgm = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=NUM_HEADS * HEAD_DIM,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend=backend.upper()),
        skip_create_weights_in_init=True,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "trtllm" else None
    )
    config.visual_gen_mapping = vgm
    attn = Attention(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        config=config,
        layer_idx=0,
    )
    assert attn.attn.support_kv_cache()
    cache = make_cache(chunk)
    group = vgm.ulysses_group
    try:
        cache.open(pin_tokens=PROMPT)
        _, pk, pv = rand_qkv(PROMPT)
        cache.write_range(0, 0, to_head_layout(pk, rank, group), to_head_layout(pv, rank, group))
        cache.commit(PROMPT)
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        for real in (chunk, 11):  # a full chunk, then 11 tokens padded to 12
            pad = (-real) % world_size
            q, k, v = rand_qkv(real)
            padded = [torch.cat([x, torch.randn_like(x[:pad])]) for x in (q, k, v)]
            per = (real + pad) // world_size
            flat = [token_slice(x, rank).flatten(2) for x in padded]  # [1, per, heads * dim]
            out = attn._attn_impl(*flat, kv_cache=cache, seq_len=real).view(
                per, NUM_HEADS, HEAD_DIM
            )
            torch.cuda.synchronize()
            full = exact_reference(q, pk, pv, empty, empty, k, v, 0, real, WINDOW)
            full = torch.cat([full, full.new_zeros((pad, NUM_HEADS, HEAD_DIM))])
            torch.testing.assert_close(
                out, full[rank * per : (rank + 1) * per], rtol=2e-2, atol=2e-2
            )
    finally:
        cache.shutdown()


def _logic_all(rank, world_size, backend):
    """Every scenario in one process group per backend: spawning costs more than the tests."""
    _logic_rollout(rank, world_size, backend)
    _logic_causal_blocks(rank, world_size, backend, chunk=40, num_causal_blocks=4)
    _logic_padded_first_chunk(rank, world_size, backend)
    _logic_head_count_guard(rank, world_size, backend)
    _logic_per_rank_seq_len_refused(rank, world_size, backend)
    _logic_through_the_attention_module(rank, world_size, backend)


@pytest.mark.parametrize("backend", ["cudnn", "trtllm"])
def test_kv_cache_attention_under_ulysses(backend):
    """A rollout through rotations, the clean pass in four page-unaligned causal blocks
    cut after the all-to-all, a padded single-frame first chunk, and two refusals: a
    cache built for the full head count, and a per-rank seq_len."""
    run_distributed(functools.partial(_logic_all, backend=backend))
