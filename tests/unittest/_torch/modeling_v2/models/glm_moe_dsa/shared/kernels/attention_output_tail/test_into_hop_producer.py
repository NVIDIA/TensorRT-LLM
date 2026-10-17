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
"""attention_output_into_hop producer (single GPU): the publishing attention_output_tail o_proj and the tail protocol.

A plain device buffer stands in for the post_attention_hop workspace (multicast view and
local copy are the same memory on one GPU), filled with the Lamport sentinel,
with a protocol state selecting the current buffer. On the real DSA ``MLA``
of TP8 rank 0 with real layer-3 weights and synthetic latents:

* ``request_into_hop`` → ``arm`` → ``defer_w_uv`` → ``project`` publishes exactly
  slot ``[row][rank]`` of the current buffer for every output row (live rows
  and the zero rows of padding the post_attention_hop also sums), as 32-bit words equal
  to the BF16 partial with ``-0.0`` mapped to ``+0.0``; every other word
  stays the sentinel; ``take_into_hop_published`` reports it once;
* the ordinary partial is bit-identical to the attention_output_tail without publishing;
* a request is consumed by the next ``arm`` even when the call declines;
  non-attention_output_tail engines never publish; a request whose hop row count differs from
  the output rows publishes nothing (the layer then runs the post_attention_hop);
* CUDA-graph capture and replay publish into the same slots, also with
  padded rows;
* both publish widths (32-bit words on 152 CTAs or on request, 16-byte
  stores on 128 CTAs) write the same words;
* ``make_attn_tail`` engages the attention_output_into_hop schedules only next to the
  post_attention_hop (``INTO_HOP_SPARSE_DECODE_SCHEDULE`` behind ``sparse_mla_decode``,
  ``INTO_HOP_SCHEDULE`` behind the native FMHA); the plan turns attention_output_into_hop
  off with either path it extends.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "attention_output_tail"))
sys.path.insert(0, os.path.dirname(_HERE))
import plan_testlib  # noqa: E402
import test_attn_tail_mla_adapter as tail_adapter  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E402
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    DEFAULT_CONFIG,
    INTO_HOP_SCHEDULE,
    INTO_HOP_SPARSE_DECODE_SCHEDULE,
    TAIL_PREFETCH_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
    TailConfig,
    attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E402, E501
    AttnTail,
    make_attn_tail,
)

pytestmark = tail_adapter.pytestmark

H = 6144
RANKS = 8
MAX_ROWS = 8
BUF_WORDS = MAX_ROWS * RANKS * H * 2 // 4
SENTINEL = -(2**31)  # 0x80000000 as int32


def _transport(cur=1, rank=3):
    mc = torch.full((3 * BUF_WORDS,), SENTINEL, dtype=torch.int32, device="cuda")
    flags = torch.tensor([cur, 0, 0, 0], dtype=torch.int32, device="cuda")
    return SimpleNamespace(mc_words=mc, flags=flags, rank=rank, cur=cur)


def _sanitized_words(partial):
    bits = partial.contiguous().view(torch.int16).clone()
    bits[bits == -32768] = 0  # bf16 -0.0 (0x8000) -> +0.0
    return bits.view(torch.int32)


def _check_slots(tr, partial, rows, live=None):
    """Slots ``[:rows, rank]`` hold ``partial``'s words; rows past ``live``
    are zero words; every other word of every buffer is the sentinel."""
    buf = tr.mc_words[tr.cur * BUF_WORDS : (tr.cur + 1) * BUF_WORDS].view(MAX_ROWS, RANKS, H // 2)
    want = _sanitized_words(partial[:rows])
    assert torch.equal(buf[:rows, tr.rank], want)
    assert not bool((buf[:rows, tr.rank] == SENTINEL).any())
    if live is not None:
        assert bool((buf[live:rows, tr.rank] == 0).all())
    others = torch.ones(MAX_ROWS, RANKS, dtype=torch.bool, device="cuda")
    others[:rows, tr.rank] = False
    assert bool((buf[others] == SENTINEL).all())
    for b in range(3):
        if b != tr.cur:
            assert bool((tr.mc_words[b * BUF_WORDS : (b + 1) * BUF_WORDS] == SENTINEL).all())


def _step(tail, mla, latent, tr, out_rows=None, hop_rows=None):
    """One armed call; the attention output has ``out_rows`` rows (live rows
    first, as in ``MLA.forward``) and the request names ``hop_rows``."""
    rows = latent.shape[0]
    out_rows = rows if out_rows is None else out_rows
    hop_rows = out_rows if hop_rows is None else hop_rows
    out = torch.empty((out_rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    tail.request_into_hop(tr, hop_rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point="attn")
    assert tail.defer_w_uv(mla, meta, latent, out[:rows])
    from tensorrt_llm._torch.distributed import AllReduceParams

    return tail.project(mla, out, AllReduceParams(enable_allreduce=False))


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("grid,vector", [(0, True), (128, True), (128, False)])
def test_publish_slots_and_partial(rows, grid, vector):
    mla = tail_adapter._mla()
    cfg = TailConfig(oproj_grid=grid, into_hop_vector_publish=vector)
    tail = AttnTail("layer3", cfg)
    tr = _transport()
    latent = tail_adapter._latent(rows, seed=rows)
    partial = _step(tail, mla, latent, tr)
    torch.cuda.synchronize()
    assert tail.take_into_hop_published() and not tail.take_into_hop_published()
    ref = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, rows, config=cfg)
    assert torch.equal(partial, ref)
    _check_slots(tr, partial, rows)


@pytest.mark.parametrize("live,out_rows", [(1, 4), (2, 3), (6, 8), (7, 8), (1, 8)])
@pytest.mark.parametrize("grid,vector", [(0, True), (128, True), (128, False)])
def test_publish_padded_rows(live, out_rows, grid, vector):
    """Padded output rows (``hidden_states`` rows > ``num_tokens``): every
    output row is published, the padding as zero words, so the hop, which
    polls the residual's rows, sees all of them (and sums zeros, as the post_attention_hop
    hop does from the zero rows of the ordinary partial)."""
    mla = tail_adapter._mla()
    cfg = TailConfig(oproj_grid=grid, into_hop_vector_publish=vector)
    tail = AttnTail("layer3", cfg)
    tr = _transport(cur=0, rank=6)
    latent = tail_adapter._latent(live, seed=30 + live)
    partial = _step(tail, mla, latent, tr, out_rows=out_rows)
    torch.cuda.synchronize()
    assert tail.take_into_hop_published()
    assert tail.into_hop_published_calls == 1 and tail.into_hop_declined_calls == 0
    ref = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, live, out_rows=out_rows, config=cfg)
    assert partial.shape[0] == out_rows
    assert torch.equal(partial, ref)
    assert bool((partial[live:] == 0).all())
    _check_slots(tr, partial, out_rows, live=live)


@pytest.mark.parametrize("live,out_rows,hop_rows", [(6, 8, 6), (6, 6, 8), (4, 4, 3)])
def test_hop_row_mismatch_publishes_nothing(live, out_rows, hop_rows):
    """A request whose hop row count differs from the output rows is not
    honoured: nothing is published (the hop resets exactly the rows it
    polls, so extra or missing rows would corrupt the transport), the
    ordinary partial is unchanged and the layer runs the post_attention_hop."""
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", INTO_HOP_SCHEDULE)
    tr = _transport()
    latent = tail_adapter._latent(live, seed=70 + live)
    partial = _step(tail, mla, latent, tr, out_rows=out_rows, hop_rows=hop_rows)
    torch.cuda.synchronize()
    assert not tail.take_into_hop_published()
    assert tail.into_hop_published_calls == 0 and tail.into_hop_declined_calls == 1
    assert tail.engaged_calls == 1
    ref = attn_tail(
        latent, mla.v_b_proj, mla.o_proj.weight, live, out_rows=out_rows, config=INTO_HOP_SCHEDULE
    )
    assert torch.equal(partial, ref)
    assert bool((tr.mc_words == SENTINEL).all())


@pytest.mark.parametrize("live,out_rows", [(6, 6), (6, 8), (1, 4), (8, 8)])
@pytest.mark.parametrize("grid,vector", [(0, True), (128, True), (128, False)])
def test_publisher_synthetic_inputs(live, out_rows, grid, vector):
    """The publishing o_proj alone on fully initialized synthetic inputs
    (no MLA module, so a compute-sanitizer run sees only this kernel and
    its reference): ordinary partial bit-identical to ``stream_oproj``,
    every output row published."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.cute_oproj import (  # noqa: E501
        stream_oproj,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.into_hop import (  # noqa: E501
        stream_oproj_into_hop,
    )

    gen = torch.Generator(device="cuda").manual_seed(17 * live + out_rows)
    v = (torch.randn(live, 2048, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    w_o = (torch.randn(H, 2048, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    tr = _transport(cur=2, rank=1)
    out = torch.empty((out_rows, H), dtype=torch.bfloat16, device="cuda")
    stream_oproj_into_hop(
        v,
        w_o,
        out,
        True,
        tr.mc_words,
        tr.flags,
        tr.rank,
        grid=grid or None,
        trigger_dependents=True,
        vector_publish=vector,
    )
    ref = torch.empty((out_rows, H), dtype=torch.bfloat16, device="cuda")
    stream_oproj(v, w_o, ref, use_pdl=True, grid=grid or None)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    _check_slots(tr, out, out_rows, live=live)


def test_publisher_rejects_rows_outside_transport():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.into_hop import (  # noqa: E501
        stream_oproj_into_hop,
    )

    mla = tail_adapter._mla()
    tr = _transport()
    v = torch.zeros((6, 2048), dtype=torch.bfloat16, device="cuda")
    for out_rows in (9, 5):
        out = torch.empty((out_rows, H), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError):
            stream_oproj_into_hop(v, mla.o_proj.weight, out, True, tr.mc_words, tr.flags, tr.rank)


def test_request_consumed_and_engines():
    mla = tail_adapter._mla()
    tr = _transport()
    # Declined call: the request is consumed, nothing is published.
    tail = AttnTail("layer3", INTO_HOP_SCHEDULE)
    tail.request_into_hop(tr, 6)
    assert not tail.arm(mla, tail_adapter._meta(6, num_contexts=1))
    assert tail._into_hop_request is None and tail._into_hop is None
    assert not tail.take_into_hop_published()
    # Without a request the attention_output_tail does not publish.
    latent = tail_adapter._latent(6, seed=9)
    out = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(6)
    assert tail.arm(mla, meta)
    assert tail.defer_w_uv(mla, meta, latent, out)
    tail.project(mla, out, None)
    torch.cuda.synchronize()
    assert not tail.take_into_hop_published()
    assert bool((tr.mc_words == SENTINEL).all())
    # A non-attention_output_tail engine never publishes.
    native = AttnTail("layer3", TailConfig(oproj_engine="triton"))
    native.request_into_hop(tr, 6)
    assert native.arm(mla, meta)
    assert native._into_hop is None


def test_request_dropped_after_a_compile_declined_call():
    """Under torch.compile ``arm`` returns before it reads the request; the
    layer's ``take_into_hop_published`` drops it, so a later call without a request
    never honours it (no publication into the transport)."""
    from tensorrt_llm._torch.utils import torch_compiling

    mla = tail_adapter._mla()
    tr = _transport()
    tail = AttnTail("layer3", INTO_HOP_SCHEDULE)
    tail.request_into_hop(tr, 6)
    with torch_compiling(True):
        assert not tail.arm(mla, tail_adapter._meta(6))
    assert tail._into_hop_request is not None  # arm touches no state under compile
    assert not tail.take_into_hop_published()
    assert tail._into_hop_request is None
    latent = tail_adapter._latent(6, seed=9)
    out = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(6)
    assert tail.arm(mla, meta)
    assert tail._into_hop is None
    assert tail.defer_w_uv(mla, meta, latent, out)
    tail.project(mla, out, None)
    torch.cuda.synchronize()
    assert not tail.take_into_hop_published()
    assert bool((tr.mc_words == SENTINEL).all())


@pytest.mark.parametrize("out_rows", [6, 8])
def test_graph_capture_and_replay_publish(out_rows):
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", INTO_HOP_SCHEDULE)
    rows = 6
    tr = _transport(cur=2, rank=5)
    lat = tail_adapter._latent(rows, seed=21).contiguous()
    _step(tail, mla, lat, tr, out_rows=out_rows)
    tail.take_into_hop_published()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _step(tail, mla, lat, tr, out_rows=out_rows)
    assert tail.take_into_hop_published()
    for seed in range(4):
        tr.mc_words.fill_(SENTINEL)
        lat.copy_(tail_adapter._latent(rows, seed=800 + seed))
        graph.replay()
        torch.cuda.synchronize()
        ref = attn_tail(
            lat, mla.v_b_proj, mla.o_proj.weight, rows, out_rows=out_rows, config=INTO_HOP_SCHEDULE
        )
        assert torch.equal(captured, ref)
        _check_slots(tr, captured, out_rows, live=rows)


def test_make_attn_tail_selects_into_hop(monkeypatch):
    mla = tail_adapter._build_mla(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
    cases = [
        (
            "attention_output_tail,attention_output_into_hop,post_attention_hop,sparse_mla_decode",
            INTO_HOP_SPARSE_DECODE_SCHEDULE,
        ),
        (
            "attention_output_tail,attention_output_tail_prefetch,attention_output_into_hop,post_attention_hop,sparse_mla_decode",
            INTO_HOP_SPARSE_DECODE_SCHEDULE,
        ),
        ("attention_output_tail,attention_output_into_hop,post_attention_hop", INTO_HOP_SCHEDULE),
        (
            "attention_output_tail,attention_output_tail_prefetch,attention_output_into_hop,post_attention_hop",
            INTO_HOP_SCHEDULE,
        ),
        (
            "attention_output_tail,attention_output_tail_prefetch,post_attention_hop,sparse_mla_decode",
            TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
        ),
        (
            "attention_output_tail,attention_output_tail_prefetch,sparse_mla_decode",
            TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
        ),
        # attention_output_tail_prefetch has no schedule next to the post_attention_hop
        # without sparse_mla_decode; the tail keeps its default schedule.
        ("attention_output_tail,attention_output_tail_prefetch,post_attention_hop", DEFAULT_CONFIG),
        ("attention_output_tail,attention_output_tail_prefetch", TAIL_PREFETCH_SCHEDULE),
        ("attention_output_tail", DEFAULT_CONFIG),
    ]
    for features, want in cases:
        plan_testlib.select(dispatch, features, monkeypatch)
        tail = make_attn_tail(mla)
        assert tail is not None and tail.config == want, features


def test_attention_output_into_hop_is_off_without_the_paths_it_extends():
    for base in ("attention_output_tail", "post_attention_hop"):
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: base})
        assert not plan.enabled("attention_output_into_hop"), base
