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
"""oproj_weight_staging ``oproj_weight_staging`` (single GPU): the staging o_proj against the served
attention_output_tail.

Oracle: the served attention_output_tail/attention_output_into_hop kernels
(``attention_output_tail.kernels.attn_tail``, its ``stream_oproj`` and ``stream_oproj_into_hop``) on
the same inputs. oproj_weight_staging changes no arithmetic, so every check is bit-for-bit
(``torch.equal``), on real GLM-5.2 TP8 weight slices (``tail_fixtures.load_tail_weights``) and
real-weight latents (``tail_fixtures.mix_latent``):

* every staging configuration (L2 prefetch or shared-memory staging, at
  entry or once the early rows landed, from the o_proj or from the W_UV
  kernel, cache-policy hints, partial staging), on the served schedule
  (``INTO_HOP_SPARSE_DECODE_SCHEDULE``: 128 CTAs, 32 early rows) and on other ``early_rows``
  / grids; the BF16 ``v`` intermediate as well;
* the attention_output_into_hop publish (a device buffer stands in for the post_attention_hop
  transport, as in ``attention_output_tail/test_into_hop_producer.py``): same words, same untouched
  sentinels;
* padded rows (live < out rows, also out rows 9..16);
* CUDA-graph capture and replay with fresh contents, alternating graphs of
  different row counts;
* a delayed producer of the latent (a kernel that releases its PDL
  dependents at entry and writes the latent only after a ``%globaltimer``
  spin; the tail's W_UV kernel waits for it) and a delayed producer of the
  o_proj's own input ``v`` (the o_proj is its direct PDL dependent, so every
  o_proj CTA is resident and done staging long before its wait returns), and
  a co-running kernel on another stream;
* ``%globaltimer`` stamps of the diagnostic build are ordered;
* ``StagedAttnTail`` on the real DSA ``MLA`` module: the attention_output_tail protocol (arm,
  defer_w_uv, project, into-hop request) is unchanged and the partial equals
  attention_output_tail's; schedules oproj_weight_staging does not take keep attention_output_tail's
  tail; geometry declines run attention_output_tail.

Opt-in (``GLM_MOE_DSA_OPROJ_WEIGHT_STAGING_ADAPTER_PATCH_APPLIED=1``): the adopted attention
(``GlmMLA.adopt``) wraps the tail;
every set of fast paths containing ``oproj_weight_staging`` that a plan can run engages it (the
closure is the engaged domain, so no runnable set leaves it inert); deferred_input_prep's gate
maps ``oproj_weight_staging`` to its directories and side selector.
"""

import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "attention_output_tail"))
import tail_fixtures  # noqa: E402
import test_attn_tail_mla_adapter as tail_adapter  # noqa: E402

sys.path.insert(0, os.path.dirname(_HERE))
import plan_testlib  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E402
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    INTO_HOP_SCHEDULE,
    INTO_HOP_SPARSE_DECODE_SCHEDULE,
    TAIL_PREFETCH_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    TailConfig,
    attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E402, E501
    AttnTail,
    make_attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.oproj_weight_staging import (  # noqa: E402, E501
    DEFAULT_STAGE,
    StageConfig,
    StagedAttnTail,
    stage_tail,
    wrap_attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.oproj_weight_staging.cute_oproj_stage import (  # noqa: E402, E501
    TRACE_POINTS,
    stream_oproj_stage,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.oproj_weight_staging.kernels import (  # noqa: E402, E501
    check_stage_config,
    wuv_stage_geometry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.oproj_weight_staging.tail import (  # noqa: E402, E501
    config_reason,
    geometry_reason,
)

_CKPT = tail_fixtures.checkpoint_dir()
_PATCHED = os.environ.get("GLM_MOE_DSA_OPROJ_WEIGHT_STAGING_ADAPTER_PATCH_APPLIED") == "1"
needs_patch = pytest.mark.skipif(not _PATCHED, reason="opt-in test of the adopted attention")
# Served GEN selector (without deferred_input_prep, which the eligibility test adds).
B1_GEN = (
    "absorbed_query,indexer_topk,attention_output_tail,post_moe_hop,"
    "draft_front,router_gemm,post_attention_hop,draft_tail,"
    "sparse_mla_decode,draft_absorbed_query,rope_kv_append,draft_loop,"
    "shared_indexer_absorbed_query,routed_moe,cross_rank_topk,"
    "attention_output_tail_prefetch,attention_output_into_hop,"
    "routed_moe_into_hop,deferred_norm,dense_mlp"
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="oproj_weight_staging targets SM100/SM103",
    ),
    pytest.mark.skipif(not (_CKPT and os.path.isdir(_CKPT)), reason="checkpoint not available"),
]

TP_SIZE = 8
LOCAL_HEADS = 8
H = 6144
RANKS = 8
MAX_ROWS = 8
BUF_WORDS = MAX_ROWS * RANKS * H * 2 // 4
SENTINEL = -(2**31)

S = StageConfig
ARMS = {
    "none": S(stage="none"),
    "l2_entry": S(stage="l2", when="entry"),
    "l2_landed": S(stage="l2", when="landed"),
    "l2_landed_el": S(stage="l2", when="landed", prefetch_policy="evict_last"),
    "l2_half": S(stage="l2", when="landed", rows=8),
    "smem_entry": S(stage="smem", when="entry"),
    "smem_landed": S(stage="smem", when="landed"),
    "smem_half": S(stage="smem", when="landed", rows=8),
    "wuv": S(stage="l2", host="wuv"),
    "wuv_el": S(stage="l2", host="wuv", prefetch_policy="evict_last"),
    "ef": S(stage="none", copy_policy="evict_first"),
    "l2_landed_ef": S(stage="l2", when="landed", copy_policy="evict_first"),
    "smem_landed_ef": S(stage="smem", when="landed", copy_policy="evict_first"),
    "smem_chain": S(stage="smem", when="chain"),
    "smem_chain_e16": S(stage="smem", when="chain", early_rows=16),
    "smem_chain_e8_ef": S(stage="smem", when="chain", early_rows=8, copy_policy="evict_first"),
    "default": DEFAULT_STAGE,
}
PUBLISH_ARMS = (
    "none",
    "l2_landed",
    "smem_landed",
    "wuv",
    "l2_landed_ef",
    "smem_chain_e16",
    "default",
)

_W = {}
_COMP = {}


def _weights(layer, rank):
    if layer not in _W:
        _W[layer] = {
            r: (w_uv.cuda().contiguous(), w_o.cuda().contiguous())
            for r, (w_uv, w_o) in tail_fixtures.load_tail_weights(
                _CKPT, layer, range(TP_SIZE), TP_SIZE
            ).items()
        }
    return _W[layer][rank]


def _latent(layer, rows, seed):
    if layer not in _COMP:
        _COMP[layer] = tail_fixtures.compressed_kv_rows(_CKPT, layer)
    return tail_fixtures.mix_latent(_COMP[layer], rows, LOCAL_HEADS, seed=seed)


def _transport(cur=1, rank=3):
    mc = torch.full((3 * BUF_WORDS,), SENTINEL, dtype=torch.int32, device="cuda")
    flags = torch.tensor([cur, 0, 0, 0], dtype=torch.int32, device="cuda")
    return SimpleNamespace(mc_words=mc, flags=flags, rank=rank, cur=cur)


def _pair(layer, rank, rows, out_rows, cfg, stage, seed, into_hop_pair=False):
    """Run attention_output_tail's tail and oproj_weight_staging's tail on the same inputs; return both outputs
    (and the v intermediates, and the two transports when ``into_hop_pair``)."""
    w_uv, w_o = _weights(layer, rank)
    lat = _latent(layer, rows, seed)
    v_attn_tail = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    v_oproj_stage = torch.empty_like(v_attn_tail)
    tr_attn_tail = _transport() if into_hop_pair else None
    tr_oproj_stage = _transport() if into_hop_pair else None
    ref = attn_tail(
        lat,
        w_uv,
        w_o,
        rows,
        out_rows=out_rows,
        v_scratch=v_attn_tail,
        config=cfg,
        hop_transport=tr_attn_tail,
    )
    got = stage_tail(
        lat,
        w_uv,
        w_o,
        rows,
        out_rows=out_rows,
        v_scratch=v_oproj_stage,
        config=cfg,
        stage=stage,
        hop_transport=tr_oproj_stage,
    )
    torch.cuda.synchronize()
    return ref, got, v_attn_tail, v_oproj_stage, tr_attn_tail, tr_oproj_stage


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("layer,rank", [(3, 0), (77, 7)])
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_served_schedule_bit_identical(arm, layer, rank, rows):
    ref, got, v_attn_tail, v_oproj_stage, _, _ = _pair(
        layer, rank, rows, rows, INTO_HOP_SPARSE_DECODE_SCHEDULE, ARMS[arm], seed=10 * rows + rank
    )
    assert torch.equal(v_attn_tail, v_oproj_stage)
    assert torch.equal(ref, got), (arm, layer, rank, rows)


@pytest.mark.parametrize("arm", PUBLISH_ARMS)
@pytest.mark.parametrize("live,out_rows", [(1, 1), (6, 6), (8, 8), (6, 8), (1, 4)])
def test_into_hop_publish_bit_identical(arm, live, out_rows):
    ref, got, _, _, tr_attn_tail, tr_oproj_stage = _pair(
        40,
        3,
        live,
        out_rows,
        INTO_HOP_SPARSE_DECODE_SCHEDULE,
        ARMS[arm],
        seed=live + out_rows,
        into_hop_pair=True,
    )
    assert torch.equal(ref, got)
    assert torch.equal(tr_attn_tail.mc_words, tr_oproj_stage.mc_words), arm
    buf = tr_oproj_stage.mc_words[
        tr_oproj_stage.cur * BUF_WORDS : (tr_oproj_stage.cur + 1) * BUF_WORDS
    ].view(MAX_ROWS, RANKS, H // 2)
    assert not bool((buf[:out_rows, tr_oproj_stage.rank] == SENTINEL).any())


@pytest.mark.parametrize(
    "arm", ["none", "l2_entry", "l2_landed", "smem_landed", "smem_chain_e16", "wuv", "default"]
)
@pytest.mark.parametrize("live,out_rows,publish", [(6, 6, True), (1, 4, True), (6, 12, False)])
def test_synthetic_inputs_bit_identical(arm, live, out_rows, publish):
    """Every input written by plain stores (``torch.randn``/``full``), for
    compute-sanitizer initcheck: the real-weight fixtures run cuBLAS GEMMs
    whose outputs initcheck reports as uninitialized when torch reads them."""
    gen = torch.Generator(device="cuda").manual_seed(live + out_rows)
    w_uv = (torch.randn((8, 256, 512), generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    w_o = (torch.randn((H, 2048), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    lat = torch.randn((live, 8, 512), generator=gen, device="cuda").to(torch.bfloat16)
    tr_attn_tail = _transport() if publish else None
    tr_oproj_stage = _transport() if publish else None
    ref = attn_tail(
        lat,
        w_uv,
        w_o,
        live,
        out_rows=out_rows,
        config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
        hop_transport=tr_attn_tail,
    )
    got = stage_tail(
        lat,
        w_uv,
        w_o,
        live,
        out_rows=out_rows,
        config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
        stage=ARMS[arm],
        hop_transport=tr_oproj_stage,
    )
    torch.cuda.synchronize()
    assert torch.equal(ref, got)
    if publish:
        assert torch.equal(tr_attn_tail.mc_words, tr_oproj_stage.mc_words)


@pytest.mark.parametrize(
    "cfg_name,cfg",
    [
        ("attn_tail_default", TailConfig()),
        ("g128_e24", TailConfig(oproj_grid=128, early_rows=24)),
        ("g128_e40", TailConfig(oproj_grid=128, early_rows=40)),
        ("g128_e48", TailConfig(oproj_grid=128, early_rows=48)),
        ("attention_output_into_hop", INTO_HOP_SCHEDULE),
    ],
)
@pytest.mark.parametrize(
    "arm", ["l2_entry", "l2_landed", "smem_landed", "smem_chain", "default", "ef"]
)
def test_other_schedules_bit_identical(cfg_name, cfg, arm):
    ref, got, v_attn_tail, v_oproj_stage, _, _ = _pair(3, 5, 6, 6, cfg, ARMS[arm], seed=7)
    assert torch.equal(v_attn_tail, v_oproj_stage)
    assert torch.equal(ref, got), (cfg_name, arm)


@pytest.mark.parametrize("early", [16, 24, 40])
def test_early_rows_override(early):
    stage = S(stage="l2", when="landed", early_rows=early)
    cfg = dataclasses.replace(INTO_HOP_SPARSE_DECODE_SCHEDULE, early_rows=early)
    w_uv, w_o = _weights(3, 2)
    lat = _latent(3, 6, 99)
    ref = attn_tail(lat, w_uv, w_o, 6, config=cfg)
    got = stage_tail(lat, w_uv, w_o, 6, config=INTO_HOP_SPARSE_DECODE_SCHEDULE, stage=stage)
    torch.cuda.synchronize()
    assert torch.equal(ref, got)


@pytest.mark.parametrize("live,out_rows", [(6, 12), (8, 16), (1, 9)])
@pytest.mark.parametrize("arm", ["l2_landed", "smem_landed", "wuv", "default"])
def test_wide_padded_rows(live, out_rows, arm):
    ref, got, _, _, _, _ = _pair(
        2, 1, live, out_rows, INTO_HOP_SPARSE_DECODE_SCHEDULE, ARMS[arm], seed=live
    )
    assert torch.equal(ref, got)
    assert bool((got[live:] == 0).all())


@pytest.mark.parametrize("arm", ["l2_landed", "smem_landed", "wuv", "default"])
def test_graph_replay_changing_rows(arm):
    """Graphs for 1 live row (4 out rows) and 6 live rows (8 out rows),
    replayed alternately with fresh latents; each replay equals attention_output_tail."""
    w_uv, w_o = _weights(3, 0)
    stage = ARMS[arm]
    graphs = {}
    for rows, out_rows in ((1, 4), (6, 8)):
        lat = _latent(3, rows, 0).clone()
        stage_tail(
            lat,
            w_uv,
            w_o,
            rows,
            out_rows=out_rows,
            config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
            stage=stage,
        )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = stage_tail(
                lat,
                w_uv,
                w_o,
                rows,
                out_rows=out_rows,
                config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
                stage=stage,
            )
        graphs[rows] = (g, lat, out, out_rows)
    for step in range(12):
        rows = (1, 6)[step % 2]
        g, lat, out, out_rows = graphs[rows]
        lat.copy_(_latent(3, rows, 100 + step))
        g.replay()
        ref = attn_tail(
            lat, w_uv, w_o, rows, out_rows=out_rows, config=INTO_HOP_SPARSE_DECODE_SCHEDULE
        )
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (arm, step)


@triton.jit
def _globaltimer(like):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [like], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _slow_copy_kernel(src, dst, n, delay_ns, stamps, BLOCK: tl.constexpr):
    """One CTA, launched with PDL (Triton emits the ``griddepcontrol``
    instructions only for such launches): waits for its predecessor,
    releases its own dependents, spins ``delay_ns`` on ``%globaltimer``, then
    copies ``src`` into ``dst`` (the latent). One CTA leaves SMs free, so
    o_proj CTAs (197,400 B of shared memory each) can become resident next
    to the parked W_UV grid."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    start = _globaltimer(pid)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(pid)
    for base in range(0, n, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)
    # Start of the spin and end of the latent's stores (``%globaltimer``).
    tl.store(stamps, start)
    tl.store(stamps + 1, _globaltimer(pid))


@triton.jit
def _hold_kernel(delay_ns, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = _globaltimer(pid)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(pid)


def _hold_stream(delay_ns=2_000_000):
    """Occupy the stream with one spinning CTA (no PDL) while the host
    enqueues the launches that follow. Without it the host's launch latency
    (tens of us per Triton / CuTe launch) exceeds the producer's delay, the
    producer has completed before its PDL dependents are even enqueued, and
    they never park in ``griddepcontrol.wait``."""
    _hold_kernel[(1,)](delay_ns, BLOCK=1)


def _slow_produce(src, dst, delay_ns, stamps=None):
    if stamps is None:
        stamps = torch.zeros(2, dtype=torch.int64, device=src.device)
    _slow_copy_kernel[(1,)](src, dst, src.numel(), delay_ns, stamps, BLOCK=1024, launch_pdl=True)
    return stamps


@pytest.mark.parametrize(
    "arm",
    ["l2_entry", "l2_landed", "smem_landed", "wuv", "l2_landed_ef", "smem_chain_e16", "default"],
)
@pytest.mark.parametrize("delay_ns", [0, 3000, 20000])
def test_delayed_producer(arm, delay_ns):
    """The latent's producer is late (it releases its PDL dependents, then
    spins ``delay_ns`` before writing the latent; a held stream lets the
    host enqueue the tail before the producer starts). Outputs equal attention_output_tail's
    under the same producer, and no o_proj CTA leaves its wait before the
    producer's last store (``%globaltimer`` stamps of both kernels)."""
    w_uv, w_o = _weights(40, 6)
    rows = 6
    src = _latent(40, rows, 5).contiguous()
    lat_attn_tail = torch.zeros_like(src)
    lat_oproj_stage = torch.zeros_like(src)
    tr = torch.zeros(128 * TRACE_POINTS, dtype=torch.int64, device="cuda")
    for _ in range(2):
        _hold_stream()
        _slow_produce(src, lat_attn_tail, delay_ns)
        ref = attn_tail(lat_attn_tail, w_uv, w_o, rows, config=INTO_HOP_SPARSE_DECODE_SCHEDULE)
        _hold_stream()
        ps = _slow_produce(src, lat_oproj_stage, delay_ns)
        got = stage_tail(
            lat_oproj_stage,
            w_uv,
            w_o,
            rows,
            config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
            stage=ARMS[arm],
            trace=tr,
        )
        torch.cuda.synchronize()
        assert torch.equal(ref, got), (arm, delay_ns)
        st = tr.view(128, TRACE_POINTS).cpu()
        p_start, p_end = (int(x) for x in ps.cpu())
        assert p_end - p_start >= delay_ns
        # Every o_proj CTA passed its wait only after the producer's stores
        # (through W_UV's completion), however early it became resident.
        assert int(st[:, 1].min()) >= p_end, (arm, delay_ns)
    parked = (st[:, 1] - st[:, 0]).float()
    print(
        f"delayed producer {delay_ns} ns, {arm}: o_proj resident "
        f"{float(parked.mean()):.0f} ns (mean) / {float(parked.max()):.0f} "
        f"ns (max) before its wait returned"
    )


@pytest.mark.parametrize(
    "arm",
    ["none", "l2_entry", "l2_landed", "smem_landed", "l2_landed_ef", "smem_chain_e16", "default"],
)
@pytest.mark.parametrize("publish", [False, True])
def test_parked_oproj_bit_identical(arm, publish):
    """The o_proj is the direct PDL dependent of a one-CTA producer that
    releases its dependents at entry and writes ``v`` only after a 20 us
    spin (behind a held stream, so the host has enqueued the o_proj before
    the producer starts). All 128 o_proj CTAs fit next to it, so each one
    issues its pre-wait copies and staging and then stays parked in
    ``griddepcontrol.wait`` for most of the spin (the case where the staging
    has long completed when the wait returns). Outputs (and the into-hop words) equal the served
    tail's on the same latent, for the plain and the stamped build; no CTA leaves its wait
    before the producer's last store; the median CTA was parked at least
    half the delay."""
    delay_ns = 20000
    w_uv, w_o = _weights(40, 6)
    rows = 6
    stage = ARMS[arm]
    early = (
        INTO_HOP_SPARSE_DECODE_SCHEDULE.early_rows if stage.early_rows is None else stage.early_rows
    )
    lat = _latent(40, rows, 21)
    v_src = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    tr_attn_tail = _transport() if publish else None
    ref = attn_tail(
        lat,
        w_uv,
        w_o,
        rows,
        v_scratch=v_src,
        config=INTO_HOP_SPARSE_DECODE_SCHEDULE,
        hop_transport=tr_attn_tail,
    )
    torch.cuda.synchronize()
    for trace in (False, True, True):
        # The first stamped run compiles that build (host time beyond the
        # held stream); the parked time is asserted on the second.
        tr = torch.zeros(128 * TRACE_POINTS, dtype=torch.int64, device="cuda") if trace else None
        v = torch.zeros_like(v_src)
        got = torch.empty_like(ref)
        tr_oproj_stage = _transport() if publish else None
        torch.cuda.synchronize()
        _hold_stream()
        ps = _slow_produce(v_src, v, delay_ns)
        stream_oproj_stage(
            v,
            w_o,
            got,
            use_pdl=True,
            early_rows=early,
            grid=128,
            trigger_dependents=True,
            stage=stage.stage,
            when=stage.when,
            stage_rows=stage.rows,
            prefetch_policy=stage.prefetch_policy,
            copy_policy=stage.copy_policy,
            hop_transport=tr_oproj_stage,
            trace=tr,
        )
        torch.cuda.synchronize()
        assert torch.equal(ref, got), (arm, publish, trace)
        if publish:
            assert torch.equal(tr_attn_tail.mc_words, tr_oproj_stage.mc_words), (arm, trace)
        if not trace:
            continue
        st = tr.view(128, TRACE_POINTS).cpu()
        p_start, p_end = (int(x) for x in ps.cpu())
        assert p_end - p_start >= delay_ns
        assert int(st[:, 1].min()) >= p_end, arm
        parked = (st[:, 1] - st[:, 0]).float()
        print(
            f"parked o_proj {arm} publish={publish}: resident "
            f"{float(parked.median()):.0f} ns (median) / "
            f"{float(parked.min()):.0f} ns (min) before its wait returned"
        )
    assert float(parked.median()) >= delay_ns / 2, arm


@pytest.mark.parametrize("arm", ["l2_landed", "smem_landed", "wuv", "default"])
def test_co_running_kernel_progress(arm):
    """A long GEMM on another stream occupies SMs while the tail runs; the
    tail completes with attention_output_tail's result (no wait in oproj_weight_staging depends on
    co-residency)."""
    w_uv, w_o = _weights(3, 4)
    rows = 6
    lat = _latent(3, rows, 17)
    ref = attn_tail(lat, w_uv, w_o, rows, config=INTO_HOP_SPARSE_DECODE_SCHEDULE)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    outs = []
    for _ in range(3):
        with torch.cuda.stream(side):
            c = a @ a
        outs.append(
            stage_tail(
                lat, w_uv, w_o, rows, config=INTO_HOP_SPARSE_DECODE_SCHEDULE, stage=ARMS[arm]
            )
        )
    torch.cuda.synchronize()
    del c
    for got in outs:
        assert torch.equal(ref, got)


def test_trace_stamps_ordered():
    w_uv, w_o = _weights(3, 0)
    lat = _latent(3, 6, 3)
    tr = torch.zeros(128 * TRACE_POINTS, dtype=torch.int64, device="cuda")
    out = stage_tail(
        lat, w_uv, w_o, 6, config=INTO_HOP_SPARSE_DECODE_SCHEDULE, stage=ARMS["l2_landed"], trace=tr
    )
    torch.cuda.synchronize()
    ref = attn_tail(lat, w_uv, w_o, 6, config=INTO_HOP_SPARSE_DECODE_SCHEDULE)
    assert torch.equal(out, ref)
    st = tr.view(128, TRACE_POINTS).cpu()
    assert bool((st > 0).all())
    for k in range(TRACE_POINTS - 1):
        assert bool((st[:, k + 1] >= st[:, k]).all()), k


def test_stage_rejects_bad_config():
    w_uv, w_o = _weights(3, 0)
    v = torch.zeros(6, 2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(6, H, dtype=torch.bfloat16, device="cuda")
    for bad in (
        dict(stage="tma"),
        dict(when="late"),
        dict(prefetch_policy="evict_first"),
        dict(copy_policy="evict_last"),
        dict(when="chain"),
    ):
        kwargs = dict(stage="l2", when="landed", prefetch_policy="normal", copy_policy="none")
        kwargs.update(bad)
        with pytest.raises(ValueError):
            stream_oproj_stage(v, w_o, out, use_pdl=True, early_rows=32, grid=128, **kwargs)


def test_wuv_geometry():
    assert wuv_stage_geometry(H, 2048, 128, 32, -1) == (32, 16 * 4096)
    assert wuv_stage_geometry(H, 2048, 128, 32, 8) == (32, 8 * 4096)
    # 152 CTAs: 40 or 42 rows per CTA, not covered evenly.
    assert wuv_stage_geometry(H, 2048, 152, 16, -1) is None
    assert wuv_stage_geometry(H, 2048, 128, 48, -1) is None


def test_the_served_stage_is_valid():
    assert check_stage_config(DEFAULT_STAGE) is None
    assert (DEFAULT_STAGE.stage, DEFAULT_STAGE.when, DEFAULT_STAGE.early_rows) == (
        "smem",
        "landed",
        24,
    )
    assert check_stage_config(dataclasses.replace(DEFAULT_STAGE, stage="l2", when="chain"))


# ---- StagedAttnTail on the real DSA MLA module (attention_output_tail protocol unchanged) -------


def _step(tail, mla, latent, tr=None, out_rows=None):
    from tensorrt_llm._torch.distributed import AllReduceParams

    rows = latent.shape[0]
    out_rows = rows if out_rows is None else out_rows
    out = torch.empty((out_rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    if tr is not None:
        tail.request_into_hop(tr, out_rows)
    assert tail.arm(mla, meta)
    assert tail.defer_w_uv(mla, meta, latent, out[:rows])
    return tail.project(mla, out, AllReduceParams(enable_allreduce=False))


@pytest.mark.parametrize("arm", ["l2_landed", "smem_landed", "wuv", "default"])
@pytest.mark.parametrize("rows,out_rows", [(6, 6), (6, 8), (1, 1)])
def test_oproj_stage_tail_matches_attn_tail(arm, rows, out_rows):
    mla = tail_adapter._mla()
    served_tail = AttnTail("layer3", INTO_HOP_SPARSE_DECODE_SCHEDULE)
    oproj_stage_tail = wrap_attn_tail(
        mla, AttnTail("layer3", INTO_HOP_SPARSE_DECODE_SCHEDULE), ARMS[arm]
    )
    assert isinstance(oproj_stage_tail, StagedAttnTail)
    lat = tail_adapter._latent(rows, seed=rows + out_rows)
    tr_attn_tail, tr_oproj_stage = _transport(cur=0, rank=2), _transport(cur=0, rank=2)
    ref = _step(served_tail, mla, lat, tr_attn_tail, out_rows)
    got = _step(oproj_stage_tail, mla, lat, tr_oproj_stage, out_rows)
    torch.cuda.synchronize()
    assert torch.equal(ref, got)
    assert torch.equal(tr_attn_tail.mc_words, tr_oproj_stage.mc_words)
    assert served_tail.take_into_hop_published() and oproj_stage_tail.take_into_hop_published()
    assert oproj_stage_tail.staging_calls == 1 and oproj_stage_tail.engaged_calls == 1
    assert oproj_stage_tail.staging_reason is None
    # No packet left: the native projection would follow.
    assert oproj_stage_tail.project(mla, torch.empty(rows, 2048, device="cuda"), None) is None


def test_oproj_stage_tail_declines_like_attn_tail():
    mla = tail_adapter._mla()
    tail = wrap_attn_tail(
        mla, AttnTail("layer3", INTO_HOP_SPARSE_DECODE_SCHEDULE), ARMS["l2_landed"]
    )
    lat = tail_adapter._latent(6, seed=0)
    out = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    assert not tail.defer_w_uv(mla, tail_adapter._meta(6), lat, out)
    assert not tail.arm(mla, tail_adapter._meta(6, num_contexts=1))
    assert not tail.arm(mla, tail_adapter._meta(9))
    assert tail.declined_calls == 2 and tail.staging_calls == 0
    # torch.compile policy: arm declines without touching state.
    from tensorrt_llm._torch.utils import torch_compiling

    with torch_compiling(True):
        assert not tail.arm(mla, tail_adapter._meta(6))
    assert tail.declined_calls == 2


def test_wrap_keeps_attn_tail_for_other_schedules():
    mla = tail_adapter._mla()
    for cfg in (
        TAIL_PREFETCH_SCHEDULE,
        TailConfig(oproj_engine="native"),
        TailConfig(oproj_engine="fused"),
        TailConfig(trigger_late=True),
        TailConfig(),
        INTO_HOP_SCHEDULE,
    ):
        tail = AttnTail("layer3", cfg)
        assert config_reason(cfg, StageConfig()) is not None
        assert wrap_attn_tail(mla, tail, StageConfig()) is tail
    assert wrap_attn_tail(mla, None, StageConfig()) is None
    served = AttnTail("layer3", INTO_HOP_SPARSE_DECODE_SCHEDULE)
    assert wrap_attn_tail(mla, served, StageConfig(host="sparse_mla_decode")) is served
    wrapped = wrap_attn_tail(mla, served, None)
    assert isinstance(wrapped, StagedAttnTail)
    assert wrapped.stage == DEFAULT_STAGE


def test_geometry_decline_runs_attn_tail():
    """W_UV host on the 152-CTA attention_output_tail default schedule cannot cover the rows:
    the layer logs an oproj_weight_staging decline and runs attention_output_tail's own project."""
    mla = tail_adapter._mla()
    cfg = TailConfig()
    assert geometry_reason(mla, cfg, ARMS["wuv"]) is not None
    assert geometry_reason(mla, INTO_HOP_SPARSE_DECODE_SCHEDULE, ARMS["wuv"]) is None
    tail = StagedAttnTail("layer3", cfg, ARMS["wuv"])
    lat = tail_adapter._latent(6, seed=4)
    got = _step(tail, mla, lat)
    ref = _step(AttnTail("layer3", cfg), mla, lat)
    torch.cuda.synchronize()
    assert torch.equal(ref, got)
    assert tail.staging_calls == 0 and tail.staging_reason is not None


@needs_patch
def test_adoption_attaches_oproj_stage_tail(monkeypatch):
    """With ``oproj_weight_staging`` running on its closure (alone and with
    attention_output_tail_prefetch, as served), the attention adoption wraps the
    attention_output_tail of a target layer; the MTP layer keeps no tail; without
    ``oproj_weight_staging`` the plain attention_output_tail ``AttnTail``.
    """
    for feats in (
        "attention_output_tail,post_attention_hop,"
        "sparse_mla_decode,attention_output_into_hop,"
        "oproj_weight_staging",
        "attention_output_tail,post_attention_hop,"
        "sparse_mla_decode,attention_output_tail_prefetch,"
        "attention_output_into_hop,oproj_weight_staging",
    ):
        plan_testlib.select(dispatch, feats, monkeypatch)
        mla = tail_adapter._build_adopted_mla(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
        assert isinstance(mla._attn_tail, StagedAttnTail), feats
        assert mla._attn_tail.config == INTO_HOP_SPARSE_DECODE_SCHEDULE
        assert mla._attn_tail.stage == DEFAULT_STAGE
        mtp = tail_adapter._build_adopted_mla(layer_idx=78, num_hidden_layers=78)
        assert mtp._attn_tail is None
    plan_testlib.select(
        dispatch,
        "attention_output_tail,post_attention_hop,sparse_mla_decode,"
        "attention_output_tail_prefetch,attention_output_into_hop",
        monkeypatch,
    )
    mla = tail_adapter._build_adopted_mla(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
    assert type(mla._attn_tail) is AttnTail
    assert mla._attn_tail.config == INTO_HOP_SPARSE_DECODE_SCHEDULE


HOST_SCHEDULE_PATHS = (
    "attention_output_tail",
    "post_attention_hop",
    "sparse_mla_decode",
    "attention_output_into_hop",
)


@needs_patch
def test_oproj_stage_runs_only_with_its_host_schedule():
    """``oproj_weight_staging`` stages only in ``INTO_HOP_SPARSE_DECODE_SCHEDULE``
    (attention_output_tail's tail with attention_output_into_hop and post_attention_hop behind
    sparse_mla_decode): the plan turns it off with any of attention_output_tail,
    post_attention_hop, sparse_mla_decode, attention_output_into_hop instead of running
    attention_output_tail under oproj_weight_staging's name. This includes disabling only
    ``attention_output_into_hop``: the attention_output_tail_prefetch schedule that then runs has
    the same value as ``INTO_HOP_SPARSE_DECODE_SCHEDULE`` but no into-hop publication."""
    assert TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE == INTO_HOP_SPARSE_DECODE_SCHEDULE
    assert dispatch.REQUIRED_FEATURES["oproj_weight_staging"] == set(HOST_SCHEDULE_PATHS)
    for base in HOST_SCHEDULE_PATHS:
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: base})
        assert not plan.enabled("oproj_weight_staging"), base
    assert plan_testlib.plan_running(
        dispatch, HOST_SCHEDULE_PATHS + ("oproj_weight_staging",)
    ).enabled("oproj_weight_staging")
    b1 = frozenset(B1_GEN.split(","))
    for paths in (b1, b1 | {"deferred_input_prep"}):
        assert plan_testlib.plan_running(dispatch, paths | {"oproj_weight_staging"}).enabled(
            "oproj_weight_staging"
        )


@needs_patch
def test_every_runnable_selection_engages_oproj_stage(monkeypatch):
    """Over every subset of the paths ``make_attn_tail`` reads (attention_output_tail, post_attention_hop,
    sparse_mla_decode, attention_output_tail_prefetch, attention_output_into_hop) plus
    ``oproj_weight_staging``: each set a plan can run gives a target layer an
    attention_output_tail with ``INTO_HOP_SPARSE_DECODE_SCHEDULE`` that ``wrap_attn_tail`` wraps, so
    ``oproj_weight_staging`` never runs inert."""
    import itertools

    mla = SimpleNamespace(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
    others = (
        "attention_output_tail",
        "post_attention_hop",
        "sparse_mla_decode",
        "attention_output_tail_prefetch",
        "attention_output_into_hop",
    )
    runnable = []
    for n in range(len(others) + 1):
        for combo in itertools.combinations(others, n):
            ids = combo + ("oproj_weight_staging",)
            if not dispatch.REQUIRED_FEATURES["oproj_weight_staging"] <= set(ids):
                continue
            plan_testlib.select(dispatch, ids, monkeypatch)
            tail = wrap_attn_tail(mla, make_attn_tail(mla))
            assert isinstance(tail, StagedAttnTail), ids
            assert tail.config == INTO_HOP_SPARSE_DECODE_SCHEDULE, ids
            runnable.append(",".join(ids))
    assert sorted(runnable) == sorted(
        [
            "attention_output_tail,post_attention_hop,sparse_mla_decode,attention_output_tail_prefetch,attention_output_into_hop,oproj_weight_staging",
            "attention_output_tail,post_attention_hop,sparse_mla_decode,attention_output_into_hop,oproj_weight_staging",
        ]
    )
