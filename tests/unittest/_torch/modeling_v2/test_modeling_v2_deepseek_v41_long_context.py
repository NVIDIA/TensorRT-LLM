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
"""The DeepSeek-V4.1-Flash-NVFP4 target's long-context paths, on the real kernels.

* The prefill index pass runs its query rows in tiles bounded by a logits budget; every
  op in it is per row, so the top-k and the candidate-block masks are bitwise those of
  one untiled pass.
* Under decoder replay the decoder pass scores its tail rows' candidate-block masks from
  the candidate source's query rows it carried, against the step's gathered index K.
  Those are the masks (and the re-index top-k) a full prefill gives the same rows, even
  where the request's keys sit at another offset of the gather.
* Under native KV storage the attention reads the compressed rows from one buffer in either
  of two forms, per request or per query (the selected top-k rows only); the attention
  output is bitwise the same, and each step takes the form it fits that dequantizes less.
* Past a byte budget a step keeps no candidate masks: each re-index layer re-scores its
  tiles' masks from the candidate source's query rows, with the same result.
* The decoder-replay planner takes prompts past the candidate blocks' 2048 x 8 positions.

Each case drives the target's own methods on a stand-in core holding just the attributes
they read, and a stand-in prefill target holding just the entries they call, bound as the
prefill target binds them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    modeling,
)

Core = modeling.DeepseekV41UnifiedModelingV2
Prefill = modeling.PrefillTarget
CAND_SRC, REINDEX, PLAIN = 20, 24, 14
CANDIDATE_POSITIONS = modeling._CAND_TOPK_BLOCKS * modeling._CAND_BLOCK


@pytest.mark.parametrize("rows", [1, 63, 500, 8192])
@pytest.mark.parametrize("s_len", [1, 16384, 20500, 1 << 20, 1 << 23])
def test_index_row_tiles(rows, s_len):
    """The tiles cover the rows once, in order, each within the budget (one row at least)."""
    tiles = modeling._index_row_tiles(rows, s_len)
    width = -(-(s_len + 256) // 256) * 256
    assert tiles[0][0] == 0 and tiles[-1][1] == rows
    assert all(a < b for a, b in tiles) and all(
        tiles[n][1] == tiles[n + 1][0] for n in range(len(tiles) - 1)
    )
    if rows * width <= modeling._INDEX_TILE_ELEMS:
        assert tiles == [(0, rows)]
    for a, b in tiles:
        assert (b - a) * width <= modeling._INDEX_TILE_ELEMS or b - a == 1
    if len(tiles) > 1 and modeling._INDEX_TILE_ELEMS // width >= 64:
        assert all((b - a) % 64 == 0 for a, b in tiles[:-1])


def _core():
    """A stand-in prefill target over a stand-in core: the index pass's entries."""
    core = SimpleNamespace(
        cand_src=CAND_SRC,
        ratios=[1] * 40,
        _const={"zero_i32_rows": torch.zeros(8192, dtype=torch.int32, device="cuda")},
    )
    target = SimpleNamespace(core=core)
    target._mqa_logits = modeling.DeepgemmFp8Fp4MqaLogits()
    target._mqa_logits.bind_const(clean_logits=True, max_seqlen_k=0, logits_dtype=torch.float32)
    target._cand = modeling.CandidateBlockSelect()
    target._cand.bind_const(
        topk_blocks=modeling._CAND_TOPK_BLOCKS, block_size=modeling._CAND_BLOCK, next_n=1
    )
    target._topk = modeling.FlashinferTopKRaggedDeterministic()
    target._topk.bind_const(k=modeling._TOPK)
    target._cand_masks = lambda *a: Prefill._cand_masks(target, *a)
    return target


def _query(rows, seed):
    """MXFP4 index query rows and fp32 head weights, as the target's _index makes them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 32, 128, generator=g, device="cuda").to(torch.bfloat16)
    qd, qsf = modeling.Fp4Quantize()(q, None, 32, True, False)
    w = torch.randn(rows, 32, generator=g, device="cuda") * 2.0**-6
    return (
        qd.view(torch.int8).contiguous(),
        qsf.view(torch.int32).reshape(rows, 32).contiguous(),
        w.contiguous(),
    )


def _keys(n, seed):
    """n gathered index-K rows (MXFP4 codes, one packed UE8M0 word each)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(n, 128, generator=g, device="cuda").to(torch.bfloat16)
    kd, ksf = modeling.Fp4Quantize()(k, None, 32, True, False)
    return kd.view(torch.int8).contiguous(), ksf.view(torch.int32).reshape(-1).contiguous()


def _rows(q, sel):
    return tuple(t[sel].contiguous() for t in q)


class _Step:
    """Context rows of requests in step order; request r's keys at offset ks_off[r] of the
    gather. Ratio 1: a row at position p sees p + 1 keys."""

    def __init__(self, reqs, key_order):
        # reqs: list of (first position, rows); key_order: request indices in gather order.
        self.reqs = reqs
        total = [p0 + n for p0, n in reqs]
        off, self.ks_off = 0, {}
        for r in key_order:
            self.ks_off[r] = off
            off += total[r]
        self.s_len = off
        pos, ks = [], []
        for r, (p0, n) in enumerate(reqs):
            pos += list(range(p0, p0 + n))
            ks += [self.ks_off[r]] * n
        self.pos = torch.tensor(pos, dtype=torch.int32, device="cuda")
        self.ks = torch.tensor(ks, dtype=torch.int32, device="cuda")
        self.vis = self.pos + 1
        self.ke = self.ks + self.vis
        self.rows = len(pos)

    def gather(self, keys_by_req):
        order = sorted(self.ks_off, key=self.ks_off.get)
        return (
            torch.cat([keys_by_req[r][0] for r in order]).contiguous(),
            torch.cat([keys_by_req[r][1] for r in order]).contiguous(),
        )


def _run(core, layer, q, step, kvc, cache, tail_bound):
    out = torch.empty(step.rows, modeling._TOPK, dtype=torch.int32, device="cuda")
    Prefill._index_prefill(
        core,
        layer,
        *q,
        kvc,
        step.ks,
        step.ke,
        step.vis,
        step.s_len,
        {"tail_bound": tail_bound},
        cache,
        out,
    )
    return out


def _masks(cache):
    return torch.cat(cache["cand_ctx"], 0)


# Request 0 is long enough for its candidate blocks to bind; request 1 is short.
REQS = [(20000, 300), (0, 200)]


@pytest.fixture(scope="module")
def data():
    keys = {0: _keys(REQS[0][0] + REQS[0][1], 1), 1: _keys(REQS[1][0] + REQS[1][1], 2)}
    rows = sum(n for _, n in REQS)
    return keys, _query(rows, 3), _query(rows, 4)


@skip_pre_blackwell
@pytest.mark.parametrize("tile_rows", [64, 37])
def test_prefill_index_tiling_is_exact(data, tile_rows, monkeypatch):
    """The candidate source's masks and the re-index layer's top-k are the same in one
    pass and in row tiles (64-row tiles, and 37-row ones that start at odd offsets)."""
    keys, q20, q24 = data
    step = _Step(REQS, [0, 1])
    assert step.s_len > CANDIDATE_POSITIONS
    kvc = step.gather(keys)
    core = _core()

    def passes():
        cache = {}
        idx20 = _run(core, CAND_SRC, q20, step, kvc, cache, False)
        idx24 = _run(core, REINDEX, q24, step, kvc, cache, False)
        return idx20, idx24, _masks(cache), len(cache["cand_ctx"])

    one = passes()
    assert one[3] == 1
    width = -(-(step.s_len + 256) // 256) * 256
    monkeypatch.setattr(modeling, "_INDEX_TILE_ELEMS", width * tile_rows)
    tiled = passes()
    assert tiled[3] == -(-step.rows // tile_rows)
    for a, b in zip(one[:3], tiled[:3]):
        assert torch.equal(a, b)

    # The masks bind on the long request's rows (so the re-index top-k depends on them):
    # the same query without masks (a layer below the source) selects differently there.
    masks = one[2]
    long_rows = slice(0, REQS[0][1])
    visible = [masks[t, : int(step.vis[t])] for t in range(REQS[0][1])]
    assert all(int(v.sum()) <= CANDIDATE_POSITIONS < v.numel() for v in visible)
    unmasked = _run(core, PLAIN, q24, step, kvc, {}, False)
    assert not torch.equal(unmasked[long_rows], one[1][long_rows])
    assert torch.equal(unmasked[REQS[0][1] :], one[1][REQS[0][1] :])


@skip_pre_blackwell
def test_tail_masks_match_full_prefill(data):
    """Decoder replay: the decoder pass's masks and re-index top-k of each request's last
    128 rows equal the rows of a full prefill, although the final step gathers the
    requests' keys in the other order (every tail row's range moves)."""
    keys, q20, q24 = data
    core = _core()
    full = _Step(REQS, [0, 1])
    kvc = full.gather(keys)
    cache = {}
    _run(core, CAND_SRC, q20, full, kvc, cache, False)
    idx24_full = _run(core, REINDEX, q24, full, kvc, cache, False)
    masks_full = _masks(cache)

    w = 128
    tail_reqs = [(p0 + n - w, w) for p0, n in REQS]
    sel = torch.cat(
        [torch.arange(n - w, n) for _, n in REQS[:1]]
        + [torch.arange(REQS[0][1] + REQS[1][1] - w, REQS[0][1] + REQS[1][1])]
    ).cuda()
    tail = _Step(tail_reqs, [1, 0])
    kvc_t = tail.gather(keys)
    cache_t = {"cand_q": _rows(q20, sel)}
    # Under decoder replay the candidate source scores no mask over the engine's rows.
    probe = {}
    _run(core, CAND_SRC, q20, full, kvc, probe, True)
    assert "cand_ctx" not in probe
    idx24_tail = _run(core, REINDEX, _rows(q24, sel), tail, kvc_t, cache_t, True)
    assert torch.equal(idx24_tail, idx24_full[sel])
    masks_tail = _masks(cache_t)
    for t in range(tail.rows):
        v = int(tail.vis[t])
        src = int(sel[t])
        a, b = int(tail.ks[t]), int(full.ks[src])
        assert torch.equal(masks_tail[t, a : a + v], masks_full[src, b : b + v])
        assert not masks_tail[t, :a].any() and not masks_tail[t, a + v :].any()


@skip_pre_blackwell
@pytest.mark.parametrize("tail", [False, True], ids=["full_rows", "decoder_tail"])
def test_candidate_masks_rescored_past_budget_are_exact(data, tail, monkeypatch):
    """With the masks over the budget no step keeps them; the re-index layer re-scores
    each tile's masks from the candidate source's query rows and selects the same top-k,
    in a full-row step and in the decoder pass, in 37-row tiles."""
    keys, q20, q24 = data
    core = _core()
    if tail:
        w = 128
        sel = torch.cat(
            [
                torch.arange(REQS[0][1] - w, REQS[0][1]),
                torch.arange(REQS[0][1] + REQS[1][1] - w, REQS[0][1] + REQS[1][1]),
            ]
        ).cuda()
        step = _Step([(p0 + n - w, w) for p0, n in REQS], [1, 0])
        q20, q24 = _rows(q20, sel), _rows(q24, sel)
    else:
        step = _Step(REQS, [0, 1])
    kvc = step.gather(keys)
    width = -(-(step.s_len + 256) // 256) * 256
    monkeypatch.setattr(modeling, "_INDEX_TILE_ELEMS", width * 37)

    def passes():
        cache = {"cand_q": q20} if tail else {}
        if not tail:
            _run(core, CAND_SRC, q20, step, kvc, cache, False)
        return _run(core, REINDEX, q24, step, kvc, cache, tail), cache

    kept, cache_kept = passes()
    assert len(cache_kept["cand_ctx"]) == -(-step.rows // 37)
    monkeypatch.setattr(modeling, "_CAND_MASK_BYTES", step.rows * step.s_len - 1)
    rescored, cache_rescored = passes()
    assert "cand_ctx" not in cache_rescored
    assert torch.equal(kept, rescored)


def _native_target(core):
    """A stand-in prefill target: the core it reads and the read-back entries it calls."""
    return SimpleNamespace(
        core=core,
        _swa_read=modeling.Mxfp8Dequantize(),
        _swa_read8=modeling.Mxfp8ToE4m3(),
        _comp_read=modeling.Nvfp4KvToBf16(),
        _comp_read8=modeling.Nvfp4KvToE4m3(),
    )


def _native_core(max_t):
    return SimpleNamespace(
        w={"final_norm": torch.empty(1, device="cuda")},
        head_dim=512,
        ratios={CAND_SRC: 1},
        kv_sources=[CAND_SRC],
        kv_src={CAND_SRC: CAND_SRC},
        local=frozenset(range(40)),  # one pipeline stage: every layer is local
        rank=0,
        fp8_attn=False,
        _const={"one_f32": torch.ones(1, dtype=torch.float32, device="cuda")},
    ), SimpleNamespace(swa_local_indices_cuda=torch.empty(max_t, 128, device="cuda"))


@skip_pre_blackwell
def test_both_read_buffer_forms_are_exact():
    """Native compressed KV: in one read buffer, attention through the per-request form and
    through the per-query form (the selected rows only, renumbered) is bitwise the same on the
    same step, including a request with one compressed entry and top-k rows padded with -1
    whose rows the per-query form leaves stale."""
    torch.manual_seed(0)
    H, D, bk, pages, cols = 16, 512, 128, 200, 40
    lens = [3000, 5100, 4097, 1]
    seqs = len(lens)
    # The manager's 288-byte rows: 256 bytes of packed E2M1, 32 E4M3 scales (finite).
    pool = torch.randint(0, 256, (pages, bk, 288), dtype=torch.uint8, device="cuda")
    pool[..., 256:] = torch.randint(0, 0x70, (pages, bk, 32), dtype=torch.uint8, device="cuda")
    codes, scales = pool[..., :256], pool[..., 256:]
    comp_q4 = {CAND_SRC: (codes.reshape(pages, bk, 1, 256), scales.reshape(pages, bk, 1, 32))}
    comp_q1 = {
        CAND_SRC: (codes.reshape(pages * bk, 1, 1, 256), scales.reshape(pages * bk, 1, 1, 32))
    }
    assert comp_q1[CAND_SRC][0].data_ptr() == codes.data_ptr()
    tbl = (
        torch.randperm(pages, device="cuda")[: seqs * cols]
        .view(seqs, cols)
        .to(torch.int32)
        .contiguous()
    )
    klens = torch.tensor(lens, dtype=torch.int32, device="cuda")

    topk = torch.full((seqs, modeling._TOPK), -1, dtype=torch.int32, device="cuda")
    for b, n in enumerate(lens):
        pick = torch.randperm(n, device="cuda")[: min(n, modeling._TOPK)].sort().values
        topk[b, : pick.numel()] = pick.to(torch.int32)
    count = (topk >= 0).sum(1).to(torch.int32)
    valid = topk >= 0
    e = topk.clamp(min=0).long()
    b_idx = torch.arange(seqs, device="cuda").view(-1, 1)
    per_request = torch.where(valid, b_idx * cols * bk + e, -1).to(torch.int32)
    slots = torch.where(valid, tbl.long()[b_idx, e // bk] * bk + e % bk, -1).to(torch.int32)

    swa = torch.randn(4, 1, 128, D, device="cuda").to(torch.bfloat16)
    win = torch.randint(0, 4 * 128, (seqs, 128), dtype=torch.int32, device="cuda")
    q = torch.randn(seqs, 1, H, D, device="cuda").to(torch.bfloat16)
    sinks = torch.randn(H, device="cuda")
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    topk_lens = (128 + count).contiguous()

    def attend(gidx, comp_pool, n_lens=topk_lens):
        return modeling.TrtllmSparseMlaDsv4()(
            q, swa, comp_pool, ws, gidx, n_lens, klens.clamp(min=128), sinks, D**-0.5, 1.0
        )

    # One buffer that holds either form: 4 request slots of 40 x 128 rows, 40 query rows.
    core, md = _native_core(max_t=64)
    md.compress_block_tables = {1: tbl}
    md.compressed_kv_lens_cuda = {1: klens}
    pools = Core._native_read_buffers(core, md)
    assert (pools["comp_req_fit"], pools["comp_q_fit"]) == (
        seqs,
        seqs * cols * bk // modeling._TOPK,
    )
    pools.update(comp_q4=comp_q4, comp_q1=comp_q1)

    # Per request: every entry of every request, dequantized once per KV source.
    target = _native_target(core)
    Prefill._native_compressed(target, CAND_SRC, {"md": md, "B": seqs}, pools)
    out_request = attend(torch.cat([win, per_request], 1), pools["comp_c4_req"][1])

    # Per query, in the same buffer, over stale finite rows (never attended).
    pools["topk_out"].fill_(3.0)
    cache = {}
    gidx = Prefill._native_topk(
        target, CAND_SRC, torch.cat([win, slots], 1), topk, seqs, pools, cache
    )
    assert torch.equal(gidx[:, 128:], pools["topk_ren"][:seqs])
    out_query = attend(gidx, pools["comp_c4_q"][1])
    assert torch.equal(out_request, out_query)
    # The layers that share the top-k set and KV source reuse the rows.
    assert cache["comp_rows"][0] is topk
    pools["topk_out"].fill_(0.0)
    Prefill._native_topk(target, CAND_SRC, torch.cat([win, slots], 1), topk, seqs, pools, cache)
    assert not pools["topk_out"].any()

    # The compressed rows matter: without them the output differs.
    assert not torch.equal(
        out_request,
        attend(torch.cat([win, slots], 1), pools["comp_c4_q"][1], torch.full_like(topk_lens, 128)),
    )


def _form_core():
    return SimpleNamespace(
        ratios={2: 2, 8: 2, 14: 2, 20: 1},
        kv_sources=[2, 8, 14, 20],
        index_sources=[2, 8, 14, 20, 24, 28, 32, 36],
        local=frozenset(range(40)),
        late=21,
    )


def _form(caps, B, T, nc, kv, max_seq_len, max_t=8192, tail=None):
    pools = {"comp_req_fit": caps[0], "comp_q_fit": caps[1]}
    md = SimpleNamespace(
        cached_token_lens_cpu=torch.tensor([k - 1 for k in kv], dtype=torch.int32),
        seq_lens_kv=torch.ones(len(kv), dtype=torch.int32),
        max_seq_len=max_seq_len,
        swa_local_indices_cuda=torch.empty(max_t, 0),
    )
    st = {"B": B, "T": T, "nc": nc}
    return Core._comp_form(_form_core(), pools, md, st, tail)


def test_read_buffer_form_per_step():
    """The step takes a form it fits, the one that dequantizes fewer rows: prefill of a few
    requests the per-request form (batch 256 / 16K / 8192 included), long-context decode the
    per-query form, short-context decode the per-request form."""
    # Batch 256 / 16K / 8192: 248 request slots of 132 x 128 rows, 8192 query rows.
    caps = (248, 8192)
    assert _form(caps, 4, 8192, 4, [4096, 2048, 8192, 16000], 16384) == "request"
    assert _form(caps, 64, 8192, 64, [16000] * 64, 16384) == "request"
    assert _form(caps, 256, 8192, 8, [16000] * 256, 16384) == "topk"  # 256 > 248 slots
    assert _form(caps, 32, 32 * 6, 0, [0] * 32, 16384) == "request"  # decode, 16K
    # 1M / batch 32 / 8192: 3 request slots of 8196 x 128 rows, 8192 query rows.
    caps = (3, 8192)
    assert _form(caps, 1, 8192, 1, [1 << 20], 1 << 20) == "request"  # 2.6M rows vs 33.5M
    assert _form(caps, 32, 32 * 6, 0, [0] * 32, 1 << 20) == "topk"
    assert _form(caps, 2, 2 * 6, 0, [0] * 2, 1 << 20) == "topk"  # 2.6M rows vs 49K
    # Bounded step: the decoder rows count for the per-query form.
    tail = SimpleNamespace(bound=True, rows=128)
    assert _form((3, 8192), 1, 8192, 1, [1 << 20], 1 << 20, tail=tail) == "request"
    # A form the step does not fit is never taken.
    assert _form((0, 8192), 1, 100, 1, [100], 1 << 20) == "topk"
    assert _form((4, 64), 4, 100, 4, [1 << 20] * 4, 1 << 20) == "request"


def _plan_md(rid, cached, length, last):
    return SimpleNamespace(
        context_last_chunk=[last],
        context_return_logits=[False],
        request_ids=[rid],
        seq_lens=torch.tensor([length], dtype=torch.int32),
        cached_token_lens_cpu=torch.tensor([cached], dtype=torch.int32),
        v41_num_replay_tokens_cpu=torch.tensor([0], dtype=torch.int32),
    )


@skip_pre_blackwell
def test_decoder_tail_plans_long_prompts():
    """A prompt past 2048 x 8 positions -- first seen whole, or recomputed after a pause
    with its generated tokens in its prompt, over several chunks -- plans a bounded tail
    of its last 128 positions."""
    core = SimpleNamespace(
        replay=True, _witness=False, replay_window=128, _tail_records={}, _ar32=None
    )
    core._ensure_ar = lambda n, dev: Core._ensure_ar(core, n, dev)
    dev = torch.device("cuda")

    def plan(rid, cached, length, last):
        st = {"T": length, "tc": length, "nc": 1, "gen": 0}
        return modeling._DecoderTail(core, _plan_md(rid, cached, length, last), st, dev)

    tail = plan(7, 0, CANDIDATE_POSITIONS + 616, True)
    r = tail.reqs[0]
    assert tail.bound and r.t0 == r.s == CANDIDATE_POSITIONS + 616 - 128

    # A recomputed request: chunks of 8192 up to prompt + generated = 17000.
    n = CANDIDATE_POSITIONS + 616
    first = plan(9, 0, 8192, False)
    assert first.bound and first.reqs[0].keep == 8192 - 128
    first.counted = True
    first.finish()
    assert core._tail_records[9].end == 8192
    second = plan(9, 8192, 8192, False)
    second.counted = True
    second.finish()
    last = plan(9, 2 * 8192, n - 2 * 8192, True)
    r = last.reqs[0]
    assert last.bound and r.t0 == n - 128 and r.rec is core._tail_records[9]


@skip_pre_blackwell
@pytest.mark.parametrize("chunk", [1000, 4096])
def test_index_k_gather_in_chunks_is_exact(chunk, monkeypatch):
    """The index-K gather in chunks (at most _GATHER_MAX_ROWS rows a call) returns the rows of
    one call, codes and scale words, for any chunk size; skipped (negative) slots too."""
    bk, kb, pages, rows = 128, 68, 64, 6000
    g = torch.Generator(device="cuda").manual_seed(11)
    pool = torch.randint(0, 256, (pages, bk, 1, kb), dtype=torch.uint8, device="cuda", generator=g)
    slot = torch.randperm(pages * bk, device="cuda", generator=g)[:rows]
    page, s = slot // bk, slot % bk
    base = page * (bk * kb)
    data = (base + s * 64).to(torch.int64).contiguous()
    scale = (base + bk * 64 + s * 4).to(torch.int64).contiguous()
    data[17] = -1  # a skipped row
    scale[17] = -1
    target = SimpleNamespace(_idx_gather=modeling.IndexerKCacheGather())
    target._idx_gather.bind_const(head_dim=modeling._IDX_DIM // 2)
    one = Prefill._gather_index_k(target, pool, data, scale, rows, 1)
    monkeypatch.setattr(modeling, "_GATHER_MAX_ROWS", chunk)
    chunked = Prefill._gather_index_k(target, pool, data, scale, rows, 1)
    assert chunked[0].shape == (rows, 64) and chunked[1].shape == (rows,)
    keep = torch.ones(rows, dtype=torch.bool, device="cuda")
    keep[17] = False
    assert torch.equal(one[0][keep], chunked[0][keep]) and torch.equal(
        one[1][keep], chunked[1][keep]
    )
    flat = pool.view(-1)
    ref = flat[data[keep, None] + torch.arange(64, device="cuda")].view(torch.int8)
    assert torch.equal(chunked[0][keep], ref)
    # The DeepGEMM logits op needs the scale words' base 16-byte aligned.
    assert chunked[1].data_ptr() % 16 == 0


def _tail_core(hidden=8):
    w = 128
    a = torch.arange(w, device="cuda")
    core = SimpleNamespace(
        replay=True,
        _witness=False,
        replay_window=w,
        _tail_records={},
        _ar32=None,
        ds=None,
        hidden=hidden,
        hc_k=4 * hidden,
        late=21,
        num_layers=40,
        _const={
            "tri": (a.view(1, -1) <= a.view(-1, 1)).contiguous(),
            "col0": (a == 0).view(1, -1).contiguous(),
            "zero_bf16": torch.zeros((1, 1), dtype=torch.bfloat16, device="cuda"),
        },
    )
    core._ensure_ar = lambda n, dev: Core._ensure_ar(core, n, dev)
    return core


def _overflow_step(core, max_rows):
    """The step that outgrows itself: request 0 finishes a 2049-token prompt on a 1-token chunk
    (its first 127 tail rows are stashed from the chunk before), 16 fresh 127-token prompts
    finish too. T = 2033 engine rows; the tail holds 128 + 16 x 127 = 2160 rows."""
    dev = torch.device("cuda")
    w, hid = core.replay_window, core.hidden
    reqs = [(0, 2048, 1)] + [(k + 1, 0, 127) for k in range(16)]  # (rid, cached, chunk)
    T = sum(n for _, _, n in reqs)
    nc = len(reqs)
    pos = torch.cat([torch.arange(c, c + n) for _, c, n in reqs]).to(torch.int32).cuda()
    rid_row = torch.cat([torch.full((n,), rid) for rid, _, n in reqs]).cuda()

    def val(rid, p):  # an exact value per (request, position)
        return (rid * 100000 + p).to(torch.float32)

    rows = {
        "h": val(rid_row, pos).view(T, 1, 1).expand(T, 4, hid).to(torch.float32).contiguous(),
        "yF": val(rid_row, pos).view(T, 1).expand(T, 24).contiguous(),
        "rF": val(rid_row, pos).contiguous(),
        "img": torch.zeros(T, dtype=torch.bool, device="cuda"),
    }
    cache = {
        "topk": {1: val(rid_row, pos).to(torch.int32).view(T, 1).expand(T, 512).contiguous()},
        "cand_q": (
            torch.zeros(T, 32, 64, dtype=torch.int8, device="cuda"),
            val(rid_row, pos).to(torch.int32).view(T, 1).expand(T, 32).contiguous(),
            val(rid_row, pos).view(T, 1).expand(T, 32).contiguous(),
        ),
        "ctx_k": {},
        "sched": {},
    }
    md = SimpleNamespace(
        context_last_chunk=[True] * nc,
        context_return_logits=[False] * nc,
        request_ids=[rid for rid, _, _ in reqs],
        seq_lens=torch.tensor([n for _, _, n in reqs], dtype=torch.int32),
        cached_token_lens_cpu=torch.tensor([c for _, c, _ in reqs], dtype=torch.int32),
        v41_num_replay_tokens_cpu=torch.zeros(nc, dtype=torch.int32),
        sparse_mla_topk_lens={1: (128 + torch.clamp(pos + 1, max=512)).to(torch.int32)},
        compressed_visible_lens_cuda={1: (pos + 1).to(torch.int32)},
        v41_ctx_index_ks={1: torch.zeros(T, dtype=torch.int32, device="cuda")},
        kv_lens_cuda_runtime=torch.tensor(
            [c + n for _, c, n in reqs], dtype=torch.int32, device="cuda"
        ),
        indexer_compress_block_tables={1: torch.zeros(nc, 4, dtype=torch.int32, device="cuda")},
        req_idx_per_token=torch.cat([torch.full((n,), b) for b, (_, _, n) in enumerate(reqs)])
        .to(torch.int32)
        .cuda(),
        swa_local_indices_cuda=torch.zeros(T, w, dtype=torch.int32, device="cuda"),
    )
    # Request 0's record: its previous chunk ended at 2048 and stashed positions 1921..2047.
    rec = modeling._TailRecord(0)
    rec.end, rec.start = 2048, 2048 - 127
    sp = torch.arange(rec.start, 2048, device="cuda")
    rec.stash = {
        "h": val(torch.zeros_like(sp), sp).view(-1, 1, 1).expand(127, 4, hid).contiguous(),
        "yF": val(torch.zeros_like(sp), sp).view(-1, 1).expand(127, 24).contiguous(),
        "rF": val(torch.zeros_like(sp), sp).contiguous(),
        "img": torch.zeros(127, dtype=torch.bool, device="cuda"),
        "topk": val(torch.zeros_like(sp), sp)
        .to(torch.int32)
        .view(-1, 1)
        .expand(127, 512)
        .contiguous(),
        "lens": torch.full((127,), 640, dtype=torch.int32, device="cuda"),
        "vis": (sp + 1).to(torch.int32),
        "cq": torch.zeros(127, 32, 64, dtype=torch.int8, device="cuda"),
        "cqs": val(torch.zeros_like(sp), sp)
        .to(torch.int32)
        .view(-1, 1)
        .expand(127, 32)
        .contiguous(),
        "cw": val(torch.zeros_like(sp), sp).view(-1, 1).expand(127, 32).contiguous(),
    }
    core._tail_records = {0: rec}
    st = {"T": T, "tc": T, "nc": nc, "G": 0, "gen": 0, "B": nc, "next_n": 1, "pos": pos}
    tail = modeling._DecoderTail(core, md, st, dev)
    return tail, tail.enter(rows, st, cache, max_rows), T


@skip_pre_blackwell
def test_decoder_pass_slices_at_step_capacity():
    """A tail larger than the step capacity runs in slices of at most max_num_tokens rows, cut
    at request boundaries; the slices hold exactly the rows, per-request fields and output rows
    one pass would."""
    one_tail, one, T = _overflow_step(_tail_core(), 1 << 20)
    tail, passes, _ = _overflow_step(_tail_core(), 2048)
    assert [p[1]["T"] for p in one] == [2160]
    assert [p[1]["T"] for p in passes] == [128 + 15 * 127, 127]
    assert all(p[1]["T"] <= 2048 for p in passes)
    for name in ("h", "yF", "rF"):
        assert torch.equal(torch.cat([p[0][name] for p in passes]), one[0][0][name])
    for key in ("topk",):
        assert torch.equal(torch.cat([p[2][key][1] for p in passes]), one[0][2][key][1])
    assert torch.equal(torch.cat([p[2]["cand_q"][1] for p in passes]), one[0][2]["cand_q"][1])
    for field in (
        "req_idx_per_token",
        "swa_local_indices_cuda",
        "kv_lens_cuda_runtime",
        "v41_swa_floor_cuda",
        "v41_swa_seq_lens_cuda",
    ):
        assert torch.equal(
            torch.cat([getattr(p[1]["md"], field) for p in passes]), getattr(one[0][1]["md"], field)
        )
    assert [p[1]["md"].num_contexts for p in passes] == [16, 1]
    assert [p[1]["nc"] for p in passes] == [16, 1] and all(p[1]["B"] == p[1]["nc"] for p in passes)
    cu = [p[1]["md"].cu_seq_lens_cuda.tolist() for p in passes]
    assert cu[0][-1] == 2033 and cu[1] == [0, 127]
    # Request 0's tail: its 127 stashed rows, then its 1-token chunk.
    assert passes[0][0]["rF"][:128].tolist() == list(range(1921, 2049))
    # Each slice's output rows land where one pass puts them.
    for t, ps in ((one_tail, one), (tail, passes)):
        for k, (dec, _, _) in enumerate(ps):
            t.activate(k)
            t.scatter(dec["rF"].view(-1, 1).expand(-1, t.core.hidden).to(torch.bfloat16))
    assert torch.equal(tail.out, one_tail.out)
    assert tail.out[0, 0].item() == 2048  # request 0's only row of this step: position 2048


class _Capture:
    """DSpark's capture: each call overwrites the layer's rows [0, n)."""

    def __init__(self):
        self.rows = {}

    def maybe_capture_hidden_states(self, layer_id, hidden_states):
        self.rows[layer_id] = hidden_states.clone()


@skip_pre_blackwell
def test_decoder_pass_slices_capture_every_layer():
    """DSpark captures the input of layers 37-39 over the step's rows. Run in slices, every
    decoder row still holds its own input of each captured layer once the last slice has
    captured: an earlier slice's rows do not carry a later layer's input into this one."""
    tail, passes, T = _overflow_step(_tail_core(), 2048)
    hid = tail.core.hidden
    spec = _Capture()
    capture = modeling._TailCapture(spec, tail)
    layers = (37, 38, 39)
    for k, (dec, _, _) in enumerate(passes):
        tail.activate(k)
        for layer in layers:
            capture.maybe_capture_hidden_states(
                layer, (dec["h"] + 1000.0 * layer).reshape(-1, 4 * hid)
            )
    for layer in layers:
        got = spec.rows[layer].view(T, 4, hid)
        for k, (dec, _, _) in enumerate(passes):
            for r in tail.slices[k].fin:
                want = dec["h"][r.q + r.n_old : r.q + r.n] + 1000.0 * layer
                assert torch.equal(got[r.off + r.a : r.off + r.L], want), (layer, k, r.rid)
