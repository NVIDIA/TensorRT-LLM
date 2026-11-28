# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder recovery (stock's CED replay) on the tep8 target, as the dep2 / dep targets and stock apply it.

With block reuse a context request whose Encoder pages were not kept gets an `EncoderReplay`: its chunk starts at the
replay start and its rows below the replay's Global end are historical. The core reads the plans of the engine's
`context_requests` before the forward and consumes them only after it succeeds (stock's
`DeepseekV41ForCausalLM.forward`); a context step's Encoder SWA reads start at the recovery floor (`_step_ctx`, per
request); the historical rows publish no GLOBAL / INDEX rows (stock compresses each owner from
`hist_end - hist_end % ratio` on, `CSA2TrtllmMetadata._prepare_owner_tables`); and `on_step` seeds a new request's
Engram look-back before the replay start. Before this the target planned only the Decoder pass: historical rows
rewrote the reused GLOBAL rows, the Encoder read below its floor, and the unconsumed plan failed the next chunk's
`prepare_context_replay` check."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_history_slots import (
    EngramHistorySlots,
)
from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling"
)
att = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.attention"
)
DEV = "cuda"


def _requests():
    replay = EncoderReplay(request_id=7, cache_identity=1, global_end=40, start=8)
    checkpoint = EncoderCheckpoint(request_id=8, cache_identity=2, global_end=512)
    return [
        NS(py_request_id=7, py_ced_replay=replay),
        NS(py_request_id=8, py_ced_replay=checkpoint),
    ]


def _core(fail=False):
    """The real core's `forward` (its recovery bookkeeping) around a stub body that records what it ran with."""
    core = object.__new__(mod.ModelingV2Core)
    nn.Module.__init__(core)
    core._recovery, core._recovery_logged, core.seen = None, False, []

    def body(attn_metadata, input_ids, position_ids, inputs_embeds, lora_params, **kwargs):
        core.seen.append((core._recovery, dict(kwargs)))
        if fail:
            raise RuntimeError("forward failed")
        return torch.zeros(1)

    core._forward = body
    return core


def test_the_replay_is_read_before_the_body_and_consumed_after_it():
    core, reqs = _core(), _requests()
    core(NS(seq_lens=[40, 208]), input_ids=torch.zeros(4, dtype=torch.int32), context_requests=reqs)
    ((recovery, kwargs),) = core.seen
    assert recovery == (
        [40, 0],
        [8, 0],
    )  # the replay's Global end and floor (its start), per request row
    assert (
        kwargs["context_requests"] is reqs
    )  # the replay plan of the Decoder pass still reads them
    assert core._recovery is None
    assert reqs[0].py_ced_replay.consumed and reqs[0].py_ced_replay.num_tokens == 0


def test_a_failed_forward_leaves_the_replay_pending():
    core, reqs = _core(fail=True), _requests()
    with pytest.raises(RuntimeError):
        core(
            NS(seq_lens=[40, 208]),
            input_ids=torch.zeros(4, dtype=torch.int32),
            context_requests=reqs,
        )
    assert core._recovery is None
    assert not reqs[0].py_ced_replay.consumed and reqs[0].py_ced_replay.num_tokens == 32


def test_a_step_without_context_requests_has_no_recovery():
    core = _core()
    core(NS(seq_lens=[1, 1]), input_ids=torch.zeros(2, dtype=torch.int32))
    assert core.seen == [(None, {})]


def test_the_context_step_reads_from_the_recovery_floor_and_knows_the_global_end():
    pos = torch.tensor([8, 9, 10, 100, 101], dtype=torch.int32, device=DEV)
    rid = torch.tensor([0, 0, 0, 1, 1], device=DEV)
    step = NS(
        positions=pos, tok_req=rid, cs=None, ce=None, nreq=2, uniform=False, bounds=[(0, 3), (3, 5)]
    )
    core = NS(_recovery=([10, 0], [8, 0]), rope=None, cfg=NS(num_hidden_layers=2))
    md = NS(
        kv_cache_manager=None, kv_cache_block_offsets=torch.zeros(2, 2, 2, 4, dtype=torch.int32)
    )
    ctx = mod.ModelingV2Core._step_ctx(core, step, md, segs=[], paged=False)
    assert ctx.floor.tolist() == [8, 0] and ctx.hist_end.tolist() == [
        10,
        0,
    ]  # per request, as the replay's floors
    assert ctx.floor.dtype == torch.int64
    paged = mod.ModelingV2Core._step_ctx(core, step, md, segs=[], paged=True)
    assert paged.floor is None and paged.hist_end is None  # generation steps carry no recovery
    core._recovery = None
    plain = mod.ModelingV2Core._step_ctx(core, step, md, segs=[], paged=False)
    assert plain.floor is None and plain.hist_end is None


def _engine_slots(starts, lengths, ratio, pages, page_size):
    """The engine's owner slots and compressed batch (`CSA2TrtllmMetadata._refresh_owner_slots`)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    n, nreq = int(lengths.sum()), lengths.numel()
    slots, first = (
        torch.empty(n, dtype=torch.int64, device=DEV),
        torch.empty(n, dtype=torch.int32, device=DEV),
    )
    batch = tuple(torch.empty(k, dtype=torch.int32, device=DEV) for k in (nreq, nreq, nreq + 1))
    CSA2TrtllmMetadata._refresh_owner_slots(
        starts, lengths, torch.zeros_like(starts), ratio, page_size, pages, slots, first, *batch
    )
    cu = batch[2] if ratio == 2 else torch.cat([lengths.new_zeros(1), lengths.cumsum(0).int()])
    return slots, first, cu


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("hist_end", [(0, 0, 0), (40, 0, 77), (41, 3, 64), (48, 9, 100)])
def test_historical_groups_publish_nothing_as_in_stock(ratio, hist_end):
    """The slots the step writes are stock's: an Encoder replay's owner compresses from `prefix - prefix % ratio` on,
    so the groups wholly before the Global end write no GLOBAL row; every other group keeps its slot."""
    page_size = 128 // ratio
    starts = torch.tensor(
        [8, 0, 60], dtype=torch.int32, device=DEV
    )  # each chunk starts at its replay start
    lengths = torch.tensor([45, 12, 41], dtype=torch.int32, device=DEV)
    pages = torch.arange(3 * 8, dtype=torch.int32, device=DEV).view(3, 8).flip(1).contiguous()
    slots, first, cu = _engine_slots(starts, lengths, ratio, pages, page_size)
    ends = (starts + lengths).tolist()
    h = torch.tensor(hist_end, dtype=torch.int32, device=DEV)
    got = att._unpublished_history(slots, first, cu, h, ratio)
    # stock's encoder-replay owner geometry: source starts at max(start, prefix - prefix % ratio)
    owner_starts = torch.tensor(
        [s if p == 0 else min(e, max(s, p - p % ratio)) for s, e, p in zip(starts.tolist(), ends, hist_end)],
        dtype=torch.int32, device=DEV)  # fmt: skip
    stock, _, _ = _engine_slots(
        owner_starts,
        torch.tensor(ends, dtype=torch.int32, device=DEV) - owner_starts,
        ratio,
        pages,
        page_size,
    )
    assert sorted(v for v in got.tolist() if v >= 0) == sorted(v for v in stock.tolist() if v >= 0)
    # nothing at or past a group of the historical range survives
    n_out = int(cu[-1])
    req = torch.searchsorted(cu[1:].long(), torch.arange(n_out, device=DEV), right=True)
    hist = first[:n_out].long() + ratio <= h.long()[req]
    assert (got[:n_out][hist] == -1).all() and torch.equal(got[:n_out][~hist], slots[:n_out][~hist])


@pytest.mark.parametrize("ratio", [1, 2])
def test_compress_masks_the_owner_slots_of_a_replay(monkeypatch, ratio):
    """`_compress` (the owner layers' and the bounded-replay boundary's publish) hands the stores the masked slots."""
    slots = torch.arange(6, dtype=torch.int64, device=DEV) + 100
    first = torch.tensor([0, 2, 4, 6, 0, 2], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, 4, 6], dtype=torch.int32, device=DEV)
    plan = dict(
        owners={3: (None, slots, first, None, None, None, None, cu)},
        cu_seq=cu,
        ends=None,
        max_outputs=3,
    )
    monkeypatch.setattr(att, "_owner_plan", lambda ctx, sources, ratios: plan)
    monkeypatch.setattr(att, "_pool", lambda ctx, layer, role: None)
    monkeypatch.setattr(att, "_table", lambda ctx, layer, role: None)
    monkeypatch.setattr(att, "csa2_bf16_lowm_proj", lambda x, w: x)
    monkeypatch.setattr(att, "csa2_compressor_project", lambda *a: ("lat", "k"))
    layer = NS(layer=3, sources=(3,), ratios={3: ratio}, ratio=ratio, c_wkv=None, c_wkvg=0, c_wkvg_bf16=0, c_norm=None,
               eps=None, i_wk=None)  # fmt: skip
    ctx = NS(hist_end=torch.tensor([5, 0], dtype=torch.int32, device=DEV))
    _, _, got, _ = att._Attention._compress(layer, ctx, torch.zeros(1), None)
    # request 0 (outputs 0-3) keeps the groups ending past 5; request 1 has no replay
    assert got.tolist() == (
        [-1, -1, 102, 103, 104, 105] if ratio == 2 else [-1, -1, -1, 103, 104, 105]
    )
    ctx.hist_end = None
    assert att._Attention._compress(layer, ctx, torch.zeros(1), None)[2] is slots


class _Core(NS):
    hist = property(lambda self: self.engram_slots.hist)
    req_slot = property(lambda self: self.engram_slots.req_slot)


def test_on_step_seeds_the_look_back_before_the_replay_start():
    vocab = 64
    core = _Core(
        engram_slots=EngramHistorySlots(mod._RING, -7, mod._MAX_REQS, DEV),
        cfg=NS(engram_max_ngram_size=3), _prev_reqs=0, _prev_ids=None, _staging={},
        engram_hash=NS(token_map=torch.arange(vocab, device=DEV, dtype=torch.int64) * 3),
    )  # fmt: skip
    tokens = [(7 * i + 1) % vocab for i in range(64)]
    ranges = []
    req = NS(py_request_id=11, is_context_init_state=True, context_current_position=40,
             py_ced_replay=EncoderReplay(request_id=11, cache_identity=1, global_end=40, start=24),
             get_tokens_range=lambda beam, a, b: ranges.append((a, b)) or tokens[a:b])  # fmt: skip
    mod.ModelingV2Core.on_step(core, NS(context_requests=[req], generation_requests=[]), None)
    torch.cuda.synchronize()
    assert ranges == [(22, 24)]  # the step's first position is the replay start, 24
    slot = core.engram_slots.row_of(11)
    assert [int(core.hist[slot, p % mod._RING]) for p in (22, 23)] == [
        tokens[22] * 3,
        tokens[23] * 3,
    ]
