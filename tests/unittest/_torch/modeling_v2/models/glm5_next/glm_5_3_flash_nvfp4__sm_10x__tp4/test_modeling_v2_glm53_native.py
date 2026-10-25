# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's torch bindings: the step program's structs as torch classes, and the
launchers the target calls itself against PyTorch references.

Importing the target package loads ``libth_modeling_v2_glm5_next.so`` through
``_native.load_target_ops``, as routing does: a pass here means the family library was
built for this GPU, packaged, opened, carries the target, and registered the fakes. The
launchers take device addresses as integers and run on the current stream.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_10x__tp4 import (  # noqa: E501
    layout,
    native_classes,
    native_ops,
)

assert torch.cuda.is_available(), "the glm5_next family library needs a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="built for sm_100a and sm_103a only",
)


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def test_layout():
    assert (layout.mla_block, layout.kpool, layout.sel_pools) == (128, 4, 512)
    assert layout.idx_kg_row0 == layout.mla_block // layout.kpool
    assert layout.sample_params_bytes == 40
    assert layout.wfmt_nvfp4 != layout.wfmt_bf16
    tp, max_elems = 4, 4096
    assert layout.allreduce_buf_elems(tp, max_elems) == 5 * tp * max_elems + 6 * max_elems
    assert 1 <= layout.attn_tile_tokens(16) <= 64


def test_structs_read_and_write_their_fields():
    w = native_classes.Bf16W()
    w.w, w.n, w.k = 0x7F12_3456_7890, 3, 5
    assert (w.w, w.n, w.k) == (0x7F12_3456_7890, 3, 5)
    with pytest.raises(RuntimeError, match="out of the field's range"):
        w.n = 1 << 40
    f = native_classes.Fp4W()
    f.alpha = 0.25
    assert f.alpha == 0.25
    layer = native_classes.LayerW()
    layer.kda_layer = False
    assert not layer.kda_layer
    # a struct member reads as a reference into its owner: writing through it sets the owner's
    layer.mla.q_b16.w = 77
    assert layer.mla.q_b16.w == 77
    # assigning a struct copies it in
    layer.mla.q_b16 = w
    w.n = 9
    assert (layer.mla.q_b16.w, layer.mla.q_b16.n) == (0x7F12_3456_7890, 3)
    s = native_classes.StepIn()
    s.force_accept, s.graph = 2.5, True
    s.set_h_idx_tiles([(0, 4), (4, 4)])
    assert (s.force_accept, s.graph) == (2.5, True)


def test_step_program_weights_are_its_own():
    mc = native_classes.ModelConfig()
    mc.set_layers([True, False], [False, True])
    step = native_classes.GlmStep(
        mc,
        native_classes.ParallelConfig(),
        native_classes.RunnerConfig(),
        torch.cuda.current_device(),
    )
    step.weights().layer(1).mla.q_b16.w = 99
    assert step.weights().layer(1).mla.q_b16.w == 99
    with pytest.raises(RuntimeError, match="no layer 2"):
        step.weights().layer(2)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    assert step.sm_count() == sms


@pytest.mark.parametrize("optional", [True, False])
def test_step_meta(optional):
    g = torch.Generator().manual_seed(0)
    # 3 contexts, then 4 sequences decoding 4 tokens each
    seq_lens = torch.tensor([5, 1, 9, 4, 4, 4, 4], dtype=torch.int32)
    B, nctx, T = len(seq_lens), 3, int(seq_lens.sum())
    has_initial = torch.tensor([True, False, True])
    slots = torch.randperm(16, generator=g)[:B].to(torch.int32)
    stride = 3
    prev_accepted = torch.randint(0, 4, (16 * stride,), generator=g, dtype=torch.int32)
    i32 = dict(dtype=torch.int32, device="cuda")
    cu, token_seq = torch.full((B + 1,), -1, **i32), torch.full((T,), -1, **i32)
    has_state, replay, tiles = (
        torch.full((B,), -1, **i32),
        torch.full((B,), -1, **i32),
        torch.full((B, 2), -1, **i32),
    )
    dev = {name: t.cuda() for name, t in dict(lens=seq_lens, slots=slots).items()}
    hi = has_initial.cuda() if optional else None
    pa = prev_accepted.cuda() if optional else None
    outs = [
        _ptr(cu),
        _ptr(token_seq),
        _ptr(has_state),
        _ptr(replay),
        _ptr(tiles if optional else None),
    ]
    ins = [_ptr(dev["lens"]), _ptr(hi), _ptr(dev["slots"]), _ptr(pa)]
    native_ops.step_meta(*outs, *ins, stride, B, nctx, T)
    torch.cuda.synchronize()
    ref_cu = torch.cat([torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0, dtype=torch.int32)])
    assert torch.equal(cu.cpu(), ref_cu)
    assert torch.equal(
        token_seq.cpu(), torch.arange(B, dtype=torch.int32).repeat_interleave(seq_lens.long())
    )
    ctx_state = has_initial.to(torch.int32) if optional else torch.zeros(nctx, dtype=torch.int32)
    assert torch.equal(
        has_state.cpu(), torch.cat([ctx_state, torch.ones(B - nctx, dtype=torch.int32)])
    )
    gen_replay = (
        prev_accepted[slots[nctx:].long() * stride]
        if optional
        else torch.zeros(B - nctx, dtype=torch.int32)
    )
    assert torch.equal(replay.cpu(), torch.cat([torch.zeros(nctx, dtype=torch.int32), gen_replay]))
    ref_tiles = (
        torch.stack([ref_cu[:-1], seq_lens], 1)
        if optional
        else torch.full((B, 2), -1, dtype=torch.int32)
    )
    assert torch.equal(tiles.cpu(), ref_tiles)


def test_spec_sample_params():
    g = torch.Generator().manual_seed(1)
    S, greedy_below = 37, 1e-3
    out = torch.randint(0, 256, (S + 1, layout.sample_params_bytes), generator=g, dtype=torch.uint8)
    temps = torch.rand(S, generator=g)
    temps[::4] = 0.0
    temps[1::7] = greedy_below / 2
    seeds = torch.randint(0, 1 << 62, (S,), generator=g, dtype=torch.int64)
    offsets = torch.randint(0, 1 << 40, (S,), generator=g, dtype=torch.int64)
    d_out, d_temps, d_seeds, d_offsets = out.cuda(), temps.cuda(), seeds.cuda(), offsets.cuda()
    native_ops.spec_sample_params(
        _ptr(d_out), _ptr(d_temps), _ptr(d_seeds), _ptr(d_offsets), S, greedy_below
    )
    torch.cuda.synchronize()
    # k::SampleParams: temperature, top_p, top_k, min_p (4 bytes each), seed, step (8 bytes each),
    # num_logprobs; the launcher writes the temperature, the seed and the step of rows [0, S)
    ref = out.clone()
    ref[:S, 0:4] = torch.where(temps < greedy_below, 0.0, temps).view(torch.uint8).view(S, 4)
    ref[:S, 16:24] = seeds.view(torch.uint8).view(S, 8)
    ref[:S, 24:32] = offsets.view(torch.uint8).view(S, 8)
    assert torch.equal(d_out.cpu(), ref)


@pytest.mark.parametrize("rows", ["row0", "packed"])
def test_spec_accept(rows):
    g = torch.Generator().manual_seed(2)
    B, nctx, D = 6, 2, 3
    n = D + 1
    S = B * n
    tokens = torch.randint(0, 1000, (S,), generator=g, dtype=torch.int32)
    drafts = torch.randint(0, 1000, (B, D), generator=g, dtype=torch.int32)
    accepted = torch.randint(0, D + 1, (B,), generator=g, dtype=torch.int32)
    # the rows of a sequence: given (some sequences sample fewer rows), or n per sequence
    row0 = torch.tensor([0, 1, 5, 9, 13, 17], dtype=torch.int32) if rows == "row0" else None
    batch_idx = torch.randperm(B, generator=g).to(torch.int32) if rows == "row0" else None
    slots = torch.randperm(10, generator=g)[:B].to(torch.int32)
    dummy = torch.tensor([False, False, False, True, False, False])
    stride = 2
    prev_accepted = torch.full((10 * stride,), -7, dtype=torch.int32)
    i32 = dict(dtype=torch.int32, device="cuda")
    new_tokens, lens, nxt = (
        torch.full((B, n), -1, **i32),
        torch.full((B,), -1, **i32),
        torch.full((B, n), -1, **i32),
    )
    host = dict(tokens=tokens, drafts=drafts, accepted=accepted, slots=slots, dummy=dummy)
    d = {k: v.cuda() for k, v in dict(host, prev=prev_accepted).items()}
    d_row0 = None if row0 is None else row0.cuda()
    d_bi = None if batch_idx is None else batch_idx.cuda()
    outs = [_ptr(new_tokens), _ptr(lens), _ptr(nxt)]
    ins = [_ptr(d["tokens"]), _ptr(d["drafts"]), _ptr(d["accepted"]), _ptr(d_row0), _ptr(d_bi)]
    state = [_ptr(d["slots"]), _ptr(d["dummy"]), _ptr(d["prev"]), stride]
    native_ops.spec_accept(*outs, *ins, S, *state, B, nctx, D)
    torch.cuda.synchronize()
    first = row0.long() if row0 is not None else torch.arange(B) * n
    ref_new = tokens[(first[:, None] + torch.arange(n)).clamp(max=S - 1)]
    assert torch.equal(new_tokens.cpu(), ref_new)
    assert torch.equal(lens.cpu(), accepted + 1)
    src = batch_idx.long() if batch_idx is not None else torch.arange(B)
    last = tokens[(first[src] + accepted.long()).clamp(max=S - 1)]
    assert torch.equal(nxt.cpu(), torch.cat([last[:, None], drafts], 1))
    ref_prev = prev_accepted.clone()
    for b in range(nctx, B):
        if not dummy[b]:
            ref_prev[int(slots[b]) * stride] = int(accepted[b])
    assert torch.equal(d["prev"].cpu(), ref_prev)


def test_allreduce_lamport_init():
    tp, max_elems = 4, 512
    elems = layout.allreduce_buf_elems(tp, max_elems)
    buf = torch.zeros(elems, dtype=torch.int16, device="cuda")
    native_ops.allreduce_lamport_init(_ptr(buf), tp, max_elems)
    torch.cuda.synchronize()
    # the one-shot and two-shot Lamport buffers hold bf16 -0.0 (0x8000), the flag-protocol
    # parities between them stay as they were
    sentinel = torch.iinfo(torch.int16).min
    one_shot, parities = 3 * tp * max_elems, 2 * tp * max_elems
    out = buf.cpu()
    assert (out[:one_shot] == sentinel).all()
    assert (out[one_shot : one_shot + parities] == 0).all()
    assert (out[one_shot + parities :] == sentinel).all()
