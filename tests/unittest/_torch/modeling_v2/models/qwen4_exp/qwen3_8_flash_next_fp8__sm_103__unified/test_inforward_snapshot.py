# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-forward Mamba snapshots in the FP8 unified target's prefill: one step over [.., L) leaves
the state at P = L - tail in the live slot and stages the state at L, checked against the main
chunk [.., P) then the tail [P, L) alone in the next step, at a tep2 rank's GDN geometry."""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.embedding.ngram_hash import (  # noqa: E501
    CKPT_MULTIPLIERS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ngram_token_history import (  # noqa: E501
    ngram_token_history,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_gate_conv_inject_norm import (  # noqa: E501
    CKPT_CLAMP_MIN,
    CKPT_DILATION,
    CKPT_HC_COUNT,
    CKPT_HIDDEN,
    ple_gate_conv_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as _m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm import (  # noqa: E501
    flashinfer_chunk_gated_delta_rule as fi,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.gdn_inforward_tail import (  # noqa: E501
    gdn_inforward_tail_reference,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import InforwardSnapshotEntry

assert torch.cuda.is_available(), "the in-forward snapshot test requires a CUDA device"

DEV = "cuda"
H, HV, DK, DV = fi.RANK2_GEOMETRY
CONV_DIM, D_CONV, SLOTS, ROWS, EOS = 2 * H * DK + HV * DV, 4, 24, 8, 2  # ROWS: staging rows
NGRAM, PLE_STATE, STREAM = len(CKPT_MULTIPLIERS), 3 * CKPT_DILATION, CKPT_HC_COUNT * CKPT_HIDDEN

# (request id, live slot, chunk length, carries a state, tail or None). One request: CP pipeline.
BATCHES = {
    "1req": [(31, 4, 3000, True, 2)],
    "1fresh": [(32, 9, 16384, False, 2)],
    "3req": [(11, 5, 300, True, None), (12, 2, 1000, True, 2), (13, 7, 150, False, 2)],
    "8req": [(21, 3, 96, True, 2), (22, 9, 700, True, None), (23, 1, 67, True, 2),
             (24, 17, 2048, False, None), (25, 12, 130, True, 2), (26, 0, 515, True, 2),
             (27, 20, 64, True, None), (28, 6, 1500, False, 2)],
}  # fmt: skip


def _batch_split(batch):
    plan, row = {}, 0
    for rid, _, length, _, tail in batch:
        if tail is not None:
            plan[rid] = InforwardSnapshotEntry(row, length - tail, length - tail, tail)
            row += 1
    slots = {rid: slot for rid, slot, *_ in batch}
    manager = types.SimpleNamespace(
        conv_state_shape=[CONV_DIM, D_CONV - 1],
        inforward_snapshot_rows=ROWS,
        inforward_snapshot_max_tail=2,
        inforward_snapshot_plan=plan,
        state_index=lambda rid: slots[rid],
    )
    attn = types.SimpleNamespace(
        request_ids=[rid for rid, *_ in batch],
        seq_lens=torch.tensor([length for _, _, length, _, _ in batch], dtype=torch.int32),
    )
    metadata = Mamba2Metadata(max_batch_size=32, chunk_size=128)
    metadata._prepare_inforward_split(attn, manager, plan, len(batch), len(batch))
    return metadata


class _Rows:
    def __init__(self, batch):
        self.main, self.tail, self.main_cu, self.tail_cu = [], [], [0], [0]
        self.tail_slots, self.starts = [], []
        start = 0
        for _, slot, length, _, tail in batch:
            before = length - (tail or 0)
            self.starts.append(start)
            self.main += range(start, start + before)
            self.main_cu.append(self.main_cu[-1] + before)
            if tail:
                self.tail += range(start + before, start + length)
                self.tail_cu.append(self.tail_cu[-1] + tail)
                self.tail_slots.append(slot)
            start += length
        self.total = start
        self.slots = [slot for _, slot, *_ in batch]
        self.carry = [c for *_, c, _ in batch]
        self.split = [(slot, start, length - tail, start + length) for (_, slot, length, _, tail), start
                      in zip(batch, self.starts) if tail]  # fmt: skip

    def t(self, values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=DEV)


def _fake_target(metadata, batch, pools, stage, input_ids=None):
    core = types.SimpleNamespace(_bound=types.SimpleNamespace(pools=pools, stage=stage))
    target = types.SimpleNamespace(core=core, _step=None)
    hi = torch.tensor([c for *_, c, _ in batch], dtype=torch.bool, device=DEV)
    total = sum(length for _, _, length, _, _ in batch)
    split = _m.PrefillTarget._split_of(target, metadata, len(batch), 0, hi, total)
    target._step = types.SimpleNamespace(split=split, input_ids=input_ids)
    return target


def _strided_pool(gen, *shape, dtype=torch.bfloat16):
    """`[SLOTS, *shape]` whose slot stride is two states: the coalesced page's layout."""
    page = (torch.randn(SLOTS, 2, *shape, generator=gen, device=DEV) * 0.1).to(dtype)
    return page[:, 1]


def _stage(gdn):
    bf = torch.bfloat16
    ssm = {0: torch.full((ROWS, HV, DV, DK), 7.0, dtype=bf, device=DEV)} if gdn else {}
    conv = {0: torch.full((ROWS, CONV_DIM, D_CONV - 1), 7.0, dtype=bf, device=DEV)} if gdn else {}
    return _m._InforwardStage(
        ssm=ssm,
        conv=conv,
        ple_conv=torch.full((ROWS, STREAM, PLE_STATE), 7.0, dtype=bf, device=DEV),
        ple_ngram=torch.full((ROWS, NGRAM - 1), -1, dtype=torch.long, device=DEV),
    )


def _gdn_inputs(total, gen):
    q = torch.randn(1, total, H, DK, generator=gen, device=DEV)
    k = torch.randn(1, total, H, DK, generator=gen, device=DEV)
    q = (q / q.norm(dim=-1, keepdim=True)).bfloat16()
    k = (k / k.norm(dim=-1, keepdim=True)).bfloat16()
    v = torch.randn(1, total, HV, DV, generator=gen, device=DEV).bfloat16()
    g = torch.exp(-torch.rand(1, total, HV, generator=gen, device=DEV) * 0.2)
    beta = torch.rand(1, total, HV, generator=gen, device=DEV)
    return q, k, v, g, beta


def _fi_step(qkvgb, sel, pool, slots, cu, carry):
    out, _ = fi.flashinfer_chunk_gated_delta_rule(
        *(x.index_select(1, sel) for x in qkvgb), scale=None, initial_state=pool,
        initial_state_indices=slots, inplace_indexed_state_update=True, output_final_state=False,
        cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=False, output=None,
        state_workspace=None, has_initial_state=carry, linear_gate=True,
    )  # fmt: skip
    return out


def _ple_inputs(gen, n):
    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=DEV) * scale).bfloat16()

    hc, key, value = rand(n, STREAM, scale=2.0), rand(n, STREAM), rand(n, CKPT_HIDDEN, scale=0.5)
    norms = [rand(STREAM, scale=0.5) for _ in range(4)]
    return hc, key, value, norms, rand(STREAM, 1, 4, scale=0.3)


def _ple(inputs, sel, pool, slots, cu, carry, max_len, **kw):
    """`ple_gate_conv_inject_norm` over rows `sel` (None: every row) at the checkpoint's shapes."""
    hc, key, value, norms, conv_w = inputs
    if sel is not None:
        hc, key, value = (t.index_select(0, sel) for t in (hc, key, value))
    return ple_gate_conv_inject_norm(
        hc, key, value, norms[0], norms[1], norms[2], conv_w, pool, slots, cu, norms[3], 1e-6,
        CKPT_HC_COUNT, CKPT_CLAMP_MIN, CKPT_DILATION, -1, max_len, carry, **kw,
    )  # fmt: skip


def _gdn_split(batch, metadata, rows):
    """`_gdn_split`: the state at P, main rows and conv windows bitwise the main step's; the tails'
    outputs and staged states within rounding of the tail step and the float64 reference."""
    gen = torch.Generator(device=DEV).manual_seed(7 + len(batch))
    qkvgb = _gdn_inputs(rows.total, gen)
    ssm = _strided_pool(gen, HV, DV, DK)
    conv = _strided_pool(gen, CONV_DIM, D_CONV - 1)
    proj = torch.randn(rows.total, CONV_DIM + 96, generator=gen, device=DEV).bfloat16()
    mixed_qkv = proj[:, :CONV_DIM]
    main, tail = rows.t(rows.main), rows.t(rows.tail)
    tail_slots = rows.t(rows.tail_slots)
    tail_carry = torch.ones(len(rows.tail_slots), dtype=torch.bool, device=DEV)

    ref_ssm = ssm.clone()
    slots32, carry = rows.t(rows.slots, torch.int32), rows.t(rows.carry, torch.bool)
    out_main = _fi_step(qkvgb, main, ref_ssm, slots32, rows.t(rows.main_cu), carry)
    state_p = ref_ssm.clone()
    tail32 = rows.t(rows.tail_slots, torch.int32)
    out_tail = _fi_step(qkvgb, tail, ref_ssm, tail32, rows.t(rows.tail_cu), tail_carry)
    state_l = ref_ssm.index_select(0, tail_slots)

    for slot, start, _, end in rows.split:
        conv[slot] = mixed_qkv[end - (D_CONV - 1) : end].t()
    conv_before = conv.clone()
    stage = _stage(gdn=True)
    pools = types.SimpleNamespace(ssm={0: ssm}, conv={0: conv})
    target = _fake_target(metadata, batch, pools, stage)
    out = torch.full((1, rows.total, HV, DV), -7.0, dtype=torch.bfloat16, device=DEV)
    _m.PrefillTarget._gdn_split(target, *qkvgb, mixed_qkv, 0, out)
    torch.cuda.synchronize()

    assert torch.equal(ssm, state_p), "live slots differ from the main step's states"
    assert torch.equal(out[0].index_select(0, main), out_main[0]), "main rows' outputs differ"
    nsplit = len(rows.split)
    staged = stage.ssm[0][:nsplit]
    got_tail = out[0].index_select(0, tail)
    torch.testing.assert_close(staged.float(), state_l.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_tail.float(), out_tail[0].float(), rtol=2e-2, atol=2e-2)
    ref_stage, ref_out = torch.zeros_like(stage.ssm[0]), torch.zeros_like(out)
    sp = target._step.split
    gdn_inforward_tail_reference(
        *qkvgb, sp.tail_tok, sp.cu_tail, state_p, sp.tail_slots, ref_stage, sp.stage_rows, ref_out
    )
    torch.testing.assert_close(staged.float(), ref_stage[:nsplit].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        got_tail.float(), ref_out[0].index_select(0, tail).float(), rtol=1e-2, atol=1e-2
    )
    for row, (slot, start, before, end) in enumerate(rows.split):
        p = start + before
        assert torch.equal(conv[slot], mixed_qkv[p - (D_CONV - 1) : p].t())
        assert torch.equal(stage.conv[0][row], conv_before[slot])
    others = [s for s in range(SLOTS) if s not in rows.tail_slots]
    assert torch.equal(conv[others], conv_before[others])


def _ple_split(batch, metadata, rows):
    """`_split_ple_states`: PLE conv slots and n-gram histories bitwise the two-step schedule's."""
    gen = torch.Generator(device=DEV).manual_seed(100 + len(batch))
    n = rows.total
    inputs = _ple_inputs(gen, n)
    conv_pool = torch.randn(SLOTS, STREAM, PLE_STATE, generator=gen, device=DEV).bfloat16()
    ids = torch.randint(3, 50000, (n,), generator=gen, device=DEV, dtype=torch.int32)
    ngram_pool = torch.randint(3, 50000, (SLOTS, NGRAM - 1), generator=gen, device=DEV)
    carry = rows.t(rows.carry, torch.bool)
    slots32 = rows.t(rows.slots, torch.int32)

    def conv_step(pool, sel, cu, slots, carries, conv_input=None):
        max_len = int((cu[1:] - cu[:-1]).max())
        _ple(inputs, sel, pool, slots, cu, carries, max_len, conv_input=conv_input)

    main, tail = rows.t(rows.main), rows.t(rows.tail)
    tail_slots32 = rows.t(rows.tail_slots, torch.int32)
    tail_carry = torch.ones(len(rows.tail_slots), dtype=torch.bool, device=DEV)
    ref_conv = conv_pool.clone()
    conv_step(ref_conv, main, rows.t(rows.main_cu), slots32, carry)
    conv_p = ref_conv.clone()
    conv_step(ref_conv, tail, rows.t(rows.tail_cu), tail_slots32, tail_carry)
    _, ngram_p = ngram_token_history.reference(
        ids.index_select(0, main), ngram_pool, slots32, carry, rows.t(rows.main_cu), EOS, NGRAM
    )
    _, ngram_l = ngram_token_history.reference(
        ids.index_select(0, tail), ngram_p, tail_slots32, tail_carry, rows.t(rows.tail_cu), EOS,
        NGRAM,
    )  # fmt: skip

    whole_cu = rows.t([0] + [s + length for s, (_, _, length, _, _) in zip(rows.starts, batch)])
    conv_input = torch.empty(n, STREAM, dtype=torch.bfloat16, device=DEV)
    conv_step(conv_pool, None, whole_cu, slots32, carry, conv_input=conv_input)
    _, ngram_whole = ngram_token_history.reference(
        ids, ngram_pool, slots32, carry, whole_cu, EOS, NGRAM
    )
    ngram_pool = ngram_whole.clone()
    stage = _stage(gdn=False)
    target = _fake_target(metadata, batch, types.SimpleNamespace(), stage, input_ids=ids)
    _m.PrefillTarget._split_ple_states(target, conv_pool, ngram_pool, conv_input)
    torch.cuda.synchronize()

    k = len(rows.split)
    assert torch.equal(conv_pool, conv_p), "PLE conv slots differ from the main step's"
    assert torch.equal(stage.ple_conv[:k], ref_conv.index_select(0, rows.t(rows.tail_slots)))
    assert torch.equal(ngram_pool, ngram_p), "n-gram histories differ from the main step's"
    assert torch.equal(stage.ple_ngram[:k], ngram_l.index_select(0, rows.t(rows.tail_slots)))


@pytest.mark.parametrize("state", ["gdn", "ple"])
@pytest.mark.parametrize("name", list(BATCHES))
def test_split_states_are_the_two_iteration_schedule(name, state) -> None:
    batch = BATCHES[name]
    (_gdn_split if state == "gdn" else _ple_split)(batch, _batch_split(batch), _Rows(batch))


def _padded_gdn_call(rows, padded, seqs, zero_pad) -> None:
    """`_gdn_core_graph`'s CP and carried calls over a bucket's `padded` rows are bitwise the
    unpadded call; rows past the batch are zero or, without `zero_pad`, values never read."""
    gen = torch.Generator(device=DEV).manual_seed(rows + padded)
    q, k, v, g, beta = _gdn_inputs(padded, gen)
    lens = [rows // seqs] * seqs
    lens[-1] += rows - sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.long, device=DEV)
    slots = torch.tensor([3 + 5 * i for i in range(seqs)], dtype=torch.int32, device=DEV)
    carry = torch.tensor([i % 2 == 0 for i in range(seqs)], dtype=torch.bool, device=DEV)
    dev = torch.device(DEV)
    cp = fi._reaches_cp(seqs, HV, dev)
    keep = (torch.arange(padded, device=DEV) < rows).view(1, padded)
    padded_in = [
        torch.where(keep.view(1, padded, *([1] * (t.dim() - 2))), t, 0) if zero_pad else t
        for t in (q, k, v, g, beta)
    ]
    chunk = fi.cp_chunk_len_for(rows, max(H, HV), dev) if cp else None
    fi_kwargs = {"_cp_chunk_len": chunk, "max_seqlen": padded} if cp else None
    for path in [fi._chunk_carried] + ([fi._chunk_cp_inplace] if cp else []):
        results = []
        for unpadded in (True, False):
            pool = _strided_pool(torch.Generator(device=DEV).manual_seed(5), HV, DV, DK)
            out = torch.full((1, padded, HV, DV), -7.0, dtype=torch.bfloat16, device=DEV)
            if unpadded:
                path(
                    *(t[:, :rows] for t in (q, k, v, g, beta)), None, pool, slots, carry, cu,
                    out[:, :rows], True,
                )  # fmt: skip
            else:
                path(*padded_in, None, pool, slots, carry, cu, out, True, fi_kwargs=fi_kwargs)
            results.append((out[:, :rows].clone(), pool.clone()))
        torch.cuda.synchronize()
        assert torch.equal(results[0][0], results[1][0]), f"{path.__name__}: outputs"
        assert torch.equal(results[0][1], results[1][1]), f"{path.__name__}: states"


def _ple_publish_end() -> None:
    """`publish_end=k` (`DecodeTarget._ple_from_pool`): outputs bitwise the full launch's; conv
    pool rows of sequences below `k` published, the later ones left exactly as they were."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    # A 5-row context-shaped prefix, then three 3-row verify requests.
    inputs = _ple_inputs(gen, 14)
    pool0 = torch.randn(SLOTS, STREAM, PLE_STATE, generator=gen, device=DEV).bfloat16()
    slots = torch.tensor([2, 5, 7, 9], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, 5, 8, 11, 14], dtype=torch.int32, device=DEV)
    carry = torch.ones(4, dtype=torch.bool, device=DEV)
    outs, pools = [], [pool0.clone(), pool0.clone()]
    for publish_end, pool in zip((None, 1), pools):
        outs.append(_ple(inputs, None, pool, slots, cu, carry, 5, publish_end=publish_end))
    torch.cuda.synchronize()
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]), "outputs"
    assert torch.equal(pools[1][2], pools[0][2]), "the sequence below publish_end publishes"
    for slot in (5, 7, 9):
        assert torch.equal(pools[1][slot], pool0[slot]), f"slot {slot} must stay unpublished"
        assert not torch.equal(pools[0][slot], pool0[slot]), "the full launch publishes it"
    others = [s for s in range(SLOTS) if s not in (2, 5, 7, 9)]
    assert torch.equal(pools[1][others], pool0[others])


# (eager rows, padded bucket rows, sequences): every FlashInfer CP chunk-length class below 4096.
PADDED_CALLS = [
    (126, 128, 1), (300, 512, 1), (1000, 1024, 1), (1300, 2048, 1), (2046, 2048, 1),
    (3070, 4096, 1), (900, 1024, 2), (3000, 4096, 3),
]  # fmt: skip
_VARIANTS = [
    (f"padded-{r}-{p}-{s}-{'zero' if z else 'stale'}", _padded_gdn_call, (r, p, s, z))
    for r, p, s in PADDED_CALLS
    for z in (True, False)
] + [("ple_publish_end", _ple_publish_end, ())]


@pytest.mark.parametrize("helper,args", [v[1:] for v in _VARIANTS], ids=[v[0] for v in _VARIANTS])
def test_launch_variants_are_bitwise_the_plain_launch(helper, args) -> None:
    """A padded CUDA-graph GDN call and a partly publishing PLE launch against the plain ones."""
    helper(*args)
