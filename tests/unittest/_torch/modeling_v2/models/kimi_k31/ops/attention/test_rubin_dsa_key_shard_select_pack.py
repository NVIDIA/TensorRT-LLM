# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_dsa_key_shard_select_pack (helix8's key-sharded Top-K in one launch of helix16's
select / pack / merge over helix8's workspace) and the first-use check it is used behind.

CPU (``cpu_only``): the per-step cluster rule; the check's cases (the step's logits, ties,
forced fallback, full-width rows); the check's verdict with stand-in sides (equal, a difference
on one case, timed-out polls, a peer's failure), the step's logits restored either way.

GPU (sm_107): every certified cell with the group emulated on one GPU (every rank's workspace on
it over plain local buffers, every rank's select / pack on its own stream): each rank's KV rows,
counts and mask equal helix8's chain on the same inputs (the CuTe local Top-K, the pack, the merge
with ``failing_only``, the remap) word for word, at every cluster size of the cell, over two
steps through the same workspaces (both LL parities), and the chain equals the reference; no
exchange poll timed out; the skewed cells rescue short ranks.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_dsa_key_shard import (  # noqa: E501
    KEY_SHARD_LOCAL_K,
    key_shard_bytes_needed,
    key_shard_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E501
    rubin_dsa_key_shard_select_pack as mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_merge as merge_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_pack as pack_op,
)

op = mod.rubin_dsa_key_shard_select_pack

__extra_import_path__ = ["../../../..", "."]
from _validating import validating  # noqa: E402

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

TPB = 64
STRIDE_ROWS, ROW_OFFSET = 24 * TPB, 5 * TPB


# --- CPU -------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_cluster_is_sized_per_step():
    got = {rows: mod.select_cluster(rows, 212) for rows in range(1, 213)}
    assert {r for r, c in got.items() if c == 4} == set(range(1, 54))
    assert {r for r, c in got.items() if c == 3} == set(range(54, 71))
    assert {r for r, c in got.items() if c == 2} == set(range(71, 107))
    assert {r for r, c in got.items() if c == 1} == set(range(107, 213))
    assert all(rows * c <= 212 for rows, c in got.items())
    assert all(mod.select_cluster(rows, 212, cap=1) == 1 for rows in (1, 24, 54))
    # helix8's k3..k7 row counts up to b13: 4..104 rows.
    assert {mod.select_cluster(b * (k + 1), 212) for b in range(1, 14) for k in range(3, 8)} == {
        2,
        3,
        4,
    }


@pytest.mark.cpu_only
def test_the_check_cases():
    gen = torch.Generator().manual_seed(0)
    real = torch.randn(5, 96, generator=gen)
    lens = torch.tensor([96, 40, 0, 7, 96, 3], dtype=torch.int32)
    cases = list(mod.key_shard_check_cases(real, lens, 1, 2, 96))
    assert [c[0] for c in cases] == ["step", "ties", "fallback", "long"]
    step, ties, fallback, full = cases
    assert step[1] is real and torch.equal(step[2], lens[:5])
    assert torch.equal(ties[2], lens[:5]) and torch.equal(fallback[2], lens[:5])
    assert not bool((ties[1].view(torch.int32) & 0xFFFFF).any()), "3 mantissa bits left"
    assert ties[1].unique().numel() < real.unique().numel()
    # Rank 1 of 2 holds rows 1 and 3 whole: raised past every other rank's scores.
    raised = (fallback[1] >= 2.0**23).all(dim=1)
    assert raised.tolist() == [False, True, False, True, False]
    assert torch.equal(fallback[1][0::2], ties[1][0::2])
    assert full[2].dtype == torch.int32 and full[2].tolist() == [96] * 5
    assert torch.equal(full[1] * 4, torch.round(full[1] * 4)) and full[1].shape == real.shape


def _stub_ws(rows=4, width=32, rank=0, world=2):
    from types import SimpleNamespace

    logits = torch.randn(rows, width + 8, generator=torch.Generator().manual_seed(rank))
    return SimpleNamespace(local_logits=logits, rank=rank, world=world, owner=None)


def _sides(ws, rows, width, differ_on=None, timeout=None):
    """Stand-in sides: today's rows are a function of the case's logits; the select's equal them
    except on the case named ``differ_on``; ``timeout`` (a state tensor) counts one poll there."""
    seen = []

    def today(lens):
        lg = ws.local_logits[:rows, :width]
        seen.append(lg.clone())
        return (lg.argsort(dim=1).int(), lens.clone(), lg[:, 0] > 0)

    def select(lens):
        out = today(lens)
        seen.pop()
        name = ["step", "ties", "fallback", "long"][len(seen) - 1]
        if name == differ_on:
            out[1][0] += 1
            if timeout is not None:
                timeout[3] += 1
        return out

    return today, select, seen


@pytest.mark.cpu_only
@pytest.mark.parametrize("differ_on", [None, "step", "ties", "fallback", "long"])
def test_the_check_verdict(differ_on):
    ws = _stub_ws()
    before = ws.local_logits.clone()
    lens = torch.tensor([32, 10, 0, 31], dtype=torch.int32)
    votes = []

    def agree(w, ok):
        votes.append(ok)
        return ok

    today, select, seen = _sides(ws, 4, 32, differ_on)
    why = mod.key_shard_select_check(ws, 4, 32, lens, today, select, agree=agree)
    assert torch.equal(ws.local_logits, before), "the step's logits are restored"
    assert len(seen) == 4 and torch.equal(seen[0], before[:4, :32])
    assert votes == [True] * 4 + [differ_on is None], "a barrier per case, then the verdict"
    if differ_on is None:
        assert why == ""
    else:
        assert why == f"counts differ on the {differ_on} logits"


@pytest.mark.cpu_only
def test_the_check_fails_on_timed_out_polls_and_on_a_peer():
    ws = _stub_ws()
    lens = torch.full((4,), 32, dtype=torch.int32)
    state = torch.zeros(4, dtype=torch.int32)
    today, select, _ = _sides(ws, 4, 32)

    def select_timing_out(lens_):
        state[3] += 1
        return select(lens_)

    why = mod.key_shard_select_check(ws, 4, 32, lens, today, select_timing_out, (state,))
    assert why == "4 exchange polls timed out"
    today, select, _ = _sides(ws, 4, 32)
    votes = iter([True, True, True, True, False])
    why = mod.key_shard_select_check(ws, 4, 32, lens, today, select,
                                     agree=lambda w, ok: next(votes))  # fmt: skip
    assert why == "a peer's check failed"


@pytest.mark.cpu_only
def test_is_valid_refuses_keys_past_the_block_table():
    from types import SimpleNamespace

    runner = SimpleNamespace(merge=SimpleNamespace(rows_max=4, local_width=1024))
    kv = SimpleNamespace(
        req_idx=torch.zeros(4, dtype=torch.int32), block_table=torch.zeros(2, 3, dtype=torch.int32)
    )
    lens = torch.tensor([192, 0, 5, 64], dtype=torch.int32)
    op.is_valid(runner, lens, 4, None, None, None, None, kv)
    with pytest.raises(AssertionError, match="past the block table"):
        op.is_valid(runner, lens + 1, 4, None, None, None, None, kv)
    op.is_valid(runner, lens + 1, 4, None, None, None, None, None)


# --- GPU: every rank on one GPU --------------------------------------------------------------


def _scores(kind, world, rows, width, gen):
    if kind == "fp8":
        q = (torch.randn(rows, 4, 16, generator=gen) * 0.5).to(torch.float8_e4m3fn).float()
        out = []
        for _ in range(world):
            k = (torch.randn(width, 16, generator=gen) * 0.5).to(torch.float8_e4m3fn).float()
            s = torch.relu(torch.einsum("thd,jd->thj", q, k))
            out.append((s * torch.tensor([0.25, 0.5, 0.75, 1.0])[None, :, None]).sum(dim=1))
        return torch.stack(out)
    if kind == "zeros":
        return torch.zeros(world, rows, width)
    x = torch.randn(world, rows, width, generator=gen)
    if kind == "coarse":
        x = torch.round(x * 4) / 4
    if kind == "skew":
        x[0] += 4.0
    return x


class _Group:
    """``world`` ranks' helix8 workspaces (sized for the select / pack) over one GPU's buffers."""

    def __init__(self, spec, seed):
        gen = torch.Generator().manual_seed(seed)
        world, rows, top_k = spec["world"], spec["rows"], spec["top_k"]
        lo, hi = spec["row_lens"]
        self.world, self.rows, self.top_k = world, rows, top_k
        self.width = width = -(-max(hi, 256) // 256) * 256
        need = max(
            key_shard_bytes_needed(rows, top_k, world, KEY_SHARD_LOCAL_K, width),
            mod.select_bytes_needed(rows, top_k, world, width),
        )
        base = torch.zeros(world, -(-need // 256) * 256, dtype=torch.uint8, device="cuda")
        bufs = [base[r] for r in range(world)]
        self.ws = [
            key_shard_workspace(bufs, r, rows, top_k, width, max_spin=1 << 22) for r in range(world)
        ]
        self.sel = [mod.rubin_dsa_key_shard_select_ws(ws) for ws in self.ws]
        assert all(s is not None for s in self.sel), "the select / pack takes the workspace"
        self.lens = [
            torch.randint(lo, hi + 1, (rows,), generator=gen, dtype=torch.int32).clamp(max=width)
            for _ in range(world)
        ]
        self.logits = _scores(spec["kind"], world, rows, width, gen)
        self.kv = [None] * world
        self.tables = [None] * world
        req_idx = (torch.arange(rows) // 2).int().cuda()
        if spec["kv_rows"]:
            pages = -(-width // TPB) + 1
            for r in range(world):
                table = torch.randperm(60_000, generator=gen)[: (rows // 2 + 1) * pages]
                table = table.view(rows // 2 + 1, pages).int()
                table[0, 1] = -1  # a negative entry: its keys map to -1
                self.tables[r] = table.cuda()
                self.kv[r] = mod.key_shard_kv_rows(self.tables[r], req_idx, STRIDE_ROWS, ROW_OFFSET)
        self.req_idx = req_idx
        self.bounds = None
        if spec["bounds"]:
            self.bounds = torch.randint(0, 3, (rows,), generator=gen, dtype=torch.int32).cuda()
        self.streams = [torch.cuda.Stream() for _ in range(world)]

    def outputs(self):
        rows = self.rows
        return [
            (
                torch.full((rows, self.top_k), -7, dtype=torch.int32, device="cuda"),
                torch.full((rows,), -3, dtype=torch.int32, device="cuda"),
                None if self.bounds is None else torch.zeros(rows, dtype=torch.bool, device="cuda"),
            )
            for _ in range(self.world)
        ]

    def load(self):
        for r, ws in enumerate(self.ws):
            ws.local_logits[: self.rows, : self.width].copy_(self.logits[r])

    def today(self):
        """helix8's chain on every rank: local Top-K and pack, then the merges, then the remap."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E501
            convert_req_index_to_global,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_topk_decode_cute import (  # noqa: E501
            rubin_topk_decode_cute,
        )

        rows = self.rows
        lens = [ln.cuda() for ln in self.lens]
        for r, ws in enumerate(self.ws):
            local = ws.local_topk[:rows]
            rubin_topk_decode_cute(
                ws.local_logits[:rows, : self.width], lens[r], local, ws.local_k, 1
            )
            pack_op(local, lens[r], ws)
        results = []
        for r, (ws, (out, counts, mask)) in enumerate(zip(self.ws, self.outputs())):
            merge_op(out, counts, self.bounds, mask, ws, failing_only=True)
            if self.kv[r] is not None:
                out = convert_req_index_to_global(
                    self.req_idx,
                    self.tables[r],
                    out,
                    TPB,
                    self.top_k,
                    STRIDE_ROWS,
                    ROW_OFFSET // TPB,
                )
            results.append((out, counts, mask))
        torch.cuda.synchronize()
        return results

    def select(self, sms):
        """The select / pack on every rank, each on its own stream (one served call)."""
        outs = self.outputs()
        cur = torch.cuda.current_stream()
        lens = [ln.cuda() for ln in self.lens]
        for r, sel in enumerate(self.sel):
            sel.sms = sms
            self.streams[r].wait_stream(cur)
            with torch.cuda.stream(self.streams[r]):
                out, counts, mask = outs[r]
                op(sel, lens[r], self.rows, out, counts, self.bounds, mask, self.kv[r])
        for s in self.streams:
            cur.wait_stream(s)
        torch.cuda.synchronize()
        return outs


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda c: c.why[:48])
def test_certified_cells(cell):
    spec = cell.spec
    grp = _Group(spec, seed=spec["world"] * 101 + spec["rows"] + len(cell.why))
    for step in range(2):
        grp.load()
        want = grp.today()
        if step == 0:
            for r in range(grp.world):
                ref = op.reference(
                    grp.sel[r], grp.lens[r], grp.rows, None, None, grp.bounds,
                    want[r][2], grp.kv[r],
                    group=[(grp.logits[w], grp.lens[w]) for w in range(grp.world)],
                )  # fmt: skip
                assert torch.equal(want[r][0].cpu(), ref[0]), f"rank {r}: the chain's rows"
                assert torch.equal(want[r][1].cpu(), ref[1]), f"rank {r}: the chain's counts"
        for sms in spec["sms"]:
            with validating(op):
                got = grp.select(sms)
            clusters = {s.cluster for s in grp.sel}
            assert len(clusters) == 1
            for r in range(grp.world):
                for label, a, b in zip(("rows", "counts", "mask"), want[r], got[r]):
                    assert (a is None) == (b is None), label
                    if a is not None:
                        assert torch.equal(a, b), f"rank {r} step {step} sms {sms}: {label}"
        grp.logits = grp.logits.flip(-1)  # another step's scores through the same workspaces
    states = [s.merge.state.tolist() for s in grp.sel]
    assert all(st[3] == 0 for st in states), "an exchange poll timed out"
    if spec["kind"] == "skew":
        assert sum(st[2] for st in states) > 0, "the skewed rows rescue a short rank"


@on_sm107
def test_every_cluster_size_compiles_before_use():
    grp = _Group(op.CELLS[0].spec, seed=3)
    sel = grp.sel[0]
    assert sel.precompile(78, True, False) == (2, 3, 4)
    assert sel.precompile(130, False, False) == (1, 2, 3, 4)
