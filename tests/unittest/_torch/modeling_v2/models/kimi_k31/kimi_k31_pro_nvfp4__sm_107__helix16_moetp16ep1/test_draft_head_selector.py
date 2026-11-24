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
"""The drafter head with the selector's hidden projection appended.

Load: every TP16 rank's bank is its own lm_head vocabulary rows, then the whole selector
projection, and the head weight and the projection become views of it. Step: one head GEMM over
the bank gives the shard logits bitwise the separate head's, so the top-32 pairs written into the
exchange row and the merged candidates are bitwise too (the top-32 reads the logits in place at
the bank's row pitch); the gate columns are within the separate gate's envelope against the FP64
product (another kernel's FP32 order), and the selector reads them in place. No gate GEMM runs on
that path; a CUDA graph of the step replays its eager bits; row counts without a head-table row
keep the separate head and gate.
"""

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = _FAMILY + ".kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
SHARD, RANK, H, TP = 10240, 256, 7168, 16
VOCAB = SHARD * TP
SLOTS = 7  # one request's draft rows
TP_RANK = 3


def _sm100_family():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _sm100_family(), reason="needs an SM 10.x GPU")


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_PKG + ".modeling")


@pytest.fixture(scope="module")
def weights_mod():
    return importlib.import_module(_PKG + ".weights")


@pytest.fixture(scope="module")
def spec_ops():
    return SimpleNamespace(
        top32=importlib.import_module(_FAMILY + ".ops.spec.dflash2_global_top32"),
        tokens=importlib.import_module(
            _FAMILY + ".ops.spec.dflash2_selector_greedy_tokens"
        ).k31_dflash2_selector_greedy_tokens,
    )


def _bits(t):
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _same(u, v):
    return u.shape == v.shape and torch.equal(_bits(u), _bits(v))


def _rel(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max()).item()


def _rel_l2(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


# ---- load: the bank under the TP16 vocabulary sharding -------------------------------------------


@dataclass(frozen=True)
class _ModelWeights:
    """The one field of ``FlatModelWeights`` the bank replaces."""

    lm_head: object


def test_bank_layout_on_every_rank(mod, weights_mod):
    """Each rank's bank is [its lm_head rows ; the whole selector projection]; the head weight and
    the projection are its two row blocks (one storage), every other drafter weight is untouched,
    and the drafter's check accepts it (and refuses a copy)."""
    cols = 64
    g = torch.Generator(device="cuda").manual_seed(1)
    full = torch.randn(VOCAB, cols, generator=g, device="cuda").bfloat16()
    selector = torch.randn(RANK, cols, generator=g, device="cuda").bfloat16()
    checkpoint = {
        "lm_head.weight": full,
        "candidate_selector.hidden_projection.weight": selector,
    }
    for rank in range(TP):
        loader = weights_mod.FlatCheckpointLoader(checkpoint, {}, device="cuda", rank=rank)
        head = loader.bf16("lm_head", (VOCAB, cols), shard_dim=0)
        sel = loader.read(
            "candidate_selector.hidden_projection.weight", (RANK, cols), torch.bfloat16
        )
        other = torch.zeros(4, cols, dtype=torch.bfloat16, device="cuda")
        model, draft = mod._bank_head_selector(
            _ModelWeights(head), {"d_sel_hidden_proj": sel, "d_fc": other}
        )
        bank = draft[mod.DRAFT_HEAD_SELECTOR_BANK]
        assert bank.shape == (SHARD + RANK, cols) and bank.is_contiguous()
        assert torch.equal(bank[:SHARD], full[rank * SHARD : (rank + 1) * SHARD])
        assert torch.equal(bank[SHARD:], selector)
        assert mod._same_view(model.lm_head.weight, bank[:SHARD])
        assert mod._same_view(draft["d_sel_hidden_proj"], bank[SHARD:])
        assert draft["d_fc"] is other
        w = dict(draft)
        fake = SimpleNamespace(
            _core=lambda w=w: SimpleNamespace(w=w),
            lm_head=SimpleNamespace(weight=model.lm_head),
            _w=w.__getitem__,
        )
        mod.DraftModel._check_head_selector_bank(fake)
        fake.lm_head = SimpleNamespace(weight=mod.Bf16Weight(bank[:SHARD].clone()))
        with pytest.raises(RuntimeError, match="one storage"):
            mod.DraftModel._check_head_selector_bank(fake)


# ---- the step: one head GEMM for the logits and the gate ------------------------------------------


@pytest.fixture(scope="module")
def site():
    g = torch.Generator(device="cuda").manual_seed(2)
    head = (0.02 * torch.randn(SHARD, H, generator=g, device="cuda")).bfloat16()
    selector = (0.02 * torch.randn(RANK, H, generator=g, device="cuda")).bfloat16()
    # The block's eight rows; the drafter reads rows 1..7 (a packed slice at a 14336 B offset).
    block = torch.randn(8, H, generator=g, device="cuda").bfloat16()
    others = torch.randperm(VOCAB, device="cuda", generator=g)[: TP * 32 * SLOTS].view(SLOTS, -1)
    exchange = torch.stack(
        [others.float(), torch.randn(SLOTS, TP * 32, generator=g, device="cuda")], -1
    ).flatten(-2)
    return SimpleNamespace(
        head=head,
        selector=selector,
        block=block,
        exchange=exchange.contiguous(),
        pred=(0.1 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16(),
        succ=(0.1 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16(),
        anchor=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def _bank(mod, site):
    return mod.draft_head_selector_bank(site.head, site.selector)


def _candidates(spec_ops, logits, exchange):
    """The shard's top-32 pairs into this rank's columns of the exchange row, then the merge."""
    own = exchange[:, TP_RANK * 64 : (TP_RANK + 1) * 64]
    spec_ops.top32.k31_dflash2_shard_top32(
        logits.view(1, SLOTS, SHARD), 32, TP_RANK * SHARD, True, out=own
    )
    unary, ids = spec_ops.top32.k31_dflash2_merge_top32(exchange.view(1, SLOTS, -1), 32)
    return unary, ids


def _separate(mod, spec_ops, site, bank):
    """Off: the head on the shard, the gate on the projection (each a view of the same bank)."""
    rows = site.block[1:]
    exchange = site.exchange.clone()
    logits = mod.head_linear(rows, bank[:SHARD])
    unary, ids = _candidates(spec_ops, logits, exchange)
    gate = mod.selector_gate_linear(rows.view(1, SLOTS, H), bank[SHARD:])
    tokens = spec_ops.tokens(
        site.pred,
        site.succ,
        bank[SHARD:],
        ids,
        unary,
        rows.view(1, SLOTS, H),
        site.anchor,
        torch.bfloat16,
        gate_linear=mod.selector_gate_linear,
    )
    return logits, gate.view(SLOTS, RANK), exchange, unary, ids, tokens


def _banked(mod, spec_ops, site, bank, gate_linear):
    """On: one GEMM over the bank, both consumers reading its output in place."""
    rows = site.block[1:]
    exchange = site.exchange.clone()
    out = mod.draft_head_selector_gemm(rows, bank)
    logits, gate = out[:, :SHARD], out[:, SHARD:]
    unary, ids = _candidates(spec_ops, logits, exchange)
    tokens = spec_ops.tokens(
        site.pred,
        site.succ,
        bank[SHARD:],
        ids,
        unary,
        rows.view(1, SLOTS, H),
        site.anchor,
        torch.bfloat16,
        gate_linear=gate_linear,
        gate=gate.reshape(1, SLOTS, RANK),
    )
    return logits, gate, exchange, unary, ids, tokens


def _no_gate_gemm(*args, **kwargs):
    raise AssertionError("the selector gate GEMM ran with the gate in the head GEMM")


def test_on_matches_off(mod, spec_ops, site):
    """Logits, the exchange row (pairs), unary values and candidates bitwise; the gate within the
    separate gate's envelope against the FP64 product; the tokens are the selector's on that gate
    (read in place equals read from a packed copy)."""
    bank = _bank(mod, site)
    off = [t.clone() for t in _separate(mod, spec_ops, site, bank)]
    on = _banked(mod, spec_ops, site, bank, _no_gate_gemm)
    torch.cuda.synchronize()
    for name, i in (("logits", 0), ("pairs", 2), ("unary", 3), ("candidates", 4)):
        assert _same(on[i], off[i]), name
    rows = site.block[1:].double()
    ref = rows @ site.selector.double().t()
    assert _rel(on[1], ref) < 2e-2
    assert _rel_l2(on[1], ref) <= 1.25 * _rel_l2(off[1], ref)
    packed = spec_ops.tokens(
        site.pred,
        site.succ,
        bank[SHARD:],
        on[4],
        on[3],
        site.block[1:].view(1, SLOTS, H),
        site.anchor,
        torch.bfloat16,
        gate=on[1].contiguous().view(1, SLOTS, RANK),
    )
    assert torch.equal(on[5], packed)
    want = spec_ops.tokens.reference(
        site.pred,
        site.succ,
        bank[SHARD:],
        on[4],
        on[3],
        site.block[1:].view(1, SLOTS, H),
        site.anchor,
        torch.bfloat16,
        gate=on[1].view(1, SLOTS, RANK),
    )
    assert torch.equal(on[5], want)


def test_graph_replay_matches_eager(mod, spec_ops, site):
    bank = _bank(mod, site)
    _banked(mod, spec_ops, site, bank, _no_gate_gemm)  # prepares the bank's engine eagerly
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _banked(mod, spec_ops, site, bank, _no_gate_gemm)
    saved = site.block.clone()
    try:
        for seed in (31, 32):
            g = torch.Generator(device="cuda").manual_seed(seed)
            site.block.copy_(torch.randn(8, H, generator=g, device="cuda").bfloat16())
            graph.replay()
            torch.cuda.synchronize()
            eager = _banked(mod, spec_ops, site, bank, _no_gate_gemm)
            torch.cuda.synchronize()
            for got, want in zip(captured, eager):
                assert _same(got, want)
    finally:
        site.block.copy_(saved)


def _worker():
    head = SimpleNamespace(shard_width=SHARD)
    worker = SimpleNamespace(_vocab_parallel=SimpleNamespace(head=head))
    return worker, head


def test_worker_takes_the_gate_from_the_head_gemm(mod, site, monkeypatch):
    """The walk step's shard logits come from the bank GEMM (the only head-kernel launch, on the
    bank's N) and the drafter's walk runs no gate GEMM; without the walk, or without a bank, the
    head alone."""
    bank = _bank(mod, site)
    rows = site.block[1:]
    worker, head = _worker()
    head.local_logits = lambda x: mod.head_linear(x, bank[:SHARD])
    launched = []
    real = mod._head_gemm

    def spy(x, weight, choice):
        launched.append(tuple(weight.shape))
        return real(x, weight, choice)

    monkeypatch.setattr(mod, "_head_gemm", spy)
    monkeypatch.setattr(mod, "selector_gate_linear", _no_gate_gemm)
    draft = SimpleNamespace(head_selector_bank=bank)
    logits, gate = mod.DFlashWorker._draft_shard_logits(worker, draft, rows, True)
    assert gate is not None and launched == [(SHARD + RANK, H)]
    assert logits.data_ptr() == gate.data_ptr() - SHARD * 2 and logits.stride(0) == SHARD + RANK
    w = {
        "d_sel_pred": site.pred,
        "d_sel_succ": site.succ,
        "d_sel_hidden_proj": bank[SHARD:],
    }
    fake = SimpleNamespace(_w=w.__getitem__)
    ids = torch.stack([torch.randperm(VOCAB, device="cuda")[:32] for _ in range(SLOTS)]).view(
        1, SLOTS, 32
    )
    unary = torch.randn(1, SLOTS, 32, device="cuda").sort(dim=-1, descending=True).values
    mod.DraftModel.select_candidate_tokens(
        fake,
        ids,
        unary,
        rows.view(1, SLOTS, H),
        site.anchor,
        torch.bfloat16,
        gate=gate.reshape(1, SLOTS, RANK),
    )
    launched.clear()
    logits, gate = mod.DFlashWorker._draft_shard_logits(worker, draft, rows, False)
    assert gate is None and launched == [(SHARD, H)]
    launched.clear()
    logits, gate = mod.DFlashWorker._draft_shard_logits(
        worker, SimpleNamespace(head_selector_bank=None), rows, True
    )
    assert gate is None and launched == [(SHARD, H)]


def test_rows_without_a_table_row_keep_the_separate_head(mod, site):
    bank = _bank(mod, site)
    table = mod._HEAD_GEMM_ROWS_RUBIN if mod._rubin_rows(bank.device) else mod._HEAD_GEMM_ROWS
    rows = table[-1][0] + 1
    g = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(rows, H, generator=g, device="cuda").bfloat16()
    assert mod.draft_head_selector_gemm(x, bank) is None
    worker, head = _worker()
    head.local_logits = lambda h: mod.head_linear(h, bank[:SHARD])
    logits, gate = mod.DFlashWorker._draft_shard_logits(
        worker, SimpleNamespace(head_selector_bank=bank), x, True
    )
    assert gate is None
    assert _same(logits, mod.head_linear(x, bank[:SHARD]))
