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
"""``spec_step_copies`` (the one-model speculative decoding step's copy kernels) against the torch ops each replaces,
bit for bit (int32 copies), at every split of R = 1 .. 8 requests x T = 1, 2, 4, 8 tokens, on buffers sized like an
engine's for max batch 8 and max draft length 7 (sampler stores [8, slots, 1], [8, slots, 1], [slots], [slots, 7] over
8 or 16 slots; draft-token buffer 56, KV-length offsets 8, per-token buffers max_num_tokens).

* ``SlotScatter.scatter`` vs SpecSampler's torch store update (each output padded with zeros or cut to its store's
  width, the new tokens at or past a row's accepted length zeroed, then four ``index_copy_`` by slot): outputs
  narrower than (dynamic draft length), as wide as and wider than the stores, and mixed per field; row_begin 0 and
  > 0; shuffled distinct slot tables; accepted lengths from 0 to past the output width. A mixed context / generation
  batch laid out as the one-model worker writes it (a context row's tokens past column 0 never written) stores zeros
  there.
* ``StepInputGather.gather`` vs PyTorchModelEngine._prepare_tp_inputs's torch overlap gathers (the stores by slot into
  the input ids and draft tokens, the lengths by the per-token index list into the position offsets,
  ``new_tokens_lens - T`` by slot into the KV-length offsets): the engine's offsets (the requests without a previous
  batch first), zero and arbitrary offsets, a draft width below T - 1, the engine's and arbitrary index lists.
* The calls the kernels do not cover launch nothing and return False, so the caller keeps its torch path.

Every buffer starts random and is compared whole (what an op must not touch included), every case runs twice (reruns
bit-identical), and one CUDA graph per split family (R = 1 .. 8 at one T) captures each step's scatter and gather; it
is replayed with every input rewritten in place, bit-identical to the eager ops and to the torch ops on every replay.

Table: ``python3 test_spec_step_copies.py report``. Timing: ``python3 test_spec_step_copies.py time`` (CUDA graphs of
back-to-back calls, median us per call over 15 replays, each kernel vs the torch ops it replaces, at every split).
"""

import statistics
import sys

import pytest
import torch

MAX_BATCH = 8  # max_batch_size
MAX_DRAFT = 7  # max_draft_len (linear speculation: max_total_draft_tokens == max_draft_len)
STORE_WIDTH = MAX_DRAFT + 1  # the sampler's new_tokens / next_new_tokens stores
MAX_NUM_TOKENS = 8192  # the engine's per-token buffers (input ids, positions, previous_*)
DRAFT_BUFFER = MAX_DRAFT * MAX_BATCH  # draft_tokens_cuda: max_draft_loop_tokens * batch_size

ROWS = list(range(1, MAX_BATCH + 1))
TOKENS = [1, 2, 4, 8]
SPLITS = [(r, t) for t in TOKENS for r in ROWS]


def _op():
    from tensorrt_llm._torch.cute_dsl_kernels.spec_step_copies import op

    return op


def _supported() -> bool:
    return torch.cuda.is_available() and _op().is_supported()


pytestmark = pytest.mark.skipif(
    not _supported(), reason="the step-copy kernels run on SM 100 with the CuTe DSL"
)


def kernel_scatter(args) -> None:
    """``SlotScatter.scatter`` on ``args``, which it must cover."""
    assert _op().SlotScatter().scatter(*args), "scatter declined a covered call"


def kernel_gather(args) -> None:
    """``StepInputGather.gather`` on ``args``, which it must cover."""
    assert _op().StepInputGather().gather(*args), "gather declined a covered call"


# ----------------------------------------------------------------------------------------------------------------
# The torch ops each kernel replaces
# ----------------------------------------------------------------------------------------------------------------


def torch_scatter(
    outputs,
    num_skip,
    num_sampling_requests,
    slot_table,
    store_new_tokens,
    store_next_new_tokens,
    store_new_tokens_lens,
    store_next_draft_tokens,
):
    """SpecSampler's torch store update (SlotScatter.scatter's arguments; the slots as the long tensor it builds from
    the requests' seq slots)."""
    slots = slot_table[:num_sampling_requests].long()
    end = num_skip + num_sampling_requests
    o_new_tokens = outputs["new_tokens"][num_skip:end]
    o_new_tokens_lens = outputs["new_tokens_lens"][num_skip:end]
    o_next_draft_tokens = outputs["next_draft_tokens"][num_skip:end]
    o_next_new_tokens = outputs["next_new_tokens"][num_skip:end]

    def fit(t, width):  # pad with zeros or truncate to the store width
        if t.shape[1] < width:
            return torch.nn.functional.pad(t, (0, width - t.shape[1]))
        return t[:, :width]

    o_new_tokens = fit(o_new_tokens, store_new_tokens.shape[0])
    columns = torch.arange(o_new_tokens.shape[1], device=o_new_tokens.device)
    o_new_tokens = torch.where(columns < o_new_tokens_lens[:, None], o_new_tokens, 0)
    o_next_draft_tokens = fit(o_next_draft_tokens, store_next_draft_tokens.shape[1])
    o_next_new_tokens = fit(o_next_new_tokens, store_next_new_tokens.shape[0])
    store_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
    store_next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
    store_new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
    store_next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)


def torch_gather(
    new_tokens_device,
    next_draft_tokens_device,
    new_tokens_lens_device,
    previous_batch_indices_cuda,
    previous_pos_indices_cuda,
    previous_batch_len,
    runtime_tokens_per_gen_step,
    runtime_draft_token_buffer_width,
    input_ids_cuda,
    num_tokens,
    draft_tokens_cuda,
    num_draft_tokens,
    previous_pos_id_offsets_cuda,
    pos_begin,
    previous_kv_lens_offsets_cuda,
    kv_begin,
):
    """PyTorchModelEngine._prepare_tp_inputs's torch overlap gathers (StepInputGather.gather's arguments)."""
    tokens = runtime_tokens_per_gen_step
    width = runtime_draft_token_buffer_width
    previous_slots = previous_batch_indices_cuda[:previous_batch_len]
    previous_batch_tokens = previous_batch_len * tokens
    new_tokens = new_tokens_device.transpose(0, 1)[previous_slots, :tokens].flatten()
    input_ids_cuda[num_tokens : num_tokens + previous_batch_tokens].copy_(
        new_tokens, non_blocking=True
    )
    previous_batch_draft_tokens = previous_batch_len * width
    if width > 0:
        draft_tokens_cuda[num_draft_tokens : num_draft_tokens + previous_batch_draft_tokens].copy_(
            next_draft_tokens_device[previous_slots, :width].flatten(), non_blocking=True
        )
    kv_len_offsets_device = new_tokens_lens_device - tokens
    previous_pos_id_offsets_cuda[pos_begin : pos_begin + previous_batch_tokens].copy_(
        new_tokens_lens_device[previous_pos_indices_cuda[0:previous_batch_tokens]],
        non_blocking=True,
    )
    previous_kv_lens_offsets_cuda[kv_begin : kv_begin + previous_batch_len].copy_(
        kv_len_offsets_device[previous_slots], non_blocking=True
    )


# ----------------------------------------------------------------------------------------------------------------
# One decode step's buffers
# ----------------------------------------------------------------------------------------------------------------


def rand_i32(shape, g, low=-(1 << 30), high=1 << 30):
    """Random host int32 (``g``: a CPU generator)."""
    return torch.randint(low, high, tuple(shape), generator=g, dtype=torch.int32)


def shuffled_distinct(count, total, g):
    """``count`` distinct indices below ``total`` (>= 2) in a shuffled order: never ascending and never index r at
    position r, so a kernel that takes a row for its slot (or ignores the order) fails."""
    while True:
        picked = torch.randperm(total, generator=g)[:count].tolist()
        if all(p != r for r, p in enumerate(picked)) and (count < 2 or picked != sorted(picked)):
            return picked


def engine_begins(rows, tokens):
    """The engine's gather offsets (input ids, draft tokens, position offsets, KV-length offsets) when the batch's
    other MAX_BATCH - rows generation requests (no previous batch; T tokens, T - 1 draft tokens each) come first."""
    first = MAX_BATCH - rows
    return first * tokens, first * (tokens - 1), first * tokens, first


class Step:
    """One decode step of ``rows`` requests x ``tokens`` tokens on engine-sized int32 device buffers, all random so
    that a stray or a missing write shows (index buffers hold valid indices past their used part):

    * the forward's outputs: new_tokens [N, a], next_new_tokens [N, b], new_tokens_lens [N], next_draft_tokens [N, c]
      (default a = b = T, c = T - 1);
    * the sampler's slot table [slots] and stores new_tokens / next_new_tokens [8, slots, 1], new_tokens_lens [slots],
      next_draft_tokens [slots, 7];
    * the engine's previous_batch_indices / previous_pos_indices and the gathers' outputs input_ids, draft_tokens [56],
      previous_pos_id_offsets and previous_kv_lens_offsets [8] (grown only when an offset needs it).

    ``chain``: the gather reads the scatter's slots in another order (the next step's inputs from this step's stores).
    """

    def __init__(
        self,
        rows,
        tokens,
        seed,
        *,
        num_slots=MAX_BATCH,
        widths=None,
        row_begin=0,
        out_rows=None,
        begins=(0, 0, 0, 0),
        draft_width=None,
        pos_list="engine",
        chain=False,
    ):
        self.rows, self.tokens, self.num_slots = rows, tokens, num_slots
        self.widths = widths or (tokens, tokens, tokens - 1)
        self.row_begin, self.begins = row_begin, begins
        self.draft_width = tokens - 1 if draft_width is None else draft_width
        self.pos_list, self.chain = pos_list, chain
        self.g = torch.Generator().manual_seed(seed)
        n_out = max(MAX_BATCH, row_begin + rows) if out_rows is None else out_rows
        assert n_out >= row_begin + rows
        n_draft = max(DRAFT_BUFFER, begins[1] + rows * self.draft_width)
        n_kv = max(MAX_BATCH, begins[3] + rows)
        a, b, c = self.widths
        shapes = {
            "out_new": (n_out, a),
            "out_next": (n_out, b),
            "out_lens": (n_out,),
            "out_draft": (n_out, c),
            "slot_table": (num_slots,),
            "st_new": (STORE_WIDTH, num_slots, 1),
            "st_next": (STORE_WIDTH, num_slots, 1),
            "st_lens": (num_slots,),
            "st_draft": (num_slots, MAX_DRAFT),
            "prev_slots": (MAX_NUM_TOKENS,),
            "prev_pos": (MAX_NUM_TOKENS,),
            "input_ids": (MAX_NUM_TOKENS,),
            "draft": (n_draft,),
            "pos_off": (MAX_NUM_TOKENS,),
            "kv_off": (n_kv,),
        }
        self.bufs = {k: torch.empty(v, dtype=torch.int32, device="cuda") for k, v in shapes.items()}
        self.rewrite()

    def rewrite(self) -> None:
        """New random contents for every buffer, written in place (a captured graph keeps the addresses)."""
        g, rows, tokens, num_slots = self.g, self.rows, self.tokens, self.num_slots
        b = self.bufs
        for t in b.values():
            t.copy_(rand_i32(t.shape, g))
        # Accepted lengths from none to past the new-token width, so every row cuts its new tokens somewhere.
        b["out_lens"].copy_(rand_i32(b["out_lens"].shape, g, 0, self.widths[0] + 2))
        slots = shuffled_distinct(rows, num_slots, g)
        table = rand_i32((num_slots,), g, 0, num_slots)
        table[:rows] = torch.tensor(slots, dtype=torch.int32)
        b["slot_table"].copy_(table)
        previous = slots[1:] + slots[:1] if self.chain else shuffled_distinct(rows, num_slots, g)
        previous = torch.tensor(previous, dtype=torch.int32)
        index = rand_i32((MAX_NUM_TOKENS,), g, 0, num_slots)
        index[:rows] = previous
        b["prev_slots"].copy_(index)
        per_token = rand_i32((MAX_NUM_TOKENS,), g, 0, num_slots)
        if self.pos_list == "engine":  # each row's slot, once per token
            per_token[: rows * tokens] = previous.repeat_interleave(tokens)
        b["prev_pos"].copy_(per_token)

    def outputs(self, b):
        return {
            "new_tokens": b["out_new"],
            "next_new_tokens": b["out_next"],
            "new_tokens_lens": b["out_lens"],
            "next_draft_tokens": b["out_draft"],
        }

    def scatter_args(self, b):
        """SlotScatter.scatter's arguments on the buffers ``b``."""
        return (self.outputs(b), self.row_begin, self.rows, b["slot_table"], b["st_new"], b["st_next"],
                b["st_lens"], b["st_draft"])  # fmt: skip

    def gather_args(self, b):
        """StepInputGather.gather's arguments on the buffers ``b``."""
        ib, db, pb, kb = self.begins
        return (b["st_next"], b["st_draft"], b["st_lens"], b["prev_slots"], b["prev_pos"], self.rows, self.tokens,
                self.draft_width, b["input_ids"], ib, b["draft"], db, b["pos_off"], pb, b["kv_off"], kb)  # fmt: skip


def clone_bufs(bufs):
    """A copy of every buffer."""
    return {k: t.clone() for k, t in bufs.items()}


def mismatches(x, y):
    """Names of the buffers whose contents differ."""
    return [k for k in x if not torch.equal(x[k], y[k])]


def compare(bufs, run_torch, run_kernel):
    """The buffers where the kernel differs from the torch ops and where a rerun differs from the kernel, each run on
    its own copy of ``bufs``."""
    want, got, again = clone_bufs(bufs), clone_bufs(bufs), clone_bufs(bufs)
    run_torch(want)
    run_kernel(got)
    run_kernel(again)
    torch.cuda.synchronize()
    return dict(torch=mismatches(want, got), rerun=mismatches(got, again))


def result(case, **bad):
    """A table row; ``bad`` maps each comparison (torch, eager, rerun) to the buffers where it differed."""
    failed = {k: v for k, v in bad.items() if v}
    if not failed:
        return dict(case=case, identical="yes (= " + ", = ".join(bad) + ")", ok=True)
    detail = "; ".join(f"!= {k}: {', '.join(v)}" for k, v in failed.items())
    return dict(case=case, identical=f"no: {detail}", ok=False)


def _seed(op_index, rows, tokens, case):
    return 100000 * op_index + 1000 * rows + 10 * tokens + case


# ----------------------------------------------------------------------------------------------------------------
# The cases of one split
# ----------------------------------------------------------------------------------------------------------------


def scatter_cases(rows, tokens):
    """(label, Step arguments): the forward's outputs narrower than or as wide as the stores (dynamic draft length;
    the engine's batch of 8 rows, the first 8 - R skipped), wider than every store, and mixed per field."""
    t = tokens
    fit = "pad" if t < STORE_WIDTH else "exact"
    return [
        (f"{fit}: widths ({t}, {t}, {t - 1}) -> (8, 8, 7), 8 slots, row_begin {MAX_BATCH - rows} of 8 rows",
         dict(num_slots=MAX_BATCH, row_begin=MAX_BATCH - rows, out_rows=MAX_BATCH)),
        (f"cut: widths ({t + 8}, {t + 8}, {t + 7}) -> (8, 8, 7), 16 slots, row_begin 0",
         dict(num_slots=2 * MAX_BATCH, widths=(t + 8, t + 8, t + 7), out_rows=rows + 1)),
        (f"mixed: widths ({t + 8}, {t}, {t + 7}) -> (8, 8, 7), 16 slots, row_begin 3",
         dict(num_slots=2 * MAX_BATCH, widths=(t + 8, t, t + 7), row_begin=3, out_rows=rows + 4)),
    ]  # fmt: skip


def gather_cases(rows, tokens):
    """(label, Step arguments): the engine's layout, zero offsets, and arbitrary offsets with a narrower draft width
    and an arbitrary per-token index list."""
    begins = engine_begins(rows, tokens)
    return [
        (f"engine: offsets {begins}, draft width {tokens - 1}, per-token list = slots x T, 8 slots",
         dict(num_slots=MAX_BATCH, begins=begins)),
        (f"offsets 0, draft width {tokens - 1}, per-token list = slots x T, 16 slots",
         dict(num_slots=2 * MAX_BATCH)),
        (f"offsets (37, 5, 11, 3), draft width {(tokens - 1) // 2}, arbitrary per-token list, 16 slots",
         dict(num_slots=2 * MAX_BATCH, begins=(37, 5, 11, 3), draft_width=(tokens - 1) // 2, pos_list="any")),
    ]  # fmt: skip


def measure_scatter(rows, tokens):
    out = []
    for i, (case, kw) in enumerate(scatter_cases(rows, tokens)):
        st = Step(rows, tokens, _seed(1, rows, tokens, i), **kw)
        bad = compare(
            st.bufs,
            lambda b: torch_scatter(*st.scatter_args(b)),
            lambda b: kernel_scatter(st.scatter_args(b)),
        )
        out.append(result(case, **bad))
    return out


def measure_gather(rows, tokens):
    out = []
    for i, (case, kw) in enumerate(gather_cases(rows, tokens)):
        st = Step(rows, tokens, _seed(2, rows, tokens, i), **kw)
        bad = compare(
            st.bufs,
            lambda b: torch_gather(*st.gather_args(b)),
            lambda b: kernel_gather(st.gather_args(b)),
        )
        out.append(result(case, **bad))
    return out


def measure_graph(tokens, replays=6):
    """One CUDA graph for the split family R = 1 .. 8 at ``tokens``: each step's scatter and its gather (from the
    stores the scatter wrote, as the next step reads them), captured once and replayed with every input rewritten in
    place; each replay against the eager ops and the torch ops on the same inputs, then a replay of the last inputs
    against the first."""
    steps = [
        Step(r, tokens, _seed(4, r, tokens, 0), num_slots=2 * MAX_BATCH, row_begin=MAX_BATCH - r,
             out_rows=MAX_BATCH, begins=engine_begins(r, tokens), chain=True)
        for r in ROWS
    ]  # fmt: skip

    def run_kernels(st, b):
        kernel_scatter(st.scatter_args(b))
        kernel_gather(st.gather_args(b))

    def run_torch(st, b):
        torch_scatter(*st.scatter_args(b))
        torch_gather(*st.gather_args(b))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for st in steps:  # the kernels compile on their first call, outside capture
            run_kernels(st, clone_bufs(st.bufs))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for st in steps:
                run_kernels(st, st.bufs)
    torch.cuda.synchronize()
    out = []
    before = None
    for rep in range(replays):
        for st in steps:
            st.rewrite()
        before = [clone_bufs(st.bufs) for st in steps]
        graph.replay()
        torch.cuda.synchronize()
        bad = dict(torch=[], eager=[])
        for st, start in zip(steps, before):
            want, got = clone_bufs(start), clone_bufs(start)
            run_torch(st, want)
            run_kernels(st, got)
            torch.cuda.synchronize()
            bad["torch"] += [f"{st.rows}x{tokens} {k}" for k in mismatches(want, st.bufs)]
            bad["eager"] += [f"{st.rows}x{tokens} {k}" for k in mismatches(got, st.bufs)]
        out.append(result(f"replay {rep}: scatter, gather, every input rewritten", **bad))
    after = [clone_bufs(st.bufs) for st in steps]
    for st, start in zip(steps, before):
        for k, t in st.bufs.items():
            t.copy_(start[k])
    graph.replay()
    torch.cuda.synchronize()
    rerun = [
        f"{st.rows}x{tokens} {k}" for st, a in zip(steps, after) for k in mismatches(a, st.bufs)
    ]
    out.append(result("the last inputs replayed again", rerun=rerun))
    return [dict(family=f"R x {tokens}", **r) for r in out]


# ----------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_scatter(rows, tokens):
    bad = [r for r in measure_scatter(rows, tokens) if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("rows,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_gather(rows, tokens):
    bad = [r for r in measure_gather(rows, tokens) if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("tokens", TOKENS, ids=[f"Rx{t}" for t in TOKENS])
def test_graph_replay(tokens):
    bad = [r for r in measure_graph(tokens) if not r["ok"]]
    assert not bad, bad


POISON = 0x5EEDF00D  # what the never-written columns hold in the poisoned case


@pytest.mark.parametrize("poison", [True, False], ids=["poisoned", "never_written"])
def test_scatter_zeros_past_accepted_length(poison):
    """A mixed batch as the one-model worker's acceptance leaves its outputs (``new_tokens`` is ``torch.empty``
    [N, K + 1]; a context row writes its first token only and accepts 1, a generation row writes every column): the
    store holds each row's accepted tokens, then zeros, never a never-written column. ``poisoned``: those columns hold
    POISON. ``never_written``: they keep the allocation's contents, so compute-sanitizer initcheck reports any read of
    them. Row 0 is a context row whose chunk is not its last (row_begin 1 skips it)."""
    op = _op()
    g = torch.Generator().manual_seed(49)
    skipped, num_contexts, accepted = 1, 3, [1, 3, MAX_DRAFT + 1, 5]
    n = skipped + num_contexts + len(accepted)
    rows, row_begin, num_slots = n - skipped, skipped, 2 * MAX_BATCH
    width = MAX_DRAFT + 1
    contexts = skipped + num_contexts
    tokens = rand_i32((n, width), g, 0, 1 << 20)
    lens = torch.tensor([1] * contexts + accepted, dtype=torch.int32)
    new_tokens = torch.empty((n, width), dtype=torch.int32, device="cuda")
    if poison:
        new_tokens.fill_(POISON)
    new_tokens[:contexts, 0] = tokens[:contexts, 0].cuda()
    new_tokens[contexts:] = tokens[contexts:].cuda()
    outputs = {
        "new_tokens": new_tokens,
        "new_tokens_lens": lens.cuda(),
        "next_new_tokens": rand_i32((n, width), g).cuda(),
        "next_draft_tokens": rand_i32((n, MAX_DRAFT), g).cuda(),
    }
    slots = shuffled_distinct(rows, num_slots, g)
    slot_table = torch.tensor(slots + [0] * (num_slots - rows), dtype=torch.int32, device="cuda")
    stores = [
        torch.full(shape, -1, dtype=torch.int32, device="cuda")
        for shape in (
            (STORE_WIDTH, num_slots, 1),
            (STORE_WIDTH, num_slots, 1),
            (num_slots,),
            (num_slots, MAX_DRAFT),
        )
    ]
    assert op.SlotScatter().scatter(outputs, row_begin, rows, slot_table, *stores)
    store_new, store_lens = stores[0][:, :, 0].T.cpu(), stores[2].cpu()
    want = torch.full((num_slots, STORE_WIDTH), -1, dtype=torch.int32)
    for r, s in enumerate(slots):
        src = row_begin + r
        want[s] = 0
        want[s, : lens[src]] = tokens[src, : lens[src]]
        assert store_lens[s] == lens[src], (r, s)
    assert not (store_new == POISON).any(), "a never-written column reached the store"
    assert torch.equal(store_new, want), (store_new, want)


def test_contract():
    """rows == 0 launches nothing; every call the kernels do not cover returns False and launches nothing, so the
    caller keeps its torch path."""
    op = _op()
    assert op.is_supported()
    st = Step(MAX_BATCH, STORE_WIDTH, _seed(6, 0, 0, 0))
    b = clone_bufs(st.bufs)
    before = clone_bufs(b)
    scatter = list(st.scatter_args(b))
    gather = list(st.gather_args(b))
    assert op.SlotScatter().scatter(*scatter[:2], 0, *scatter[3:])
    assert op.StepInputGather().gather(*gather[:5], 0, *gather[6:])
    torch.cuda.synchronize()
    assert not mismatches(before, b), "rows == 0 wrote"
    # Argument 7 (draft width) == tokens, argument 6 (tokens) wider than the 8-wide store, argument 15 (KV begin)
    # ending past the engine's 8 KV-length offsets, and an int64 slot table (argument 3).
    for index, value in (
        (7, STORE_WIDTH),
        (6, STORE_WIDTH + 1),
        (15, 1),
        (3, b["prev_slots"].long()),
    ):
        args = list(gather)
        args[index] = value
        assert not op.StepInputGather().gather(*args), (
            f"gather argument {index} = {value!r} was launched"
        )
    # Outputs of 8 rows with rows 1 .. 8 asked, an int64 slot table, a non-contiguous output.
    assert not op.SlotScatter().scatter(scatter[0], 1, *scatter[2:])
    assert not op.SlotScatter().scatter(*scatter[:3], b["slot_table"].long(), *scatter[4:])
    strided = dict(
        scatter[0], next_draft_tokens=torch.empty_like(b["out_draft"]).t().contiguous().t()
    )
    assert not op.SlotScatter().scatter(strided, *scatter[1:])
    torch.cuda.synchronize()
    assert not mismatches(before, b), "a declined call wrote"


# ----------------------------------------------------------------------------------------------------------------
# Error table (python3 test_spec_step_copies.py report) and timing (python3 test_spec_step_copies.py time)
# ----------------------------------------------------------------------------------------------------------------


def report() -> int:
    print(f"{torch.cuda.get_device_name()}")
    ok_all = True
    sections = [
        ("SlotScatter.scatter", measure_scatter),
        ("StepInputGather.gather", measure_gather),
    ]
    for name, measure in sections:
        print(f"\n## {name}\n")
        print("| split | case | identical | result |")
        print("| :-- | :-- | :-- | :-- |")
        for rows, tokens in SPLITS:
            for r in measure(rows, tokens):
                ok_all &= r["ok"]
                print(f"| {rows}x{tokens} | {r['case']} | {r['identical']} | {'PASS' if r['ok'] else 'FAIL'} |",
                      flush=True)  # fmt: skip
    print("\n## CUDA graph replays (one capture per family R = 1 .. 8)\n")
    print("| family | case | identical | result |")
    print("| :-- | :-- | :-- | :-- |")
    for tokens in TOKENS:
        for r in measure_graph(tokens):
            ok_all &= r["ok"]
            print(f"| {r['family']} | {r['case']} | {r['identical']} | {'PASS' if r['ok'] else 'FAIL'} |",
                  flush=True)  # fmt: skip
    print("\nALL PASS" if ok_all else "\nFAIL")
    return 0 if ok_all else 1


def time_graph(body, calls, replays=15):
    """Per-call us of a CUDA graph of ``calls`` back-to-back ``body(i)``: median (min, max) over ``replays`` replays.
    ``body(-1)`` runs first, outside capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body(-1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls):
                body(i)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_call.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(per_call), min(per_call), max(per_call)


def timing() -> None:
    calls = 32
    print(f"{torch.cuda.get_device_name()}; graphs of {calls} back-to-back calls on one step's buffers (the engine's "
          "layout), 15 replays: median (min-max) us per call")  # fmt: skip
    print("| split | scatter: torch | scatter: kernel | gather: torch | gather: kernel |")
    print("| :-- | --: | --: | --: | --: |")
    for rows, tokens in SPLITS:
        st = Step(rows, tokens, _seed(5, rows, tokens, 0), row_begin=MAX_BATCH - rows, out_rows=MAX_BATCH,
                  begins=engine_begins(rows, tokens))  # fmt: skip
        b = st.bufs
        arms = [
            lambda i: torch_scatter(*st.scatter_args(b)),
            lambda i: kernel_scatter(st.scatter_args(b)),
            lambda i: torch_gather(*st.gather_args(b)),
            lambda i: kernel_gather(st.gather_args(b)),
        ]
        res = [[] for _ in arms]
        for rep in range(3):  # alternating order
            order = range(len(arms)) if rep % 2 == 0 else reversed(range(len(arms)))
            for a in order:
                res[a].append(time_graph(arms[a], calls))
        cells = []
        for timings in res:
            meds = sorted(x[0] for x in timings)
            lo, hi = min(x[1] for x in timings), max(x[2] for x in timings)
            cells.append(f"{meds[1]:.2f} ({lo:.2f}-{hi:.2f})")
        print(f"| {rows}x{tokens} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
