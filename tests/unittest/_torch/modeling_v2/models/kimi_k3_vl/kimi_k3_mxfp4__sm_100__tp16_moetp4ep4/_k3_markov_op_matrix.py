# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``spec/k3_markov`` entry of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``: the
DSpark vanilla-Markov draft chain over a vocab-sharded draft head, on W ranks of one NVLink domain (default 4).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_markov_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _k3_markov_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job (the cells share the op's workspaces). The
collected entry point is ``test_modeling_v2_k3_markov_op_matrix.py``.

Every rank draws every rank's inputs from one seed per cell, so each rank holds the whole reference. Each cell runs
the op under ``validating`` on this rank's shard and compares its corrected logits (fp32 bits), tokens, next_new and
the rewound kv_lens bit for bit with the reference's, and tokens, next_new and kv_lens bit for bit across the ranks.
The TP16 slice (S = 10240) pushes every exchange entry into 16 / W slots, the exchange volume of 16 ranks.
"""

import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_markov requires CUDA devices"

DEADLINE_S = 1200
VOCAB = 163840  # the draft vocabulary: rows of markov_w1 and markov_w2
MARKOV_RANK = 256
TP16_SHARD = VOCAB // 16
# Local rows tied on every rank (with S / 2 + 3 and S - 5): 3 and 5 share an 8-row block, 11 is the warp's next block,
# then the next warps and CTAs.
TIE_ROWS = (3, 5, 11, 19, 67, 131, 323)
TIE_VALUE = 40.0

R = None
validating = None
OP = None
WORKSPACE = None  # markov_workspace of the kernel's op module
W1 = None
W2 = None


def _shard(kind: str):
    """(S, push copies) of a cell's shard kind."""
    if kind == "tp16":
        return TP16_SHARD, max(1, 16 // R.world)
    return VOCAB // R.world, 1


def _gen(*key) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(repr(key).encode()))


def _inputs(cell):
    """Every rank's (base, markov_w2 shard), the replicated arguments, the expected tokens of a tie cell, and the
    workspace copies, all the same on every rank."""
    spec = cell.spec
    batch, block, contexts = spec["batch"], spec["block"], spec["contexts"]
    shard, copies = _shard(spec["shard"])
    shared = _gen(cell.why, "shared")
    bases = [
        torch.randn(batch, block, shard, generator=_gen(cell.why, "base", r), device="cuda") * 3.0
        for r in range(R.world)
    ]
    w2 = W2
    expect = None
    if spec.get("ties"):
        rows = [r for r in TIE_ROWS if r < shard] + [shard // 2 + 3, shard - 5]
        w2 = W2.clone()
        for r in range(R.world):
            w2[[r * shard + row for row in rows]] = W2[
                11
            ]  # one markov_w2 row, so one bias, at every tie row
        expect = torch.empty(batch, block, dtype=torch.int32)
        for b in range(batch):
            for k in range(block):
                # Every rank from the lead one holds the tie, minus the lead's first `drop` rows: the winner (the
                # lead's first kept row) moves over the levels of the reduction and over the ranks.
                lead, drop = (b + k) % R.world, (b + 2 * k) % (len(rows) - 1)
                expect[b, k] = lead * shard + rows[drop]
                for r in range(lead, R.world):
                    bases[r][b, k, rows[drop:] if r == lead else rows] = TIE_VALUE
        expect = expect.cuda()
    bases = [b.to(spec["dtype"]) for b in bases]
    shards = [w2[r * shard : (r + 1) * shard] for r in range(R.world)]
    first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
    if spec.get("anchors"):
        first[0], first[-1] = -1, VOCAB
    requests = contexts + batch
    accepted = torch.randint(
        0, VOCAB, (requests + 1, block + 1), generator=shared, device="cuda", dtype=torch.int32
    )
    num_accepted = torch.randint(
        1, block + 2, (requests,), generator=shared, device="cuda", dtype=torch.int32
    )
    accepted_rows = torch.randperm(requests + 1, generator=shared, device="cuda")[:requests].to(
        torch.int32
    )
    rewind = None
    if spec["rewind"]:
        kv_lens = torch.randint(
            0, 50, (requests + 1,), generator=shared, device="cuda", dtype=torch.int32
        )
        amounts = torch.randint(-6, 9, (batch,), generator=shared, device="cuda", dtype=torch.int32)
        amounts[0] = kv_lens[contexts] + 5  # past the length: clamped to 0
        if batch > 1:
            amounts[1] = -3  # negative: the length grows
        if batch > 2:
            amounts[2] = kv_lens[contexts + 2]  # exactly to 0
        rewind = (kv_lens, amounts, contexts)
    replicated = dict(
        first_prev=first,
        markov_w1=W1,
        accepted=accepted,
        # The worker's [num_contexts:] slices of the step's per-request tensors.
        num_accepted=num_accepted[contexts:],
        accepted_rows=accepted_rows[contexts:],
    )
    return bases, shards, shard, copies, replicated, rewind, expect


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def check_cells() -> None:
    """Every cell of the entry, op against reference bit for bit on every rank, the same tokens, next_new and kv_lens
    on every rank."""
    for cell in OP.CELLS:
        bases, shards, shard, copies, rep, rewind, expect = _inputs(cell)
        ws = WORKSPACE(R.mapping, copies)
        kv_op = kv_ref = None
        amounts, first_row = None, 0
        if rewind is not None:
            kv_lens, amounts, first_row = rewind
            kv_op, kv_ref = kv_lens.clone(), kv_lens.clone()
        with validating(OP):
            corrected, tokens, next_new = OP(
                bases[R.rank],
                rep["first_prev"],
                rep["markov_w1"],
                shards[R.rank],
                R.rank * shard,
                rep["accepted"],
                rep["num_accepted"],
                rep["accepted_rows"],
                ws,
                kv_lens=kv_op,
                rewind=amounts,
                rewind_first=first_row,
            )
        ref_corrected, ref_tokens, ref_next_new = OP.reference(
            bases,
            rep["first_prev"],
            rep["markov_w1"],
            shards,
            rep["accepted"],
            rep["num_accepted"],
            rep["accepted_rows"],
            kv_lens=kv_ref,
            rewind=amounts,
            rewind_first=first_row,
        )
        torch.cuda.synchronize()
        want = ref_corrected[R.rank]
        assert corrected.shape == want.shape and corrected.dtype == torch.float32, cell.why
        differ = int((_bits(corrected) != _bits(want)).sum())
        assert differ == 0, f"{cell.why}: {differ} corrected words differ on rank {R.rank}"
        assert torch.equal(tokens, ref_tokens), f"{cell.why}: tokens differ on rank {R.rank}"
        assert torch.equal(next_new, ref_next_new), f"{cell.why}: next_new differs on rank {R.rank}"
        if expect is not None:
            assert torch.equal(tokens, expect), f"{cell.why}: a tie did not go to the lowest index"
        shown = [tokens, next_new]
        if kv_op is not None:
            assert torch.equal(kv_op, kv_ref), f"{cell.why}: kv_lens differ on rank {R.rank}"
            shown.append(kv_op)
        assert R.same_on_ranks(*shown), f"{cell.why}: the ranks disagree"


def check_is_valid_rejects() -> None:
    """Under ``validating`` an accepted row outside accepted, a num_accepted past its width and a shard past the
    vocabulary raise ValueError on every rank before any launch (so no rank waits on a peer); the next call is
    correct."""
    cell = OP.CELLS[0]
    bases, shards, shard, copies, rep, _, _ = _inputs(cell)
    ws = WORKSPACE(R.mapping, copies)
    rows_out = rep["accepted_rows"].clone()
    rows_out[0] = rep["accepted"].shape[0]
    counts_out = rep["num_accepted"].clone()
    counts_out[0] = rep["accepted"].shape[1] + 1
    bad = [
        ("accepted row past accepted", dict(accepted_rows=rows_out), R.rank * shard),
        ("num_accepted past accepted's width", dict(num_accepted=counts_out), R.rank * shard),
        ("shard past the vocabulary", {}, VOCAB - shard + 1),
    ]
    for name, override, offset in bad:
        args = dict(rep, **override)
        raised = False
        try:
            with validating(OP):
                OP(bases[R.rank], args["first_prev"], args["markov_w1"], shards[R.rank], offset,
                   args["accepted"], args["num_accepted"], args["accepted_rows"], ws)  # fmt: skip
        except ValueError:
            raised = True
        assert R.all_true(raised), f"{name}: is_valid did not raise on every rank"
    _one_call_is_correct(cell)


def _one_call_is_correct(cell) -> None:
    bases, shards, shard, copies, rep, _, _ = _inputs(cell)
    ws = WORKSPACE(R.mapping, copies)
    _, tokens, next_new = OP(bases[R.rank], rep["first_prev"], rep["markov_w1"], shards[R.rank], R.rank * shard,
                             rep["accepted"], rep["num_accepted"], rep["accepted_rows"], ws)  # fmt: skip
    _, ref_tokens, ref_next_new = OP.reference(bases, rep["first_prev"], rep["markov_w1"], shards, rep["accepted"],
                                               rep["num_accepted"], rep["accepted_rows"])  # fmt: skip
    assert torch.equal(tokens, ref_tokens) and torch.equal(next_new, ref_next_new), (
        "the call after the rejected ones is wrong"
    )


CHECKS = [check_cells, check_is_valid_rejects]


def _run_one_rank(args) -> int:
    global R, OP, WORKSPACE, W1, W2, validating
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_markov.op import (  # noqa: E501
        markov_workspace,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.spec.k3_markov import (  # noqa: E501
        k3_markov,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from _validating import validating as validating_context

    validating, OP, WORKSPACE = validating_context, k3_markov, markov_workspace
    # The markov weights, the same on every rank and shared by every cell.
    weights = torch.Generator(device="cuda").manual_seed(7)
    W1 = (torch.randn(VOCAB, MARKOV_RANK, generator=weights, device="cuda") * 0.25).bfloat16()
    W2 = (torch.randn(VOCAB, MARKOV_RANK, generator=weights, device="cuda") * 0.25).bfloat16()
    with torch.inference_mode():
        return ls.run_checks(R, CHECKS)


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
