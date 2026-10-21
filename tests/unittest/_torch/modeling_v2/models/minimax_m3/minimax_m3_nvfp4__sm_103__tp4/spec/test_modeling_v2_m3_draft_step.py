# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_draft_step entry (one fused greedy draft pass: the BF16 draft head, streamed from its lossless
12-bit coding or as BF16 fragments, the cross-rank argmax and the draft loop's bookkeeping), and its tokens against the
in-tree drafter's head and greedy draft sampling.

The in-tree path: each rank's LM head shard GEMM (BF16 out), ``LogitsProcessor``'s float cast, then either the argmax
of the gathered logits (``torch.argmax``) or ``greedy_sample_draft_with_tp_gather``'s per-rank ``torch.max`` and the
pick among the ranks' (index, value) pairs. The edge cases use integer data, whose fp32 sums are exact in any order,
so the kernel and the in-tree path see the same BF16 logits and must agree bit for bit; every in-tree comparison runs
with the coded head (the default) and with the BF16 fragments, and on integer data the coded head's winning keys are
the BF16 fragments' bit for bit.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    m3_draft_step as draft_step_kernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_target_head import (  # noqa: E501
    encode_chunked,
    round_trip_ok,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_draft_step import (  # noqa: E501
    DRAFT_LEN,
    DRAFT_SHARD,
    HIDDEN,
    MAX_TOKENS,
    bf16_fragments,
    from_fragments,
    head_weight,
    shard_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_draft_step import (  # noqa: E501
    m3_draft_step as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    exchange_tokens,
    local_argmax,
    make_keys,
)
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402 — needs the path declared above
    WORLD,
    gen,
    head_workspace,
    num_sms,
    plant_peers,
    randn,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_draft_step requires a CUDA device"

T = DRAFT_LEN + 1
_LOW32 = 0xFFFFFFFF

# ------------------------------------------------------------------ the per-row rule against the in-tree token
#
# The kernel and the in-tree head (a cuBLAS BF16 GEMM) both round an FP32 sum of the same exact BF16 products to BF16,
# in different summation orders. A row's token can differ from the in-tree one only where some candidate's FP32 sum
# can land on either side of a BF16 rounding midpoint. Each row is held to:
#
#   kernel token == in-tree token, or the row is provably undecidable: with the fp64 reference logit L_i (exact BF16
#   products, fp64 sum) and the FP32-order bound E_i = gamma_K * sum_k |w_ik * x_k|, the BF16 value of any FP32 sum of
#   candidate i lies in [lo_i, hi_i] = [bf16(L_i - E_i), bf16(L_i + E_i)]; the kernel's token must win for some
#   choice of values within those intervals (larger value, then the lower global id), and a BF16 rounding midpoint
#   must lie within E of the kernel's or the in-tree token's L.
#
# gamma_K = K * 2^-22 is la-xh2's bound (4x the round-to-nearest unit roundoff 2^-24 per addition: any summation
# order, and a truncating tensor-core accumulator); K counts the row's non-zero products (adding an exact zero does
# not round). lo / hi take one more FP32 step outward before the BF16 rounding, which covers rounding L -+ E to FP32.
# Each undecidable row is reported as a warning with its midpoint distances, and each test reports its row count; any
# other mismatch fails.
GAMMA_UNIT = 2.0**-22


def _bf16_outward(v: torch.Tensor, direction: float) -> torch.Tensor:
    f = v.float()
    f = torch.nextafter(f, torch.full_like(f, direction))
    b = f.to(torch.bfloat16).float()
    return torch.where(b == 0, torch.zeros_like(b), b)


def bounds(h: torch.Tensor, w: torch.Tensor):
    """(fp64 logits L, bound E, lo, hi) [rows, N] of the head input rows h against the head rows w."""
    cols = (h != 0).any(dim=0)  # zero columns add exact zeros
    hd, wd = h[:, cols].double(), w[:, cols].double()
    logits = hd @ wd.t()
    bound = (h != 0).sum(dim=1, keepdim=True).double() * GAMMA_UNIT * (hd.abs() @ wd.abs().t())
    del wd
    inf = float("inf")
    return logits, bound, _bf16_outward(logits - bound, -inf), _bf16_outward(logits + bound, inf)


def midpoint_distance(v: float) -> float:
    """Distance of v to the nearest BF16 rounding midpoint."""
    b = torch.tensor(v, dtype=torch.float64).float().to(torch.bfloat16).float()
    if float(b) == 0.0:
        return abs(v)
    ulp = 2.0 ** (int(torch.frexp(b).exponent) - 8)
    return min(abs(v - (float(b) + ulp / 2)), abs(v - (float(b) - ulp / 2)))


def _winnable(tok: int, ids: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> bool:
    """Whether candidate ``tok`` wins the row for some values within the candidates' [lo, hi]."""
    k = int((ids == tok).nonzero()[0, 0])
    others = ids != tok
    beat = (hi[k] > lo[others]) | ((hi[k] == lo[others]) & (tok < ids[others]))
    return bool(beat.all())


def check_row(name: str, kernel_tok: int, stock_tok: int, ids, logits, bound, lo, hi) -> bool:
    """The per-row rule for one row over the candidates ``ids`` [C] (logits / bound / lo / hi [C]); True when the
    row is undecidable (reported as a warning), False when the tokens match; a mismatch outside the bound fails."""
    if kernel_tok == stock_tok:
        return False
    assert _winnable(kernel_tok, ids, lo, hi), (
        f"{name}: token {kernel_tok} differs from the in-tree {stock_tok} outside the FP32-order bound"
    )
    ik = int((ids == kernel_tok).nonzero()[0, 0])
    i_stock = int((ids == stock_tok).nonzero()[0, 0])
    lk, ls, ek, es = (
        float(logits[ik]),
        float(logits[i_stock]),
        float(bound[ik]),
        float(bound[i_stock]),
    )
    dk, ds = midpoint_distance(lk), midpoint_distance(ls)
    assert dk <= ek or ds <= es, f"{name}: neither logit is within its bound of a BF16 midpoint"
    warnings.warn(
        f"[draft-step undecidable] {name}: kernel {kernel_tok} (L {lk:.9g}, midpoint distance {dk:.3e}, bound "
        f"{ek:.3e}) vs in-tree {stock_tok} (L {ls:.9g}, midpoint distance {ds:.3e}, bound {es:.3e})"
    )
    return True


def report(name: str, rows: int, undecidable: int) -> None:
    warnings.warn(f"[draft-step rows] {name}: {rows} rows, {undecidable} undecidable")


def key_value(keys: torch.Tensor) -> torch.Tensor:
    """The fp32 value a key orders by (the inverse of the kernel's order-preserving image)."""
    u = (keys >> 32) & _LOW32
    bits = torch.where(u >= 0x80000000, u - 0x80000000, _LOW32 - u)
    bits = torch.where(bits >= 2**31, bits - 2**32, bits)  # as a signed 32-bit pattern
    return bits.to(torch.int32).view(torch.float32)


def key_index(keys: torch.Tensor) -> torch.Tensor:
    return _LOW32 - (keys & _LOW32)


def _bookkeeping(nreq: int, step: int, n_x_rows: int, g: torch.Generator):
    num_acc = (
        torch.randint(1, T + 1, (nreq,), generator=g, device="cuda", dtype=torch.int32)
        if step == 0
        else None
    )
    seq_lens = torch.full((nreq,), T, dtype=torch.int32, device="cuda") if step == 0 else None
    next_new = torch.full((nreq, T), -5, dtype=torch.int32, device="cuda")
    pos_src = torch.randint(100, 3000, (n_x_rows,), generator=g, device="cuda", dtype=torch.int32)
    kv_lens = torch.randint(200, 3000, (nreq,), generator=g, device="cuda", dtype=torch.int32)
    ids_next = torch.full((nreq,), -9, dtype=torch.int32, device="cuda")
    pos_next = torch.full((nreq,), -9, dtype=torch.int32, device="cuda")
    return num_acc, seq_lens, next_new, pos_src, kv_lens, ids_next, pos_next


def _head(w: torch.Tensor, coded: bool):
    """(wf, (exc_off, exc_pos, exc_w, exp_base)) of a BF16 shard: its 12-bit coding (checked to round-trip) or its
    BF16 fragments."""
    if not coded:
        return bf16_fragments(w), (None, None, None, 0)
    packed, base, exc_off, exc_pos, exc_w = encode_chunked(w)
    assert round_trip_ok(packed, base, exc_off, exc_pos, exc_w, w)
    return packed, (exc_off, exc_pos, exc_w.view(-1), base)


def _args(
    x,
    x_rows,
    nreq,
    wf,
    rank,
    world,
    step,
    g,
    uc=None,
    keys=None,
    counters=None,
    num_acc=None,
    coded=(None, None, None, 0),
):
    """The op's argument tuple for one rank (fresh workspace and bookkeeping buffers unless given); ``coded`` = the
    coded head's (exc_off, exc_pos, exc_w, exp_base)."""
    acc, seq_lens, next_new, pos_src, kv_lens, ids_next, pos_next = _bookkeeping(
        nreq, step, x.shape[0], g
    )
    if num_acc is None:
        num_acc = acc
    if uc is None:
        uc, keys, counters = head_workspace(world, slot_rows=MAX_TOKENS)
    out_keys = torch.zeros(nreq, dtype=torch.int64, device="cuda")
    tiles = wf.shape[0] if coded[0] is None else coded[0].numel() - 1
    return (x, x_rows, nreq, wf, rank * tiles * 16, keys, counters, uc, uc.data_ptr(), rank, world,
            next_new[:, 1:], step, ids_next, pos_src, pos_next, kv_lens, num_acc, seq_lens, DRAFT_LEN, num_sms(),
            True, out_keys, *coded)  # fmt: skip


def _case(spec, seed):
    g = gen(seed)
    nreq, step, rank, world, vs, k = (
        spec["nreq"],
        spec["step"],
        spec["rank"],
        spec["world"],
        spec["vs"],
        spec["k"],
    )
    pass0 = step == 0
    x = randn((nreq * T if pass0 else nreq, k), g)
    wf, coded = _head(randn((vs, k), g, 0.02), spec["coded"])
    num_acc = (
        torch.randint(1, T + 1, (nreq,), generator=g, device="cuda", dtype=torch.int32)
        if pass0
        else None
    )
    x_rows = (torch.arange(nreq, device="cuda") * T + num_acc.long() - 1) if pass0 else None
    uc, keys, counters = head_workspace(world, slot_rows=MAX_TOKENS)
    h = x[x_rows] if pass0 else x
    own = shard_logits(h, wf, *coded).max(dim=1).values
    plant_peers(
        uc, counters, own, rank, world, vs, g, slot_rows=MAX_TOKENS, ties=spec.get("ties", False)
    )
    return _args(x, x_rows, nreq, wf, rank, world, step, g, uc, keys, counters, num_acc, coded)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The bookkeeping is exact on every row (next positions, KV and sequence lengths). Each row's token is the
    in-tree one (this rank's BF16 shard logits from the head's GEMM, their first maximum, against the peers' keys) or
    the row is provably undecidable (check_row); the winning key's index is the token, and its value lies in the
    token's [lo, hi] (or is the peer's pushed key when a peer wins)."""
    args = _case(cell.spec, seed_of(cell.why))
    x, x_rows, nreq, wf, offset, _, counters, uc = args[:8]
    rank, world = args[9], args[10]
    half = int(counters[1].item()) & 1
    pushed = (
        uc[half * MAX_TOKENS * world : (half * MAX_TOKENS + nreq) * world].view(nreq, world).clone()
    )
    h = x[x_rows] if x_rows is not None else x[:nreq]
    w = head_weight(wf, x.shape[1], *args[23:27])
    best, idx = local_argmax(torch.nn.functional.linear(h, w).float(), offset)
    stock = exchange_tokens(make_keys(best, idx), uc, counters, rank, world, MAX_TOKENS)
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    assert args[6].tolist() == [0, 1] and not bool(args[5].any()), "counters / keys not re-armed"
    step = args[12]
    others = [c for c in range(args[11].shape[1]) if c != step]
    assert torch.equal(args[11][:, others], ref["draft_out"][:, others])
    assert torch.equal(args[11][:, step], args[13]), "draft_out and ids_next disagree"
    assert torch.equal(args[15], ref["pos_next"])
    assert torch.equal(args[16], ref["kv_lens"])
    if args[18] is not None:
        assert torch.equal(args[18], ref["seq_lens"])
    # Candidates per row: this rank's head rows (fp64 bounds) and each peer's pushed key (an exact value).
    logits, bound, lo, hi = bounds(h, w)
    own_ids = offset + torch.arange(w.shape[0], device="cuda")
    peers = [p for p in range(world) if p != rank]
    undecidable = 0
    for r in range(nreq):
        pv = key_value(pushed[r, peers]).double()
        ids = torch.cat([own_ids, key_index(pushed[r, peers])])
        undecidable += check_row(
            f"{cell.why[:40]} row {r}",
            int(args[13][r]),
            int(stock[r]),
            ids,
            torch.cat([logits[r], pv]),
            torch.cat([bound[r], torch.zeros_like(pv)]),
            torch.cat([lo[r], pv.float()]),
            torch.cat([hi[r], pv.float()]),
        )
    report(cell.why[:40], nreq, undecidable)
    out_keys = args[22]
    assert torch.equal(key_index(out_keys), args[13].long()), "the winning key's index is the token"
    got = key_value(out_keys)
    for r in range(nreq):
        c = int(args[13][r]) - offset
        if 0 <= c < w.shape[0]:
            assert float(lo[r, c]) <= float(got[r]) <= float(hi[r, c]), (
                f"row {r}: winning value outside [lo, hi]"
            )
        else:
            assert int(out_keys[r]) in pushed[r].tolist(), (
                f"row {r}: a peer won with a key it did not push"
            )


def test_fragments_round_trip() -> None:
    g = gen(seed_of("fragments"))
    w = randn((64, 96), g)
    wf = bf16_fragments(w)
    assert wf.shape == (4, 6, 256)
    assert torch.equal(from_fragments(wf), w)
    head = SimpleNamespace(weight=w, bias=None)
    # K 96: never coded, the BF16 fragments.
    prepared = draft_step_kernels.prepare_head(head, 3, True)
    assert not prepared.coded
    assert torch.equal(prepared.wf, wf) and prepared.vocab_offset == 3 * 64 and prepared.weight is w


# ------------------------------------------------------------------ against the in-tree drafter, bit for bit


def _stock_tokens(shards: list, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The in-tree greedy draft tokens of the rows x over the vocab-sharded BF16 head: (argmax of the gathered float
    logits, greedy_sample_draft_with_tp_gather's per-rank (index, value) pick)."""
    logits = [torch.nn.functional.linear(x, w).float() for w in shards]
    gathered = torch.argmax(torch.cat(logits, dim=-1), dim=-1).to(torch.int32)
    combined = []
    for rank, shard_logit in enumerate(logits):
        worker = SimpleNamespace(mapping=SimpleNamespace(tp_rank=rank))
        combined.append(Eagle3OneModelWorker._get_local_max_and_combined(worker, shard_logit))
    pick = Eagle3OneModelWorker._get_draft_tokens_from_gathered
    pick = getattr(pick, "_torchdynamo_orig_callable", pick)
    tp_gather = pick(None, torch.cat(combined, dim=-1)).to(torch.int32)
    return gathered, tp_gather


def _fused_tokens(shards: list, x: torch.Tensor, coded: bool, step: int = 1) -> list:
    """Each rank's fused draft tokens: every rank's own key from a one-rank run, then each rank's four-rank pass with
    its peers' keys in its slots, as their multicast pushes would put them."""
    g = gen(seed_of("fused tokens"))
    nreq = x.shape[0]
    heads = [_head(w, coded) for w in shards]
    own_keys = []
    for rank, (wf, cd) in enumerate(heads):
        args = list(_args(x, None, nreq, wf, rank, 1, step, g, coded=cd))
        args[9] = 0  # one-rank run: this rank's slot only, global ids of `rank`
        op(*args)
        own_keys.append(args[22].clone())
    tokens = []
    for rank, (wf, cd) in enumerate(heads):
        uc, keys, counters = head_workspace(WORLD, slot_rows=MAX_TOKENS)
        for peer in range(WORLD):
            if peer != rank:
                for t in range(nreq):
                    uc[t * WORLD + peer] = own_keys[peer][t]
        args = _args(x, None, nreq, wf, rank, WORLD, step, g, uc, keys, counters, coded=cd)
        with validating(op):
            op(*args)
        torch.cuda.synchronize()
        assert torch.equal(args[11][:, step], args[13]), "draft_out and ids_next disagree"
        tokens.append(args[13].clone())
    return tokens


def _integer_head(g: torch.Generator, rows: int = WORLD * DRAFT_SHARD) -> torch.Tensor:
    """A [rows, 6144] head of small integers over the first 4 columns (zero elsewhere): its logits against integer
    rows are exact integers in any summation order, many of which share a BF16 value."""
    w = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    w[:, :4] = torch.randint(-60, 61, (rows, 4), generator=g, device="cuda").to(torch.bfloat16)
    return w


def _shards(w: torch.Tensor) -> list:
    return [w[r * DRAFT_SHARD : (r + 1) * DRAFT_SHARD].contiguous() for r in range(WORLD)]


def _assert_stock(w: torch.Tensor, x: torch.Tensor, coded: bool, expect=None) -> None:
    shards = _shards(w)
    gathered, tp_gather = _stock_tokens(shards, x)
    assert torch.equal(gathered, tp_gather), "the two in-tree pickers disagree"
    if expect is not None:
        assert gathered.tolist() == expect, (gathered.tolist(), expect)
    for rank, tokens in enumerate(_fused_tokens(shards, x, coded)):
        assert torch.equal(tokens, gathered), (
            f"rank {rank}: {tokens.tolist()} vs stock {gathered.tolist()}"
        )


def _ones_rows(nreq: int) -> torch.Tensor:
    x = torch.zeros((nreq, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x[:, :4] = 1.0
    return x


@pytest.mark.parametrize("coded", [True, False], ids=["coded", "bf16"])
def test_forced_bf16_ties_take_the_lowest_global_id(coded) -> None:
    """A head row copied onto every rank that wins every row of ones (logit 244, every other at most 240): the lowest
    global id must win, inside a shard and across ranks. Then integer rows in [-2, 2], whose logits (up to 480 in
    magnitude, BF16 spacing 2 above 256) collide often, exactly or after BF16 rounding."""
    g = gen(seed_of("forced ties"))
    w = _integer_head(g)
    win = torch.tensor([61.0, 61.0, 61.0, 61.0], device="cuda").to(torch.bfloat16)
    for gid in (3 * DRAFT_SHARD + 5, DRAFT_SHARD + 77, DRAFT_SHARD + 9000, 2 * DRAFT_SHARD + 1):
        w[gid, :4] = win
    _assert_stock(w, _ones_rows(5), coded, expect=[DRAFT_SHARD + 77] * 5)
    # Random integer rows: ties after BF16 rounding happen naturally.
    x = torch.randint(-2, 3, (8, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
    x[:, 4:] = 0
    _assert_stock(w, x, coded)


@pytest.mark.parametrize("coded", [True, False], ids=["coded", "bf16"])
def test_signed_zero_maxima_take_the_lowest_global_id(coded) -> None:
    """Rows whose maximum is zero: an all-zero head input (every logit +-0) and logits <= 0 with zeros by cancellation
    on several ranks. -0 and +0 compare equal, so the lowest global id reaching zero wins."""
    w = torch.zeros((WORLD * DRAFT_SHARD, HIDDEN), dtype=torch.bfloat16, device="cuda")
    w[:, 0] = -1.0
    w[:, 1] = -1.0
    for gid in (2 * DRAFT_SHARD + 5, DRAFT_SHARD + 40000, 3 * DRAFT_SHARD + 7, 49999):
        w[gid, 0] = 1.0
        w[gid, 1] = -1.0
    x = torch.zeros((3, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x[1, :2] = 1.0
    x[2, :2] = 2.0
    x[2, 2] = -1.0  # w[:, 2] is zero: a -0 product on every row
    _assert_stock(w, x, coded, expect=[0, 49999, 49999])


@pytest.mark.parametrize(
    "lower,higher,winner",
    [
        (1000, 1001, "lower"),  # 1001 rounds to 1000 in BF16: a tie, the lower id wins
        (1001, 1002, "lower"),  # 1001 and the tie-to-even midpoint 1002 both round to 1000
        (1000, 1004, "higher"),  # adjacent BF16 values: the larger wins
        (1003, 1000, "lower"),  # 1003 rounds up to 1004 > 1000
    ],
    ids=["one_fp32_step", "midpoint", "adjacent_bf16", "rounds_up"],
)
@pytest.mark.parametrize("where", ["in_shard", "across_ranks"])
@pytest.mark.parametrize("coded", [True, False], ids=["coded", "bf16"])
def test_near_equal_top2_follow_the_bf16_logits(lower, higher, winner, where, coded) -> None:
    """Top-2 logits one to four fp32 steps apart: the winner is the one the BF16 logits decide (the in-tree logits'
    dtype), not the fp32 sum. The pair sits inside rank 3's shard or on ranks 1 and 2; every other logit is at most
    240 in magnitude."""
    g = gen(seed_of(f"near top2 {lower} {higher} {where}"))
    w = _integer_head(g)
    if where == "in_shard":
        lo_id, hi_id = 3 * DRAFT_SHARD + 100, 3 * DRAFT_SHARD + 200
    else:
        lo_id, hi_id = DRAFT_SHARD + 49000, 2 * DRAFT_SHARD + 3
    for gid, value in ((lo_id, lower), (hi_id, higher)):
        q, r = divmod(value, 4)  # four integers <= 251, exact in BF16
        w[gid, :4] = torch.tensor([q + (i < r) for i in range(4)], device="cuda").to(torch.bfloat16)
    expect = lo_id if winner == "lower" else hi_id
    _assert_stock(w, _ones_rows(2), coded, expect=[expect, expect])


# ------------------------------------------------------------------ non-integer rows at a BF16 rounding midpoint

#: A BF16 rounding midpoint (between 6.0 and 6.03125; a tie there rounds to the even 6.0) and the FP32 spacing there.
MIDPOINT = 6.0 + 2.0**-6
FP32_ULP = 2.0**-21


def _near_midpoint_head(g: torch.Generator, targets: dict):
    """A full-vocabulary head [4 * 50016, 6144] and one head-input row x with 65 non-zero entries: 1, 2^-12, 2^-20
    and 2^-28 on columns 0..3 and 61 random non-integer BF16 values. Every head row carries random non-integer BF16
    values on those 61 columns (logits about 0.2 in magnitude); each row of ``targets`` (global id -> fp64 value) also
    gets columns 0..3 set so its fp64 logit is the target to within 2^-30 (a coarse value, then three finer
    corrections, each BF16-rounded)."""
    x = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
    scales = (1.0, 2.0**-12, 2.0**-20, 2.0**-28)
    for col, scale in enumerate(scales):
        x[0, col] = scale
    x[0, 4:65] = randn((61,), g, 0.5)
    w = torch.zeros((WORLD * DRAFT_SHARD, HIDDEN), dtype=torch.bfloat16, device="cuda")
    w[:, 4:65] = randn((WORLD * DRAFT_SHARD, 61), g, 0.05)
    xd = x[0].double()
    for gid, target in targets.items():
        rest = float(target - (w[gid].double() * xd).sum())
        for col, scale in enumerate(scales):
            v = torch.tensor(rest / scale, dtype=torch.float64).to(torch.bfloat16)
            w[gid, col] = v
            rest -= float(v) * scale
        assert abs(rest) <= 2.0**-30, (gid, rest)
    return w, x


def _check_fused_against_stock(
    name: str, w: torch.Tensor, x: torch.Tensor, coded: bool, expect=None
) -> int:
    """Each rank's fused tokens against the in-tree tokens under the per-row rule (full-vocabulary candidates);
    ``expect`` (a token per row) additionally pins rows the bound decides. Returns the undecidable rows."""
    shards = _shards(w)
    gathered, tp_gather = _stock_tokens(shards, x)
    assert torch.equal(gathered, tp_gather), "the two in-tree pickers disagree"
    fused = _fused_tokens(shards, x, coded)
    for tokens in fused[1:]:
        assert torch.equal(tokens, fused[0]), "the ranks disagree on the global token"
    logits, bound, lo, hi = bounds(x, w)
    ids = torch.arange(w.shape[0], device="cuda")
    undecidable = 0
    for r in range(x.shape[0]):
        undecidable += check_row(
            f"{name} row {r}",
            int(fused[0][r]),
            int(gathered[r]),
            ids,
            logits[r],
            bound[r],
            lo[r],
            hi[r],
        )
        if expect is not None:
            assert int(gathered[r]) == expect[r] and int(fused[0][r]) == expect[r], (
                name,
                int(gathered[r]),
                int(fused[0][r]),
                expect[r],
            )
            others = ids != expect[r]
            assert _winnable(expect[r], ids, lo[r], hi[r]), (
                f"{name}: the bound does not let {expect[r]} win"
            )
            rivals = [int(i) for i in ids[others][hi[r][others] >= lo[r][expect[r]]]]
            assert not any(_winnable(i, ids, lo[r], hi[r]) for i in rivals), (
                f"{name}: the row is not decided"
            )
    report(name, x.shape[0], undecidable)
    return undecidable


@pytest.mark.parametrize(
    "lower,higher,winner",
    [
        (
            -2,
            2,
            None,
        ),  # straddling the midpoint by 2 FP32 ulps: either side is in-tree up to the order
        (2, -2, None),
        (0, -2, None),  # exactly on the midpoint (ties to the even 6.0) against just below it
        (
            -2048,
            2048,
            "higher",
        ),  # ~1e-3 either side, well outside the bound: decided by BF16 rounding
        (
            -4096,
            -2048,
            "lower",
        ),  # both below the midpoint, ~1e-3 apart: a BF16 tie the lower id wins
    ],
    ids=["straddle", "straddle_reversed", "on_midpoint", "decided_up", "decided_tie"],
)
@pytest.mark.parametrize("where", ["in_shard", "across_ranks"])
@pytest.mark.parametrize("coded", [True, False], ids=["coded", "bf16"])
def test_top2_at_a_bf16_midpoint(lower, higher, winner, where, coded) -> None:
    """Non-integer rows whose top-2 fp64 logits sit the given number of FP32 ulps from a BF16 rounding midpoint, inside
    rank 3's shard or on ranks 1 and 2, every other logit far below. The per-row rule holds on every row; outside the
    bound the token is the one the BF16 rounding decides, bit for bit."""
    g = gen(seed_of(f"midpoint {lower} {higher} {where}"))
    if where == "in_shard":
        lo_id, hi_id = 3 * DRAFT_SHARD + 100, 3 * DRAFT_SHARD + 200
    else:
        lo_id, hi_id = DRAFT_SHARD + 49000, 2 * DRAFT_SHARD + 3
    w, x = _near_midpoint_head(
        g, {lo_id: MIDPOINT + lower * FP32_ULP, hi_id: MIDPOINT + higher * FP32_ULP}
    )
    expect = None if winner is None else [lo_id if winner == "lower" else hi_id]
    _check_fused_against_stock(f"midpoint {where} {lower:+d}/{higher:+d} ulp", w, x, coded, expect)


# ------------------------------------------------------------------ the coded head against the BF16 fragments


@pytest.mark.parametrize("nreq", [1, 2, 8])
def test_coded_head_is_the_bf16_head_bit_for_bit(nreq) -> None:
    """Integer data (fp32 sums exact in any order) with exceptions on both sides of the exponent window: the coded
    head's winning keys (value and global id) and its bookkeeping are the BF16 fragments' bit for bit, on one rank and
    across the ranks."""
    g = gen(seed_of(f"coded vs bf16 {nreq}"))
    sign = torch.randint(0, 2, (DRAFT_SHARD, HIDDEN), generator=g, device="cuda") * 2 - 1
    w = (torch.randint(1, 41, (DRAFT_SHARD, HIDDEN), generator=g, device="cuda") * sign).to(
        torch.bfloat16
    )
    # Exceptions on both sides of the window of 1..40: exact zeros, and 2^15 (fp32 sums stay below 2^24).
    w[::5, ::512] = 0.0
    w[::13, 7::600] = 32768.0
    x = torch.randint(-2, 3, (nreq, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
    wf_c, cd = _head(w, True)
    assert cd[1].numel() > 0, "the shard must carry exceptions"
    wf_b, cb = _head(w, False)
    outs = []
    for wf, c in ((wf_c, cd), (wf_b, cb)):
        args = _args(x, None, nreq, wf, 2, 1, 1, gen(seed_of("bookkeeping")), coded=c)
        args = list(args)
        args[9] = 0
        with validating(op):
            op(*args)
        torch.cuda.synchronize()
        outs.append(args)
    coded_args, bf16_args = outs
    for i in (11, 13, 15, 16, 22):  # draft_out, ids_next, pos_next, kv_lens, out_keys
        assert torch.equal(coded_args[i], bf16_args[i]), i
    best, idx = local_argmax(torch.nn.functional.linear(x, w).float(), 2 * DRAFT_SHARD)
    assert torch.equal(coded_args[22], make_keys(best, idx)), (
        "the winning keys are not the in-tree ones"
    )


@pytest.mark.parametrize("nreq", [1, 2, 8])
def test_tiny_exception_products_are_exact(nreq) -> None:
    """Weights of 1 (exponent window base 113) with column 0 zero except 2^-30 / 2^-29 on rows 0 / 1, and a one-hot
    activation on column 0: the exceptions' exact products are the only non-zero logits, so the coded head must pick
    row 1 with the BF16 head's key bit for bit (a correction against a non-zero placeholder cancels them to 0)."""
    n, k = 256, 1024
    w = torch.ones((n, k), dtype=torch.bfloat16, device="cuda")
    w[:, 0] = 0.0
    w[0, 0] = 2.0**-30
    w[1, 0] = 2.0**-29
    x = torch.zeros((nreq, k), dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1.0
    wf_c, cd = _head(w, True)
    assert cd[3] == 113
    keys = []
    for wf, c in ((wf_c, cd), _head(w, False)):
        args = _args(x, None, nreq, wf, 0, 1, 1, gen(seed_of("tiny exceptions")), coded=c)
        with validating(op):
            op(*args)
        torch.cuda.synchronize()
        assert args[13].tolist() == [1] * nreq
        keys.append(args[22])
    best, idx = local_argmax(torch.nn.functional.linear(x, w).float(), 0)
    assert torch.equal(keys[0], keys[1]) and torch.equal(keys[0], make_keys(best, idx))


def test_prepare_head_codes_the_shard_and_fails_closed(monkeypatch) -> None:
    """prepare_head keeps the coded shard when it decodes to the weight bit for bit; a coding that does not is
    dropped for the BF16 fragments."""
    g = gen(seed_of("prepare coded"))
    w = randn((4096, 1024), g, 0.02)
    head = SimpleNamespace(weight=w, bias=None)
    prepared = draft_step_kernels.prepare_head(head, 2, True)
    assert prepared.coded and prepared.vocab_offset == 2 * 4096 and prepared.weight is w
    assert torch.equal(head_weight(prepared.wf, 1024, prepared.exc_off, prepared.exc_pos, prepared.exc_w,
                                   prepared.exp_base), w)  # fmt: skip

    def corrupt(weight, *args, **kwargs):
        packed, base, exc_off, exc_pos, exc_w = encode_chunked(weight)
        packed = packed.clone()
        packed[11] ^= 0x10
        return packed, base, exc_off, exc_pos, exc_w

    monkeypatch.setattr(draft_step_kernels, "encode_chunked", corrupt)
    dropped = draft_step_kernels.prepare_head(head, 2, True)
    assert not dropped.coded and dropped.exc_off is None
    assert torch.equal(dropped.wf, bf16_fragments(w))


def test_prepare_head_keeps_bf16_when_coding_is_off(monkeypatch) -> None:
    """``coded`` False (``DRAFT_HEAD_CODED`` off) keeps the BF16 fragments even for a shard whose coding would decode
    bit for bit."""
    g = gen(seed_of("prepare bf16"))
    w = randn((4096, 1024), g, 0.02)
    prepared = draft_step_kernels.prepare_head(SimpleNamespace(weight=w, bias=None), 2, False)
    assert not prepared.coded and prepared.exc_off is None
    assert torch.equal(prepared.wf, bf16_fragments(w))


def test_guard_refuses_head_rows_past_x() -> None:
    args = list(_case(op.CELLS[0].spec, 1))
    args[1] = args[1] + 1000
    with pytest.raises(AssertionError, match="x_rows must index x"):
        with validating(op):
            op(*args)


def test_guard_refuses_too_few_ctas() -> None:
    args = list(_case(op.CELLS[1].spec, 2))
    args[20] = 8
    with pytest.raises(AssertionError, match="tiles per CTA"):
        with validating(op):
            op(*args)
