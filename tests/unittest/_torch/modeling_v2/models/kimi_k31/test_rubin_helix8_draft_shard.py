# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter takes the selector's candidates from its vocab shard.

The drafter head's logits stay this rank's ``[rows, 20480]`` shard. ``_dflash2_global_top_k``'s
sharded branch takes the shard's top-32, puts the 32 ``(id, value)`` pairs in id order
(``_pairs_in_id_order``), all-gathers them and merges, where the drafter used to all-gather the
full ``[rows, 163840]`` logits, cast them to FP32 and take their top-32. The candidates must be
the full-vocab top-32's ids, values and order bit for bit; these tests hold it to that.

CPU (``cpu_only``), on modeling.py's own functions (read from the source, run over stand-ins):

* the sharded branch over 8 emulated ranks returns, on every rank, the full-vocab branch's ids,
  value bits and order, on rows with planted ties (inside a shard, across shard edges, at the
  32nd place, equal values with different ids) and NaN-free extremes (a ``-inf`` mask, ``+inf``
  ties, the largest finite values, subnormals, ``+-0``). The top-32 stand-ins emulate
  ``torch.topk``'s CUDA order (the selection, ``gatherTopK``'s order, ``sortKeyValueInplace``'s
  32-element bitonic network), which the family's top-32 kernels reproduce bit for bit;
* without the id order the same rows differ: the re-order is what makes the merge exact;
* each rank all-gathers its 32 pairs in ascending id order, the pairs intact; the block to
  score into is FP32 and full-vocab;
* ``_draft_shard_head`` keeps the shard only when the selector drafts, the target's vocab-parallel
  head is the drafter's LM head behind the target's logits processor, the shard is this
  worker's, and the engine takes a row of the shard's width for a vocab shard;
* the drafter step reads the shard through that head and logs its INFO line, and calls the
  logits processor otherwise; its vocabulary width stays the full vocabulary.

One sm_103 / sm_107 GPU, the family's kernels with 8 shards on one GPU (the all-gather a
concatenation in rank order): at the helix8 decode shapes (16 and 24 requests, K = 5) the sharded
branch equals the full-vocab branch on the gathered FP32 logits bit for bit, and so do the
selector's greedy block logits and the argmax draft tokens; a CUDA graph of the sharded branch
replays the eager result.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"
_MODELING = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31" / _TARGET / "modeling.py"

TOP_K = 32
VOCAB = 163840
TP = 8
SHARD = VOCAB // TP
EDGES = [r * SHARD for r in range(1, TP)]
_BF16_MAX = float(torch.finfo(torch.bfloat16).max)
_BF16_TINY = float(torch.finfo(torch.bfloat16).tiny)


# --- torch.topk's CUDA order, emulated per row ------------------------------------------------------


def _keys(v: np.ndarray) -> np.ndarray:
    """``TopKTypeConfig<float>::convert``: order-preserving uint32 keys, every NaN the largest.

    A BF16 value's key and its FP32 upcast's key order alike, so FP32 keys serve both."""
    bits = v.view(np.uint32)
    mask = np.where(bits & np.uint32(0x80000000), np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return np.where(np.isnan(v), np.uint32(0xFFFFFFFF), bits ^ mask).astype(np.uint32)


def _gt(a: float, b: float) -> bool:
    """``GTOp<float, true>``: NaN above every number, ``-0 == +0``."""
    return (a != a and b == b) or a > b


def _network(values: list) -> list:
    """``bitonicSortKVInPlace`` of 32 valid elements, descending: the order of the positions."""
    keys, pos = list(values), list(range(TOP_K))

    def swap(i, j, direction):
        if _gt(keys[i], keys[j]) == direction:
            keys[i], keys[j] = keys[j], keys[i]
            pos[i], pos[j] = pos[j], pos[i]

    size = 2
    while size < TOP_K:
        stride = size // 2
        while stride > 0:
            for t in range(TOP_K // 2):
                swap(
                    2 * t - (t & (stride - 1)),
                    2 * t - (t & (stride - 1)) + stride,
                    (t & (size // 2)) != 0,
                )
            stride //= 2
        size *= 2
    stride = TOP_K // 2
    while stride > 0:
        for t in range(TOP_K // 2):
            swap(2 * t - (t & (stride - 1)), 2 * t - (t & (stride - 1)) + stride, False)
        stride //= 2
    return pos


def _topk_row(v: np.ndarray) -> np.ndarray:
    """The positions ``torch.topk(v, 32)`` returns on CUDA, in its order."""
    keys = _keys(v)
    kth = np.partition(keys, keys.size - TOP_K)[keys.size - TOP_K]
    above = np.nonzero(keys > kth)[0]
    at = np.nonzero(keys == kth)[0][: TOP_K - above.size]
    gathered = np.concatenate([above, at])
    return gathered[_network([float(x) for x in v[gathered]])]


def _shard_top32(gen_logits, top_k, id_offset, pairs):
    """``k31_dflash2_shard_top32`` on CPU (torch.topk's CUDA order)."""
    assert top_k == TOP_K and gen_logits.is_contiguous()
    n, slots, width = gen_logits.shape
    rows = gen_logits.reshape(-1, width).float().numpy()
    picks = np.stack([_topk_row(r) for r in rows])
    ids = torch.from_numpy(picks).long().view(n, slots, TOP_K)
    values = torch.gather(gen_logits, -1, ids)
    ids = ids + id_offset
    if not pairs:
        return values, ids
    return torch.stack([ids.float(), values.float()], dim=-1).flatten(-2)


def _merge_top32(gathered, top_k):
    """``k31_dflash2_merge_top32`` on CPU: torch.topk of the value column in its CUDA order."""
    assert top_k == TOP_K and gathered.dtype == torch.float32 and gathered.is_contiguous()
    values = gathered[..., 1::2]
    n, slots, cols = values.shape
    picks = np.stack([_topk_row(r) for r in values.reshape(-1, cols).numpy()])
    sel = torch.from_numpy(picks).long().view(n, slots, TOP_K)
    return torch.gather(values, -1, sel), torch.gather(gathered[..., 0::2], -1, sel).long()


# --- rows with planted ties -------------------------------------------------------------------------

_KINDS = (
    "randn",
    "levels",
    "all_equal",
    "inside_shard",
    "across_edges",
    "kth_split",
    "dup_above",
    "masked_ninf",
    "pos_inf",
    "max_finite",
    "subnormal",
    "hot",
)


def _row(kind: str, rng: np.random.Generator) -> np.ndarray:
    """One FP32 row of BF16-representable logits with the kind's ties planted."""
    x = rng.standard_normal(VOCAB).astype(np.float32) * 2.0
    if kind == "levels":
        x = 0.25 * rng.integers(-3, 6, VOCAB).astype(np.float32)
    elif kind == "all_equal":
        x[:] = 10.0
    elif kind == "inside_shard":
        rank, n_hi = int(rng.integers(0, TP)), int(rng.integers(0, 28))
        x[rng.choice(VOCAB, n_hi, replace=False)] = 20.0 + 0.5 * rng.integers(0, 3, n_hi)
        x[rank * SHARD + rng.choice(SHARD, 45, replace=False)] = 12.0
    elif kind == "across_edges":
        n_hi = int(rng.integers(0, 31))
        x[rng.choice(VOCAB, n_hi, replace=False)] = 20.0 + rng.integers(0, 4, n_hi)
        x[[e + d for e in EDGES for d in (-3, -2, -1, 0, 1, 2)]] = 9.0
        x[rng.choice(VOCAB, 20, replace=False)] = 9.0
    elif kind == "kth_split":
        e = EDGES[int(rng.integers(0, len(EDGES)))]
        x[rng.choice(VOCAB, 29, replace=False)] = 30.0
        x[[e - 2, e - 1, e, e + 1, e + 2]] = 15.0
    elif kind == "dup_above":
        for v in (25.0, 24.0, 23.0):
            x[rng.choice(VOCAB, 7, replace=False)] = v
        x[EDGES] = 24.0
        x[[e - 1 for e in EDGES]] = 23.0
    elif kind == "masked_ninf":
        x[:] = -np.inf
        n_fin = int(rng.integers(0, 31))
        x[rng.choice(VOCAB, n_fin, replace=False)] = rng.standard_normal(n_fin)
    elif kind == "pos_inf":
        x[rng.choice(VOCAB, 40, replace=False)] = np.inf
        x[EDGES] = np.inf
    elif kind == "max_finite":
        x[rng.choice(VOCAB, 20, replace=False)] = _BF16_MAX
        x[rng.choice(VOCAB, 20, replace=False)] = _BF16_MAX * 0.99
        x[rng.choice(VOCAB, 50, replace=False)] = -_BF16_MAX
        x[EDGES] = _BF16_MAX
    elif kind == "subnormal":
        x = -np.abs(x) - 1.0
        x[rng.choice(VOCAB, 10, replace=False)] = _BF16_TINY / 4
        x[rng.choice(VOCAB, 10, replace=False)] = -_BF16_TINY / 4
        x[rng.choice(VOCAB, 12, replace=False)] = 0.0
        x[rng.choice(VOCAB, 12, replace=False)] = -0.0
        x[EDGES] = -0.0
        x[[e - 1 for e in EDGES]] = 0.0
    elif kind == "hot":
        x[rng.choice(VOCAB, 60, replace=False)] = 30.0 + 0.25 * rng.integers(0, 6, 60)
    else:
        assert kind == "randn", kind
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def _logits(n: int, slots: int, seed: int) -> torch.Tensor:
    """``[n, slots, VOCAB]`` BF16 logits, the kinds dealt over the rows."""
    rng = np.random.default_rng(seed)
    rows = [_row(_KINDS[i % len(_KINDS)], rng) for i in range(n * slots)]
    return torch.from_numpy(np.stack(rows)).to(torch.bfloat16).view(n, slots, VOCAB)


_INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit patterns, so NaN payloads and signed zeros compare too."""
    return t.view(_INT_VIEW[t.dtype]) if t.dtype in _INT_VIEW else t


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


# --- modeling.py's functions over stand-ins -----------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _source() -> tuple[dict, dict, dict]:
    tree = ast.parse(_MODELING.read_text(), filename=str(_MODELING))
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {
        t.id: n
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    (worker,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"]
    methods = {n.name: n for n in worker.body if isinstance(n, ast.FunctionDef)}
    return top, assigns, methods


class _Log:
    def __init__(self):
        self.lines = []

    def info_once(self, msg, key):
        if key not in {k for _, k in self.lines}:
            self.lines.append((msg, key))


def _worker_class(namespace: dict):
    """``DFlashWorker``'s shard top-32 methods and their module names, compiled into ``namespace``."""
    top, assigns, methods = _source()
    parts = [ast.unparse(assigns[c]) for c in ("_DRAFT_SHARD_TOP32", "_DRAFT_SHARD_LINE")]
    parts.append(ast.unparse(top["_pairs_in_id_order"]))
    body = "\n".join(
        "    " + line
        for m in ("_dflash2_global_top_k", "_draft_shard_head")
        for line in ast.unparse(methods[m]).splitlines()
    )
    parts.append(f"class Worker:\n{body}")
    exec(compile("\n\n".join(parts), str(_MODELING), "exec"), namespace)
    return namespace["Worker"]


def _group(namespace: dict, shards: list, pairs_reorder=None):
    """Each rank's sharded branch over an emulated group: two passes, the gather in rank order."""
    if pairs_reorder is not None:
        namespace["_pairs_in_id_order"] = pairs_reorder
    recorded = []

    def record(tensor, mapping, dim, sizes):
        assert dim == -1 and sizes is None
        recorded.append(tensor.clone())
        return torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * TP))

    def gather(tensor, mapping, dim, sizes):
        return torch.cat(recorded, dim=-1)

    Worker = namespace["Worker"]
    out = []
    for fn in (record, gather):
        namespace["allgather_module"] = fn
        out = []
        for rank in range(TP):
            w = Worker()
            w._vocab_shards, w._vocab_shard, w.mapping = TP, rank, None
            w._draft_logits_are_sharded = lambda logits, md: logits.shape[-1] < VOCAB
            out.append(w._dflash2_global_top_k(shards[rank], None, TOP_K, VOCAB))
    return out, recorded


def _cpu_namespace() -> dict:
    ns = {
        "torch": torch,
        "logger": _Log(),
        "k31_dflash2_shard_top32": _shard_top32,
        "k31_dflash2_merge_top32": _merge_top32,
        # draft_index_pairs off: the shard's top-32 in id order through _pairs_in_id_order.
        "_DRAFT_INDEX_PAIRS": False,
        "_SELECTOR_TOP_K": TOP_K,
    }
    _worker_class(ns)
    return ns


def _full_branch(ns: dict, logits: torch.Tensor):
    w = ns["Worker"]()
    w._vocab_shards, w._vocab_shard = TP, 0
    return w._dflash2_global_top_k(logits.float(), None, TOP_K, VOCAB)


@pytest.mark.cpu_only
def test_the_sharded_branch_returns_the_full_vocab_top32():
    ns = _cpu_namespace()
    logits = _logits(len(_KINDS), 2, seed=20261112)
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    want_ids, want_unary, want_block = _full_branch(ns, logits)
    assert want_block.dtype == torch.float32
    out, _ = _group(ns, shards)
    for rank, (ids, unary, block) in enumerate(out):
        assert _equal(ids, want_ids), f"rank {rank}: ids or their order differ"
        assert _equal(unary, want_unary), f"rank {rank}: values differ"
        assert block.dtype == torch.float32 and block.shape == want_block.shape


@pytest.mark.cpu_only
def test_without_the_id_order_the_merge_differs():
    ns = _cpu_namespace()
    logits = _logits(len(_KINDS), 2, seed=7)
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    want_ids, want_unary, _ = _full_branch(ns, logits)
    out, _ = _group(ns, shards, pairs_reorder=lambda pairs: pairs)
    ids, unary, _ = out[0]
    same = (ids == want_ids).all(-1) & (_bits(unary) == _bits(want_unary)).all(-1)
    differing = int((~same).sum())
    assert differing >= ids.shape[0] * ids.shape[1] // 2, (
        f"only {differing} rows differ without the id order: the planted ties do not bite"
    )


@pytest.mark.cpu_only
def test_each_rank_gathers_its_pairs_in_id_order():
    ns = _cpu_namespace()
    logits = _logits(len(_KINDS), 1, seed=3)
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    _, recorded = _group(ns, shards)
    for rank, pairs in enumerate(recorded):
        ids = pairs[..., 0::2]
        assert bool((ids[..., 1:] > ids[..., :-1]).all()), f"rank {rank}: ids not ascending"
        assert int(ids.min()) >= rank * SHARD and int(ids.max()) < (rank + 1) * SHARD
        raw = _shard_top32(shards[rank], TOP_K, rank * SHARD, True)
        assert _pair_bits(pairs) == _pair_bits(raw), f"rank {rank}: the re-order changed the pairs"


def _pair_bits(pairs: torch.Tensor) -> list:
    """The ``(id, value)`` pairs of each row as sorted bit patterns."""
    rows = pairs.reshape(-1, TOP_K, 2).view(torch.int32).tolist()
    return [sorted(map(tuple, row)) for row in rows]


@pytest.mark.cpu_only
def test_pairs_in_id_order_is_a_permutation_by_id():
    ns = _cpu_namespace()
    g = torch.Generator().manual_seed(11)
    ids = torch.stack([torch.randperm(SHARD, generator=g)[:TOP_K] for _ in range(6)])
    ids = ids.view(2, 3, TOP_K)
    values = torch.randn(2, 3, TOP_K, generator=g)
    pairs = torch.stack([ids.float(), values], dim=-1).flatten(-2)
    out = ns["_pairs_in_id_order"](pairs)
    assert out.shape == pairs.shape and out.dtype == torch.float32 and out.is_contiguous()
    order = ids.argsort(dim=-1)
    assert torch.equal(out[..., 0::2], ids.gather(-1, order).float())
    assert torch.equal(out[..., 1::2].view(torch.int32), values.gather(-1, order).view(torch.int32))


def _case(rank: int = 3) -> SimpleNamespace:
    """A worker on rank ``rank`` whose drafter may keep its shard: every condition holds."""
    lm_head, processor = object(), object()
    return SimpleNamespace(
        lever=True,
        head=SimpleNamespace(
            lm_head=lm_head, shard_width=SHARD, vocab_size=VOCAB, col_base=rank * SHARD
        ),
        draft=SimpleNamespace(
            has_candidate_selector=True, lm_head=lm_head, logits_processor=processor
        ),
        target=SimpleNamespace(logits_processor=processor),
        rank=rank,
        shards=TP,
        sharded=True,
    )


def _decide(ns: dict, c: SimpleNamespace):
    """``_draft_shard_head`` of the case, and the rows the engine's shard check was asked about."""
    ns["_DRAFT_SHARD_TOP32"] = c.lever
    w = ns["Worker"]()
    w.target_vocab_parallel_head = c.head
    w._target_model_ref = None if c.target is None else (lambda: c.target)
    w._vocab_shards, w._vocab_shard = c.shards, c.rank
    seen = []

    def sharded(logits, spec_metadata):
        seen.append((tuple(logits.shape), logits.device.type))
        return c.sharded

    w._draft_logits_are_sharded = sharded
    return w._draft_shard_head(c.draft, SimpleNamespace()), seen


_REFUSALS = {
    "the lever off": lambda c: setattr(c, "lever", False),
    "no selector": lambda c: setattr(c.draft, "has_candidate_selector", False),
    "no vocab-parallel head": lambda c: setattr(c, "head", None),
    "no target": lambda c: setattr(c, "target", None),
    "another LM head": lambda c: setattr(c.draft, "lm_head", object()),
    "another logits processor": lambda c: setattr(c.draft, "logits_processor", object()),
    "another rank's shard": lambda c: setattr(c, "rank", 2),
    "another shard count": lambda c: setattr(c, "shards", 4),
    "another vocabulary size": lambda c: setattr(c.head, "vocab_size", VOCAB + 8),
    "not a vocab shard to the engine": lambda c: setattr(c, "sharded", False),
}


@pytest.mark.cpu_only
def test_the_shard_is_kept_where_it_is_the_drafter_s():
    c = _case()
    got, seen = _decide(_cpu_namespace(), c)
    assert got is c.head
    assert seen == [((0, SHARD), "meta")], "the engine's check must see a row of the shard's width"


@pytest.mark.cpu_only
@pytest.mark.parametrize("why", sorted(_REFUSALS))
def test_the_gathered_logits_stay_otherwise(why):
    c = _case()
    _REFUSALS[why](c)
    got, _ = _decide(_cpu_namespace(), c)
    assert got is None, why


def _forward_impl() -> ast.FunctionDef:
    _, _, methods = _source()
    return methods["_forward_impl"]


@pytest.mark.cpu_only
def test_the_drafter_step_reads_the_shard_through_the_head():
    fn = _forward_impl()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    heads = [c for c in calls if ast.unparse(c.func) == "self._draft_shard_head"]
    assert [ast.unparse(c) for c in heads] == ["self._draft_shard_head(draft_model, spec_metadata)"]
    (branch,) = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "shard_head is None"
    ]
    gathered = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
    shard = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
    assert (
        "draft_model.logits_processor(gen_hidden_states, draft_model.lm_head, attn_metadata, True)"
        in gathered
    )
    assert "shard_head.local_logits(gen_hidden_states)" in shard
    assert "logger.info_once(_DRAFT_SHARD_LINE, key=_DRAFT_SHARD_LINE)" in shard
    processors = [c for c in calls if ast.unparse(c.func).endswith("logits_processor")]
    assert len(processors) == 1, (
        "the drafter calls its logits processor outside the gathered branch"
    )
    sizes = [
        ast.unparse(n.value)
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["vocab_size"]
    ]
    assert "gen_logits.shape[-1] if shard_head is None else shard_head.vocab_size" in sizes


@pytest.mark.cpu_only
def test_the_drafter_shares_the_target_s_head_and_processor():
    """The shell builds the drafter on its own LM head and logits processor, which the engine's
    vocab-parallel head is bound over."""
    tree = ast.parse(_MODELING.read_text())
    built = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "DraftModel"
    ]
    assert built == ["DraftModel(core, self.draft_config, self.lm_head, self.logits_processor)"]


# --- one sm_103 / sm_107 GPU ---------------------------------------------------------------------------

_ON_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the family's top-32 kernels are certified on sm_103 and sm_107",
)


def _gpu_group(mod, monkeypatch, shards):
    recorded = []

    def record(tensor, mapping, dim, sizes):
        recorded.append(tensor.clone())
        return torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * TP), device=tensor.device)

    def gather(tensor, mapping, dim, sizes):
        return torch.cat(recorded, dim=-1)

    def worker(rank):
        return SimpleNamespace(
            _vocab_shards=TP,
            _vocab_shard=rank,
            mapping=None,
            _draft_logits_are_sharded=lambda logits, md: logits.shape[-1] < VOCAB,
        )

    out = []
    for fn in (record, gather):
        monkeypatch.setattr(mod, "allgather_module", fn)
        out = [
            mod.DFlashWorker._dflash2_global_top_k(worker(r), shards[r], None, TOP_K, VOCAB)
            for r in range(TP)
        ]
    return out


@_ON_GPU
@pytest.mark.parametrize("index_pairs", [False, True], ids=["sorted_pairs", "index_pairs"])
@pytest.mark.parametrize("n, slots", [(16, 5), (24, 5), (1, 5), (9, 5), (13, 5), (9, 8)])
def test_gpu_sharded_branch_is_the_full_vocab_branch(n, slots, index_pairs, monkeypatch):
    """Both id-order routes (draft_index_pairs off: the shard's top-32 sorted by
    ``_pairs_in_id_order``; on: the selection kernel's index-order pairs) give the full-vocab
    branch's candidates bit for bit."""
    mod = importlib.import_module(f"{_PKG}.{_TARGET}.modeling")
    monkeypatch.setattr(mod, "_DRAFT_INDEX_PAIRS", index_pairs)
    logits = _logits(n, slots, seed=n * 100 + slots).cuda()
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    full = torch.cat(shards, dim=-1).float()  # the gathered head logits, cast by the processor
    today = SimpleNamespace(_vocab_shards=TP, _vocab_shard=0, mapping=None)
    want_ids, want_unary, want_block = mod.DFlashWorker._dflash2_global_top_k(
        today, full, None, TOP_K, VOCAB
    )
    assert want_block is full
    out = _gpu_group(mod, monkeypatch, shards)
    for rank, (ids, unary, block) in enumerate(out):
        assert _equal(ids, want_ids), f"rank {rank}: ids or their order differ"
        assert _equal(unary, want_unary), f"rank {rank}: values differ"
        assert block.dtype == torch.float32 and block.shape == full.shape

    # The selector's greedy walk scores both blocks the same way.
    g = torch.Generator(device="cuda").manual_seed(5)
    rank_dim, hidden = 256, 512
    pred = (torch.randn(VOCAB, rank_dim, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    succ = (torch.randn(VOCAB, rank_dim, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    proj = (torch.randn(rank_dim, hidden, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    block_hidden = torch.randn(n, slots, hidden, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda")
    walk = mod.k31_dflash2_selector_greedy_path
    want = walk(pred, succ, proj, want_ids, want_unary, block_hidden, anchor, want_block)
    ids, unary, block = out[TP - 1]
    block.fill_(float("nan"))  # whatever the empty block held, the walk overwrites it all
    got = walk(pred, succ, proj, ids, unary, block_hidden, anchor, block)
    assert _equal(got, want), "the selector's block logits differ"
    assert torch.equal(got.view(-1, VOCAB).argmax(-1), want.view(-1, VOCAB).argmax(-1))


@_ON_GPU
def test_gpu_sharded_branch_replays_in_a_cuda_graph(monkeypatch):
    mod = importlib.import_module(f"{_PKG}.{_TARGET}.modeling")
    n, slots, rank = 16, 5, 3
    logits = _logits(n, slots, seed=1).cuda()
    others = [
        mod.k31_dflash2_shard_top32(
            logits[..., r * SHARD : (r + 1) * SHARD].contiguous(), TOP_K, r * SHARD, True
        )
        for r in range(TP)
    ]
    others = [mod._pairs_in_id_order(p) for p in others]

    def gather(tensor, mapping, dim, sizes):
        return torch.cat([*others[:rank], tensor, *others[rank + 1 :]], dim=-1)

    monkeypatch.setattr(mod, "allgather_module", gather)
    worker = SimpleNamespace(
        _vocab_shards=TP,
        _vocab_shard=rank,
        mapping=None,
        _draft_logits_are_sharded=lambda logits, md: True,
    )
    shard = logits[..., rank * SHARD : (rank + 1) * SHARD].contiguous()
    mod.DFlashWorker._dflash2_global_top_k(worker, shard, None, TOP_K, VOCAB)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            ids, unary, _ = mod.DFlashWorker._dflash2_global_top_k(
                worker, shard, None, TOP_K, VOCAB
            )
    torch.cuda.current_stream().wait_stream(stream)
    shard.copy_(_logits(n, slots, seed=2)[..., rank * SHARD : (rank + 1) * SHARD].cuda())
    graph.replay()
    want_ids, want_unary, _ = mod.DFlashWorker._dflash2_global_top_k(
        worker, shard, None, TOP_K, VOCAB
    )
    torch.cuda.synchronize()
    assert _equal(ids, want_ids) and _equal(unary, want_unary)
