# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter-tail switches on CPU.

``draft_select_accept``, ``draft_greedy_tokens``, ``draft_index_pairs``, ``draft_block_rows``,
``draft_tail_rewind``, ``draft_history_aux`` and ``draft_noise_block`` are helix8 generation
constants (on but for ``draft_block_rows`` and ``draft_history_aux``), stated in the startup
constants line and read once into module aliases. Off, every worker method they gate hands
the step back to today's path; on, each takes its path only where it gives today's values:

* the block rows: the strided view equals the slot-id gather row for row;
* the greedy tokens: only on all-greedy drafting with the 32-candidate selector, no d2t and no
  draft-confidence record;
* the index pairs: the selection kernel's id-order pairs in place of ``_pairs_in_id_order`` of the
  shard's top-32;
* the fused acceptance: refused (with its reason logged once) off an all-greedy shard step, past 16
  requests, or when the KDA record is not the engine's ``KimiK3HybridCacheManager`` /
  ``KDAReplayState`` one; the forced-acceptance operand reproduces the engine's three modes;
* the tail rewind: never on a graph warmup or without a pending rewind;
* the history on the aux stream: its positions are the preparation's ``ctx_positions`` (each
  request's history length before the advance, plus 0 .. K), and the preparation's
  ``precompute_context_kv`` returns the aux stream's latents only for the early projection itself.

Read from the target's modeling.py (``ast``) and run over stand-ins; nothing is imported from it.
The GPU class checks (bitwise against the off path at helix8 shapes) are the ops' tests and the
drafter-tail microbench.
"""

from __future__ import annotations
import __future__

import ast
import copy
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
from test_rubin_helix8_draft_window import _MODELING, _constants, _exec, _tree  # noqa: E402

pytestmark = pytest.mark.cpu_only

#: Each switch: (generation constant, module alias).
_SWITCHES = {
    "draft_select_accept": "_DRAFT_SELECT_ACCEPT",
    "draft_greedy_tokens": "_DRAFT_GREEDY_TOKENS",
    "draft_index_pairs": "_DRAFT_INDEX_PAIRS",
    "draft_block_rows": "_DRAFT_BLOCK_ROWS",
    "draft_tail_rewind": "_DRAFT_TAIL_REWIND",
    "draft_history_aux": "_DRAFT_HISTORY_AUX",
    "draft_noise_block": "_DRAFT_NOISE_BLOCK",
}
#: The switches the target ships on (each cuts >= 1 % of the drafter's step time); the others are
#: off.
_SHIPPED_ON = frozenset(
    {
        "draft_select_accept",
        "draft_greedy_tokens",
        "draft_index_pairs",
        "draft_tail_rewind",
        "draft_noise_block",
    }
)


@pytest.fixture(autouse=True)
def _no_capture(monkeypatch):
    """No CUDA device on a CPU-only node, so nothing is being captured (torch's query would raise
    without a device)."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _methods(tree: ast.Module, cls_name: str, names: tuple, ns: dict) -> type:
    """Class ``cls_name`` of ``tree`` cut down to ``names`` (no bases), executed in ``ns``."""
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name]
    body = [
        copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in body} == set(names), f"{cls_name} lacks {set(names)}"
    kwargs = {"type_params": []} if "type_params" in ast.ClassDef._fields else {}
    node = ast.ClassDef(
        name=cls_name, bases=[], keywords=[], body=body, decorator_list=[], **kwargs
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns[cls_name]


def _ns(**switches) -> dict:
    tree = _tree(_MODELING)
    ns = _constants(tree)
    ns.update({alias: False for alias in _SWITCHES.values()})
    ns.update(switches)
    return ns


def test_the_switches_are_logged_generation_constants_as_shipped():
    tree = _tree(_MODELING)
    ns = _constants(tree)
    generation = ns["_CONSTANTS"]["generation"]
    ns.update(_exec(ns, tree, "constants_line"))
    line = ns["constants_line"]("generation")
    for key, alias in _SWITCHES.items():
        shipped = key in _SHIPPED_ON
        assert generation[key] is shipped, key
        assert ns[alias] is shipped, alias
        assert f"{key}={shipped}" in line, key
        (assign,) = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == [alias]
        ]
        assert ast.unparse(assign.value) == f"_CONSTANTS['generation']['{key}']"


# --- block rows -------------------------------------------------------------------------------------------------


def _block_worker(ns, block):
    def slot_ids(dm, n, block_size, k):
        return (torch.arange(n)[:, None] * block_size + 1 + torch.arange(k)[None]).flatten()

    ns["_EngineDFlashWorker"] = SimpleNamespace(_draft_slot_ids=slot_ids)
    cls = _methods(_tree(_MODELING), "DFlashWorker", ("_block_rows", "_block_gather_ids"), ns)
    cls._draft_slot_ids = staticmethod(slot_ids)
    ns["_EngineDFlashWorker"]._draft_slot_ids = cls._draft_slot_ids
    worker = cls()
    worker._compute_block_size = block
    return worker


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("n, k", [(1, 5), (4, 5), (9, 5), (13, 5), (4, 3), (2, 8), (9, 1), (1, 1)])
def test_block_rows_equal_the_slot_gather(on, n, k):
    block = max(k + 1, 6)
    ns = _ns(_DRAFT_BLOCK_ROWS=on)
    worker = _block_worker(ns, block)
    out = torch.randn(n * block, 16).to(torch.bfloat16)
    ids = (torch.arange(n)[:, None] * block + 1 + torch.arange(k)[None]).flatten()
    got = worker._block_rows(None, out, n, k)
    assert got.is_contiguous() and torch.equal(got.view(torch.int16), out[ids].view(torch.int16))


def test_block_rows_gather_when_the_rows_are_not_whole_blocks():
    ns = _ns(_DRAFT_BLOCK_ROWS=True)
    worker = _block_worker(ns, 6)
    out = torch.randn(4 * 6 + 2, 8)
    got = worker._block_rows(None, out, 4, 5)
    ids = (torch.arange(4)[:, None] * 6 + 1 + torch.arange(5)[None]).flatten()
    assert torch.equal(got, out[ids])


# --- greedy tokens ----------------------------------------------------------------------------------------------


def _greedy(**state):
    ns = _ns(_DRAFT_GREEDY_TOKENS=state.pop("on", True))
    cls = _methods(_tree(_MODELING), "DFlashWorker", ("_greedy_tokens_apply",), ns)
    worker = cls()
    worker._d2t = state.pop("d2t", None)
    worker._accept_stats = state.pop("stats", None)
    dm = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=state.pop("top_k", 32)))
    md = SimpleNamespace(wants_advanced_draft_sampling=state.pop("advanced", False))
    return worker._greedy_tokens_apply(dm, md)


def test_greedy_tokens_apply_only_to_greedy_drafting_with_the_32_candidate_selector():
    assert _greedy() is True
    assert _greedy(on=False) is False
    assert _greedy(advanced=True) is False
    assert _greedy(d2t=torch.zeros(4)) is False
    assert _greedy(top_k=16) is False
    assert _greedy(stats=SimpleNamespace(confidence_provider=object())) is False
    assert _greedy(stats=SimpleNamespace(confidence_provider=None)) is True


# --- index pairs ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("on", [False, True])
def test_index_pairs_replace_the_id_order_sort(on):
    calls, made = [], {}

    def pairs(name):
        made[name] = torch.zeros(2, 5, 64)
        return made[name]

    def name_of(t):
        return next(n for n, v in made.items() if v is t)

    ns = _ns(_DRAFT_INDEX_PAIRS=on)
    ns.update(
        k31_dflash2_shard_top32=lambda x, k, off, p: calls.append(("top32", off)) or pairs("top32"),
        _pairs_in_id_order=lambda p: calls.append(("sort", name_of(p))) or pairs("sorted"),
        k31_dflash2_shard_index_pairs=lambda x, off: calls.append(("index", off)) or pairs("index"),
        allgather_module=lambda t, mapping, dim, sizes: calls.append(("gather", name_of(t)))
        or torch.zeros(1, 1, 64),
        k31_dflash2_merge_top32=lambda g, k: ("unary", "ids"),
    )
    cls = _methods(_tree(_MODELING), "DFlashWorker", ("_dflash2_global_top_k",), ns)
    worker = cls()
    worker._vocab_shards, worker._vocab_shard, worker.mapping = 8, 3, None
    worker._draft_logits_are_sharded = lambda logits, md: True
    shard = torch.zeros(2, 5, 20480)
    ids, unary, block = worker._dflash2_global_top_k(shard, None, 32, 163840, block=False)
    assert (ids, unary, block) == ("ids", "unary", None)
    if on:
        assert calls == [("index", 3 * 20480), ("gather", "index")]
    else:
        assert calls == [("top32", 3 * 20480), ("sort", "top32"), ("gather", "sorted")]
    _, _, block = worker._dflash2_global_top_k(shard, None, 32, 163840)
    assert block.dtype == torch.float32 and block.shape == (2, 5, 163840)


# --- fused acceptance -------------------------------------------------------------------------------------------


class _Manager:
    def update_mamba_states(self, *args, **kwargs):
        pass


class _Replay:
    def update(self, batch):
        pass

    def record_acceptance(self, *args):
        pass


def _accept_ns(**switches):
    ns = _ns(**switches)
    logged = []
    ns.update(
        VocabShardedLogits=type("VocabShardedLogits", (), {}),
        dflash2_sampled_step=lambda worker, md: False,
        KimiK3HybridCacheManager=_Manager,
        KDAReplayState=_Replay,
        logger=SimpleNamespace(info=logged.append),
        # The engine's forced-acceptance strides (speculative/interface.py).
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE=6007,
        _FORCE_ACCEPT_RNG_SLOT_STRIDE=1009,
    )
    cls = _methods(
        _tree(_MODELING),
        "DFlashWorker",
        (
            "_fused_select_accept",
            "_log_select_accept",
            "_kda_replay_operands",
            "_force_accept_kernel_args",
            "_fused_kv_rewind_operands",
        ),
        ns,
    )
    return ns, cls, logged


def _md(**kw):
    base = dict(
        is_ragged_verify=False,
        is_all_greedy_sample=True,
        enable_penalty=False,
        is_cuda_graph=False,
        runtime_draft_len=5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_fused_acceptance_off_is_todays():
    ns, cls, logged = _accept_ns()
    worker = cls()
    assert worker._fused_select_accept(None, None, None) is None
    assert logged == []


@pytest.mark.parametrize(
    "why, logits, md, gens",
    [
        ("not a greedy shard step", object(), _md(), 4),
        ("not a greedy shard step", "shard", _md(is_all_greedy_sample=False), 4),
        ("not a greedy shard step", "shard", _md(is_ragged_verify=True), 4),
        ("not a greedy shard step", "shard", _md(enable_penalty=True), 4),
        ("batch 0/17", "shard", _md(), 17),
    ],
)
def test_fused_acceptance_refuses_and_says_why(why, logits, md, gens):
    ns, cls, logged = _accept_ns(_DRAFT_SELECT_ACCEPT=True)
    worker = cls()
    if logits == "shard":
        logits = ns["VocabShardedLogits"]()
    attn = SimpleNamespace(num_seqs=gens, num_contexts=0)
    assert worker._fused_select_accept(logits, attn, md) is None
    assert worker._fused_select_accept(logits, attn, md) is None
    assert logged == [f"Kimi K3.1 Rubin DFlash2 select_accept: fused=0 reason={why}"]


def _replay_case(**kw):
    prev = kw.pop("prev", torch.zeros(8, dtype=torch.int32))
    replay_cls = kw.pop("replay_cls", _Replay)
    replay = replay_cls()
    replay.prev_num_accepted_tokens = prev
    mgr = kw.pop("mgr_cls", _Manager)()
    mgr.kda_context_only = kw.pop("context_only", False)
    mgr.local_num_mamba_layers = kw.pop("layers", 12)
    mgr._speculative_state = replay
    mgr._dummy_request_mask = torch.zeros(8, dtype=torch.bool)
    mgr.get_state_indices = lambda: torch.arange(8, dtype=torch.int32)
    return SimpleNamespace(kv_cache_manager=mgr, mamba_metadata=SimpleNamespace(state_indices=None))


def test_kda_replay_operands_are_the_engine_record_only():
    ns, cls, _ = _accept_ns(_DRAFT_SELECT_ACCEPT=True)

    class OwnUpdate(_Manager):
        def update_mamba_states(self, *args, **kwargs):
            pass

    class OwnRecord(_Replay):
        def record_acceptance(self, *args):
            pass

    class Seeded(
        _Replay
    ):  # a subclass that keeps the base's update and record (RubinKdaReplayState)
        pass

    why, ops = cls._kda_replay_operands(_replay_case())
    assert why is None and ops[0].dtype == torch.int32 and ops[1].shape == (8,)
    assert cls._kda_replay_operands(_replay_case(replay_cls=Seeded))[0] is None
    assert cls._kda_replay_operands(SimpleNamespace(kv_cache_manager=object()))[0] == "manager"
    assert cls._kda_replay_operands(_replay_case(mgr_cls=OwnUpdate))[0] == "manager"
    assert cls._kda_replay_operands(_replay_case(context_only=True)) == (None, None)
    assert cls._kda_replay_operands(_replay_case(layers=0)) == (None, None)
    assert cls._kda_replay_operands(_replay_case(replay_cls=OwnRecord))[0] == "replay_class"
    assert cls._kda_replay_operands(_replay_case(prev=None))[0] == "replay_unbound"
    assert cls._kda_replay_operands(_replay_case(prev=torch.zeros(8)))[0] == "layout"


@pytest.mark.parametrize(
    "force, k, want",
    [
        (0.0, 5, None),
        (2.0, 5, (3,)),
        (5.5, 5, (6,)),
        (2.43, 5, (3, 0.43, 6)),
        (2.43, 1, (2,)),
        (3.25, 8, (4, 0.25, 9)),
    ],
)
def test_force_operand_reproduces_the_engine_modes(force, k, want):
    ns, cls, _ = _accept_ns(_DRAFT_SELECT_ACCEPT=True)
    worker = cls()
    worker.force_num_accepted_tokens = force
    worker._force_accept_rng_pool = "pool"
    worker._force_accept_rng_counter = "counter"
    worker._ensure_force_accept_rng_state = lambda device: None
    got = worker._force_accept_kernel_args(k, _md(), None)
    if want is None or len(want) == 1:
        assert got == want
    else:
        assert got[0] == want[0] and abs(got[1] - want[1]) < 1e-12 and got[2] == want[2]
        assert got[3:5] == ("pool", "counter")
        assert got[5:] == (
            ns["_FORCE_ACCEPT_RNG_COUNTER_STRIDE"],
            ns["_FORCE_ACCEPT_RNG_SLOT_STRIDE"],
        )
    # A graph warmup forces nothing (the engine's rule).
    assert worker._force_accept_kernel_args(k, _md(is_cuda_graph=True), None) is None


# --- tail rewind ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("on", [False, True])
def test_tail_rewind_only_with_a_pending_rewind_off_warmup(on):
    ns, cls, _ = _accept_ns(_DRAFT_TAIL_REWIND=on)
    worker = cls()
    amount = torch.zeros(4, dtype=torch.int32)
    kv = torch.zeros(6, dtype=torch.int32)
    worker.__dict__.update(
        _kv_rewind_pending=True, _kv_rewind_amount=amount, _kv_rewind_bs=4, _kv_rewind_nc=0
    )
    attn = SimpleNamespace(kv_lens_cuda=kv)
    accepted = torch.zeros(4, 6, dtype=torch.int32)
    drafts = torch.zeros(4, 5, dtype=torch.int32)
    got = worker._fused_kv_rewind_operands(attn, _md(), 4, accepted, drafts)
    if not on:
        assert got is None
        return
    assert got[0] is kv and got[1] is amount and got[2] == 0
    assert (
        worker._fused_kv_rewind_operands(attn, _md(is_cuda_graph=True), 4, accepted, drafts) is None
    )
    worker._kv_rewind_pending = False
    assert worker._fused_kv_rewind_operands(attn, _md(), 4, accepted, drafts) is None


# --- history on the aux stream ------------------------------------------------------------------------------------


def test_history_positions_are_the_preparation_s():
    """The aux stream's positions equal dflash_paged_latent_prepare's ctx_positions: ``old + j`` for
    ``j < width``, ``old`` the request's history length before the advance (its slot's)."""
    ns = _ns(_DRAFT_HISTORY_AUX=True)
    issued = []
    ns.update(torch=torch)
    cls = _methods(_tree(_MODELING), "DFlashWorker", ("_issue_draft_projection",), ns)
    worker = cls()
    gens, width = 4, 6
    worker._draft_tokens_per_req = width
    worker._fused_gen_prepare_applies = lambda md: True
    worker._ctx_len = torch.tensor([100, 7, 0, 55, 9, 31], dtype=torch.int64)
    worker._batch_to_slot_buffer = SimpleNamespace(
        tensor=torch.tensor([3, 0, 5, 2, 1], dtype=torch.int32)
    )
    captured = torch.zeros(gens * width, 12)
    md = SimpleNamespace(is_cuda_graph=True, get_hidden_states=lambda n: captured)
    attn = SimpleNamespace(num_contexts=0, num_ctx_tokens=0)
    dm = SimpleNamespace(
        context_kv_only=False, issue_projection=lambda h, positions: issued.append(positions)
    )
    assert worker._issue_draft_projection(dm, attn, md, gens * width, gens) is True
    (positions,) = issued
    got = positions()
    old = torch.tensor([55, 100, 31, 0], dtype=torch.int64)
    want = (old[:, None] + torch.arange(width)[None]).reshape(-1)
    assert got.dtype == torch.int64 and torch.equal(got, want)


def test_history_positions_are_not_read_with_the_switch_off():
    ns = _ns()
    issued = []
    cls = _methods(_tree(_MODELING), "DFlashWorker", ("_issue_draft_projection",), ns)
    worker = cls()
    worker._draft_tokens_per_req = 6
    worker._fused_gen_prepare_applies = lambda md: True
    md = SimpleNamespace(is_cuda_graph=True, get_hidden_states=lambda n: torch.zeros(12, 4))
    attn = SimpleNamespace(num_contexts=0, num_ctx_tokens=0)
    dm = SimpleNamespace(
        context_kv_only=False, issue_projection=lambda h, positions: issued.append(positions)
    )
    worker._issue_draft_projection(dm, attn, md, 12, 2)
    assert issued == [None]


def test_the_preparation_takes_the_aux_latents_of_the_early_projection_only():
    ns = _ns(_DRAFT_HISTORY_AUX=True)
    logged = []
    ns.update(logger=SimpleNamespace(info_once=lambda msg, key: logged.append(msg)))
    cls = _methods(_tree(_MODELING), "DraftModel", ("precompute_context_kv",), ns)
    dm = cls()
    computed = []
    dm._context_kv = lambda x, pos: computed.append(x) or "computed"
    projected, latent = torch.zeros(3), torch.ones(3)
    dm._early_latent = (projected, latent)
    assert dm.precompute_context_kv(projected, None) == (latent, None)
    assert dm._early_latent is None and computed == [] and len(logged) == 1
    dm._early_latent = (projected, latent)
    other = torch.zeros(3)
    assert dm.precompute_context_kv(other, None) == ("computed", None)
    assert computed == [other] and dm._early_latent is None
    dm._early_latent = None
    assert dm.precompute_context_kv(projected, None) == ("computed", None)


# --- noise block --------------------------------------------------------------------------------------------------


def _body(fn: ast.FunctionDef) -> list:
    """``fn``'s statements without its docstring."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def test_the_noise_block_preparation_is_the_engine_s_but_the_block():
    """Past its switch, helix8's ``_prepare_paged_latent_gen_inputs`` is the engine's statement for
    statement, but for the input block: ``k31_dflash2_noise_block`` over the embedding table in
    place of ``dflash_noise_block_embedding`` (without a trained mask row)."""
    engine_path = _MODELING.parents[5] / "speculative" / "dflash.py"
    engine = ast.parse(engine_path.read_text())
    (engine_cls,) = [
        n for n in engine.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"
    ]
    (want,) = [
        n
        for n in engine_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_paged_latent_gen_inputs"
    ]
    tree = _tree(_MODELING)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"]
    (got,) = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_paged_latent_gen_inputs"
    ]
    gate, *rest = _body(got)
    assert ast.unparse(gate.test) == "not _DRAFT_NOISE_BLOCK or trained_mask_embedding is not None"
    assert ast.unparse(gate.body[0]).startswith("return super()._prepare_paged_latent_gen_inputs(")
    swapped = []
    for stmt in rest:
        text = ast.unparse(stmt)
        if text.startswith("noise_embedding = k31_dflash2_noise_block("):
            assert ast.unparse(stmt.value.args[0]) == (
                "draft_model.draft_model_full.model.embed_tokens.table"
            )
            assert [ast.unparse(a) for a in stmt.value.args[1:]] == [
                "prep.bonus",
                "mask_token_id",
                "block_size",
            ]
            swapped.append(stmt)
            continue
        swapped.append(stmt)
    engine_text = [ast.unparse(s) for s in _body(want)]
    got_text = [ast.unparse(s) for s in swapped]
    (i,) = [j for j, t in enumerate(engine_text) if t.startswith("noise_embedding = ")]
    assert engine_text[i].startswith("noise_embedding = dflash_noise_block_embedding(")
    assert got_text[i].startswith("noise_embedding = k31_dflash2_noise_block(")
    assert got_text[:i] == engine_text[:i] and got_text[i + 1 :] == engine_text[i + 1 :]


def test_the_target_embedding_exposes_its_table():
    tree = _tree(_MODELING)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_TargetEmbedding"]
    (table,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "table"]
    assert [ast.unparse(d) for d in table.decorator_list] == ["property"]
    assert ast.unparse(_body(table)[0]) == "return self._core().w['embed']"
