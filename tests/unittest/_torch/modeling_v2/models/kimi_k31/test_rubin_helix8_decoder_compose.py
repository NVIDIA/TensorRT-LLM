# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's decoder with the generation levers composed: ``DecodeTarget.forward`` and
``DecodeTarget._run_layers`` from modeling.py, executed on the CPU over exact stand-ins.

Five generation constants act in the decoder loop or around it:

* ``mix_bank_in_mix``: layer 0's embedding gathers into bank row 0, and a block boundary's mix
  stores its snapshot;
* ``boundary_ar_push``: a boundary KDA verify's o_proj pushes once its shape passed the check;
* ``dsa_oproj_push``: a DSA layer's o_proj pushes on its bands' rows;
* ``dsa_ks_select_pack``: the first decode step builds the select / pack runner;
* ``moe_shared_late``: a band step's layers run on the raised streams.

``kda_verify_fused_fb`` acts inside the KDA verify only. Each callee's own contract is tested in
its lever's file. Here the callees are stand-ins that keep that contract over float64 integers
modulo a prime (exact), so a lever may change where bytes go but never what they are. Memory the
step allocates starts as -1, a value no stand-in produces, so a read before a write shows.

* ``_run_layers`` runs over K3.1's 93 layers: DSA at every fourth layer and the last, a bank
  snapshot every 12 layers, layer 0 dense, the drafter tapping three layers. With the levers as
  shipped and with them off, it returns the same hidden states, bank and drafter captures. This
  holds on a KDA verify step's first (checking) and later (pushing) call at 6, 12, 24, 54 and 132
  rows, and on a decode step.
* With the levers on:

  * the embedding is bank row 0, never copied;
  * a boundary's snapshot is copied only where its mix did not store it (a tapped layer before
    it, or rows past the PDL mix);
  * every boundary checks its shape once, then its KDA verify pushes and the boundary finish
    takes the pushed partial;
  * every DSA layer pushes on its bands' rows;
  * no boundary is a DSA layer.
* ``DecodeTarget.forward``'s first decode step builds the key-shard workspace and the select /
  pack runner before the band fork. A band step's runtime therefore carries them, with the core's
  push slot, boundary push and taps, on the raised streams. Later steps reuse them.

Whether the composed levers keep the target's bits on the device (ordering under graph replay,
the collective first-use checks) is checked by a gen-only run, not here.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import pytest
import test_rubin_helix8_moe_cp_tail as T  # noqa: E402 -- the constants with the band parser
import test_rubin_helix8_moe_shared_late as S  # noqa: E402 -- the stream stand-ins
import torch
from torch.overrides import TorchFunctionMode

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MODELING = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31"
    / "kimi_k31_pro_nvfp4__sm_107__helix8"
    / "modeling.py"
)
# The generation levers on the target side, as the stack ships them.
_LEVERS = (
    "kda_verify_fused_fb",
    "dsa_oproj_push",
    "mix_bank_in_mix",
    "boundary_ar_push",
    "dsa_ks_select_pack",
    "moe_shared_late",
)
# The loop's callees that run as modeling.py defines them; the mixers and the slots are stand-ins.
_REAL = (
    "_Pushed",
    "_MoeOut",
    "_MoeTail",
    "_DensePartial",
    "_tap",
    "_capture",
    "_mix_in",
    "_mix_out",
)
# The stand-ins' modulus, and the value of memory the step allocated but did not write.
_P, _UNWRITTEN = 1009.0, -1.0
_H, _VOCAB = 4, 64
# Layers whose output stream the drafter captures: the ones before boundaries 12 and 48, the last.
_TAPS = frozenset({11, 47, 92})
# k5 verify rows: b1, b2, b4, b9 (a moe_shared_late band step) and b22 (past the PDL mixes).
_VERIFY_ROWS = (6, 12, 24, 54, 132)


@functools.lru_cache(maxsize=None)
def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


@functools.lru_cache(maxsize=None)
def _shipped() -> dict:
    return T._constants(_tree())


def _constants() -> dict:
    """modeling.py's module-level constants (a copy: callers may add to it)."""
    return dict(_shipped())


def _gen() -> dict:
    return dict(_constants()["_CONSTANTS"]["generation"])


def _layout() -> tuple:
    """K3.1's layers as helix8 reads them: (count, attn_res block, DSA layers, dense layers)."""
    k31 = _constants()["_K31"]
    n, block = k31["num_hidden_layers"], k31["attn_res_block_size"]
    # The checkpoint's full-attention layers 4, 8, ..., 92 and 93 (one-based).
    mla = frozenset([*range(3, n - 1, 4), n - 1])
    return n, block, mla, frozenset(range(k31["first_k_dense_replace"]))


def _exec(ns: dict, nodes: list) -> dict:
    flags = __future__.annotations.compiler_flag
    module = ast.fix_missing_locations(ast.Module([copy.deepcopy(n) for n in nodes], []))
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _top(names: tuple) -> list:
    nodes = [
        n
        for n in _tree().body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert sorted(n.name for n in nodes) == sorted(names)
    return nodes


def _method(cls: str, name: str) -> ast.FunctionDef:
    (c,) = [n for n in _tree().body if isinstance(n, ast.ClassDef) and n.name == cls]
    (fn,) = [n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def test_the_levers_are_on_as_shipped():
    gen = _gen()
    for key in _LEVERS:
        assert gen[key] is True, key
    # What the rules below read: the push slot's and the boundary push's rows, dsa_oproj_push's bands.
    assert gen["kda_oproj_push"] and gen["attn_res_pdl"] == 2 and gen["attn_res_pdl_max_m"] == 128
    assert gen["dsa_oproj_push_bands"] == "1-72"
    n, block, mla, dense = _layout()
    assert (n, block, len(mla), dense) == (93, 12, 24, frozenset({0}))
    # A boundary is always a KDA layer: the boundary push acts on KDA verifies only.
    assert not {i for i in mla if i % block == 0}


# --- _run_layers ---


class _Memory(TorchFunctionMode):
    """Fresh allocations start as ``_UNWRITTEN``; the bank (the step's one 3-D allocation) and the
    rows the loop copies into it are recorded."""

    def __init__(self):
        super().__init__()
        self.bank, self.copies = None, []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if name == "copy_" and self.bank is not None:
            dst = args[0]
            if dst.untyped_storage().data_ptr() == self.bank.untyped_storage().data_ptr():
                row_bytes = self.bank.stride(0) * self.bank.element_size()
                self.copies.append((dst.data_ptr() - self.bank.data_ptr()) // row_bytes)
        out = func(*args, **kwargs)
        if name in ("new_empty", "empty", "empty_like"):
            out.fill_(_UNWRITTEN)
            if out.dim() == 3 and self.bank is None:
                self.bank = out
        return out


class _Rig:
    """One arm's runtime and mixers over exact stand-ins, and the log of what the loop asked for.

    The push workspace holds one pushed partial at a time (``slot``): a mixer that pushes fills it,
    and the next mix (``finish_norm``, or the boundary push's ``finish``) takes it.
    """

    def __init__(self, on: bool, rows: int):
        self.on, self.rows = on, rows
        gen, consts = _gen(), _constants()
        self.max_rows = gen["attn_res_pdl_max_m"]
        self.dsa_bands = consts["_moe_row_bands"](
            gen["dsa_oproj_push_bands"], consts["_DSA_OPROJ_PUSH_MAX_M"]
        )
        self.slot, self.verdict, self.log, self.captures, self.lines = None, {}, [], {}, []
        self.ns = {**consts, "Any": Any, "Callable": Callable, "NamedTuple": NamedTuple,
                   "Optional": Optional, "torch": torch, "nn": torch.nn}  # fmt: skip
        _exec(self.ns, _top(_REAL))
        self.ns.update(
            _MIX_BANK_IN_MIX=on,
            kda_verify=self.kda_verify,
            kda_decode=self.kda_decode,
            mla_generate=self.mla_generate,
            latent_moe=self.latent_moe,
            dense_mlp=self.dense_mlp,
            logger=SimpleNamespace(info_once=lambda msg, key: self.lines.append(msg)),
        )
        _exec(self.ns, [_method("DecodeTarget", "_run_layers")])

    # The mixes; the TP all-reduce over one stand-in rank is the partial itself.
    def norm(self, prefix, bank, mw):
        return (prefix * 3 + bank.sum(0) + mw.k) % _P

    def reduce(self, partial):
        return partial.clone()

    def reduce_norm(self, prefix, partial, bank, mw):
        new = (prefix + partial) % _P
        return new, self.norm(new, bank, mw)

    def moe_add(self, prefix, moe):
        return (prefix + moe.routed + moe.shared) % _P

    def moe_norm(self, prefix, moe, bank, mw, updated=None):
        """The MoE mix; a PDL mix (rows up to attn_res_pdl_max_m) stores the updated prefix into
        ``updated`` and returns that row."""
        new = self.moe_add(prefix, moe)
        if updated is not None and prefix.shape[0] <= self.max_rows:
            updated[...] = new
            new = updated
        return new, self.norm(new, bank, mw)

    def attn_res_mix(self, prefix, bank, mw):
        return (prefix + bank.sum(0) + mw.k) % _P

    def rms_norm(self, mix, weight):
        return (mix * 5 + weight) % _P

    def finish_norm(self, prefix, bank, mw):
        assert self.slot is not None, "a finish with nothing pushed"
        partial, self.slot = self.slot, None
        self.log.append(("finish_norm",))
        return self.reduce_norm(prefix, partial, bank, mw)

    # The boundary push (boundary_ar_push): a shape is checked once, then pushed.
    def mode(self, rows, candidates):
        if rows == 0 or rows > self.max_rows:
            return None
        return {None: "check", True: "push", False: None}[self.verdict.get((rows, candidates))]

    def check(self, partial, prefix, bank, mw):
        self.verdict[(prefix.shape[0], bank.shape[0])] = True
        self.log.append(("check", bank.shape[0]))
        new = self.reduce(partial)
        return new, self.norm(new, bank, mw)

    def finish(self, prefix, bank, mw):
        assert self.slot is not None, "a boundary finish with nothing pushed"
        partial, self.slot = self.slot, None
        self.log.append(("finish", bank.shape[0]))
        new = self.reduce(partial)
        return new, self.norm(new, bank, mw)

    # The mixers.
    def _push(self, o, pushed: bool):
        if not pushed:
            return o
        assert self.slot is None, "two partials in the push workspace"
        self.slot = o
        return self.ns["_Pushed"](o.shape[0])

    def kda_verify(self, L, x, st, rt, push=False):
        pushed = push and rt.oproj_push is not None and x.shape[0] <= self.max_rows
        self.log.append(("kda", L.layer, push, pushed))
        return self._push((x * 7 + L.layer) % _P, pushed)

    def kda_decode(self, L, x, st, rt):
        self.log.append(("kda", L.layer, False, False))
        return (x * 7 + L.layer) % _P

    def mla_generate(self, L, i, x, st, rt, push=False):
        rows = x.shape[0]
        in_bands = any(lo <= rows <= hi for lo, hi in self.dsa_bands)
        pushed = push and self.on and rt.oproj_push is not None and in_bands
        self.log.append(("mla", i, push, pushed))
        return self._push((x * 11 + i) % _P, pushed)

    def latent_moe(self, L, i, x, st, rt):
        return self.ns["_MoeOut"](routed=(x * 2 + i) % _P, shared=(x * 3) % _P)

    def dense_mlp(self, L, x, st, rt):
        return (x * 13) % _P

    def run(self, steps: int, rows_per_gen: int = 6) -> list:
        """``_run_layers`` of this arm ``steps`` times over the same rows: per call the hidden
        states, the bank, the embedding rows, the drafter captures, the bank rows the loop copied
        and the log."""
        n, block, mla, dense = _layout()
        layers = [
            SimpleNamespace(
                kda=None if i in mla else SimpleNamespace(layer=i),
                mla=SimpleNamespace(layer=i) if i in mla else None,
                moe=None if i in dense else SimpleNamespace(layer=i),
                dense=SimpleNamespace(layer=i) if i in dense else None,
                pre=SimpleNamespace(k=2 * i + 1, out_norm=2 * i + 1),
                mid=SimpleNamespace(k=2 * i + 2, out_norm=2 * i + 2),
                # _layer_views' rules.
                boundary=i % block == 0,
                n_bank=-(-i // block),
            )
            for i in range(n)
        ]
        embed = torch.arange(_VOCAB * _H, dtype=torch.float64).view(_VOCAB, _H) % _P
        out_mix = SimpleNamespace(k=3, out_norm=5)
        head = SimpleNamespace(embed=embed, n_bank=-(-n // block), out=out_mix)
        target = SimpleNamespace(_head=head, _layers=layers)
        rig = self

        class _SpecMd:
            def is_layer_capture(self, layer):
                return layer in _TAPS

            def maybe_capture_hidden_states(self, layer, rows, _):
                rig.captures[layer] = rows.clone()

        st = SimpleNamespace(rows_per_gen=rows_per_gen, spec_md=_SpecMd(), cuda_graph=True)
        bp = SimpleNamespace(mode=self.mode, check=self.check, finish=self.finish)
        rt = SimpleNamespace(
            attn_res_norm=self.norm,
            add_norm=self.reduce_norm,
            reduce_norm=self.reduce_norm,
            reduce=self.reduce,
            moe_norm=self.moe_norm,
            moe_fold_norm=None,
            moe_norm_mx=None,
            moe_add=self.moe_add,
            attn_res_mix=self.attn_res_mix,
            rms_norm=self.rms_norm,
            taps=None,
            oproj_push=SimpleNamespace(finish_norm=self.finish_norm),
            boundary_push=bp if self.on else None,
        )
        ids = torch.arange(self.rows, dtype=torch.int64) * 5 % _VOCAB
        calls = []
        for _ in range(steps):
            self.log, self.captures, self.lines = [], {}, []
            mem = _Memory()
            with mem:
                hidden = self.ns["_run_layers"](target, st, rt, ids, None)
            assert self.slot is None, "a pushed partial nothing finished"
            calls.append(
                dict(
                    hidden=hidden,
                    bank=mem.bank.clone(),
                    embed=embed[ids],
                    captures=self.captures,
                    copies=mem.copies,
                    log=self.log,
                    lines=self.lines,
                )
            )
        return calls


def _arms(rows: int, steps: int = 2, rows_per_gen: int = 6) -> tuple:
    return (
        _Rig(True, rows).run(steps, rows_per_gen),
        _Rig(False, rows).run(steps, rows_per_gen),
    )


def _same_bytes(on: dict, off: dict) -> None:
    for t in (on["hidden"], on["bank"], off["hidden"], off["bank"]):
        assert not (t == _UNWRITTEN).any(), "memory read before it was written"
    assert torch.equal(on["hidden"], off["hidden"])
    assert torch.equal(on["bank"], off["bank"])
    assert set(on["captures"]) == set(off["captures"]) == _TAPS
    for layer in _TAPS:
        assert torch.equal(on["captures"][layer], off["captures"][layer]), layer
    assert torch.equal(on["bank"][0], on["embed"]), "bank row 0 is layer 0's input"


@pytest.mark.parametrize("rows", _VERIFY_ROWS)
def test_the_levers_keep_a_verify_steps_bytes(rows):
    on, off = _arms(rows)
    for got_on, got_off in zip(on, off):
        _same_bytes(got_on, got_off)


def _want_log(rows: int, on: bool, later: bool) -> list:
    """The mixer calls and finishes of a verify step, in layer order, by the decoder's rules."""
    n, block, mla, _ = _layout()
    pdl = rows <= _gen()["attn_res_pdl_max_m"]
    c3 = on and 1 <= rows <= 72
    want = []
    for i in range(n):
        n_bank = -(-i // block)
        if i in mla:
            # dsa_oproj_push: the push-path rule of a non-boundary layer with snapshot rows.
            want.append(("mla", i, True, c3))
            want += [("finish_norm",)] if c3 else []
        elif i % block == 0:
            # boundary_ar_push: the first call at a shape checks it, the later ones push.
            push = on and pdl and later
            want.append(("kda", i, push, push))
            if on and pdl:
                want.append(("finish" if later else "check", n_bank + 1))
        else:
            # kda_oproj_push (on in both arms).
            push = n_bank > 0
            want.append(("kda", i, push, push and pdl))
            want += [("finish_norm",)] if push and pdl else []
    return want


@pytest.mark.parametrize("rows", _VERIFY_ROWS)
def test_the_loop_pushes_and_stores_where_the_rules_say(rows):
    n, block, _, _ = _layout()
    line = _constants()["_MIX_BANK_LINE"]
    pdl = rows <= _gen()["attn_res_pdl_max_m"]
    on, off = _arms(rows)
    for later, (got_on, got_off) in enumerate(zip(on, off)):
        assert got_on["log"] == _want_log(rows, True, bool(later))
        assert got_off["log"] == _want_log(rows, False, bool(later))
        # Off: every boundary's snapshot is a copy (row 0 the embedding's).
        assert got_off["copies"] == list(range(-(-n // block))) and not got_off["lines"]
        # On: row 0 is the embedding's gather, and a boundary's mix stores its row unless the
        # drafter taps the layer before it or the rows are past the PDL mixes.
        copied = [i // block for i in range(block, n, block) if not pdl or i - 1 in _TAPS]
        assert got_on["copies"] == copied
        assert got_on["lines"] == [line] * (-(-n // block) - len(copied))


@pytest.mark.parametrize("rows", [9, 80])
def test_a_decode_step_pushes_the_dsa_layers_only(rows):
    _, _, mla, _ = _layout()
    (on,), (off,) = _arms(rows, steps=1, rows_per_gen=1)
    _same_bytes(on, off)
    c3 = rows <= 72
    for got, arm_on in ((on, True), (off, False)):
        kda = [e for e in got["log"] if e[0] == "kda"]
        mla_calls = [e for e in got["log"] if e[0] == "mla"]
        assert kda and all(e[2:] == (False, False) for e in kda)
        assert [e[1] for e in mla_calls] == sorted(mla)
        assert all(e[2:] == (True, arm_on and c3) for e in mla_calls)
        assert got["log"].count(("finish_norm",)) == (len(mla) if arm_on and c3 else 0)


# --- DecodeTarget.forward ---


def test_the_first_decode_step_builds_the_select_runner_before_the_band_fork():
    cuda = S._Cuda(capturing=True)
    ns = S._streams_ns(cuda)
    assert ns["_KS_SELECT_PACK"] is True
    serves = T._block(S._on(), {}, ("_moe_in_bands", "_moe_shared_late_serves"))
    built, bound, ran = [], [], []

    def key_shard_ws(mapping, md):
        built.append(("ws", mapping, md.key_shard_world))
        return SimpleNamespace(name="key-shard workspace")

    def key_shard_select(ws):
        built.append(("sel", ws))
        return SimpleNamespace(name="select / pack runner", ws=ws)

    ns.update(
        _key_shard_ws=key_shard_ws,
        _key_shard_select=key_shard_select,
        _build_step_args=lambda md, spec, pools, decode: SimpleNamespace(cuda_graph=md.graph),
        _moe_shared_late_serves=serves["_moe_shared_late_serves"],
    )
    _exec(ns, [_method("DecodeTarget", "forward")])
    rt_cls, comm_cls = ns["_Rt"], ns["_MoeComm"]
    required = [f for f in rt_cls._fields if f not in rt_cls._field_defaults and f != "moe_comm"]
    core_rt = rt_cls(
        **{f: f"rt.{f}" for f in required},
        moe_comm=comm_cls(*(f"comm.{f}" for f in comm_cls._fields)),
    )._replace(
        dsa_pools=SimpleNamespace(bind=bound.append),
        ks_ws=None,
        oproj_push="push slot",
        boundary_push="boundary push",
        taps="taps",
    )
    late = ns["_late_streams"]()
    core = SimpleNamespace(
        tp_mapping="tp mapping",
        late_streams=late,
        streams=SimpleNamespace(aux="core aux"),
        _rt=core_rt,
    )

    def run_layers(st, rt, input_ids, inputs_embeds):
        ran.append((cuda.current_stream().name, rt))
        return SimpleNamespace(record_stream=lambda stream: None)

    target = SimpleNamespace(core=core, _rt=core_rt, _run_layers=run_layers)

    def step(rows: int, graph: bool) -> tuple:
        md = SimpleNamespace(kv_cache_manager="cache", key_shard_world=8, graph=graph)
        ns["forward"](target, md, torch.zeros(rows, dtype=torch.int32), "pos", None, "spec")
        return ran[-1]

    # The first decode step: k5 b9 (54 rows), a CUDA graph step in moe_shared_late's band.
    stream, band = step(54, True)
    ws, sel = target._rt.ks_ws, target._rt.ks_sel
    assert built == [("ws", "tp mapping", 8), ("sel", ws)] and sel.ws is ws
    assert core._rt is target._rt and core._rt.ks_sel is sel
    assert stream == late.fwd.name
    moved = {
        "aux_stream": late.aux,
        "fork_stream": late.fork,
        "idx_stream": late.idx,
        "shared_stream": "core aux",
    }
    for field in rt_cls._fields:
        if field == "moe_comm":
            want = core._rt.moe_comm._replace(
                latent_stream=late.latent, prezero_stream=late.prezero
            )
            assert band.moe_comm == want
        else:
            assert getattr(band, field) == moved.get(field, getattr(core._rt, field)), field
    assert (band.ks_ws, band.ks_sel) == (ws, sel)
    assert (band.oproj_push, band.boundary_push, band.taps) == (
        "push slot",
        "boundary push",
        "taps",
    )
    # Later steps reuse the runner: off the band (k5 b4), and an eager step in the band.
    for rows, graph in ((24, True), (54, False)):
        assert step(rows, graph) == ("main", core._rt)
    assert len(built) == 2 and bound == ["cache"] * 3
