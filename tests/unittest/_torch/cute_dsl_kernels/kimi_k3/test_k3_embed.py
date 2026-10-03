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
"""``trtllm::k3_embed`` (a decode step's embedding rows, one 16-byte load per thread) and ``trtllm::k3_embed_norm``
(the rows into a snapshot-bank slot plus layer 0's input RMSNorm, one launch) at the in-model shape: Kimi K3's
replicated bf16 embedding table [163840, 7168], M = 1 .. 64 tokens (every M; each compiles its own kernel).

The reference is what the model runs without them (KimiLinearModel._embed and layer 0's input_layernorm):
nn.Embedding's ``F.embedding`` (an index_select) for the rows, and the RMSNorm module, whose bf16 path is
``trtllm::flashinfer_rmsnorm`` (``flashinfer.norm.rmsnorm``), for the norm. Both ops claim bit-identity (bf16 bits).
k3_embed_norm reproduces flashinfer's CuTe DSL RMSNormKernel (one 128-thread CTA per row for 6144 < H <= 16384, see
k3_embed_kernel), so its checks need flashinfer (the model's gate for the op) on that kernel. The table is replicated
(no vocabulary shard), so the model never passes an id outside [0, V); the ops define one as a zero row, checked
against the masked reference.

Checks (in the first three, every call runs twice and the reruns must be bit-identical):

* k3_embed: int32 ids (the engine's) at every M, int64 ids at a few; ids with 0, V - 1, repeats, one id M times;
* k3_embed_norm at every M, called as the model calls it (the module's weight and eps, a bank slot as ``raw``): the
  normed rows against the module on the F.embedding rows, the raw rows, the bank's other slots untouched; eps 1e-6
  and 1e-5, bank slots 0 (the model's) and 1;
* ids outside [0, V) (negative, V, the int32 extremes, int64 ids past 2^32): zero rows, normed as the module norms
  them;
* every table row through both ops (64 consecutive ids per call);
* CUDA graphs of both ops at M = 1 .. 16, 17 .. 32, 33 .. 48 and 49 .. 64, captured once and replayed with the ids
  rewritten in place, bit-identical to the eager ops and the reference on every replay; the same ids replayed again
  give the same bits.

Table: ``python3 test_k3_embed.py report``. Timing: ``python3 test_k3_embed.py time`` (CUDA graphs of back-to-back
calls, each on its own random ids, median us per call over 15 replays at every M: F.embedding vs k3_embed, and the
rows plus the snapshot copy plus the RMSNorm module vs k3_embed_norm).
"""

import math
import os
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

VOCAB = 163840  # Kimi K3's vocabulary (a replicated nn.Embedding)
HIDDEN = 7168
EPS = 1e-6
EPS_ALT = 1e-5
MAX_TOKENS = 64  # k3_embed's MAX_TOKENS
TOKENS = list(range(1, MAX_TOKENS + 1))
INT64_TOKENS = [1, 7, 8, 16, 33, 64]
OUTSIDE_TOKENS = [1, 8, 64]
GRAPH_FAMILIES = [list(range(lo, lo + 16)) for lo in range(1, MAX_TOKENS + 1, 16)]
BANK_SLOTS = 4  # attention-residual snapshots; the model passes slot 0 as raw
SENTINEL = 7.0
ID_KINDS = ("0, V - 1, repeats", "V - 1, 0", "random", "one id x M")
GRAPH_KINDS = ("random", "0, V - 1, repeats", "outside [0, V)", "one id x M", "V - 1, 0")
OUTSIDE_IDS = (-1, VOCAB, VOCAB + 7, -VOCAB, 2**31 - 1, -(2**31))
OUTSIDE_IDS_64 = OUTSIDE_IDS + (2**32, 2**32 + 5, 3 - 2**32, 2**40)


def _sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


pytestmark = pytest.mark.skipif(
    not _sm90(), reason="needs SM90 or newer (griddepcontrol, programmatic dependent launch)"
)


def _op():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_embed import op

    return op


def _flashinfer_norm() -> bool:
    """Whether layer 0's RMSNorm runs flashinfer's CuTe DSL RMSNormKernel: the model takes k3_embed_norm only with
    flashinfer, and FLASHINFER_USE_CUDA_NORM=1 selects flashinfer's CUDA kernel, which is not the one the op
    reproduces."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

    return IS_FLASHINFER_AVAILABLE and os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") != "1"


def _need_norm():
    if not _flashinfer_norm():
        pytest.skip("needs flashinfer's CuTe DSL RMSNorm (the kernel k3_embed_norm reproduces)")


# ----------------------------------------------------------------------------------------------------------------
# Inputs and the model's path
# ----------------------------------------------------------------------------------------------------------------

_state = {}


def table() -> torch.Tensor:
    """The embedding table, built once: N(0, 1) rows at log-uniform scales 1e-4 .. 1e2, row 1 all zero."""
    if "table" not in _state:
        with torch.inference_mode(False):
            gen = torch.Generator(device="cuda").manual_seed(20260930)
            t = torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
            scale = torch.empty(VOCAB, 1, device="cuda")
            scale.uniform_(math.log(1e-4), math.log(1e2), generator=gen)
            t.mul_(scale.exp_().bfloat16())
            t[1].zero_()
        _state["table"] = t
    return _state["table"]


def layer0_norm(eps):
    """Layer 0's input_layernorm as KimiLinearDecoderLayer builds it (the RMSNorm module, bf16): weights around 1,
    every 101st negative."""
    key = ("norm", eps)
    if key not in _state:
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        with torch.inference_mode(False):
            gen = torch.Generator(device="cuda").manual_seed(7)
            w = 1.0 + 0.2 * torch.randn(HIDDEN, generator=gen, device="cuda")
            w[::101] *= -1.0
            norm = RMSNorm(hidden_size=HIDDEN, eps=eps, dtype=torch.bfloat16, device="cuda")
            norm.requires_grad_(False)
            norm.weight.copy_(w)
        _state[key] = norm
    return _state[key]


def make_ids(m, kind, gen, dtype=torch.int32):
    """``m`` ids in [0, V) of one of ID_KINDS."""
    ids = torch.randint(0, VOCAB, (m,), generator=gen, device="cuda")
    if kind == "0, V - 1, repeats":
        ids[0] = 0
        if m >= 2:
            ids[-1] = VOCAB - 1
        if m >= 4:
            ids[1] = VOCAB - 1
            ids[2] = 0
        if m >= 6:
            ids[4] = ids[3]
    elif kind == "V - 1, 0":
        ids[0] = VOCAB - 1
        if m >= 2:
            ids[-1] = 0
    elif kind == "one id x M":
        ids = ids[:1].repeat(m)
    return ids.to(dtype)


def make_outside_ids(m, dtype, gen):
    """``m`` ids: the even positions outside [0, V) (cycling through the dtype's out-of-range values), the others
    random in [0, V)."""
    bad = OUTSIDE_IDS_64 if dtype == torch.int64 else OUTSIDE_IDS
    ids = torch.randint(0, VOCAB, (m,), generator=gen, device="cuda")
    ids[0::2] = torch.tensor([bad[i % len(bad)] for i in range((m + 1) // 2)], device="cuda")
    return ids.to(dtype)


def rows_ref(ids):
    """The model's rows: F.embedding (nn.Embedding's index_select); an id outside [0, V) gives the ops' zero row."""
    valid = (ids >= 0) & (ids < VOCAB)
    if bool(valid.all()):
        return F.embedding(ids, table())
    rows = F.embedding(torch.where(valid, ids, torch.zeros_like(ids)), table())
    return rows.masked_fill_(~valid[:, None], 0.0)


def embed(ids):
    """k3_embed as KimiLinearModel._embed calls it."""
    return torch.ops.trtllm.k3_embed(ids, table())


def new_bank(m):
    """A snapshot bank [BANK_SLOTS, m, H] filled with SENTINEL."""
    return torch.full((BANK_SLOTS, m, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")


def embed_norm_into(ids, norm, raw):
    """k3_embed_norm as KimiLinearModel._embed_norm calls it: the norm module's weight and eps, ``raw`` a bank slot."""
    return torch.ops.trtllm.k3_embed_norm(ids, table(), norm.weight, norm.variance_epsilon, raw)


def embed_norm(ids, eps=EPS, slot=0):
    """k3_embed_norm into ``slot`` of a fresh bank: (normed rows, bank)."""
    bank = new_bank(ids.numel())
    return embed_norm_into(ids, layer0_norm(eps), bank[slot]), bank


def same(a, b) -> bool:
    """Bit-identical bf16 tensors."""
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def differ(a, b):
    """The number of rows of ``a`` that differ from ``b`` in any bit (on the device)."""
    return (a.view(torch.int16) != b.view(torch.int16)).any(dim=1).sum()


def check_embed(ids, kind):
    got = embed(ids)
    return {f"{kind}: rows": same(got, rows_ref(ids)), f"{kind}: rerun": same(embed(ids), got)}


def check_norm(ids, kind, eps=EPS, slot=0):
    normed, bank = embed_norm(ids, eps, slot)
    normed_again, bank_again = embed_norm(ids, eps, slot)
    rows = rows_ref(ids)
    others = [s for s in range(BANK_SLOTS) if s != slot]
    return {
        f"{kind}: normed": same(normed, layer0_norm(eps)(rows)),
        f"{kind}: raw": same(bank[slot], rows),
        f"{kind}: other slots": bool((bank[others] == SENTINEL).all()),
        f"{kind}: rerun": same(normed_again, normed) and same(bank_again, bank),
    }


def result(op_name, m, case, checks):
    """A table row: ``checks`` maps each comparison to whether it held."""
    failed = [k for k, ok in checks.items() if not ok]
    identical = "yes" if not failed else "no: " + ", ".join(failed)
    return dict(op=op_name, m=m, case=case, identical=identical, ok=not failed)


# ----------------------------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------------------------


def measure_embed(m):
    """k3_embed at M tokens: int32 ids of every kind, int64 ids at INT64_TOKENS."""
    gen = torch.Generator(device="cuda").manual_seed(1000 + m)
    checks = {}
    for kind in ID_KINDS:
        checks.update(check_embed(make_ids(m, kind, gen), kind))
    out = [result("k3_embed", m, "int32 ids: " + " / ".join(ID_KINDS), checks)]
    if m in INT64_TOKENS:
        ids = make_ids(m, ID_KINDS[0], gen, torch.int64)
        out.append(result("k3_embed", m, f"int64 ids: {ID_KINDS[0]}", check_embed(ids, "int64")))
    return out


def measure_norm(m):
    """k3_embed_norm at M tokens: int32 ids of every kind into slot 0 at EPS, random ids into slot 1 at EPS_ALT,
    int64 ids at INT64_TOKENS."""
    gen = torch.Generator(device="cuda").manual_seed(2000 + m)
    checks = {}
    for kind in ID_KINDS:
        checks.update(check_norm(make_ids(m, kind, gen), kind))
    case = f"int32 ids: {' / '.join(ID_KINDS)}; eps {EPS:g}, slot 0"
    out = [result("k3_embed_norm", m, case, checks)]
    checks = check_norm(make_ids(m, "random", gen), "random", EPS_ALT, 1)
    out.append(result("k3_embed_norm", m, f"int32 ids: random; eps {EPS_ALT:g}, slot 1", checks))
    if m in INT64_TOKENS:
        checks = check_norm(make_ids(m, ID_KINDS[0], gen, torch.int64), "int64")
        case = f"int64 ids: {ID_KINDS[0]}; eps {EPS:g}, slot 0"
        out.append(result("k3_embed_norm", m, case, checks))
    return out


def measure_outside(m, dtype, with_norm):
    """Both ops on ids outside [0, V) at the even positions."""
    gen = torch.Generator(device="cuda").manual_seed(3000 + m)
    ids = make_outside_ids(m, dtype, gen)
    values = ", ".join(map(str, OUTSIDE_IDS_64 if dtype == torch.int64 else OUTSIDE_IDS))
    case = f"{str(dtype).split('.')[-1]} ids, even positions outside [0, V): {values}"
    out = [result("k3_embed", m, case, check_embed(ids, "outside"))]
    if with_norm:
        out.append(result("k3_embed_norm", m, case, check_norm(ids, "outside")))
    return out


def measure_every_row(with_norm):
    """Every table row through k3_embed (and k3_embed_norm), MAX_TOKENS consecutive ids per call."""
    tab = table()
    norm = layer0_norm(EPS) if with_norm else None
    bank = new_bank(MAX_TOKENS)
    every_id = torch.arange(VOCAB, dtype=torch.int32, device="cuda")
    bad = torch.zeros(3, dtype=torch.int64, device="cuda")
    for lo in range(0, VOCAB, MAX_TOKENS):
        ids = every_id[lo : lo + MAX_TOKENS]
        rows = F.embedding(ids, tab)
        bad[0] += differ(embed(ids), rows)
        if with_norm:
            normed = embed_norm_into(ids, norm, bank[0])
            bad[1] += differ(bank[0], rows)
            bad[2] += differ(normed, norm(rows))
    bad = bad.tolist()
    case = f"all {VOCAB} rows, {MAX_TOKENS} consecutive int32 ids per call"
    out = [result("k3_embed", "all", case, {f"rows ({bad[0]} differ)": bad[0] == 0})]
    if with_norm:
        checks = {f"raw ({bad[1]} rows differ)": bad[1] == 0}
        checks[f"normed ({bad[2]} rows differ)"] = bad[2] == 0
        out.append(result("k3_embed_norm", "all", f"{case}, eps {EPS:g}", checks))
    return out


def measure_graph(ms, with_norm):
    """One CUDA graph with both ops at every M of ``ms``, captured once and replayed with the ids rewritten in place
    (one GRAPH_KINDS kind per replay), each replay against the eager ops and the reference; then the last ids
    replayed again against the first replay of them."""
    tab = table()
    norm = layer0_norm(EPS) if with_norm else None
    gen = torch.Generator(device="cuda").manual_seed(4000 + ms[0])
    ids_buf = {m: torch.zeros(m, dtype=torch.int32, device="cuda") for m in ms}
    banks = {m: new_bank(m) for m in ms}
    outs = {}

    def body():
        for m in ms:
            outs["k3_embed", m] = torch.ops.trtllm.k3_embed(ids_buf[m], tab)
            if with_norm:
                outs["k3_embed_norm", m] = embed_norm_into(ids_buf[m], norm, banks[m][0])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()  # every M compiles on its first call, outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
    torch.cuda.synchronize()
    family = f"{ms[0]}-{ms[-1]}"
    out = []
    for rep, kind in enumerate(GRAPH_KINDS):
        for m in ms:
            if kind == "outside [0, V)":
                ids_buf[m].copy_(make_outside_ids(m, torch.int32, gen))
            else:
                ids_buf[m].copy_(make_ids(m, kind, gen))
            banks[m].fill_(SENTINEL)
        graph.replay()
        torch.cuda.synchronize()
        checks = {"k3_embed": {}, "k3_embed_norm": {}}
        for m in ms:
            ids = ids_buf[m]
            rows = rows_ref(ids)
            got = outs["k3_embed", m]
            checks["k3_embed"][f"M {m} = reference"] = same(got, rows)
            checks["k3_embed"][f"M {m} = eager"] = same(got, embed(ids))
            if with_norm:
                got = outs["k3_embed_norm", m]
                eager, eager_bank = embed_norm(ids)
                checks["k3_embed_norm"].update({
                    f"M {m} normed = reference": same(got, norm(rows)),
                    f"M {m} raw = reference": same(banks[m][0], rows),
                    f"M {m} = eager": same(got, eager) and same(banks[m], eager_bank),
                })  # fmt: skip
        for name, op_checks in checks.items():
            if op_checks:
                out.append(result(name, family, f"replay {rep}: {kind} ids rewritten", op_checks))
    first = {k: v.clone() for k, v in outs.items()}
    first_banks = {m: b.clone() for m, b in banks.items()}
    for b in banks.values():
        b.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    for name in ("k3_embed", "k3_embed_norm") if with_norm else ("k3_embed",):
        rerun = {f"M {m}": same(outs[name, m], first[name, m]) for m in ms}
        if name == "k3_embed_norm":
            rerun.update({f"M {m} bank": same(banks[m], first_banks[m]) for m in ms})
        out.append(result(name, family, "the last ids replayed again", rerun))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", TOKENS)
def test_embed(m):
    _op()
    with torch.inference_mode():
        bad = [r for r in measure_embed(m) if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("m", TOKENS)
def test_embed_norm(m):
    _op()
    _need_norm()
    with torch.inference_mode():
        bad = [r for r in measure_norm(m) if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("m", OUTSIDE_TOKENS)
def test_ids_outside_vocab(m, dtype):
    _op()
    with torch.inference_mode():
        bad = [r for r in measure_outside(m, dtype, _flashinfer_norm()) if not r["ok"]]
    assert not bad, bad


def test_every_row():
    _op()
    with torch.inference_mode():
        bad = [r for r in measure_every_row(_flashinfer_norm()) if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("ms", GRAPH_FAMILIES, ids=[f"M{f[0]}-{f[-1]}" for f in GRAPH_FAMILIES])
def test_graph_replay(ms):
    _op()
    with torch.inference_mode():
        bad = [r for r in measure_graph(ms, _flashinfer_norm()) if not r["ok"]]
    assert not bad, bad


def test_supports():
    """The model's calls are supported at every M in 1 .. 64 (hidden 7168, int32 or int64 ids); M = 0 or 65, other id
    dtypes and widths outside the norm's geometry are refused."""
    op = _op()
    with torch.inference_mode():
        tab = table()
        norm = layer0_norm(EPS)
        bank = torch.empty(2, MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def ids(n, dtype=torch.int32):
            return torch.zeros(n, dtype=dtype, device="cuda")

        for m in TOKENS:
            raw = bank[0, :m]
            assert op.supports(ids(m), tab) and op.supports(ids(m, torch.int64), tab), m
            assert op.supports_norm(ids(m), tab, norm.weight, raw), m
        assert op.norm_supports_hidden(HIDDEN)
        assert not op.norm_supports_hidden(6144) and not op.norm_supports_hidden(HIDDEN + 512)
        assert not op.supports(ids(0), tab) and not op.supports(ids(MAX_TOKENS + 1), tab)
        assert not op.supports(ids(8, torch.int16), tab)
        assert not op.supports_norm(ids(8), tab, norm.weight, bank[0])  # raw [64, H] for 8 ids
        with pytest.raises((ValueError, RuntimeError)):
            torch.ops.trtllm.k3_embed(ids(MAX_TOKENS + 1), tab)


# ----------------------------------------------------------------------------------------------------------------
# Error table (python3 test_k3_embed.py report) and timing (python3 test_k3_embed.py time)
# ----------------------------------------------------------------------------------------------------------------


def report() -> int:
    _op()
    with_norm = _flashinfer_norm()
    rows = []
    with torch.inference_mode():
        print(f"{torch.cuda.get_device_name()}; table [{VOCAB}, {HIDDEN}] bf16")
        for m in TOKENS:
            rows += measure_embed(m)
            if with_norm:
                rows += measure_norm(m)
        for m in OUTSIDE_TOKENS:
            for dtype in (torch.int32, torch.int64):
                rows += measure_outside(m, dtype, with_norm)
        rows += measure_every_row(with_norm)
        for ms in GRAPH_FAMILIES:
            rows += measure_graph(ms, with_norm)
    what = {
        "k3_embed": "identical: rows = F.embedding (zero rows for ids outside [0, V)); rerun = first run; in a "
        "graph replay also = the eager op",
        "k3_embed_norm": "identical: normed = the RMSNorm module on the reference rows; raw (the bank slot) = the "
        "reference rows; the bank's other slots untouched; rerun = first run; in a graph replay also = the eager op",
    }
    for name in ("k3_embed", "k3_embed_norm"):
        print(f"\n## {name}\n\n{what[name]}\n")
        if name == "k3_embed_norm" and not with_norm:
            print(
                "skipped: needs flashinfer's CuTe DSL RMSNorm (the kernel k3_embed_norm reproduces)"
            )
            continue
        print("| M | case | identical | result |")
        print("| :-- | :-- | :-- | :-- |")
        for r in (r for r in rows if r["op"] == name):
            verdict = "PASS" if r["ok"] else "FAIL"
            print(f"| {r['m']} | {r['case']} | {r['identical']} | {verdict} |")
    ok_all = all(r["ok"] for r in rows)
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
    _op()
    with_norm = _flashinfer_norm()
    calls = 32
    with torch.inference_mode():
        tab = table()
        norm = layer0_norm(EPS) if with_norm else None
        gen = torch.Generator(device="cuda").manual_seed(11)
        print(f"{torch.cuda.get_device_name()}; table [{VOCAB}, {HIDDEN}] bf16; graphs of {calls} back-to-back calls, "
              "each on its own random int32 ids, 15 replays: median (min-max) us per call")  # fmt: skip
        print("| M | F.embedding | k3_embed | F.embedding + snapshot + RMSNorm | k3_embed + snapshot + RMSNorm "
              "| k3_embed_norm |")  # fmt: skip
        print("| --: | --: | --: | --: | --: | --: |")
        for m in TOKENS:
            idss = [
                torch.randint(0, VOCAB, (m,), generator=gen, device="cuda", dtype=torch.int32)
                for _ in range(calls)
            ]
            bank = new_bank(m)

            def unfused(rows, bank=bank):
                """Layer 0 without k3_embed_norm: the bank's snapshot of the rows and the input RMSNorm."""
                bank[0].copy_(rows)
                return norm(rows)

            arms = [
                lambda i: F.embedding(idss[i], tab),
                lambda i: torch.ops.trtllm.k3_embed(idss[i], tab),
            ]
            if with_norm:
                arms += [
                    lambda i: unfused(F.embedding(idss[i], tab)),
                    lambda i: unfused(torch.ops.trtllm.k3_embed(idss[i], tab)),
                    lambda i: embed_norm_into(idss[i], norm, bank[0]),
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
            cells += ["n/a"] * (5 - len(cells))
            print(f"| {m} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
