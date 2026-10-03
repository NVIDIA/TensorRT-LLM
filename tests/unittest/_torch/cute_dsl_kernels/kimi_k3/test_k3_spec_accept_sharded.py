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
"""``trtllm::k3_spec_accept`` with vocabulary-sharded bf16 target logits (the ranks exchange their row maxima over the
multicast Lamport buffer of ``op.workspace(mapping, push_copies)``) against the same kernel on the gathered fp32 logits,
on W GPUs of one NVLink domain (MPI, one rank per GPU, the W ranks one TP group).

The gathered logits are the bf16 values in fp32 (what the all-gather and ``.float()`` give), so both must agree bit for
bit: every output and every in-place state tensor, on every rank; the gathered kernel is also checked against the torch
op sequence it replaces (``test_k3_spec_accept.reference``). Every rank draws the same logits (checked) and passes its
shard.

Every split B = 1 .. 8 requests x K + 1 in {2, 4, 8} tokens (block K + 1), 6 consecutive calls per case with the state
carried (the exchange's three buffers rotate twice): normal (accepted prefixes 0 .. K mixed over the requests), ties
(across CTAs and shard boundaries), rank_ties (the same maximum in every rank's shard: rank 0's index wins), nan (a
whole row, NaNs in two shards, a NaN in the last shard only: the first NaN wins), negzero (+-0 maxima: -0.0 travels as
+0.0) and forced (integer and fractional values alternating). Then sharded calls of 64, 2, 2 and 64 rows with every
rank but rank 0 launching the last one late (the last call reads the buffer of the first after a 2-row call re-armed
it), a batch that dips and grows back (64, 64, 64, 16, 56 and 64 rows, then 50 calls of a random B x 8 with one random
rank late each call), and CUDA graphs of 3 sharded calls replayed 10 times with the logits, drafts and dummy mask
rewritten in place (the buffer rotation inside a graph).

Shapes: the rank's shard V / W of V = 163840 (W exchange slots), and TP16's 10240-column shard with every rank filling
16 / W slots (the exchange of 16 ranks; ``--copies`` overrides).

    srun -n W --mpi=pmix python3 test_k3_spec_accept_sharded.py [report] [--copies C] [--time [--base]]

``--time`` also reports us per call at batch 1 and every split of the sharded kernel and of the kernel on the gathered
fp32 logits (CUDA graphs of 20 back-to-back calls, the slowest rank of each replay, median over 12 replays in
alternating order; the all-gather, cat and cast that the sharded path removes are not in the second number);
``--base`` adds the installed base package's kernel (``$K3_BASE_TRTLLM``) on the same sharded calls and workspace.
Under pytest (``python3 -m pytest test_k3_spec_accept_sharded.py``, 4 GPUs visible) one test runs the checks on 4
ranks: this file under a local ``mpirun -n 4``, started from a fresh interpreter, under a deadline; with fewer GPUs the
module is skipped.
"""

import argparse
import contextlib
import importlib.util
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback

import pytest
import torch

__extra_import_path__ = ["."]
from test_k3_spec_accept import (  # noqa: E402  (the single-GPU test's state, inputs, reference and tallies)
    FORCES,
    OUTPUTS,
    SPLITS,
    STATE_KEYS,
    TP16_COLUMNS,
    V,
    _op,
    _sm100,
    cell,
    clone_state,
    drafts_for,
    embedding,
    finish,
    fused,
    make_state,
    new_result,
    record,
    reference,
    set_dummies,
    shape_logits,
    state_of,
    tally,
)

STEPS = 6
# (name, logits kind)
CASES = (
    ("normal", "plain"),
    ("ties", "ties"),
    ("rank_ties", "rank_ties"),
    ("nan", "nan"),
    ("negzero", "negzero"),
    ("forced", "plain"),
)
GRAPH_SPLITS = [(1, 2), (2, 8), (4, 4), (8, 8)]
GRAPH_KINDS = ("plain", "ties", "rank_ties", "nan", "negzero")


# The ranks of the pytest run (one GB200 tray) and its deadline: a broken exchange hangs rather than raising.
WORLD = 4
DEADLINE_S = 1200

pytestmark = pytest.mark.skipif(
    not _sm100() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} SM100 GPUs (MNNVL multicast)",
)

_env = {}


def mpi_env() -> dict:
    """The job's communicator, rank and world, the op and a TP mapping of every rank, set up once (the device: the
    rank's index on its node)."""
    if not _env:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        op = _op()
        from tensorrt_llm.mapping import Mapping

        mapping = Mapping(
            world_size=world, rank=rank, gpus_per_node=torch.cuda.device_count(), tp_size=world
        )
        _env.update(comm=comm, rank=rank, world=world, op=op, mapping=mapping)
    return _env


def on_every_rank(env: dict, fn, *args):
    """``fn(*args)``, run by every rank; an exception on one rank aborts the job, whose other ranks would otherwise
    wait forever in the exchange or in the next collective."""
    try:
        return fn(*args)
    # Whatever failed: report it, then take the whole job down rather than leave it hanging.
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        env["comm"].Abort(1)
        raise


def configs(world: int, copies=None) -> list:
    """(name, total columns, push copies): every rank's shard of V, and TP16's shard with 16 / W slots per rank."""
    return [("V / W", V, 1), ("TP16 shard", TP16_COLUMNS * world, copies or max(1, 16 // world))]


def shard_of(env: dict, vocab: int):
    """(columns per rank, this rank's first column)."""
    shard = vocab // env["world"]
    return shard, env["rank"] * shard


def same_logits_everywhere(env: dict, x: torch.Tensor) -> bool:
    """Whether every rank drew the same logits (the shards are slices of one tensor only then): row checksums."""
    sums = x.view(torch.int16).sum(dim=1, dtype=torch.int64).tolist()
    return all(other == sums for other in env["comm"].allgather(sums))


def sharded_state(gen, batch: int, vocab: int) -> dict:
    st = make_state(gen, batch)
    st["embed"] = embedding()[:vocab]  # the drafter embedding covers the vocabulary
    return st


def base_kernel():
    """The unmodified kernel module of the installed base package (``$K3_BASE_TRTLLM``), for before/after timing."""
    path = os.path.join(os.environ["K3_BASE_TRTLLM"], "_torch", "cute_dsl_kernels", "k3_spec_accept",
                        "k3_spec_accept_kernel.py")  # fmt: skip
    spec = importlib.util.spec_from_file_location("k3_spec_accept_kernel_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rearm(env: dict, ws: dict) -> None:
    """Every rank's Lamport words empty again, with no call in flight on any rank (between the base and the new
    kernel on one workspace: the base kernel re-arms only its own call's rows)."""
    torch.cuda.synchronize()
    env["comm"].Barrier()
    ws["uc"].fill_(env["op"]._kernel_module().EMPTY_WORD)
    torch.cuda.synchronize()
    env["comm"].Barrier()


@contextlib.contextmanager
def kernel_module(op, module, compiled: dict):
    """The op's calls inside run ``module``'s kernel, compiled into ``compiled``."""
    saved = op._modules.get("kernel"), op._compiled
    op._modules["kernel"], op._compiled = module, compiled
    try:
        yield
    finally:
        op._modules["kernel"], op._compiled = saved


# ----------------------------------------------------------------------------------------------------------------
# Checks (every rank runs the same calls in the same order: the exchange and the collectives need it)
# ----------------------------------------------------------------------------------------------------------------


def check_cases(env: dict, ws: dict, vocab: int, config: str, steps: int = STEPS) -> list:
    """Every split x case: ``steps`` calls with the state carried; the sharded kernel against the gathered one, and the
    gathered one against the torch reference."""
    shard, lo = shard_of(env, vocab)
    results = []
    for batch, tokens in SPLITS:
        drafts = tokens - 1
        for index, (name, kind) in enumerate(CASES):
            # One seed on every rank: the same logits, drafts and state everywhere.
            seed = 4000 + 100 * batch + 10 * tokens + index
            gen = torch.Generator(device="cuda").manual_seed(seed)
            st_sh = sharded_state(gen, batch, vocab)
            st_g, st_t = clone_state(st_sh), clone_state(st_sh)
            res = new_result(config=config, split=f"{batch}x{tokens}", case=name)
            for step in range(steps):
                x = torch.randn(batch * tokens, vocab, generator=gen, device="cuda").bfloat16()
                shape_logits(x, gen, kind, step, env["world"])
                draft = drafts_for(gen, x, batch, drafts, step)
                force = FORCES[drafts][step % 2] if name == "forced" else 0.0
                set_dummies((st_sh, st_g, st_t), batch, step)
                tally(res, "logits", same_logits_everywhere(env, x), step)
                x_shard, x32 = x[:, lo : lo + shard].contiguous(), x.float()
                got = fused(st_sh, x_shard, draft, force, tokens, (ws, lo))
                ref = fused(st_g, x32, draft, force, tokens)
                want = reference(st_t, x32, draft, force, tokens)
                record(res, step, {
                    "outputs": (OUTPUTS, got, ref),
                    "state": (STATE_KEYS, state_of(st_sh), state_of(st_g)),
                    "torch": (OUTPUTS + STATE_KEYS, ref + state_of(st_g), want + state_of(st_t)),
                })  # fmt: skip
            results.append(finish(res))
    return results


def check_transition(env: dict, ws: dict, vocab: int, config: str) -> dict:
    """Sharded calls of 64, 2, 2 and 64 rows. The 4th call uses the buffer the 1st filled (row maxima of 50.0), which
    the 2nd re-armed for its own 2 rows; every rank but rank 0 launches the 4th call 50 ms late, so rank 0 must wait for
    its peers' pushes in every row rather than take what the 1st call left."""
    comm = env["comm"]
    shard, lo = shard_of(env, vocab)
    gen = torch.Generator(device="cuda").manual_seed(4999)
    states = {}
    for batch in (8, 1):
        st = sharded_state(gen, batch, vocab)
        states[batch] = (st, clone_state(st))
    res = new_result(config=config, split="8x8, 1x2", case="rows 64, 2, 2, 64 (peers late)")
    for call, (batch, tokens) in enumerate(((8, 8), (1, 2), (1, 2), (8, 8))):
        rows = batch * tokens
        x = torch.randn(rows, vocab, generator=gen, device="cuda").bfloat16()
        if call == 0:
            r = torch.arange(rows, device="cuda")
            x[r, vocab - 1 - r] = 50.0
        draft = drafts_for(gen, x, batch, tokens - 1, call)
        tally(res, "logits", same_logits_everywhere(env, x), call)
        st_sh, st_g = states[batch]
        torch.cuda.synchronize()
        comm.Barrier()
        if call == 3 and env["rank"] > 0:
            time.sleep(0.05)
        got = fused(st_sh, x[:, lo : lo + shard].contiguous(), draft, 0.0, tokens, (ws, lo))
        ref = fused(st_g, x.float(), draft, 0.0, tokens)
        record(res, call, {
            "outputs": (OUTPUTS, got, ref),
            "state": (STATE_KEYS, state_of(st_sh), state_of(st_g)),
        })  # fmt: skip
    torch.cuda.synchronize()
    comm.Barrier()
    return finish(res)


def check_dip_regrow(env: dict, ws: dict, vocab: int, config: str, calls: int = 50) -> dict:
    """A batch that dips and grows back: sharded calls of 8, 8, 8, 2, 7 and 8 requests x 8 tokens, then ``calls``
    calls of a random B x 8, one random rank launching each of those 20 ms late (the same draws on every rank). Every
    row of call c peaks at 100 - c in a random column, so a word left from an earlier call outranks the fresh ones."""
    comm = env["comm"]
    shard, lo = shard_of(env, vocab)
    gen = torch.Generator(device="cuda").manual_seed(5999)
    draws = random.Random(5999)
    tokens = 8
    states = {}
    for batch in range(1, 9):
        st = sharded_state(gen, batch, vocab)
        states[batch] = (st, clone_state(st))
    batches = [8, 8, 8, 2, 7, 8] + [draws.randint(1, 8) for _ in range(calls)]
    late = [None] * 6 + [draws.randrange(env["world"]) for _ in range(calls)]
    res = new_result(config=config, split="B x 8, B varying",
                     case=f"rows 64, 64, 64, 16, 56, 64 + {calls} random (one rank late)")  # fmt: skip
    for call, batch in enumerate(batches):
        rows = batch * tokens
        x = torch.randn(rows, vocab, generator=gen, device="cuda").bfloat16()
        peak = torch.randint(0, vocab, (rows,), generator=gen, device="cuda")
        x[torch.arange(rows, device="cuda"), peak] = 100.0 - call
        draft = drafts_for(gen, x, batch, tokens - 1, call)
        tally(res, "logits", same_logits_everywhere(env, x), call)
        st_sh, st_g = states[batch]
        torch.cuda.synchronize()
        comm.Barrier()
        if late[call] == env["rank"]:
            time.sleep(0.02)
        got = fused(st_sh, x[:, lo : lo + shard].contiguous(), draft, 0.0, tokens, (ws, lo))
        ref = fused(st_g, x.float(), draft, 0.0, tokens)
        record(res, call, {
            "outputs": (OUTPUTS, got, ref),
            "state": (STATE_KEYS, state_of(st_sh), state_of(st_g)),
        })  # fmt: skip
    torch.cuda.synchronize()
    comm.Barrier()
    return finish(res)


def check_graph(env: dict, ws: dict, vocab: int, config: str, batch: int, tokens: int, calls: int = 3,
                replays: int = 10) -> dict:  # fmt: skip
    """``calls`` sharded calls captured in one CUDA graph, replayed ``replays`` times with the logits (every kind),
    drafts and dummy mask rewritten in place: every call of every replay against the gathered kernel."""
    comm = env["comm"]
    shard, lo = shard_of(env, vocab)
    drafts, rows = tokens - 1, batch * tokens
    gen = torch.Generator(device="cuda").manual_seed(5000 + 10 * batch + tokens)
    st_sh = sharded_state(gen, batch, vocab)
    st_g = clone_state(st_sh)
    x_buf = torch.zeros(rows, shard, dtype=torch.bfloat16, device="cuda")
    d_buf = torch.zeros(batch, drafts, dtype=torch.int32, device="cuda")
    # Compiled outside capture (an exchange: every rank makes this call).
    fused(clone_state(st_sh), x_buf, d_buf, 0.0, tokens, (ws, lo))
    torch.cuda.synchronize()
    comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [fused(st_sh, x_buf, d_buf, 0.0, tokens, (ws, lo)) for _ in range(calls)]
    torch.cuda.synchronize()
    res = new_result(
        config=config, split=f"{batch}x{tokens}", case=f"graph of {calls} calls x {replays}"
    )
    for rep in range(replays):
        x = torch.randn(rows, vocab, generator=gen, device="cuda").bfloat16()
        shape_logits(x, gen, GRAPH_KINDS[rep % len(GRAPH_KINDS)], rep, env["world"])
        draft = drafts_for(gen, x, batch, drafts, rep)
        set_dummies((st_sh, st_g), batch, rep)
        tally(res, "logits", same_logits_everywhere(env, x), rep)
        x_buf.copy_(x[:, lo : lo + shard])
        d_buf.copy_(draft)
        graph.replay()
        x32 = x.float()
        for c in range(calls):
            record(res, rep, {"outputs": (OUTPUTS, outs[c], fused(st_g, x32, draft, 0.0, tokens))})
        record(res, rep, {"state": (STATE_KEYS, state_of(st_sh), state_of(st_g))})
    torch.cuda.synchronize()
    comm.Barrier()
    del graph
    return finish(res)


def time_split(env: dict, ws: dict, vocab: int, config: str, batch: int, tokens: int, calls: int = 20,
               rounds: int = 12, base=None) -> dict:  # fmt: skip
    """us per call at one split of the sharded kernel and of the kernel on the gathered fp32 logits (and of the
    ``base`` kernel module on the sharded logits, twice: the second graph is the noise control): CUDA graphs of
    ``calls`` back-to-back calls, the slowest rank of each replay, median over ``rounds`` replays in alternating
    order."""
    comm = env["comm"]
    shard, lo = shard_of(env, vocab)
    gen = torch.Generator(device="cuda").manual_seed(6000 + 10 * batch + tokens)
    st_sh = sharded_state(gen, batch, vocab)
    # The CUDA-graph padding requests (one KDA slot) are dummies, as in the model: no two lanes write one slot.
    set_dummies((st_sh,), batch, 0)
    st_g = clone_state(st_sh)
    x = torch.randn(batch * tokens, vocab, generator=gen, device="cuda").bfloat16()
    x_shard, x32 = x[:, lo : lo + shard].contiguous(), x.float()
    draft = drafts_for(gen, x, batch, tokens - 1, 3)
    arms = [
        lambda: fused(st_sh, x_shard, draft, 0.0, tokens, (ws, lo)),
        lambda: fused(st_g, x32, draft, 0.0, tokens),
    ]
    identical = None
    if base is not None:
        rearm(env, ws)
        compiled = env.setdefault("base_compiled", {})
        st_b, st_b2 = clone_state(st_sh), clone_state(st_sh)

        def base_arm(st=st_b):
            with kernel_module(env["op"], base, compiled):
                return fused(st, x_shard, draft, 0.0, tokens, (ws, lo))

        # Identity: one call of each kernel on the same state and logits, every output and state tensor.
        st_new, st_old = clone_state(st_sh), clone_state(st_sh)
        new = fused(st_new, x_shard, draft, 0.0, tokens, (ws, lo))
        with kernel_module(env["op"], base, compiled):
            old = fused(st_old, x_shard, draft, 0.0, tokens, (ws, lo))
        identical = all(
            torch.equal(a, b) for a, b in zip(new + state_of(st_new), old + state_of(st_old))
        )
        identical = all(comm.allgather(identical))
        arms += [base_arm, lambda: base_arm(st_b2)]
    graphs = []
    for body in arms:
        body()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                body()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)
    per_call = [[] for _ in arms]
    for rd in range(rounds):
        for arm in range(len(arms)) if rd % 2 == 0 else reversed(range(len(arms))):
            comm.Barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[arm].replay()
            end.record()
            torch.cuda.synchronize()
            per_call[arm].append(max(comm.allgather(start.elapsed_time(end) * 1e3 / calls)))
    comm.Barrier()
    return dict(config=config, split=f"{batch}x{tokens}", sharded=statistics.median(per_call[0]),
                gathered=statistics.median(per_call[1]),
                base=statistics.median(per_call[2]) if base is not None else None,
                base2=statistics.median(per_call[3]) if base is not None else None, identical=identical)  # fmt: skip


def run_config(env: dict, name: str, vocab: int, copies: int, time_it: bool = False, base=None, rounds: int = 12,
               time_splits=None, checks: bool = True):  # fmt: skip
    """Every check of one shape (and its timing with ``time_it``): (skip reason or None, results, timings)."""
    world = env["world"]
    shard, slots = vocab // world, world * copies
    config = f"{name}: {shard} columns x {world} ranks, {slots} slots"
    if vocab % world or not env["op"].supports_columns(shard, slots):
        reason = f"the kernel does not split {vocab} columns over {world} ranks of {copies} slots"
        return reason, [], []
    try:
        ws = env["op"].workspace(env["mapping"], copies)
    # Raised on every rank: the ranks agree on the allocation's outcome.
    except RuntimeError as exc:
        return f"no multicast workspace ({exc})", [], []
    results = []
    if checks:
        results = check_cases(env, ws, vocab, config)
        results.append(check_transition(env, ws, vocab, config))
        results.append(check_dip_regrow(env, ws, vocab, config))
        results += [check_graph(env, ws, vocab, config, b, t) for b, t in GRAPH_SPLITS]
    timings = []
    if time_it:
        for b, t in time_splits or SPLITS:
            timings.append(time_split(env, ws, vocab, config, b, t, rounds=rounds, base=base))
    return None, results, timings


# ----------------------------------------------------------------------------------------------------------------
# pytest (python3 -m pytest test_k3_spec_accept_sharded.py): the checks on WORLD ranks of this node.
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.no_xdist
def test_k3_spec_accept_sharded():
    """``report`` on WORLD ranks: ``launch`` in a fresh interpreter, which starts this file under ``mpirun`` (a
    process that has initialized MPI, as pytest's has, cannot start mpirun)."""
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    devices = (visible or [str(i) for i in range(torch.cuda.device_count())])[:WORLD]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "launch"],
        env=env,
        capture_output=True,
        text=True,
        timeout=DEADLINE_S + 300,
    )
    print(done.stdout, flush=True)
    assert done.returncode == 0 and "ALL PASS" in done.stdout, (
        done.stdout[-20000:] + done.stderr[-20000:]
    )


def launch() -> int:
    """This file's ``report`` under ``mpirun -n WORLD`` (one rank per visible device), killed with its process
    group at the deadline."""
    command = ["mpirun", "-n", str(WORLD), sys.executable, os.path.abspath(__file__), "report"]
    print(f"[launch] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(
            f"[launch] the {WORLD}-rank run did not finish in {DEADLINE_S} s (wedged)", flush=True
        )
        return 1


# ----------------------------------------------------------------------------------------------------------------
# srun -n W --mpi=pmix python3 test_k3_spec_accept_sharded.py [report] [--copies C] [--time]
# ----------------------------------------------------------------------------------------------------------------


def print_report(per_rank: list, timings: list, skips: list, world: int) -> None:
    """The checks as a markdown table (every count the fewest over the ranks), then the timing."""
    print(f"{torch.cuda.get_device_name()} x {world} ranks; {STEPS} consecutive calls per case, state carried; "
          "counts: the fewest over the ranks")  # fmt: skip
    print("| config | split | case | outputs identical (sharded = gathered) | state identical | gathered = torch "
          "| same logits on every rank | result |")  # fmt: skip
    print("| :-- | :-- | :-- | --: | --: | --: | --: | :-- |")
    for rows in zip(*per_rank):
        checks = {
            k: (min(r["checks"][k][0] for r in rows), total)
            for k, (_, total) in rows[0]["checks"].items()
        }
        res = dict(rows[0], checks=checks)
        failed = [(rank, r["bad"]) for rank, r in enumerate(rows) if not r["ok"]]
        print(f"| {res['config']} | {res['split']} | {res['case']} | {cell(res, 'outputs')} | {cell(res, 'state')} | "
              f"{cell(res, 'torch')} | {cell(res, 'logits')} | {'FAIL ' + str(failed[:2]) if failed else 'PASS'} |",
              flush=True)  # fmt: skip
    for skip in skips:
        print(f"skipped: {skip}")
    if timings:
        print(
            "\nus per call: the slowest rank, median over the replays of a CUDA graph of 20 calls (alternating order)"
        )
        with_base = timings[0]["base"] is not None
        print("| config | split | sharded (bf16 shard + exchange) | gathered fp32 logits |"
              + (" sharded, base kernel | sharded - base | base2 - base | outputs = base |"
                 if with_base else ""))  # fmt: skip
        print("| :-- | :-- | --: | --: |" + (" --: | --: | --: | :-- |" if with_base else ""))
        for t in timings:
            extra = (f" {t['base']:.2f} | {t['sharded'] - t['base']:+.2f} | {t['base2'] - t['base']:+.2f} | "
                     f"{t['identical']} |" if with_base else "")  # fmt: skip
            print(
                f"| {t['config']} | {t['split']} | {t['sharded']:.2f} | {t['gathered']:.2f} |{extra}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="trtllm::k3_spec_accept: vocabulary-sharded vs gathered logits"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("report", "launch"),
        help="the checks (the default), or the checks on WORLD ranks under a local mpirun (launch; the pytest form)",
    )
    parser.add_argument("--copies", type=int, default=None,
                        help="exchange slots every rank fills in the TP16 shape (default 16 / W)")  # fmt: skip
    parser.add_argument("--time", action="store_true",
                        help="also time the sharded kernel and the kernel on the gathered fp32 logits")  # fmt: skip
    parser.add_argument("--base", action="store_true",
                        help="with --time: also the base package's kernel ($K3_BASE_TRTLLM), sharded")  # fmt: skip
    parser.add_argument("--rounds", type=int, default=12, help="with --time: replays per arm")
    parser.add_argument(
        "--splits", default=None, help="with --time: only these splits, e.g. 1x2,1x8"
    )
    parser.add_argument("--time-only", action="store_true", help="the timing without the checks")
    args = parser.parse_args()
    if args.mode == "launch":
        return launch()
    time_splits = None
    if args.splits:
        time_splits = [tuple(int(v) for v in sp.split("x")) for sp in args.splits.split(",")]
    env = mpi_env()
    comm, world = env["comm"], env["world"]
    if world < 2:
        print(
            "test_k3_spec_accept_sharded: needs an MPI job of 2 or more ranks (srun -n W --mpi=pmix ...)"
        )
        return 1
    results, timings, skips = [], [], []
    with torch.inference_mode():
        for name, vocab, copies in configs(world, args.copies):
            skip, res, tim = on_every_rank(env, run_config, env, name, vocab, copies, args.time or args.time_only,
                                           base_kernel() if args.base else None, args.rounds, time_splits,
                                           not args.time_only)  # fmt: skip
            if skip:
                skips.append(f"{name}: {skip}")
            results += res
            timings += tim
    per_rank = comm.gather(results, root=0)
    oks = comm.allgather(all(r["ok"] for r in results))
    passed = all(oks) and not skips
    if env["rank"] == 0:
        print_report(per_rank, timings, skips, world)
        print(
            "ALL PASS" if passed else f"FAIL (ranks ok: {oks}; skipped: {len(skips)})", flush=True
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
