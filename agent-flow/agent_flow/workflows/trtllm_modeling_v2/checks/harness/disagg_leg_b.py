#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leg B of the equivalence witness: leg A's engine split into a context and a generation engine.

Usage, as the witness runs it when its block names no ``contrast_command``
(one process that holds twice the member's devices, under the tree's
interpreter)::

    <python> disagg_leg_b.py

and as that process starts each of its two engines::

    <engine launch> disagg_leg_b.py engine --role ctx|gen

Everything it needs comes from the ``MV2FLOW_WITNESS_*`` environment the
witness hands any leg B (``checks/witness.py``), so it restates nothing of
leg A's workload.

**The coordinator** (no arguments) splits the devices it can see in two --
the first half for the context engine, the second for the generation
engine, each as many as leg A's argv has ranks (``--tp_size`` x
``--pp_size``) -- and refuses with fewer. It starts both engines at once,
each with :data:`ENGINE_LAUNCH_ENV` (``mpirun ... trtllm-llmapi-launch``
above one rank, as the witness's own environment launches engine work), and
waits for each to connect back. Then, per prompt of
:data:`PROMPTS_FILE_ENV`, it sends a context-only request to the context
engine and hands the ``DisaggregatedParams`` that comes back, as a
generation-only request with the prompt's token ids, to the generation
engine. With :data:`RANK_PAIRS_ENV` set, prompt ``i``'s two requests are
pinned to the ``i``-th (context, generation) attention-DP rank pair with
``SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)``. At
``max_batch_size: 1`` (:data:`MAX_BATCH_SIZE_ENV`) one prompt crosses at a
time; otherwise every prompt goes to the context engine at once, then every
hand-over to the generation engine. Once both engines have shut down it
prints, per prompt, the lines a quickstart leg prints and the witness parses
(:func:`format_hop`). It exits 1 when any prompt failed, and 2 when it could
not run at all.

**The engines** are built from leg A's argv by the tree's own
``examples/llm-api/quickstart_advanced.py`` (``parse_arguments`` and
``setup_llm``), with what the gates add to every quickstart engine
(``engine_driver.engine_kwargs``: the ``modeling_v2`` target and the unit's
KV-cache keys) and the cache transceiver of :data:`TRANSCEIVER_ENV`
(:data:`DEFAULT_TRANSCEIVER` when the block declares none). The context
engine runs without the overlap scheduler, as the disaggregated-serving
documentation's context servers do. Each engine receives the block's
``legs_env`` (:data:`LEGS_ENV_ENV`) last.

**The ablation.** Under :data:`ABLATE_ENV` = :data:`ABLATE_STATE` the
generation engine starts with a ``sitecustomize`` (:data:`SITECUSTOMIZE`)
first on its ``PYTHONPATH``, so every process of it, each MPI rank
included, installs :func:`install_ablation_hook`. Where the executor takes
up a generation-only request whose transfer completed
(``PyExecutor._prepare_disagg_gen_transmission_complete``), before its
first decode step, the hook destroys what the request received
(:func:`resolve_ablation`): the conv and SSM state of every recurrent layer
at the request's slot, else -- a model with no recurrent state -- its KV
pages in every attention layer. It prints :data:`ABLATED` once per request
and process. A cache manager it has no way to reach is refused when the
executor is built, with :data:`REFUSED`, so the engine never comes up.

Standard library only at import time: the ``sitecustomize`` imports this file
in every process of the generation engine. ``tensorrt_llm`` and the sibling
harness modules are imported where an engine is built.
"""

from __future__ import annotations

import argparse
import copy
import importlib.abc
import json
import os
import pickle
import queue
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: What the witness hands a leg B (``checks/witness.py``, ``LEG_B_*_ENV``);
#: named again here because nothing in this package imports ``agent_flow``.
PROMPTS_FILE_ENV = "MV2FLOW_WITNESS_PROMPTS_FILE"
MAX_TOKENS_ENV = "MV2FLOW_WITNESS_MAX_TOKENS"
MEMBER_ENV = "MV2FLOW_WITNESS_MEMBER"
ARGV_ENV = "MV2FLOW_WITNESS_LEG_A_ARGV"
TIMEOUT_ENV = "MV2FLOW_WITNESS_TIMEOUT_S"
MAX_BATCH_SIZE_ENV = "MV2FLOW_WITNESS_MAX_BATCH_SIZE"
LEGS_ENV_ENV = "MV2FLOW_WITNESS_LEGS_ENV"
RANK_PAIRS_ENV = "MV2FLOW_WITNESS_RANK_PAIRS"
ABLATE_ENV = "MV2FLOW_WITNESS_ABLATE"
ABLATE_STATE = "state"
ENGINE_LAUNCH_ENV = "MV2FLOW_WITNESS_ENGINE_LAUNCH"
TRANSCEIVER_ENV = "MV2FLOW_WITNESS_TRANSCEIVER"
FULL_OUTPUT_ENV = "TRTLLM_WITNESS_FULL_OUTPUT"

#: Where an engine connects back to the coordinator, and the key it proves
#: itself with; set by the coordinator on each engine it starts.
ADDRESS_ENV = "MV2FLOW_LEG_B_ADDRESS"
AUTHKEY_ENV = "MV2FLOW_LEG_B_AUTHKEY"

CTX, GEN = "ctx", "gen"
ROLES = (CTX, GEN)
ENGINE_NAMES = {CTX: "context", GEN: "generation"}

#: The cache transceiver both engines run when the witness block declares
#: none: the backend TensorRT-LLM resolves by default, and the transceiver
#: runtime the model prefers.
DEFAULT_TRANSCEIVER: dict[str, Any] = {"backend": "DEFAULT", "transceiver_runtime": "auto"}

#: What the generation engine prints for each request whose received state it
#: destroyed, and what it prints when it cannot destroy it.
ABLATED = "Witness ablated:"
REFUSED = "Witness ablation refused:"

#: How long engines get to exit once told to, before they are terminated.
SHUTDOWN_GRACE_S = 120.0


class LegBError(RuntimeError):
    """Leg B cannot run as handed; the message says why."""


class AblationRefused(RuntimeError):
    """The generation engine's cache cannot be reached to destroy what a request received."""


# ------------------------------------------------------------ the workload


@dataclass(frozen=True)
class Workload:
    """What leg B runs, read from the witness's environment (:func:`read_workload`)."""

    prompts: tuple[str, ...]
    quickstart: str
    argv: tuple[str, ...]
    max_tokens: int | None = None
    pairs: tuple[tuple[int, int], ...] = ()
    ablate: bool = False
    one_at_a_time: bool = False
    legs_env: Mapping[str, str] = field(default_factory=dict)
    transceiver: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_TRANSCEIVER))
    launch: str = ""
    timeout_s: float = 3600.0
    full_output: bool = False
    member: str = ""


def parse_rank_pairs(text: str) -> tuple[tuple[int, int], ...]:
    """``c:g,c:g,...`` (:data:`RANK_PAIRS_ENV`) as (context rank, generation rank) pairs."""
    pairs = []
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        ctx, sep, gen = entry.partition(":")
        if not sep or not ctx.strip().isdigit() or not gen.strip().isdigit():
            raise LegBError(f"{RANK_PAIRS_ENV}={text!r}: {entry!r} is not `<ctx rank>:<gen rank>`")
        pairs.append((int(ctx), int(gen)))
    return tuple(pairs)


def _json_mapping(environ: Mapping[str, str], name: str) -> dict[str, Any] | None:
    text = environ.get(name, "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise LegBError(f"{name} is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LegBError(f"{name} is not a JSON object: {text[:200]!r}")
    return value


def read_workload(environ: Mapping[str, str]) -> Workload:
    """The workload the witness handed over, or :class:`LegBError` naming what is missing."""
    missing = [name for name in (PROMPTS_FILE_ENV, ARGV_ENV) if not environ.get(name)]
    if missing:
        raise LegBError(
            f"{', '.join(missing)} not set: this is leg B of the equivalence witness, which "
            f"hands it leg A's workload; run it through `trtllm-modeling-v2 gate witness`"
        )
    path = Path(environ[PROMPTS_FILE_ENV])
    try:
        prompts = json.loads(path.read_text(encoding="utf-8")).get("prompts")
    except (OSError, ValueError, AttributeError) as exc:
        raise LegBError(f"cannot read the prompts file {path}: {exc}") from exc
    if not isinstance(prompts, list) or not prompts:
        raise LegBError(f"the prompts file {path} holds no prompts")
    words = shlex.split(environ[ARGV_ENV])
    if not words or not words[0].endswith(".py"):
        raise LegBError(f"{ARGV_ENV} does not start with the quickstart: {environ[ARGV_ENV]!r}")
    pairs = parse_rank_pairs(environ.get(RANK_PAIRS_ENV, ""))
    if pairs and len(pairs) != len(prompts):
        raise LegBError(
            f"{RANK_PAIRS_ENV} names {len(pairs)} rank pair(s) for {len(prompts)} prompt(s); "
            f"the witness hands one per prompt"
        )
    ablate = environ.get(ABLATE_ENV, "").strip()
    if ablate and ablate != ABLATE_STATE:
        raise LegBError(
            f"{ABLATE_ENV}={ablate!r}: the one ablation leg B performs is {ABLATE_STATE}"
        )
    transceiver = _json_mapping(environ, TRANSCEIVER_ENV)
    legs_env = _json_mapping(environ, LEGS_ENV_ENV) or {}
    max_tokens = environ.get(MAX_TOKENS_ENV, "").strip()
    batch = environ.get(MAX_BATCH_SIZE_ENV, "").strip()
    timeout = environ.get(TIMEOUT_ENV, "").strip()
    return Workload(
        prompts=tuple(str(p) for p in prompts),
        quickstart=words[0],
        argv=tuple(words[1:]),
        max_tokens=int(max_tokens) if max_tokens else None,
        pairs=pairs,
        ablate=ablate == ABLATE_STATE,
        one_at_a_time=batch == "1",
        legs_env={str(k): str(v) for k, v in legs_env.items()},
        transceiver=dict(DEFAULT_TRANSCEIVER if transceiver is None else transceiver),
        launch=environ.get(ENGINE_LAUNCH_ENV, "").strip(),
        timeout_s=float(timeout) if timeout else 3600.0,
        full_output=environ.get(FULL_OUTPUT_ENV) == "1",
        member=environ.get(MEMBER_ENV, ""),
    )


def engine_ranks(argv: Sequence[str]) -> int:
    """The ranks one engine built from leg A's quickstart ``argv`` has: tensor x pipeline."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    known, _ = parser.parse_known_args(list(argv))
    return max(1, known.tp_size) * max(1, known.pp_size)


def count_devices() -> int:
    """The GPUs ``nvidia-smi`` lists here; 0 when it cannot say."""
    try:
        listed = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in listed.stdout.splitlines() if line.startswith("GPU "))


def visible_devices(
    environ: Mapping[str, str], count: Callable[[], int] = count_devices
) -> list[str]:
    """The devices this process may hand out: ``CUDA_VISIBLE_DEVICES``, else every GPU here."""
    named = environ.get("CUDA_VISIBLE_DEVICES")
    if named is not None and named.strip():
        return [d.strip() for d in named.split(",") if d.strip()]
    return [str(i) for i in range(count())]


def split_devices(devices: Sequence[str], ranks: int) -> tuple[list[str], list[str]]:
    """The context engine's devices and the generation engine's: ``ranks`` each, in order."""
    if len(devices) < 2 * ranks:
        raise LegBError(
            f"leg B starts a context and a generation engine of {ranks} rank(s) each, so it "
            f"needs {2 * ranks} devices and sees {len(devices)} ({','.join(devices) or 'none'}). "
            f"The witness asks for the whole node; a member wider than half of it has no "
            f"disaggregated leg here"
        )
    return list(devices[:ranks]), list(devices[ranks : 2 * ranks])


# ------------------------------------------------------- the coordinator side


@dataclass
class Hop:
    """One prompt across the hop: what each engine answered, or why it did not."""

    index: int
    prompt: str
    pair: tuple[int, int] | None = None
    #: The context rank the context engine reports serving (``ctx_dp_rank``).
    ctx_rank: int | None = None
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    error: str = ""


def engine_environment(
    base: Mapping[str, str],
    *,
    role: str,
    devices: Sequence[str],
    address: tuple[str, int],
    authkey: bytes,
    workload: Workload,
    site: Path | None = None,
) -> dict[str, str]:
    """The environment one engine starts under.

    The coordinator's own, then the block's ``legs_env``, then what places the
    engine: its devices, where it connects back and, for the generation
    engine of an ablation run, the ``sitecustomize`` in ``site`` first on
    ``PYTHONPATH``. The context engine is not handed the ablation at all.
    """
    env = {**base, **workload.legs_env}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env[ADDRESS_ENV] = f"{address[0]}:{address[1]}"
    env[AUTHKEY_ENV] = authkey.hex()
    if role == GEN and workload.ablate and site is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(site), env.get("PYTHONPATH")]))
    else:
        env.pop(ABLATE_ENV, None)
    return env


def engine_command(launch: str, role: str, python: str = sys.executable) -> str:
    """The shell command that starts one engine: ``launch`` (else this interpreter) and this script."""
    prefix = launch or shlex.quote(python)
    return f"{prefix} {shlex.quote(str(SCRIPT))} engine --role {role}"


class EngineClient:
    """The coordinator's end of one engine: send it requests, wait for its answers."""

    def __init__(self, role: str, process: Any, conn: Connection) -> None:
        self.role, self.process, self.conn = role, process, conn

    @property
    def name(self) -> str:
        return f"the {ENGINE_NAMES[self.role]} engine"

    def call(self, requests: list[dict[str, Any]], deadline: float) -> list[dict[str, Any]]:
        """``requests`` answered, one reply each in order; :class:`LegBError` if the engine is lost."""
        try:
            self.conn.send({"requests": requests})
        except (OSError, EOFError) as exc:
            raise LegBError(f"{self.name} hung up: {exc}") from exc
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LegBError(f"{self.name} did not answer before leg B's deadline")
            if self.conn.poll(min(5.0, remaining)):
                try:
                    return list(self.conn.recv())
                except (OSError, EOFError) as exc:
                    raise LegBError(f"{self.name} hung up: {exc}") from exc
            code = self.process.poll()
            if code is not None:
                raise LegBError(f"{self.name} exited {code} while serving requests")

    def close(self) -> None:
        try:
            self.conn.send(None)
        except (OSError, EOFError, ValueError):
            pass
        self.conn.close()


def _ctx_request(index: int, prompt: str, workload: Workload) -> dict[str, Any]:
    pair = workload.pairs[index] if workload.pairs else None
    return {
        "index": index,
        "prompt": prompt,
        "max_tokens": workload.max_tokens,
        "rank": pair[0] if pair else None,
    }


def _gen_request(hop: Hop, reply: Mapping[str, Any], workload: Workload) -> dict[str, Any]:
    return {
        "index": hop.index,
        "params": reply["params"],
        "prompt_token_ids": list(reply["prompt_token_ids"]),
        "max_tokens": workload.max_tokens,
        "rank": hop.pair[1] if hop.pair else None,
    }


def _take_ctx(hop: Hop, reply: Mapping[str, Any]) -> bool:
    """Record the context engine's ``reply`` on ``hop``; whether it can be handed over."""
    if reply.get("error"):
        hop.error = f"the context-only request failed: {reply['error']}"
        return False
    if reply.get("params") is None or reply.get("prompt_token_ids") is None:
        hop.error = (
            "the context engine returned no disaggregated params: it ran without a cache "
            "transceiver, or the tree does not hand context-only requests over"
        )
        return False
    rank = reply.get("ctx_dp_rank")
    hop.ctx_rank = int(rank) if rank is not None else None
    return True


def _take_gen(hop: Hop, reply: Mapping[str, Any]) -> None:
    if reply.get("error"):
        hop.error = f"the generation-only request failed: {reply['error']}"
        return
    hop.token_ids = [int(t) for t in reply.get("token_ids") or ()]
    hop.text = str(reply.get("text") or "")


def drive(ctx: EngineClient, gen: EngineClient, workload: Workload, deadline: float) -> list[Hop]:
    """Every prompt context-only on ``ctx``, then generation-only on ``gen``; one hop each."""
    hops = [
        Hop(index=i, prompt=p, pair=workload.pairs[i] if workload.pairs else None)
        for i, p in enumerate(workload.prompts)
    ]
    if workload.one_at_a_time:
        for hop in hops:
            [reply] = ctx.call([_ctx_request(hop.index, hop.prompt, workload)], deadline)
            if _take_ctx(hop, reply):
                [answer] = gen.call([_gen_request(hop, reply, workload)], deadline)
                _take_gen(hop, answer)
        return hops
    replies = ctx.call([_ctx_request(h.index, h.prompt, workload) for h in hops], deadline)
    ready = [(hop, reply) for hop, reply in zip(hops, replies) if _take_ctx(hop, reply)]
    if ready:
        answers = gen.call([_gen_request(hop, reply, workload) for hop, reply in ready], deadline)
        for (hop, _), answer in zip(ready, answers):
            _take_gen(hop, answer)
    return hops


def format_hop(hop: Hop, *, full_output: bool) -> str:
    """The lines one prompt's hop prints: what a quickstart leg prints, and the ranks it crossed.

    ``[i] Witness ranks: ctx=<c> gen=<g>`` where the hop was pinned (the
    context rank as the context engine reports serving it, where it does);
    quickstart's ``[i] Prompt: ..., Generated text: ...``; and, under
    :data:`FULL_OUTPUT_ENV`, ``[i] Witness token ids: [...]`` and
    ``[i] Witness text: '...'``. A hop that failed prints why instead.
    """
    i = hop.index
    if hop.error:
        return f"[{i}] Witness leg B failed: {hop.error}\n"
    lines = []
    if hop.pair is not None:
        ctx = hop.pair[0] if hop.ctx_rank is None else hop.ctx_rank
        lines.append(f"[{i}] Witness ranks: ctx={ctx} gen={hop.pair[1]}")
    lines.append(f"[{i}] Prompt: {hop.prompt[:80]!r}..., Generated text: {hop.text[:200]!r}...")
    if full_output:
        lines.append(f"[{i}] Witness token ids: {list(hop.token_ids)}")
        lines.append(f"[{i}] Witness text: {hop.text!r}")
    return "\n".join(lines) + "\n"


def accept_engines(
    listener: Listener, processes: Mapping[str, Any], deadline: float
) -> dict[str, EngineClient]:
    """Each engine's connection, once it is up; :class:`LegBError` if one exits or time runs out."""
    accepted: queue.Queue = queue.Queue()

    def accept() -> None:
        for _ in range(len(processes)):
            try:
                accepted.put(listener.accept())
            except OSError:
                return

    threading.Thread(target=accept, daemon=True, name="leg-b-accept").start()
    clients: dict[str, EngineClient] = {}
    while len(clients) < len(processes):
        for role, process in processes.items():
            code = process.poll() if role not in clients else None
            if code is not None:
                raise LegBError(
                    f"the {ENGINE_NAMES[role]} engine exited {code} before it was up; its "
                    f"output is above"
                )
        if time.monotonic() > deadline:
            waiting = [ENGINE_NAMES[r] for r in processes if r not in clients]
            raise LegBError(
                f"the {' and '.join(waiting)} engine(s) did not come up before leg B's "
                f"deadline. A disaggregated launch fails by hanging more often than by raising: "
                f"check the cache transceiver and the interpreter each engine runs"
            )
        try:
            conn = accepted.get(timeout=2.0)
            hello = conn.recv()
        except queue.Empty:
            continue
        except (OSError, EOFError):
            # Connected and gone: the exit is named on the next pass.
            continue
        role = hello.get("role") if isinstance(hello, dict) else None
        if role not in processes:
            conn.close()
            continue
        clients[role] = EngineClient(role, processes[role], conn)
    return clients


def stop_engines(
    clients: Mapping[str, EngineClient], processes: Mapping[str, Any], grace: float
) -> None:
    """Tell each engine to shut down; terminate the ones still running after ``grace`` seconds."""
    for client in clients.values():
        client.close()
    for process in processes.values():
        try:
            process.wait(timeout=grace)
            continue
        except subprocess.TimeoutExpired:
            pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue


#: The ``sitecustomize`` the generation engine of an ablation run starts with
#: (``@SCRIPT@`` is this file). It installs :func:`install_ablation_hook` in
#: every process of the engine, then runs the next ``sitecustomize`` on
#: ``sys.path``, as a site or session hook it shadows would have run. Every
#: one it ran is remembered, so a hook that chains back to this one runs it
#: once and the chain moves on.
SITECUSTOMIZE = '''\
"""Leg B's generation engine: install the state ablation, then run the next sitecustomize."""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAN = "_mv2flow_sitecustomize_ran"


def _install():
    if os.environ.get("@ABLATE_ENV@") != "@ABLATE_STATE@":
        return
    if getattr(sys, "_mv2flow_ablation_hook", False):
        return
    sys._mv2flow_ablation_hook = True
    try:
        spec = importlib.util.spec_from_file_location("mv2flow_disagg_leg_b", @SCRIPT@)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install_ablation_hook()
    except Exception as exc:  # noqa: BLE001 -- said on stdout, where the witness reads
        print(f"@REFUSED@ the hook could not be installed: {type(exc).__name__}: {exc}", flush=True)


def _chain():
    ran = sys.__dict__.setdefault(_RAN, set())
    ran.add(os.path.join(_HERE, "sitecustomize.py"))
    for entry in list(sys.path):
        candidate = os.path.join(os.path.abspath(entry or "."), "sitecustomize.py")
        if candidate in ran or not os.path.isfile(candidate):
            continue
        ran.add(candidate)
        with open(candidate, encoding="utf-8") as handle:
            code = compile(handle.read(), candidate, "exec")
        exec(code, {"__name__": "sitecustomize", "__file__": candidate})
        return


_install()
_chain()
'''


def write_site(directory: Path) -> Path:
    """``directory`` holding :data:`SITECUSTOMIZE`, ready to lead a ``PYTHONPATH``."""
    directory.mkdir(parents=True, exist_ok=True)
    text = (
        SITECUSTOMIZE.replace("@SCRIPT@", repr(str(SCRIPT)))
        .replace("@ABLATE_ENV@", ABLATE_ENV)
        .replace("@ABLATE_STATE@", ABLATE_STATE)
        .replace("@REFUSED@", REFUSED)
    )
    (directory / "sitecustomize.py").write_text(text, encoding="utf-8")
    return directory


def _start_engine(command: str, env: Mapping[str, str]) -> subprocess.Popen:
    """One engine as a process group of its own, its output on this process's."""
    return subprocess.Popen(["bash", "-c", command], env=dict(env), start_new_session=True)


def coordinate(
    environ: Mapping[str, str],
    *,
    out: Any = None,
    start: Callable[[str, Mapping[str, str]], Any] = _start_engine,
    count: Callable[[], int] = count_devices,
    grace: float = SHUTDOWN_GRACE_S,
) -> int:
    """Run leg B: 0 when every prompt crossed, 1 when one did not, 2 when it could not run."""
    out = out or sys.stdout
    started = time.monotonic()
    try:
        workload = read_workload(environ)
        ranks = engine_ranks(workload.argv)
        ctx_devices, gen_devices = split_devices(visible_devices(environ, count), ranks)
    except LegBError as exc:
        print(f"disagg_leg_b: {exc}", file=out, flush=True)
        return 2
    # The engines' own deadline sits inside the witness's, so a hang is named
    # here rather than cut off by the runner.
    deadline = started + max(60.0, workload.timeout_s - 60.0)
    print(
        f"disagg_leg_b: context engine on devices {','.join(ctx_devices)}, generation engine "
        f"on {','.join(gen_devices)}, {ranks} rank(s) each"
        + (f" (member {workload.member})" if workload.member else "")
        + f"; cache_transceiver_config {json.dumps(dict(workload.transceiver), sort_keys=True)}"
        + (
            "; destroying the state each generation-only request receives"
            if workload.ablate
            else ""
        ),
        file=out,
        flush=True,
    )
    authkey = secrets.token_bytes(16)
    processes: dict[str, Any] = {}
    clients: dict[str, EngineClient] = {}
    hops: list[Hop] = []
    with (
        tempfile.TemporaryDirectory(prefix="leg-b-") as scratch,
        Listener(("127.0.0.1", 0), authkey=authkey) as listener,
    ):
        site = write_site(Path(scratch) / "site") if workload.ablate else None
        try:
            for role, devices in ((CTX, ctx_devices), (GEN, gen_devices)):
                env = engine_environment(
                    environ,
                    role=role,
                    devices=devices,
                    address=listener.address,
                    authkey=authkey,
                    workload=workload,
                    site=site,
                )
                processes[role] = start(engine_command(workload.launch, role), env)
            clients = accept_engines(listener, processes, deadline)
            hops = drive(clients[CTX], clients[GEN], workload, deadline)
        except LegBError as exc:
            print(f"disagg_leg_b: {exc}", file=out, flush=True)
            return 2
        finally:
            stop_engines(clients, processes, grace)
    # Printed once both engines are down, so no engine's log splits a line.
    for hop in hops:
        out.write(format_hop(hop, full_output=workload.full_output))
    out.flush()
    return 1 if any(hop.error for hop in hops) else 0


# ------------------------------------------------------------ the engine side


def build_llm(
    quickstart: Any, argv: Sequence[str], extra: Mapping[str, Any], driver: Any
) -> tuple[Any, Any, Any]:
    """``(llm, sampling_params, args)`` as the quickstart builds them from ``argv``, plus ``extra``.

    ``driver`` is ``engine_driver``: its ``engine_kwargs`` add what the gates
    add to every quickstart engine, and ``extra`` (the cache transceiver, the
    context engine's scheduler) is applied over that.
    """
    saved = sys.argv
    sys.argv = [str(getattr(quickstart, "__file__", "quickstart_advanced.py")), *argv]
    try:
        args = quickstart.parse_arguments()
    finally:
        sys.argv = saved
    mode = os.environ.get(driver.MODELING_V2_ENV)
    kv_cache = driver.kv_cache_overrides()
    llm_class = quickstart.LLM

    def build(*positional: Any, **kwargs: Any) -> Any:
        merged = driver.engine_kwargs(kwargs, mode=mode, kv_cache=kv_cache)
        return llm_class(*positional, **{**merged, **extra})

    quickstart.LLM = build
    try:
        llm, sampling = quickstart.setup_llm(args)
    finally:
        quickstart.LLM = llm_class
    return llm, sampling, args


@dataclass(frozen=True)
class Api:
    """The LLM-API types an engine builds its requests from."""

    disaggregated_params: Any
    scheduling_params: Any


def load_api() -> Api:
    from tensorrt_llm.llmapi import DisaggregatedParams, SchedulingParams

    return Api(disaggregated_params=DisaggregatedParams, scheduling_params=SchedulingParams)


def _render(llm: Any, args: Any, prompt: str) -> str:
    """``prompt`` as the quickstart sends it: through the chat template when its argv says so."""
    if not getattr(args, "apply_chat_template", False):
        return prompt
    messages = [{"role": "user", "content": f"{prompt}"}]
    return llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def submit(llm: Any, base: Any, args: Any, role: str, request: Mapping[str, Any], api: Api) -> Any:
    """One request on this engine: context-only on ``ctx``, generation-only on ``gen``.

    The generation-only request carries the context engine's
    ``DisaggregatedParams`` as it pickled them (:func:`reply_of`): the
    coordinator between the two engines never imports ``tensorrt_llm``.
    """
    sampling = copy.deepcopy(base)
    if request.get("max_tokens"):
        sampling.max_tokens = int(request["max_tokens"])
    kwargs: dict[str, Any] = {}
    if request.get("rank") is not None:
        kwargs["scheduling_params"] = api.scheduling_params(
            attention_dp_rank=int(request["rank"]), attention_dp_relax=False
        )
    if role == CTX:
        params = api.disaggregated_params(request_type="context_only")
        inputs: Any = _render(llm, args, str(request["prompt"]))
    else:
        params = pickle.loads(request["params"])
        params.request_type = "generation_only"
        inputs = {"prompt_token_ids": [int(t) for t in request["prompt_token_ids"]]}
    return llm.generate_async(inputs, sampling, disaggregated_params=params, **kwargs)


def reply_of(role: str, request: Mapping[str, Any], output: Any) -> dict[str, Any]:
    """What the coordinator needs of one finished request."""
    sequence = output.outputs[0]
    reply: dict[str, Any] = {
        "index": request["index"],
        "token_ids": [int(t) for t in sequence.token_ids],
        "text": sequence.text,
    }
    if role == CTX:
        params = getattr(sequence, "disaggregated_params", None)
        if params is None:
            params = getattr(output, "disaggregated_params", None)
        prompt_ids = getattr(output, "prompt_token_ids", None)
        reply["params"] = None if params is None else pickle.dumps(params)
        reply["prompt_token_ids"] = None if prompt_ids is None else [int(t) for t in prompt_ids]
        reply["ctx_dp_rank"] = getattr(params, "ctx_dp_rank", None)
    return reply


def serve(
    llm: Any,
    base: Any,
    args: Any,
    role: str,
    requests: Sequence[Mapping[str, Any]],
    api: Api,
) -> list[dict[str, Any]]:
    """Answer ``requests`` on this engine, all submitted before any is awaited, in order."""
    pending = []
    for request in requests:
        try:
            pending.append((request, submit(llm, base, args, role, request, api), None))
        # A request the engine refuses is the target's to report, in the reply.
        except Exception as exc:  # noqa: BLE001
            pending.append((request, None, f"{type(exc).__name__}: {exc}"))
    replies = []
    for request, future, error in pending:
        if future is not None:
            try:
                replies.append(reply_of(role, request, future.result()))
                continue
            except Exception as exc:  # noqa: BLE001 -- as above
                error = f"{type(exc).__name__}: {exc}"
        replies.append({"index": request["index"], "error": str(error)[:2000]})
    return replies


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host, int(port)


def engine_main(role: str, environ: Mapping[str, str]) -> int:
    """Build this engine, connect back to the coordinator, and serve until told to stop."""
    workload = read_workload(environ)
    sys.path.insert(0, str(HERE))
    import engine_driver  # by path, see the package docstring
    from tensorrt_llm.llmapi import CacheTransceiverConfig

    quickstart = engine_driver.load_quickstart(workload.quickstart)
    extra: dict[str, Any] = {
        "cache_transceiver_config": CacheTransceiverConfig(**dict(workload.transceiver))
    }
    if role == CTX:
        extra["disable_overlap_scheduler"] = True
    llm, base, args = build_llm(quickstart, workload.argv, extra, engine_driver)
    api = load_api()
    try:
        with Client(
            _address(environ[ADDRESS_ENV]), authkey=bytes.fromhex(environ[AUTHKEY_ENV])
        ) as conn:
            conn.send({"role": role})
            while True:
                try:
                    message = conn.recv()
                except EOFError:
                    break
                if message is None:
                    break
                conn.send(serve(llm, base, args, role, message["requests"], api))
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return 0


# --------------------------------------------------------------- the ablation

#: Where the generation engine takes up a request whose transfer completed.
EXECUTOR_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"
HOOKED_METHOD = "_prepare_disagg_gen_transmission_complete"


def _layer_map(manager: Any, name: str) -> Mapping[int, Any] | None:
    """``manager.<name>`` (layer -> offset), or its implementation's."""
    for holder in (manager, getattr(manager, "_impl", None)):
        value = getattr(holder, name, None) if holder is not None else None
        if isinstance(value, Mapping):
            return value
    return None


class RecurrentAblation:
    """Destroy a request's recurrent state: conv and SSM, every recurrent layer, at its slot."""

    def __init__(self, manager: Any, layers: Sequence[int]) -> None:
        self.manager, self.layers = manager, list(layers)

    def slot(self, request_id: int) -> int:
        for holder in (self.manager, getattr(self.manager, "_impl", None)):
            for name in ("mamba_cache_index", "_request_id_to_state_index"):
                mapping = getattr(holder, name, None) if holder is not None else None
                if isinstance(mapping, Mapping) and request_id in mapping:
                    return int(mapping[request_id])
        return int(self.manager.get_state_indices([request_id], [False])[0])

    def destroy(self, request: Any) -> str | None:
        if not self.layers:
            return None
        slot = self.slot(int(request.py_request_id))
        for layer in self.layers:
            self.manager.get_conv_states(layer)[slot].zero_()
            self.manager.get_ssm_states(layer)[slot].zero_()
        return f"the conv and SSM state of {len(self.layers)} recurrent layer(s)"


class PageAblation:
    """Destroy a request's KV pages in every attention layer."""

    def __init__(self, manager: Any, layers: Sequence[int]) -> None:
        self.manager, self.layers = manager, list(layers)

    def destroy(self, request: Any) -> str | None:
        request_id = int(request.py_request_id)
        pages = 0
        for layer in self.layers:
            buffer = self.manager.get_buffers(layer)
            if buffer is None:
                continue
            blocks = self.manager.get_batch_cache_indices([request_id], layer_idx=layer)[0]
            for block in blocks:
                # One page at a time: an integer index is a view, so this
                # writes the pool; a list index would zero a copy.
                buffer[int(block)].zero_()
            pages = max(pages, len(blocks))
        if not pages:
            return None
        return f"the KV pages of {len(self.layers)} attention layer(s) ({pages} page(s) each)"


def _kv_cache_manager(executor: Any) -> tuple[Any, Mapping[Any, Any]]:
    """``executor``'s KV-cache manager (``None`` without one), and all its resource managers."""
    managers = getattr(getattr(executor, "resource_manager", None), "resource_managers", None)
    if not isinstance(managers, Mapping):
        return None, {}
    for key, manager in managers.items():
        if getattr(key, "value", key) == "KV_CACHE_MANAGER":
            return manager, managers
    return None, managers


def resolve_ablation(executor: Any) -> RecurrentAblation | PageAblation:
    """What destroys a request's received state on ``executor``'s caches.

    Recurrent state where a manager holds it (``get_conv_states``,
    ``get_ssm_states`` and its recurrent layers), the KV manager first;
    else KV pages, where the KV manager can name and reach them
    (``get_batch_cache_indices``, ``get_buffers`` and its layers). Anything
    else raises :class:`AblationRefused`.
    """
    kv, managers = _kv_cache_manager(executor)
    first = [kv] if kv is not None else []
    for manager in [*first, *(m for m in managers.values() if m is not kv)]:
        layers = _layer_map(manager, "mamba_layer_offsets")
        if (
            layers
            and callable(getattr(manager, "get_conv_states", None))
            and callable(getattr(manager, "get_ssm_states", None))
        ):
            return RecurrentAblation(manager, sorted(layers))
    layers = _layer_map(kv, "layer_offsets") if kv is not None else None
    if (
        layers
        and callable(getattr(kv, "get_buffers", None))
        and callable(getattr(kv, "get_batch_cache_indices", None))
    ):
        return PageAblation(kv, sorted(layers))
    kinds = sorted({type(m).__name__ for m in managers.values()}) or ["none"]
    raise AblationRefused(
        f"leg B cannot reach the state a request received on this engine's caches "
        f"({', '.join(kinds)}): no manager has recurrent layers with conv and SSM state, and "
        f"the KV-cache manager does not name a request's pages "
        f"(get_batch_cache_indices, get_buffers, layer_offsets)"
    )


def _refuse(why: str) -> AblationRefused:
    print(f"{REFUSED} {why}", flush=True)
    return AblationRefused(f"{REFUSED} {why}")


def patch_executor(module: Any) -> None:
    """Make ``module.PyExecutor`` destroy each request's received state before its first decode.

    Wraps :data:`HOOKED_METHOD`, which the executor runs for the requests
    whose transfer completed before their first generation step, and the
    constructor, which resolves :func:`resolve_ablation` once the caches
    exist -- so an engine whose caches cannot be reached does not come up.
    Without the method, the constructor refuses.
    """
    executor_class = getattr(module, "PyExecutor", None)
    if executor_class is None or getattr(executor_class, "_mv2flow_ablation", False):
        return
    executor_class._mv2flow_ablation = True
    init = executor_class.__init__
    original = getattr(executor_class, HOOKED_METHOD, None)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        if original is None:
            raise _refuse(
                f"PyExecutor has no {HOOKED_METHOD}, where a generation-only request is taken "
                f"up once its transfer completes"
            )
        try:
            self._mv2flow_ablate = resolve_ablation(self)
        except AblationRefused as exc:
            raise _refuse(str(exc)) from exc

    executor_class.__init__ = __init__
    if original is None:
        return

    def prepare(self: Any, scheduled_batch: Any, *args: Any, **kwargs: Any) -> Any:
        ablation = getattr(self, "_mv2flow_ablate", None)
        if ablation is not None:
            for request in getattr(scheduled_batch, "generation_requests", ()) or ():
                if not getattr(request, "is_disagg_generation_transmission_complete", False):
                    continue
                what = ablation.destroy(request)
                if what:
                    print(f"{ABLATED} {what}", flush=True)
        return original(self, scheduled_batch, *args, **kwargs)

    setattr(executor_class, HOOKED_METHOD, prepare)


class _PatchingLoader(importlib.abc.Loader):
    """A module's own loader, with :func:`patch_executor` run once the module has executed."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def create_module(self, spec: Any) -> Any:
        return self.loader.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.loader.exec_module(module)
        patch_executor(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


class _ExecutorFinder(importlib.abc.MetaPathFinder):
    """Find :data:`EXECUTOR_MODULE` as the other finders do, and patch it as it loads."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name != EXECUTOR_MODULE:
            return None
        spec = None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader)
        return spec


def install_ablation_hook() -> None:
    """Patch the executor in this process: now if it is imported, else as it is imported."""
    module = sys.modules.get(EXECUTOR_MODULE)
    if module is not None:
        patch_executor(module)
        return
    if not any(isinstance(f, _ExecutorFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ExecutorFinder())


def main(argv: Sequence[str] | None = None) -> int:
    words = list(sys.argv[1:] if argv is None else argv)
    if words and words[0] == "engine":
        parser = argparse.ArgumentParser(prog="disagg_leg_b.py engine")
        parser.add_argument("--role", choices=ROLES, required=True)
        return engine_main(parser.parse_args(words[1:]).role, os.environ)
    if words:
        print(f"usage: {SCRIPT.name} [engine --role ctx|gen]", file=sys.stderr)
        return 2
    return coordinate(os.environ)


if __name__ == "__main__":
    sys.exit(main())
