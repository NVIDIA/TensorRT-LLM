# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate bundles: the checks of one unit at one member measured on one engine.

A cold engine start is most of what a boot, logit or accuracy check costs,
and each check of a unit at a member starts its own on the same commit. A
bundle starts one. The checks of a unit at a member that are ready together
measure one snapshot commit, and those whose gates would build the same
engine run their workloads on it one after another, in one job
(``harness/bundle_driver.py``). Each check still gets its own verdict from
its own gate, over the output its own job would have printed.

**What joins** (:func:`bundled`): the boot, logit, acceptance and accuracy
gates. Each gate runs once with a runner that records the command it would
launch first and launches nothing (:func:`capture`). An engine-driver or
logit-driver command builds its engine with the quickstart from its argv; two
of them build one engine when their launch (environment, launcher,
interpreter), quickstart and argv agree but for ``--max_tokens``, which sets
only the sampling budget (:attr:`Launch.engine`). The acceptance gate's
``=require`` leg is its first command, so it joins; its ``=off`` leg is
another engine, and runs as its own job when the gate asks for it. An
accuracy gate runs ``trtllm-eval``, which builds its engine from LLM-API
arguments, not quickstart flags. It joins the group whose engine its argv and
options file describe the same way on every setting either launch path can
state (:func:`eval_engine_difference`), and keeps its section only if the
engine that ran is the one its options file declares on every attested field
(:func:`engine_holds`).

**Verdicts.** The job's output is cut into sections (:func:`split`): what was
printed while the engine was built, then between each section's markers what
that gate's workload printed. Each gate then runs again with a runner that
returns, for the command it recorded, the build output and its own section
as that command's result, and runs any other command as before
(:func:`replaying`). So each verdict is its own gate's, over its own records,
and every member's carries the one engine record they share. The verdict
names the bundle (:class:`BundledResult`). The sections share the engine's KV
cache: a later one may resume a prompt from blocks an earlier one left, which
adds reuse and takes none away, and the logit driver salts each request, so
its prompt log-probs come from a context step of their own.

**Failure isolation.** A section that ended with no error is replayed. One
that raised or ran past its time, and every section after it, runs again
singly, as does every section of a job lost with its session or that printed
no section at all. A job whose engine could not be built says so
(``MV2FLOW_BUILD``): every quickstart gate gets the job's failure as its
result, since each would have built that engine from the same argv and
failed the same way, and an evaluation, which builds its own, runs singly.

**Formation** (:class:`Board`). Checks join a bundle by unit and member as
they are admitted. The first to arrive leads and holds the bundle's one
admission slot; the others wait in it without holding one. Once admitted, the
leader waits until :data:`GATHER_S` after it arrived, seals the bundle,
captures every member and runs one group of two or more: its own, or else the
largest. Every member it runs no job for runs singly, admitted on its own.
"""

from __future__ import annotations

import json
import shlex
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.execution.commands import CommandResult
from . import gates as gates_mod
from . import logit_agreement
from .harness import workload_io

#: The job, run as a file under the measured tree's interpreter.
BUNDLE_DRIVER = Path(__file__).resolve().parent / "harness" / "bundle_driver.py"
#: Before and after each section's own output (``bundle_driver.SECTION_MARKER``).
SECTION_MARKER = "MV2FLOW_SECTION"
#: Once, when the shared engine was not built (``bundle_driver.BUILD_MARKER``).
BUILD_MARKER = "MV2FLOW_BUILD"
#: How long a bundle's leader waits after it arrived for the checks submitted with it.
GATHER_S = 2.0

KIND_DRIVER = "driver"
KIND_LOGITS = "logits"
KIND_EVAL = "eval"
#: The script each kind of command runs.
SCRIPTS: dict[str, Path] = {
    KIND_DRIVER: gates_mod.ENGINE_DRIVER,
    KIND_LOGITS: logit_agreement.LOGIT_DRIVER,
    KIND_EVAL: gates_mod.EVAL_SHIM,
}
#: The launch variable that hands KV-cache keys to the quickstart's engine.
HANDED_ENV = gates_mod.KV_CACHE_CONFIG_ENV


def bundled(gate: str) -> bool:
    """Whether a check of ``gate`` may join a bundle."""
    return gate in ("boot", "acceptance", logit_agreement.GATE_NAME) or gate.startswith("accuracy[")


# ------------------------------------------------------------------ commands


@dataclass(frozen=True)
class Launch:
    """The command one check's gate launches first, as a bundle reads it."""

    key: str
    kind: str
    command: str
    timeout: int
    #: The command up to its script: the environment, the launcher, the interpreter.
    launch: str
    #: The driver's spec file (driver and logits).
    spec: str = ""
    quickstart: str = ""
    #: The quickstart's argv (driver and logits), or ``trtllm-eval``'s (eval).
    argv: tuple[str, ...] = ()
    #: Where the logit driver writes its records (logits).
    out: str = ""

    @property
    def engine(self) -> tuple[str, str, tuple[str, ...]] | None:
        """What decides the engine a quickstart command builds; ``None`` for an evaluation."""
        if self.kind == KIND_EVAL:
            return None
        return (self.launch, self.quickstart, without_max_tokens(self.argv))


def without_max_tokens(words: Sequence[str]) -> tuple[str, ...]:
    """``words`` without ``--max_tokens``: the quickstart reads it into the sampling parameters only."""
    kept: list[str] = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word == "--max_tokens":
            skip = True
            continue
        if word.startswith("--max_tokens="):
            continue
        kept.append(word)
    return tuple(kept)


def _driver_words(words: Sequence[str]) -> tuple[str, str | None, list[str] | None]:
    """``--spec S [--quickstart Q] [-- argv]`` as the drivers read it."""
    spec, quickstart, argv = "", None, None
    index = 0
    while index < len(words):
        word = words[index]
        if word == "--":
            argv = list(words[index + 1 :])
            break
        if word in ("--spec", "--quickstart") and index + 1 < len(words):
            if word == "--spec":
                spec = words[index + 1]
            else:
                quickstart = words[index + 1]
            index += 2
            continue
        index += 1
    return spec, quickstart, argv


def parse_launch(key: str, command: str, timeout: int) -> Launch | None:
    """``command`` as a bundle runs it; ``None`` when it is none of the drivers a bundle runs.

    A driver command's quickstart and argv are read off its command line,
    as the driver itself reads them, else from its spec.
    """
    for kind, script in SCRIPTS.items():
        marker = f" {shlex.quote(str(script))} "
        at = command.find(marker)
        if at < 0:
            continue
        try:
            words = shlex.split(command[at + len(marker) :])
        except ValueError:
            return None
        launch = command[:at]
        if kind == KIND_EVAL:
            return Launch(key, kind, command, int(timeout), launch, argv=tuple(words))
        spec, quickstart, argv = _driver_words(words)
        try:
            data = json.loads(Path(spec).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Launch(
            key,
            kind,
            command,
            int(timeout),
            launch,
            spec=spec,
            quickstart=str(quickstart or data.get("quickstart") or ""),
            argv=tuple(
                str(w) for w in (argv if argv is not None else data.get("quickstart_argv") or ())
            ),
            out=str(data.get("out") or "") if kind == KIND_LOGITS else "",
        )
    return None


class _Launched(BaseException):
    """What a gate run under :func:`capture` asked to launch.

    A ``BaseException``, so a gate's own handler of ordinary errors lets it through.
    """

    def __init__(self, command: str, timeout: int) -> None:
        super().__init__(command)
        self.command = command
        self.timeout = timeout


def capture(key: str, call: Callable[[gates_mod.ShellRunner], Any]) -> Launch | Any:
    """What ``call(runner)`` (one check's gate) launches first, launching nothing.

    The :class:`Launch`; ``None`` when that command is none a bundle runs; the
    gate's own result when it returned without launching anything.
    """

    def runner(command: str, env: Any = None, timeout: int = gates_mod.BOOT_TIMEOUT_S) -> Any:
        raise _Launched(command, int(timeout))

    try:
        return call(runner)
    except _Launched as launched:
        return parse_launch(key, launched.command, launched.timeout)


# ------------------------------------------------------- the evaluation's engine

#: Quickstart flags the engine comparison reads that take a value.
_QUICKSTART_VALUES = frozenset(
    {
        "--model_dir",
        "--max_tokens",
        "--tp_size",
        "--moe_ep_size",
        "--max_batch_size",
        "--max_num_tokens",
        "--max_seq_len",
        "--kv_cache_fraction",
        "--kv_cache_dtype",
        "--mamba_ssm_cache_dtype",
        "--tokens_per_block",
        "--use_kv_cache_manager_v2",
        "--mamba_ssm_philox_rounds",
        "--attention_backend",
        "--moe_backend",
        "--spec_decode_algo",
        "--spec_decode_max_draft_len",
        "--cuda_graph_batch_sizes",
    }
)
_QUICKSTART_SWITCHES = frozenset(
    {
        "--use_cuda_graph",
        "--cuda_graph_padding_enabled",
        "--enable_chunked_prefill",
        "--disable_overlap_scheduler",
        "--disable_kv_cache_reuse",
        "--enable_attention_dp",
        "--mamba_ssm_stochastic_rounding",
        "--log_kv_cache_events",
        "--print_iter_log",
    }
)
#: ``trtllm-eval``'s own options the comparison reads, before its benchmark subcommand.
_EVAL_VALUES = frozenset(
    {
        "--model",
        "--tp_size",
        "--ep_size",
        "--max_batch_size",
        "--max_num_tokens",
        "--max_seq_len",
        "--kv_cache_free_gpu_memory_fraction",
        "--extra_llm_api_options",
        "--config",
        "--log_level",
    }
)
_EVAL_SWITCHES = frozenset({"--disable_kv_cache_reuse", "--telemetry", "--no-telemetry"})
#: Options-file keys the comparison reads; any other key sets something it does not.
_OPTION_KEYS = frozenset(
    {
        "kv_cache_config",
        "cuda_graph_config",
        "speculative_config",
        "enable_chunked_prefill",
        "max_num_tokens",
        "max_batch_size",
        "max_seq_len",
        "enable_attention_dp",
        "modeling_v2",
        "attn_backend",
        "moe_config",
        "disable_overlap_scheduler",
        "print_iter_log",
        "enable_iter_perf_stats",
    }
)
_SPECULATIVE_KEYS = frozenset(
    {"decoding_type", "max_draft_len", "num_nextn_predict_layers", "block_size"}
)
_CUDA_GRAPH_KEYS = frozenset({"batch_sizes", "max_batch_size", "enable_padding"})
_MOE_KEYS = frozenset({"backend"})
#: The LLM-API defaults of the settings compared, where the two launch paths leave them unset.
_DEFAULTS: dict[str, Any] = {
    "free_gpu_memory_fraction": 0.9,
    "tokens_per_block": 32,
    "use_kv_cache_manager_v2": True,
    "mamba_ssm_philox_rounds": 10,
    "max_num_tokens": 8192,
    "max_batch_size": 2048,
    "attn_backend": "TRTLLM",
    "moe_backend": "AUTO",
}


def _read_flags(
    words: Sequence[str], values: frozenset[str], switches: frozenset[str], *, subcommand: bool
) -> tuple[dict[str, Any], list[str]]:
    """The flags of ``words`` (the last of each wins), and those the comparison does not read.

    ``--flag=value`` and ``--flag value``; ``--cuda_graph_batch_sizes`` takes
    the words up to the next flag. With ``subcommand`` the first word that is
    not a flag ends the reading (``trtllm-eval``'s benchmark and its protocol).
    """
    found: dict[str, Any] = {}
    unread: list[str] = []
    index = 0
    while index < len(words):
        word = str(words[index])
        index += 1
        if not word.startswith("--"):
            if subcommand:
                break
            unread.append(word)
            continue
        name, equals, value = word.partition("=")
        if name in switches and not equals:
            found[name] = True
        elif name in values:
            if equals:
                found[name] = value
                continue
            taken: list[str] = []
            while index < len(words) and not str(words[index]).startswith("--"):
                taken.append(str(words[index]))
                index += 1
                if name != "--cuda_graph_batch_sizes":
                    break
            found[name] = " ".join(taken)
        else:
            unread.append(name)
    return found, unread


def _launch_tokens(launch: str) -> tuple[list[str], dict[str, str]]:
    """``launch`` as words without the handed-over KV-cache keys, and its ``KEY=value`` words."""
    try:
        words = shlex.split(launch)
    except ValueError:
        words = launch.split()
    assigned = {}
    for word in words:
        name, equals, value = word.partition("=")
        if equals and name.isidentifier():
            assigned[name] = value
    return [w for w in words if not w.startswith(f"{HANDED_ENV}=")], assigned


def _tri(value: Any) -> Any:
    """``use_kv_cache_manager_v2``, which is true, false or ``auto``."""
    text = str(value).strip().lower()
    return {"true": True, "false": False}.get(text, value if isinstance(value, bool) else text)


def _int(value: Any) -> int | None:
    return None if value in (None, "", "None") else int(value)


def _expert(value: int | None) -> int | None:
    """An expert-parallel width, ``None`` when unset (``-1``, or not given)."""
    return value if value is not None and value > 0 else None


def quickstart_engine(
    words: Sequence[str], handed: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """The settings the quickstart builds its engine with from ``words``, and the flags it does not read."""
    flags, unread = _read_flags(words, _QUICKSTART_VALUES, _QUICKSTART_SWITCHES, subcommand=False)
    capture_list = gates_mod.quickstart_capture(list(words))
    algorithm = flags.get("--spec_decode_algo")
    settings = {
        "model": flags.get("--model_dir"),
        "tp": int(flags.get("--tp_size", 1)),
        "ep": _expert(_int(flags.get("--moe_ep_size"))),
        "attention_dp": "--enable_attention_dp" in flags,
        "block_reuse": "--disable_kv_cache_reuse" not in flags,
        "kv_cache_dtype": str(flags.get("--kv_cache_dtype", "auto")).lower(),
        "mamba_ssm_cache_dtype": str(flags.get("--mamba_ssm_cache_dtype", "auto")).lower(),
        "free_gpu_memory_fraction": float(
            flags.get("--kv_cache_fraction", _DEFAULTS["free_gpu_memory_fraction"])
        ),
        "tokens_per_block": int(flags.get("--tokens_per_block", _DEFAULTS["tokens_per_block"])),
        "use_kv_cache_manager_v2": _tri(
            flags.get("--use_kv_cache_manager_v2", _DEFAULTS["use_kv_cache_manager_v2"])
        ),
        "mamba_ssm_stochastic_rounding": "--mamba_ssm_stochastic_rounding" in flags,
        "mamba_ssm_philox_rounds": int(
            flags.get("--mamba_ssm_philox_rounds", _DEFAULTS["mamba_ssm_philox_rounds"])
        ),
        "event_buffer_max_size": 1024 if "--log_kv_cache_events" in flags else 0,
        "cuda_graph": (
            None
            if capture_list is None
            else (tuple(capture_list.batch_sizes), capture_list.enable_padding)
        ),
        "speculative": (
            None
            if not algorithm
            else (str(algorithm).upper(), int(flags.get("--spec_decode_max_draft_len", 1)))
        ),
        "max_num_tokens": int(flags.get("--max_num_tokens", _DEFAULTS["max_num_tokens"])),
        "max_batch_size": int(flags.get("--max_batch_size", _DEFAULTS["max_batch_size"])),
        "max_seq_len": _int(flags.get("--max_seq_len")),
        "chunked_prefill": "--enable_chunked_prefill" in flags,
        "overlap_scheduler": "--disable_overlap_scheduler" not in flags,
        "attn_backend": str(flags.get("--attention_backend", _DEFAULTS["attn_backend"])).upper(),
        "moe_backend": str(flags.get("--moe_backend", _DEFAULTS["moe_backend"])).upper(),
        "handed": dict(handed),
    }
    return settings, [f"quickstart flag {name}" for name in unread]


def eval_options(argv: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """The options file ``trtllm-eval``'s ``argv`` names, and what it holds."""
    flags, _ = _read_flags(argv[1:], _EVAL_VALUES, _EVAL_SWITCHES, subcommand=True)
    path = str(flags.get("--extra_llm_api_options") or flags.get("--config") or "")
    return path, gates_mod.load_llm_api_options(path) if path else {}


def eval_engine(argv: Sequence[str]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """The settings ``trtllm-eval`` builds its engine with from ``argv``, what it does not read, its options.

    An explicit flag wins over the options file, and the file over the LLM
    API's default, as ``trtllm-eval`` merges them.
    """
    flags, unread = _read_flags(argv[1:], _EVAL_VALUES, _EVAL_SWITCHES, subcommand=True)
    path, options = eval_options(argv)
    refused = [f"trtllm-eval flag {name}" for name in unread]
    refused += [f"`{key}` in {path}" for key in sorted(set(options) - _OPTION_KEYS)]
    kv = (
        options.get("kv_cache_config")
        if isinstance(options.get("kv_cache_config"), Mapping)
        else {}
    )
    speculative = options.get("speculative_config")
    graphs_block = options.get("cuda_graph_config")
    moe = options.get("moe_config") if isinstance(options.get("moe_config"), Mapping) else {}
    for name, block, allowed in (
        ("speculative_config", speculative, _SPECULATIVE_KEYS),
        ("cuda_graph_config", graphs_block, _CUDA_GRAPH_KEYS),
        ("moe_config", moe, _MOE_KEYS),
    ):
        if isinstance(block, Mapping):
            refused += [f"`{name}.{key}` in {path}" for key in sorted(set(block) - allowed)]
    try:
        graphs = gates_mod.options_capture(options, path or "the options")
    except ValueError as exc:
        refused.append(str(exc))
        graphs = None
    depth = None
    if isinstance(speculative, Mapping):
        for key in ("max_draft_len", "num_nextn_predict_layers", "block_size"):
            if speculative.get(key) is not None:
                depth = max(1, int(speculative[key]))
                break

    def setting(flag: str, key: str) -> Any:
        if flag in flags:
            return flags[flag]
        return options.get(key, _DEFAULTS.get(key))

    settings = {
        "model": flags.get("--model"),
        "tp": int(flags.get("--tp_size", 1)),
        "ep": _expert(_int(flags.get("--ep_size"))),
        "attention_dp": bool(options.get("enable_attention_dp", False)),
        "block_reuse": "--disable_kv_cache_reuse" not in flags
        and bool(kv.get("enable_block_reuse", True)),
        "kv_cache_dtype": str(kv.get("dtype", "auto")).lower(),
        "mamba_ssm_cache_dtype": str(kv.get("mamba_ssm_cache_dtype", "auto")).lower(),
        "free_gpu_memory_fraction": float(
            flags.get(
                "--kv_cache_free_gpu_memory_fraction",
                kv.get("free_gpu_memory_fraction", _DEFAULTS["free_gpu_memory_fraction"]),
            )
        ),
        "tokens_per_block": int(kv.get("tokens_per_block", _DEFAULTS["tokens_per_block"])),
        "use_kv_cache_manager_v2": _tri(
            kv.get("use_kv_cache_manager_v2", _DEFAULTS["use_kv_cache_manager_v2"])
        ),
        "mamba_ssm_stochastic_rounding": bool(kv.get("mamba_ssm_stochastic_rounding", False)),
        "mamba_ssm_philox_rounds": int(
            kv.get("mamba_ssm_philox_rounds", _DEFAULTS["mamba_ssm_philox_rounds"])
        ),
        "event_buffer_max_size": int(kv.get("event_buffer_max_size", 0)),
        "cuda_graph": (
            None if graphs is None else (tuple(graphs.batch_sizes), graphs.enable_padding)
        ),
        "speculative": (
            None
            if not isinstance(speculative, Mapping)
            else (str(speculative.get("decoding_type", "")).upper(), depth or 1)
        ),
        "max_num_tokens": int(setting("--max_num_tokens", "max_num_tokens")),
        "max_batch_size": int(setting("--max_batch_size", "max_batch_size")),
        "max_seq_len": _int(setting("--max_seq_len", "max_seq_len")),
        "chunked_prefill": bool(options.get("enable_chunked_prefill", False)),
        "overlap_scheduler": not bool(options.get("disable_overlap_scheduler", False)),
        "attn_backend": str(options.get("attn_backend", _DEFAULTS["attn_backend"])).upper(),
        "moe_backend": str(moe.get("backend", _DEFAULTS["moe_backend"])).upper(),
        "handed": gates_mod.quickstart_kv_cache_config(options),
    }
    return settings, refused, options


def eval_engine_difference(evaluation: Launch, quickstart: Launch) -> list[str]:
    """How the engine ``evaluation``'s ``trtllm-eval`` builds differs from ``quickstart``'s; empty: one engine.

    Compared setting by setting, each as its own launch path resolves it
    (:func:`eval_engine`, :func:`quickstart_engine`), with the launch
    environments, the target selection and the KV-cache keys handed to the
    quickstart's engine. A flag or an options key the comparison does not
    read is a difference: what it sets is not known to agree.
    """
    eval_words, eval_env = _launch_tokens(evaluation.launch)
    quick_words, quick_env = _launch_tokens(quickstart.launch)
    differences: list[str] = []
    if eval_words != quick_words:
        differences.append("their launches differ (environment, launcher or interpreter)")
    try:
        handed = json.loads(quick_env.get(HANDED_ENV) or "{}")
    except ValueError:
        handed = {"unreadable": quick_env.get(HANDED_ENV)}
    quick, unread = quickstart_engine(quickstart.argv, handed)
    theirs, refused, options = eval_engine(evaluation.argv)
    differences += unread + refused
    mode = quick_env.get(gates_mod.MODELING_V2_ENV)
    if options.get("modeling_v2") != mode:
        differences.append(
            f"modeling_v2: quickstart {mode}, trtllm-eval {options.get('modeling_v2')}"
        )
    for name, value in quick.items():
        if theirs.get(name) != value:
            differences.append(f"{name}: quickstart {value!r}, trtllm-eval {theirs.get(name)!r}")
    return differences


def eval_declaration(evaluation: Launch, quickstart: Launch) -> dict[str, Any]:
    """The engine ``evaluation``'s options file declares, as the gates attest one (``gates.declared_engine``)."""
    _, options = eval_options(evaluation.argv)
    _, env = _launch_tokens(quickstart.launch)
    return gates_mod.declared_engine(
        list(quickstart.argv), mode=env.get(gates_mod.MODELING_V2_ENV), options=options
    )


# -------------------------------------------------------------------- groups


@dataclass
class Group:
    """The checks that run on one engine: quickstart commands that build it, evaluations that share it."""

    launches: list[Launch]
    #: Per evaluation: the engine its options file declares (:func:`eval_declaration`).
    declared: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def head(self) -> Launch:
        return next(launch for launch in self.launches if launch.kind != KIND_EVAL)

    @property
    def keys(self) -> list[str]:
        return [launch.key for launch in self.launches]

    @property
    def timeout(self) -> int:
        """Long enough for every member's own bound, one after another."""
        return sum(launch.timeout for launch in self.launches)

    def launch_of(self, key: str) -> Launch | None:
        return next((launch for launch in self.launches if launch.key == key), None)


def form_groups(launches: Sequence[Launch]) -> tuple[list[Group], dict[str, str]]:
    """The groups ``launches`` fall into, largest first, and why each evaluation joined none.

    Quickstart commands group by :attr:`Launch.engine`. Each evaluation joins
    the largest group whose engine it describes (:func:`eval_engine_difference`).
    """
    groups: dict[Any, Group] = {}
    for launch in launches:
        if launch.kind != KIND_EVAL:
            groups.setdefault(launch.engine, Group([])).launches.append(launch)
    engines = sorted(groups.values(), key=lambda g: -len(g.launches))
    alone: dict[str, str] = {}
    singles: list[Group] = []
    for launch in launches:
        if launch.kind != KIND_EVAL:
            continue
        reasons: list[str] = []
        for group in engines:
            difference = eval_engine_difference(launch, group.head)
            if not difference:
                group.launches.append(launch)
                group.declared[launch.key] = eval_declaration(launch, group.head)
                break
            reasons = reasons or difference
        else:
            alone[launch.key] = (
                "its engine differs from the bundle's: " + "; ".join(reasons)
                if reasons
                else "no quickstart command of the bundle builds an engine for it to share"
            )
            singles.append(Group([launch]))
    return sorted([*engines, *singles], key=lambda g: -len(g.launches)), alone


def write_spec(path: str | Path, group: Group, outs: Mapping[str, str]) -> Path:
    """The bundle driver's spec for ``group``; ``outs`` is where each logits section writes."""
    head = group.head
    quick = [launch for launch in group.launches if launch.kind != KIND_EVAL]
    sections = []
    for launch in group.launches:
        section: dict[str, Any] = {
            "key": launch.key,
            "kind": launch.kind,
            "timeout": launch.timeout,
        }
        if launch.kind == KIND_EVAL:
            section["argv"] = list(launch.argv)
        else:
            section["spec"] = launch.spec
        if launch.kind == KIND_LOGITS:
            section["out"] = str(outs[launch.key])
        sections.append(section)
    spec = {
        "quickstart": head.quickstart,
        "quickstart_argv": list(head.argv),
        # Every quickstart gate's own job would have ended by then.
        "build_timeout": min(launch.timeout for launch in quick),
        "sections": sections,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def bundle_command(group: Group, spec_path: str | Path) -> str:
    """The bundle driver over ``spec_path``, launched as the group's quickstart commands are."""
    return (
        f"{group.head.launch} {shlex.quote(str(BUNDLE_DRIVER))} "
        f"--spec {shlex.quote(str(spec_path))}"
    )


# ------------------------------------------------------------------- the job


@dataclass(frozen=True)
class Section:
    """One member's part of a bundle job: the result to judge, or why it runs singly."""

    key: str
    result: CommandResult | None = None
    why: str = ""


def _cut(stdout: str) -> tuple[str, dict[str, str], dict[str, dict[str, Any]], list[str]]:
    """``(build output, each section's own output, each section's end record, sections begun)``."""
    pattern = workload_io.parse_marked
    common: list[str] = []
    own: dict[str, list[str]] = {}
    ended: dict[str, dict[str, Any]] = {}
    begun: list[str] = []
    current: str | None = None
    for line in stdout.splitlines(keepends=True):
        marks = pattern(line, SECTION_MARKER)
        if marks:
            record = marks[0]
            key = str(record.get("key"))
            if record.get("begin"):
                current = key
                begun.append(key)
                own.setdefault(key, [])
            elif record.get("end"):
                ended[key] = record
                current = None
            continue
        if current is not None:
            own[current].append(line)
        elif not begun:
            common.append(line)
    return "".join(common), {k: "".join(v) for k, v in own.items()}, ended, begun


def engine_holds(output: str, declared: Mapping[str, Any]) -> str:
    """Why the engine whose record is in ``output`` is not the ``declared`` one; ``""`` when it is."""
    attested = gates_mod.attest_engine(output, declared)
    if not attested.reported:
        return f"the bundle's engine was not attested: {attested.note}"
    return attested.difference() if attested.mismatches else ""


def split(output: CommandResult, group: Group) -> dict[str, Section]:
    """Each member's :class:`Section` of the job that printed ``output``; see the module docstring."""
    keys = group.keys
    if output.infrastructure:
        why = f"the bundle did not run to an end: {output.infrastructure}"
        return {key: Section(key, why=why) for key in keys}
    built = workload_io.parse_marked(output.stdout, BUILD_MARKER)
    if any(record.get("error") for record in built):
        failed = CommandResult(
            returncode=output.returncode or 1, stdout=output.stdout, stderr=output.stderr
        )
        why = "the bundle's engine was not built, and an evaluation builds its own"
        return {
            launch.key: Section(launch.key, why=why)
            if launch.kind == KIND_EVAL
            else Section(launch.key, failed)
            for launch in group.launches
        }
    common, own, ended, begun = _cut(output.stdout)
    sections: dict[str, Section] = {}
    stopped = ""
    for launch in group.launches:
        key = launch.key
        end = ended.get(key)
        if stopped:
            sections[key] = Section(key, why=stopped)
            continue
        if end is not None and end.get("ok"):
            result = CommandResult(
                returncode=0, stdout=common + own.get(key, ""), stderr=output.stderr
            )
            declared = group.declared.get(key)
            held = engine_holds(result.stdout + result.stderr, declared) if declared else ""
            sections[key] = Section(key, result) if not held else Section(key, why=held)
            continue
        if end is not None:
            why = (
                "its section ran past its own time in the bundle"
                if end.get("timed_out")
                else f"its section failed in the bundle: {end.get('error') or 'no reason given'}"
            )
            stopped = f"an earlier section ({key}) ended the bundle"
        elif key in begun:
            why = (
                f"the bundle ended inside its section "
                f"({'timed out' if output.timed_out else f'exited {output.returncode}'})"
            )
            stopped = f"the bundle ended inside an earlier section ({key})"
        elif any(record.get("timed_out") for record in built):
            why = "the bundle's engine took longer to build than its gates allow"
            stopped = why
        else:
            why = (
                f"the bundle ended before its section "
                f"({'timed out' if output.timed_out else f'exited {output.returncode}'})"
            )
            stopped = why
        sections[key] = Section(key, why=why)
    return sections


def replaying(
    launch: Launch,
    section: CommandResult,
    run: gates_mod.ShellRunner,
    observe: Callable[[str], None] = lambda output: None,
    out: str = "",
) -> gates_mod.ShellRunner:
    """A runner that answers ``launch``'s command with ``section`` and runs any other.

    ``observe`` hears the output answered, as the snapshot's runner hears a
    command's; ``out`` is where the job wrote a logits section's records,
    copied to where the gate reads them each time it is answered.
    """

    def runner(command: str, env: Any = None, timeout: int = gates_mod.BOOT_TIMEOUT_S) -> Any:
        if command != launch.command:
            return run(command, env, timeout)
        if out and launch.out and Path(out).is_file():
            shutil.copyfile(out, launch.out)
        observe(f"{section.stdout}\n{section.stderr}")
        return section

    return runner


@dataclass(frozen=True)
class BundledResult(gates_mod.GateResult):
    """A gate's verdict measured in a bundle: the gate's own, naming the bundle."""

    #: The bundle's name and the checks whose workloads ran on its one engine.
    bundle: str = ""

    def summary(self) -> str:
        return f"{super().summary()} [bundle {self.bundle}]"


def mark(result: Any, label: str) -> Any:
    """``result`` naming the bundle ``label`` it was measured in; anything but a gate result as it is."""
    if not isinstance(result, gates_mod.GateResult):
        return result
    values = {f.name: getattr(result, f.name) for f in fields(gates_mod.GateResult)}
    return BundledResult(**values, bundle=label)


class Job:
    """One bundle job: its group, run by the leader, its sections read by every member."""

    def __init__(self, group: Group, context: Any = None, name: str = "") -> None:
        self.group = group
        #: What every member needs to judge its section (the commit measured, say).
        self.context = context
        self.name = name or f"b{uuid.uuid4().hex[:8]}"
        #: Per logits member, where the job writes its records.
        self.outs: dict[str, str] = {}
        self._sections: dict[str, Section] = {}
        self._done = threading.Event()

    @property
    def label(self) -> str:
        return f"{self.name}: {' + '.join(self.group.keys)}"

    def publish(self, sections: Mapping[str, Section]) -> None:
        self._sections = dict(sections)
        self._done.set()

    def fail(self, why: str) -> None:
        """Every member that has no section yet runs singly, for ``why``."""
        if not self._done.is_set():
            self.publish({key: Section(key, why=why) for key in self.group.keys})

    def section(self, key: str) -> Section:
        """``key``'s section, once the job is in."""
        self._done.wait()
        return self._sections.get(key) or Section(key, why="the bundle job has no section for it")


# ------------------------------------------------------------------ formation

SINGLE = "single"
AWAIT = "await"
LEAD = "lead"


@dataclass(frozen=True)
class Assignment:
    """What one member of a sealed bundle does: run singly, or judge its section of ``job``."""

    role: str
    why: str = ""
    job: Job | None = None


class Seat:
    """One check's place in a bundle, from its admission to its verdict."""

    def __init__(self, board: "Board", bundle: "_Open", check: Any, leader: bool) -> None:
        self.board = board
        self.bundle = bundle
        self.check = check
        self.leader = leader
        self.assignment: Assignment | None = Assignment(LEAD) if leader else None
        self._assigned = threading.Event()
        if leader:
            self._assigned.set()

    def assign(self, assignment: Assignment) -> None:
        self.assignment = assignment
        self._assigned.set()

    def wait(self) -> Assignment:
        """What the leader assigned this member, once it has."""
        self._assigned.wait()
        assert self.assignment is not None
        return self.assignment

    def seal(self) -> list["Seat"]:
        """(The leader.) Close the bundle to new members, once its gathering time is up."""
        return self.board._seal(self)

    def leave(self) -> None:
        """The check's admission ended; see :meth:`Board._leave`."""
        self.board._leave(self)


@dataclass
class _Open:
    key: Any
    opened: float
    seats: list[Seat] = field(default_factory=list)
    sealed: bool = False


class Board:
    """The bundles taking members, by unit and member; shared by every check thread."""

    def __init__(
        self,
        gather_s: float = GATHER_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.gather_s = gather_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._open: dict[Any, _Open] = {}

    def join(self, key: Any, check: Any) -> Seat:
        """``check``'s seat in the bundle open for ``key``, which it leads when there is none."""
        with self._lock:
            bundle = self._open.get(key)
            if bundle is not None and not bundle.sealed:
                seat = Seat(self, bundle, check, leader=False)
                bundle.seats.append(seat)
                return seat
            bundle = _Open(key, self._clock())
            self._open[key] = bundle
            seat = Seat(self, bundle, check, leader=True)
            bundle.seats.append(seat)
            return seat

    def _seal(self, seat: Seat) -> list[Seat]:
        bundle = seat.bundle
        remaining = bundle.opened + self.gather_s - self._clock()
        if remaining > 0:
            self._sleep(remaining)
        with self._lock:
            bundle.sealed = True
            if self._open.get(bundle.key) is bundle:
                del self._open[bundle.key]
            return list(bundle.seats)

    def _leave(self, seat: Seat) -> None:
        """A member that leaves before the bundle is sealed is not in it.

        A leader that leaves before it sealed the bundle (its check was
        skipped) abandons it: every member runs singly. After sealing,
        leaving changes nothing: the leader assigned every member.
        """
        with self._lock:
            bundle = seat.bundle
            if bundle.sealed:
                return
            if not seat.leader:
                bundle.seats.remove(seat)
                return
            bundle.sealed = True
            if self._open.get(bundle.key) is bundle:
                del self._open[bundle.key]
            others = [s for s in bundle.seats if s is not seat]
        for other in others:
            other.assign(Assignment(SINGLE, why="the check that led its bundle did not run"))


__all__ = [
    "AWAIT",
    "BUILD_MARKER",
    "BUNDLE_DRIVER",
    "Assignment",
    "Board",
    "BundledResult",
    "GATHER_S",
    "Group",
    "Job",
    "KIND_DRIVER",
    "KIND_EVAL",
    "KIND_LOGITS",
    "LEAD",
    "Launch",
    "SECTION_MARKER",
    "SINGLE",
    "Seat",
    "Section",
    "bundle_command",
    "bundled",
    "capture",
    "engine_holds",
    "eval_declaration",
    "eval_engine",
    "eval_engine_difference",
    "form_groups",
    "mark",
    "parse_launch",
    "quickstart_engine",
    "replaying",
    "split",
    "without_max_tokens",
    "write_spec",
]
