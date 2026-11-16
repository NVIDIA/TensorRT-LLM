# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast path: the run's spec derived in seconds from the checkpoint, ``site.yaml`` and a goal.

``--checkpoint``, ``--site`` and ``--goal`` write ``task.derived.yaml`` (the spec
``--task`` would take), each deployment's native options beside it
(``engines/<name>.yaml``) and ``derive.report.md``, which says every value the goal did
not state and where it came from. What cannot be derived is an open question: the run
stops before it starts and names each, and the operator answers it in the spec. A feature
is brought up only when the goal names it: what the checkpoint carries beside (draft
layers, a vision or audio tower) the report lists with the word that adds it.

What a goal comes to is a list of deployments. Each member the goal names (``tp1``,
``tep4``, ``dep4``) is one deployment of the core configuration at that layout: CUDA
graphs and the overlap scheduler on, the KV cache in the checkpoint's own format, block
reuse and chunked prefill off. Each feature is one more deployment whose options turn it
on, each on its own, never a cross product: a drafter (``speculative_config``), a tower
(``inputs: [text, image]``), disaggregation (``ctx`` and ``gen``), each extra KV dtype
(``kv=``), chunked prefill and prefix reuse; the last four are judged against the first
deployment's own score (``relative_to``). The goal is one line of words, separated by
spaces, commas or semicolons; :func:`parse_goal` defines them.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.compute import pool as pool_mod
from ..infra.git import git_output
from . import deployments as deployments_mod
from .schema_keys import reject_unknown_keys
from .task_schema import DEFAULT_BOOT_MAX_TOKENS

#: The derived spec and its report, in the workspace.
SPEC_FILENAME = "task.derived.yaml"
REPORT_FILENAME = "derive.report.md"
#: Where each deployment's native options are written, beside the spec.
ENGINES_DIRNAME = "engines"
#: A parallel layout on N GPUs: tensor parallel (``tp``), with expert parallel too
#: (``tep``), and with attention data parallel as well (``dep``).
LAYOUT = re.compile(r"^(tp|tep|dep)([1-9]\d*)$")
#: The features that draft tokens a target model verifies.
SPECULATORS: frozenset[str] = frozenset({"mtp", "eagle3", "dflash", "dspark"})
DISAGG_FEATURE = "disagg"
#: The name of the disaggregated deployment a goal's ``disagg`` comes to.
DISAGG_DEPLOYMENT = "pd"
#: The features a goal may name. A drafter, a tower and disaggregation are brought up
#: by options this module writes; for the others it writes a deployment of the core
#: configuration and asks which options turn the feature on.
FEATURE_WORDS: frozenset[str] = frozenset(
    {
        *SPECULATORS,
        "multimodal",
        DISAGG_FEATURE,
        "kv_quant",
        "host_offload",
        "decoder_replay",
        "encoder_replay",
    }
)

#: The layout of the first deployment when neither the goal nor the site names any.
DEFAULT_MEMBERS: tuple[str, ...] = ("tp1",)
#: The workload when neither the goal nor the site states one. Stated rather
#: than left to the engine, which would size its cache for the checkpoint's
#: whole ``max_position_embeddings``.
DEFAULT_WORKLOAD: dict[str, int] = {"isl": 4096, "osl": 1024, "max_batch_size": 64}
#: The boot gate's prompts when the site declares none, as INTAKE states them: one fact
#: and one arithmetic word problem, each checked by a keyword any working target states.
DEFAULT_BOOT: dict[str, Any] = {
    "prompts": [
        {"prompt": "The capital of France is", "keywords_any": ["Paris"]},
        {
            "prompt": "Question: A crate holds 12 bottles. How many bottles do 6 crates "
            "hold? Answer:",
            "keywords_any": ["72"],
        },
    ],
    "max_tokens": DEFAULT_BOOT_MAX_TOKENS,
}
DEFAULT_BENCHMARK = "gsm8k"

#: Goal word -> its runtime switch: ``-<word>`` turns a core one off; naming a non-core
#: one (:data:`NON_CORE_SWITCHES`) adds a deployment that turns it on.
RUNTIME_WORDS: dict[str, str] = {
    "cuda_graph": "cuda_graph",
    "overlap_scheduler": "overlap_scheduler",
    "chunked_prefill": "chunked_prefill",
    "prefix_reuse": "kv_cache_reuse",
}
#: The runtime switches of the core configuration: on unless the goal turns them off.
CORE_SWITCHES: frozenset[str] = frozenset({"cuda_graph", "overlap_scheduler"})
#: Non-core runtime switch -> the deployment that turns it on: off in every other.
NON_CORE_SWITCHES: dict[str, str] = {"chunked_prefill": "chunked", "kv_cache_reuse": "reuse"}
#: Tokens per prefill chunk in the ``chunked`` deployment when the goal names none: below the
#: benchmark's few-shot prompts, so their prefill is split.
DEFAULT_CHUNK_TOKENS = 512
#: The features a goal may name.
UNIT_WORDS: frozenset[str] = FEATURE_WORDS
#: ``<feature>:<depth>``: how much the feature's deployment is judged by. ``smoke``: its
#: session's own checks and no benchmark; ``standard``: its benchmarks too.
DEPTH_SMOKE, DEPTH_STANDARD = "smoke", "standard"
DEPTH_WORDS: dict[str, str] = {"smoke": DEPTH_SMOKE, "standard": DEPTH_STANDARD}
#: Speculative algorithm -> its reference rows' ``spec_dec_algo`` and the LLM
#: API's ``speculative_config.decoding_type`` (the two spell them alike).
SPEC_DEC_ALGO = {"mtp": "MTP", "eagle3": "Eagle3", "dflash": "DFlash", "dspark": "DSpark"}
#: Where a config states its multi-token-prediction layers.
DRAFT_LAYER_KEYS: tuple[str, ...] = (
    "num_nextn_predict_layers",
    "num_mtp_modules",
    "mtp_num_hidden_layers",
)
WORKLOAD_KEYS = ("isl", "osl", "max_seq_len", "max_batch_size", "concurrency")
SITE_KEYS = (
    "trtllm_repo_path",
    "environment",
    "agents",
    "commit_policy",
    # The cluster's devices, and what a derived spec takes when its goal is silent.
    "device",
    "defaults",
)
SITE_DEVICE_KEYS = ("arch", "gpus_per_node", "nodes")
SITE_DEFAULTS_KEYS = ("members", "workload", "accuracy", "boot", "disagg")
#: What a site may state for the disaggregated deployment: its environment (the transport
#: the cluster needs), its timeout, and native tests that also judge it.
SITE_DISAGG_KEYS = ("env", "timeout", "tests")
#: ``reference`` (how the anchor was measured) is the operator's record; nothing reads it.
SITE_ACCURACY_KEYS = ("benchmark", "protocol", "reference", "reference_yaml", "model_key")

#: ``key=value`` goal words -> (what they set, type).
_PARAM_WORDS: dict[str, tuple[str, type]] = {
    "draft_len": ("draft_len", int),
    "drafter": ("drafter", str),
    "kv": ("kv_dtype", str),
    "memory": ("kv_fraction", float),
    "chunk": ("chunk_tokens", int),
    "snapshot": ("snapshot_interval", int),
}
_ACCURACY_WORDS = {
    "benchmark": "benchmark",
    "model_key": "model_key",
    "reference": "reference_yaml",
}
_KV_DTYPES = ("fp8", "nvfp4")
#: KV dtypes a goal may name for the checkpoint's own KV format.
_KV_AUTO = frozenset({"auto", "bf16", "bfloat16"})
#: Weight formats a reference row states no ``quant_algo`` for.
_UNQUANTIZED = frozenset({"", "none", "auto", "bf16", "bfloat16", "fp16", "float16", "fp32"})
_REPO_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class DeriveError(ValueError):
    """A goal or a ``site.yaml`` the spec cannot be derived from, every problem listed."""


@dataclass
class Goal:
    """What a ``--goal`` line asks for."""

    text: str = ""
    #: Unit -> whether the goal builds it (True) or puts it out of scope (False).
    units: dict[str, bool] = field(default_factory=dict)
    #: Unit -> the members the goal narrows it to.
    unit_members: dict[str, list[str]] = field(default_factory=dict)
    #: Feature -> the depth the goal gives it (``multimodal:smoke``).
    unit_profiles: dict[str, str] = field(default_factory=dict)
    #: The goal asks for speculative decoding by the checkpoint's own drafter (True),
    #: or for none at all (False).
    speculative: bool | None = None
    #: Runtime switch -> off (``-cuda_graph``).
    switches_off: set[str] = field(default_factory=set)
    #: Non-core runtime switches the goal names (``chunked_prefill``): one variant each.
    switches_on: set[str] = field(default_factory=set)
    members: list[str] = field(default_factory=list)
    workload: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, str] = field(default_factory=dict)
    arch: str | None = None


def parse_goal(text: str | None) -> Goal:
    """Read a goal line. Raises :class:`DeriveError` naming every unknown word."""
    goal = Goal(text=" ".join((text or "").split()))
    errors: list[str] = []
    unknown: list[str] = []
    for word in re.split(r"[\s,;]+", goal.text):
        if not word:
            continue
        if "=" in word:
            _key_value(goal, word, errors)
            continue
        lowered = word.lower()
        if LAYOUT.match(lowered):
            goal.members.append(lowered)
            continue
        if lowered == "text_only":
            goal.units["multimodal"] = False
            continue
        out = lowered.startswith(("-", "no-", "no_"))
        lowered, _, depth = lowered.partition(":")
        name, _, members = (
            lowered.lstrip("-").removeprefix("no-").removeprefix("no_").partition("@")
        )
        if depth and (out or name not in UNIT_WORDS or depth not in DEPTH_WORDS):
            errors.append(
                f"goal: {word!r}: a depth follows a feature and is one of {sorted(DEPTH_WORDS)}"
            )
            continue
        if depth:
            goal.unit_profiles[name] = DEPTH_WORDS[depth]
        if name in ("spec", "speculative"):
            goal.speculative = not out
        elif name in RUNTIME_WORDS and out:
            goal.switches_off.add(RUNTIME_WORDS[name])
        elif name in RUNTIME_WORDS:
            # A core switch is on already; a non-core one is tested as its own variant.
            if RUNTIME_WORDS[name] in NON_CORE_SWITCHES:
                goal.switches_on.add(RUNTIME_WORDS[name])
        elif name in UNIT_WORDS:
            goal.units[name] = not out
            if members and not out:
                listed = [m for m in members.split("/") if m]
                if not listed or not all(LAYOUT.match(m) for m in listed):
                    errors.append(
                        f"goal: {word!r} names no members of the form tp<N>, tep<N>, dep<N>"
                    )
                else:
                    goal.unit_members[name] = listed
        else:
            unknown.append(word)
            errors.append(f"goal: {word!r} is not a feature, a member or key=value")
    if unknown:
        known = sorted({*UNIT_WORDS, *RUNTIME_WORDS, "speculative", "text_only"})
        errors.append(
            f"goal: the words a goal knows are {', '.join(known)}; a member is tp<N>, "
            "tep<N> or dep<N>; a word is left out with a leading `-`"
        )
    if errors:
        raise DeriveError("\n".join(errors))
    return goal


def _key_value(goal: Goal, word: str, errors: list[str]) -> None:
    key, _, value = word.partition("=")
    key = key.lower()
    try:
        if key == "concurrency":
            goal.workload[key] = [int(v) for v in value.split("/") if v]
        elif key in WORKLOAD_KEYS:
            goal.workload[key] = int(value)
        elif key == "arch":
            goal.arch = value.lower()
        elif key in _ACCURACY_WORDS:
            goal.accuracy[_ACCURACY_WORDS[key]] = value
        elif key == "kv":
            dtypes = ["auto" if v in _KV_AUTO else v for v in value.lower().split("/") if v]
            wrong = [v for v in dtypes if v not in ("auto", *_KV_DTYPES)]
            if wrong or not dtypes:
                errors.append(f"goal: kv={value}: each of {['auto', 'bf16', *_KV_DTYPES]}")
            elif "kv_dtype" in goal.params:
                errors.append(
                    f"goal: {word!r}: kv= is named once, its dtypes in one word "
                    "(kv=bf16/fp8/nvfp4): the first is every deployment's, each other its own"
                )
            else:
                goal.params["kv_dtype"] = dtypes[0]
                goal.params["kv_variants"] = list(
                    dict.fromkeys(d for d in dtypes[1:] if d != dtypes[0])
                )
        elif key in _PARAM_WORDS:
            param, kind = _PARAM_WORDS[key]
            goal.params[param] = kind(value)
        else:
            errors.append(f"goal: {key!r} in {word!r} is not a key the goal takes")
    except ValueError:
        errors.append(f"goal: {word!r} needs a number")


def read_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """The checkpoint's ``config.json``, its ``text_config`` lifted to the top; None if unreadable."""
    try:
        data = json.loads((Path(str(checkpoint)) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # A multimodal checkpoint keeps the language model's shape under text_config.
    if isinstance(data.get("text_config"), dict):
        data = {**data["text_config"], **{k: v for k, v in data.items() if k != "text_config"}}
    return data


#: How the engine's sources name a registry of its built-in custom tokenizers: a mapping
#: from an alias to a class. A pattern to search the checkout with, never a path.
_TOKENIZER_REGISTRY = r"^[A-Z_]*TOKENIZER[A-Z_]*ALIAS[A-Z_]* *= *\{"


def registered_tokenizer(repo: str | Path, model_type: Any) -> str | None:
    """The alias the engine registers a custom tokenizer under for this ``model_type``, if any.

    Searched in the engine's checkout as derive runs: a module-level mapping whose name
    says tokenizer aliases, one of whose keys is the checkpoint's own ``model_type``. Such
    a model's chat template is the tokenizer's, so a session that leaves the option out
    asks the API with another prompt than the model was trained on. ``None`` when the
    engine registers none for it or the checkout cannot be searched.
    """
    if not isinstance(model_type, str) or not model_type:
        return None
    found = git_output(repo, "grep", "-l", "-E", _TOKENIZER_REGISTRY, "--", "*.py") or ""
    for name in found.split():
        try:
            tree = ast.parse((Path(repo) / name).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                if re.match(_TOKENIZER_REGISTRY, ast.unparse(node)) and model_type in keys:
                    return model_type
    return None


#: Where the engine keeps the recipes it serves and tests a model with.
RECIPE_PLACES: tuple[str, ...] = ("tests/integration/defs", "examples")


def engine_recipes(repo: str | Path, model_type: Any, limit: int = 5) -> list[str]:
    """The engine's test and example files named for ``model_type``: where its own recipe is.

    Searched in the engine's checkout as derive runs, as the tokenizer is: the flow
    keeps no table of models. Only their paths: which of the options such a file sets
    a deployment needs is for a person to say, so derive names the files and reads none.
    """
    name = str(model_type or "").strip().lower()
    if not name:
        return []
    listed = git_output(repo, "ls-files", "--", *RECIPE_PLACES) or ""
    return [path for path in listed.splitlines() if name in path.lower()][:limit]


def draft_layers(config: Mapping[str, Any]) -> int:
    """How many draft (MTP) layers the checkpoint carries; 0 for none."""
    for key in DRAFT_LAYER_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


@dataclass(frozen=True)
class CheckpointFacts:
    """What a derivation reads from a checkpoint directory."""

    config: dict[str, Any]
    architecture: str
    #: The weight format and the KV-cache format the checkpoint was quantized to.
    quant_algo: str | None
    kv_cache_quant_algo: str | None
    #: A quantization the checkpoint declares without a readable ``quant_algo``.
    unread_quantization: str | None
    #: ``mtp`` or ``dspark`` for draft layers in the checkpoint, and their depth.
    speculator: str | None
    draft_len: int | None
    towers: tuple[str, ...]
    #: The ``<org>/<model>`` name the config states, the reference rows' key.
    name: str | None


def inspect_checkpoint(path: Path) -> CheckpointFacts:
    """Read ``path``'s config files. Raises :class:`DeriveError` without a ``config.json``."""
    config = read_config(path)
    if config is None:
        raise DeriveError(f"{path}: no readable config.json; this is not a HuggingFace checkpoint")
    quant = config.get("quantization_config")
    quant = dict(quant) if isinstance(quant, Mapping) else {}
    try:
        hf_quant = json.loads((Path(path) / "hf_quant_config.json").read_text(encoding="utf-8"))
        quant = {**dict(hf_quant.get("quantization") or {}), **quant}
    except (OSError, ValueError, AttributeError):
        pass
    quant_algo = quant.get("quant_algo")
    method = str(quant.get("quant_method") or "").lower()
    if not quant_algo and method == "fp8":
        quant_algo = "FP8_BLOCK_SCALES" if quant.get("weight_block_size") else "FP8"
    speculator, draft_len = None, None
    if any(str(k).startswith("dspark") for k in config):
        speculator, draft_len = "dspark", config.get("dspark_block_size")
    elif draft_layers(config):
        speculator, draft_len = "mtp", draft_layers(config)
    name = config.get("_name_or_path")
    return CheckpointFacts(
        config=config,
        architecture=str((config.get("architectures") or ["?"])[0]),
        quant_algo=str(quant_algo) if quant_algo else None,
        kv_cache_quant_algo=str(quant.get("kv_cache_quant_algo") or "") or None,
        unread_quantization=(method or "a quantization_config without quant_algo")
        if quant and not quant_algo
        else None,
        speculator=speculator,
        draft_len=int(draft_len) if isinstance(draft_len, int) and draft_len > 0 else None,
        towers=tuple(k for k in ("vision_config", "audio_config") if k in config),
        name=str(name) if isinstance(name, str) and _REPO_NAME.match(name) else None,
    )


def _state_environment(environment: dict[str, Any], unstated: list) -> None:
    """Write the environment's kind into the spec when the site does not state it, and say so.

    What the site's block names decides (Slurm fields, a spool somebody else's watchers
    serve). A block that names nothing is Slurm where ``sbatch`` is on this host and this one
    host otherwise: detected here, written down, never chosen silently at the start.
    """
    if environment.get("kind"):
        return
    named = any(environment.get(key) for key in ("slurm", "spool"))
    kind = pool_mod.environment_kind(environment)
    if named:
        source = "the site's environment block"
    elif shutil.which("sbatch"):
        source = "detected: sbatch is on this host; state environment.slurm"
    else:
        kind = pool_mod.KIND_LOCAL
        source = "detected: no sbatch on this host, so this one host"
    environment["kind"] = kind
    unstated.append(("environment.kind", kind, source))


def _environment_owed(environment: Mapping[str, Any]) -> list[str]:
    """What a run cannot start without and the environment does not state, one question each.

    The same values the spec's load and the start's preflight ask for, said here together:
    a user with one host and no site file has nobody else to state them.
    """
    owed = []
    if not str(environment.get("python") or "").strip():
        owed.append(
            "environment.python: the absolute path of the Python that has TensorRT-LLM "
            "installed (`<venv>/bin/python`); every check runs under it"
        )
    env = environment.get("env")
    unset = [name for name in SCRATCH_VARIABLES if not (isinstance(env, Mapping) and env.get(name))]
    if unset:
        owed.append(
            f"environment.env: set {' and '.join(unset)} to a scratch directory of the run's "
            "own, not under /tmp and short (a socket is bound under TMPDIR)"
        )
    return owed


def read_site(path: str | Path) -> Any:
    """The site file at ``path``. Raises :class:`DeriveError`, naming ``--site``, when none can be read."""
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        why = getattr(exc, "strerror", None) or str(exc).splitlines()[0]
        raise DeriveError(f"--site {path}: no site file can be read there ({why})") from exc


def validate_site(site: Any) -> list[str]:
    """Every problem with the shape of a ``site.yaml``."""
    if not isinstance(site, Mapping):
        return ["site.yaml must be a mapping"]
    errors: list[str] = []
    reject_unknown_keys(site, SITE_KEYS, errors, where="site.yaml")
    for key in ("trtllm_repo_path", "environment"):
        if not site.get(key):
            errors.append(f"site.yaml: '{key}' is required")
    environment = site.get("environment")
    if isinstance(environment, Mapping):
        kind = environment.get("kind")
        if environment.get("pool") is not None or (kind and kind not in pool_mod.KINDS):
            # Neither Slurm nor one host: the same message the spec's own load gives.
            named = f"kind: {kind}" if kind and kind not in pool_mod.KINDS else "pool"
            errors.append(
                f"site.yaml: 'environment' names an environment the flow has no launcher for "
                f"(`environment.{named}`): {pool_mod.OTHER_ENVIRONMENT}"
            )
    for key, keys in (("device", SITE_DEVICE_KEYS),):
        block = site.get(key)
        if block is not None and not isinstance(block, Mapping):
            errors.append(f"site.yaml: '{key}' must be a mapping")
        elif block:
            reject_unknown_keys(block, keys, errors, where=f"site.yaml {key}")
    defaults = site.get("defaults")
    if defaults is None:
        return errors
    if not isinstance(defaults, Mapping):
        return [*errors, "site.yaml: 'defaults' must be a mapping"]
    reject_unknown_keys(defaults, SITE_DEFAULTS_KEYS, errors, where="site.yaml defaults")
    members = defaults.get("members")
    if members is not None and (
        not isinstance(members, list)
        or not members
        or not all(isinstance(m, str) and LAYOUT.match(m) for m in members)
    ):
        errors.append(
            "site.yaml: 'defaults.members' must be a non-empty list of tp<N>, tep<N>, dep<N>"
        )
    for key, keys in (
        ("workload", WORKLOAD_KEYS),
        ("accuracy", SITE_ACCURACY_KEYS),
        ("boot", None),
        ("disagg", SITE_DISAGG_KEYS),
    ):
        block = defaults.get(key)
        if block is not None and not isinstance(block, Mapping):
            errors.append(f"site.yaml: 'defaults.{key}' must be a mapping")
        elif block and keys:
            reject_unknown_keys(block, keys, errors, where=f"site.yaml defaults.{key}")
    return errors


@dataclass
class Derivation:
    """A derived spec, what it holds that the goal did not state, and what is still open."""

    #: The spec without its ``deployments``: :func:`write_derivation` adds them.
    spec: dict[str, Any]
    #: The deployments, the first one first: each as a spec states it, its ``options``
    #: (or each role's) still the mapping :func:`write_derivation` writes to a file.
    deployments: list[dict[str, Any]]
    #: ``(field, value, source)`` for every value read from the checkpoint or the
    #: site, or defaulted.
    unstated: list[tuple[str, Any, str]] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    #: The checkpoint, site and goal it was derived from, and when.
    derived: dict[str, str] = field(default_factory=dict)


def layout_options(layout: str, base: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` (the core configuration) at one parallel layout, as native options.

    ``base`` states the batch as the global total: under attention data parallel the
    engine's batch, and its CUDA graphs', are each rank's share of it.
    """
    found = LAYOUT.match(layout)
    if not found:
        raise DeriveError(f"{layout!r} is not a parallel layout: tp<N>, tep<N> or dep<N>")
    kind, size = found.group(1), int(found.group(2))
    options = json.loads(json.dumps(dict(base)))
    total = options.get("max_batch_size")
    if total and kind == "dep":
        options["max_batch_size"] = max(1, -(-int(total) // size))
    graphs = options.get("cuda_graph_config")
    if isinstance(graphs, dict) and options.get("max_batch_size"):
        graphs["max_batch_size"] = options["max_batch_size"]
    options.update(
        tensor_parallel_size=size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1 if kind == "tp" else size,
        enable_attention_dp=kind == "dep",
    )
    return options


def _overlaid(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` laid over ``base``, mapping by mapping."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _overlaid(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_spec(
    checkpoint: Path, site: Mapping[str, Any], goal: Goal, *, site_path: Path | None = None
) -> Derivation:
    """The spec ``goal`` asks of ``checkpoint`` on ``site``; see the module docstring."""
    errors = validate_site(site)
    if errors:
        raise DeriveError("\n".join(errors))
    checkpoint = Path(checkpoint).resolve()
    facts = inspect_checkpoint(checkpoint)
    defaults = dict(site.get("defaults") or {})
    unstated: list[tuple[str, Any, str]] = []
    questions: list[str] = []

    def note(where: str, value: Any, source: str) -> Any:
        unstated.append((where, value, source))
        return value

    arch = goal.arch or (site.get("device") or {}).get("arch")
    if arch and not goal.arch:
        note("arch", arch, "site")
    if not arch:
        questions.append(
            "arch: name the GPU architecture (arch=sm_100 in --goal, or `device: {arch: ...}` "
            "in site.yaml); receipts are keyed by it and it is never guessed"
        )
    if facts.unread_quantization:
        questions.append(
            f"the checkpoint is quantized ({facts.unread_quantization}) and states no "
            "quant_algo: set accuracy.discriminators to the reference row's (quant_algo), so "
            "the anchor is not the unquantized model's"
        )

    # The layouts: one deployment of the core configuration each.
    if goal.members:
        members = list(dict.fromkeys(goal.members))
    elif defaults.get("members"):
        members = note("members", list(dict.fromkeys(defaults["members"])), "site")
    else:
        members = note("members", list(DEFAULT_MEMBERS), "default")

    # The workload.
    workload = dict(defaults.get("workload") or {})
    source = "site" if workload else "default"
    if not workload:
        workload = dict(DEFAULT_WORKLOAD)
        reach = facts.config.get("max_position_embeddings")
        if isinstance(reach, int) and workload["isl"] + workload["osl"] > reach:
            # A default must fit the model; a stated workload that does not is a block.
            workload["osl"] = min(workload["osl"], reach // 4)
            workload["isl"] = reach - workload["osl"]
    for key, value in workload.items():
        if key not in goal.workload and key != "max_batch_size":
            note(f"workload.{key}", value, source)
    workload.update(goal.workload)
    concurrency = workload.get("concurrency") or []
    max_seq_len = workload.get("max_seq_len") or (
        workload["isl"] + workload["osl"] if "isl" in workload and "osl" in workload else None
    )
    max_batch_size = workload.get("max_batch_size") or (max(concurrency) if concurrency else None)
    if concurrency and "max_batch_size" not in goal.workload:
        # A default batch below the goal's concurrency would queue the requests it names.
        max_batch_size = max(int(max_batch_size or 0), max(concurrency))
    if max_batch_size and "max_batch_size" not in goal.workload:
        # The value the spec carries, not the default it was raised from.
        note(
            "workload.max_batch_size",
            max_batch_size,
            source if max_batch_size == workload.get("max_batch_size") else "concurrency",
        )

    # The features: those the goal names. What the checkpoint carries beside them (its own
    # drafter, its towers) is certain; whether this run onboards it is the user's to say, and
    # a first run is one deployment. The report names each with the word that adds it.
    built: list[str] = []
    speculator = next((u for u, on in goal.units.items() if on and u in SPECULATORS), None)
    own = facts.speculator
    if speculator == "mtp" and own and own != "mtp" and not goal.params.get("drafter"):
        # "MTP" is what a goal calls a checkpoint's own draft layers. This checkpoint's are
        # another algorithm's and the goal names no drafter of its own: the checkpoint's is
        # written, and the report says so. Ordinary MTP at depth 1 would be written
        # otherwise, for layers the engine drives another way.
        speculator = note(
            "features (speculator)",
            own,
            f"checkpoint: the goal says `mtp`, and the checkpoint's own draft layers are "
            f"{SPEC_DEC_ALGO[own]}'s; say `{own}` in --goal, or name another drafter with "
            "`drafter=`",
        )
        for said in (goal.units, goal.unit_members, goal.unit_profiles):
            if "mtp" in said:
                said[own] = said.pop("mtp")
    if speculator is None and goal.speculative:
        if own:
            speculator = note("features (speculator)", own, "checkpoint")
        else:
            questions.append(
                "speculative: the goal asks for speculative decoding and the checkpoint carries "
                "no draft layers; name the algorithm (mtp, eagle3, dflash or dspark)"
            )
    if speculator:
        built.append(speculator)
        if "draft_len" not in goal.params and facts.draft_len and speculator == facts.speculator:
            goal.params["draft_len"] = note("draft_len", facts.draft_len, "checkpoint")
    elif own and goal.speculative is None and goal.units.get(own) is None:
        note(
            "features (left out)",
            own,
            f"the checkpoint carries {SPEC_DEC_ALGO[own]} draft layers: `{own}` in --goal adds "
            "their deployment",
        )
    multimodal = goal.units.get("multimodal")
    if multimodal:
        built.append("multimodal")
    elif multimodal is None and facts.towers:
        note(
            "features (left out)",
            "multimodal",
            "the checkpoint carries a tower: `multimodal` in --goal adds its deployment",
        )
    built += [
        u for u, on in goal.units.items() if on and u not in SPECULATORS and u != "multimodal"
    ]

    # The core configuration: the core switches on unless the goal turns them off, the rest off.
    cuda_graph = "cuda_graph" not in goal.switches_off
    for switch in sorted(set(NON_CORE_SWITCHES) - goal.switches_on - goal.switches_off):
        note(switch, False, "core configuration: a non-core feature")
    chunked = "chunked_prefill" in goal.switches_on
    snapshot = goal.params.get("snapshot_interval")
    kv_dtype = str(goal.params.get("kv_dtype") or "").lower()
    native_kv = str(facts.kv_cache_quant_algo or "").lower()
    if kv_dtype in ("", "auto") and native_kv in _KV_DTYPES:
        kv_dtype = note("kv_cache dtype", native_kv, "checkpoint")
    kv: dict[str, Any] = {"enable_block_reuse": False}
    if kv_dtype in _KV_DTYPES:
        kv["dtype"] = kv_dtype
    if goal.params.get("kv_fraction"):
        kv["free_gpu_memory_fraction"] = float(goal.params["kv_fraction"])
    core: dict[str, Any] = {
        "cuda_graph_config": {"enable_padding": True} if cuda_graph else None,
        "enable_chunked_prefill": False,
        "kv_cache_config": kv,
        "speculative_config": None,
    }
    if "overlap_scheduler" in goal.switches_off:
        core["disable_overlap_scheduler"] = True
    if max_seq_len:
        core["max_seq_len"] = int(max_seq_len)
    if max_batch_size:
        core["max_batch_size"] = int(max_batch_size)
    if goal.params.get("chunk_tokens") and not chunked:
        core["max_num_tokens"] = int(goal.params["chunk_tokens"])
    # A model the engine registers a tokenizer of its own for is served with it.
    tokenizer = registered_tokenizer(site["trtllm_repo_path"], facts.config.get("model_type"))
    if tokenizer:
        core["custom_tokenizer"] = note(
            "custom_tokenizer",
            tokenizer,
            "the engine registers one for the checkpoint's model_type",
        )
    # Code the checkpoint ships (`auto_map`) is loaded only when the options allow it.
    for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        try:
            shipped = json.loads((checkpoint / name).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(shipped, Mapping) and "auto_map" in shipped:
            core["trust_remote_code"] = True
    # What else this architecture needs to serve is in no file derive can read for certain:
    # the engine's own recipes for it are named, for a person to compare.
    model_type = facts.config.get("model_type")
    recipes = engine_recipes(site["trtllm_repo_path"], model_type)
    if recipes:
        questions.append(
            f"engine options: the files under `{ENGINES_DIRNAME}/` hold what every model takes"
            + (" and the tokenizer the engine registers" if tokenizer else "")
            + ", and nothing of this architecture's own (a sparse-attention algorithm, say, or "
            "`trust_remote_code` where the checkpoint names no `auto_map`). Derive reads no "
            f"recipe. The engine's tree has files named for `{model_type}`: "
            f"{', '.join(recipes)}. Add to each options file what they set and your "
            "deployment needs"
        )

    # The accuracy defaults every benchmark takes.
    site_accuracy = dict(defaults.get("accuracy") or {})
    for key, value in site_accuracy.items():
        if key not in goal.accuracy:
            note(f"accuracy.{key}", value, "site")
    anchor = {**site_accuracy, **goal.accuracy}
    benchmark = anchor.get("benchmark") or note("accuracy.benchmark", DEFAULT_BENCHMARK, "default")
    accuracy: dict[str, Any] = {"benchmark": benchmark}
    if anchor.get("reference_yaml"):
        accuracy["reference_yaml"] = anchor["reference_yaml"]
    model_key = anchor.get("model_key") or (
        note("accuracy.model_key", facts.name, "checkpoint") if facts.name else None
    )
    if model_key:
        accuracy["model_key"] = model_key
    else:
        questions.append(
            "accuracy.model_key: name the model's HuggingFace id (`org/model`, model_key= in "
            "--goal): the reference rows are keyed by it, and config.json names none (the "
            "fallback, the checkpoint path's last two components, is no model's name)"
        )
    benchmark_args = (
        ["--apply_chat_template"] if (anchor.get("protocol") or {}).get("chat_template") else []
    )
    weights = str(facts.quant_algo or "").strip()
    if weights.lower() not in _UNQUANTIZED:
        # The checkpoint's weight format; a deployment's KV dtype and drafter tag its own row.
        accuracy["discriminators"] = {"quant_algo": weights.upper()}
    if defaults.get("boot"):
        accuracy["boot"] = note("accuracy.boot", dict(defaults["boot"]), "site")
    else:
        accuracy["boot"] = note("accuracy.boot", json.loads(json.dumps(DEFAULT_BOOT)), "default")

    def scored(depth: str | None) -> dict[str, Any]:
        """What a deployment of ``depth`` states of its judging, beside its options."""
        if depth == DEPTH_SMOKE:
            return {"benchmarks": []}
        entry: dict[str, Any] = {}
        if benchmark_args:
            entry["benchmarks"] = [{"name": benchmark, "args": list(benchmark_args)}]
        return entry

    # The deployments: the core configuration at each layout, then each feature on its own.
    first = members[0]
    found: list[dict[str, Any]] = [
        {"name": member, "options": layout_options(member, core), **scored(None)}
        for member in members
    ]

    def feature(name: str, overlay: Mapping[str, Any], **entry: Any) -> None:
        """One deployment per layout the goal gives ``name``: the first layout unsaid."""
        layouts = list(dict.fromkeys(goal.unit_members.get(name) or [first]))
        for member in layouts:
            options = _overlaid(layout_options(member, core), overlay)
            label = name if len(layouts) == 1 else f"{name}_{member}"
            found.append(
                {
                    "name": label,
                    "options": options,
                    **scored(goal.unit_profiles.get(name)),
                    **entry,
                }
            )

    for name in built:
        if name in SPEC_DEC_ALGO:
            draft: dict[str, Any] = {
                "decoding_type": SPEC_DEC_ALGO[name],
                "max_draft_len": int(goal.params.get("draft_len") or 1),
            }
            if goal.params.get("drafter"):
                draft["speculative_model"] = str(goal.params["drafter"])
            feature(name, {"speculative_config": draft})
        elif name == "multimodal":
            feature(name, {}, inputs=["text", "image"])
        elif name == DISAGG_FEATURE:
            environment = site.get("environment") or {}
            per_node = (
                (site.get("device") or {}).get("gpus_per_node")
                or (environment.get("slurm") or {}).get("gpus_per_node")
                or environment.get("gpus_per_node")
            )
            found.append(_disaggregated(defaults, core, goal, first, scored, note, per_node))
        else:
            feature(name, {})
            questions.append(
                f"{name}: its deployment `{name}` was written with the core options; state in "
                f"`{ENGINES_DIRNAME}/{name}.yaml` the options that turn {name} on"
            )
    for dtype in goal.params.get("kv_variants") or ():
        found.append(
            {
                "name": f"kv_{dtype}",
                "options": _overlaid(
                    layout_options(first, core), {"kv_cache_config": {"dtype": dtype}}
                ),
                "relative_to": first,
            }
        )
    for switch in sorted(goal.switches_on):
        if switch == "chunked_prefill":
            tokens = int(goal.params.get("chunk_tokens") or DEFAULT_CHUNK_TOKENS)
            overlay: dict[str, Any] = {"enable_chunked_prefill": True, "max_num_tokens": tokens}
        else:
            cache: dict[str, Any] = {"enable_block_reuse": True}
            if snapshot:
                cache["mamba_state_config"] = {"periodic_snapshot_interval": int(snapshot)}
            overlay = {"kv_cache_config": cache}
        found.append(
            {
                "name": NON_CORE_SWITCHES[switch],
                "options": _overlaid(layout_options(first, core), overlay),
                "relative_to": first,
            }
        )

    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": site["trtllm_repo_path"],
        **({"arch": arch} if arch else {}),
        "environment": dict(site["environment"]),
        "accuracy": accuracy,
    }
    _state_environment(spec["environment"], unstated)
    questions += _environment_owed(spec["environment"])
    _fit_sessions(spec, found, note)
    _no_launcher(spec, found, note)
    if site.get("commit_policy"):
        spec["commit_policy"] = dict(site["commit_policy"])
    if site.get("agents"):
        spec["agents"] = dict(site["agents"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    derived = {
        "checkpoint": str(checkpoint),
        "architecture": facts.architecture,
        "site": str(Path(site_path).resolve()) if site_path else "",
        "goal": goal.text,
        "at": stamp,
    }
    return Derivation(spec, found, unstated, questions, derived)


def _no_launcher(spec: Mapping[str, Any], found: list[dict[str, Any]], note: Any) -> None:
    """List each deployment a start would refuse for want of a launcher, in the start's words.

    A deployment wider than one node where no ``environment.slurm`` says how a step crosses
    nodes. Not a question: the spec as written starts, and that deployment's session is
    refused before any load.
    """
    from ..checks import session as session_mod

    probe = {**spec, "deployments": found}
    try:
        resolved = deployments_mod.resolve(probe)
    except Exception:  # a draft that does not resolve yet is the lint's to report
        return
    for deployment in resolved:
        try:
            session_mod.place(probe, deployment)
        except session_mod.WiderThanOneNode as exc:
            note(f"deployments.{deployment.name}", "refused before any load", str(exc))


def _fit_sessions(spec: Mapping[str, Any], found: list[dict[str, Any]], note: Any) -> None:
    """Give each deployment whose session outlasts one holder of the site a limit to start with.

    The spool never admits a job that cannot finish in its allocation, so a session longer
    than one holder would wait for ever, and the next command refuses the spec. A
    deployment that states no ``timeout`` is allowed what the flow allows its checks (half
    an hour to boot, four hours a benchmark): where that is more than one holder admits
    (``pool.fit_wall_s``: the rule preflight, the pool and the scheduler's fit hold a job
    to), it gets half of what one holder admits. Derive does not know how long this
    checkpoint loads and scores, and the spec as written must start. The longest limit a
    holder admits starts only in the holder's first minutes and, hung, costs all of it;
    half starts anywhere in about a holder's first half and costs half. It is a value the
    spec leaves unstated, listed with that reason and with where the real one is read
    (``note``); it is no question. A site whose holders have no end, or state none,
    changes nothing.
    """
    from ..checks import session as session_mod

    try:
        holder = pool_mod.holder_time_s(spec.get("environment") or {})
    except (KeyError, TypeError, ValueError):
        holder = None
    if not holder:
        return
    probe = {**spec, "deployments": found}
    try:
        resolved = {d.name: d for d in deployments_mod.resolve(probe)}
    except Exception:  # a draft that does not resolve yet is the lint's to report
        return
    longest = pool_mod.fit_wall_s(holder)
    fits = int(longest - session_mod.TEARDOWN_S) // 60 * 60
    start = fits // 2 // 60 * 60
    for entry in found:
        deployment = resolved.get(str(entry.get("name")))
        if deployment is None or entry.get("timeout") or start <= 0:
            continue
        wall = session_mod.session_timeout(probe, deployment) + session_mod.TEARDOWN_S
        if wall <= longest:
            continue
        entry["timeout"] = note(
            f"deployments.{entry['name']}.timeout",
            start,
            f"a starting value: half of what one holder of the site admits ({fits}s of its "
            f"{int(holder)}s; unstated, the session would be allowed {int(wall)}s, which is "
            "never admitted). With it a session starts anywhere in about a holder's first "
            "half, and one that hangs costs half a holder. The line the first session prints "
            "ends with its seconds: put that, with room, in its place",
        )


def _disaggregated(
    defaults: Mapping[str, Any],
    core: Mapping[str, Any],
    goal: Goal,
    first: str,
    scored: Any,
    note: Any,
    per_node: Any = None,
) -> dict[str, Any]:
    """The disaggregated deployment a goal's ``disagg`` comes to.

    A context and a generation server, each at one of the layouts the goal gives
    ``disagg`` (``disagg@tp1/tp2``: context at the first, generation at the second;
    one layout, or none, for both), each stating what moves the cache between them, and
    judged against the first deployment's own score where no reference row fits. The
    site's ``defaults.disagg`` gives its environment (the transport the cluster needs),
    its timeout and native tests that also judge it.

    A goal that gives no layout puts the first layout on both sides: twice its devices.
    The pair is written as the goal's words say. Where the site states a node's devices
    (``per_node``) and the pair does not fit one node, the report says so and names the
    word that states the two layouts: it runs as a job of whole nodes.
    """
    stated = goal.unit_members.get(DISAGG_FEATURE)
    layouts = stated or [first]
    sides = {"ctx": layouts[0], "gen": layouts[-1]}

    def width(pair: Mapping[str, str]) -> int:
        return sum(int(LAYOUT.match(layout).group(2)) for layout in pair.values())

    node = int(per_node) if isinstance(per_node, int) and per_node > 0 else 0
    if node and width(sides) > node:
        note(
            f"{DISAGG_DEPLOYMENT} (wider than a node)",
            f"{width(sides)} devices",
            f"context `{sides['ctx']}` and generation `{sides['gen']}` together, on a site "
            f"whose node has {node}: its session runs as one job of whole nodes "
            f"(`{DISAGG_FEATURE}@<ctx>/<gen>` in --goal names the two layouts)",
        )
    moves = {deployments_mod.TRANSCEIVER: {"backend": "DEFAULT"}}
    entry: dict[str, Any] = {
        "name": DISAGG_DEPLOYMENT,
        **{
            role: {"options": _overlaid(layout_options(layout, core), moves)}
            for role, layout in sides.items()
        },
        "relative_to": first,
        **scored(goal.unit_profiles.get(DISAGG_FEATURE)),
    }
    block = dict(defaults.get("disagg") or {})
    if block:
        note(DISAGG_DEPLOYMENT, dict(block), "site")
        entry.update({key: block[key] for key in SITE_DISAGG_KEYS if block.get(key)})
    return entry


#: The scratch variables a run's environment must state (``environment.check_tmpdir``).
SCRATCH_VARIABLES = ("TMPDIR", "CLAUDE_CODE_TMPDIR")
#: What running the deriving command again does to the files it wrote, by the command.
#: ``intake derive`` writes them anew; the command that derives and runs keeps a spec it finds.
REWRITTEN = (
    "`intake derive` writes the spec, each deployment's options under `engines/` and this "
    "report anew every time it runs: what you edit here is lost when it runs again into the "
    "same directory. Edit a copy, or derive into another directory."
)
KEPT = (
    "Edit it and run the same command to go on with the spec as it stands, your edits "
    "included; `--clean` derives it again, over them."
)


def render_report(derivation: Derivation, spec_path: Path, again: str = REWRITTEN) -> str:
    """``derive.report.md``: what was read, what was defaulted, and what is still open.

    ``again`` says what running the command again does to these files (:data:`REWRITTEN`,
    :data:`KEPT`): it differs by the command that derived them.
    """
    derived = derivation.derived
    lines = [
        "# The derived spec",
        "",
        f"Derived from `{derived.get('checkpoint')}` ({derived.get('architecture')}), the site "
        f"`{derived.get('site') or '(unnamed)'}` and the goal `{derived.get('goal') or '(none)'}` "
        f"at {derived.get('at')}, into `{spec_path}`. {again}",
        "",
        "## What the goal did not state",
        "",
        "| field | from | value |",
        "|---|---|---|",
    ]
    for where, value, source in derivation.unstated:
        shown = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        lines.append(f"| `{where}` | {source} | `{shown}` |")
    lines += ["", f"## Open questions ({len(derivation.questions)})", ""]
    lines += [f"- {q}" for q in derivation.questions] or ["None."]
    return "\n".join(lines) + "\n"


def write_derivation(workspace: Path, derivation: Derivation, again: str = REWRITTEN) -> Path:
    """Write the spec, each deployment's options and the report into ``workspace``; the spec's path.

    ``engines/<name>.yaml`` for an aggregated deployment, ``engines/<name>.ctx.yaml``
    and ``.gen.yaml`` for a disaggregated one; the spec names each beside itself. Every
    file is written whole, over one that was there; the report says so (``again``).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    engines = workspace / ENGINES_DIRNAME
    engines.mkdir(parents=True, exist_ok=True)

    def written(name: str, options: Mapping[str, Any]) -> str:
        (engines / f"{name}.yaml").write_text(
            yaml.safe_dump(dict(options), sort_keys=False), encoding="utf-8"
        )
        return f"{ENGINES_DIRNAME}/{name}.yaml"

    stated = []
    for entry in derivation.deployments:
        entry = dict(entry)
        if "options" in entry:
            entry["options"] = written(entry["name"], entry["options"])
        for role in deployments_mod.ROLES:
            if role in entry:
                options = written(f"{entry['name']}.{role}", entry[role]["options"])
                entry[role] = {**entry[role], "options": options}
        stated.append(entry)
    spec = {**derivation.spec, deployments_mod.DEPLOYMENTS_FIELD: stated}
    path = workspace / SPEC_FILENAME
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (workspace / REPORT_FILENAME).write_text(
        render_report(derivation, path, again), encoding="utf-8"
    )
    return path


__all__ = [
    "DEFAULT_BOOT",
    "DEFAULT_MEMBERS",
    "DEFAULT_WORKLOAD",
    "DRAFT_LAYER_KEYS",
    "ENGINES_DIRNAME",
    "REPORT_FILENAME",
    "SPEC_DEC_ALGO",
    "SPEC_FILENAME",
    "CheckpointFacts",
    "DeriveError",
    "Derivation",
    "Goal",
    "derive_spec",
    "draft_layers",
    "inspect_checkpoint",
    "layout_options",
    "parse_goal",
    "read_config",
    "render_report",
    "validate_site",
    "write_derivation",
]
