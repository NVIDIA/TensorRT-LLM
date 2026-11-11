# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec's shorthand: a small table from its older keys to deployments.

    parallel: tp1                  # or `members: [tp1, tep4, dep16]`
    features: [mtp, disagg]
    variants: {kv_fp8: {llm_api_options: {kv_cache_config: {dtype: fp8}}}}
    runtime: {cuda_graph: true, extra_flags: [--kv_cache_fraction=0.7]}
    accuracy: {llm_api_options: {core: llm_api/core.yaml}, max_seq_len: 9216}

- each member listed is one deployment, named by the member; the first is core's;
- each feature and each variant is one more deployment on the first member, named by
  itself, judged against the first's own score where no reference row fits;
- ``disagg`` is one disaggregated deployment whose two servers take core's options at the
  member ``members: {disagg: [<m>]}`` names, else the first; ``accuracy.disagg`` gives its
  environment, its timeout and the native tests that also judge it.

Nothing accumulates from one deployment to the next. What the shorthand implies (the
switches' defaults, the token and batch limits, an attention-DP rank's share of a global
batch, a drafter's speculative config) is written into the options here, once. What this
table cannot state is refused in one line, ``state deployments: <why>``.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .deployments import (
    BENCHMARK_DEFAULTS,
    DEFAULT_BENCHMARK,
    INPUTS,
    Deployment,
    Server,
    benchmark_of,
    load_options,
    validate_env,
    validate_judging,
)
from .task_schema import ACCURACY_FIELD, reject_unknown_keys

MEMBER = re.compile(r"^(tp|tep|dep)(\d+)$")
UNIFIED = "unified"
DISAGG = "disagg"
#: A drafter feature's ``speculative_config.decoding_type``.
DRAFTERS = {"mtp": "MTP", "eagle3": "Eagle3", "dflash": "DFlash", "dspark": "DSpark"}
_OTHERS = ("multimodal", "host_offload", "kv_quant", "decoder_replay", "encoder_replay", DISAGG)
FEATURES: tuple[str, ...] = (*DRAFTERS, *_OTHERS)
#: The keys that state a run in shorthand; a spec that states ``deployments`` has none.
SHORTHAND_KEYS: tuple[str, ...] = ("members", "features", "variants", "runtime")
#: The ``accuracy`` keys only shorthand reads.
ACCURACY_KEYS = ("llm_api_options", "benchmark_args", "max_seq_len", "max_batch_size", DISAGG)
VARIANT_KEYS = ("unit", "llm_api_options", "runtime", "env", "accuracy", "note")
DISAGG_KEYS = ("test", "tests", "env", "timeout")
#: A runtime switch: the option it sets, whether "on" writes true, and what it is unsaid.
SWITCHES: dict[str, tuple[str, bool, bool]] = {
    "cuda_graph": ("cuda_graph_config", True, True),
    "chunked_prefill": ("enable_chunked_prefill", True, False),
    "overlap_scheduler": ("disable_overlap_scheduler", False, True),
    "kv_cache_reuse": ("kv_cache_config.enable_block_reuse", True, False),
}
#: ``runtime.extra_flags``: ``--<flag>=<value>`` and the LLM-API option it sets.
FLAGS: dict[str, str] = {
    "max_seq_len": "max_seq_len",
    "max_batch_size": "max_batch_size",
    "max_num_tokens": "max_num_tokens",
    "max_beam_width": "max_beam_width",
    "trust_remote_code": "trust_remote_code",
    "kv_cache_fraction": "kv_cache_config.free_gpu_memory_fraction",
    "kv_cache_dtype": "kv_cache_config.dtype",
    "tokens_per_block": "kv_cache_config.tokens_per_block",
    "mamba_ssm_cache_dtype": "kv_cache_config.mamba_ssm_cache_dtype",
    "attention_backend": "attn_backend",
    "moe_backend": "moe_config.backend",
    "spec_decode_max_draft_len": "speculative_config.max_draft_len",
    "draft_model_dir": "speculative_config.speculative_model",
}
SAMPLING: tuple[str, ...] = ("temperature", "top_k", "top_p")
_STATE = "state deployments: "


def _get(options: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    for key in path.split("."):
        if not isinstance(options, Mapping) or key not in options:
            return False, None
        options = options[key]
    return True, options


def _put(options: dict[str, Any], path: str, value: Any) -> None:
    *parents, last = path.split(".")
    for key in parents:
        options = options.setdefault(key, {})
    options[last] = value


def parse_flags(listed: Any) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """``runtime.extra_flags`` as ``{option path: value}``, the sampling keys, and chat rendering."""
    words = [word for flag in listed or () for word in shlex.split(str(flag))]
    options: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    chat = False
    while words:
        name, stated, value = words.pop(0).partition("=")
        if not stated and words and not words[0].startswith("--"):
            stated, value = "=", words.pop(0)
        name, parsed = name.removeprefix("--"), yaml.safe_load(value) if stated else True
        if name == "apply_chat_template":
            chat = True
        elif name in SAMPLING:
            sampling[name] = parsed
        elif name in FLAGS:
            options[FLAGS[name]] = parsed
        else:
            raise ValueError(
                f"{_STATE}runtime.extra_flags has --{name}, which no option of the shorthand "
                f"states; write it in an options file"
            )
    return options, sampling, chat


def merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` over ``base``, mapping into mapping, so a nested block keeps its other keys."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def per(accuracy: Mapping[str, Any], key: str, *names: str) -> Any:
    """``accuracy[key]``: its entry for the first of ``names`` it has, else ``default``; or itself.

    A value is per deployment when it is a mapping (for ``discriminators``, a mapping of
    mappings): ``{default: ..., mtp: ...}``.
    """
    value = accuracy.get(key)
    nested = isinstance(value, Mapping) and (
        key != "discriminators" or any(isinstance(v, Mapping) for v in value.values())
    )
    if not nested:
        return value
    return next((value[n] for n in names if value.get(n) is not None), value.get("default"))


def remote_code(checkpoint: Any) -> bool:
    """Whether the checkpoint ships its own code (an ``auto_map``): every launch must trust it."""
    for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        try:
            data = json.loads((Path(str(checkpoint)) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping) and data.get("auto_map"):
            return True
    return False


def topology(member: str) -> dict[str, Any]:
    """A member's name as the engine's own options: ``tep4`` is 4 ranks with the experts split."""
    kind, size = MEMBER.match(member).groups()
    return {
        "tensor_parallel_size": int(size),
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": int(size) if kind in ("tep", "dep") else 1,
        "enable_attention_dp": kind == "dep",
    }


def share_batch(options: dict[str, Any]) -> None:
    """Turn the batch sizes, global totals, into each attention-DP rank's share, in place."""
    ranks = int(options.get("tensor_parallel_size") or 1)
    if not options.get("enable_attention_dp") or ranks <= 1:
        return

    def share(value: Any) -> int:
        return max(1, -(-int(value) // ranks))

    options["max_batch_size"] = limit = share(options["max_batch_size"])
    graphs = options.get("cuda_graph_config")
    if isinstance(graphs, dict):
        if graphs.get("max_batch_size"):
            graphs["max_batch_size"] = limit = min(limit, share(graphs["max_batch_size"]))
        if isinstance(graphs.get("batch_sizes"), list):
            kept = {int(b) for b in graphs["batch_sizes"] if int(b) <= limit} | {limit}
            graphs["batch_sizes"] = sorted(kept)


def _server(
    data: Mapping[str, Any], unit: str, member: str, variant: Mapping[str, Any] | None = None
) -> tuple[Server, dict[str, Any], bool]:
    """The one server ``unit`` (core or a feature) runs at ``member``, a variant's overlay over it."""
    accuracy = data.get(ACCURACY_FIELD) or {}
    base, own = data.get("runtime") or {}, (variant or {}).get("runtime") or {}
    runtime = {**base, **own}
    stated, sampling, chat = parse_flags(
        [*(base.get("extra_flags") or ()), *(own.get("extra_flags") or ())]
    )
    source = per(accuracy, "llm_api_options", unit)
    options = merge(
        load_options(source) if source else {}, (variant or {}).get("llm_api_options") or {}
    )
    for path, value in stated.items():
        found, there = _get(options, path)
        if found and there != value:
            raise ValueError(
                f"{unit}: two values for `{path}`: runtime.extra_flags asks {value!r} and the "
                f"options ask {there!r}; state it once"
            )
    for key in ("max_seq_len", "max_batch_size"):
        # The limits the benchmark and the server were given are the deployment's.
        if accuracy.get(key) is not None:
            stated[key] = accuracy[key]
    for switch, (path, positive, default) in SWITCHES.items():
        on = bool(runtime.get(switch, default))
        if switch == "cuda_graph":
            if on:
                options.setdefault(path, {})
            else:
                options[path] = None
        elif switch in runtime or not _get(options, path)[0]:
            _put(options, path, on if positive else not on)
    for path, value in stated.items():
        _put(options, path, value)
    options.setdefault("max_num_tokens", 8192)
    options.setdefault("max_batch_size", 2048)
    options["kv_cache_config"].setdefault("free_gpu_memory_fraction", 0.9)
    if remote_code(data.get("checkpoint_path")):
        options.setdefault("trust_remote_code", True)
    options.update(topology(member))
    share_batch(options)
    drafter = DRAFTERS.get(unit)
    draft = dict(options.get("speculative_config") or {}) if drafter else None
    if drafter:
        if str(draft.setdefault("decoding_type", drafter)).lower() != drafter.lower():
            raise ValueError(f"{unit}: its options state decoding_type {draft['decoding_type']}")
        draft.setdefault("max_draft_len", 1)
        if drafter == "MTP":
            draft.setdefault("speculative_model", str(data.get("checkpoint_path")))
    options["speculative_config"] = draft
    return Server(options=options), sampling, chat


def members_of(data: Mapping[str, Any]) -> tuple[list[str], str]:
    """The members listed (the first is core's), and the member ``disagg`` is served at."""
    parallel = str(data.get("parallel") or UNIFIED)
    block = data.get("members")
    listed = block.get("default") if isinstance(block, Mapping) else block
    members = [str(m) for m in listed or ()] or ([parallel] if parallel != UNIFIED else [])
    named = (block.get(DISAGG) or [None])[0] if isinstance(block, Mapping) else None
    return members, str(named or (members[0] if members else ""))


def validate(data: Mapping[str, Any], errors: list[str]) -> None:
    """Every problem with a spec's shorthand keys, batched into ``errors``."""
    parallel, block = data.get("parallel"), data.get("members")
    if parallel is not None and not (
        isinstance(parallel, str) and (parallel == UNIFIED or MEMBER.match(parallel))
    ):
        errors.append(f"'parallel' must be tp<N>, tep<N>, dep<N> or '{UNIFIED}'; got {parallel!r}")
        return
    extra = sorted(set(block) - {"default", DISAGG}) if isinstance(block, Mapping) else []
    if extra:
        errors.append(
            f"{_STATE}'members' lists members for {extra}; in shorthand a feature or a variant "
            f"is one deployment on the first member"
        )
    members, disagg_member = members_of(data)
    if block is not None and parallel not in (None, UNIFIED) and members != [parallel]:
        errors.append(f"'members' beside 'parallel: {parallel}': list the members, or name one")
    bad = [m for m in (*members, disagg_member) if not MEMBER.match(m)]
    if not members or bad:
        errors.append(
            "missing members: what to serve at; fix: `parallel: tp1`, or `members: [tp1, tep4]`"
            + (f" (not tp<N>, tep<N> or dep<N>: {bad})" if bad else "")
        )
        return
    features = data.get("features") or []
    if not isinstance(features, list) or any(
        not isinstance(f, str) or f not in FEATURES for f in features
    ):
        errors.append(f"'features' must be a list drawn from {list(FEATURES)}; got {features!r}")
        return
    if {"mtp", "dspark"} <= set(features):
        errors.append("'features' lists mtp and dspark: the checkpoint's `mtp.*` weights are one")
    runtime = data.get("runtime") or {}
    if not isinstance(runtime, Mapping) or any(isinstance(v, Mapping) for v in runtime.values()):
        errors.append(f"{_STATE}'runtime' differs by unit; a deployment states its own options")
        return
    reject_unknown_keys(runtime, (*SWITCHES, "extra_flags"), errors, where="runtime")
    accuracy = data.get(ACCURACY_FIELD) or {}
    disagg = accuracy.get(DISAGG) or {}
    if not isinstance(disagg, Mapping):
        errors.append(f"'{ACCURACY_FIELD}.{DISAGG}' must be a mapping")
        return
    reject_unknown_keys(disagg, DISAGG_KEYS, errors, where=f"{ACCURACY_FIELD}.{DISAGG}")
    validate_env(disagg.get("env"), errors, f"{ACCURACY_FIELD}.{DISAGG}.env")
    variants = data.get("variants") or {}
    if not isinstance(variants, Mapping):
        errors.append("'variants' must be a mapping from a variant's name to its entry")
        return
    for name, entry in variants.items():
        where = f"variants.{name}"
        if not re.match(r"^[a-z][a-z0-9_]{0,40}$", str(name)) or name in (
            *members,
            *FEATURES,
            "core",
        ):
            errors.append(f"'{where}': a variant needs a name of its own, lowercase with '_'")
        if not isinstance(entry, Mapping):
            errors.append(f"'{where}' must be a mapping")
            continue
        reject_unknown_keys(entry, VARIANT_KEYS, errors, where=where)
        unit = entry.get("unit") or "core"
        if unit != "core" and (unit not in features or unit == DISAGG):
            errors.append(f"'{where}.unit' is {unit!r}, not core or a feature of this spec")
        if not (entry.get("llm_api_options") or entry.get("runtime") or entry.get("env")):
            errors.append(
                f"'{where}' changes nothing: give it `llm_api_options`, `runtime` or `env`"
            )
        validate_env(entry.get("env"), errors, f"{where}.env")
    if not errors:
        try:
            for deployment in expand(data):
                for benchmark in deployment.benchmarks:
                    validate_judging(benchmark.as_dict(), errors, ACCURACY_FIELD)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))


def expand(data: Mapping[str, Any]) -> tuple[Deployment, ...]:
    """The deployments a spec's shorthand keys ask for, core's first (the module's table)."""
    accuracy = data.get(ACCURACY_FIELD)
    members, disagg_member = members_of(data)
    first = members[0]

    def benchmarks(*names: str, own: Mapping[str, Any] | None = None):
        if not isinstance(accuracy, Mapping):
            return ()
        block = {**accuracy, **{k: v for k, v in (own or {}).items() if v is not None}}
        if isinstance((own or {}).get("discriminators"), Mapping):
            # A variant's row is its unit's, with the tags it adds.
            unit_tags = per(accuracy, "discriminators", *names[1:]) or {}
            block["discriminators"] = {**unit_tags, **own["discriminators"]}
        entry = {
            "name": per(block, "benchmark", *names) or DEFAULT_BENCHMARK,
            "args": per(block, "benchmark_args", *names),
        }
        return (benchmark_of(entry, {k: per(block, k, *names) for k in BENCHMARK_DEFAULTS}),)

    def deployment(name: str, unit: str, member: str, variant=None, **more: Any) -> Deployment:
        server, sampling, chat = _server(data, unit, member, variant)
        sides = {"ctx": server, "gen": server} if name == DISAGG else {"server": server}
        related = first if name != first else None
        return Deployment(
            name=name, relative_to=related, sampling=sampling, chat_template=chat, **sides, **more
        )

    out = [deployment(m, "core", m, benchmarks=benchmarks("core")) for m in members]
    out[0] = replace(out[0], core=True)
    for feature in data.get("features") or ():
        if feature == DISAGG:
            block = (accuracy or {}).get(DISAGG) or {}
            more = dict(
                env={str(k): str(v) for k, v in (block.get("env") or {}).items()},
                timeout=block.get("timeout"),
                tests=tuple(block.get("tests") or filter(None, [block.get("test")])),
            )
            out.append(
                deployment(
                    DISAGG, "core", disagg_member, benchmarks=benchmarks(DISAGG, "core"), **more
                )
            )
            continue
        images = feature == "multimodal" and data.get("scope") != "text_only"
        inputs = INPUTS if images else ("text",)
        out.append(
            deployment(feature, feature, first, inputs=inputs, benchmarks=benchmarks(feature))
        )
    for name, entry in (data.get("variants") or {}).items():
        unit = str(entry.get("unit") or "core")
        env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
        judged = benchmarks(str(name), unit, own=entry.get("accuracy"))
        out.append(deployment(str(name), unit, first, entry, env=env, benchmarks=judged))
    return tuple(out)


__all__ = [
    "ACCURACY_KEYS",
    "DISAGG",
    "DRAFTERS",
    "FEATURES",
    "MEMBER",
    "SHORTHAND_KEYS",
    "SWITCHES",
    "expand",
    "members_of",
    "merge",
    "parse_flags",
    "per",
    "remote_code",
    "share_batch",
    "topology",
    "validate",
]
