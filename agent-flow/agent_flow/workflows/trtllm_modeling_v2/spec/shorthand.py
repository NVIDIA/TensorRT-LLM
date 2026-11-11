# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec's shorthand keys, expanded into deployments with native options.

``parallel``, ``members``, ``features``, ``variants``, ``runtime`` and
``accuracy.llm_api_options`` state a run as units, members and variants. This
module is the one place they are turned into what TensorRT-LLM takes: each unit
and each variant at a member is a :class:`~.deployments.Deployment` whose
options are the unit's options file, the runtime switches and flags, the
member's topology and the unit's speculator (:func:`engine_options`). The
defaults the shorthand implies (block reuse and chunked prefill off, CUDA graphs
as the switch says, the batch and token limits, an attention-DP rank's share of
a global batch) are written into those options here, once; a deployment a spec
states itself is taken as written and gets none of them.

:func:`deployment_for` answers for any unit or variant at any member, which is
how a check named ``<unit>:<gate>@<member>`` finds what it runs; :func:`expand`
lists the deployments a shorthand spec asks for, core's first.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from .deployments import INPUTS, Benchmark, Deployment, Server, name_for
from .task_schema import (
    ACCURACY_FIELD,
    DISAGG_FEATURE,
    SCOPE_FIELD,
    TaskSchemaError,
    assembly_units,
    features_through,
    members_for,
    parse_parallel_segment,
    resolve_per_unit,
    widest_world_size,
)
from .variants import (
    ANCHOR_RELATIVE,
    _deep_merge,
    anchor_state,
    base_unit,
    is_variant,
    variant_gates,
    variants,
)

DEFAULT_BENCHMARK = "gsm8k"
#: The world size of a shorthand spec that names no member with one.
DEFAULT_WORLD_SIZE = 4


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(
            f"runtime.extra_flags: {message}; use public LLM keys in accuracy.llm_api_options"
        )


# Only the workflow's input spelling differs from the public constructor.
_FLAGS = {
    "kv_cache_fraction": ("kv_cache_config.free_gpu_memory_fraction", float),
    "kv_cache_dtype": ("kv_cache_config.dtype", str),
    "tokens_per_block": ("kv_cache_config.tokens_per_block", int),
    "mamba_ssm_cache_dtype": ("kv_cache_config.mamba_ssm_cache_dtype", str),
    "mamba_ssm_philox_rounds": ("kv_cache_config.mamba_ssm_philox_rounds", int),
    "max_seq_len": ("max_seq_len", int),
    "max_batch_size": ("max_batch_size", int),
    "max_num_tokens": ("max_num_tokens", int),
    "max_beam_width": ("max_beam_width", int),
    "attention_backend": ("attn_backend", str),
    "moe_backend": ("moe_config.backend", str),
    "load_format": ("load_format", str),
    "checkpoint_format": ("checkpoint_format", str),
    "custom_tokenizer": ("custom_tokenizer", str),
    "orchestrator_type": ("orchestrator_type", str),
    "spec_decode_algo": ("speculative_config.decoding_type", str),
    "spec_decode_max_draft_len": ("speculative_config.max_draft_len", int),
    "draft_model_dir": ("speculative_config.speculative_model", str),
}


_SWITCHES = {
    "trust_remote_code": ("trust_remote_code", True),
    "disable_overlap_scheduler": ("disable_overlap_scheduler", True),
    "enable_chunked_prefill": ("enable_chunked_prefill", True),
    "disable_kv_cache_reuse": ("kv_cache_config.enable_block_reuse", False),
    "mamba_ssm_stochastic_rounding": ("kv_cache_config.mamba_ssm_stochastic_rounding", True),
}


_SAMPLING = {"temperature": float, "top_k": int, "top_p": float, "n": int, "best_of": int}


def _put(options: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        options = options.setdefault(key, {})
    options[keys[-1]] = value


def parse_runtime_flags(flags: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Public engine keys, sampling keys and chat rendering from workflow runtime flags."""
    parser = _Parser(add_help=False, argument_default=argparse.SUPPRESS, allow_abbrev=False)
    for name, (_, kind) in _FLAGS.items():
        parser.add_argument(f"--{name}", type=kind)
    for name in (*_SWITCHES, "use_cuda_graph", "cuda_graph_padding_enabled", "apply_chat_template"):
        parser.add_argument(f"--{name}", action="store_true")
    for name, kind in _SAMPLING.items():
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--cuda_graph_batch_sizes", type=int, nargs="+")
    parser.add_argument("--use_kv_cache_manager_v2", choices=("auto", "true", "false"))
    values = vars(parser.parse_args([w for flag in flags for w in shlex.split(str(flag))]))
    options: dict[str, Any] = {}
    sampling = {name: values.pop(name) for name in _SAMPLING if name in values}
    chat = values.pop("apply_chat_template", False)
    for name, (path, _) in _FLAGS.items():
        if name in values:
            _put(options, path, values.pop(name))
    for name, (path, value) in _SWITCHES.items():
        if values.pop(name, False):
            _put(options, path, value)
    manager = values.pop("use_kv_cache_manager_v2", None)
    if manager is not None:
        _put(
            options,
            "kv_cache_config.use_kv_cache_manager_v2",
            manager if manager == "auto" else manager == "true",
        )
    graphs = values.pop("use_cuda_graph", False)
    padding = values.pop("cuda_graph_padding_enabled", None)
    batches = values.pop("cuda_graph_batch_sizes", None)
    if graphs or padding is not None or batches is not None:
        options["cuda_graph_config"] = {}
        if padding is not None:
            options["cuda_graph_config"]["enable_padding"] = padding
        if batches is not None:
            options["cuda_graph_config"]["batch_sizes"] = batches
    return options, sampling, chat


def share_batch(options: dict[str, Any], member: str | None, ranks: int | None = None) -> None:
    """Turn ``options``' batch sizes, global totals, into each attention-DP rank's share.

    A run states a batch size (and a concurrency) as the global total across the
    ranks. Under attention DP every rank schedules its own batch, so the engine's
    ``max_batch_size`` and its CUDA graphs' are the total over the ranks, rounded up;
    the captured batch sizes stop at that share.
    """
    if not member:
        return
    kind, size = parse_parallel_segment(member)
    ranks = ranks or size or 1
    if kind != "dep" or ranks <= 1:
        return

    def share(value: Any) -> int:
        return max(1, -(-int(value) // ranks))

    if options.get("max_batch_size"):
        options["max_batch_size"] = share(options["max_batch_size"])
    graphs = options.get("cuda_graph_config")
    if not isinstance(graphs, dict):
        return
    limit = share(graphs["max_batch_size"]) if graphs.get("max_batch_size") else None
    if options.get("max_batch_size"):
        limit = min(limit or options["max_batch_size"], options["max_batch_size"])
    if limit is None:
        return
    if graphs.get("max_batch_size"):
        graphs["max_batch_size"] = limit
    if isinstance(graphs.get("batch_sizes"), list):
        kept = sorted({int(b) for b in graphs["batch_sizes"] if int(b) <= limit} | {limit})
        graphs["batch_sizes"] = kept


#: Startup cost of a member's topology, cheapest first: ``tep`` partitions the
#: experts and slices attention, ``dep`` replicates attention on every rank,
#: ``tp`` holds a slice of every expert on every rank.
_MEMBER_COST_RANK: dict[str, int] = {"tep": 0, "dep": 1, "tp": 2}


# Unknown cost (``unified`` included) is not cheap cost: it sorts last.
_UNKNOWN_MEMBER_COST = len(_MEMBER_COST_RANK)


def order_members_by_cost(members: Sequence[str]) -> list[str]:
    """Sort members by cost, then world size and name; unknown members sort last."""

    def key(member: str) -> tuple[int, int, str]:
        try:
            kind, size = parse_parallel_segment(member)
        except TaskSchemaError:
            return (_UNKNOWN_MEMBER_COST, 0, member)
        return (
            _MEMBER_COST_RANK.get(kind, _UNKNOWN_MEMBER_COST),
            size if size is not None else 0,
            member,
        )

    return sorted(members, key=key)


def resolve_runtime(task_data: Mapping[str, Any], unit: str | None) -> Mapping[str, Any]:
    """The ``runtime`` block for ``unit``, with the per-unit form merged.

    Accepts the flat form (the whole run) or a mapping from unit name to a
    partial block layered key by key over ``default``. ``extra_flags`` is an
    ordinary key and therefore REPLACES rather than extends.
    """
    block = task_data.get("runtime") or {}
    if not any(isinstance(v, dict) for v in block.values()):
        return block
    merged = dict(block.get("default") or {})
    chosen = block.get(unit)
    if isinstance(chosen, dict):
        merged.update(chosen)
    return merged


#: The checkpoint files whose ``auto_map`` says the model's code ships with its weights.
REMOTE_CODE_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def checkpoint_remote_code(checkpoint: str | Path | None) -> bool:
    """Whether the checkpoint ships its own code (an ``auto_map``): every launcher must trust it."""
    if not checkpoint:
        return False
    for name in REMOTE_CODE_FILES:
        path = Path(str(checkpoint)) / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping) and data.get("auto_map"):
            return True
    return False


def launch_conflicts(
    task_data: Mapping[str, Any], unit: str | None, overlay: Mapping[str, Any] | None = None
) -> list[str]:
    """Refuse conflicting input spellings before a check constructs its public options.

    ``overlay`` is merged over the unit's options file first, as a variant's options are.
    """
    flags, _, _ = parse_runtime_flags(resolve_runtime(task_data, unit).get("extra_flags") or ())
    options = merge_options(_unit_options(task_data, unit), overlay or {})

    def differences(left: Mapping[str, Any], right: Mapping[str, Any], path: str = "") -> list[str]:
        errors = []
        for key in left.keys() & right.keys():
            name = f"{path}.{key}" if path else key
            if isinstance(left[key], Mapping) and isinstance(right[key], Mapping):
                errors.extend(differences(left[key], right[key], name))
            elif left[key] != right[key]:
                errors.append(
                    f"{name}: runtime.extra_flags asks {left[key]!r}, options file asks {right[key]!r}"
                )
        return errors

    return differences(flags, options)


def _unit_options(task_data: Mapping[str, Any], unit: str | None) -> dict[str, Any]:
    return load_llm_api_options(resolve_llm_api_options(task_data.get("accuracy") or {}, unit))


def merge_options(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` over ``base``, mapping into mapping, so a nested block keeps its other keys."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_options(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_max_draft_len(accuracy: Mapping[str, Any], unit: str | None) -> int:
    """How many draft tokens this unit's gates must ask for, from its options YAML; 1 without one.

    All checks use the same ``speculative_config``: a shallower speculator
    accepts a larger fraction of what it drafts. Read in the order
    the engine resolves it: ``max_draft_len``, its deprecated MTP alias
    ``num_nextn_predict_layers``, and a DSpark drafter's ``block_size``.
    """
    options = resolve_llm_api_options(accuracy, unit)
    if not options:
        return 1
    source = Path(options)
    if not source.is_file():
        return 1
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    spec = data.get("speculative_config") or {}
    for key in ("max_draft_len", "num_nextn_predict_layers", "block_size"):
        try:
            return max(1, int(spec[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return 1


def resolve_llm_api_options(accuracy: Mapping[str, Any], unit: str | None) -> str | None:
    """Pick the runtime-configuration YAML for ``unit``.

    ``llm_api_options`` may be a single path (the whole run) or a mapping
    from unit name to path, with ``default`` as the fallback key.
    """
    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        return options.get(unit) or options.get("default")
    return options


def load_llm_api_options(path: str | Path | None) -> dict[str, Any]:
    """The LLM-API options file at ``path`` as a mapping; empty when there is none."""
    if not path or not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def engine_options(
    task_data: Mapping[str, Any],
    *,
    unit: str | None,
    member: str | None,
    features: Any | None = None,
    world_size: int | None = None,
    overlay: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """The public LLM-API options ``unit`` runs under at ``member``, from the spec's shorthand.

    The runtime switches and flags, the unit's options file (``overlay`` merged over it, as
    a variant's options are), the member's topology and the unit's speculator become one
    options mapping; the sampling keys and the chat-template switch the flags state come
    beside it. ``world_size`` is the member's when its segment names none.
    """
    runtime = resolve_runtime(task_data, unit)
    extra, sampling, chat = parse_runtime_flags(runtime.get("extra_flags") or ())
    conflicts = launch_conflicts(task_data, unit, overlay)
    if conflicts:
        raise ValueError(f"{unit}: conflicting check inputs: " + "; ".join(conflicts))
    options = merge_options(merge_options(_unit_options(task_data, unit), overlay or {}), extra)
    for switch, key, default in (
        ("chunked_prefill", "enable_chunked_prefill", False),
        ("overlap_scheduler", "disable_overlap_scheduler", True),
    ):
        value = bool(runtime.get(switch, default))
        value = not value if switch == "overlap_scheduler" else value
        if switch in runtime:
            options[key] = value
        else:
            options.setdefault(key, value)
    kv = options.setdefault("kv_cache_config", {})
    if "kv_cache_reuse" in runtime:
        kv["enable_block_reuse"] = bool(runtime["kv_cache_reuse"])
    else:
        # Block reuse is a non-core feature: the core configuration keeps it off.
        kv.setdefault("enable_block_reuse", False)
    if runtime.get("cuda_graph") is False:
        options["cuda_graph_config"] = None
    elif "cuda_graph_config" not in options:
        options["cuda_graph_config"] = {} if runtime.get("cuda_graph") else None
    # Explicit extra flags are the final runtime spelling, validated at preflight.
    options = merge_options(options, extra)
    options.setdefault("max_num_tokens", 8192)
    options.setdefault("max_batch_size", 2048)
    options.setdefault("kv_cache_config", {}).setdefault("free_gpu_memory_fraction", 0.9)
    if checkpoint_remote_code(task_data.get("checkpoint_path")):
        options.setdefault("trust_remote_code", True)
    kind, size = parse_parallel_segment(member or str(task_data["parallel"]))
    size = size or world_size or 1
    options.update(
        model=str(task_data["checkpoint_path"]),
        tensor_parallel_size=size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=size if kind in ("tep", "dep") else 1,
        enable_attention_dp=kind == "dep",
    )
    share_batch(options, member or str(task_data["parallel"]), size)
    if features is None:
        actual_unit = base_unit(task_data, unit) if is_variant(task_data, unit) else unit
        features = (
            features_through(task_data, actual_unit) if actual_unit else task_data.get("features")
        )
    names = {str(f).lower() for f in features or ()}
    algorithm = next(
        (a for f, a in (("mtp", "MTP"), ("eagle3", "Eagle3"), ("dspark", "DSpark")) if f in names),
        None,
    )
    if algorithm:
        speculative = options.setdefault("speculative_config", {})
        if speculative is None:
            raise ValueError(f"{unit}: {algorithm} feature has speculative_config: null")
        stated = speculative.setdefault("decoding_type", algorithm)
        if str(stated).lower() != algorithm.lower():
            raise ValueError(f"{unit}: {algorithm} feature conflicts with decoding_type {stated}")
        speculative["decoding_type"] = algorithm
        speculative.setdefault(
            "max_draft_len", resolve_max_draft_len(task_data.get("accuracy") or {}, unit)
        )
        if algorithm == "MTP":
            speculative.setdefault("speculative_model", str(task_data["checkpoint_path"]))
    else:
        options["speculative_config"] = None
    return options, sampling, chat


def _per_unit(key: str, value: Any, name: str, chosen: Any) -> dict[str, Any]:
    """Accuracy key ``key`` in its per-unit form, with ``name`` set to ``chosen``.

    ``acceptance`` holds the workload and the per-unit floors side by side;
    ``discriminators`` is per-unit when its values are mappings; every other
    key is per-unit exactly when it is a mapping.
    """
    if key == "acceptance":
        return {**dict(value or {}), name: chosen}
    if key == "discriminators":
        per_unit = isinstance(value, Mapping) and any(
            isinstance(v, Mapping) for v in value.values()
        )
    else:
        per_unit = isinstance(value, Mapping)
    out = dict(value) if per_unit else ({} if value is None else {"default": value})
    out[name] = chosen
    return out


def _unit_discriminators(block: Any, unit: str) -> Mapping[str, Any]:
    """``unit``'s discriminators in ``accuracy.discriminators``, flat or per unit; {} for none."""
    if not isinstance(block, Mapping):
        return {}
    if any(isinstance(v, Mapping) for v in block.values()):
        chosen = block.get(unit, block.get("default"))
        return chosen if isinstance(chosen, Mapping) else {}
    return block


def variant_spec(
    data: Mapping[str, Any],
    name: str,
    *,
    world_size: int,
    options_dir: Path | None = None,
) -> dict[str, Any]:
    """The spec a variant's checks run under: the unit's, with the variant's overlay.

    Every per-unit lookup the gates make (``resolve_per_unit``, the runtime
    block, ``members``) then answers for the variant's name what it answers
    for its unit, except where the variant says otherwise. With
    ``options_dir`` the unit's LLM-API YAML merged with the variant's options
    is written there; without it (compiling the contract) the options are not
    resolved.
    """
    entry = variants(data)[name]
    unit = base_unit(data, name)
    out = copy.deepcopy(dict(data))
    accuracy = dict(out.get("accuracy") or {})
    for key, value in list(accuracy.items()):
        if isinstance(value, dict) and unit in value:
            value[name] = copy.deepcopy(value[unit])

    if options_dir is not None:
        base_options = resolve_llm_api_options(accuracy, unit)
        loaded: dict[str, Any] = {}
        if base_options and Path(base_options).is_file():
            loaded = yaml.safe_load(Path(base_options).read_text(encoding="utf-8")) or {}
        merged = _deep_merge(loaded, entry.get("llm_api_options") or {})
        options_dir.mkdir(parents=True, exist_ok=True)
        path = options_dir / f"{name}.llm_api.yaml"
        # Replaced whole: a gate of another variant may be reading it now.
        write_text_atomic(path, yaml.safe_dump(merged, sort_keys=False), mode=0o600)
        accuracy["llm_api_options"] = _per_unit(
            "llm_api_options", accuracy.get("llm_api_options"), name, str(path)
        )
    for key, value in (entry.get("accuracy") or {}).items():
        if key == "discriminators" and isinstance(value, Mapping):
            # Over the unit's row: a KV-dtype variant adds `kv_cache_quant_algo` to the
            # unit's `quant_algo`, it does not drop it.
            value = {**dict(_unit_discriminators(accuracy.get(key), unit)), **value}
        accuracy[key] = _per_unit(key, accuracy.get(key), name, copy.deepcopy(value))
    out["accuracy"] = accuracy

    runtime = dict(resolve_runtime(data, unit))
    overlay = dict(entry.get("runtime") or {})
    flags = [*runtime.get("extra_flags", ()), *overlay.pop("extra_flags", ())]
    runtime.update(overlay)
    runtime["extra_flags"] = flags
    block = data.get("runtime") or {}
    if any(isinstance(v, dict) for v in block.values()):
        out["runtime"] = {**copy.deepcopy(dict(block)), name: runtime}
    else:
        out["runtime"] = {"default": copy.deepcopy(dict(block)), name: runtime}

    members = entry.get("members") or members_for(data, unit, world_size)
    declared = data.get("members")
    if isinstance(declared, dict):
        out["members"] = {**declared, name: list(members)}
    elif str(data.get("parallel")) == "unified":
        default = list(declared or members_for(data, None, world_size))
        out["members"] = {"default": default, name: list(members)}

    if entry.get("env"):
        environment = dict(out.get("environment") or {})
        environment["env"] = {**(environment.get("env") or {}), **entry["env"]}
        out["environment"] = environment
    return out


def deployment_for(
    data: Mapping[str, Any], name: str, member: str, world_size: int | None = None
) -> Deployment:
    """The deployment unit or variant ``name`` is at ``member``.

    A unit's is one server on its options. ``disagg`` is a context and a generation
    server, both on the options of the unit before it, judged by the test the spec
    names. A variant's is its unit's options with its own overlay, judged against its
    unit's deployment at the same member unless it names a reference row of its own.
    A unit's benchmark is on it wherever ``accuracy.benchmark_members`` allows; a
    variant's when ``accuracy`` is one of its gates.
    """
    world = world_size or widest_world_size(data, DEFAULT_WORLD_SIZE)
    accuracy = data.get(ACCURACY_FIELD) or {}
    named = accuracy.get("benchmark_members")

    def server(
        spec: Mapping[str, Any], unit: str, overlay: Any = None
    ) -> tuple[Server, dict, bool]:
        options, sampling, chat = engine_options(
            spec, unit=unit, member=member, world_size=world, overlay=overlay
        )
        # The model is the spec's `checkpoint_path`, not an option of a served configuration.
        options.pop("model", None)
        return Server(options=options), sampling, chat

    def benchmarks(spec: Mapping[str, Any], unit: str, at: Sequence[str]) -> tuple[Benchmark, ...]:
        if named in at and member != named:
            return ()
        block = spec.get(ACCURACY_FIELD) or {}
        benchmark = str(resolve_per_unit(block, "benchmark", unit) or DEFAULT_BENCHMARK)
        args = tuple(str(a) for a in resolve_per_unit(block, "benchmark_args", unit) or ())
        return (Benchmark(name=benchmark, args=args),)

    def inputs(unit: str) -> tuple[str, ...]:
        images = "multimodal" in features_through(data, unit)
        return INPUTS if images and data.get(SCOPE_FIELD) != "text_only" else ("text",)

    if is_variant(data, name):
        entry = variants(data)[name]
        unit = base_unit(data, name)
        spec = variant_spec(data, name, world_size=world)
        unit_members = members_for(data, unit, world)
        at = [str(m) for m in entry.get("members") or ()] or [member]
        scored = "accuracy" in variant_gates(data, name)
        relative = scored and anchor_state(data, name) == ANCHOR_RELATIVE
        found, sampling, chat = server(spec, name, entry.get("llm_api_options"))
        return Deployment(
            name=name_for(name, member),
            server=found,
            env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
            relative_to=name_for(unit, member) if relative and member in unit_members else None,
            inputs=inputs(unit),
            benchmarks=benchmarks(spec, name, at) if scored else (),
            sampling=sampling,
            chat_template=chat,
        )
    units = assembly_units(data)
    if name == DISAGG_FEATURE:
        block = accuracy.get("disagg") or {}
        side, sampling, chat = server(data, units[units.index(name) - 1])
        tests = block.get("tests") or ([block["test"]] if block.get("test") else [])
        return Deployment(
            name=name_for(name, member),
            ctx=side,
            gen=side,
            env={str(k): str(v) for k, v in (block.get("env") or {}).items()},
            timeout=block.get("timeout"),
            tests=tuple(str(test) for test in tests),
            inputs=inputs(name),
            sampling=sampling,
            chat_template=chat,
        )
    found, sampling, chat = server(data, name)
    return Deployment(
        name=name_for(name, member),
        server=found,
        inputs=inputs(name),
        benchmarks=benchmarks(data, name, members_for(data, name, world)),
        sampling=sampling,
        chat_template=chat,
    )


def expand(data: Mapping[str, Any], world_size: int | None = None) -> tuple[Deployment, ...]:
    """The deployments a spec's shorthand keys ask for, core's first.

    Each unit is one deployment per member, in the spec's order, so core at its first
    member comes first. ``disagg`` is one, at the member ``accuracy.benchmark_members``
    names, else its first. A variant is one at each member it names, else one at the
    benchmark member or its unit's first; a variant of ``disagg`` is none. ``profile``
    and ``coverage`` change what a run checks, not what is deployed, and are not read.
    """
    from dataclasses import replace

    world = world_size or widest_world_size(data, DEFAULT_WORLD_SIZE)
    named = (data.get(ACCURACY_FIELD) or {}).get("benchmark_members")

    def one(members: Sequence[str]) -> str:
        return named if named in members else members[0]

    out: list[Deployment] = []
    for unit in assembly_units(data):
        members = members_for(data, unit, world)
        for member in [one(members)] if unit == DISAGG_FEATURE else members:
            out.append(deployment_for(data, unit, member, world))
    for name, entry in variants(data).items():
        unit = base_unit(data, name)
        if unit == DISAGG_FEATURE:
            continue
        at = [str(m) for m in entry.get("members") or ()] or [one(members_for(data, unit, world))]
        out.extend(deployment_for(data, name, member, world) for member in at)
    if out:
        out[0] = replace(out[0], core=True)
    return tuple(out)


__all__ = [
    "REMOTE_CODE_FILES",
    "checkpoint_remote_code",
    "deployment_for",
    "engine_options",
    "expand",
    "launch_conflicts",
    "load_llm_api_options",
    "merge_options",
    "order_members_by_cost",
    "parse_runtime_flags",
    "resolve_llm_api_options",
    "resolve_max_draft_len",
    "resolve_runtime",
    "share_batch",
    "variant_spec",
]
