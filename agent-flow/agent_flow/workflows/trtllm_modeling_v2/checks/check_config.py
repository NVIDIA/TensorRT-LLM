# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve shared check configuration and preview requested engine settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.shorthand import (
    engine_options,
    load_llm_api_options,
    merge_options,
    parse_runtime_flags,
    resolve_llm_api_options,
    resolve_runtime,
    variant_spec,
)
from ..spec.task_schema import assembly_units
from ..spec.variants import _deep_merge, base_unit, is_variant, variants


def declared_engine(options: Mapping[str, Any]) -> dict[str, Any]:
    """Requested fields, never evidence that a worker actually applied them."""
    declared = {}
    names = {
        "modeling_v2": "modeling_v2",
        "max_seq_len": "max_seq_len",
        "max_batch_size": "max_batch_size",
        "max_num_tokens": "max_num_tokens",
        "enable_chunked_prefill": "chunked_prefill",
        "tensor_parallel_size": "tp",
        "pipeline_parallel_size": "pp",
        "moe_expert_parallel_size": "ep",
        "enable_attention_dp": "attention_dp",
    }
    for key, field in names.items():
        if key in options:
            declared[field] = options[key]
    if "disable_overlap_scheduler" in options:
        declared["overlap_scheduler"] = not options["disable_overlap_scheduler"]
    kv = options.get("kv_cache_config") or {}
    for key, field in (
        ("enable_block_reuse", "block_reuse"),
        ("dtype", "kv_cache_dtype"),
        ("mamba_ssm_cache_dtype", "mamba_ssm_cache_dtype"),
    ):
        if key in kv:
            declared[field] = kv[key]
    if "mamba_state_config" in kv:
        state = kv["mamba_state_config"] or {}
        declared["mamba_snapshot"] = {
            "periodic_snapshot_interval": state.get("periodic_snapshot_interval", 0),
            "additional_snapshot_offsets_from_start": state.get(
                "additional_snapshot_offsets_from_start", []
            ),
            "additional_snapshot_offsets_from_end": state.get(
                "additional_snapshot_offsets_from_end", []
            ),
        }
    if "cuda_graph_config" in options:
        graph = options["cuda_graph_config"]
        declared["cuda_graph"] = (
            None
            if graph is None
            else {
                **(
                    {"batch_sizes": sorted(graph["batch_sizes"])}
                    if graph.get("batch_sizes")
                    else {}
                ),
                **(
                    {"max_batch_size": graph["max_batch_size"]}
                    if graph.get("max_batch_size")
                    else {}
                ),
                "enable_padding": graph.get("enable_padding", False),
            }
        )
    if "speculative_config" in options:
        spec = options["speculative_config"]
        declared["speculative"] = (
            None
            if spec is None
            else {
                "algorithm": spec["decoding_type"],
                "max_draft_len": spec.get("max_draft_len", spec.get("num_nextn_predict_layers", 1)),
            }
        )
    return declared


#: The checks' explicit scheduler-token default.
DEFAULT_MAX_NUM_TOKENS = 8192


@dataclass(frozen=True)
class EngineLimits:
    """What one request may be in the unit's engine, read from its public constructor configuration."""

    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS
    max_seq_len: int | None = None
    chunked_prefill: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EngineLimits":
        """The request limits explicitly sent to the public LLM API."""
        options = config["llm"]
        return cls(
            max_num_tokens=int(options["max_num_tokens"]),
            max_seq_len=options.get("max_seq_len"),
            chunked_prefill=bool(options.get("enable_chunked_prefill", False)),
        )

    def refuses(self, prompt_tokens: int, new_tokens: int) -> str:
        """Why a prompt of ``prompt_tokens`` that generates ``new_tokens`` does not fit; ``""`` when it does."""
        if self.max_seq_len is not None and prompt_tokens + new_tokens > self.max_seq_len:
            return f"{prompt_tokens} + {new_tokens} tokens exceed max_seq_len {self.max_seq_len}"
        if prompt_tokens > self.max_num_tokens and not self.chunked_prefill:
            return (
                f"{prompt_tokens} tokens exceed max_num_tokens {self.max_num_tokens} and "
                f"chunked prefill is off"
            )
        return ""


def speculation_flags(features: Any, *, max_draft_len: int = 1) -> list[str]:
    """Describe a speculative feature using the workflow runtime input spelling.

    ``dspark`` gets its own algorithm and never ``MTP``, even though a
    checkpoint may store its DSpark stages as ``mtp.*``: ``MTP`` over those
    weights drafts garbage that verification rejects, which passes accuracy.
    """
    names = {str(f).lower() for f in (features or ())}
    for feature, algo in (("mtp", "MTP"), ("eagle3", "Eagle3"), ("dspark", "DSPARK")):
        if feature in names:
            return [f"--spec_decode_algo={algo}", f"--spec_decode_max_draft_len={max_draft_len}"]
    return []


# ------------------------------------------------------------ CUDA-graph capture
#
# All checks receive the same public configuration. These CPU-only previews
# support preflight and planning; only worker observations attest actual capture.

#: The capture limit ``CudaGraphConfig`` uses when neither ``batch_sizes`` nor
#: ``max_batch_size`` is set (``BaseCudaGraphConfig.validate_base_cuda_graph_config``).
LLM_API_CAPTURE_MAX_BATCH_SIZE = 128


def llm_api_batch_sizes(
    batch_sizes: Sequence[int] | None = None,
    max_batch_size: int = 0,
    enable_padding: bool = False,
) -> tuple[int, ...]:
    """The decode batch sizes ``CudaGraphConfig(...)`` resolves to.

    The LLM API's own rule, restated so the orchestrator never imports
    TensorRT-LLM: ``BaseCudaGraphConfig.validate_base_cuda_graph_config`` and
    ``BaseCudaGraphConfig._generate_cuda_graph_batch_sizes`` in
    ``tensorrt_llm/llmapi/llm_args.py``. Raises ``ValueError`` where it does:
    on a ``max_batch_size`` beside explicit ``batch_sizes`` that is not their
    maximum. The engine's later filter by each launch's own limits
    (``filter_cuda_graph_batch_sizes``) is not part of this list.
    """
    if batch_sizes:
        sizes = sorted(int(b) for b in batch_sizes)
        if max_batch_size and int(max_batch_size) != sizes[-1]:
            raise ValueError(
                f"cuda_graph_config.max_batch_size {max_batch_size} is not "
                f"max(cuda_graph_config.batch_sizes) {sizes[-1]}; the LLM API refuses the pair"
            )
        return tuple(sizes)
    limit = int(max_batch_size) or LLM_API_CAPTURE_MAX_BATCH_SIZE
    if enable_padding:
        generated = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        while generated[-1] + 64 <= limit:
            generated.append(generated[-1] + 64)
    else:
        generated = list(range(1, 32)) + [32, 64, 128]
        generated += [2**i for i in range(8, math.ceil(math.log(limit, 2)))]
    sizes = sorted(size for size in generated if size <= limit)
    if not sizes or sizes[-1] != limit:
        sizes.append(limit)
    return tuple(sizes)


def format_batches(sizes: Sequence[int]) -> str:
    """``1-32, 64, 128``: a batch list with each run of three or more consecutive sizes joined."""
    parts: list[str] = []
    run: list[int] = []
    for size in [*sizes, None]:
        if run and size == run[-1] + 1:
            run.append(size)
            continue
        if len(run) >= 3:
            parts.append(f"{run[0]}-{run[-1]}")
        else:
            parts.extend(str(s) for s in run)
        run = [size] if size is not None else []
    return ", ".join(parts)


@dataclass(frozen=True)
class CaptureList:
    """The decode CUDA-graph batch sizes one engine captures, and whether it pads to them.

    Two lists are equal when they capture the same sizes with the same
    padding; ``source`` says where one was read and takes no part in that.
    """

    batch_sizes: tuple[int, ...]
    enable_padding: bool
    source: str = dataclass_field(default="", compare=False)

    def describe(self) -> str:
        padded = "padded" if self.enable_padding else "unpadded"
        return f"{len(self.batch_sizes)} batches [{format_batches(self.batch_sizes)}], {padded}"


#: What a user who sets no ``cuda_graph_config`` gets from the LLM API.
LLM_API_DEFAULT_CAPTURE = CaptureList(
    llm_api_batch_sizes(), False, source="the LLM API default, `CudaGraphConfig()`"
)


def options_capture(
    options: Mapping[str, Any], source: str = "the options file"
) -> CaptureList | None:
    """What an engine built from LLM-API ``options`` captures; ``None`` when it captures nothing.

    Options that do not name ``cuda_graph_config`` get ``CudaGraphConfig()``,
    the field's default in ``TorchLlmArgs``; ``cuda_graph_config: null``
    turns CUDA graphs off. A ``speculative_config.draft_len_schedule`` turns
    padding on and adds its batch thresholds to the list, as ``TorchLlmArgs``'s
    speculative-config validator does (``DecodeCudaGraphConfig._merge_schedule_keys``).
    """
    if "cuda_graph_config" not in options:
        config: Any = {}
        where = f"{LLM_API_DEFAULT_CAPTURE.source}: {source} sets no `cuda_graph_config`"
    else:
        config = options["cuda_graph_config"]
        where = f"`cuda_graph_config` in {source}"
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError(f"`cuda_graph_config` in {source} is not a mapping: {config!r}")
    padding = bool(config.get("enable_padding", False))
    sizes = llm_api_batch_sizes(
        config.get("batch_sizes"), int(config.get("max_batch_size") or 0), padding
    )
    speculative = options.get("speculative_config")
    schedule = speculative.get("draft_len_schedule") if isinstance(speculative, Mapping) else None
    if isinstance(schedule, Mapping) and schedule:
        padding = True
        sizes = tuple(sorted({*sizes, *(int(k) for k in schedule if int(k) <= sizes[-1])}))
        where += " with `speculative_config.draft_len_schedule`"
    return CaptureList(sizes, padding, source=where)


@dataclass(frozen=True)
class LaunchCapture:
    """A planning preview of one shared configuration, not actual capture evidence."""

    name: str
    unit: str
    options: str
    resolved: CaptureList | None


def _launch_capture(
    name: str, unit: str, options: Mapping[str, Any], runtime: Mapping[str, Any], source: str
) -> LaunchCapture:
    extra, _, _ = parse_runtime_flags(runtime.get("extra_flags") or ())
    flag_graph = extra.get("cuda_graph_config") or {}
    file_graph = options.get("cuda_graph_config") or {}
    if flag_graph.get("batch_sizes") and file_graph:
        expected = options_capture(options, source)
        if expected and tuple(sorted(flag_graph["batch_sizes"])) != expected.batch_sizes:
            raise ValueError(
                f"`{name}`: runtime batch sizes {flag_graph['batch_sizes']} conflict "
                f"with {expected.describe()} in {source}; state `cuda_graph_config` "
                "in the unit's options file only"
            )
    merged = merge_options(options, extra)
    if runtime.get("cuda_graph") is False:
        merged["cuda_graph_config"] = None
    elif "cuda_graph_config" not in merged:
        merged["cuda_graph_config"] = {} if runtime.get("cuda_graph") else None
    try:
        resolved = options_capture(merged, source)
    except ValueError as exc:
        raise ValueError(f"`{name}`: {exc}") from exc
    return LaunchCapture(name=name, unit=unit, options=source, resolved=resolved)


def _launch_configurations(
    task_data: Mapping[str, Any],
) -> list[tuple[str, str, dict[str, Any], Mapping[str, Any], str]]:
    """``(name, unit, options, runtime, source)`` for each unit, then each configuration variant.

    A variant's options are its unit's file with the variant's
    ``llm_api_options`` merged in, as :func:`variant_spec` writes
    them, and its runtime block is its unit's with the variant's overlay.
    """
    accuracy = task_data.get("accuracy") or {}
    configurations: list[tuple[str, str, dict[str, Any], Mapping[str, Any], str]] = []
    for unit in assembly_units(task_data):
        path = resolve_llm_api_options(accuracy, unit)
        configurations.append(
            (
                unit,
                unit,
                load_llm_api_options(path),
                resolve_runtime(task_data, unit),
                str(path) if path else "the unit (no options file)",
            )
        )
    for name, entry in variants(task_data).items():
        unit = base_unit(task_data, name)
        path = resolve_llm_api_options(accuracy, unit)
        options = _deep_merge(load_llm_api_options(path), entry.get("llm_api_options") or {})
        # The members it resolves are not read here, so any world size does.
        spec = variant_spec(task_data, name, world_size=1)
        configurations.append(
            (
                name,
                unit,
                options,
                resolve_runtime(spec, name),
                f"{path or 'no options file'} with variants.{name}.llm_api_options",
            )
        )
    return configurations


def launch_captures(task_data: Mapping[str, Any]) -> list[LaunchCapture]:
    """Each unit's capture lists by launch path, then each configuration variant's.

    Raises ``ValueError`` on a ``cuda_graph_config`` the LLM API refuses.
    """
    return [_launch_capture(*configuration) for configuration in _launch_configurations(task_data)]


# ------------------------------------------------------------ KV-cache configuration
#
# Hybrid engines need a snapshot policy to enable recurrent-state reuse.


def has_snapshot_policy(kv_cache: Mapping[str, Any]) -> bool:
    """Whether a ``kv_cache_config`` sets a Mamba state snapshot policy, which block reuse needs.

    The engine's test (``_validate_and_adjust_mamba_snapshot_config``): a
    positive ``mamba_state_config.periodic_snapshot_interval``, or any
    additional snapshot offset. ``mamba_state_cache_interval`` is the
    interval's deprecated alias, and ``block_reuse_config.policy:
    per_conversation`` sets the interval to zero.
    """
    state = kv_cache.get("mamba_state_config")
    state = state if isinstance(state, Mapping) else {}
    interval = state.get("periodic_snapshot_interval", kv_cache.get("mamba_state_cache_interval"))
    reuse = kv_cache.get("block_reuse_config")
    if isinstance(reuse, Mapping) and reuse.get("policy") == "per_conversation":
        interval = 0
    try:
        periodic = int(interval or 0) > 0
    except (TypeError, ValueError):
        periodic = False
    offsets = state.get("additional_snapshot_offsets_from_start") or state.get(
        "additional_snapshot_offsets_from_end"
    )
    return periodic or bool(offsets)


@dataclass(frozen=True)
class LaunchKvCache:
    """The shared requested KV-cache configuration, not evidence of actual reuse."""

    name: str
    unit: str
    options: str
    config: Mapping[str, Any]
    reuse: bool
    snapshots: bool


def launch_kv_caches(task_data: Mapping[str, Any]) -> list[LaunchKvCache]:
    """Each unit's KV-cache configuration, then each configuration variant's."""
    launches = []
    for name, unit, options, runtime, source in _launch_configurations(task_data):
        extra, _, _ = parse_runtime_flags(runtime.get("extra_flags") or ())
        kv = dict(options.get("kv_cache_config") or {})
        if "kv_cache_reuse" in runtime:
            kv["enable_block_reuse"] = bool(runtime["kv_cache_reuse"])
        kv = merge_options(kv, extra.get("kv_cache_config") or {})
        launches.append(
            LaunchKvCache(
                name,
                unit,
                source,
                kv,
                # Block reuse is a non-core feature: off unless asked for.
                bool(kv.get("enable_block_reuse", False)),
                has_snapshot_policy(kv),
            )
        )
    return launches


def engine_declaration(config: Mapping[str, Any], *, mode: str | None) -> dict[str, Any]:
    """The exact public options sent to the engine, separate from actual facts."""
    options = dict(config["llm"])
    if mode:
        options["modeling_v2"] = mode
    return declared_engine(options)


def check_configuration(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    features: Any | None,
    max_tokens: int,
    member: str | None,
) -> dict[str, Any]:
    """One public API configuration for request and eval checks.

    Runtime flags are an input spelling only. The complete options mapping
    (:func:`engine_options`), including nested KV and speculative settings,
    reaches every check intact.
    """
    options, sampling, chat = engine_options(
        task_data, unit=unit, member=member, features=features, world_size=environment.world_size
    )
    sampling.setdefault("temperature", 0.0)
    sampling.setdefault("top_k", 1)
    sampling.setdefault("max_tokens", int(max_tokens))
    width = int(options.get("max_beam_width", 1))
    if width > 1:
        sampling.setdefault("n", width)
        sampling["use_beam_search"] = True
    return {"llm": options, "sampling": sampling, "apply_chat_template": chat}


#: The runtime input flag each engine field is set by, as a variant's runtime block names it.
_FLAG_FIELDS: dict[str, str] = {
    "--kv_cache_dtype": "kv_cache_dtype",
    "--mamba_ssm_cache_dtype": "mamba_ssm_cache_dtype",
    "--max_num_tokens": "max_num_tokens",
    "--enable_chunked_prefill": "chunked_prefill",
    "--max_batch_size": "max_batch_size",
    "--disable_kv_cache_reuse": "block_reuse",
    "--use_cuda_graph": "cuda_graph",
    "--cuda_graph_batch_sizes": "cuda_graph",
    "--cuda_graph_padding_enabled": "cuda_graph",
    "--spec_decode_algo": "speculative",
    "--spec_decode_max_draft_len": "speculative",
}


#: The runtime-block switch each engine field is set by.
_SWITCH_FIELDS: dict[str, str] = {
    "chunked_prefill": "chunked_prefill",
    "kv_cache_reuse": "block_reuse",
    "cuda_graph": "cuda_graph",
    "overlap_scheduler": "overlap_scheduler",
}


#: The LLM-API option each engine field is set by; ``kv_cache_config`` by key.
_OPTION_FIELDS: dict[str, str] = {
    "max_num_tokens": "max_num_tokens",
    "enable_chunked_prefill": "chunked_prefill",
    "max_batch_size": "max_batch_size",
    "speculative_config": "speculative",
    "cuda_graph_config": "cuda_graph",
    "disable_overlap_scheduler": "overlap_scheduler",
}


_KV_FIELDS: dict[str, str] = {
    "dtype": "kv_cache_dtype",
    "mamba_ssm_cache_dtype": "mamba_ssm_cache_dtype",
    "enable_block_reuse": "block_reuse",
    "mamba_state_config": "mamba_snapshot",
}


#: What a variant setting this module cannot place is listed as.
UNKNOWN = "?"


def subject_fields(task_data: Mapping[str, Any], unit: str | None) -> tuple[str, ...]:
    """The engine fields a configuration variant is about: those its overlay sets.

    A setting no field expresses is listed as :data:`UNKNOWN`. A unit that is not a
    variant is about none.
    """
    if not is_variant(task_data, unit):
        return ()
    entry = variants(task_data)[str(unit)]
    found: list[str] = []
    runtime = entry.get("runtime") or {}
    for key in runtime:
        if key != "extra_flags":
            found.append(_SWITCH_FIELDS.get(str(key), UNKNOWN))
    for flag in runtime.get("extra_flags") or ():
        name = str(flag).split("=", 1)[0].split()[0] if str(flag).strip() else ""
        found.append(_FLAG_FIELDS.get(name, UNKNOWN))
    options = entry.get("llm_api_options") or {}
    for key, value in options.items():
        if key == "kv_cache_config" and isinstance(value, Mapping):
            found.extend(_KV_FIELDS.get(str(k), UNKNOWN) for k in value)
        else:
            found.append(_OPTION_FIELDS.get(str(key), UNKNOWN))
    if entry.get("env"):
        found.append(UNKNOWN)
    return tuple(dict.fromkeys(found))


__all__ = [
    "CaptureList",
    "DEFAULT_MAX_NUM_TOKENS",
    "EngineLimits",
    "LLM_API_CAPTURE_MAX_BATCH_SIZE",
    "LLM_API_DEFAULT_CAPTURE",
    "LaunchCapture",
    "LaunchKvCache",
    "UNKNOWN",
    "check_configuration",
    "declared_engine",
    "engine_declaration",
    "format_batches",
    "has_snapshot_policy",
    "launch_captures",
    "launch_kv_caches",
    "llm_api_batch_sizes",
    "options_capture",
    "speculation_flags",
    "subject_fields",
]
