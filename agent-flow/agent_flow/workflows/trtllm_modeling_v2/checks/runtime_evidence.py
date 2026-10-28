# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse actual engine observations and hold them to the requested configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .check_config import format_batches, has_snapshot_policy
from .harness import workload_io

#: How the stock path says *"I have never heard of this architecture"*: the
#: invariant half of each message, matched in the engine's log.
UNSUPPORTED_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    # `AutoModelForCausalLM` / the multimodal encoder (`_torch/models/modeling_auto.py`)
    # and `tensorrt_llm/models/automodel.py`.
    "Unknown architecture for AutoModelForCausalLM",
    "Unknown architecture for AutoModelForMultimodalEncoder",
    "is not supported in TRT-LLM yet",
    # `get_model_architecture`'s registry miss (`_torch/models/modeling_utils.py`).
    "Unknown model architecture:",
    # `transformers`' refusal of a checkpoint that ships its own code (`auto_map`).
    "contains custom code which must be executed to correctly load the model",
    # `transformers`' refusal of an unknown `model_type` with no code to fall back on.
    "but Transformers does not recognize this architecture",
    # A registration shim: a name registered only so the runtime finds something
    # keyed on it, which raises when built without a modeling_v2 target.
    "has no built-in implementation",
)


def reports_unsupported_architecture(detail: str, output: str) -> bool:
    """Whether this text says *"the stock path does not know this model"*."""
    haystack = f"{detail}\n{output}"
    return any(marker in haystack for marker in UNSUPPORTED_ARCHITECTURE_MARKERS)


#: The head of what the engine logs when it turns block reuse off for a hybrid
#: model with no snapshot policy: "Disabling KV cache block reuse for the hybrid
#: Mamba model because no Mamba state snapshot policy is configured. ..." The
#: engine turns reuse off for other reasons too (some MLA configurations), with
#: other messages; this one is the reason an options file can fix.
REUSE_DISABLED_MARKER = "Disabling KV cache block reuse for the hybrid Mamba model"


def reuse_disabled_detail(output: str, *, options: str | None) -> str | None:
    """Why a gate fails when its engine turned block reuse off (``options``: the unit's file).

    ``None`` when the log does not say so. The engine logs it only for an
    engine asked for block reuse.
    """
    at = output.find(REUSE_DISABLED_MARKER)
    if at < 0:
        return None
    end = output.find("\n", at)
    line = output[output.rfind("\n", 0, at) + 1 : end if end >= 0 else None].strip()
    where = f"the unit's options file {options}" if options else "an options file for the unit"
    return (
        f"the unit asks for KV-cache block reuse and the engine turned it off: "
        f"{line[:240]!r}. Set `kv_cache_config.mamba_state_config` (a positive "
        f"`periodic_snapshot_interval` or snapshot offsets) in {where} "
        f"(accuracy.llm_api_options); the gates take it from there. Or turn "
        f"`kv_cache_reuse` off for the unit"
    )


# ------------------------------------------------------------ engine attestation
#
# The engine can run something other than what a gate asks: the LLM API fills
# in the arguments and the executor's worker changes them while it loads the
# model. So the harness prints what the engine runs once it is up (one
# `MV2FLOW_ENGINE` record per engine, `checks/harness/engine_facts.py`) and
# what it reused over the workload (one `MV2FLOW_REUSE` record,
# `checks/harness/reuse_probe.py`), and a gate compares them, field by field,
# with what it declared.

#: The fields a gate holds its engine to, in the order a difference is named.
ATTESTED_FIELDS: tuple[str, ...] = (
    "modeling_v2",
    "block_reuse",
    "mamba_snapshot",
    "kv_cache_dtype",
    "mamba_ssm_cache_dtype",
    "cuda_graph",
    "speculative",
    "max_num_tokens",
    "max_batch_size",
    "chunked_prefill",
    "tp",
    "ep",
    "attention_dp",
    "pp",
    "max_seq_len",
    "overlap_scheduler",
)


#: The harness's record of one engine (``checks/harness/engine_facts.py``).
ENGINE_MARKER = workload_io.ENGINE_MARKER


#: The harness's record of what one engine reused (``checks/harness/reuse_probe.py``).
REUSE_MARKER = workload_io.REUSE_MARKER


#: What each rank's own worker prints of its engine (``checks/harness/rankhook``).
RANK_ENGINE_MARKER = workload_io.RANK_ENGINE_MARKER


#: Why an accuracy verdict's engine is not attested: the run printed no engine
#: record (the eval shim found no ``LLM`` class to wrap in the tree's
#: ``trtllm-eval``).
EVAL_NOT_ATTESTED = "the evaluation printed no engine record"


#: ``modeling_v2 newmodel: bound MambaHybridCacheManagerV2 #2: 4096 recurrent
#: slots, ...`` -- what a target prints, once per rank, each time it binds a
#: recurrent-state pool; the highest-numbered one is the pool the engine serves from.
_BIND_LINE = re.compile(r"\bbound (\w+) #(\d+): (\d+) recurrent slots")


#: ``[MemUsageChange] Allocated 29.11 GiB for max tokens in paged KV cache
#: (524288).`` -- the C++ KV-cache manager, once per rank and pool, at INFO.
_KV_ALLOCATED = re.compile(r"Allocated [0-9.]+ GiB for max tokens in paged KV cache \((\d+)\)")


#: What the worker resolves each KV-cache dtype a gate can name to.
_KV_QUANT_FOR: dict[str, str] = {"fp8": "FP8", "fp8_ds_mla": "FP8", "nvfp4": "NVFP4"}


_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
}


_GIB_PER_MIB = 1.0 / 1024.0


def _dtype_name(value: Any) -> str:
    text = str(value).lower().replace("torch.", "")
    return _DTYPE_ALIASES.get(text, text)


def _show(field_name: str, value: Any) -> str:
    """One field's value as a verdict names it."""
    if value is None:
        return {"cuda_graph": "no CUDA graphs", "speculative": "no speculation"}.get(
            field_name, "none"
        )
    if isinstance(value, bool):
        return "on" if value else "off"
    if field_name == "cuda_graph" and isinstance(value, Mapping):
        sizes = [int(b) for b in value.get("batch_sizes") or ()]
        padded = "padded" if value.get("enable_padding") else "unpadded"
        return f"{len(sizes)} batches [{format_batches(sizes)}], {padded}"
    if field_name == "speculative" and isinstance(value, Mapping):
        return f"{value.get('algorithm')} at draft length {value.get('max_draft_len')}"
    if field_name == "mamba_snapshot" and isinstance(value, Mapping):
        return (
            f"interval {value.get('periodic_snapshot_interval')}, offsets from start "
            f"{value.get('additional_snapshot_offsets_from_start')}, from end "
            f"{value.get('additional_snapshot_offsets_from_end')}"
        )
    return str(value)


def _engine_difference(field_name: str, declared: Any, config: Mapping[str, Any]) -> str | None:
    """How the engine's ``field_name`` differs from ``declared``; ``None`` when it does not.

    The two cache dtypes are compared as asked for (the dtype the LLM API
    holds) and, where the rank-0 worker resolved a named one, as resolved: an
    engine asked for NVFP4 KV that falls back to FP8 ran FP8.
    """
    got = config[field_name]
    if field_name in ("kv_cache_dtype", "mamba_ssm_cache_dtype"):
        want = _dtype_name(declared)
        if _dtype_name(got) != want:
            return f"engine {got}"
        if field_name == "kv_cache_dtype":
            expected = _KV_QUANT_FOR.get(want)
            resolved = config.get("kv_cache_quant")
            if expected and "kv_cache_quant" in config and str(resolved).upper() != expected:
                return f"engine asked for {got} and resolved {resolved}"
        else:
            resolved = config.get("mamba_ssm_cache_dtype_resolved")
            if want != "auto" and resolved is not None and _dtype_name(resolved) != want:
                return f"engine asked for {got} and resolved {resolved}"
        return None
    if field_name == "speculative" and isinstance(declared, Mapping) and isinstance(got, Mapping):
        same = str(declared.get("algorithm")).upper() == str(got.get("algorithm")).upper() and int(
            declared.get("max_draft_len") or 0
        ) == int(got.get("max_draft_len") or 0)
        return None if same else f"engine {_show(field_name, got)}"
    if field_name == "cuda_graph" and isinstance(declared, Mapping) and isinstance(got, Mapping):
        sizes = [int(b) for b in got.get("batch_sizes") or ()]
        same = bool(declared.get("enable_padding")) == bool(got.get("enable_padding"))
        if "batch_sizes" in declared:
            same = same and sorted(declared["batch_sizes"]) == sizes
        if declared.get("max_batch_size"):
            same = same and bool(sizes) and max(sizes) == declared["max_batch_size"]
        return None if same else f"engine {_show(field_name, got)}"
    return None if declared == got else f"engine {_show(field_name, got)}"


def parse_engine_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_ENGINE`` record in ``output``, in order."""
    return workload_io.parse_marked(output, ENGINE_MARKER)


def parse_recurrent_slots(output: str) -> list[int]:
    """The recurrent-state slots of the pool each rank serves from: its last bind line's."""
    binds = [(int(n), int(slots)) for _, n, slots in _BIND_LINE.findall(output)]
    if not binds:
        return []
    last = max(n for n, _ in binds)
    return [slots for n, slots in binds if n == last]


def parse_kv_allocations(output: str) -> list[int]:
    """The max tokens of every paged KV-cache pool the C++ manager logged allocating."""
    return [int(tokens) for tokens in _KV_ALLOCATED.findall(output)]


def parse_rank_records(output: str) -> dict[int, dict[str, Any]]:
    """Each rank's last ``MV2FLOW_RANK_ENGINE`` record in ``output``, by rank."""
    ranks: dict[int, dict[str, Any]] = {}
    for record in workload_io.parse_marked(output, RANK_ENGINE_MARKER):
        rank = record.get("rank")
        if isinstance(rank, int) and not isinstance(rank, bool):
            ranks[rank] = record
    return ranks


def parse_reuse_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_REUSE`` record in ``output``, in order."""
    return workload_io.parse_marked(output, REUSE_MARKER)


def _count(value: Any) -> int | None:
    """``value`` as a count, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def reuse_evidence(reuse: Mapping[str, Any]) -> tuple[bool, str] | None:
    """Positive public cache hits prove reuse; zero does not prove a disabled engine."""
    probe = reuse.get("probe")
    if isinstance(probe, Mapping) and (_count(probe.get("reused")) or 0) > 0:
        return True, SOURCE_PROBE
    if (_count(reuse.get("hits")) or 0) > 0:
        return True, SOURCE_CACHED
    return None


def _holds_recurrent_state(slots: Sequence[int], config: Mapping[str, Any]) -> bool:
    """Whether the engine keeps recurrent state: it bound a recurrent pool, or runs a snapshot policy."""
    if slots:
        return True
    policy = config.get("mamba_snapshot")
    return isinstance(policy, Mapping) and has_snapshot_policy({"mamba_state_config": policy})


@dataclass(frozen=True)
class EngineAttestation:
    """What one engine a gate built ran, held against what the gate declared."""

    #: Which of the gate's engines this is (``require`` or a witness leg);
    #: empty for a gate with one.
    leg: str
    #: Whether the harness printed a record for this engine.
    reported: bool
    declared: Mapping[str, Any] = field(default_factory=dict)
    #: What the engine runs, by field (``config`` of the harness record).
    config: Mapping[str, Any] = field(default_factory=dict)
    #: Actual worker/public/log observation source, never requested constructor values.
    sources: Mapping[str, str] = field(default_factory=dict)
    #: One line per declared field the engine ran differently.
    mismatches: tuple[str, ...] = ()
    #: Declared fields the engine's record does not carry.
    not_attested: tuple[str, ...] = ()
    #: Per rank, from the bind lines: the recurrent slots of the pool served from.
    recurrent_slots: tuple[int, ...] = ()
    #: The rank-0 worker's KV-cache pool, in tokens.
    kv_max_tokens: int | None = None
    #: Per rank and pool, from the KV-cache manager's INFO line, when logged.
    kv_log_tokens: tuple[int, ...] = ()
    #: Per device: ``{"device", "free_gib", "total_gib"}`` once the engine was up.
    devices: tuple[Mapping[str, Any], ...] = ()
    memory_source: str = ""
    #: Why the engine is not attested, or what the harness could not read.
    note: str = ""
    #: What the engine reused over the workload: the harness's ``MV2FLOW_REUSE``
    #: record (``checks/harness/reuse_probe.py``), plus ``recurrent`` (the
    #: engine keeps recurrent state). Cache hits do not attest snapshot restoration.
    #: Empty when the run printed none.
    reuse: Mapping[str, Any] = field(default_factory=dict)
    #: The ranks whose own worker reported its engine (``MV2FLOW_RANK_ENGINE``).
    ranks: tuple[int, ...] = ()
    #: How many ranks the engine runs: its TP width times its PP width.
    world: int = 1

    def unreported_ranks(self) -> tuple[int, ...]:
        """The engine's ranks whose own worker printed no readable record."""
        return tuple(r for r in range(self.world) if r not in self.ranks)

    def record(self) -> dict[str, Any]:
        """The JSON-able form a :class:`GateResult` carries in ``engine``."""
        return {
            "leg": self.leg,
            "reported": self.reported,
            "declared": dict(self.declared),
            "attested": dict(self.config),
            "sources": dict(self.sources),
            "mismatches": list(self.mismatches),
            "not_attested": list(self.not_attested),
            "pools": {
                "recurrent_slots": list(self.recurrent_slots),
                "kv_max_tokens": self.kv_max_tokens,
                "kv_log_tokens": list(self.kv_log_tokens),
            },
            "memory": {"source": self.memory_source, "devices": [dict(d) for d in self.devices]},
            "reuse": dict(self.reuse),
            "ranks": list(self.ranks),
            "world": self.world,
            "note": self.note,
        }

    def difference(self) -> str:
        """The failing verdict's detail: every field the engine ran differently."""
        where = f" of the {self.leg} leg" if self.leg else ""
        return (
            f"the engine{where} did not run the declared configuration -- "
            + "; ".join(self.mismatches)
            + ". The gate's argv and handed-over KV-cache keys come from the unit's runtime "
            "block and options file; the engine's values were read back once it was up "
            f"({ENGINE_MARKER}). A verdict on this engine would certify a configuration "
            "nobody declared"
        )


def attest_engine(
    output: str,
    declared: Mapping[str, Any] | None,
    *,
    leg: str = "",
    absent: str = "",
) -> EngineAttestation:
    """Hold the engine whose log is ``output`` to ``declared``.

    The last ``MV2FLOW_ENGINE`` record is the engine's and each rank's last
    ``MV2FLOW_RANK_ENGINE`` record its worker's; a field every reporting rank
    carries is held to ``declared`` on each rank. A declared field no record
    carries is not attested, never counted as agreeing. ``declared`` ``None``
    compares nothing, and ``absent`` says why.
    """
    want = dict(declared or {})
    slots = tuple(parse_recurrent_slots(output))
    kv_log = tuple(parse_kv_allocations(output))
    reused = parse_reuse_records(output)
    reuse = dict(reused[-1]) if reused else {}
    records = parse_engine_records(output)
    record = records[-1] if records else None
    ranks = parse_rank_records(output)
    rank_configs = {
        r: dict(rec["config"])
        for r, rec in sorted(ranks.items())
        if isinstance(rec.get("config"), Mapping)
    }
    rank_notes = [
        f"rank {r}'s worker could not be read: {rec.get('error') or rec}"
        for r, rec in sorted(ranks.items())
        if r not in rank_configs
    ]
    harness_read = record is not None and isinstance(record.get("config"), Mapping)
    if not harness_read and not rank_configs:
        if record is not None:
            note = f"the harness could not read the engine: {record.get('error') or record}"
        else:
            note = absent or f"the run printed no {ENGINE_MARKER} record"
        if reuse:
            reuse["recurrent"] = bool(slots)
        return EngineAttestation(
            leg=leg,
            reported=False,
            declared=want,
            not_attested=tuple(f for f in ATTESTED_FIELDS if f in want),
            recurrent_slots=slots,
            kv_log_tokens=kv_log,
            note="; ".join([note, *rank_notes])[:300],
            reuse=reuse,
        )
    record = record if harness_read else {}
    config = dict(record.get("config") or {})
    sources = {str(k): str(v) for k, v in (record.get("sources") or {}).items()}
    expected_world = int(want.get("tp", 1)) * int(want.get("pp", 1))
    if rank_configs and set(rank_configs) != set(range(expected_world)):
        rank_notes.append(
            f"worker coverage UNKNOWN: observed {sorted(rank_configs)}, expected {expected_world} ranks"
        )
    if rank_configs:
        lowest = rank_configs[min(rank_configs)]
        for name, value in lowest.items():
            if all(name in c for c in rank_configs.values()):
                config[name] = value
                sources[name] = SOURCE_RANKS
    if "block_reuse" not in config and REUSE_DISABLED_MARKER in output:
        config["block_reuse"] = False
        sources["block_reuse"] = SOURCE_LOG
    evidence = reuse_evidence(reuse)
    if "block_reuse" not in config and evidence is not None:
        config["block_reuse"], sources["block_reuse"] = evidence
    if reuse:
        reuse["recurrent"] = _holds_recurrent_state(slots, config)
    mismatches: list[str] = []
    unattested: list[str] = []
    for name in ATTESTED_FIELDS:
        if name not in want:
            continue
        if name not in config:
            unattested.append(name)
            continue
        resolved_key = {
            "kv_cache_dtype": "kv_cache_quant",
            "mamba_ssm_cache_dtype": "mamba_ssm_cache_dtype_resolved",
            "cuda_graph": "cuda_graph_captured",
        }.get(name)
        evidence_configs = rank_configs.values() if sources.get(name) == SOURCE_RANKS else [config]
        if resolved_key and want[name] is not None and str(want[name]).lower() != "auto":
            if any(resolved_key not in c for c in evidence_configs):
                unattested.append(name)
        if sources.get(name) == SOURCE_RANKS:
            if set(rank_configs) != set(range(expected_world)) and name not in unattested:
                unattested.append(name)
            differing: dict[str, list[int]] = {}
            for rank, rank_config in rank_configs.items():
                difference = _engine_difference(name, want[name], rank_config)
                if difference:
                    differing.setdefault(difference, []).append(rank)
            for difference, at in differing.items():
                mismatches.append(
                    f"{name}: declared {_show(name, want[name])}, {difference} on rank(s) "
                    f"{', '.join(str(r) for r in at)} (read from {SOURCE_RANKS})"
                )
            continue
        difference = _engine_difference(name, want[name], config)
        if difference:
            mismatches.append(
                f"{name}: declared {_show(name, want[name])}, {difference} "
                f"(read from {sources.get(name, 'the engine')})"
            )
    memory = record.get("memory") if isinstance(record.get("memory"), Mapping) else {}
    devices = tuple(
        {
            "device": str(d.get("device")),
            "free_gib": round(float(d["free_mib"]) * _GIB_PER_MIB, 2),
            "total_gib": round(float(d["total_mib"]) * _GIB_PER_MIB, 2),
        }
        for d in memory.get("devices") or ()
        if isinstance(d, Mapping) and d.get("free_mib") is not None and d.get("total_mib")
    )
    pools = record.get("pools") if isinstance(record.get("pools"), Mapping) else {}
    errors = record.get("errors") if isinstance(record.get("errors"), Mapping) else {}
    unread = sorted(set(errors) & set(unattested))
    note = "; ".join(
        filter(
            None, [str(record.get("note") or ""), *(f"{name}: {errors[name]}" for name in unread)]
        )
    )
    if errors.get("worker"):
        note = "; ".join(
            filter(None, [note, f"rank 0's worker could not be read: {errors['worker']}"])
        )
    note = "; ".join(filter(None, [note, *rank_notes]))
    if memory.get("error") and not devices:
        note = "; ".join(filter(None, [note, f"free memory not read: {memory['error']}"]))
    kv_tokens = pools.get("kv_max_tokens")
    if kv_tokens is None and rank_configs:
        first = ranks[min(rank_configs)].get("pools")
        kv_tokens = first.get("kv_max_tokens") if isinstance(first, Mapping) else None
    world = 1
    for width in ("tp", "pp"):
        value = config.get(width, want.get(width))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            world *= value
    return EngineAttestation(
        leg=leg,
        reported=True,
        declared=want,
        config=config,
        sources=sources,
        mismatches=tuple(mismatches),
        not_attested=tuple(unattested),
        recurrent_slots=slots,
        kv_max_tokens=int(kv_tokens) if isinstance(kv_tokens, (int, float)) else None,
        kv_log_tokens=kv_log,
        devices=devices,
        memory_source=str(memory.get("source") or ""),
        note=note[:300],
        reuse=reuse,
        ranks=tuple(rank_configs),
        world=world,
    )


def reuse_shortfall(attestation: EngineAttestation, *, asks: bool) -> str | None:
    """Judge only an explicit repeat experiment, never inferred eval prefix matches."""
    probe = attestation.reuse.get("probe")
    if not asks or not isinstance(probe, Mapping):
        return None
    leg = f" of the {attestation.leg} leg" if attestation.leg else ""
    reused = _count(probe.get("reused"))
    if reused is None:
        why = probe.get("skipped") or probe.get("error") or "cached_tokens was not reported"
        return f"KV-cache reuse UNKNOWN{leg}: the controlled repeat probe could not validate reuse ({why})"
    if reused == 0:
        return (
            f"the engine{leg} reused nothing: the reuse probe sent one {probe.get('tokens')}-token prompt "
            "twice after completion on the same cache, but cached_tokens was 0 on the second response. "
            "Check the declared snapshot policy and engine's actual block-reuse configuration"
        )
    return None


def engine_failure(
    output: str,
    attested: EngineAttestation,
    *,
    options: str | None,
    asks: bool,
    prefix: str = "",
) -> str | None:
    """Why a gate fails on the engine it ran, in this order; ``None`` when it does not.

    The engine turned block reuse off (:func:`reuse_disabled_detail`, after
    ``prefix``), reused nothing although the unit ``asks`` for reuse
    (:func:`reuse_shortfall`), or ran a field other than declared.
    """
    reuse_off = reuse_disabled_detail(output, options=options)
    if reuse_off:
        return prefix + reuse_off
    shortfall = reuse_shortfall(attested, asks=asks)
    if shortfall:
        return shortfall
    return attested.difference() if attested.mismatches else None


def _describe_reuse(reuse: Mapping[str, Any]) -> list[str]:
    """The ``[engine ...]`` parts that say what the engine reused."""
    if reuse.get("error"):
        return [f"reuse not read ({str(reuse['error'])[:80]})"]
    parts: list[str] = []
    requests = _count(reuse.get("requests")) or 0
    hits = _count(reuse.get("hits")) or 0
    if requests and not _count(reuse.get("reported")):
        parts.append("reuse not reported by the engine")
    elif requests:
        text = f"reuse {hits}/{requests} requests"
        if hits:
            text += f", {_count(reuse.get('reused_tokens')) or 0} cached tokens"
        parts.append(text)
    probe = reuse.get("probe")
    if isinstance(probe, Mapping):
        resumed = _count(probe.get("reused"))
        if resumed is not None:
            parts.append(f"reuse probe cached {resumed} of {probe.get('tokens')} tokens")
        else:
            why = probe.get("skipped") or probe.get("error") or "no reading"
            parts.append(f"reuse probe not read ({str(why)[:80]})")
    return parts


def _describe_engine_record(record: Mapping[str, Any]) -> str:
    leg = f"{record['leg']}: " if record.get("leg") else ""
    declared = record.get("declared") or {}
    compared = [name for name in ATTESTED_FIELDS if name in declared]
    differ = record.get("mismatches") or ()
    missing = record.get("not_attested") or ()
    parts: list[str] = []
    if not record.get("reported"):
        parts.append(
            f"NOT ATTESTED ({record.get('note') or 'no record'})"
            if compared
            else "configuration not attested"
        )
    elif compared:
        agreed = len(compared) - len(differ) - len(missing)
        head = f"{agreed}/{len(compared)} fields as declared"
        if differ:
            head += f", {len(differ)} DIFFER"
        if missing:
            head += f", NOT ATTESTED {', '.join(missing)}"
        parts.append(head)
    pools = record.get("pools") or {}
    slots = [int(s) for s in pools.get("recurrent_slots") or ()]
    if slots:
        same = len(set(slots)) == 1
        shown = (
            f"{slots[0]} x{len(slots)}" if same and len(slots) > 1 else "/".join(map(str, slots))
        )
        parts.append(f"recurrent slots {shown}")
    kv_tokens = pools.get("kv_max_tokens")
    if kv_tokens is None and pools.get("kv_log_tokens"):
        kv_tokens = pools["kv_log_tokens"][-1]
    if kv_tokens is not None:
        parts.append(f"KV {int(kv_tokens)} tokens")
    parts.extend(_describe_reuse(record.get("reuse") or {}))
    devices = (record.get("memory") or {}).get("devices") or ()
    if devices:
        low = min(devices, key=lambda d: float(d["free_gib"]))
        parts.append(f"min free {float(low['free_gib']):.2f} of {float(low['total_gib']):.2f} GiB")
    return f"{leg}{'; '.join(parts)}" if parts else ""


def describe_engine(records: Sequence[Mapping[str, Any]]) -> str:
    """A verdict summary's `` [engine ...]`` segment; empty for a gate that built none."""
    shown = [text for text in (_describe_engine_record(r) for r in records) if text]
    return f" [engine {' | '.join(shown)}]" if shown else ""


def engine_records(attestations: Sequence[EngineAttestation]) -> tuple[dict[str, Any], ...]:
    """A :class:`GateResult`'s ``engine``: one record per engine the gate built."""
    return tuple(a.record() for a in attestations)


#: Where a value read from the rank-0 worker in the harness's own process comes
#: from (``checks/harness/engine_facts.py``).
SOURCE_WORKER = "worker"


#: Where one every rank's own worker reported comes from (``checks/harness/rankhook``).
SOURCE_RANKS = "every rank"


#: Where a block-reuse reading taken from the log, not from an object, comes from.
SOURCE_LOG = "engine log"


#: Where one taken from the reuse probe comes from: one prompt sent twice, and
#: the engine resumed the second from its KV cache or computed it from the start.
SOURCE_PROBE = "reuse probe"


#: Where one taken from the workload comes from: a request the engine resumed
#: from its KV cache.
SOURCE_CACHED = "cached tokens"


__all__ = [
    "ATTESTED_FIELDS",
    "ENGINE_MARKER",
    "EVAL_NOT_ATTESTED",
    "EngineAttestation",
    "RANK_ENGINE_MARKER",
    "REUSE_DISABLED_MARKER",
    "REUSE_MARKER",
    "SOURCE_CACHED",
    "SOURCE_LOG",
    "SOURCE_PROBE",
    "SOURCE_RANKS",
    "SOURCE_WORKER",
    "UNSUPPORTED_ARCHITECTURE_MARKERS",
    "attest_engine",
    "describe_engine",
    "engine_failure",
    "engine_records",
    "parse_engine_records",
    "parse_kv_allocations",
    "parse_rank_records",
    "parse_recurrent_slots",
    "parse_reuse_records",
    "reports_unsupported_architecture",
    "reuse_disabled_detail",
    "reuse_evidence",
    "reuse_shortfall",
]
