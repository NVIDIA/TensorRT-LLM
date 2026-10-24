# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a check must observe of the feature it is about, and what it judges that against.

An engine's report proves a setting took effect only where it comes from the
engine: read from ``llm.args`` it is the request repeated. Each kind of
feature a check can be about is registered here with its evidence (what must
be observed, and where it may be read) and its oracle (what the evidence is
judged against). A configuration variant is about the engine fields its
overlay sets (:func:`subject_fields`); a field it is about that its gate did
not observe fails the gate (:func:`unobserved`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class Observable:
    """One feature a check can be about: the evidence it must observe, and its oracle."""

    feature: str
    #: The engine fields (``gates.ATTESTED_FIELDS``) whose change makes a check about it.
    fields: tuple[str, ...]
    evidence: str
    oracle: str
    #: Where an observation of a field may be read; never ``llm.args``.
    sources: tuple[str, ...] = ()
    #: The HF reference measures a configuration that changes only these fields.
    lossless: bool = False


_ENGINE = (SOURCE_WORKER, SOURCE_RANKS)
KV_DTYPE = Observable(
    "KV-cache dtype",
    ("kv_cache_dtype",),
    "the KV-cache dtype each rank's worker resolved once its engine was up",
    "the declared dtype, on every rank",
    _ENGINE,
)
SSM_DTYPE = Observable(
    "SSM-state dtype",
    ("mamba_ssm_cache_dtype",),
    "the recurrent-state dtype each rank's worker resolved once its engine was up",
    "the declared dtype, on every rank",
    _ENGINE,
)
CHUNKED = Observable(
    "chunked prefill",
    ("chunked_prefill", "max_num_tokens"),
    "a request whose prompt, less what the KV cache resumed, is longer than max_num_tokens, "
    "served whole: the engine computed it in two or more context chunks",
    "the HF reference's greedy tokens on such a prompt, where the logit reference has one",
    _ENGINE,
    lossless=True,
)
BLOCK_REUSE = Observable(
    "block reuse",
    ("block_reuse", "mamba_snapshot"),
    "requests the engine resumed from its KV cache",
    "the reuse probe: one prompt sent twice",
    (*_ENGINE, SOURCE_LOG, SOURCE_PROBE, SOURCE_CACHED),
    lossless=True,
)
SPECULATIVE = Observable(
    "speculative decoding",
    ("speculative",),
    "accepted draft tokens per iteration, per prompt (the acceptance gate)",
    "the stock speculator on the same prompts; without one, the same target one draft depth "
    "shallower, which each depth must beat",
    _ENGINE,
)
CUDA_GRAPHS = Observable(
    "CUDA graphs",
    ("cuda_graph",),
    "the batches each rank's worker captured",
    "the declared capture list",
    _ENGINE,
    lossless=True,
)
#: Every registered feature the engine fields express, in the order a failing
#: verdict names them.
OBSERVABLES: tuple[Observable, ...] = (
    KV_DTYPE,
    SSM_DTYPE,
    CHUNKED,
    BLOCK_REUSE,
    SPECULATIVE,
    CUDA_GRAPHS,
)


#: The engine fields a lossless change is made of: a variant changing only
#: these is compared with the HF reference's greedy tokens.
LOSSLESS_FIELDS: frozenset[str] = frozenset(
    f for o in OBSERVABLES if o.lossless for f in o.fields
) | {"max_batch_size", "overlap_scheduler"}

#: What a feature the engine fields do not express is judged by.
MULTIMODAL = Observable(
    "multimodal",
    (),
    "the engine's output on prompts that send an image",
    "the HF reference's logits on its image prompts (the logit gate), or a benchmark that "
    "sends images",
)
#: Accuracy benchmarks that send images, so their score judges a multimodal unit.
IMAGE_BENCHMARKS: frozenset[str] = frozenset(
    {"mmmu", "mmmu_pro", "mmstar", "mmbench", "chartqa", "docvqa", "textvqa", "ai2d", "mathvista"}
)


#: The quickstart flag each engine field is set by, as a variant's runtime block names it.
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
#: Runtime keys that change no engine field: the gate's own memory margin.
_INERT_SWITCHES = frozenset({"extra_flags", "min_free_memory_gib", "min_free_memory_fraction"})
#: What a variant setting this module cannot place is listed as.
UNKNOWN = "?"


def subject_fields(task_data: Mapping[str, Any], unit: str | None) -> tuple[str, ...]:
    """The engine fields a configuration variant is about: those its overlay sets.

    A setting no field expresses is listed as :data:`UNKNOWN`, so a variant
    setting one is never taken for lossless. A unit that is not a variant is
    about none.
    """
    from ..spec.variants import is_variant, variants

    if not is_variant(task_data, unit):
        return ()
    entry = variants(task_data)[str(unit)]
    found: list[str] = []
    runtime = entry.get("runtime") or {}
    for key in runtime:
        if key not in _INERT_SWITCHES:
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


def observable_for(field_name: str) -> Observable | None:
    """The registered feature ``field_name`` belongs to, if any."""
    return next((o for o in OBSERVABLES if field_name in o.fields), None)


def lossless(subjects: Sequence[str]) -> bool:
    """Whether a variant about ``subjects`` changes only what the HF reference measures."""
    return bool(subjects) and all(f in LOSSLESS_FIELDS for f in subjects)


def about(subjects: Sequence[str], feature: Observable) -> bool:
    """Whether a check about ``subjects`` is about ``feature``."""
    return any(f in feature.fields for f in subjects)


def unobserved(attestation: Any, subjects: Sequence[str]) -> list[str]:
    """Why each field of ``subjects`` the engine report does not observe fails the check.

    ``attestation`` is a ``gates.EngineAttestation``. A field is observed when
    it was read from a source its feature registers and, when read from every
    rank, every rank of the engine reported.
    """
    failures: list[str] = []
    missing = tuple(attestation.unreported_ranks())
    for name in subjects:
        observable = observable_for(name)
        if observable is None:
            continue
        if name not in attestation.config:
            failures.append(f"{name} ({observable.feature}): no record carries it")
            continue
        source = attestation.sources.get(name, "")
        if source not in observable.sources:
            failures.append(
                f"{name} ({observable.feature}): read only from {source or 'nowhere'}, which "
                f"repeats the request; the evidence is {observable.evidence}"
            )
        elif source == SOURCE_RANKS and missing:
            failures.append(
                f"{name} ({observable.feature}): rank(s) {', '.join(str(r) for r in missing)} "
                f"of {attestation.world} printed no record of their engine"
            )
    return failures


def unobserved_detail(failures: Sequence[str]) -> str:
    """The failing verdict's detail for :func:`unobserved`'s answer."""
    return (
        "the check cannot show the setting it exists to prove took effect -- "
        + "; ".join(failures)
        + ". Each rank's worker prints what its engine runs (MV2FLOW_RANK_ENGINE, through the "
        "gates' rank hook on PYTHONPATH); a rank that printed none ran without the hook, or "
        "its worker could not be read (the log says why)"
    )


def chunk_shortfall(records: Sequence[Mapping[str, Any] | None], max_num_tokens: int) -> str | None:
    """Why no request shows chunked prefill ran; ``None`` when one does.

    A request whose prompt, less the tokens the KV cache resumed, is longer
    than ``max_num_tokens`` cannot be computed in one step: one served is the
    evidence.
    """
    longest = 0
    for record in records:
        if not record:
            continue
        prompt = record.get("prompt_tokens")
        if not isinstance(prompt, int):
            continue
        cached = record.get("cached_tokens")
        computed = prompt - (cached if isinstance(cached, int) else 0)
        if computed > max_num_tokens:
            return None
        longest = max(longest, computed)
    return (
        f"no request was longer than max_num_tokens {max_num_tokens} past what the KV cache "
        f"resumed (the longest computed {longest} prompt tokens), so the engine never split a "
        f"context into chunks and this check shows nothing of chunked prefill"
    )


def chunk_probe_tokens(max_num_tokens: int, max_seq_len: int | None) -> int | None:
    """The length of a prompt the engine must split: twice ``max_num_tokens``, within ``max_seq_len``.

    ``None`` when the engine's limits leave no room for one.
    """
    want = 2 * int(max_num_tokens) + 1
    if max_seq_len is not None:
        want = min(want, int(max_seq_len) - 2)
    return want if want > max_num_tokens else None


def judges_images(benchmark: str | None) -> bool:
    """Whether an accuracy benchmark's prompts send images."""
    return str(benchmark or "").lower() in IMAGE_BENCHMARKS


__all__ = [
    "BLOCK_REUSE",
    "CHUNKED",
    "CUDA_GRAPHS",
    "IMAGE_BENCHMARKS",
    "KV_DTYPE",
    "LOSSLESS_FIELDS",
    "MULTIMODAL",
    "OBSERVABLES",
    "SOURCE_CACHED",
    "SOURCE_LOG",
    "SOURCE_PROBE",
    "SOURCE_RANKS",
    "SOURCE_WORKER",
    "SPECULATIVE",
    "SSM_DTYPE",
    "UNKNOWN",
    "Observable",
    "about",
    "chunk_probe_tokens",
    "chunk_shortfall",
    "judges_images",
    "lossless",
    "observable_for",
    "subject_fields",
    "unobserved",
    "unobserved_detail",
]
