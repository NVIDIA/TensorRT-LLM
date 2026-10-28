# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature coverage policy: required observations, accepted sources and reference eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .runtime_evidence import (
    SOURCE_CACHED,
    SOURCE_LOG,
    SOURCE_PROBE,
    SOURCE_RANKS,
    SOURCE_WORKER,
    EngineAttestation,
)


@dataclass(frozen=True)
class FeatureEvidence:
    """One feature a check can be about: the evidence it must observe, and its oracle."""

    feature: str
    #: The engine fields (``runtime_evidence.ATTESTED_FIELDS``) whose change makes a check about it.
    fields: tuple[str, ...]
    evidence: str
    oracle: str
    #: Where an observation of a field may be read; never ``llm.args``.
    sources: tuple[str, ...] = ()
    #: The HF reference measures a configuration that changes only these fields.
    reference_compatible: bool = False


_ENGINE = (SOURCE_WORKER, SOURCE_RANKS)


KV_DTYPE = FeatureEvidence(
    "KV-cache dtype",
    ("kv_cache_dtype",),
    "the KV-cache dtype each rank's worker resolved once its engine was up",
    "the declared dtype, on every rank",
    _ENGINE,
)


SSM_DTYPE = FeatureEvidence(
    "SSM-state dtype",
    ("mamba_ssm_cache_dtype",),
    "the recurrent-state dtype each rank's worker resolved once its engine was up",
    "the declared dtype, on every rank",
    _ENGINE,
)


CHUNKED = FeatureEvidence(
    "chunked prefill",
    ("chunked_prefill", "max_num_tokens"),
    "a request whose prompt, less what the KV cache resumed, is longer than max_num_tokens, "
    "served whole: the engine computed it in two or more context chunks",
    "the HF reference's greedy tokens on such a prompt, where the logit reference has one",
    _ENGINE,
    reference_compatible=True,
)


BLOCK_REUSE = FeatureEvidence(
    "block reuse",
    ("block_reuse", "mamba_snapshot"),
    "requests the engine resumed from its KV cache",
    "the reuse probe: one prompt sent twice",
    (*_ENGINE, SOURCE_LOG, SOURCE_PROBE, SOURCE_CACHED),
    reference_compatible=True,
)


SPECULATIVE = FeatureEvidence(
    "speculative decoding",
    ("speculative",),
    "accepted draft tokens per iteration, per prompt (the acceptance gate)",
    "the stock speculator on the same prompts; without one, the same target one draft depth "
    "shallower, which each depth must beat",
    _ENGINE,
)


CUDA_GRAPHS = FeatureEvidence(
    "CUDA graphs",
    ("cuda_graph",),
    "the batches each rank's worker captured",
    "the declared capture list",
    _ENGINE,
    reference_compatible=True,
)


#: Every registered feature the engine fields express, in the order a failing
#: verdict names them.
FEATURES: tuple[FeatureEvidence, ...] = (
    KV_DTYPE,
    SSM_DTYPE,
    CHUNKED,
    BLOCK_REUSE,
    SPECULATIVE,
    CUDA_GRAPHS,
)


#: The engine fields a reference_compatible change is made of: a variant changing only
#: these is compared with the HF reference's greedy tokens.
REFERENCE_COMPATIBLE_FIELDS: frozenset[str] = frozenset(
    f for o in FEATURES if o.reference_compatible for f in o.fields
) | {"max_batch_size", "overlap_scheduler"}


#: What a feature the engine fields do not express is judged by.
MULTIMODAL = FeatureEvidence(
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


def feature_for(field_name: str) -> FeatureEvidence | None:
    """The registered feature ``field_name`` belongs to, if any."""
    return next((o for o in FEATURES if field_name in o.fields), None)


def reference_compatible(subjects: Sequence[str]) -> bool:
    """Whether the HF reference is eligible to judge a variant about ``subjects``.

    Eligibility is a coverage policy, not a promise of bitwise-identical execution:
    the selected comparison still applies its own numerical tolerance.
    """
    return bool(subjects) and all(f in REFERENCE_COMPATIBLE_FIELDS for f in subjects)


def about(subjects: Sequence[str], feature: FeatureEvidence) -> bool:
    """Whether a check about ``subjects`` is about ``feature``."""
    return any(f in feature.fields for f in subjects)


def unobserved(attestation: EngineAttestation, subjects: Sequence[str]) -> list[str]:
    """Why each field of ``subjects`` the engine report does not observe fails the check.

    ``attestation`` is a ``runtime_evidence.EngineAttestation``. A field is observed when
    it was read from a source its feature registers and, when read from every
    rank, every rank of the engine reported.
    """
    failures: list[str] = []
    missing = tuple(attestation.unreported_ranks())
    for name in subjects:
        observable = feature_for(name)
        if observable is None:
            continue
        if name not in attestation.config or name in attestation.not_attested:
            failures.append(f"{name} ({observable.feature}): actual evidence is UNKNOWN")
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


def judges_images(benchmark: str | None) -> bool:
    """Whether an accuracy benchmark's prompts send images."""
    return str(benchmark or "").lower() in IMAGE_BENCHMARKS


__all__ = [
    "BLOCK_REUSE",
    "CHUNKED",
    "CUDA_GRAPHS",
    "FEATURES",
    "FeatureEvidence",
    "IMAGE_BENCHMARKS",
    "KV_DTYPE",
    "MULTIMODAL",
    "REFERENCE_COMPATIBLE_FIELDS",
    "SPECULATIVE",
    "SSM_DTYPE",
    "about",
    "feature_for",
    "judges_images",
    "reference_compatible",
    "unobserved",
    "unobserved_detail",
]
