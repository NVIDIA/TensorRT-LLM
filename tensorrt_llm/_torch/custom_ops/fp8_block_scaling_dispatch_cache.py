# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON persistence for validated FP8 block-scaling dispatch decisions."""

import json
from pathlib import Path

from .fp8_block_scaling_dispatch import (
    ActivationScaleLayout,
    CacheIdentity,
    DispatchBackend,
    DispatchCache,
    DispatchEntry,
    DispatchKey,
    MatrixLayout,
    WeightScaleLayout,
)


def _identity_to_json(identity: CacheIdentity) -> dict[str, int | str | bool | list[str]]:
    return {
        "sm": identity.sm,
        "device_class": identity.device_class,
        "trtllm_version": identity.trtllm_version,
        "trtllm_build_id": identity.trtllm_build_id,
        "deep_gemm_version": identity.deep_gemm_version,
        "deep_gemm_available": identity.deep_gemm_available,
        "policy_version": identity.policy_version,
        "backend_candidates": list(identity.backend_candidates),
    }


def _entry_to_json(entry: DispatchEntry) -> dict[str, int | str]:
    return {
        "m": entry.key.m,
        "n": entry.key.n,
        "k": entry.key.k,
        "activation_scale_layout": entry.key.activation_scale_layout.value,
        "weight_scale_layout": entry.key.weight_scale_layout.value,
        "matrix_layout": entry.key.matrix_layout.value,
        "backend": entry.backend.value,
    }


def write_dispatch_cache(path: Path, cache: DispatchCache) -> None:
    """Atomically write a validated dispatch cache."""
    payload = {
        "schema_version": 1,
        "identity": _identity_to_json(cache.identity),
        "entries": [_entry_to_json(entry) for entry in cache.entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def load_dispatch_cache(path: Path) -> DispatchCache:
    """Load a dispatch cache, rejecting unsupported schema versions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported FP8 dispatch cache schema: {payload['schema_version']}")

    identity_payload = payload["identity"]
    identity = CacheIdentity(
        sm=identity_payload["sm"],
        device_class=identity_payload["device_class"],
        trtllm_version=identity_payload["trtllm_version"],
        trtllm_build_id=identity_payload["trtllm_build_id"],
        deep_gemm_version=identity_payload["deep_gemm_version"],
        deep_gemm_available=identity_payload["deep_gemm_available"],
        policy_version=identity_payload["policy_version"],
        backend_candidates=tuple(identity_payload["backend_candidates"]),
    )
    entries = tuple(
        DispatchEntry(
            key=DispatchKey(
                m=entry["m"],
                n=entry["n"],
                k=entry["k"],
                activation_scale_layout=ActivationScaleLayout(entry["activation_scale_layout"]),
                weight_scale_layout=WeightScaleLayout(entry["weight_scale_layout"]),
                matrix_layout=MatrixLayout(entry["matrix_layout"]),
            ),
            backend=DispatchBackend(entry["backend"]),
        )
        for entry in payload["entries"]
    )
    return DispatchCache(identity=identity, entries=entries)
