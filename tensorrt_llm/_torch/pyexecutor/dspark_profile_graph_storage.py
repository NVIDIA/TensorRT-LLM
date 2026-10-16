# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline-profiler metadata storage checks; no model or graph operations."""

from __future__ import annotations

import json
from typing import Any

import torch

_V4_METADATA = (
    "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata",
    "DeepseekV4TrtllmAttentionMetadata",
)
_TRT_METADATA_TYPES = {
    _V4_METADATA,
    ("tensorrt_llm._torch.attention.backends.sparse.dsa.metadata", "DSAtrtllmAttentionMetadata"),
    ("tensorrt_llm._torch.attention.backends.trtllm", "TrtllmAttentionMetadata"),
}
_V4_BUFFER_MAPS = (
    "new_comp_kv_lens_cuda",
    "cu_new_comp_kv_cuda",
    "compressed_kv_lens_cuda",
    "past_kv_lens_cuda",
    "compressed_position_ids_cuda",
    "compressed_mask_cuda",
    "compress_block_tables",
    "sparse_mla_topk_lens",
)


def _metadata_type(obj: Any) -> tuple[str, str]:
    return type(obj).__module__, type(obj).__name__


def _signature(value: torch.Tensor) -> dict[str, Any]:
    return {
        "address": value.data_ptr(),
        "storage_offset": value.storage_offset(),
        "shape": tuple(value.shape),
        "stride": tuple(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _role(obj: Any, name: str) -> str:
    metadata_type = _metadata_type(obj)
    if (
        name == "gen_indexer_kv_lens_cuda_runtime"
        and metadata_type == _V4_METADATA
        and obj.is_cuda_graph
        and obj.num_contexts == 0
        and obj.num_generations > 0
        and obj._indexer_compress_ratio == 4
    ):
        return "graph_produced_compressed_lengths"
    if name == "workspace" and metadata_type in _TRT_METADATA_TYPES and obj.is_cuda_graph:
        return "eager_only_workspace"
    return "persistent"


def _refuse(
    obj: Any, name: str, reason: str, before: Any, after: Any, context: dict[str, Any] | None = None
) -> None:
    context = {} if context is None else context
    details = {
        "reason": reason,
        "field": name,
        "metadata_type": _metadata_type(obj),
        "expected": before,
        "actual": after,
        "rank": context.get("rank"),
        "graph_key": context.get("graph_key"),
    }
    raise RuntimeError(
        "Profile graph storage contract failed: " + json.dumps(details, sort_keys=True)
    )


def _cuda_tensors(obj: Any, *, context: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    tensors = {
        name: value
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor) and value.is_cuda
    }
    if _metadata_type(obj) == _V4_METADATA:
        for name in _V4_BUFFER_MAPS:
            values = getattr(obj, name, None)
            if not isinstance(values, dict) or any(type(key) is not int for key in values):
                _refuse(
                    obj,
                    name,
                    "Invalid or missing required profile storage map",
                    "dict with integer keys",
                    type(values).__name__,
                    context,
                )
            for key, value in values.items():
                if not isinstance(value, torch.Tensor) or not value.is_cuda:
                    _refuse(
                        obj,
                        f"{name}[{key}]",
                        "Invalid profile CUDA storage",
                        "CUDA tensor",
                        type(value).__name__,
                        context,
                    )
                tensors[f"{name}[{key}]"] = value
    return tensors


def _validate_intermediate(obj: Any, value: torch.Tensor) -> None:
    if (
        tuple(value.shape) != (obj.num_generations,)
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.dtype != obj.kv_lens_cuda.dtype
        or value.device != obj.kv_lens_cuda.device
    ):
        raise RuntimeError("Invalid profile graph-produced compressed-length tensor")


def snapshot_metadata_storage(obj: Any) -> dict[str, Any]:
    """Record top-level CUDA tensors and audited V4 persistent buffer maps.

    This is a finite profiler contract, not an exhaustive traversal of arbitrary
    runtime objects. Unknown tensors are persistent; no user-provided exemption
    list or baseline reset is accepted.
    """
    records = {}
    for name, value in _cuda_tensors(obj).items():
        role = _role(obj, name)
        if role == "graph_produced_compressed_lengths":
            _validate_intermediate(obj, value)
        records[name] = {"role": role, "signature": _signature(value)}
    return {"metadata_type": _metadata_type(obj), "fields": records}


def validate_metadata_storage(
    obj: Any, expected: dict[str, Any], *, context: dict[str, Any] | None = None
) -> None:
    """Refuse genuine storage/layout drift with finite, field-specific details."""

    def refuse(name: str, reason: str, before: Any, after: Any) -> None:
        _refuse(obj, name, reason, before, after, context)

    if _metadata_type(obj) != tuple(expected["metadata_type"]):
        refuse(
            "<metadata_type>",
            "metadata type changed",
            expected["metadata_type"],
            _metadata_type(obj),
        )
    actual = _cuda_tensors(obj, context=context)
    before_fields = expected["fields"]
    if set(actual) != set(before_fields):
        changed = sorted(set(actual) ^ set(before_fields))[0]
        refuse(
            changed,
            "CUDA field missing or introduced",
            before_fields.get(changed),
            _signature(actual[changed]) if changed in actual else None,
        )
    for name, value in actual.items():
        before = before_fields[name]
        role = _role(obj, name)
        after = _signature(value)
        if role != before["role"]:
            refuse(name, "field role changed", before, {"role": role, "signature": after})
        if role == "graph_produced_compressed_lengths":
            try:
                _validate_intermediate(obj, value)
            except RuntimeError as error:
                refuse(name, str(error), before, after)
            comparable_before = {
                key: item for key, item in before["signature"].items() if key != "address"
            }
            comparable_after = {key: item for key, item in after.items() if key != "address"}
        elif role == "eager_only_workspace":
            # The graph uses cuda_graph_workspace, which remains persistent.
            comparable_before = {key: before["signature"][key] for key in ("dtype", "device")}
            comparable_after = {key: after[key] for key in ("dtype", "device")}
        else:
            comparable_before, comparable_after = before["signature"], after
        if comparable_after != comparable_before:
            refuse(name, "storage or layout changed", before, {"role": role, "signature": after})
