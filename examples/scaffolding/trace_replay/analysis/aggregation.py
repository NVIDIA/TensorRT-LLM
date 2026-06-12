r"""Dataset-level aggregation of per-trace cache-hit records."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .branch_summary import merge_rollup_arrays

DATASET_SCHEMA = "scaffolding.cache_hit_dataset.v4"


_TRACE_LEVEL_RATE_FIELDS = (
    ("optimal_overall_cache_hit_rate", "optimal_trace_cache_hit_rate"),
    ("optimal_overall_cache_block_hit_rate", "optimal_trace_cache_block_hit_rate"),
)

_SUMMABLE_SUMMARY_FIELDS = (
    "event_count",
    "llm_request_count",
    "total_prompt_tokens",
    "optimal_total_cache_hit_blocks",
    "optimal_total_cache_miss_blocks",
    "optimal_total_cache_hit_tokens",
    "optimal_total_cache_miss_tokens",
    "minimal_cache_blocks",
    "preloaded_system_blocks",
    "distinct_system_prompts",
)

_ROLLUP_KINDS = (
    ("by_branch_root", False),
    ("by_branch_depth", True),
    ("by_system_prompt", False),
)


def aggregate_dataset_record(trace_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-trace records into a flat dataset-level summary record."""
    sums: Dict[str, int] = {field: 0 for field in _SUMMABLE_SUMMARY_FIELDS}
    max_prompt_tokens = 0
    rate_buckets: Dict[str, List[float]] = {
        out_field: [] for _, out_field in _TRACE_LEVEL_RATE_FIELDS
    }

    for record in trace_records:
        summary = record["summary"]
        for field in _SUMMABLE_SUMMARY_FIELDS:
            sums[field] += summary.get(field, 0)
        max_prompt_tokens = max(max_prompt_tokens, summary.get("max_prompt_tokens", 0))
        for in_field, out_field in _TRACE_LEVEL_RATE_FIELDS:
            rate_buckets[out_field].append(summary.get(in_field, 0.0))

    optimal_overall_cache_hit_rate = _safe_div(
        sums["optimal_total_cache_hit_tokens"], sums["total_prompt_tokens"]
    )
    optimal_overall_cache_block_hit_rate = _safe_div(
        sums["optimal_total_cache_hit_blocks"],
        sums["optimal_total_cache_hit_blocks"] + sums["optimal_total_cache_miss_blocks"],
    )

    algorithm = dict(trace_records[0]["algorithm"]) if trace_records else {}
    if algorithm:
        algorithm["cache_scope"] = "per_trace_independent"
    tokens_per_block = algorithm.get("tokens_per_block") or 0

    record: Dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "algorithm": algorithm,
        "trace_count": len(trace_records),
        **{field: sums[field] for field in _SUMMABLE_SUMMARY_FIELDS},
        "tokens_per_block": algorithm.get("tokens_per_block"),
        "decode_kv_reuse": algorithm.get("decode_kv_reuse"),
        "cot_pollutes_cache": algorithm.get("cot_pollutes_cache"),
        "preloaded_system_tokens": sums["preloaded_system_blocks"] * tokens_per_block,
        "minimal_cache_tokens": sums["minimal_cache_blocks"] * tokens_per_block,
        "optimal_overall_cache_hit_rate": optimal_overall_cache_hit_rate,
        "optimal_overall_cache_block_hit_rate": optimal_overall_cache_block_hit_rate,
        "max_prompt_tokens": max_prompt_tokens,
    }

    for _, out_field in _TRACE_LEVEL_RATE_FIELDS:
        record.update(_rate_stats(rate_buckets[out_field], out_field))

    rollups: Dict[str, List[Dict[str, Any]]] = {}
    for kind, sort_numeric in _ROLLUP_KINDS:
        per_trace = [
            rec.get("rollups", {}).get(kind, [])
            for rec in trace_records
            if rec.get("rollups") is not None
        ]
        if per_trace:
            rollups[kind] = merge_rollup_arrays(per_trace, sort_numeric=sort_numeric)
    if rollups:
        record["rollups"] = rollups
    return record


def _rate_stats(values: Sequence[float], field_prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"mean_{field_prefix}": 0.0,
            f"min_{field_prefix}": 0.0,
            f"max_{field_prefix}": 0.0,
        }
    return {
        f"mean_{field_prefix}": sum(values) / len(values),
        f"min_{field_prefix}": min(values),
        f"max_{field_prefix}": max(values),
    }


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0
