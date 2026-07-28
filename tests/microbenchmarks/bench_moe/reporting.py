# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape as _xml_escape

import torch


def build_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group serialized result rows by workload and rank candidates by score."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        workload = row.get("workload") or {}
        requested_cfg = row.get("requested_config") or {}
        num_tokens = int(workload.get("num_tokens") or 0)
        parallel_mode = str(requested_cfg.get("parallel_mode") or "")
        grouped.setdefault((num_tokens, parallel_mode), []).append(row)

    rankings: list[dict[str, Any]] = []
    for (num_tokens, parallel_mode), items in sorted(grouped.items()):
        ranking_entries: list[dict[str, Any]] = []
        for row in items:
            actual_cfg = row.get("actual_config") or {}
            requested_cfg = row.get("requested_config") or {}
            instrumentation = row.get("instrumentation") or {}
            latency = row.get("latency_ms") or {}
            score = latency.get("score") if isinstance(latency, dict) else None
            raw_score = latency.get("raw_score") if isinstance(latency, dict) else None
            outliers = latency.get("iter_max_outliers") if isinstance(latency, dict) else {}
            ranking_entries.append(
                {
                    "backend": actual_cfg.get("backend") or requested_cfg.get("backend"),
                    "requested_backend": requested_cfg.get("backend"),
                    "comm_method": actual_cfg.get("comm_method"),
                    "cuda_graph": requested_cfg.get("cuda_graph"),
                    "use_low_precision_moe_combine": requested_cfg.get(
                        "use_low_precision_moe_combine"
                    ),
                    "score_ms": float(score) if isinstance(score, (int, float)) else None,
                    "raw_score_ms": float(raw_score)
                    if isinstance(raw_score, (int, float))
                    else None,
                    "outlier_count": outliers.get("count") if isinstance(outliers, dict) else None,
                    "status": row.get("status"),
                    "skip_reason": row.get("skip_reason"),
                    "autotune_status": instrumentation.get("autotune_status"),
                }
            )
        ranking_entries.sort(
            key=lambda e: (
                e["score_ms"] is None,
                e["score_ms"] if e["score_ms"] is not None else 0.0,
            )
        )
        best = next(
            (
                e
                for e in ranking_entries
                if e["score_ms"] is not None
                and e["status"] == "success"
                and not (
                    isinstance(e.get("autotune_status"), str)
                    and e["autotune_status"].startswith("failed")
                )
            ),
            None,
        )
        rankings.append(
            {
                "num_tokens": int(num_tokens),
                "parallel_mode": parallel_mode,
                "best": best,
                "ranking": ranking_entries,
            }
        )
    return rankings


def _trtllm_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _build_environment_block(
    world_size: int, cuda_graph_default: bool, repo_root: Path
) -> dict[str, Any]:
    device_name = None
    sm = None
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            device_name = props.name
            sm = int(getattr(props, "major", 0) * 10 + getattr(props, "minor", 0))
        except Exception:
            pass
    cuda_version = None
    try:
        cuda_version = torch.version.cuda
    except Exception:
        pass
    driver_version = None
    torch_driver_version = getattr(torch.cuda, "_get_driver_version", None)
    if torch_driver_version is not None:
        try:
            driver_version = torch_driver_version()
        except Exception:
            pass

    try:
        host = socket.gethostname()
    except Exception:
        host = None
    try:
        user = getpass.getuser()
    except Exception:
        user = None

    return {
        "world_size": int(world_size),
        "world_size_per_node": int(min(world_size, max(torch.cuda.device_count(), 1))),
        "hostname": host,
        "username": user,
        "device_name": device_name,
        "sm": sm,
        "cuda_version": str(cuda_version) if cuda_version else None,
        "driver_version": str(driver_version) if driver_version else None,
        "torch_version": str(torch.__version__),
        "trtllm_commit": _trtllm_commit(repo_root),
        "platform": platform.platform(),
        "nvlink_topology": "unknown",
        "memory_type": "unknown",
        "clock_locked": False,
        "cuda_graph_default": bool(cuda_graph_default),
    }


def build_report_payload(
    *,
    ctx: Any,
    rows: list[dict[str, Any]],
    world_size: int,
    cuda_graph_default: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the dashboard JSON payload from already-serialized result rows."""
    return {
        "benchmark": "bench_moe",
        "environment": _build_environment_block(world_size, cuda_graph_default, repo_root),
        "model": ctx.model.to_dict(),
        "search": ctx.search.to_dict(),
        "base_config": ctx.base_config.to_dict(),
        "results": list(rows),
        "rankings": build_rankings(rows),
    }


def _excel_safe_sheet_name(name: str, used: set[str]) -> str:
    invalid = set("[]:*?/\\")
    base = "".join("_" if ch in invalid else ch for ch in str(name)).strip() or "sheet"
    base = base[:31]
    candidate = base
    suffix = 1
    while candidate in used:
        tail = f"_{suffix}"
        candidate = f"{base[: 31 - len(tail)]}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


def _excel_col_name(index: int) -> str:
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _excel_cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{_excel_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        text = "TRUE" if value else "FALSE"
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            text = str(value)
            return f'<c r="{ref}" t="inlineStr"><is><t>{_xml_escape(text)}</t></is></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = str(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<c r="{ref}" t="inlineStr"><is><t{preserve}>{_xml_escape(text)}</t></is></c>'


def _excel_sheet_xml(rows: list[list[Any]]) -> str:
    body: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(
            _excel_cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row)
        )
        body.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>" + "".join(body) + "</sheetData></worksheet>"
    )


def _write_xlsx_workbook(path: str, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    """Write a minimal XLSX workbook using only the Python standard library."""
    used_names: set[str] = set()
    named_sheets = [(_excel_safe_sheet_name(name, used_names), rows) for name, rows in sheets]
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    for idx, (name, _rows) in enumerate(named_sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{_xml_escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(content_overrides)
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>" + "".join(workbook_sheets) + "</sheets></workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels)
        + "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, (_name, rows) in enumerate(named_sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _excel_sheet_xml(rows))


def _latency_rank_value(row: dict[str, Any], rank_name: str, metric: str) -> Optional[float]:
    per_rank = ((row.get("latency_ms") or {}).get("per_rank") or {}).get(rank_name) or {}
    value = per_rank.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _flatten_result_for_analysis(row: dict[str, Any]) -> dict[str, Any]:
    workload = row.get("workload") or {}
    requested = row.get("requested_config") or {}
    actual = row.get("actual_config") or {}
    instrumentation = row.get("instrumentation") or {}
    latency = row.get("latency_ms") or {}
    routing_actual = (row.get("routing_control") or {}).get("actual") or {}
    dispatch_summary = routing_actual.get("observed_dispatch_matrix_summary") or {}
    hist_summary = routing_actual.get("observed_expert_histogram_summary") or {}
    kernel_breakdown = row.get("kernel_breakdown") or {}
    iter_max_stats = latency.get("iter_max_stats") or {}
    iter_max_outliers = latency.get("iter_max_outliers") or {}
    return {
        "num_tokens": workload.get("num_tokens"),
        "per_rank_num_tokens": json.dumps(workload.get("per_rank_num_tokens")),
        "requested_backend": requested.get("backend"),
        "requested_comm_method": requested.get("comm_method"),
        "requested_parallel_mode": requested.get("parallel_mode"),
        "requested_cuda_graph": requested.get("cuda_graph"),
        "requested_low_precision_combine": requested.get("use_low_precision_moe_combine"),
        "actual_backend": actual.get("backend"),
        "actual_comm_method": actual.get("comm_method"),
        "scheduler_kind": actual.get("scheduler_kind"),
        "actual_moe_ep_size": actual.get("moe_ep_size"),
        "actual_moe_tp_size": actual.get("moe_tp_size"),
        "actual_attention_dp": actual.get("enable_attention_dp"),
        "num_chunks": actual.get("num_chunks"),
        "status": row.get("status"),
        "skip_reason": row.get("skip_reason"),
        "score_ms": latency.get("score"),
        "score_type": latency.get("score_type"),
        "raw_score_ms": latency.get("raw_score"),
        "raw_score_type": latency.get("raw_score_type"),
        "iter_max_median_ms": iter_max_stats.get("median"),
        "iter_max_p90_ms": iter_max_stats.get("p90"),
        "iter_max_max_ms": iter_max_stats.get("max"),
        "iter_max_outlier_count": iter_max_outliers.get("count"),
        "rank0_mean_ms": _latency_rank_value(row, "rank0", "mean"),
        "rank1_mean_ms": _latency_rank_value(row, "rank1", "mean"),
        "rank2_mean_ms": _latency_rank_value(row, "rank2", "mean"),
        "rank3_mean_ms": _latency_rank_value(row, "rank3", "mean"),
        "rank0_p90_ms": _latency_rank_value(row, "rank0", "p90"),
        "rank1_p90_ms": _latency_rank_value(row, "rank1", "p90"),
        "rank2_p90_ms": _latency_rank_value(row, "rank2", "p90"),
        "rank3_p90_ms": _latency_rank_value(row, "rank3", "p90"),
        "autotune_status": instrumentation.get("autotune_status"),
        "latency_source": instrumentation.get("latency_source"),
        "analysis_level": instrumentation.get("level"),
        "nsys_capture": instrumentation.get("nsys_capture"),
        "kernel_breakdown_available": instrumentation.get("kernel_breakdown_available"),
        "bottleneck": row.get("bottleneck"),
        "moe_forward_kernel_count": len(kernel_breakdown.get("moe_forward_kernels") or []),
        "other_kernel_count": len(kernel_breakdown.get("other_kernels") or []),
        "routing_path": routing_actual.get("routing_path"),
        "routing_realization_status": (routing_actual.get("routing_realization") or {}).get(
            "status"
        ),
        "routing_observation_source": routing_actual.get("observation_source"),
        "routing_off_diagonal_ratio": dispatch_summary.get("off_diagonal_ratio"),
        "routing_active_experts": hist_summary.get("active_experts"),
    }


_ANALYSIS_COLUMNS: tuple[str, ...] = (
    "num_tokens",
    "requested_parallel_mode",
    "requested_backend",
    "requested_comm_method",
    "requested_cuda_graph",
    "requested_low_precision_combine",
    "actual_backend",
    "actual_comm_method",
    "scheduler_kind",
    "actual_moe_ep_size",
    "actual_moe_tp_size",
    "actual_attention_dp",
    "num_chunks",
    "status",
    "skip_reason",
    "score_ms",
    "score_type",
    "raw_score_ms",
    "raw_score_type",
    "iter_max_median_ms",
    "iter_max_p90_ms",
    "iter_max_max_ms",
    "iter_max_outlier_count",
    "rank0_mean_ms",
    "rank1_mean_ms",
    "rank2_mean_ms",
    "rank3_mean_ms",
    "rank0_p90_ms",
    "rank1_p90_ms",
    "rank2_p90_ms",
    "rank3_p90_ms",
    "autotune_status",
    "latency_source",
    "analysis_level",
    "nsys_capture",
    "kernel_breakdown_available",
    "bottleneck",
    "moe_forward_kernel_count",
    "other_kernel_count",
    "routing_path",
    "routing_realization_status",
    "routing_observation_source",
    "routing_off_diagonal_ratio",
    "routing_active_experts",
    "per_rank_num_tokens",
)


def _analysis_table_rows(flat_rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [list(_ANALYSIS_COLUMNS)] + [
        [row.get(col) for col in _ANALYSIS_COLUMNS] for row in flat_rows
    ]


def _best_by_workload_rows(rankings: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = [
        [
            "num_tokens",
            "parallel_mode",
            "backend",
            "requested_backend",
            "comm_method",
            "cuda_graph",
            "score_ms",
            "status",
            "skip_reason",
            "autotune_status",
        ]
    ]
    for ranking in rankings:
        best = ranking.get("best") or {}
        rows.append(
            [
                ranking.get("num_tokens"),
                ranking.get("parallel_mode"),
                best.get("backend"),
                best.get("requested_backend"),
                best.get("comm_method"),
                best.get("cuda_graph"),
                best.get("score_ms"),
                best.get("status"),
                best.get("skip_reason"),
                best.get("autotune_status"),
            ]
        )
    return rows


def _status_summary_rows(flat_rows: list[dict[str, Any]]) -> list[list[Any]]:
    counts: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in flat_rows:
        key = (
            row.get("num_tokens"),
            row.get("requested_backend"),
            row.get("requested_comm_method"),
            row.get("status"),
        )
        entry = counts.setdefault(key, {"count": 0, "reasons": set()})
        entry["count"] += 1
        if row.get("skip_reason"):
            entry["reasons"].add(str(row["skip_reason"]))
    out: list[list[Any]] = [
        ["num_tokens", "requested_backend", "requested_comm_method", "status", "count", "reasons"]
    ]
    for (num_tokens, backend, comm_method, status), entry in sorted(
        counts.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), str(item[0][2]), str(item[0][3])),
    ):
        out.append(
            [
                num_tokens,
                backend,
                comm_method,
                status,
                entry["count"],
                " | ".join(sorted(entry["reasons"])),
            ]
        )
    return out


_KERNEL_BREAKDOWN_BASE_COLUMNS: tuple[str, ...] = (
    "result_index",
    "num_tokens",
    "requested_parallel_mode",
    "requested_backend",
    "requested_comm_method",
    "requested_cuda_graph",
    "requested_low_precision_combine",
    "actual_backend",
    "actual_comm_method",
    "scheduler_kind",
    "status",
    "score_ms",
    "category",
    "kernel_name",
    "kernel_count",
)


def _rank_sort_key(rank_name: str) -> tuple[int, str]:
    prefix = "rank"
    if rank_name.startswith(prefix) and rank_name[len(prefix) :].isdigit():
        return (int(rank_name[len(prefix) :]), rank_name)
    return (10**9, rank_name)


def _kernel_stat(stats: dict[str, Any], metric: str) -> Optional[float]:
    value = stats.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _kernel_breakdown_rank_names(rows: list[dict[str, Any]]) -> list[str]:
    rank_names: set[str] = set()
    for row in rows:
        kernel_breakdown = row.get("kernel_breakdown") or {}
        for category in ("moe_forward_kernels", "other_kernels"):
            for kernel in kernel_breakdown.get(category) or []:
                per_rank = kernel.get("per_rank") or {}
                rank_names.update(str(rank_name) for rank_name in per_rank)
    return sorted(rank_names, key=_rank_sort_key)


def _kernel_breakdown_columns(rank_names: list[str]) -> list[str]:
    return (
        list(_KERNEL_BREAKDOWN_BASE_COLUMNS)
        + [f"{rank_name}_mean_ms" for rank_name in rank_names]
        + [f"{rank_name}_median_ms" for rank_name in rank_names]
    )


def _kernel_rank_metric_values(
    per_rank: dict[str, Any], rank_names: list[str], metric: str
) -> list[Optional[float]]:
    values: list[Optional[float]] = []
    for rank_name in rank_names:
        stats = per_rank.get(rank_name)
        values.append(_kernel_stat(stats, metric) if isinstance(stats, dict) else None)
    return values


def _kernel_breakdown_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    rank_names = _kernel_breakdown_rank_names(rows)
    out: list[list[Any]] = [_kernel_breakdown_columns(rank_names)]
    for result_index, row in enumerate(rows):
        workload = row.get("workload") or {}
        requested = row.get("requested_config") or {}
        actual = row.get("actual_config") or {}
        latency = row.get("latency_ms") or {}
        common = [
            result_index,
            workload.get("num_tokens"),
            requested.get("parallel_mode"),
            requested.get("backend"),
            requested.get("comm_method"),
            requested.get("cuda_graph"),
            requested.get("use_low_precision_moe_combine"),
            actual.get("backend"),
            actual.get("comm_method"),
            actual.get("scheduler_kind"),
            row.get("status"),
            latency.get("score"),
        ]
        kernel_breakdown = row.get("kernel_breakdown") or {}
        for category in ("moe_forward_kernels", "other_kernels"):
            for kernel in kernel_breakdown.get(category) or []:
                per_rank = kernel.get("per_rank") or {}
                out.append(
                    common
                    + [
                        category,
                        kernel.get("name"),
                        kernel.get("count"),
                    ]
                    + _kernel_rank_metric_values(per_rank, rank_names, "mean")
                    + _kernel_rank_metric_values(per_rank, rank_names, "median")
                )
    return out


_RAW_DATA_COLUMNS: tuple[str, ...] = (
    "result_index",
    "num_tokens",
    "requested_parallel_mode",
    "requested_backend",
    "requested_comm_method",
    "requested_cuda_graph",
    "requested_low_precision_combine",
    "actual_backend",
    "actual_comm_method",
    "scheduler_kind",
    "status",
    "score_ms",
    "record_type",
    "category",
    "kernel_name",
    "rank",
    "iteration",
    "time_ms",
)


def _result_common_row(result_index: int, row: dict[str, Any]) -> list[Any]:
    workload = row.get("workload") or {}
    requested = row.get("requested_config") or {}
    actual = row.get("actual_config") or {}
    latency = row.get("latency_ms") or {}
    return [
        result_index,
        workload.get("num_tokens"),
        requested.get("parallel_mode"),
        requested.get("backend"),
        requested.get("comm_method"),
        requested.get("cuda_graph"),
        requested.get("use_low_precision_moe_combine"),
        actual.get("backend"),
        actual.get("comm_method"),
        actual.get("scheduler_kind"),
        row.get("status"),
        latency.get("score"),
    ]


def _raw_sample_value(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _raw_data_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    out: list[list[Any]] = [list(_RAW_DATA_COLUMNS)]
    for result_index, row in enumerate(rows):
        common = _result_common_row(result_index, row)
        raw_data = row.get("raw_data") or {}
        forward_per_rank = (raw_data.get("forward_times_ms") or {}).get("per_rank") or {}
        for rank_name, times in sorted(
            forward_per_rank.items(), key=lambda item: _rank_sort_key(item[0])
        ):
            for iteration, value in enumerate(times or []):
                out.append(
                    common
                    + [
                        "forward",
                        None,
                        None,
                        rank_name,
                        iteration,
                        _raw_sample_value(value),
                    ]
                )

        kernel_times = raw_data.get("kernel_times_ms") or {}
        for category in ("moe_forward_kernels", "other_kernels"):
            for kernel in kernel_times.get(category) or []:
                per_rank = kernel.get("per_rank") or {}
                for rank_name, times in sorted(
                    per_rank.items(), key=lambda item: _rank_sort_key(item[0])
                ):
                    for iteration, value in enumerate(times or []):
                        out.append(
                            common
                            + [
                                "kernel",
                                category,
                                kernel.get("name"),
                                rank_name,
                                iteration,
                                _raw_sample_value(value),
                            ]
                        )
    return out


def _workload_sheets(flat_rows: list[dict[str, Any]]) -> list[tuple[str, list[list[Any]]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in flat_rows:
        grouped.setdefault(row.get("num_tokens"), []).append(row)
    sheets: list[tuple[str, list[list[Any]]]] = []
    for num_tokens, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                row.get("score_ms") is None,
                row.get("score_ms") if row.get("score_ms") is not None else 0.0,
                str(row.get("requested_backend")),
                str(row.get("requested_comm_method")),
            ),
        )
        sheets.append((f"workload_{num_tokens}", _analysis_table_rows(sorted_rows)))
    return sheets


def _build_analysis_workbook_sheets(payload: dict[str, Any]) -> list[tuple[str, list[list[Any]]]]:
    rows = payload.get("results") or []
    flat_rows = [_flatten_result_for_analysis(row) for row in rows]
    return [
        ("all_results", _analysis_table_rows(flat_rows)),
        ("best_by_workload", _best_by_workload_rows(payload.get("rankings") or [])),
        ("status_summary", _status_summary_rows(flat_rows)),
        ("kernel_breakdown", _kernel_breakdown_rows(rows)),
        ("raw_data", _raw_data_rows(rows)),
        *_workload_sheets(flat_rows),
    ]


def default_analysis_workbook_path(output_file: Optional[str]) -> Optional[str]:
    if not output_file:
        return None
    root, _ext = os.path.splitext(output_file)
    return f"{root}.analysis.xlsx"


def write_analysis_workbook(payload: dict[str, Any], path: str) -> None:
    _write_xlsx_workbook(path, _build_analysis_workbook_sheets(payload))


def write_final_report(
    *,
    args: Any,
    ctx: Any,
    rows: list[dict[str, Any]],
    world_size: int,
    repo_root: Path,
) -> None:
    """Produce the final report payload and write it to disk + workbook."""
    out_payload = build_report_payload(
        ctx=ctx,
        rows=rows,
        world_size=world_size,
        cuda_graph_default=bool(args.cuda_graph),
        repo_root=repo_root,
    )
    if args.output_file:
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp = args.output_file + ".final.tmp"
        with open(tmp, "w") as f:
            json.dump(out_payload, f, indent=2)
        os.replace(tmp, args.output_file)
        print(f"Report written to {args.output_file}", flush=True)
        workbook_file = getattr(
            args, "analysis_workbook_file", None
        ) or default_analysis_workbook_path(args.output_file)
        if workbook_file:
            write_analysis_workbook(out_payload, workbook_file)
            print(f"Analysis workbook written to {workbook_file}", flush=True)
    else:
        print(json.dumps({"rankings": out_payload["rankings"]}, indent=2), flush=True)
        workbook_file = getattr(args, "analysis_workbook_file", None)
        if workbook_file:
            write_analysis_workbook(out_payload, workbook_file)
            print(f"Analysis workbook written to {workbook_file}", flush=True)
