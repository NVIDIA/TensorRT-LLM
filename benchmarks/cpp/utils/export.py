import json
import os
from typing import Any, Dict

from utils.abstractions import (ExportFormat, GptManagerBenchmarkExportFormat,
                                TllmBenchExportFormat, Workload)


def _safe_create_parent_dir(file_path: str) -> None:
    real_path = os.path.realpath(file_path)
    parent_dir = os.path.dirname(real_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def export_workload_for_gpt_manager_benchmark(
    workload: Workload,
    gpt_manager_benchmark_export_format: GptManagerBenchmarkExportFormat,
) -> None:
    _safe_create_parent_dir(
        gpt_manager_benchmark_export_format.output_file_path)
    with open(gpt_manager_benchmark_export_format.output_file_path, "w") as f:
        json.dump(workload.model_dump(), f)


def export_workload_for_tllm_bench(
        workload: Workload,
        tllm_bench_export_format: TllmBenchExportFormat) -> None:
    _safe_create_parent_dir(tllm_bench_export_format.output_file_path)
    with open(tllm_bench_export_format.output_file_path, "w") as f:
        for sample in workload.samples:
            json.dump(sample.model_dump(), f)
            f.write("\n")


def export_workload(workload: Workload, export_format: ExportFormat) -> None:
    match export_format:
        case GptManagerBenchmarkExportFormat():
            export_workload_for_gpt_manager_benchmark(workload, export_format)
        case TllmBenchExportFormat():
            export_workload_for_tllm_bench(workload, export_format)
        case _:
            raise ValueError(f"Unsupported export format: {export_format}")


def print_to_stdout(workload: Workload) -> None:
    for sample in workload.samples:
        d = {
            "task_id": sample.task_id,
            "input_ids": sample.input_ids,
            "output_tokens": sample.output_len,
        }
        print(json.dumps(d, separators=(",", ":"), ensure_ascii=False))


def export_workload_from_args(root_args: Dict[str, Any], workload: Workload):
    match (root_args.std_out, root_args.export_format):
        case (True, _):
            print_to_stdout(workload)
        case (_, "gpt-manager-benchmark"):
            export_format = GptManagerBenchmarkExportFormat(
                output_file_path=root_args.output, )
            export_workload(workload, export_format)
        case (_, "tllm-bench"):
            export_format = TllmBenchExportFormat(
                output_file_path=root_args.output, )
            export_workload(workload, export_format)
        case _:
            raise ValueError(
                f"Unsupported export format: {root_args.export_format}")
