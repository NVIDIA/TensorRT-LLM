from abstractions import (GptManagerBenchmarkExportFormat, TextSample,
                          TllmBenchExportFormat, Workload)
from export import export_workload

_expected_output_for_gpt_manager_benchmark = """{"metadata": {"num_requests": 100, "tokenize_vocabsize": 10000, "max_input_len": 100, "max_output_len": 100}, "samples": [{"input_len": 10, "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "output_len": 10, "task_id": 0}, {"input_len": 10, "input_ids": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "output_len": 10, "task_id": 1}]}"""


def test_export_for_gpt_manager_benchmark():
    workload = Workload(
        metadata={
            "num_requests": 100,
            "tokenize_vocabsize": 10000,
            "max_input_len": 100,
            "max_output_len": 100,
        },
        samples=[
            TextSample(
                input_len=10,
                input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                output_len=10,
                task_id=0,
            ),
            TextSample(
                input_len=10,
                input_ids=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                output_len=10,
                task_id=1,
            ),
        ],
    )
    file_path = "test_export_for_gpt_manager_benchmark.json"
    export_workload(workload,
                    GptManagerBenchmarkExportFormat(output_file_path=file_path))
    with open(file_path, "r") as f:
        assert f.read() == _expected_output_for_gpt_manager_benchmark


_expected_output_for_tllm_bench = """{"input_len": 10, "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "output_len": 10, "task_id": 0}
{"input_len": 10, "input_ids": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "output_len": 10, "task_id": 1}
"""


def test_export_for_tllm_bench():
    workload = Workload(
        metadata={
            "num_requests": 100,
            "tokenize_vocabsize": 10000,
            "max_input_len": 100,
            "max_output_len": 100,
        },
        samples=[
            TextSample(
                input_len=10,
                input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                output_len=10,
                task_id=0,
            ),
            TextSample(
                input_len=10,
                input_ids=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                output_len=10,
                task_id=1,
            ),
        ],
    )
    file_path = "test_export_for_tllm_bench.json"
    export_workload(workload, TllmBenchExportFormat(output_file_path=file_path))
    with open(file_path, "r") as f:
        assert f.read() == _expected_output_for_tllm_bench
