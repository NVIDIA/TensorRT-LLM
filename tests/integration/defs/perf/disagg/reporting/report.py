import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.common import GPU_RESOURCE_CONFIG, EnvManager
from utils.logger import logger


class LogWriter(object):
    def __init__(self, log_path: str):
        self.log_path = log_path

    def print_to_console(self, file_name):
        log_file_name = os.path.join(self.log_path, file_name)
        logger.info(f"Log file: {log_file_name}")
        try:
            with open(log_file_name, "r", encoding="utf-8", errors="replace") as log_file:
                for line in log_file:
                    logger.debug(line.rstrip("\n"))
        except FileNotFoundError:
            logger.error(f"File not found: {log_file_name}")
        except PermissionError:
            logger.error(f"Permission denied: {log_file_name}")
        except Exception as e:
            logger.error(f"Error reading file: {e}")


class LogParser(object):
    def __init__(self, benchmark_type: str, config, metrics_config, result_dir: str):
        """Log parser with metrics config support.

        Args:
            benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
            config: Configuration (BenchmarkConfig object or dict)
            metrics_config: MetricsConfig object (default or custom)
            result_dir: Result directory
        """
        self.benchmark_type = benchmark_type
        self.config = config
        self.metrics_config = metrics_config
        self.result_dir = result_dir

    def _extract_log(self, pattern: str, metric_names: List[str], log_content: str):
        """Extract log according to pattern and metrics names."""
        compiled = re.compile(pattern, re.MULTILINE | re.VERBOSE)
        results = []
        for match in compiled.finditer(log_content):
            logger.debug(f"Found match: {match.group(0)[:100]}...")
            logger.debug(f"All groups: {match.groups()}")

            if len(match.groups()) < 3:
                logger.warning(f"Expected at least 3 groups but got {len(match.groups())}")
                continue

            try:
                values = [float(x) for x in match.groups()[:-1]]
                concurrency = int(match.groups()[-1])
                item = dict(zip(metric_names, values))
                item["concurrency"] = concurrency
                results.append(item)
                logger.debug(
                    f"Extracted: concurrency={concurrency}, {dict(zip(metric_names, values))}"
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing match: {e}")
                continue
        return results

    def _extract_request_counts_from_log(self, log_content: str) -> Tuple[int, int]:
        """Extract failed/total from log via regex (TRT-LLM benchmark_serving.py format).

        Sums all matches to handle multi-concurrency logs correctly.
        """
        failed_requests = 0
        total_requests = 0
        # Match "Failed requests:" (capital F) from summary block, not
        # "Total failed requests:" (lowercase f) which can report 0 incorrectly
        failed_matches = re.findall(r"Failed requests:\s+(\d+)", log_content)
        total_matches = re.findall(r"Total requests:\s+(\d+)", log_content)
        if failed_matches:
            failed_requests = sum(int(x) for x in failed_matches)
        if total_matches:
            total_requests = sum(int(x) for x in total_matches)
        return failed_requests, total_requests

    def _extract_request_counts_from_json(self, concurrencies: List[int]) -> Tuple[int, int]:
        """Extract failed/total from result.json files (bench_serving format).

        Used when use_nv_sa_benchmark is true, since bench_serving logs do not
        contain "Total requests" / "Total failed requests" fields.
        """
        total_requests = 0
        failed_requests = 0
        for concurrency in concurrencies:
            result_json_path = os.path.join(
                self.result_dir, f"concurrency_{concurrency}", "result.json"
            )
            if not os.path.exists(result_json_path):
                logger.warning(f"result.json not found: {result_json_path}")
                continue
            try:
                with open(result_json_path, "r") as f:
                    data = json.load(f)
                num_prompts = data.get("num_prompts", 0)
                completed = data.get("completed", 0)
                total_requests += num_prompts
                failed_requests += num_prompts - completed
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading result.json: {result_json_path}, {e}")
                continue
        return failed_requests, total_requests

    def parse(
        self,
        model_name: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ):
        """Parse logs using configured metrics."""
        log_file_name = os.path.join(self.result_dir, self.metrics_config.log_file)

        if not os.path.exists(log_file_name):
            logger.error(f"Log file not found: {log_file_name}")
            return {"status": False, "df": None, "failed_requests": 0, "total_requests": 0}

        with open(log_file_name, "r", encoding="utf-8", errors="replace") as log_file:
            log_content = log_file.read()

        raw_results = self._extract_log(
            self.metrics_config.extractor_pattern, self.metrics_config.metric_names, log_content
        )

        # Determine request count extraction strategy based on benchmark backend
        use_nv_sa = False
        if isinstance(self.config, dict):
            use_nv_sa = self.config.get("benchmark", {}).get("use_nv_sa_benchmark", False)

        if use_nv_sa:
            concurrencies = [item.get("concurrency", 0) for item in raw_results]
            failed_requests, total_requests = self._extract_request_counts_from_json(concurrencies)
        else:
            failed_requests, total_requests = self._extract_request_counts_from_log(log_content)

        if len(raw_results) == 0:
            logger.warning("No metrics extracted from log file")
            return {
                "status": False,
                "df": None,
                "failed_requests": failed_requests,
                "total_requests": total_requests,
            }

        df = self._convert_to_perf_result_format(raw_results, model_name, timestamps, test_name)

        return {
            "status": True,
            "df": df,
            "failed_requests": failed_requests,
            "total_requests": total_requests,
        }

    def _convert_to_perf_result_format(
        self,
        raw_results: List[dict],
        model_name: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ):
        """Convert raw results to perf result format (one row per metric)."""
        expanded_rows = []

        if timestamps:
            start_time = timestamps.get(
                "start_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            end_time = timestamps.get("end_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            total_time = timestamps.get("total_time__sec", np.nan)
        else:
            current_time = datetime.now()
            start_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            total_time = np.nan

        gpu_type = EnvManager.get_gpu_type()
        gpu_config = GPU_RESOURCE_CONFIG[gpu_type]
        lock_freq_graphics = gpu_config.get("lock_freq_graphics_mhz", 0) or 0
        lock_freq_memory = gpu_config.get("lock_freq_memory_mhz", 0) or 0

        if isinstance(self.config, dict):
            precision = self.config.get("metadata", {}).get("precision", "unknown")
        else:
            precision = "unknown"

        for item in raw_results:
            concurrency = item.get("concurrency", "1")
            base_test_name = f"{test_name}_con:{concurrency}"

            for metric_name, metric_value in item.items():
                if metric_name == "concurrency":
                    continue

                row = {
                    "network_name": self._get_network_name(base_test_name),
                    "network_hash": base_test_name,
                    "sm_clk": lock_freq_graphics,
                    "mem_clk": lock_freq_memory,
                    "gpu_idx": np.nan,
                    "perf_case_name": base_test_name,
                    "test_name": base_test_name,
                    "original_test_name": base_test_name,
                    "perf_metric": float(metric_value),
                    "metric_type": metric_name,
                    "total_time__sec": total_time,
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "state": "valid",
                    "command": f"disagg_benchmark --model={model_name} --{precision} --concurrency={concurrency}",
                    "threshold": np.nan,
                    "absolute_threshold": np.nan,
                }
                expanded_rows.append(row)

        expected_columns = [
            "network_name",
            "network_hash",
            "sm_clk",
            "mem_clk",
            "gpu_idx",
            "perf_case_name",
            "test_name",
            "original_test_name",
            "perf_metric",
            "total_time__sec",
            "start_timestamp",
            "end_timestamp",
            "state",
            "command",
            "threshold",
            "absolute_threshold",
            "metric_type",
        ]

        df = pd.DataFrame(expanded_rows)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_columns]

        return df

    def _get_network_name(self, base_test_name: str):
        """Extract network name from test name.

        e.g. "...::test_benchmark[deepseek-r1_1k1k_...]-con-1" -> "deepseek-r1_1k1k_...-con-1"
        """
        match = re.search(r"\[([^\]]+)\](-con-\d+)", base_test_name)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return base_test_name.replace("/", "-")


class ResultSaver(object):
    """Append benchmark results to a shared CSV file."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def append_a_df(self, df: pd.DataFrame):
        """Append DataFrame to CSV, writing header only on first write."""
        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0

        if file_exists:
            df.to_csv(self.output_path, mode="a", index=False, header=False)
            logger.success(f"Appended {len(df)} rows to {self.output_path}")
        else:
            df.to_csv(self.output_path, mode="w", index=False, header=True)
            logger.success(f"Created new file with {len(df)} rows: {self.output_path}")

    def save_all(self, results: List[Tuple[pd.DataFrame, str]]):
        """Append each (DataFrame, benchmark_type) pair to CSV."""
        for df, btype in results:
            logger.info(f"Writing benchmark type: {btype}")
            self.append_a_df(df)
