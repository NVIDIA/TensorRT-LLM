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
        self.metrics_config = metrics_config  # 保存 metrics 配置
        self.result_dir = result_dir

    def _extract_log(self, pattern: str, metric_names: List[str], log_content: str):
        """Extract log according to pattern and metrics names."""
        compiled = re.compile(pattern, re.MULTILINE | re.VERBOSE)
        results = []
        for match in compiled.finditer(log_content):
            logger.debug(f"Found match: {match.group(0)[:100]}...")
            logger.debug(f"All groups: {match.groups()}")
            logger.debug(f"Number of groups: {len(match.groups())}")

            if len(match.groups()) < 3:
                logger.warning(f"Expected 3 groups but got {len(match.groups())}")
                continue

            try:
                values = [float(x) for x in match.groups()[:-1]]
                concurrency = int(match.groups()[-1])  # Use groups()[-1] instead of group(-1)
                item = dict(zip(metric_names, values))
                item["concurrency"] = concurrency  # Concurrency used to make test names
                results.append(item)
                logger.debug(
                    f"Successfully extracted: E2EL={values[0]}, TTFT={values[1]}, concurrency={concurrency}"
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing match: {e}")
                continue
        return results

    def parse(
        self,
        model_name: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ):
        """Parse logs using configured metrics."""
        # Build log file path using metrics_config.log_file
        log_file_name = os.path.join(self.result_dir, self.metrics_config.log_file)

        if not os.path.exists(log_file_name):
            logger.error(f"Log file not found: {log_file_name}")
            return {"status": False, "df": None}

        with open(log_file_name, "r", encoding="utf-8", errors="replace") as log_file:
            log_content = log_file.read()

        # Use metrics_config for extraction
        raw_results = self._extract_log(
            self.metrics_config.extractor_pattern, self.metrics_config.metric_names, log_content
        )
        if len(raw_results) == 0:
            logger.warning("No metrics extracted from log file")
            return {"status": False, "df": None}

        # Convert to perf result format
        df = self._convert_to_perf_result_format(raw_results, model_name, timestamps, test_name)

        return {"status": True, "df": df}

    def _convert_to_perf_result_format(
        self,
        raw_results: List[dict],
        model_name: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ):
        """Convert raw results to perf result format.

        Each test result is expanded into multiple rows, one row per metric.

        Args:
            raw_results: Raw performance results
            model_name: Model name
            timestamps: Optional timestamps dict
            test_name: Optional pytest test name (e.g., "test_benchmark[deepseek-r1_1k1k_...]")
        """
        expanded_rows = []
        test_prefix = test_name
        # Use provided timestamps or fallback to current time
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

        # Get precision from YAML config metadata
        if isinstance(self.config, dict):
            precision = self.config.get("metadata", {}).get("precision", "unknown")
        else:
            # Fallback if config is not a dict (should not happen in current system)
            precision = "unknown"

        for item in raw_results:
            concurrency = item.get("concurrency", "1")
            base_test_name = f"{test_prefix}_con:{concurrency}"

            # Create a separate row for each performance metric
            for metric_name, metric_value in item.items():
                if metric_name == "concurrency":
                    continue

                # Create new row
                row = {
                    # Network related fields (use test_name)
                    "network_name": self._get_network_name(base_test_name),
                    "network_hash": base_test_name,
                    # Hardware related fields (leave empty)
                    "sm_clk": lock_freq_graphics,
                    "mem_clk": lock_freq_memory,
                    "gpu_idx": np.nan,
                    # Test related fields
                    "perf_case_name": base_test_name,
                    "test_name": base_test_name,
                    "original_test_name": base_test_name,
                    # Performance metrics
                    "perf_metric": float(metric_value),
                    "metric_type": metric_name,
                    # Time related fields - use actual timestamps from TestCaseTracker
                    "total_time__sec": total_time,
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    # State and configuration
                    "state": "valid",
                    "command": f"disagg_benchmark --model={model_name} --{precision} --concurrency={concurrency}",
                    # Threshold related fields
                    "threshold": np.nan,
                    "absolute_threshold": np.nan,
                }

                expanded_rows.append(row)

        # Create DataFrame and ensure column order
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

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Rearrange column order
        df = df[expected_columns]

        return df

    def _get_network_name(self, base_test_name: str):
        """Extract network name from test name.

        Input format:
            test_disagg_simple.py::TestDisaggBenchmark::test_benchmark[deepseek-r1_1k1k_...]-con-1
        Output format:
            deepseek-r1_1k1k_...-con-1
        """
        # Pattern to extract content inside brackets and the trailing -con-X
        # Group 1: content inside []
        # Group 2: -con-X suffix
        pattern = r"\[([^\]]+)\](-con-\d+)"
        match = re.search(pattern, base_test_name)

        if match:
            # Combine the bracket content with -con-X suffix
            return f"{match.group(1)}{match.group(2)}"
        else:
            # Fallback: if pattern doesn't match, use original logic
            return base_test_name.replace("/", "-")


class ResultSaver(object):
    """All of the benchmarks append to the same csv, add header to it each time.

    No matter whether the columns are of the same count.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path

    def append_a_df(self, df: pd.DataFrame):
        """Seamlessly append DataFrame to CSV without headers or extra line breaks.

        Ideal for unified format data where consistency is maintained across appends.
        """
        # Check if file exists and has content
        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0

        if file_exists:
            # File exists, append data only (no header)
            df.to_csv(self.output_path, mode="a", index=False, header=False)
            logger.success(f"Seamlessly appended {len(df)} rows to {self.output_path}")
        else:
            # First write, include header
            df.to_csv(self.output_path, mode="w", index=False, header=True)
            logger.success(f"Created new file with {len(df)} rows: {self.output_path}")

    def save_all(self, results: List[Tuple[pd.DataFrame, str]]):
        """Save in batch manner: Append each dataframe with header.

        The 2nd parameter can print to logs.
        ex: [(df1, '1k1k'), (df2, '8k1k')]
        """
        for df, btype in results:
            logger.info(f"Writing benchmark type: {btype}")
            self.append_a_df(df)
