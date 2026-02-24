"""Accuracy test result parser."""

import glob
import os
import re
from typing import Dict, List

from utils.config_loader import AccuracyConfig, MetricsConfig
from utils.logger import logger

from reporting.accuracy_types import AccuracyValidationResult, DatasetValidation, RunValidation


class AccuracyParser:
    """Accuracy test parser (extracts results from accuracy_eval.log)."""

    def __init__(
        self, metrics_config: MetricsConfig, accuracy_config: AccuracyConfig, result_dir: str
    ):
        """Initialize AccuracyParser.

        Args:
            metrics_config: Metrics configuration for log file and regex pattern
            accuracy_config: Accuracy configuration with dataset thresholds
            result_dir: Directory containing test results
        """
        self.metrics_config = metrics_config
        self.accuracy_config = accuracy_config
        self.result_dir = result_dir

    def parse_and_validate(self) -> AccuracyValidationResult:
        """Parse accuracy_eval.log(s) and validate all configured datasets for all runs.

        Supports multiple runs (e.g., pre-benchmark and post-benchmark).
        Supports wildcard patterns in log_file (e.g., "7_accuracy_eval_*.log").
        All runs must pass for the validation to succeed.

        Returns:
            AccuracyValidationResult with validation results for all runs
        """
        log_pattern = self.metrics_config.log_file

        # Check if pattern contains wildcards
        if "*" in log_pattern or "?" in log_pattern or "[" in log_pattern:
            # Use glob to match multiple files
            log_files = sorted(glob.glob(os.path.join(self.result_dir, log_pattern)))

            if not log_files:
                return {
                    "success": False,
                    "all_passed": False,
                    "runs": [],
                    "raw_results": [],
                    "error": f"No log files found matching pattern: {log_pattern}",
                }

            logger.info(f"Found {len(log_files)} log file(s) matching pattern '{log_pattern}'")
        else:
            # Single file (backward compatible)
            log_file = os.path.join(self.result_dir, log_pattern)

            if not os.path.exists(log_file):
                return {
                    "success": False,
                    "all_passed": False,
                    "runs": [],
                    "raw_results": [],
                    "error": f"Log file not found: {log_file}",
                }

            log_files = [log_file]

        # Read and merge all log files
        combined_log_content = ""
        failed_files = []

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    combined_log_content += content
                    if not content.endswith("\n"):
                        combined_log_content += "\n"  # Ensure separation between files
                    logger.info(f"Successfully read log file: {os.path.basename(log_file)}")
            except Exception as e:
                failed_files.append((log_file, str(e)))
                logger.warning(f"Failed to read {log_file}: {e}")

        if not combined_log_content:
            error_msg = "No valid log content found."
            if failed_files:
                error_msg += f" Failed files: {failed_files}"
            return {
                "success": False,
                "all_passed": False,
                "runs": [],
                "raw_results": [],
                "error": error_msg,
            }

        # Extract accuracy values for all runs
        all_runs_results = self._extract_accuracy_values(combined_log_content)

        if not all_runs_results:
            return {
                "success": False,
                "all_passed": False,
                "runs": [],
                "raw_results": [],
                "error": "No accuracy values found in log",
            }

        logger.info(f"Found {len(all_runs_results)} accuracy test run(s)")

        # Validate each run
        runs_validation: List[RunValidation] = []
        all_runs_passed = True

        for run_idx, parsed_results in enumerate(all_runs_results):
            run_id = f"run-{run_idx + 1}"
            run_name = f"Run {run_idx + 1}"

            logger.info(f"Validating {run_name}: {parsed_results}")

            # Validate all datasets for this run
            validation_results: List[DatasetValidation] = []
            run_passed = True

            for dataset_config in self.accuracy_config.datasets:
                dataset_name = dataset_config.dataset_name.lower()

                # Check if dataset result was found in log
                if dataset_name not in parsed_results:
                    validation_results.append(
                        DatasetValidation(
                            dataset=dataset_config.dataset_name,
                            filter="",
                            passed=False,
                            actual=0.0,
                            expected=dataset_config.expected_value,
                            threshold=dataset_config.get_computed_threshold(),
                            threshold_type=dataset_config.threshold_type,
                            message="",
                            error=f"Dataset {dataset_config.dataset_name} not found in {run_name}",
                        )
                    )
                    run_passed = False
                    continue

                # Get results for this dataset's filter type
                filter_results = parsed_results[dataset_name]
                filter_type = dataset_config.filter_type

                if filter_type not in filter_results:
                    validation_results.append(
                        DatasetValidation(
                            dataset=dataset_config.dataset_name,
                            filter=filter_type,
                            passed=False,
                            actual=0.0,
                            expected=dataset_config.expected_value,
                            threshold=dataset_config.get_computed_threshold(),
                            threshold_type=dataset_config.threshold_type,
                            message="",
                            error=(
                                f"Filter '{filter_type}' not found for dataset "
                                f"{dataset_config.dataset_name} in {run_name}"
                            ),
                        )
                    )
                    run_passed = False
                    continue

                actual_value = filter_results[filter_type]
                passed, msg = dataset_config.validate(actual_value)

                validation_results.append(
                    DatasetValidation(
                        dataset=dataset_config.dataset_name,
                        filter=filter_type,
                        passed=passed,
                        actual=actual_value,
                        expected=dataset_config.expected_value,
                        threshold=dataset_config.get_computed_threshold(),
                        threshold_type=dataset_config.threshold_type,
                        message=msg,
                    )
                )

                if not passed:
                    run_passed = False

            runs_validation.append(
                RunValidation(
                    run_id=run_id,
                    run_name=run_name,
                    all_passed=run_passed,
                    results=validation_results,
                )
            )

            if not run_passed:
                all_runs_passed = False

        return {
            "success": True,
            "all_passed": all_runs_passed,
            "runs": runs_validation,
            "raw_results": all_runs_results,
        }

    def _extract_accuracy_values(self, log_content: str) -> List[Dict[str, Dict[str, float]]]:
        """Extract accuracy values from log content for multiple runs.

        Parses markdown table format from lm_eval output.
        Detects multiple runs by finding repeated datasets.

        Args:
            log_content: Content of the accuracy_eval.log file

        Returns:
            List of dictionaries, each representing one run:
            [
                {  # Run 1 (pre-benchmark)
                    'gsm8k': {
                        'flexible-extract': 0.9454,
                        'strict-match': 0.9431
                    },
                    'mmlu': { ... }
                },
                {  # Run 2 (post-benchmark)
                    'gsm8k': {
                        'flexible-extract': 0.9450,
                        'strict-match': 0.9428
                    },
                    'mmlu': { ... }
                }
            ]
        """
        all_runs = []
        current_run = {}

        # Regex to match table rows
        # Format: |dataset|version|filter|n-shot|metric|arrow|value|±|stderr|
        pattern = re.compile(self.metrics_config.extractor_pattern, re.IGNORECASE)

        matches = pattern.findall(log_content)

        for match in matches:
            dataset_name = match[0].strip().lower()
            filter_type = match[1].strip()
            accuracy_value = float(match[2].strip())

            # Skip table header rows (dataset might be "Tasks")
            if dataset_name in ["tasks", "task"]:
                continue

            # Only keep flexible-extract and strict-match filters
            if filter_type not in self.metrics_config.metric_names:
                continue

            # Check if this dataset already exists in current run
            if dataset_name in current_run and filter_type in current_run[dataset_name]:
                # This is a new run - save current run and start a new one
                if current_run:  # Only save if current_run has data
                    all_runs.append(current_run)
                    current_run = {}

            # Add to current run
            if dataset_name not in current_run:
                current_run[dataset_name] = {}

            current_run[dataset_name][filter_type] = accuracy_value

        # Don't forget to add the last run
        if current_run:
            all_runs.append(current_run)

        return all_runs
