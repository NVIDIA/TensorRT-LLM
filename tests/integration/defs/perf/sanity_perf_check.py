# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import difflib
import sys
from pathlib import Path

import pandas as pd
from diff_tools import get_csv_lines, get_diff, load_file


class SanityPerfCheck():

    def __init__(self, target_perf_csv, base_perf_csv=None, threshold=0.1):
        self.target_perf_csv = Path(target_perf_csv)
        self.base_perf_csv = Path(base_perf_csv)
        self.threshold = threshold

    def report_diff(self, full_diff: pd.DataFrame) -> None:
        print("=" * 40)
        for diff in full_diff.itertuples():
            if pd.isna(diff.perf_metric_base):
                print(f"perf_case_name: {diff.Index} is missing from base")
            elif pd.isna(diff.perf_metric_target):
                print(f"perf_case_name: {diff.Index} is missing from target")
            else:
                print(
                    f"perf_case_name: {diff.Index}, base->target: {diff.perf_metric_base}->{diff.perf_metric_target}"
                )
        print("=" * 40)

    def write_patch(self, old_lines: list[str], new_lines: list[str],
                    output_path: str, base_perf_filename: str) -> None:
        with open(output_path, 'w') as f:
            for diff_line in difflib.unified_diff(
                    old_lines, new_lines,
                    f'a/tests/integration/defs/perf/{base_perf_filename}',
                    f'b/tests/integration/defs/perf/{base_perf_filename}'):
                f.write(diff_line)

    def _check_autodeploy_failures(self, full_diff: pd.DataFrame,
                                   base_perf: pd.DataFrame,
                                   current_perf: pd.DataFrame) -> bool:
        """
        Check if any of the performance regressions are from autodeploy tests.
        Only considers actual regressions (worse performance), not improvements.
        Returns True if there are autodeploy regressions, False otherwise.
        """
        # Create mappings for network_name, threshold, absolute_threshold, and metric_type
        base_network_mapping = dict(
            zip(base_perf['perf_case_name'], base_perf['network_name']))
        current_network_mapping = dict(
            zip(current_perf['perf_case_name'], current_perf['network_name']))
        base_threshold_mapping = dict(
            zip(base_perf['perf_case_name'], base_perf['threshold']))
        base_abs_threshold_mapping = dict(
            zip(base_perf['perf_case_name'], base_perf['absolute_threshold']))
        base_metric_type_mapping = dict(
            zip(base_perf['perf_case_name'], base_perf['metric_type']))

        # Check each performance difference
        for idx, row in full_diff.iterrows():
            # Look up network_name from either base or current (they should be the same)
            network_name = base_network_mapping.get(
                idx) or current_network_mapping.get(idx)

            # Only check autodeploy tests
            if network_name and "_autodeploy" in str(network_name):
                # Check if this is actually a regression (worse performance)
                if hasattr(row, 'perf_metric_base') and hasattr(
                        row, 'perf_metric_target'):
                    base_value = row.perf_metric_base
                    target_value = row.perf_metric_target
                    threshold = base_threshold_mapping.get(idx, 0)
                    abs_threshold = base_abs_threshold_mapping.get(idx, 50)
                    metric_type = base_metric_type_mapping.get(idx, '')

                    # Skip if we don't have the necessary data
                    if pd.isna(base_value) or pd.isna(target_value):
                        continue

                    # Determine if this is a regression based on metric type and threshold sign
                    is_regression = self._is_performance_regression(
                        base_value, target_value, threshold, abs_threshold,
                        metric_type)

                    if is_regression:
                        return True

        return False

    def _is_performance_regression(self, base_value: float, target_value: float,
                                   threshold: float, abs_threshold: float,
                                   metric_type: str) -> bool:
        """
        Determine if a performance change represents a regression (worse performance)
        that exceeds the acceptable threshold.

        Args:
            base_value: Baseline performance value
            target_value: Current performance value
            threshold: Performance threshold (sign indicates better direction)
            abs_threshold: Absolute threshold for tolerance calculation
            metric_type: Type of metric (for context)

        Returns:
            True if target_value represents worse performance than base_value
            AND the change exceeds the threshold
        """
        import numpy as np

        # First check if the change exceeds the threshold (same logic as diff_tools.py)
        # Use absolute value of threshold for relative tolerance calculation
        rel_threshold = abs(threshold)

        # If values are within threshold tolerance, no significant change
        if np.isclose(base_value,
                      target_value,
                      rtol=rel_threshold,
                      atol=abs(abs_threshold)):
            return False

        # Now check if it's a regression (worse performance) in the expected direction
        if threshold > 0:
            # Positive threshold: lower is better - regression if target > base
            return target_value > base_value
        else:
            # Negative threshold: higher is better - regression if target < base
            return target_value < base_value

    def _filter_by_device_subtype(
            self, base_perf: pd.DataFrame,
            current_perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter performance data to match device subtypes for autodeploy tests.

        For autodeploy tests, only compare against baselines with the same device subtype.
        For non-autodeploy tests, use the original behavior.

        Args:
            base_perf: Baseline performance DataFrame
            current_perf: Current performance DataFrame

        Returns:
            Tuple of (filtered_base_perf, filtered_current_perf)
        """
        # If current performance data doesn't have device_subtype column, return as-is
        if 'device_subtype' not in current_perf.columns:
            return base_perf, current_perf

        # Get the current device subtype from current performance data
        current_device_subtypes = current_perf['device_subtype'].dropna(
        ).unique()

        if len(current_device_subtypes) == 0:
            # No device subtype info in current data, return as-is
            return base_perf, current_perf

        current_device_subtype = current_device_subtypes[
            0]  # Assume single device type per run
        print(
            f"Filtering performance data for device subtype: {current_device_subtype}"
        )

        # Filter base performance data to only include entries with matching device subtype
        # or entries without device subtype info (for backward compatibility)
        if 'device_subtype' in base_perf.columns:
            # Filter base data: keep entries with matching subtype or null subtype
            base_filtered = base_perf[
                (base_perf['device_subtype'] == current_device_subtype) |
                (base_perf['device_subtype'].isna())].copy()
        else:
            # Base data doesn't have device subtype column, keep all entries
            base_filtered = base_perf.copy()

        # For autodeploy tests, only keep current entries with device subtype
        autodeploy_mask = current_perf['network_name'].str.contains(
            '_autodeploy', na=False)
        current_filtered = current_perf.copy()

        # For autodeploy tests, ensure device subtype is present
        if autodeploy_mask.any():
            autodeploy_entries = current_perf[autodeploy_mask]
            non_autodeploy_entries = current_perf[~autodeploy_mask]

            # Keep only autodeploy entries that have device subtype
            autodeploy_with_subtype = autodeploy_entries[
                autodeploy_entries['device_subtype'].notna()]

            # Combine filtered autodeploy entries with non-autodeploy entries
            current_filtered = pd.concat(
                [autodeploy_with_subtype, non_autodeploy_entries],
                ignore_index=True)

        return base_filtered, current_filtered

    def __call__(self, *args, **kwargs):
        # Check if the base_perf_csv file exists
        if not self.base_perf_csv.exists():
            print(
                f"{self.base_perf_csv.name} doesn't exist, skip check the perf result."
            )
            return 0

        base_perf = load_file(self.base_perf_csv.as_posix())
        current_perf = load_file(self.target_perf_csv.as_posix())

        # Filter performance data by device subtype for autodeploy tests
        base_perf, current_perf = self._filter_by_device_subtype(
            base_perf, current_perf)

        full_diff, new_base = get_diff(base_perf, current_perf)
        if not full_diff.empty:
            self.report_diff(full_diff)
            output_patch = self.target_perf_csv.with_name(
                'perf_patch.patch').as_posix()
            self.write_patch(get_csv_lines(base_perf), get_csv_lines(new_base),
                             output_patch, self.base_perf_csv.name)
            print(f"patch_file was written to {output_patch}")
            print(
                "You can download the file and update base_perf.csv by `git apply <patch file>`"
            )

            # Check if any of the failed tests are autodeploy tests
            autodeploy_failures = self._check_autodeploy_failures(
                full_diff, base_perf, current_perf)

            if autodeploy_failures:
                print(
                    "Sanity perf check failed for autodeploy tests - failing the build"
                )
                return 1
            else:
                print(
                    "Sanity perf check failed, but it has been disabled for non-autodeploy tests"
                )
        return 0


if __name__ == '__main__':
    exit_code = SanityPerfCheck(sys.argv[1], sys.argv[2])()
    sys.exit(exit_code)
