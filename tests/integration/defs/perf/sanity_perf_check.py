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

    def __call__(self, *args, **kwargs):
        # Check if the base_perf_csv file exists
        if not self.base_perf_csv.exists():
            print(
                f"{self.base_perf_csv.name} doesn't exist, skip check the perf result."
            )
            return 0

        base_perf = load_file(self.base_perf_csv.as_posix())
        current_perf = load_file(self.target_perf_csv.as_posix())

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
            print("Sanity perf check failed, but it has been disabled")
        return 0


if __name__ == '__main__':
    SanityPerfCheck(sys.argv[1], sys.argv[2])()
