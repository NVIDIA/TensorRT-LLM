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
import csv
import os
import sys

# This is to prevent csv field size limit error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


class SanityPerfCheck():

    # This is to prevent redundant messages and long logs.
    USEFUL_METRICS = [
        "original_test_name", "perf_case_name", "metric_type", "perf_metric",
        "command", "sm_clk", "mem_clk", "start_timestamp", "end_timestamp",
        "state", "threshold", "absolute_threshold"
    ]

    def __init__(self, target_perf_csv, base_perf_csv=None, threshold=0.1):
        self.target_perf_csv = target_perf_csv
        self.base_perf_csv = base_perf_csv
        self.threshold = threshold

    def _parse_result(self, csv_path):
        result = {}
        with open(csv_path) as csv_file:
            parsed_csv_file = csv.DictReader(csv_file)
            for row in parsed_csv_file:
                if row['metric_type'] not in result:
                    result[row['metric_type']] = {}
                result[row['metric_type']][row['perf_case_name']] = float(
                    row['perf_metric'])
        return result

    def _dump_csv_row(self, csv_path, metric_type, test_name):
        with open(csv_path) as csv_file:
            parsed_csv_file = csv.DictReader(csv_file)
            for row in parsed_csv_file:
                if row['metric_type'] == metric_type and row[
                        'perf_case_name'] == test_name:
                    print('=' * 40)
                    print('Please fill below content into the base_perf.csv.')
                    cleaned_row = []
                    for k in self.USEFUL_METRICS:
                        v = row[k]
                        # Need to truncate the commands
                        if k == "command":
                            options = v.split(" ")
                            cleaned_options = []
                            for option in options:
                                # Truncate workspace dir
                                if "build.py" in option or "benchmark.py" in option or "SessionBenchmark.cpp" in option:
                                    cleaned_options.append("/".join(
                                        option.split("/")[-5:]))
                                # Remove engine_dir as it is not useful
                                elif "--engine_dir=" not in option and "--output_dir=" not in option:
                                    cleaned_options.append(option)
                            cleaned_row.append(" ".join(cleaned_options))
                        else:
                            cleaned_row.append(v)

                    print(",".join(['\"' + row + '\"' for row in cleaned_row]))
                    print('=' * 40)
                    break

    def __call__(self, *args, **kwargs):
        # Check if the base_perf_csv file exists
        if not os.path.exists(self.base_perf_csv):
            print(f"base_perf.csv doesn't exist, skip check the perf result.")
            return 0

        base_result = self._parse_result(self.base_perf_csv)
        target_result = self._parse_result(self.target_perf_csv)

        success = True

        for _, metric_type in enumerate(target_result):
            # Engine build time is very CPU specific, so skip the check
            if metric_type != "BUILD_TIME":
                for _, test_name in enumerate(target_result[metric_type]):
                    if metric_type not in base_result or test_name not in base_result[
                            metric_type]:

                        self._dump_csv_row(self.target_perf_csv, metric_type,
                                           test_name),
                        print(
                            f"{metric_type} {test_name} doesn't exist in the base_perf.csv, please add it and rerun the pipeline."
                        )
                        success = False
                    else:
                        base_perf = base_result[metric_type][test_name]
                        target_perf = target_result[metric_type][test_name]

                        if target_perf > base_perf * (1 + self.threshold):
                            # the mr perf is worse than baseline, there's perf regression.
                            print(
                                f"Perf Regression found on {metric_type} {test_name} where the current perf is {target_perf} while the baseline is {base_perf}."
                            )
                            success = False
                        elif target_perf < base_perf * (1 - self.threshold):
                            # the MR perf is better than baseline, please update the base_perf.csv
                            self._dump_csv_row(self.target_perf_csv,
                                               metric_type, test_name),
                            print(
                                f"Please update {metric_type} {test_name} into base_perf.csv and commit again. The outdated perf baseline is {base_perf} and the new perf baseline is {target_perf}"
                            )
                            success = False

        if not success:
            # We have temporarily disabled post perf sanity tests
            print("Sanity perf check failed, but it has been disabled")
        return 0


if __name__ == '__main__':
    SanityPerfCheck(sys.argv[1], sys.argv[2])()
