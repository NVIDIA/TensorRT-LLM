# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import re

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    with open(args.log_file) as f:
        log = f.read()

    test_cases = re.search(r"(?<=items in this shard: ).+", log).group()
    test_cases = test_cases.split(", ")
    test_case_to_score = {case: None for case in test_cases}

    log = log.split("\n")
    i = -1
    for line in log:
        if i + 1 < len(test_cases) and line.startswith(test_cases[i + 1]):
            i += 1
            continue
        if test_case_to_score[test_cases[i]] is not None:
            continue
        matched = re.search(r"(?<=rouge1 : )\d+\.\d+", line)
        if matched:
            test_case_to_score[test_cases[i]] = float(matched.group())
            if i >= len(test_cases):
                break

    test_case_to_score = pd.Series(test_case_to_score)
    print(test_case_to_score)

    if args.output_file:
        test_case_to_score.to_csv(args.output_file)
