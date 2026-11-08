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

metric_regex = {
    "rouge1": r"(?<=rouge1: )\d+\.\d+",
    "mmlu": r"(?<=MMLU weighted average accuracy: )\d+\.\d+",
    "gsm8k": r"(?<=gsm8k average accuracy: )\d+\.\d+",
    "gpqa_diamond":
    r"(?<=gpqa_diamond_cot_zeroshot_aa average accuracy: )\d+\.\d+",
    "perplexity": r"(?<=Per-token perplexity: )\d+\.\d+",
    "passkey": r"(?<=passkey accuracy: )\d+\.\d+"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    with open(args.log_file) as f:
        log = f.read()

    test_cases = re.search(r"(?<=items in this shard: ).+", log).group()
    test_cases = test_cases.split(", ")
    data = [{} for _ in test_cases]

    log = log.split("\n")
    i = -1
    for line in log:
        if i + 1 < len(test_cases) and line.startswith(test_cases[i + 1]):
            # Advance to next test case
            i += 1
            continue
        if i < 0:
            continue

        entry = data[i]
        for metric, regex in metric_regex.items():
            if metric in entry:
                continue
            matched = re.search(regex, line)
            if matched:
                entry[metric] = float(matched.group())

    df = pd.DataFrame(data, index=test_cases)
    print(df)

    if args.output_file:
        df.to_csv(args.output_file)
