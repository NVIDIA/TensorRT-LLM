# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#!/usr/bin/env python3

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    output_o = []

    with open(args.input, 'r') as infile:
        for _l in infile:
            l = _l.strip()
            if len(l) == 0:
                continue
            o = json.loads(l)
            output_o.append({
                "input": o["prompt"],
                "instruction": "",
                "output": o["completion"]
            })

    with open(args.output, 'w') as outfile:
        json.dump(output_o, outfile)


if __name__ == "__main__":
    main()
