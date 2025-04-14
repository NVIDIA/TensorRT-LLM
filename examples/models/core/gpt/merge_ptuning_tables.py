#! /usr/bin/env python3
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
import argparse
from pathlib import Path
from typing import List

import numpy as np


def combine_tables(input_tables: List[Path], output_table: Path):
    tables = [np.load(table) for table in input_tables]
    max_vocab_size = max(table.shape[1] for table in tables)
    padded_tables = [
        np.pad(table, [(0, 0), (0, max_vocab_size - table.shape[1]), (0, 0)])
        for table in tables
    ]
    merged_table = np.concatenate(padded_tables, axis=0)
    np.save(output_table, merged_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tables",
                        type=Path,
                        nargs="+",
                        help="paths to tables to merge.")
    parser.add_argument("output_table",
                        type=Path,
                        help="path where to save the combined table")
    args = parser.parse_args()
    combine_tables(args.input_tables, args.output_table)
