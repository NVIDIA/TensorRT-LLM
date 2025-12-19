#!/usr/bin/env python3
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
import sys

import yaml


def sort_entries(data: list) -> list:
    """Sort entries by fields in the order they appear in the first entry."""
    if not data:
        return data

    fields = list(data[0].keys())

    def sort_key(entry):
        return tuple(entry.get(field, "") for field in fields)

    return sorted(data, key=sort_key)


def main():
    parser = argparse.ArgumentParser(description="Sort a YAML file by fields in order")
    parser.add_argument("file", help="Path to the database lookup.yaml file to sort")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = yaml.safe_load(f)

    sorted_data = sort_entries(data)

    if data != sorted_data:
        with open(args.file, "w") as f:
            yaml.dump(sorted_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Sorted {args.file}")

        return True  # file was modified

    return False  # file was not modified


if __name__ == "__main__":
    sys.exit(main())
