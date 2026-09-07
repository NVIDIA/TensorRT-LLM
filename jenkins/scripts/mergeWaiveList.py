# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import sys

# Generate the merged waive list:
# 1. Parse the current MR waive list, and get the removed lines from the diff
# 2. Parse the TOT waive list
# 3. Merge the current MR waive list and TOT waive list, and remove the removed lines from the step 1


def get_remove_lines_from_diff_file(diff_file):
    with open(diff_file, 'r') as f:
        diff = f.read()
    lines = diff.split('\n')
    remove_lines = [
        line[1:] + '\n' for line in lines
        if len(line) > 1 and line.startswith('-')
    ]
    return remove_lines


def parse_waive_txt(waive_txt):
    with open(waive_txt, 'r') as f:
        lines = f.readlines()
    waive_list = [line for line in lines if line.strip()]
    return waive_list


def write_waive_list(waive_list, output_file):
    with open(output_file, 'w') as f:
        for line in waive_list:
            f.write(line)


def merge_waive_list(cur_list, main_list, remove_lines, output_file):
    merged = list(dict.fromkeys(cur_list + main_list))
    for line in reversed(remove_lines):
        for i in range(len(merged) - 1, -1, -1):
            if merged[i] == line:
                merged.pop(i)
                break
    write_waive_list(merged, output_file)


def parse_maintenance_config(content, source):
    """Parse maintenance entries keyed by stage name or pattern."""
    entries = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        fields = [field.strip() for field in line.split('|', 1)]
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(
                f"Invalid maintenance entry at {source}:{line_number}; "
                "expected '<stage-or-pattern> | <reason>'.")
        pattern, reason = fields
        if pattern in entries:
            print(
                f"WARNING: Duplicate maintenance pattern '{pattern}' at "
                f"{source}:{line_number}; the first entry is used.",
                file=sys.stderr)
            continue
        entries[pattern] = {'pattern': pattern, 'reason': reason}
    return entries


def merge_maintenance_config(cur_config, main_config, diff_file, output_file):
    """Apply only the PR's maintenance additions and deletions to target TOT."""
    with open(diff_file, 'r', encoding='utf-8') as f:
        diff = f.read()

    if not os.path.isfile(cur_config) and diff.strip():
        raise ValueError(
            'Deleting or renaming the maintenance config is not allowed.')

    with open(main_config, 'r', encoding='utf-8') as f:
        effective = parse_maintenance_config(f.read(), 'target TOT')
    if os.path.isfile(cur_config):
        with open(cur_config, 'r', encoding='utf-8') as f:
            current = parse_maintenance_config(f.read(), 'PR file')
    else:
        current = {}

    addition_lines = [
        line[1:] for line in diff.splitlines()
        if line.startswith('+') and not line.startswith('+++')
    ]
    deletion_lines = [
        line[1:] for line in diff.splitlines()
        if line.startswith('-') and not line.startswith('---')
    ]
    additions = parse_maintenance_config('\n'.join(addition_lines),
                                         'PR additions')
    deletions = parse_maintenance_config('\n'.join(deletion_lines),
                                         'PR deletions')

    for pattern in deletions.keys() - additions.keys():
        effective.pop(pattern, None)
    for pattern in additions:
        if pattern not in current:
            raise ValueError(
                f"Maintenance pattern '{pattern}' was added in the diff but "
                'is missing from the PR file.')
        effective[pattern] = current[pattern]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(effective.values()), f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maintenance-config', action='store_true')
    parser.add_argument('--cur-waive-list',
                        required=True,
                        help='Current waive list')
    parser.add_argument('--latest-waive-list',
                        required=True,
                        help='Latest waive list')
    parser.add_argument('--diff-file',
                        required=True,
                        help='File containing diff of the waive list')
    parser.add_argument('--output-file', required=True, help='Output file')
    args = parser.parse_args(sys.argv[1:])

    if args.maintenance_config:
        try:
            merge_maintenance_config(args.cur_waive_list,
                                     args.latest_waive_list, args.diff_file,
                                     args.output_file)
        except ValueError as error:
            parser.error(str(error))
    else:
        cur_list = parse_waive_txt(args.cur_waive_list)
        main_list = parse_waive_txt(args.latest_waive_list)
        remove_lines = get_remove_lines_from_diff_file(args.diff_file)
        merge_waive_list(cur_list, main_list, remove_lines, args.output_file)
