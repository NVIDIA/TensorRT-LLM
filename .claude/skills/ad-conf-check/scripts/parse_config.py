#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Parse and flatten AutoDeploy YAML configs for agent-based log checking.

Usage:
    python3 parse_config.py <yaml_path> [<yaml_path2> ...] [--json-output <path>]

Accepts one or more YAML config files. When multiple YAMLs are provided,
they are deep-merged left-to-right: later files override earlier ones for
overlapping keys (like AutoDeploy's own layering of default + user configs).

Outputs a flat list of config key-value pairs as JSON — either to stdout
or to the file specified by --json-output.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates and returns *base*).

    For overlapping keys:
    - If both values are dicts, merge recursively.
    - Otherwise the override value wins.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def flatten_yaml(cfg: dict, prefix: str = "") -> List[Tuple[str, object]]:
    """Flatten YAML config into (dotted_key, value) pairs."""
    items = []
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_yaml(v, full_key))
        else:
            items.append((full_key, v))
    return items


def main():
    parser = argparse.ArgumentParser(description="Parse and flatten AutoDeploy YAML configs")
    parser.add_argument(
        "yaml_paths",
        nargs="+",
        help="One or more YAML config files (later files override earlier for overlapping keys)",
    )
    parser.add_argument(
        "--json-output",
        help="Path to write JSON output (default: print to stdout)",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    for yp in args.yaml_paths:
        if not Path(yp).exists():
            print(f"Error: YAML file not found: {yp}", file=sys.stderr)
            sys.exit(1)

    # Parse and merge
    cfg: dict = {}
    for yp in args.yaml_paths:
        with open(yp) as f:
            layer = yaml.safe_load(f) or {}
        deep_merge(cfg, layer)

    # Flatten
    flat = flatten_yaml(cfg)
    configs = [{"key": k, "value": str(v)} for k, v in flat]

    output = {
        "yaml_files": args.yaml_paths,
        "total_configs": len(configs),
        "configs": configs,
    }

    json_str = json.dumps(output, indent=2)
    if args.json_output:
        with open(args.json_output, "w") as f:
            f.write(json_str)
            f.write("\n")
        print(f"Parsed {len(configs)} configs -> {args.json_output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
