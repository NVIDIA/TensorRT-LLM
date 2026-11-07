# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge layered settings files into one effective configuration.

The layers in LAYERS are applied in that order, and a layer whose file is missing is skipped.
Inside a layer, ``[name]`` starts a section, a line starting with ``#`` is a comment, and every
other non-empty line is one of:

* ``key = value``: sets the key, replacing any earlier value;
* ``key += value``: appends value to the key's comma-separated list, or sets it if the key is
  not set yet;
* ``key = !unset``: removes the key, whatever earlier layers set.

Run ``python3 loader.py [settings_dir]`` to print the merged settings.
"""

import os
import sys

LAYERS = ["defaults.ini", "site.ini", "local.ini", "override.ini"]
UNSET = "!unset"


def parse_layer(text):
    """Return a layer's lines as (section, key, op, value) tuples, op being "=" or "+="."""
    operations = []
    section = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section is None:
            raise ValueError(f"line {number}: a setting outside any section")
        if "+=" in line:
            key, value = line.split("+=", 1)
            op = "+="
        elif "=" in line:
            key, value = line.split("=", 1)
            op = "="
        else:
            raise ValueError(f"line {number}: expected key = value")
        operations.append((section, key.strip(), op, value.strip()))
    return operations


def apply_layer(merged, operations):
    """Apply one layer's operations to merged, a {section: {key: value}} dict."""
    for section, key, op, value in operations:
        settings = merged.setdefault(section, {})
        if op == "=" and value == UNSET:
            settings.pop(key, None)
        elif op == "+=" and settings.get(key):
            settings[key] = f"{settings[key]},{value}"
        else:
            settings[key] = value
    return merged


def load(directory):
    """Merge the layers found in directory, in the order of LAYERS."""
    merged = {}
    for name in LAYERS:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            apply_layer(merged, parse_layer(fh.read()))
    return merged


def main(argv):
    directory = argv[1] if len(argv) > 1 else "settings"
    for section, settings in load(directory).items():
        for key, value in settings.items():
            print(f"{section}.{key} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
