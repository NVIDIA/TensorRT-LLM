# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Build the perf client's pinned wheel without installing it into the image."""

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path


def cache_wheel(client_file: Path, output_dir: Path) -> None:
    """Record a wheel and its source pin for offline installation by the benchmark client."""
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        print(
            "Skipping benchmark client wheel cache: the pinned fork requires Python 3.11 through 3.13."
        )
        return
    source_ref = runpy.run_path(str(client_file))["DEFAULT_AIPERF_REF"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Benchmark client wheel cache output directory must be empty: {output_dir}"
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output_dir),
            f"aiperf @ {source_ref}",
        ],
        check=True,
    )
    wheels = list(output_dir.glob("aiperf-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one aiperf wheel in {output_dir}, found {len(wheels)}")
    wheel = wheels[0]
    record = {
        "source_ref": source_ref,
        "wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache_wheel(args.client_file, args.output_dir)


if __name__ == "__main__":
    main()
