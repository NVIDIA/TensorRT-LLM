# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the experimental host driver from a clean, exact Snapshot checkout."""

import argparse
import subprocess
from pathlib import Path

from qualify import HERE, SNAPSHOT_REVISION, inventory, write_json


def main() -> None:
    """Verify source identity, compile against it, and record immutable inputs.

    Raises:
        ValueError: If the source checkout is dirty or at another revision.
        subprocess.CalledProcessError: If compilation or unit tests fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-source", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    source = args.snapshot_source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if head != SNAPSHOT_REVISION or dirty:
        raise ValueError("Build requires a clean checkout of the exact Snapshot pin")
    args.evidence.mkdir(parents=True, exist_ok=False)
    write_json(
        args.evidence / "inputs.json",
        {"snapshot_revision": head, "driver": inventory(HERE / "host_driver")},
    )
    with (args.evidence / "build.log").open("w") as log:
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--load",
                "--progress",
                "plain",
                "--build-context",
                f"snapshot={source}",
                "--tag",
                args.tag,
                str(HERE / "host_driver"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    image = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.tag], text=True
    ).strip()
    write_json(args.evidence / "result.json", {"image": image, "snapshot_revision": head})
    print(image)


if __name__ == "__main__":
    main()
