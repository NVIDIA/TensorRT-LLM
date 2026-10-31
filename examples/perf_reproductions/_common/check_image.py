# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the running container matches an entry's pinned TensorRT-LLM version.

Run inside the container. To check the image digest on the host instead, compare
``docker image inspect --format '{{index .RepoDigests 0}}' <image>`` with ``image.digest`` in source.yaml.

Usage: python3 check_image.py <entry>/source.yaml
"""

import sys
from importlib import metadata
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    image = yaml.safe_load(Path(sys.argv[1]).read_text())["image"]
    expected = image.get("tensorrt_llm_version")
    installed = metadata.version("tensorrt_llm")
    ok = installed == expected
    print(
        f"{'OK  ' if ok else 'FAIL'} tensorrt_llm {installed} (expected {expected} from {image['name']})"
    )
    for package, version in (
        yaml.safe_load(Path(sys.argv[1]).read_text()).get("dependencies") or {}
    ).items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            found = None
        ok &= found == version
        print(f"{'OK  ' if found == version else 'FAIL'} {package} {found} (expected {version})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
