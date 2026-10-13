# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

INSTALL_NIXL = Path(__file__).resolve().parents[3] / "docker/common/install_nixl.sh"


@pytest.mark.parametrize("with_header", (False, True))
@pytest.mark.parametrize("with_library", (False, True))
def test_libfabric_path_requires_efa_development_files(
    tmp_path: Path, with_header: bool, with_library: bool
) -> None:
    efa_path = tmp_path / "efa"
    for present, name in (
        (with_header, "include/rdma/fabric.h"),
        (with_library, "lib/libfabric.so"),
    ):
        if present:
            dependency = efa_path / name
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.touch()

    # Relocate the installation prefix without needing /opt write access.
    installer = tmp_path / "install_nixl.sh"
    installer.write_text(INSTALL_NIXL.read_text().replace("/opt/amazon/efa", str(efa_path)))
    (tmp_path / "github_auth.sh").write_text(":\n")
    meson_args = tmp_path / "meson_args"
    command = r"""
pip3() { :; }
git() { mkdir nixl; }
find() { printf '%s\n' /usr/local/cuda/lib64/libcuda.so.1; }
meson() {
    printf '%s\n' "$@" > "$MESON_ARGS"
    exit 0
}
source "$INSTALL_NIXL"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "INSTALL_NIXL": str(installer),
            "MESON_ARGS": str(meson_args),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    args = meson_args.read_text().splitlines()
    expected_path = str(efa_path) if with_header and with_library else ""
    assert [arg for arg in args if arg.startswith("-Dlibfabric_path=")] == [
        f"-Dlibfabric_path={expected_path}"
    ]
