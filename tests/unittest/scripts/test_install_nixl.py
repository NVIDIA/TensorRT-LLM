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
@pytest.mark.parametrize(
    "missing_hwloc",
    (None, "all", "include/hwloc.h", "lib/libhwloc.so", "lib/pkgconfig/hwloc.pc"),
)
@pytest.mark.parametrize("pkg_config_path", (None, "/custom/lib/pkgconfig"))
def test_bundled_dependencies_require_development_files(
    tmp_path: Path,
    with_header: bool,
    with_library: bool,
    missing_hwloc: str | None,
    pkg_config_path: str | None,
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

    mpi_path = tmp_path / "mpi"
    for name in ("include/hwloc.h", "lib/libhwloc.so", "lib/pkgconfig/hwloc.pc"):
        if missing_hwloc not in ("all", name):
            dependency = mpi_path / name
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.touch()

    # Relocate NGC PyTorch's bundled EFA and MPI prefixes without root access.
    installer = tmp_path / "install_nixl.sh"
    installer.write_text(
        INSTALL_NIXL.read_text()
        .replace("/opt/amazon/efa", str(efa_path))
        .replace("/usr/local/mpi", str(mpi_path))
    )
    (tmp_path / "github_auth.sh").write_text(":\n")
    meson_args = tmp_path / "meson_args"
    command = r"""
pip3() { :; }
git() { mkdir nixl; }
find() { printf '%s\n' /usr/local/cuda/lib64/libcuda.so.1; }
meson() {
    printf '%s\n' "$@" > "$MESON_ARGS"
    printf '%s' "${PKG_CONFIG_PATH-<unset>}" > "$MESON_PKG_CONFIG_PATH"
    exit 0
}
source "$INSTALL_NIXL"
"""
    env = {
        "PATH": os.environ["PATH"],
        "INSTALL_NIXL": str(installer),
        "MESON_ARGS": str(meson_args),
        "MESON_PKG_CONFIG_PATH": str(tmp_path / "meson_pkg_config_path"),
    }
    if pkg_config_path is not None:
        env["PKG_CONFIG_PATH"] = pkg_config_path
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=env,
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
    expected_pkg_config_path = pkg_config_path or "<unset>"
    if with_header and with_library and missing_hwloc is None:
        expected_pkg_config_path = str(mpi_path / "lib/pkgconfig")
        if pkg_config_path:
            expected_pkg_config_path += f":{pkg_config_path}"
    assert (tmp_path / "meson_pkg_config_path").read_text() == expected_pkg_config_path
