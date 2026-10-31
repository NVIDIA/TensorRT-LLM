# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
INSTALL_EFA = Path(__file__).resolve().parents[3] / "docker/common/install_efa_installer.sh"

_COMMAND = r"""
curl() {
    printf '%s\n' "$@" > "$CURL_ARGS"
    [ "$FAIL_STAGE" != download ]
}
tar() {
    mkdir aws-efa-installer
    cat > aws-efa-installer/efa_installer.sh <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > "$EFA_ARGS"
[ "$FAIL_STAGE" != install ]
EOF
    chmod +x aws-efa-installer/efa_installer.sh
}
apt-get() { printf '%s\n' "$@" > "$APT_ARGS"; }
source "$INSTALL_EFA"
"""


@pytest.mark.parametrize(
    ("os_id", "version", "fail_stage"),
    (
        ("rocky", None, ""),
        ("ubuntu", None, ""),
        ("ubuntu", "1.52.0", ""),
        ("ubuntu", None, "download"),
        ("ubuntu", None, "install"),
    ),
)
def test_efa_container_install(
    tmp_path: Path, os_id: str, version: str | None, fail_stage: str
) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text(f'ID="{os_id}"\n')
    installer = tmp_path / "install_efa.sh"
    installer.write_text(INSTALL_EFA.read_text().replace("/etc/os-release", str(os_release)))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "INSTALL_EFA": str(installer),
        "TMPDIR": str(scratch),
        "FAIL_STAGE": fail_stage,
        "CURL_ARGS": str(tmp_path / "curl_args"),
        "EFA_ARGS": str(tmp_path / "efa_args"),
        "APT_ARGS": str(tmp_path / "apt_args"),
    }
    if version is not None:
        env["EFA_INSTALLER_VERSION"] = version
    result = subprocess.run(
        ["bash", "-c", _COMMAND], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert (result.returncode != 0) == bool(fail_stage), result.stderr
    assert not list(scratch.iterdir())
    if fail_stage == "download":
        assert (tmp_path / "curl_args").exists()
        assert not (tmp_path / "efa_args").exists()
    elif fail_stage == "install":
        assert (tmp_path / "efa_args").exists()
    if os_id == "rocky":
        assert not (tmp_path / "curl_args").exists()
        assert not (tmp_path / "apt_args").exists()
    elif not fail_stage:
        curl_args = (tmp_path / "curl_args").read_text().splitlines()
        assert curl_args[curl_args.index("--connect-timeout") + 1] == "30"
        assert curl_args[curl_args.index("--max-time") + 1] == "1800"
        assert (tmp_path / "efa_args").read_text().splitlines() == [
            "--skip-mpi",
            "--skip-rdma-core",
            "--build-ngc",
        ]
        assert (tmp_path / "apt_args").read_text().splitlines() == ["update"]
        assert (
            f"https://efa-installer.amazonaws.com/aws-efa-installer-{version or '1.51.0'}.tar.gz"
            in (tmp_path / "curl_args").read_text().splitlines()
        )
