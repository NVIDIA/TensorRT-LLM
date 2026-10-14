# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
INSTALLER = Path(__file__).resolve().parents[3] / "docker/common/install_libfabric.sh"


def run_installer(tmp_path: Path, *, present: bool = True, **overrides: str):
    efa = tmp_path / "efa"
    if present:
        for filename in ("include/rdma/fabric.h", "lib/libfabric.so"):
            path = efa / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        binary = efa / "bin/fi_info"
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\nprintf 'efa:\\nshm:\\n'\n")
        binary.chmod(0o755)
    installer = tmp_path / "install_libfabric.sh"
    installer.write_text(
        INSTALLER.read_text()
        .replace("/opt/amazon/efa", str(efa))
        .replace("/third-party-source", str(tmp_path / "source"))
    )
    patch = tmp_path / "patches/libfabric-2.4.0amzn5.0-cuda-vmm.patch"
    patch.parent.mkdir()
    patch.touch()
    command = r"""
pkg-config() { printf '%s\n' "$VERSION"; }
curl() { touch "$CALLED"; }
sha256sum() { cat >/dev/null; return "$CHECKSUM_RC"; }
patch() { return "$PATCH_RC"; }
nproc() { echo 1; }
tar() {
    if [ "$1" = -xzf ]; then
        mkdir libfabric-2.4.0amzn5.0
        printf '#!/bin/sh\nexit 0\n' > libfabric-2.4.0amzn5.0/autogen.sh
        cp libfabric-2.4.0amzn5.0/autogen.sh libfabric-2.4.0amzn5.0/configure
        chmod +x libfabric-2.4.0amzn5.0/{autogen.sh,configure}
        for feature in HAVE_CUDA HAVE_CUDA_CTX_SYNC_MEMOPS HAVE_GDRCOPY HAVE_NEURON; do
            [ "$feature" = "$MISSING_FEATURE" ] || echo "#define $feature 1"
        done > libfabric-2.4.0amzn5.0/config.h
    fi
}
make() {
    if [[ "${1:-}" = DESTDIR=* ]]; then
        target="${1#DESTDIR=}$EFA_TEST_PATH"
        mkdir -p "$target/lib" "$target/bin"
        echo patched > "$target/lib/libfabric.so"
        printf '#!/bin/sh\nprintf "%s"\n' "$NEW_PROVIDERS" > "$target/bin/fi_info"
        chmod +x "$target/bin/fi_info"
    fi
}
ldconfig() { touch "$INSTALLED"; }
source "$INSTALLER"
"""
    env = {
        "PATH": os.environ["PATH"],
        "INSTALLER": str(installer),
        "EFA_TEST_PATH": str(efa),
        "CALLED": str(tmp_path / "download-started"),
        "INSTALLED": str(tmp_path / "installed"),
        "VERSION": "2.4.0amzn5.0",
        "CHECKSUM_RC": "0",
        "PATCH_RC": "0",
        "MISSING_FEATURE": "",
        "NEW_PROVIDERS": "shm:\\nefa:\\n",
        **overrides,
    }
    return subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, timeout=30
    )


def test_no_bundled_efa_skips_download(tmp_path: Path) -> None:
    result = run_installer(tmp_path, present=False)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "download-started").exists()


def test_version_change_requires_revalidation(tmp_path: Path) -> None:
    result = run_installer(tmp_path, VERSION="2.7.0amzn1.0")
    assert result.returncode != 0
    assert "Revalidate" in result.stderr
    assert not (tmp_path / "download-started").exists()


@pytest.mark.parametrize(
    "failure",
    [
        {"CHECKSUM_RC": "1"},
        {"PATCH_RC": "1"},
        {"MISSING_FEATURE": "HAVE_NEURON"},
        {"NEW_PROVIDERS": "efa:\\n"},
    ],
)
def test_failed_validation_preserves_original_library(
    tmp_path: Path, failure: dict[str, str]
) -> None:
    result = run_installer(tmp_path, **failure)
    assert result.returncode != 0
    assert not (tmp_path / "installed").exists()
    assert (tmp_path / "efa/lib/libfabric.so").read_text() == "original"


def test_matching_providers_allow_installation(tmp_path: Path) -> None:
    result = run_installer(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "installed").exists()
    assert (tmp_path / "efa/lib/libfabric.so").read_text().strip() == "patched"
