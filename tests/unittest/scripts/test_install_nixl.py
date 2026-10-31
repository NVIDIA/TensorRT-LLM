# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[3]
INSTALL_NIXL = ROOT / "docker/common/install_nixl.sh"

# Execute the full installer while replacing network, compilation and package installation.
_COMMAND = r"""
wget() {
    printf '%s\n' "$@" > "$WGET_ARGS"
    [ "$FAIL_STAGE" != download ] || return 23
    : > libfabric.tar.bz2
}
tar() {
    mkdir libfabric-1.21.0
    printf '#!/bin/bash\nexit 0\n' > libfabric-1.21.0/autogen.sh
    cat > libfabric-1.21.0/configure <<'EOF'
#!/bin/bash
[ "$FAIL_STAGE" != configure ] || exit 23
printf '%s\n' "$@" > "$CONFIGURE_ARGS"
for arg in "$@"; do
    case "$arg" in --prefix=*) mkdir -p "$(printf '%s' "$arg" | cut -c10-)" ;; esac
done
EOF
    chmod +x libfabric-1.21.0/autogen.sh libfabric-1.21.0/configure
}
make() { :; }
nproc() { echo 2; }
pip3() {
    printf '%s\n' "$@" >> "$PIP_ARGS"
    if [ "$2" = "--no-deps" ] && [ "$FAIL_STAGE" = shim ]; then return 23; fi
}
git() { printf '%s\n' "$@" > "$GIT_ARGS"; mkdir nixl; }
find() { printf '%s\n' /usr/local/cuda/lib64/libcuda.so.1; }
uname() { echo "$TEST_ARCH"; }
meson() {
    printf '%s\n' "$@" > "$MESON_ARGS"
    if [ -n "$PKG_CONFIG_PATH" ]; then
        printf '%s' "$PKG_CONFIG_PATH" > "$MESON_PKG_CONFIG_PATH"
    else
        printf '<unset>' > "$MESON_PKG_CONFIG_PATH"
    fi
    mkdir builddir
}
ninja() { [ "$FAIL_STAGE" != ninja ]; }
source "$INSTALL_NIXL"
printf '%s' "$LD_LIBRARY_PATH" > "$RESTORED_LD_PATH"
"""


def _run_installer(
    tmp_path: Path,
    missing_hwloc: str | None = None,
    pkg_config_path: str | None = None,
    arch: str = "x86_64",
    fail_stage: str = "",
    github_mirror: str | None = None,
) -> subprocess.CompletedProcess[str]:
    mpi_path = tmp_path / "mpi"
    for name in ("include/hwloc.h", "lib/libhwloc.so", "lib/pkgconfig/hwloc.pc"):
        if missing_hwloc not in ("all", name):
            dependency = mpi_path / name
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.touch()
    installer = tmp_path / "install_nixl.sh"
    installer.write_text(INSTALL_NIXL.read_text().replace("/usr/local/mpi", str(mpi_path)))
    (tmp_path / "github_auth.sh").write_text(":\n")
    (tmp_path / "scratch").mkdir()
    # These names used to be vulnerable to the installer's wildcard cleanup.
    for name in ("libfabric-keep", "nixl-keep"):
        (tmp_path / name).write_text("keep\n")
    env = {
        "PATH": os.environ["PATH"],
        "INSTALL_NIXL": str(installer),
        "TMPDIR": str(tmp_path / "scratch"),
        "ENV": str(tmp_path / "shinit"),
        "LD_LIBRARY_PATH": "/existing/lib",
        "TEST_ARCH": arch,
        "FAIL_STAGE": fail_stage,
    }
    for name in (
        "CONFIGURE_ARGS",
        "PIP_ARGS",
        "GIT_ARGS",
        "WGET_ARGS",
        "MESON_ARGS",
        "MESON_PKG_CONFIG_PATH",
        "RESTORED_LD_PATH",
    ):
        env[name] = str(tmp_path / name.lower())
    if pkg_config_path is not None:
        env["PKG_CONFIG_PATH"] = pkg_config_path
    if github_mirror is not None:
        env["GITHUB_MIRROR"] = github_mirror
    return subprocess.run(
        ["bash", "-c", _COMMAND],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_cleaned(tmp_path: Path) -> None:
    assert not list((tmp_path / "scratch").iterdir())
    for name in ("libfabric-keep", "nixl-keep"):
        assert (tmp_path / name).read_text() == "keep\n"


@pytest.mark.parametrize("arch", ("x86_64", "aarch64"))
@pytest.mark.parametrize(
    "missing_hwloc", (None, "all", "include/hwloc.h", "lib/libhwloc.so", "lib/pkgconfig/hwloc.pc")
)
@pytest.mark.parametrize("pkg_config_path", (None, "/custom/lib/pkgconfig"))
def test_build_only_libfabric_and_hwloc(
    tmp_path: Path, arch: str, missing_hwloc: str | None, pkg_config_path: str | None
) -> None:
    result = _run_installer(tmp_path, missing_hwloc, pkg_config_path, arch)
    assert result.returncode == 0, result.stderr
    configure_args = (tmp_path / "configure_args").read_text().splitlines()
    prefix = next(
        arg.removeprefix("--prefix=") for arg in configure_args if arg.startswith("--prefix=")
    )
    assert Path(prefix).is_relative_to(tmp_path / "scratch")
    assert "--disable-rstream" in configure_args
    args = (tmp_path / "meson_args").read_text().splitlines()
    assert f"-Dlibfabric_path={prefix}" in args
    expected_pkg_config_path = pkg_config_path or "<unset>"
    if missing_hwloc is None:
        expected_pkg_config_path = str(tmp_path / "mpi/lib/pkgconfig")
        if pkg_config_path:
            expected_pkg_config_path += f":{pkg_config_path}"
    assert (tmp_path / "meson_pkg_config_path").read_text() == expected_pkg_config_path
    assert (tmp_path / "restored_ld_path").read_text() == "/existing/lib"
    shinit = (tmp_path / "shinit").read_text()
    assert f"/lib/{arch}-linux-gnu:" in shinit
    assert prefix not in shinit
    git_args = (tmp_path / "git_args").read_text().splitlines()
    version = git_args[git_args.index("-b") + 1].removeprefix("v")
    assert f"nixl=={version}" in (tmp_path / "pip_args").read_text().splitlines()
    assert f"nixl-cu13=={version}" in (ROOT / "requirements-dev.txt").read_text().splitlines()
    _assert_cleaned(tmp_path)


@pytest.mark.parametrize("fail_stage", ("download", "configure", "ninja", "shim"))
def test_failed_install_cleans_build_files(tmp_path: Path, fail_stage: str) -> None:
    result = _run_installer(tmp_path, fail_stage=fail_stage)
    assert result.returncode != 0
    _assert_cleaned(tmp_path)


@pytest.mark.parametrize("github_mirror", (None, "", "https://mirror.example.test/github"))
def test_libfabric_release_mirror(tmp_path: Path, github_mirror: str | None) -> None:
    result = _run_installer(tmp_path, github_mirror=github_mirror)
    assert result.returncode == 0, result.stderr
    base = github_mirror or "https://github.com"
    assert (
        f"{base}/ofiwg/libfabric/releases/download/v1.21.0/libfabric-1.21.0.tar.bz2"
        in (tmp_path / "wget_args").read_text().splitlines()
    )
    # The release-asset mirror does not serve the Git protocol.
    assert (
        "https://github.com/ai-dynamo/nixl.git" in (tmp_path / "git_args").read_text().splitlines()
    )
