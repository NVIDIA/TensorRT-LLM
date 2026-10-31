# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark launchers must not import packages from the inference environment."""

import functools
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import threading
import venv
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def installer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "install_benchmark_tools", REPO_ROOT / "scripts/install_benchmark_tools.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def supported_python() -> str:
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        pytest.skip("The benchmark client requires Python 3.11-3.13")
    return sys.executable


def make_environment(destination: Path) -> Path:
    venv.EnvBuilder(with_pip=False).create(destination)
    executable = destination / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(executable), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    package = site / "aiperf"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "import json,sys\nprint(json.dumps({'prefix': sys.prefix, 'module': __file__}))\n"
    )
    metadata = site / "aiperf-0.13.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: aiperf\nVersion: 0.13.0\n")
    return package


def make_pip_environment(destination: Path) -> None:
    """Create a pip-enabled fixture using bundled seed wheels."""
    subprocess.check_call(
        [
            sys.executable,
            "-I",
            "-m",
            "virtualenv",
            "--python",
            sys.executable,
            "--no-download",
            "--no-periodic-update",
            "--app-data",
            str(destination.parent / ".virtualenv"),
            str(destination),
        ]
    )


def test_launcher_ignores_inference_package_path(
    installer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supported_python: str
) -> None:
    root = tmp_path / "tools with spaces"
    package = make_environment(root / "stock")
    contaminated = tmp_path / "inference"
    (contaminated / "aiperf").mkdir(parents=True)
    (contaminated / "aiperf/__init__.py").write_text("")
    (contaminated / "aiperf/__main__.py").write_text("raise RuntimeError('ambient package used')\n")
    monkeypatch.setenv("PYTHONPATH", str(contaminated))
    monkeypatch.setenv("VIRTUAL_ENV", str(contaminated))
    real_call = subprocess.check_call

    def check_call(argv: list[str], **kwargs: object) -> int:
        # Package resolution is exercised by the real CLI smoke; keep this test offline.
        if argv[1:4] == ["-I", "-m", "pip"]:
            return 0
        return real_call(argv, **kwargs)

    real_output = subprocess.check_output

    def check_output(argv: list[str], **kwargs: object) -> str:
        if argv[1:4] == ["-I", "-m", "pip"]:
            return json.dumps({"install": [{"metadata": {"name": "aiperf"}, "is_direct": False}]})
        return real_output(argv, **kwargs)

    monkeypatch.setitem(installer.REQUIREMENTS, "stock", ["aiperf==0.13.0"])
    monkeypatch.setattr(installer.subprocess, "check_call", check_call)
    monkeypatch.setattr(installer.subprocess, "check_output", check_output)
    original = tmp_path / "existing-client"
    original.write_text("original interpreter environment entrypoint")
    (root / "bin").mkdir()
    (root / "bin/aiperf").symlink_to(original)
    installer.install("stock", root, supported_python, root / "bin")
    assert original.read_text() == "original interpreter environment entrypoint"
    assert not (root / "bin/aiperf").is_symlink()
    result = json.loads(subprocess.check_output([str(root / "bin/aiperf")], text=True))
    assert result["prefix"] == str(root / "stock")
    assert Path(result["module"]).parent == package
    assert not (root / "bin/python").exists()
    assert not (root / "bin/pip").exists()


def test_rejects_environment_with_inference_site_packages(
    installer: ModuleType, tmp_path: Path, supported_python: str
) -> None:
    destination = tmp_path / "stock"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(destination)
    with pytest.raises(RuntimeError, match="must not inherit inference site-packages"):
        installer.install("stock", tmp_path, supported_python, tmp_path / "bin")


def test_rejects_unsupported_python_before_creating_client_environment(
    installer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer.subprocess, "check_output", lambda *args, **kwargs: "[3, 10]")
    with pytest.raises(RuntimeError, match="Python 3.11-3.13"):
        installer.install("stock", tmp_path, "python3.10", tmp_path / "bin")
    assert not (tmp_path / "stock").exists()


def write_wheel(
    directory: Path,
    name: str,
    version: str,
    dependency: str = "",
    marker: str = "local test client",
) -> Path:
    distribution = name.replace("-", "_")
    metadata_dir = f"{distribution}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    files = {
        f"{distribution}/__init__.py": "",
        f"{distribution}/__main__.py": f"print({marker!r})\n",
        f"{metadata_dir}/METADATA": metadata,
        f"{metadata_dir}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = f"{metadata_dir}/RECORD"
    files[record] = "".join(f"{path},,\n" for path in [*files, record])
    wheel = directory / f"{distribution}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, contents in files.items():
            archive.writestr(path, contents)
    return wheel


@pytest.mark.parametrize("variable", ["PIP_PYTHON", "PIP_ROOT", "PIP_TARGET", "PIP_PREFIX"])
def test_pip_destination_overrides_cannot_change_runtime(
    installer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported_python: str,
    variable: str,
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    write_wheel(wheels, "aiperf", "0.8.0", marker="unchanged runtime client")
    write_wheel(wheels, "aiperf", "0.13.0", marker="isolated client")
    runtime = tmp_path / "inference"
    make_pip_environment(runtime)
    runtime_python = str(runtime / "bin/python")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "cache"))
    env = installer.clean_environment()
    subprocess.check_call([runtime_python, "-I", "-m", "pip", "install", "aiperf==0.8.0"], env=env)
    before = {path.relative_to(runtime) for path in runtime.rglob("*")}
    monkeypatch.setenv(variable, runtime_python if variable == "PIP_PYTHON" else str(runtime))
    monkeypatch.setitem(installer.REQUIREMENTS, "stock", ["aiperf==0.13.0"])
    root = tmp_path / "clients"
    installer.install("stock", root, supported_python, root / "bin")
    assert {path.relative_to(runtime) for path in runtime.rglob("*")} == before
    assert (
        subprocess.check_output(
            [
                runtime_python,
                "-I",
                "-c",
                "import importlib.metadata as m;print(m.version('aiperf'))",
            ],
            text=True,
        ).strip()
        == "0.8.0"
    )
    assert (
        subprocess.check_output([str(root / "bin/aiperf")], text=True).strip() == "isolated client"
    )


def test_clean_environment_keeps_package_transport_settings(
    installer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {
        "PIP_INDEX_URL": "https://packages.example.invalid/simple",
        "PIP_EXTRA_INDEX_URL": "https://extra.example.invalid/simple",
        "PIP_CACHE_DIR": "/test/cache",
        "PIP_CERT": "/test/ca.pem",
        "PIP_CLIENT_CERT": "/test/client.pem",
        "PIP_KEYRING_PROVIDER": "subprocess",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PIP_USER", "1")
    env = installer.clean_environment()
    assert {key: env[key] for key in settings} == settings
    assert "PIP_USER" not in env


def test_runtime_preflight_is_read_only_and_explicit(tmp_path: Path, installer: ModuleType) -> None:
    runtime = tmp_path / "inference"
    package = make_environment(runtime)
    metadata = package.parent / "aiperf-0.13.0.dist-info/METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: aiperf\nVersion: 0.8.0\n")
    before = metadata.read_bytes()
    root = tmp_path / "unused-clients"
    command = [
        str(runtime / "bin/python"),
        str(REPO_ROOT / "scripts/install_benchmark_tools.py"),
        "--root",
        str(root),
        "--check-runtime",
    ]
    result = subprocess.run(
        command, env=installer.clean_environment(), capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "AIPerf 0.8.0 is installed in the runtime environment" in result.stderr
    assert "explicitly remove AIPerf" in result.stderr
    assert metadata.read_bytes() == before
    assert not root.exists()
    clean = tmp_path / "clean-runtime"
    venv.EnvBuilder(with_pip=False).create(clean)
    command[0] = str(clean / "bin/python")
    subprocess.check_call(command, env=installer.clean_environment())
    assert not root.exists()


@pytest.mark.parametrize("mode", ["wheel", "source"])
def test_local_setup_rejects_ambient_client_before_runtime_install(
    installer: ModuleType, tmp_path: Path, mode: str
) -> None:
    runtime = tmp_path / "inference"
    make_environment(runtime)
    marker = tmp_path / "runtime-mutated"
    env = installer.clean_environment()
    env.update(
        PATH=f"{runtime / 'bin'}:{os.environ['PATH']}",
        llmSrcNode=str(REPO_ROOT),
        INSTALL_MODE=mode,
        SLURM_LOCALID="0",
        SLURM_JOB_ID=f"benchmark-preflight-{os.getpid()}",
        MUTATION_MARKER=str(marker),
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; retry_command() { touch "$MUTATION_MARKER"; }; slurm_install_setup',
            "test",
            str(REPO_ROOT / "jenkins/scripts/perf/local/slurm_install.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "AIPerf 0.13.0 is installed in the runtime environment" in result.stderr
    assert not marker.exists()


def test_reused_environment_upgrades_transitive_dependencies(
    installer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported_python: str,
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    write_wheel(wheels, "benchmark-leaf", "1.0")
    write_wheel(wheels, "benchmark-leaf", "2.0")
    write_wheel(wheels, "aiperf", "0.13.0", "benchmark-leaf>=1")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))
    monkeypatch.setitem(installer.REQUIREMENTS, "stock", ["aiperf==0.13.0"])
    destination = tmp_path / "stock"
    make_pip_environment(destination)
    python = str(destination / "bin/python")
    env = installer.clean_environment()
    subprocess.check_call(
        [python, "-I", "-m", "pip", "install", "aiperf==0.13.0", "benchmark-leaf==1.0"],
        env=env,
    )
    # An ordinary reinstall retains the old, still-compatible transitive package.
    subprocess.check_call([python, "-I", "-m", "pip", "install", "aiperf==0.13.0"], env=env)
    version = [
        python,
        "-I",
        "-c",
        "import importlib.metadata as m;print(m.version('benchmark-leaf'))",
    ]
    assert subprocess.check_output(version, text=True).strip() == "1.0"
    installer.install("stock", tmp_path, supported_python, tmp_path / "bin")
    assert subprocess.check_output(version, text=True).strip() == "2.0"
    manifest = json.loads((destination / "trtllm-benchmark.json").read_text())
    assert {item["name"]: item["version"] for item in manifest["packages"]}[
        "benchmark-leaf"
    ] == "2.0"


def test_same_version_source_change_updates_client_metadata(
    installer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported_python: str,
) -> None:
    wheels = tmp_path / "wheels"
    updated = tmp_path / "updated"
    wheels.mkdir()
    updated.mkdir()
    write_wheel(wheels, "benchmark-leaf", "1.0")
    write_wheel(wheels, "benchmark-leaf", "2.0")
    write_wheel(wheels, "aiperf", "0.13.0", "benchmark-leaf<2", "old client")
    replacement = write_wheel(updated, "aiperf", "0.13.0", "benchmark-leaf>=2", "new client")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    destination = tmp_path / "stock"
    make_pip_environment(destination)
    python = str(destination / "bin/python")
    env = installer.clean_environment()
    subprocess.check_call([python, "-I", "-m", "pip", "install", "aiperf==0.13.0"], env=env)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(updated))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/{replacement.name}"
        requirement = f"aiperf @ {url}"
        monkeypatch.setitem(installer.REQUIREMENTS, "stock", [requirement, "benchmark-leaf==1.0"])
        with pytest.raises(subprocess.CalledProcessError):
            installer.install("stock", tmp_path, supported_python, tmp_path / "bin")
        assert (
            subprocess.check_output([python, "-I", "-m", "aiperf"], text=True).strip()
            == "old client"
        )
        monkeypatch.setitem(installer.REQUIREMENTS, "stock", [requirement])
        installer.install("stock", tmp_path, supported_python, tmp_path / "bin")
        assert (
            subprocess.check_output([python, "-I", "-m", "aiperf"], text=True).strip()
            == "new client"
        )
        direct_url = json.loads(
            subprocess.check_output(
                [
                    python,
                    "-I",
                    "-c",
                    "import importlib.metadata as m;print(m.distribution('aiperf').read_text('direct_url.json'))",
                ],
                text=True,
            )
        )
        assert direct_url["url"] == url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("explicit", [True, False])
def test_environment_export_uses_selected_client_without_installing(
    installer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    client = tmp_path / "custom-client"
    package = make_environment(client)
    metadata = package.parent / "aiperf-0.13.0.dist-info"
    (metadata / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://user:do-not-export@example.invalid/client.git?token=do-not-export",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            }
        )
    )
    unused_root = tmp_path / "not-created"
    output = tmp_path / "artifacts/client_environment.json"
    argv = [
        "install_benchmark_tools.py",
        "--profile",
        "stock",
        "--root",
        str(unused_root),
        "--export-environment",
        str(output),
    ]
    if explicit:
        argv += ["--client-python", str(client / "bin/python")]
    else:
        monkeypatch.setenv("TRTLLM_AIPERF_PYTHON", str(client / "bin/python"))
    monkeypatch.setattr(sys, "argv", argv)

    def no_install(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Export must not install packages or create a client environment")

    monkeypatch.setattr(installer, "install", no_install)
    monkeypatch.setattr(installer, "check_runtime", no_install)
    monkeypatch.setattr(installer.subprocess, "check_call", no_install)
    installer.main()
    result = json.loads(output.read_text())
    assert result["aiperf_commit"] == "a" * 40
    assert result["packages"] == [{"name": "aiperf", "version": "0.13.0"}]
    assert result["profile"] == "stock"
    assert "do-not-export" not in output.read_text()
    assert "example.invalid" not in output.read_text()
    assert "_source" not in result
    assert not unused_root.exists()
