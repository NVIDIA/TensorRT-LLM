# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install benchmark clients without changing the inference environment."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/opt/trtllm-benchmark-tools")
STOCK_REF = "git+https://github.com/BowenFu/aiperf.git@9900c8ad61fea80afbe09c33a00ce693a562c798"
AGENTX_REF = "git+https://github.com/BowenFu/aiperf.git@2d03814bd25dcadca867b82bbe2cdbd1aaa773fa"
REQUIREMENTS = {
    "stock": [f"aiperf @ {STOCK_REF}", "aiohttp>=3.14.4,<4"],
    "agentx": [
        f"aiperf @ {AGENTX_REF}",
        "datasets>=4",
        "transformers==5.8.1",
        "pydantic==2.12.5",
    ],
}


def clean_environment() -> dict[str, str]:
    """Keep package discovery inside the selected virtual environment."""
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_CONSTRAINT",
    ):
        env.pop(key, None)
    # Tool requirements must not inherit the inference image's pip configuration.
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def _environment_metadata(executable: Path, env: dict[str, str]) -> dict[str, object]:
    """Read installed package versions and source identity from the client interpreter."""
    query = (
        "import importlib.metadata as m,json,sys; "
        "distributions=list(m.distributions()); "
        "client=next((d for d in distributions if d.metadata['Name'].lower()=='aiperf'),None); "
        "source=json.loads(client.read_text('direct_url.json') or '{}') if client else {}; "
        "print(json.dumps({'python_version':sys.version.split()[0], "
        "'_aiperf_present':client is not None,'_source':source, "
        "'aiperf_commit':source.get('vcs_info',{}).get('commit_id'), "
        "'packages':sorted([{'name':d.metadata['Name'],'version':d.version} "
        "for d in distributions],key=lambda d:d['name'].lower())}))"
    )
    return json.loads(
        subprocess.check_output([str(executable), "-I", "-c", query], env=env, text=True)
    )


def _write_environment(path: Path, profile: str, metadata: dict[str, object]) -> None:
    """Write comparison provenance without URLs, credentials or environment variables."""
    public = {key: metadata[key] for key in ("python_version", "aiperf_commit", "packages")}
    public["profile"] = profile
    public["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".client-environment-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(public, indent=2) + "\n")
        staged.replace(path)


def export_environment(profile: str, executable: str, destination: Path) -> None:
    """Record the selected client's current software without installing anything."""
    metadata = _environment_metadata(Path(executable), clean_environment())
    if not metadata["_aiperf_present"]:
        raise RuntimeError("The selected interpreter does not have an AIPerf client installed.")
    _write_environment(destination, profile, metadata)


def install(profile: str, root: Path, python: str, bin_dir: Path) -> None:
    """Resolve one client's dependencies and expose only its executable."""
    env = clean_environment()
    version = json.loads(
        subprocess.check_output(
            [python, "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
            env=env,
            text=True,
        )
    )
    if not (3, 11) <= tuple(version) < (3, 14):
        raise RuntimeError("Benchmark tools need Python 3.11-3.13; pass --python to select it.")

    destination = root / profile
    executable = destination / "bin" / "python"
    if not executable.exists():
        subprocess.check_call([python, "-I", "-m", "venv", str(destination)], env=env)
    installed_version = json.loads(
        subprocess.check_output(
            [
                str(executable),
                "-I",
                "-c",
                "import json,sys; print(json.dumps(list(sys.version_info[:2])))",
            ],
            env=env,
            text=True,
        )
    )
    if installed_version != version:
        raise RuntimeError(
            f"{destination} uses Python {installed_version}; select a new --root for Python {version}."
        )
    config = (destination / "pyvenv.cfg").read_text()
    if "include-system-site-packages = false" not in config.lower():
        raise RuntimeError(f"{destination} must not inherit inference site-packages")
    requirements = list(REQUIREMENTS[profile])
    if profile == "agentx" and os.environ.get("AGENTX_AIPERF_REF"):
        requirements[0] = f"aiperf @ {os.environ['AGENTX_AIPERF_REF']}"
    install_command = [
        str(executable),
        "-I",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "eager",
        *requirements,
    ]
    before = _environment_metadata(executable, env)
    plan = json.loads(
        subprocess.check_output(
            [*install_command, "--dry-run", "--ignore-installed", "--report", "-", "--quiet"],
            env=env,
            text=True,
        )
    )
    client = next(
        (item for item in plan["install"] if item["metadata"]["name"].lower() == "aiperf"),
        None,
    )
    if client is None:
        raise RuntimeError("Dependency resolution did not select an AIPerf client.")
    resolved_source = client["download_info"] if client["is_direct"] else {}
    # Pip can retain a same-version client while resolving its new source's dependencies.
    # Resolve first, then refresh only that client when its provenance changes.
    if before["_aiperf_present"] and (
        before["_source"] != resolved_source or "dir_info" in resolved_source
    ):
        subprocess.check_call(
            [str(executable), "-I", "-m", "pip", "uninstall", "--yes", "aiperf"], env=env
        )
    subprocess.check_call(install_command, env=env)
    subprocess.check_call([str(executable), "-I", "-m", "pip", "check"], env=env)
    subprocess.check_call(
        [str(executable), "-I", "-m", "aiperf", "--help"], env=env, stdout=subprocess.DEVNULL
    )
    metadata = _environment_metadata(executable, env)
    if not metadata.pop("_aiperf_present") or metadata.pop("_source") != resolved_source:
        raise RuntimeError("Installed AIPerf source does not match the resolved source.")
    _write_environment(destination / "trtllm-benchmark.json", profile, metadata)
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / ("aiperf" if profile == "stock" else "agentx-aiperf")
    with tempfile.TemporaryDirectory(prefix=".trtllm-benchmark-", dir=bin_dir) as temporary:
        staged = Path(temporary) / launcher.name
        staged.write_text(
            "#!/bin/sh\n"
            "# TensorRT-LLM isolated benchmark launcher.\n"
            "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX\n"
            f'exec {shlex.quote(str(executable))} -I -m aiperf "$@"\n'
        )
        staged.chmod(0o755)
        staged.replace(launcher)
    print(f"Installed {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("stock", "agentx", "all"), default="stock")
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("TRTLLM_BENCHMARK_ENV_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bin-dir", type=Path, help="Directory for client launchers only")
    parser.add_argument(
        "--export-environment", type=Path, help="Record an existing client; do not install"
    )
    parser.add_argument("--client-python", help="Interpreter to inspect in export mode")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export_environment is not None:
        if args.profile == "all":
            parser.error("--export-environment requires one client profile")
        override = "AGENTX_PYTHON" if args.profile == "agentx" else "TRTLLM_AIPERF_PYTHON"
        executable = args.client_python
        if executable is None:
            executable = os.environ.get(override, str(root / args.profile / "bin/python"))
        export_environment(args.profile, executable, args.export_environment)
        print(f"Recorded {args.profile} client environment in {args.export_environment}")
        return
    if args.client_python is not None:
        parser.error("--client-python is only used with --export-environment")
    root.mkdir(parents=True, exist_ok=True)
    bin_dir = (args.bin_dir or root / "bin").resolve()
    profiles = ("stock", "agentx") if args.profile == "all" else (args.profile,)
    for profile in profiles:
        install(profile, root, args.python, bin_dir)
    print(f"Add {bin_dir} to PATH; inference Python and pip are unchanged.")


if __name__ == "__main__":
    main()
