#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Generates pyproject.toml and poetry.lock files from requirements.txt

Black Duck requires poetry lock files to perform security scans. TensorRT uses requirements.txt to define
Python dependencies.
This script parses through the requirements.txt files in the project and generates the poetry lock files
required to perform the security scans.

Pip install requirements:
pip3 install poetry

To generate pyproject.toml and poetry.lock recursively for all requirements.txt in the project:
python3 scripts/generate_lock_files.py

To generate pyproject.toml and poetry.lock for a single requirements.txt:
python3 scripts/generate_lock_files.py --path <path>/requirements.txt
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())

FOLDER_SECURITY_SCANNING = "security_scanning"

url_mapping = {}

# Constraints added to a project before its requirements.txt entries, keyed by
# the project's directory relative to the repo root. They steer the resolver
# away from versions that make it backtrack through hundreds of releases.
project_constraints = {
    # botocore (via nemo-toolkit) requires python-dateutil<3.0.0.
    "examples/quantization": ["python-dateutil<3.0.0"],
}

target_env = {
    "python_version": 312,
    "platform_system": "",
    "platform_machine": "x86_64",
    "sys_platform": "linux",
}


def get_project_info(path: str):
    path_project = re.sub(rf"^{os.getcwd()}\/?", "", path)
    name = "unknown-package"
    version = "0.1.0"
    if not path_project:
        name = "tensorrt-llm"
        import importlib.util

        # get trtllm version from tensorrt_llm/version.py
        module_path = os.path.join("tensorrt_llm", "version.py")
        spec = importlib.util.spec_from_file_location("trtllm_version",
                                                      module_path)
        if spec and spec.loader:
            version_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(version_module)
            version = version_module.__version__
    else:
        matches = re.match(r"^(?:([\w\-]+)?\/)?([\w\-]+)$", path_project)
        if matches:
            if matches.group(1):
                name = f"{matches.group(2)}-{matches.group(1)}"
            else:
                name = matches.group(2)
    return {"name": name, "version": version}


def _normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_unresolvable_spec(output, specs):
    """Return the index of the spec that Poetry blames for a failed resolution.

    Poetry ends a failed resolution with
    "So, because <project> depends on <pkg> (<constraint>), version solving
    failed." (or "... on both <a> (..) and <b> (..)", or "Could not find a
    matching version of package <pkg>"). When several packages are named, the
    one listed last in `specs` is blamed, so that earlier entries win, as when
    adding one by one. Returns None if no listed package can be identified.
    """
    names = set()
    match = re.search(
        r"So, because .*? depends on (.*?), version solving failed", output,
        re.S)
    if match:
        names.update(
            _normalize_name(n) for n in re.findall(
                r"([A-Za-z0-9][A-Za-z0-9_.\-]*)(?:\[[^\]]*\])? \(",
                match.group(1)))
    match = re.search(r"Could not find a matching version of package (\S+)",
                      output)
    if match:
        names.add(_normalize_name(match.group(1)))
    indexes = []
    for i, spec in enumerate(specs):
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", spec)
        if match and _normalize_name(match.group(0)) in names:
            indexes.append(i)
    return max(indexes) if indexes else None


def _is_resolution_failure(output):
    """Whether Poetry's output reports a dependency-resolution failure.

    Only such failures justify skipping a spec. Anything else (network or index
    errors, auth problems, a Poetry crash) says nothing about the spec itself.
    """
    return bool(
        re.search(
            r"version solving failed|"
            r"Could not find a matching version of package", output))


def poetry_add(specs, cwd):
    specs = list(specs)
    while specs:
        result = subprocess.run(["poetry", "add", "--lock", *specs],
                                cwd=cwd,
                                capture_output=True,
                                text=True)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode == 0:
            return
        output = result.stdout + result.stderr
        if not _is_resolution_failure(output):
            raise RuntimeError(
                f"'poetry add' failed in {cwd} (exit code {result.returncode}) "
                "for a reason other than dependency resolution; refusing to "
                "produce an incomplete lock file")
        culprit = _find_unresolvable_spec(output, specs)
        if culprit is None:
            if len(specs) == 1:
                print(f"Skipping {specs[0]}: dependency resolution failed")
                return
            mid = len(specs) // 2
            poetry_add(specs[:mid], cwd)
            poetry_add(specs[mid:], cwd)
            return
        print(f"Skipping {specs[culprit]}: dependency resolution failed")
        del specs[culprit]


def generate_metadata_json():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving git commit hash: {e}")
        raise

    data = {
        "commit_hash": commit_hash,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(f"{FOLDER_SECURITY_SCANNING}/metadata.json",
              "w",
              encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lock files generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", help="Path to requirements.txt")
    args, _ = parser.parse_known_args()

    if args.path:
        _, filename = os.path.split(args.path)
        assert filename == 'requirements.txt'
        realpath = Path(args.path).resolve()
        paths = [realpath]
    else:
        # get paths to all files names requirements.txt
        paths = Path.cwd().rglob('requirements.txt')

    if os.path.exists(FOLDER_SECURITY_SCANNING):
        shutil.rmtree(FOLDER_SECURITY_SCANNING)
    os.mkdir(FOLDER_SECURITY_SCANNING)
    generate_metadata_json()

    # generate pyproject.toml and poetry.lock files in the same location
    for path in paths:
        file_path, file_name = os.path.split(path)
        curr_path = Path.cwd()
        if "3rdparty" in file_path:
            continue

        # init poetry
        save_path = os.path.join(FOLDER_SECURITY_SCANNING,
                                 Path(file_path).relative_to(curr_path))
        os.makedirs(save_path, exist_ok=True)
        print(f"Initializing PyProject.toml in {file_path}")
        project_info = get_project_info(file_path)
        name = project_info["name"]
        author = '"TensorRT LLM [90828364+tensorrt-cicd@users.noreply.github.com]"'
        version = project_info["version"]
        py_version = '">=3.10,<3.13"'
        poetry_init_cmd = f'poetry init --no-interaction --name {name} --author {author} --python {py_version}'
        if name == "tensorrt-llm":
            poetry_init_cmd += " -l Apache-2.0"
        subprocess.run(poetry_init_cmd, shell=True, cwd=save_path)
        if version != "0.1.0":
            subprocess.run(f"poetry version {version}",
                           shell=True,
                           cwd=file_path)
        output = subprocess.run(f'cat {path}',
                                shell=True,
                                capture_output=True,
                                text=True).stdout
        # Constraints go first so they steer the resolver for the whole batch.
        specs = list(
            project_constraints.get(str(Path(file_path).relative_to(curr_path)),
                                    []))
        packages = output.split('\n')

        if packages[-1] == '':  # last entry is newline
            packages = packages[:-1]

        for package in packages:
            package = re.sub(r'\s#.*$', '', package).rstrip()
            # WAR: ignore lines with "-f": No tool exists to parse complex requirements.txt
            if not package or \
                package.startswith('#') or \
                '-f' in package or \
                package.startswith('-'):
                continue

            curr_env = None
            if ';' in package:
                curr_env = package.split(';')[1]
                curr_env = curr_env.replace("sys.", "sys_")

                # WAR for "3.8" < "3.10" evaluating to False:
                # convert to int and remove decimal 38 < 310 is True
                while '.' in curr_env:
                    py_version_str = curr_env.split(
                        '.')[0][-1] + '.' + curr_env.split('.')[1][0]
                    if curr_env.split('.')[1][1] != '"' and curr_env.split(
                            '.')[1][1] != "'":
                        py_version_str += curr_env.split('.')[1][1]

                    py_version_int = py_version_str.replace('.', '')

                    curr_env = curr_env.replace(f'"{py_version_str}"',
                                                py_version_int)

            if curr_env is None or eval(curr_env, target_env):
                data = package.split(';')
                package_name = data[0].replace(" ", "")
                if (not package_name.startswith("tensorrt_llm")
                        and not package_name.startswith("3rdparty")):
                    specs.append(url_mapping.get(package_name, package_name))

        poetry_add(list(dict.fromkeys(specs)), save_path)
