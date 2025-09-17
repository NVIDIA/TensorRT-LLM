#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

FOLDER_SECURITY_SCANNING = "security_scanning"

url_mapping = {}

target_env = {
    "python_version": 310,
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
        packages = output.split('\n')

        if packages[-1] == '':  # last entry is newline
            packages = packages[:-1]

        for package in packages:
            # WAR: ignore lines with "-f": No tool exists to parse complex requirements.txt
            if '-f' in package or \
                "#" in package or \
                package.startswith('--'):
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
                    poetry_cmd = f"poetry add '{package_name}'"
                    if package_name in url_mapping:
                        poetry_cmd = f"poetry add '{url_mapping[package_name]}'"
                    subprocess.run(poetry_cmd, shell=True, cwd=save_path)
