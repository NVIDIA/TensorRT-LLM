# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Self-contained wrapper tests for the verl repo.

All setup (dependency installation, repo cloning, env vars) is handled by
a session-scoped pytest fixture. Configuration is read from verl_config.yml.
"""

import os
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "verl_config.yml")
VERL_ROOT = os.path.join(_HERE, "verl_repo")


def _load_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["verl_config"]


def _export_env_vars(config):
    """Export env vars from config into the current process environment."""
    for entry in config.get("env_vars", []):
        key, val = entry.split("=", 1)
        val = val.strip('"')
        val = os.path.expandvars(val)
        os.environ[key] = val


def _run_install_commands(config):
    """Run install commands from config with env vars already set."""
    for cmd in config.get("install_commands", []):
        print(f"[verl setup] Running: {cmd}")
        subprocess.check_call(cmd, shell=True)


def _clone_verl_repo(config):
    """Clone the verl repo and checkout the specified tag."""
    if os.path.isdir(VERL_ROOT):
        print(f"[verl setup] Repo already exists at {VERL_ROOT}, skipping clone")
        return
    repo_url = config["repo_url"]
    repo_tag = config["repo_tag"]
    print(f"[verl setup] Cloning {repo_url} (tag={repo_tag}) into {VERL_ROOT}")
    subprocess.check_call(
        f"git clone {repo_url} {VERL_ROOT} && cd {VERL_ROOT} && git checkout {repo_tag}",
        shell=True,
    )
    assert os.path.isdir(VERL_ROOT), f"Failed to clone verl repo to {VERL_ROOT}"
    print(f"[verl setup] Installing verl package from {VERL_ROOT}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", VERL_ROOT],
    )


def _setup_model_symlinks(config):
    """Create symlinks from HF-style paths to CI cache paths.

    Verl tests expect models at {model_root}/Qwen/ModelName but the CI cache
    stores them at {ci_cache}/ModelName (flat structure). We create symlinks
    in a writable staging directory that point to the read-only CI cache.
    """
    model_root = os.environ.get("TRTLLM_TEST_MODEL_PATH_ROOT", "")
    ci_cache = config.get("ci_model_cache", "")
    if not model_root or not ci_cache:
        return
    for model_id in config.get("required_models", []):
        if "/" not in model_id:
            continue
        namespace, name = model_id.split("/", 1)
        ns_dir = os.path.join(model_root, namespace)
        src = os.path.join(ci_cache, name)
        dst = os.path.join(ns_dir, name)
        if os.path.exists(dst):
            print(f"[verl setup] Model symlink already exists: {dst}")
            continue
        if not os.path.isdir(src):
            print(f"[verl setup] Model not found in CI cache: {src}, skipping")
            continue
        os.makedirs(ns_dir, exist_ok=True)
        os.symlink(src, dst)
        print(f"[verl setup] Created symlink: {dst} -> {src}")


@pytest.fixture(scope="session", autouse=True)
def verl_setup():
    """Session-scoped fixture: install deps, set env vars, clone verl repo."""
    config = _load_config()
    _export_env_vars(config)
    _run_install_commands(config)
    _clone_verl_repo(config)
    _setup_model_symlinks(config)
    yield VERL_ROOT


def _run_verl_test(test_path, extra_args=None, timeout=600):
    """Run a test from the verl repo via subprocess."""
    full_path = os.path.join(VERL_ROOT, test_path)
    assert os.path.exists(full_path), f"Verl test not found: {full_path}"
    cmd = [sys.executable, "-m", "pytest", full_path, "-v", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd,
        cwd=VERL_ROOT,
        env=os.environ.copy(),
        timeout=timeout,
    )
    assert result.returncode == 0, f"Verl test failed with return code {result.returncode}"


def test_async_server():
    _run_verl_test("tests/workers/rollout/rollout_trtllm/test_async_server.py")


def test_adapter():
    _run_verl_test("tests/workers/rollout/rollout_trtllm/test_adapter.py")


def test_rollout_utils():
    _run_verl_test(
        "tests/workers/rollout/rollout_trtllm/test_trtllm_rollout_utils.py",
        extra_args=[
            "-k",
            "not (test_unimodal_generate or test_unimodal_batch_generate)",
        ],
        timeout=900,
    )
