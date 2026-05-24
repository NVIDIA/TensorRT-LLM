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

Each wrapper function maps 1-to-1 to a single verl pytest case, enabling
fine-grained waiving without blanket-skipping a whole file.
"""

import os
import re
import subprocess
import sys
import tempfile

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "verl_config.yml")
VERL_ROOT = os.path.join(_HERE, "verl_repo")

_ROLLOUT = "tests/workers/rollout/rollout_trtllm"
_ASYNC_SERVER = f"{_ROLLOUT}/test_async_server.py"
_ADAPTER = f"{_ROLLOUT}/test_adapter.py"
_ROLLOUT_UTILS = f"{_ROLLOUT}/test_trtllm_rollout_utils.py"
_INTER_NODE = f"{_ROLLOUT}/test_inter_node_rollout.py"
_ABORT = f"{_ROLLOUT}/test_trtllm_abort.py"


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


def _run_single(verl_file, case_name, timeout=600):
    """Run exactly one verl pytest case by ``-k`` filter.

    Tried switching to pytest node-ID syntax (``<file>::<case>``) per
    CodeRabbit's suggestion, but it returns pytest exit 4 (usage error)
    under verl's own pytest config — likely because verl's conftest /
    pytest plugins don't cleanly accept the absolute-path node-ID format.
    Substring collision under ``-k`` is a theoretical risk we accept
    because verl's case names are distinctive and don't share substrings
    in practice.
    """
    _run_verl_test(verl_file, extra_args=["-k", case_name], timeout=timeout)


# ---------------------------------------------------------------------------
# Shared helpers for E2E training tests (test_verl_E2E_*.py)
# ---------------------------------------------------------------------------

_STEP_LINE = re.compile(r"step:\d+ - ")


def _dump_log_tail(log_file, lines=100, label=""):
    """Print last ``lines`` lines of ``log_file`` to stdout for CI diagnostics.

    The verl E2E tests redirect subprocess stdout/stderr to per-test log
    files inside the container's ``/tmp``. When the container exits those
    files are lost, so on subprocess failure we dump the tail to stdout
    (which the CI runner captures into Jenkins console output) so the
    actual error is preserved.
    """
    print(f"=== {label} log tail ({log_file}, last {lines} lines) ===")
    try:
        with open(log_file) as fh:
            for line in fh.readlines()[-lines:]:
                print(line, end="")
    except OSError as e:
        print(f"(could not read log: {e})")
    print("=== end log tail ===")


def _run_verl_train(extra_args, log_file=None, timeout=1800):
    """Run ``python3 -m verl.trainer.main_ppo`` in VERL_ROOT with Hydra overrides.

    Stdout/stderr are written to ``log_file`` (default: a fresh mktemp).
    Asserts the subprocess return code is 0 and returns the log path so the
    caller can grep per-step metrics out of it.
    """
    if log_file is None:
        # ``mkstemp`` rather than the deprecated ``mktemp`` to avoid TOCTOU.
        fd, log_file = tempfile.mkstemp(suffix="-verl-train.log")
        os.close(fd)
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *extra_args]
    with open(log_file, "w") as fh:
        result = subprocess.run(
            cmd,
            cwd=VERL_ROOT,
            env=os.environ.copy(),
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if result.returncode != 0:
        _dump_log_tail(log_file, label=f"verl trainer (exit {result.returncode})")
    assert result.returncode == 0, f"verl trainer exited {result.returncode}; see {log_file}"
    return log_file


def _ensure_gsm8k(local_dir):
    """Resolve a directory containing GSM8K's ``train.parquet`` + ``test.parquet``.

    Checks (in order): the requested ``local_dir`` and
    ``$TRTLLM_TEST_DATA_PATH/gsm8k``. Falls back to runtime preprocess via
    ``examples/data_preprocess/gsm8k.py`` (network download; ~10 MB, finishes
    in well under a minute on CI nodes).
    """
    candidates = [
        local_dir,
        os.path.join(os.environ.get("TRTLLM_TEST_DATA_PATH", ""), "gsm8k"),
    ]
    for path in candidates:
        if (
            path
            and os.path.exists(os.path.join(path, "train.parquet"))
            and os.path.exists(os.path.join(path, "test.parquet"))
        ):
            return path
    os.makedirs(local_dir, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "examples/data_preprocess/gsm8k.py", "--local_dir", local_dir],
        cwd=VERL_ROOT,
    )
    return local_dir


def _check_convergence(log_file, target=0.01, ppl_ratio_max=1.05, min_kl_readings=1):
    """Parse the verl trainer's stdout log for per-step metrics and assert bounds.

    Used by both colocated (``test_e2e_convergence_qwen25_05b_grpo_trtllm``)
    and standalone-async (``test_verl_E2E_standalone``) E2E tests.

      1. **Convergence (learning signal)**: ``max(critic/rewards/mean) > target``
         — the model has to show *any* learning signal during the short loop.
         Mirrors verl's own ``tests/special_e2e/check_results.py`` (which uses
         ``best_reward > args.target`` with default 0.2). We use a much lower
         target (0.01) because GSM8K reward is binary 0/1 and our gate runs
         only ~30 steps on Qwen2.5-0.5B with effective batch
         ``train_batch_size(16) × rollout.n(4) = 64``, so non-zero rewards
         can only be ``1/64=0.0156``, ``2/64=0.0313``, ``4/64=0.0625`` …
         Higher thresholds (e.g. ``> 0.05``) proved empirically flaky
         (heavy-tail success spikes), so we trade a tighter convergence
         claim for run-to-run reliability. **Set ``target=None`` to skip**
         (e.g. the standalone-async test where reward is not the focus).

      2. **Per-step PPL ratio near 1**: ``max(exp(|actor/ppo_kl|)) < ppl_ratio_max``
         — the trainer's policy must stay close to the rollout policy.
         ``actor/ppo_kl`` is verl's per-step ``mean(-log(ratio))`` where
         ``ratio = π_train / π_rollout``; the PPL ratio is ``exp(|kl|)``,
         interpretable as "by what factor does perplexity drift between
         trainer and rollout per step". Symmetric absolute value catches
         negative drift too. Default 1.05 leaves ~13× margin over the worst
         ``|kl|`` we've measured across our colocated Lyris dry-runs
         (``|kl|=3.8e-3`` ⇒ ratio=1.0038), absorbing run-to-run variance.
         The standalone-async path uses a tighter 1.01 bound because under
         bypass_mode (where ``old_log_probs`` come directly from rollout)
         any drift IS the regression to catch. A weight-sync / dtype /
         NaN-gradient regression typically drives ``|kl|`` to ~1.0+ on the
         first PPO step (ratio ≫ 2.7), so both bounds catch them.

      3. **Minimum kl readings**: ``len(kls) >= min_kl_readings`` —
         guards against degenerate runs where the trainer didn't emit
         enough PPO steps to assert on (e.g. verl's fully_async pipeline
         logs PPO metrics with one-step lag and may emit 0 readings if
         only one trainer step actually fires; bump ``min_kl_readings``
         to require multiple).

    Verl prints one summary line per step in the form
    ``step:N - key1:val1 - key2:val2 - …``. When the trainer runs as a Ray
    actor the line is prefixed by ``(TaskRunner pid=N)`` plus ANSI colour
    escapes, so we anchor at the ``step:N - `` substring via regex rather
    than assuming the line starts with ``step:`` (verl's own
    ``check_results.py`` uses ``startswith("step")`` and would miss
    Ray-prefixed lines).
    """
    import math

    rewards, kls = [], []
    with open(log_file) as fh:
        for line in fh:
            m = _STEP_LINE.search(line)
            if not m:
                continue
            for kv in line[m.start() :].split(" - "):
                try:
                    key, val = kv.strip().split(":", 1)
                except ValueError:
                    continue
                try:
                    f = float(val)
                except ValueError:
                    continue
                if key == "critic/rewards/mean":
                    rewards.append(f)
                elif key == "actor/ppo_kl":
                    kls.append(f)
    if target is not None:
        assert rewards, f"No critic/rewards/mean lines parsed from {log_file}"
        best = max(rewards)
        assert best > target, (
            f"Convergence target not met: best critic/rewards/mean={best:.4f} "
            f"<= {target}; see {log_file}"
        )
    assert len(kls) >= min_kl_readings, (
        f"Expected >= {min_kl_readings} actor/ppo_kl readings, got {len(kls)}: "
        f"{kls}; see {log_file}"
    )
    worst_kl = max(abs(kl) for kl in kls)
    worst_ratio = math.exp(worst_kl)
    assert worst_ratio < ppl_ratio_max, (
        f"PPL ratio out of band: max(exp(|actor/ppo_kl|))={worst_ratio:.4f} "
        f"(|kl|={worst_kl:.4f}) >= {ppl_ratio_max}; see {log_file}"
    )


# ---------------------------------------------------------------------------
# test_async_server.py wrappers
# ---------------------------------------------------------------------------


def test_placement_group_with_sub_ray_resource_pool():
    _run_single(_ASYNC_SERVER, "test_placement_group_with_sub_ray_resource_pool")


def test_placement_group_with_ray_resource_pool():
    _run_single(_ASYNC_SERVER, "test_placement_group_with_ray_resource_pool")


def test_placement_group_multi_node_ray_resource_pool():
    _run_single(_ASYNC_SERVER, "test_placement_group_multi_node_ray_resource_pool")


def test_placement_group_multi_node_multi_replica():
    _run_single(_ASYNC_SERVER, "test_placement_group_multi_node_multi_replica")


def test_async_generate():
    _run_single(_ASYNC_SERVER, "test_async_generate")


def test_async_memory_management():
    _run_single(_ASYNC_SERVER, "test_async_memory_management")


# ---------------------------------------------------------------------------
# test_adapter.py wrappers
# ---------------------------------------------------------------------------


def test_make_async_request_get_method():
    _run_single(_ADAPTER, "test_make_async_request_get_method")


def test_make_async_request_post_method():
    _run_single(_ADAPTER, "test_make_async_request_post_method")


def test_make_async_request_http_error():
    _run_single(_ADAPTER, "test_make_async_request_http_error")


def test_make_async_request_max_attempts_exceeded():
    _run_single(_ADAPTER, "test_make_async_request_max_attempts_exceeded")


def test_init_without_device_mesh():
    _run_single(_ADAPTER, "test_init_without_device_mesh")


# ---------------------------------------------------------------------------
# test_trtllm_rollout_utils.py wrappers  (900 s — multimodal cases are slow)
# ---------------------------------------------------------------------------


def test_unimodal_generate():
    _run_single(_ROLLOUT_UTILS, "test_unimodal_generate", timeout=900)


def test_unimodal_batch_generate():
    _run_single(_ROLLOUT_UTILS, "test_unimodal_batch_generate", timeout=900)


def test_multimodal_generate_with_image():
    _run_single(_ROLLOUT_UTILS, "test_multimodal_generate_with_image", timeout=900)


def test_multimodal_different_image_sizes():
    _run_single(_ROLLOUT_UTILS, "test_multimodal_different_image_sizes", timeout=900)


def test_multimodal_text_only_fallback():
    _run_single(_ROLLOUT_UTILS, "test_multimodal_text_only_fallback", timeout=900)


def test_wake_sleep_cycle():
    _run_single(_ROLLOUT_UTILS, "test_wake_sleep_cycle", timeout=900)


# ---------------------------------------------------------------------------
# test_inter_node_rollout.py wrappers  (900 s — multi-node)
# ---------------------------------------------------------------------------


def test_inter_node_trtllm_rollout():
    _run_single(_INTER_NODE, "test_inter_node_trtllm_rollout", timeout=900)


# ---------------------------------------------------------------------------
# test_trtllm_abort.py wrappers
# ---------------------------------------------------------------------------


def test_trtllm_abort():
    _run_single(_ABORT, "test_trtllm_abort")
