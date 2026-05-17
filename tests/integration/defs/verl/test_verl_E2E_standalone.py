# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E weight-sync correctness guard for verl STANDALONE-mode async-RL.

Runs the verl-shipped ``tests/special_e2e/run_fully_async_policy.sh`` script
inside the cloned verl checkout for 10 GRPO steps on Qwen2.5-0.5B-Instruct,
asserting that the trainer-side PPO ratio (measured via ``actor/ppo_kl``)
stays within ±5% of 1 on **every** step — i.e. that the rollout→trainer
weight sync is bit-correct.

Why bypass_mode makes ppo_kl the right signal: the fully_async configs ship
``bypass_mode=True`` by default
(``verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml``,
verified at pin ``d324b01``). Under bypass_mode PPO's ``old_log_probs`` come
directly from the rollout engine, so ``actor/ppo_kl`` directly measures
``KL(π_trainer || π_rollout)``. With a correct weight sync the ratio is
theoretically exactly 1; floating-point + dtype-cast noise stays well under
±5%. A measurable weight-sync regression (broken parameter group, dtype
mismatch, dropped allreduce) drives ``actor/ppo_kl`` well above this
threshold on the very first step, so the test catches it within ~1 minute
of training start.

Tighter than ``test_e2e_convergence_qwen25_05b_grpo_trtllm``'s 0.1 bound
because that test runs in colocated mode where some legitimate per-step
policy drift is expected; this test runs with bypass_mode=True where any
drift IS the regression to catch.
"""

import os
import subprocess
import tempfile

import pytest

from .test_verl_cases import _STEP_LINE, VERL_ROOT, _ensure_gsm8k

# Per-step upper bound on ``actor/ppo_kl``.
#
# ``mean(-log(ratio)) < 0.05`` ↔ ``ratio ∈ [exp(-0.05), exp(0.05)] ≈
# [0.951, 1.051]`` (i.e. trainer policy within ±5% of rollout policy).
PPO_KL_MAX = 0.05

# How many actor/ppo_kl readings the test asserts on. verl's fully_async trainer
# logs PPO metrics with one-step lag (step N's metrics are emitted at step N+1's
# weight-sync log call), and the very last step's metrics are never logged. So
# total trainer steps needed = EXPECTED_STEPS + 1. We use EXPECTED_STEPS=2 to
# keep wall time on 4xB200 under ~15 min.
EXPECTED_STEPS = 2


def _parse_actor_ppo_kl(log_file):
    """Pull every per-step ``actor/ppo_kl`` value from the trainer console log.

    Anchors on the ``step:N - `` substring via the shared ``_STEP_LINE``
    regex rather than ``str.startswith("step:")``, so Ray-prefixed lines
    (``(TaskRunner pid=N) step:N - ...``) are also captured. Mirrors
    ``_check_convergence`` in ``test_verl_cases.py``.
    """
    values = []
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
                if key != "actor/ppo_kl":
                    continue
                try:
                    values.append(float(val))
                except ValueError:
                    continue
    return values


@pytest.mark.timeout(1800)
def test_verl_E2E_standalone():
    """STANDALONE async-RL + TRT-LLM rollout — per-step weight-sync guard."""
    model_path = os.path.join(
        os.environ["TRTLLM_TEST_MODEL_PATH_ROOT"], "Qwen/Qwen2.5-0.5B-Instruct"
    )
    data_dir = _ensure_gsm8k("/tmp/verl-data/gsm8k")
    # Suffix matches ``_run_verl_train``'s naming so any out-of-band log
    # rescue tooling that greps for ``*-verl-train.log`` (e.g. local Lyris
    # post-mortem scripts) picks this up too.
    log_file = tempfile.mktemp(prefix="standalone-", suffix="-verl-train.log")

    env = os.environ.copy()
    env.update(
        {
            # 4 GPUs: 2 for the FSDP2 trainer, 2 for the standalone TRT-LLM
            # rollout. The script splits NUM_GPUS evenly when N_GPUS_ROLLOUT /
            # N_GPUS_TRAINING are unset; we set them both for clarity.
            "NUM_GPUS": "4",
            "N_GPUS_ROLLOUT": "2",
            "N_GPUS_TRAINING": "2",
            "ACTOR_STRATEGY": "fsdp2",
            "ROLLOUT_NAME": "trtllm",
            "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
            "MODEL_PATH": model_path,
        }
    )

    script = os.path.join(VERL_ROOT, "tests/special_e2e/run_fully_async_policy.sh")
    overrides = [
        # Use the CI-resolved GSM8K parquets instead of the script's default
        # $HOME/data/gsm8k.
        f"data.train_files=['{data_dir}/train.parquet']",
        f"data.val_files=['{data_dir}/test.parquet']",
        # ``total_rollout_steps`` counts ROLLOUT PROMPTS (not trainer steps).
        # With ppo_mini_batch_size=8 → 8 prompts/trainer-step. 24 prompts ⇒
        # 3 trainer steps; combined with verl's one-step lag in metric
        # logging (step N logged at step N+1's sync, last step never logged),
        # 3 trainer steps emit 2 ``actor/ppo_kl`` readings == EXPECTED_STEPS.
        "rollout.total_rollout_steps=24",
        # Force trainer↔rollout weight sync every step (staleness=1, sync_step=1)
        # so ppo_kl is maximally sensitive to a broken sync on every iteration.
        "async_training.staleness_threshold=1",
        "async_training.trigger_parameter_sync_step=1",
        # Compact sequence lengths keep wall time under ~15 min on 4xB200.
        "data.max_prompt_length=512",
        "data.max_response_length=256",
        # Default ``rollout.n=16`` (responses per prompt) × 24 prompts = 384
        # generations — too expensive for the CI budget. n=4 still gives GRPO
        # enough samples for advantage estimation while keeping rollout time
        # ~4x lower.
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        # Skip the startup validation pass — saves ~1 min, doesn't affect the
        # ppo_kl signal we care about.
        "trainer.val_before_train=False",
    ]
    cmd = ["bash", script, *overrides]

    with open(log_file, "w") as fh:
        result = subprocess.run(
            cmd,
            cwd=VERL_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    assert result.returncode == 0, (
        f"run_fully_async_policy.sh exited {result.returncode}; see {log_file}"
    )

    values = _parse_actor_ppo_kl(log_file)
    assert len(values) >= EXPECTED_STEPS, (
        f"expected >= {EXPECTED_STEPS} actor/ppo_kl readings, got {len(values)}: "
        f"{values}; see {log_file}"
    )

    for step_idx, kl in enumerate(values[:EXPECTED_STEPS], start=1):
        assert kl < PPO_KL_MAX, (
            f"actor/ppo_kl out of band at step {step_idx}: {kl:.4f} >= {PPO_KL_MAX} "
            f"-- likely STANDALONE weight-sync regression. First {EXPECTED_STEPS} "
            f"readings: {values[:EXPECTED_STEPS]}; full log: {log_file}"
        )
