# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E convergence + IS-ratio sanity gates for verl GRPO training with TRT-LLM rollout.

Two test cases on Qwen2.5-0.5B-Instruct + GSM8K against the pinned verl
revision, sharing the ``_check_convergence`` parser/assertion helper:

  - ``test_e2e_convergence_qwen25_05b_grpo_trtllm`` — **sync / colocated**
    mode: actor + rollout share the same 4 GPUs; trainer ↔ rollout
    coupling is direct. Asserts ``max(critic/rewards/mean) > 0.01`` (any
    learning signal) and ``max(exp(|actor/ppo_kl|)) < 1.05`` (PPL ratio
    near 1, accommodates the legitimate per-step drift colocated training
    introduces). Empirical Lyris worst case: ratio ≤ 1.0038 (~13× margin).

  - ``test_e2e_convergence_qwen25_05b_grpo_trtllm_async_standalone`` —
    **async / standalone** mode: trainer (FSDP2) and rollout (TRT-LLM)
    occupy disjoint GPU pools, sync via ``run_fully_async_policy.sh``.
    Under ``bypass_mode=True`` (verl default) PPO's ``old_log_probs`` come
    directly from the rollout engine, so ``actor/ppo_kl`` directly measures
    ``KL(π_trainer || π_rollout)`` — a near-zero quantity unless the
    weight-sync regresses. Asserts ``max(exp(|actor/ppo_kl|)) < 1.01``
    (tighter than colocated: any drift IS the regression). Reward
    convergence is not the focus and is skipped via ``target=None``.
    Empirical worst case: ratio ≤ 1.0003 (~38× margin).
"""

import os
import subprocess
import tempfile

import pytest

from .test_verl_cases import (
    VERL_ROOT,
    _check_convergence,
    _dump_log_tail,
    _ensure_gsm8k,
    _run_verl_train,
)

# Min ``actor/ppo_kl`` readings the standalone test asserts on. verl's
# fully_async trainer logs PPO metrics with a one-step lag (step N's
# metrics are emitted at step N+1's weight-sync log call) and the very
# last step's metrics are never logged, so total trainer steps in this
# config = ``rollout.total_rollout_steps / ppo_mini_batch_size`` ⇒
# kl readings = trainer_steps - 1. With ``total_rollout_steps=24`` and
# ``ppo_mini_batch_size=8`` we get 3 trainer steps ⇒ 2 readings.
_STANDALONE_EXPECTED_STEPS = 2


@pytest.mark.timeout(1800)
def test_e2e_convergence_qwen25_05b_grpo_trtllm() -> None:
    """10-step GRPO on GSM8K + Qwen2.5-0.5B-Instruct + TRT-LLM rollout (colocated/sync)."""
    model = os.path.join(os.environ["TRTLLM_TEST_MODEL_PATH_ROOT"], "Qwen/Qwen2.5-0.5B-Instruct")
    data_dir = _ensure_gsm8k("/tmp/verl-data/gsm8k")
    log_file = _run_verl_train(
        [
            "algorithm.adv_estimator=grpo",
            f"actor_rollout_ref.model.path={model}",
            f"data.train_files=['{data_dir}/train.parquet']",
            f"data.val_files=['{data_dir}/test.parquet']",
            "data.train_batch_size=16",
            "data.max_prompt_length=512",
            # GSM8K chain-of-thought + final answer typically needs 200-400
            # tokens; at 128 we observed 98.4% of rollouts getting clipped
            # → 0 valid answers → 0 reward → 0 advantage → 0 gradient → policy
            # never updates (ppo_kl=0). 512 gives the 0.5B model room to
            # actually produce answers and learn.
            "data.max_response_length=512",
            "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.name=trtllm",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.n=4",
            "actor_rollout_ref.rollout.max_num_seqs=16",
            # Per-batch token budget; 1024 matches verl's own
            # e2e_ppo_grpo_trainer_trtllm.yml CI workflow and is more than
            # enough for our prompt(512) + response(512) sequence shape with
            # in-flight batching.
            "actor_rollout_ref.rollout.max_num_batched_tokens=1024",
            "actor_rollout_ref.rollout.max_model_len=1024",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "trainer.logger=['console']",
            "trainer.n_gpus_per_node=4",
            "trainer.nnodes=1",
            "trainer.total_training_steps=10",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
        ],
        timeout=1800,
    )
    _check_convergence(log_file, target=0.01, ppl_ratio_max=1.05)


@pytest.mark.timeout(1800)
def test_e2e_convergence_qwen25_05b_grpo_trtllm_async_standalone() -> None:
    """3-trainer-step GRPO + standalone async-RL (FSDP2 trainer + TRT-LLM rollout)."""
    model_path = os.path.join(
        os.environ["TRTLLM_TEST_MODEL_PATH_ROOT"], "Qwen/Qwen2.5-0.5B-Instruct"
    )
    data_dir = _ensure_gsm8k("/tmp/verl-data/gsm8k")
    # ``mkstemp`` rather than the deprecated ``mktemp`` to avoid the TOCTOU
    # race. Suffix matches ``_run_verl_train``'s naming so out-of-band log
    # rescue tooling that greps for ``*-verl-train.log`` picks this up too.
    fd, log_file = tempfile.mkstemp(prefix="standalone-", suffix="-verl-train.log")
    os.close(fd)

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
        # 3 trainer steps emit 2 ``actor/ppo_kl`` readings.
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
    if result.returncode != 0:
        _dump_log_tail(log_file, label=f"run_fully_async_policy.sh (exit {result.returncode})")
    assert result.returncode == 0, (
        f"run_fully_async_policy.sh exited {result.returncode}; see {log_file}"
    )

    # ``target=None`` skips reward check (bypass_mode test isn't about reward
    # convergence). ``min_kl_readings=_STANDALONE_EXPECTED_STEPS`` guards
    # against the fully_async one-step-lag emitting fewer readings than
    # expected (e.g. degenerate single-trainer-step runs).
    _check_convergence(
        log_file,
        target=None,
        ppl_ratio_max=1.01,
        min_kl_readings=_STANDALONE_EXPECTED_STEPS,
    )
