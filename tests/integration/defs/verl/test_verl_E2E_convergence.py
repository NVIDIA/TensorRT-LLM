# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E smoke + IS-ratio sanity gate for verl GRPO training with TRT-LLM rollout.

Runs 30 GRPO training steps on Qwen2.5-0.5B-Instruct + GSM8K against the
pinned verl revision, parses the trainer's per-step stdout, and asserts:

  1. ``max(critic/rewards/mean) > 0.01`` — the model shows *any* learning
     signal (i.e. at least one step where it gets >0 correct out of 64).
     This is not a real convergence assertion — verl's own
     ``check_results.py`` uses 0.2 with much longer training. Calibrated
     instead to GSM8K's binary 0/1 reward × effective batch=64, so
     non-zero rewards are 1/64=0.0156, 2/64=0.0313, 4/64=0.0625, …;
     the threshold survives runs that only hit 1/64 once.
  2. ``max(actor/ppo_kl) < 0.1`` — the importance-sampling ratio
     ``π_train/π_rollout`` stays in ``[exp(-0.1), exp(0.1)] ≈ [0.905,
     1.105]`` (i.e. near 1 within ±10%). Catches weight-sync / dtype /
     NaN-gradient regressions on the trainer↔rollout boundary.

Empirical 20-step dry-runs on this config measured
``max(critic/rewards/mean) ∈ {0.0156, 0.0625}`` (heavy-tail spikes) and
``max(actor/ppo_kl) ≤ 5e-4`` — both bounds clear with significant margin.
"""

import os

import pytest

from .test_verl_cases import _check_convergence, _ensure_gsm8k, _run_verl_train


@pytest.mark.timeout(1800)
def test_e2e_convergence_qwen25_05b_grpo_trtllm():
    """30-step GRPO on GSM8K + Qwen2.5-0.5B-Instruct + TRT-LLM rollout."""
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
            # 16 seqs × (512 prompt + 512 response) = 16384 worst-case
            # batched tokens; bump from the trtllm rollout default (1024)
            # so generation isn't artificially serialized.
            "actor_rollout_ref.rollout.max_num_batched_tokens=16384",
            "actor_rollout_ref.rollout.max_model_len=1024",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "trainer.logger=['console']",
            "trainer.n_gpus_per_node=4",
            "trainer.nnodes=1",
            "trainer.total_training_steps=30",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
        ],
        timeout=1800,
    )
    _check_convergence(log_file, target=0.01, ppo_kl_max=0.1)
