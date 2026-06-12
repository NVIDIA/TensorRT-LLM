# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E feature-combination smoke gate for verl + TRT-LLM rollout.

Exercises the most-feature-loaded recipe currently shipped by verl —
``examples/grpo_trainer/run_qwen3_8b_megatron.sh`` — with Megatron
(TP=2/PP=1) actor + TRT-LLM rollout (TP=2) on Qwen2.5-7B-Instruct, runs
3 training steps, and passes when the subprocess exits 0.

Why 3 steps (not 1):
  - step 1 covers the cold-start path: resource pool, Megatron init,
    initial weight transfer trainer→rollout, first rollout, first PPO update.
  - step 2 covers the post-update path: weight resync after the actor
    has been updated, second rollout with refreshed weights.
  - step 3 covers steady state: confirms it's not a "only the first
    iteration works" regression.

Purpose is a smoke gate over the Megatron-actor + TRT-LLM-rollout
combination. No per-step metric assertion — that's the convergence-style
gate in ``test_verl_E2E_convergence.py``.

The shell recipe defaults to ``NGPUS_PER_NODE=8``; the verl CI stage runs
on 4 GPUs (DGX_B200-4_GPUs-Verl-Post-Merge-1), so we override to 4 +
``NNODES=1`` to keep the resource pool sizeable.
"""

import os
import subprocess

import pytest

from .test_verl_cases import VERL_ROOT, _ensure_gsm8k


@pytest.mark.timeout(1500)
def test_e2e_smoke_qwen25_7b_megatron_trtllm():
    """1-step Megatron(TP=2,PP=1) GRPO + Qwen2.5-7B + TRT-LLM rollout."""
    model = os.path.join(os.environ["TRTLLM_TEST_MODEL_PATH_ROOT"], "Qwen/Qwen2.5-7B-Instruct")
    data_dir = _ensure_gsm8k("/tmp/verl-data/gsm8k")
    env = os.environ.copy()
    # The recipe builds its TRAINER array from these env vars; default is
    # NGPUS_PER_NODE=8 which makes verl's resource pool manager refuse to
    # allocate ("Total available GPUs 4 < total desired GPUs 8") on the
    # 4-GPU verl post-merge stage.
    env.update(
        {
            "ACTOR_TP": "2",
            "ACTOR_PP": "1",
            "ROLLOUT_TP": "2",
            "NGPUS_PER_NODE": "4",
            "NNODES": "1",
            "INFER_BACKEND": "trtllm",
            "MODEL_PATH": model,
        }
    )
    cmd = [
        "bash",
        os.path.join(VERL_ROOT, "examples/grpo_trainer/run_qwen3_8b_megatron.sh"),
        "trainer.total_training_steps=3",
        "data.train_batch_size=32",
        "data.max_prompt_length=128",
        "data.max_response_length=64",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.max_num_batched_tokens=1024",
        "actor_rollout_ref.rollout.max_model_len=256",
        f"data.train_files=['{data_dir}/train.parquet']",
        f"data.val_files=['{data_dir}/test.parquet']",
        "trainer.logger=['console']",
    ]
    result = subprocess.run(cmd, cwd=VERL_ROOT, env=env, timeout=1500)
    assert result.returncode == 0, f"run_qwen3_8b_megatron.sh exited {result.returncode}"
