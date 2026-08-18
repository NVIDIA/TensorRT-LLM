# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 SA (suffix automaton) spec-dec integration test (truncated model).

Runs the kimi_k3_sa_harness (same directory) on the first 4 layers (KDA +
the first MLA layer) with SA spec dec and logits-parity checking: baseline
and spec logprobs must agree along the shared output prefix (hard failure
on drift — the state-bug signature), while non-tie divergences only warn
(benign reduction-order rounding flips argmax on truncated-model noise
logits; see the harness docstring).

Requirements: 4 GPUs and the Kimi K3 checkpoint (env KIMI_K3_CKPT or
<LLM_MODELS_ROOT>/Kimi-K3). Skips cleanly when the
checkpoint is absent. The MoE backend defaults to VANILLA (the reference
dequant path — the bit-parity oracle; slow but fine at 4 layers) so the
test has no fused-kernel dependency and runs on any arch.
"""

import os
import subprocess
import sys

import pytest

from defs.conftest import llm_models_root
from defs.trt_test_alternative import print_info


def _find_checkpoint():
    ckpt = os.environ.get("KIMI_K3_CKPT")
    if ckpt and os.path.isdir(ckpt):
        return ckpt
    models_root = llm_models_root()
    if models_root:
        candidate = os.path.join(models_root, "Kimi-K3")
        if os.path.isdir(candidate):
            return candidate
    return None


@pytest.mark.skip_less_device(4)
def test_kimi_k3_sa_specdec_logits_parity():
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip(
            "Kimi K3 checkpoint not available (set KIMI_K3_CKPT or stage under LLM_MODELS_ROOT)"
        )

    env = os.environ.copy()
    env.update(
        {
            "KIMI_K3_CKPT": ckpt,
            "KIMI_K3_TP": "4",
            "KIMI_K3_NUM_LAYERS": "4",
            "KIMI_K3_SPEC_MODE": "sa",
            "KIMI_K3_SPEC_DRAFT_LEN": "2",
            "KIMI_K3_SPEC_PARITY": "logits",
            # ~50 trajectories make the aggregate divergence statistics
            # meaningful; loading dominates runtime so this is nearly free.
            "KIMI_K3_SPEC_NUM_PROMPTS": "48",
            # No KIMI_K3_MOE_BACKEND override: KimiK3MoERuntime pins the
            # routed MoE backend to TRTLLM regardless of moe_config.backend,
            # so a VANILLA default here would not take effect. Parity holds
            # because the baseline and spec runs share the same backend.
        }
    )
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kimi_k3_sa_harness.py")
    print_info(f"running {script} with ckpt={ckpt}")
    result = subprocess.run(
        [sys.executable, script], env=env, capture_output=True, text=True, timeout=1800
    )
    sys.stdout.write(result.stdout[-8000:])
    sys.stderr.write(result.stderr[-4000:])
    assert result.returncode == 0, "sanity harness reported FAIL"
    assert "[sanity] PASS" in result.stdout


def test_kimi_k3_disagg_parity_selftest():
    """Self-test of the two-endpoint (aggregated vs disagg) parity harness.

    Comparison logic only: canned responses, no servers or GPUs.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kimi_k3_disagg_parity.py")
    # Hard failure, not a skip: this test is CI-listed (l0_cpu), so a moved or
    # renamed harness must surface as a regression instead of a silent skip.
    assert os.path.exists(script), f"kimi_k3_disagg_parity.py harness missing at {script}"
    result = subprocess.run(
        [sys.executable, script, "--self-test"], capture_output=True, text=True, timeout=120
    )
    sys.stdout.write(result.stdout[-4000:])
    sys.stderr.write(result.stderr[-2000:])
    assert result.returncode == 0, "parity harness self-test FAILED"
    assert "[self-test] PASS" in result.stdout
