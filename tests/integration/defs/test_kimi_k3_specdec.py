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
<LLM_MODELS_ROOT>/Kimi-K3). Fails — deliberately does not skip — when the
checkpoint is absent: the test is CI-listed (GB300 post-merge), and a
checkpoint that vanishes from the runners' models mount must surface as a
regression rather than an indistinguishable green skip. The MoE backend is
routed to TRTLLM by KimiK3MoERuntime regardless of any
KIMI_K3_MOE_BACKEND / moe_config.backend override (see the comment on the
env block below); parity holds because the baseline and spec runs share
the same backend.
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


_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _find_lfs_pointer_files(ckpt):
    """Top-level checkpoint files that are still git-lfs pointers.

    The checkpoint is staged from a git-lfs clone; a models mirror that has
    not been hydrated (or has lagged the hydrated source) serves ~130-byte
    pointer files instead of the real blobs, and the resulting failures are
    deep and misleading (e.g. tiktoken parsing the pointer text as a vocab).
    """
    offenders = []
    for name in sorted(os.listdir(ckpt)):
        path = os.path.join(ckpt, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                head = f.read(len(_LFS_MAGIC))
        except OSError:
            continue
        if head == _LFS_MAGIC:
            offenders.append(name)
    return offenders


@pytest.mark.skip_less_device(4)
def test_kimi_k3_sa_specdec_logits_parity():
    ckpt = _find_checkpoint()
    # Hard failure, not a skip: on the post-merge stage a skip is
    # indistinguishable from a pass, so a checkpoint dropped from the
    # runners' models mount would silently end this coverage.
    assert ckpt is not None, (
        "Kimi K3 checkpoint not found (set KIMI_K3_CKPT or stage under LLM_MODELS_ROOT)"
    )
    lfs_pointers = _find_lfs_pointer_files(ckpt)
    assert not lfs_pointers, (
        f"Kimi K3 checkpoint at {ckpt} is not hydrated on this runner's models "
        f"mirror — these files are still git-lfs pointers: {lfs_pointers}. "
        f"This is a checkpoint-staging/mirror-sync problem, not a code failure; "
        f"re-sync the mirror or 'git lfs pull' the staging copy."
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
