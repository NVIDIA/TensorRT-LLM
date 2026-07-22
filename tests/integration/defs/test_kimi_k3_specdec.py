# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 SA (suffix automaton) speculative-decoding integration test
(truncated model).

Runs the examples/kimi_k3 sanity harness on the first 4 layers (KDA + the
first MLA layer) with SA spec dec and logits-parity checking: baseline
and spec logprobs must agree along the shared output prefix (hard failure
on drift — the state-bug signature), while non-tie divergences only warn
(benign reduction-order rounding flips argmax on truncated-model noise
logits; see the harness docstring).

Requirements: 4 GPUs and the goldenprairie checkpoint (env KIMI_K3_CKPT or
<LLM_MODELS_ROOT>/goldenprairie-final-weights_vv1). Skips cleanly when the
checkpoint is absent. KIMI_K3_FUSED_MOE defaults to 0 here so the test does
not depend on the private FlashInfer snapshot (reference MoE is slow but
fine at 4 layers).
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
        candidate = os.path.join(models_root,
                                 "goldenprairie-final-weights_vv1")
        if os.path.isdir(candidate):
            return candidate
    return None


@pytest.mark.skip_less_device(4)
def test_kimi_k3_sa_specdec_logits_parity(llm_root):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("goldenprairie checkpoint not available "
                    "(set KIMI_K3_CKPT or stage under LLM_MODELS_ROOT)")

    env = os.environ.copy()
    env.update({
        "KIMI_K3_CKPT": ckpt,
        "KIMI_K3_TP": "4",
        "KIMI_K3_NUM_LAYERS_OVERRIDE": "4",
        "KIMI_K3_SPEC_MODE": "sa",
        "KIMI_K3_SPEC_DRAFT_LEN": "2",
        "KIMI_K3_SPEC_PARITY": "logits",
        # ~50 trajectories make the aggregate divergence statistics
        # meaningful; loading dominates runtime so this is nearly free.
        "KIMI_K3_SPEC_NUM_PROMPTS": "48",
        # Default to the in-tree reference MoE so the test has no
        # dependency on the private FlashInfer snapshot; opt in to the
        # fused path by exporting KIMI_K3_FUSED_MOE=1 + snapshot env.
        "KIMI_K3_FUSED_MOE": env.get("KIMI_K3_FUSED_MOE", "0"),
    })
    script = os.path.join(llm_root, "examples", "kimi_k3",
                          "sanity_kimi_k3.py")
    print_info(f"running {script} with ckpt={ckpt}")
    result = subprocess.run([sys.executable, script],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=1800)
    sys.stdout.write(result.stdout[-8000:])
    sys.stderr.write(result.stderr[-4000:])
    assert result.returncode == 0, "sanity harness reported FAIL"
    assert "[sanity] PASS" in result.stdout
