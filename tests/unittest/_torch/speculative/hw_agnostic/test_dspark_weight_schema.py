# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Executable spec for the DeepSeek-V4-Pro-DSpark draft (``mtp.*``) weight schema.

This pins the exact per-stage parameter layout the DSpark backbone (DSparkBlock x
n_mtp_layers) must define, so the weight loader (Phase 2/3) routes every checkpoint
tensor. It is derived from the V4-Pro reference (``inference/model.py``) and was
confirmed against the real ``model.safetensors.index.json``:

  - stage 0 only:    main_proj.{weight,scale}, main_norm.weight  (fp8 main_proj)
  - last stage only: norm.weight, markov_head.markov_w1.weight,
                     markov_head.markov_w2.weight, confidence_head.proj.weight,
                     hc_head_{fn,base,scale}
  - every stage:     a full V4 block (attn.*, attn_norm, ffn.{gate,experts.*,
                     shared_experts}, ffn_norm, hc_attn_*, hc_ffn_*)

If the real checkpoint is present (default path or $DSPARK_CKPT), the test also
validates the live index against this schema; otherwise that check is skipped.
"""

import json
import os
import re

import pytest

N_MTP_LAYERS = 3
N_ROUTED_EXPERTS = 384  # DeepSeek-V4-Pro
DEFAULT_CKPT = "/home/scratch.jonasl_gpu_1/DeepSeek-V4-Pro-DSpark"

# Per-stage canonical (expert index collapsed to "<i>") non-? key suffixes.
_ATTN = {
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.weight",
    "attn.wkv.scale",
    "attn.wo_a.weight",
    "attn.wo_a.scale",
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "attn.wq_a.weight",
    "attn.wq_a.scale",
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn_norm.weight",
}
_FFN = {
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.experts.<i>.w1.weight",
    "ffn.experts.<i>.w1.scale",
    "ffn.experts.<i>.w2.weight",
    "ffn.experts.<i>.w2.scale",
    "ffn.experts.<i>.w3.weight",
    "ffn.experts.<i>.w3.scale",
    "ffn_norm.weight",
}
_HC = {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"}
_STAGE0_ONLY = {"main_proj.weight", "main_proj.scale", "main_norm.weight"}
_LAST_ONLY = {
    "norm.weight",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
}


def expected_stage_suffixes(stage: int) -> set:
    s = set(_ATTN) | set(_FFN) | set(_HC)
    if stage == 0:
        s |= _STAGE0_ONLY
    if stage == N_MTP_LAYERS - 1:
        s |= _LAST_ONLY
    return s


def _canon(key: str) -> str:
    # mtp.<s>.ffn.experts.<i>.w1.weight -> ("<s>", "ffn.experts.<i>.w1.weight")
    m = re.match(r"mtp\.(\d+)\.(.+)", key)
    assert m, key
    stage, rest = m.group(1), m.group(2)
    rest = re.sub(r"ffn\.experts\.\d+\.", "ffn.experts.<i>.", rest)
    return stage, rest


def test_schema_internal_consistency():
    # Stage 0 carries the capture proj; only the last stage carries the heads.
    assert "main_proj.weight" in expected_stage_suffixes(0)
    assert "markov_head.markov_w1.weight" not in expected_stage_suffixes(0)
    assert "markov_head.markov_w1.weight" in expected_stage_suffixes(N_MTP_LAYERS - 1)
    assert "confidence_head.proj.weight" in expected_stage_suffixes(N_MTP_LAYERS - 1)
    # Middle stage is a plain V4 block (no capture proj, no heads).
    mid = expected_stage_suffixes(1)
    assert "main_proj.weight" not in mid and "norm.weight" not in mid


def _ckpt_index_path():
    root = os.environ.get("DSPARK_CKPT", DEFAULT_CKPT)
    idx = os.path.join(root, "model.safetensors.index.json")
    return idx if os.path.exists(idx) else None


@pytest.mark.skipif(
    _ckpt_index_path() is None, reason="DeepSeek-V4-Pro-DSpark checkpoint index not present"
)
def test_real_checkpoint_matches_schema():
    with open(_ckpt_index_path()) as f:
        weight_map = json.load(f)["weight_map"]
    mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
    assert mtp_keys, "no mtp.* keys in checkpoint index"

    by_stage: dict[str, set] = {}
    max_expert = -1
    for k in mtp_keys:
        stage, rest = _canon(k)
        by_stage.setdefault(stage, set()).add(rest)
        m = re.search(r"ffn\.experts\.(\d+)\.", k)
        if m:
            max_expert = max(max_expert, int(m.group(1)))

    assert sorted(by_stage) == [str(s) for s in range(N_MTP_LAYERS)], sorted(by_stage)
    assert max_expert + 1 == N_ROUTED_EXPERTS, f"expert count {max_expert + 1}"
    for stage in range(N_MTP_LAYERS):
        got = by_stage[str(stage)]
        want = expected_stage_suffixes(stage)
        assert got == want, (
            f"stage {stage} schema mismatch\n"
            f"  missing: {sorted(want - got)}\n  extra: {sorted(got - want)}"
        )
