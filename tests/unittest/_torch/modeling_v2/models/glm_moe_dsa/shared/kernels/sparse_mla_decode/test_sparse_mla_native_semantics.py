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
"""Characterization of the native sparse MLA generation kernel (FP8 KV).

The native trtllm-gen kernel ships as a binary, so its rounding and masking
rules are measured here against FP64 models: which probability rounding it
feeds to P*V, whether it ignores index-list positions past a row's causal KV
length, what it writes for a row without valid entries, and whether repeated
indices count once per occurrence. sparse_mla_decode must reproduce whatever these tests find.
"""

import json
import os

import pytest
import torch
from sparse_mla_harness import (
    TOP_K,
    DsaGenHarness,
    bf16_round,
    e4m3_round,
    error_stats,
    fp16_round,
    oracle_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU",
)

PROB_MODELS = {
    "exact": {},
    "fp16_split256_step128": dict(prob_round=fp16_round, split=256, step=128),
    "bf16_split256_step128": dict(prob_round=bf16_round, split=256, step=128),
    "e4m3x1_split256_step128": dict(prob_round=e4m3_round(1.0), split=256, step=128),
    "e4m3x448_split256_step128": dict(prob_round=e4m3_round(448.0), split=256, step=128),
    "e4m3x256_split256_step128": dict(prob_round=e4m3_round(256.0), split=256, step=128),
    "e4m3x1_split2048_step128": dict(prob_round=e4m3_round(1.0), split=2048, step=128),
    "e4m3x448_split2048_step128": dict(prob_round=e4m3_round(448.0), split=2048, step=128),
    "e4m3x1_split128_step128": dict(prob_round=e4m3_round(1.0), split=128, step=128),
    "e4m3x448_split128_step128": dict(prob_round=e4m3_round(448.0), split=128, step=128),
}


def record(case: str, payload: dict) -> None:
    line = json.dumps({"case": case, **payload}, sort_keys=True)
    print(line)
    results_dir = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_RESULTS")
    if results_dir:
        with open(os.path.join(results_dir, "native_semantics.jsonl"), "a") as f:
            f.write(line + "\n")


def run_native(harness: DsaGenHarness, topk: torch.Tensor, seed: int = 1, q_std: float = 1.0):
    step = harness.make_step(0, seed=seed, q_std=q_std)
    harness.select_library(0, use_sparse_mla=False)
    out = harness.forward(0, step, topk).clone()
    global_idx = harness.global_indices(0, topk)
    return step, out, global_idx


@pytest.mark.parametrize("ctx_len,q_std", [(32768, 1.0), (32768, 3.0), (32768, 0.3), (4096, 1.0)])
def test_native_probability_rounding(ctx_len, q_std):
    harness = DsaGenHarness([ctx_len], seq_len_q=6, seed=11)
    try:
        topk = harness.random_topk(seed=3, overlap=0.7, recent=32)
        step, native, global_idx = run_native(harness, topk, seed=5, q_std=q_std)
        b1 = float(step.bmm1_scale[0])
        b2 = float(step.bmm2_scale[0])
        native = native.double()
        distances = {}
        for name, kwargs in PROB_MODELS.items():
            ref = oracle_attention(step.quant_q, harness.pool_rows(), global_idx, b1, b2, **kwargs)
            ref_bf16 = ref.to(torch.bfloat16).double()
            stats = error_stats(native, ref_bf16)
            stats["exact_match_frac"] = float(
                (native.reshape(ref_bf16.shape) == ref_bf16).double().mean().item()
            )
            distances[name] = stats
        best = min(distances, key=lambda n: distances[n]["nrms"])
        record(
            "prob_rounding",
            {
                "ctx_len": ctx_len,
                "q_std": q_std,
                "bmm1_scale": [float(x) for x in step.bmm1_scale.tolist()],
                "bmm2_scale": b2,
                "best": best,
                "distances": distances,
            },
        )
        assert distances["exact"]["nonfinite"] == 0
    finally:
        harness.shutdown()


def test_native_kv_length_clamp():
    """Valid entries placed after a row's causal length in the index list."""
    harness = DsaGenHarness([100], seq_len_q=6, seed=12)
    try:
        topk = harness.random_topk(seed=4)
        extra_positions = [700, 1500, TOP_K - 1]
        for row in range(harness.num_rows):
            for pos in extra_positions:
                topk[row, pos] = 3 + row
        step, native, global_idx = run_native(harness, topk, seed=6)
        b1 = float(step.bmm1_scale[0])
        b2 = float(step.bmm2_scale[0])
        limits = [harness.causal_len(r) for r in range(harness.num_rows)]
        clamped = oracle_attention(
            step.quant_q, harness.pool_rows(), global_idx, b1, b2, list_limit=limits
        )
        unclamped = oracle_attention(step.quant_q, harness.pool_rows(), global_idx, b1, b2)
        record(
            "kv_length_clamp",
            {
                "vs_clamped": error_stats(native.double(), clamped),
                "vs_unclamped": error_stats(native.double(), unclamped),
                "limits": limits,
            },
        )
    finally:
        harness.shutdown()


def test_native_invalid_and_duplicate_entries():
    harness = DsaGenHarness([8192], seq_len_q=6, seed=13)
    try:
        topk = harness.random_topk(seed=5)
        # Row 1: no valid entry. Row 2: invalid entries interleaved with valid ones.
        # Row 3: two tokens repeated with unequal multiplicity.
        topk[1, :] = -1
        topk[2, ::3] = -1
        topk[3, :1500] = 17
        topk[3, 1500:] = 4242
        step, native, global_idx = run_native(harness, topk, seed=7)
        b1 = float(step.bmm1_scale[0])
        b2 = float(step.bmm2_scale[0])
        exact = oracle_attention(step.quant_q, harness.pool_rows(), global_idx, b1, b2)
        dedup_idx = global_idx.clone()
        dedup_idx[3, 1:1500] = -1
        dedup_idx[3, 1501:] = -1
        dedup = oracle_attention(step.quant_q, harness.pool_rows(), dedup_idx, b1, b2)
        row1 = native[1].double()
        record(
            "invalid_and_duplicates",
            {
                "all_invalid_row": {
                    "nonfinite": int((~torch.isfinite(row1)).sum().item()),
                    "max_abs": float(row1.nan_to_num().abs().max().item()),
                },
                "interleaved_invalid_row": error_stats(native[2].double(), exact[2]),
                "duplicates_counted": error_stats(native[3].double(), exact[3]),
                "duplicates_dedup": error_stats(native[3].double(), dedup[3]),
            },
        )
    finally:
        harness.shutdown()
