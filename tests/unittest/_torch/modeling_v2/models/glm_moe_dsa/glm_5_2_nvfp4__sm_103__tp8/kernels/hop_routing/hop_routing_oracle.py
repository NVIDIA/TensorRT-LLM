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
"""Native routing oracle and boundary logits for the hop_routing tests.

``native_routing`` runs the served TRT-LLM-Gen routing runner (DeepSeekV3,
n_group = topk_group = 1, top_k = 8, routed scaling 2.5, bf16 bias) through
``trtllm::moe_topk_sort`` and reads the expert of every ``(t, k)`` back
through the permutation: ``expanded_idx_to_permuted_idx[t, k]`` lies in
tile ``p // TILE`` whose expert is ``tile_idx_to_expert_idx``. The weights
are the runner's bf16 ``token_final_scales`` in top-k order.

``boundary_rows`` builds FP32 logit rows whose native selection sits on an
exact tie or a one-ulp key difference at the 8th/9th boundary (same warp,
different warps, different clusters of the hop), plus NaN, +-Inf, -0.0,
huge and subnormal logits; each row is labelled.
"""

import os
import sys
from typing import List, Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
import routed_moe_fixtures as fx  # noqa: E402

NUM_EXPERTS = 256
TOP_K = 8
TILE = 8
ROUTED_SCALING = 2.5
DEEPSEEK_V3 = 2  # RoutingMethodType.DeepSeekV3


def native_routing(
    logits: torch.Tensor, bias_bf16: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(ids int32 [R, 8], weights bf16 [R, 8])`` of the native runner."""
    logits = logits.float().contiguous()
    out = torch.ops.trtllm.moe_topk_sort(
        logits,
        bias_bf16.contiguous(),
        NUM_EXPERTS,
        TOP_K,
        1,
        1,
        0,
        NUM_EXPERTS,
        ROUTED_SCALING,
        TILE,
        DEEPSEEK_V3,
    )
    tile_expert, tile_limit, exp_to_perm = out[0], out[1], out[2]
    scales = out[6]
    perm = exp_to_perm.long()
    tile = perm // TILE
    ids = tile_expert[tile]
    assert bool((perm < tile_limit[tile]).all()), "permuted index outside its tile"
    return ids.to(torch.int32), scales


def reference_routing(logits: torch.Tensor, bias_bf16: torch.Tensor):
    """The bit-faithful PyTorch model of routed_moe's routing (secondary check)."""
    return fx.routing_reference(logits.float(), bias_bf16)


def weight_bits(w_fp32: torch.Tensor) -> torch.Tensor:
    return w_fp32.contiguous().view(torch.int32)


def native_weight_bits(w_bf16: torch.Tensor) -> torch.Tensor:
    return w_bf16.float().contiguous().view(torch.int32)


def compare_routing(ids, weights, n_ids, n_w) -> Optional[str]:
    """None when ids are equal and weights are bit-identical (NaN included)."""
    if not torch.equal(ids, n_ids):
        bad = (ids != n_ids).any(-1).nonzero().flatten().tolist()
        return f"ids differ in rows {bad[:8]}: got {ids[bad[0]].tolist()} native {n_ids[bad[0]].tolist()}"
    a, b = weight_bits(weights), native_weight_bits(n_w)
    if not torch.equal(a, b):
        bad = (a != b).any(-1).nonzero().flatten().tolist()
        return f"weights differ in rows {bad[:8]}: got {weights[bad[0]].tolist()} native {n_w[bad[0]].float().tolist()}"
    if not torch.equal(weight_bits(weights.to(torch.bfloat16).float()), a):
        return "weights are not bf16-representable"
    return None


def _key(logit: float, bias_val: torch.Tensor) -> torch.Tensor:
    lg = torch.tensor([logit], dtype=torch.float32, device=bias_val.device)
    return fx._key_of(lg, bias_val)[0]


def _logit_for_sigmoid(s: float) -> float:
    return float(torch.logit(torch.tensor(s, dtype=torch.float64)))


LOW_LOGIT = -12.0


def boundary_rows(bias_bf16: torch.Tensor, seed: int = 0) -> List[Tuple[str, torch.Tensor]]:
    """Labelled boundary rows (each ``[256]`` FP32 on the bias's device).

    Every expert not placed explicitly gets a logit near ``LOW_LOGIT`` (key
    ~ its bias). With ``k_low`` the largest such key, the placed experts aim
    at keys ``k_low + 0.25`` (the 8th/9th pair) and ``k_low + 0.4 + ...``
    (seven clear winners), so only experts whose bias lets their sigmoid
    reach those keys are placed.
    """
    dev = bias_bf16.device
    b = bias_bf16.float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows: List[Tuple[str, torch.Tensor]] = []
    low_sig = torch.sigmoid(torch.tensor(LOW_LOGIT + 0.5)).item()
    k_low = float((b + low_sig).max().item())
    pair_key = k_low + 0.25
    win_key = k_low + 0.40
    reach = [
        e
        for e in range(NUM_EXPERTS)
        if 1e-3 < pair_key - float(b[e]) and win_key + 0.08 - float(b[e]) < 1 - 1e-3
    ]

    def base():
        return (torch.rand(NUM_EXPERTS, generator=g) * 0.5 + LOW_LOGIT).to(dev)

    def place_winners(row, avoid):
        winners = [e for e in reach if e not in avoid]
        perm = torch.randperm(len(winners), generator=g).tolist()
        for i, j in enumerate(perm[:7]):
            e = winners[j]
            row[e] = _logit_for_sigmoid(win_key + 0.01 * i - float(b[e]))

    if len(reach) >= 9:
        # 8th/9th boundary pairs (e1 < e2): same warp, different warps, different hop clusters.
        cand = sorted(reach)
        pairs = [
            (cand[0], cand[1]),
            (cand[0], cand[-1]),
            (cand[len(cand) // 2], cand[len(cand) // 2 + 1]),
            (cand[1], cand[-2]),
        ]
        same_warp = [(a, c) for a, c in zip(cand, cand[1:]) if a // 32 == c // 32]
        if same_warp:
            pairs.append(same_warp[0])
        cross_cluster = [(a, c) for a in cand for c in cand if a < 64 and c >= 192]
        if cross_cluster:
            pairs.append(cross_cluster[0])
        for e1, e2 in pairs:
            for offset, label in ((0, "tie"), (1, "e2+1ulp"), (-1, "e2-1ulp")):
                row = base()
                place_winners(row, (e1, e2))
                l1 = _logit_for_sigmoid(pair_key - float(b[e1]))
                k1 = _key(l1, b[e1])
                l2 = fx.logit_for_key(float(k1.item()), b[e2], offset)
                if l2 is None:
                    continue
                row[e1] = l1
                row[e2] = l2
                rows.append((f"{label}({e1},{e2})", row))
        # Exact ties inside the top 8 (order among equal keys) across warps and clusters.
        picks = [cand[(i * len(cand)) // 8] for i in range(8)]
        if len(set(picks)) == 8:
            row = base()
            k = _key(_logit_for_sigmoid(pair_key - float(b[picks[0]])), b[picks[0]])
            ok = True
            for e in picks:
                lg = fx.logit_for_key(float(k.item()), b[e])
                if lg is None:
                    ok = False
                    break
                row[e] = lg
            if ok:
                rows.append(("tie-top8", row))

    row = base()
    row[77] = float("nan")
    rows.append(("nan1", row))
    row = base()
    row[200] = float("nan")
    row[13] = float("nan")
    rows.append(("nan2", row))
    rows.append(("nan-all", torch.full((NUM_EXPERTS,), float("nan"), device=dev)))
    row = base()
    row[[1, 9, 100, 250]] = float("inf")
    row[[2, 3]] = float("-inf")
    rows.append(("inf", row))
    rows.append(("neginf-all", torch.full((NUM_EXPERTS,), float("-inf"), device=dev)))
    rows.append(("posinf-all", torch.full((NUM_EXPERTS,), float("inf"), device=dev)))
    rows.append(("zero-all", torch.zeros(NUM_EXPERTS, device=dev)))
    row = torch.zeros(NUM_EXPERTS, device=dev)
    row[::2] = -0.0
    rows.append(("negzero-mix", row))
    row = base()
    row[[4, 44, 144]] = 3.0e38
    row[[5, 55]] = -3.0e38
    row[[6, 66]] = 1.0e-42
    row[[7, 77]] = -1.0e-42
    rows.append(("huge-subnormal", row))
    for scale in (0.5, 2.0, 8.0):
        rows.append((f"randn*{scale}", (torch.randn(NUM_EXPERTS, generator=g) * scale).to(dev)))
    return rows
