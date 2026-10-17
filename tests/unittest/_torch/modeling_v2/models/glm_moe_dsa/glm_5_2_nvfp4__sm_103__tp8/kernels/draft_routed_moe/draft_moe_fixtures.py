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
"""Real-weight fixtures and the native oracle for the draft_routed_moe tests.

The fixture is the MTP layer (``model.layers.78``, BF16 in the GLM-5.2 NVFP4
checkpoint) as TP rank ``r`` of TP8/EP1 sees it: every routed expert's
gate/up rows ``[256r, 256r+256)`` and down columns ``[256r, 256r+256)``, the
router weight and the BF16 router bias. The native module is the stock
FlashInfer TRT-LLM-Gen BF16 leaf (``trtllm_gen_leaf(None)``) loaded through its
own ``load_weights``/``post_load_weights``, so its buffers carry exactly the
shuffled BlockMajorK layouts of a served model. Slicing the checkpoint before
loading a ``tp_size=1`` leaf is what the TP8 loader's ``load_weight_shard``
does (COLUMN for w1/w3, ROW for w2).
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from utils.llm_data import llm_models_root

CKPT_DIR = os.environ.get(
    "GLM_MOE_DSA_DRAFT_ROUTED_MOE_CKPT_DIR",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
CACHE_DIR = os.environ.get(
    "GLM_MOE_DSA_DRAFT_ROUTED_MOE_FIXTURE_CACHE",
    os.path.join(tempfile.gettempdir(), "glm_moe_dsa_draft_routed_moe_fixtures"),
)

MTP_LAYER = 78
HIDDEN = 6144
INTER = 2048
NUM_EXPERTS = 256
TOP_K = 8
TP_SIZE = 8
ROUTED_SCALING = 2.5
RMS_EPS = 1e-5


class Reader:
    """Opens each safetensors shard once and slices tensors lazily."""

    def __init__(self):
        from safetensors import safe_open

        self._safe_open = safe_open
        with open(os.path.join(CKPT_DIR, "model.safetensors.index.json")) as f:
            self._map = json.load(f)["weight_map"]
        self._open = {}

    def _handle(self, name: str):
        shard = self._map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(
                os.path.join(CKPT_DIR, shard), framework="pt", device="cpu"
            )
        return self._open[shard]

    def full(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self._handle(name).get_slice(name)[start:stop].clone()

    def cols(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self._handle(name).get_slice(name)[:, start:stop].clone()


def load_rank(tp_rank: int, layer: int = MTP_LAYER) -> Dict[str, torch.Tensor]:
    """Rank-sliced raw tensors of the MTP layer's MoE (CPU, cached under ``CACHE_DIR``)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"glm52_L{layer}_bf16_tp{TP_SIZE}r{tp_rank}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    rd = Reader()
    inter_rank = INTER // TP_SIZE
    r0, r1 = tp_rank * inter_rank, (tp_rank + 1) * inter_rank
    pre = f"model.layers.{layer}.mlp"
    out: Dict[str, torch.Tensor] = {}
    for e in range(NUM_EXPERTS):
        out[f"{e}.w1.weight"] = rd.rows(f"{pre}.experts.{e}.gate_proj.weight", r0, r1)
        out[f"{e}.w3.weight"] = rd.rows(f"{pre}.experts.{e}.up_proj.weight", r0, r1)
        out[f"{e}.w2.weight"] = rd.cols(f"{pre}.experts.{e}.down_proj.weight", r0, r1)
    out["gate.weight"] = rd.full(f"{pre}.gate.weight")
    out["gate.e_score_correction_bias"] = rd.full(f"{pre}.gate.e_score_correction_bias")
    out["post_attention_layernorm.weight"] = rd.full(
        f"model.layers.{layer}.post_attention_layernorm.weight"
    )
    tmp = path + f".tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, path)
    return out


@dataclass
class NativeBf16Layer:
    """The native FlashInfer TRT-LLM-Gen BF16 MoE leaf of one TP rank plus its router."""

    module: torch.nn.Module
    gate_weight: torch.Tensor  # [256, 6144] bf16
    bias: torch.Tensor  # [256] bf16 (the TRTLLM gate dtype)
    norm_weight: torch.Tensor  # [6144] bf16
    tp_rank: int

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 router logits of ``x`` (FP32 GEMM; any FP32 logits are valid oracle inputs)."""
        return (x.float() @ self.gate_weight.float().t()).contiguous()

    def native(self, x: torch.Tensor, logits: torch.Tensor, do_finalize: bool = True):
        """``trtllm_bf16_moe`` exactly as ``FlashinferTrtllmGenBf16Impl.run_moe`` calls it."""
        from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.kernel_inputs import (
            to_trtllm_gen_act_type,
        )

        m = self.module
        return m.op_backend.run_bf16_moe(
            router_logits=logits,
            routing_bias=self.bias,
            hidden_states=x,
            gemm1_weights=m.w3_w1_weight,
            gemm2_weights=m.w2_weight,
            num_experts=m.num_slots,
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            intermediate_size=m.intermediate_size_per_partition,
            local_expert_offset=m.slot_start,
            local_num_experts=m.expert_size_per_partition,
            routed_scaling_factor=ROUTED_SCALING,
            routing_method_type=m.routing_method.routing_method_type,
            topk_weights=None,
            topk_ids=None,
            gated_act_type=to_trtllm_gen_act_type(m.activation_type),
            output=None,
            use_shuffled_weight=m.quant_method.use_shuffled_weight,
            weight_layout=m.quant_method.weight_layout,
            do_finalize=do_finalize,
        )


def build_native(
    tp_rank: int = 0, max_num_tokens: int = 128, identity_fc2: bool = False
) -> NativeBf16Layer:
    """The rank's native leaf.

    ``identity_fc2`` replaces every expert's down projection by ``W[i, i % 256] = 1`` so that
    ``gemm2_output[:, :256]`` is the native FC1 + SwiGLU intermediate (diagnostics only).
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import trtllm_gen_leaf
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    raw = load_rank(tp_rank)
    bias = raw["gate.e_score_correction_bias"].to(torch.bfloat16).cuda()
    routing = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    model_config = ModelConfig(
        quant_config=QuantConfig(),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        moe_backend="TRTLLM",
        max_num_tokens=max_num_tokens,
    )
    leaf_cls = trtllm_gen_leaf(None)
    module = leaf_cls(
        routing_method=routing,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER // TP_SIZE,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=model_config,
        init_load_balancer=False,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    )
    weights = {k: v for k, v in raw.items() if k[0].isdigit()}
    if identity_fc2:
        eye = torch.zeros(HIDDEN, INTER // TP_SIZE, dtype=torch.bfloat16)
        eye[torch.arange(HIDDEN), torch.arange(HIDDEN) % (INTER // TP_SIZE)] = 1.0
        weights = {k: (eye if k.endswith(".w2.weight") else v) for k, v in weights.items()}
    module.load_weights([weights])
    module.post_load_weights()
    module = module.cuda()
    return NativeBf16Layer(
        module=module,
        gate_weight=raw["gate.weight"].to(torch.bfloat16).cuda().contiguous(),
        bias=bias,
        norm_weight=raw["post_attention_layernorm.weight"].to(torch.bfloat16).cuda(),
        tp_rank=tp_rank,
    )


def hidden_rows(layer: NativeBf16Layer, num: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    """``num`` RMS-normalized Gaussian rows times the layer's post-attention norm weight (BF16)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(num, HIDDEN, device="cuda", generator=g) * scale
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (y * layer.norm_weight.float()).to(torch.bfloat16).contiguous()


def chain_like_rows(layer: NativeBf16Layer, num: int, seed: int) -> torch.Tensor:
    """hidden_rows with the distinct-expert count of the draft chain at R = 6.

    Independent rows select about 40 distinct experts, the chain's six draft rows about 31: rows 4 and 5 repeat
    rows 0 and 1. Other row counts are hidden_rows unchanged.
    """
    x = hidden_rows(layer, num, seed)
    if num == 6:
        x[4:] = x[:2]
    return x


def distinct_experts(layer: NativeBf16Layer, logits: torch.Tensor) -> int:
    """Distinct experts of the top-8 selection (sigmoid + correction bias), for reports."""
    scores = torch.sigmoid(logits.float()) + layer.bias.float()
    return int(scores.topk(8, dim=-1).indices.unique().numel())


def top8_sets(bias: torch.Tensor, logits: torch.Tensor) -> List[frozenset]:
    """Per row, the set of its top-8 experts (sigmoid + correction bias) as frozensets (for row selection)."""
    scores = torch.sigmoid(logits.float()) + bias.float().to(logits.device)
    return [frozenset(r) for r in scores.topk(TOP_K, dim=-1).indices.cpu().tolist()]


def select_distinct(
    sets: List[frozenset], rows: int, target: int, seed: int, steps: int = 600, sample: int = 256
) -> Tuple[List[int], int]:
    """Indices into ``sets`` (repeats allowed) of ``rows`` rows whose top-8 sets cover ``target`` distinct experts.

    Seeded local search: start from random rows, then repeatedly replace one row by the candidate (a random sample
    of the pool plus the selection's other rows, whose repeat lowers the count) that brings the union closest to the
    target. Returns ``(indices, achieved count)``; the count differs from the target only when the search does not
    reach it.
    """
    import random

    rng = random.Random(seed)
    n = len(sets)
    pick = [rng.randrange(n) for _ in range(rows)]

    def count(p):
        u = set()
        for i in p:
            u |= sets[i]
        return len(u)

    best, best_d = list(pick), count(pick)
    for _ in range(steps):
        d = count(pick)
        if abs(d - target) < abs(best_d - target):
            best, best_d = list(pick), d
        if d == target:
            return pick, d
        pos = rng.randrange(rows)
        cands = rng.sample(range(n), min(n, sample)) + [pick[j] for j in range(rows) if j != pos]
        scored = [
            (abs(count(pick[:pos] + [c] + pick[pos + 1 :]) - target), rng.random(), c)
            for c in cands
        ]
        pick[pos] = min(scored)[2]
    d = count(pick)
    if abs(d - target) < abs(best_d - target):
        best, best_d = list(pick), d
    return best, best_d


class RowPool:
    """Candidate rows and their top-8 sets, from which ``take`` draws rows with a chosen distinct-expert count."""

    def __init__(self, xs: torch.Tensor, sets: List[frozenset]):
        self.xs, self.sets = xs, sets

    @classmethod
    def gaussian(cls, layer: NativeBf16Layer, num: int = 1024, seed: int = 10_000) -> "RowPool":
        xs = hidden_rows(layer, num, seed)
        return cls(xs, top8_sets(layer.bias, layer.logits(xs)))

    def take(self, rows: int, target: int, seed: int) -> Tuple[torch.Tensor, int]:
        """``rows`` rows (repeats allowed) covering ``target`` distinct experts, and the count achieved."""
        idx, d = select_distinct(self.sets, rows, target, seed)
        return self.xs[torch.tensor(idx, device=self.xs.device)].contiguous(), d


def embedding_rows(layer: NativeBf16Layer, token_ids: List[int]) -> torch.Tensor:
    """RMSNorm(embedding) * post_attention_layernorm.weight for real token ids (BF16)."""
    rd = Reader()
    name = "model.embed_tokens.weight"
    sl = rd._handle(name).get_slice(name)
    emb = torch.stack([sl[t : t + 1][0] for t in token_ids]).cuda().float()
    y = emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (y * layer.norm_weight.float()).to(torch.bfloat16).contiguous()
