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
"""Real-weight sparse_attention contract inputs from GLM-5.2 layer 0 on real text.

Layer 0 is the one target layer whose input is fully determined by the token
ids (embedding + input RMSNorm), so its attention inputs are exactly the
model's. The projections use the checkpoint's BF16 attention weights for the
TP rank 0 head slice; RoPE, the FP8 latent cache and the FP8 query come from
the native TRT-LLM ops driven by ``DsaGenHarness``. Two top-k sources exist:

* ``indexer_topk``: the native indexer projections (``Indexer.pre_indexer_proj``)
  with the checkpoint's indexer weights, followed by a *reference* FP32
  evaluation of the indexer logits ``sum_h w_h * relu(q_h . k) * k_scale`` and an
  exact top-k (not the native top-k kernels);
* ``indexer_projections`` + ``DsaGenHarness.forward_native_indexer``: the native
  indexer K cache written by the native ``_update_k_cache`` and the unmodified
  native ``Indexer.forward_from_projected`` (paged MQA logits + decode top-k).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from utils.llm_data import llm_models_root

import tensorrt_llm

NUM_LOCAL_HEADS = 8
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
KV_LORA = 512
NUM_HEADS = 64
RMS_EPS = 1e-5
CHECKPOINT_NAME = "GLM-5.2-NVFP4"


def checkpoint_dir() -> Optional[Path]:
    root = llm_models_root()
    if root is None:
        return None
    path = Path(root) / CHECKPOINT_NAME
    return path if (path / "config.json").is_file() else None


def load_tensors(ckpt: Path, names: Iterable[str], device: str = "cuda") -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: Dict[str, list] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(ckpt / shard), framework="pt", device=device) as f:
            for name in shard_names:
                tensors[name] = f.get_tensor(name)
    return tensors


def real_token_ids(ckpt: Path, count: int) -> torch.Tensor:
    """Token ids of real text: the installed TensorRT-LLM Python sources, in path order."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    ids = []
    for path in sorted(Path(tensorrt_llm.__file__).resolve().parent.rglob("*.py")):
        ids.extend(tokenizer.encode(path.read_text(errors="ignore")).ids)
        if len(ids) >= count:
            break
    if len(ids) < count:
        raise ValueError(f"only {len(ids)} tokens of text available, {count} requested")
    return torch.tensor(ids[:count], dtype=torch.long)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


@dataclass
class LayerInputs:
    """Token-wise projections of one layer for every token of the sequence."""

    hidden: torch.Tensor  # [N, 6144] post input-RMSNorm
    qr: torch.Tensor  # [N, 2048] post q_a RMSNorm
    fused_q: torch.Tensor  # [N, 8 * 576] absorbed q_nope | raw q_pe
    q_pe: torch.Tensor  # [N, 8, 64] raw q_pe
    latent: torch.Tensor  # [N, 576] normalized latent | raw k_pe


def layer_weight_names(layer: int) -> list:
    prefix = f"model.layers.{layer}."
    attn = prefix + "self_attn."
    return [
        "model.embed_tokens.weight",
        prefix + "input_layernorm.weight",
        attn + "q_a_proj.weight",
        attn + "q_a_layernorm.weight",
        attn + "q_b_proj.weight",
        attn + "kv_a_proj_with_mqa.weight",
        attn + "kv_a_layernorm.weight",
        attn + "kv_b_proj.weight",
        attn + "indexer.wq_b.weight",
        attn + "indexer.wk.weight",
        attn + "indexer.k_norm.weight",
        attn + "indexer.k_norm.bias",
        attn + "indexer.weights_proj.weight",
    ]


@torch.inference_mode()
def project_layer0(weights: Dict[str, torch.Tensor], ids: torch.Tensor) -> LayerInputs:
    """Layer-0 attention inputs of TP rank 0 (heads 0-7) for every token."""
    attn = "model.layers.0.self_attn."
    hidden = rms_norm(
        weights["model.embed_tokens.weight"][ids.cuda()],
        weights["model.layers.0.input_layernorm.weight"],
    )
    kv = hidden @ weights[attn + "kv_a_proj_with_mqa.weight"].T
    compressed = rms_norm(kv[:, :KV_LORA], weights[attn + "kv_a_layernorm.weight"])
    k_pe = kv[:, KV_LORA:]
    qr = rms_norm(
        hidden @ weights[attn + "q_a_proj.weight"].T, weights[attn + "q_a_layernorm.weight"]
    )
    q_b = weights[attn + "q_b_proj.weight"][: NUM_LOCAL_HEADS * (QK_NOPE + QK_ROPE)]
    q = (qr @ q_b.T).view(-1, NUM_LOCAL_HEADS, QK_NOPE + QK_ROPE)
    q_nope, q_pe = q.split([QK_NOPE, QK_ROPE], dim=-1)
    kv_b = weights[attn + "kv_b_proj.weight"].view(NUM_HEADS, QK_NOPE + V_HEAD, KV_LORA)
    w_uk = kv_b[:NUM_LOCAL_HEADS, :QK_NOPE, :]
    q_absorbed = torch.einsum("nhd,hdk->nhk", q_nope.float(), w_uk.float()).to(torch.bfloat16)
    fused_q = torch.cat([q_absorbed, q_pe], dim=-1).reshape(
        -1, NUM_LOCAL_HEADS * (KV_LORA + QK_ROPE)
    )
    return LayerInputs(
        hidden=hidden.contiguous(),
        qr=qr.contiguous(),
        fused_q=fused_q.contiguous(),
        q_pe=q_pe.contiguous(),
        latent=torch.cat([compressed, k_pe], dim=-1).contiguous(),
    )


def load_indexer_weights(indexer, weights: Dict[str, torch.Tensor], layer: int = 0) -> None:
    prefix = f"model.layers.{layer}.self_attn.indexer."
    with torch.no_grad():
        for name, module_param in (
            ("wq_b.weight", indexer.wq_b.weight),
            ("wk.weight", indexer.wk.weight),
            ("weights_proj.weight", indexer.weights_proj.weight),
            ("k_norm.weight", indexer.k_norm.weight),
            ("k_norm.bias", indexer.k_norm.bias),
        ):
            module_param.data = (
                weights[prefix + name].to(device="cuda", dtype=module_param.dtype).contiguous()
            )
    indexer.cache_derived_state()


@torch.inference_mode()
def reference_indexer_logits(indexer, inputs: LayerInputs, ctx_len: int, seq_len_q: int) -> list:
    """FP32 indexer logits ``sum_h w_h * relu(q_h . k) * k_scale`` of each generation row.

    Uses the native indexer projections; row ``r`` covers its ``ctx_len + r + 1`` visible tokens.
    """
    num_tokens = ctx_len + seq_len_q
    positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    q_fp8, k_fp8, k_scale, weights, _ = indexer.pre_indexer_proj(
        inputs.qr[:num_tokens], inputs.hidden[:num_tokens], positions
    )
    q = q_fp8[ctx_len:num_tokens].float().reshape(seq_len_q, indexer.n_heads, indexer.head_dim)
    keys = (
        k_fp8.float().reshape(num_tokens, indexer.head_dim)
        * k_scale.float().reshape(num_tokens, -1)[:, :1]
    )
    head_weights = weights[ctx_len:num_tokens].float().reshape(seq_len_q, indexer.n_heads)
    logits = []
    for row in range(seq_len_q):
        visible = ctx_len + row + 1
        logits.append(
            (head_weights[row][:, None] * torch.relu(q[row] @ keys[:visible].T)).sum(dim=0)
        )
    return logits


@torch.inference_mode()
def indexer_topk(
    indexer, inputs: LayerInputs, ctx_len: int, seq_len_q: int, topk: int = 2048
) -> torch.Tensor:
    """Reference local top-k of the generation rows: exact top-k of the FP32 reference logits."""
    out = torch.full((seq_len_q, topk), -1, dtype=torch.int32, device="cuda")
    for row, logits in enumerate(reference_indexer_logits(indexer, inputs, ctx_len, seq_len_q)):
        count = min(topk, logits.numel())
        out[row, :count] = logits.topk(count).indices.to(torch.int32)
    return out


def selection_vs_reference(topk_local: torch.Tensor, logits: list, topk: int = 2048) -> list:
    """How a selection ranks under the FP32 reference logits, per row.

    ``kth`` is the reference top-k boundary value; ``ties_at_kth`` counts visible tokens whose
    reference logit equals it; ``selected_at_or_above_kth`` is the fraction of the selection whose
    reference logit is at least ``kth`` (1.0 for any exact top-k up to tie order), and
    ``selected_within_1pct`` the same with a tolerance of 1 % of the logit range.
    """
    rows = []
    for r, row_logits in enumerate(logits):
        count = min(topk, row_logits.numel())
        kth = row_logits.topk(count).values[-1]
        ids = topk_local[r][topk_local[r] >= 0].long()
        chosen = row_logits[ids]
        span = (row_logits.max() - row_logits.min()).clamp_min(1e-30)
        rows.append(
            {
                "kth": float(kth),
                "max": float(row_logits.max()),
                "ties_at_kth": int((row_logits == kth).sum()),
                "zero_logits": int((row_logits == 0).sum()),
                "selected_at_or_above_kth": float((chosen >= kth).float().mean()),
                "selected_within_1pct": float((chosen >= kth - 0.01 * span).float().mean()),
            }
        )
    return rows


@torch.inference_mode()
def indexer_projections(indexer, inputs: LayerInputs, start: int, end: int) -> list:
    """Native ``pre_indexer_proj`` of tokens ``[start, end)``: ``[q_fp8, k_fp8, k_scale, w, q_scale]``."""
    positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
    return list(indexer.pre_indexer_proj(inputs.qr[start:end], inputs.hidden[start:end], positions))


def topk_contract(topk_local: torch.Tensor, ctx_len: int) -> list:
    """Per generation row: where the local top-k holds negative, future and repeated ids.

    Row ``r`` of a verify window sees ``ctx_len + r + 1`` tokens; ``negative_below_causal``
    counts ``-1`` entries at list positions inside that causal length (where the native
    sparse MLA kernel reads the row ``-1`` addresses and sparse_mla_decode skips the entry).
    """
    rows = []
    for r in range(topk_local.shape[0]):
        causal = ctx_len + r + 1
        row = topk_local[r].long()
        head = row[: min(causal, row.numel())]
        valid = head[head >= 0]
        rows.append(
            {
                "causal_len": causal,
                "negative_below_causal": int((head < 0).sum()),
                "negative_total": int((row < 0).sum()),
                "ids_at_or_past_causal": int((row >= causal).sum()),
                "duplicates": int(valid.numel() - valid.unique().numel()),
                "valid": int((row >= 0).sum()),
            }
        )
    return rows


def selection_vs_kernel_scores(
    topk_local: torch.Tensor, topk_inputs: dict, reference_logits: list, topk: int = 2048
) -> list:
    """The native decode scores the top-k kernel received, against its output and the reference.

    Row ``r`` of a verify window of ``next_n`` rows scores ``seq_len - next_n + r + 1`` tokens
    (the native decode top-k rule). ``exact_agreement``: overlap of the selection with the exact
    top-k of those scores; ``scores_reference_corr``: Pearson correlation of the scores with the
    FP32 reference logits of the same tokens.
    """
    scores = topk_inputs["scores"].float()
    seq_len = int(topk_inputs["sequence_lengths"].reshape(-1)[0])
    next_n = int(topk_inputs["next_n"])
    rows = []
    for r in range(topk_local.shape[0]):
        visible = seq_len - next_n + r + 1
        row_scores = scores[r, :visible]
        count = min(topk, visible)
        exact = set(row_scores.topk(count).indices.tolist())
        chosen = set(x for x in topk_local[r].tolist() if x >= 0)
        ref = reference_logits[r][:visible].float()
        corr = torch.corrcoef(torch.stack([row_scores, ref]))[0, 1]
        rows.append(
            {
                "scores_shape": list(scores.shape),
                "visible": visible,
                "exact_agreement": len(chosen & exact) / max(len(exact), 1),
                "scores_reference_corr": float(corr),
            }
        )
    return rows


def recent_window_topk(num_tokens: int, topk: int = 2048) -> torch.Tensor:
    """Causal context selection used only to run the native context step."""
    rows = torch.arange(num_tokens, dtype=torch.int32, device="cuda")[:, None]
    cols = torch.arange(topk, dtype=torch.int32, device="cuda")[None, :]
    return (rows - cols).clamp_(min=-1)
