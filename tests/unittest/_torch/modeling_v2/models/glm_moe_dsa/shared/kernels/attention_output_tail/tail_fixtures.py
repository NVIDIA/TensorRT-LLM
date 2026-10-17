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
"""Real-weight fixtures for the attention_output_tail, read from a GLM-5.x checkpoint.

The TP slicing mirrors ``DeepseekV3WeightLoader``: ``kv_b_proj`` is unflattened
to ``[heads, qk_nope + v, kv_lora]``, chunked along heads per rank, and the
``v`` rows form ``v_b_proj [heads / tp, v, kv_lora]``; the ROW-parallel
``o_proj`` takes the rank's chunk of input features (columns).

``compressed_kv_rows`` and ``mix_latent`` build a latent attention output from
real layer weights (token embeddings → input RMSNorm → ``kv_a_proj_with_mqa``
→ ``kv_a_layernorm``), mixed by random softmax weights. It is a real-weight, synthetic-activation
fixture: the rows are convex combinations of real compressed-KV vectors, but
the attention probabilities are random, not model activations.
"""

import json
import os
from typing import Dict, Iterable, Tuple

import torch
from utils.llm_data import llm_models_root

GLM52_CKPT_ENV = "GLM52_NVFP4_CKPT"


def checkpoint_dir() -> str:
    """The GLM-5.2 NVFP4 checkpoint: ``$GLM52_NVFP4_CKPT``, else under ``llm_models_root()``."""
    path = os.environ.get(GLM52_CKPT_ENV)
    if path:
        return path
    root = llm_models_root()
    return os.path.join(str(root), "GLM-5.2-NVFP4") if root is not None else ""


def _index(ckpt_dir: str) -> Dict[str, str]:
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _read(
    ckpt_dir: str, weight_map: Dict[str, str], name: str, rows: slice = slice(None)
) -> torch.Tensor:
    from safetensors import safe_open

    path = os.path.join(ckpt_dir, weight_map[name])
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_slice(name)[rows].contiguous()


def read_tensor(ckpt_dir: str, name: str) -> torch.Tensor:
    """Read one checkpoint tensor by its safetensors name (CPU)."""
    return _read(ckpt_dir, _index(ckpt_dir), name)


def read_prefix(ckpt_dir: str, prefix: str) -> Dict[str, torch.Tensor]:
    """Read every checkpoint tensor whose name starts with ``prefix`` (CPU)."""
    from safetensors import safe_open

    weight_map = _index(ckpt_dir)
    by_file: Dict[str, list] = {}
    for name, fname in weight_map.items():
        if name.startswith(prefix):
            by_file.setdefault(fname, []).append(name)
    out = {}
    for fname, names in by_file.items():
        with safe_open(os.path.join(ckpt_dir, fname), framework="pt", device="cpu") as f:
            for name in names:
                out[name] = f.get_tensor(name)
    return out


def normed_embeddings(
    ckpt_dir: str, layer: int, first_token: int, num_tokens: int, device: str = "cuda"
) -> torch.Tensor:
    """``input_layernorm`` of ``layer`` over the embeddings of consecutive
    token IDs ``first_token ... first_token + num_tokens - 1`` (BF16)."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        eps = json.load(f).get("rms_norm_eps", 1e-6)
    weight_map = _index(ckpt_dir)
    emb = _read(
        ckpt_dir,
        weight_map,
        "model.embed_tokens.weight",
        slice(first_token, first_token + num_tokens),
    ).to(device)
    ln = _read(ckpt_dir, weight_map, f"model.layers.{layer}.input_layernorm.weight").to(device)
    return _rms_norm(emb, ln, eps).contiguous()


def load_tail_weights(
    ckpt_dir: str, layer: int, ranks: Iterable[int], tp_size: int = 8
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Load ``(v_b_proj [H/tp, V, C], o_proj [N, H*V/tp])`` per TP rank (CPU)."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    heads = cfg["num_attention_heads"]
    nope = cfg["qk_nope_head_dim"]
    v_dim = cfg["v_head_dim"]
    weight_map = _index(ckpt_dir)
    prefix = f"model.layers.{layer}.self_attn"
    kv_b = _read(ckpt_dir, weight_map, f"{prefix}.kv_b_proj.weight")
    kv_b = kv_b.unflatten(0, [heads, nope + v_dim])
    o_proj = _read(ckpt_dir, weight_map, f"{prefix}.o_proj.weight")
    out = {}
    for rank in ranks:
        kv_b_rank = torch.chunk(kv_b, tp_size, dim=0)[rank]
        _, v_rank = kv_b_rank.split([nope, v_dim], dim=1)
        o_rank = torch.chunk(o_proj, tp_size, dim=1)[rank]
        out[rank] = (v_rank.contiguous(), o_rank.contiguous())
    return out


def load_full_o_proj(ckpt_dir: str, layer: int) -> torch.Tensor:
    """Return the unsharded ``o_proj`` weight ``[N, H*V]`` (CPU)."""
    return _read(ckpt_dir, _index(ckpt_dir), f"model.layers.{layer}.self_attn.o_proj.weight")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def compressed_kv_rows(
    ckpt_dir: str, layer: int, context: int = 2048, token_offset: int = 1000, device: str = "cuda"
) -> torch.Tensor:
    """Return ``[context, kv_lora]`` normalized compressed-KV rows of ``layer``.

    Computed from the embeddings of token IDs ``token_offset`` to
    ``token_offset + context - 1``: input RMSNorm, the compressed part of
    ``kv_a_proj_with_mqa``, then ``kv_a_layernorm``.
    """
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    eps = cfg.get("rms_norm_eps", 1e-6)
    kv_lora = cfg["kv_lora_rank"]
    weight_map = _index(ckpt_dir)
    prefix = f"model.layers.{layer}"
    emb = _read(
        ckpt_dir,
        weight_map,
        "model.embed_tokens.weight",
        slice(token_offset, token_offset + context),
    ).to(device)
    ln = _read(ckpt_dir, weight_map, f"{prefix}.input_layernorm.weight").to(device)
    kv_a = _read(ckpt_dir, weight_map, f"{prefix}.self_attn.kv_a_proj_with_mqa.weight").to(device)
    kv_ln = _read(ckpt_dir, weight_map, f"{prefix}.self_attn.kv_a_layernorm.weight").to(device)
    hidden = _rms_norm(emb, ln, eps)
    compressed = torch.nn.functional.linear(hidden, kv_a)[:, :kv_lora]
    return _rms_norm(compressed, kv_ln, eps)


def mix_latent(
    compressed: torch.Tensor,
    num_rows: int,
    num_local_heads: int = 8,
    score_scale: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """Return a ``[num_rows, num_local_heads, kv_lora]`` BF16 latent.

    Each (row, head) is a softmax-weighted average of the ``compressed`` rows
    with random scores, like a sparse-attention output over a real cache.
    """
    device = compressed.device
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = (
        torch.randn(
            (num_rows, num_local_heads, compressed.shape[0]),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )
        * score_scale
    )
    probs = torch.softmax(scores, dim=-1)
    latent = torch.einsum("rht,tc->rhc", probs, compressed.float())
    return latent.to(torch.bfloat16).contiguous()
