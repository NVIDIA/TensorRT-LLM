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
"""Native oracle and fixtures shared by the draft_tail tests and benchmarks.

The oracle runs the native TensorRT-LLM code itself: ``F.linear`` (what
``UnquantizedLinearMethod`` runs for the BF16 LM head) per vocab shard, then
``SpecWorkerBase._get_local_max_and_combined`` per rank, the all-gather along
the last dimension (a concatenation in rank order, as ``allgather(dim=-1)``
returns) and the compiled ``SpecWorkerBase._get_draft_tokens_from_gathered``.
On one GPU the eight ranks are emulated shard by shard.
"""

import json
import os
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn.functional as F
from utils.llm_data import llm_models_root

CKPT = os.environ.get(
    "GLM_MOE_DSA_DRAFT_TAIL_CKPT",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
VOCAB = 154880
HIDDEN = 6144
TP = 8
SHARD = VOCAB // TP


def checkpoint_available() -> bool:
    return os.path.exists(os.path.join(CKPT, "model.safetensors.index.json"))


def _tensor_file(name: str) -> str:
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    return os.path.join(CKPT, index["weight_map"][name])


def load_lm_head_rows(begin: int, end: int, device="cuda") -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(_tensor_file("lm_head.weight"), framework="pt", device="cpu") as f:
        return f.get_slice("lm_head.weight")[begin:end].to(device).contiguous()


def load_lm_head_shards(tp: int = TP, device="cuda") -> List[torch.Tensor]:
    rows = VOCAB // tp
    full = load_lm_head_rows(0, VOCAB, device=device)
    return [full[r * rows : (r + 1) * rows].contiguous() for r in range(tp)]


def load_tensor(name: str, device="cuda") -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(_tensor_file(name), framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device)


def mtp_norm_weight(device="cuda") -> torch.Tensor:
    return load_tensor("model.layers.78.shared_head.norm.weight", device)


def rms_normalize(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * gamma.float()).to(torch.bfloat16)


def synthetic_hidden(
    rows: int,
    gamma: torch.Tensor,
    generator: torch.Generator,
    peaked_from: Optional[torch.Tensor] = None,
    peak_scale: float = 0.0,
) -> torch.Tensor:
    """Normed hidden rows as the MTP shared-head RMSNorm emits them.

    ``peaked_from`` (rows of the LM head) adds a component along chosen token
    directions so the logits have a clear winner, like real decoding states;
    without it the logits are flat and near-ties are frequent.
    """
    x = torch.randn(rows, HIDDEN, generator=generator, device="cuda", dtype=torch.float32)
    if peaked_from is not None:
        d = peaked_from.float()
        d = d / d.norm(dim=-1, keepdim=True)
        x = x + peak_scale * d * x.norm(dim=-1, keepdim=True)
    return rms_normalize(x, gamma)


class _Rank(SimpleNamespace):
    pass


def native_combined(logits: torch.Tensor, tp_rank: int) -> torch.Tensor:
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    return SpecWorkerBase._get_local_max_and_combined(
        None, logits, mapping_lm_tp=_Rank(tp_rank=tp_rank)
    )


def native_from_gathered(gathered: torch.Tensor) -> torch.Tensor:
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    return SpecWorkerBase._get_draft_tokens_from_gathered(None, gathered)


def native_tokens_from_logits(shard_logits: List[torch.Tensor]) -> torch.Tensor:
    """Native greedy TP draft tokens from per-rank BF16 vocab-shard logits."""
    combined = [native_combined(lg, r) for r, lg in enumerate(shard_logits)]
    return native_from_gathered(torch.cat(combined, dim=-1))


def native_tokens(hidden: torch.Tensor, shards: List[torch.Tensor]) -> torch.Tensor:
    return native_tokens_from_logits([F.linear(hidden, w) for w in shards])


def keys_to_tokens(keys: torch.Tensor) -> torch.Tensor:
    low = keys & 0xFFFFFFFF
    return (0xFFFFFFFF - low).to(torch.int32)


def global_key(keys_per_rank: List[torch.Tensor]) -> torch.Tensor:
    """Unsigned max of packed keys (all keys are < 2**63 here)."""
    stacked = torch.stack(keys_per_rank, dim=0)
    assert bool((stacked >= 0).all()), "keys must fit in int64 as non-negative"
    return stacked.max(dim=0).values


def fp64_logits(
    hidden: torch.Tensor, rows: torch.Tensor, weight_full_rows: torch.Tensor
) -> torch.Tensor:
    """Exact-ish logits of selected vocab rows (FP64 accumulation)."""
    return hidden.double() @ weight_full_rows.double().t()
