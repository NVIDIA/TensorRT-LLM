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
"""All-reduce sites of the DSpark MLA drafter on pushed MNNVL all-reduces.

``TLLM_DSPARK_AR_PUSH=1`` (with the drafter's fused all-reduce, ``TLLM_DSPARK_DRAFT_FUSION=1``):

* o_proj and down_proj run as the low-M BF16 split-K GEMM whose epilogue stores the partial
  straight into the Lamport slots of the all-reduce (``dspark_draft_gemm_push``), and one site
  kernel finishes the all-reduce, adds the residual and applies the following RMSNorm
  (``dspark_draft_ar.DraftArNormKernel``). The site kernel is launched behind the GEMM with PDL and
  polls while the GEMM runs; its outputs are bitwise the MNNVL fused all-reduce
  (``RESIDUAL_RMS_NORM``) of the GEMM output. The GEMM is the split-K kernel at the tactics below
  instead of the module's GEMM, i.e. a different FP32 accumulation order of the partial (drafter
  rounding only).
* the embedding of the draft block: the owner rank of each token multicasts its table row and every
  rank polls it (``DraftEmbedKernel``), bitwise the vocab-sharded lookup + all-reduce.

A site that does not apply (more than 64 rows, no single-load launch shape, no MNNVL workspace view
during capture) returns None and the caller keeps its path.
"""

import os
from typing import Optional, Tuple

import torch

DRAFT_AR_PUSH_ENV_VAR = "TLLM_DSPARK_AR_PUSH"


def draft_ar_push_enabled() -> bool:
    """Whether ``TLLM_DSPARK_AR_PUSH=1`` selects the pushed drafter all-reduce sites."""
    return os.environ.get(DRAFT_AR_PUSH_ENV_VAR, "0") == "1"


# Split-K tactics (mma_m, mma_n, split_k, ab_stages) of the push GEMM per (role, max rows); the
# first entry whose row bound covers the call applies. Rubin, TP8 drafter shapes: o_proj
# 7168 x 1024, down_proj 7168 x 1792. TP4 sweep, chain of 10 sites (GEMM + site) with cold
# weights, us per site at 7 / 14 / 28 rows: split 1 at mma_n = the row tile is best for both roles
# (o_proj 6.92 / 7.43 / 10.03, down 7.12 / 7.21 / 10.09; split 2: 7.60 / 7.90 / 11.16; cuBLAS +
# C++ fused all-reduce: 9.76 / 8.83 / 10.86).
_PUSH_TACTICS = {
    "o_proj": (
        (8, (64, 8, 1, 0)),
        (16, (64, 16, 1, 0)),
        (32, (64, 32, 1, 0)),
        (64, (64, 64, 1, 0)),
    ),
    "down": ((8, (64, 8, 1, 0)), (16, (64, 16, 1, 0)), (32, (64, 32, 1, 0)), (64, (64, 64, 1, 0))),
}


def push_tactic(role: str, m: int, n: int, k: int, table=None):
    """The ``SplitKTactic`` of the push GEMM for ``role`` at ``m`` rows, or None. A table entry
    with zero stages takes the deepest pipeline that fits."""
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    from ..kernels.blackwell.dspark_draft_gemm_push import validate_push_tactic

    entry = next((t for rows, t in (table or _PUSH_TACTICS).get(role, ()) if m <= rows), None)
    if entry is None:
        return None
    mma_m, mma_n, split_k, stages = entry
    for ab_stages in [stages] if stages else range(12, 0, -1):
        tactic = SplitKTactic(mma_m, mma_n, split_k, ab_stages)
        try:
            validate_push_tactic(tactic, m, n, k)
        except ValueError:
            continue
        return tactic
    return None


class DraftArSites:
    """The pushed all-reduce sites of one drafter, on its o_proj ``AllReduce`` (whose MNNVL
    workspace every drafter all-reduce shares)."""

    def __init__(self, all_reduce):
        self.all_reduce = all_reduce

    def linear_ar_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        role: str,
        residual: torch.Tensor,
        norm,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``norm(all_reduce(x @ weight^T) + residual)`` and the new residual, or None if the site
        does not apply (the caller then runs its own path)."""
        from ..flashinfer_utils import get_env_enable_pdl
        from ..kernels.blackwell.dspark_draft_ar import ar_norm_config, run_draft_ar_norm
        from ..kernels.blackwell.dspark_draft_gemm_push import run_gemm_push

        if (
            x.dim() != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or not residual.is_contiguous()
            or residual.dtype != torch.bfloat16
            or getattr(norm, "weight", None) is None
            or x.data_ptr() % 32
            or weight.data_ptr() % 32
        ):
            return None
        m, k = x.shape
        n = weight.shape[0]
        if residual.shape != (m, n) or m == 0:
            return None
        tactic = push_tactic(role, m, n, k)
        if tactic is None or not ar_norm_config(m, n, x.device)[0]:
            return None
        target = self.all_reduce.lamport_push_target(m, n, force=True)
        if target is None:
            return None
        pdl = get_env_enable_pdl()
        run_gemm_push(x.detach(), weight.detach(), target.output, target, tactic, pdl=pdl)
        out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        run_draft_ar_norm(
            target.workspace,
            target.one_shot,
            target.cpp_one_shot,
            residual,
            norm.weight.detach(),
            norm.variance_epsilon,
            out,
            residual_out,
            use_pdl=pdl,
        )
        return out, residual_out


def draft_embed_site(embedding, ids: torch.Tensor) -> Optional[torch.Tensor]:
    """``embedding(ids)`` of a COLUMN (vocab-sharded) TP ``Embedding`` whose all-reduce runs over
    MNNVL, as one owner-multicast kernel; None where it does not apply."""
    from ..flashinfer_utils import get_env_enable_pdl
    from ..kernels.blackwell.dspark_draft_ar import ELTS, EMBED_THREADS, run_draft_embed
    from ..modules.linear import TensorParallelMode

    weight = getattr(embedding, "weight", None)
    all_reduce = getattr(embedding, "all_reduce", None)
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if (
        weight is None
        or mnnvl is None
        or getattr(embedding, "tp_size", 1) <= 1
        or getattr(embedding, "tp_mode", None) != TensorParallelMode.COLUMN
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or ids.dim() != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.shape[0] == 0
        or weight.shape[1] % (ELTS * EMBED_THREADS)
    ):
        return None
    # The module's shards: ceil(vocab / tp) rows per rank, the last one possibly shorter.
    vocab = embedding.num_embeddings
    slice_width = -(-vocab // embedding.tp_size)
    if embedding.vocab_start_index != embedding.tp_rank * slice_width:
        return None
    m, hidden = ids.shape[0], weight.shape[1]
    target = mnnvl.lamport_push_target(m, hidden, one_shot=True)
    if target is None:
        return None
    run_draft_embed(
        target.workspace,
        ids,
        weight.detach(),
        slice_width,
        vocab,
        target.output,
        use_pdl=get_env_enable_pdl(),
    )
    return target.output
