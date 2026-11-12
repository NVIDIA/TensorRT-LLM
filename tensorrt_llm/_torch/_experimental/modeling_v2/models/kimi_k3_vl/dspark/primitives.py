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
"""DSpark drafter primitives shared by the K3 drafter and worker.

* FP8 E4M3 storage of the drafter's context latent (``to_draft_fp8``).
* ``HelixDraftShard``: one helix rank's share of the paged context latent.
* ``DraftArSites`` / ``draft_embed_site``: the drafter's o_proj / down_proj all-reduce
  sites and block embedding on pushed MNNVL all-reduces. o_proj and down_proj run as the
  low-M BF16 split-K GEMM whose epilogue stores the partial straight into the Lamport
  slots, and one site kernel finishes the all-reduce, adds the residual and applies the
  following RMSNorm; its outputs are bitwise the MNNVL fused all-reduce
  (``RESIDUAL_RMS_NORM``) of the GEMM output. The block embedding multicasts each
  token's owner row, bitwise the vocab-sharded lookup + all-reduce. A site that does not
  apply (more than 64 rows, no single-load launch shape, no MNNVL workspace view during
  capture) returns None and the caller keeps its path.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..ops.common.sm107_launch import USE_PDL

# FP8 E4M3 storage of an MLA drafter's context latent: one per-tensor scale of
# 1.0 for the latent and for the query that reads it -- the scheme of the
# target MLA's FP8 KV cache without calibrated kv scales (quant_scale_kv and
# quant_scale_q default to 1, bmm1_scale = softmax_scale * dq_q * dq_kv,
# bmm2_scale = dq_kv). Values saturate at the format's largest finite value, as
# the target's FP8 cache writes do, instead of overflowing to NaN.
DRAFT_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
DRAFT_FP8_KV_SCALE = 1.0


def to_draft_fp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` as saturated FP8 E4M3 at the drafter's unit per-tensor scale."""
    return x.clamp(-DRAFT_FP8_MAX, DRAFT_FP8_MAX).to(torch.float8_e4m3fn)


@dataclass(frozen=True)
class HelixDraftShard:
    """One rank's share of a drafter's paged context KV under helix.

    Global page ``g`` of a request belongs to CP rank ``g % cp_size``, which
    stores its pages back to back: local page ``g // cp_size``. Every helper
    maps global token positions or lengths into that rank-local layout.
    """

    cp_size: int
    cp_rank: int
    page_size: int

    def local_slots(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-local slot of each global position, and whether this rank owns it.

        An unowned position maps to the slot this rank keeps for the token one
        or more pages away, never one of the ``page_size`` tokens around it.
        """
        page = torch.div(positions, self.page_size, rounding_mode="floor")
        slots = torch.div(page, self.cp_size, rounding_mode="floor") * self.page_size
        return slots + positions % self.page_size, page % self.cp_size == self.cp_rank

    def local_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Number of owned positions below each global length."""
        span = self.page_size * self.cp_size
        rem = lengths % span - self.cp_rank * self.page_size
        return torch.div(lengths, span, rounding_mode="floor") * self.page_size + rem.clamp(
            0, self.page_size
        )


def combine_helix_partials(
    out: torch.Tensor, lse: torch.Tensor, empty: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Merge per-rank partial attention over disjoint KV shards.

    Args:
        out: ``[cp, N, H, D]`` partial outputs, one per CP rank.
        lse: ``[cp, N, H]`` base-2 log-sum-exp of the matching scores, as
            TRTLLM-Gen returns it.
        empty: optional ``[cp, N]`` bool, True where a rank held no keys for
            a row; its partial is ignored whatever the kernel wrote.
    Returns:
        ``[N, H, D]`` in ``out``'s dtype.
    """
    lse = lse.float()
    if empty is not None:
        lse = lse.masked_fill(empty.unsqueeze(-1), float("-inf"))
    peak = lse.amax(dim=0, keepdim=True)
    weight = torch.exp2(lse - peak.clamp_min(torch.finfo(torch.float32).min))
    weight = weight.nan_to_num(nan=0.0)
    total = weight.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
    merged = (weight.unsqueeze(-1) * out.float().nan_to_num(0.0)).sum(dim=0)
    return (merged / total.unsqueeze(-1)).to(out.dtype)


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
    from ..kernels.k3r_gemm.low_m_bf16_splitk import SplitKTactic
    from ..kernels.k3r_spec.dspark_draft_gemm_push import validate_push_tactic

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
    """The pushed all-reduce sites of one drafter, on its o_proj all-reduce (a
    ``layers.TpAllReduce`` whose MNNVL workspace every drafter all-reduce shares)."""

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

        from ..kernels.k3r_spec.dspark_draft_ar import ar_norm_config
        from ..ops.comm.sm107_allreduce import lamport_push_target
        from ..ops.spec.k3_draft_ar_norm import k3_draft_ar_norm
        from ..ops.spec.k3_draft_gemm_push import k3_draft_gemm_push

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
        target = lamport_push_target(self.all_reduce, m, n)
        if target is None:
            return None
        pdl = USE_PDL
        k3_draft_gemm_push(x.detach(), weight.detach(), target.output, target, tactic, pdl=pdl)
        out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        k3_draft_ar_norm(
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
    """``embedding(ids)`` of a vocab-parallel ``layers.EmbeddingTable`` whose all-reduce
    runs over MNNVL, as one owner-multicast kernel; None where it does not apply."""
    weight = getattr(embedding, "weight", None)
    all_reduce = getattr(embedding, "all_reduce", None)
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if weight is None or mnnvl is None or not getattr(embedding, "vocab_parallel", False):
        return None

    from ..kernels.k3r_spec.dspark_draft_ar import ELTS, EMBED_THREADS
    from ..ops.comm.sm107_allreduce import mnnvl_lamport_push_target
    from ..ops.spec.k3_draft_embed import k3_draft_embed

    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or ids.dim() != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.shape[0] == 0
        or weight.shape[1] % (ELTS * EMBED_THREADS)
    ):
        return None
    # The table's shards: ceil(vocab / tp) rows per rank (this rank's padded to that), the last
    # one possibly shorter.
    vocab = embedding.num_embeddings
    slice_width = weight.shape[0]
    if slice_width != -(-vocab // mnnvl.mapping.tp_size) or embedding.vocab_start % slice_width:
        return None
    m, hidden = ids.shape[0], weight.shape[1]
    target = mnnvl_lamport_push_target(mnnvl, m, hidden, one_shot=True)
    if target is None:
        return None
    k3_draft_embed(
        target.workspace,
        ids,
        weight.detach(),
        slice_width,
        vocab,
        target.output,
        use_pdl=USE_PDL,
    )
    return target.output
