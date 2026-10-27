# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC_SHARD: the DSpark drafter's fc as a tensor-parallel column shard plus an all-gather, bit for bit the full fc.

The drafter's fc ([M, layers x H] -> [M, H], bf16, no bias) is replicated on every rank, so each rank reads the whole
weight. Here each rank computes its H / tp output columns and the ranks all-gather them. Each column is computed by
``trtllm::minimax_m31_fc_shard_gemm`` with the cuBLASLt algo the full fc uses (``PINNED_CFG``), so every output column
is the same sum of the same K-split partials as in the full GEMM and the gathered result equals
``torch.nn.functional.linear(x, w)``. ``FcShard.build`` checks that on the device before the first use (real-scale and
2^+-12 block-scaled probes, every row count up to ``MAX_ROWS``) and returns None, keeping the stock fc, when the op is
not built, the shape does not apply, or any probe differs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm.logger import logger

# cuBLASLt's algo for the full fc (K = 30720 -> N = 6144, bf16 in / out, fp32 compute) in the VR image's cuBLASLt:
# ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID,
# CLUSTER_SHAPE_ID. The build self-check rejects it if this cuBLASLt picks a different one for the full fc.
PINNED_CFG = (66, 393, 22, 2, 0, 1, 35, 0, 3)
MAX_ROWS = 8  # capacity: fixed: the K7 verify rows at batch 1 (c1; c2+ keeps the stock fc)

_STATE: dict = {}
_OP = "minimax_m31_fc_shard_gemm"


def fc_shard_op():
    """``trtllm::minimax_m31_fc_shard_gemm`` if this build carries it, else None."""
    if "op" not in _STATE:
        op = getattr(torch.ops.trtllm, _OP, None)
        if op is not None:
            # The op's fake lives in the catalog's M3.1 fake table (registered once, whoever loads it first).
            from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops

            m31_fake_ops.register()
        _STATE["op"] = op
    return _STATE["op"]


def _probes(rows: int, k: int, device: torch.device):
    """A real-scale and a 2^+-12 block-scaled bf16 probe [rows, k] (128-column blocks), seeded."""
    g = torch.Generator(device="cpu").manual_seed(31 * rows + 7)
    x = torch.randn(rows, k, generator=g)
    yield x.to(torch.bfloat16).to(device)
    e = torch.randint(-12, 13, (rows, (k + 127) // 128), generator=g).float()
    yield (x * torch.pow(2.0, e).repeat_interleave(128, dim=1)[:, :k]).to(torch.bfloat16).to(device)


class FcShard:
    """The fc split by output columns over the tensor-parallel group; ``__call__(x)`` = the full fc's output."""

    def __init__(self, w: torch.Tensor, w_shard: torch.Tensor, mapping, op) -> None:
        self.source = w
        self.w_shard = w_shard
        self.mapping = mapping
        self._op = op
        self._cfg = list(PINNED_CFG)
        self._logged = False

    @classmethod
    def build(cls, fc: torch.nn.Module, mapping) -> Optional["FcShard"]:
        """This rank's shard of ``fc``, or None (the stock fc stays) with the reason logged. Every rank of the
        tensor-parallel group takes the shard only if every rank's check passed, so the ranks agree on whether the fc
        all-gathers (the agreement is one tiny all-gather that every rank issues, whatever its own outcome)."""
        tp, rank = int(mapping.tp_size), int(mapping.tp_rank)
        shard, why = cls._check(fc, tp, rank, mapping)
        if tp >= 2 and torch.cuda.is_available():
            ok = torch.tensor([0 if why else 1], dtype=torch.int32, device="cuda")
            if not bool(allgather(ok, mapping, dim=0).all()):
                why = why or "the check failed on another rank"
        if why:
            logger.info(f"[M3 dispatch] FC_SHARD not taken on rank {rank}: {why}")
            return None
        logger.info(
            f"[M3 prebuild] FC_SHARD set up on rank {rank}/{tp}: fc {tuple(shard.source.shape)} -> shard "
            f"{tuple(shard.w_shard.shape)}, bitwise to the full fc at 1..{MAX_ROWS} rows"
        )
        return shard

    @classmethod
    def _check(cls, fc: torch.nn.Module, tp: int, rank: int, mapping):
        """(shard, None) when this rank's pinned shard GEMM is bitwise to the full fc on every probe, else
        (None, reason)."""
        op = fc_shard_op()
        if op is None:
            return None, f"{_OP} is not built"
        w = getattr(fc, "weight", None)
        if (
            w is None
            or getattr(fc, "bias", None) is not None
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or not w.is_cuda
            or tp < 2
            or w.shape[0] % tp != 0
        ):
            return None, f"fc shape / dtype / tp do not apply (tp {tp})"
        cols = w.shape[0] // tp
        shard = cls(w, w[rank * cols : (rank + 1) * cols].contiguous(), mapping, op)
        try:
            for rows in range(1, MAX_ROWS + 1):
                for x in _probes(rows, w.shape[1], w.device):
                    want = F.linear(x, w)[:, rank * cols : (rank + 1) * cols]
                    if not torch.equal(op(x, shard.w_shard, shard._cfg), want):
                        return (
                            None,
                            f"the pinned shard GEMM differs from the full fc at {rows} rows",
                        )
        except RuntimeError as e:
            return None, f"the pinned cuBLASLt algo is rejected ({e})"
        torch.cuda.synchronize()
        return shard, None

    def applies(self, fc: torch.nn.Module, rows: int) -> bool:
        """Whether ``fc`` is the fc this shard was built from and ``rows`` is within the checked row counts."""
        return getattr(fc, "weight", None) is self.source and 0 < rows <= MAX_ROWS

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's output columns [rows, H / tp] (no collective: safe on a side stream)."""
        if not self._logged:
            self._logged = True
            logger.info(
                f"[M3 dispatch] FC_SHARD taken: drafter fc as {self.w_shard.shape[0]}-column shards + all-gather on "
                f"rank {self.mapping.tp_rank}"
            )
        return self._op(x.contiguous(), self.w_shard, self._cfg)

    def gather(self, y: torch.Tensor) -> torch.Tensor:
        """The full fc output from every rank's ``partial`` (the tensor-parallel all-gather, in rank order)."""
        return allgather(y, self.mapping, dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.gather(self.partial(x))
