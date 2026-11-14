# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC_SHARD16 (TP2): the DSpark drafter's fc as a tensor-parallel column shard plus an all-gather, bit for bit
the full fc, at up to 16 rows (the K7 verify rows of the batch-1 and batch-2 steps).

The TP4-geometry FC_SHARD with this target's row bound:
the drafter's fc ([M, layers x H] -> [M, H], bf16, no bias) is replicated on every rank, so each rank reads the whole
weight. Here each rank computes its H / tp output columns (3072 at TP2) and the ranks all-gather them. Each column is
computed by ``mv2_minimax_m3::minimax_m31_fc_shard_gemm`` with the cuBLASLt algo the full fc uses (``PINNED_CFG``), so
every output column is the same sum of the same K-split partials as in the full GEMM and the gathered result equals the
fc module's own output. ``FcShard.build`` checks that on the device before the first use, against ``fc(x)`` itself
(real-scale and 2^+-12 block-scaled probes at each row count it is asked to serve: the K7 verify rows 8 and 16 in
serving, every row count up to ``MAX_ROWS`` by default), per row count: the shard takes only the row counts whose probes
are equal on every rank, and returns None, keeping the stock fc, when the op is not built, the shape does not apply, or
no row count passes. The self-check decides, on every boot.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm.logger import logger

# cuBLASLt's algo for the full fc (K = 30720 -> N = 6144, bf16 in / out, fp32 compute) on SM107:
# ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID,
# CLUSTER_SHAPE_ID. The build self-check rejects it if this cuBLASLt picks a different one for the full fc.
PINNED_CFG = (66, 393, 22, 2, 0, 1, 35, 0, 3)
MAX_ROWS = (
    16  # capacity: fixed: the K7 verify rows at batches 1 and 2 (wider steps keep the stock fc)
)

_STATE: dict = {}
_OP = "minimax_m31_fc_shard_gemm"
_AGREE_DEVICE = "cuda"  # the per-row-count pass mask's all-gather (NCCL)


def fc_shard_op():
    """``mv2_minimax_m3::minimax_m31_fc_shard_gemm`` if this build carries it, else None."""
    if "op" not in _STATE:
        _STATE["op"] = getattr(family_ops(), _OP, None)
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

    def __init__(
        self, w: torch.Tensor, w_shard: torch.Tensor, mapping, op, rows: Sequence[int] = ()
    ) -> None:
        self.source = w
        self.w_shard = w_shard
        self.mapping = mapping
        self._op = op
        self._cfg = list(PINNED_CFG)
        self.rows: Tuple[int, ...] = tuple(
            rows
        )  # the row counts the self-check passed on every rank
        self._logged = set()  # "captured" / "eager": the engagement lines this rank logged

    @staticmethod
    def _rows(rows: Optional[Sequence[int]]) -> Tuple[int, ...]:
        """The row counts to check: ``rows`` within 1..MAX_ROWS (sorted, once each), or every one of them."""
        if rows is None:
            return tuple(range(1, MAX_ROWS + 1))
        return tuple(sorted({int(r) for r in rows if 0 < int(r) <= MAX_ROWS}))

    @classmethod
    def build(
        cls, fc: torch.nn.Module, mapping, rows: Optional[Sequence[int]] = None
    ) -> Optional["FcShard"]:
        """This rank's shard of ``fc`` for the row counts ``rows`` (default 1..MAX_ROWS) whose self-check passed on
        every rank, or None (the stock fc stays at every row count) with the reason logged. The ranks agree per row
        count with one tiny all-gather of a pass mask that every rank issues whatever its own outcome, so they agree
        on which row counts all-gather."""
        tp, rank = int(mapping.tp_size), int(mapping.tp_rank)
        want = cls._rows(rows)
        shard, why, bad = cls._check(fc, tp, rank, mapping, want)
        ok = [r for r in want if shard is not None and r not in bad]
        if tp >= 2 and torch.cuda.is_available() and want:
            mask = torch.tensor(
                [1 if r in ok else 0 for r in want], dtype=torch.int32, device=_AGREE_DEVICE
            )
            every = allgather(mask, mapping, dim=0).view(tp, len(want)).min(dim=0).values.tolist()
            others = [r for r, e in zip(want, every) if not e and r in ok]
            ok = [r for r, e in zip(want, every) if e]
            for r in others:
                bad[r] = "the check failed on another rank"
        if not ok:
            why = why or (
                "; ".join(f"{r} rows: {bad[r]}" for r in want if r in bad)
                or "no row count to serve"
            )
            logger.info(f"[M3 dispatch] FC_SHARD16 not taken on rank {rank}: {why}")
            return None
        shard.rows = tuple(ok)
        dropped = "".join(f"; {r} rows kept on the full fc: {bad[r]}" for r in want if r in bad)
        logger.info(
            f"[M3 prebuild] FC_SHARD16 set up on rank {rank}/{tp}: fc {tuple(shard.source.shape)} -> shard "
            f"{tuple(shard.w_shard.shape)}, bitwise to the full fc at rows {shard.rows}{dropped}"
        )
        return shard

    @classmethod
    def _check(
        cls, fc: torch.nn.Module, tp: int, rank: int, mapping, rows: Optional[Sequence[int]] = None
    ) -> Tuple[Optional["FcShard"], Optional[str], Dict[int, str]]:
        """(shard, None, {row count: why}) with the row counts whose probes differ from ``fc(x)``'s columns (the fc
        module's own output, what the stock projection computes); (None, reason, {}) when the shard does not apply
        at all (no op, shape / dtype / tp, the pinned algo rejected). The shard's ``rows`` are the passing ones."""
        op = fc_shard_op()
        if op is None:
            return None, f"{_OP} is not built", {}
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
            return None, f"fc shape / dtype / tp do not apply (tp {tp})", {}
        cols = w.shape[0] // tp
        shard = cls(w, w[rank * cols : (rank + 1) * cols].contiguous(), mapping, op)
        bad: Dict[int, str] = {}
        want = cls._rows(rows)
        try:
            with torch.no_grad():
                for m in want:
                    for x in _probes(m, w.shape[1], w.device):
                        ref = fc(x)[:, rank * cols : (rank + 1) * cols]
                        if not torch.equal(op(x, shard.w_shard, shard._cfg), ref):
                            bad[m] = "the pinned shard GEMM differs from the full fc"
                            break
        except RuntimeError as e:
            return None, f"the pinned cuBLASLt algo is rejected ({e})", {}
        torch.cuda.synchronize()
        shard.rows = tuple(m for m in want if m not in bad)
        return shard, None, bad

    def applies(self, fc: torch.nn.Module, rows: int) -> bool:
        """Whether ``fc`` is the fc this shard was built from and ``rows`` is one of the row counts it was checked
        bitwise at (on every rank)."""
        return getattr(fc, "weight", None) is self.source and rows in self.rows

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's output columns [rows, H / tp] (no collective: safe on a side stream). The engagement key line
        once per rank at the first call under CUDA-graph capture (the captured path); an eager call logs its own
        line."""
        kind = "captured" if torch.cuda.is_current_stream_capturing() else "eager"
        if kind not in self._logged:
            self._logged.add(kind)
            if kind == "captured":
                logger.info(
                    f"[M3 dispatch] FC_SHARD16 taken: drafter fc as {self.w_shard.shape[0]}-column shards + all-gather "
                    f"({x.shape[0]} rows, under CUDA-graph capture) on rank {self.mapping.tp_rank}"
                )
            else:
                logger.info(
                    f"[M3 sm107 tp2] FC_SHARD16 at an eager step: drafter fc as {self.w_shard.shape[0]}-column shards "
                    f"+ all-gather ({x.shape[0]} rows) on rank {self.mapping.tp_rank}"
                )
        return self._op(x.contiguous(), self.w_shard, self._cfg)

    def gather(self, y: torch.Tensor) -> torch.Tensor:
        """The full fc output from every rank's ``partial`` (the tensor-parallel all-gather, in rank order)."""
        return allgather(y, self.mapping, dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.gather(self.partial(x))
