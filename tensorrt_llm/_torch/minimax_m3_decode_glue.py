# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 / M3.1 TP2 decode glue on Blackwell (SM100/SM103): switches and the multicast layer-boundary all-reduce.

The pieces, each in the module it changes:

- select (msa_indexer): decode top-16 block selects of 1..72 rows over 129..9216 blocks run
  torch.ops.trtllm.minimax_m3_select_long (one 1024-thread CTA per row holding the row's keys in registers) instead
  of the histogram kernel. Same table bit for bit, except on a row where more than 1024 valid blocks tie at the
  16th-largest score, where select_long follows the selection contract and the histogram kernel does not.
- nvfp4_hoist (MiniMaxM3MoE): in decode CUDA-graph forwards the routed experts' NVFP4 input quantize runs at the head
  of the shared-expert aux stream instead of behind the router on the main stream. Bitwise.
- router_tc (MiniMaxM3MoE): decode MoE forwards of 2..48 rows run torch.ops.trtllm.minimax_m3_router_tc, one kernel
  for the router logits (TF32 tensor cores) and the experts' NVFP4 input, in place of the router GEMM and the
  quantize. The NVFP4 bytes equal fp4_quantize's; the logits differ from the stock router by TF32 rounding and
  summation order.
- mc_boundary (MiniMaxM3DecoderLayer): at each post-feed-forward boundary of a decode CUDA-graph forward of 1..48 rows
  the shared + routed add, the AllReduce(RESIDUAL_RMS_NORM) and the next layer's MXFP8 input quantize become one
  torch.ops.trtllm.minimax_m3_mc_allreduce_norm call (multicast all-reduce, residual add, RMSNorm, MXFP8 copy), and
  the next layer's fused QKV projection runs only its GEMM on that copy. The residual is rounded to BF16 once
  (rank sum + residual) instead of twice, so some residual / normed elements differ from the stock path by one BF16
  ulp. Kept stock: speculative capture layers, the last layer (final norm), eager / mixed / prefill /
  breakable-graph forwards.

Rows are the forward's token rows: at TP2 with DSpark K drafts a decode step has (K + 1) * batch rows, so with K=3
the 48-row parts (mc_boundary, router_tc) cover batch <= 12 and select covers batch <= 18.

Switches (read once at import; "0", "false", "off", "no" = off): M3AX_L5 (master, off by default), then
M3AX_L5_SELECT, M3AX_L5_NVFP4_HOIST, M3AX_L5_ROUTER_TC, M3AX_L5_MC_BOUNDARY, each on once the master is on; the row
ranges M3AX_L5_SELECT_ROWS (default 1-72), M3AX_L5_ROUTER_TC_ROWS (2-48), M3AX_L5_MC_ROWS (1-48), "lo-hi[,lo-hi...]",
and M3AX_L5_HOIST_MAX_ROWS (1024).
"""

import os
from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

_OFF = ("0", "false", "off", "no")
ARCHS = frozenset((100, 103))
MC_MAX_ROWS = 48  # minimax_m3_mc_allreduce_norm's largest bucket
_MC_CLUSTER = 8  # the only cluster size the multicast workspaces build call counters for


def _switch(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in _OFF


def _rows_spec(name: str, default: str, cap: Optional[int] = None) -> Tuple[Tuple[int, int], ...]:
    text = os.environ.get(name, default)
    ranges = []
    try:
        for part in text.replace(" ", "").split(","):
            if not part:
                continue
            lo, _, hi = part.partition("-")
            ranges.append((int(lo), int(hi or lo)))
    except ValueError as e:
        raise ValueError(f"{name}={text!r}: expected lo-hi[,lo-hi...]") from e
    if cap is not None:
        ranges = [(max(lo, 1), min(hi, cap)) for lo, hi in ranges if lo <= cap]
    return tuple(r for r in ranges if r[0] <= r[1])


MASTER = _switch("M3AX_L5", "0")
SELECT = MASTER and _switch("M3AX_L5_SELECT")
NVFP4_HOIST = MASTER and _switch("M3AX_L5_NVFP4_HOIST")
ROUTER_TC = MASTER and _switch("M3AX_L5_ROUTER_TC")
MC_BOUNDARY = MASTER and _switch("M3AX_L5_MC_BOUNDARY")
SELECT_ROWS = _rows_spec("M3AX_L5_SELECT_ROWS", "1-72")
ROUTER_TC_ROWS = _rows_spec("M3AX_L5_ROUTER_TC_ROWS", "2-48", cap=MC_MAX_ROWS)
MC_ROWS = _rows_spec("M3AX_L5_MC_ROWS", "1-48", cap=MC_MAX_ROWS)
HOIST_MAX_ROWS = int(os.environ.get("M3AX_L5_HOIST_MAX_ROWS", "1024"))
USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1").strip() != "0"


def arch_ok() -> bool:
    return get_sm_version() in ARCHS


def rows_in(ranges: Tuple[Tuple[int, int], ...], rows: int) -> bool:
    return any(lo <= rows <= hi for lo, hi in ranges)


def describe() -> str:
    def fmt(ranges):
        return ",".join(f"{lo}-{hi}" for lo, hi in ranges)

    return (
        f"select={int(SELECT)} (rows {fmt(SELECT_ROWS)}) nvfp4_hoist={int(NVFP4_HOIST)} "
        f"router_tc={int(ROUTER_TC)} (rows {fmt(ROUTER_TC_ROWS)}) mc_boundary={int(MC_BOUNDARY)} "
        f"(rows {fmt(MC_ROWS)}) pdl={int(USE_PDL)}"
    )


class McBoundary:
    """The process-wide multicast workspaces of minimax_m3_mc_allreduce_norm (buckets of 8 / 16 / 48 rows).

    Every condition that decides whether a boundary takes the op depends only on shapes and configuration, so all
    TP ranks issue the same calls in the same order, which the op's exchange protocol requires. The workspaces are
    created collectively by the first eligible call, outside CUDA-graph capture (the decode-graph warmup forward);
    a call during capture before that returns None on every rank alike.
    """

    _state = {"state": "new", "ws": None}

    @classmethod
    def _module(cls):
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops import (  # noqa: E501
            m3_mc_allreduce,
        )

        return m3_mc_allreduce

    @classmethod
    def workspace(cls, mapping, hidden: int):
        st = cls._state
        if st["state"] == "ok":
            return st["ws"]
        if st["state"] == "failed" or torch.cuda.is_current_stream_capturing():
            return None
        mc = cls._module()
        local_ok = (
            hasattr(torch.ops.trtllm, "minimax_m3_mc_allreduce_norm")
            and hidden % 128 == 0
            and torch.cuda.get_device_capability()[0] == 10
        )
        if not mc.tp_agree(mapping, local_ok):
            st["state"] = "failed"
            logger.warning(
                "MiniMax-M3 multicast boundary all-reduce unavailable on some TP rank; stock all-reduce"
            )
            return None
        built = None
        try:
            built = mc.get_workspace(mapping, hidden)
        except RuntimeError as e:
            logger.warning(f"MiniMax-M3 multicast boundary workspace creation failed: {e}")
        built = mc.tp_agree_built(mapping, built, "MiniMax-M3 multicast boundary workspace")
        if built is None or any(w.flags.get(_MC_CLUSTER) is None for w in built.workspaces):
            st["state"] = "failed"
            return None
        st["state"], st["ws"] = "ok", built
        logger.info(
            f"MiniMax-M3 multicast boundary all-reduce on: hidden {hidden}, buckets "
            f"{[w.max_tokens for w in built.workspaces]}, world {built.workspaces[0].world_size}"
        )
        return built

    @classmethod
    def allreduce_norm(
        cls,
        ws,
        x: torch.Tensor,
        x2: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ):
        """[normed, updated, act_e4m3, act_sf] for bf16(x + x2) summed over the TP ranks, plus residual, RMSNorm'd;
        None when the operands are outside the op (decided from shapes and layouts only, alike on every rank)."""
        rows, hidden = int(residual.shape[0]), int(residual.shape[1])
        bucket = ws.for_tokens(rows)
        if bucket is None:
            return None
        flags = bucket.flags.get(_MC_CLUSTER)
        if (
            flags is None
            or hidden != bucket.hidden
            or not residual.is_contiguous()
            or residual.dtype != torch.bfloat16
            or x.dim() != 2
            or tuple(x.shape) != tuple(residual.shape)
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or (
                x2 is not None
                and (
                    tuple(x2.shape) != tuple(x.shape)
                    or x2.dtype != torch.bfloat16
                    or not x2.is_contiguous()
                )
            )
        ):
            return None
        return torch.ops.trtllm.minimax_m3_mc_allreduce_norm(
            x,
            x2,
            residual,
            norm_weight,
            float(eps),
            bucket.uc,
            bucket.mc.data_ptr(),
            flags,
            bucket.rank,
            bucket.world_size,
            _MC_CLUSTER,
            USE_PDL,
            None,
            None,
            bucket.max_tokens,
            True,
            0,
            cls._module()._ar_opts(rows, bucket.max_tokens),
        )
