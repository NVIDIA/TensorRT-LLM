# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe block-sparse verify decode for the MiniMax-M3 TP2 NVFP4 shape on GB300 (SM103).

minimax_m3_sparse_attn_decode hands a call here first; try_decode() runs torch.ops.trtllm.minimax_m3_cute_sparse_decode
and returns True when the device is SM103, the rank holds two KV heads (TP2), the call's (decode_query_len, local batch)
key is in the policy table below and the call is one the kernel covers, else False and the caller keeps the Triton
kernel.

The kernel: one CTA per (top-k split, KV head, query row), clusters of S CTAs merging over DSMEM, an ST-stage ring of
bulk copies feeding m16n8k16 HMMAs with the NVFP4 dequantized in registers (cpp/tensorrt_llm/kernels/minimaxM3/
cuteSparseDecode.cu). Policy = (S splits, ST ring stages, DQ dequant mode).

Table: a key is listed only where the kernel measured at most 0.95x the faster of the tuned and the default Triton
policy, on 57-layer CUDA graphs over a ~160 GiB pool (150k / 300k-token contexts at 85% block
overlap), and on every robustness scenario. Draft length 3 (decode_query_len 4): b1 and b2 measured slower; S4 at b3-b9
keeps one resident wave of at most 288 CTAs, S1 from b10 (0.40-0.72x). Draft length 5 (decode_query_len 6): b1 measured
0.975x; S4 at b2-b6, S1 from b7 (0.35-0.48x). Every listed policy passed an fp64-oracle tolerance gate. Any other
decode_query_len (e.g. 8 for a seven-token draft) keeps Triton.

Switches (read once at import): M3_CSD=1 enables the kernel (default off: the Triton kernel runs every call);
M3_CSD_TABLE=<json path> replaces the table ({"table": {"dql4": {"20": [1, 4, 0], ...}}}, benchmarking only).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

POLICIES = tuple((s, st, dq) for s in (1, 2, 4, 8) for st in (2, 3, 4) for dq in (0, 1))

# (decode_query_len, first local batch, last local batch, policy)
_RANGES = (
    (4, 3, 9, (4, 4, 0)),
    (4, 10, 45, (1, 4, 0)),
    (6, 2, 6, (4, 4, 0)),
    (6, 7, 25, (1, 4, 0)),
)

_ENABLED = os.environ.get("M3_CSD", "0").strip().lower() in ("1", "true", "on", "yes")
_USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
_PAGE = 128
_HEAD_DIM = 128
_GROUP = 16
_KV_HEADS = 2
_MAX_TOPK = 32


def builtin_table() -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    table = {}
    for dql, lo, hi, policy in _RANGES:
        for batch in range(lo, hi + 1):
            table[(dql, batch)] = policy
    return table


def table_from_json(obj) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """{(dql, batch): policy} from {"table": {"dql4": {"20": [1, 4, 0]}}} or its inner dict."""
    inner = obj.get("table", obj) if isinstance(obj, dict) else None
    if not isinstance(inner, dict):
        raise ValueError("M3_CSD_TABLE must hold a JSON object")
    table = {}
    for dql_key, rows in inner.items():
        if not (isinstance(dql_key, str) and dql_key.startswith("dql") and dql_key[3:].isdigit()):
            raise ValueError(f"M3_CSD_TABLE: bad key {dql_key!r}")
        if not isinstance(rows, dict):
            raise ValueError(f"M3_CSD_TABLE: {dql_key} must map batches to policies")
        dql = int(dql_key[3:])
        for batch_key, policy in rows.items():
            batch = int(batch_key)
            policy = tuple(int(x) for x in policy)
            if policy not in POLICIES or batch < 1 or dql < 1:
                raise ValueError(f"M3_CSD_TABLE: bad entry {dql_key} b{batch_key} {policy}")
            table[(dql, batch)] = policy
    return table


def _load_table() -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    path = os.environ.get("M3_CSD_TABLE", "").strip()
    if not path:
        return builtin_table()
    with open(path) as f:
        return table_from_json(json.load(f))


_TABLE = _load_table() if _ENABLED else {}
_state = {"ready": None, "logged": set()}


def _ready(device: torch.device) -> bool:
    """Whether the op exists and the device is SM103; sets every table policy's kernel attributes once, eagerly.

    Outside a CUDA-graph capture the answer is cached; during a capture an uninitialised state answers False (that
    graph keeps Triton) so the kernels are never first loaded inside a capture.
    """
    ready = _state["ready"]
    if ready is not None:
        return ready
    ready = (
        torch.cuda.get_device_capability(device) == (10, 3)
        and hasattr(torch.ops.trtllm, "minimax_m3_cute_sparse_decode")
        and not torch.cuda.is_current_stream_capturing()
    )
    if ready:
        with torch.cuda.device(device):
            for policy in sorted(set(_TABLE.values())):
                torch.ops.trtllm.minimax_m3_cute_sparse_decode_attrs(*policy)
        logger.info(
            f"MiniMax-M3 CuTe sparse decode on: {len(_TABLE)} (decode_query_len, batch) keys, "
            f"policies {sorted(set(_TABLE.values()))}"
        )
    if ready or not torch.cuda.is_current_stream_capturing():
        _state["ready"] = ready
    return ready


def _byte_view_ok(t: torch.Tensor, inner: int) -> bool:
    return (
        t.dim() == 4
        and t.stride(3) == 1
        and t.stride(2) == inner
        and t.stride(1) % 16 == 0
        and t.stride(0) % 16 == 0
        and t.data_ptr() % 16 == 0
    )


def try_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    k_block_scale: Optional[torch.Tensor],
    v_block_scale: Optional[torch.Tensor],
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
) -> bool:
    """Run the CuTe kernel when the call's key is listed and its operands are covered; True when it ran."""
    if not _TABLE or decode_query_len <= 0:
        return False
    num_kv_heads = int(k_paged.shape[1])
    if num_kv_heads != _KV_HEADS:
        return False
    if any(t is None for t in (k_block_scale, v_block_scale, k_global_scale, v_global_scale)):
        return False
    # Initialise on the first eager NVFP4 call of the TP2 shape, whatever its batch, so that it precedes capture.
    if not _ready(q.device):
        return False
    total_q = int(q.shape[0])
    policy = _TABLE.get((decode_query_len, total_q // decode_query_len))
    if policy is None:
        return False
    q8 = q.view(torch.uint8) if q.dtype == torch.float8_e4m3fn else None
    k8, v8 = k_paged.view(torch.uint8), v_paged.view(torch.uint8)
    ks8, vs8 = k_block_scale.view(torch.uint8), v_block_scale.view(torch.uint8)
    covered = (
        q8 is not None
        and q.dim() == 3
        and q.shape[2] == _HEAD_DIM
        and q.shape[1] == _GROUP * num_kv_heads
        and q8.stride(2) == 1
        and q8.stride(1) % 16 == 0
        and q8.stride(0) % 16 == 0
        and q8.data_ptr() % 16 == 0
        and int(k_paged.shape[2]) == _PAGE
        and int(k_paged.shape[-1]) == _HEAD_DIM // 2
        and _byte_view_ok(k8, _HEAD_DIM // 2)
        and _byte_view_ok(v8, _HEAD_DIM // 2)
        and _byte_view_ok(ks8, _HEAD_DIM // 16)
        and _byte_view_ok(vs8, _HEAD_DIM // 16)
        and topk_idx.dtype == torch.int32
        and topk_idx.dim() == 3
        and topk_idx.shape[-1] <= _MAX_TOPK
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and seq_lens.dtype == torch.int32
        and output.dtype == torch.bfloat16
        and output.dim() == 3
        and output.stride(2) == 1
        and output.stride(1) % 8 == 0
        and output.stride(0) % 8 == 0
        and output.data_ptr() % 16 == 0
    )
    if not covered:
        key = (decode_query_len, total_q // decode_query_len)
        if key not in _state["logged"]:
            _state["logged"].add(key)
            # Unaligned eager slices (mixed steps, the sparse-decode prewarm) keep the Triton kernel.
            logger.info(
                f"MiniMax-M3 CuTe sparse decode: key {key} call not covered (layout); Triton kernel"
            )
        return False
    torch.ops.trtllm.minimax_m3_cute_sparse_decode(
        q8,
        k8,
        v8,
        ks8,
        vs8,
        k_global_scale,
        v_global_scale,
        topk_idx,
        block_table,
        seq_lens,
        output,
        int(decode_query_len),
        float(sm_scale),
        *policy,
        _USE_PDL,
    )
    return True
