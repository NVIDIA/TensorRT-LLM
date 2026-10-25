# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GB300 launch policy for the MiniMax-M3 NVFP4 sparse decode at TP2 (o8 sparse tables).

The in-tree GB300 tables in triton_sparse_decode (split-K, Eagle query pairing,
producer warps/stages, merge warps) are gated on ``num_kv_heads == 4``. Under
TP2 each rank holds 2 of the model's 4 KV heads, so every TP2 decode step falls
through to ``resolve_num_topk_chunks`` and Triton's default launch (4 warps, 3
stages, merge 4 warps, no pairing). This module supplies the TP2 policy that the
o8 op bench measured on GB300 at the AgentX shapes (2 KV heads, GQA 16, top-16,
four-token Eagle3 verify, NVFP4 cache, FP8 q), keyed on the local batch, which is
fixed for a captured CUDA graph. MiniMax-M3.1 shares this attention geometry,
and its four-token verify (DSpark or Eagle3 with three draft tokens) takes the
same table; any other decode_query_len keeps the default routing.

Pure Python and stdlib only, so the CPU self-test can load it without torch.

Switches (read once, at import):
  M3AX_STACK_A=0         GB300 stack master switch (also false/off/no): as
                         M3_O8_SPARSE_TABLES=0.
  M3_O8_SPARSE_TABLES=0  no policy at all: minimax_m3_sparse_attn_decode uses
                         its default routing (bitwise identical output).
  M3_O8_PAIR=0           use the best measured policy that does not use the
                         Eagle query-pair producer (for an A/B of the pairing).
A local batch the table does not list also keeps the default routing.

Evidence: each time a local batch's
policy is used for the 1st, 2nd, 4th, 8th, ... time in a process, one line
"[o8] sparse decode TP2 policy hit: ..." goes to stderr. Captured CUDA graphs
call this at capture only; eager (mixed ctx+gen) steps call it every time, so
the line keeps appearing at a logarithmic rate while the policy is in use.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ENV_SWITCH = "M3_O8_SPARSE_TABLES"
ENV_PAIR = "M3_O8_PAIR"
ENV_MASTER = "M3AX_STACK_A"

# Read once: the policy is consulted on every sparse layer of every eager
# (mixed) step, and a captured graph freezes whatever it saw at capture anyway.
MASTER_ENABLED = os.environ.get(ENV_MASTER, "1").strip().lower() not in ("0", "false", "off", "no")
ENABLED = MASTER_ENABLED and os.environ.get(ENV_SWITCH, "1") != "0"
PAIR_ENABLED = os.environ.get(ENV_PAIR, "1") != "0"

# Measured on GB300 (SM103) only; SM100 keeps the defaults, as in-tree.
TUNED_CAPABILITIES = ((10, 3),)
NUM_KV_HEADS = 2
GQA_GROUP_SIZE = 16
MAX_TOPK = 16
DECODE_QUERY_LEN = 4


@dataclass(frozen=True)
class SparseDecodePolicy:
    """One launch of the producer (+ merge) of minimax_m3_sparse_attn_decode.

    ``None`` warps/stages leave Triton's default, exactly as the default
    routing's ``{}`` launch options do.
    """

    num_topk_chunks: int
    query_group_size: int = 1
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    merge_num_warps: Optional[int] = None

    def __post_init__(self):
        c = self.num_topk_chunks
        if c < 1 or c & (c - 1) or c > MAX_TOPK:
            raise ValueError(f"num_topk_chunks must be a power of two in [1, {MAX_TOPK}]; got {c}")
        if self.query_group_size not in (1, 2):
            raise ValueError(f"query_group_size must be 1 or 2; got {self.query_group_size}")
        for name in ("num_warps", "merge_num_warps"):
            w = getattr(self, name)
            if w is not None and (w < 1 or w & (w - 1) or w > 16):
                raise ValueError(f"{name} must be a power of two in [1, 16]; got {w}")
        if self.num_stages is not None and not 1 <= self.num_stages <= 6:
            raise ValueError(f"num_stages must be in [1, 6]; got {self.num_stages}")

    def launch_options(self) -> Dict[str, int]:
        out = {}
        if self.num_warps is not None:
            out["num_warps"] = self.num_warps
        if self.num_stages is not None:
            out["num_stages"] = self.num_stages
        return out

    def merge_launch_options(self) -> Dict[str, int]:
        return {} if self.merge_num_warps is None else {"num_warps": self.merge_num_warps}

    def as_tuple(self) -> Tuple:
        return (
            self.num_topk_chunks,
            self.query_group_size,
            self.num_warps,
            self.num_stages,
            self.merge_num_warps,
        )


def _p(chunks, group=1, warps=None, stages=None, merge_warps=None) -> SparseDecodePolicy:
    return SparseDecodePolicy(chunks, group, warps, stages, merge_warps)


# Local batch (requests per rank; total_q = 4 x local batch) -> policy.
# TP2_TABLE is the fastest measured policy; TP2_TABLE_NO_PAIR the fastest one
# with query_group_size 1. Both come from the op-bench result named in
# TABLE_SOURCE; each comment gives the time relative to the default routing.
# >>> O8 TABLE (generated)
TABLE_SOURCE = (
    "o8 op bench, job 3410271 (2026-10-20 20:22 ET), GB300 theia0209.lyris.clusters.nvidia.com"
)
TP2_TABLE: Dict[int, SparseDecodePolicy] = {
    3: _p(16, 1, warps=4, stages=1),  # x0.863 primary, x0.863 all 6 (stock split 8)
    5: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.769 primary, x0.767 all 6 (stock split 4)
    6: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.768 primary, x0.764 all 6 (stock split 4)
    7: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.767 primary, x0.763 all 6 (stock split 4)
    8: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.917 primary, x0.917 all 6 (stock split 4)
    9: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.707 primary, x0.707 all 6 (stock split 2)
    10: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.687 primary, x0.682 all 6 (stock split 2)
    11: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.689 primary, x0.684 all 6 (stock split 2)
    12: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.734 primary, x0.729 all 6 (stock split 2)
    13: _p(4, 1, warps=4, stages=2, merge_warps=2),  # x0.745 primary, x0.741 all 6 (stock split 2)
    14: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.744 primary, x0.741 all 6 (stock split 2)
    15: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.892 primary, x0.891 all 6 (stock split 2)
    16: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.895 primary, x0.894 all 6 (stock split 2)
    17: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.689 primary, x0.686 all 6 (stock split 1)
    18: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.691 primary, x0.689 all 6 (stock split 1)
    19: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.699 primary, x0.695 all 6 (stock split 1)
    20: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.645 primary, x0.636 all 6 (stock split 1)
    21: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.648 primary, x0.641 all 6 (stock split 1)
    22: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.689 primary, x0.682 all 6 (stock split 1)
    23: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.690 primary, x0.683 all 6 (stock split 1)
    24: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.754 primary, x0.750 all 6 (stock split 1)
    25: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.762 primary, x0.756 all 6 (stock split 1)
    26: _p(2, 1, warps=4, stages=2),  # x0.768 primary, x0.761 all 6 (stock split 1)
    27: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.770 primary, x0.763 all 6 (stock split 1)
    28: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.772 primary, x0.766 all 6 (stock split 1)
    29: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.882 primary, x0.876 all 6 (stock split 1)
    30: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.897 primary, x0.889 all 6 (stock split 1)
    31: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.928 primary, x0.921 all 6 (stock split 1)
    32: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.943 primary, x0.934 all 6 (stock split 1)
    35: _p(2, 1, warps=2, stages=1, merge_warps=1),  # x0.956 primary, x0.949 all 6 (stock split 1)
    40: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.878 primary, x0.876 all 6 (stock split 1)
    45: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.968 primary, x0.966 all 6 (stock split 1)
}
TP2_TABLE_NO_PAIR: Dict[int, SparseDecodePolicy] = {
    3: _p(16, 1, warps=4, stages=1),  # x0.863 primary, x0.863 all 6 (stock split 8)
    5: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.769 primary, x0.767 all 6 (stock split 4)
    6: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.768 primary, x0.764 all 6 (stock split 4)
    7: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.767 primary, x0.763 all 6 (stock split 4)
    8: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.917 primary, x0.917 all 6 (stock split 4)
    9: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.707 primary, x0.707 all 6 (stock split 2)
    10: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.687 primary, x0.682 all 6 (stock split 2)
    11: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.689 primary, x0.684 all 6 (stock split 2)
    12: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.734 primary, x0.729 all 6 (stock split 2)
    13: _p(4, 1, warps=4, stages=2, merge_warps=2),  # x0.745 primary, x0.741 all 6 (stock split 2)
    14: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.744 primary, x0.741 all 6 (stock split 2)
    15: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.892 primary, x0.891 all 6 (stock split 2)
    16: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.895 primary, x0.894 all 6 (stock split 2)
    17: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.689 primary, x0.686 all 6 (stock split 1)
    18: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.691 primary, x0.689 all 6 (stock split 1)
    19: _p(4, 1, warps=2, stages=1, merge_warps=1),  # x0.699 primary, x0.695 all 6 (stock split 1)
    20: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.645 primary, x0.636 all 6 (stock split 1)
    21: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.648 primary, x0.641 all 6 (stock split 1)
    22: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.689 primary, x0.682 all 6 (stock split 1)
    23: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.690 primary, x0.683 all 6 (stock split 1)
    24: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.754 primary, x0.750 all 6 (stock split 1)
    25: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.762 primary, x0.756 all 6 (stock split 1)
    26: _p(2, 1, warps=4, stages=2),  # x0.768 primary, x0.761 all 6 (stock split 1)
    27: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.770 primary, x0.763 all 6 (stock split 1)
    28: _p(2, 1, warps=4, stages=2, merge_warps=1),  # x0.772 primary, x0.766 all 6 (stock split 1)
    29: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.882 primary, x0.876 all 6 (stock split 1)
    30: _p(8, 1, warps=4, stages=2, merge_warps=1),  # x0.897 primary, x0.889 all 6 (stock split 1)
    31: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.928 primary, x0.921 all 6 (stock split 1)
    32: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.943 primary, x0.934 all 6 (stock split 1)
    35: _p(2, 1, warps=2, stages=1, merge_warps=1),  # x0.956 primary, x0.949 all 6 (stock split 1)
    40: _p(4, 1, warps=4, stages=2, merge_warps=1),  # x0.878 primary, x0.876 all 6 (stock split 1)
    45: _p(8, 1, warps=2, stages=1, merge_warps=1),  # x0.968 primary, x0.966 all 6 (stock split 1)
}
# <<< O8 TABLE (generated)


def tp2_policy(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Tuple[int, int],
) -> Optional[SparseDecodePolicy]:
    """The measured TP2 policy for this NVFP4 decode shape, or None for the default routing."""
    if (
        not ENABLED
        or tuple(capability) not in TUNED_CAPABILITIES
        or num_kv_heads != NUM_KV_HEADS
        or gqa_group_size != GQA_GROUP_SIZE
        or max_topk != MAX_TOPK
        or decode_query_len != DECODE_QUERY_LEN
        or total_q <= 0
        or total_q % decode_query_len != 0
    ):
        return None
    local_batch = total_q // decode_query_len
    table = TP2_TABLE if PAIR_ENABLED else TP2_TABLE_NO_PAIR
    policy = table.get(local_batch)
    if policy is not None:
        n = _HITS.get(local_batch, 0) + 1
        _HITS[local_batch] = n
        if n & (n - 1) == 0:
            print(
                f"[o8] sparse decode TP2 policy hit: local_batch={local_batch} n={n} "
                f"policy={policy.as_tuple()} pair_enabled={int(PAIR_ENABLED)}",
                file=sys.stderr,
                flush=True,
            )
    return policy


# Uses per local batch in this process (evidence logging only).
_HITS: Dict[int, int] = {}


def summary() -> dict:
    """What the switches select, for logs."""
    return {
        "enabled": ENABLED,
        "master_enabled": MASTER_ENABLED,
        "pair_enabled": PAIR_ENABLED,
        "table_source": TABLE_SOURCE,
        "table": {b: p.as_tuple() for b, p in sorted(TP2_TABLE.items())},
        "table_no_pair": {b: p.as_tuple() for b, p in sorted(TP2_TABLE_NO_PAIR.items())},
    }


__all__ = [
    "ENABLED",
    "ENV_MASTER",
    "ENV_PAIR",
    "ENV_SWITCH",
    "MASTER_ENABLED",
    "PAIR_ENABLED",
    "SparseDecodePolicy",
    "TP2_TABLE",
    "TP2_TABLE_NO_PAIR",
    "summary",
    "tp2_policy",
]
