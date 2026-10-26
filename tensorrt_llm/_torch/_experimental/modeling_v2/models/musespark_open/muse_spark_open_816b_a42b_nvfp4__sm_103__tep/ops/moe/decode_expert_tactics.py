# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed TRTLLM-gen tactics for the NVFP4 expert call of tep4 decode steps (sm103, sm107).

``trtllm::fp4_block_scale_moe_runner`` takes its ``[tile_n, config]`` tactic from the
autotuner, which profiles one tactic per power-of-two row bucket at warmup, on a dummy
routing, separately on every rank: every row count of 256 to 511 runs the tactic
profiled at 256 rows, where it settles on 64-token tiles.

This table pins the tactic of the stacked expert call (``ring_buddy._nvfp4_call``) per
``(expert window, rows)`` for listed decode row counts of tep4 (8 rows per request at the
DFlash K = 7 verify width), one table per GPU:

* sm107: the 32-token tile the autotuner serves at 128 to 255 rows, for 256 to 384 rows.
  It computes every expert row bitwise as the autotuned tactics whose FC2 has the same
  MMA K; tactics whose FC2 MMA K differs can round a rare row differently in its last
  place (the autotuner picks from both).
* sm103: the 128-token tile the autotuner serves at 512 rows, for 256 to 504 rows. Every
  NVFP4 expert kernel sm103 loads has an MMA K of 64, and every tactic valid at these row
  counts computes every expert row bitwise as the autotuned 64-token tactics.

Row counts and windows not listed, and tuning passes, keep the autotuner. A GPU without a
table is not supported.

The config index is an index into the TRTLLM-gen MoE runner's list of (FC1, FC2) kernel
pairs for one tile N, built from the kernels the running GPU loads, so an entry is only
meaningful for the GPU and kernel library it was measured on; ``fixed_tactic`` also
requires the entry to be a valid config of the call's shape (``get_valid_configs``) and
returns None otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch

#: The 32-token tile the autotuner serves at 128..255 rows (FC1
#: ``t128x32x512u2_s5 … m256x32x128``, FC2 ``t128x32x512_s5 … m256x32x128``), pinned for the
#: 256..384-row decode steps (32 to 48 requests) of the 68-expert ``hot_replicas`` window.
#: Its FC2's MMA K is 128, as is that of the tile-64 tactics the autotuner picks there
#: most, so every expert row is bitwise theirs.
_TILE32_DECODE = (32, 236)

#: ``(local_num_experts, rows) -> (tile_n, config)`` on sm107. Window 68 = the 64 own
#: experts plus 4 hot copies (``hot_replicas``, 33..512 rows).
SM107_TACTICS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (68, rows): _TILE32_DECODE for rows in range(256, 385, 8)
}

#: The 128-token tile the autotuner serves at 512 rows on sm103 (FC1
#: ``t128x128x512_s3x3x3x3x1x3 … m256x128x64``, FC2 ``t128x128x256_s6 … m256x128x64``), pinned
#: for the 256..504-row decode steps (32 to 63 requests) of the 68-expert window.
_SM103_TILE128_DECODE = (128, 4)

#: ``(local_num_experts, rows) -> (tile_n, config)`` on sm103.
SM103_TACTICS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (68, rows): _SM103_TILE128_DECODE for rows in range(256, 505, 8)
}

_TABLES = {(10, 3): SM103_TACTICS, (10, 7): SM107_TACTICS}


def table_for(capability: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """The table of a device capability: sm103 or sm107; any other GPU is not supported."""
    table = _TABLES.get(tuple(capability))
    if table is None:
        raise NotImplementedError(
            f"decode expert tactics: no table for compute capability {tuple(capability)} "
            "(measured for sm103 and sm107 only)"
        )
    return table


@lru_cache(maxsize=None)
def _valid(act_type: int, top_k: int, hidden: int, intermediate: int, window: int, rows: int):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner.runner_dict.get((act_type,))
    if runner is None:
        runner = torch.classes.trtllm.FP4BlockScaleMoERunner(act_type)
        FP4BlockScaleMoERunner.runner_dict[(act_type,)] = runner
    configs = runner.get_valid_configs(top_k, hidden, intermediate, window, rows)
    return frozenset(tuple(int(v) for v in c) for c in configs)


def fixed_tactic(
    device: torch.device,
    act_type: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    window: int,
    rows: int,
) -> Optional[List[int]]:
    """The pinned ``[tile_n, config]`` of an expert call, or None to leave it to the autotuner.

    None when the ``(window, rows)`` is not listed, the call is not the SwiGLU expert
    (``act_type`` 0), or the listed tactic is not a valid config of the call's shape; a
    device without a table raises (``table_for``).
    """
    if act_type != 0:
        return None
    tactic = table_for(torch.cuda.get_device_capability(device)).get((window, rows))
    if tactic is None:
        return None
    if tactic not in _valid(act_type, top_k, hidden, intermediate, window, rows):
        return None
    return list(tactic)


def run_fixed(
    tactic: List[int],
    inputs: List[Optional[torch.Tensor]],
    num_experts: int,
    top_k: int,
    intermediate: int,
    window: int,
    routing_method_type: int,
    act_type: int,
    tune_max_num_tokens: int,
) -> List[torch.Tensor]:
    """The unfinalized pre-routed expert call on ``tactic``, without the autotuner.

    ``inputs`` are ``FP4BlockScaleMoEInputs``'s 18 tensors in order (``routing_logits``
    None, ``topk_weights`` / ``topk_ids`` the caller's). Returns what
    ``trtllm::fp4_block_scale_moe_runner`` returns for ``do_finalize=False``.
    """
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner(
        num_experts,
        top_k,
        None,
        None,
        intermediate,
        0,
        window,
        None,
        routing_method_type,
        False,
        act_type,
        tune_max_num_tokens=tune_max_num_tokens,
    )
    return runner(inputs, tactic=tactic)
