# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-off decode module for large mixed batches (TLLM_MOE_KF_DECODE_MODULE=
tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_offline): two alternative MoE kernels with the same pre-routed
contract, picked by the call's token count (T < TLLM_MOE_KF_OFFLINE_SPLIT_TOKENS, default 7296: kf_moe_w4a8_offline_lo;
otherwise kf_moe_w4a8_offline_hi). The window floor and routing come from the wrapper below."""
import os
import sys

from tensorrt_llm._torch.cute_dsl_kernels import kf_moe_w4a8_offline_hi as _hi
from tensorrt_llm._torch.cute_dsl_kernels import kf_moe_w4a8_offline_lo as _lo

_SPLIT = int(os.environ.get("TLLM_MOE_KF_OFFLINE_SPLIT_TOKENS", "7296"))
print(f"[TRT-LLM] [W] [TLLM_MOE_KF_DECODE] offline composition: T < {_SPLIT} on kf_moe_w4a8_offline_lo, "
      f"T >= {_SPLIT} on kf_moe_w4a8_offline_hi",
      file=sys.stderr, flush=True)


def run(*args):
    return (_lo.run if args[0].shape[0] < _SPLIT else _hi.run)(*args)


# ---------------------------------------------------------------------------------------------------------------------
# Token window around the two kernels above:
# - T < TLLM_MOE_KF_OFFLINE_MIN_TOKENS (default 2048; below it the in-tree decode module was faster in testing): the
#   in-tree decode module (kf_moe_w4a8) runs the call, router_logits passed through, i.e. exactly what runs without
#   this module;
# - T >= it: the kernels above, pre-routed. Routing ids/bf16 weights come from the in-tree module's route() (k_route
#   alone) or, where the in-tree module has none, the hook's own torch routing (top-4 over float logits, softmax over
#   the four, bf16).
# route / run_unfinalized / SUPPORTS_ROUTER_LOGITS mirror the in-tree module: prefill-route calls (TLLM_MOE_KF_PREFILL)
# keep their routing, and the unfinalized form (TLLM_GPTOSS_FUSE_FINALIZE_NORM) runs the in-tree kernel for every T.
import os as _os
import sys as _sys

import torch as _torch

from tensorrt_llm._torch.cute_dsl_kernels import kf_moe_w4a8 as _tree

_offline_run = run
_MIN = int(_os.environ.get("TLLM_MOE_KF_OFFLINE_MIN_TOKENS", "2048"))
_tree_route = getattr(_tree, "route", None)
SUPPORTS_ROUTER_LOGITS = bool(getattr(_tree, "SUPPORTS_ROUTER_LOGITS", False))
if _tree_route is not None:
    route = _tree_route
if hasattr(_tree, "run_unfinalized"):
    run_unfinalized = _tree.run_unfinalized
print(f"[TRT-LLM] [W] [TLLM_MOE_KF_DECODE] offline module composed: T >= {_MIN} on the offline kernels, below on the "
      f"in-tree kernel; routing={'in-kernel' if _tree_route is not None else 'torch'}", file=_sys.stderr, flush=True)


def _route(router_logits):
    if _tree_route is not None:
        return _tree_route(router_logits)
    values, ids = _torch.topk(router_logits.float(), 4, dim=-1)
    return ids.to(_torch.int32), _torch.softmax(values, dim=-1).to(_torch.bfloat16)


def run(*args, router_logits=None):
    if args[0].shape[0] < _MIN:
        if router_logits is not None:
            return _tree.run(*args, router_logits=router_logits)
        return _tree.run(*args)
    if router_logits is not None:
        ids, weights = _route(router_logits)
        args = args[:2] + (ids, weights) + args[4:]
    return _offline_run(*args)
