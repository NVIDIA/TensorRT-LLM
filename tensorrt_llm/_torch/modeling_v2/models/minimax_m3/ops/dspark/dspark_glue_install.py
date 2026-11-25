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
# =============================================================================
# DSpark / DFlash drafter glue: the one install every M3.1 target can call
# =============================================================================
#
# The TP targets' prebuild set (the lean glue, the full-slot arena, the fused drafter glue: ctx K / V store, drafter
# inputs, ctx advance (+ ADV_PAR), block-table refresh (+ its grid form), forward indices; the q / k norm + RoPE (+ the
# K / V append form) and the draft KV-length rewind), installed through the in-tree DFlashWorker hooks. Every hook keeps
# the stock lines for any layout its op does not reproduce, and every op here is CUDA C++ (or the existing CuTe DSL
# rewind): without the op the catalog wrapper returns None and the stock torch lines run, never Triton. Each item is
# bitwise to the stock lines, except the lean glue at draft lengths above 3 on the TP targets (see S10_MAX_DRAFT_LEN),
# which stays off there; a target without the drafter all-reduce (DEP) passes ``s10_any_draft_len``.
#
# Switches (read by the caller's ``disabled``, by default the TRTLLM_M31_DISABLE_ environment words): DRAFTER_LEAN_GLUE,
# DRAFTER_FUSED_GLUE (all of the fused glue), DRAFTER_CTX_KV_STORE_FUSED, DRAFTER_BLOCK_TABLE_REFRESH_FUSED,
# DRAFTER_BLOCK_TABLE_REFRESH_GRID, DRAFTER_INPUTS_FUSED, DRAFTER_CTX_ADVANCE_FUSED, ADV_PAR,
# DRAFTER_FORWARD_INDICES_FUSED, EX11 (the q / k norm + RoPE), DRAFT_APPEND_FUSE, STRICT_ACCEPT_FUSED,
# DRAFT_KV_REWIND_FUSED.

import functools
import os
from typing import Callable, Dict, Optional

import torch

from . import strict_accept as _strict_accept
from .dspark_block_tables import enable_parallel_refresh, refresh_block_tables
from .dspark_ctx_advance import advance_ctx_and_read_view
from .dspark_ctx_store import fused_ctx_store
from .dspark_drafter_inputs import drafter_inputs as _drafter_inputs
from .dspark_forward_indices import forward_indices
from .dspark_qk_norm_rope import qk_norm_rope, qk_norm_rope_append

#: The lean glue (the drafter's strided-view glue) is bitwise on SM107 at draft length 3, not at 7 (a TP4 draft length
#: 7 run diverges from its run without the glue); 4..6 are not checked. The divergence is the TP drafter all-reduce;
#: DEP has none, and its lean-glue sites at draft length 7 are bitwise (``s10_any_draft_len``).
S10_MAX_DRAFT_LEN = 3

_SET = ("1", "true", "on", "yes")


def _env_disabled(item: str) -> bool:
    return os.environ.get(f"TRTLLM_M31_DISABLE_{item}", "").strip().lower() in _SET


def replicated_embedding_inputs(
    embed_tokens,
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    mask_token_id,
    trained_mask_embedding,
    block_size,
    tokens_per_request,
):
    """The fused drafter inputs for the in-tree ``Embedding`` when it is an exact gather of a replicated table: one rank
    (tp_size 1, the attention-DP form) and no padding. Its forward then masks an id outside [0, vocab) to +0, as the op
    does. None for any other embedding (a TP target passes its own wrapper): the stock path runs."""
    from tensorrt_llm._torch.modules.embedding import Embedding

    if (
        type(embed_tokens) is not Embedding
        or getattr(embed_tokens, "tp_size", 1) != 1
        or getattr(embed_tokens, "padding_size", 0) != 0
    ):
        return None
    table = embed_tokens.weight
    if table.dim() != 2:
        return None
    mask_row = (
        trained_mask_embedding if trained_mask_embedding is not None else table[int(mask_token_id)]
    )
    return _drafter_inputs(
        accepted,
        num_accepted,
        slots,
        ctx_len,
        max_ctx,
        table,
        mask_row,
        block_size,
        tokens_per_request,
    )


class KvRewind:
    """The draft KV-length rewind for the DFlash hook: ``1 - num_accepted[nc:bs]`` (a view of a persistent int32 buffer)
    with ``kv_lens[nc:bs] += 1`` in one launch, or None (the stock lines) for operands or a variant it does not take."""

    def __init__(self, device: torch.device):
        _strict_accept.precompile_kv_rewind(device)
        self.buf = torch.empty(_strict_accept.max_requests(), dtype=torch.int32, device=device)

    def __call__(self, num_accepted, kv_lens, nc: int, bs: int):
        n = bs - nc
        if not _strict_accept.supports_kv_rewind(num_accepted, kv_lens, self.buf, nc, n):
            return None
        if not _strict_accept.is_compiled_kv_rewind(num_accepted, kv_lens):
            return None
        return _strict_accept.spec_kv_rewind(num_accepted, kv_lens, self.buf, nc, n)


def install_dspark_glue(
    spec_worker,
    draft_model,
    *,
    draft_len: int,
    device: Optional[torch.device],
    inputs: Optional[Callable] = replicated_embedding_inputs,
    s10_any_draft_len: bool = False,
    disabled: Callable[[str], bool] = _env_disabled,
) -> Dict[str, bool]:
    """Install the set on ``spec_worker`` (an in-tree DFlashWorker) and ``draft_model``; return {item: on}. Empty for a
    worker without the hooks. ``inputs`` is the fused drafter-inputs wrapper for the target's embedding (None: off);
    ``device`` None skips the KV-length rewind (its compiles need one); ``s10_any_draft_len`` lifts the lean glue's
    draft-length bound (a target without the drafter all-reduce; DISABLE_DRAFTER_LEAN_GLUE still wins)."""
    on: Dict[str, bool] = {}
    if spec_worker is None or not hasattr(spec_worker, "enable_fused_ctx_store"):
        return on
    lean = not disabled("DRAFTER_LEAN_GLUE") and (
        s10_any_draft_len or int(draft_len) <= S10_MAX_DRAFT_LEN
    )
    spec_worker.enable_lean_glue(draft_model, lean)
    on["S10"] = lean
    spec_worker.enable_full_slot_arena()
    on["ARENA"] = True
    s10c = not disabled("DRAFTER_FUSED_GLUE")

    def pick(item: str, fn):
        return fn if s10c and not disabled(item) else None

    on["S10C_A"] = spec_worker.enable_fused_ctx_store(
        pick("DRAFTER_CTX_KV_STORE_FUSED", fused_ctx_store)
    )
    on["S10C_C"] = spec_worker.enable_fused_drafter_inputs(pick("DRAFTER_INPUTS_FUSED", inputs))
    on["S10C_D"] = spec_worker.enable_fused_ctx_advance(
        pick(
            "DRAFTER_CTX_ADVANCE_FUSED",
            functools.partial(advance_ctx_and_read_view, parallel=not disabled("ADV_PAR")),
        )
    )
    on["S10C_B"] = spec_worker.enable_fused_block_refresh(
        pick("DRAFTER_BLOCK_TABLE_REFRESH_FUSED", refresh_block_tables)
    )
    on["S10C_B2"] = bool(
        enable_parallel_refresh(on["S10C_B"] and not disabled("DRAFTER_BLOCK_TABLE_REFRESH_GRID"))
    )
    on["S10C_E"] = spec_worker.enable_fused_forward_indices(
        draft_model, pick("DRAFTER_FORWARD_INDICES_FUSED", forward_indices)
    )
    on["EX11"] = spec_worker.enable_fused_qk_norm_rope(
        draft_model, None if disabled("EX11") else qk_norm_rope
    )
    on["APPEND"] = spec_worker.enable_fused_qk_norm_rope_append(
        draft_model,
        qk_norm_rope_append if on["EX11"] and not disabled("DRAFT_APPEND_FUSE") else None,
    )
    rewind = None
    if (
        device is not None
        and not disabled("STRICT_ACCEPT_FUSED")
        and not disabled("DRAFT_KV_REWIND_FUSED")
    ):
        rewind = KvRewind(device)
    on["N3A_KV"] = spec_worker.enable_fused_kv_rewind(rewind)
    return on


def engagement(on: Dict[str, bool]) -> str:
    """One line for the install log: the items on and off."""
    if not on:
        return "not installed (no DFlash worker)"
    return " ".join(f"{k}={'on' if v else 'off'}" for k, v in on.items())
