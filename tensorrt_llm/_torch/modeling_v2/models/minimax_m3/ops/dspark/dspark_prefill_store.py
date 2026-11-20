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
# DSpark / DFlash drafter PREFILL context K / V store: the decode store (dspark_ctx_store) per context chunk
# =============================================================================
#
# DEP2CTX_DSPARK_STORE. DFlashWorker._store_prefill_context runs, per context chunk,
# precompute_context_kv (GEMM, two K / V copies, k_norm, weight multiply, the dummy-q flashinfer RoPE) and
# _store_context_kv_paged (two index casts, per drafter layer two copies + one flashinfer append): 28 launches. This
# hook hands the chunk to dspark_ctx_store.fused_ctx_store instead: the same GEMM and k_norm (torch's own), the weight
# multiply folded into the kernel (fold_k_norm_weight, bitwise torch's bf16 multiply), RoPE + store in one launch of
# family_ops().minimax_m31_dspark_ctx_store (its generic build): 4 launches. Bitwise the stock
# lines; the one exception is a NaN payload (the kernel's x * 1.0 canonicalizes it), which needs an already-broken
# draft.
#
# How a chunk maps onto the decode op (one request of tokens_per_request = chunk length rows):
# - positions = cols = arange(start, end): the RoPE position and the column of every row, exactly what the stock
#   caller passes to both precompute_context_kv and the append. Given columns (not derived from positions), so the
#   kernel's capacity clamp never applies: the address is block_table[row, col // page], col % page, flashinfer's
#   append addressing (dense table, indptr = arange * width; dflash.py _ctx_paged_index_args).
# - num_accepted = INT32_MAX: every row is accepted, so the accept mask is 1.0 (x * 1.0 is x for every non-NaN bf16).
# - request_rows = [row].
# The three index tensors are views of persistent int64 buffers grown outside capture: no launch per chunk.
#
# Kill switches (read once, at import, by the target's m31_switches): TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE=1 (the
# target installs nothing: today's path, its position copy included); TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE_FOLD=1
# (the weight multiply stays torch's); TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1 (lean glue off for the chunks the store
# declines); the op's own TRTLLM_M31_DISABLE_DSPARK_CTX_STORE_CPP=1 (fused_ctx_store returns False: the stock lines run
# per chunk).
# TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE=1 is the only switch that restores the stock path exactly: under the op
# switch alone every chunk declines into the stock lines, but with lean glue on, the hook's call
# per chunk and the eager position copy gone (the same bytes, not the same launches).
#
# The fold flag is process-wide and also read by the decode store. ``install`` sets it only on a worker with no
# decode store installed (a context-only role); with one present it leaves the flag to that store's installer and
# reports the flag as it stands (FOLD=<on|off> (left to S10C_A)): fused_ctx_store reads it for the prefill chunks too.

from typing import Callable, Dict, Optional, Sequence, Union

import torch

from .dspark_ctx_store import _fold_w, fold_k_norm_weight, fused_ctx_store

#: m = off < num_accepted is 1 for every row of any chunk the op takes (n <= INT32_MAX, minimaxM31DsparkCtxStoreOp.cpp).
_ALL_ACCEPTED = torch.iinfo(torch.int32).max


class PrefillStore:
    """The ``enable_fused_prefill_store`` hook: one ``fused_ctx_store`` call per context chunk, or False (the stock
    lines run) for a drafter / cache / state it does not take."""

    def __init__(self, store: Callable[..., bool] = fused_ctx_store):
        self._store = store
        self._arange: Optional[torch.Tensor] = (
            None  # int64 [0, n): positions / cols, and request rows
        )
        self._nacc: Optional[torch.Tensor] = None  # int64 [1] = INT32_MAX
        self.calls = 0
        self.taken = 0
        self._logged: set = set()

    def _index_buffers(self, n: int, device: torch.device) -> None:
        """Grow the persistent arange to at least ``n`` (geometric; the caller has ruled out capture)."""
        a = self._arange
        if a is not None and a.device == device and a.numel() >= n:
            return
        size = max(n, 2 * a.numel() if a is not None and a.device == device else 0, 4096)
        self._arange = torch.arange(size, dtype=torch.int64, device=device)
        if self._nacc is None or self._nacc.device != device:
            self._nacc = torch.full((1,), _ALL_ACCEPTED, dtype=torch.int64, device=device)

    def __call__(
        self,
        draft_model,
        projected: torch.Tensor,
        start: int,
        end: int,
        row: int,
        block_table: torch.Tensor,
        kv_buf: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> bool:
        self.calls += 1
        n = end - start
        if n <= 0 or start < 0 or row < 0 or projected.shape[0] != n:
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # The hook contract (DFlashWorker.enable_fused_prefill_store): decline under capture, before anything can
            # allocate (the geometry build, the index buffers) or launch. The store runs after the captured decoder
            # body (breakable prefill graphs capture only the layer forward), so this is a guard, not a path.
            return False
        # ctx_store_supported reads the drafter's fused-KV geometry (_head_dim, _k_norm_stacked, ...), which
        # precompute_context_kv builds lazily on its first call; build it here so the first chunk is taken too.
        if getattr(draft_model, "_fused_kv_weight", 0) is None and hasattr(
            draft_model, "_build_fused_kv_buffers"
        ):
            draft_model._build_fused_kv_buffers()
        self._index_buffers(max(end, row + 1), projected.device)
        pos = self._arange[start:end]
        ok = self._store(
            draft_model,
            projected,
            pos,
            self._nacc,
            self._arange[row : row + 1],
            pos,  # given columns = positions: no capacity clamp
            block_table,
            kv_buf,
            n,
        )
        self.taken += bool(ok)
        self._log_first(bool(ok), n)
        return bool(ok)

    def _log_first(self, ok: bool, n: int) -> None:
        """One engagement line for the first chunk taken and one for the first chunk declined (taken shows the op
        ran on a real chunk, declined names a layout that fell back)."""
        if ok in self._logged:
            return
        self._logged.add(ok)
        from tensorrt_llm.logger import logger

        if ok:
            logger.info(
                f"[M3 dispatch] DEP2CTX_DSPARK_STORE taken: first prefill chunk n={n} (call {self.calls})"
            )
        else:
            logger.info(
                f"[M3 dispatch] DEP2CTX_DSPARK_STORE declined: prefill chunk n={n}; stock lines"
            )


def install(
    spec_worker,
    draft_model,
    *,
    disabled: Callable[[str], bool],
    store: Optional[PrefillStore] = None,
) -> Dict[str, bool]:
    """DEP2CTX_DSPARK_STORE on a context-only target: the lean glue (the stock lines' copies, for chunks the store
    declines), the per-chunk fused prefill store, and the k_norm weight fold. {} for a worker without the hook;
    every item off under TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE. ``fold_k_norm_weight`` is process-wide and the decode
    store reads it too, so it is set only while this worker has no decode store installed; with one present
    the flag is left to that store's installer and FOLD reports the flag as it stands, since the prefill chunks read
    it too (the fold is bitwise either way)."""
    if spec_worker is None:
        return {}
    if not hasattr(spec_worker, "enable_fused_prefill_store"):
        from tensorrt_llm.logger import logger

        logger.info(
            "[M3 dispatch] DEP2CTX_DSPARK_STORE not installed: "
            f"{type(spec_worker).__name__} has no enable_fused_prefill_store"
        )
        return {}
    if disabled("DEP2CTX_DSPARK_STORE"):
        return {"S10": False, "STORE": False, "FOLD": False}
    on: Dict[str, bool] = {}
    lean = not disabled("DRAFTER_LEAN_GLUE")
    spec_worker.enable_lean_glue(draft_model, lean)
    on["S10"] = lean
    on["STORE"] = spec_worker.enable_fused_prefill_store(
        store if store is not None else PrefillStore()
    )
    if getattr(spec_worker, "_fused_ctx_store", None) is not None:
        on["FOLD"] = on["STORE"] and bool(_fold_w[0])
        from tensorrt_llm.logger import logger

        logger.info(
            f"[M3 dispatch] DEP2CTX_DSPARK_STORE FOLD={'on' if on['FOLD'] else 'off'} (left to S10C_A): the "
            "process-wide k_norm fold flag is the decode store installer's"
        )
    else:
        on["FOLD"] = fold_k_norm_weight(on["STORE"] and not disabled("DEP2CTX_DSPARK_STORE_FOLD"))
    return on


def engagement(on: Dict[str, bool]) -> str:
    """One line for the install log."""
    if not on:
        return "not installed (no spec worker, or it has no enable_fused_prefill_store)"
    return " ".join(f"{k}={'on' if v else 'off'}" for k, v in on.items())
