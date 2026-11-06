# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared PrimTS plan construction; geometry, reducer policy and caches stay target-owned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch


@dataclass(frozen=True)
class FullCtxGeometry:
    heads: int
    kv_heads: int
    head_dim: int
    page: int
    policy_kv_len: int
    heads_per_kv: int


class FullCtxPlanBase:
    """Construct one plan with its own workspace and target compile geometry."""

    GEOMETRY: FullCtxGeometry

    def __init__(
        self,
        device: torch.device,
        qlen: int,
        nreq: int,
        max_kv_len: int,
        recipe: Mapping,
        own_reducer: bool,
        pdl: bool,
    ):
        from tensorrt_llm._torch.attention.backends.prims_ts import decode as tsdec
        from tensorrt_llm._torch.attention.backends.prims_ts.kernels.fmha_decode.fmha_decode_config import (
            get_max_active_clusters_for_cluster_size,
            make_decode_config,
        )

        self._tsdec = tsdec
        self.qlen = qlen
        self.nreq = nreq
        self.pdl = pdl
        self.recipe = dict(recipe)
        splits = int(self.recipe["splits"])
        fp8 = tsdec._cutlass_dtype("float8_e4m3fn")
        # PDL contract of the main kernel (prims_ts fmha_decode_kernel.py): every CTA acquires its producer
        # (griddepcontrol.wait) before any task reads q, lengths or tables, and triggers its dependents
        # (launch_dependents) only after its own tail: an active CTA after its last store and TMEM dealloc, an
        # inactive split or zero-work CTA right after its wait. The trigger lets the merge launch before the
        # slowest CTA ends, so a dependent may read only step-invariant data (weights) before its own
        # griddepcontrol.wait, never this step's partial O/LSE, output, q, lengths or tables; the merge
        # (minimax_m3_sm107_tp2_fullctx_reduce) waits as its first instruction.
        cfg = make_decode_config(
            headdim=self.GEOMETRY.head_dim,
            args={"use_pdl": pdl, **self.recipe.get("config", {})},
            seq_len_q=qlen,
            seq_len_kv=min(max_kv_len, self.GEOMETRY.policy_kv_len),
            batch_size=nreq,
            num_heads_q=self.GEOMETRY.heads,
            num_heads_kv=self.GEOMETRY.kv_heads,
            q_dtype=fp8,
            k_dtype=fp8,
            v_dtype=fp8,
            o_dtype=tsdec._cutlass_dtype("float16"),
            qkv_layout="pagedKv",
            num_tokens_per_page=self.GEOMETRY.page,
            storage_tokens_per_page=self.GEOMETRY.page,
            split_kv_mode="gmem_reduction_with_separate_kernel",
            splits_kv=splits,
            max_splits_kv=splits,
            min_loop_iters_per_split=1,
            mask_type="causal" if qlen > 1 else "dense",
            auto_tuner=True,
            split_kv=True,
        )
        check_layout(cfg, qlen, splits, heads_per_kv=self.GEOMETRY.heads_per_kv)
        with torch.cuda.device(device):
            max_active = get_max_active_clusters_for_cluster_size(1)
        spec = tsdec._decode_launch_spec_from_config(
            cfg,
            batch_size=nreq,
            num_qo_heads=self.GEOMETRY.heads,
            num_kv_heads=self.GEOMETRY.kv_heads,
            head_dim=self.GEOMETRY.head_dim,
            seq_len_q=qlen,
            max_active_clusters=int(max_active),
        )
        compile_spec = tsdec._make_decode_compile_spec(
            spec,
            device_index=device_index(device),
            num_qo_heads=self.GEOMETRY.heads,
            num_kv_heads=self.GEOMETRY.kv_heads,
            head_dim=self.GEOMETRY.head_dim,
            page_size=self.GEOMETRY.page,
            max_kv_len=max_kv_len,
            seq_len_q=qlen,
            q_dtype_key="float8_e4m3fn",
            k_dtype_key="float8_e4m3fn",
            v_dtype_key="float8_e4m3fn",
            output_dtype_key="float16",
            use_packed_q=False,
            kv_prefix_mode="dynamic",
            kv_lengths_mode="dynamic",
        )
        self._main, self._primts_reducer = tsdec._get_compiled_decode(compile_spec)
        layout = tsdec._make_decode_workspace_layout(
            spec.scratch_shapes,
            torch.float16,
            use_separate_reduction_kernel=True,
            use_split_kv=True,
        )
        self._workspace_buffer = torch.empty(layout.total_bytes, device=device, dtype=torch.int8)
        self.workspace = tsdec._bind_decode_workspace(self._workspace_buffer, layout)
        self.workspace.split_kv_counter.zero_()
        self.workspace.cu_seqlens_q.zero_()
        self.workspace.attention_sinks.zero_()
        rows = self.GEOMETRY.heads_per_kv * qlen
        if tuple(self.workspace.partial_o.shape) != (
            nreq,
            self.GEOMETRY.kv_heads,
            splits,
            rows,
            self.GEOMETRY.head_dim,
        ):
            raise RuntimeError(
                f"PrimTS split scratch {tuple(self.workspace.partial_o.shape)} is not the "
                f"[requests, KV heads, splits, {rows} rows, {self.GEOMETRY.head_dim}] layout the BF16 merge reads"
            )
        self.own_reducer = own_reducer
        self.q_tokens_per_cta = max(int(cfg.q_tokens_per_cta), 1)
        self.tile_kv = int(cfg.tile_size_kv)
        self.insts_kv = int(cfg.num_insts_kv)
        self.causal = qlen > 1
        shape = (
            (nreq, qlen, self.GEOMETRY.heads, self.GEOMETRY.head_dim)
            if qlen > 1
            else (nreq, self.GEOMETRY.heads, self.GEOMETRY.head_dim)
        )
        self.staging = torch.empty(shape, dtype=torch.float16, device=device)


def check_layout(cfg, qlen: int, splits: int, *, heads_per_kv: int) -> None:
    """The scratch layout and split ranges the BF16 merge (``fullCtxReduce.cu`` ``activeSplits``) assumes.

    Rows are [token, local head]; a row's split ranges are those of its grouped Q CTA, whose first token is
    ``(token // q_tokens_per_cta) * q_tokens_per_cta``; tiles are grouped ``num_insts_kv`` per loop step; no sliding
    window, attention sinks or packed Q.
    """
    problems = []
    if not (cfg.use_split_kv and cfg.use_separate_reduction_kernel):
        problems.append("not split-KV with the separate GMEM reducer")
    if int(cfg.splits_kv) != splits or int(cfg.max_splits_kv) != splits:
        problems.append(f"splits {cfg.splits_kv}/{cfg.max_splits_kv}, recipe {splits}")
    if cfg.use_cluster_smem_reduction or cfg.use_persistent_scheduler:
        problems.append("cluster-SMEM reduction or a persistent scheduler")
    if cfg.use_variable_seqlens_q or cfg.use_sliding_window_causal or cfg.use_attention_sinks:
        problems.append("packed Q, a sliding window or attention sinks")
    if int(cfg.heads_q_per_kv) not in (0, heads_per_kv):
        problems.append(f"{cfg.heads_q_per_kv} query heads per KV head")
    if qlen > 1:
        if not (cfg.groups_tokens_heads_q and cfg.uses_nontrivial_grouped_q_layout):
            problems.append("multi-token Q without the grouped [token, head] tile layout")
        elif int(cfg.q_tokens_per_cta) * heads_per_kv != int(cfg.tile_size_q):
            problems.append(
                f"{cfg.q_tokens_per_cta} tokens per CTA in a {cfg.tile_size_q}-row tile"
            )
    if problems:
        raise RuntimeError("PrimTS recipe outside the BF16 merge's layout: " + "; ".join(problems))


def kv_page_views(pool: torch.Tensor, subpages_per_slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """K and V page views of a flat sub-page pool (K of slot s at sub-page s * f, V one later), indexed by slot."""
    return pool[0::subpages_per_slot], pool[1::subpages_per_slot]


def device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()
