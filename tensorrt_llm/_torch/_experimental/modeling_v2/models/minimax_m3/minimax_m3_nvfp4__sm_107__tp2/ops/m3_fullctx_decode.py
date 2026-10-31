# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 full-context FP8 decode attention on Rubin: ``fullctx_decode``.

The dense layers (0-2) attend every token of their FP8 E4M3 (unscaled) paged KV cache. On SM 10.7 the stock call
(flashinfer trtllm-gen, ``minimax_m3_trtllm_gen_dense_decode``) caps its split at 16 cluster CTAs per (request, KV
head): at TP2 (2 KV heads) one request runs on 32 CTAs of 212 and takes 28-30 us for 85 MB at 165k tokens.

This runs the same arithmetic -- FP8 Q (the same E4M3 rounding of q), FP8 K/V, P quantized to E4M3 per split -- on
the vendored PrimTS decode kernel (``attention/backends/prims_ts``, SM 10.7 through the sm_100f family) at a pinned
split-KV recipe, which fans one request out over 96-144 CTAs. The split partials are merged by
``torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce`` (``cpp/tensorrt_llm/kernels/minimaxM3Sm107Tp2/
fullCtxReduce.cu``): unnormalized fp32 accumulation over the active splits, one final division, one BF16 rounding,
the merge form of stock trtllm-gen. Only the split changes against stock (modeling's ``FULLCTX_DECODE_RECIPES``).

A plan is compiled per (device, query rows per request, requests) by ``prebuild`` before the executor's warmup; the
forward only reads plans (``plan_for``). The plan is built from PrimTS's own config, compile and workspace helpers at
an explicit recipe; the PrimTS wrapper's planning policy is not used, so no PrimTS state is replaced. With
``own_reducer`` off the PrimTS reducer merges into FP16 and the result is cast to BF16.
"""

from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional, Tuple

import torch

HEADS = 32
KV_HEADS = 2  # capacity: fixed: the tp2 KV heads per rank
HEAD_DIM = 128  # capacity: fixed: the checkpoint's head dim
PAGE = 128
# The per-request K/V length PrimTS's config selection sizes its policy for (Q/KV tiles, launch mode); the pinned
# recipes were measured with it. The compiled capacity (``max_kv_len``) may be larger.
POLICY_KV_LEN = (
    1 << 20
)  # capacity: fixed: the PrimTS policy's K/V length bound (the plans' compile shape)
HEADS_PER_KV = HEADS // KV_HEADS  # capacity: fixed: the tp2 query heads per KV head

_plans: Dict[Tuple[int, int, int, bool], "FullCtxPlan"] = {}
_lock = threading.Lock()


class FullCtxPlan:
    """One compiled PrimTS split-KV decode at a pinned recipe, its workspace and its FP16 staging output."""

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
            headdim=HEAD_DIM,
            args={"use_pdl": pdl, **self.recipe.get("config", {})},
            seq_len_q=qlen,
            seq_len_kv=min(max_kv_len, POLICY_KV_LEN),
            batch_size=nreq,
            num_heads_q=HEADS,
            num_heads_kv=KV_HEADS,
            q_dtype=fp8,
            k_dtype=fp8,
            v_dtype=fp8,
            o_dtype=tsdec._cutlass_dtype("float16"),
            qkv_layout="pagedKv",
            num_tokens_per_page=PAGE,
            storage_tokens_per_page=PAGE,
            split_kv_mode="gmem_reduction_with_separate_kernel",
            splits_kv=splits,
            max_splits_kv=splits,
            min_loop_iters_per_split=1,
            mask_type="causal" if qlen > 1 else "dense",
            auto_tuner=True,
            split_kv=True,
        )
        _check_layout(cfg, qlen, splits)
        with torch.cuda.device(device):
            max_active = get_max_active_clusters_for_cluster_size(1)
        spec = tsdec._decode_launch_spec_from_config(
            cfg,
            batch_size=nreq,
            num_qo_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            seq_len_q=qlen,
            max_active_clusters=int(max_active),
        )
        compile_spec = tsdec._make_decode_compile_spec(
            spec,
            device_index=_index(device),
            num_qo_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=PAGE,
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
        rows = HEADS_PER_KV * qlen
        if tuple(self.workspace.partial_o.shape) != (nreq, KV_HEADS, splits, rows, HEAD_DIM):
            raise RuntimeError(
                f"PrimTS split scratch {tuple(self.workspace.partial_o.shape)} is not the "
                f"[requests, KV heads, splits, {rows} rows, {HEAD_DIM}] layout the BF16 merge reads"
            )
        self.own_reducer = own_reducer
        self.q_tokens_per_cta = max(int(cfg.q_tokens_per_cta), 1)
        self.tile_kv = int(cfg.tile_size_kv)
        self.insts_kv = int(cfg.num_insts_kv)
        self.causal = qlen > 1
        shape = (nreq, qlen, HEADS, HEAD_DIM) if qlen > 1 else (nreq, HEADS, HEAD_DIM)
        self.staging = torch.empty(shape, dtype=torch.float16, device=device)

    def run(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
    ) -> None:
        # The kernel takes a compact query; a column view of the fused QKV projection is copied once.
        q = q.reshape(self.staging.shape).contiguous()
        runtime = self._tsdec._prepare_decode_runtime_unchecked(
            q,
            (k_pages, v_pages),
            None,
            output_dtype=torch.float16,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            out=self.staging,
        )
        ws = self.workspace
        self._main(
            runtime.q,
            runtime.k_cache,
            runtime.v_cache,
            runtime.k_sf_cache,
            runtime.v_sf_cache,
            runtime.out,
            seq_lens,
            ws.cu_seqlens_q,
            block_table,
            block_table,  # Dense decode constexpr-elides the membership table.
            ws.partial_o,
            ws.partial_stats,
            ws.split_kv_counter,
            ws.attention_sinks,
            runtime.num_physical_pages,
            runtime.k_page_stride,
            runtime.k_head_stride,
            runtime.k_token_stride,
            runtime.v_page_stride,
            runtime.v_head_stride,
            runtime.v_token_stride,
            runtime.bmm1_scale,
            runtime.bmm2_scale,
            (),
        )
        if self.own_reducer:
            torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce(
                ws.partial_o,
                ws.partial_stats,
                seq_lens,
                output,
                self.qlen,
                self.q_tokens_per_cta,
                self.tile_kv,
                self.insts_kv,
                self.causal,
                self.pdl,
            )
            return
        self._primts_reducer(
            runtime.out,
            seq_lens,
            ws.cu_seqlens_q,
            ws.partial_o,
            ws.partial_stats,
            ws.attention_sinks,
            runtime.bmm1_scale,
            runtime.bmm2_scale,
            (),
        )
        output.view(self.staging.shape).copy_(self.staging)


def _check_layout(cfg, qlen: int, splits: int) -> None:
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
    if int(cfg.heads_q_per_kv) not in (0, HEADS_PER_KV):
        problems.append(f"{cfg.heads_q_per_kv} query heads per KV head")
    if qlen > 1:
        if not (cfg.groups_tokens_heads_q and cfg.uses_nontrivial_grouped_q_layout):
            problems.append("multi-token Q without the grouped [token, head] tile layout")
        elif int(cfg.q_tokens_per_cta) * HEADS_PER_KV != int(cfg.tile_size_q):
            problems.append(
                f"{cfg.q_tokens_per_cta} tokens per CTA in a {cfg.tile_size_q}-row tile"
            )
    if problems:
        raise RuntimeError("PrimTS recipe outside the BF16 merge's layout: " + "; ".join(problems))


def has_own_reducer() -> bool:
    """Require the BF16 split-KV merge op supplied by this target build."""
    torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce
    return True


def plan_for(
    device: torch.device, qlen: int, nreq: int, own_reducer: bool = True
) -> Optional[FullCtxPlan]:
    """The prebuilt plan for this shape, or None."""
    return _plans.get((_index(device), qlen, nreq, own_reducer))


def prebuild(
    device: torch.device,
    recipes: Mapping[Tuple[int, int], Mapping],
    max_kv_len: int,
    own_reducer: bool,
    pdl: bool,
) -> int:
    """Compile the plan of every (query rows per request, requests) in ``recipes``. Call before warmup."""
    built = 0
    for (qlen, nreq), recipe in recipes.items():
        key = (_index(device), qlen, nreq, own_reducer)
        with _lock:
            if key not in _plans:
                _plans[key] = FullCtxPlan(device, qlen, nreq, max_kv_len, recipe, own_reducer, pdl)
                built += 1
    return built


def kv_page_views(pool: torch.Tensor, subpages_per_slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """K and V page views of a flat sub-page pool (K of slot s at sub-page s * f, V one later), indexed by slot."""
    return pool[0::subpages_per_slot], pool[1::subpages_per_slot]


def _index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()
