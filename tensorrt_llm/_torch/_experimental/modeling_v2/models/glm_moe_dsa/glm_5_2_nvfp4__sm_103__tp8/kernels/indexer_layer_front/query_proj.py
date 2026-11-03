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
"""indexer_layer_front entry into the absorbed_query / draft_absorbed_query path of full-indexer layers.

``dsa_proj`` serves the full-indexer target layers (absorbed_query's domain) and the
one-model MTP draft layer (draft_absorbed_query's domain). It runs the native bridges absorbed_query
runs there: the kv_a GEMM (``kv_a_proj_with_mqa``) and, with
``IndexerFrontConfig.norm="native"`` (default), the native ``q_a_layernorm`` on the
current stream and, with absorbed_query's ``indexer_k_on_aux`` and multi-stream enabled,
the BF16 ``wk|weights_proj`` GEMM and the native ``k_norm`` on the indexer's
aux stream, joined before the kernel. With ``norm="inline"`` or
``"programs"`` the native ``q_a_layernorm`` launch is dropped and the indexer_layer_front
kernel (``kernels.front_qproj_indexer_kernel``) normalizes q_a itself. Every
mode writes the same query_front packet as absorbed_query's PRENORM path, byte for byte:
``q``, ``latent`` (``compressed_kv`` / ``k_pe`` views), ``fused_q[..., :512]``
(preabsorbed form) and the indexer outputs ``q_fp8``, ``k_fp8``, ``k_scale``,
``weights``, ``q_scale``.

Every decline happens before any GPU work and returns ``None``; the adapter
then runs absorbed_query's / draft_absorbed_query's entry unchanged. The decision is a static check
of replicated module state (identical on every rank) plus absorbed_query's per-call check.

Option ``skip_reused_iq`` (``IndexerFrontQueryProjection.skip_reused_iq``, off
in this target; tests set it on a runner) is a declared ABI change: on an MTP
draft step whose indexer reuses the shared top-k (the indexer's own ``reuse_topk`` condition:
``mtp_index_share``, ``in_mtp_draft_loop``, ``indexer_skip_topk`` and shared
rows present), the kernel skips the indexer ``wq_b`` producers and the indexer
query post-processing, leaving ``q_fp8``, ``q_scale`` and ``weights`` of the
packet unwritten (the reusing indexer reads none of them). Every other output is
unchanged.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ...dispatch import is_enabled, log_capability
from ..absorbed_query.query_proj import (
    _HEAD_DIM,
    _KV_LORA,
    _LOCAL_HEADS,
    _NOPE,
    _Q_LORA,
    _ROPE,
    AbsorbedQueryPacket,
    AbsorbedQueryProjection,
    KernelConfig,
    absorb_for,
    absorb_layout,
    call_eligibility,
    check_package_selectors,
    module_eligibility,
)
from . import kernels

FEATURE_ID = "indexer_layer_front"
# Host-side decisions of skip_reused_iq modules ("skip" / "full"), for tests and decode boots.
SKIP_DECISIONS: Counter = Counter()


# Where q_a is normalized (kernel constexpr NORM_MODE).
NORM_MODES = {"native": 0, "inline": 1, "programs": 2}


@dataclass(frozen=True)
class IndexerFrontConfig:
    # "native": the native q_a_layernorm launch feeds the kernel (absorbed_query's PRENORM input);
    # "inline": every q_b / indexer wq_b producer normalizes its own K tiles;
    # "programs": norm programs publish qn to the producers through a counter.
    norm: str = "native"
    # Rows per norm program (norm="programs"; FlashInfer uses one warp per row).
    norm_rows: int = 2
    # L2 prefetch of each producer's weight tile before the grid-dependency wait.
    prefetch_weights: bool = False
    # Trigger the dependents (the generation rope) right after each program's last global load.
    early_trigger: bool = True
    # Tests only: skew producers / consumers by up to this many nanoseconds.
    debug_delay_ns: int = 0


def config_for(mode: str, **kwargs) -> IndexerFrontConfig:
    """Tests / benches: ``native``, ``inline`` or ``nr<k>`` (norm programs of ``k`` rows), with
    an optional ``+pf`` suffix (weight L2 prefetch)."""
    base, _, suffix = mode.partition("+")
    if suffix not in ("", "pf"):
        raise ValueError(f"unknown indexer_layer_front mode suffix {mode!r}")
    kwargs.setdefault("prefetch_weights", suffix == "pf")
    if base in ("native", "inline"):
        return IndexerFrontConfig(norm=base, **kwargs)
    if base.startswith("nr"):
        return IndexerFrontConfig(norm="programs", norm_rows=int(base[2:]), **kwargs)
    raise ValueError(f"unknown indexer_layer_front norm mode {mode!r}")


def reuses_shared_topk(mla, attn_metadata) -> bool:
    """The indexer's ``reuse_topk`` condition of this call (host state only)."""
    indexer = mla.mqa.indexer
    return bool(
        getattr(indexer, "mtp_index_share", False)
        and getattr(attn_metadata, "in_mtp_draft_loop", False)
        and getattr(attn_metadata, "indexer_skip_topk", False)
        and getattr(attn_metadata, "shared_topk_indices", None) is not None
    )


def _native_q_norm_reproduced(mla) -> Tuple[bool, str]:
    from ..absorbed_query.shared_indexer import _native_rmsnorm_is_flashinfer_cute

    ok, reason = _native_rmsnorm_is_flashinfer_cute(mla.q_a_layernorm, _Q_LORA)
    return ok, (f"q_a_layernorm: {reason}" if not ok else "")


def front_module_eligibility(mla) -> Tuple[bool, str, bool]:
    """Static per-module check; returns ``(ok, reason, draft)``. No GPU work."""
    num_layers = getattr(mla, "num_hidden_layers", None)
    draft = num_layers is not None and mla.layer_idx is not None and mla.layer_idx >= num_layers
    if draft and not is_enabled("draft_absorbed_query"):
        return False, "draft layer without draft_absorbed_query", draft
    if not draft and not is_enabled("absorbed_query"):
        return False, "target layer without absorbed_query", draft
    ok, reason = module_eligibility(mla, draft=draft)
    if not ok:
        return False, reason, draft
    if mla.mqa.indexer is None:
        return (
            False,
            "shared-indexer layer (no q_a RMSNorm launch to remove; shared_indexer_absorbed_query / "
            "absorbed_query serve it)",
            draft,
        )
    ok, reason = _native_q_norm_reproduced(mla)
    if not ok:
        return False, reason, draft
    return True, "", draft


class IndexerFrontQueryProjection(AbsorbedQueryProjection):
    """absorbed_query's full-layer projection with the indexer_layer_front query kernel
    (``IndexerFrontConfig.norm`` selects the q_a norm)."""

    def __init__(
        self,
        mla,
        config: IndexerFrontConfig = IndexerFrontConfig(),
        kernel_config: KernelConfig = KernelConfig(),
    ):
        # kernel_config: absorbed_query's tiling and options (tests pass debug options).
        super().__init__(mla, kernel_config)
        self.front_config = config
        # skip_reused_iq on this module (off in this target; tests set it on MTP draft layers).
        self.skip_reused_iq = False
        indexer = mla.mqa.indexer
        # [2 * H absorption, IH indexer-head, norm released, norm consumers].
        self.indexer_front_sync = torch.zeros(
            2 * _LOCAL_HEADS + indexer.n_heads + 2,
            dtype=torch.int32,
            device=mla.q_b_proj.weight.device,
        )

    def supported(self) -> Optional[str]:
        cfg = self.config
        if cfg.prefetch or cfg.own_hidden_gemv or not cfg.prenorm:
            return "absorbed_query kernel configuration is not the PRENORM pipelined default"
        if not self.indexer_fusable():
            return "absorbed_query fused indexer kernel not eligible"
        if self.front_config.norm not in NORM_MODES:
            return f"norm mode {self.front_config.norm!r} unknown"
        if self.front_config.norm == "programs" and (
            self.front_config.norm_rows not in (1, 2, 4, 8, 16)
            or self.front_config.norm_rows > cfg.block_m
        ):
            return f"norm_rows {self.front_config.norm_rows} unsupported"
        chunks = cfg.bn * _Q_LORA * 2 // 1024  # 1 KiB L2 prefetches per producer weight tile
        if self.front_config.prefetch_weights and (
            chunks * 1024 != cfg.bn * _Q_LORA * 2 or chunks & (chunks - 1)
        ):
            return f"weight tile of {cfg.bn} x {_Q_LORA} is not a power-of-two number of KiB"
        return None

    def bridges(self, hidden_states: torch.Tensor):
        """absorbed_query's ``native_bridges``; the native ``q_a_layernorm`` only with ``norm="native"``.

        Returns ``(kva, qn, kw, k_ln)``: the native kv_a GEMM output, the
        native ``q_a_layernorm`` of its q_a columns (None unless ``native``),
        the native BF16 fused ``wk|weights_proj`` output and the native
        ``k_norm`` of its key columns (the latter two on the indexer's aux
        stream when absorbed_query's ``indexer_k_on_aux`` is set and multi-stream is
        enabled, joined before returning).
        """
        mla = self.mla
        indexer = mla.mqa.indexer
        native = self.front_config.norm == "native"

        def main():
            kva = mla.kv_a_proj_with_mqa(hidden_states)
            return kva, (self.native_q_norm(kva) if native else None)

        def aux():
            kw = torch.nn.functional.linear(hidden_states, indexer._fused_wk_wp_weight)
            return kw, indexer.k_norm(kw[:, : indexer.head_dim])

        if self.config.indexer_k_on_aux:
            from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

            (kva, qn), (kw, k_ln) = maybe_execute_in_parallel(
                main, aux, self.events[0], self.events[1], indexer.aux_stream
            )
        else:
            kva, qn = main()
            kw, k_ln = aux()
        return kva, qn, kw, k_ln

    def launch_front(
        self,
        kva: torch.Tensor,
        kw: torch.Tensor,
        k_ln: torch.Tensor,
        position_ids: torch.Tensor,
        absorb: bool,
        early_trigger: Optional[bool] = None,
        qn: Optional[torch.Tensor] = None,
        skip_iq: bool = False,
    ) -> AbsorbedQueryPacket:
        """The indexer_layer_front kernel on the bridge outputs (same packet as ``launch_indexer`` with PRENORM).

        ``qn``: the native ``q_a_layernorm`` output (``norm="native"`` only).
        ``skip_iq``: ``skip_reused_iq`` (``q_fp8``, ``q_scale``, ``weights`` left unwritten).
        """
        mla = self.mla
        indexer = mla.mqa.indexer
        cfg = self.config
        rows = kva.shape[0]
        assert (
            kva.stride(1) == 1 and kw.stride(1) == 1 and k_ln.stride(1) == 1 and rows <= cfg.block_m
        )
        ih, ihd = indexer.n_heads, indexer.head_dim
        q = kva.new_empty(rows, _LOCAL_HEADS * _HEAD_DIM)
        mode = NORM_MODES[self.front_config.norm]
        keep_qn = cfg.debug_keep_intermediates
        if mode == 0:
            assert qn is not None and qn.shape == (rows, _Q_LORA) and qn.is_contiguous()
        elif mode == 1:
            # Formed in registers; stored only for tests.
            qn = kva.new_empty(rows, _Q_LORA) if keep_qn else kva
        else:
            qn = kva.new_empty(rows, _Q_LORA)
        latent = kva.new_empty(rows, _KV_LORA + _ROPE)
        fused_q = kva.new_empty(rows, _LOCAL_HEADS, _KV_LORA + _ROPE) if absorb else None
        iq = kva.new_empty(rows, ih * ihd)
        q_fp8 = torch.empty(rows, ih, ihd, dtype=torch.float8_e4m3fn, device=kva.device)
        q_scale = torch.empty(rows, ih, 1, dtype=torch.float32, device=kva.device)
        k_fp8 = torch.empty(rows, ihd, dtype=torch.float8_e4m3fn, device=kva.device)
        k_scale = torch.empty(rows, 1, dtype=torch.float32, device=kva.device)
        weights = torch.empty(rows, ih, dtype=torch.float32, device=kva.device)
        pos = position_ids.reshape(-1)
        assert pos.numel() >= rows
        n_qb = (_LOCAL_HEADS * _HEAD_DIM) // cfg.bn
        n_iq = 0 if skip_iq else (ih * ihd) // cfg.bn
        n_qpost = 0 if skip_iq else ih
        in_producer = absorb and cfg.absorb_in_producer
        layout = absorb_layout(n_qb, absorb, in_producer, cfg.absorb_prev_head, cfg.bc)
        n_norm = kernels.num_norm_programs(rows, self.front_config.norm_rows) if mode == 2 else 0
        grid = (kernels.grid_size(n_norm, n_qb, n_iq, layout["N_ABS"], n_qpost),)
        trigger = self.front_config.early_trigger if early_trigger is None else early_trigger
        kernels.front_qproj_indexer_kernel[grid](
            kva,
            kva.stride(0),
            mla.q_a_layernorm.weight,
            mla.kv_a_layernorm.weight,
            mla.q_b_proj.weight,
            mla.k_b_proj_trans,
            q,
            latent,
            fused_q if absorb else q,
            fused_q.stride(0) if absorb else 0,
            self.indexer_front_sync,
            qn,
            n_norm,
            rows,
            mla.q_a_layernorm.variance_epsilon,
            mla.kv_a_layernorm.variance_epsilon,
            indexer.wq_b.weight,
            iq,
            kw,
            kw.stride(0),
            k_ln,
            k_ln.stride(0),
            pos,
            indexer.rotary_emb.rotary_cos_sin,
            q_fp8,
            q_scale,
            k_fp8,
            k_scale,
            weights,
            indexer.weight_scale_factor,
            Q_LORA=_Q_LORA,
            KV_LORA=_KV_LORA,
            ROPE=_ROPE,
            HEAD_DIM=_HEAD_DIM,
            NOPE=_NOPE,
            H=_LOCAL_HEADS,
            IHD=ihd,
            IH=ih,
            ROPE_I=indexer.rope_dim,
            BLOCK_M=cfg.block_m,
            BN=cfg.bn,
            BK=cfg.bk,
            N_QB=n_qb,
            N_IQ=n_iq,
            N_QPOST=n_qpost,
            DO_ABSORB=absorb,
            NORM_ROWS=self.front_config.norm_rows,
            USE_PDL=self.use_pdl,
            EARLY_TRIGGER=trigger,
            ABSORB_IN_PRODUCER=in_producer,
            NORM_MODE=mode,
            PREFETCH_W=self.front_config.prefetch_weights,
            WRITE_QN=mode == 1 and keep_qn,
            DELAY_NS=self.front_config.debug_delay_ns,
            **layout,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            launch_pdl=self.use_pdl,
        )
        if cfg.debug_keep_intermediates:
            self.debug["iq"] = iq
            self.debug["qn"] = qn
        outs = [
            q,
            latent[:, :_KV_LORA],
            latent[:, _KV_LORA:],
            latent,
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            q_scale,
        ]
        return AbsorbedQueryPacket(outs, fused_q)

    def project_front(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        absorb: bool,
        skip_iq: bool = False,
    ) -> AbsorbedQueryPacket:
        kva, qn, kw, k_ln = self.bridges(hidden_states)
        return self.launch_front(kva, kw, k_ln, position_ids, absorb, qn=qn, skip_iq=skip_iq)


def _runner(mla):
    """Per-module ``IndexerFrontQueryProjection`` (cached in ``mla._indexer_front_projection``) or False."""
    runner = getattr(mla, "_indexer_front_projection", None)
    if runner is not None:
        return runner
    check_package_selectors()
    ok, reason, draft = front_module_eligibility(mla)
    runner = False
    if ok:
        proj = IndexerFrontQueryProjection(mla)
        why = proj.supported()
        if why is not None:
            ok, reason = False, why
        else:
            runner = proj
    log_capability(FEATURE_ID, f"{'draft' if draft else 'target'}_layer{mla.layer_idx}", ok, reason)
    mla._indexer_front_projection = runner
    return runner


def dsa_proj(
    mla, position_ids: Optional[torch.Tensor], hidden_states: torch.Tensor, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """indexer_layer_front: the query_front packet of a full-indexer target layer (absorbed_query)
    or the MTP draft layer (draft_absorbed_query), or ``None`` (before any GPU work) to run the
    next path."""
    if not is_enabled(FEATURE_ID):
        return None
    runner = _runner(mla)
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, hidden_states, attn_metadata)
    if not ok or position_ids is None:
        return None
    skip_iq = runner.skip_reused_iq and reuses_shared_topk(mla, attn_metadata)
    if runner.skip_reused_iq:
        SKIP_DECISIONS["skip" if skip_iq else "full"] += 1
    return runner.project_front(position_ids, hidden_states, absorb_for(mla), skip_iq=skip_iq)
