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
"""absorbed_query: target query/projection region of GLM-5.x DSA MLA (the query_front contract).

``dsa_proj`` is the package entry point. It replaces the token-wise work of
the native ``forward_dsa_proj`` for one generation-only target layer and
returns a :class:`AbsorbedQueryPacket`. ``mtp_dsa_proj`` (``draft_absorbed_query``) does
the same for the one-model MTP draft layer at every draft step. The packet
holds:

* ``proj_outputs``: the native 4/9-tensor list ``[q, compressed_kv, k_pe,
  latent_cache(, q_fp8, k_fp8, k_scale, weights, q_scale)]`` with native
  shapes, dtypes and values (``compressed_kv``/``k_pe`` are views of
  ``latent_cache``);
* ``fused_q``: in the preabsorbed form, the ``[R, H, kv_lora + rope]`` buffer
  whose ``[..., :kv_lora]`` already holds ``q_nope @ W_UK`` (BF16). The
  caller hands it to the native ``forward_absorption_generation`` in place of
  its own allocation and skips only the absorption BMM; native RoPE, cache
  append, FP8 query quantization and scheduler buffers still write the rest.
  ``None`` in the native-q form.

The kv_a GEMM and, on full indexer layers, ``Indexer.pre_indexer_proj`` run
natively around the absorbed_query kernel (adapter bridges). ``dsa_proj`` returns ``None``
before any GPU work when the call is outside the support domain, so the
caller runs the unchanged native path.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ...dispatch import log_capability
from . import kernels

#: How the form's capability line names it (``<label>=<form>``).
FORM_LABEL = "TRTLLM_STAIRCASE_GLM_LL_ABSORBED_QUERY_FORM"
# "auto": preabsorbed on full indexer layers, native-q on shared layers (the
# default form for each layer type); the other two force one form. Both
# forms are bit-identical to native; the form is logged once per process.
FORMS = ("auto", "preabsorbed", "native_q")
#: The form this target runs.
SERVED_FORM = FORMS[0]

# Support domain (GLM-5.2 DSA MLA, TP8, non-ADP).
_Q_LORA = 2048
_KV_LORA = 512
_ROPE = 64
_NOPE = 192
_HEAD_DIM = _NOPE + _ROPE
_LOCAL_HEADS = 8
_MAX_ROWS = 16


def selected_form() -> str:
    """The form this target run, ``SERVED_FORM``."""
    return SERVED_FORM


def check_package_selectors() -> str:
    """Log (once per process) this package's form and return it.

    Called by every entry point before its per-module decision, so it runs
    before any GPU work.
    """
    form = selected_form()
    log_capability("absorbed_query", "form", True, f"{FORM_LABEL}={form}")
    return form


def absorb_for(mla, form: Optional[str] = None) -> bool:
    """Whether the call returns the preabsorbed form for this module."""
    form = form or selected_form()
    if form == "auto":
        return mla.mqa.indexer is not None
    return form == "preabsorbed"


@dataclass
class AbsorbedQueryPacket:
    """Explicit query_front packet of one absorbed_query invocation."""

    proj_outputs: List[torch.Tensor]
    fused_q: Optional[torch.Tensor]


@dataclass(frozen=True)
class KernelConfig:
    # True: kernels.qproj_prefetch_kernel (whole weight tiles requested
    # before the grid-dependency wait); False: the cp.async-pipelined
    # kernels.qproj_kernel / qproj_indexer_kernel.
    prefetch: bool = False
    # Normalize q_a once with the native q_a_layernorm (FlashInfer RMSNorm)
    # and feed its BF16 output to the q_b / indexer wq_b MMA loops, instead of
    # re-normalizing the padded q_a block inside every producer program.
    prenorm: bool = True
    # Preabsorbed form: q_b programs perform absorption jobs after
    # publishing their own tile (instead of separate spinning programs).
    absorb_in_producer: bool = True
    # With absorb_in_producer: the producers of head h + 1 absorb head h and
    # HEAD_DIM // BN extra programs after all producers absorb the last head,
    # so every spin-wait targets lower program ids (forward progress with one
    # resident program slot; see kernels.py). False is the legacy layout
    # (producer p absorbs job p of its own head group), which needs
    # NOPE // BN co-resident programs; kept only as a stress-test reference.
    absorb_prev_head: bool = True
    # Tests only: skew producers/consumers by up to this many nanoseconds
    # (kernels._stress_delay). 0 compiles the delay out.
    debug_delay_ns: int = 0
    # Tests only: keep the fused indexer kernel's BF16 wq_b output
    # (``self.debug["iq"]``) for comparison with the native GEMM.
    debug_keep_intermediates: bool = False
    prefetch_bn: int = 16
    prefetch_warps: int = 8
    # Pipelined query-projection kernel; tuned launch configuration: BN=32,
    # 4 stages.
    block_m: int = 16
    bn: int = 32
    bk: int = 256
    bc: int = 64
    num_warps: int = 4
    num_stages: int = 4
    # Hidden-state GEMV (kernels.hidden_gemv_kernel) replacing the native
    # kv_a GEMM and, on full indexer layers, the indexer wk/weights_proj GEMM.
    # Experimental, off: it would replace the native kv_a GEMM, and its
    # results are outside the frozen error budget.
    own_hidden_gemv: bool = False
    # Full indexer layers: compute the indexer query/key/weights tail in the
    # query kernel from the native fused wk/weights_proj output and the
    # native k_norm output; otherwise the native Indexer.pre_indexer_proj
    # runs on the absorbed_query qr.
    fuse_indexer: bool = True
    # Full indexer layers, multi-stream (CUDA-graph capture): run the native
    # fused wk/weights_proj GEMM and k_norm (bridges) on the indexer's aux
    # stream, concurrently with the native kv_a GEMM and q_a_layernorm, and
    # join before the absorbed_query kernel. False (or no multi-stream): in stream order.
    indexer_k_on_aux: bool = True
    gemv_programs: int = 0  # 0: one program per SM
    gemv_bk: int = 256
    gemv_warps: int = 4
    gemv_stages: int = 6


# The DSA hooks of a DSA MLA module: the built-in class and the one this target adopts it as.
_DSA_HOOKS = ("DSAHooks", "GlmDsaHooks")


def module_eligibility(mla, draft: bool = False) -> Tuple[bool, str]:
    """Static (per-module) support check; no GPU work.

    ``draft=False`` (absorbed_query): target layers, ``layer_idx < num_hidden_layers``.
    ``draft=True`` (draft_absorbed_query): the one-model MTP draft layer,
    ``layer_idx >= num_hidden_layers``. Every other check is shared.
    """
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        return False, f"sm{get_sm_version()} not validated"
    if (
        getattr(mla, "sparse_attn_hooks", None) is None
        or type(mla.sparse_attn_hooks).__name__ not in _DSA_HOOKS
    ):
        return False, "not a DSA MLA module"
    if getattr(mla, "is_lite", True) or not getattr(mla, "fuse_qkv_a_proj", False):
        return False, "needs the fused [q_a | kv_a] projection"
    num_layers = getattr(mla, "num_hidden_layers", None)
    if draft:
        if num_layers is None or mla.layer_idx is None or mla.layer_idx < num_layers:
            return False, "draft (MTP) layers only"
    elif num_layers is None or mla.layer_idx is None or mla.layer_idx >= num_layers:
        return False, "target layers only"
    if mla.mapping.enable_attention_dp or mla.mapping.cp_size != 1:
        return False, "attention DP / CP not supported"
    if (
        mla.q_lora_rank,
        mla.kv_lora_rank,
        mla.qk_rope_head_dim,
        mla.qk_nope_head_dim,
        mla.num_heads_tp,
    ) != (_Q_LORA, _KV_LORA, _ROPE, _NOPE, _LOCAL_HEADS):
        return False, "shape outside the GLM-5.2 TP8 domain"
    if mla.apply_rotary_emb or mla.kv_cache_dtype == "fp8_ds_mla":
        return False, "needs the fused-RoPE TRTLLM backend cache path"
    if getattr(mla, "llama_4_scaling", False):
        return False, "llama-4 query scaling not supported"
    for name in ("kv_a_proj_with_mqa", "q_b_proj"):
        lin = getattr(mla, name)
        if lin.has_any_quant or lin.weight.dtype != torch.bfloat16 or (lin.bias is not None):
            return False, f"{name} must be unquantized BF16 without bias"
    if mla.k_b_proj_trans.dtype != torch.bfloat16 or tuple(mla.k_b_proj_trans.shape) != (
        _LOCAL_HEADS,
        _KV_LORA,
        _NOPE,
    ):
        return False, "k_b_proj_trans must be BF16 [8, 512, 192]"
    if mla.q_a_layernorm.use_gemma or mla.kv_a_layernorm.use_gemma:
        return False, "Gemma-style RMSNorm not supported"
    if getattr(mla.q_a_layernorm, "is_nvfp4", False):
        return False, "q_a_layernorm fused with NVFP4 quantization"
    return True, ""


def call_eligibility(mla, hidden_states, attn_metadata) -> Tuple[bool, str]:
    """Per-call support check; no GPU work."""
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
    from tensorrt_llm._torch.utils import is_torch_compiling

    if is_torch_compiling():
        return False, "torch.compile graph (piecewise custom ops) stays native"
    if is_in_breakable_cuda_graph():
        return False, "breakable CUDA graph keeps the native custom-op break"
    if not isinstance(hidden_states, torch.Tensor) or type(hidden_states) is not torch.Tensor:
        return False, "quantized hidden-state input"
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        return False, "hidden states must be 2-D BF16"
    rows = hidden_states.shape[0]
    if not 1 <= rows <= _MAX_ROWS:
        return False, f"{rows} rows outside [1, {_MAX_ROWS}]"
    if (
        attn_metadata is None
        or attn_metadata.num_contexts != 0
        or (attn_metadata.num_generations == 0)
    ):
        return False, "generation-only batches only"
    return True, ""


class AbsorbedQueryProjection:
    """query_front producer bound to one native DSA ``MLA`` module."""

    def __init__(self, mla, config: KernelConfig = KernelConfig()):
        self.mla = mla
        self.config = config
        device = mla.q_b_proj.weight.device
        # Per-layer ready/done counters (see kernels.py): 2 * H absorption
        # counters, then one per indexer head; always zero between launches.
        indexer = mla.mqa.indexer
        n_index_heads = indexer.n_heads if indexer is not None else 0
        self.sync = torch.zeros(2 * _LOCAL_HEADS + n_index_heads, dtype=torch.int32, device=device)
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.use_pdl = get_env_enable_pdl()
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # Fork/join events of the indexer-key bridges (``native_bridges``).
        self.events = [torch.cuda.Event(), torch.cuda.Event()]
        self.debug = {}

    def launch(
        self,
        kva: torch.Tensor,
        need_qr: bool,
        absorb: bool,
        ss: Optional[torch.Tensor] = None,
        n_parts: int = 0,
    ):
        """Run the query kernel on a kv_a output ``[R, 2624]``.

        ``ss``/``n_parts`` are the hidden GEMV's RMSNorm partial sums; without
        them the kernel recomputes the row norms from ``kva``.
        """
        mla = self.mla
        cfg = self.config
        rows = kva.shape[0]
        assert kva.stride(1) == 1 and rows <= cfg.block_m
        q = kva.new_empty(rows, _LOCAL_HEADS * _HEAD_DIM)
        prenorm = cfg.prenorm and ss is None
        qn = self.native_q_norm(kva) if prenorm else None
        if prenorm:
            qr = qn
            need_qr = False
        else:
            qr = kva.new_empty(rows, _Q_LORA) if need_qr else q
        latent = kva.new_empty(rows, _KV_LORA + _ROPE)
        fused_q = kva.new_empty(rows, _LOCAL_HEADS, _KV_LORA + _ROPE) if absorb else None
        n_qb = (_LOCAL_HEADS * _HEAD_DIM) // cfg.bn
        in_producer = absorb and cfg.absorb_in_producer
        layout = absorb_layout(n_qb, absorb, in_producer, cfg.absorb_prev_head, cfg.bc)
        grid = (n_qb + layout["N_ABS"] + 1,)
        kernels.qproj_kernel[grid](
            kva,
            kva.stride(0),
            mla.q_a_layernorm.weight,
            mla.kv_a_layernorm.weight,
            mla.q_b_proj.weight,
            mla.k_b_proj_trans,
            q,
            qr,
            latent,
            fused_q if absorb else q,
            fused_q.stride(0) if absorb else 0,
            self.sync,
            ss if ss is not None else self.sync,
            n_parts,
            rows,
            mla.q_a_layernorm.variance_epsilon,
            mla.kv_a_layernorm.variance_epsilon,
            Q_LORA=_Q_LORA,
            KV_LORA=_KV_LORA,
            ROPE=_ROPE,
            HEAD_DIM=_HEAD_DIM,
            NOPE=_NOPE,
            H=_LOCAL_HEADS,
            BLOCK_M=cfg.block_m,
            BN=cfg.bn,
            BK=cfg.bk,
            N_QB=n_qb,
            WRITE_QR=need_qr,
            DO_ABSORB=absorb,
            SS_PARTIALS=ss is not None,
            MAX_PARTS=triton_next_pow2(max(n_parts, 1)),
            USE_PDL=self.use_pdl,
            PRENORM=prenorm,
            qn_ptr=qn if prenorm else q,
            stride_qn=qn.stride(0) if prenorm else 0,
            ABSORB_IN_PRODUCER=in_producer,
            DELAY_NS=cfg.debug_delay_ns,
            **layout,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            launch_pdl=self.use_pdl,
        )
        return q, (qr if (need_qr or prenorm) else None), latent, fused_q

    def native_q_norm(self, kva: torch.Tensor) -> torch.Tensor:
        """The native ``q_a_layernorm`` call on the q_a slice (bridge)."""
        return self.mla._q_a_layernorm_maybe_fused(kva[:, :_Q_LORA])

    def fuse_indexer_kw(self) -> bool:
        """Whether the hidden GEMV also produces the indexer wk/weights_proj
        output (BF16 indexer projection, FP8 indexer keys only)."""
        indexer = self.mla.mqa.indexer
        return (
            indexer is not None
            and indexer._indexer_bf16
            and not indexer.use_fp4
            and indexer._fused_wk_wp_weight is not None
            and indexer._fused_wk_wp_weight.dtype == torch.bfloat16
        )

    def hidden_gemv(self, hidden_states: torch.Tensor, fuse_indexer_kw: bool):
        """``hidden @ [W_kv_a; W_indexer_kw]^T`` in one balanced launch.

        Returns ``(kva, kw, ss, n_parts)``: the BF16 kv_a output, the BF16
        indexer wk/weights_proj output (or None) and the per-program RMSNorm
        partial sums ``[n_parts, 2, block_m]`` of the q_a / compressed_kv
        column groups.
        """
        mla = self.mla
        cfg = self.config
        rows = hidden_states.shape[0]
        w0 = mla.kv_a_proj_with_mqa.weight
        n0 = w0.shape[0]
        kva = hidden_states.new_empty(rows, n0)
        programs = cfg.gemv_programs or self.num_sms
        if fuse_indexer_kw:
            w1 = mla.mqa.indexer._fused_wk_wp_weight
            n1 = w1.shape[0]
            kw = hidden_states.new_empty(rows, n1)
            p1 = max(1, round(programs * n1 / (n0 + n1)))
            cpp1 = triton_cdiv(n1, p1)
            p1 = triton_cdiv(n1, cpp1)
        else:
            w1, kw, n1, p1, cpp1 = w0, kva, 0, 0, 1
        cpp0 = triton_cdiv(n0, programs - p1)
        p0 = triton_cdiv(n0, cpp0)
        ss = torch.empty(p0, 2, cfg.block_m, dtype=torch.float32, device=hidden_states.device)
        kernels.hidden_gemv_kernel[(p0 + p1,)](
            hidden_states,
            hidden_states.stride(0),
            w0,
            w1,
            kva,
            kva.stride(0),
            kw,
            kw.stride(0),
            ss,
            rows,
            n0,
            n1,
            P0=p0,
            CPP0=cpp0,
            CPP1=cpp1,
            SS_SPLIT=_Q_LORA,
            SS_END=_Q_LORA + _KV_LORA,
            WRITE_SS=True,
            K=hidden_states.shape[1],
            BLOCK_M=cfg.block_m,
            BN=triton_next_pow2(max(cpp0, cpp1)),
            BK=cfg.gemv_bk,
            USE_PDL=self.use_pdl,
            num_warps=cfg.gemv_warps,
            num_stages=cfg.gemv_stages,
            launch_pdl=self.use_pdl,
        )
        return kva, (kw if fuse_indexer_kw else None), ss, p0

    def indexer_fusable(self) -> bool:
        """Whether the query kernel can produce the indexer tail itself."""
        indexer = self.mla.mqa.indexer
        if not (self.config.fuse_indexer and self.fuse_indexer_kw()):
            return False
        rope = indexer.rotary_emb
        return (
            indexer.scale_fmt == "ue8m0"
            and indexer.head_dim == 128
            and indexer.rope_dim == 64
            and indexer.n_heads <= 64
            and not rope.is_neox
            and not rope.inverse
            and rope.head_dim == indexer.rope_dim
            and rope.rotary_cos_sin.dtype == torch.float32
            and rope.rotary_cos_sin.is_contiguous()
            and tuple(rope.rotary_cos_sin.shape[1:]) == (2, 32)
            and indexer.wq_b.weight.dtype == torch.bfloat16
            and not indexer.wq_b.has_any_quant
        )

    def launch_indexer(
        self,
        kva: torch.Tensor,
        kw: torch.Tensor,
        k_ln: torch.Tensor,
        ss: Optional[torch.Tensor],
        n_parts: int,
        position_ids: torch.Tensor,
        absorb: bool,
        qn: Optional[torch.Tensor] = None,
    ):
        """Query kernel with the fused indexer tail (full indexer layers).

        ``kw`` is the native fused wk/weights_proj output, ``k_ln`` the
        native ``k_norm`` of its key columns and ``qn`` (optional) the native
        ``q_a_layernorm`` output (computed here when absent).
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
        prenorm = cfg.prenorm and ss is None
        if prenorm and qn is None:
            qn = self.native_q_norm(kva)
        elif not prenorm:
            qn = q
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
        n_iq = (ih * ihd) // cfg.bn
        in_producer = absorb and cfg.absorb_in_producer
        layout = absorb_layout(n_qb, absorb, in_producer, cfg.absorb_prev_head, cfg.bc)
        grid = (n_qb + n_iq + layout["N_ABS"] + ih + 2,)
        kernels.qproj_indexer_kernel[grid](
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
            self.sync,
            ss if ss is not None else self.sync,
            n_parts,
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
            DO_ABSORB=absorb,
            SS_PARTIALS=ss is not None,
            MAX_PARTS=triton_next_pow2(max(n_parts, 1)),
            USE_PDL=self.use_pdl,
            PRENORM=prenorm,
            qn_ptr=qn,
            stride_qn=qn.stride(0) if prenorm else 0,
            ABSORB_IN_PRODUCER=in_producer,
            DELAY_NS=cfg.debug_delay_ns,
            **layout,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            launch_pdl=self.use_pdl,
        )
        if cfg.debug_keep_intermediates:
            self.debug["iq"] = iq
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

    def launch_prefetch(
        self,
        kva: torch.Tensor,
        kw: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        absorb: bool,
        k_ln: Optional[torch.Tensor] = None,
    ) -> AbsorbedQueryPacket:
        """Prefetching query kernel (both layer types; experimental).

        ``kw`` is the native fused indexer wk/weights_proj output on full
        indexer layers whose tail is fused (``k_ln``: its native ``k_norm``,
        computed here when absent); with ``kw`` None on a full layer the
        native indexer tail runs on the returned ``qr``.
        """
        mla = self.mla
        indexer = mla.mqa.indexer
        cfg = self.config
        rows = kva.shape[0]
        assert kva.stride(1) == 1 and rows <= cfg.block_m
        bn = cfg.prefetch_bn
        n_qb = (_LOCAL_HEADS * _HEAD_DIM) // bn
        # The prefetching kernel always absorbs inside the producers.
        layout = absorb_layout(
            n_qb, absorb, True, cfg.absorb_prev_head, (_LOCAL_HEADS * _KV_LORA) // n_qb
        )
        fuse = kw is not None
        need_qr = indexer is not None and not fuse
        q = kva.new_empty(rows, _LOCAL_HEADS * _HEAD_DIM)
        qr = kva.new_empty(rows, _Q_LORA) if need_qr else q
        latent = kva.new_empty(rows, _KV_LORA + _ROPE)
        fused_q = kva.new_empty(rows, _LOCAL_HEADS, _KV_LORA + _ROPE) if absorb else None
        if fuse:
            ih, ihd = indexer.n_heads, indexer.head_dim
            iq = kva.new_empty(rows, ih * ihd)
            q_fp8 = torch.empty(rows, ih, ihd, dtype=torch.float8_e4m3fn, device=kva.device)
            q_scale = torch.empty(rows, ih, 1, dtype=torch.float32, device=kva.device)
            k_fp8 = torch.empty(rows, ihd, dtype=torch.float8_e4m3fn, device=kva.device)
            k_scale = torch.empty(rows, 1, dtype=torch.float32, device=kva.device)
            weights = torch.empty(rows, ih, dtype=torch.float32, device=kva.device)
            pos = position_ids.reshape(-1)
            assert pos.numel() >= rows
            n_iq = (ih * ihd) // bn
            if k_ln is None:
                k_ln = indexer.k_norm(kw[:, :ihd])
            iargs = (
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
            )
            iconst = dict(IHD=ihd, IH=ih, ROPE_I=indexer.rope_dim)
        else:
            ih, n_iq = 0, 0
            dummy = q
            iargs = (
                dummy,
                dummy,
                dummy,
                0,
                dummy,
                0,
                self.sync,
                self.sync,
                dummy,
                self.sync,
                dummy,
                self.sync,
                self.sync,
                0.0,
            )
            iconst = dict(IHD=128, IH=0, ROPE_I=64)
        grid = (n_qb + n_iq + layout["N_ABS"] + ih + 1 + (1 if fuse else 0),)
        kernels.qproj_prefetch_kernel[grid](
            kva,
            kva.stride(0),
            mla.q_a_layernorm.weight,
            mla.kv_a_layernorm.weight,
            mla.q_b_proj.weight,
            mla.k_b_proj_trans,
            q,
            qr,
            latent,
            fused_q if absorb else q,
            fused_q.stride(0) if absorb else 0,
            self.sync,
            rows,
            mla.q_a_layernorm.variance_epsilon,
            mla.kv_a_layernorm.variance_epsilon,
            *iargs,
            Q_LORA=_Q_LORA,
            KV_LORA=_KV_LORA,
            ROPE=_ROPE,
            HEAD_DIM=_HEAD_DIM,
            NOPE=_NOPE,
            H=_LOCAL_HEADS,
            BLOCK_M=cfg.block_m,
            BN=bn,
            N_QB=n_qb,
            N_IQ=n_iq,
            WRITE_QR=need_qr,
            DO_ABSORB=absorb,
            FUSE_INDEXER=fuse,
            USE_PDL=self.use_pdl,
            DELAY_NS=cfg.debug_delay_ns,
            **layout,
            num_warps=cfg.prefetch_warps,
            num_stages=1,
            launch_pdl=self.use_pdl,
            **iconst,
        )
        outs = [q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent]
        if fuse:
            outs += [q_fp8, k_fp8, k_scale, weights, q_scale]
        elif indexer is not None:
            outs.extend(indexer.pre_indexer_proj(qr, hidden_states, position_ids))
        return AbsorbedQueryPacket(outs, fused_q)

    def native_bridges(self, hidden_states: torch.Tensor):
        """The native token-wise calls the fused full-layer kernel consumes.

        Returns ``(kva, qn, kw, k_ln)``: the native kv_a GEMM output, the
        native ``q_a_layernorm`` of its q_a columns (None without
        ``prenorm``), the native BF16 fused wk/weights_proj GEMM output (the
        call ``Indexer.pre_indexer_proj`` makes with
        ``TRTLLM_DSA_INDEXER_BF16``) and the native ``k_norm`` of its key
        columns. The last two depend only on ``hidden_states``; with
        ``indexer_k_on_aux`` and multi-stream enabled (CUDA-graph capture)
        they run on the indexer's aux stream concurrently with the first two
        and are joined before returning, as the native region overlaps its
        own aux-stream work.
        """
        mla = self.mla
        indexer = mla.mqa.indexer
        prenorm = self.config.prenorm

        def main():
            kva = mla.kv_a_proj_with_mqa(hidden_states)
            return kva, (self.native_q_norm(kva) if prenorm else None)

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

    def project(
        self, position_ids: Optional[torch.Tensor], hidden_states: torch.Tensor, absorb: bool
    ) -> AbsorbedQueryPacket:
        """Native bridges (or the experimental hidden GEMV), absorbed_query kernel,
        indexer tail."""
        mla = self.mla
        indexer = mla.mqa.indexer
        fusable = indexer is not None and self.indexer_fusable()
        kw, ss, n_parts = None, None, 0
        if self.config.own_hidden_gemv:
            kva, kw, ss, n_parts = self.hidden_gemv(hidden_states, self.fuse_indexer_kw())
        elif fusable:
            kva, qn, kw, k_ln = self.native_bridges(hidden_states)
            if self.config.prefetch:
                return self.launch_prefetch(kva, kw, hidden_states, position_ids, absorb, k_ln)
            return self.launch_indexer(kva, kw, k_ln, None, 0, position_ids, absorb, qn)
        else:
            kva = mla.kv_a_proj_with_mqa(hidden_states)
        if self.config.prefetch and not self.config.own_hidden_gemv:
            return self.launch_prefetch(kva, None, hidden_states, position_ids, absorb)
        if fusable:
            # Experimental own hidden GEMV: its kw, the native k_norm.
            return self.launch_indexer(
                kva,
                kw,
                indexer.k_norm(kw[:, : indexer.head_dim]),
                ss,
                n_parts,
                position_ids,
                absorb,
            )
        q, qr, latent, fused_q = self.launch(
            kva, need_qr=indexer is not None, absorb=absorb, ss=ss, n_parts=n_parts
        )
        outs = [q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent]
        if indexer is not None:
            if kw is not None:
                outs.extend(indexer_tail_from_projected(indexer, qr, kw, position_ids))
            else:
                outs.extend(indexer.pre_indexer_proj(qr, hidden_states, position_ids))
        return AbsorbedQueryPacket(outs, fused_q)

    def forward_preabsorbed(self, position_ids, hidden_states):
        return self.project(position_ids, hidden_states, absorb=True)


def absorb_layout(
    n_qb: int, absorb: bool, in_producer: bool, prev_head: bool, bc_separate: int
) -> dict:
    """Absorption program layout of one query-kernel launch.

    Returns the kernel's ``BC``, ``N_ABS`` (separate absorption programs),
    ``ABS_JOB0`` (first job of those programs) and ``PRODUCER_JOB_SHIFT``
    (producer ``p`` runs job ``p - shift`` when ``ABSORB_IN_PRODUCER``).
    """
    tiles_per_head = n_qb // _LOCAL_HEADS
    if not absorb:
        return dict(BC=bc_separate, N_ABS=0, ABS_JOB0=0, PRODUCER_JOB_SHIFT=0)
    if not in_producer:
        return dict(
            BC=bc_separate,
            N_ABS=_LOCAL_HEADS * (_KV_LORA // bc_separate),
            ABS_JOB0=0,
            PRODUCER_JOB_SHIFT=0,
        )
    # One job per producer: BC latent columns, KV_LORA // BC == tiles/head.
    bc = (_LOCAL_HEADS * _KV_LORA) // n_qb
    assert _KV_LORA // bc == tiles_per_head
    if not prev_head:
        return dict(BC=bc, N_ABS=0, ABS_JOB0=0, PRODUCER_JOB_SHIFT=0)
    return dict(
        BC=bc,
        N_ABS=tiles_per_head,
        ABS_JOB0=n_qb - tiles_per_head,
        PRODUCER_JOB_SHIFT=tiles_per_head,
    )


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def triton_next_pow2(n: int) -> int:
    return 1 << max(4, (n - 1).bit_length())


def indexer_tail_from_projected(
    indexer, qr: torch.Tensor, fused_kw: torch.Tensor, position_ids: Optional[torch.Tensor]
):
    """Native ``Indexer.pre_indexer_proj`` after its fused wk/weights_proj
    GEMM, fed with that GEMM's output ``fused_kw [R, head_dim + n_heads]``.

    Calls the native indexer methods in the native order (FP8 keys, BF16
    projection), so only the source of ``fused_kw`` differs.
    """
    from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

    indexer_k, weights = fused_kw.split([indexer.head_dim, indexer.n_heads], dim=-1)
    q_pe, q_nope, k_pe, k_nope = indexer._qk_projection_and_rope(qr, indexer_k, position_ids)
    q, k = maybe_execute_in_parallel(
        lambda: indexer._prep_q_or_k(q_pe, q_nope),
        lambda: indexer._prep_q_or_k(k_pe, k_nope),
        indexer.ln_events[0],
        indexer.ln_events[1],
        indexer.aux_stream,
    )
    q_fp8, q_scale = q
    k_fp8, k_scale = k
    q_fp8 = q_fp8.view(-1, indexer.n_heads, indexer.head_dim)
    q_scale = q_scale.view(-1, indexer.n_heads, 1)
    weights = indexer._weight_scale(weights, q_scale)
    return q_fp8, k_fp8, k_scale, weights, q_scale


def dsa_proj(
    mla, position_ids: Optional[torch.Tensor], hidden_states: torch.Tensor, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """absorbed_query replacement of the token-wise ``forward_dsa_proj`` work.

    Returns ``None`` (before any GPU work) when the module or the call is
    outside the support domain; the caller then runs the native path.
    """
    runner = getattr(mla, "_absorbed_query", None)
    if runner is None:
        check_package_selectors()
        ok, reason = module_eligibility(mla)
        log_capability("absorbed_query", f"layer{mla.layer_idx}", ok, reason)
        runner = AbsorbedQueryProjection(mla) if ok else False
        mla._absorbed_query = runner
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, hidden_states, attn_metadata)
    if not ok:
        return None
    return runner.project(position_ids, hidden_states, absorb=absorb_for(mla))


def mtp_dsa_proj(
    mla, position_ids: Optional[torch.Tensor], hidden_states: torch.Tensor, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """draft_absorbed_query: the absorbed_query query_front producer on the one-model MTP draft layer.

    The MTP layer (``layer_idx >= num_hidden_layers``, GLM-5.2: layer 78)
    is called once per draft step of the MTP Eagle draft loop (``R = 6``
    rows at step 0, one row per request at steps 1-4 for one request). Its
    attention weights have the target-layer shapes and dtypes and it always
    owns a full indexer, so the unchanged absorbed_query kernel path serves every draft
    step. The native RoPE, cache appends (latent and indexer key), FP8 query
    quantization, scheduler buffers and the indexer's draft-loop top-k reuse
    (``mtp_index_share``) all stay in the native code after this call. Every
    output of the native 9-tensor list is written at every draft step, also
    on steps whose indexer reuses the step-0 top-k.

    Returns ``None`` (before any GPU work) for target layers and outside the
    absorbed_query support domain; the caller then runs the native path. The per-module
    decision is cached in ``mla._draft_absorbed_query`` and logged once under
    ``draft_absorbed_query``.
    """
    runner = getattr(mla, "_draft_absorbed_query", None)
    if runner is None:
        check_package_selectors()
        ok, reason = module_eligibility(mla, draft=True)
        log_capability("draft_absorbed_query", f"layer{mla.layer_idx}", ok, reason)
        runner = AbsorbedQueryProjection(mla) if ok else False
        mla._draft_absorbed_query = runner
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, hidden_states, attn_metadata)
    if not ok:
        return None
    return runner.project(position_ids, hidden_states, absorb=absorb_for(mla))
