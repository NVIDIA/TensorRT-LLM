# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stock stages of one M3 sparse/MoE decoder layer (vLLM fe1d68261 op sequence) and the swap registry.

Every stage is built by a factory ``make(blk) -> fn`` and ``fn()`` takes no arguments: it reads its inputs from and
writes its outputs to attributes of the per-layer state ``blk`` (see HARNESS.md for the buffer contract). A swap
replaces a factory; the name ``none`` removes a stage (for a fused op that covers several stages).
"""

from __future__ import annotations

import importlib

import torch

from .model_geometry import EPS, ROUTE_SCALE, SWIGLU_ALPHA, SWIGLU_BETA, SWIGLU_LIMIT, TOP_K, H

ATTN_STAGES = ("boundary_in", "qkv", "producer", "scorer", "topk", "decode", "oproj")
MOE_STAGES = ("boundary_post", "router", "moe_quant", "moe_experts", "shared", "combine")
ALL_STAGES = ATTN_STAGES + MOE_STAGES


# ----------------------------------------------------------------------------- helpers
def mx_quant(x):
    from flashinfer import mxfp8_quantize

    return mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl")


def mx_mm(xq, xs, lin):
    from flashinfer import mm_mxfp8

    # vLLM selects FlashInferCutedslMxfp8LinearKernel on sm_107 (serve logs: "Using FlashInferCutedslMxfp8LinearKernel")
    return mm_mxfp8(
        xq, lin.w.t(), xs, lin.s, out=None, out_dtype=torch.bfloat16, backend="cute-dsl"
    )


# ----------------------------------------------------------------------------- attention block
def make_boundary_in(blk):
    def fn():
        blk.ar.ar_norm(
            blk.ffn_partial,
            blk.residual_in,
            blk.aw.in_norm,
            blk.normed,
            residual_out=blk.residual_mid,
        )

    return fn


def make_qkv(blk):
    c, lin = blk.chain, blk.aw.qkv

    def fn():
        xq, xs = mx_quant(blk.normed)
        c.qkv = mx_mm(xq, xs, lin)

    return fn


def make_producer(blk):
    c = blk.chain
    from . import host_hooks

    insert_kwargs = host_hooks.get(host_hooks.STEP_META)

    def fn():
        c.insert(**insert_kwargs(blk.indexer.layer_name, blk.index_prefix))

    return fn


def make_scorer(blk):
    c, ix = blk.chain, blk.indexer

    def fn():
        md = c.idx_md
        nd = md.num_decode_tokens
        iq = c.index_q[: md.num_actual_tokens].view(-1, ix.num_index_heads, ix.index_head_dim)
        ix._decode_score_msa(md, iq[:nd], ix.index_cache.kv_cache, md.unified_scores[:nd])

    return fn


def make_topk(blk):
    c, ix = blk.chain, blk.indexer

    def fn():
        ix._topk_route_a(c.idx_md, c.idx_md.unified_scores)

    return fn


def make_decode(blk):
    c = blk.chain

    def fn():
        c.main.forward(blk.attn_layer, c.q, c.kv_cache, c.out, query_fp8=c.q_fp8)

    return fn


def make_oproj(blk):
    c, lin = blk.chain, blk.aw.o

    def fn():
        xq, xs = mx_quant(c.out[: blk.t])
        blk.attn_partial = mx_mm(xq, xs, lin)

    return fn


# ----------------------------------------------------------------------------- MoE block
def make_boundary_post(blk):
    def fn():
        blk.ar.ar_norm(
            blk.attn_partial,
            blk.residual_mid,
            blk.aw.post_norm,
            blk.normed2,
            residual_out=blk.residual_out,
        )

    return fn


def make_router(blk):
    import vllm._custom_ops as ops

    w = blk.mw.router

    def fn():
        blk.logits = ops.fp32_router_gemm(blk.normed2, w)

    return fn


def make_moe_quant(blk):
    import vllm._custom_ops as ops

    gs = blk.mw.a1_gscale

    def fn():
        q, sf = ops.scaled_fp4_quant(blk.normed2, gs, is_sf_swizzled_layout=False)
        blk.xq4, blk.xsf4 = q, sf

    return fn


def make_moe_experts(blk):
    import flashinfer.fused_moe as fim

    try:
        mm2 = int(fim.RoutingMethodType.MiniMax2)
    except AttributeError:
        from flashinfer.tllm_enums import RoutingMethodType

        mm2 = int(RoutingMethodType.MiniMax2)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    act = int(activation_to_flashinfer_int(MoEActivation.SWIGLUOAI_UNINTERLEAVE))
    mw = blk.mw
    w13, w13s, w2, w2s = mw.experts
    t = blk.t

    def fn():
        r = fim.trtllm_fp4_block_scale_moe(
            routing_logits=blk.logits,
            routing_bias=mw.bias,
            hidden_states=blk.xq4.view(torch.uint8).reshape(t, -1),
            hidden_states_scale=blk.xsf4.view(torch.float8_e4m3fn).reshape(t, -1),
            gemm1_weights=w13,
            gemm1_weights_scale=w13s,
            gemm1_bias=None,
            gemm1_alpha=mw.gemm1_alpha,
            gemm1_beta=mw.gemm1_beta,
            gemm1_clamp_limit=mw.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=w2s,
            gemm2_bias=None,
            output1_scale_scalar=mw.g1_scale_c,
            output1_scale_gate_scalar=mw.g1_alphas,
            output2_scale_scalar=mw.g2_alphas,
            num_experts=128,
            top_k=TOP_K,
            n_group=0,
            topk_group=0,
            intermediate_size=mw.inter,
            local_expert_offset=0,
            local_num_experts=128,
            routed_scaling_factor=ROUTE_SCALE,
            routing_method_type=mm2,
            do_finalize=True,
            activation_type=act,
        )
        blk.routed = r[0] if isinstance(r, (list, tuple)) else r

    return fn


def make_shared(blk):
    """vLLM SharedExperts.maybe_forward_async: input-ready event on main, aux waits on it, shared MLP on aux,
    output-ready event on aux; main waits on that event before the add (runner/shared_experts.py:127-146)."""
    mw = blk.mw
    op = torch.ops._C.silu_and_mul_with_clamp
    it = mw.inter
    blk.ev_in, blk.ev_out = torch.cuda.Event(), torch.cuda.Event()

    def fn():
        blk.ev_in.record(torch.cuda.current_stream())
        with torch.cuda.stream(blk.aux):
            blk.ev_in.wait(blk.aux)
            xq, xs = mx_quant(blk.normed2)
            gu = mx_mm(xq, xs, mw.shared_gu)
            a = torch.empty(blk.t, it, dtype=torch.bfloat16, device="cuda")
            op(a, gu, SWIGLU_LIMIT, SWIGLU_ALPHA, SWIGLU_BETA)
            aq, as_ = mx_quant(a)
            blk.shared_out = mx_mm(aq, as_, mw.shared_dn)
            blk.ev_out.record(blk.aux)
        blk.shared_pending = True

    return fn


def make_combine(blk):
    def fn():
        join_aux(blk)
        torch.add(blk.routed, blk.shared_out, out=blk.ffn_out)

    return fn


def join_aux(blk):
    """Join the aux stream only if this run forked it (capturing streams may only wait on captured work)."""
    if getattr(blk, "shared_pending", False):
        blk.ev_out.wait(torch.cuda.current_stream())
        blk.shared_pending = False


STOCK = {n: globals()[f"make_{n}"] for n in ALL_STAGES}


def resolve(spec: str):
    """``module:attr`` -> factory; ``none`` -> stage removed; ``stock`` -> stock factory (caller handles)."""
    if spec == "none":
        return None
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


class Stock_AR:
    """vLLM fused_allreduce_gemma_rms_norm: FlashInfer trtllm one-shot AR + residual + Gemma RMSNorm (fp32 acc)."""

    def __init__(self, world, rank, cpu_group):
        import flashinfer.comm as fc
        from flashinfer.comm.mnnvl import TorchDistBackend

        self.fc = fc
        self.ws = fc.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world,
            rank=rank,
            max_token_num=(64 << 20) // (H * 2),
            hidden_dim=H,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group=cpu_group),
            group=cpu_group,
        )

    def ar_norm(self, x, residual, gamma, norm_out, residual_out):
        fc = self.fc
        fc.allreduce_fusion(
            input=x,
            workspace=self.ws,
            pattern=fc.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            residual_in=residual,
            rms_gamma=gamma,
            rms_eps=EPS,
            scale_factor=None,
            layout_code=fc.QuantizationSFLayout.SWIZZLED_128x4,
            use_oneshot=True,
            fp32_acc=True,
            weight_bias=1.0,
            trigger_completion_at_end=True,
        )
