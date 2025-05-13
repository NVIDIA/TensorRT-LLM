# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, Tuple, Union

import fp8_mha_api
import torch
import transformer_engine.pytorch.cpp_extensions as ext
import transformer_engine.pytorch.fp8 as fp8
import transformer_engine_extensions as tex
from torch.nn.parameter import Parameter
from transformer_engine.pytorch.module import TransformerEngineBaseModule

_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True

#FP8Tensors indices used (in this case 8)
# GEMM1_INPUT  - unrelated
# GEMM1_WEIGHT - unrelated
# GEMM2_WEIGHT - unrelated
# GRAD_OUTPUT2
# GEMM1_OUTPUT - should be QKV
# GEMM2_INPUT  - should be O
# GRAD_INPUT1  - should be dO
# GRAD_OUTPUT1 - should be dQKV
# need Index for:
# S  8
# dP 9

# Make sure no unintended scales are accessed.
for name in tex.FP8Tensors.__entries:
    val = int(tex.FP8Tensors.__dict__[name])
    if val >= 10:
        print(name, val)
assert all([
    int(tex.FP8Tensors.__dict__[name]) < 10 for name in tex.FP8Tensors.__entries
])
# Map names to make it easier to read.
META_QKV = tex.FP8Tensors.GEMM1_OUTPUT
META_O = tex.FP8Tensors.GEMM2_INPUT
META_DO = tex.FP8Tensors.GRAD_INPUT1
META_DQKV = tex.FP8Tensors.GRAD_OUTPUT1

# New scales.
META_S = 10
META_DP = 11  #TODO this is E5M2!


class _MHA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: torch.Tensor, qkv_weight: torch.Tensor,
                qkv_bias: torch.Tensor, proj_weight: torch.Tensor,
                proj_bias: torch.Tensor, cu_seqlens: torch.Tensor,
                num_attention_heads: int, p_dropout: float, max_s: int,
                set_zero: bool, fp8_meta: Dict[str, Any],
                workspace: torch.Tensor, is_training: bool) -> torch.Tensor:
        assert inp.dim() == 2
        # Make sure input dimensions are compatible
        in_features = qkv_weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        h = num_attention_heads
        d = in_features // h
        n_tokens = inp.shape[0]

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"],
                                                 fprop_tensor=True)

        npad = 256 - (n_tokens % 256)
        if npad < 256:
            inp = torch.nn.functional.pad(inp, (0, 0, 0, npad))
        inputmat, inputmat_t = ext.fp8_cast_transpose_fused(
            inp,
            fp8_meta["scaling"],
            tex.FP8Tensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )
        ext.fp8_cast_transpose_fused(
            qkv_weight,
            fp8_meta["scaling"],
            tex.FP8Tensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            cast_out=qkv_weight.cast,
            transpose_out=qkv_weight.transposed,
        )
        qkv_out = torch.empty(
            inputmat.shape[0],
            qkv_weight.shape[0],
            dtype=torch.int8,
            device="cuda",
        )
        ext.fp8_gemm(
            qkv_weight.cast,
            tex.FP8Tensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            inputmat,
            tex.FP8Tensors.GEMM1_INPUT,
            fp8_dtype_forward,
            fp8_meta["scaling"],
            torch.int8,
            workspace,
            bias=qkv_bias,
            use_bias=True,
            out=qkv_out,
            out_index=tex.FP8Tensors.GEMM1_OUTPUT,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        ##################FP8_FMHA change begins for FPROP ##############################
        #### [FP8_FMHA] cast_to_fp16 -> FP16_FMHA can be replaced with FP8_FMHA

        #qkv_out = ext.cast_from_fp8(
        #    qkv_out,
        #    fp8_meta["scaling"],
        #    tex.FP8Tensors.GEMM1_OUTPUT,
        #    fp8_dtype_forward,
        #    ext.TE_DType[torch.float16]
        #)
        #qkv_out = qkv_out[:n_tokens,:]

        ## FMHA
        #b = cu_seqlens.numel() - 1
        #is_nl = False
        #if b < 4 and b > 1:
        #    max_s = 512
        #    is_nl = True
        #qkv_out = qkv_out.view(-1, 3, h, d)

        #context, S_dmask = fmha.fwd(qkv_out, cu_seqlens, p_dropout, max_s, is_training, is_nl, set_zero, None)
        #context = context.view(-1, in_features)
        #if npad < 256:
        #    context = torch.nn.functional.pad(context, (0,0,0,npad))
        #context, context_t = ext.fp8_cast_transpose_fused(
        #    context,
        #    fp8_meta["scaling"],
        #    tex.FP8Tensors.GEMM2_INPUT,
        #    fp8_dtype_forward,
        #)

        qkv_out = qkv_out[:n_tokens, :]
        qkv_out = qkv_out.view(-1, 3, h, d)
        rng_state = torch.get_rng_state()

        context_, M, Z = fp8_mha_api.fwd(
            qkv_out,
            cu_seqlens,
            fp8_meta["scaling"].scale_inv[META_QKV],  #d_scale_qkv
            fp8_meta["scaling"].scale[META_O],  #q_scale_o
            fp8_meta["scaling"].amax_history[0][META_S],  #amax_s
            fp8_meta["scaling"].amax_history[0][META_O],  #amax_o
            p_dropout,
            max_s,
            is_training,
            set_zero,
            None,  # gen
        )

        context = context_.view(-1, in_features)

        if npad < 256:
            context = torch.nn.functional.pad(context, (0, 0, 0, npad))
        # unfortunately can't get rid of this transpose as this is needed for bwd.
        context_t = tex.fp8_transpose(
            context,
            fp8_dtype_forward,
        )

        ##################FP8_FMHA change ends for FPROP ##############################
        ext.fp8_cast_transpose_fused(
            proj_weight,
            fp8_meta["scaling"],
            tex.FP8Tensors.GEMM2_WEIGHT,
            fp8_dtype_forward,
            cast_out=proj_weight.cast,
            transpose_out=proj_weight.transposed,
        )
        proj_out = ext.fp8_gemm(
            proj_weight.cast,
            tex.FP8Tensors.GEMM2_WEIGHT,
            fp8_dtype_forward,
            context,
            tex.FP8Tensors.GEMM2_INPUT,
            fp8_dtype_forward,
            fp8_meta["scaling"],
            torch.float16,
            workspace,
            bias=proj_bias,
            use_bias=True,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        proj_out = proj_out[:n_tokens, :]

        ctx.save_for_backward(
            inputmat_t,
            qkv_weight,
            workspace,
            fp8_meta["scaling"].scale_inv[
                tex.FP8Tensors.GEMM1_WEIGHT].clone().detach(),
            fp8_meta["scaling"].scale_inv[
                tex.FP8Tensors.GEMM1_INPUT].clone().detach(),
            qkv_out,
            M,
            Z,  #S_dmask,
            context_,
            context_t,
            proj_weight,
            fp8_meta["scaling"].scale_inv[
                tex.FP8Tensors.GEMM2_WEIGHT].clone().detach(),
            fp8_meta["scaling"].scale_inv[
                tex.FP8Tensors.GEMM2_INPUT].clone().detach(),
            #TODO remove duplicates.
            fp8_meta["scaling"].scale_inv[META_QKV].clone().detach(
            ),  # d_scale_qkv
            fp8_meta["scaling"].scale_inv[META_S].clone().detach(),  # d_scale_s
            fp8_meta["scaling"].scale_inv[META_O].clone().detach(),  # d_scale_o
            fp8_meta["scaling"].scale[META_S].clone().detach(),  # q_scale_s
        )
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.set_zero = set_zero
        #ctx.is_nl = is_nl
        ctx.hidden_size = in_features
        ctx.num_attention_heads = num_attention_heads
        ctx.rng_state = rng_state

        return proj_out

    @staticmethod
    def backward(
            ctx,
            grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        (
            inputmat_t,
            qkv_weight,
            workspace,
            qkv_fwd_weight_scale_inv,
            qkv_fwd_inp_scale_inv,
            qkv_out,
            M,
            Z,  #S_dmask,
            context,
            context_t,
            proj_weight,
            proj_fwd_weight_scale_inv,
            proj_fwd_inp_scale_inv,
            d_scale_qkv,
            d_scale_s,
            d_scale_o,
            q_scale_s,
        ) = ctx.saved_tensors
        #grad_output, grad_output_c, grad_output_t, grad_bias = grad_output_preprocess(
        #    ctx, grad_output, ctx.parallel_mode == "row"
        #)
        fp8_dtype_forward = fp8.get_fp8_te_dtype(ctx.fp8_meta["recipe"],
                                                 fprop_tensor=True)
        fp8_dtype_backward = fp8.get_fp8_te_dtype(ctx.fp8_meta["recipe"],
                                                  fprop_tensor=False)
        n_tokens = grad_output.shape[0]
        npad = 256 - (n_tokens % 256)
        if npad < 256:
            grad_output = torch.nn.functional.pad(grad_output, (0, 0, 0, npad))
        proj_bgrad, proj_grad_output_c, proj_grad_output_t = ext.fp8_cast_transpose_bgrad_fused(
            grad_output,
            ctx.fp8_meta["scaling"],
            tex.FP8Tensors.GRAD_OUTPUT2,
            fp8_dtype_backward,
        )
        # PROJ DGRAD
        proj_dgrad = torch.empty(
            grad_output.shape[0],
            ctx.hidden_size,
            dtype=torch.int8,
            device="cuda",
        )
        #        print ('PROJ_DGRAD')
        ext.fp8_gemm(
            proj_weight.transposed,
            tex.FP8Tensors.GEMM2_WEIGHT,
            fp8_dtype_forward,
            proj_grad_output_c,
            tex.FP8Tensors.GRAD_OUTPUT2,
            fp8_dtype_backward,
            ctx.fp8_meta["scaling"],
            torch.int8,  #float16,
            workspace,
            bias=proj_bgrad,
            use_bias=False,
            out=proj_dgrad,
            out_index=tex.FP8Tensors.GRAD_INPUT1,
            use_split_accumulator=_2X_ACC_DGRAD,
            A_scale_inv_override=proj_fwd_weight_scale_inv,
        )
        #        proj_dgrad = ext.cast_to_fp8(
        #            proj_dgrad,
        #            ctx.fp8_meta["scaling"],
        #            tex.FP8Tensors.GRAD_INPUT1,
        #            fp8_dtype_backward)
        # PROJ WGRAD
        proj_wgrad = ext.fp8_gemm(
            context_t,
            tex.FP8Tensors.GEMM2_INPUT,
            fp8_dtype_forward,
            proj_grad_output_t,
            tex.FP8Tensors.GRAD_OUTPUT2,
            fp8_dtype_backward,
            ctx.fp8_meta["scaling"],
            torch.float16,
            workspace,
            use_split_accumulator=_2X_ACC_WGRAD,
            A_scale_inv_override=proj_fwd_inp_scale_inv,
        )
        ####################################################################################
        ##################FP8_FMHA change begins for BPROP #################################
        #### [FP8_FMHA] cast_to_fp16 -> FP16_FMHA dgrad can be replaced with FP8_FMHA dgrad

        #proj_dgrad = ext.cast_from_fp8(
        #    proj_dgrad,
        #    ctx.fp8_meta["scaling"],
        #    tex.FP8Tensors.GRAD_INPUT1,
        #    fp8_dtype_backward,
        #    ext.TE_DType[torch.float16]
        #)
        #proj_dgrad = proj_dgrad[:n_tokens,:]
        #proj_dgrad = proj_dgrad.view(-1, ctx.num_attention_heads, ctx.hidden_size//ctx.num_attention_heads)
        #if ctx.is_nl:
        #    dqkv, dp, dkv = fmha.bwd_nl(proj_dgrad, qkv_out, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.set_zero)
        #else:
        #    dqkv, dp = fmha.bwd(proj_dgrad, qkv_out, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.set_zero)

        rng_state_old = torch.get_rng_state()
        torch.set_rng_state(ctx.rng_state)

        dqkv, = fp8_mha_api.bwd(
            proj_dgrad.view_as(context),
            qkv_out,
            context,
            M,
            Z,
            ctx.cu_seqlens,
            d_scale_qkv,
            d_scale_s,
            d_scale_o,
            ctx.fp8_meta['scaling'].scale_inv[META_DO],  # d_scale_do
            ctx.fp8_meta['scaling'].scale_inv[META_DP],  # d_scale_dp
            q_scale_s,
            ctx.fp8_meta['scaling'].scale[META_DP],  # q_scale_dp
            ctx.fp8_meta['scaling'].scale[META_DQKV],  # q_scale_dqkv
            ctx.fp8_meta['scaling'].amax_history[0][META_DP],  # amax_dp
            ctx.fp8_meta['scaling'].amax_history[0][META_DQKV],  # amax_dqkv
            ctx.p_dropout,
            ctx.max_s,
            ctx.set_zero,
            None)

        torch.set_rng_state(rng_state_old)

        dqkv = dqkv.view(-1, 3 * ctx.hidden_size)
        if npad < 256:
            dqkv = torch.nn.functional.pad(dqkv, (0, 0, 0, npad))
        ####################################################################################
        qkv_bgrad, dqkv_grad_output_c, dqkv_grad_output_t = ext.fp8_cast_transpose_bgrad_fused(
            dqkv,
            ctx.fp8_meta["scaling"],
            tex.FP8Tensors.GRAD_OUTPUT1,
            fp8_dtype_backward,
        )
        # QKV DGRAD
        qkv_dgrad = ext.fp8_gemm(
            qkv_weight.transposed,
            tex.FP8Tensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            dqkv_grad_output_c,
            tex.FP8Tensors.GRAD_OUTPUT1,
            fp8_dtype_backward,
            ctx.fp8_meta["scaling"],
            torch.float16,
            workspace,
            use_split_accumulator=_2X_ACC_DGRAD,
            A_scale_inv_override=qkv_fwd_weight_scale_inv,
        )
        # QKV WGRAD
        qkv_wgrad = ext.fp8_gemm(
            inputmat_t,
            tex.FP8Tensors.GEMM1_INPUT,
            fp8_dtype_forward,
            dqkv_grad_output_t,
            tex.FP8Tensors.GRAD_OUTPUT1,
            fp8_dtype_backward,
            ctx.fp8_meta["scaling"],
            torch.float16,
            workspace,
            use_split_accumulator=_2X_ACC_WGRAD,
            A_scale_inv_override=qkv_fwd_inp_scale_inv,
        )
        qkv_dgrad = qkv_dgrad[:n_tokens, :]
        fp8.fp8_updates(
            ctx.fp8_meta,
            reduce_amax_across_tp_group=False,
            tp_group=None,
            fwd_bwd_update=False,
            fwd_only_update=False,
        )
        return (qkv_dgrad, qkv_wgrad, qkv_bgrad, proj_wgrad, proj_bgrad, None,
                None, None, None, None, None, None, None)

        #grad_output_c, grad_output_t = fp8_cast_transpose_fused(
        #    grad_output,
        #    ctx.fp8_meta["scaling"],
        #    tex.FP8Tensors.GRAD_OUTPUT1,
        #    fp8_dtype_backward,
        #)


class FP8_MHA(TransformerEngineBaseModule):

    def __init__(self, config, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        self.set_zero = config.packed_samples  # TODO read this from config
        assert self.d * self.h == self.hidden_size, "Invalid hidden size/num_heads"

        self.qkv_weight = Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            ))
        self.qkv_bias = Parameter(
            torch.empty(
                self.hidden_size * 3,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            ))
        self.proj_weight = Parameter(
            torch.empty(
                self.hidden_size,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            ))
        self.proj_bias = Parameter(
            torch.empty(
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            ))
        with torch.no_grad():
            self.qkv_bias.zero_()
            self.qkv_weight.fill_(1.0)
            self.proj_bias.zero_()
            self.proj_weight.fill_(1.0)
        # workspace for cublasLt
        self.workspace = torch.empty(_CUBLASLT_WORKSPACE_SIZE_BYTES,
                                     dtype=torch.int8,
                                     device="cuda")
        self.max_adjusted = False

    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop"""
        super().fp8_init(num_gemms)
        if self.max_adjusted:
            return
        self.fp8_meta['fp8_max'][META_DP] = 57344.0
        self.max_adjusted = True

    def forward(self, inp: torch.Tensor, cu_seqlens, max_s) -> torch.Tensor:
        self.pre_forward(inp, num_gemms=3)

        out = _MHA.apply(inp, self.qkv_weight, self.qkv_bias, self.proj_weight,
                         self.proj_bias, cu_seqlens, self.h, self.p_dropout,
                         max_s, self.set_zero, self.fp8_meta, self.workspace,
                         self.training)

        if torch.is_grad_enabled() and self.training:
            fp8.fp8_updates(
                self.fp8_meta,
                reduce_amax_across_tp_group=False,
                tp_group=None,
                fwd_bwd_update=False,
                fwd_only_update=True,
            )
#        out = out.view(-1, self.hidden_size)

        return out  #, self.fp8_meta["scaling"].amax_history


#fp8_recipe = recipe.DelayedScaling(
#    margin=0,
#    interval=1,
#    fp8_format=recipe.Format.E4M3,
#    amax_history_len=1,
#    amax_compute_algo="most_recent",
#)
#
#bs = 1
#seq_len = 333
#a = torch.empty(bs*seq_len,1024,dtype=torch.half).cuda()
#a.fill_(0.1)
#seqlen = torch.empty(bs,dtype=torch.int32).cuda()
#seqlen.fill_(seq_len)
##A_index = tex.FP8Tensors.GEMM1_INPUT
##b = torch.ones(20,10,dtype=torch.half).cuda()
##B_index = tex.FP8Tensors.GEMM1_WEIGHT
#class Config():
#    def __init__(self):
#        self.hidden_size = 1024
#        self.attention_probs_dropout_prob = 0.1
#        self.num_attention_heads = 16
#        self.d = self.hidden_size // self.num_attention_heads
#        self.packed_samples = False # TODO read this from config
#mha = FP8_MHA(Config()).half()
#
#with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#    cu_seqlens = torch.zeros(bs+1, device=a.device, dtype=torch.int32)
#    cu_seqlens[1:] = torch.cumsum(seqlen, dim=0)
#    op = mha(a, cu_seqlens, seq_len)
#    op_grad = torch.ones(bs*seq_len, 1024, dtype=torch.float16).cuda()
#    op.backward(op_grad)
#    print (mha.qkv_weight.grad)
#print ('op {}:{} {} '.format(op.shape, op.dtype, op))
