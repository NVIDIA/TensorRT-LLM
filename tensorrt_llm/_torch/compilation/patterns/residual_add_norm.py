from operator import getitem

import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

aten = torch.ops.aten
from torch._higher_order_ops.auto_functionalize import auto_functionalized


def register_add_norm(custom_pass: PatternMatcherPass):
    residual = KeywordArg("residual")
    add_Tensor = CallFunction(aten.add.Tensor,
                              KeywordArg("input"),
                              residual,
                              _users=MULTIPLE)
    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    add_norm_pattern = MultiOutputPattern([flashinfer_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
            input=input,
            residual=residual,
            weight=norm_weight,
            eps=eps)
        return at[1], at[2]

    def extra_check(match: Match):
        # Check the original residual and hidden has no other users since we will inplace update them
        residual_node = match.ctx.pattern_to_node[add_Tensor]
        if not isinstance(residual_node, torch.fx.graph.Node):
            return False

        # torch uses dict here to guarantee the order of the uses
        if list(residual_node.args[0].users.keys()
                )[-1] != residual_node or list(
                    residual_node.args[1].users.keys())[-1] != residual_node:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_pattern,
        extra_check=extra_check,
    )


def register_add_norm_quant(custom_pass: PatternMatcherPass):
    residual_out = CallFunction(aten.add.Tensor,
                                KeywordArg("input"),
                                KeywordArg("residual"),
                                _users=MULTIPLE)

    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        residual_out,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=1)

    static_quantize = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        flashinfer_norm_default,
        KeywordArg("scale"),
        _users=1)

    quant_out = CallFunction(getitem, static_quantize, 0, _users=1)
    add_norm_quant_pattern = MultiOutputPattern([quant_out, residual_out])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm_quant.default,
            out=out,
            input=input,
            residual=residual,
            weight=norm_weight,
            scale=scale,
            eps=eps,
        )
        # at[1]=out (fp8 quant), at[2]=residual (updated)
        return at[1], at[2]

    def extra_check(match: Match) -> bool:
        # flashinfer_fused_add_rmsnorm_quant mutates residual in-place. Check that the original
        # residual tensor has the add node as its last user so no downstream node sees a stale pre-mutation value.
        add_node = match.ctx.pattern_to_node[residual_out]
        if not isinstance(add_node, torch.fx.graph.Node):
            return False

        if list(add_node.args[1].users.keys())[-1] != add_node:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_quant_pattern,
        extra_check=extra_check,
    )
