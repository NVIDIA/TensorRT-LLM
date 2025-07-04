from operator import getitem

import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, Ignored,
                                             KeywordArg, Match,
                                             MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

import tensorrt_llm

from ...distributed import AllReduceFusionOp

aten = torch.ops.aten
from tensorrt_llm.mapping import Mapping


def register_ar_residual_norm(custom_pass: PatternMatcherPass):
    # TODO: add pp + tp support
    mapping = Mapping(
        world_size=tensorrt_llm.mpi_world_size(),
        tp_size=tensorrt_llm.mpi_world_size(),
        rank=tensorrt_llm.mpi_rank(),
    )
    residual_key = KeywordArg("residual")
    trtllm_allreduce_default = CallFunction(
        torch.ops.trtllm.allreduce.default, KeywordArg("input"), None, None,
        None, None, KeywordArg("workspace"), mapping.tp_group,
        KeywordArg("strategy"), int(AllReduceFusionOp.NONE), Ignored(),
        KeywordArg("trigger_completion_at_end"))
    getitem_x = CallFunction(getitem, trtllm_allreduce_default, 0)
    add_Tensor = CallFunction(aten.add.Tensor,
                              getitem_x,
                              residual_key,
                              _users=MULTIPLE)
    _torch_rms_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    ar_residual_norm_pattern = MultiOutputPattern(
        [_torch_rms_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        workspace: torch.LongTensor,
        residual: torch.Tensor,
        strategy: int,
        norm_weight: torch.nn.Parameter,
        eps: float,
        trigger_completion_at_end: bool,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        workspace: torch.LongTensor,
        residual: torch.Tensor,
        strategy: int,
        norm_weight: torch.nn.Parameter,
        eps: float,
        trigger_completion_at_end: bool,
    ):
        all_reduce_output = torch.ops.trtllm.allreduce(
            input, residual, norm_weight, None, None, workspace,
            mapping.tp_group, int(strategy),
            int(AllReduceFusionOp.RESIDUAL_RMS_NORM), float(eps),
            trigger_completion_at_end)
        return all_reduce_output[0], all_reduce_output[1]

    def extra_check(match: Match) -> bool:
        # Residual should be a tensor
        residual_node = match.ctx.pattern_to_node[residual_key]
        if not isinstance(residual_node, torch.fx.graph.Node):
            return False
        getitem_node = match.ctx.pattern_to_node[getitem_x]
        if not isinstance(getitem_node, torch.fx.graph.Node):
            return False

        getitem_node_shape = getitem_node.meta["tensor_meta"].shape
        residual_node_shape = residual_node.meta["tensor_meta"].shape

        if getitem_node_shape != residual_node_shape:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=ar_residual_norm_pattern,
        extra_check=extra_check,
    )
