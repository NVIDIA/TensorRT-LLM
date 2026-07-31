import contextlib
from typing import Callable, List, Optional, Union

import torch
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE


def get_symint_val(i: Union[torch.SymInt | int]):
    if isinstance(i, int):
        return i
    elif isinstance(i, torch.SymInt):
        node = i.node
        expr = node.expr
        shape_env: ShapeEnv = node.shape_env
        var_val = shape_env.var_to_val.get(expr, None) or expr.xreplace(
            shape_env.var_to_val)
        return var_val
    else:
        raise Exception("Only support int or torch.SymInt")


def get_arg(node, idx, arg_name):
    return node.args[idx] if len(node.args) > idx else node.kwargs[arg_name]


def is_call_function(node: Node, target: Union[List[Callable], Callable]):
    if isinstance(target, list):
        return node.op == "call_function" and node.target in target
    else:
        return node.op == "call_function" and node.target == target


def get_optional_trtllm_op(op_name: str) -> Optional[Callable]:
    try:
        return getattr(torch.ops.trtllm, op_name).default
    except AttributeError:
        return None


_enable_piecewise_cuda_graph_capture = False


def set_capture_piecewise_cuda_graph_flag(enable: bool):
    global _enable_piecewise_cuda_graph_capture
    _enable_piecewise_cuda_graph_capture = enable


def get_capture_piecewise_cuda_graph_flag() -> bool:
    global _enable_piecewise_cuda_graph_capture
    return _enable_piecewise_cuda_graph_capture


@contextlib.contextmanager
def capture_piecewise_cuda_graph(enable: bool):
    prev_enable = get_capture_piecewise_cuda_graph_flag()
    set_capture_piecewise_cuda_graph_flag(enable)
    try:
        yield
    finally:
        set_capture_piecewise_cuda_graph_flag(prev_enable)


def inplace_info():
    inplace_map = {
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default: {
            1: "input",
            2: "residual"
        },
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm_quant.default: {
            1: "out",
            2: "residual"
        },
        torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8.default: {
            1: "quant_q_out",
            2: "q_pe_out"
        },
        torch.ops.trtllm.fused_qk_norm_rope.default: {
            1: "qkv"
        },
        torch.ops.trtllm.fused_dit_qk_norm_rope.default: {
            1: "qkv"
        },
        torch.ops.trtllm.fused_dit_split_norm_rope.default: {
            1: "tensor"
        },
        torch.ops.trtllm.fused_dit_split_norm.default: {
            1: "tensor"
        },
        torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace.default:
        {
            1: "query",
            2: "key"
        },
        torch.ops.trtllm.logits_bitmask.default: {
            1: "logits"
        },
        torch.ops.trtllm.moe_unpermute_inplace.default: {
            1: "output"
        },
        torch.ops.trtllm.moe_output_memset_inplace.default: {
            1: "input"
        },
        torch.ops.trtllm.megamoe_prepare.default: {
            1: "x_out",
            2: "x_sf_out",
            3: "topk_idx_out",
            4: "topk_weights_out"
        },
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell.default:
        {
            1: "output"
        },
        torch.ops.trtllm.pp_recv_tensors.default: {
            1: "tensors"
        },
        torch.ops.trtllm.pp_send_tensors.default: {
            1: "tensors"
        },
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell.default: {
            1: "output"
        },
        torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell.default: {
            1: "output"
        },
        torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell.default: {
            1: "output"
        },
        torch.ops.trtllm.compressor_paged_kv_compress.default: {
            1: "paged_kv",
            2: "paged_score",
            3: "output"
        },
        torch.ops.trtllm.compressor_prefill_reduction.default: {
            1: "paged_kv",
            2: "paged_score",
            3: "output"
        },
        torch.ops.trtllm.compressor_postprocess_scatter.default: {
            1: "kv_out",
            2: "kv_cache",
            3: "quant_output",
            4: "scale_output"
        },
        torch.ops.trtllm.mhc_big_fuse.default: {
            1: "post_mix",
            2: "comb_mix",
            3: "layer_input"
        },
        torch.ops.trtllm.mhc_gemm_sqrsum_fma.default: {
            1: "y",
            2: "r"
        },
        torch.ops.trtllm.mhc_hc_head_apply.default: {
            1: "out"
        },
        torch.ops.trtllm.mhc_post_mapping.default: {
            1: "out"
        },
        torch.ops.trtllm.mhc_fused_hc.default: {
            1: "residual_cur",
            2: "post_mix_cur",
            3: "comb_mix_cur",
            4: "layer_input_cur",
            5: "y_acc_workspace",
            6: "r_acc_workspace",
            7: "done_counter_workspace"
        },
        torch.ops.trtllm.inplace_slice_copy.default: {
            1: "dest"
        },
        torch.ops.trtllm.verify_dynamic_tree_rejection_out_op.default: {
            5: "acceptIndex",
            6: "acceptTokenNum",
            7: "acceptToken"
        }
    }
    optional_inplace_infos = {
        "attn_custom_op_inplace": {
            1: "output",
            2: "output_sf"
        },
        "mla_custom_op_inplace": {
            1: "output"
        },
        "mla_dsa_attn_inplace": {
            1: "output"
        },
        "gdn_custom_op_inplace": {
            1: "output"
        },
        "minimax_m3_attn_custom_op_inplace": {
            1: "output"
        },
    }
    for op_name, mutates_args in optional_inplace_infos.items():
        op = get_optional_trtllm_op(op_name)
        if op is not None:
            inplace_map[op] = mutates_args
    if IS_CUDA_TILE_AVAILABLE:
        # cuda.tile availability depends on GPU capability thus runtime check.
        inplace_map[
            torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_.default] = {
                1: "x",
                2: "residual"
            }
    return inplace_map
