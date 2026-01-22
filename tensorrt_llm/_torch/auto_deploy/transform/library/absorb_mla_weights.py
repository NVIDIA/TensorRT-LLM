from typing import Tuple

import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_mla import (
    compute_w_uq_uk,
    compute_w_uv_o,
)

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _absorb_mla_weights(gm: GraphModule) -> int:
    """Compute the absorbed weights of all MLA operations in the graph

    For each MLA operation, compute the absorbed weights and set them as inputs of the MLA operation.
        W_UQ_UK <= W_UQ * W_UK
        W_UV_O <= W_UV * W_O
    """

    def _register_parameter(
        gm: GraphModule, absorbed_weights: torch.Tensor, param_name: str, param_idx: int
    ) -> str:
        param_key = f"{param_name}_{param_idx}"
        new_param = torch.nn.Parameter(absorbed_weights)
        gm.register_parameter(param_key, new_param)
        return param_key

    def _get_dims(mla_inputs) -> Tuple[int, int, int, int, int, int]:
        """Get the dimensions of the MLA operation"""
        wkv_b, wq_b, w_uq_uk, wo_proj, w_uv_o = mla_inputs[5:10]
        wkv_b = gm.get_parameter(
            wkv_b.target
        )  # [num_heads * (qk_nope_head_dim+v_head_dim), kv_lora_rank]
        wq_b = gm.get_parameter(wq_b.target)  # [num_heads * q_head_dim, q_lora_rank]
        assert w_uq_uk is None
        assert w_uv_o is None
        wo_proj = gm.get_parameter(wo_proj.target)  # [hidden_size, num_heads * v_head_dim]

        qk_nope_head_dim = 128
        q_lora_rank = wq_b.shape[-1]
        assert q_lora_rank == 1536
        kv_lora_rank = wkv_b.shape[-1]
        assert kv_lora_rank == 512
        k_pe = mla_inputs[2]  # [bsz, 1, kv_len, qk_rope_head_dim]
        qk_rope_head_dim = k_pe.meta["val"].shape[-1]
        assert qk_rope_head_dim == 64
        q_head_dim = qk_rope_head_dim + qk_nope_head_dim
        num_heads = wq_b.shape[0] // q_head_dim
        assert num_heads == 128
        v_head_dim = wo_proj.shape[-1] // num_heads
        assert v_head_dim == 128
        return qk_nope_head_dim, q_lora_rank, kv_lora_rank, q_head_dim, num_heads, v_head_dim

    def _extract_weights_from_inputs(
        mla_inputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # MLA inputs order is defined in torch_deepseek_mla_no_cache
        wkv_b, wq_b, w_uq_uk, wo_proj, w_uv_o = mla_inputs[5:10]
        wkv_b = gm.get_parameter(
            wkv_b.target
        )  # [num_heads * (qk_nope_head_dim+v_head_dim), kv_lora_rank]
        wq_b = gm.get_parameter(wq_b.target)  # [num_heads * q_head_dim, q_lora_rank]
        assert w_uq_uk is None
        assert w_uv_o is None
        wo_proj = gm.get_parameter(wo_proj.target)  # [hidden_size, num_heads * v_head_dim]

        qk_nope_head_dim, q_lora_rank, kv_lora_rank, q_head_dim, num_heads, v_head_dim = _get_dims(
            mla_inputs
        )

        w_uq_qr = wq_b  # W_{UQ + QR}: wq_b ~ [num_heads * q_head_dim, q_lora_rank]
        w_uq_qr_t = (
            w_uq_qr.transpose(0, 1).reshape(q_lora_rank, num_heads, q_head_dim).contiguous()
        )  # [q_lora_rank, num_heads, q_head_dim]
        w_uq = w_uq_qr_t[
            :, :, :qk_nope_head_dim
        ].contiguous()  # [q_lora_rank, num_heads, qk_nope_head_dim]

        w_ukv = wkv_b
        w_ukv = w_ukv.view(
            num_heads, -1, kv_lora_rank
        )  # [num_heads, qk_nope_head_dim+v_head_dim, kv_lora_rank]
        w_uk = w_ukv[
            :, :qk_nope_head_dim, :
        ].contiguous()  # W_UKV = [num_heads, qk_nope_head_dim, kv_lora_rank]

        w_uv = w_ukv[:, -v_head_dim:]  # [num_heads, v_head_dim, kv_lora_rank]
        return w_uq, w_uk, w_uv, wo_proj

    def _is_mla_op(node: torch.fx.Node) -> bool:
        return is_op(node, torch.ops.auto_deploy.torch_deepseek_mla_no_cache)

    def _absorb_weights(node: torch.fx.Node, idx: int) -> None:
        mla_inputs = list(node.args)
        w_uq, w_uk, w_uv, wo_proj = _extract_weights_from_inputs(mla_inputs)
        _, _, kv_lora_rank, _, num_heads, v_head_dim = _get_dims(mla_inputs)
        w_uq_uk = compute_w_uq_uk(w_uq, w_uk)
        w_uq_uk_key = _register_parameter(gm, w_uq_uk, "w_uq_uk", idx)
        w_uv_o = compute_w_uv_o(w_uv, wo_proj, kv_lora_rank, num_heads, v_head_dim)
        w_uv_o_key = _register_parameter(gm, w_uv_o, "w_uv_o", idx)

        with graph.inserting_before(node):
            new_mla_inputs = mla_inputs
            new_mla_inputs[7] = graph.get_attr(w_uq_uk_key)
            new_mla_inputs[9] = graph.get_attr(w_uv_o_key)
            new_node = graph.call_function(
                torch.ops.auto_deploy.torch_deepseek_mla_no_cache,
                args=tuple(new_mla_inputs),
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    graph = gm.graph
    mla_nodes = [node for node in list(graph.nodes) if _is_mla_op(node)]
    for idx, node in enumerate(mla_nodes):
        _absorb_weights(node, idx)

    return len(mla_nodes)


@TransformRegistry.register("absorb_mla_weights")
class AbsorbMLAWeights(BaseTransform):
    """Absorb MLA weights by computing W_UQ_UK and W_UV_O for all MLA operations."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            num_matches = _absorb_mla_weights(gm)

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=num_matches > 0, has_valid_shapes=False
        )
        return gm, info
