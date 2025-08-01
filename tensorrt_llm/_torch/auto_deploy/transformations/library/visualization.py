"""Transformation to the graph to render nicely in model_explorer."""

import json
from typing import Tuple

import model_explorer
import torch
import torch.export as te
from model_explorer.graph_builder import GraphNode, KeyValue, MetadataItem
from model_explorer.pytorch_exported_program_adater_impl import PytorchExportedProgramAdapterImpl
from torch import fx


def print_tensor(self, tensor: torch.Tensor, size_limit: int = 16):
    shape = tensor.shape
    total_size = 1
    for dim in shape:
        total_size *= dim

    if size_limit < 0 or size_limit >= total_size:
        return json.dumps(tensor.cpu().detach().to(torch.float32).numpy().tolist())

    return json.dumps(
        (tensor.cpu().detach().to(torch.float32).numpy().flatten())[:size_limit].tolist()
    )


def _get_shape(val):
    return json.dumps(
        list(
            map(
                lambda x: int(x) if str(x).isdigit() else str(x),
                val.shape,
            )
        )
    )


def add_outputs_metadata(self, fx_node: torch.fx.node.Node, node: GraphNode):
    out_vals = fx_node.meta.get("val")
    if out_vals is None:
        return

    if isinstance(out_vals, (tuple, list)):
        for idx, val in enumerate(out_vals):
            metadata = MetadataItem(id=str(idx), attrs=[])
            if val is None:
                continue
            dtype = str(val.dtype)
            shape = _get_shape(val)
            metadata.attrs.append(KeyValue(key="tensor_shape", value=dtype + shape))
            node.outputsMetadata.append(metadata)
    elif isinstance(out_vals, torch.Tensor):
        dtype = str(out_vals.dtype)
        shape = _get_shape(out_vals)
        metadata = MetadataItem(id="0", attrs=[KeyValue(key="tensor_shape", value=dtype + shape)])
        node.outputsMetadata.append(metadata)
    elif isinstance(out_vals, bool):
        metadata = MetadataItem(id="0", attrs=[KeyValue(key="tensor_shape", value="bool[1]")])
        node.outputsMetadata.append(metadata)
    else:
        raise ValueError(f"Unsupported output type: {type(out_vals)}")


PytorchExportedProgramAdapterImpl.print_tensor = print_tensor
PytorchExportedProgramAdapterImpl.add_outputs_metadata = add_outputs_metadata

# TODO(yudong): make custom_ops configurable
CUSTOM_OPS = (
    torch.ops.auto_deploy.torch_dist_all_reduce.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.auto_deploy.triton_attention_fused_mha_with_cache.default,
    torch.ops.auto_deploy.trtllm_dist_fused_linear_all_reduce.default,
    torch.ops.auto_deploy.torch_linear_simple.default,
    torch.ops.aten.split_with_sizes.default,
)


# TODO(yudong): make viz as non-block call.
def visualize_namespace(gm: fx.GraphModule, args: Tuple[torch.Tensor, ...], dynamic_shapes):
    ep = te.export(gm, args=args, dynamic_shapes=dynamic_shapes)
    graph = ep.graph
    # Ensure the ops land up in the right module for better viz
    for n in graph.nodes:
        if n.target in CUSTOM_OPS:
            n.meta["nn_module_stack"] = n.args[0].meta["nn_module_stack"]

    model_explorer.visualize_pytorch("model-viz", ep)
