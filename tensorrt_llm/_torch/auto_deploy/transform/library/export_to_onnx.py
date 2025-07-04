# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Type

import onnx
import onnxscript
import torch
from onnx.defs import OpSchema
from onnxscript import ir, opset17, opset20
from onnxscript.values import Opset
from pydantic import Field
from torch.export import Dim
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from ._chat_template import process_chat_template
from ._config_export import export_llm_config


class ExportToONNXConfig(TransformConfig):
    """Configuration for the export to ONNX transform."""

    output_dir: Path = Field(
        description="The directory to save the exported ONNX model.",
    )
    output_name: str = Field(
        description="The name of the exported ONNX model.",
        default="model.onnx",
    )
    is_eagle_base: bool = Field(
        description="Whether the model is an Eagle base model.",
        default=False,
    )


custom_rope_schema = OpSchema(
    name="rope_with_explicit_cos_sin",
    domain="auto_deploy",
    since_version=1,
    doc="Rope with explicit cos and sin caches.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Q tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="K tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cos",
            description="Cos cache",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="sin",
            description="Sin cache",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="unsqueeze_dim",
            type=OpSchema.AttrType.INT,
            description="Unsqueeze dimension. Must be 1 or 2.",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(custom_rope_schema)


def custom_rope_op(
    q: ir.Tensor, k: ir.Tensor, cos: ir.Tensor, sin: ir.Tensor, unsqueeze_dim: int = 1
):
    auto_deploy_op = onnxscript.values.Opset(domain="auto_deploy", version=1)
    return auto_deploy_op.rope_with_explicit_cos_sin(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


# ============================================================================
# ONNX Custom Op Registration for simple_linear
# ============================================================================


def custom_simple_linear_op(input: ir.Tensor, weight: ir.Tensor, bias: Optional[ir.Tensor]):
    weight = opset20.Transpose(weight, perm=[1, 0])
    if bias is None:
        return opset20.MatMul(input, weight)
    return opset20.Add(opset20.MatMul(input, weight), bias)


auto_deploy_opset = Opset("auto_deploy", 1)
trt_domain_name = "trt"
trt_opset = Opset(trt_domain_name, 1)

# ============================================================================
# ONNX Custom Op Registration for GatherND
# ============================================================================


def custom_gather_nd_op(data: ir.Tensor, indices: ir.Tensor, batch_dims: int):
    return opset17.GatherND(data, indices, batch_dims=batch_dims)


# ============================================================================
# ONNX Custom Op Registration for AttentionPlugin
# ============================================================================

# Define ONNX OpSchema for AttentionPlugin
rope_attention_schema = OpSchema(
    name="AttentionPlugin",
    domain=trt_domain_name,
    since_version=1,
    doc="Fused RoPE + Attention operation for efficient inference.",
    inputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description="Concatenated Q, K, V tensors in shape [batch, seq_len, qkv_hidden_size]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_values",
            description="Concatenated past K and V cache in shape [batch, 2, num_kv_heads, past_len, head_size]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context lengths for each sequence in shape [batch]",
            type_str="T1",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="Concatenated cos and sin values for RoPE in shape [max_seq_len, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="kvcache_start_index",
            description="KV cache start index for each sequence in shape [batch]",
            type_str="T1",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Attention output in shape [batch, seq_len, hidden_size]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="updated_past_key_values",
            description="Updated past K and V cache",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type for floating point tensors.",
        ),
        (
            "T1",
            ["tensor(int32)", "tensor(int64)"],
            "Input data type for integer tensors.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0 or 1).",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head.",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads.",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads.",
            required=True,
        ),
    ],
)


def onnx_rope_attention_op(
    qkv: ir.Tensor,
    past_key_values: ir.Tensor,
    context_lengths: ir.Tensor,
    rope_rotary_cos_sin: ir.Tensor,
    kvcache_start_index: ir.Tensor,
    enable_tree_attention: int,
    head_size: int,
    num_kv_heads: int,
    num_q_heads: int,
):
    """
    ONNX custom op translation function for AttentionPlugin.

    This function creates a custom ONNX op node in the auto_deploy domain.
    The actual implementation will need to be provided in the inference engine
    (e.g., ONNX Runtime or TensorRT) that loads this ONNX model.

    Note: This is a translation function for torch.onnx.export's custom_translation_table,
    so it should NOT have @script() decorator.
    """
    # Call the custom op from the auto_deploy domain
    return trt_opset.AttentionPlugin(
        qkv,
        past_key_values,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        enable_tree_attention=enable_tree_attention,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
    )


# Register the schema
onnx.defs.register_schema(rope_attention_schema)

# ============================================================================
# ONNX Custom Op Registration for torch_attention
# ============================================================================

# Define ONNX OpSchema for torch_attention
torch_attention_schema = OpSchema(
    name="torch_attention",
    domain="auto_deploy",
    since_version=1,
    doc="SDPA attention (with optional GQA) that supports bnsd and bsnd memory layouts.",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Query tensor [batch, seq_len_q/num_heads, num_heads/seq_len_q, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key",
            description="Key tensor [batch, seq_len_k/num_kv_heads, num_kv_heads/seq_len_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value tensor [batch, seq_len_k/num_kv_heads, num_kv_heads/seq_len_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="attn_mask",
            description="Optional attention mask in [batch, num_heads, seq_len_q, seq_len_k] layout",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="sinks",
            description="Optional sinks tensor",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Attention output in the same layout as inputs",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type for floating point tensors.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="dropout_p",
            type=OpSchema.AttrType.FLOAT,
            description="Dropout probability.",
            required=False,
        ),
        OpSchema.Attribute(
            name="is_causal",
            type=OpSchema.AttrType.INT,
            description="Whether to apply causal masking (0 or 1).",
            required=False,
        ),
        OpSchema.Attribute(
            name="scale",
            type=OpSchema.AttrType.FLOAT,
            description="Attention scale factor.",
            required=False,
        ),
        OpSchema.Attribute(
            name="sliding_window",
            type=OpSchema.AttrType.INT,
            description="Sliding window size for attention.",
            required=False,
        ),
        OpSchema.Attribute(
            name="logit_cap",
            type=OpSchema.AttrType.FLOAT,
            description="Logit capping value.",
            required=False,
        ),
        OpSchema.Attribute(
            name="layout",
            type=OpSchema.AttrType.STRING,
            description="Memory layout: 'bnsd' or 'bsnd'.",
            required=False,
        ),
    ],
)


def onnx_torch_attention_op(
    query: ir.Tensor,
    key: ir.Tensor,
    value: ir.Tensor,
    attn_mask: Optional[ir.Tensor] = None,
    sinks: Optional[ir.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    layout: str = "bnsd",
):
    """
    ONNX custom op translation function for torch_attention.

    This function creates a custom ONNX op node in the auto_deploy domain.
    The actual implementation will need to be provided in the inference engine
    (e.g., ONNX Runtime or TensorRT) that loads this ONNX model.

    Note: This is a translation function for torch.onnx.export's custom_translation_table,
    so it should NOT have @script() decorator.
    """
    # Call the custom op from the auto_deploy domain
    return auto_deploy_opset.torch_attention(
        query,
        key,
        value,
        attn_mask,
        sinks,
        dropout_p=dropout_p,
        is_causal=1 if is_causal else 0,
        scale=scale,
        sliding_window=sliding_window,
        logit_cap=logit_cap,
        layout=layout,
    )


# Register the schema
onnx.defs.register_schema(torch_attention_schema)

# ============================================================================


@TransformRegistry.register("export_to_onnx")
class ExportToONNX(BaseTransform):
    config: ExportToONNXConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ExportToONNXConfig

    def _export_onnx_model(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> None:
        args = []
        kwargs = {}
        placeholders = gm.graph.find_nodes(op="placeholder")
        for ph in placeholders:
            kwargs[ph.name] = ph.meta["val"]
        args = tuple(args)

        ad_logger.info("Placeholders args:")
        for i, e in enumerate(args):
            ad_logger.info(f"  {i}: {placeholders[i].name:20} {e}")

        ad_logger.info("Placeholders kwargs:")
        for k, v in kwargs.items():
            ad_logger.info(f"  {k}: {v}")

        # Build output path
        output_path = self.config.output_dir / self.config.output_name

        # Build dynamic_shapes for dynamo export
        # For dynamo, we need to specify dynamic dimensions for each input tensor
        dynamic_shapes = {}
        dynamic_shapes["input_ids"] = {
            0: Dim("batch_size", min=0, max=cm.info.max_batch_size),
            1: Dim("seq_len", min=0, max=cm.info.max_seq_len),
        }
        # Add dynamic shapes for context_lengths and rope_rotary_cos_sin
        dynamic_shapes["context_lengths"] = {
            0: Dim("batch_size", min=0, max=cm.info.max_batch_size)
        }
        dynamic_shapes["rope_rotary_cos_sin"] = {
            0: Dim("rope_batch_size", min=1, max=16),
            1: Dim("max_position_embeddings", min=1, max=4096),
        }
        dynamic_shapes["kvcache_start_index"] = {
            0: Dim("kv_cache_start_batch_size", min=0, max=cm.info.max_batch_size)
        }
        # Add dynamic shapes for past_key_values
        num_layers = len(
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.auto_deploy.AttentionPlugin.default
            )
        )
        for i in range(num_layers):
            dynamic_shapes[f"past_key_values_{i}"] = {
                0: Dim("batch_size", min=0, max=cm.info.max_batch_size),
                3: Dim("past_len", min=1, max=4096),
            }
        dynamic_shapes["last_token_ids"] = {
            0: Dim("batch_size", min=0, max=cm.info.max_batch_size),
            1: Dim("num_selected_tokens", min=1, max=cm.info.max_seq_len),
        }

        ad_logger.info(f"Exporting GraphModule to ONNX with dynamo: {output_path}")

        # Register ONNX function for custom ops before export
        ad_logger.info("Registering ONNX custom op handlers...")

        # Create custom translation table for ONNX export
        # Map the torch custom op to the onnxscript function
        # Use the global onnx_prepare_metadata_op directly
        custom_translation_table = {
            # Before fuse rope attention
            # NOTE (yoco): This 2 ops will be fused into the AttentionPlugin operation
            # in the fuse_rope_attention transform.
            # However, when TensorRT-LLM changed, the fusion might not be applied.
            # And for debug purpose we might want to export the .onnx to check the graph.
            # So let's just keep them here.
            torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default: custom_rope_op,
            torch.ops.auto_deploy.torch_attention.default: onnx_torch_attention_op,
            # Before and after fuse rope attention
            torch.ops.auto_deploy.torch_linear_simple.default: custom_simple_linear_op,
            # After fuse rope attention
            torch.ops.auto_deploy.AttentionPlugin.default: onnx_rope_attention_op,
            torch.ops.auto_deploy.GatherND.default: custom_gather_nd_op,
        }

        # Prepare output names
        output_names = []
        output_node = gm.graph.find_nodes(op="output")[0]
        outputs = output_node.args[0]
        for output in outputs:
            output_names.append(output.name)
        output_names[0] = "logits"

        # Export the graph module to ONNX using dynamo (more advanced tracer)
        torch.onnx.export(
            gm,
            tuple(args),
            output_path,
            opset_version=20,
            kwargs=kwargs,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            report=False,
            output_names=output_names,
            custom_translation_table=custom_translation_table,
        )
        # export_output.save(output_path)

        ad_logger.info(f"Successfully exported ONNX model to {output_path}")
        return True

    def _export_config_json(self, gm: GraphModule, output_dir: str, is_eagle_base: bool) -> None:
        model_type = "eagle3_base" if is_eagle_base else "llm"
        model_config = export_llm_config(gm.config, model_type)
        # Add reduced_vocab_size to config if vocabulary reduction is used
        reduced_vocab_size = None  # TODO: Implement this
        if reduced_vocab_size is not None:
            model_config["reduced_vocab_size"] = reduced_vocab_size
            ad_logger.info(f"Added reduced_vocab_size={reduced_vocab_size} to config")

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        ad_logger.info(f"Model configuration saved to {config_path}")

    def _export_json_files(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> None:
        # Handle config.json
        is_eagle_base = self.config.is_eagle_base
        output_dir = self.config.output_dir
        self._export_config_json(gm, output_dir, is_eagle_base)

        # Handle: added_tokens.json, special_tokens_map.json, tokenizer.json,
        # tokenizer_config.json, vocab.json
        tokenizer = factory.init_tokenizer()
        tokenizer.save_pretrained(output_dir)

        # Handle processed_chat_template.json
        model_dir = factory.model
        process_chat_template(model_dir, output_dir)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Create output directory
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Export JSON files
        self._export_json_files(gm, cm, factory, shared_config)

        # Export the ONNX model
        success = self._export_onnx_model(gm, cm, factory, shared_config)

        # Return the original graph module unchanged and transform info
        # This transform doesn't modify the graph, it only exports it
        info = TransformInfo(
            skipped=not success,
            num_matches=1 if success else 0,
            is_clean=True,  # We don't modify the graph
            has_valid_shapes=True,  # We don't affect shapes
        )

        return gm, info
