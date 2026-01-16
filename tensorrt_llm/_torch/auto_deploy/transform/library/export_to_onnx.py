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

import torch
from onnxscript import ir, opset20
from pydantic import Field
from torch.export import Dim
from torch.fx import GraphModule

from ...models import hf
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
from . import _onnx_schemas
from ._chat_template import process_chat_template
from ._config_export import export_llm_config


class ExportToONNXConfig(TransformConfig):
    """Configuration for the export to ONNX transform."""

    output_dir: Path = Field(
        description="The directory to save the exported ONNX model.",
    )
    is_eagle_base: bool = Field(
        description="Whether the model is an Eagle base model.",
        default=False,
    )


# ============================================================================
# Custom translation functions for ONNX export
# ============================================================================


def _translate_rope_op(
    q: ir.Tensor, k: ir.Tensor, cos: ir.Tensor, sin: ir.Tensor, unsqueeze_dim: int
):
    return _onnx_schemas.auto_deploy_opset.rope_with_explicit_cos_sin(
        q, k, cos, sin, unsqueeze_dim=unsqueeze_dim
    )


def _translate_simple_linear_op(input: ir.Tensor, weight: ir.Tensor, bias: Optional[ir.Tensor]):
    weight = opset20.Transpose(weight, perm=[1, 0])
    if bias is None:
        return opset20.MatMul(input, weight)
    return opset20.Add(opset20.MatMul(input, weight), bias)


def _translate_gather_nd_op(data: ir.Tensor, indices: ir.Tensor, batch_dims: int):
    return opset20.GatherND(data, indices, batch_dims=batch_dims)


def _translate_rope_attention_op(
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

    This function creates a custom ONNX op node in the trt domain.
    The actual implementation will need to be provided in the inference engine
    (e.g., ONNX Runtime or TensorRT) that loads this ONNX model.

    Note: This is a translation function for torch.onnx.export's custom_translation_table,
    so it should NOT have @script() decorator.
    """
    # Call the custom op from the trt domain
    return _onnx_schemas.trt_opset.AttentionPlugin(
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


def _translate_torch_attention_op(
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
    return _onnx_schemas.auto_deploy_opset.torch_attention(
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


@TransformRegistry.register("export_to_onnx")
class ExportToONNX(BaseTransform):
    """Transform that exports a PyTorch GraphModule to ONNX format for deployment.

    This transform is responsible for:
    1. Exporting the model graph to ONNX format with dynamic shapes support
    2. Generating configuration files (config.json) for the exported model
    3. Saving tokenizer files (tokenizer.json, vocab.json, etc.)
    4. Processing and exporting chat templates

    The exported ONNX model includes custom ops from the auto_deploy. These custom ops include:
    - torch_onnx_attention_plugin: Fused RoPE + Attention for efficient inference(exported as EdgeLLM's custom op)
    - torch_onnx_gather_nd: N-dimensional gather operation (exported as onnxscript.opset20.GatherND)

    Note:
        This transform does NOT modify the input graph. It only exports the graph
        to external files and returns the original graph unchanged.

    Attributes:
        config: ExportToONNXConfig containing output directory, and is_eagle_base flag.
    """

    config: ExportToONNXConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        """Return the configuration class for this transform."""
        return ExportToONNXConfig

    def _export_onnx_model(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        _factory: ModelFactory,
        _shared_config: SharedConfig,
    ) -> bool:
        """Export the GraphModule to ONNX format using torch.onnx.export with dynamo.

        This method handles the core ONNX export logic:
        1. Extracts input placeholders and their metadata from the graph
        2. Configures dynamic shapes for batch size, sequence length, and KV cache
        3. Sets up custom translation table for auto_deploy custom ops
        4. Exports the model using torch.onnx.export with dynamo=True

        Args:
            gm: The PyTorch FX GraphModule to export.
            cm: CachedSequenceInterface containing max batch size and sequence length info.
            _factory: ModelFactory instance (unused in this method).
            _shared_config: SharedConfig instance (unused in this method).

        Returns:
            bool: True if export was successful.
        """
        # Extract input placeholders from graph to build kwargs for export
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
        output_path = self.config.output_dir / "model.onnx"

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
                op="call_function", target=torch.ops.auto_deploy.torch_onnx_attention_plugin.default
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

        # Create custom translation table for ONNX export
        # Map torch custom ops to their corresponding onnxscript translation functions
        custom_translation_table = {
            # Before fuse rope attention
            # NOTE (yoco): This 2 ops will be fused into the AttentionPlugin operation
            # in the fuse_rope_attention transform.
            # However, when TensorRT-LLM changed, the fusion might not be applied.
            # And for debug purpose we might want to export the .onnx to check the graph.
            # So let's just keep them here.
            torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default: _translate_rope_op,
            torch.ops.auto_deploy.torch_attention.default: _translate_torch_attention_op,
            # Before and after fuse rope attention
            torch.ops.auto_deploy.torch_linear_simple.default: _translate_simple_linear_op,
            # After fuse rope attention
            torch.ops.auto_deploy.torch_onnx_attention_plugin.default: _translate_rope_attention_op,
            torch.ops.auto_deploy.torch_onnx_gather_nd.default: _translate_gather_nd_op,
        }

        # Prepare output names
        output_names = []
        output_node = gm.graph.find_nodes(op="output")[0]
        outputs = output_node.args[0]
        for output in outputs:
            output_names.append(output.name)
        output_names[0] = "logits"

        # Register ONNX custom ops
        _onnx_schemas.register_onnx_schemas()

        # Export the graph module to ONNX using dynamo (more advanced tracer)
        ad_logger.info(f"Exporting GraphModule to ONNX with dynamo: {output_path}")
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

        ad_logger.info(f"Successfully exported ONNX model to {output_path}")
        return True

    def _export_config_json(
        self, factory: ModelFactory, output_dir: str, is_eagle_base: bool
    ) -> None:
        """Export model configuration to config.json.

        Generates a configuration file containing model architecture parameters
        such as hidden size, number of layers, attention heads, etc.

        Args:
            factory: The ModelFactory containing model configuration.
            output_dir: Directory path where config.json will be saved.
            is_eagle_base: If True, exports as "eagle3_base" model type,
                otherwise exports as "llm" model type.
        """
        model_type = "eagle3_base" if is_eagle_base else "llm"
        assert isinstance(factory, hf.AutoModelFactory)
        model_config, _ = factory._get_model_config()
        model_config = export_llm_config(model_config, model_type)
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
        _cm: CachedSequenceInterface,
        factory: ModelFactory,
        _shared_config: SharedConfig,
    ) -> None:
        """Export all JSON configuration and tokenizer files required for deployment.

        This method orchestrates the export of:
        1. config.json - Model architecture configuration
        2. Tokenizer files - added_tokens.json, special_tokens_map.json,
           tokenizer.json, tokenizer_config.json, vocab.json
        3. processed_chat_template.json - Processed chat template for inference

        Args:
            gm: The GraphModule containing model configuration.
            _cm: CachedSequenceInterface (unused, kept for interface consistency).
            factory: ModelFactory used to initialize tokenizer and get model directory.
            _shared_config: SharedConfig (unused, kept for interface consistency).
        """
        # Export model configuration (architecture params, layer counts, etc.)
        is_eagle_base = self.config.is_eagle_base
        output_dir = self.config.output_dir
        self._export_config_json(factory, output_dir, is_eagle_base)

        # Export tokenizer files for text processing during inference
        # Includes: added_tokens.json, special_tokens_map.json, tokenizer.json,
        # tokenizer_config.json, vocab.json
        tokenizer = factory.init_tokenizer()
        tokenizer.save_pretrained(output_dir)

        # Export processed chat template for conversational inference
        model_dir = factory.model
        process_chat_template(model_dir, output_dir)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Apply the ONNX export transform to the graph module.

        This is the main entry point that orchestrates the export process:
        1. Creates the output directory if it doesn't exist
        2. Exports all JSON configuration files (config, tokenizer, chat template)
        3. Exports the ONNX model with dynamic shapes and custom ops

        Note:
            Unlike other transforms, this does NOT modify the graph.
            It only performs side effects (writing files to disk).

        Args:
            gm: The PyTorch FX GraphModule to export.
            cm: CachedSequenceInterface containing runtime constraints.
            factory: ModelFactory for tokenizer initialization.
            shared_config: SharedConfig for transform coordination.

        Returns:
            Tuple containing:
            - gm: The original GraphModule (unchanged)
            - info: TransformInfo with export status (is_clean=True since graph is unmodified)
        """
        # Ensure output directory exists before writing any files
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Export all auxiliary JSON files (config, tokenizer, chat template)
        self._export_json_files(gm, cm, factory, shared_config)

        # Step 2: Export the ONNX model with dynamic shapes
        success = self._export_onnx_model(gm, cm, factory, shared_config)

        # Return original graph unchanged with export status info
        # This transform is "clean" because it doesn't modify the graph structure
        info = TransformInfo(
            skipped=not success,
            num_matches=1 if success else 0,
            is_clean=True,  # Graph is not modified
            has_valid_shapes=True,  # Shape validity is preserved
        )

        return gm, info
