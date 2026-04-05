# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from typing import Optional, Sequence, Tuple, Type

import onnx
import torch
from onnx import TensorProto, helper
from onnxscript import ir, opset21
from pydantic import Field
from torch.export import Dim
from torch.fx import GraphModule

from ...models import hf
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import sync_weight_meta_dtype
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


def _convert_nvfp4_weight_initializers_to_float4e2m1(model_proto) -> int:
    """Convert packed uint8 FP4 weight initializers to float4e2m1 [N, K] in-place.

    Finds initializers that are used as the 'data' input of DequantizeLinear with
    block_size=16 (NVFP4 path). Converts them from uint8 [N, K_packed] to
    float4e2m1 [N, K] (same raw bytes; ONNX stores 2 FP4 values per byte).
    Returns the number of initializers converted.
    """
    graph = model_proto.graph
    data_inputs_to_convert = set()
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        block_size = None
        for attr in node.attribute:
            if attr.name == "block_size":
                block_size = attr.i
                break
        if block_size != 16:
            continue
        if len(node.input) >= 1:
            data_inputs_to_convert.add(node.input[0])

    converted = 0
    for init in graph.initializer:
        if init.name not in data_inputs_to_convert:
            continue
        if init.data_type != TensorProto.UINT8:
            continue
        if len(init.dims) < 2:
            continue
        k_packed = init.dims[-1]
        init.data_type = TensorProto.FLOAT4E2M1
        new_dims = list(init.dims)[:-1] + [k_packed * 2]
        del init.dims[:]
        init.dims.extend(new_dims)
        converted += 1
    return converted


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
    """Translate RoPE operation to ONNX custom op."""
    return _onnx_schemas.auto_deploy_opset.rope_with_explicit_cos_sin(
        q, k, cos, sin, unsqueeze_dim=unsqueeze_dim
    )


def _translate_simple_linear_op(input: ir.Tensor, weight: ir.Tensor, bias: Optional[ir.Tensor]):
    """Translate linear operation to ONNX MatMul + optional Add."""
    weight = opset21.Transpose(weight, perm=[1, 0])
    if bias is None:
        return opset21.MatMul(input, weight)
    return opset21.Add(opset21.MatMul(input, weight), bias)


def _translate_gather_nd_op(data: ir.Tensor, indices: ir.Tensor, batch_dims: int):
    """Translate GatherND operation to ONNX GatherND op."""
    return opset21.GatherND(data, indices, batch_dims=batch_dims)


def _translate_rope_attention_op(
    q: ir.Tensor,
    k: ir.Tensor,
    v: ir.Tensor,
    past_key_values: ir.Tensor,
    context_lengths: ir.Tensor,
    rope_rotary_cos_sin: ir.Tensor,
    kvcache_start_index: ir.Tensor,
    enable_tree_attention: int,
    head_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    enable_fp8_kv_cache: int = 0,
    sliding_window_size: int = -1,
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
        q,
        k,
        v,
        past_key_values,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        enable_tree_attention=enable_tree_attention,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        enable_fp8_kv_cache=enable_fp8_kv_cache,
        sliding_window_size=sliding_window_size,
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


def _get_fp8e4m3fn_zero_point(zp: Sequence[ir.Tensor]) -> ir.Tensor:
    """Return zero point tensor, creating FP8 zero constant if zp is empty.

    Args:
        zp: Sequence of zero point tensors. If non-empty, returns zp[0].
            If empty or None, creates and returns a FP8 zero constant.

    Returns:
        Zero point tensor for QuantizeLinear/DequantizeLinear ops.

    Note:
        Currently, the zero point for FP8 fake quantized is not supported.
    """
    if zp is not None and len(zp) > 0:
        raise ValueError("Currently, the zero point for FP8 fake quantized is not supported")

    # Create FP8 zero constant when zp is empty/None
    fp8_const_proto = helper.make_tensor(
        name="fp8_zero_point",
        data_type=TensorProto.FLOAT8E4M3FN,
        dims=[1],
        vals=[0.0],
    )
    return opset21.Constant(value=fp8_const_proto)


def _translate_fake_quant_fp8_linear_op(
    input: ir.Tensor,
    weight_quantized: ir.Tensor,
    bias: Optional[ir.Tensor],
    input_scale: Sequence[ir.Tensor],
    weight_scale: Sequence[ir.Tensor],
    input_zp: Sequence[ir.Tensor],
    weight_zp: Sequence[ir.Tensor],
):
    """
    ONNX translation for FP8 fake quantized linear operation.

    Converts torch_fake_quant_fp8_linear to standard ONNX ops:
    - Input: QuantizeLinear -> DequantizeLinear (fake quantization)
    - Weight: DequantizeLinear (weight is already FP8)
    - Linear: Transpose + MatMul + Add(bias)

    Note 1: This is a translation function for torch.onnx.export's custom_translation_table,
    so it should NOT have @script() decorator.

    Note 2: The result quanted type is defined by input zp, and it should be float8_e4m3fn.
    However, onnxscript does not support torch.float8_e4m3fn type, so we need to use the
    onnxscript's Constant op to create the zero point.
    """

    if len(input_scale) != 1 or len(weight_scale) != 1:
        raise ValueError(
            f"FP8 fake quantized linear requires exactly one scale per tensor, "
            f"got input_scale={len(input_scale)}, weight_scale={len(weight_scale)}"
        )
    s_in = input_scale[0]
    s_w = weight_scale[0]
    s_in = opset21.Cast(s_in, to=TensorProto.FLOAT)

    # Get zero points (creates FP8 zero constant if empty)
    input_zp = _get_fp8e4m3fn_zero_point(input_zp)
    weight_zp = _get_fp8e4m3fn_zero_point(weight_zp)

    # Input fake quantization: quantize to FP8, then dequantize back
    input_q = opset21.QuantizeLinear(input, s_in, y_zero_point=input_zp)
    input_dq = opset21.DequantizeLinear(input_q, s_in, x_zero_point=input_zp)
    input_dq = opset21.Cast(input_dq, to=TensorProto.FLOAT16)

    # Weight dequantization (weight is already FP8)
    weight_dq = opset21.DequantizeLinear(weight_quantized, s_w, x_zero_point=weight_zp)
    weight_dq = opset21.Cast(weight_dq, to=TensorProto.FLOAT16)

    # Linear: Transpose weight [N, K] -> [K, N], then MatMul
    weight_t = opset21.Transpose(weight_dq, perm=[1, 0])
    out = opset21.MatMul(input_dq, weight_t)

    if bias is not None:
        out = opset21.Add(out, bias)

    return out


def _translate_fake_quant_nvfp4_linear_op(
    input: ir.Tensor,
    weight_quantized: ir.Tensor,
    bias: Optional[ir.Tensor],
    input_scale: Sequence[ir.Tensor],
    weight_scale: Sequence[ir.Tensor],
    input_zp: Sequence[ir.Tensor],
    weight_zp: Sequence[ir.Tensor],
):
    """
    ONNX translation for NVFP4 fake quantized linear operation.

    Expands torch_fake_quant_nvfp4_linear to the target ONNX subgraph:
    - Activation path: Reciprocal(x_scale) -> TRT_FP4DynamicQuantize -> two-stage
      DequantizeLinear (first dequant scale, second dequant data) -> Cast to float16.
    - Weight path: Div(w_alpha, ixscale) -> DequantizeLinear(w_scale, wscale_2) ->
      DequantizeLinear(w, combined_scale, axis=-1, block_size=16) -> Cast -> Transpose.
    - Linear: MatMul(cx, weight_t) + optional Add(bias).

    Scale semantics: FP4 uses per-block scale (scale itself may be quantized); see
    .ai/knowledge/fp4-two-stage-dequant-scale-compression.md. Arguments are lists:
    input_scale[0]=x_scale, weight_scale[0]=w_scale, weight_scale[1]=w_alpha.
    """
    if len(input_scale) != 1 or len(weight_scale) != 2:
        raise ValueError(
            f"NVFP4 fake quantized linear requires input_scale length 1 and weight_scale length 2, "
            f"got input_scale={len(input_scale)}, weight_scale={len(weight_scale)}"
        )
    x_scale = input_scale[0]
    w_scale = weight_scale[0]
    w_alpha = weight_scale[1]

    # Activation path: ixscale = 1 / x_scale
    ixscale = opset21.Reciprocal(x_scale)

    # TRT_FP4DynamicQuantize(x, ixscale) -> (packed, per_block_scale)
    fp4dq_packed, fp4dq_block_scale = _onnx_schemas.trt_opset.TRT_FP4DynamicQuantize(
        input, ixscale, axis=-1, block_size=16, scale_type=TensorProto.FLOAT8E4M3FN
    )

    # Two-stage dequant (activation): xdq1 then xdq2 per user graph
    xdq1 = opset21.DequantizeLinear(fp4dq_block_scale, ixscale)
    xdq2 = opset21.DequantizeLinear(fp4dq_packed, xdq1, axis=-1, block_size=16)
    cx = opset21.Cast(xdq2, to=TensorProto.FLOAT16)

    # Weight path: wscale_2 = w_alpha / ixscale, then two-stage dequant
    wscale_2 = opset21.Div(w_alpha, ixscale)
    wdq1 = opset21.DequantizeLinear(w_scale, wscale_2)
    wdq2 = opset21.DequantizeLinear(weight_quantized, wdq1, axis=-1, block_size=16)
    cw = opset21.Cast(wdq2, to=TensorProto.FLOAT16)
    weight_t = opset21.Transpose(cw, perm=[1, 0])

    out = opset21.MatMul(cx, weight_t)
    if bias is not None:
        out = opset21.Add(out, bias)
    return out


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
    - torch_onnx_gather_nd: N-dimensional gather operation (exported as onnxscript.opset21.GatherND)

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
        #
        # IMPORTANT: The order of keys in dynamic_shapes dict must match the order of
        # placeholders in graph.inputs. This is because PyTorch's ONNX exporter internally
        # flattens the dynamic_shapes dict to a list using _flatten_dynamic_shapes_to_axes(),
        # and then zips it with graph.inputs by index in create_rename_mapping().
        # If the order doesn't match, it will try to access wrong dimensions and cause
        # IndexError (e.g., accessing shape[1] on a 1D tensor).
        #
        # Python 3.7+ guarantees dict preserves insertion order, so we must add keys
        # in the same order as placeholders appear in the graph.
        # Reusable Dim factories
        batch_dim = Dim("batch_size", min=0, max=cm.info.max_batch_size)
        seq_dim = Dim("seq_len", min=0, max=cm.info.max_seq_len)

        # Dynamic shape specs: pattern -> shape_spec
        # Pattern can be exact string or prefix with "*" suffix
        shape_specs = {
            "context_lengths": {0: batch_dim},
            "kvcache_start_index": {
                0: Dim("kv_cache_start_batch_size", min=0, max=cm.info.max_batch_size)
            },
            "rope_rotary_cos_sin": {
                0: Dim("rope_batch_size", min=1, max=16),
                1: Dim("max_position_embeddings", min=1, max=4096),
            },
            "past_key_values_*": {
                0: batch_dim,
                3: Dim("past_len", min=1, max=4096),
            },
            "last_token_ids": {
                0: batch_dim,
                1: Dim("num_selected_tokens", min=1, max=cm.info.max_seq_len),
            },
            "inputs_embeds": {0: batch_dim, 1: seq_dim},
        }

        def match_shape_spec(name: str) -> Optional[dict]:
            """Match placeholder name against shape specs (exact or prefix)."""
            if name in shape_specs:
                return shape_specs[name]
            for pattern, spec in shape_specs.items():
                if pattern.endswith("*") and name.startswith(pattern[:-1]):
                    return spec
            return None

        dynamic_shapes = {}
        for ph in placeholders:
            if spec := match_shape_spec(ph.name):
                dynamic_shapes[ph.name] = spec

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
            # FP8 quantized linear
            torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default: _translate_fake_quant_fp8_linear_op,
            # NVFP4 quantized linear (single-op expansion to ONNX subgraph)
            torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default: _translate_fake_quant_nvfp4_linear_op,
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

        # Export to ONNX IR in memory (f=None returns ONNXProgram), then optionally
        ad_logger.info(f"Exporting GraphModule to ONNX with dynamo: {output_path}")
        onnx_program = torch.onnx.export(
            gm,
            tuple(args),
            None,  # f=None: get ONNXProgram instead of writing file
            opset_version=21,
            kwargs=kwargs,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            report=False,
            output_names=output_names,
            custom_translation_table=custom_translation_table,
        )

        # Keep a single reference to model_proto: ONNXProgram.model_proto may return a new
        # copy on each access, so mutating the passed object would not be reflected in save().
        # We mutate this reference and save it explicitly with onnx.save_model.
        model_proto = onnx_program.model_proto

        # convert NVFP4 weight initializers to float4e2m1, and save.
        n_converted = _convert_nvfp4_weight_initializers_to_float4e2m1(model_proto)
        if n_converted:
            ad_logger.info(f"Converted {n_converted} NVFP4 weight initializers to float4e2m1")

        # location must be relative to the model file; when None, onnx uses uuid.uuid1().data.
        # Use "<model_basename>.data" so external file is e.g. model.onnx.data next to model.onnx.
        external_location = output_path.name + ".data"
        onnx.save_model(
            model_proto,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_location,
            size_threshold=1024,
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

        # Step 1.5: Sync .meta["val"] dtype with actual state_dict dtype for weight nodes
        # This ensures ONNX export sees the correct dtype (e.g., FP8) instead of the
        # dtype from tracing time (e.g., FP16)
        sync_weight_meta_dtype(gm)

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
