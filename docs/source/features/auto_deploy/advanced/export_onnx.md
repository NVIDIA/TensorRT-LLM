# Export ONNX for EdgeLLM

AutoDeploy provides a mode to export PyTorch/HuggingFace models to ONNX format specifically designed for EdgeLLM deployment. This mode performs graph transformations to fuse RoPE (Rotary Position Embedding) and attention operations into a single `AttentionPlugin` operation, then exports the optimized graph to ONNX.

## Overview

The `export_edgellm_onnx` mode differs from the standard AutoDeploy workflow in several key ways:

1. **Operation Fusion**: Fuses `torch_rope_with_explicit_cos_sin` and `torch_cached_attention_with_cache` into a single `AttentionPlugin` operation
1. **Multimodal Input Support**: Rewrites the model to accept `inputs_embeds` instead of `input_ids`, enabling multimodal model support
1. **Embedding Export**: Exports the embedding table as `embedding.safetensors` for runtime embedding lookup
1. **ONNX Export**: Outputs an ONNX model file instead of a TensorRT Engine

### Multimodal Input Changes

To support multimodal models (e.g., vision-language models), the exported ONNX model now accepts `inputs_embeds` (float16 tensor of shape `[batch_size, seq_len, hidden_size]`) instead of `input_ids` (int32 tensor of shape `[batch_size, seq_len]`). This allows EdgeLLM runtime to:

- Perform embedding lookup for text tokens using the exported `embedding.safetensors`
- Fuse multimodal embeddings (from vision/audio encoders) with text embeddings
- Pass the combined embeddings directly to the TensorRT engine

The embedding table is exported separately so that the runtime can handle both text-only and multimodal inputs efficiently.

## Quick Start

Use the `onnx_export_llm.py` script to export a model:

```bash
cd examples/auto_deploy
python onnx_export_llm.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

This will export the model to ONNX format in the current directory.

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | Required | HuggingFace model name or path to a local checkpoint |
| `--device` | str | `cpu` | Device to use for export (`cpu` or `cuda`) |
| `--output_dir` | str | `.` | Directory to save the exported ONNX model |

## Examples

### Basic Export

Export a DeepSeek model with default settings:

```bash
python onnx_export_llm.py --model "Qwen/Qwen2.5-0.5B-Instruct"
```

### Custom Output Location

Export to a specific directory with a custom filename:

```bash
python onnx_export_llm.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --output_dir "./exported_models"
```

## Output Files

The export process generates the following files in the output directory:

| File | Description |
|------|-------------|
| `model.onnx` | The exported ONNX model with fused attention operations |
| `embedding.safetensors` | Embedding table weights (for multimodal input support) |
| `config.json` | Model configuration (architecture, hidden size, etc.) |
| `tokenizer.json` | Tokenizer vocabulary and configuration |
| `tokenizer_config.json` | Tokenizer settings |
| `special_tokens_map.json` | Special token mappings |
| `processed_chat_template.json` | Processed chat template for inference |

## Programmatic Usage

You can also use the ONNX export functionality programmatically:

```python
from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig

# Create AutoDeploy config with export_edgellm_onnx mode
ad_config = AutoDeployConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    mode="export_edgellm_onnx",
    max_batch_size=8,
    max_seq_len=512,
    device="cpu",
)

# Configure attention backend
ad_config.attn_backend = "torch"

# Optionally customize output location
ad_config.transforms["rewrite_embedding_to_inputs_embeds"]["output_dir"] = "./my_output"
ad_config.transforms["export_to_onnx"]["output_dir"] = "./my_output"

# Run the export
LLM(**ad_config.to_llm_kwargs())
```

## Notes

- **Device Selection**: Using `cpu` for the `--device` option is recommended to reduce GPU memory footprint during export.
- **Custom Operations**: The exported ONNX model contains custom operations (e.g., `AttentionPlugin`) in the `trt` domain that require corresponding implementations in the target inference runtime.
