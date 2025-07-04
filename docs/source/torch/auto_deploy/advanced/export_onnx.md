# Export ONNX for DriveOS LLM

AutoDeploy provides a mode to export PyTorch/HuggingFace models to ONNX format specifically designed for DriveOS LLM deployment. This mode performs graph transformations to fuse RoPE (Rotary Position Embedding) and attention operations into a single `AttentionPlugin` operation, then exports the optimized graph to ONNX.

## Overview

The `export_driveos_llm_onnx` mode differs from the standard AutoDeploy workflow in two key ways:

1. **Operation Fusion**: Fuses `torch_rope_with_explicit_cos_sin` and `torch_cached_attention_with_cache` into a single `AttentionPlugin` operation
1. **ONNX Export**: Outputs an ONNX model file instead of a TensorRT Engine

## Quick Start

Use the `onnx_export_llm.py` script to export a model:

```bash
cd examples/auto_deploy
python onnx_export_llm.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

This will export the model to ONNX format in the current directory.

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model name or path to a local checkpoint |
| `--max_seq_len` | int | `4` | Maximum sequence length for the model |
| `--max_batch_size` | int | `13` | Maximum batch size for the model |
| `--device` | str | `cpu` | Device to use for export (`cpu` or `cuda`) |
| `--output_dir` | str | `.` | Directory to save the exported ONNX model |
| `--output_name` | str | `model.onnx` | Name of the exported ONNX model file |

## Examples

### Basic Export

Export a TinyLlama model with default settings:

```bash
python onnx_export_llm.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Custom Output Location

Export to a specific directory with a custom filename:

```bash
python onnx_export_llm.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output_dir "./exported_models" \
    --output_name "tinyllama.onnx"
```

### Configure Sequence and Batch Size

Export with custom maximum sequence length and batch size:

```bash
python onnx_export_llm.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --max_seq_len 2048 \
    --max_batch_size 8 \
    --output_dir "./llama2_onnx"
```

## Output Files

The export process generates the following files in the output directory:

| File | Description |
|------|-------------|
| `model.onnx` | The exported ONNX model with fused attention operations |
| `config.json` | Model configuration (architecture, hidden size, etc.) |
| `tokenizer.json` | Tokenizer vocabulary and configuration |
| `tokenizer_config.json` | Tokenizer settings |
| `special_tokens_map.json` | Special token mappings |
| `processed_chat_template.json` | Processed chat template for inference |

## Programmatic Usage

You can also use the ONNX export functionality programmatically:

```python
from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig

# Create AutoDeploy config with export_driveos_llm_onnx mode
ad_config = AutoDeployConfig(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    mode="export_driveos_llm_onnx",
    max_batch_size=8,
    max_seq_len=512,
    device="cpu",
)

# Configure attention backend
ad_config.attn_backend = "torch"

# Optionally customize output location
ad_config.transforms["export_to_onnx"]["output_dir"] = "./my_output"
ad_config.transforms["export_to_onnx"]["output_name"] = "my_model.onnx"

# Run the export
llm = LLM(**ad_config.to_llm_kwargs())
```

## Notes

- **Device Selection**: Using `cpu` for the `--device` option is recommended to reduce GPU memory footprint during export.
- **Dynamic Shapes**: The exported ONNX model supports dynamic batch size and sequence length within the specified maximum bounds.
- **Custom Operations**: The exported ONNX model contains custom operations (e.g., `AttentionPlugin`) in the `trt` domain that require corresponding implementations in the target inference runtime.
