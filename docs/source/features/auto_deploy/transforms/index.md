# AutoDeploy Transforms

AutoDeploy optimizes models through a pipeline of graph transforms applied in a fixed stage order. Each transform performs a specific optimization -- from pattern matching and operator fusion to sharding, quantization, and compilation.

## Transform Pipeline

The inference optimizer applies transforms in the following stages:

1. **Factory** -- Build the model architecture on a meta device.
1. **Export** -- Export the PyTorch model to an FX GraphModule via `torch.export`.
1. **Post-Export** -- Low-level cleanups (remove no-ops, fix constraints).
1. **Pattern Matcher** -- High-level pattern matching to standardize and fuse operations.
1. **Sharding** -- Auto-sharding for multi-GPU parallelism.
1. **Weight Load** -- Load model weights from checkpoints.
1. **Post-Load Fusion** -- Performance optimizations (KV-cache, quantization, kernel fusion).
1. **Cache Init** -- Initialize cached attention and KV cache structures.
1. **Visualize** -- Graph visualization for debugging.
1. **Compile** -- Graph compilation with backends like `torch.compile` + CUDA graphs.

## Configuring Transforms

Transforms are configured via YAML in the `transforms` section of the AutoDeploy config. Each transform has a name (its registry key) and optional config fields:

```yaml
transforms:
  fuse_swiglu:
    enabled: true
  insert_cached_mla_attention:
    backend: trtllm_mla
  compile_model:
    piecewise_enabled: true
    piecewise_num_tokens: [256, 512, 1024, 2048, 4096]
```

All transforms share these base config fields:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable or disable the transform. |
| `skip_on_error` | `false` | Gracefully skip if an error occurs instead of failing. |
| `run_per_gm` | `true` | Run per sub-graph module or on the whole module. |

See [Expert Configurations](../advanced/expert_configurations.md) for more details on YAML configuration.

## Full Reference

```{toctree}
:maxdepth: 1

reference
```
