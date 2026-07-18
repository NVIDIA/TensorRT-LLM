# Checkpoint Loading

The PyTorch backend provides a flexible and extensible infrastructure for loading model checkpoints from different formats, such as HuggingFace (HF). This system allows you to load models from various sources (e.g., HuggingFace or custom formats) by implementing the required components, such as the checkpoint’s weight loader, mapper, and configuration parser.

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Built-in Checkpoint Formats](#built-in-checkpoint-formats)
4. [Experimental Weight-Load Plans](#experimental-weight-load-plans)
5. [Using Checkpoint Loaders](#using-checkpoint-loaders)
6. [Creating Custom Checkpoint Loaders](#creating-custom-checkpoint-loaders)

## Overview

The checkpoint loading design is built around a plugin-like architecture that is separated into four distinct components:

- **Checkpoint Loaders**: Orchestrate the loading process for specific formats
- **Config Loaders**: Handle model configuration parsing and validation
- **Weight Loaders**: Manage the actual loading of model weights from storage into memory
- **Weight Mappers**: Map and transform loaded weights to TensorRT LLM model's definition

This modular design allows for easy extension to support new checkpoint formats while maintaining backward compatibility and performance optimizations. By separating the checkpoint loading components into four different subcomponents, any user can employ any relevant previous work while also introducing their own custom checkpoint-specific components.

If one wishes to support a new checkpoint format, they must implement all four components.
Likewise, if the format shares some components with an already supported framework (e.g., HF), only the custom-specific components need to be implemented.

## Core Components

### BaseCheckpointLoader

The `BaseCheckpointLoader` is the central base interface for all checkpoint loading required operators. It provides a unified API regardless of the underlying checkpoint format. This interface is responsible for holding and exposing all objects required for the loading and parsing process.

**Key Methods:**
- `load_config(checkpoint_dir, **kwargs)`: Loads and returns a `ModelConfig` object
- `load_weights(checkpoint_dir, mapping, **kwargs)`: Loads and returns a dictionary of weights
- `get_initialized_weight_mapper(model, config)`: Returns a runtime initialized weight mapper for the model
- `cleanup()`: Releases resources and cleans up internal state

### BaseConfigLoader

Responsible for loading model configurations from checkpoint directories and parsing them into TRTLLM `ModelConfig`:

```python
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader

class CustomConfigLoader(BaseConfigLoader):
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        # Load and parse configuration from your custom format
        pretrained_config = self._get_pretrained_config(checkpoint_dir, **kwargs)

        return ModelConfig(pretrained_config=pretrained_config,
                            ...)

    def _get_pretrained_config(self, checkpoint_dir, **kwargs):
        ...

```

### BaseWeightLoader

Handles the loading of model weights from storage:

```python
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader

class CustomWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping) -> dict[str, Any]:
        # Load weights from your custom format
        # Return a dictionary mapping parameter names to tensors
        return weights_dict
```

### BaseWeightMapper

Transforms weights between different naming conventions and applies model-specific transformations into TRTLLM model's object.

## Built-in Checkpoint Formats

### HuggingFace Format

Currently, HF checkpoint loader is the primary built-in format, supporting:

- **Weights loading** (`.safetensors/.bin/.pth`) - Loading HF compatible weights from disk
- **Configuration parser** - Parsing HF stored configuration information to TRTLLM `ModelConfig` object
- **Weights Mapping** - Converting HF weights into TRTLLM compatible representation

### ModelExpress (MX) Loading Path

The PyTorch backend can use ModelExpress (MX) for peer-to-peer weight transfer
from a running TensorRT-LLM source instance before falling back to Hugging Face
checkpoint loading. Selecting MX does not require an MX-specific on-disk
checkpoint or conversion of the Hugging Face checkpoint. For installation, MX
service deployment, and configuration details, see
[ModelExpress (MX) Checkpoint Loading](./model-express.md).

## Experimental Weight-Load Plans

The HF SafeTensors loader resolves an ordered `WeightLoadPlan` before starting
storage-specific collectives. The default plan is:

```text
direct_rank_read
shared_host_producer
gpu_broadcast
legacy_fallback
```

The first eligible and implemented policy is selected. Qualification and
fallback happen before policy-specific I/O begins; the loader does not switch
policies after a collective read or transfer has started. In this first
version, the two host policies share the same qualification rules and
`gpu_broadcast` is unavailable. The implicit plan therefore selects
`direct_rank_read` for a qualified model and otherwise reaches
`legacy_fallback`; select `shared_host_producer` explicitly to compare it.
An explicitly selected single policy is strict.

Set a single policy for controlled benchmarking, or a comma-separated ordered
plan when fallback is desired:

```bash
# Strict: fail if this policy is not eligible.
export TRTLLM_HF_WEIGHT_LOAD_PLAN=shared_host_producer

# Ordered fallback.
export TRTLLM_HF_WEIGHT_LOAD_PLAN=direct_rank_read,legacy_fallback
trtllm-serve <model>
```

The same selection is available when constructing a checkpoint loader:

```python
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader

checkpoint_loader = HfCheckpointLoader(
    weight_loader=HfWeightLoader(
        weight_load_plan=("shared_host_producer", "legacy_fallback")
    )
)
```

| Policy | Status | Current behavior |
| --- | --- | --- |
| `direct_rank_read` | Implemented; default primary policy | Node-local ranks own disjoint file extents and issue parallel direct reads into the shared OS page cache. |
| `shared_host_producer` | Implemented page-cache staging v0 | Node-local rank 0 reads all extents with a bounded thread pool; peer ranks wait and then mmap the shared cached pages. |
| `gpu_broadcast` | Recognized future policy | Preflight reports it unavailable until rank-aware final tensor placement and a topology-aware GPU transport exist. |
| `legacy_fallback` | Implemented compatibility policy | Uses the pre-existing whole-file prefetch and SafeTensors loading path. |

The policy is orthogonal to the weight source. This first implementation
applies only to HF disk loading with `LoadFormat.AUTO`; MX artifacts, GMS
transfers, and format-specific Mistral loaders keep their existing paths under
the default plan. A later source-neutral planner can pair the same placement
policies with local files, object-store streams, snapshots, MX, or GMS without
encoding those sources as more policy modes.

`.bin` and `.pth` checkpoints support only `legacy_fallback`. A strict plan
that omits that policy fails instead of silently measuring the legacy path.

Both implemented cooperative policies retain the existing 90%-of-available-
host-memory guard. When full prefetch is safe, they divide selected files into
256 MiB extents and read each extent through bounded 8 MiB temporary buffers.
After a node-local barrier, every rank uses the unchanged mmap-backed
SafeTensors and model-loading paths. If the memory guard rejects full prefetch,
the mmap path demand-pages the required ranges instead.

`direct_rank_read` is a storage-byte assignment, not yet a TP/PP-aware tensor
loader. It aims for one logical checkpoint read per node by striping extents
across ranks, but each rank subsequently maps the checkpoint, selects its model
weights, performs its TP/PP transforms, and copies its own parameters to its
GPU. Each node independently reads a complete checkpoint; PP ownership does
not partition storage traffic across nodes. "Direct" means that ranks own
regular buffered `pread` responsibilities; it does not mean Linux `O_DIRECT`,
GPUDirect Storage, or direct placement into final GPU tensors.

`shared_host_producer` currently uses the Linux page cache as shared host
storage. It is a useful single-producer comparison policy, but it is not the
complete pinned shared-memory producer/consumer design. That later design
requires partial model loading, a bounded pinned buffer, per-rank consumption
tracking, NUMA placement, cancellation, and crash-safe reclamation.

`gpu_broadcast` is shorthand for topology-aware GPU fan-out rather than a
literal full-model broadcast. Replicated weights could use NCCL broadcast, but
TP/EP-sharded weights require scatter or grouped point-to-point transfers and
PP ranks must receive only their owned layers. A future implementation would
pipeline storage to bounded pinned host buffers, copy once to a producer GPU,
and fan out rank-ready chunks over NVLink, NVSwitch, or PCIe into destination
parameter buffers. The current HF loader returns raw CPU tensors before
model-specific slicing and placement, so implementing this policy efficiently
requires a new rank-aware materialization interface.

The cooperative policies are initially restricted to unquantized dense
`LlamaForCausalLM`, `Qwen2ForCausalLM`, and `Qwen3ForCausalLM` checkpoints with
standard HF mapping, `LoadFormat.AUTO`, TP, and PP. Distributed cooperative
loading requires MPI-launched ranks. Context parallelism, expert parallelism,
attention data parallelism, DWDP, speculative decoding, VLMs, custom models or
mappers, quantized checkpoints, LoRA-enabled models, dynamic quantization, and
`.bin`/`.pth` weights resolve to `legacy_fallback` under the default plan.
Every participating rank must use the same plan, load format, and world size.

Page-cache reuse is best-effort and depends on the filesystem, mount, memory
pressure, and cache state. Cooperating ranks must resolve paths to the same
backing files. When the optional HF raw-weight cache is enabled and no plan is
explicitly configured, the loader preserves that request by selecting
`legacy_fallback`. An explicitly configured cooperative policy overrides the
cache with a warning because the cache does not yet mirror its collective
sequence. More I/O issuers often help Lustre or high-bandwidth NVMe, while one
producer can help a filesystem that penalizes concurrent clients. Measure both
against a verified cold cache on the target deployment.

## Using Checkpoint Loaders

### Basic Usage

There are two main approaches to trigger the use of checkpoint loading objects.

The first approach, through llm-api, as shown in the following example:

```python
from tensorrt_llm import LLM

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir)
```

In this example, `HfCheckpointLoader` will be selected by default.

To explicitly set the checkpoint loader, you need to call the required checkpoint-specific loader

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir,
          checkpoint_loader=HfCheckpointLoader())
```

Similarly, if one wants to use a basic implemented checkpoint loader, but with a specific subcomponent, they can provide any specific subcomponent upon need

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir,
          checkpoint_loader=HfCheckpointLoader(weight_loader=MyCustomWeightLoader()))
```

In the second approach, one can directly use the components of the checkpoint loading.

```python
from tensorrt_llm._torch.models.checkpoints.hf.gemma3_weight_mapper import \
    Gemma3HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM

gemma3 = Gemma3ForCausalLM(model_config)
weight_mapper = Gemma3HfWeightMapper()
weight_mapper.init_model_and_config(gemma3, model_config)
gemma3.load_weights(hf_gemma3.state_dict(), weight_mapper)
```
## Creating Custom Checkpoint Loaders

To support a new checkpoint format, you need to implement all four components. This section provides minimal templates for each component.

### When to Create Custom Components

- **Complete New Format**: Implement all four components when supporting a completely new checkpoint format
- **Custom Weight Storage**: Only implement a custom weight loader if you have a unique weight storage format (e.g., custom binary format, database storage, etc.)
- **Custom Configuration**: Only implement a custom config loader if your configuration format cannot be parsed by existing parsers.
- **Custom Weight Mapping**: Only implement a custom weight mapper if your model has unique weight naming or transformation requirements that are checkpoint-specific.

### Step 1: Create the Checkpoint Loader

```python
from typing import Optional
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader

@register_checkpoint_loader("CUSTOM_FORMAT")
class CustomCheckpointLoader(BaseCheckpointLoader):
    def __init__(self,
                 *,
                 weight_loader: Optional[BaseWeightLoader] = None,
                 weight_mapper: Optional[BaseWeightMapper] = None,
                 config_loader: Optional[BaseConfigLoader] = None):
        self._weight_loader = weight_loader or self.get_default_weight_loader()
        self._config_loader = config_loader or self.get_default_config_loader()
        self._weight_mapper = weight_mapper
        self._checkpoint_format = "CUSTOM_FORMAT"

    def get_default_weight_loader(self) -> BaseWeightLoader:
        return CustomWeightLoader()

    def get_default_config_loader(self) -> BaseConfigLoader:
        return CustomConfigLoader()
```

### Step 2: Create the Checkpoint Weight Loader

```python
from typing import Any
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_weight_loader

@register_checkpoint_weight_loader("CUSTOM_FORMAT")
class CustomWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        """
        Load weights from your custom format.
        Args:
            checkpoint_dir: Directory containing checkpoint files
            mapping: A mapping object containing the distributed configuration.
            **kwargs: Additional loading parameters
        Returns:
            Dictionary mapping parameter names to tensors
        """
        weights = {}

        # Implement your custom weight loading logic here
        # Examples:
        # - Load from custom binary files
        # - Load from databases
        # - Load from compressed archives
        # - Apply custom preprocessing

        return weights
```

### Step 3: Create the Checkpoint Config Loader

```python
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader

@register_config_loader("CUSTOM_FORMAT")
class CustomConfigLoader(BaseConfigLoader):
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        """
        Load and parse configuration from your custom format.
        Args:
            checkpoint_dir: Directory containing configuration files
            **kwargs: Additional loading parameters
        Returns:
            ModelConfig object containing parsed configuration
        """
        # Load your custom configuration format
        # Examples:
        # - Parse YAML/TOML files
        # - Convert from proprietary formats

        pretrained_config = self._load_pretrained_config(checkpoint_dir, **kwargs)

        return ModelConfig(
            pretrained_config=pretrained_config,
            # Add other ModelConfig parameters as needed
        )

    def _load_pretrained_config(self, checkpoint_dir: str, **kwargs):
        """Load the raw configuration from your custom format."""
        pass
```

### Step 4: Create the Checkpoint Weight Mapper

```python
from torch import nn
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

@register_mapper("CUSTOM_FORMAT")
class CustomWeightMapper(BaseWeightMapper):
    def __init__(self):
        super().__init__()
        # Define any weight transformation callbacks
        self._callbacks = [
            # Add your custom weight transformation functions
            # self._custom_transform_function,
        ]

    def map_weights(self) -> None:
        """
        Define mappings between source and target weight names.
        """
        self.mapping.update({
            # Map source names to target names
            # 'target_module_name': ['source_param1', 'source_param2'],
            # Example: 'qkv_proj': ['q_proj', 'k_proj', 'v_proj']
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """
        Apply weight transformations for modules that require special handling.
        Args:
            module: The target module
            module_name: The specific module name being processed
            module_names_breakdown: Module path components
            weights: Source weights dictionary
        Returns:
            List of transformed weight dictionaries
        """
        module_weights = []

        for new_name in self._mapping[module_name]:
            # Filter weights for this specific parameter
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)

            # Apply transformation callbacks
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)

            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        """
        Define which modules should be skipped during loading.
        """
        # Add logic to skip specific modules based on your requirements
        # Examples:
        # - Skip LoRA-specific modules
        # - Skip temporary/auxiliary modules

        return super().should_skip_module(module_name)
```

Note: when creating a custom mapper, you can either define a checkpoint-format-specific mapper. For example:

```python
@register_mapper("CUSTOM_FORMAT")
class CustomWeightMapper(BaseWeightMapper)
```

Alternatively, you can define a checkpoint-model-specific mapper. For example:

```python
@register_mapper("CUSTOM_FORMAT", "Gemma3ForCausalLM")
class CustomWeightMapper(BaseWeightMapper)
```

By setting the model name, the registered mapper will be associated with the specific model.
