# Checkpoint Loading

The PyTorch backend provides a flexible and extensible infrastructure for loading model checkpoints from different formats, such as HuggingFace (HF). This system allows you to load models from various sources (e.g., HuggingFace or custom formats) by implementing the required components, such as the checkpoint’s weight loader, mapper, and configuration parser.

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Built-in Checkpoint Formats](#built-in-checkpoint-formats)
4. [Using Checkpoint Loaders](#using-checkpoint-loaders)
5. [Creating Custom Checkpoint Loaders](#creating-custom-checkpoint-loaders)
6. [Checkpoint I/O Policies](#checkpoint-io-policies)
7. [Startup Telemetry](#startup-telemetry)

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

Select a format with `checkpoint_format` in the LLM API or the YAML config:

| `checkpoint_format` | Checkpoint it loads |
|---|---|
| `HF` (default) | Hugging Face checkpoint |
| `mistral` | Mistral native checkpoint |
| `mistral_large_3` | Mistral Large 3 native checkpoint |
| `MX` | ModelExpress peer-to-peer (P2P) weight transfer with Hugging Face checkpoint as fallback |

### HuggingFace Format

Currently, HF checkpoint loader is the primary built-in format, supporting:

- **Weights loading** (`.safetensors/.bin/.pth`) - Loading HF compatible weights from disk
- **Configuration parser** - Parsing HF stored configuration information to TRTLLM `ModelConfig` object
- **Weights Mapping** - Converting HF weights into TRTLLM compatible representation
- **Weights Streaming** - Support experimental weight streaming options (see [Checkpoint I/O Policies](#checkpoint-io-policies))

### Mistral Native Format

`checkpoint_format: mistral` loads checkpoints in Mistral's own layout: a
`params.json` config and `consolidated*.safetensors` weights.

### Mistral Large 3 Format

`checkpoint_format: mistral_large_3` reads the same layout as `mistral`. It
uses the Mistral Large 3 weight mapper for every architecture in the
checkpoint, including the multimodal `PixtralForConditionalGeneration`
wrapper.

### ModelExpress (MX) Loading Path

ModelExpress (MX) is a P2P weight transfer service that lets one GPU transfer
its loaded weights to another, skipping slow weight loading on the second GPU from storage.
This checkpoint option checks for MX service first for P2P weight transfer, and
falls back to loading from HuggingFace checkpoint if MX is not available.
For MX service details, see [ModelExpress (MX) Checkpoint Loading](./model-express.md).

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
    
    ...
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

## Checkpoint I/O Policies

The experimental `checkpoint_io_policy` option controls how the built-in HF
loader reads SafeTensors files from storage into GPU memory. It provides
options to speed up weight loading via efficient streaming from storage.

- `native`: It reads the checkpoint files into the OS page cache if there is
  enough CPU memory. It then opens each tensor via mmap and loads them to GPU.
  There could be many read contentions.
- `rank_striped_read_ahead`: It divides the checkpoint files into pieces and
  reads each piece into the OS page cache while copying tensors to GPU at the
  same time. This assumes most tensors being copied are from the OS page cache.
  It runs only when the whole checkpoint fits in the available host memory,
  minus some headroom. Otherwise it falls back to `native`.
- `auto` (default): use `rank_striped_read_ahead` when the load supports it,
  and `native` I/O otherwise.

Apart from built-in HF checkpoint and built-in checkpoint loader, to enable
non-native I/O policy, the following must all be true:
- `load_format` is `auto`.
- All of the model is set to be loaded (no partial model loading).
- The checkpoint uses SafeTensors files.

### Re-reads in `rank_striped_read_ahead`

The OS can drop read-ahead pages from the page cache before their tensors are
copied to GPU, for example when a Lustre client cancels its aged DLM locks.
When the node then takes major page faults while copying tensors to GPU, the
reader threads read their pieces into the page cache again. Each such re-read
is a refresh.

- Wait between refreshes: after the first refresh, the next refresh starts at
  least 10 seconds after the previous one completes. The wait then grows to 20
  and 40 seconds, up to 60 seconds. A warning is logged once when the wait
  reaches 60 seconds. The wait limits how often refreshes run, not how many
  refreshes run.
- Fault checks during the wait: the reader still checks page faults every 5
  seconds. A burst of faults queues one refresh, which starts when the wait
  ends.
- Memory check: before each refresh, the reader checks again that the
  available host memory can hold the full checkpoint plus the startup
  headroom. If the check fails, or the reader cannot read the memory status,
  the refresh stays queued and the reader checks again at later polls. A
  warning is logged once each time a refresh starts to wait on memory. This
  check is conservative and does not reserve memory: pages that are already in
  memory and charged to a cgroup can defer a refresh even when only a small
  part of the checkpoint is missing from the page cache.
- Tensor copies to GPU continue while a refresh waits.

### Logs

Each rank logs one line per checkpoint read (draft checkpoint loading has
its own line):

```text
Checkpoint I/O policy: requested=auto, selected=rank_striped_read_ahead, activated=True, effective=rank_striped_read_ahead, fallback_reason=none.
```

- `requested`: the configured policy.
- `selected`: the policy chosen by TRTLLM before loading started. Various checks like
config incompatibility could lead to `native` fallback.
- `activated`: whether the background reader threads in `rank_striped_read_ahead`
started.
- `effective`: the final policy that read the checkpoint. If there is a runtime issue
like communication error, it could fall back to `native`.
- `fallback_reason`: why loading switched to native I/O.

The `/server_info` response also reports these five fields in
`startup_metrics.model_loader`: `checkpoint_io_policy` for the target
checkpoint read, and `draft_checkpoint_io_policy` for a separate draft
checkpoint read. The value is `null` when the built-in HF weight loader did not
read the checkpoint from storage, for example with preloaded weights, a custom
weight loader, or a GMS reader.

## Startup Telemetry

CI perf-sanity runs that upload telemetry record how each server loaded its
checkpoint, so startup comparisons can account for the loader that ran.

The primary model's finalized policy is also published as
`startup_metrics.model_loader.checkpoint_io_policy` in `/server_info`.
This detached snapshot is captured after weight-session finalization, before a
separate draft checkpoint can reuse the loader. It preserves activation even
when a subsequent advisory read failure makes the effective policy native.
HF-derived formats such as Mistral and MX also report observed native disk
loads; this does not enable rank-striped read-ahead for those formats.
GMS writers using the built-in HF disk path capture the native policy after
target materialization, before draft loading; GMS readers remain unknown.
Unsupported, preloaded, or unobserved paths report an explicit unknown (`null`),
not an inferred native policy. The collector prefers this structured primary
status; only older runtimes without the field use `legacy_logs` fallback.
Draft statuses do not override the structured primary status.

Use the following fields to qualify a comparison (with `ctx_` or `gen_`
prefixes for disaggregated server roles):

- `s_checkpoint_io_policy_source`: `server_info`, `legacy_logs`, or `unknown`.
- `b_checkpoint_io_policy_complete`: every expected server provided one valid
  structured primary policy. Legacy log attribution is not considered complete.
- `b_startup_metrics_timing_complete`: every expected server provided finite,
  nonnegative preparation, population, finalization, and total model-loading
  times. The existing collection-complete flag only checks successful fetches
  and server counts; it does not imply these fields were present.

CI records include image provenance so startup comparisons can account for
runtime-environment differences instead of attributing them to the loader.
This describes the benchmark environment, not which images customers deploy.

The existing `s_runtime_image`, `s_runtime_image_source`,
`s_runtime_image_digest`, and `s_runtime_image_identity_strength` fields
describe the original OCI/base-image reference selected for the test
container, not the Jenkins dispatcher image. A valid digest-pinned reference
reports `digest_pinned` for that OCI reference only; a tagged reference reports
`reference_only` with an unknown digest. Missing provenance remains `unknown`.

Enroot may launch a derived fat `.sqsh` with dependencies and TRT-LLM already
installed. These fields do not identify or verify that derived artifact or
the environment after package installation. A tag hash or `.sqsh` cache
filename is not a content digest. Group comparisons by the recorded base
reference as well as model/configuration, GPU and parallelism, but do not
treat matching base references as proof of identical runtime environments.
Final launch-artifact provenance remains a separate follow-up.

These additions do not change loader policy or add startup collectives. Timing
remains rank-0-based per server, and failed launches are not comprehensively
persisted. Native counterfactual eligibility, checkpoint fingerprints,
slowest-rank timing, and process-launch-to-first-token measurement remain
separate follow-ups.
