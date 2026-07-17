<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Checkpoint Loading

The PyTorch backend provides a flexible and extensible infrastructure for loading model checkpoints from different formats, such as HuggingFace (HF). This system allows you to load models from various sources (e.g., HuggingFace or custom formats) by implementing the required components, such as the checkpoint’s weight loader, mapper, and configuration parser.

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Built-in Checkpoint Formats](#built-in-checkpoint-formats)
4. [Distributing checkpoints and engine artifacts from object storage](#distributing-checkpoints-and-engine-artifacts-from-object-storage)
5. [Using Checkpoint Loaders](#using-checkpoint-loaders)
6. [Creating Custom Checkpoint Loaders](#creating-custom-checkpoint-loaders)

## Overview

The checkpoint loading design is built around a plugin-like architecture that is separated into four distinct components:

- **Checkpoint Loaders**: Orchestrate the loading process for specific formats
- **Config Loaders**: Handle model configuration parsing and validation
- **Weight Loaders**: Manage the actual loading of model weights from storage into memory
- **Weight Mappers**: Map and transform loaded weights to TensorRT-LLM model's definition

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

## Distributing checkpoints and engine artifacts from object storage

TensorRT-LLM reads checkpoints and engines from a local directory and does
not include an object-store client. To distribute artifacts across a GPU
fleet you stage them in object storage and pull them onto each serving node
with standard S3 tooling. This suits object storage that exposes the Amazon
S3 API, because TensorRT engines are large and specific to a given GPU,
driver, and TensorRT version, so an organization typically ends up hosting a
matrix of engines keyed by model, GPU SKU, and precision. The workflow below
is written against the S3 API, so the same `aws` CLI and `boto3` code work
against Amazon S3 directly, or against any S3-compatible object store (for
example Amazon S3, Backblaze B2, Cloudflare R2, and MinIO) once you point the
tooling at that store's endpoint.

The workflow has three stages: build the engine (or obtain a prebuilt one),
upload the artifact directory to a bucket, and sync it to a local cache on
each serving node before pointing `LLM(model=...)` or `trtllm-serve` at the
cached path.

### Credentials and endpoint

The `aws` CLI and `boto3` read AWS-style environment variables. Set your
access key, secret key, region, and bucket. For a non-AWS store, also set the
S3 endpoint. Adjust the region and endpoint to match your bucket:

```bash
# S3 credentials, mapped onto the AWS_* names the tooling reads.
export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${S3_REGION:-us-east-1}"

# Bucket.
export S3_BUCKET_NAME="my-trtllm-artifacts"

# For non-AWS S3-compatible storage, uncomment and set the endpoint.
# Leave S3_ENDPOINT unset for Amazon S3.
# export S3_ENDPOINT="https://your-s3-endpoint.example.com"

S3_ENDPOINT_ARGS=()
if [ -n "${S3_ENDPOINT:-}" ]; then
    S3_ENDPOINT_ARGS=(--endpoint-url "$S3_ENDPOINT")
fi
```

For Amazon S3, leave `S3_ENDPOINT` unset and use AWS credentials directly. For
any other S3-compatible store, set `S3_ENDPOINT` so the examples pass
`--endpoint-url "$S3_ENDPOINT"` on each command.

### Build and upload an engine

Build an engine from a checkpoint with `trtllm-build` (the PyTorch backend can
also load checkpoints directly, in which case you upload the checkpoint
directory instead of a built engine):

```bash
trtllm-build --checkpoint_dir ./llama-3.1-8b-ckpt \
             --output_dir ./engines/llama-3.1-8b-fp8

# Sync the engine directory (config.json + rank*.engine shards) to the bucket.
aws s3 sync ./engines/llama-3.1-8b-fp8 \
    "s3://${S3_BUCKET_NAME}/engines/llama-3.1-8b-fp8" \
    "${S3_ENDPOINT_ARGS[@]}"
```

### Download on the serving node

On each serving node, sync the artifact directory into a local cache and serve
from the cached path. `aws s3 sync` only transfers objects that are missing or
changed, so a warm cache skips the download:

```bash
export TRTLLM_MODEL_CACHE="${TRTLLM_MODEL_CACHE:-/models/cache}"
S3_PREFIX="${S3_PREFIX:-engines/llama-3.1-8b-fp8}"
S3_PREFIX="$(printf '%s' "$S3_PREFIX" | sed 's#^/*##; s#/*$##')"
LOCAL_DIR="${TRTLLM_MODEL_CACHE}/${S3_PREFIX}"

aws s3 sync "s3://${S3_BUCKET_NAME}/${S3_PREFIX}" \
    "$LOCAL_DIR" "${S3_ENDPOINT_ARGS[@]}"

# Engine directories use the TensorRT backend. For checkpoint directories,
# omit --backend tensorrt so trtllm-serve uses the PyTorch backend.
trtllm-serve "$LOCAL_DIR" --backend tensorrt --host 0.0.0.0 --port 8000
```

### Equivalent download with boto3

When you prefer to pull artifacts from application code (for example, inside a
container entrypoint), `boto3` against the same endpoint is equivalent. Setting
`user_agent_extra` identifies the traffic to the storage service:

```python
import os
from pathlib import Path

import boto3
from botocore.config import Config

from tensorrt_llm._tensorrt_engine import LLM

CACHE_DIR = Path(os.environ.get("TRTLLM_MODEL_CACHE", "/models/cache"))
BUCKET = os.environ["S3_BUCKET_NAME"]
PREFIX = os.environ.get("S3_PREFIX", "engines/llama-3.1-8b-fp8/").strip("/") + "/"
LOCAL_DIR = CACHE_DIR / PREFIX.rstrip("/")
CACHE_ROOT = LOCAL_DIR.resolve()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    config=Config(user_agent_extra="tensorrt-llm-docs"),
)

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel = key[len(PREFIX):]
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Unsafe S3 object key: {key}")
        dest = (LOCAL_DIR / rel_path).resolve()
        if not dest.is_relative_to(CACHE_ROOT):
            raise ValueError(f"Unsafe S3 object key: {key}")
        remote_mtime = obj["LastModified"].timestamp()
        if dest.exists():
            stat = dest.stat()
            if stat.st_size == obj["Size"] and int(stat.st_mtime) >= int(remote_mtime):
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(dest))
        os.utime(dest, (remote_mtime, remote_mtime))

llm = LLM(model=str(LOCAL_DIR))
```

### Recommended practices

- **Bucket-scoped credentials.** Issue read-only credentials scoped to the
  artifact bucket for serving nodes; never deploy with root or account-wide
  keys.
- **Match the cache key to the build.** TensorRT engines are not portable
  across GPU SKU, driver, or TensorRT version. Encode those values in the
  object prefix (for example, `engines/<model>/<gpu>/<precision>/`) so a node
  never loads an engine built for a different configuration.
- **Multipart transfers.** The default `boto3` and `aws` CLI transfer
  configuration uses multipart requests, which suits the large `.safetensors`
  checkpoint shards and `rank*.engine` files typical of LLM artifacts.
- **Local cache reuse.** Mount `TRTLLM_MODEL_CACHE` on persistent storage so a
  cold pull happens once per node rather than once per pod restart.
- **Custom weight loader.** To stream checkpoint weights directly from the
  object store without staging to disk, implement a `BaseWeightLoader`
  subclass that reads each shard from `s3.get_object(...)["Body"]`. See
  *Creating Custom Checkpoint Loaders* below.

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
