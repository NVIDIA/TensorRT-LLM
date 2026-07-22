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
fallback happen before checkpoint payload I/O or shared-window allocation; the
loader does not switch policies after a collective read or transfer has
started. The implicit plan selects `direct_rank_read` for HF/AUTO SafeTensors
and otherwise reaches `legacy_fallback`; select `shared_host_producer`
explicitly to compare the producer/consumer design. An explicitly selected
single policy is strict. Shared-host preflight parses only SafeTensors headers
and requires both partial model loading and a mapper-provided atomic dependency
manifest before it can be selected.
The ordered plan is an eligibility/fallback mechanism, not a runtime
performance-adaptive selector: one policy handles the whole checkpoint, and
policies are not mixed per tensor.

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
| `direct_rank_read` | Implemented; default primary policy | Node-local ranks own disjoint file extents and issue parallel background reads into the shared OS page cache while ModelLoader materializes tensors. |
| `shared_host_producer` | Implemented for qualified model mappers | One producer per node fills an adaptively sized, bounded shared-memory double buffer and attempts CUDA host registration. All local ranks incrementally materialize complete dependency groups while the producer fills the next slot. |
| `single_producer_page_cache_prefetch` | Implemented comparison policy | Node-local rank 0 reads all extents into the Linux page cache; peers wait at a node barrier and then use the unchanged mmap path. This preserves the earlier policy that was temporarily named `shared_host_producer`. |
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

`direct_rank_read` and `single_producer_page_cache_prefetch` retain the existing
90%-of-available-host-memory guard. When full prefetch is safe, they divide
selected files into 256 MiB extents and read each extent through bounded 8 MiB
temporary buffers. The single-producer page-cache policy completes those reads
and a node-local barrier before every rank uses the unchanged mmap-backed
SafeTensors and model-loading paths. For `direct_rank_read`, ModelLoader starts
a background read-ahead session,
maps the full raw weight dictionary immediately, and keeps the session open
through mapper initialization and model weight materialization. Session exit
joins the reads, coordinates errors, and performs the node barrier. This is
system-level overlap between filesystem I/O and existing materialization/H2D
work; it is not pinned-memory asynchronous DMA. Direct callers of
`HfWeightLoader.load_weights()` retain synchronous behavior. If the memory
guard rejects full prefetch, the mmap path demand-pages the required ranges
instead.

`direct_rank_read` is a storage-byte assignment, not yet a TP/PP-aware tensor
loader. It aims for one logical checkpoint read per node by striping extents
across ranks, but each rank subsequently maps the checkpoint, selects its model
weights, performs its TP/PP transforms, and copies its own parameters to its
GPU. Each node independently reads a complete checkpoint; PP ownership does
not partition storage traffic across nodes. "Direct" means that ranks own
regular buffered `pread` responsibilities; it does not mean Linux `O_DIRECT`,
GPUDirect Storage, or direct placement into final GPU tensors.

`shared_host_producer` is the producer/consumer implementation. SafeTensors
headers and an atomic mapper manifest are validated first. Each node then
allocates an MPI shared-memory double buffer, and node-local rank 0 reads the
next ordered batch with a bounded worker pool while all local ranks consume the
current batch. The configured 256 MiB slot size is a baseline: planning grows
each slot to the packed size of the largest atomic group, subject to a 64 GiB
total two-slot budget. This keeps the allocation bounded by one dependency
group rather than the full checkpoint while allowing groups larger than 256
MiB to remain in one published lease.

The producer uses ordinary buffered `pread` operations. Source data therefore
still traverses the Linux page cache before the kernel copies it into the
shared arena; this policy is bounded streaming, not `O_DIRECT` or GDS. The
initial protocol publishes one atomic dependency group per batch (or multiple
batches when one group exceeds a slot) and does not yet pack independent small
groups together, so benchmark group/batch counts and acknowledgement overhead.

After collective shared-window allocation, every node-local rank opens one
MPI passive-target epoch with `Win.Lock_all`; all producer/consumer
`Win.Sync` calls occur inside that epoch. Normal teardown closes the epoch
with `Win.Unlock_all` on every rank, coordinates the result node-locally, and
only then enters collective `Win.Free`. A rank-local lock or unlock failure is
coordinated before any collective free. Unsafe windows are quarantined rather
than freed; a known lifetime quarantine intentionally leaves its epoch open so
escaped storage cannot outlive the MPI window.

When every rank on a node successfully registers that node's shared arena with
CUDA, a group fits in one slot, and the mapper declares the runtime profile's
source-tensor lifetime safe, consumers borrow immutable tensor views directly
from that single shared copy. The existing model-specific transformations and
H2D copies consume those views, and each rank synchronizes its current CUDA
stream before acknowledging the lease. No per-rank full-group host copy is
made on this path. A registration failure disables direct views only on the
affected node, not on independently registered nodes.

If CUDA host registration is unavailable, an atomic group exceeds the
configured buffer budget, or the source-lifetime contract is not qualified,
the correctness fallback assembles that group in rank-local pinned host
storage (with a logged pageable fallback). Quantized profiles currently take
this staging path because some Linear and MoE methods retain temporary source
views until final checkpoint processing; integrated-GPU profiles stage as well
to keep node-shared bytes outside mmap page-eviction hooks. Unquantized,
static-loading Qwen 3.5 and Llama 4 profiles on discrete GPUs may use direct
views. An all-rank completion or error consensus gates slot reuse in both
cases. Telemetry reports the configured and effective slot sizes, largest
group, single-slot coverage, node-local all-ranks registration result, and
direct-versus-staged group and byte counts; benchmark runs should retain these
fields with their timing results.

Direct views have a strict lifetime contract: model loading must neither mutate
nor retain source storage after the incremental load call returns. If a loader
or its exception traceback retains borrowed storage, the loader raises a
dedicated retention error and the completion consensus propagates a quarantine
decision to every rank. All ranks then intentionally retain the CUDA
registration, MPI window, and bounded two-slot arena for the rest of the
process lifetime. Startup still fails, but no rank reuses or frees memory that
an escaped tensor may reference. This bounded leak occurs only on the failed
contract-violation path; normal completion and other coordinated failures keep
their usual collective teardown.

After the last dependency group is loaded and mapper coverage is validated,
the stream runs each eligible module's deferred
`process_weights_after_loading` hook exactly once before the final batch
consensus. Wrapper-owned MoE backends are de-duplicated. Any deferred
quantization, fusion, or CUDA synchronization failure is therefore reported to
all ranks before the final slot can be reused. The ordinary
`post_load_weights` walk still runs afterward through the common model-loader
lifecycle.

For this policy, `HfWeightLoader.prefetch_chunk_size` sets the baseline slot
size, `shared_host_buffer_budget` caps the total double buffer, and
`prefetch_workers_per_rank` sets the single node producer's I/O worker count.
The producer defaults to the existing 64-worker node budget. The buffer budget
can also be set in bytes with
`TRTLLM_HF_SHARED_HOST_BUFFER_BUDGET_BYTES`; the constructor value takes
precedence. Increase the default only after confirming host-memory headroom.

This shared window contains immutable raw checkpoint bytes. It is not a
transformed-weight cache, does not require the producer to retain a full model,
and provides no restart reuse after the load session closes. On a multi-node
job, every node has its own producer and shared window; checkpoint payload I/O
is deduplicated within a node, not across nodes. The stream and placement
interfaces remain source-neutral so a future MX, GMS, snapshot, or Model
Streamer source can reuse the incremental materialization lifecycle without
being implemented as another policy mode.

`gpu_broadcast` is shorthand for topology-aware GPU fan-out rather than a
literal full-model broadcast. Replicated weights could use NCCL broadcast, but
TP/EP-sharded weights require scatter or grouped point-to-point transfers and
PP ranks must receive only their owned layers. A future implementation would
pipeline storage to bounded pinned host buffers, copy once to a producer GPU,
and fan out rank-ready chunks over NVLink, NVSwitch, or PCIe into destination
parameter buffers. The current HF loader returns raw CPU tensors before
model-specific slicing and placement, so implementing this policy efficiently
requires a new rank-aware materialization interface.

The two page-cache policies return the same complete raw tensor dictionary as
the legacy mmap path, so their eligibility is not tied to a model class,
mapper, quantization mode, or TP/PP/CP/EP layout. Shared-host streaming is more
selective: `model.load_weights` must explicitly accept
`allow_partial_loading`, and the initialized mapper must partition every
checkpoint source key exactly once into dependency-safe groups. This prevents
half of a fused QKV projection, quantization scale family, MoE layer, vision
tower, or MTP dependency from reaching model code. Unsupported models fail
before parameter mutation in a strict plan or explicitly advance to the next
policy in an ordered plan.

Falling back to the exact generic `HfWeightMapper` is not qualification by
itself. After an end-to-end audit, a model class can explicitly declare
`_supports_generic_hf_incremental_loading = True`; the marker is intentionally
not inherited by derived architectures. Without that model-level opt-in, the
generic mapper publishes no manifest, does not borrow transport-owned tensors,
and does not use incremental subtree dispatch.

Qualified mappers use the smallest audited dependency: a fused QKV or gate/up
family, a Qwen linear-attention qkvz/ba family, or an ordinary parameter
family. Routed-MoE tensors remain atomic at the complete MLP/MoE module, and
multimodal vision plus projector loaders that require a complete state dict
remain one group. Incremental dispatch walks only those destination subtrees
and avoids creating a full-model thread pool for each group.
Loads that require a separately opened speculative-draft checkpoint are also
conservatively ineligible until target and draft streams share one preflight
and failure transaction; integrated MTP groups remain mapper-qualified.

An atomic manifest is necessary but is not end-to-end qualification. Every
model, quantization backend, and parallelism configuration still requires
correctness and cold-start benchmark coverage. Distributed cooperative loading
requires MPI-launched ranks. `.bin`/`.pth`, MX, GMS, and format-specific
loaders retain their existing paths. Every participating rank must use the
same plan, load format, checkpoint metadata, group manifest, and world size.
Before header parsing or shared-window allocation, the loader also walks nested
Linear and MoE modules and requires their concrete quantization/backend methods
to advertise partial-load support. Unsupported backends fail a strict shared
plan or advance an ordered plan before model mutation. Dynamic EPLB is
currently ineligible because it deliberately retains complete raw expert
tensors beyond a bounded batch. Llama 4 min-latency loading is also ineligible
because it eagerly derives FP8 layouts before deferred partial-load
finalization. Static EPLB does not have the raw-source retention behavior and
remains eligible when its selected backend passes the nested capability check.

The initial model-specific manifests cover Qwen 3.5 text/VLM and Llama 4.
Architectures that use the unmodified generic HF mapper remain ineligible until
their model class explicitly opts into the audited contract above. DeepSeek V4
is deliberately not enabled for `shared_host_producer` yet: its bespoke loader
performs whole-checkpoint remapping, synthesized defaults, direct key lookups,
and MTP/shared-layer finalization without a partial-load contract. It continues
to use `direct_rank_read` or an explicit fallback until that loader is
refactored and qualified across BF16/FP8/NVFP4, TP/EP, and MTP configurations.

Flagship qualification should exercise the real downstream loaders rather
than add model-name checks to the byte reader:

| Family | Minimum coverage |
| --- | --- |
| Qwen 3.5 | Dense and MoE HF checkpoints; BF16, FP8, and NVFP4 where supported; TP/EP and attention-DP on and off; the Qwen 3.5 custom mapper. |
| DeepSeek V4 | Native HF SafeTensors; supported quantization; TP/EP with attention-DP; MTP target and draft loading. |
| Llama 4 | Scout and Maverick; text and multimodal construction; FP8; TP/EP and a PP configuration that verifies layer ownership. |

For each configuration, run strict `direct_rank_read`, strict
`shared_host_producer`, the default ordered plan, and `legacy_fallback`. The
single-producer page-cache policy can be retained as an additional diagnostic,
but it is not the shared-memory producer/consumer result. A strict optimized
run fails qualification if it falls back. Validate
deterministic inference parity, clean worker and communicator teardown, and
peak host memory, then measure cold-cache model initialization and first-
inference latency. Performance claims require a system trace showing storage
reads overlapping model materialization or H2D activity.

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
