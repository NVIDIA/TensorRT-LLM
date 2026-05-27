# LoRA (Low-Rank Adaptation)

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that enables adapting large language models to specific tasks without modifying the original model weights. Instead of fine-tuning all parameters, LoRA introduces small trainable rank decomposition matrices that are added to existing weights during inference.

## Table of Contents
1. [Background](#background)
2. [Basic Usage](#basic-usage)
   - [Single LoRA Adapter](#single-lora-adapter)
   - [Multi-LoRA Support](#multi-lora-support)
3. [Advanced Usage](#advanced-usage)
   - [LoRA with Quantization](#lora-with-quantization)
   - [NeMo LoRA Format](#nemo-lora-format)
   - [Routed-Expert MoE LoRA](#routed-expert-moe-lora)
   - [Cache Management](#cache-management)
4. [TRTLLM serve with LoRA](#trtllm-serve-with-lora)
   - [YAML Configuration](#yaml-configuration)
   - [Starting the Server](#starting-the-server)
   - [Client Usage](#client-usage)
5. [TRTLLM bench with LoRA](#trtllm-bench-with-lora)
   - [YAML Configuration](#yaml-configuration)
   - [Run trtllm-bench](#run-trtllm-bench)

## Background

The PyTorch backend provides LoRA support, allowing you to:
- Load and apply multiple LoRA adapters simultaneously
- Switch between different adapters for different requests
- Use LoRA with quantized models
- Support both HuggingFace and NeMo LoRA formats

## Basic Usage

### Single LoRA Adapter

```python
from tensorrt_llm import LLM
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.sampling_params import SamplingParams

# Configure LoRA
lora_config = LoraConfig(
    lora_dir=["/path/to/lora/adapter"],
    max_lora_rank=8,
    max_loras=1,
    max_cpu_loras=1
)

# Initialize LLM with LoRA support
llm = LLM(
    model="/path/to/base/model",
    lora_config=lora_config
)

# Create LoRA request
lora_request = LoRARequest("my-lora-task", 0, "/path/to/lora/adapter")

# Generate with LoRA
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=50)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_request]
)
```

### Multi-LoRA Support

```python
# Configure for multiple LoRA adapters
lora_config = LoraConfig(
    lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
    max_lora_rank=8,
    max_loras=4,
    max_cpu_loras=8
)

llm = LLM(model="/path/to/base/model", lora_config=lora_config)

# Create multiple LoRA requests
lora_req1 = LoRARequest("task-1", 0, "/path/to/adapter1")
lora_req2 = LoRARequest("task-2", 1, "/path/to/adapter2")

prompts = [
    "Translate to French: Hello world",
    "Summarize: This is a long document..."
]

# Apply different LoRAs to different prompts
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_req1, lora_req2]
)
```

## Advanced Usage

### LoRA with Quantization

```python
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# Configure quantization
quant_config = QuantConfig(
    quant_algo=QuantAlgo.FP8,
    kv_cache_quant_algo=QuantAlgo.FP8
)

# LoRA works with quantized models
llm = LLM(
    model="/path/to/model",
    quant_config=quant_config,
    lora_config=lora_config
)
```

### NeMo LoRA Format

```python
# For NeMo-format LoRA checkpoints
lora_config = LoraConfig(
    lora_dir=["/path/to/nemo/lora"],
    lora_ckpt_source="nemo",
    max_lora_rank=8
)

lora_request = LoRARequest(
    "nemo-task",
    0,
    "/path/to/nemo/lora",
    lora_ckpt_source="nemo"
)
```

### Routed-Expert MoE LoRA

LoRA can be applied to the routed-expert projections of a Mixture-of-Experts (MoE) layer in addition to the attention modules. The PyTorch backend's Cutlass MoE kernel fuses the LoRA application with the MoE forward pass, so multi-adapter batches and CUDA-graph-captured decode steps both work without an extra GEMM pass per layer.

#### Supported configuration

| Aspect | Supported in MVP |
|---|---|
| MoE backend | `CUTLASS` only (other backends raise an error at construction). |
| Base-weight dtype | bf16 / fp16. Quantized base weights (FP8, NVFP4, INT4, INT8) are not yet supported. |
| Adapter modules | `moe_h_to_4h` (gate side of SwiGLU), `moe_gate` (up side), `moe_4h_to_h` (down). At minimum, `moe_gate` and `moe_4h_to_h` must be present together. |
| Adapter layout | Per-expert and shared-outer (via load-time replication; see below). |
| Multi-LoRA in flight | Yes. Reuses the existing slot manager. |
| Execution paths | Eager and CUDA-graph decode. |
| min-latency mode | Not supported with MoE LoRA. |
| Alltoall (WideEP) | Not supported with MoE LoRA. |
| DoRA on MoE modules | Not supported (and rejected at load time). |
| `register_to_config` + `torch.compile` | Not supported with MoE LoRA. |

#### Enabling routed-expert MoE LoRA

```python
from tensorrt_llm import LLM
from tensorrt_llm.lora_manager import LoraConfig

lora_config = LoraConfig(
    lora_target_modules=[
        "attn_q", "attn_k", "attn_v",  # optional: standard attention LoRA
        "moe_gate",                    # up projection (required for MoE LoRA)
        "moe_4h_to_h",                 # down projection (required for MoE LoRA)
        "moe_h_to_4h",                 # gate projection (SwiGLU; optional)
    ],
    max_lora_rank=16,
    max_loras=8,
    max_cpu_loras=8,
)

llm = LLM(
    model="/path/to/moe_base_model",
    lora_config=lora_config,
    # The MoE LoRA path requires the Cutlass backend. Other moe_backend values
    # raise ValueError at construction.
    moe_backend="CUTLASS",
)
```

Adapters are loaded via `LoRARequest` exactly like attention-only LoRA — no API change.

#### Adapter layouts

Two adapter layouts are supported:

- **Per-expert.** Each expert has its own `(A, B)` matrices stored as stacked `[num_experts, rank, in_dim]` and `[num_experts, out_dim, rank]` tensors. This is the standard HuggingFace PEFT export shape for MoE LoRA.
- **Shared-outer.** The matrix on the residual-stream side is shared across experts:
  - For up-projections (`moe_h_to_4h`, `moe_gate`), `A` is shared (`[rank, hidden_size]`) and `B` is per expert.
  - For the down-projection (`moe_4h_to_h`), `B` is shared (`[hidden_size, rank]`) and `A` is per expert.

In the MVP, shared-outer is supported via **load-time replication**: replicate the shared tensor `num_experts` times before saving the adapter so it looks per-expert on disk. The MoE kernel reads each expert's slice at offset `expert_index * dim * rank`, so a correctly replicated checkpoint is bit-identical in behavior to a per-expert one. A native single-copy kernel path is planned for a later release.

A helper for assembling synthetic adapters (for unit tests and experimentation) is provided at `tensorrt_llm._torch.peft.lora.moe_layout`:

```python
from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora

# Shared-A adapter for the up projection (residual-stream-side shared).
fc1_adapter = make_per_expert_lora(
    num_experts=8, rank=16, in_dim=2048, out_dim=5632,
    shared_side="A",   # or "B" for down-projection, or None for fully per-expert
    dtype=torch.bfloat16,
)
# fc1_adapter["A"].shape == (8, 16, 2048)  -- replicated 8 times, all slices identical
# fc1_adapter["B"].shape == (8, 5632, 16)  -- independent per expert
```

#### CUDA-graph decode

In this release, MoE LoRA runs in eager mode. CUDA-graph **capture** of LoRA-active routed-expert MoE layers is not supported: the fused MoE kernel's LoRA path performs a host-side `cudaEventSynchronize` after a device-to-host pointer-expansion copy, which is not capturable. When CUDA-graph decode is enabled and an MoE LoRA is active, the fused MoE op detects the capturing stream and raises a clear error.

The slot-indexed input plumbing (a `token_to_slot` map plus per-slot pointer tables that live in persistent pinned host memory) is already in place inside the op and provides stable input layouts for eager replay. The kernel-side host-sync removal that unlocks full graph capture is tracked as a follow-up; it will be transparent to callers when it lands. No user-facing API change is needed — `LLM(...)` with `lora_config` already wires everything through; you just need to disable CUDA-graph capture for now (e.g. by running with `--cuda_graph_disabled` or the equivalent config option) if MoE LoRA is active.

#### What is rejected, and where

If you supply MoE LoRA on a non-Cutlass backend or with quantization, `create_moe` raises at construction with a message pointing at the offending setting. At runtime, the fused MoE op also rejects min-latency mode + LoRA, alltoall + LoRA, (per-request, slot-indexed) mixed inputs, and CUDA-graph capture + LoRA.

### Cache Management

```python
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig

# Fine-tune cache sizes
peft_cache_config = PeftCacheConfig(
    host_cache_size=1024*1024*1024,  # 1GB CPU cache
    device_cache_percent=0.1          # 10% of free GPU memory
)

llm = LLM(
    model="/path/to/model",
    lora_config=lora_config,
    peft_cache_config=peft_cache_config
)
```

## TRTLLM serve with LoRA

### YAML Configuration

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

Create a `config.yaml` file:

```yaml
lora_config:
  lora_target_modules: ['attn_q', 'attn_k', 'attn_v']
  max_lora_rank: 8
```
### Starting the Server
```bash
python -m tensorrt_llm.commands.serve
     /path/to/model \
    --config config.yaml
```

### Client Usage

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.completions.create(
    model="/path/to/model",
    prompt="What is the capital city of France?",
    max_tokens=20,
    extra_body={
        "lora_request": {
            "lora_name": "lora-example-0",
            "lora_int_id": 0,
            "lora_path": "/path/to/lora_adapter"
        }
    },
)
```

## TRTLLM bench with LoRA

### YAML Configuration

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

Create a `config.yaml` file:

```yaml
lora_config:
  lora_dir:
    - /workspaces/tensorrt_llm/loras/0
  max_lora_rank: 64
  max_loras: 8
  max_cpu_loras: 8
  lora_target_modules:
    - attn_q
    - attn_k
    - attn_v
  trtllm_modules_to_hf_modules:
    attn_q: q_proj
    attn_k: k_proj
    attn_v: v_proj
```
### Run trtllm-bench
```bash
trtllm-bench --model $model_path throughput --dataset $dataset_path --config config.yaml --num_requests 64 --concurrency 16
```
