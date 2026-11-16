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
   - [Options for Performance Tuning](#options-for-performance-tuning)
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
from tensorrt_llm._torch.peft.lora.config import LoraConfig
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

#### Native FP8 adapter support

The base model's quantization and the LoRA adapter's data type are independent. Dense and routed-expert MoE LoRA modules can keep
FP8 E4M3 adapter weights and execute them with native FP8 grouped GEMM kernels on the following architectures:

| GPU architecture | Native FP8 adapter support |
|---|---|
| Hopper (SM90) | Yes |
| Blackwell B200 (SM100) | Yes |
| Blackwell (SM103/SM107) | No |
| Blackwell (SM120/SM121) | No |

If native FP8 LoRA kernels are unavailable for the current device or were excluded from the TensorRT-LLM build,
the adapter weights are converted to the model compute data type. Routed-expert modules require the CUTLASS
backend; see [Routed-Expert MoE LoRA](#routed-expert-moe-lora).

For routed-expert adapters, native FP8 is selected from the checkpoint's `float8_e4m3fn` dtype.
All adapter matrices must have one dtype, and the rank, input dimension, and output dimension must be
multiples of 16. DoRA is unsupported. Save the adapter matrices as BF16 or FP16 to use the compute-dtype
path instead.

Native FP8 reduces adapter tensor storage from two bytes per element (BF16/FP16) to one byte. This
halves the weight payload in the PEFT cache, not total engine memory or necessarily allocated cache capacity.
This storage reduction does not imply a latency or throughput improvement: each routed-expert FC1/FC2 LoRA stage casts its
input to FP8, adding a full input read/write pass. These casts reuse existing LoRA result buffers;
they do not add full-size workspace allocations. The final grouped-GEMM epilogue writes directly to
BF16/FP16, so no result-conversion pass is needed.

This execution path is lossy. Activations and the low-rank intermediate are rounded to E4M3 with
unit scale and saturation at magnitude 448. The final LoRA delta is accumulated in FP32 and stored
in the model's BF16/FP16 compute dtype; it is not rounded or saturated to FP8. There is no dynamic
activation scaling. Check application accuracy before deploying an FP8 adapter, especially when
activations or low-rank intermediates approach the saturation limit. Accuracy depends on the adapter
and workload; there is no universal error bound.

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

LoRA can be applied to the routed-expert projections of a Mixture-of-Experts (MoE) layer in addition to the attention modules. The PyTorch backend's Cutlass MoE kernel fuses the LoRA application into the MoE forward pass, so multi-adapter batches run without an extra GEMM pass per layer.

#### Supported configuration

| Aspect | Supported |
|---|---|
| MoE backend | `CUTLASS` only (other backends raise an error at construction). |
| Base-weight dtype | bf16 / fp16 or per-tensor FP8 (QDQ). FP8 block-scale, NVFP4, MXFP8, INT4, and INT8 are not supported. |
| Adapter modules | `moe_h_to_4h` (gate side of SwiGLU), `moe_gate` (up side), `moe_4h_to_h` (down). `moe_h_to_4h` and `moe_4h_to_h` must both be present; `moe_gate` is optional (gated activations only). |
| Adapter layout | Per-expert (stacked `[num_experts, ...]`). |
| Multi-LoRA in flight | Yes. Reuses the existing slot manager. |
| Execution modes | Eager and CUDA-graph capture/replay (see below). Both feed the same grouped-GEMM LoRA core. |
| min-latency mode | Not supported with MoE LoRA. |
| Alltoall (WideEP) | Not supported with MoE LoRA. |
| DoRA on MoE modules | Not supported (and rejected at load time). |
| `register_to_config` + `torch.compile` | Not supported with MoE LoRA. |

#### Enabling routed-expert MoE LoRA

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig

lora_config = LoraConfig(
    lora_target_modules=[
        "attn_q", "attn_k", "attn_v",  # optional: standard attention LoRA
        "moe_h_to_4h",                 # gate/SiLU projection (required for MoE LoRA)
        "moe_4h_to_h",                 # down projection (required for MoE LoRA)
        "moe_gate",                    # up/linear projection (optional; gated activations)
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

Adapters are loaded via `LoRARequest` exactly like attention-only LoRA; no API change.

#### Adapter layout

Each routed expert has its own `(A, B)` matrices, stored as stacked `[num_experts, rank, in_dim]` and `[num_experts, out_dim, rank]` tensors. This is the standard HuggingFace PEFT export shape for MoE LoRA; the MoE kernel reads each expert's slice at offset `expert_index * dim * rank`.

A helper for assembling synthetic per-expert adapters (for unit tests and experimentation) is provided at `tensorrt_llm._torch.peft.lora.moe_layout`:

```python
from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora

fc1_adapter = make_per_expert_lora(
    num_experts=8, rank=16, in_dim=2048, out_dim=5632,
    dtype=torch.bfloat16,
)
# fc1_adapter["A"].shape == (8, 16, 2048)  -- independent per expert
# fc1_adapter["B"].shape == (8, 5632, 16)  -- independent per expert
```

#### Execution modes

Routed-expert MoE LoRA always runs through a single capture-safe grouped-GEMM core (on-stream pointer expansion, problem building, and grouped GEMMs). Two input schemas feed that core, and both produce identical results:

- **Eager** uses the per-request input schema: the op expands the per-request adapter tables into per-token `(rank, A, B)` arrays before launching the core. This mode handles both context (prefill) and decode batches.
- **CUDA-graph decode** uses the slot-indexed input schema: `CudaGraphLoraManager` maintains stable per-slot adapter tables and a `token_to_slot` map, and the slot→token expansion runs entirely on the stream. Because the tables live at stable addresses and are refreshed in place, reassigning a slot's adapter is reflected on replay without re-capture. CUDA-graph decode is generation-only.

The per-request schema is not itself CUDA-graph capturable (its host-side adapter expansion would be frozen at capture time), so under capture the op uses the slot-indexed schema; supplying per-request inputs while capturing raises a clear error.

#### What is rejected, and where

If you supply MoE LoRA on a non-Cutlass backend or with unsupported base quantization, `create_moe` raises at construction with a message pointing at the offending setting. At runtime, the fused MoE op also rejects min-latency mode + LoRA, alltoall + LoRA, and per-request (eager-schema) inputs under CUDA-graph capture (use the slot-indexed schema for capture).

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

### Options for Performance Tuning

`LoraConfig` exposes several options for performance tuning depending on the user's workload. The first is `overlap_lora_and_base`:
setting this to `True` enables concurrent execution of LoRA adapter and base model computations, which are scheduled on
separate CUDA streams. This allows to significantly reduce the latency overhead introduced by LoRA, especially in low-concurrency
(i.e., memory-bound) regimes. This option covers the LoRA path for the `mlp`, `gated_mlp`, `attention` and `linear` modules, but does
not apply to MoE LoRA.

The second option is `cuda_graph_specialize_lora`: enabling this triggers the capture of two sets of CUDA graphs, having the LoRA path
enabled or disabled for all requests in the batch; by default, a single set of CUDA graphs is captured with the LoRA path active, using
dummy requests without `lora_task_id`, with empty adapter slots that have rank `0` and zeroed weight pointers. This option is particularly 
effective when the majority of requests in the workload does not use LoRA, since it allows most batches to follow the non-LoRA path and 
therefore eliminate all of the associated overhead.

However, it should be noted that enabling `cuda_graph_specialize_lora` will result in a slight memory overhead due to the additional
number of CUDA graphs that need to be captured and stored (about 50MB for 21 generation graphs with Qwen3 32B).

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
