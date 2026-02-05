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
   - [Cache Management](#cache-management)
4. [TRTLLM serve with LoRA](#trtllm-serve-with-lora)
   - [YAML Configuration](#yaml-configuration)
   - [Starting the Server](#starting-the-server)
   - [Client Usage](#client-usage)
5. [TRTLLM bench with LORA](#trtllm-bench-with-lora)
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

Create an `extra_llm_api_options.yaml` file:

```yaml
lora_config:
  lora_target_modules: ['attn_q', 'attn_k', 'attn_v']
  max_lora_rank: 8
```
### Starting the Server
```bash
python -m tensorrt_llm.commands.serve
     /path/to/model \
    --extra_llm_api_options extra_llm_api_options.yaml
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

## TRTLLM bench with LORA

### YAML Configuration

Create an `extra_llm_api_options.yaml` file:

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
trtllm-bench --model $model_path throughput --dataset $dataset_path --extra_llm_api_options extra_llm_api_options.yaml --num_requests 64 --concurrency 16
```
