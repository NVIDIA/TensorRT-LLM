# TriAttention KV-Cache Compression

This document describes enabling TriAttention KV-cache compression in TensorRT LLM.

TriAttention is a training-free, decode-time KV-cache eviction method for long-context LLM inference. During generation it periodically scores the cached tokens by a trigonometric importance measure derived from offline per-head query statistics (calibration), keeps the most important `budget` tokens, and physically compacts the cache — reducing KV-cache memory so more sequences fit on a GPU at once.

For technical details see the paper [TriAttention](https://arxiv.org/abs/2604.04921) and the official implementation [github.com/WeianMao/triattention](https://github.com/WeianMao/triattention).

## Overview

TriAttention runs entirely in the generation phase and reuses the standard dense attention kernel over the compacted cache:

1. **Calibration (offline, one-time per model).** The importance score needs each attention head's mean and magnitude of the pre-RoPE query, gathered over a small calibration corpus. **TensorRT LLM does not compute calibration** — you produce it once with the official tool and pass the resulting `.pt` file. TensorRT LLM loads it and converts it to its runtime schema at the first request.
2. **Periodic eviction (Stage during generation).** Every `beta` confirmed generation tokens, once a sequence is over budget, TriAttention scores the whole cache, selects `budget` tokens to keep (the prompt tokens are preserved on top of the budget), and physically compacts the KV cache down to the kept set. A speculative iteration may confirm multiple tokens; crossing multiple periods in one update is coalesced into one eviction.

TriAttention is integrated into TensorRT LLM as a KV-cache compression manager on top of the `KVCacheManagerV2`. The scoring and compaction kernels are implemented in **Triton**.

## Support Matrix

* GPU Compute Capability >= 9.0 (Hopper or newer)
* FP16 / BF16
* Paged KV Cache (`KVCacheManagerV2`)
* Tensor Parallel
* PyTorch backend

**Notes:**
1. TriAttention requires `enable_block_reuse=False` in the KV-cache configuration — the eviction physically rewrites stored keys, which is incompatible with block reuse. The construction step rejects a cache manager that has block reuse enabled.
2. TriAttention requires the V2 KV-cache manager (`use_kv_cache_manager_v2=True`).
3. TriAttention does not compute calibration. Bring the official tool's calibration `.pt`; see [Calibration](#calibration).
4. Requires full-attention KVCacheManagerV2 lifecycles; attention-DP, disaggregated serving, native SWA/VSWA/SSM pools, and MLA caches are unsupported.

## Calibration

The calibration file is produced once per model with the official tool, then reused for every inference run with that model.

Generate the calibration file for your model with the official repository (for
example `qwen3-8b-calibration.pt` for Qwen3-8B), keep it anywhere on disk, and
point `calibration_path` at it:

```bash
# Clone + install the official tool
git clone https://github.com/WeianMao/triattention.git
cd triattention && pip install -e .

# Calibrate (writes the official {metadata, stats} .pt)
python3 scripts/calibrate.py \
    --model <path_to_model> \
    --input data/calibration_text.txt \
    --output <model>_calibration.pt \
    --max-length 32768 \
    --device cuda
```

TensorRT LLM accepts that file directly: it reads the official `{metadata, stats}` layout and derives the model's RoPE tables from the model config, then converts everything to its runtime schema at load. (An already-converted flat `.pt` is also accepted.)

## Usage

To enable TriAttention, pass a `TriAttentionKvCacheCompressionConfig` (the eviction knobs + the calibration file) to the `LLM` constructor. TriAttention is a pure compression method — there is **no** sparse-attention config and no custom attention backend; decode runs the model's standard attention over the compacted cache.

### Python API

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (KvCacheConfig,
                                 TriAttentionKvCacheCompressionConfig)

# 1. Configure the eviction manager + point it at the calibration file.
compression_config = TriAttentionKvCacheCompressionConfig(
    budget=2048,            # tokens kept at each eviction (prompt is kept on top)
    beta=64,               # eviction period, in confirmed generation tokens
    eviction_mode="union",
    calibration_path="/path/to/qwen3-8b-calibration.pt",  # official tool's output
    model_path="<path_to_model>",                # used to derive the RoPE tables
)

# 2. TriAttention needs the V2 KV-cache manager and block reuse disabled.
kv_config = KvCacheConfig(enable_block_reuse=False, use_kv_cache_manager_v2=True)

llm = LLM(
    model="<path_to_model>",
    backend="pytorch",
    kv_cache_compression_config=compression_config,
    kv_cache_config=kv_config,
)

# 4. Generate
prompts = ["To be or not to be, that is the question."]
sampling_params = SamplingParams(max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### Usage with `trtllm-bench` and `trtllm-serve`

Pass the configs via `--config config.yaml`. The field names match the Python configs:

```yaml
backend: pytorch
kv_cache_compression_config:
  algorithm: triattention
  budget: 2048
  beta: 64
  eviction_mode: union
  calibration_path: /path/to/qwen3-8b-calibration.pt
  model_path: <path_to_model>
kv_cache_config:
  enable_block_reuse: false
  use_kv_cache_manager_v2: true
```

```bash
trtllm-eval --model <path_to_model> --config config.yaml longbench_v2 --max_output_length 1024 ...
```

## Configuration Arguments

`TriAttentionKvCacheCompressionConfig` controls the compression ratio and the eviction algorithm:

* **`budget`** (int, default=2048): Tokens kept at each eviction. Prompt tokens are always preserved on top of this. Smaller `budget` → more compression.
* **`beta`** (int, default=128): Eviction period, in confirmed generation tokens (the upstream `divide_length`). Speculative acceptance advances the counter by `1 + accepted_draft_tokens`; at most one eviction is coalesced per final update.
* **`eviction_mode`** (str, default=`union`): Which token set each eviction keeps.
    * `union`: union of each KV head's top-B, re-ranked by the per-token max score. Matches the official base setting.
    * `per_head`: each KV head keeps its own set, shared across layers (mean of per-layer maxima).
    * `per_layer_perhead`: each head keeps its own set, fully independent per layer.
* **`normalize_scores`** (bool, default=True): Z-normalize each head's scores over the decode region before selection (upstream default). `union` eviction requires `True` (the fused union pipeline always z-normalizes; construction rejects `False`).
* **`calibration_path`** (str): Path to the calibration `.pt` from the official tool. Required — TensorRT LLM does not compute calibration.
* **`model_path`** (str): Checkpoint path, used to derive the model's RoPE tables when converting the official calibration file and to classify kernel-masked sliding-window (SWA) layers from the model config.
