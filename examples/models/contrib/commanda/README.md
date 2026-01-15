# Cohere Command-A

This document shows how to build and run [Cohere Command-A](https://huggingface.co/CohereLabs/c4ai-command-a-03-2025) models.

## Hardware
With 111B parameters, the command-a model will require more than 60GB of GPU memory for 4-bit weights and over 220GB for full precision.

## Support Matrix
  * INT8 weight-only
  * INT4 weight-only
  * Tensor Parallel

Other options have not been tested except for Pipeline Parallel, which does not work.

## TensorRT-LLM Backend Usage

```python
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import KvCacheConfig

# max attention window is 4096 for sliding attention layers and 131072 for full attention layers, where 131072 is the max limit for position embeddings.
# Sliding window attention is alternated with full attention in a 3:1 ratio.
# from: https://huggingface.co/CohereLabs/c4ai-command-a-03-2025/blob/main/config.json
kvc_cfg = KvCacheConfig(max_attention_window=[4096, 4096, 4096, 131072])
# max_attention_window (RuntimeDefaults) cannot be serialized to config.json
# until serialization is added to the nanobind definitions. This parameter will
# be necessary at inference time if sequence lengths exceeds 4096.
model = LLM("CohereLabs/command-a-translate-08-2025", kv_cache_config=kvc_cfg)
output = model.generate("Hello from Command-A")

```
