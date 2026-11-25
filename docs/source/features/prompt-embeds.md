# Prompt Embeddings

> **Prototype.** The prompt embedding inputs may change in later releases.

`LLM.generate` and `LLM.generate_async` accept precomputed input embeddings for some or all prompt positions, on the PyTorch backend. The rows replace the token embeddings at those positions; everything after the embedding layer is unchanged. Use this to feed embeddings produced outside the model (for example soft prompts or a separate encoder) without writing a custom model.

## Usage

A prompt given entirely as embeddings:

```python
import torch
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-8B")
rows = torch.load("prompt_rows.pt")  # [num_tokens, hidden_size], fp16/bf16/fp32
out = llm.generate({"prompt_embeds": rows}, SamplingParams(max_tokens=32))
```

Embeddings for some positions of a token prompt: `prompt_embeds_mask` marks, in order, the positions that take the next row. The ids at those positions are placeholders.

```python
item_rows = torch.load("item_rows.pt")  # [num_item_rows, hidden_size], e.g. from a product encoder
prefix = llm.tokenizer.encode("Describe the product:")
suffix = llm.tokenizer.encode(" in one sentence.", add_special_tokens=False)
ids = prefix + [0] * len(item_rows) + suffix  # 0 is the placeholder id at the item positions
mask = [False] * len(prefix) + [True] * len(item_rows) + [False] * len(suffix)
out = llm.generate(
    {"prompt_token_ids": ids, "prompt_embeds": item_rows, "prompt_embeds_mask": mask},
    SamplingParams(max_tokens=32),
)
```

On Qwen2-VL, Qwen2.5-VL and Qwen3-VL the same keys combine with `multi_modal_data`. The mask then indexes the ids before the image or video placeholder expands, and must not select a placeholder.

## Supported models

A model accepts prompt embeddings when its class sets `supports_prompt_embeds = True`; the request is rejected on any other model, and the error names the reason. Opted-in models include the Llama, Mistral and Mixtral, Qwen2 to Qwen3.5 (dense, MoE and VL), Gemma3 and Gemma4, DeepSeek-V3, GLM, GPT-OSS, Nemotron and Phi-3 families. `modeling_v2` implementations do not accept them yet.

## Behaviour

- Rows replace the input embedding layer's output at their positions, so they must be in that layer's output space, including any scaling it applies. They are copied when the request is submitted.
- Placeholder ids still feed repetition penalties and prompt logprobs. A prompt given only as embeddings schedules id 0 at every position.
- KV cache blocks are reused only across identical ids and rows. With the V2 KV cache manager and no KV connector, blocks before the first row are shared with plain-token prompts; otherwise the whole request is scoped by a salt derived from the rows. KV cache events from the V2 manager report embedding positions as hexadecimal row keys in `token_ids`.
- Gemma4 per-layer embeddings read the token id after the embedding layer; at embedding positions they see the pad id, so the output can depend on that choice.

## Limitations

The request is rejected with speculative decoding, attention data parallelism, context parallelism, a model that does not support prompt embeddings, rows whose hidden size differs from the model's, or rows outside the range of a float16 or bfloat16 model's dtype. Disaggregated serving requires `kv_cache_config.use_kv_cache_manager_v2=True`. The context result's `disaggregated_params` carries the identity of the rows, so the generation-only request needs only the prompt token ids; rows it passes anyway must match. With `schedule_style` `generation_first` the generation request arrives before any context result, so it must carry the same rows as the context request. CSA2 remote-tail replay (DeepSeek-V4.1 disaggregated serving) does not take prompt embeddings. `LLM.encode` and `LLM.preprocess` do not accept prompt embeddings.
