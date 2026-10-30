<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sampling

The PyTorch backend supports most of the sampling features that are supported on the C++ backend, such as temperature, top-k and top-p sampling, stop words, bad words, penalty, context and generation logits, and log probs.

The following example prepares two identical prompts which will give different results due to the sampling parameters chosen:

```python
from tensorrt_llm import LLM
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

When using speculative decoders such as MTP or Eagle-3 the subset of sampling options available is more restricted.

## Context logits and prompt log probabilities

`SamplingParams(return_context_logits=True)` requests the full vocabulary logits
for every prompt position. The returned tensor can be large: an 8192-token prompt
with a 131072-token vocabulary requires 4 GiB of FP32 result storage.

For ordinary, non-speculative decoding with a dense LM head and the standard
logits processor, decoder warmup reserves and initializes a 32-row projection
workspace before KV capacity estimation. Both single-rank heads and uniformly
vocabulary-sharded tensor-parallel heads with gathered outputs are supported.
The context-output path prepares request storage before projecting hidden states. It first attempts GPU
storage and falls back to CPU storage if that allocation raises a CUDA OOM.
Consequently, callers must support either device for returned context logits.
CPU storage still requires enough host memory for the full result.

When GPU result storage succeeds, projection attempts a larger temporary output
and halves its row count after allocation failure, with a bounded number of
attempts. It then falls back to the resident workspace. CPU result storage uses
the resident workspace directly. Copies from reusable GPU scratch to CPU storage
complete before that scratch is overwritten. Mixed batches project all context
rows only for requests that need them; other requests retain their final row.

For tensor parallelism, each rank reserves local projection, all-gather receive,
and full-vocabulary FP32 conversion buffers. All ranks agree on the smallest
successful chunk capacity before projecting, so allocation failure or CPU result
storage on one rank cannot change the collective sequence independently. The
all-gather writes directly into the reserved receive buffer; vocabulary assembly
and padding removal copy into the conversion buffer without allocating a second
gathered output. Allocation failures for required sampling rows and GEMM failures
are reported on all TP ranks before the next logits collective. These failures
abort the operation; they do not trigger an independent per-rank retry. TP
capacity and failure agreement introduces additional synchronization.

Requests that only specify `prompt_logprobs` retain the requested probabilities
and token IDs rather than a full context-logits tensor. Projection chunks are
reduced as they are produced. The legacy projection path uses the same probability
accumulation, so changing batch composition between prefill chunks does not change
the result format or lose positions. One final vocabulary row is retained on the CPU
until the first generated token is known. Context probability postprocessing is
also chunked when consuming complete CPU or GPU logits; a GPU reduction allocation
failure falls back to CPU for that chunk.

The bounded projection path does not apply to context/attention-data parallel
execution, ROW-parallel or unevenly sharded LM heads, speculative decoding,
quantized or LoRA LM heads, locality partitioned heads, Gemma4 softcapping,
Cohere2 scaling, or custom/model-specific logits processors.
Unsupported configurations do not reserve the projection workspace. These paths
retain their existing projection and CUDA graph selection behavior. Eager
execution is required only when the bounded projection plan is active. The
workspace bounds projection, gather-output and dtype-conversion storage; it does
not guarantee that model activations, sampling outputs, or unrelated runtime
allocations fit. Ordinary generation without
context outputs continues to use the existing LM-head path.
