<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Beam sampling CUDA graphs

Beam sampling CUDA graphs are opt-in and independent of the model-forward
`cuda_graph_config`. The default (`beam_search_cuda_graph_config=None`) keeps
sampling eager and allocates no beam graph buffers.

The initial implementation supports one GPU, a decoder-only model, and the
synchronous `TorchSampler`. The overlap scheduler can remain enabled. Async
sampler workers, speculative decoding, encoder-decoder execution, and KV
transfer/connector execution do not use these graphs. Startup must be quiescent:
application CUDA threads and concurrent model construction must finish before
starting an executor with this option. Another live native executor prevents
beam capture in the new executor.

## Configuration

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import BeamSearchCudaGraphConfig

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_batch_size=4,
    max_beam_width=4,
    max_seq_len=1024,
    beam_search_cuda_graph_config=BeamSearchCudaGraphConfig(
        profiles=[
            {"batch_sizes": [1, 2], "beam_widths": [4], "max_tokens": 8},
        ],
        max_graphs=192,
        max_static_input_bytes=512 * 1024 * 1024,
    ),
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(n=4, best_of=4, use_beam_search=True, max_tokens=8),
)
llm.shutdown()
```

The same nested configuration is accepted by `trtllm-serve --config`:

```yaml
max_batch_size: 4
max_beam_width: 4
max_seq_len: 1024
beam_search_cuda_graph_config:
  profiles:
    - batch_sizes: [1, 2]
      beam_widths: [4]
      max_tokens: 8
  max_graphs: 192
  max_static_input_bytes: 536870912
```

- `batch_sizes` are exact counts of requests in a sampling group, excluding
  beam rows. There is no beam-sampling batch padding.
- `beam_widths` describes output widths for successive generated tokens. One
  width repeats for `max_tokens` steps. A variable schedule must provide exactly
  `max_tokens` widths, for example `[20, 300, 500]` with `max_tokens: 3` and
  `max_beam_width: 500`.
- `early_stopping` is `0` (false), `1` (true, the default), or `2` (never).
- Requests without logprobs and with raw logprobs use the same graphs.
  Processed logprobs are not prepared and take the existing eager path.
- The initial profiles cover zero length/diversity penalties. Nonzero
  per-request penalties take the eager path. Temperature scaling remains
  outside the captured region.

Graphs are prepared before the final executor worker starts; the temporary KV
capacity-estimation executor skips beam capture. Preparation restores persistent
sampling state before admitting requests. Uncovered batches, steps, parameters,
or changed state addresses execute eagerly. Requests never capture new graphs.
An `ignore_eos` setting does not change graph eligibility; EOS and stop-word
state remain inputs to the native algorithm.

## Memory and failures

`max_graphs` bounds prepared entries. Profiles are considered in declaration
order. `max_static_input_bytes` bounds persistent logits staging storage, shared
where shapes permit. It is **not** a total GPU-memory limit: graph scratch,
persistent sampler state, temporary warmup inputs/state backups and a private
1 MiB radix workspace also consume memory. Leave room for those allocations
when sizing the KV cache.

Shape or input-budget misses execute eagerly. Capture failures are reported and
the affected entry remains eager. A replay failure is propagated and invalidates
that sampler's runner: retrying eagerly could apply a state update twice.
Shutdown drains submitted work before releasing graph buffers and workspaces.

## Reproduction

Use a TensorRT-LLM build matching the checkout. Set `LLM_MODELS_ROOT` for tests
requiring weights. The sampler component suite does not download a model:

```bash
pytest tests/unittest/_torch/sampler/test_beam_search_cuda_graph.py
pytest -m cpu_only tests/unittest/llmapi/test_beam_search_cuda_graph_config.py
pytest -m cpu_only tests/unittest/llmapi/test_trtllm_serve_config_schemas.py
```

The included benchmark uses the public `Qwen/Qwen3-0.6B` checkpoint; `--model`
can point to an already downloaded copy. Validate small beams, fallback cases,
raw logprob output and variable large beams before measuring performance:

```bash
python examples/llm-api/beam_search_cuda_graph.py --mode graph --suite small \
  --shadow --fallbacks --warmup 2 --repeats 3 --output shadow-small.json
python examples/llm-api/beam_search_cuda_graph.py --mode graph --suite small \
  --shadow --raw-scores --warmup 2 --repeats 3 --output shadow-raw.json
python examples/llm-api/beam_search_cuda_graph.py --mode graph --suite large \
  --shadow --warmup 2 --repeats 3 --output shadow-large.json
```

Shadow mode compares graph/eager outputs and the complete persistent sampler
state on identical logits from actual model forwards. It runs outside the timed
loop. Independent end-to-end runs can still have model-forward floating-point
variation; the benchmark retains output hashes and token sequences to inspect it.

Run eager/graph modes in alternating **separate processes**, after compilation
caches are warm. Repeat multiple process pairs and rotate case order:

```bash
python examples/llm-api/beam_search_cuda_graph.py --mode eager --suite large \
  --warmup 12 --repeats 60 --rotate 0 --output eager-0.json
python examples/llm-api/beam_search_cuda_graph.py --mode graph --suite large \
  --warmup 12 --repeats 60 --rotate 0 --output graph-0.json
```

This is a single-client, closed-loop latency benchmark, with each batch completed
before the next is submitted. It is not an offered-QPS or concurrent-serving
throughput test. Compare process-level medians, report uncertainty across
independent processes, and retain the JSON files. They include the source commit,
script hash, startup time, allocation/reservation peaks, graph coverage and
shutdown checks. Startup time includes model loading and all warmup; report
whether compilation caches were warm. Performance is workload-dependent and is
not promised for small beams or uncovered profiles.
