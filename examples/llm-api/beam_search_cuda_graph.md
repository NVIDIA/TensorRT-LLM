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

The service declares capture ranges at startup. Requests retain their own
sampling parameters and generation limits. Providing
`beam_search_cuda_graph_config` opts in; omitting it or setting it to `None`
keeps beam sampling eager.

```yaml
max_batch_size: 64
max_beam_width: 500
max_seq_len: 1024
beam_search_cuda_graph_config:
  max_batch_size: 16
  capture_configs:
    - beam_width_array: [4]
      max_capture_tokens: 8
    - beam_width_array: [20, 300, 500]
      max_capture_tokens: 3
```

Pass this YAML with `trtllm-serve --config`. The service still admits batches
up to 64; beam capture considers request groups up to 16. Both schedules share
the batch sizes and resource budgets. Preparation is bounded by those budgets,
so declaring a range does not guarantee every entry will be captured.

The same nested configuration is accepted by the Python API:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import BeamSearchCudaGraphConfig

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_batch_size=4,
    max_beam_width=4,
    max_seq_len=1024,
    beam_search_cuda_graph_config=BeamSearchCudaGraphConfig(
        capture_configs=[
            {"beam_width_array": [4], "max_capture_tokens": 8},
        ],
    ),
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(n=4, best_of=4, use_beam_search=True, max_tokens=12),
)
llm.shutdown()
```

This prepares the first eight steps, subject to resource limits. The request
can still generate up to twelve tokens; uncovered steps execute eagerly.

- `batch_sizes` optionally lists exact counts of requests in a sampling group,
  excluding beam rows. Sizes are sorted and deduplicated. There is no
  beam-sampling batch padding, and these sizes are independent of the
  model-forward `cuda_graph_config`.
- `max_batch_size` optionally sets the beam capture limit. With explicit
  `batch_sizes`, it defaults to their maximum; if both are specified, they
  must agree. Without a list, sizes use the existing decode CUDA graph policy
  with padding disabled: dense sizes through 32, then sparse sizes, always
  including the chosen maximum. When both options are omitted, the limit is
  `min(128, executor max_batch_size)`. Explicit sizes or limits exceeding the
  executor capacity are rejected.
- Each `capture_configs` entry supplies a non-decreasing `beam_width_array`.
  A single width describes fixed-width beam search. As with the request
  parameter of the same name, steps past the array repeat its last width.
  For example, `[2, 4]` with `max_capture_tokens: 4` prepares `[2, 4, 4, 4]`.
  A longer array is truncated for capture; all its widths must still fit
  `max_beam_width`.
- `max_capture_tokens` is the number of generation steps to prepare for that
  schedule, not a request generation limit. It must be smaller than the
  executor's `max_seq_len`. A request's `max_tokens` remains independent.
- `early_stopping` selects `0` (false), `1` (true, the default), or `2` (never).
  To prepare another mode, add a capture configuration for it.
- Requests without logprobs and with raw logprobs use the same graphs.
  Processed logprobs are not prepared and take the existing eager path.
- Capture configurations cover zero length/diversity penalties. Nonzero
  per-request penalties take the eager path. Temperature scaling remains
  outside the captured region.

Graphs are prepared before the final executor worker starts; the temporary KV
capacity-estimation executor skips beam capture. Preparation restores persistent
sampling state before admitting requests. Uncovered batches, steps, parameters,
or changed state addresses execute eagerly. Requests never capture new graphs.
An `ignore_eos` setting does not change graph eligibility; EOS and stop-word
state remain inputs to the native algorithm.

## Memory and failures

`max_graphs` (default 192) bounds prepared entries. Capture configurations
are considered in declaration order, with batch sizes in ascending order. `max_static_input_bytes` (default 512 MiB) bounds persistent logits staging storage, shared
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
