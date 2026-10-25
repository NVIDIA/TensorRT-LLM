# Beam with ModelingV2

This experimental target runs the complete 52-layer Beam model through TRT-LLM's
native PyTorch executor using the StairCase/ModelingV2 approach. It includes dense
and MoE decoder blocks, checkpoint loading, cached generation, shared-prefix KV
reuse, and a text chat-completions endpoint. The supported launch below requires
ModelingV2 explicitly; it does not fall back to the built-in model implementation.

## Supported deployment

| Setting | Supported configuration |
|---|---|
| Hardware | Four GB300 GPUs (SM103) on one node |
| Parallel layout | Attention TP4, MoE TP4/EP1, PP1; attention DP disabled |
| Parameters and arithmetic | Packed NVFP4 experts with native W4A4 GEMMs; BF16 excluded modules/activations; FP32 routing |
| Attention | TRTLLM backend; 512-token sliding attention and full attention according to the checkpoint's layer schedule |
| KV cache | BF16, native `KVCacheManagerV2`, full-block prefix reuse |
| Generation | Native `TorchSampler`; greedy decoding is the evaluated policy |
| Execution | Eager; CUDA graphs, overlap scheduling, and chunked prefill disabled |
| Serving | Native OpenAI-compatible server, one frontend, streaming and non-streaming text |

The expert provider currently dispatches selected experts in a host-driven loop.
The GEMMs use native NVFP4 kernels, but expert dispatch and rank agreement introduce
synchronization. This is a functional baseline, not a throughput-optimized MoE path.

## Launch and request

Use a TensorRT-LLM installation built from the commit containing this target, with
matching native libraries and access to all four supported GPUs. Supply an
authorized, compatible checkpoint directory containing its original configuration,
tokenizer, chat template, and indexed safetensors weights. No checkpoint download
or conversion is performed by this recipe.

```bash
python -m tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving \
  --checkpoint /path/to/checkpoint \
  --served-model-name beam-nvfp4 \
  --host 127.0.0.1 --port 8000 \
  --max-seq-len 69632 --max-num-tokens 8192 \
  --max-batch-size 1
```

The launcher selects the target, checkpoint loader, tokenizer, channel parser,
cache manager, and execution settings together. Plain `trtllm-serve` with default
settings is not an equivalent launch recipe. Analysis-channel text is returned as
`reasoning_content`; final-channel text is returned as `content`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"beam-nvfp4","messages":[{"role":"user","content":"What is 12 plus 7?"}],"temperature":0,"max_tokens":4096,"stream":false,"chat_template_kwargs":{"reasoning_effort":128}}'
```

Set `stream` to `true` for streaming. `max_tokens` includes reasoning tokens, so a
short budget can terminate before final-channel text is produced. The request above
is an invocation example, not a promise of a particular response or token length.

The sequence limit includes prompt and generated tokens. With chunked prefill off,
the prompt must also fit the per-step token budget; the model accepts at most 8192
new token rows per call. Configured sequence capacity does not establish accuracy
at the maximum sequence length. The launcher defaults to one request per step;
native batching can be enabled with `--max-batch-size`. The benchmark used 128 and
`--kv-cache-max-tokens 1048576`, with workload-specific memory budgeting. Shared KV
capacity must cover concurrently active requests; do not infer capacity from batch
size alone. `--iter-stats` enables native iteration statistics.

## Validation and regression coverage

The initial full-model evaluation completed all 1,319 GSM8K test questions with
10-shot prompting, greedy decoding, reasoning effort 128, and a 65,536-token
completion allowance. It scored **1,263/1,319 (95.75%)**, with no execution errors
or truncated completions. Evaluation used disjoint subsets on two four-GPU
replicas, followed by resumptions of unfinished questions; completed samples were
retained and each question was scored once.

The comparison baseline is the recorded BF16 mean of **95.53%** across three runs,
not a same-precision NVFP4 reference. Precision, runtime, and batching differ.
Greedy token trajectories are not guaranteed to be identical across batch sizes,
and this small score difference is not evidence that quantization improves
accuracy. Private checkpoint identifiers, evaluation traces, and reference captures
are not part of this source distribution.

Portable generated-input tests cover the component implementations and their
composition. The full-model directory contains CPU parser/routing tests and GPU
metadata/request-row tests:

```bash
python -m pytest -m cpu_only \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_serving.py
python -m pytest \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_batching.py
```

The second command requires a compatible CUDA environment but no checkpoint.
These unit tests do not reproduce the full 52-layer accuracy evaluation. A
checkpoint-backed end-to-end regression and its CI model staging remain separate
integration work; generated tests must not be presented as that coverage.

## File organization and limits

- `full_model_nvfp4__sm_103__tp4/` contains the engine-facing target, full-depth
  execution, native checkpoint loader adapter, and serving launcher/parser.
- `checkpoint_layout_nvfp4__sm_103__tp4/` supplies packed weights and expert kernels;
  it reuses BF16 layout/loading and block helpers from
  `checkpoint_layout_bf16__sm_103__tp4/`.
- Router, expert, dispatch/combine, dense/sparse block, and six-layer directories
  retain component implementations and generated-input regression coverage. Some
  are runtime dependencies of the full target, not disposable tier artifacts.
- `tensorrt_llm/_torch/models/modeling_beam.py` supplies `BeamConfig` and the earlier
  built-in component implementation. The result above belongs to the routed
  ModelingV2 target, not to that built-in model path.

Other GPU architectures, parallel layouts, FP8 KV cache, speculative decoding,
CUDA graphs, overlap scheduling, chunked prefill, tool calling, and disaggregated
serving are not qualified by this implementation. Subsequent feature/performance
work should preserve this functional and accuracy baseline in follow-up PRs.
