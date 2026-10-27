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
| KV cache | BF16 by default; opt-in FP8 E4M3 for target and drafter with explicit per-layer scales; native `KVCacheManagerV2`, full-block prefix reuse |
| Generation | Native `TorchSampler` for ordinary decoding; native DFlash worker and speculative sampler for optional DFlash2 greedy decoding |
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

## Optional DFlash2 speculative decoding

Supply a compatible DFlash2 drafter directory with `--draft-checkpoint`. The
launcher uses the native DFlash worker, greedy verifier, and separate target/draft
KV cache pools. The target supplies completed-block captures at the layers named
by the drafter; the drafter shares the target's embedding and vocabulary head.

```bash
python -m tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving \
  --checkpoint /path/to/target-checkpoint \
  --draft-checkpoint /path/to/dflash2-checkpoint \
  --served-model-name beam-nvfp4 \
  --host 127.0.0.1 --port 8000 \
  --max-seq-len 8192 --max-num-tokens 8192 --max-batch-size 4
```

The evaluated configuration uses four GB300 GPUs, TP4, BF16 KV for both models,
seven proposals per step (eight target verification rows), and eager execution
without overlap or chunked prefill. Validation covers batches of one and four,
prompts through 4097 tokens, and at most 64 generated tokens; the configured 8192
sequence capacity is not a claim of qualification at every length. The default
shared KV capacity can queue a request when four long prompts compete for space.

Draft context uses the native paged cache. Non-causal windowed draft block
attention uses Torch SDPA over that same pool because TRTLLM-Gen does not support
that mask. Target verification retains TRTLLM attention. Accepted proposals must
match the target's greedy choice on the verification logits; independent ordinary
and speculative runs can still diverge at near ties when GEMM/collective shapes
change. The GSM8K result below is for ordinary decoding, not DFlash2.

CUDA graphs and overlap scheduling remain future work. Disaggregated serving is
covered separately below.
The bounded eager timing measurements are not a production throughput claim.

## Optional FP8 KV cache

Add `--kv-cache-dtype fp8 --kv-cache-scales /path/to/kv-scales.json` to the
bounded DFlash2 launch above to store both target and drafter KV caches in FP8
E4M3. Ordinary decoding also supports this option without `--draft-checkpoint`.
BF16 remains the default. Cache precision does not change the NVFP4 expert weights.

The scale file supplies a `target` object and, when using DFlash2, a `draft`
object. Each contains `layers` and a `scales` map keyed by every logical layer
index as a string. Each value is a positive scale `s`, shared by K and V:
storage quantizes `x / s`, and reads recover `code * s`. Supply scales validated
for the selected checkpoint and workload; NVFP4 weight scales are not KV-cache
scales. Neither checkpoint weights nor a calibrated scale artifact are included
here. Missing scales, unsupported cache formats, and scales or reciprocals that
are not finite normal FP32 values are rejected.

Validation covers the same four-GB300, TP4, eager deployment with batches 1/4,
prompts through 4097 tokens, and at most 64 generated tokens. KV storage halves
at equal token capacity without a persistent BF16 shadow cache. Independent
full-model reruns reproduced all 22 recorded FP8 outputs; generated-input tests
exercise scaled writes/reads, window boundaries, rollback, isolation, and bad
scale detection. This does not establish a throughput improvement.

FP8 can change greedy token choices and serving responses relative to BF16 KV.
Some native target attention outputs exceed the original FP32-reference limits
(relative RMS 0.0314 and worst-row normalized maximum error 0.142). Isolated
native-operator replays reproduce those outputs bitwise; query and intermediate
probability rounding in the native FP8 kernel explains the localized deviation.
The bounded implementation was accepted with an explicit numerical exception;
the strict failures are retained, and no kernel correction is claimed. The
initial diagnostic probe also has a reconstructed-source provenance limitation.
No FP8-KV GSM8K result or general answer-quality equivalence is claimed. Evaluate
quality and performance for the intended workload before promoting FP8 to a
recommended default.

## Disaggregated serving with DFlash2 and prefix reuse

The qualified deployment uses **two nodes with four GB300 GPUs each**: one TP4
context worker and one TP4 generation worker, both running the same target,
drafter, tokenizer and FP8 KV scale table. The Python NIXL transceiver moves both
target and draft KV caches, plus the draft-context cursor. The generation worker
continues from that state without replaying the target prompt. Peer checks compare
checkpoint manifests, layout and scale digests; they do not hash all weight bytes,
so provision identical checkpoint contents on both workers.

On the context node, launch:

```bash
python -m tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving \
  --checkpoint /path/to/target-checkpoint \
  --draft-checkpoint /path/to/dflash2-checkpoint \
  --kv-cache-dtype fp8 --kv-cache-scales /path/to/kv-scales.json \
  --served-model-name beam-nvfp4 --server-role context \
  --host 0.0.0.0 --port 8001 \
  --max-seq-len 8192 --max-num-tokens 8192 --max-batch-size 1 \
  --kv-transfer-timeout-ms 60000
```

On the generation node, use the same command with `--server-role generation`
and `--port 8002`. Both nodes need matching native libraries and network access
for NIXL and the worker endpoints. Omit `--draft-checkpoint` on both workers for
ordinary decoding. Route client requests through `trtllm-serve disaggregated -c
disagg_config.yaml`, using the context and generation worker addresses as described
in the [disaggregated serving guide](../../../../../../docs/source/features/disagg-serving.md).
The same streaming/non-streaming chat-completions request format applies.

Independent qualification covered ordinary decoding, DFlash2, shared-prefix reuse,
stops and completion budgets, request isolation, cancellation and transfer failures.
The fresh DFlash2 checks matched all 24 executed requests to the co-located FP8
reference; all 1,464 matched verifier rows were bitwise equal. KV transfer checks
compared 70,792,704 logical bytes with no mismatches. The configuration remains eager,
TP4/EP1/PP1 per worker, with no attention DP, overlap, graphs or chunked prefill.

**Limits:** batch-four ordinary and speculative runs can produce different tokens
from the co-located schedule. This was accepted as a documented limitation, not
exact-parity success; the differing operation remains unlocalized. Greedy verification
against each step's actual target logits passed. Context-side timeout observation
can be delayed while its loop is idle; the configured transfer deadline is not a
hard wall-clock response guarantee. These checks do not establish disaggregated
GSM8K accuracy, a throughput gain, other layouts or arbitrary feature combinations.
The inherited FP8 numerical exception remains in force, and BF16 stays the default.

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

L0 includes the CPU parser/routing and attention-selection checks, plus the
single-GPU generated-input operator, block, six-layer and metadata regressions.
The GPU tests are selected by the ModelingV2 directory entry in `l0_b300.yml`;
the six-layer suites need up to approximately 200 GiB of free device memory and
release their cached models at module teardown. Real-checkpoint multi-GPU smoke
coverage is outside L0 and may run manually or in a separately provisioned
nightly/long-running job. Full Beam GSM8K evaluation is manual-only and must not
be registered in L0, nightly or long-running CI. No Beam multi-GPU CI job is
registered by this change.

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

Other GPU architectures, parallel layouts, other speculative modes,
CUDA graphs, overlap scheduling, chunked prefill and tool calling are not
qualified by this implementation. Subsequent feature/performance
work should preserve this functional and accuracy baseline in follow-up PRs.
