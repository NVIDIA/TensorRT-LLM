# Beam with ModelingV2

This experimental target runs the complete 52-layer Beam model through TRT-LLM's
native PyTorch executor using the StairCase/ModelingV2 approach. It includes dense
and MoE decoder blocks, checkpoint loading, cached generation, shared-prefix KV
reuse, optional DFlash2 speculative decoding and an optional FP8 KV cache. It is
served by the standard `trtllm-serve` command or the Python `LLM` API with
`modeling_v2: require`, which does not fall back to the built-in model
implementation.

## Supported deployment

| Setting | Supported configuration |
|---|---|
| Hardware | Four GB300 GPUs (SM103) per node |
| Parallel layout | Attention TP4, MoE TP4/EP1, PP1; attention DP disabled |
| Parameters and arithmetic | Packed NVFP4 experts with native W4A4 GEMMs; BF16 excluded modules/activations; FP32 routing |
| Attention | TRTLLM backend; 512-token sliding attention and full attention according to the checkpoint's layer schedule |
| KV cache | BF16 by default; opt-in FP8 E4M3 for target and drafter with explicit per-layer scales; native `KVCacheManagerV2`, full-block prefix reuse |
| Generation | Native `TorchSampler` for ordinary decoding; native DFlash worker and speculative sampler for optional DFlash2 greedy decoding |
| Execution | Eager; CUDA graphs, overlap scheduling, and chunked prefill disabled |
| Serving | `trtllm-serve`: one aggregated server, or a context and a generation worker behind `trtllm-serve disaggregated`; streaming and non-streaming text |

The expert provider currently dispatches selected experts in a host-driven loop.
The GEMMs use native NVFP4 kernels, but expert dispatch and rank agreement introduce
synchronization. This is a functional baseline, not a throughput-optimized MoE path.

## Launch and request

Use a TensorRT-LLM installation built from the commit containing this target, with
matching native libraries and access to the four GPUs of a node. Supply an
authorized, compatible checkpoint directory containing its original configuration,
tokenizer, chat template, and indexed safetensors weights. No checkpoint download
or conversion is performed.

The curated recipes under `examples/configs/curated/` hold the qualified
configuration: the NVFP4 target with the DFlash2 drafter and FP8 target/draft KV
caches, aggregated (`beam-nvfp4-dflash2-fp8kv-aggregated.yaml`) or disaggregated
(`beam-nvfp4-dflash2-fp8kv-disagg-ctx.yaml` and `-disagg-gen.yaml`). The
[deployment guide](../../../../../../docs/source/deployment-guide/deployment-guide-for-beam-on-trtllm.md)
gives their inputs, limits, launch, request, verification and shutdown steps, and
the equivalent `LLM()` example. Aggregated, after replacing `<DRAFTER_CHECKPOINT>`
in a copy of the recipe:

```bash
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export TRTLLM_MODELING_V2_KV_CACHE_SCALES=/path/to/kv-scales.json
trtllm-serve /path/to/checkpoint \
  --config beam-nvfp4-dflash2-fp8kv-aggregated.yaml \
  --chat_template /path/to/checkpoint/chat_template.jinja \
  --served_model_name beam-nvfp4 --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"beam-nvfp4","messages":[{"role":"user","content":"What is 12 plus 7?"}],"temperature":0,"max_tokens":4096,"stream":false,"chat_template_kwargs":{"reasoning_effort":128}}'
```

Set `stream` to `true` for streaming. Analysis-channel text is returned as
`reasoning_content`; final-channel text is returned as `content`. `max_tokens`
includes reasoning tokens, so a short budget can terminate before final-channel
text is produced. The request above is an invocation example, not a promise of a
particular response or token length.

The recipes reach this target through TensorRT LLM's shared registries:

- `checkpoint_format: beam_nvfp4` is registered in
  `tensorrt_llm/_torch/models/checkpoints/beam/checkpoint_loader.py`, with the HF
  weight and configuration loaders; it builds this target's checkpoint loader
  (`full_model_nvfp4__sm_103__tp4/weights.py`), which reads the FP8 scale table
  named by `TRTLLM_MODELING_V2_KV_CACHE_SCALES`.
- `custom_tokenizer: tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer`
  loads the checkpoint's fast tokenizer by its concrete class.
- `reasoning_parser: beam_harmony_text` is the channel parser in
  `tensorrt_llm/llmapi/reasoning_parser.py`.

The sequence limit includes prompt and generated tokens. With chunked prefill off,
the prompt must also fit the per-step token budget; the model accepts at most 8192
new token rows per call (`max_num_tokens`), and selection refuses a larger budget.
Configured sequence capacity does not establish accuracy at the maximum sequence
length. Shared KV capacity must cover concurrently active requests; do not infer
capacity from batch size alone.

For ordinary decoding with the default BF16 KV cache, start from the aggregated
recipe without `speculative_config`, with `kv_cache_config.dtype: auto`, and leave
`TRTLLM_MODELING_V2_KV_CACHE_SCALES` unset: a BF16 cache takes no scale table. The
GSM8K evaluation below used that configuration with `max_batch_size: 128` and
`kv_cache_config.max_tokens: 1048576`, with workload-specific memory budgeting.
`enable_iter_perf_stats: true` enables native iteration statistics.

## Optional DFlash2 speculative decoding

`speculative_config` with `decoding_type: DFlash`, `max_draft_len: 7`,
`attention_backend: TRTLLM` and a compatible DFlash2 drafter directory as
`speculative_model` adds the native DFlash worker and greedy verifier. The target
supplies completed-block captures at the layers named by the drafter; the drafter
shares the target's embedding and vocabulary head. The drafter's five KV layers
are placed in the target's `KVCacheManagerV2` after the target's 52 layers, so one
KV-cache capacity holds both models' context. Seven proposals per step (eight
target verification rows) is the only draft length the target serves; selection
refuses other draft lengths, schedules, rejection sampling and the overlap
scheduler.

Non-causal windowed draft block attention uses Torch SDPA over the drafter's
paged KV layers because TRTLLM-Gen does not support that mask. Target verification
retains TRTLLM attention. Accepted proposals must match the target's greedy choice
on the verification logits; independent ordinary and speculative runs can still
diverge at near ties when GEMM/collective shapes change.

CUDA graphs and overlap scheduling remain future work. The bounded eager timing
measurements are not a production throughput claim.

## Optional FP8 KV cache

`kv_cache_config.dtype: fp8` stores both target and drafter KV caches in FP8 E4M3;
ordinary decoding also supports it. The per-layer scales come from the JSON table
named by `TRTLLM_MODELING_V2_KV_CACHE_SCALES` (or a checkpoint loader instance
built with `kv_cache_scales`). BF16 remains the default. Cache precision does not
change the NVFP4 expert weights.

The scale file supplies a `target` object and, when using DFlash2, a `draft`
object. Each contains `layers` and a `scales` map keyed by every logical layer
index as a string. Each value is a positive scale `s`, shared by K and V:
storage quantizes `x / s`, and reads recover `code * s`. Supply scales validated
for the selected checkpoint and workload; NVFP4 weight scales are not KV-cache
scales. Neither checkpoint weights nor a calibrated scale artifact are included
here. Missing scales, a table given with a BF16 cache, unsupported cache formats,
ranks holding different tables, and scales or reciprocals that are not finite
normal FP32 values are rejected.

KV storage halves at equal token capacity without a persistent BF16 shadow cache.
Generated-input tests exercise scaled writes/reads, window boundaries, rollback,
isolation, and bad scale detection. This does not establish a throughput
improvement.

FP8 can change greedy token choices and serving responses relative to BF16 KV.
Some native target attention outputs exceed the original FP32-reference limits
(relative RMS 0.0314 and worst-row normalized maximum error 0.142). Isolated
native-operator replays reproduce those outputs bitwise; query and intermediate
probability rounding in the native FP8 kernel explains the localized deviation.
The bounded implementation was accepted with an explicit numerical exception;
the strict failures are retained, and no kernel correction is claimed. Evaluate
quality and performance for the intended workload before promoting FP8 to a
recommended default.

## Disaggregated serving with DFlash2 and prefix reuse

The disaggregated recipes use **two nodes with four GB300 GPUs each**: one TP4
context worker and one TP4 generation worker, both running the same target,
drafter, tokenizer and FP8 KV scale table, behind the native
`trtllm-serve disaggregated` router (see the
[disaggregated serving guide](../../../../../../docs/source/features/disagg-serving.md)).
The Python NIXL transceiver moves each request's KV cache, including the drafter's
layers, plus the draft-context cursor. The generation worker continues from that
state without replaying the target prompt. Peer checks compare checkpoint
manifests, layout, cache layers and scale digests; they do not hash all weight
bytes, so provision identical checkpoint contents on both workers. Both workers
must keep the drafter's layers in the target's cache (the default); with
`TRTLLM_DISAGG_SHARED_DRAFT_KV=0` they refuse to start. Omit `speculative_config`
on both workers for ordinary decoding.

**Limits:** greedy outputs can differ between batch sizes, between fresh and
reused-cache execution, between speculative and ordinary decoding, and between
aggregated and disaggregated serving, because the target's own numerics differ
across these execution shapes; exact token parity across these modes is not
established. Context-side timeout observation can be delayed while its loop is idle; the
configured transfer deadline is not a hard wall-clock response guarantee. The
inherited FP8 numerical exception remains in force, and BF16 stays the default.

## Validation and regression coverage

The initial full-model evaluation (ordinary decoding, BF16 KV cache) completed all
1,319 GSM8K test questions with 10-shot prompting, greedy decoding, reasoning
effort 128, and a 65,536-token completion allowance. It scored
**1,263/1,319 (95.75%)**, with no execution errors or truncated completions.
Evaluation used disjoint subsets on two four-GPU replicas, followed by resumptions
of unfinished questions; completed samples were retained and each question was
scored once. It does not qualify the DFlash2 + FP8 KV recipes.

The comparison baseline is the recorded BF16 mean of **95.53%** across three runs,
not a same-precision NVFP4 reference. Precision, runtime, and batching differ.
Greedy token trajectories are not guaranteed to be identical across batch sizes,
and this small score difference is not evidence that quantization improves
accuracy. Private checkpoint identifiers, evaluation traces, and reference captures
are not part of this source distribution.

Portable generated-input tests cover the component implementations and their
composition. The full-model directory contains CPU parser/routing/registration and
recipe tests and GPU metadata/request-row and disaggregated-handoff tests:

```bash
python -m pytest -m cpu_only \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_serving.py \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_serving_cli.py
python -m pytest \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_batching.py \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_disagg_handoff.py
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
be registered in L0, nightly or long-running CI.

## File organization and limits

- `full_model_nvfp4__sm_103__tp4/` contains the engine-facing target, full-depth
  execution and the target's checkpoint loader (`weights.py`).
- `tensorrt_llm/_torch/models/checkpoints/beam/` registers the `beam_nvfp4`
  checkpoint format and holds the checkpoint's tokenizer class;
  `tensorrt_llm/llmapi/reasoning_parser.py` holds the `beam_harmony_text` parser.
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
