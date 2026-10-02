# TRT-LLM Telemetry Schema Reference

Schema version: **0.8** | Client ID: `616561816355034` | Protocol: GXT Event Protocol v1.6

## Overview

TRT-LLM collects anonymous, session-level deployment telemetry to understand
how the library is used in production (GPU types, LLM and VisualGen parallelism,
and bounded model or pipeline categories). No PII, model weights, prompts, outputs, model paths, tokenizer
paths, or raw free-form configuration strings are collected.

**Opt-out** (any one of these disables telemetry):
- `TRTLLM_NO_USAGE_STATS=1`
- `TELEMETRY_DISABLED=true`
- `DO_NOT_TRACK=1`
- Create file `~/.config/trtllm/do_not_track`
- `TelemetryConfig(disabled=True)` in code

**Auto-disabled** in CI/test environments (detects `CI`, `GITHUB_ACTIONS`,
`JENKINS_URL`, `GITLAB_CI`, `PYTEST_CURRENT_TEST`, etc.). Override with
`TRTLLM_USAGE_FORCE_ENABLED=1` for staging deployments.

## GXT Envelope

Every payload is wrapped in a GXT v1.6 envelope. Dashboard builders will see
these top-level fields in Kibana alongside the event parameters.

| Field | Type | Description |
|-------|------|-------------|
| `clientId` | string | Always `"616561816355034"`. Identifies TRT-LLM in the GXT system. |
| `clientType` | string | Always `"Native"`. |
| `clientVer` | string | TRT-LLM version, e.g. `"1.3.0rc9"`. |
| `eventProtocol` | string | Always `"1.6"`. |
| `eventSchemaVer` | string | Schema version, currently `"0.8"`. |
| `eventSysVer` | string | Always `"trtllm-telemetry/1.0"`. |
| `sessionId` | string | Unique hex UUID per telemetry session. Use this to correlate initial, heartbeat, and terminal events. |
| `sentTs` | string | ISO 8601 UTC timestamp of when the payload was sent. |

Privacy/identity fields (`osVersion`, `geoInfo`, `deviceGUID`, etc.) are
hardcoded to `"undefined"` — TRT-LLM is a server-side SDK with no browser or
login context.

## Events

### `trtllm_initial_report`

Sent once after the first successful LLM initialization. Contains system info
and the first successfully reported LLM's serving configuration. A process that
fails earlier can send a terminal report without an initial report.

#### System fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `trtllmVersion` | ShortString | TRT-LLM package version. | `"1.3.0rc9"` |
| `platform` | LongString | OS platform string. | `"Linux-5.15.0-88-generic-x86_64"` |
| `pythonVersion` | ShortString | Python version. | `"3.12.3"` |
| `cpuArchitecture` | ShortString | CPU architecture. | `"x86_64"`, `"aarch64"` |
| `cpuCount` | PositiveInt | Number of logical CPUs (from `os.cpu_count()`). | `128` |

#### GPU fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `gpuCount` | PositiveInt | Number of GPUs **visible to the process** (`torch.cuda.device_count()`). Reflects `CUDA_VISIBLE_DEVICES`, not total system GPUs. | `8` |
| `gpuName` | LongString | Name of GPU 0. | `"NVIDIA H100 80GB HBM3"` |
| `gpuMemoryMB` | PositiveInt | Total memory of GPU 0 in MB. | `81559` |
| `cudaVersion` | ShortString | CUDA toolkit version. | `"12.4"` |

#### Parallelism fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tensorParallelSize` | PositiveInt | Tensor parallelism degree. | `8` |
| `pipelineParallelSize` | PositiveInt | Pipeline parallelism degree. | `1` |
| `contextParallelSize` | PositiveInt | Context parallelism degree. | `1` |
| `moeExpertParallelSize` | PositiveInt | MoE expert parallelism. **`0` = auto/unset** (runtime decides). Positive value = explicitly configured. | `0`, `8` |
| `moeTensorParallelSize` | PositiveInt | MoE tensor parallelism. **`0` = auto/unset** (runtime decides). Positive value = explicitly configured. | `0`, `2` |

#### Model & config fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `architectureClassName` | LongString | Exact model architecture class when it appears in the checked-in public Hugging Face architecture allowlist; otherwise empty. | `"MixtralForCausalLM"`, `"LlamaForCausalLM"`, `""` |
| `architectureClassHash` | LongString | Pseudonymous, deterministic SHA-256 grouping key for a non-empty architecture outside the public allowlist; otherwise empty. | `"sha256:76873a...ccd04"`, `""` |
| `backend` | ShortString | Execution backend. | `"pytorch"`, `"tensorrt"` |
| `dtype` | ShortString | Model data type. | `"float16"`, `"bfloat16"`, `"auto"` |
| `quantizationAlgo` | ShortString | Quantization algorithm. Empty string if none. | `""`, `"fp8"`, `"w4a16_awq"` |
| `kvCacheDtype` | ShortString | KV cache data type. Empty string if default. | `""`, `"fp8"`, `"auto"` |

#### Serving context fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `ingressPoint` | ShortString | How TRT-LLM was invoked. See [Ingress point values](#ingress-point-values). | `"cli_serve"` |
| `featuresJson` | string | Legacy JSON-serialized summary of feature flags. See [featuresJson keys](#featuresjson-keys). | `'{"lora":false,...}'` |
| `llmApiConfigJson` | string | JSON-serialized sanitized, type-driven effective LLM API configuration. See [runtime config capture](#runtime-config-capture). | `'{"tensor_parallel_size":2,...}'` |
| `llmApiConfigMetaJson` | string | JSON-serialized metadata for LLM API configuration capture. | `'{"capture_succeeded":true,...}'` |
| `disaggRole` | ShortString | Disaggregated serving role. Empty if not disaggregated. | `""`, `"context"`, `"generation"`, `"coordinator"`, `"server_coordinator"`, `"ctx0"`, `"gen0"` |
| `deploymentId` | ShortString | Shared ID across disaggregated workers. Empty if not disaggregated. | `""`, `"dep-abc123"` |

##### Architecture privacy policy

`architectureClassName` and `architectureClassHash` are mutually exclusive.
Plaintext is reported only for an exact match in
`tensorrt_llm/usage/architecture_allowlist.py`. Other valid names are omitted and
represented by `sha256:` plus SHA-256 of
`"trtllm-architecture-class-v1\0<UTF-8 name>"`, allowing repeated unknown names to
be grouped without transmitting them.

This hash provides pseudonymization, not confidentiality. Architecture names are
often predictable, and the domain separator is public, so a telemetry recipient
can hash candidate names offline and identify a match. The fixed domain also
makes the same unknown architecture globally correlatable across deployments.
The intended guarantee is only that the raw name is not transmitted while stable
grouping remains possible; the hash must not be treated as anonymous or secret.

Extraction uses the first Hugging Face `architectures` value and also supports
legacy singular values and nested engine configs. Invalid or empty values leave
both fields empty. The conservative allowlist is maintained manually; custom
names remain excluded until reviewed.

#### Aggregate LLM lifecycle counters

These process-local snapshots are present on the initial report, every
heartbeat, and the terminal report. Cumulative counters saturate at uint32 max;
`activeLlmInstances` is a gauge.

| Field | Type | Description |
|-------|------|-------------|
| `llmInitializationAttempts` | PositiveInt | Number of entries into `LLM.__init__`. |
| `llmInstancesCreated` | PositiveInt | Number of LLM objects initialized successfully. |
| `activeLlmInstances` | PositiveInt | Successfully initialized objects not yet shut down. |
| `maxConcurrentLlmInstances` | PositiveInt | Maximum active objects observed in the process session. |
| `llmInitializationFailures` | PositiveInt | Initialization attempts that raised a handled Python exception. |

### `trtllm_heartbeat`

Sent periodically (default: every 600s) to track session duration. Up to 1000
heartbeats per session.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `seq` | PositiveInt | Zero-based heartbeat sequence number. | `0`, `1`, `42` |
| `ingressPoint` | ShortString | Invocation boundary for the process session. | `"cli_serve"` |
| `disaggRole` | ShortString | Disaggregated role: `context`, `generation`, `coordinator`, `server_coordinator`, or compatible legacy values such as `ctx0`/`gen0`. | `"context"` |
| `deploymentId` | ShortString | Optional shared disaggregated deployment ID. | `"dep-abc123"` |

Every heartbeat also contains the five aggregate LLM lifecycle counters above.

### VisualGen events

`trtllm_visual_gen_initial_report` contains the common system/GPU fields above
and the first successfully loaded VisualGen pipeline's sanitized configuration.
Only the explicit fields below are collected. There is no generic
`VisualGenArgs` dump and no VisualGen architecture hash.

| Priority | Fields | Collection |
|----------|--------|------------|
| P0 | `runtimeKind`, `ingressPoint` | Fixed runtime and entry-point categories. |
| P0 | `modelId`, `pipelineClassName`, `resolvedPipelineClass`, `modality` | Exact public registry IDs and built-in classes marked `telemetry_safe`; other identities become `other`. Modality is a fixed pipeline capability, or `unknown`. |
| P0 | `launchMode`, `nodeCount`, `nWorkers` | `local_spawn`, `torchrun`, `slurm`, or `unknown`, plus bounded counts. No hostnames, ranks, or addresses. |
| P0 | `cfgSize`, `ulyssesSize`, `ringSize`, `attn2dRowSize`, `attn2dColSize`, `tensorParallelSize`, `parallelVaeSize`, `parallelVaeSplitDim` | Typed topology. The design's `attn2dSize` uses two fields; `tpSize` reuses the existing `tensorParallelSize` wire name. VAE split is `width` or `height`. |
| P0 | `quantizationAlgo`, `dynamicWeightQuant`, `quantizedComponentsJson` | Known `QuantAlgo` names, empty/other/mixed sentinels, a boolean, and only `transformer`/`transformer_2`. `quantizationAlgo` is the wire name for the design's `quantAlgo`. |
| P0 | `featuresJson` | Only four booleans: `parallelVae`, `sparseAttention`, `quantAttention`, `quantizedWeights`. |
| P0 | `attentionBackend`, `sparseAttentionAlgorithm`, `vsaSparsity`, `targetSparsity`, `cacheBackend` | Reviewed backend/algorithm enums; unsupported backends become `other`. Sparsity uses upper-edge buckets 0.25, 0.5, 0.75, 1.0, overflow, or unknown. No exact coefficients or thresholds. |
| P0 | `torchCompileEnable`, `enableFullgraph`, `enableAutotune`, `cudaGraphEnable` | Typed booleans, not compilation shapes or raw configuration. |
| P1 | `componentsPresentJson`, `transformerCount`, `checkpointFormat` | Fixed VAE, audio-VAE, text-encoder, and vocoder flags; bounded transformer count; `diffusers`, `single_safetensors`, or `other`. No paths. |
| P1 | `quantAttentionEnabled`, `qkDtype`, `vDtype`, `qBlockSize`, `kBlockSize`, `vBlockSize` | Known dtype labels and bounded block sizes. No `clamp_val`. |
| P1 | `pipelineLoadDurationSec`, `warmupDurationSec` | Rank-zero load and warmup durations. Load includes checkpoint resolution and compilation but excludes warmup. |

`trtllm_visual_gen_heartbeat` carries `seq`, `runtimeKind=visual_gen`,
`ingressPoint`, `nWorkers`, `gpuCount`, and cumulative session summaries.
The initial and terminal events also include those summaries. Sending a snapshot
does not reset counters. Consumers must use the latest snapshot per session
rather than summing snapshots.

| Priority | Summary field | Collection |
|----------|---------------|------------|
| P0 | `visualGenMetricsJson.endpointRequests` | POST counts for `/v1/images/generations`, `/v1/images/edits`, `/v1/videos`, and `/v1/videos/generations`. The current `/v1/videos/sync` route maps to the last category. GET polling and arbitrary paths are excluded. |
| P0 | `visualGenMetricsJson.requestsByModality` | One count per submitted generation call/batch, keyed by pipeline capability. |
| P0 | `visualGenMetricsJson.resolution`, `.numFrames` | Disjoint upper-edge buckets: longest image side 512/1024/2048/4096 pixels; frame count 16/64/128/256. Includes overflow and unknown buckets. Collected after request preparation resolves defaults/shapes. |
| P0 | `peakNumQueuedRequests`, `peakNumActiveRequests` | Maximum observed executor queue and in-flight counts over the session. Queue acceptance is observed before dispatch to include fast requests. |
| P1 | `visualGenMetricsJson.numInferenceSteps`, `.batchSize` | Upper-edge buckets 10/30/50/100 steps and 1/2/4/8 batch items, plus overflow/unknown. |
| P1 | `visualGenMetricsJson.inputReferenceKind`, `.extraParamsKeysUsed` | Counts for none/image/video reference presence; only explicitly supplied `stg_scale` key usage. Unknown keys and all values are dropped. |
| P1 | `visualGenMetricsJson.latencySec` | Successful-request generation/pre_denoise/denoise/post_denoise p50/p95 and sample counts. Fixed logarithmic bins give approximate upper-edge percentiles (10% spacing); values above the last bin carry an overflow flag. Zero/unmeasured phases are excluded. No raw timing records are retained. |
| P1 | `visualGenMetricsJson.errors` | Cumulative client/capacity/unclassified/timeout counts from submission failures, server schema validation, worker responses, and abandoned result waits. No exception text. Late/duplicate responses do not add completion samples. |
| P1 | `visualGenInitializationAttempts`, `visualGenInstancesCreated`, `activeVisualGenInstances`, `visualGenInitializationFailures` | Attempts, successful constructions, active gauge, and handled construction failures. |
| P1 | `failedComponent`, `sessionDurationSec`, `seq` | Failure stage is currently `none` or `unclassified`; duration uses a monotonic clock; heartbeat sequence is zero-based. |

Counters saturate at unsigned 32-bit maximum. Histogram labels such as
`le_1024` denote the disjoint bucket ending at 1024, not an additional cumulative
bucket. Only aggregate snapshots leave the coordinator. Request IDs, prompts,
reference content, outputs, exact shapes, arbitrary extra keys, and exception
details never enter the telemetry payload.

### `trtllm_exit_report`

Sent at most once when TRT-LLM or a surviving observer can classify the session
outcome. Missing terminal events remain unknown; they are not confirmed crashes.

| Field | Type | Description |
|-------|------|-------------|
| `exitCodeKnown` | boolean | Whether an authoritative process-style exit code is available. |
| `exitCode` | PositiveInt | Exit code, or `0` when unknown. |
| `signalNumber` | PositiveInt | Signal number, or `0` when not applicable or unknown. |
| `terminationKind` | enum | `clean`, `exception`, `signal`, `worker_failure`, `timeout`, or `unknown`. |
| `lifecyclePhase` | enum | Last known phase reached before termination: `cli_parsing`, `config_validation`, `model_initialization`, `serving`, or `unknown`. |
| `component` | enum | `llm`, `visual_gen`, `server`, `engine_worker`, `disagg_worker`, or `unknown`. |
| `reportingSource` | enum | `self`, `supervisor`, or `executor_proxy`. |
| `runtimeKind` | enum | Runtime families observed in the process: `llm`, `visual_gen`, `mixed`, or `unknown`. |
| `ingressPoint` | ShortString | Entry point copied onto the terminal event so terminal-only early failures remain attributable. |
| `disaggRole` | ShortString | Disaggregated role (`context`, `generation`, `coordinator`, `server_coordinator`, or compatible legacy `ctx0`/`gen0`), or empty when unavailable/not applicable. |
| `deploymentId` | ShortString | Optional shared disaggregated deployment ID. |

Every terminal report also contains the five aggregate LLM lifecycle counters
and the VisualGen cumulative summaries above. Pure VisualGen sessions leave
LLM disaggregation identity fields empty. Delivery is best-effort and waits
no more than 0.5 seconds; the local
terminal lock permits at most one delivery attempt per process session.

When a surviving parent observes a subprocess return code such as `-9`, it is
normalized to shell-style `exitCode=137` with `signalNumber=9`.

Terminal fields are deliberately bounded and categorical. Exception messages,
stack traces, commands, paths, model identifiers, and configuration text are
not collected.

V1 signal coverage is boundary-specific. The generic CLI boundary observes
Ctrl+C/SIGINT, while Uvicorn serving and explicitly instrumented disaggregated
boundaries observe both SIGINT and SIGTERM. A default SIGTERM delivered to
another CLI, such as `trtllm-bench` or `trtllm-eval`, may terminate the process
before it can send a terminal report.

## Type Reference

| Type | JSON type | Constraints |
|------|-----------|-------------|
| ShortString | string | 0–128 characters |
| LongString | string | 0–256 characters |
| PositiveInt | integer | 0–4,294,967,295 |
| Fraction | number | 0.0–1.0 |

## Ingress Point Values

The `ingressPoint` field identifies which TRT-LLM entry point started the session.

| Value | Meaning |
|-------|---------|
| `"cli_serve"` | Started via `trtllm-serve` CLI |
| `"cli_bench"` | Started via `trtllm-bench` CLI |
| `"cli_eval"` | Started via evaluation CLI |
| `"llm_class"` | Started via `LLM()` Python API directly |
| `"visual_gen_class"` | Started via `VisualGen()` Python API directly |
| `"disaggregated"` | Started as a disaggregated coordinator or fleet worker |
| `"unknown"` | Entry point not identified |

## `featuresJson` Keys

The `featuresJson` field is a JSON-serialized dict. All keys are always present
with safe defaults. This list may evolve as features are added.

TODO: Deduplicate `featuresJson` with `llmApiConfigJson` after derived-only
flags such as LoRA/speculative decoding have explicit safe config fields.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lora` | bool | `false` | LoRA adapter enabled (`enable_lora=True` or `lora_config` provided). |
| `speculative_decoding` | bool | `false` | Speculative decoding enabled (`speculative_config` is not None). Covers MTP, EAGLE, Medusa, etc. |
| `prefix_caching` | bool | `false` | KV cache block reuse / prefix caching enabled. |
| `cuda_graphs` | bool | `false` | CUDA graphs enabled for reduced launch overhead. |
| `chunked_context` | bool | `false` | Chunked prefill enabled (`enable_chunked_prefill=True`). |
| `data_parallel_size` | int | `1` | Data parallel degree. `1` = no data parallelism. Derived from `tp_size` when attention DP is enabled. |

## Runtime Config Capture

The `llmApiConfigJson` field is a JSON-serialized dict containing a type-driven
subset of the validated, effective LLM API configuration. Capture is
automatic for `bool`, `int`, finite `float`, `Literal`, `Enum`, supported unions,
and homogeneous sequences. Unsafe scalar `str`, `Any`, and `object` branches
require `TelemetryField.categorical(...)`; paths, mappings, callables, and
unsupported structures always fail closed. Use `telemetry=False` to exclude a
field.

Captured values must be safe primitives. Raw strings are excluded unless the
field is a `Literal[...]`/`Enum` or matches explicit `allowed_values`. Paths,
tokenizer locations, dicts, objects, callables, raw `Any` values, non-finite
floats (`nan`/`inf`), and unsafe or heterogeneous sequences are excluded.
Union branches are compiled and sanitized independently: explicit
`allowed_values` opt in only otherwise unsafe scalar branches and do not filter
safe numeric, boolean, `Literal`, or `Enum` branches in the same union.
Captured sequences are capped at a fixed length and any clipping is reported in
the corresponding metadata field. Exclusion is fail-closed: the value is
omitted instead of being serialized, and the metadata reports whether any
resolved field was excluded as unsafe.

The table below gives common dashboard examples. The committed manifest is the
canonical list of `TorchLlmArgs` capturable paths, merged policies, and
categorical domains. `llmApiConfigMetaJson` identifies that manifest by digest.
The docs build renders the full table under **Developer Guide > Telemetry**.

| Key | Description |
|-----|-------------|
| `tensor_parallel_size` | Tensor parallelism degree from the effective LLM args. |
| `pipeline_parallel_size` | Pipeline parallelism degree from the effective LLM args. |
| `context_parallel_size` | Context parallelism degree from the effective LLM args. |
| `moe_expert_parallel_size` | MoE expert parallelism degree (None/unset when runtime decides). |
| `moe_tensor_parallel_size` | MoE tensor parallelism degree (None/unset when runtime decides). |
| `moe_cluster_parallel_size` | MoE cluster parallelism degree (None/unset when runtime decides). |
| `backend` | Execution backend. Captured as the `Literal["pytorch"]` value on the PyTorch args. |
| `dtype` | Model dtype, captured through an explicit allowlist. |
| `load_format` | Weight load format, captured as a low-cardinality enum/string value. |
| `quant_config.quant_algo` | Quantization algorithm, captured as a closed `QuantAlgo` enum value (TRT args only). Empty/absent when unquantized. |
| `kv_cache_config.dtype` | KV cache dtype, captured through an explicit allowlist. |
| `kv_cache_config.enable_block_reuse` | Whether KV cache block reuse/prefix caching is enabled. |
| `cuda_graph_config.batch_sizes` | CUDA graph batch sizes when configured. |
| `scheduler_config.capacity_scheduler_policy` | Scheduler capacity policy. |
| `scheduler_config.enable_prefix_aware_scheduling` | Whether scheduler admission and token budgeting use KV prefix-reuse estimates. |
| `torch_compile_config.enable_inductor` | Whether Torch Inductor compilation is enabled. |
| `moe_config.backend` | MoE backend selection (`AUTO`, `CUTLASS`, `TRTLLM`, ...), an annotation-derived categorical. |
| `speculative_config.decoding_type` | Speculative decoding mode discriminator (e.g. `User_Provided`); other arms expose their own numeric/boolean knobs under `speculative_config.*`. |
| `sparse_attention_config.algorithm` | Sparse attention algorithm discriminator; arm-specific knobs appear under `sparse_attention_config.*`. |
| `reasoning_parser` | Reasoning parser selection, captured through an allowlist mirroring the `ReasoningParserFactory` registry. |

The matching config metadata field describes the capture process itself. It includes
contract/version fields, schema and manifest digests, source args class, field
counts (`capturable_field_count`, `captured_field_count`, `excluded_field_count`), capture
success, unsafe-exclusion status, a `sequence_truncated` flag set when any captured
sequence was clipped to the length cap, and a `payload_truncated` flag set when the
total serialized config exceeded the size budget and fields were dropped. The metadata
is intended to make dashboards robust when the safe capture manifest changes
over time.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRTLLM_NO_USAGE_STATS` | unset | Set to `1` to disable telemetry. |
| `TELEMETRY_DISABLED` | unset | Set to `true` to disable telemetry. |
| `DO_NOT_TRACK` | unset | Set to `1` to disable telemetry. |
| `TRTLLM_USAGE_STATS_SERVER` | `https://events.gfe.nvidia.com/v1.1/events/json` | Override the GXT endpoint URL. Use for staging. |
| `TRTLLM_USAGE_HEARTBEAT_INTERVAL` | `600` | Heartbeat interval in seconds. |
| `TRTLLM_USAGE_FORCE_ENABLED` | `0` | Set to `1` to force-enable telemetry in CI/test environments. |
| `TRTLLM_DISAGG_ROLE` | unset | Disaggregated serving role (`context`, `generation`, `coordinator`, `server_coordinator`, or compatible legacy values such as `ctx0`/`gen0`). |
| `TRTLLM_DISAGG_DEPLOYMENT_ID` | unset | Shared deployment ID across disaggregated workers. |

## For Developers: Adding a New Field

Checklist for adding a telemetry field:

1. **`tensorrt_llm/usage/schema.py`** — Add the field to the appropriate event Pydantic model with an alias.
2. **`tensorrt_llm/usage/schemas/trtllm_usage_event_schema.json`** — Add to `properties` and `required` array.
3. **`tensorrt_llm/usage/usage_lib.py`** — Populate the field in `_background_reporter()` and add extraction logic in `_extract_trtllm_config()` or `_collect_gpu_info()` as appropriate.
4. **`tests/unittest/usage/test_schema.py`** — Update test fixtures and expected field sets.
5. **`tests/unittest/usage/test_collectors.py`** — Add extraction test.
6. **`tests/unittest/usage/test_e2e_capture.py`** — Update e2e payload assertions if needed.
7. **SMS schema upload** — Upload the updated JSON schema to the NvTelemetry Schema Management Service and toggle "on stage" / "on prod".
8. **Update this README** — Add the field to the appropriate table above.

Checklist for adding a runtime config capture field inside `llmApiConfigJson`:

1. **Add the field with its natural type.** If it is categorical
   (`Literal`/`Enum`/`bool`) or numeric (`int`/`float`) — or a safe collection of
   those — it is captured automatically; no marker is needed.
2. **Bounded bare-string fields opt in via an allowlist.** If a free-form
   `str`/`Any` field should be captured, mark it
   `telemetry=TelemetryField.categorical(<allowed_values>)`, mirroring its real
   recognized domain. **Prefer tightening the type (e.g. `str` -> `Literal`) over
   an allowlist** when the API contract allows it; the allowlist is the fallback
   when the annotation cannot be narrowed without a breaking validation change.
3. **Type-safe but sensitive? Opt out with `telemetry=False`.** This honored
   exclusion sentinel keeps a categorical/numeric field out of capture.
4. **Do not capture unsafe data.** No model/tokenizer/file paths, prompts,
   outputs, secrets/tokens/URLs/hostnames, free-form user strings, raw
   dict/object payloads, or callables. Paths, mappings, callables, arbitrary
   non-scalars, heterogeneous sequences, and non-finite floats always fail
   closed. A `str`/`Any`/`object` scalar branch is emitted only when it exactly
   matches the field's finite `allowed_values`.
5. **`tests/unittest/usage/test_llmapi_config_capture.py`** — Add behavior
   coverage: assert the value is captured, and for a categorical bare-string
   field assert that an out-of-allowlist value is redacted (dropped) while an
   in-allowlist value is captured.
6. **Regenerate the manifest golden**:
   `python3 scripts/generate_llm_args_golden_manifest.py`
   Review the golden diff — **it is the privacy review.** A newly captured field
   requires sign-off from the GitHub telemetry/privacy CODEOWNER (`.github/CODEOWNERS`).
7. **`docs/source/developer-guide/telemetry.md` is generated** from the committed
   golden at docs-build time; do not hand-edit it.
8. **Update this README** — Add a common-key row above when the field is
   important enough for dashboard users to know by name.

Dashboard note: payloads carry `capture_version` and `field_policy_version` in
`llmApiConfigMetaJson`. During release adoption, v1 (opt-in), v2 (initial
type-driven), and v3 (composed branch-policy) payloads coexist in the same index
— **bucket by these before aggregating**
`captured_field_count` or any `llmApiConfigJson.<field>`.

### VisualGen collection limits

- Resolved pipeline metadata is available only after READY. Earlier failures can produce lifecycle/terminal data without an initial report.
- `failedComponent` does not yet distinguish individual loader stages. `extraParamsKeysUsed` starts with the reviewed `stg_scale` key only.
- P2 tuning/output fields are not collected.
- A process has one reporter. The first runtime/object supplies static metadata. VisualGen request and lifecycle summaries aggregate across VisualGen objects, while queue peaks are the maximum observed on any individual executor, not a sum of executor gauges.
- If an LLM starts the reporter first, VisualGen aggregates are available in the final terminal snapshot but not the LLM heartbeat. If VisualGen starts first, its heartbeats carry the VisualGen summaries. Mixed sessions retain the existing LLM telemetry contract.
- Telemetry reuses the existing opt-outs, CI/test suppression, notification, heartbeat limits, and best-effort terminal delivery. Schema registration and production approval are separate deployment steps.

### Conventions

- Use **camelCase** aliases for JSON wire format (Pydantic `alias=`).
- Use **snake_case** for Python field names.
- String fields: use `ShortString` (128 chars) or `LongString` (256 chars).
- Integer fields: use `PositiveInt` (0–4B). Use `0` for "auto/unset" semantics.
- All fields must be **required** in the JSON schema (no optional fields).
- Empty string `""` is the sentinel for "not applicable" string fields.
- The telemetry code is **fail-silent in two layers.** The LLM API config
  collector catches only the expected sanitizer/walk error family
  (`AttributeError`, `TypeError`, `ValueError`, `KeyError`) and emits an empty
  config plus `capture_succeeded=false`; unexpected exceptions are left to
  propagate so genuine collector bugs are not masked. They are then caught by
  the outer daemon-thread reporter guard in `usage_lib.py`, which keeps the
  reporting thread from ever taking down the host process.
- No PII. No model weights. No prompts. No outputs. No model/tokenizer paths.
