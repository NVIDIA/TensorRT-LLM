# TRT-LLM Telemetry Schema Reference

Schema version: **0.2** | Client ID: `616561816355034` | Protocol: GXT Event Protocol v1.6

## Overview

TRT-LLM collects anonymous, session-level deployment telemetry to understand
how the library is used in production (GPU types, parallelism configs, model
architectures). No PII, model weights, prompts, outputs, model paths, tokenizer
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
| `eventSchemaVer` | string | Schema version, currently `"0.2"`. |
| `eventSysVer` | string | Always `"trtllm-telemetry/1.0"`. |
| `sessionId` | string | Unique hex UUID per server lifetime. Use this to correlate initial report with heartbeats. |
| `sentTs` | string | ISO 8601 UTC timestamp of when the payload was sent. |

Privacy/identity fields (`osVersion`, `geoInfo`, `deviceGUID`, etc.) are
hardcoded to `"undefined"` — TRT-LLM is a server-side SDK with no browser or
login context.

## Events

### `trtllm_initial_report`

Sent once at server startup. Contains system info and serving configuration.

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
| `architectureClassName` | LongString | HuggingFace model architecture class. | `"MixtralForCausalLM"`, `"LlamaForCausalLM"` |
| `backend` | ShortString | Execution backend. | `"pytorch"`, `"tensorrt"` |
| `dtype` | ShortString | Model data type. | `"float16"`, `"bfloat16"`, `"auto"` |
| `quantizationAlgo` | ShortString | Quantization algorithm. Empty string if none. | `""`, `"fp8"`, `"w4a16_awq"` |
| `kvCacheDtype` | ShortString | KV cache data type. Empty string if default. | `""`, `"fp8"`, `"auto"` |

#### Serving context fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `ingressPoint` | ShortString | How TRT-LLM was invoked. See [Ingress point values](#ingress-point-values). | `"cli_serve"` |
| `featuresJson` | string | Legacy JSON-serialized summary of feature flags. See [featuresJson keys](#featuresjson-keys). | `'{"lora":false,...}'` |
| `llmApiConfigJson` | string | JSON-serialized sanitized, type-driven effective LLM API configuration. See [LLM API config capture](#llm-api-config-capture). | `'{"tensor_parallel_size":2,...}'` |
| `llmApiConfigMetaJson` | string | JSON-serialized metadata for LLM API configuration capture. | `'{"capture_succeeded":true,...}'` |
| `disaggRole` | ShortString | Disaggregated serving role. Empty if not disaggregated. | `""`, `"context"`, `"generation"` |
| `deploymentId` | ShortString | Shared ID across disaggregated workers. Empty if not disaggregated. | `""`, `"dep-abc123"` |

### `trtllm_heartbeat`

Sent periodically (default: every 600s) to track session duration. Up to 1000
heartbeats per session.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `seq` | PositiveInt | Zero-based heartbeat sequence number. | `0`, `1`, `42` |

## Type Reference

| Type | JSON type | Constraints |
|------|-----------|-------------|
| ShortString | string | 0–128 characters |
| LongString | string | 0–256 characters |
| PositiveInt | integer | 0–4,294,967,295 |

## Ingress Point Values

The `ingressPoint` field identifies which TRT-LLM entry point started the session.

| Value | Meaning |
|-------|---------|
| `"cli_serve"` | Started via `trtllm-serve` CLI |
| `"cli_bench"` | Started via `trtllm-bench` CLI |
| `"cli_eval"` | Started via evaluation CLI |
| `"llm_class"` | Started via `LLM()` Python API directly |
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

## LLM API Config Capture

The `llmApiConfigJson` field is a JSON-serialized dict containing a type-driven
subset of the validated, effective LLM API configuration. Capture is
**type-driven**: a field is captured automatically when its type is categorical
(`Literal`/`Enum`/`bool`) or numeric (`int`/`float`), or a safe collection of
those. Free-form `str`/`Any`/`Path`/`dict`/`Callable` are not captured unless the
field carries an explicit allowlist (`TelemetryField.categorical(...)`). Any field
can opt out with `telemetry=False`.

Captured values must be safe primitives. Raw strings are excluded unless the
field is a `Literal[...]` or uses an explicit `allowlist` converter. Paths,
tokenizer locations, dicts, objects, callables, raw `Any` values, non-finite
floats (`nan`/`inf`), and unsafe or heterogeneous sequences are excluded.
Captured sequences are capped at a fixed length and any clipping is reported in
`llmApiConfigMetaJson`. Exclusion is fail-closed: the value is omitted instead
of being serialized, and `llmApiConfigMetaJson` reports whether any resolved field
was excluded as unsafe.

The table below is a non-exhaustive set of examples for readers building
dashboards. The exhaustive source of truth is
`tensorrt_llm/usage/llm_args_golden_manifest.json` (regenerated from
`build_capture_manifest`), after the safety sanitizer has excluded unsafe values.
Use `llmApiConfigMetaJson` digests and field counts to track the exact capture
manifest for a given release. The rendered documentation generates the
exhaustive field table at docs build time under **Developer Guide > Telemetry**.

| Key | Description |
|-----|-------------|
| `tensor_parallel_size` | Tensor parallelism degree from the effective LLM args. |
| `pipeline_parallel_size` | Pipeline parallelism degree from the effective LLM args. |
| `context_parallel_size` | Context parallelism degree from the effective LLM args. |
| `moe_expert_parallel_size` | MoE expert parallelism degree (None/unset when runtime decides). |
| `moe_tensor_parallel_size` | MoE tensor parallelism degree (None/unset when runtime decides). |
| `moe_cluster_parallel_size` | MoE cluster parallelism degree (None/unset when runtime decides). |
| `backend` | Execution backend. Captured as the `Literal["pytorch"]` value on the PyTorch args, and through an explicit allowlist (`pytorch`, `tensorrt`, `_autodeploy`) on the base/TRT args. |
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
| `sampler_type` | Sampler selection, captured through an allowlist mirroring the `SamplerType` enum. |

`llmApiConfigMetaJson` describes the capture process itself. It includes
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
| `TRTLLM_DISAGG_ROLE` | unset | Disaggregated serving role (`context` or `generation`). |
| `TRTLLM_DISAGG_DEPLOYMENT_ID` | unset | Shared deployment ID across disaggregated workers. |

## For Developers: Adding a New Field

Checklist for adding a telemetry field:

1. **`tensorrt_llm/usage/schema.py`** — Add field to `TrtllmInitialReport` (or `TrtllmHeartbeat`) Pydantic model with alias.
2. **`tensorrt_llm/usage/schemas/trtllm_usage_event_schema.json`** — Add to `properties` and `required` array.
3. **`tensorrt_llm/usage/usage_lib.py`** — Populate the field in `_background_reporter()` and add extraction logic in `_extract_trtllm_config()` or `_collect_gpu_info()` as appropriate.
4. **`tests/unittest/usage/test_schema.py`** — Update test fixtures and expected field sets.
5. **`tests/unittest/usage/test_collectors.py`** — Add extraction test.
6. **`tests/unittest/usage/test_e2e_capture.py`** — Update e2e payload assertions if needed.
7. **SMS schema upload** — Upload the updated JSON schema to the NvTelemetry Schema Management Service and toggle "on stage" / "on prod".
8. **Update this README** — Add the field to the appropriate table above.

Checklist for adding an LLM API config capture field inside `llmApiConfigJson`:

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
   dict/object payloads, or callables. The sanitizer fails closed regardless:
   bare `str`, `Any`, `object`, `Path`, `dict`, callables, permissive unions, and
   non-finite floats are dropped unless an approved `allowlist` converter applies.
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
`llmApiConfigMetaJson`. During release adoption, v1 (opt-in) and v2 (type-driven)
payloads coexist in the same index — **bucket by these before aggregating**
`captured_field_count` or any `llmApiConfigJson.<field>`.

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
