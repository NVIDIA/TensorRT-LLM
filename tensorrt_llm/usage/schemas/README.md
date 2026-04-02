# TRT-LLM Telemetry Schema Reference

Schema version: **0.1** | Client ID: `616561816355034` | Protocol: GXT Event Protocol v1.6

## Overview

TRT-LLM collects anonymous, session-level deployment telemetry to understand
how the library is used in production (GPU types, parallelism configs, model
architectures). No PII, model weights, prompts, or outputs are collected.

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
| `eventSchemaVer` | string | Schema version, currently `"0.1"`. |
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
| `featuresJson` | string | JSON-serialized dict of feature flags. See [featuresJson keys](#featuresjson-keys). | `'{"lora":false,...}'` |
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

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lora` | bool | `false` | LoRA adapter enabled (`enable_lora=True` or `lora_config` provided). |
| `speculative_decoding` | bool | `false` | Speculative decoding enabled (`speculative_config` is not None). Covers MTP, EAGLE, Medusa, etc. |
| `prefix_caching` | bool | `false` | KV cache block reuse / prefix caching enabled. |
| `cuda_graphs` | bool | `false` | CUDA graphs enabled for reduced launch overhead. |
| `chunked_context` | bool | `false` | Chunked prefill enabled (`enable_chunked_prefill=True`). |
| `data_parallel_size` | int | `1` | Data parallel degree. `1` = no data parallelism. Derived from `tp_size` when attention DP is enabled. |

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

### Conventions

- Use **camelCase** aliases for JSON wire format (Pydantic `alias=`).
- Use **snake_case** for Python field names.
- String fields: use `ShortString` (128 chars) or `LongString` (256 chars).
- Integer fields: use `PositiveInt` (0–4B). Use `0` for "auto/unset" semantics.
- All fields must be **required** in the JSON schema (no optional fields).
- Empty string `""` is the sentinel for "not applicable" string fields.
- The telemetry code is **fail-silent** — exceptions are caught and swallowed.
- No PII. No model weights. No prompts. No outputs. Architecture class names only.
