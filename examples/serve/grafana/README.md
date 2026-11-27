<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grafana dashboards for Kubernetes serving

Render and publish role-aware TensorRT-LLM monitoring dashboards using Python 3.10+
and its standard library. Copy this whole directory, including `templates/`, to
use it separately from the TensorRT-LLM checkout. Rendering does not import
TensorRT-LLM, resolve credentials, contact Grafana, or invoke Kubernetes.

The metrics dashboard has E2E, Prefill, Decode and All views. Engine workers are
discovered from current scrape targets on refresh. The 80 default panel templates
include request/queue latency, throughput, rank-local KV reuse/load, transfer
outcomes, executor timing and physical tier pressure. Optional trace-PVC charts
add two templates. Optional Loki support adds a separate 16-panel Logs dashboard.
Auto-grid panels without observations are hidden; missing native tier/timing
measurements do not become zero-valued metrics.

## Requirements

- A Grafana server exposing the **dashboard schema v2** API and dynamic layouts.
  `grafana.api_version` selects `v2` (the example) or `v2beta1`. This is not the
  Classic dashboard JSON API. Confirm the selected version in your server's
  Swagger and run the validation command before publication. The `12.3.1` plugin
  versions in templates do not establish server API compatibility: Grafana's
  [v12.3.1 schema snapshots](https://github.com/grafana/grafana/tree/v12.3.1/pkg/tests/apis/openapi_snapshots)
  expose `v2alpha1`, which this example does not target. The current
  [Grafana dashboard API documentation](https://grafana.com/docs/grafana/latest/developer-resources/api-reference/http-api/dashboard/)
  recommends `v2`.
- A Prometheus datasource and application series with `namespace`, `pod`, `job`
  and `endpoint` scrape labels. Set `metric_job_regex` to **one application scrape
  path**, especially when both frontend and engine endpoints expose overlapping
  counters. Selectors are deployment inputs, not inferred from the current cluster.
- kube-state-metrics for deployment topology; DCGM with `exported_namespace` and
  `exported_pod` labels for GPU charts. Dynamo frontend metrics enable the E2E
  charts. Missing optional exporters do not establish workload health.
- `kubectl` is needed for ConfigMap publication, Kubernetes Secret authentication,
  or optional Service/ServiceMonitor creation. API publication with environment
  authentication and scrape setup disabled makes no Kubernetes calls.
- Loki is optional and disabled in the example. Its charts expect the Dynamo/TRTLLM
  iteration/trace log fields represented in `templates/logs.json`; this tool does
  not install a log collector. Optional distinct-session queries count observed
  trace IDs, not online connections, and can encounter Loki cardinality limits.

## Configure and render

From the repository root:

```sh
cp examples/serve/grafana/example.config.json /tmp/trtllm-dashboard.json
# Edit /tmp/trtllm-dashboard.json for your deployment and datasource UIDs.
python3 examples/serve/grafana/dashboard.py render \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered
```

Rendering produces `metrics.json`, optionally `logs.json`,
`dashboards.configmap.json`, and `scrape-resources.json`. These are local artifacts;
rendering alone does not create a dashboard. Keep deployment configs and rendered
resource identities outside the distributable example.

| Configuration | Meaning |
| --- | --- |
| `kubernetes.context`, `namespace` | Explicit cluster context and workload namespace; no current-context fallback. |
| `grafana.url`, `api_namespace`, `api_version` | Grafana endpoint, API namespace and supported schema-v2 API version. |
| `grafana.namespace`, `configmap`, `sidecar_labels` | Monitoring namespace and existing sidecar configuration for file publication. |
| `grafana.publication` | `api` or `configmap-v2`, according to who owns the dashboard. |
| `grafana.dashboard_keys` | Existing metrics/logs JSON filenames in the ConfigMap; preserve them when updating a file-owned dashboard. |
| `dashboard.*_uid`, `title`, `refresh` | Stable dashboard identities, display title and refresh period. |
| `dashboard.include_logs` | Enable the optional Logs dashboard; defaults to false in this example. |
| `datasources.prometheus`, `loki` | Grafana datasource UIDs; the Loki value is unused when Logs is disabled. |
| `workload.*_pod_regex`, `metric_job_regex` | Workload selection and the application scrape path. |
| `workload.metric_role_label`, `decode_metric_role_value` | Exporter role label and decode value; ServiceMonitors can inject this label. |
| `workload.engine_endpoint`, `engine_container`, `frontend_container` | Actual endpoint and container labels. |
| `workload.log_*`, `frontend_container` | Loki job, role and container contract, when Logs is enabled. |
| `workload.trace_pvc` | Optional trace volume; null omits both PVC charts. |
| `scrape.enabled` | Generate/apply scoped Services and ServiceMonitors; false by default. |

Use the supported RE2-compatible subset: ordinary groups and `[.]` for literal
periods; no lookaround, inline flags, backslashes, quotes, colons or commas. Use
JSON booleans, not strings such as `"false"`.

`dashboard.counter_lookback` defaults to `1m`; choose at least two actual scrape
intervals. Existing `irate` panels use the last two scrapes within that lookup
window. New tier turnover/drop panels use `rate`, averaging resets-aware deltas
across the window. Migration/transfer latency percentiles use a fixed 5-minute
bucket-rate window. Refresh cadence, scrape cadence and statistics collection
cadence are independent.

## Authentication and publication

`grafana.auth` accepts references, not inline credentials:

```json
{"token_env": "GRAFANA_TOKEN"}
```

Or use `username_env` / `password_env`, or an existing Kubernetes Secret reference:

```json
{"secret": "grafana-auth", "username_key": "admin-user", "password_key": "admin-password"}
```

Supply values through your environment or secret manager. Use HTTPS or a loopback
administrative tunnel. The client rejects redirects and omits credentials, raw
HTTP error bodies and Kubernetes stderr from output.

Validate the configured server/API and any enabled Kubernetes resources:

```sh
python3 examples/serve/grafana/dashboard.py deploy \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered --dry-run
```

This uses server-side `dryRun=All` / Kubernetes dry-run and writes local artifacts;
it does not publish dashboards or scrape resources. Publication is explicit:

```sh
python3 examples/serve/grafana/dashboard.py deploy \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered
```

`api` saves directly in Grafana. Use `configmap-v2` when an existing file
sidecar owns these UIDs; an API-only update can be overwritten by that provisioner.
The latter mode preserves unrelated ConfigMap entries/metadata, applies the
checkpoint, waits for its source checksum, then finalizes and verifies the native
v2 spec. Applying the rendered ConfigMap alone does not complete this workflow.

The script checks ConfigMap data/resource versions and dashboard resource versions
before writes, including after a sidecar checkpoint. Final native PUTs retain
`resourceVersion` for optimistic concurrency. This is not a transaction spanning
Kubernetes and Grafana: partial checkpoints can exist after an error. Resume only
with the same config/rendered checkpoint:

```sh
python3 examples/serve/grafana/dashboard.py deploy \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered --finalize-only
```

A successful publication reads back layouts, queries, variables, links, title and
time settings and writes `receipt.json` with `verified: true`. Each invocation
removes an old receipt first. Always check the current command's exit status.
Publication changes only the configured dashboard/scrape resources; the example
neither installs Grafana/Prometheus nor changes inference configuration.

## Interpreting KV-cache panels

Resident occupancy includes retained evictable prefixes. Combine it with turnover,
final capacity/priority drops and recent misses; high occupancy alone does not
prove thrashing. Capacity ratios weight pool groups by logical slot bytes. Legacy
GPU active/available charts are not physical tier occupancy, and available blocks
can contain useful prefixes. No host-only occupancy is inferred from aggregated
secondary-tier statistics.

Restore/evict turnover is `rate(moved_bytes) / capacity` in turnovers per second;
it can exceed one per second. Capacity/priority drops measure final prefix loss,
not ordinary request release. Native counters count accepted submissions and do
not confirm successful asynchronous DMA. Detailed tier rank coverage can be limited.

Sampled migration latency is disabled unless
`TRTLLM_KV_CACHE_TIMING_SAMPLE_INTERVAL` is positive before manager construction.
It measures batch stream intervals, including codec/staging, and is distinct from
completed-request KV-transfer duration. Neither is necessarily a request's restore
stall. Compare source/destination paths and skipped timing samples; sparse or
absent samples do not mean zero latency. Ratios and latencies are not stacked.

## Standalone tests

From the repository root, the utility tests need Python, `pytest` for the CI marker,
and no TensorRT-LLM runtime import. They mock publication; they do not deploy:

```sh
python3 -m unittest discover -s tests/unittest/metrics -p test_grafana_dashboard.py -v
```

A successful offline render or mocked publisher test is not a live Grafana,
Prometheus-query or native KV-instrumentation integration test.
