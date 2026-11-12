# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render and publish role-aware inference dashboards for an explicit K8s target.

Python standard library only. Credentials are resolved at runtime and never saved.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Callable, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

HERE = Path(__file__).resolve().parent


class SetupError(Exception):
    """Invalid configuration, failed transport, or concurrent modification."""


def require(condition: object, message: str) -> None:
    """Raise a setup diagnostic when a validated condition is false."""
    if not condition:
        raise SetupError(message)


def load_config(path: Path) -> dict:
    """Load and validate an explicit, credential-reference-only configuration."""
    c = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(c, dict), "Configuration must be a JSON object")
    require(c.get("schema_version") == 1, "Expected schema_version 1")
    for section in ("kubernetes", "grafana", "dashboard", "datasources", "workload", "scrape"):
        require(isinstance(c.get(section), dict), "Missing config section: " + section)
    k, g, d, w = (c[x] for x in ("kubernetes", "grafana", "dashboard", "workload"))
    require(
        isinstance(k.get("context"), str)
        and bool(k["context"])
        and not k["context"].startswith("-"),
        "An explicit Kubernetes context is required",
    )
    for value in (k.get("namespace"), g.get("namespace"), g.get("configmap")):
        require(
            isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9.-]*", value),
            "Invalid Kubernetes resource name",
        )
    for uid in (d.get("metrics_uid"), d.get("logs_uid")):
        require(
            isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", uid),
            "Invalid dashboard UID",
        )
    require(d["metrics_uid"] != d["logs_uid"], "Dashboard UIDs must differ")
    require(
        g.get("publication") in ("api", "configmap-v2"), "publication must be api or configmap-v2"
    )
    keys = g.get("dashboard_keys", {"metrics": "metrics.json", "logs": "logs.json"})
    require(isinstance(keys, dict), "dashboard_keys must be a JSON object")
    require(
        set(keys) == {"metrics", "logs"}
        and len(set(keys.values())) == 2
        and all(
            isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", x) and x.endswith(".json")
            for x in keys.values()
        ),
        "dashboard_keys must contain distinct metrics/logs JSON filenames",
    )
    parsed = urllib.parse.urlsplit(g.get("url", ""))
    require(
        parsed.scheme in ("https", "http")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "Grafana URL must have no embedded credentials, query, or fragment",
    )
    require(
        parsed.scheme == "https" or parsed.hostname in ("localhost", "127.0.0.1", "::1"),
        "Use HTTPS or a loopback administrative tunnel",
    )
    require(
        g.get("api_version", "v2beta1") in ("v2beta1", "v2"), "api_version must be v2beta1 or v2"
    )
    for section, key in ((d, "include_logs"), (c["scrape"], "enabled")):
        require(isinstance(section.get(key, False), bool), key + " must be a JSON boolean")
    for key in ("prometheus", "loki"):
        require(
            isinstance(c["datasources"].get(key), str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", c["datasources"][key]),
            "Invalid datasource UID",
        )
    auth = g.get("auth", {})
    require(isinstance(auth, dict), "grafana.auth must be a JSON object")
    require(
        set(auth)
        in (
            {"token_env"},
            {"username_env", "password_env"},
            {"secret", "username_key", "password_key"},
        ),
        "Use token_env, username_env/password_env, or a Kubernetes Secret reference for Grafana auth",
    )
    require(
        all(isinstance(v, str) and v for v in auth.values()),
        "Authentication references must be nonempty strings",
    )
    if "secret" in auth:
        require(
            bool(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", auth["secret"])),
            "Invalid Kubernetes Secret name",
        )
    for key in (
        "prefill_pod_regex",
        "decode_pod_regex",
        "frontend_pod_regex",
        "all_pod_regex",
        "metric_job_regex",
    ):
        value = w.get(key)
        require(
            isinstance(value, str)
            and value
            and not any(x in value for x in ('"', "\\", "\n", "\r", ",", ":", "@@")),
            key
            + (
                " must be a nonempty RE2-compatible regex without quotes, commas, colons or "
                "backslashes (use [.] for a literal dot)"
            ),
        )
        require(
            "(?" not in value, key + " supports ordinary groups, not lookaround or inline flags"
        )
        try:
            re.compile(value)
        except re.error:
            raise SetupError("Invalid regex: " + key) from None
    for key in ("metric_role_label", "log_role_label"):
        require(
            bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", w.get(key, ""))),
            "Invalid metric/log label name",
        )
    require(
        isinstance(w.get("log_containers"), list) and bool(w["log_containers"]),
        "log_containers must be a nonempty list",
    )
    for value in w["log_containers"]:
        require(
            isinstance(value, str) and bool(re.fullmatch(r"[a-z0-9][a-z0-9-]*", value)),
            "Invalid container name",
        )
    for duration in (
        d.get("counter_lookback"),
        d.get("refresh"),
        c["scrape"].get("interval"),
        c["scrape"].get("timeout"),
    ):
        require(
            isinstance(duration, str) and bool(re.fullmatch(r"[1-9][0-9]*[smhd]", duration)),
            "Durations must be positive integers followed by s/m/h/d",
        )
    return c


def walk(value: JsonValue, fn: Callable[[str], str]) -> JsonValue:
    """Replace strings recursively in a JSON value."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [walk(x, fn) for x in value]
    if isinstance(value, dict):
        return {fn(k): walk(v, fn) for k, v in value.items()}
    return value


def add_unique_sessions(spec: dict, c: dict) -> None:
    """Optional exact rolling distinct session count, sourced from frontend traces."""
    settings = c["dashboard"].get("unique_sessions", {})
    if not settings.get("enabled", False):
        return
    windows = settings.get("windows", ["1m", "5m", "15m", "30m", "60m"])
    default = settings.get("default_window", "30m")
    require(
        isinstance(windows, list)
        and windows
        and all(
            isinstance(x, str) and re.fullmatch(r"[1-9][0-9]*m", x) and int(x[:-1]) <= 60
            for x in windows
        ),
        "Session windows must be minute durations up to 60m",
    )
    require(
        default in windows and len(set(windows)) == len(windows),
        "Session default must be one of the distinct window choices",
    )
    spec["variables"] = [v for v in spec["variables"] if v["spec"]["name"] != "session_window"]
    spec["variables"].append(
        {
            "kind": "CustomVariable",
            "spec": {
                "name": "session_window",
                "label": "Session window",
                "query": ",".join(windows),
                "current": {"text": default, "value": default},
                "options": [{"text": x, "value": x, "selected": x == default} for x in windows],
                "multi": False,
                "includeAll": False,
                "allowCustomValue": False,
                "hide": "dontHide",
                "skipUrlSync": False,
                "description": "Rolling session observation window ending at the dashboard time-range end.",
            },
        }
    )
    w = c["workload"]
    selector = (
        '{namespace="$namespace",job='
        + json.dumps(w["log_job"])
        + ",pod=~"
        + json.dumps(w["frontend_pod_regex"])
        + ",container="
        + json.dumps(w["frontend_container"])
        + "}"
    )
    # Disjoint partitions avoid Loki's per-query series ceiling. Deduplication
    # still uses the complete session ID, so hash collisions do not merge IDs.
    partition = (
        "{{ $s := .session_id }}{{ $h := 0 }}{{ range $i := len $s }}{{ $h = mod (add "
        "(mul $h 31) (index $s $i)) 8 }}{{ end }}{{ $h }}"
    )
    queries = []
    for i in range(8):
        ref = chr(65 + i)
        pipeline = (
            selector
            + (
                ' |= "dynamo.request.trace.v1" | json session_id="agent_context.session_id", '
                'trace_schema="schema" | __error__="" | trace_schema="dynamo.request.trace.v1" '
                '| session_id!="" | label_format session_partition=`'
            )
            + partition
            + '` | session_partition="'
            + str(i)
            + '" | keep session_id'
        )
        expr = (
            "count(sum by (session_id) (count_over_time("
            + pipeline
            + " [${session_window:raw}]))) or vector(0)"
        )
        query = {
            "kind": "DataQuery",
            "group": "loki",
            "version": "v0",
            "datasource": {"name": c["datasources"]["loki"]},
            "spec": {
                "expr": expr,
                "queryType": "instant",
                "instant": True,
                "range": False,
                "refId": ref,
                "legendFormat": "partition " + str(i),
            },
        }
        queries.append(
            {"kind": "PanelQuery", "spec": {"query": query, "refId": ref, "hidden": False}}
        )
    queries.append(
        {
            "kind": "PanelQuery",
            "spec": {
                "refId": "TOTAL",
                "hidden": False,
                "query": {
                    "kind": "DataQuery",
                    "group": "__expr__",
                    "version": "v0",
                    "datasource": {"name": "__expr__"},
                    "spec": {
                        "type": "math",
                        "expression": " + ".join("$" + chr(65 + i) for i in range(8)),
                        "refId": "TOTAL",
                    },
                },
            },
        }
    )
    key = "session-unique-observed"
    panel_id = (
        spec["elements"]
        .get(key, {})
        .get("spec", {})
        .get("id", max(e["spec"]["id"] for e in spec["elements"].values()) + 1)
    )
    spec["elements"][key] = {
        "kind": "Panel",
        "spec": {
            "id": panel_id,
            "title": "Observed unique sessions - last ${session_window}",
            "description": (
                "Distinct nonempty agent_context.session_id values in frontend "
                "dynamo.request.trace.v1 logs during the selected Session window, ending at the"
                " dashboard time-range end. Repeated requests/events for one session count "
                "once. Observed sessions, not online connections, people or agent instances. A "
                "session appears only after a trace event is logged; request-end traces do not "
                "expose an unfinished first request. Independent of log search, container, "
                "worker and rank filters. Missing IDs are excluded. Empty successful queries "
                "return zero; query errors remain errors. Exact partitioned deduplication; "
                "excessive cardinality can still exceed Loki limits."
            ),
            "links": [],
            "data": {
                "kind": "QueryGroup",
                "spec": {
                    "queries": queries,
                    "transformations": [
                        {
                            "kind": "filterByRefId",
                            "spec": {"id": "filterByRefId", "options": {"include": "TOTAL"}},
                        }
                    ],
                    "queryOptions": {},
                },
            },
            "vizConfig": {
                "kind": "VizConfig",
                "group": "stat",
                "version": "12.3.1",
                "spec": {
                    "options": {
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "graphMode": "none",
                        "textMode": "value",
                        "colorMode": "value",
                        "justifyMode": "center",
                    },
                    "fieldConfig": {
                        "defaults": {"unit": "short", "decimals": 0, "min": 0, "noValue": "—"},
                        "overrides": [],
                    },
                },
            },
        },
    }
    roles = next(v["spec"] for v in spec["variables"] if v["spec"]["name"] == "role")
    all_role = next(o["value"] for o in roles["options"] if o["text"] == "All")
    rows = spec["layout"]["spec"]["rows"]
    rows[:] = [r for r in rows if r["spec"].get("title") != "E2E - Session activity"]
    rows.insert(
        0,
        {
            "kind": "RowsLayoutRow",
            "spec": {
                "title": "E2E - Session activity",
                "collapse": False,
                "hideHeader": False,
                "conditionalRendering": {
                    "kind": "ConditionalRenderingGroup",
                    "spec": {
                        "visibility": "show",
                        "condition": "or",
                        "items": [
                            {
                                "kind": "ConditionalRenderingVariable",
                                "spec": {"variable": "role", "operator": "equals", "value": v},
                            }
                            for v in ("e2e", all_role)
                        ],
                    },
                },
                "layout": {
                    "kind": "GridLayout",
                    "spec": {
                        "items": [
                            {
                                "kind": "GridLayoutItem",
                                "spec": {
                                    "x": 0,
                                    "y": 0,
                                    "width": 24,
                                    "height": 4,
                                    "element": {"kind": "ElementReference", "name": key},
                                },
                            }
                        ]
                    },
                },
            },
        },
    )


def render(c: dict) -> tuple[dict, dict, list[dict]]:
    """Render portable dashboard specs and optional scrape resources offline."""
    w, d, g = c["workload"], c["dashboard"], c["grafana"]
    engine = "(" + w["prefill_pod_regex"] + "|" + w["decode_pod_regex"] + ")"
    values = {
        "NAMESPACE": c["kubernetes"]["namespace"],
        "TITLE": d["title"],
        "METRICS_UID": d["metrics_uid"],
        "LOGS_UID": d["logs_uid"],
        "PREFILL_PODS": w["prefill_pod_regex"],
        "DECODE_PODS": w["decode_pod_regex"],
        "ENGINE_PODS": engine,
        "FRONTEND_PODS": w["frontend_pod_regex"],
        "ALL_PODS": w["all_pod_regex"],
        "METRIC_JOBS": w["metric_job_regex"],
        "METRIC_ROLE_LABEL": w["metric_role_label"],
        "DECODE_ROLE_VALUE": w["decode_metric_role_value"],
        "ENGINE_ENDPOINT": w["engine_endpoint"],
        "ENGINE_CONTAINER": w["engine_container"],
        "FRONTEND_CONTAINER": w["frontend_container"],
        "LOG_JOB": w["log_job"],
        "LOG_ROLE_LABEL": w["log_role_label"],
        "PREFILL_LOG_ROLE": w["prefill_log_role_value"],
        "DECODE_LOG_ROLE": w["decode_log_role_value"],
        "CONTAINERS": ",".join(w["log_containers"]),
        "TRACE_PVC": w.get("trace_pvc") or "",
        "PROMETHEUS_UID": c["datasources"]["prometheus"],
        "LOKI_UID": c["datasources"]["loki"],
    }
    for key in (
        "DECODE_ROLE_VALUE",
        "ENGINE_ENDPOINT",
        "ENGINE_CONTAINER",
        "FRONTEND_CONTAINER",
        "LOG_JOB",
        "PREFILL_LOG_ROLE",
        "DECODE_LOG_ROLE",
        "TRACE_PVC",
    ):
        values[key] = json.dumps(values[key], ensure_ascii=False)[1:-1]

    def replace(text: str) -> str:
        return re.sub(r"@@([A-Z_]+)@@", lambda m: values[m[1]], text)

    result = {}
    for name in ("metrics", "logs"):
        if name == "logs" and not d.get("include_logs", True):
            continue
        spec = walk(
            json.loads((HERE / "templates" / f"{name}.json").read_text(encoding="utf-8")), replace
        )
        spec["timeSettings"]["autoRefresh"] = d["refresh"]
        for panel in spec["elements"].values():
            for q in panel["spec"]["data"]["spec"]["queries"]:
                s = q["spec"]["query"]["spec"]
                if q["spec"]["query"]["group"] == "prometheus" and "expr" in s:
                    s["expr"] = s["expr"].replace("[1m]", "[" + d["counter_lookback"] + "]")
                    if "kubelet_volume_stats_" in s["expr"]:
                        s["expr"] = s["expr"].replace(
                            "persistentvolumeclaim=",
                            'namespace="$namespace",persistentvolumeclaim=',
                        )
        if not w.get("trace_pvc"):
            removed = {
                k for k, v in spec["elements"].items() if "kubelet_volume_stats_" in json.dumps(v)
            }
            spec["elements"] = {k: v for k, v in spec["elements"].items() if k not in removed}
            for row in spec["layout"]["spec"]["rows"]:
                items = row["spec"]["layout"]["spec"]["items"]
                row["spec"]["layout"]["spec"]["items"] = [
                    x for x in items if x["spec"]["element"]["name"] not in removed
                ]
            # Repack shared fixed-grid rows after dropping the optional PVC charts.
            for row in spec["layout"]["spec"]["rows"]:
                layout = row["spec"]["layout"]
                if layout["kind"] != "GridLayout":
                    continue
                x = y = height = 0
                for item in layout["spec"]["items"]:
                    s = item["spec"]
                    if x + s["width"] > 24:
                        y += height
                        x = height = 0
                    s.update(x=x, y=y)
                    x += s["width"]
                    height = max(height, s["height"])
        if not d.get("include_logs", True):
            spec["links"] = [x for x in spec["links"] if x.get("url") != "/d/" + d["logs_uid"]]
        if name == "logs":
            add_unique_sessions(spec, c)
        require("@@" not in json.dumps(spec), "Unresolved dashboard placeholder")
        refs = [
            i["spec"]["element"]["name"]
            for r in spec["layout"]["spec"]["rows"]
            for i in r["spec"]["layout"]["spec"]["items"]
        ]
        require(
            len(refs) == len(set(refs)) == len(spec["elements"]),
            "Dashboard layout references are inconsistent",
        )
        result[name] = spec
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": g["configmap"],
            "namespace": g["namespace"],
            "labels": g.get("sidecar_labels", {}),
        },
        # The legacy file provisioner needs uid even though native v2 spec omits it.
        "data": {
            g.get("dashboard_keys", {}).get(name, name + ".json"): json.dumps(
                dict(spec, uid=d[name + "_uid"]), separators=(",", ":")
            )
            for name, spec in result.items()
        },
    }
    return result, cm, monitors(c)


def monitors(c: dict) -> list[dict]:
    """Build Services and ServiceMonitors scoped to the configured workload."""
    scrape = c["scrape"]
    if not scrape.get("enabled", False):
        return []
    result = []
    for role in scrape["roles"]:
        require(role.get("selector"), "Each scrape role needs an explicit Service selector")
        name = c["grafana"]["configmap"] + "-" + role["name"] + "-metrics"
        require(
            bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name)),
            "Generated Service name is invalid or too long",
        )
        label = {"app.kubernetes.io/name": name}
        endpoints = role["endpoints"]
        require(bool(endpoints), "Each scrape role needs an endpoint")
        for e in endpoints:
            require(isinstance(e["port"], int) and 0 < e["port"] < 65536, "Invalid metrics port")
        result.append(
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": name,
                    "namespace": c["kubernetes"]["namespace"],
                    "labels": label,
                },
                "spec": {
                    "selector": role["selector"],
                    "ports": [
                        {
                            "name": e["name"],
                            "port": e["port"],
                            "targetPort": e.get("target_port", e["port"]),
                        }
                        for e in endpoints
                    ],
                },
            }
        )
        result.append(
            {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "ServiceMonitor",
                "metadata": {
                    "name": name,
                    "namespace": scrape["monitor_namespace"],
                    "labels": scrape.get("service_monitor_labels", {}),
                },
                "spec": {
                    "selector": {"matchLabels": label},
                    "namespaceSelector": {"matchNames": [c["kubernetes"]["namespace"]]},
                    "endpoints": [
                        {
                            "port": e["name"],
                            "path": e["path"],
                            "interval": scrape["interval"],
                            "scrapeTimeout": scrape["timeout"],
                            "relabelings": [
                                {
                                    "targetLabel": c["workload"]["metric_role_label"],
                                    "replacement": role["metric_role_value"],
                                }
                            ],
                        }
                        for e in endpoints
                    ],
                },
            }
        )
    return result


def write_outputs(output: Path, dashboards: dict, cm: dict, resources: list[dict]) -> None:
    """Write rendered JSON artifacts without authentication material."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = {name + ".json": spec for name, spec in dashboards.items()}
    values.update(
        {
            "dashboards.configmap.json": cm,
            "scrape-resources.json": {"apiVersion": "v1", "kind": "List", "items": resources},
        }
    )
    for name, value in values.items():
        (output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


class Client:
    """Explicitly scoped Grafana and Kubernetes client with runtime authentication."""

    def __init__(self, config: dict) -> None:
        """Resolve configured credential references in memory."""
        self.c = config
        auth = config["grafana"]["auth"]
        if "token_env" in auth:
            token = os.environ.get(auth["token_env"], "")
            require(bool(token), "Grafana token environment variable is unset")
            self.authorization = "Bearer " + token
        else:
            if "secret" in auth:
                secret = self.kube(config["grafana"]["namespace"], "get", "secret", auth["secret"])[
                    "data"
                ]
                try:
                    user, password = (
                        base64.b64decode(secret[auth[k]], validate=True).decode()
                        for k in ("username_key", "password_key")
                    )
                except (KeyError, ValueError, UnicodeError):
                    raise SetupError(
                        "Grafana Secret is missing valid authentication keys"
                    ) from None
            else:
                user, password = (
                    os.environ.get(auth[k], "") for k in ("username_env", "password_env")
                )
            require(bool(user) and bool(password), "Grafana authentication is unset")
            self.authorization = (
                "Basic " + base64.b64encode((user + ":" + password).encode()).decode()
            )

    def kube(
        self, namespace: str, *args: str, payload: dict | None = None, missing_ok: bool = False
    ) -> dict | None:
        command = [
            self.c["kubernetes"].get("kubectl", "kubectl"),
            "--context",
            self.c["kubernetes"]["context"],
            "-n",
            namespace,
            *args,
        ]
        if payload is None:
            command += ["-o", "json"]
        p = subprocess.run(
            command,
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if missing_ok and p.returncode and "NotFound" in p.stderr:
            return None
        require(
            p.returncode == 0,
            "Kubernetes operation failed (exit "
            + str(p.returncode)
            + "); check context, access and resource names",
        )
        return json.loads(p.stdout) if p.stdout.strip() else None

    def api(
        self,
        path: str,
        payload: dict | None = None,
        method: str | None = None,
        missing_ok: bool = False,
    ) -> dict | None:
        req = urllib.request.Request(
            self.c["grafana"]["url"].rstrip("/") + path,
            method=method,
            headers={"Authorization": self.authorization, "Content-Type": "application/json"},
            data=json.dumps(payload).encode() if payload is not None else None,
        )

        # Credentials may not follow a redirect to another URL.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(
                self,
                req: urllib.request.Request,
                fp: object,
                code: int,
                msg: str,
                headers: object,
                newurl: str,
            ) -> None:
                return None

        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            raise SetupError(
                "Grafana request failed (HTTP " + str(error.code) + "); response body omitted"
            ) from None
        except urllib.error.URLError:
            raise SetupError("Grafana connection failed; check the configured URL/tunnel") from None


def endpoint(c: dict, uid: str = "") -> str:
    """Return the selected native dashboard API endpoint."""
    ns = urllib.parse.quote(c["grafana"].get("api_namespace", "default"), safe="")
    return (
        "/apis/dashboard.grafana.app/"
        + c["grafana"].get("api_version", "v2beta1")
        + "/namespaces/"
        + ns
        + "/dashboards"
        + ("/" + uid if uid else "")
    )


def resource(c: dict, uid: str, desired: dict, current: dict | None) -> dict:
    """Build a native dashboard resource retaining optimistic-concurrency metadata."""
    metadata = (
        deepcopy(current["metadata"])
        if current
        else {"name": uid, "namespace": c["grafana"].get("api_namespace", "default")}
    )
    return {
        "apiVersion": "dashboard.grafana.app/" + c["grafana"].get("api_version", "v2beta1"),
        "kind": "Dashboard",
        "metadata": metadata,
        "spec": desired,
    }


def normalized(value: dict) -> dict:
    """Normalize Grafana defaults before comparing persisted dashboard content."""
    result = deepcopy(value)
    for e in result["elements"].values():
        defaults = e["spec"]["vizConfig"]["spec"].get("fieldConfig", {}).get("defaults", {})
        for step in defaults.get("thresholds", {}).get("steps", []):
            if step.get("value") is None:
                step["value"] = 0
    for link in result.get("links", []):
        link.setdefault("tooltip", "")
    return result


def _revision(value: dict | None) -> str | None:
    """Return the opaque resource version used for optimistic-concurrency checks."""
    return (value or {}).get("metadata", {}).get("resourceVersion")


def deploy(
    c: dict,
    dashboards: dict,
    cm: dict,
    scrape: list[dict],
    client: Client,
    dry_run: bool = False,
    finalize_only: bool = False,
) -> dict:
    """Validate, publish, and verify dashboards using the selected persistence mode."""
    g = c["grafana"]
    names = {name: c["dashboard"][name + "_uid"] for name in dashboards}
    initial = {}
    # Check native-schema API support and all desired dashboards before mutation.
    for name, uid in names.items():
        current = client.api(endpoint(c, uid), missing_ok=True)
        initial[name] = current
        path = endpoint(c, uid) if current else endpoint(c)
        client.api(
            path + "?dryRun=All",
            resource(c, uid, dashboards[name], current),
            "PUT" if current else "POST",
        )
    checkpoint = None
    baseline = None
    if g["publication"] == "configmap-v2":
        baseline = client.kube(g["namespace"], "get", "configmap", g["configmap"], missing_ok=True)
        checkpoint = deepcopy(cm)
        if baseline:
            # Preserve unrelated dashboards and metadata in a shared ConfigMap.
            checkpoint["data"] = dict(baseline.get("data", {}), **cm["data"])
            checkpoint["metadata"]["labels"] = dict(
                baseline["metadata"].get("labels", {}), **checkpoint["metadata"].get("labels", {})
            )
            annotations = {
                k: v
                for k, v in baseline["metadata"].get("annotations", {}).items()
                if k != "kubectl.kubernetes.io/last-applied-configuration"
            }
            if annotations:
                checkpoint["metadata"]["annotations"] = annotations
            if "ownerReferences" in baseline["metadata"]:
                checkpoint["metadata"]["ownerReferences"] = baseline["metadata"]["ownerReferences"]
            checkpoint["metadata"]["resourceVersion"] = baseline["metadata"]["resourceVersion"]
        client.kube(
            g["namespace"], "apply", "--dry-run=server", "-f", "-", "-o", "json", payload=checkpoint
        )
    for item in scrape:
        client.kube(
            item["metadata"]["namespace"],
            "apply",
            "--dry-run=server",
            "-f",
            "-",
            "-o",
            "json",
            payload=item,
        )
    if dry_run:
        return {
            "dry_run": True,
            "dashboards": list(names.values()),
            "scrape_resources": len(scrape),
        }
    provisioned = {}
    if checkpoint:
        if not finalize_only:
            for name, uid in names.items():
                current = client.api(endpoint(c, uid), missing_ok=True)
                require(
                    _revision(current) == _revision(initial[name]),
                    "Dashboard changed before the file checkpoint; retry from fresh state",
                )
        latest = client.kube(g["namespace"], "get", "configmap", g["configmap"], missing_ok=True)
        require(
            _revision(latest) == _revision(baseline)
            and (latest or {}).get("data", {}) == (baseline or {}).get("data", {}),
            "Dashboard ConfigMap changed during validation; retry from fresh state",
        )
        if finalize_only:
            require(
                latest is not None
                and all(latest.get("data", {}).get(k) == v for k, v in cm["data"].items()),
                "Resume requires the exact rendered ConfigMap checkpoint",
            )
        else:
            client.kube(g["namespace"], "apply", "-f", "-", "-o", "json", payload=checkpoint)
        deadline = time.monotonic() + 90
        while True:
            matches = []
            for name, uid in names.items():
                current = client.api(endpoint(c, uid), missing_ok=True)
                key = g.get("dashboard_keys", {}).get(name, name + ".json")
                checksum = hashlib.md5(cm["data"][key].encode(), usedforsecurity=False).hexdigest()
                provisioned[name] = current
                matches.append(
                    bool(current)
                    and current["metadata"].get("annotations", {}).get("grafana.app/sourceChecksum")
                    == checksum
                )
            if all(matches):
                break
            require(
                time.monotonic() < deadline,
                "File provisioner did not acknowledge the source; check sidecar labels and resume with --finalize-only",
            )
            time.sleep(2)
    for item in scrape:
        client.kube(item["metadata"]["namespace"], "apply", "-f", "-", "-o", "json", payload=item)
    for name, uid in names.items():
        current = client.api(endpoint(c, uid), missing_ok=True)
        expected = initial[name] if g["publication"] == "api" else provisioned[name]
        require(
            _revision(current) == _revision(expected),
            "Dashboard changed during validation; retry from fresh state",
        )
        client.api(
            endpoint(c, uid) if current else endpoint(c),
            resource(c, uid, dashboards[name], current),
            "PUT" if current else "POST",
        )
        saved = client.api(endpoint(c, uid))
        require(
            not saved.get("status", {}).get("conversion", {}).get("failed"),
            "Grafana saved an incompatible dashboard schema",
        )
        for field in ("elements", "layout", "variables", "links", "title", "timeSettings"):
            require(
                normalized(saved["spec"]).get(field) == normalized(dashboards[name]).get(field),
                "Saved dashboard verification failed: " + field,
            )
    return {
        "dry_run": False,
        "dashboards": list(names.values()),
        "scrape_resources": len(scrape),
        "verified": True,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("render", "deploy"))
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args(argv)
    (args.output / "receipt.json").unlink(missing_ok=True)
    require(
        args.command == "deploy" or not (args.dry_run or args.finalize_only),
        "--dry-run and --finalize-only require deploy",
    )
    c = load_config(args.config)
    dashboards, cm, scrape = render(c)
    write_outputs(args.output, dashboards, cm, scrape)
    if args.command == "render":
        print("Rendered " + str(len(dashboards)) + " dashboards; no cluster access")
    else:
        require(
            not args.finalize_only or c["grafana"]["publication"] == "configmap-v2",
            "--finalize-only requires configmap-v2 publication",
        )
        result = deploy(c, dashboards, cm, scrape, Client(c), args.dry_run, args.finalize_only)
        (args.output / "receipt.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("Dry run passed" if args.dry_run else "Dashboards published and verified")


if __name__ == "__main__":
    try:
        main()
    except (
        SetupError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        # Do not echo kube stderr, HTTP bodies, config values or authentication.
        print(
            str(error)
            if isinstance(error, SetupError)
            else "Invalid/incomplete configuration or unavailable dependency",
            file=sys.stderr,
        )
        sys.exit(1)
