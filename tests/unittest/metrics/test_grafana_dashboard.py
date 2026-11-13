# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portability, safe publication, and authentication boundary tests."""

import base64
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.response
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DIR = ROOT / "examples/serve/grafana"
_spec = importlib.util.spec_from_file_location(
    "trtllm_grafana_dashboard", DASHBOARD_DIR / "dashboard.py"
)
assert _spec is not None and _spec.loader is not None
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def config() -> dict:
    c = app.load_config(DASHBOARD_DIR / "example.config.json")
    c["dashboard"]["include_logs"] = True
    return c


class SessionPanelTests(unittest.TestCase):
    def test_optional_logs_only_and_idempotent(self) -> None:
        c = config()
        baseline, _, _ = app.render(c)
        c["dashboard"]["unique_sessions"] = {"enabled": True}
        dashboards, _, _ = app.render(c)
        self.assertEqual(baseline["metrics"], dashboards["metrics"])
        logs = dashboards["logs"]
        self.assertEqual(len(logs["elements"]), len(baseline["logs"]["elements"]) + 1)
        before = deepcopy(logs)
        app.add_unique_sessions(logs, c)
        self.assertEqual(before, logs)
        panel = logs["elements"]["session-unique-observed"]["spec"]
        queries = panel["data"]["spec"]["queries"]
        self.assertEqual(len(queries), 9)
        for i, q in enumerate(queries[:8]):
            expr = q["spec"]["query"]["spec"]["expr"]
            self.assertIn("sum by (session_id)", expr)
            self.assertIn('session_partition="' + str(i) + '"', expr)
            self.assertNotIn("$search", expr)
            self.assertIn("[${session_window:raw}]", expr)
        self.assertEqual(
            queries[-1]["spec"]["query"]["spec"]["expression"],
            "$A + $B + $C + $D + $E + $F + $G + $H",
        )
        values = [
            x["spec"]["value"]
            for x in logs["layout"]["spec"]["rows"][0]["spec"]["conditionalRendering"]["spec"][
                "items"
            ]
        ]
        self.assertIn("e2e", values)

    def test_invalid_window_rejected(self) -> None:
        c = config()
        c["dashboard"]["unique_sessions"] = {
            "enabled": True,
            "windows": ["120m"],
            "default_window": "120m",
        }
        with self.assertRaises(app.SetupError):
            app.render(c)


class FakeClient:
    def __init__(self, c: dict) -> None:
        self.c = c
        self.calls = []
        self.saved = {}
        self.cm = None
        self.edit_on_get = False

    def api(
        self,
        path: str,
        payload: dict | None = None,
        method: str | None = None,
        missing_ok: bool = False,
    ) -> dict | None:
        self.calls.append(("api", path, method))
        uid = path.split("?")[0].rsplit("/", 1)[-1]
        if payload is not None:
            if "dryRun=All" in path:
                return deepcopy(payload)
            uid = payload["metadata"]["name"]
            self.saved[uid] = deepcopy(payload)
            self.saved[uid]["metadata"]["resourceVersion"] = "2"
            return deepcopy(self.saved[uid])
        result = deepcopy(self.saved.get(uid))
        if self.edit_on_get and result:
            result["metadata"]["resourceVersion"] = str(len(self.calls))
        return result

    def kube(
        self, namespace: str, *args: str, payload: dict | None = None, missing_ok: bool = False
    ) -> dict | None:
        self.calls.append(("kube", namespace, args))
        if args[0] == "get":
            return deepcopy(self.cm)
        if "--dry-run=server" not in args:
            self.cm = deepcopy(payload)
            self.cm["metadata"]["resourceVersion"] = "2"
            for name in ("metrics", "logs"):
                if name + ".json" not in self.cm["data"]:
                    continue
                uid = self.c["dashboard"][name + "_uid"]
                self.saved[uid] = {
                    "metadata": {
                        "name": uid,
                        "resourceVersion": "1",
                        "annotations": {
                            "grafana.app/sourceChecksum": hashlib.md5(
                                self.cm["data"][name + ".json"].encode(), usedforsecurity=False
                            ).hexdigest()
                        },
                    },
                    "spec": {},
                }
        return deepcopy(payload)


class DashboardTests(unittest.TestCase):
    def test_offline_render_has_no_auth_or_kubernetes_access(self) -> None:
        with patch("subprocess.run", side_effect=AssertionError("Cluster access during render")):
            dashboards, cm, scrape = app.render(config())
        self.assertEqual(len(dashboards["metrics"]["elements"]), 80)
        panels = dashboards["metrics"]["elements"]
        self.assertTrue({"panel-106", "panel-132", "panel-104", "panel-129"}.isdisjoint(panels))
        for key in ("panel-114", "panel-140"):
            queries = panels[key]["spec"]["data"]["spec"]["queries"]
            self.assertEqual(len(queries), 2)
            for q, percentile in zip(queries, ("0.5", "0.95")):
                expr = q["spec"]["query"]["spec"]["expr"]
                self.assertIn("histogram_quantile(" + percentile + ",", expr)
                self.assertIn("trtllm_request_inference_time_seconds_bucket", expr)
                self.assertIn("sum by(le,workload_role,pod)", expr)
        self.assertEqual(len(dashboards["logs"]["elements"]), 16)
        self.assertEqual(scrape, [])
        self.assertNotIn("auth", cm)
        self.assertEqual(json.loads(cm["data"]["metrics.json"])["uid"], "inference-metrics")
        self.assertNotIn("uid", dashboards["metrics"])

    def test_second_cluster_uses_only_its_own_inputs(self) -> None:
        c = config()
        c["kubernetes"].update(context="second-context", namespace="model-team")
        c["dashboard"].update(
            metrics_uid="model-metrics", logs_uid="model-logs", title="Model team"
        )
        c["datasources"] = {"prometheus": "metrics-other", "loki": "logs-other"}
        w = c["workload"]
        w.update(
            prefill_pod_regex="context-worker-.*",
            decode_pod_regex="generation-worker-.*",
            frontend_pod_regex="api-worker-.*",
            all_pod_regex=".*-worker-.*",
            metric_job_regex="team-model-.*",
            metric_role_label="engine_role",
            trace_pvc="trace-data",
        )
        dashboards, cm, _ = app.render(c)
        text = json.dumps(dashboards)
        for unwanted in (
            "inference-prefill-",
            "inference-decode-",
            "inference-frontend-",
            "inference-metrics",
            "workload_role",
            "@@",
        ):
            self.assertNotIn(unwanted, text)
        self.assertEqual(len(dashboards["metrics"]["elements"]), 82)
        self.assertIn("/d/model-logs", text)
        self.assertIn("engine_role", text)
        for panel in dashboards["metrics"]["elements"].values():
            for q in panel["spec"]["data"]["spec"]["queries"]:
                self.assertEqual(q["spec"]["query"]["datasource"]["name"], "metrics-other")
                expr = q["spec"]["query"]["spec"].get("expr", "")
                if "kubelet_volume_stats_" in expr:
                    self.assertIn('namespace="$namespace",persistentvolumeclaim="trace-data"', expr)
        self.assertNotIn("second-context", json.dumps(cm))

    def test_no_logs_option_removes_navigation_without_deleting_other_dashboards(self) -> None:
        c = config()
        c["dashboard"]["include_logs"] = False
        dashboards, cm, _ = app.render(c)
        self.assertEqual(set(dashboards), {"metrics"})
        self.assertNotIn("/d/inference-logs", json.dumps(dashboards))
        self.assertEqual(set(cm["data"]), {"metrics.json"})

    def test_existing_configmap_file_keys_can_be_retained(self) -> None:
        c = config()
        c["grafana"]["dashboard_keys"] = {
            "metrics": "owned-metrics.json",
            "logs": "owned-logs.json",
        }
        _, cm, _ = app.render(c)
        self.assertEqual(set(cm["data"]), {"owned-metrics.json", "owned-logs.json"})

    def test_optional_monitors_have_scoped_selectors_and_labels(self) -> None:
        c = config()
        c["scrape"]["enabled"] = True
        _, _, resources = app.render(c)
        self.assertEqual(len(resources), 6)
        for service, monitor in zip(resources[::2], resources[1::2]):
            self.assertEqual(service["metadata"]["namespace"], "inference")
            self.assertEqual(monitor["metadata"]["namespace"], "monitoring")
            self.assertEqual(monitor["spec"]["namespaceSelector"], {"matchNames": ["inference"]})
            self.assertEqual(
                monitor["spec"]["selector"]["matchLabels"], service["metadata"]["labels"]
            )

    def test_dry_run_never_publishes(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        receipt = app.deploy(c, dashboards, cm, scrape, client, dry_run=True)
        self.assertTrue(receipt["dry_run"])
        self.assertFalse(client.saved)
        self.assertIsNone(client.cm)
        for call in client.calls:
            if call[0] == "api" and call[2]:
                self.assertIn("dryRun=All", call[1])
            if call[0] == "kube" and call[2][0] == "apply":
                self.assertIn("--dry-run=server", call[2])

    def test_api_creation_and_postsave_verification(self) -> None:
        c = config()
        ds, cm, s = app.render(c)
        client = FakeClient(c)
        result = app.deploy(c, ds, cm, s, client)
        self.assertTrue(result["verified"])
        self.assertEqual(len(client.saved), 2)
        self.assertIsNone(client.cm)

    def test_shared_configmap_preserves_other_dashboards_binary_data_and_labels(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        ds, cm, s = app.render(c)
        client = FakeClient(c)
        client.cm = {
            "metadata": {
                "resourceVersion": "1",
                "labels": {"keep": "yes"},
                "annotations": {"folder": "keep"},
            },
            "data": {"unrelated.json": "untouched"},
            "binaryData": {"unrelated.bin": "AP8="},
        }
        app.deploy(c, ds, cm, s, client)
        self.assertEqual(client.cm["data"]["unrelated.json"], "untouched")
        self.assertEqual(client.cm["binaryData"], {"unrelated.bin": "AP8="})
        self.assertEqual(client.cm["metadata"]["labels"]["keep"], "yes")
        self.assertEqual(client.cm["metadata"]["annotations"]["folder"], "keep")

    def test_concurrent_dashboard_change_stops_publication(self) -> None:
        c = config()
        ds, cm, s = app.render(c)
        client = FakeClient(c)
        for name, spec in ds.items():
            uid = c["dashboard"][name + "_uid"]
            client.saved[uid] = app.resource(c, uid, spec, None)
            client.saved[uid]["metadata"]["resourceVersion"] = "1"
        client.edit_on_get = True
        with self.assertRaisesRegex(app.SetupError, "changed during validation"):
            app.deploy(c, ds, cm, s, client)
        self.assertFalse(
            any(x[0] == "api" and x[2] and "dryRun=All" not in x[1] for x in client.calls)
        )

    def test_auth_errors_do_not_echo_server_body_or_secret(self) -> None:
        c = config()
        secret = "private-sentinel"
        with patch.dict(os.environ, {"GRAFANA_TOKEN": secret}):
            client = app.Client(c)
        error = urllib.error.HTTPError(
            "https://internal.invalid", 403, "denied", {}, io.BytesIO(secret.encode())
        )
        with patch("urllib.request.OpenerDirector.open", side_effect=error):
            with self.assertRaises(app.SetupError) as caught:
                client.api("/test")
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("internal.invalid", str(caught.exception))

    def test_kubernetes_context_and_namespace_are_explicit(self) -> None:
        c = config()
        with patch.dict(os.environ, {"GRAFANA_TOKEN": "test-only"}):
            client = app.Client(c)
        with patch(
            "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "{}", "")
        ) as run:
            client.kube("monitoring", "get", "configmap", "test")
        command = run.call_args.args[0]
        self.assertEqual(
            command[:5], ["kubectl", "--context", "example-context", "-n", "monitoring"]
        )
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_kubernetes_errors_do_not_expose_stderr(self) -> None:
        c = config()
        with patch.dict(os.environ, {"GRAFANA_TOKEN": "test-only"}):
            client = app.Client(c)
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "", "private-error-sentinel"),
        ):
            with self.assertRaises(app.SetupError) as caught:
                client.kube("monitoring", "get", "secret", "auth")
        self.assertNotIn("private-error-sentinel", str(caught.exception))

    def test_redirects_are_rejected_without_echoing_location(self) -> None:
        c = config()
        with patch.dict(os.environ, {"GRAFANA_TOKEN": "test-only"}):
            client = app.Client(c)
        error = urllib.error.HTTPError(
            "https://redirect.invalid",
            302,
            "redirect",
            {"Location": "https://other.invalid"},
            io.BytesIO(b""),
        )
        with patch("urllib.request.OpenerDirector.open", side_effect=error):
            with self.assertRaisesRegex(app.SetupError, "HTTP 302"):
                client.api("/test")
        # The handler handed to urllib refuses even same-origin redirects.
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaises(app.SetupError):
                client.api("/test")
            handler = build.call_args.args[0]
            self.assertIsNone(
                handler.redirect_request(None, None, 302, "redirect", {}, "https://other.invalid")
            )

    def test_missing_target_or_inline_credentials_are_rejected(self) -> None:
        for update in ("context", "inline", "remote-http"):
            c = config()
            if update == "context":
                c["kubernetes"]["context"] = ""
            elif update == "inline":
                c["grafana"]["auth"] = {"password": "do-not-allow"}
            else:
                c["grafana"]["url"] = "http://cluster.example.invalid"
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp) / "config.json"
                p.write_text(json.dumps(c))
                with self.assertRaises(app.SetupError):
                    app.load_config(p)


class ReviewedPublicationTests(unittest.TestCase):
    def test_default_example_requires_no_loki_and_render_is_offline(self) -> None:
        c = app.load_config(DASHBOARD_DIR / "example.config.json")
        with patch.object(app, "Client", side_effect=AssertionError("Network during render")):
            dashboards, _, _ = app.render(c)
        self.assertEqual(set(dashboards), {"metrics"})
        self.assertNotIn(c["dashboard"]["logs_uid"], json.dumps(dashboards))

    def test_health_queries_use_configured_job_not_service(self) -> None:
        c = config()
        c["workload"]["metric_job_regex"] = "selected-metrics-job"
        dashboards, _, _ = app.render(c)
        for key in ("panel-1", "panel-39"):
            query = dashboards["metrics"]["elements"][key]["spec"]["data"]["spec"]["queries"][0]
            expr = query["spec"]["query"]["spec"]["expr"]
            self.assertIn('job=~"selected-metrics-job"', expr)
            self.assertNotIn("service=~", expr)

    def test_new_tier_panels_keep_scope_units_and_missing_data_rules(self) -> None:
        c = config()
        c["dashboard"]["counter_lookback"] = "2m"
        dashboards, _, _ = app.render(c)
        metrics = dashboards["metrics"]
        panels = [p for key, p in metrics["elements"].items() if key.startswith("observability-")]
        self.assertEqual(len(panels), 14)
        for p in panels:
            spec = p["spec"]
            custom = spec["vizConfig"]["spec"]["fieldConfig"]["defaults"]["custom"]
            self.assertEqual(custom["stacking"]["mode"], "none")
            for query in spec["data"]["spec"]["queries"]:
                expr = query["spec"]["query"]["spec"]["expr"]
                self.assertIn('namespace="$namespace"', expr)
                self.assertIn('job=~"inference-.*-metrics"', expr)
                self.assertIn("pod=~", expr)
                self.assertNotIn("or vector(0)", expr)
                if "Tier restore / eviction turnover" in spec["title"]:
                    self.assertIn("rate(", expr)
                    self.assertIn("[2m]", expr)
                    self.assertIn("group_left", expr)
                if "Sampled restore latency" in spec["title"]:
                    self.assertIn("[5m]", expr)
                    self.assertIn("source_cache_level,destination_cache_level", expr)
        for row in metrics["layout"]["spec"]["rows"]:
            for item in row["spec"]["layout"]["spec"]["items"]:
                key = item["spec"]["element"]["name"]
                if key.startswith("observability-"):
                    rule = item["spec"]["conditionalRendering"]["spec"]
                    self.assertEqual(rule["items"][0]["kind"], "ConditionalRenderingData")

    def test_configmap_native_edit_after_checkpoint_stops_final_put(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        original = client.api
        reads = {}

        def api(
            path: str,
            payload: dict | None = None,
            method: str | None = None,
            missing_ok: bool = False,
        ) -> dict | None:
            value = original(path, payload, method, missing_ok)
            if payload is None and client.cm is not None and value is not None:
                uid = path.rsplit("/", 1)[-1]
                reads[uid] = reads.get(uid, 0) + 1
                if reads[uid] == 2:
                    client.saved[uid]["metadata"]["resourceVersion"] = "edited"
                    client.saved[uid]["spec"]["title"] = "Concurrent user edit"
                    value = deepcopy(client.saved[uid])
            return value

        client.api = api
        with self.assertRaisesRegex(app.SetupError, "changed during validation"):
            app.deploy(c, dashboards, cm, scrape, client)
        self.assertFalse(
            any(
                k == "api" and method and "dryRun=All" not in path
                for k, path, method in client.calls
            )
        )
        self.assertTrue(
            any(v["spec"].get("title") == "Concurrent user edit" for v in client.saved.values())
        )

    def test_configmap_metadata_edit_is_not_overwritten(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        client.cm = {"metadata": {"resourceVersion": "1"}, "data": {"unrelated.json": "keep"}}
        original = client.kube
        reads = 0

        def kube(
            namespace: str, *args: str, payload: dict | None = None, missing_ok: bool = False
        ) -> dict | None:
            nonlocal reads
            if args[0] == "get":
                reads += 1
                if reads == 2:
                    client.cm["metadata"]["resourceVersion"] = "2"
            return original(namespace, *args, payload=payload, missing_ok=missing_ok)

        client.kube = kube
        with self.assertRaisesRegex(app.SetupError, "ConfigMap changed"):
            app.deploy(c, dashboards, cm, scrape, client)
        self.assertEqual(client.cm["data"], {"unrelated.json": "keep"})

    def test_api_version_can_be_selected_explicitly(self) -> None:
        c = config()
        for version in ("v2", "v2beta1"):
            c["grafana"]["api_version"] = version
            self.assertIn("/" + version + "/", app.endpoint(c))
            self.assertEqual(
                app.resource(c, "example", {}, None)["apiVersion"],
                "dashboard.grafana.app/" + version,
            )

    def test_invalid_bool_and_non_re2_groups_are_rejected(self) -> None:
        for field in ("bool", "regex"):
            c = config()
            if field == "bool":
                c["scrape"]["enabled"] = "false"
            else:
                c["workload"]["prefill_pod_regex"] = "(?=prefix).*"
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp) / "config.json"
                p.write_text(json.dumps(c))
                with self.assertRaises(app.SetupError):
                    app.load_config(p)

    def test_secret_reference_is_a_resource_name_not_a_cli_option(self) -> None:
        c = config()
        c["grafana"]["auth"] = {"secret": "-A", "username_key": "user", "password_key": "pass"}
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.json"
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(app.SetupError, "Secret name"):
                app.load_config(p)

    def test_failed_invocation_drops_stale_success_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "receipt.json"
            p.write_text('{"verified":true}')
            with patch.object(app, "load_config", side_effect=app.SetupError("invalid config")):
                with self.assertRaises(app.SetupError):
                    app.main(["render", "--config", "missing.json", "--output", temp])
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()


class CodeRabbitRegressionTests(unittest.TestCase):
    def test_omitted_logs_flag_renders_metrics_without_logs_navigation(self) -> None:
        c = config()
        del c["dashboard"]["include_logs"]
        dashboards, cm, _ = app.render(c)
        self.assertEqual(set(dashboards), {"metrics"})
        self.assertNotIn("logs.json", cm["data"])
        self.assertFalse(
            any(
                link.get("url") == "/d/" + c["dashboard"]["logs_uid"]
                for link in dashboards["metrics"]["links"]
            )
        )

    def test_loopback_credentials_bypass_environment_proxy(self) -> None:
        c = config()
        c["grafana"]["auth"] = {"token_env": "DASHBOARD_TEST_TOKEN"}
        for host in ("127.0.0.1", "localhost", "[::1]"):
            with self.subTest(host=host):
                c["grafana"]["url"] = "http://" + host + ":3000"
                requests = []

                def respond(handler: object, request: object) -> object:
                    requests.append(request)
                    response = urllib.response.addinfourl(
                        io.BytesIO(b"{}"), {}, request.full_url, 200
                    )
                    response.msg = "OK"
                    return response

                with (
                    patch.dict(os.environ, {"DASHBOARD_TEST_TOKEN": "test-token"}),
                    patch(
                        "urllib.request.getproxies",
                        return_value={"http": "http://proxy.invalid:3128"},
                    ),
                    patch("urllib.request.HTTPHandler.http_open", respond),
                ):
                    self.assertEqual(app.Client(c).api("/health"), {})
                self.assertEqual(len(requests), 1)
                self.assertFalse(requests[0].has_proxy())
                self.assertNotEqual(requests[0].host, "proxy.invalid:3128")
                self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-token")

    def test_custom_role_legend_and_instance_worker_scope(self) -> None:
        c = config()
        c["workload"]["log_role_label"] = "worker_role"
        dashboards, _, _ = app.render(c)
        logs = dashboards["logs"]
        instance = logs["elements"]["panel-15"]["spec"]
        query = instance["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]["expr"]
        self.assertIn("${worker:regex}", query)
        self.assertIn(c["workload"]["frontend_pod_regex"], query)
        self.assertEqual(query.count("pod=~"), 1)
        self.assertIn("${container:regex}", query)
        text = json.dumps(logs)
        self.assertGreaterEqual(text.count("{{worker_role}}"), 12)
        self.assertNotIn("{{role}}", text)

    def test_finalize_only_accepts_matching_checkpoint_without_applying_it(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        client.kube(c["grafana"]["namespace"], "apply", payload=cm)
        client.calls.clear()
        app.deploy(c, dashboards, cm, scrape, client, finalize_only=True)
        self.assertFalse(
            any(
                call[0] == "kube" and call[2][0] == "apply" and "--dry-run=server" not in call[2]
                for call in client.calls
            )
        )

    def test_finalize_only_rejects_changed_checkpoint(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        client.cm = deepcopy(cm)
        client.cm["metadata"]["resourceVersion"] = "1"
        client.cm["data"]["metrics.json"] = "{}"
        with self.assertRaisesRegex(
            app.SetupError, "Resume requires the exact rendered ConfigMap checkpoint"
        ):
            app.deploy(c, dashboards, cm, scrape, client, finalize_only=True)

    def test_file_provisioner_timeout_is_bounded(self) -> None:
        c = config()
        c["grafana"]["publication"] = "configmap-v2"
        dashboards, cm, scrape = app.render(c)
        client = FakeClient(c)
        client.cm = deepcopy(cm)
        client.cm["metadata"]["resourceVersion"] = "1"
        with (
            patch.object(app.time, "monotonic", side_effect=[0, 100]),
            patch.object(
                app.time, "sleep", side_effect=AssertionError("Should fail before sleeping")
            ),
            self.assertRaisesRegex(app.SetupError, "File provisioner did not acknowledge"),
        ):
            app.deploy(c, dashboards, cm, scrape, client, finalize_only=True)

    def test_secret_auth_decodes_keys_and_rejects_missing_values(self) -> None:
        c = config()
        c["grafana"]["auth"] = {
            "secret": "grafana-auth",
            "username_key": "user",
            "password_key": "password",
        }
        with patch.object(
            app.Client,
            "kube",
            return_value={
                "data": {
                    "user": base64.b64encode(b"admin").decode(),
                    "password": base64.b64encode(b"test-password").decode(),
                }
            },
        ):
            expected = "Basic " + base64.b64encode(b"admin:test-password").decode()
            self.assertEqual(app.Client(c).authorization, expected)
        with (
            patch.object(app.Client, "kube", return_value={"data": {}}),
            self.assertRaisesRegex(app.SetupError, "missing valid authentication keys"),
        ):
            app.Client(c)
