# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Route snapshot of every trtllm-serve configuration: a new api-stability gate.

Each configuration's live routes must equal the generated reference
``references/trtllm_serve_routes.yaml``. Regenerating the reference cannot hide
the invariants at the bottom of this file: they are checked on the live apps.
The VisualGen configuration is checked by
``tests/unittest/serve/api/visual_gen/test_routes_visual_gen.py`` so that it
runs on VisualGen-only pull requests.
"""

import pytest
from test_common.serve.routes import (
    CONFIGS,
    VISUAL_GEN_CONFIGS,
    compare,
    format_failure,
    read_reference,
    snapshot_config,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def reference():
    return read_reference()


@pytest.fixture(scope="module")
def live_snapshots(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve_routes")
    return {config.name: snapshot_config(config, root / config.name) for config in CONFIGS}


def test_reference_configuration_names(reference):
    expected = [config.name for config in CONFIGS + VISUAL_GEN_CONFIGS]
    assert list(reference["configs"].keys()) == expected, (
        "The configuration names in references/trtllm_serve_routes.yaml must equal the "
        "names in CONFIGS plus VISUAL_GEN_CONFIGS of tests/test_common/serve/routes.py, in "
        "that order. Regenerate with `PYTHONPATH=tests python3 -m test_common.serve.routes "
        "--write`."
    )


@pytest.mark.parametrize("name", [config.name for config in CONFIGS])
def test_routes_match_reference(name, reference, live_snapshots):
    recorded = reference["configs"][name]
    live = live_snapshots[name]
    assert recorded.get("error") == live.get("error"), (
        f"Configuration '{name}' now fails with {live.get('error')!r} instead of "
        f"{recorded.get('error')!r}. If the change is intended, regenerate with "
        "`PYTHONPATH=tests python3 -m test_common.serve.routes --write` and request review "
        "from @NVIDIA/trt-llm-noncommitted-api-review-committee."
    )
    differences = compare(recorded.get("routes", []), live.get("routes", []))
    assert not differences, format_failure(name, differences)


# ---------------------------------------------------------------------------
# Invariants checked on the live apps, independent of the reference.
# Regenerating the reference cannot hide these.
# ---------------------------------------------------------------------------

_DOC_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _routes_of(live_snapshots, name):
    return live_snapshots[name]["routes"]


def _parse(record):
    parts = record.split(" ")
    return parts[0], parts[1], parts[2]


def test_openai_server_routes_are_msgspec(live_snapshots):
    for name, snapshot in live_snapshots.items():
        if snapshot.get("error") or snapshot["kind"] != "openai_server":
            continue
        for record in snapshot["routes"]:
            methods, path, cls = _parse(record)
            if path in _DOC_ROUTES:
                assert cls == "Route", record
            elif methods == "MOUNT":
                assert cls == "Mount", record
            else:
                assert cls == "_MsgspecRoute", (
                    f"configuration '{name}': {record}: every API route of the aggregated "
                    "server must keep the msgpack route class, or workers reject the "
                    "disaggregated frontend's msgpack bodies with 400/422 while plain JSON "
                    "still works."
                )


_API_ROUTE_CLASSES = {
    "disagg_server": (
        "KvHintsRoute",
        "every API route of the disaggregated frontend must keep KvHintsRoute: it strips "
        "kv_hints from request bodies unless the server trusts its ingress "
        "(TRTLLM_TRUST_KV_HINTS=1), and a plain APIRoute skips that policy.",
    ),
    "coordinator_server": (
        "APIRoute",
        "the coordinator app registers FastAPI's plain routes; it takes neither the "
        "disaggregated frontend's KvHintsRoute nor the aggregated server's msgpack route "
        "class.",
    ),
}


def test_disagg_and_coordinator_route_classes(live_snapshots):
    for name, snapshot in live_snapshots.items():
        if snapshot.get("error") or snapshot["kind"] == "openai_server":
            continue
        expected, reason = _API_ROUTE_CLASSES[snapshot["kind"]]
        for record in snapshot["routes"]:
            methods, path, cls = _parse(record)
            if path in _DOC_ROUTES:
                assert cls == "Route", record
            elif methods == "MOUNT":
                assert cls == "Mount", record
            else:
                assert cls == expected, f"configuration '{name}': {record}: {reason}"


def test_control_route_groups(live_snapshots):
    def paths(name):
        return {_parse(record)[1] for record in _routes_of(live_snapshots, name)}

    rl = paths("llm_rl_control")
    assert {"/release_memory", "/resume_memory", "/update_weights"} <= rl
    assert "/memory_status" not in rl, (
        "the RL control group has no /memory_status; the runtime control group owns it"
    )
    runtime = paths("llm_runtime_control")
    assert {"/release_memory", "/resume_memory", "/memory_status"} <= runtime
    assert "/update_weights" not in runtime, (
        "the runtime control group has no /update_weights; the RL control group owns it"
    )
    control_paths = {"/release_memory", "/resume_memory", "/update_weights", "/memory_status"}
    for name in ("llm", "llm_context", "llm_generation", "mm_encoder", "embedding"):
        assert not control_paths & paths(name), (
            f"configuration '{name}' must register no memory-control route: those routes "
            "exist only behind their construction-time switches."
        )


def test_metrics_presence_per_role(live_snapshots):
    embedding_paths = {_parse(r)[1] for r in _routes_of(live_snapshots, "embedding")}
    assert "/metrics" not in embedding_paths, (
        "the embedding server registers no /metrics today; keep the asymmetry until a "
        "change deliberately adds it."
    )
    mm_paths = {_parse(r)[1] for r in _routes_of(live_snapshots, "mm_encoder")}
    assert "/metrics" in mm_paths


def test_launcher_forwarding_shadows_local_routes(live_snapshots):
    """An attached frontend's forwarding routes precede the local handlers.

    Launcher-owned routes are registered before the local table with the same
    method and path (or a path parameter that matches the same requests), and
    FastAPI matches in registration order: flipping the order silently serves
    launcher-owned state from the local process instead.
    """
    records = _routes_of(live_snapshots, "llm_multi_frontend_attached")
    base = _routes_of(live_snapshots, "llm")
    assert len(records) == len(base) + 10, (
        "the attached-frontend configuration registers the ten launcher "
        "forwarding routes on top of the default table"
    )
    forward_positions = [i for i, r in enumerate(records) if r.endswith("#forward")]
    assert len(forward_positions) == 7, records
    for position in forward_positions:
        methods_path = " ".join(records[position].split(" ")[:2])
        shadowed = [
            j
            for j, record in enumerate(records)
            if j != position and " ".join(record.split(" ")[:2]) == methods_path
        ]
        assert shadowed and all(position < j for j in shadowed), (
            f"{records[position]} must stay registered before the local handler of {methods_path}"
        )
    rest_positions = [i for i, r in enumerate(records) if "{rest:path}" in _parse(r)[1]]
    batch_id_positions = [i for i, r in enumerate(records) if "{batch_id}" in _parse(r)[1]]
    assert rest_positions and batch_id_positions
    assert max(rest_positions) < min(batch_id_positions), (
        "the batches forwarding routes (a path parameter that matches every "
        "batch id) must stay registered before the local batch-id handlers"
    )


def test_metrics_shadowing_order(live_snapshots):
    records = _routes_of(live_snapshots, "llm_perf_metrics_enable_metrics_env")
    metrics = [r for r in records if _parse(r)[1] == "/metrics" and _parse(r)[0] == "GET"]
    assert len(metrics) == 2, metrics
    assert metrics[0].endswith("#get_iteration_stats") and metrics[1].endswith("#metrics"), (
        "the JSON /metrics route must stay registered before the Prometheus-text /metrics "
        "route: FastAPI matches in registration order, so swapping them silently changes "
        "the /metrics payload when ENABLE_METRICS is set."
    )


# The only configurations that fail at construction, with their exact errors.
_ERROR_CONFIGURATIONS = {
    "llm_rl_and_runtime_control": (
        "ValueError: Legacy RL control and runtime control endpoints cannot both be enabled."
    ),
    "mm_encoder_rl_control": "ValueError: RL control endpoints require AsyncLLM",
    "embedding_runtime_control": (
        "ValueError: Runtime control endpoints are not supported for embedding servers"
    ),
}


def test_valid_configurations_build(live_snapshots):
    failing = {
        name: snapshot.get("error")
        for name, snapshot in live_snapshots.items()
        if name not in _ERROR_CONFIGURATIONS and "routes" not in snapshot
    }
    assert not failing, (
        f"these configurations must build but fail at construction: {failing}. Only the "
        "configurations in _ERROR_CONFIGURATIONS may fail, and a regenerated reference that "
        "records the error does not change that."
    )


def test_error_configurations(tmp_path):
    by_name = {config.name: config for config in CONFIGS}
    for name, message in _ERROR_CONFIGURATIONS.items():
        snapshot = snapshot_config(by_name[name], tmp_path / name)
        assert snapshot.get("error") == message, (
            f"configuration '{name}' must fail at construction with exactly "
            f"{message!r}, got {snapshot.get('error')!r}: the conflicting route groups are "
            "rejected before any route is registered."
        )
