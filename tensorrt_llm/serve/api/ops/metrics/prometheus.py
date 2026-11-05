# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Prometheus mount: its ``GET /metrics`` and the ``/prometheus/metrics`` app.

Attached last of all routers, so the JSON ``GET /metrics`` of the metrics
router stays registered first (FastAPI matches in registration order), and
before the composition root adds its middleware, so the instrumentator's
middleware stays innermost.
"""

import re

from fastapi import FastAPI
from starlette.routing import Mount


def mount_metrics(app):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
    from prometheus_fastapi_instrumentator import Instrumentator
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    Instrumentator(
        should_group_status_codes=False,
        should_respect_env_var=True,
        excluded_handlers=[".*"],
        registry=registry,
    ).add().instrument(app).expose(app)
    metrics_app = make_asgi_app(registry=registry)
    metrics_route = Mount("/prometheus/metrics", metrics_app)
    metrics_route.path_regex = re.compile("^/prometheus/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def attach_router(app: FastAPI) -> None:
    # Nothing on app.state: the mount reads only the app and the process-wide
    # Prometheus registry.
    mount_metrics(app)
