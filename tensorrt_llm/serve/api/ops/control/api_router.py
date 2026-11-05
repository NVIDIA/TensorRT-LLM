# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resource-governor routes, registered by the ``ResourceGovernor`` itself.

This package has no serving class: the governor (``serve/resource_governor.py``)
keeps registering its own routes on the app, so the app's route class applies
to them. The composition root puts the governor on ``app.state`` when the
executor exposes a resource-governor queue, and the router table attaches this
module only then.
"""

from fastapi import FastAPI

APP_STATE_KEY = "ops_serving_control"


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (the ResourceGovernor) must be set before attaching "
            "tensorrt_llm.serve.api.ops.control; the composition root sets it in the LLM branch "
            "when the executor exposes a resource-governor queue."
        )
    getattr(app.state, APP_STATE_KEY).register_routes(app)
