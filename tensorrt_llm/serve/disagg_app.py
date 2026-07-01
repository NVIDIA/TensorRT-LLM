# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import-string app factory for the disaggregated OpenAI server.

uvicorn only runs multiple worker processes (``WEB_CONCURRENCY``/``workers>1``)
when the app is given as an import string it can re-import in each spawned
worker -- a constructed FastAPI instance is rejected (and is unpicklable). This
module exposes :func:`create_app`, a zero-arg factory (``factory=True``) that
each worker calls to rebuild the ``OpenAIDisaggServer`` from configuration
passed via environment variables.

Parameters are passed by env (not args) because the factory takes no arguments:

* ``TLLM_DISAGG_CONFIG_FILE``            -- path to the disagg YAML (required)
* ``TLLM_DISAGG_METADATA_CONFIG_FILE``   -- optional metadata server config path
* ``TLLM_DISAGG_REQUEST_TIMEOUT``        -- request timeout secs (default 180)
* ``TLLM_DISAGG_SERVER_START_TIMEOUT``   -- server start timeout secs (default 180)
* ``TLLM_DISAGG_METRICS_INTERVAL``       -- metrics interval secs (default 0)

Single-process serving does not need this module; it is used only on the
``workers>1`` path.
"""

import os

from tensorrt_llm.llmapi.disagg_utils import (parse_disagg_config_file,
                                              parse_metadata_server_config_file)
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

CONFIG_FILE_ENV = "TLLM_DISAGG_CONFIG_FILE"
METADATA_CONFIG_FILE_ENV = "TLLM_DISAGG_METADATA_CONFIG_FILE"
REQUEST_TIMEOUT_ENV = "TLLM_DISAGG_REQUEST_TIMEOUT"
SERVER_START_TIMEOUT_ENV = "TLLM_DISAGG_SERVER_START_TIMEOUT"
METRICS_INTERVAL_ENV = "TLLM_DISAGG_METRICS_INTERVAL"


def build_disagg_server() -> OpenAIDisaggServer:
    """Rebuild the OpenAIDisaggServer from env-provided configuration."""
    config_file = os.environ.get(CONFIG_FILE_ENV)
    if not config_file:
        raise RuntimeError(
            f"{CONFIG_FILE_ENV} must be set to build the disagg app in a "
            "worker process (workers>1 path).")
    disagg_cfg = parse_disagg_config_file(config_file)
    metadata_server_cfg = parse_metadata_server_config_file(
        os.environ.get(METADATA_CONFIG_FILE_ENV))
    return OpenAIDisaggServer(
        config=disagg_cfg,
        req_timeout_secs=int(os.environ.get(REQUEST_TIMEOUT_ENV, "180")),
        server_start_timeout_secs=int(
            os.environ.get(SERVER_START_TIMEOUT_ENV, "180")),
        metadata_server_cfg=metadata_server_cfg,
        metrics_interval_secs=int(os.environ.get(METRICS_INTERVAL_ENV, "0")))


def create_app():
    """uvicorn factory (``factory=True``): return the FastAPI app for a worker."""
    return build_disagg_server().app
