<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress (MX) Checkpoint Loading

TensorRT-LLM can use ModelExpress (MX) as a checkpoint-loading path for
PyTorch backend deployments. When `checkpoint_format="MX"` is selected,
TensorRT-LLM attempts to fetch compatible weights from another running
TensorRT-LLM instance through the MX server. If no compatible source is
available, or if MX transfer fails, loading falls back to the standard
Hugging Face checkpoint path.

This integration is intended to reduce repeated disk reads when multiple
TensorRT-LLM workers load the same model. A worker that loads from disk can
publish its weights as an MX source, and later workers can receive those
weights directly through MX.

## Installation

The MX Python client is included in the TensorRT-LLM requirements. Packaging
flows that want to declare the MX dependency explicitly can also install the MX
extra:

```bash
pip install "tensorrt_llm[mx]"
```

The automatic local server launch also requires Docker, because TensorRT-LLM
starts the MX server and Redis metadata backend as local containers.

## Local Launch

For a single-node deployment without Kubernetes, set the checkpoint format to
`MX`:

```python
from tensorrt_llm import LLM

llm = LLM(
    model="/path/to/model",
    checkpoint_format="MX",
)
```

You can also use the same option from a `trtllm-serve` config:

```yaml
checkpoint_format: MX
mx_config:
  local_server:
    enabled: true
    port: 8001
```

```bash
trtllm-serve /path/to/model --config config.yaml
```

When `checkpoint_format` is `MX` and neither `mx_config.server_url` nor
`MODEL_EXPRESS_URL` is set, TensorRT-LLM starts a local Docker-backed MX
server on `127.0.0.1:8001`. The launcher:

1. Reuses the matching TRT-LLM-managed MX server container when it is already
   listening on the configured local port.
2. Creates a Docker network for the local MX containers.
3. Starts Redis for the MX metadata backend.
4. Starts the MX server container and propagates the resulting URL to the MX
   checkpoint loader.

If the local server cannot be started, or if the configured local port is
already occupied by something other than the matching TRT-LLM-managed MX server
container, TensorRT-LLM logs a warning and continues with disk-based Hugging
Face checkpoint loading.

## External MX Server

If an MX server is managed outside TensorRT-LLM, provide its URL explicitly:

```yaml
checkpoint_format: MX
mx_config:
  server_url: http://mx-server.example.com:8001
  local_server:
    enabled: false
```

The `MODEL_EXPRESS_URL` environment variable can also provide the server URL
when `mx_config.server_url` is not set.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `mx_config.server_url` | `null` | URL of an externally managed MX server. |
| `mx_config.server_query_timeout_s` | `null` | Timeout for MX source discovery. When unset, TensorRT-LLM uses a short fallback cap when no source exists and otherwise lets MX wait for long donor loads. |
| `mx_config.local_server.enabled` | `true` | Starts a local MX server when `checkpoint_format="MX"` and no server URL is configured. |
| `mx_config.local_server.port` | `8001` | Host TCP port for the local MX server. |
| `mx_config.local_server.server_image` | `nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0` | Docker image for the local MX server. |
| `mx_config.local_server.redis_image` | `redis:8-alpine` | Docker image for the Redis metadata backend. |
| `mx_config.local_server.startup_timeout_s` | `30` | Seconds to wait for the local MX server to accept connections. |

## Notes and Limitations

- The automatic server launch is local-only. Kubernetes and other managed
  server lifecycles should provide `mx_config.server_url` or
  `MODEL_EXPRESS_URL` instead.
- Docker must be available for the local launch path.
- The first worker may still load weights from disk if no compatible MX source
  is already registered.
- This page describes the MX checkpoint-loading path only. GPU Memory Service
  (GMS) integration is configured separately.
