<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Online Serving Examples with `trtllm-serve`

We provide a CLI command, `trtllm-serve`, to launch a FastAPI server compatible with OpenAI APIs, here are some client examples to query the server, you can check the source code here or refer to the [command documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) and [examples](https://nvidia.github.io/TensorRT-LLM/examples/trtllm_serve_examples.html) for detailed information and usage guidelines.


## Grafana monitoring

The [Grafana setup example](grafana/README.md) renders and publishes role-aware
Kubernetes inference dashboards, including KV-cache pressure, transfer and
executor performance views. Start with an offline render and the neutral example
configuration; Grafana/Prometheus deployment and inference settings remain separate.
