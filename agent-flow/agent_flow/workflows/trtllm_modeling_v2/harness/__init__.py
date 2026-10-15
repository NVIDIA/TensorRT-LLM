# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programs the gates run on the compute side, under the interpreter of the tree being measured.

Nothing in this package imports ``agent_flow``: each script runs as
``<tree's python> <this directory>/<script>.py`` inside a gate's job, where
``agent_flow`` is generally not installed. Each imports its siblings by
path, and the orchestrator imports the same modules as a package to parse
what they print.

* :mod:`.engine_driver` -- builds the engine exactly as the tree's own
  ``examples/llm-api/quickstart_advanced.py`` does from the gate's argv, runs
  a declared workload (text, images, a local dataset) and prints one
  structured record per request: token ids, full text, iterations.
* :mod:`.serve_smoke` -- starts a deployment entry (``trtllm-serve`` or a
  declared disaggregated launcher), drives it over HTTP and prints one
  structured record per request: status code, text, score.
* :mod:`.workload_io` -- the record format both print, and the local
  dataset reader both use. No network access: a dataset is a local path,
  or a Hugging Face dataset id read strictly from the local cache.
"""
