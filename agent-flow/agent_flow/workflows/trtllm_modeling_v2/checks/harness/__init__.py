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
* :mod:`.bundle_driver` -- builds that same engine once and runs several
  gates' workloads on it, each between section markers: engine-driver specs,
  logit-driver specs and ``trtllm-eval`` in process (``checks/bundle.py``
  bundles them and cuts the output into one result per gate).
* :mod:`.engine_facts` -- what the engine runs once it is up (the
  configuration after the LLM API and the executor adjusted it, its pools,
  the device memory left free), read back by the engine driver and the
  shims and printed as one ``MV2FLOW_ENGINE`` record.
* ``rankhook/sitecustomize.py`` -- not a module: the engine gates put its
  directory first on ``PYTHONPATH``, and every rank's executor worker, in
  whatever process it runs, prints one ``MV2FLOW_RANK_ENGINE`` record of its
  own engine as it is set up (:func:`.engine_facts.emit_rank`).
* :mod:`.engine_reuse` -- what the engine reused from its KV cache over the
  workload, read from the ``cached_tokens`` each response carries, and a
  probe that sends one prompt twice; printed as one ``MV2FLOW_REUSE`` record.
* :mod:`.geometry_probe` -- boots the engine the run's runtime block builds
  before any kernel is certified and prints the serving geometry it runs: the
  KV page view, the limits, the snapshot policy and the context chunks of a
  few probe lengths (``run/geometry.py`` starts it in phase 0).
* :mod:`.disagg_leg_b` -- the equivalence witness's leg B when its block
  names no ``contrast_command``: leg A's engine split into a context and a
  generation engine with the KV cache handed over, printing the witness
  record format (``checks/witness.py``).
* :mod:`.quickstart_shim` -- the tree's ``quickstart_advanced.py`` as a
  witness leg runs it, with both records added.
* :mod:`.eval_shim` -- the tree's ``trtllm-eval`` as the accuracy gate runs
  it, with both records added.
* :mod:`.workload_io` -- the record format it prints, and the local
  dataset reader it uses. No network access: a dataset is a local path,
  or a Hugging Face dataset id read strictly from the local cache.
"""
