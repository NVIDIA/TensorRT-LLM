# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intake: the ``task.yaml`` a run starts from (see ``INTAKE.md``).

- ``schema.py``   the user-facing ``task.yaml`` and the per-cluster ``site.yaml``
- ``derive.py``   the fast path: a ``task.yaml`` derived from the checkpoint, the site and a goal
- ``checks.py``   feasibility checks against the checkpoint and the request
- ``freeze.py``   confirmation: a digest of what the user agreed to, kept beside the request
- ``compile.py``  ``task.yaml`` + ``site.yaml`` -> the resolved spec the pipeline runs
- ``cli.py``      ``trtllm-modeling-v2 intake derive|check|confirm|compile``
"""
