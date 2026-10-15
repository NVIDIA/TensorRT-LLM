# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tune: the performance stage of trtllm-modeling-v2.

Runs after delivery (chained) or on its own against an existing modeling_v2
target. Three layers, cheapest first:

- ``bench.py``      the measurement: one label = one serving session (runs in the container)
- ``ladder.py``     the config ladder: serving-config rungs kept by rule (Python, no agent)
- ``campaign.py``   modeling iterations: tuner turns (fusion, kernel sources, ordering),
                    micro-bench first, e2e A/B second, accuracy gate on every keep

``records.py`` holds the measurement/decision ledger and the keep rule.
"""
