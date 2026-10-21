# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tune: the performance stage of trtllm-modeling-v2.

Runs as the ``tune`` task of a planned run, or on its own against an existing
modeling_v2 target:

- ``serving_config.py`` the serving config, derived from the workload
- ``bench.py``      the measurement: one label = one serving session (runs in the container)
- ``measure.py``    queuing measurements on spool sessions and collecting them
- ``campaign.py``   modeling iterations: tuner turns (fusion, kernel sources, ordering),
                    micro-bench first, e2e A/B second, the keep gate on every keep
- ``keep_gate.py``  the keep gate: the contract's accuracy gate at the unit's members,
                    and the target's suite, on the tree a keep would make the best
- ``split.py``      a LOSS whose losses share one parallel mode: that mode, and the verdict
                    on the member-gated variant a tuner builds for it
- ``inert.py``      the members a candidate provably does not change: judged nowhere

``records.py`` holds the measurement/decision ledger, the noise floor and the keep rule.
"""
