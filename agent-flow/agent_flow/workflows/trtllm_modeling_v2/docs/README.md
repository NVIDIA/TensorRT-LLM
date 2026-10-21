<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2 documentation

`trtllm-modeling-v2` takes one Hugging Face checkpoint to a gated ModelingV2
target, then makes the target fast without making it less correct. The
documents below explain the workflow from different starting points.

## Where to start

| you want to | read |
|---|---|
| understand what the workflow is and why it is built this way | [design.md](design.md) |
| onboard or tune a model with it | [user-guide.md](user-guide.md), then [examples.md](examples.md) |
| change the workflow itself | [design.md](design.md), then [developer-guide.md](developer-guide.md) |
| know what it has achieved on real models | [results.md](results.md) |
| fix a run that fails | [TROUBLESHOOTING.md](../TROUBLESHOOTING.md), keyed on the verbatim error |

## The documents

| document | what it covers |
|---|---|
| [design.md](design.md) | goals, principles, architecture, the run's lifecycle and phase 0, the contract and its checks (gate bundles, cheaper oracles), the task graph (check tasks, soft dependencies, promote, batched regression reviews), integration, the execution substrate, transports, tuning, the improver, failure handling, design decisions, limitations |
| [user-guide.md](user-guide.md) | prerequisites, the fast path and intake, writing `task.yaml`, compute sessions, starting, watching and steering a run, the milestones, tuning, standalone gates, FAQ |
| [developer-guide.md](developer-guide.md) | package layout, the workspace on disk, the control flow, recipes for extending the workflow, testing, conventions, contributing |
| [examples.md](examples.md) | an end-to-end worked run with every artifact it writes, feature and variant examples, tune examples, recipes |
| [results.md](results.md) | measured onboarding and tuning results, how they were measured, and what is not measured yet |

Reference material next to the package:

| document | what it covers |
|---|---|
| [README.md](../README.md) | the complete reference: every gate, flag, workspace file and substrate behavior |
| [INTAKE.md](../INTAKE.md) | the user-facing `task.yaml`, `site.yaml`, and `intake check\|confirm\|compile` |
| [BUILD_AND_TEST.md](../BUILD_AND_TEST.md) | building TensorRT-LLM and running tests at each scale |
| [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) | symptoms keyed on verbatim errors, with causes and fixes |
| [LESSONS.md](../LESSONS.md) | what real runs got wrong, and the mechanism that now prevents each failure |
| [task.example.yaml](../task.example.yaml) | every key of the spec, documented |
| [examples/](../examples/) | one model's worked values for operators writing a `task.yaml`; a run never materializes them |
