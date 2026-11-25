<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VisualGen Modeling V2

## Where to add a feature

A *target* is a pipeline for a particular checkpoint or deployment. Choose where to add a feature based on the model family's migration plan.

| Model family | Where to develop |
| --- | --- |
| New family | **Prefer V2** for the initial implementation. |
| V1 only | The model owner decides whether to migrate. Development can continue in V1. |
| V2 target available, V1 retiring soon | **Add new features only in V2.** MiniMax-H3 follows this policy. |
| V1 and V2 will both remain in active use | When the model owner considers a new feature important for the family, **support and test it on every supported V1 and V2 target**. A deployment-specific optimization can stay in its target. |

For a worked example of target layout, shared ops, routing and tests, see the [MiniMax-H3 developer guide](models/minimax_h3/README.md).
