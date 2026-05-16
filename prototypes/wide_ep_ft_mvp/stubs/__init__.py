# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimum stubs that exercise the MVP integration seams.

Each stub corresponds to one row of the table in
``docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md`` §2.

These are deliberately the absolute minimum needed to make the integration
seam runnable end-to-end on a single 4-/8-GPU node. They have no telemetry,
no feature-flag gating, no error handling for degenerate cases, and intentionally
hard-code values that the production PRs will plumb through config. None of
this code is intended to be reviewable as MVP material; it is throwaway
scaffolding to validate that the seams work before six PRs land against the
wrong contract.
"""
