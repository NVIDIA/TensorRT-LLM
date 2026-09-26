# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin mirrors of torch callables.

Upstream owns their correctness, so these carry no contract, no test and
no receipt by design. They exist so a target's forward can satisfy the
closed-vocabulary rule without leaving the catalog."""
