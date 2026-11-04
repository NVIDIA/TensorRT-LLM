# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package marker.

These test basenames can repeat a target's own (`test_modeling_v2_gelu_erf.py`).
Without a package, pytest's default import mode loads both under one module
name and refuses the second.
"""
