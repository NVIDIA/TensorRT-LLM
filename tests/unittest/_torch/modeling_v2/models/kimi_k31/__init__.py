# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package marker.

The family's test directories repeat basenames: ``ops/`` and ``kernels/`` with their category directories at the family
level and in the targets, and test files named like other families' (``test_modeling_v2_vision.py``,
``test_modeling_v2_routed_experts.py``). As packages they import under ``kimi_k31.<path>``; without them, pytest's
default import mode loads two of them under one module name and refuses the second.
"""
