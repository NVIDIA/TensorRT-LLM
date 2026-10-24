# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 NVFP4 / sm_103 / tp4."""

# Routing has selected this target before its native fake registrations are needed.
from .. import fake_ops

fake_ops.register()
