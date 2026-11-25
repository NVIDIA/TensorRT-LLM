# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package marker.

`test_routing.py` repeats the glm_moe_dsa family's basename. Without a package, pytest's
default import mode loads both under one module name and refuses the second.
"""
