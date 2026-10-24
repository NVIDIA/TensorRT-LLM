# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's tooling for a running run: ``trtllm-modeling-v2 driver --workspace <ws> <command>``.

The loop itself is :mod:`.core`; handing turns out under the external transport
is ``infra/external/driver.py``.
"""
