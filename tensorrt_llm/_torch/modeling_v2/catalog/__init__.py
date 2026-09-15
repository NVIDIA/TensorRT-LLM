# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kernel vocabulary: one entry = contract .md + wrapper .py + GPU test.

Nothing is imported here -- an entry is pulled in by the target that uses
it, so a process loads only the ops its target calls."""
