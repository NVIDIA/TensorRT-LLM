# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4ExpForConditionalGeneration targets served on several ranks.

The single-rank target of the same architecture lives in ``models/qwen4_exp``; the two
families route the architecture between them, each declining the other's deployments.
"""
