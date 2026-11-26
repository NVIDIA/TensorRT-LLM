# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..._native import load_native_op

swiglu_block_fp8_kernel = load_native_op("catalog/activation", "swiglu_block_fp8")
