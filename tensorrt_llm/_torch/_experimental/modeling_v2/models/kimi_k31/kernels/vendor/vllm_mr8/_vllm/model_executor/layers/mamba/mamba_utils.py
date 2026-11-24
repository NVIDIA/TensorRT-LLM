# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vllm mamba_utils stand-in. The replica allocates its KDA conv state dim-first ([slots, 3 * dim, window])."""


def is_conv_state_dim_first():
    return True
