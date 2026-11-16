# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""drafter_silu_fold: the TP8 drafter MLP's silu(gate) * up in its gate_up GEMM's epilogue on multi-request blocks."""
