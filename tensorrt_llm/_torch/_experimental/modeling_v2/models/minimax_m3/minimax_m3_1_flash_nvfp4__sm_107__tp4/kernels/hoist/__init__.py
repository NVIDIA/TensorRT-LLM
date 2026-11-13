# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4WC_HOIST: the reference TP4 C1 MoE-boundary block, as-is (port-first).

The block is the reference router.lr4-prr4-q + moe.sht4-n48-q stages, measured in-chain as HYB (pass B,
-8.19 us / layer vs OURS25):
- the o_proj GEMV with 16 router CTAs (fused_o_proj/kernel_tp4.py, CuTe DSL);
- the prr4-q post (route_warp_tp4_fc1_per_token, CUDA C++: this target's minimax_m3_sm107tp4_hoist_post);
- the fused engine fed external routes (fused/kernel.py = ../moe_engine_kernel.py, CuTe DSL).

``tp4c1/`` holds the block's Python package (the reference modules, moved under this package's name); ``seams``
replaces only the reference's process plumbing (torch JIT extensions, cudaIpc + gloo exchanges, torch.distributed)
with this target's (the thop op, MNNVL, the TP MPI comm);
``tp4_hoist_block`` builds one layer's callables on the target's weights."""
