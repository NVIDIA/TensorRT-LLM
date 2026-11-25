# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_PROB1: the probabilistic DSpark Markov draft of an NVIDIA reference implementation (dk_mk_prob1 /
dk_mk_prob1_w8, CUDA C++ as-is:
mv2_minimax_m3::minimax_m3_sm107tp4_draft_prob1 / mv2_minimax_m3::minimax_m3_sm107tp8_draft_prob1) in the TP4 / TP8
DSpark worker.
``draft_prob1`` holds its MNNVL workspace and the per-step call; the target's ``dspark_markov_worker`` routes the
advanced-sampling (rejection) draft through it."""
