# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for the AutoDeploy Grafia compile backend."""

BACKEND_NAME = "grafia"
GRAFIA_MODES = ("decode", "prefill", "mixed")
RMSNORM_OP_KIND = "grafia.fast_low_latency_rms_norm"
SUPPORTED_RMSNORM_HIDDEN = 2880
SUPPORTED_RMSNORM_ROWS = 107
