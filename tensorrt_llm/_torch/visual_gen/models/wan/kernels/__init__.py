# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae's CuTe DSL kernels: the norm walk (:mod:`.norm`), the frame operations
(:mod:`.frame_ops`), the upsample phase gather (:mod:`.upsample_gather`), the portable spatial
Winograd transforms (:mod:`.spatial_winograd`) and the Blackwell spatial Winograd pipeline
(:mod:`.blackwell`). Each compiles on first use. All but the Blackwell kernels also go through
flashinfer's on-disk kernel cache when flashinfer is installed, so a later process loads them
instead of compiling."""
