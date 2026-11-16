# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dense_gemms: the host code of

* ``dense_gemms``: dense projections on one named kernel per shape.
* ``dense_gemm_prefetch``: L2 prefetch of the next dense projection's weight.
"""
