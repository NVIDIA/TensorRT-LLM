# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""split_sampler: the host code of

* ``split_sampler``: target head verify sampler split over two kernels.
* ``lm_head_inbox``: LM-head GEMM into a multicast inbox the sampler gathers.
"""
