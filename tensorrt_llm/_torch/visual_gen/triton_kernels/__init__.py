# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU image-processing kernels for VisualGen control preprocessing.

Implementations of published algorithms -- Canny edge detection (Canny,
1986), the bilateral filter (Tomasi & Manduchi, 1998), and bilinear /
area-average / bicubic interpolation in their standard fixed-point forms --
written so control frames can be derived on the GPU alongside the pipeline
that consumes them.

``reference`` holds torch-op implementations of the same arithmetic, used as
the executable specification the kernels are asserted against; it is not
imported here because nothing in the inference path should reach for it.
"""

from .bilateral import bilateral_filter
from .canny import canny_edges
from .resize import resize_area_u8, resize_cubic_u8, resize_linear_u8

__all__ = [
    "bilateral_filter",
    "canny_edges",
    "resize_area_u8",
    "resize_cubic_u8",
    "resize_linear_u8",
]
