#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-side storage handler for generated media assets."""


class MediaStorage:
    """Handler for storing images and videos on disk under a server-managed path.

    Tensor encoding (image/video → file/bytes) is intentionally NOT a
    responsibility of this class; use :mod:`tensorrt_llm.media.encoding` free
    functions or :meth:`tensorrt_llm.visual_gen.VisualGenOutput.save` instead.
    """
