# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media encoding and decoding utilities for TensorRT-LLM.

Free functions for encoding tensors to image / video files or in-memory bytes,
and for decoding encoded video back to frames. Internal-by-convention: not
re-exported from ``tensorrt_llm`` so the public API surface is reached via
:class:`tensorrt_llm.visual_gen.VisualGenOutput`.
"""
