# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

from PIL import Image

from tensorrt_llm.inputs.utils import load_image


def load_rgb_image(source) -> Image.Image:
    """Decode a VisualGen image input using the shared URI and media access policy."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(source))) as image:
            return image.convert("RGB")
    return load_image(source, format="pil")
