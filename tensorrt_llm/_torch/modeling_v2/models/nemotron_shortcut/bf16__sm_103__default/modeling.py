# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 text-generation entry point."""

from ..decoder import ShortcutDecoder, ShortcutForCausalLM
from .weights import load


class ShortcutBf16ForCausalLM(ShortcutForCausalLM):
    decoder_cls = ShortcutDecoder
    weight_loader = staticmethod(load)
