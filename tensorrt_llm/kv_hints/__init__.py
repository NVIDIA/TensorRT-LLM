# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, opaque request metadata for KV-cache consumers."""

from .protocol import KvHintAction, KvHintsEnvelope, KvHintsInput

__all__ = ["KvHintAction", "KvHintsEnvelope", "KvHintsInput"]
