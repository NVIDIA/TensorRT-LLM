# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.fx import Node

from .context import LoweringContext

LoweringRule = Callable[[LoweringContext, Node], Any]

LOWERINGS: dict[Any, LoweringRule] = {}
