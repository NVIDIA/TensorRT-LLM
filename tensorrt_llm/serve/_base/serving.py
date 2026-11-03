# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes of the serving layer: one serving class per endpoint."""

from typing import Any


class BaseServing:
    """Base of every serving class.

    Holds only what all serving classes share; each subclass receives its
    dependencies as explicit constructor arguments, never by reading the
    server object.
    """


class GenerateBaseServing(BaseServing):
    """Serving classes that submit requests to the generate pipeline."""

    def __init__(self, *, model: str, generator: Any, tokenizer: Any,
                 model_config: Any) -> None:
        # BaseServing has no constructor: set every attribute here, no
        # super().__init__(...).
        self.model = model
        self.generator = generator
        self.tokenizer = tokenizer
        self.model_config = model_config
