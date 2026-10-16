# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit non-serving DSpark graph-cell profiling contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel

__all__ = ["DSparkCostProfileConfig", "DSparkProfileCell"]


class DSparkProfileCell(StrictBaseModel):
    """One rank-local graph shape; zero budget denotes the native full block."""

    model_config = ConfigDict(frozen=True)
    graph_batch_size: StrictInt = Field(gt=0, description="Rank-local request count G.")
    verifier_budget: StrictInt = Field(
        ge=0, description="Verifier token count V; zero means native."
    )


class DSparkCostProfileConfig(StrictBaseModel):
    """Construction-only graph grid, never a serving admission configuration."""

    model_config = ConfigDict(frozen=True)
    physical_k: StrictInt = Field(default=3, gt=0, description="Full physical draft block K.")
    cells: tuple[DSparkProfileCell, ...] = Field(min_length=1, description="Declared graph cells.")
    record_confidence_head: StrictBool = Field(
        default=True, description="Execute the head in graphs."
    )
    measurement_scope: Literal["graph-replay", "whole-step"] = Field(
        default="graph-replay", description="Original replay or executor-loop cost boundary."
    )
    max_samples: StrictInt = Field(
        default=4096, gt=0, le=65536, description="Maximum raw replay records for the run."
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> DSparkCostProfileConfig:
        seen = set()
        native = set()
        for cell in self.cells:
            pair = (cell.graph_batch_size, cell.verifier_budget)
            if pair in seen:
                raise ValueError("Profile cells must not contain duplicates")
            seen.add(pair)
            if cell.verifier_budget == 0:
                native.add(cell.graph_batch_size)
            elif (
                self.physical_k == 1
                or not cell.graph_batch_size
                <= cell.verifier_budget
                < cell.graph_batch_size * (self.physical_k + 1)
            ):
                raise ValueError("Compact cells require K>1 and G <= V < G*(K+1); native uses V=0")
        if native != {cell.graph_batch_size for cell in self.cells}:
            raise ValueError("Every profiled G requires a V=0 native control")
        return self

    def validate_graph_ladder(self, graph_batch_sizes: Sequence[int]) -> None:
        """Require the exact runtime ladder, not an undocumented subset."""
        if any(type(value) is not int or value <= 0 for value in graph_batch_sizes):
            raise ValueError("Runtime graph row counts must be positive integers")
        if len(set(graph_batch_sizes)) != len(graph_batch_sizes):
            raise ValueError("Runtime graph ladder contains duplicates")
        if set(graph_batch_sizes) != {cell.graph_batch_size for cell in self.cells}:
            raise ValueError("Profile grid must cover the exact captured graph ladder")

    def graph_shapes(self, graph_batch_sizes: Sequence[int]) -> list[tuple[int, ...]]:
        """Return native and positive-V capture shapes in deterministic order."""
        self.validate_graph_ladder(graph_batch_sizes)
        return [
            (cell.graph_batch_size, self.physical_k)
            if cell.verifier_budget == 0
            else (cell.graph_batch_size, self.physical_k, cell.verifier_budget)
            for cell in sorted(
                self.cells, key=lambda item: (item.graph_batch_size, item.verifier_budget)
            )
        ]

    def require_cell(self, cell: DSparkProfileCell) -> None:
        """Reject an undeclared shape before request metadata can change."""
        if cell not in self.cells:
            raise ValueError("Requested profile cell is not in the construction grid")
