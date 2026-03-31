# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml
from pydantic import BaseModel, Field, RootModel, field_validator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATABASE_LIST_PATH = Path(__file__).parent / "lookup.yaml"
CURATED_LIST_PATH = Path(__file__).parent.parent / "curated" / "lookup.yaml"

LOW_LATENCY_CONCURRENCY_THRESHOLD = 8
HIGH_THROUGHPUT_CONCURRENCY_THRESHOLD = 32
KEY_PROFILES = {"Min Latency", "Balanced", "Max Throughput"}


class CuratedRecipe(BaseModel):
    """A curated (hand-tuned) recipe entry."""

    model: str = Field(description="HuggingFace model ID")
    arch: str = Field(description="Model architecture class name")
    config_path: str = Field(description="Relative path to YAML config")
    scenario: str = Field(default="", description="Deployment scenario label")
    gpu_compatibility: str = Field(default="Any", description="Compatible GPU families")
    disagg: bool = Field(default=False, description="Requires disaggregated serving")

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, v: str) -> str:
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Invalid config path: {v}")
        return v


class CuratedRecipeList(RootModel[List[CuratedRecipe]]):
    """Validated list of curated recipe entries."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CuratedRecipeList":
        """Load and validate curated recipe list from YAML file."""
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls(data)
        except Exception as e:
            logger.error("Failed to load curated recipe list from %s: %s", yaml_path, e)
            raise

    def __iter__(self) -> Iterator[CuratedRecipe]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


class Recipe(BaseModel):
    """Recipe record for scenario list."""

    model: str = Field(description="Model name")
    gpu: str = Field(description="GPU name")
    isl: int = Field(description="Input sequence length")
    osl: int = Field(description="Output sequence length")
    concurrency: int = Field(description="Concurrency")
    config_path: str = Field(description="Configuration path")
    num_gpus: int = Field(description="Number of GPUs")

    def load_config(self) -> Dict[str, Any]:
        """Load and return the YAML config at config_path."""
        config_relative_path = Path(self.config_path)
        # Ensure config path is within the repo root
        if config_relative_path.is_absolute() or ".." in config_relative_path.parts:
            raise ValueError(f"Invalid config path: {self.config_path}")
        full_path = REPO_ROOT / self.config_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config not found: {full_path}")
        with open(full_path, encoding="utf-8") as f:
            return yaml.safe_load(f)


class RecipeList(RootModel[List[Recipe]]):
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RecipeList":
        """Load and validate recipe list from YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def __iter__(self) -> Iterator[Recipe]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


def assign_profile(num_recipes: int, idx: int, concurrency: int) -> str:
    """Assign performance profile to a recipe based on its position in a concurrency-sorted list."""
    if num_recipes == 1:
        if concurrency <= LOW_LATENCY_CONCURRENCY_THRESHOLD:
            return "Low Latency"
        elif concurrency >= HIGH_THROUGHPUT_CONCURRENCY_THRESHOLD:
            return "High Throughput"
        else:
            return "Balanced"
    elif idx == 0:
        return "Min Latency"
    elif idx == num_recipes - 1:
        return "Max Throughput"
    elif idx in ((num_recipes - 1) // 2, num_recipes // 2):
        return "Balanced"
    elif idx < num_recipes // 2:
        return "Low Latency"
    else:
        return "High Throughput"


def select_key_recipes(recipes: List[Recipe]) -> List[Tuple[Recipe, str]]:
    """Select key recipes (min latency, balanced, max throughput) from a list of recipes."""
    if not recipes:
        return []

    sorted_recipes = sorted(recipes, key=lambda r: r.concurrency)
    n = len(sorted_recipes)

    result = []
    seen_profiles = set()
    for idx, recipe in enumerate(sorted_recipes):
        profile = assign_profile(n, idx, recipe.concurrency)
        # For n==1, keep whatever profile is assigned
        # For n>=2, only keep key profiles and dedupe (for even n, two indices get "Balanced")
        if n == 1 or (profile in KEY_PROFILES and profile not in seen_profiles):
            result.append((recipe, profile))
            seen_profiles.add(profile)
    return result
