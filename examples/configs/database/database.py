# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml
from pydantic import BaseModel, Field, RootModel

DATABASE_LIST_PATH = Path(__file__).parent / "lookup.yaml"


class RecipeConstraints(BaseModel):
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
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        return data if data is not None else {}


class Recipe(BaseModel):
    """Recipe that describes a single scenario."""

    constraints: RecipeConstraints = Field(description="Recipe constraints")
    env_overrides: Dict[str, Any] = Field(description="Environment overrides", default_factory=dict)
    config: Dict[str, Any] = Field(description="Configuration overrides", default_factory=dict)


class RecipeList(RootModel[List[RecipeConstraints]]):
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RecipeList":
        """Load and validate recipe list from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    def __iter__(self) -> Iterator[RecipeConstraints]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
