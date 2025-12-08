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

import pandas as pd
import yaml
from pydantic import BaseModel, Field, RootModel

DATABASE_LIST_PATH = Path(__file__).parent / "database" / "scenario_list.yaml"


class RecipeConstraints(BaseModel):
    """Recipe record for scenario list."""

    model: str = Field(description="Model name")
    gpu: str = Field(description="GPU name")
    isl: int = Field(description="Input sequence length")
    osl: int = Field(description="Output sequence length")
    concurrency: int = Field(description="Concurrency")
    config_path: str = Field(description="Configuration path")
    num_gpus: int = Field(description="Number of GPUs")

    def to_pandas_row(self) -> pd.Series:
        """Convert recipe record to pandas Series.

        Returns:
            pd.Series: Pandas Series containing the recipe record.
        """
        return pd.Series(self.model_dump())

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from the configuration path."""
        with open(self.config_path, "r") as f:
            return Recipe(**yaml.safe_load(f, Loader=yaml.FullLoader))


class Recipe(BaseModel):
    """Recipe that describes a single scenario."""

    constraints: RecipeConstraints = Field(description="Recipe constraints")
    env_overrides: Dict[str, Any] = Field(description="Environment overrides", default_factory=dict)
    config: Dict[str, Any] = Field(description="Configuration overrides", default_factory=dict)


class RecipeList(RootModel[List[RecipeConstraints]]):
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RecipeList":
        """Load recipe list from YAML file and validate the data.

        Args:
            yaml_path (Path): Path to the YAML file containing the recipe list.

        Returns:
            RecipeList: Recipe list object.
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f, Loader=yaml.FullLoader)
        return cls(data)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Convert recipe list to pandas DataFrame.

        Returns:
            pd.DataFrame: Pandas DataFrame where each row contains a recipe record.
        """
        return pd.DataFrame([record.to_pandas_row() for record in self.root])

    def __iter__(self) -> Iterator[RecipeConstraints]:
        """Iterate over the recipe list.

        Returns:
            Iterator[RecipeRecord]: Iterator over the recipe list.
        """
        for record in self.root:
            yield record

    def __len__(self) -> int:
        """Get the number of recipes in the list.

        Returns:
            int: Number of recipes in the list.
        """
        return len(self.root)
