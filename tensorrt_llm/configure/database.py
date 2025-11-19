from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, RootModel

DATABASE_LIST_PATH = Path(__file__).parent / "database" / "scenario_list.yaml"


class RecipeRecord(BaseModel):
    model: str = Field(description="Model name")
    gpu: str = Field(description="GPU name")
    isl: int = Field(description="Input sequence length")
    osl: int = Field(description="Output sequence length")
    concurrency: int = Field(description="Concurrency")
    config_path: str = Field(description="Configuration path")
    tps_per_user: Optional[int] = Field(description="TPS per user", default=None)
    tps_per_gpu: Optional[int] = Field(description="TPS per GPU", default=None)
    num_gpus: int = Field(description="Number of GPUs")

    def to_pandas_row(self) -> pd.Series:
        """Convert recipe record to pandas Series.

        Returns:
            pd.Series: Pandas Series containing the recipe record.
        """
        return pd.Series(self.model_dump())


class RecipeList(RootModel[List[RecipeRecord]]):
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RecipeList":
        """Load recipe list from YAML file and validate the data.

        Args:
            yaml_path (Path): Path to the YAML file containing the recipe list.

        Returns:
            RecipeList: Recipe list object.
        """
        with open(yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(data)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Convert recipe list to pandas DataFrame.

        Returns:
            pd.DataFrame: Pandas DataFrame where each row contains a recipe record.
        """
        return pd.DataFrame([record.to_pandas_row() for record in self.root])

    def __iter__(self) -> Iterator[RecipeRecord]:
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
