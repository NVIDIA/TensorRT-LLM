from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel

DATABASE_LIST_PATH = Path(__file__).parent / "database" / "scenario_list.yaml"


class RecipeRecord(BaseModel):
    model: str
    gpu: str
    isl: int
    osl: int
    concurrency: int
    config_path: str
    tps_per_user: Optional[int] = None
    tps_per_gpu: Optional[int] = None
    num_gpus: int

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> List["RecipeRecord"]:
        with open(yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return [cls(**item) for item in data]

    def to_pandas_row(self) -> pd.Series:
        return pd.Series(self.model_dump())


def get_recipe_dataframe(yaml_path: Path) -> pd.DataFrame:
    records = RecipeRecord.from_yaml(yaml_path)
    return pd.DataFrame([record.to_pandas_row() for record in records])


def load_recipe_database() -> pd.DataFrame:
    return get_recipe_dataframe(DATABASE_LIST_PATH)
