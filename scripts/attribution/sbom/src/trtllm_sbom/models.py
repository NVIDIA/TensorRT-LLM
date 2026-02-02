from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, HttpUrl, RootModel


class AddDependencyInput(BaseModel):
    dependency: str
    version: str
    license: Path
    copyright: Optional[Path] = None
    attribution: Optional[Path] = None
    source: Optional[HttpUrl] = None


class RegisterFilesInput(BaseModel):
    dependency: str
    version: str
    files: List[str]


class DependencyMetadataEntry(BaseModel):
    license: Optional[str] = None
    copyright: Optional[str] = None
    attribution: Optional[str] = None
    source: Optional[HttpUrl] = None


class DependencyMetadata(RootModel[dict[str, DependencyMetadataEntry]]):
    pass


class FilesToDependency(RootModel[dict[str, List[str]]]):
    pass
