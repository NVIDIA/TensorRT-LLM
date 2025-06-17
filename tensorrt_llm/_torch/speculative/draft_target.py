from dataclasses import dataclass

from .interface import SpecMetadata


@dataclass
class DraftTargetSpecMetadata(SpecMetadata):

    def __post_init__(self):
        pass

    def prepare(self):
        pass
