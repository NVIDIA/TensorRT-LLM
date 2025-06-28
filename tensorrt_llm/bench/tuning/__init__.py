from __future__ import annotations

from typing import Any, Dict, Protocol

from tensorrt_llm.bench.dataclasses.scenario import ScenarioSpecification


class HueristicProtocol(Protocol):

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        ...

