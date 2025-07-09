from typing import Dict, Optional, Type

from tensorrt_llm.bench.dataclasses.scenario import ScenarioSpecification
from tensorrt_llm.bench.tuning import DefaultLlmHeuristic, HueristicProtocol
from tensorrt_llm.bench.tuning.throughput import (PytMaxThroughputScenario,
                                                  TrtMaxThroughputScenario)


class HeuristicFactory:
    """Factory class for creating heuristic protocol instances."""

    _heuristics: Dict[str, Dict[str, Type[HueristicProtocol]]] = {
        "pytorch": {
            "default": DefaultLlmHeuristic,
            "throughput": PytMaxThroughputScenario,
        },
        "tensorrt": {
            "default": DefaultLlmHeuristic,
            "throughput": TrtMaxThroughputScenario,
        },
        "_autodeploy": {
            "default": DefaultLlmHeuristic,
            "throughput": PytMaxThroughputScenario,
        },
    }

    @classmethod
    def get_heuristic(cls, backend: str,
                      heuristic_name: str) -> Type[HueristicProtocol]:
        """
        Get a heuristic protocol class by backend and name.

        Args:
            backend: Backend name (e.g., "pytorch", "tensorrt")
            heuristic_name: Name of the heuristic to retrieve

        Returns:
            The heuristic protocol class

        Raises:
            ValueError: If the backend or heuristic name is not found
        """
        if backend not in cls._heuristics:
            available_backends = ", ".join(cls._heuristics.keys())
            raise ValueError(f"Unknown backend '{backend}'. "
                             f"Available backends: {available_backends}")

        backend_heuristics = cls._heuristics[backend]
        if heuristic_name not in backend_heuristics:
            available_heuristics = ", ".join(backend_heuristics.keys())
            raise ValueError(
                f"Unknown heuristic '{heuristic_name}' for backend '{backend}'. "
                f"Available heuristics: {available_heuristics}")

        return backend_heuristics[heuristic_name]

    @classmethod
    def get_settings(
        cls,
        backend: str,
        heuristic_name: str,
        scenario: ScenarioSpecification,
    ) -> Dict:
        """
        Get settings for a specific heuristic.

        Args:
            backend: Backend name (e.g., "pytorch", "tensorrt")
            heuristic_name: Name of the heuristic to use
            scenario: Scenario specification

        Returns:
            Dictionary containing the settings for the heuristic
        """
        heuristic_class = cls.get_heuristic(backend, heuristic_name)
        return heuristic_class.get_settings(scenario)

    @classmethod
    def list_available_heuristics(cls,
                                  backend: Optional[str] = None) -> list[str]:
        """
        Get a list of all available heuristic names.

        Args:
            backend: Optional backend name to filter heuristics. If None, returns all heuristics.

        Returns:
            List of available heuristic names
        """
        if backend is None:
            all_heuristics = []
            for backend_name, heuristics in cls._heuristics.items():
                for heuristic_name in heuristics.keys():
                    all_heuristics.append(f"{backend_name}.{heuristic_name}")
            return all_heuristics
        else:
            if backend not in cls._heuristics:
                available_backends = ", ".join(cls._heuristics.keys())
                raise ValueError(f"Unknown backend '{backend}'. "
                                 f"Available backends: {available_backends}")
            return list(cls._heuristics[backend].keys())

    @classmethod
    def register_heuristic(cls, backend: str, name: str,
                           heuristic_class: Type[HueristicProtocol]) -> None:
        """
        Register a new heuristic class.

        Args:
            backend: Backend name to register the heuristic under
            name: Name to register the heuristic under
            heuristic_class: The heuristic protocol class to register
        """
        if backend not in cls._heuristics:
            cls._heuristics[backend] = {}
        cls._heuristics[backend][name] = heuristic_class
