"""Accuracy test result types.

This module defines strongly-typed data structures for accuracy test results,
balancing type safety with flexibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from typing import NotRequired, TypedDict  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired, TypedDict  # Python < 3.11


@dataclass
class DatasetValidation:
    """Validation result for a single dataset (stable structure, suitable for dataclass).

    Attributes:
        dataset: Dataset name (e.g., 'gsm8k', 'mmlu')
        filter: Filter type (e.g., 'flexible-extract', 'strict-match')
        passed: Whether the validation passed
        actual: Actual accuracy value from log
        expected: Expected accuracy value from config
        threshold: Threshold value for validation
        threshold_type: Type of threshold ('relative' or 'absolute')
        message: Validation message
        error: Optional error message if validation failed
    """

    dataset: str
    filter: str
    passed: bool
    actual: float
    expected: float
    threshold: float
    threshold_type: str
    message: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation of the validation result
        """
        result = {
            "dataset": self.dataset,
            "filter": self.filter,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "threshold": self.threshold,
            "threshold_type": self.threshold_type,
            "message": self.message,
        }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class RunValidation:
    """Validation result for a single run (stable structure).

    Attributes:
        run_id: Run identifier (e.g., 'run-1', 'run-2')
        run_name: Human-readable run name (e.g., 'Run 1', 'Run 2')
        all_passed: Whether all datasets passed for this run
        results: List of dataset validation results
    """

    run_id: str
    run_name: str
    all_passed: bool
    results: List[DatasetValidation]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation of the run validation
        """
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
        }


class AccuracyValidationResult(TypedDict):
    """Accuracy validation result (using TypedDict for flexibility with type hints).

    TypedDict provides type hints while maintaining dict flexibility.
    This is ideal for results that may need dynamic fields in the future.
    """

    success: bool
    all_passed: bool
    runs: List[RunValidation]
    raw_results: List[Dict[str, Dict[str, float]]]
    error: NotRequired[str]  # Optional: only present when success=False
