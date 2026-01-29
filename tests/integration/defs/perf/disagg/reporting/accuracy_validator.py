"""Accuracy validation classes and dataset threshold configurations."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import scipy.stats


def compute_theta(num_samples: int, sigma: float, alpha: float = 0.05, beta: float = 0.2) -> float:
    """Compute theta (minimum detectable effect) for hypothesis testing.

    Args:
        num_samples: Number of samples
        sigma: Standard deviation
        alpha: Type I error (false positive rate)
        beta: Type II error (false negative rate)

    Returns:
        Theta value
    """
    scale = (2 * sigma**2 / num_samples) ** 0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    theta = -(z_alpha + z_beta) * scale
    return theta


def compute_threshold(
    num_samples: int,
    ref_accuracy: float,
    sigma: float,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> float:
    """Compute threshold for hypothesis testing.

    Args:
        num_samples: Number of samples
        ref_accuracy: Reference accuracy value
        sigma: Standard deviation
        alpha: Type I error (false positive rate)
        higher_is_better: Whether higher accuracy is better

    Returns:
        Threshold value
    """
    scale = (2 * sigma**2 / num_samples) ** 0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    if higher_is_better:
        return ref_accuracy + z_alpha * scale
    else:
        return ref_accuracy - z_alpha * scale


@dataclass
class HypothesisTestingParams:
    """Parameters for hypothesis testing validation."""

    ref_accuracy: float
    num_samples: int
    alpha: float = 0.05
    beta: float = 0.2
    sigma: float = 50.0
    higher_is_better: bool = True
    theta: float = field(init=False)
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute theta and threshold after initialization."""
        # Ensure unit consistency: if ref_accuracy is in decimal form (0-1)
        # and sigma is in percentage form (e.g., 50), convert ref_accuracy to percentage
        # This matches the convention in accuracy_core.py where ref_accuracy is stored as percentage
        if self.ref_accuracy <= 1.0 and self.sigma > 1.0:
            self.ref_accuracy = self.ref_accuracy * 100

        self.theta = compute_theta(
            self.num_samples, sigma=self.sigma, alpha=self.alpha, beta=self.beta
        )
        self.threshold = compute_threshold(
            self.num_samples,
            self.ref_accuracy,
            sigma=self.sigma,
            alpha=self.alpha,
            higher_is_better=self.higher_is_better,
        )


# Dataset default parameters for hypothesis testing
# Extracted from accuracy_core.py AccuracyTask subclasses
DATASET_DEFAULTS = {
    "aime25": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 30,  # AIME 2025 full sample size
        "higher_is_better": True,
    },
    "gsm8k": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 1319,
        "higher_is_better": True,
    },
    "mmlu": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 4096,
        "higher_is_better": True,
    },
    "humaneval": {
        "alpha": 0.002,
        "beta": 0.2,
        "sigma": 15.08,
        "num_samples": 164,
        "higher_is_better": True,
    },
    "cnn_dailymail": {
        "alpha": 0.002,
        "beta": 0.2,
        "sigma": 11.06,
        "num_samples": 512,
        "higher_is_better": True,
    },
    "gpqa_diamond": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 198,
        "higher_is_better": True,
    },
    # Alias for gpqa_diamond (same task, different naming convention)
    "gpqa_diamond_local": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 198,
        "higher_is_better": True,
    },
    "json_mode_eval": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 100,
        "higher_is_better": True,
    },
    "mmmu": {"alpha": 0.05, "beta": 0.2, "sigma": 50, "num_samples": 900, "higher_is_better": True},
    "passkey_retrieval_64k": {
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0,
        "num_samples": 20,
        "higher_is_better": True,
    },
    "passkey_retrieval_128k": {
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0,
        "num_samples": 20,
        "higher_is_better": True,
    },
    "zero_scrolls": {
        "alpha": 0.002,
        "beta": 0.2,
        "sigma": 6.97,
        "num_samples": 80,
        "higher_is_better": True,
    },
    "slimpajama-6b": {
        "alpha": 0.01,
        "beta": 0.2,
        "sigma": 4.48,
        "num_samples": 86,
        "higher_is_better": False,
    },
}


class HypothesisTestValidator(object):
    """Validator using hypothesis testing."""

    def __init__(
        self,
        expected_value: float,
        alpha: float,
        beta: float,
        sigma: float,
        num_samples: int,
        higher_is_better: bool = True,
    ):
        """Initialize HypothesisTestValidator.

        Args:
            expected_value: Expected (reference) accuracy value
            alpha: Type I error (false positive rate)
            beta: Type II error (false negative rate)
            sigma: Standard deviation
            num_samples: Number of samples
            higher_is_better: Whether higher accuracy is better
        """
        self.params = HypothesisTestingParams(
            ref_accuracy=expected_value,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            num_samples=num_samples,
            higher_is_better=higher_is_better,
        )

    def validate(self, actual_value: float) -> tuple[bool, str]:
        """Validate using hypothesis testing.

        Args:
            actual_value: Actual accuracy value from test
            expected_value: Expected accuracy value (for display consistency)

        Returns:
            Tuple of (passed, message)
        """
        # Normalize actual_value to same unit as ref_accuracy/threshold
        # If ref_accuracy is in percentage (>1.0) and actual_value is in decimal (<=1.0), convert actual_value
        compare_actual = actual_value * 100

        # For display: always show as percentage
        display_actual = compare_actual
        display_expected = self.params.ref_accuracy
        display_threshold = self.params.threshold

        compare_op = ">=" if self.params.higher_is_better else "<="

        # Check if passes threshold (use normalized values for comparison)
        if self.params.higher_is_better:
            passed = compare_actual >= self.params.threshold
        else:
            passed = compare_actual <= self.params.threshold

        # Build message with percentage format for readability
        msg = (
            f"Hypothesis test: actual={display_actual:.2f}%, "
            f"expected={display_expected:.2f}%, "
            f"threshold={display_threshold:.2f}%, "
            f"required: {compare_op} threshold"
        )

        return passed, msg


@dataclass
class DatasetThreshold:
    """Accuracy threshold configuration for a single dataset."""

    dataset_name: str  # Dataset name: gsm8k, mmlu, humaneval, etc.
    expected_value: float  # Expected accuracy value
    threshold_type: str = "hypothesis_test"  # Must be "hypothesis_test"
    filter_type: str = "flexible-extract"  # lm_eval filter type

    # Optional hypothesis testing parameters (override defaults from DATASET_DEFAULTS)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    sigma: Optional[float] = None
    num_samples: Optional[int] = None
    higher_is_better: Optional[bool] = None

    def _get_hypothesis_params(self) -> Dict[str, any]:
        """Get hypothesis testing parameters, using defaults if not specified.

        Returns:
            Dict with alpha, beta, sigma, num_samples, higher_is_better
        """
        # Get defaults for this dataset
        dataset_key = self.dataset_name.lower()
        defaults = DATASET_DEFAULTS.get(dataset_key, {})

        # Override with specified values
        return {
            "alpha": self.alpha if self.alpha is not None else defaults.get("alpha", 0.05),
            "beta": self.beta if self.beta is not None else defaults.get("beta", 0.2),
            "sigma": self.sigma if self.sigma is not None else defaults.get("sigma", 50),
            "num_samples": self.num_samples
            if self.num_samples is not None
            else defaults.get("num_samples", 1000),
            "higher_is_better": self.higher_is_better
            if self.higher_is_better is not None
            else defaults.get("higher_is_better", True),
        }

    def get_computed_threshold(self) -> float:
        """Get the computed threshold value for hypothesis testing.

        Returns:
            Computed threshold value
        """
        params = self._get_hypothesis_params()
        test_params = HypothesisTestingParams(
            ref_accuracy=self.expected_value,
            alpha=params["alpha"],
            beta=params["beta"],
            sigma=params["sigma"],
            num_samples=params["num_samples"],
            higher_is_better=params["higher_is_better"],
        )
        return test_params.threshold

    def validate(self, actual_value: float) -> tuple[bool, str]:
        """Validate if accuracy passes the threshold using hypothesis testing.

        Args:
            actual_value: Actual accuracy value from test

        Returns:
            Tuple of (passed, message): Whether validation passed and detail message
        """
        if self.threshold_type != "hypothesis_test":
            raise ValueError(
                f"Unsupported threshold_type: {self.threshold_type}. "
                f"Only 'hypothesis_test' is supported."
            )

        params = self._get_hypothesis_params()
        validator = HypothesisTestValidator(
            expected_value=self.expected_value,
            alpha=params["alpha"],
            beta=params["beta"],
            sigma=params["sigma"],
            num_samples=params["num_samples"],
            higher_is_better=params["higher_is_better"],
        )

        return validator.validate(actual_value)
