"""Validation logic for scenario constraints and configurations."""

from typing import Any, Dict, List

# Known GPU types (can be extended)
VALID_GPU_TYPES = {
    "H100_SXM",
    "H100",
    "H200",
    "B200",
    "A100",
    "A100_SXM",
    "L40S",
    "L4",
    "T4",
    "V100",
}


class ScenarioValidationError(Exception):
    """Raised when scenario validation fails."""


class ScenarioValidationWarning:
    """Represents a non-fatal validation warning."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f"Warning: {self.message}"


def validate_scenario(
    scenario: Dict[str, Any], strict: bool = True
) -> List[ScenarioValidationWarning]:
    """Validate scenario parameters.

    Args:
        scenario: Dictionary containing scenario parameters
        strict: If True, raise exceptions on errors; if False, collect warnings

    Returns:
        List of ScenarioValidationWarning objects for non-fatal issues

    Raises:
        ScenarioValidationError: If validation fails and strict=True
    """
    warnings: List[ScenarioValidationWarning] = []

    # Required fields check
    required_fields = ["model", "target_isl", "target_osl", "target_concurrency"]
    missing_fields = [field for field in required_fields if field not in scenario]

    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        if strict:
            raise ScenarioValidationError(error_msg)
        else:
            warnings.append(ScenarioValidationWarning(error_msg))
            return warnings

    # Validate model name
    model = scenario.get("model", "")
    if not model or not isinstance(model, str):
        error_msg = "Model must be a non-empty string"
        if strict:
            raise ScenarioValidationError(error_msg)
        warnings.append(ScenarioValidationWarning(error_msg))

    # Validate ISL (Input Sequence Length)
    isl = scenario.get("target_isl")
    if not isinstance(isl, int) or isl <= 0:
        error_msg = f"target_isl must be a positive integer, got: {isl}"
        if strict:
            raise ScenarioValidationError(error_msg)
        warnings.append(ScenarioValidationWarning(error_msg))
    elif isl > 128000:
        warnings.append(
            ScenarioValidationWarning(
                f"target_isl={isl} is very large (>128K), may cause memory issues"
            )
        )

    # Validate OSL (Output Sequence Length)
    osl = scenario.get("target_osl")
    if not isinstance(osl, int) or osl <= 0:
        error_msg = f"target_osl must be a positive integer, got: {osl}"
        if strict:
            raise ScenarioValidationError(error_msg)
        warnings.append(ScenarioValidationWarning(error_msg))
    elif osl > 16384:
        warnings.append(
            ScenarioValidationWarning(
                f"target_osl={osl} is very large (>16K), may impact performance"
            )
        )

    # Validate concurrency
    conc = scenario.get("target_concurrency")
    if not isinstance(conc, int) or conc <= 0:
        error_msg = f"target_concurrency must be a positive integer, got: {conc}"
        if strict:
            raise ScenarioValidationError(error_msg)
        warnings.append(ScenarioValidationWarning(error_msg))
    elif conc > 1024:
        warnings.append(
            ScenarioValidationWarning(
                f"target_concurrency={conc} is very high (>1024), ensure sufficient GPU memory"
            )
        )

    # Validate GPU configuration
    gpu = scenario.get("gpu")
    if gpu and gpu not in VALID_GPU_TYPES:
        warnings.append(
            ScenarioValidationWarning(
                f"GPU type '{gpu}' not in known list: {', '.join(sorted(VALID_GPU_TYPES))}"
            )
        )

    # Validate num_gpus and tp_size
    num_gpus = scenario.get("num_gpus")
    tp_size = scenario.get("tp_size")

    if num_gpus is not None:
        if not isinstance(num_gpus, int) or num_gpus <= 0:
            error_msg = f"num_gpus must be a positive integer, got: {num_gpus}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

    if tp_size is not None:
        if not isinstance(tp_size, int) or tp_size <= 0:
            error_msg = f"tp_size must be a positive integer, got: {tp_size}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

        # Check TP divisibility
        if num_gpus and tp_size > num_gpus:
            error_msg = f"tp_size ({tp_size}) cannot exceed num_gpus ({num_gpus})"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

        if num_gpus and num_gpus % tp_size != 0:
            warnings.append(
                ScenarioValidationWarning(
                    f"num_gpus ({num_gpus}) is not divisible by tp_size ({tp_size}), "
                    "which may lead to suboptimal GPU utilization"
                )
            )

        # Check if TP is a power of 2
        if tp_size > 0 and (tp_size & (tp_size - 1)) != 0:
            warnings.append(
                ScenarioValidationWarning(
                    f"tp_size ({tp_size}) is not a power of 2, which may impact performance"
                )
            )

    # Validate ep_size if provided
    ep_size = scenario.get("ep_size")
    if ep_size is not None:
        if not isinstance(ep_size, int) or ep_size <= 0:
            error_msg = f"ep_size must be a positive integer, got: {ep_size}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

    # Validate optional dataset generation parameters
    isl_stdev = scenario.get("isl_stdev")
    if isl_stdev is not None:
        if not isinstance(isl_stdev, (int, float)) or isl_stdev < 0:
            error_msg = f"isl_stdev must be a non-negative number, got: {isl_stdev}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

    osl_stdev = scenario.get("osl_stdev")
    if osl_stdev is not None:
        if not isinstance(osl_stdev, (int, float)) or osl_stdev < 0:
            error_msg = f"osl_stdev must be a non-negative number, got: {osl_stdev}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

    num_requests = scenario.get("num_requests")
    if num_requests is not None:
        if not isinstance(num_requests, int) or num_requests <= 0:
            error_msg = f"num_requests must be a positive integer, got: {num_requests}"
            if strict:
                raise ScenarioValidationError(error_msg)
            warnings.append(ScenarioValidationWarning(error_msg))

    return warnings


def validate_config(config: Dict[str, Any]) -> List[ScenarioValidationWarning]:
    """Validate generated configuration.

    Args:
        config: Generated configuration dictionary

    Returns:
        List of ScenarioValidationWarning objects
    """
    warnings: List[ScenarioValidationWarning] = []

    # Check KV cache configuration
    if "kv_cache_config" in config:
        kv_config = config["kv_cache_config"]
        mem_frac = kv_config.get("free_gpu_memory_fraction")

        if mem_frac is not None:
            if not isinstance(mem_frac, (int, float)) or mem_frac <= 0 or mem_frac > 1:
                warnings.append(
                    ScenarioValidationWarning(
                        f"free_gpu_memory_fraction should be between 0 and 1, got: {mem_frac}"
                    )
                )
            elif mem_frac > 0.95:
                warnings.append(
                    ScenarioValidationWarning(
                        f"free_gpu_memory_fraction={mem_frac} is very high, may cause OOM errors"
                    )
                )

    # Check batch size configuration
    if "cuda_graph_config" in config:
        cuda_config = config["cuda_graph_config"]
        max_batch = cuda_config.get("max_batch_size")

        if max_batch is not None:
            if not isinstance(max_batch, int) or max_batch <= 0:
                warnings.append(
                    ScenarioValidationWarning(
                        f"max_batch_size must be a positive integer, got: {max_batch}"
                    )
                )

    return warnings


# TODO: Re-enable llm_api_config validation once PR #8331 merges
# (https://github.com/NVIDIA/TensorRT-LLM/pull/8331)
#
# PR #8331 standardizes LlmArgs with Pydantic models, after which validation
# will happen automatically when LlmArgs(**kwargs) is instantiated.
#
