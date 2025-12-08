"""
Shared enums for AutoDeploy.
"""

from enum import Enum


class MLPStyle(Enum):
    """MLP style for MoE layers."""

    GATED_MLP = "gated_mlp"  # Mixtral/DeepSeek/Llama4-style: y = W2(act(W1 x) * (W3 x))
    MLP = "mlp"  # NemotronH-style 2-layer: y = W_down(act(W_up x))


class ActivationFunction(Enum):
    """Activation functions for MoE layers."""

    SILU = "silu"  # SiLU activation
    RELU2 = "relu2"  # ReLU then square


class WeightsFormat(Enum):
    """Weight tensor organization for MoE layers."""

    PER_EXPERT = "per_expert"  # Separate weight tensors per expert in lists
    STACKED = "stacked"  # All expert weights stacked in single tensors


class WeightsFusion(Enum):
    """Weight tensor ordering and storage for gated MLP layers."""

    GATE_UP_DOWN = "w1_w2_w3_separate"  # w1, w2, w3 stored separately (matches parameter order)
    GATEUP_DOWN = (
        "w1w3_w2"  # w1 and w3 concatenated as [w1, w3], w2 separate (Llama4 native format)
    )
    UPGATE_DOWN = (
        "w3w1_w2"  # w3 and w1 concatenated as [w3, w1], w2 separate
        # (TRT-LLM format, Llama4 weights swapped during load)
    )


def mlp_style_from_str(s: str) -> MLPStyle:
    """Convert string to MLPStyle enum."""
    s = s.lower()
    for style in MLPStyle:
        if style.value == s:
            return style
    valid_values = [style.value for style in MLPStyle]
    raise ValueError(f"Unknown mlp_style '{s}'. Valid values: {valid_values}")


def act_fn_from_str(s: str) -> ActivationFunction:
    """Convert string to ActivationFunction enum."""
    s = s.lower()
    for act in ActivationFunction:
        if act.value == s:
            return act
    valid_values = [act.value for act in ActivationFunction]
    raise ValueError(f"Unknown act_fn '{s}'. Valid values: {valid_values}")


def weights_format_from_str(s: str) -> WeightsFormat:
    """Convert string to WeightsFormat enum."""
    s = s.lower()
    for fmt in WeightsFormat:
        if fmt.value == s:
            return fmt
    valid_values = [fmt.value for fmt in WeightsFormat]
    raise ValueError(f"Unknown weights_format '{s}'. Valid values: {valid_values}")


def weights_fusion_from_str(s: str) -> WeightsFusion:
    """Convert string to WeightsFusion enum."""
    s = s.lower()
    for fusion in WeightsFusion:
        if fusion.value == s:
            return fusion
    valid_values = [fusion.value for fusion in WeightsFusion]
    raise ValueError(f"Unknown weights_fusion '{s}'. Valid values: {valid_values}")
