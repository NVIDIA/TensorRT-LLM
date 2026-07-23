import math

import pytest

from tensorrt_llm.sampling_params import MIN_SAMPLING_TEMPERATURE, SamplingParams


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (None, None),  # unset -> unchanged
        (0.0, 0.0),  # greedy -> untouched
        (1e-12, MIN_SAMPLING_TEMPERATURE),  # tiny -> clamped (regression #15715)
        (MIN_SAMPLING_TEMPERATURE - 1e-3, MIN_SAMPLING_TEMPERATURE),  # just below floor -> clamped
        (MIN_SAMPLING_TEMPERATURE, MIN_SAMPLING_TEMPERATURE),  # boundary -> kept
        (1.0, 1.0),  # normal -> kept
    ],
)
def test_temperature_clamped_below_floor(temperature, expected):
    # #15715: 0 < temperature < MIN reaches the backend and overflows
    # logits / temperature to inf/nan in fp16/bf16; clamp it to the floor.
    assert SamplingParams(temperature=temperature, top_k=2).temperature == expected


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.5)


@pytest.mark.parametrize("temperature", [math.inf, -math.inf, math.nan])
def test_non_finite_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        SamplingParams(temperature=temperature)
