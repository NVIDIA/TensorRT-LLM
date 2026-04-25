import pytest

from tensorrt_llm.sampling_params import SamplingParams


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_logprobs_request_limit(field):
    SamplingParams(**{field: 20})

    with pytest.raises(ValueError, match="less than or equal to 20"):
        SamplingParams(**{field: 21})
