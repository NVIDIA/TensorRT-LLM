from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams


def _capture_enqueued_max_tokens(sampling_params: SamplingParams) -> int:
    import tensorrt_llm.executor.base_worker as bw_mod

    captured = {}

    class CapturingRequest:
        BATCHED_POST_PROCESSOR_NAME = "batched_post_processor"

        def __init__(self, *args, **kwargs):
            captured["max_tokens"] = kwargs["max_tokens"]

    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
    )
    request.set_id(42)

    worker = MagicMock()
    worker.llm_args = None
    worker._executor_config = SimpleNamespace(
        max_seq_len=10,
        mapping=SimpleNamespace(cp_size=1),
    )
    worker._is_pytorch_backend = False
    worker._lora_manager = None
    worker.engine = MagicMock()
    worker.engine.enqueue_request = MagicMock(return_value=7)

    with patch.object(bw_mod.tllm, "Request", CapturingRequest):
        BaseWorker._enqueue_request(worker, request, result_wait_queue=None)

    return captured["max_tokens"]


def test_sampling_params_omitted_max_tokens_is_deduced_from_context_window():
    sampling_params = SamplingParams()

    assert sampling_params.max_tokens is None
    assert _capture_enqueued_max_tokens(sampling_params) == 7


def test_sampling_params_explicit_max_tokens_is_preserved():
    assert _capture_enqueued_max_tokens(SamplingParams(max_tokens=5)) == 5
