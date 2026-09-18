"""submit() must register the GenerationResult before the RPC send.

The response loop runs on another thread and handle_responses() drops any
response whose client_id is not in _results ("Received response for unknown
client_id"). A request that completes before the submitting thread reaches the
registration -- a short generation, or the submitter stalled between the send
and the registration -- therefore lost its final response and the caller waited
for it forever (seen as 1800 s request timeouts in disaggregated serving).
"""

import types

import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.sampling_params import SamplingParams


class _FinalResponse:
    """Duck-typed final LlmResponse (is_llm_response keys on has_error)."""

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.result = types.SimpleNamespace(is_final=True)

    def has_error(self) -> bool:
        return False


class _RacingRpcClient:
    """rpc_client stand-in that answers synchronously inside the send.

    The final response is delivered before submit() can run anything that
    follows the send: the extreme form of the response-thread race.
    """

    def __init__(self, proxy):
        self._proxy = proxy
        self.sent = []

    def submit(self, request):
        client = self

        class _Call:
            def remote(self, need_response=False):
                client.sent.append(request.id)
                client._proxy.handle_responses([[_FinalResponse(request.id)]])

        return _Call()


class _FailingRpcClient:
    def submit(self, request):
        class _Call:
            def remote(self, need_response=False):
                raise RuntimeError("send failed")

        return _Call()


class _MixinProxy(RpcExecutorMixin, GenerationExecutor):
    """Host class providing what the mixin expects from the executor."""


# GenerationExecutor is abstract; the tests never call the abstract methods
# and construct the proxy with __new__, which refuses abstract classes.
_MixinProxy.__abstractmethods__ = frozenset()


def _proxy_classes():
    classes = [_MixinProxy]
    try:
        from tensorrt_llm.executor.ray.executor import RayExecutor

        classes.append(RayExecutor)
    except Exception:  # ray not installed
        pass
    return classes


def _make_proxy(cls, rpc_client_factory):
    # No __init__: the real constructors start workers / Ray. Set only what
    # submit() and handle_responses() touch.
    proxy = cls.__new__(cls)
    proxy._results = {}
    proxy._last_client_id = 0
    proxy.postproc_config = types.SimpleNamespace(num_postprocess_workers=0)
    proxy.rpc_client = rpc_client_factory(proxy)
    return proxy


def _request() -> GenerationRequest:
    return GenerationRequest([1, 2, 3], SamplingParams(max_tokens=1))


@pytest.mark.parametrize("cls", _proxy_classes(), ids=lambda c: c.__name__)
def test_response_arriving_during_send_is_delivered(cls):
    proxy = _make_proxy(cls, _RacingRpcClient)
    request = _request()

    result = proxy.submit(request)

    assert proxy.rpc_client.sent == [request.id]
    # The final response was delivered to the result queue instead of being
    # dropped as "unknown client_id" ...
    delivered = result.queue.get_nowait()
    assert delivered.client_id == request.id
    # ... and, being final, the registration was released.
    assert request.id not in proxy._results


@pytest.mark.parametrize("cls", _proxy_classes(), ids=lambda c: c.__name__)
def test_failed_send_leaves_no_registration(cls):
    proxy = _make_proxy(cls, lambda _proxy: _FailingRpcClient())
    request = _request()

    with pytest.raises(RuntimeError, match="send failed"):
        proxy.submit(request)

    assert proxy._results == {}
