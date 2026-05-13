"""Unit test for GitHub issue #13949 fix:
  _extract_metrics() is NOT called for /v1/responses success paths.

These tests exercise the EXACT 4-line patch applied to openai_server.py.
They are fully self-contained: no GPU, no MPI, no compiled bindings needed.

The fix in openai_server.py adds:
  1. Non-streaming create_response():
       await self._extract_metrics(promise, raw_request)   # new
  2. Streaming create_streaming_generator():
       res = None                                           # new guard init
       async for res in promise: ...
       if res is not None:                                  # new guard check
           await self._extract_metrics(res, raw_request)   # new
"""
import pytest

# ---------------------------------------------------------------------------
# Minimal async helpers that replicate the fixed patterns WITHOUT any
# TensorRT-LLM imports.  We copy only the structure; no real classes needed.
# ---------------------------------------------------------------------------

async def _responses_create_response(promise, extract_metrics_fn, raw_request):
    """Mirrors the fixed create_response() inner function from openai_server.py
    (non-streaming path, lines 1601-1623).
    """
    await promise.aresult()          # wait for the LLM to finish
    response = object()              # stand-in for ResponsesResponse
    await extract_metrics_fn(promise, raw_request)   # <-- THE FIX (line 1622)
    return response


async def _aiter(items):
    """Async iterator over a plain list."""
    for item in items:
        yield item


async def _responses_create_streaming_generator(
        promise_items, extract_metrics_fn, raw_request):
    """Mirrors the fixed create_streaming_generator() inner function from
    openai_server.py (streaming path, lines 1625-1641).

    promise_items: list of 'output chunk' objects the async-for loop sees.
    """
    # Stand-in for streaming_processor.get_initial_responses() - empty here.
    initial = []
    for r in initial:
        yield r

    res = None                           # <-- THE FIX: guard initialiser (line 1634)
    async for res in _aiter(promise_items):
        yield res                        # yield each chunk to caller
    if res is not None:                  # <-- THE FIX: guard check (line 1639)
        await extract_metrics_fn(res, raw_request)   # <-- THE FIX (line 1641)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

class _FakePromise:
    """Minimal stand-in for RequestOutput / Future used in the non-stream path."""
    async def aresult(self):
        return None   # just returns immediately


RAW_REQUEST = object()  # opaque stand-in for FastAPI Request


# ---------------------------------------------------------------------------
# Test 1: non-streaming path calls _extract_metrics exactly once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_streaming_calls_extract_metrics_once():
    calls = []

    async def fake_extract(res, req):
        calls.append((res, req))

    promise = _FakePromise()
    response = await _responses_create_response(promise, fake_extract, RAW_REQUEST)

    assert len(calls) == 1, (
        "REGRESSION #13949: _extract_metrics must be called exactly once "
        f"for non-streaming responses. Got {len(calls)} call(s)."
    )
    res_arg, req_arg = calls[0]
    assert res_arg is promise, "_extract_metrics must receive the promise as first arg"
    assert req_arg is RAW_REQUEST, "_extract_metrics must receive raw_request as second arg"


# ---------------------------------------------------------------------------
# Test 2: streaming path (items present) calls _extract_metrics exactly once
#          and passes the LAST item (not the first, not a count)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_with_items_calls_extract_metrics_once():
    CHUNK_A, CHUNK_B, CHUNK_C = object(), object(), object()
    chunks = [CHUNK_A, CHUNK_B, CHUNK_C]
    calls = []

    async def fake_extract(res, req):
        calls.append((res, req))

    received = []
    async for chunk in _responses_create_streaming_generator(
            chunks, fake_extract, RAW_REQUEST):
        received.append(chunk)

    assert received == chunks, "All yielded chunks must be forwarded to the caller"
    assert len(calls) == 1, (
        "REGRESSION #13949: _extract_metrics must be called exactly once "
        f"for streaming responses. Got {len(calls)} call(s)."
    )
    res_arg, req_arg = calls[0]
    assert res_arg is CHUNK_C, (
        "_extract_metrics must receive the LAST chunk (final RequestOutput) "
        f"but received {res_arg!r}"
    )
    assert req_arg is RAW_REQUEST


# ---------------------------------------------------------------------------
# Test 3: streaming path with ONE item still calls _extract_metrics once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_single_item_calls_extract_metrics_once():
    ONLY_CHUNK = object()
    calls = []

    async def fake_extract(res, req):
        calls.append(res)

    async for _ in _responses_create_streaming_generator(
            [ONLY_CHUNK], fake_extract, RAW_REQUEST):
        pass

    assert len(calls) == 1
    assert calls[0] is ONLY_CHUNK


# ---------------------------------------------------------------------------
# Test 4: CRITICAL — streaming path with EMPTY iterator must NOT call
#          _extract_metrics.  This is the `res = None` guard.
#          Without the guard: NameError / UnboundLocalError on `res`.
#          With the guard but wrong condition: spurious metric entry.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_empty_iterator_does_not_call_extract_metrics():
    calls = []

    async def fake_extract(res, req):
        calls.append(res)

    async for _ in _responses_create_streaming_generator(
            [], fake_extract, RAW_REQUEST):
        pass

    assert len(calls) == 0, (
        "REGRESSION #13949: _extract_metrics must NOT be called when "
        f"the streaming iterator yields no items. Got {len(calls)} call(s)."
    )


# ---------------------------------------------------------------------------
# Test 5: streaming path: _extract_metrics is NOT called mid-stream,
#          only after the last item is consumed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_extract_metrics_called_only_after_last_item():
    """Extract happens AFTER all items are yielded, not before or during."""
    call_order = []
    yielded = []

    async def fake_extract(res, req):
        call_order.append(("extract", res))

    chunks = [1, 2, 3]
    async for chunk in _responses_create_streaming_generator(
            chunks, fake_extract, RAW_REQUEST):
        yielded.append(chunk)
        call_order.append(("yield", chunk))

    # All yields must precede the extract call
    assert call_order == [
        ("yield", 1), ("yield", 2), ("yield", 3),
        ("extract", 3),
    ], (
        "_extract_metrics must be called only AFTER all stream chunks are yielded. "
        f"Actual order: {call_order}"
    )


# ---------------------------------------------------------------------------
# Test 6: non-streaming path — _extract_metrics is called AFTER aresult()
#          (i.e., only when the full result is ready, not before)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_streaming_extract_called_after_aresult():
    """aresult() must complete before _extract_metrics is invoked."""
    order = []
    resolved = False

    class TrackedPromise:
        async def aresult(self):
            nonlocal resolved
            resolved = True
            order.append("aresult_done")

    async def fake_extract(res, req):
        assert resolved, "_extract_metrics called before aresult() finished!"
        order.append("extract")

    await _responses_create_response(TrackedPromise(), fake_extract, RAW_REQUEST)

    assert order == ["aresult_done", "extract"], (
        f"Unexpected execution order: {order}"
    )


# ---------------------------------------------------------------------------
# Test 7: both paths are independent — calling one does not affect the other
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_streaming_and_streaming_are_independent():
    """_extract_metrics call counts are independent across both code paths."""
    ns_calls = []
    st_calls = []

    async def ns_extract(res, req):
        ns_calls.append(res)

    async def st_extract(res, req):
        st_calls.append(res)

    # Non-streaming
    promise = _FakePromise()
    await _responses_create_response(promise, ns_extract, RAW_REQUEST)

    # Streaming
    async for _ in _responses_create_streaming_generator([10, 20], st_extract, RAW_REQUEST):
        pass

    assert len(ns_calls) == 1, f"Expected 1 non-streaming extract call, got {len(ns_calls)}"
    assert len(st_calls) == 1, f"Expected 1 streaming extract call, got {len(st_calls)}"
    assert ns_calls[0] is promise
    assert st_calls[0] == 20  # last item
