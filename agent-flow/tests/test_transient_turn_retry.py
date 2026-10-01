"""Resending a turn that produced nothing.

A campaign is hours of GPU time and dies on the first failed turn. Measured on
one reproduction, `server_error` ended 2 of 4 runs of a stage that a third run
of the SAME input completed end to end -- which is the definition of worth
retrying, and nothing in this stack did it.

The risk being managed is the opposite one. Retrying a turn that RAN and
returned something wrong replays identical inputs, fails identically, and
buries the real error under repeated attempts. So the predicate is not "the
error looks transient" -- the error string is frequently the bare word
"unknown" -- but "this message carries no tokens at all", read off the message:
`model='<synthetic>'` with every usage counter zero.
"""

from __future__ import annotations

import pytest

from agent_flow import layers
from agent_flow.backends.claude_code import TransientTurnError, _produced_nothing


class _Msg:
    """The shape `_produced_nothing` reads: a failed AssistantMessage."""

    def __init__(self, model="<synthetic>", usage=None):
        self.model = model
        self.usage = usage


ZERO = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def test_synthetic_with_zero_usage_is_the_retryable_shape():
    """The CLI's own marker for a message it manufactured, plus nothing billed."""
    assert _produced_nothing(_Msg(usage=dict(ZERO)))


def test_a_real_model_reply_is_never_retryable_however_it_failed():
    """It ran. Whatever is wrong with it will be wrong again."""
    assert not _produced_nothing(_Msg(model="claude-opus-5", usage=dict(ZERO)))


def test_synthetic_but_with_tokens_billed_is_not_retryable():
    """Tokens were produced, so the turn is not a lost request.

    Guards the sharp edge: `<synthetic>` alone is not the signal. Resending here
    could duplicate whatever those tokens already did.
    """
    assert not _produced_nothing(_Msg(usage={**ZERO, "output_tokens": 12}))


def test_a_usage_object_works_as_well_as_a_dict():
    """The SDK hands over an object; a dict is what the tests and logs show."""

    class U:
        input_tokens = output_tokens = 0
        cache_creation_input_tokens = cache_read_input_tokens = 0

    assert _produced_nothing(_Msg(usage=U()))


class _Client:
    """Fails a scripted number of times, then streams events."""

    def __init__(self, failures: int, events=("a", "b")):
        self.failures = failures
        self.events = events
        self.sent: list[str] = []

    async def send_message(self, message):
        self.sent.append(message)
        if self.failures > 0:
            self.failures -= 1
            raise TransientTurnError("Claude Code turn failed: server_error")
        for event in self.events:
            yield event


async def _drain(client, seen=None):
    out = []
    async for event in layers._resent_while_transient(
        client, "do the thing", seen if seen is not None else (lambda *a: None)
    ):
        out.append(event)
    return out


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """The backoff is 15s and 30s; a test must not actually wait 45 seconds."""
    slept: list[float] = []

    async def fake(delay):
        slept.append(delay)

    monkeypatch.setattr(layers.anyio, "sleep", fake)
    return slept


@pytest.mark.anyio
async def test_a_lost_turn_is_sent_again_and_the_events_arrive():
    client = _Client(failures=1)

    assert await _drain(client) == ["a", "b"]
    assert client.sent == ["do the thing", "do the thing"], "resent verbatim"


@pytest.mark.anyio
async def test_the_budget_is_bounded_and_the_original_error_surfaces():
    """A real outage must still end the run, with its own error rather than a hang."""
    client = _Client(failures=99)

    with pytest.raises(TransientTurnError, match="server_error"):
        await _drain(client)

    assert len(client.sent) == layers.TRANSIENT_TURN_RETRIES + 1


@pytest.mark.anyio
async def test_it_backs_off_between_attempts(_no_real_sleeping):
    """Resending instantly into a degraded endpoint is how a blip becomes a storm."""
    with pytest.raises(TransientTurnError):
        await _drain(_Client(failures=99))

    assert _no_real_sleeping == list(layers.TRANSIENT_TURN_BACKOFF_S)


@pytest.mark.anyio
async def test_a_non_transient_failure_is_not_retried():
    """A turn that produced something wrong replays identically. Fail on the first.

    Roadmap schema failures are the real instance: the file is on disk and a
    re-read returns the same bytes.
    """

    class Broken:
        def __init__(self):
            self.sent = 0

        async def send_message(self, message):
            self.sent += 1
            raise RuntimeError("roadmap.yaml failed schema validation")
            yield  # pragma: no cover - generator marker

    client = Broken()
    with pytest.raises(RuntimeError, match="schema validation"):
        await _drain(client)

    assert client.sent == 1


@pytest.mark.anyio
async def test_a_clean_turn_sends_once():
    client = _Client(failures=0)

    assert await _drain(client) == ["a", "b"]
    assert len(client.sent) == 1
