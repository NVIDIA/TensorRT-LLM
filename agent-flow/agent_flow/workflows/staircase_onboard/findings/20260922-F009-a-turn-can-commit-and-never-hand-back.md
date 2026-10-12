# F009 — a role turn can do its work, commit it, and never hand back

- **Found:** 2026-09-22 09:25 PDT, at the end of smith batch 1
- **Component:** the external transport's hand-back protocol, and the
  orchestrator's view of a turn's liveness
- **Severity:** one stalled turn blocks a whole batch for its full timeout,
  with completed work sitting in `HEAD` the orchestrator cannot see
- **Status:** recovered by hand; fix proposed

## What happened

Dispatch `006-smith` (verify `moe/mxe4m3_mxe2m1_block_scale_moe_runner`) did
its work and **committed it** — `a1da9c19c4`, 380 lines across the contract
and a new 281-line test, receipt frontmatter updated to
`sm_103: {status: passed, tests: 21}`.

It then never wrote `decisions/smith-5.yaml`, never wrote
`006-smith.result.md`, and never wrote `006-smith.done`.

Its eight siblings all handed back cleanly. The orchestrator sat blocked on
one missing zero-byte file for **58 minutes**, and would have sat there for
the full `--agent-turn-timeout` (4 hours as configured) before failing the
batch — while the work it was waiting for was already in `HEAD`.

## Why the protocol allows this

The transport's contract is deliberately minimal and correct as far as it
goes: `.done` is written last, so a reader that sees it can trust the result
beside it. The ordering rule is right.

What is missing is the other direction. **There is no signal for "this turn is
no longer running."** The orchestrator distinguishes exactly two states — the
`.done` file exists, or it does not — and an abandoned turn is indistinguishable
from a slow one. `FileDispatchTurnExecutor` has a `driver.heartbeat` for the
*driver* being gone, which is the right idea applied one level too high: it
detects "nothing is reading the mailbox at all", not "this particular turn
died".

That distinction matters more here than it would elsewhere, because these
turns are long. A smith that takes 30 minutes is normal; the batch that just
completed ran 20-58 minutes per dispatch. So a generous timeout is correct,
and a generous timeout is exactly what makes a stall expensive.

## The thing that makes it worse

The abandoned turn had **already committed**. So the run's git history says
the work is done, the catalog's receipt says the entry is certified, and the
orchestrator's `progress.yaml` says the dispatch never produced a decision.
Those two records disagree, and only one of them is reachable by the
orchestrator.

A resume would re-dispatch the turn, and a fresh smith would find its own work
already in `HEAD` — which is recoverable, but only because the smith skill
happens to handle "the receipt is already valid" gracefully. Nothing
guarantees that for other roles.

## Proposed change

Three parts, in increasing order of effort.

1. **A per-turn heartbeat.** The driver touches
   `dispatch/<id>.heartbeat` while a turn is in flight; the orchestrator fails
   that turn — not the batch — when it goes stale, with a message naming the
   request id and the prompt path. This mirrors `driver.heartbeat` exactly,
   one level down, and reuses its staleness logic.
2. **A stalled turn does not have to fail the batch.** With per-turn
   liveness, the orchestrator can re-dispatch that one request and let its
   siblings' results stand. Today a single stall takes the batch with it.
3. **Reconcile against the tree before re-dispatching.** For smith turns
   specifically, "did this entry's receipt become valid while nobody was
   looking" is a cheap, already-implemented check — `receipt_state` in
   `survey.py` does it. A re-dispatch that starts by reading the tree rather
   than the mailbox would recover this case automatically instead of relying
   on the agent noticing.

Part 1 is small and is most of the value: it turns a four-hour silent block
into a named failure in five minutes.

## What was done for this run

The turn was completed by a second agent, briefed to **verify what the first
one left on disk rather than redo it**, and required to state in its result
file that the turn was finished by a different agent — so the run's record
does not imply one continuous turn. The distinction between "the first agent
did this, verified from disk" and "I did this" is preserved in the hand-back.
