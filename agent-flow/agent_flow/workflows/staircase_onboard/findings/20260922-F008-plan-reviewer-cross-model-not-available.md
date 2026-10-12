# F008 — the PlanReviewer's cross-model property could not be honoured

- **Found:** 2026-09-22 07:20 PDT, dispatching the plan review
- **Component:** the `plan_reviewer` role, and the external transport generally
- **Severity:** a designed safeguard silently degraded to a weaker one
- **Status:** worked around and disclosed; no code change proposed yet

## What the design says

`README.md`, "The six roles":

> | `plan_reviewer` | persistent | **Codex** | a paper review of both |
>
> The PlanReviewer runs on a different model from the PlanDrafter on purpose:
> a reviewer that shares the author's blind spots reviews its own reasoning
> back at itself.

That is a real safeguard, and it is the only one the plan phase has — there is
no gate on a plan, only a review.

## What happened

Under `--agent-transport external` the orchestrator does not construct the
agent at all; it writes the turn to disk and a driver session answers it. The
role's *backend* therefore does not travel with the turn. The
`request.json` carries `role`, `what`, the prompt paths and the decision file
— it does not carry "run this one on a different model", because in-process
that is a property of the layer the orchestrator built, not of the request.

So the driver has to know, out of band, that this particular role wants a
different model. Nothing on disk says so.

Attempting it then failed for a second, independent reason: the driver's
available models did not include the one requested —
`model_not_found, HTTP 404` for `claude-fable-5-1`. Codex was not reachable
from this session either.

The review was therefore run on the **same model family as the drafter**, with
the divergence compensated for in the prompt — the reviewer was told
explicitly that it shares the author's blind spots, and instructed to verify
where it found itself agreeing easily rather than nodding.

That is a mitigation, not the property. It should not be recorded as if the
cross-model review happened.

## Why this is worth fixing rather than noting

The failure is silent by construction. An external driver that simply runs
every turn on whatever model it has produces a `plan_reviewer-1.yaml` that
validates, a `## Open risks` section that looks right, and a run that reports
a reviewed plan. Nothing downstream can tell the difference. The same is true
of any future role that depends on *which* model answers rather than on what
the answer contains.

Compare how carefully the transport already handles the other half of this
problem: `staircase_transport_note` exists because a role prompt written for
the in-process path ends with an instruction the external path cannot honour,
and the project decided the transport must **state** the difference rather
than let three auditor turns in a row quietly work it out. This is the same
class of problem — an in-process property that does not survive the hop — and
it currently has no such note.

## Proposed change

1. **The request carries the role's intended backend.** Add a `backend` field
   to the dispatch `request.json`, populated from the same role configuration
   the in-process transport uses. A driver then knows, from the file, what the
   turn wants. Cheap, and it makes the requirement discoverable rather than
   folklore.
2. **The result records what actually answered.** Either a field the driver
   fills in, or — better, since a driver is trusted with nothing — the
   orchestrator notes in `progress.yaml` that this turn went out over a
   transport that cannot guarantee the backend. The run's own record should
   not claim a cross-model review it cannot verify.
3. **`staircase_transport_note` gains a sentence for backend-sensitive roles**,
   in the same voice as the existing one: this transport delivers your prompt,
   it does not choose your model, and if you are the reviewer you should
   assume you may share the drafter's blind spots.

Point 2 is the one that matters. Points 1 and 3 help a driver do the right
thing; point 2 stops the run from *claiming* the right thing happened when
nobody checked.

## Recorded for this run

The plan review for MiMo-V2.6-Flash-RL was **not** cross-model. Any conclusion
that rests on "two independent models agreed" does not hold for this run's
plan phase.
