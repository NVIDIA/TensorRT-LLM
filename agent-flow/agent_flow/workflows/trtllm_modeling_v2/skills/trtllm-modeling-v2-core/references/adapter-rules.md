# Adapter rules — a feature that talks to a runtime-owned service

Most features are pure modeling: they change what the forward computes, so
they are written flat into `modeling.py` and there is nothing to adapt. This
file is for the other kind — a feature whose runtime side owns state that
outlives one forward call (expert placement and migration; a composite weight
view with remote prefetch). Those need a named boundary, and the boundary is a
file.

The split is the same every time:

| layer | owns |
|---|---|
| **runtime** | the service: its threads, its communicators, its state machine, its memory |
| **adapter** | the boundary — registration, the handles, the facade the runtime traverses |
| **modeling code** | the calls, at the points in the forward where they belong |

## Rules

**One file, one import site.** `<feature>_adapter.py` is the *only* file in
the target that imports the private runtime surface. `modeling.py` and
`weights.py` see the adapter's own types and nothing else. This is what makes
the blast radius of an upstream change readable: one file to re-derive. A
target directory otherwise holds `modeling.py` and `weights.py` only — this
file is the single exception a feature of this class earns.

**The adapter does no work of its own.** It creates no stream, issues no copy,
allocates no tensor, launches no kernel, and does not reach into the service's
internals to do by hand what the service does. It hands over geometry,
placement and closures, and relays calls. An adapter that starts computing has
absorbed a piece of the runtime, and the next upstream change will break it
silently.

**Pin the private surface by digest.** A public API has a version contract; a
private one does not, and in tree there is no version pin to fall back on —
the target moves with the trunk. Record the SHA-256 of every upstream file the
adapter binds against in the run journal, beside the gate record. A change to
any of them re-opens every question the adapter answers, and the digest is what
turns that from a silent drift into a visible diff.

**Naming:**

```
<feature>_adapter.py        the file
<Feature>Adapter            the public class — the entire outward surface
_<Feature><Role>            private helpers: Facade, Stub, …
```

Do not name a helper `*Shell`. In this repo "shell" already means the target's
registered `DecoderModelForCausalLM` subclass.

## What the feature's own skill must document

Two files, both required for an adapter-class feature:

- `runtime-boundary.md` — which layer owns what, in the table above's shape.
  Concretely enough that a reader can tell whether a given piece of state is
  ours
- `adapter-interface.md` — what the adapter exposes: every method and property
  the runtime calls or reads, and **when** each is called. Timing is half the
  content; a hook called at the wrong point in the forward is a silent wrong
  answer, not a crash

## Gating

An adapter-class feature does **not** automatically owe a third gate. Ask the
only question that earns one: **can a miscomputed version of this still
produce a correct answer?** For a feature that routes to different weights, or
binds a different expert set, the answer is no — accuracy sees it, and the
gate ladder is boot → accuracy like everything else.

What these features *can* hide is being **inert**: registered, called, and
doing nothing useful. That produces correct output, so it is a performance
question, and performance here is measured rather than gated — it belongs to
the later tuning workflow.

Two cheap things do belong in onboarding, and they are not gates:

- **a static variant first**, where the feature has one. Green on static, then
  green on dynamic, makes a failure attributable to migration rather than to
  wiring
- **the boot gate asserting the launch preconditions** the feature needs but
  the runtime does not check. Turning a silent hang into a named failure is
  what the minutes-scale gate is for
