# Decomposition — when an op is a macro, and what the report must contain

A requested callable is a **macro** when it cannot be wrapped as one
kernel-launching call: its body is Python-level composition — a registry
lookup delegating to a module `forward`, module calls, tensor glue beyond the
single call, a phase dispatch, or nested calls into other registered ops.

A macro verdict ends the onboarding path for this run. Onboard nothing: no
entry files. Produce the report instead, land it, and
dispatch one child per constituent.

`decompositions/mla-custom-op.md` is both a worked example and the shape to
copy.

## 2. Granularity and sourcing policy

Atomicity criterion: **Python-level indivisible**, as above.

Sourcing preference, in order:

1. **Ops exposed by tensorrt_llm as `torch.ops.trtllm.*`** — the default
   catalog entry shape. Almost all constituents resolve here.
2. **FlashInfer's public API** (`flashinfer.<module>.<name>`), when no
   trtllm op wraps the kernel — the sourcing ladder's tier 3. FlashInfer is a
   pinned dependency of the engine, so this needs no exception; the surface
   survey lists the installed API and which callables the engine already
   calls.
3. **Any other Python-callable atomic surface**, as an approved policy
   exception, when no torch-op boundary exists. This was required exactly
   once: 1.3.0rc21 registered **no** `torch.ops.trtllm.attention`
   (verified exhaustively against all 252 registered `trtllm::` schemas —
   and note that an exhaustive schema sweep is only exhaustive after the
   lazy registrations are forced, which
   `agent_flow/workflows/trtllm_modeling_v2/tools/list_trtllm_ops.py` does
   and a bare `import tensorrt_llm` does not),
   and the MLA attention core is otherwise reachable only inside the macro.
   The pybind binding `tensorrt_llm.bindings.internal.thop.attention` is
   Python-callable, single-call, fully explicit-state, and was onboarded as
   `attention/thop_attention`. The trade-off is recorded in its entry:
   it is invisible to torch.compile and is a private API with no stability
   promise.
4. **torch-native glue** (`split`, `concat`, `reshape`, `expand`, `copy_`,
   `empty`), called directly by the target: native torch calls are not
   catalog entries.

### Writing a decomposition report for a new macro

A requested op is a macro when its registered body contains Python-level
composition — registry lookup delegating to a module `forward`, module
calls, tensor glue beyond the single call, a phase dispatch, or nested
calls into other registered ops. In that case produce a report, not an
entry. The report must:

- **Prove the macro claim**: cite the op body's Python-level composition,
  as §1 does here.
- **Verify that every named symbol exists** in the installed engine before
  citing it (`hasattr(torch.ops.trtllm, name)` or the registered-schema
  list). Decomposition maps in this project once propagated a stale name
  (`torch.ops.trtllm.attention`) across several runs before an onboarding
  attempt exposed it.
- **Label every constituent with one of three states**: existing entry
  (cite the catalog path, or the `ops/` path of the target that keeps it) /
  not onboarded (op symbol plus proposed category and home — the shared
  catalog, or this target's `ops/` when only it calls the op; the sourcing
  tiers above decide the shape) / gap (no
  Python-callable atomic surface exists; say what is reachable instead).
- **Map the mainline path step by step** (the shape of §4) and list
  variant branches with their extra ops separately (the shape of §6).
- **Claim nothing that only certification can establish.** Cells and
  `ARCHS`, numerical behavior, and caller-facing hard constraints (this document's
  §4.5 and §5) exist only after the constituents are onboarded and tested
  on GPU; a pre-onboarding report marks them as unknown.
