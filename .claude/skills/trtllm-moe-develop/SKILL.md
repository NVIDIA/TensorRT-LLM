---
name: trtllm-moe-develop
description: >-
  Checklist for designing, implementing, reviewing, or debugging TensorRT-LLM
  PyTorch MoE code. Use for any change, review, or design plan that touches
  tensorrt_llm/_torch/moe/fused_moe (ConfigurableMoE, MoEScheduler, MoEImplBase
  implementations and families, can_implement and moe_resolution, quantization
  methods, EPLB, communication), MoE selection (moe_config.backend,
  moe_config.tactic, speculative_config.moe_tactic, MoESelection), model code
  that builds MoE layers or depends on the MoE backend, the MoE developer
  guide, or MoE tests.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM MoE Checklist

`tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md` and the code hold the
answers; this skill holds only the questions, because reading the guide does
not by itself bring all of them up. Read the guide first. For a design or a
review, answer each question that applies with file:line evidence from the
code; a question the code cannot answer is a finding. The section named after a
question is where the guide answers it. Where the guide and the code disagree,
the code wins: fix the guide in the same change or report it. In a review,
weigh owner boundaries and testability above local style.

## Implementations and selection

1. Is each new leaf in its family, ranked in `IMPL_PRIORITY`, and built by
   `create_moe_backend`? (Backend Selection)
2. Do its identity and its implementation sit on one concrete class, and is it
   in the lists the tests iterate (`_SINGLE_FORMAT_IMPLS`,
   `_CUTLASS_DECLARATIONS`, `_CUTEDSL_DECLARATIONS`, `_TRTLLM_GEN_IDS` in
   `test_moe_impl.py`)? (Canonical Examples)
3. Is `can_implement` pure, and are routing, EPLB, and LoRA limits declined
   there with a `MoERejectReason` (activation through `activation_support`),
   rather than only in a factory, scheduler, wrapper, or test-local skip? A
   factory check is a backstop for callers that bypass resolution.
   (`can_implement` must be pure, Anti-Patterns)
4. Is a family name matched with `issubclass` / `isinstance`, never `==`, `is`,
   or `in (...)`? (Canonical Examples, note on family names)
5. Is `resolve_moe_impl` still the only selection entry point, with capability
   knowledge only in the gates? (Anti-Patterns)

## Model code

6. Does new model code branch on the `moe_backend` literal instead of asking
   `resolve_moe_cls`? (Anti-Patterns)
7. Does a layer whose experts have their own format pass it to `create_moe` as
   `override_quant_config`, and does a layer that must select differently
   re-point with `ModelConfig.copy_with` and a warning? (A layer's format is
   fixed when it is built, Backend Selection)
8. Do draft layers select for their own weights, and does a modeling_v2 target
   declare the ids it honors in `moe_impls_honored`? (Backend Selection)

## Weights, quantization, and EPLB

9. Is every new per-expert tensor registered for online EPLB migration, from a
   host copy indexed by `get_load_expert_ids()` and not by local slot, and is it
   one of the quant method's live planes under per-layer dynamic EPLB?
   (Scheduler / EPLB Constraints)
10. Is `eplb_support_status` explicit, and does a leaf that cannot run EPLB
    decline `d.eplb_enabled` even when a parent's `_supports_load_balancer()`
    says yes?
11. For an imported kernel, are the weight layout and the scale semantics taken
    from where the kernel consumes them, not from comments or parameter names,
    and checked against an upstream invocation dump?

## Forward execution and communication

12. Does `ConfigurableMoE.forward_impl` stay free of backend, routing,
    communication, and EPLB branches, with per-forward policy in
    `MoEScheduler`, and is `run_moe` the only compute call the scheduler
    makes? (Architecture, Anti-Patterns)
13. Does host-side communication stay out of a backend unless its kernel owns
    the exchange (`FUSED_COMM`), and does a strategy that can block
    checkpointing override `checkpoint_blocked_reason()`? (Communication
    checkpointing, Scheduler / EPLB Constraints)

## Generated files, docs, and tests

14. Are the generated files current: the implementation catalog
    (`python3 scripts/generate_moe_impl_catalog.py`), the serve schemas for a
    new id segment, and the LLM-args golden manifest for a `MoeConfig` change
    (AGENTS.md)? Is each guide section the change makes stale updated (File
    Map, Implementation Families, Backend Capability Matrix, Backend Selection,
    Activation Support, Anti-Patterns, Tests)?
15. Which tests does the change need? (Tests) When a test needs a GPU, use the
    GPU allocation and test-runner skills, and record each skipped test with
    its reason.
