# Routing and target identity

How a configuration finds its target in tree, and — the part that decides
whether a proposed target is a target at all — what may and may not be a
routing criterion.

This replaces the standalone project's registration discussion. Out of tree a
target chose the class name it registered under and patched a stub
`config.json` to match; both are gone. The checkpoint is read exactly as
published, and the decision of which modeling code runs is routing's.

Source of record: `tensorrt_llm/_torch/_experimental/modeling_v2/README.md`
(§How a config finds its target, §What may decide a target) and
`_router_index.py`.

## The path a config takes

```
LLM(model=..., modeling_v2=...)  ->  ModelLoader
                                        |  model defaults applied to llm_args
                                        +-- modeling_v2_resolve(config, llm_args)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- route(config, mapping, compute_capability,
                                              |         spec_config) -> target class, or NoTarget
                                        |
                                     config.modeling_v2_target = class
                                        |
                    AutoModelForCausalLM._resolve_class(config)
                                        |
                                     returns that class; the registry is not consulted
```

Two levels, deliberately:

- `_router_index.py` maps `architectures[0]` — a string every checkpoint
  already declares — to a routing module. Adding a checkpoint, a GPU
  architecture or a parallel topology never touches it; only a new
  architecture family does.
- `models/<family>/routing.py` owns the rest of the decision as **one
  forward-reading decision tree per architecture family**: `route(
  pretrained_config, mapping, compute_capability, spec_config)`. The criteria
  are evaluated in the order a reader would ask them; the branch that fits
  returns the target class, and when none does `route` raises `NoTarget`
  stating the facts it compared. Reading that single file tells you where any
  configuration lands.

`route` returns the class itself (`ModelingV2GptOss120bSm103Tp1`), importing
it on the branch that returns it — `from .<target>.modeling import
ModelingV2<...>` — so a routing module that declines never loads modeling
code, and the class a reader sees returned is the class that runs. There is no
registry key and nothing registers.

A multimodal target's input processor takes the same road: the routing module
hands it to the launcher through `input_processor(pretrained_config)` beside
`route`, not through a decorator on the target's class, which the launcher's
lookup never reaches (`trtllm-modeling-v2-multimodal`,
`references/vision-path.md`, answer 2).

Nothing is rewritten either. The loader carries the class on
`ModelConfig.modeling_v2_target`; `pretrained_config.architectures` is **not**
mutated, so every other runtime branch keyed on the architecture string still
sees the checkpoint's published name and allocates accordingly.

To ask why a configuration landed where it did, read `route` against the
facts, or run under `require` and read the `NoTarget` message: it names them.

`modeling_v2` is an LLM-API argument, `off` (default) / `auto` / `require`.
Anything attributed to modeling_v2 runs under `require`. It is an argument
rather than an environment variable so that it reaches every rank the way the
rest of the arguments do. Under `auto` a configuration that misses a target's
criteria silently gets the built-in implementation, and a number measured that
way reads as the target's.

## What may decide a target

A quantity may be a routing criterion only if all three hold:

1. it is **known when `_resolve_class` runs**;
2. it is **constant for the engine's life**;
3. it changes the **structure of the forward**.

| quantity | may route? | why |
|---|---|---|
| config shape (`num_hidden_layers`, `hidden_size`, expert count, …) | **yes** | known at resolve, fixed, and it is what the declared geometry is written against |
| SM version | **yes** | known at resolve, fixed, and kernel families are architecture-gated |
| parallel topology (`world_size`, EP, attention DP) | **yes** | known at resolve, fixed, and it decides which collectives the forward contains |
| quant config, speculative config | **yes** | same three tests: they change which ops the forward calls |
| `max_num_tokens` | **no** | passes 1 and 2, fails 3. A tuning knob — an LLM API argument, not identity |
| `cuda_graph_config.max_batch_size` | **no** | same: per-instance constant, still a tuning knob |
| batch composition, `num_contexts`, "is this step pure decode" | **no** | fails 1 outright. They move every step, so a target chosen from them would be chosen once and then be wrong |

Per-step specialization is a separate mechanism — dispatch inside a target's
`forward`, on metadata — not a routing dimension. The distinction is the whole
reason routing can be a single decision tree read top to bottom.

## Checkpoint identity is a shape fingerprint

Routing recognizes a checkpoint by `(num_hidden_layers, hidden_size, …)` — the
same sniffing idiom as `is_mla` / `is_nemotron_hybrid` upstream. It **cannot
tell a fine-tune from the original**: a fine-tune of a routed checkpoint has
the same shape and is routed to the same target, silently.

That is a deliberate trade, and it changes what a gate record means. A passing
gate does not say "this target passed". It says "this modeling code passed **on
the checkpoint the accuracy gate names**". Running it on any other checkpoint
of the same shape is ungated — which is why the checkpoint path and hash
summary belong in every report, and why the accuracy gate's reference entry,
not the fingerprint, is what pins identity.

Make the fingerprint carry the fields the target's declared geometry actually
depends on, for the same reason: layer count alone may separate two
checkpoints, but the expert count is what makes the MoE operand shapes correct,
so it belongs in the tuple.

## Identity is the path

The target directory's name is the triple, `<checkpoint>__<gpu arch>__<parallel>/`,
one segment under the family, and the class name `route` returns carries the same
triple. `tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py` asserts that
`_router_index.py`, the routing modules and the targets they import agree — it is import-free and runs
anywhere, so a mismatch is caught without a GPU.
