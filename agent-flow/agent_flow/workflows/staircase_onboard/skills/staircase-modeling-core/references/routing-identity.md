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
LLM(model=...)  ->  ModelLoader  ->  AutoModelForCausalLM._resolve_class
                                        |
                                        +-- modeling_v2_resolve(config)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- one forward-reading decision tree
                                              +-- returns a synthetic class name
                                        |
                                     get_registered_model_class(name)
```

Two levels, deliberately:

- `_router_index.py` maps `architectures[0]` — a string every checkpoint
  already declares — to a routing module. Adding a checkpoint, a GPU
  architecture or a parallel topology never touches it; only a new
  architecture family does.
- `models/<family>/routing.py` owns the rest of the decision as **one
  forward-reading decision tree per architecture family**. The criteria are
  evaluated in the order a reader would ask them, and every branch that does
  not end in a target returns None. Reading that single file tells you where
  any configuration lands.

The name it returns (`ModelingV2GptOss120bSm103Tp1`) is a synthetic registry
key that no checkpoint declares; importing the target module is what puts a
class behind it. Upstream already does exactly this for `EAGLE3<Arch>`, which
also exists only as a `_resolve_class` rewrite.

The rewrite is local. `pretrained_config.architectures` is **not** mutated —
`_resolve_class` rewrites a local variable — so every other runtime branch
keyed on the architecture string still sees the checkpoint's published name
and allocates accordingly.

To ask why a configuration landed where it did, replay the same tree:

```
python -m tensorrt_llm._torch._experimental.modeling_v2.explain \
    --model /path/to/DeepSeek-R1-0528-NVFP4 --tp 4 --ep 4 --attention-dp
```

`TRTLLM_MODELING_V2` is `off` / `auto` / `require`. Anything attributed to
modeling_v2 runs under `require`, and it must be exported **before the ranks
start** — worker ranks receive the environment as it stood when MPI
initialized. Under `auto` a configuration that misses a target's criteria
silently gets the built-in implementation, and a number measured that way
reads as the target's.

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

`is_disagg` is declared on the routing context but **not yet plumbed**: nothing
sets it, so it reads False in every deployment. A claim test forbids any
routing module from reading it until it is, because a branch on a value that is
always False takes the wrong side silently.

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
one segment under the family, and the synthetic class name carries the same triple.
`tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py` asserts that the
routing tables and the targets they name agree — it is import-free and runs
anywhere, so a mismatch is caught without a GPU.
